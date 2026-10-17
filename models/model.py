import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import repeat


# ==========================================
# 2D Selective Scan Module (SS2D)
# ==========================================
class SSM2D_MB(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2., dt_rank="auto", dt_min=0.001, dt_max=0.1, 
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4, dropout=0., device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        
        self.x_proj = nn.ModuleList([
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(4)
        ])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj
        
        self.dt_projs = nn.ModuleList([
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(4)
        ])
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32, device=device), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge: A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge: D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L, K = H * W, 4
        
        # 4-way scanning serialization
        x_hw = x.contiguous().view(B, -1, L)
        x_wh = torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        x_hwwh = torch.stack([x_hw, x_wh], dim=1).view(B, 2, -1, L)
        
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        
        # Cast to float32 for numerical stability under AMP
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)
        
        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, 
            z=None, 
            delta_bias=dt_projs_bias, 
            delta_softplus=True, 
            return_last_state=False
        ).view(B, K, -1, L)
        
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).contiguous().view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        
        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        y1, y2, y3, y4 = self.forward_core(x)
        
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        return y


# ==========================================
# Visual State Space Block (VSSB)
# ==========================================
class VSS(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: float = 2.0, **kwargs):
        super().__init__()
        self.d_model, self.d_inner = d_model, int(expand * d_model)
        self.ln1 = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.act_top = nn.SiLU()
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner, kernel_size=3, padding=1)
        self.act_bottom = nn.SiLU()
        self.ss2d = SSM2D_MB(d_model=self.d_model, d_state=d_state, expand=expand, **kwargs)
        self.ln2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor):
        shortcut = x 
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln1(x)
        
        xz = self.in_proj(x)
        x_bot, x_top = xz.chunk(2, dim=-1)
        x_top = self.act_top(x_top)
        
        x_bot = x_bot.permute(0, 3, 1, 2).contiguous()
        x_bot = self.act_bottom(self.conv2d(x_bot))
        x_bot = self.ss2d(x_bot)
        
        x_bot = self.ln2(x_bot)
        x_merged = x_bot * x_top
        
        out = self.out_proj(x_merged).permute(0, 3, 1, 2).contiguous()
        return out + shortcut


# ==========================================
# Spectral Contextual Mamba Block (SCMB)
# ==========================================
class HaarDWT(nn.Module):
    def forward(self, x):
        x_unshuffled = F.pixel_unshuffle(x, 2)
        x00 = x_unshuffled[:, 0::4]
        x01 = x_unshuffled[:, 1::4]
        x10 = x_unshuffled[:, 2::4]
        x11 = x_unshuffled[:, 3::4]
        
        L = (x00 + x01 + x10 + x11) / 2.0
        H = torch.cat([
            (x00 - x01 + x10 - x11) / 2.0, 
            (x00 + x01 - x10 - x11) / 2.0, 
            (x00 - x01 - x10 + x11) / 2.0
        ], dim=1).contiguous()
        return L, H


class HaarIWT(nn.Module):
    def forward(self, L, H):
        B, C, Height, Width = L.shape
        H1, H2, H3 = torch.chunk(H, 3, dim=1)
        
        x00 = (L + H1 + H2 + H3) / 2.0
        x01 = (L - H1 + H2 - H3) / 2.0
        x10 = (L + H1 - H2 - H3) / 2.0
        x11 = (L - H1 - H2 + H3) / 2.0
        
        out_shuffled = torch.empty(B, C * 4, Height, Width, device=L.device, dtype=L.dtype)
        out_shuffled[:, 0::4] = x00
        out_shuffled[:, 1::4] = x01
        out_shuffled[:, 2::4] = x10
        out_shuffled[:, 3::4] = x11
        
        return F.pixel_shuffle(out_shuffled, 2).contiguous()


class MLB(nn.Module):
    """Mamba Lifting Block (MLB)"""
    def __init__(self, in_channels_L: int):
        super().__init__()
        self.C, self.C_H = in_channels_L, 3 * in_channels_L
        self.vss_H = VSS(d_model=self.C_H)
        self.vss_L = VSS(d_model=self.C)
        # Prediction (P) branch
        self.vss_P = VSS(d_model=self.C)
        self.pwconv_P = nn.Conv2d(self.C, self.C_H, kernel_size=1)
        # Update (U) branch
        self.pwconv_SU_down = nn.Conv2d(self.C_H, self.C, kernel_size=1)
        self.vss_SU = VSS(d_model=self.C)
        self.pwconv_SU_dual_head = nn.Conv2d(self.C, self.C * 2, kernel_size=1)

    def forward(self, L, H):
        H_feat, L_feat = self.vss_H(H), self.vss_L(L)
        # P branch: refine high-frequency features
        p = self.pwconv_P(self.vss_P(L_feat))
        H_updated = H_feat - p
        # U branch: scale and shift modulation for low-frequency features
        su = self.pwconv_SU_dual_head(self.vss_SU(self.pwconv_SU_down(H_updated)))
        s, u = su.chunk(2, dim=1)
        
        s = torch.exp(torch.tanh(s) * 2.0)
        L_updated = L_feat * s + u
        return L_updated, H_updated


class SCMB(nn.Module):
    """Spectral Contextual Mamba Block (SCMB)"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.wt, self.iwt = HaarDWT(), HaarIWT()
        self.mlb1, self.mlb2 = MLB(in_channels), MLB(in_channels)

    def forward(self, x):
        # 1st-order DWT & MLB
        L1, H1 = self.wt(x) 
        L1_mod, H1_mod = self.mlb1(L1, H1)
        # 2nd-order DWT & MLB
        L2, H2 = self.wt(L1_mod)
        L2_mod, H2_mod = self.mlb2(L2, H2)
        
        # IDWT reconstruction with residual connection
        L1_fused = self.iwt(L2_mod, H2_mod)
        return self.iwt(L1_fused, H1_mod) + x


# ==========================================
# Learnable Descriptive Convolution & Attention
# ==========================================
class LDC(nn.Module):
    """Learnable Descriptive Convolution (LDC)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=None, bias=False):
        super().__init__()
        if groups is None:
            groups = in_channels 
            
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        
        # Initialize learnable descriptor m to 0 (equivalent to standard conv initially)
        self.m = nn.Parameter(
            torch.zeros(out_channels, in_channels // groups, kernel_size, kernel_size), 
            requires_grad=True
        )

    def forward(self, x):
        # Modulate kernel weights: W_eff = W * (1 + M)
        effective_weight = self.conv.weight * (1.0 + self.m)
        
        return F.conv2d(
            input=x, 
            weight=effective_weight, 
            bias=self.conv.bias, 
            stride=self.conv.stride, 
            padding=self.conv.padding, 
            dilation=self.conv.dilation, 
            groups=self.conv.groups
        )


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        y = self.sigmoid(avg_out + max_out)
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


def window_partition(x, window_size):
    """Partition feature maps into non-overlapping local windows with adaptive padding."""
    B, C, H, W = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h))
        
    Hp, Wp = x.shape[2], x.shape[3]
    x = x.view(B, C, Hp // window_size, window_size, Wp // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows, (H, W), (Hp, Wp)


def window_reverse(windows, window_size, H, W, Hp, Wp):
    """Reverse local windows back to full feature maps and remove padding."""
    B = int(windows.shape[0] // ((Hp * Wp) // (window_size * window_size)))
    x = windows.view(B, Hp // window_size, Wp // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, Hp, Wp)
    
    if Hp > H or Wp > W:
        x = x[:, :, :H, :W].contiguous()
    return x


# ==========================================
# Dynamic Routing Mamba Block (DRMB)
# ==========================================
class DRS(nn.Module):
    """Dynamic Routing Scanning (DRS / SW-DRS when shift_size > 0)"""
    def __init__(self, d_model: int, expand: float = 2.0, window_size: int = 32, shift_size: int = 0):
        super().__init__()
        self.d_model, self.d_inner = d_model, int(expand * d_model)
        self.window_size = window_size
        self.shift_size = shift_size
        
        self.ln1 = nn.LayerNorm(self.d_model)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        self.ca = ChannelAttention(self.d_inner)
        self.act_top = nn.SiLU()
        self.ldc = LDC(self.d_inner, self.d_inner)
        self.act_bottom = nn.SiLU()
        
        self.s6 = Mamba(d_model=self.d_inner, d_state=16, d_conv=4, expand=1)
        
        self.ln2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln1(x)
        xz = self.in_proj(x)
        x_top, x_bot = xz.chunk(2, dim=-1)
        
        # Top branch: Channel Attention
        x_top = x_top.permute(0, 3, 1, 2).contiguous()
        x_top = self.ca(x_top)
        x_top = self.act_top(x_top)
        
        # Bottom branch: LDC-guided Dynamic Routing Scan
        x_bot = x_bot.permute(0, 3, 1, 2).contiguous()
        x_bot = self.ldc(x_bot)
        
        # Cyclic shift for SW-DRS
        if self.shift_size > 0:
            x_bot = torch.roll(x_bot, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))
            
        x_windows, (orig_H, orig_W), (Hp, Wp) = window_partition(x_bot, self.window_size)
        NW_B = x_windows.shape[0]
        L_win = self.window_size * self.window_size
        
        # Local token reordering guided by LDC saliency map
        x_flat = x_windows.contiguous().view(NW_B, self.d_inner, L_win)
        scores = x_windows.abs().sum(dim=1).view(NW_B, L_win)
        
        sorted_idx = torch.argsort(scores, dim=-1, descending=True)
        idx_exp = sorted_idx.unsqueeze(1).expand(-1, self.d_inner, -1)
        
        x_sorted = torch.gather(x_flat, 2, idx_exp)
        x_sorted = self.act_bottom(x_sorted)
        x_sorted = x_sorted.permute(0, 2, 1).contiguous()
        
        # S6 state space modeling
        x_mamba = self.s6(x_sorted)
        x_mamba = x_mamba.permute(0, 2, 1).contiguous()
        
        # Inverse sorting to restore spatial positions
        x_unsorted = torch.zeros_like(x_flat).scatter_(2, idx_exp, x_mamba)
        x_windows_out = x_unsorted.contiguous().view(NW_B, self.d_inner, self.window_size, self.window_size)
        
        x_bot = window_reverse(x_windows_out, self.window_size, orig_H, orig_W, Hp, Wp)
        
        # Reverse cyclic shift
        if self.shift_size > 0:
            x_bot = torch.roll(x_bot, shifts=(self.shift_size, self.shift_size), dims=(2, 3))
            
        x_bot = x_bot.permute(0, 2, 3, 1).contiguous()
        x_bot = self.ln2(x_bot)
        
        x_top = x_top.permute(0, 2, 3, 1).contiguous()
        x_merged = x_bot * x_top
        
        return self.out_proj(x_merged).permute(0, 3, 1, 2).contiguous() + shortcut


class DRMB(nn.Module):
    """Dynamic Routing Mamba Block (DRMB): Cascaded DRS and SW-DRS"""
    def __init__(self, d_model: int, expand: float = 2.0, window_size: int = 32):
        super().__init__()
        self.drs = DRS(d_model, expand, window_size, shift_size=0)
        self.sw_drs = DRS(d_model, expand, window_size, shift_size=window_size // 2)

    def forward(self, x):
        x = self.drs(x)
        x = self.sw_drs(x)
        return x


# ==========================================
# Discrepancy-Guided Attention Fusion (DGAF)
# ==========================================
class DGAF(nn.Module):
    """Discrepancy-Guided Attention Fusion (DGAF)"""
    def __init__(self, dim):
        super().__init__()
        self.sa_ir = SpatialAttention()
        self.sa_vi = SpatialAttention()
        self.ca = ChannelAttention(3 * dim)
        self.out_proj = nn.Conv2d(3 * dim, dim, kernel_size=1)

    def forward(self, f_spe_r, f_spe_v, f_spa):
        # Cross-modal feature residuals as spatial attention priors
        mask_ir = self.sa_ir(f_spa - f_spe_r)
        mask_vi = self.sa_vi(f_spa - f_spe_v)
        
        f_hat = f_spa + (f_spa * mask_vi) + (f_spa * mask_ir)
        f_ca = self.ca(torch.cat([f_spe_r, f_hat, f_spe_v], dim=1))
        
        return self.out_proj(f_ca) + f_spa


# ==========================================
# Overall Network Architecture (DMDFuse)
# ==========================================
class DMDFuse(nn.Module):
    def __init__(self,
                 in_chans=1, 
                 out_chans=1, 
                 embed_dim=32, 
                 depth=3):
        super().__init__()
        # Shallow feature extraction
        self.conv_ir1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_vi1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 4, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_ir12 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_vi12 = nn.Sequential(
            nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )

        self.conv_ir2 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        self.conv_vi2 = nn.Sequential(
            nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
            nn.LeakyReLU(0.2, True)
        )
        
        # Deep spatial-frequency decoupling and fusion streams
        self.scmb_ir_layers = nn.ModuleList([SCMB(embed_dim) for _ in range(depth)])
        self.scmb_vi_layers = nn.ModuleList([SCMB(embed_dim) for _ in range(depth)])
        self.drmb_layers = nn.ModuleList([DRMB(embed_dim) for _ in range(depth)])
        self.dgaf_layers = nn.ModuleList([DGAF(embed_dim) for _ in range(depth)])
        
        # Image reconstruction
        self.conv_final = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, 1, 1), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(embed_dim // 4, out_chans, 3, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, y, ir):
        ir_c1 = self.conv_ir1(ir)
        vi_c1 = self.conv_vi1(y)

        ir_c12 = self.conv_ir12(ir_c1)
        vi_c12 = self.conv_vi12(vi_c1) 
        
        f_f = torch.cat([ir_c12, vi_c12], dim=1)
        f_spe_r = self.conv_ir2(ir_c12)
        f_spe_v = self.conv_vi2(vi_c12)
        
        for i in range(len(self.scmb_ir_layers)):
            f_spe_r = self.scmb_ir_layers[i](f_spe_r)
            f_spe_v = self.scmb_vi_layers[i](f_spe_v)
            
            f_spa = self.drmb_layers[i](f_f)
            f_f = self.dgaf_layers[i](f_spe_r, f_spe_v, f_spa)
            
        fuse_out = self.conv_final(f_f)
        return fuse_out


# Alias for backward compatibility
Fusion = DMDFuse