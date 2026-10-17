import torch
import torch.nn as nn
import torch.nn.functional as F


class Sobel_xy(nn.Module):
    """Sobel operator for spatial gradient computation."""
    def __init__(self):
        super(Sobel_xy, self).__init__()
        kernel_x = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_y = [[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False).cuda()
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False).cuda()

    def forward(self, x):
        sobel_x = F.conv2d(x, self.weight_x, padding=1)
        sobel_y = F.conv2d(x, self.weight_y, padding=1)
        return torch.abs(sobel_x) + torch.abs(sobel_y)


class Grad_loss(nn.Module):
    """Maximum gradient loss (Eq. 28)."""
    def __init__(self):
        super(Grad_loss, self).__init__()
        self.sobel = Sobel_xy()

    def forward(self, image_y, image_ir, fuse_y):
        image_y_grad = self.sobel(image_y)
        image_ir_grad = self.sobel(image_ir)
        grad_max = torch.max(image_y_grad, image_ir_grad)
        fuse_y_grad = self.sobel(fuse_y)
        loss_grad = F.l1_loss(grad_max, fuse_y_grad)

        return loss_grad


class Int_loss(nn.Module):
    """Saliency-based intensity loss (Eq. 27)."""
    def __init__(self):
        super(Int_loss, self).__init__()

    def get_saliency_map(self, img):
        """
        Compute global contrast saliency map based on intensity histogram:
        S(k) = \sum_{j=0}^{255} H(j) * |I(k) - j|
        """
        B, C, H, W = img.shape
        img_255 = (img * 255).clamp(0, 255).long()
        
        saliency_maps = []
        for b in range(B):
            pixel_data = img[b].detach() 
            hist = torch.histc(pixel_data, bins=256, min=0, max=1)
            
            # Pairwise gray-level distance matrix: dist[i, j] = |i - j|
            levels = torch.arange(256).to(img.device).float()
            dist_matrix = torch.abs(levels.unsqueeze(0) - levels.unsqueeze(1))
            
            # Saliency lookup table via matrix multiplication
            saliency_lookup = torch.matmul(hist.unsqueeze(0), dist_matrix).flatten()
            
            # Map saliency values back to spatial resolution
            curr_saliency = saliency_lookup[img_255[b]].view(C, H, W)
            saliency_maps.append(curr_saliency)
            
        return torch.stack(saliency_maps)

    def forward(self, image_y, image_ir, fuse_y):
        s_v = self.get_saliency_map(image_y)
        s_r = self.get_saliency_map(image_ir)

        # Min-max normalization
        v_min = torch.amin(s_v, dim=(1, 2, 3), keepdim=True)
        v_max = torch.amax(s_v, dim=(1, 2, 3), keepdim=True)
        s_v = (s_v - v_min) / (v_max - v_min + 1e-8)

        r_min = torch.amin(s_r, dim=(1, 2, 3), keepdim=True)
        r_max = torch.amax(s_r, dim=(1, 2, 3), keepdim=True)
        s_r = (s_r - r_min) / (r_max - r_min + 1e-8)

        # Modality weight maps
        w_v = 0.5 + 0.5 * (s_v - s_r)
        w_r = 1.0 - w_v
        
        target_intensity = w_v * image_y + w_r * image_ir
        loss_int = F.l1_loss(fuse_y, target_intensity)
        
        return loss_int


class SCD_loss(nn.Module):
    """Correlation discrepancy loss (Eq. 29)."""
    def __init__(self, eps=1e-8):
        super(SCD_loss, self).__init__()
        self.eps = eps

    def corr2(self, x, y):
        """Compute 2D Pearson correlation coefficient across spatial dimensions (H, W)."""
        mean_x = torch.mean(x, dim=[2, 3], keepdim=True)
        mean_y = torch.mean(y, dim=[2, 3], keepdim=True)
        
        centered_x = x - mean_x
        centered_y = y - mean_y
        
        cov = torch.sum(centered_x * centered_y, dim=[2, 3])
        var_x = torch.sum(centered_x ** 2, dim=[2, 3])
        var_y = torch.sum(centered_y ** 2, dim=[2, 3])
        
        corr = cov / (torch.sqrt(var_x * var_y) + self.eps)
        return torch.mean(corr)

    def forward(self, image_y, image_ir, fuse_y):
        # Cross-modal residual maps
        D_y = fuse_y - image_ir
        D_ir = fuse_y - image_y
        
        corr_y = self.corr2(D_y, image_y)
        corr_ir = self.corr2(D_ir, image_ir)
        
        scd = corr_y + corr_ir
        loss_scd = 1.0 - 0.5 * scd
        
        return loss_scd