import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.dataset_msrs import MSRS_dataset, Generator
from utils.common import YCbCr2RGB, tensor2img, save_img_single
from models.model import DMDFuse as Fusion

try:
    from thop import profile
    has_thop = True
except ImportError:
    has_thop = False


def crop_to_multiple_of_8(tensor):
    """Crop spatial dimensions (H, W) to multiples of 8 for multi-scale DWT."""
    H, W = tensor.shape[-2:]
    new_H = H - (H % 8)
    new_W = W - (W % 8)
    return tensor[..., :new_H, :new_W]


def test_fusion(args):
    save_dir = './output/MSRS'
    os.makedirs(save_dir, exist_ok=True)
    fusion_model_path = 'epoch_39.pth'

    net = Fusion(in_chans=1,
                 out_chans=1,
                 embed_dim=32,
                 depth=3).cuda()

    net.load_state_dict(torch.load(fusion_model_path))
    net.eval()
    
    test_dataset = MSRS_dataset(base_dir=args.root_path, split="test",
                                transform=transforms.Compose([Generator()]))

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,   
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader.n_iter = len(test_loader)
    test_bar = tqdm(test_loader)

    total_infer_time = 0.0
    flops = 0
    params = 0

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(test_bar):
            image_vis_y_batch, image_ir_batch = sampled_batch['image_vis_y'], sampled_batch['image_ir']
            original_size = image_vis_y_batch.shape[-2:]  # (H, W)

            image_vis_y_batch = crop_to_multiple_of_8(image_vis_y_batch).cuda()
            image_ir_batch = crop_to_multiple_of_8(image_ir_batch).cuda()

            image_vis_cb, image_vis_cr = sampled_batch['image_vis_cb'], sampled_batch['image_vis_cr']
            image_vis_cb, image_vis_cr = image_vis_cb.cuda(), image_vis_cr.cuda()

            # Compute model parameters and FLOPs on the first sample
            if i_batch == 0:
                params = sum(p.numel() for p in net.parameters())
                if has_thop:
                    flops, _ = profile(net, inputs=(image_vis_y_batch, image_ir_batch), verbose=False)
                else:
                    print("\n[Info] 'thop' is not installed. Skipping FLOPs calculation. (Run 'pip install thop' to enable)")

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()

            # Model inference
            fuse_y = net(image_vis_y_batch, image_ir_batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            infer_time = time.time() - start_time
            total_infer_time += infer_time

            # Restore original spatial resolution and convert YCbCr back to RGB
            fuse_y_resized = torch.nn.functional.interpolate(fuse_y, size=original_size, mode='bilinear',
                                                             align_corners=False)
            fuse_img = YCbCr2RGB(fuse_y_resized, image_vis_cb, image_vis_cr)

            name = sampled_batch['name']
            for k in range(len(name)):
                img_name = name[k]
                save_path = os.path.join(save_dir, img_name)
                single_fuse_img = tensor2img(fuse_img[k])
                save_img_single(single_fuse_img, save_path)

    # Print evaluation summary
    num_images = len(test_dataset)
    avg_infer_time = total_infer_time / num_images if num_images > 0 else 0

    print("\n" + "=" * 50)
    print("Total testing completed!")
    print(f"Total images processed: {num_images}")
    print(f"Average inference time: {avg_infer_time:.4f} seconds")
    print(f"FPS: {1 / avg_infer_time:.2f}" if avg_infer_time > 0 else "FPS: N/A")
    if flops > 0:
        print(f"FLOPs: {flops / 1e9:.3f} G")
    print(f"Model Parameters: {params / 1e6:.3f} M")
    print("=" * 50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./datasets/MSRS')
    args = parser.parse_args()
    test_fusion(args)