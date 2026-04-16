import torch
import random
import numpy as np
from algorithms import shift
from algorithms.pipeline import pipeline


random.seed(5)
np.random.seed(5)
torch.manual_seed(5)
torch.cuda.manual_seed(5) 
torch.cuda.manual_seed_all(5)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def rgb_pixel_shift(rgb, dx=0.5, dy=0.0):
    """
    Perform pixel shifting with bilinear interpolation on RGB Tensor with shape (H, W, 3).
    
    Args:
        rgb (torch.Tensor): Input tensor with shape (H, W, 3), dtype should be float32.
        dx (float, optional): Horizontal shift amount in pixels.
        dy (float, optional): Vertical shift amount in pixels.
    Returns:
        torch.Tensor: Shifted tensor with the same shape (H, W, 3) and dtype as input.
    """
    assert rgb.ndim == 3, f"Input tensor must have shape (H, W, 3), got {rgb.shape}"
    H, W, C = rgb.shape
    device = rgb.device
    dtype = rgb.dtype

    
    input_tensor = rgb.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    x = torch.linspace(-1.0 + 1.0/W, 1.0 - 1.0/W, W, device=device, dtype=dtype)
    y = torch.linspace(-1.0 + 1.0/H, 1.0 - 1.0/H, H, device=device, dtype=dtype)
    
    y_grid, x_grid = torch.meshgrid(y, x, indexing="ij")

    # Calculate half-pixel shift (convert pixel shift to normalized coordinate shift)
    dx_norm = dx * 2.0 / W
    dy_norm = dy * 2.0 / H

    x_grid = x_grid + dx_norm
    y_grid = y_grid + dy_norm

    grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)  # (1, H, W, 2)

    # Perform bilinear interpolation
    shifted_tensor = torch.nn.functional.grid_sample(
        input_tensor,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=False
    )

    # (1, C, H, W) → (H, W, C)
    shifted_tensor = shifted_tensor.squeeze(0).permute(1, 2, 0)

    return shifted_tensor

def parse():
    import argparse
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--dataset_dir', default="./datasets/TSA-simu/")
    parser.add_argument('--height', default=256, type=int, help="")
    parser.add_argument('--width', default=256, type=int, help="")
    parser.add_argument('--nbands', default=28, type=int, help="")

    parser.add_argument('--mask_path', default="mask.mat")
    parser.add_argument('--shear_step', default=2, type=int, help="Shearing transformation step size.")
    parser.add_argument('--shear_dim', default=1, type=int, help="The dimension sheared.")
    
    parser.add_argument('--K', default=30, type=int, help="")
    parser.add_argument('--N', default=10, type=int, help="")
    parser.add_argument('--n_stage', default=30, type=int, help="")
    parser.add_argument('--mu', default=3e-2, type=float, help="")
    parser.add_argument('--rho', default=3e-2, type=float, help="")
    parser.add_argument('--alpha', default=1e-1, type=float, help="")
    parser.add_argument('--beta', default=1.2, type=float, help="")
    parser.add_argument('--use_net_init', default=True, type=bool, help="")

    args = parser.parse_args()
    return args

def load_data(path, scene, T_CA):
    foo = scio.loadmat(path + 'HSI/' + scene + '.mat')
    truth = torch.from_numpy(foo['img']).float().to(args.device)
    
    foo = scio.loadmat(path + 'RGB/AMDC/' + scene + '.mat')
    rgb = torch.from_numpy(foo['RGB']).float().to(args.device) ** 2.2
    
    cassi = shift(T_CA*truth, args.shear_step, args.shear_dim).sum(dim=-1, keepdim=True)
    return (cassi, rgb, truth)

def load_cassi(mask_path):
    mask_real = torch.from_numpy(scio.loadmat(mask_path)['mask']).float().to(args.device)
    T_CA = mask_real.unsqueeze(-1).expand([-1, -1, args.nbands])
    return T_CA

if __name__ == "__main__":
    import scipy.io as scio
    from utils.common import calculate_psnr, calculate_sam, calculate_ssim
    import time

    args = parse()
    coded_aperture_mask = load_cassi(args.dataset_dir + args.mask_path)

    dx_list = torch.tensor([0.5, 0.25, 0, -0.25, -0.5])
    dy_list = torch.tensor([0.5, 0.25, 0, -0.25, -0.5])
    Dx_list, Dy_list = torch.meshgrid(dx_list, dy_list, indexing="ij")
    
    for dx, dy in zip(Dx_list.flatten(), Dy_list.flatten()):
        scene_idx = ['scene01', 'scene02', 'scene03', 'scene04', 'scene05', 'scene06', 'scene07', 'scene08', 'scene09', 'scene10']
        nSamples = len(scene_idx)
        preds = torch.zeros(nSamples, args.height, args.width, args.nbands)
        truths = torch.zeros(nSamples, args.height, args.width, args.nbands)
        Metrics = torch.zeros(nSamples, 4)
        print("|Scene|PSNR|SSIM|SAM|Time|")
        print("|----|----|----|----|----|")
        for i in range(nSamples):
            (cassi, rgb, truth) = load_data(args.dataset_dir, scene_idx[i], coded_aperture_mask)
            
            rgb = rgb_pixel_shift(rgb, dx=dx, dy=dy)
            tic = time.time()
            X_star = pipeline(rgb, cassi, coded_aperture_mask, args).clip(0, 1)
            toc = time.time()

            Metrics[i, 0] = calculate_psnr(truth, X_star)
            Metrics[i, 1] = calculate_ssim(truth, X_star)
            Metrics[i, 2] = calculate_sam(truth, X_star)
            Metrics[i, 3] = toc - tic

            preds[i] = X_star.cpu()
            truths[i] = truth.cpu()
            print(f"|{scene_idx[i]}|{Metrics[i, 0]:.3f}|{Metrics[i, 1]:.4f}|{Metrics[i, 2]:.4f}|{Metrics[i, 3]:.4f}|")
        print("|AVG|{:.3f}|{:.4f}|{:.4f}|{:.4f}|".format(Metrics[:, 0].mean(), Metrics[:, 1].mean(), Metrics[:, 2].mean(), Metrics[:, 3].mean()))
