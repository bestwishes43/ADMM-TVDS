import numpy as np
import torch

def check_and_trans(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return x

def im2uint8(x:np.ndarray):
    # assert x.max() <= 255 and x.min() >= 0, "input should be limited in [0, 255] for int and [0, 1] for float."
    if np.issubdtype(x.dtype, np.floating):
        x = x.clip(0.0, 1.0)*255
    return x.astype(np.uint8)



def calculate_psnr(
    ground_truth: torch.Tensor, 
    predicted: torch.Tensor, 
    value_range: int = 255,
    per_band: bool = False
) -> torch.Tensor:
    """Calculates Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        ground_truth: Ground truth image tensor [H, W, C]
        predicted: Reconstructed image tensor [H, W, C]
        value_range: Maximum possible pixel value
        per_band: Return PSNR per band if True
        
    Returns:
        PSNR value(s) in dB
    """
    ground_truth = (ground_truth.clip(0, 1) * value_range).round()
    predicted = (predicted.clip(0, 1) * value_range).round()
    
    mse_per_band = torch.mean((predicted - ground_truth)**2, dim=(0, 1))
    psnr_per_band = 10 * torch.log10(value_range**2 / mse_per_band.clip(torch.finfo(float).eps))
    
    return psnr_per_band if per_band else torch.mean(psnr_per_band)


def calculate_sam(
    ground_truth: torch.Tensor, 
    predicted: torch.Tensor
) -> torch.Tensor:
    """Calculates Spectral Angle Mapper (SAM) in degrees.
    
    Args:
        ground_truth: Ground truth image tensor [H, W, C]
        predicted: Reconstructed image tensor [H, W, C]
        
    Returns:
        Mean SAM value in degrees
    """
    eps = 1e-6
    dot_product = torch.sum(ground_truth * predicted, dim=-1)
    norm_product = torch.sqrt(
        torch.sum(ground_truth**2, dim=-1) * 
        torch.sum(predicted**2, dim=-1)
    ).clip(eps)
    
    sam_per_pixel = torch.acos(dot_product / norm_product)
    return torch.mean(sam_per_pixel * 180 / torch.pi)