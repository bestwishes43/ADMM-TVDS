import torch

from algorithms.operators import shift_back
from algorithms.pipeline import pipeline
from utils.common import calculate_psnr, calculate_ssim
from utils.viz import implay
from algorithms.operators import shift

def parse():
    import argparse
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--dataset_dir', default="./datasets/RealData/")

    parser.add_argument('--shear_step', default=1, type=int, help="Shearing transformation step size.")
    parser.add_argument('--shear_dim', default=0, type=int, help="The dimension sheared.")
    
    parser.add_argument('--K', default=30, type=int, help="")
    parser.add_argument('--N', default=10, type=int, help="")
    parser.add_argument('--n_stage', default=30, type=int, help="")
    parser.add_argument('--mu', default=3e1, type=float, help="")
    parser.add_argument('--rho', default=3e1, type=float, help="")
    parser.add_argument('--alpha', default=1e-1, type=float, help="")
    parser.add_argument('--beta', default=1.2, type=float, help="")
    parser.add_argument('--rank', default=None, type=int, help="")
    parser.add_argument('--use_net_init', default=True, type=bool, help="Align the spatial weight to cassi with network")
    args = parser.parse_args()
    return args

def load_data(path, scene, args):
    foo = scio.loadmat(path + scene + '.mat')
    mask_shift = torch.from_numpy(foo['Mask']).flip(0).float().to(args.device)
    
    pan = torch.from_numpy(foo['panInput']).flip(0).float().to(args.device)
    cassi = torch.from_numpy(foo['cassiInput']).flip(0).float().to(args.device)
    # The Spectral Response Function (SRF) calibrated using the impulse function method differs from the actual SRF and requires scaling correction.
    SRF = torch.from_numpy(foo['cameraSpectralResponse']).float().to(args.device)
    
    T_CA = shift_back(mask_shift, step=args.shear_step, dim=args.shear_dim)
    cassi = cassi.unsqueeze(-1)
    pan = pan.unsqueeze(-1)
    
    H, W, C = T_CA.shape
    Lambda = shift(T_CA*T_CA, args.shear_step, args.shear_dim).sum(dim=-1, keepdim=True)
    Weight = torch.svd_lowrank(pan.reshape(-1, pan.shape[-1]).float(), q=1)[0].to(cassi.dtype)
    w_kron_I = torch.kron(Weight, torch.eye(C, device=cassi.device, dtype=cassi.dtype)).T # [C, HWC]
    w_kron_I_tensor = w_kron_I.reshape(C, H, W, C) # [C, H, W, C]
    Phi_U = shift(T_CA[None]*w_kron_I_tensor, args.shear_step, args.shear_dim + 1).sum(dim=-1).reshape(C, -1).T # [H(W+s(C-1)), C]
    
    DeLambda = (Lambda + args.alpha)**0.5
    Phi_U = Phi_U / DeLambda.reshape(-1, 1)
    ret = torch.linalg.lstsq(Phi_U.float(), (cassi/DeLambda).reshape(-1, 1).float()) 
    spectral_bias = ret.solution.to(cassi.dtype)
    x_hat = (Weight @ spectral_bias.reshape(1, C)).reshape(H, W, C)
    pan_hat = (x_hat @ SRF).reshape(-1, 1)
    alpha = pan_hat.T @ pan.reshape(-1, 1) / (pan_hat.T @ pan_hat)
    
    args.height, args.width, args.nbands = H, W, C
    return (cassi*alpha, pan, T_CA, SRF)

if __name__ == "__main__":
    import scipy.io as scio
    from pathlib import Path
    args = parse()

    scenes = ['Scene01', 'Scene02']
    results = {}
    for scene in scenes:
        (cassi, pan, coded_aperture_mask, SRF) = load_data(args.dataset_dir, scene, args)
        X_star = pipeline(pan, cassi, coded_aperture_mask, args)
        print(calculate_psnr(pan, X_star @ SRF, pan.max()))
        X_star = X_star.flip(0)
        pan = pan.flip(0)
        
        results[scene] = X_star.cpu()
        implay(X_star.cpu().numpy())
    
    save_path = Path("./result/Real")
    save_path.mkdir(parents=True, exist_ok=True)
    scio.savemat("./result/Real/recon_Real.mat", results)
