import torch

from algorithms.operators import shift_back
from algorithms.pipeline import pipeline

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
    args = parser.parse_args()
    return args

def load_data(path, scene, args):
    foo = scio.loadmat(path + scene + '.mat')
    mask_shift = torch.from_numpy(foo['Mask']).flip(0).float().to(args.device)
    
    pan = torch.from_numpy(foo['panInput']).flip(0).float().to(args.device)
    cassi = torch.from_numpy(foo['cassiInput']).flip(0).float().to(args.device)
    # The Spectral Response Function (SRF) calibrated using the impulse function method differs from the actual SRF and requires scaling correction.
    SRF = torch.from_numpy(foo['cameraSpectralResponse']).float().to(args.device) * 1.2 
    
    T_CA = shift_back(mask_shift, step=args.shear_step, dim=args.shear_dim)
    cassi = cassi.unsqueeze(-1)
    pan = pan.unsqueeze(-1)
    
    (args.height, args.width, args.nbands) = T_CA.shape
    return (cassi, pan, T_CA, SRF)

if __name__ == "__main__":
    import scipy.io as scio
    args = parse()

    scenes = ['Scene01', 'Scene02']
    results = {}
    for scene in scenes:
        (cassi, pan, coded_aperture_mask, SRF) = load_data(args.dataset_dir, scene, args)
        X_star = pipeline(pan, cassi, coded_aperture_mask, args)
        X_star = X_star.flip(0)
        pan = pan.flip(0)
        
        results[scene] = X_star.cpu()
    
    from pathlib import Path
    save_path = Path("./result/Real")
    save_path.mkdir(parents=True, exist_ok=True)
    scio.savemat("./result/Real/recon_Real.mat", results)
