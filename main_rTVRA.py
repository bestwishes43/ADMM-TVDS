import torch
from algorithms import shift_back, pipeline

def parse():
    import argparse
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--dataset_dir', default="./datasets/rTVRA/")

    parser.add_argument('--shear_step', default=1, type=int, help="Shearing transformation step size.")
    parser.add_argument('--shear_dim', default=0, type=int, help="The dimension sheared.")
    
    parser.add_argument('--K', default=30, type=int, help="")
    parser.add_argument('--N', default=10, type=int, help="")
    parser.add_argument('--n_stage', default=15, type=int, help="")
    parser.add_argument('--mu', default=5, type=int, help="")
    parser.add_argument('--rho', default=10, type=int, help="") 
    parser.add_argument('--beta', default=1.2, type=int, help="")
    args = parser.parse_args()
    return args

def load_data(path, scene, args):
    foo = scio.loadmat(path + scene + '.mat')
    mask_shift = torch.from_numpy(foo['Mask']).flip(0).float().to(args.device)
    
    pan = torch.from_numpy(foo['panInput']).flip(0).float().to(args.device)
    cassi = torch.from_numpy(foo['cassiInput']).flip(0).float().to(args.device)
    cassi = cassi/cassi.max()
    # cameraSpectralResponse = torch.from_numpy(foo['cameraSpectralResponse']).float().to(args.device)
    
    T_CA = shift_back(mask_shift, step=args.shear_step, dim=args.shear_dim)
    cassi = cassi.unsqueeze(-1)
    pan = pan.unsqueeze(-1)

    (args.height, args.width, args.nbands) = T_CA.shape
    return (cassi, pan, T_CA)

if __name__ == "__main__":
    import scipy.io as scio
    from utils.viz import implay
    args = parse()

    scenes = ['Scene01', 'Scene02']
    results = {}
    for scene in scenes:
        (cassi, pan, optical_operators) = load_data(args.dataset_dir, scene, args)
        X_star = pipeline(pan, cassi, optical_operators, args)
        if scene == 'Scene01':
            X_star = X_star.flip(0)
        implay(X_star.cpu()**(1/2.2), vmax=1, vmin=0) # Gamma correction is used for better visualization.
        results[scene] = X_star.cpu()
    scio.savemat("./results/recon_rTVRA.mat", results)