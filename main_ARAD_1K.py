import torch
from algorithms import shift, pipeline

def parse():
    import argparse
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--dataset_dir', default="./datasets/ARAD_1K_demosaic/")
    parser.add_argument('--height', default=256, type=int, help="")
    parser.add_argument('--width', default=256, type=int, help="")
    parser.add_argument('--nbands', default=31, type=int, help="")

    parser.add_argument('--mask_path', default="mask_256.mat")
    parser.add_argument('--shear_step', default=2, type=int, help="Shearing transformation step size.")
    parser.add_argument('--shear_dim', default=1, type=int, help="The dimension sheared.")

    parser.add_argument('--K', default=30, type=int, help="")
    parser.add_argument('--N', default=10, type=int, help="")
    parser.add_argument('--n_stage', default=30, type=int, help="")
    parser.add_argument('--mu', default=0.015, type=int, help="")
    parser.add_argument('--rho', default=0.03, type=int, help="")
    parser.add_argument('--beta', default=1.2, type=int, help="")
    args = parser.parse_args()
    return args

def load_data(path, T_CA):
    foo = scio.loadmat(path)
    hsi = torch.from_numpy(foo['truth']).float().to(args.device)
    rgb = torch.from_numpy(foo['rgb']).float().to(args.device)

    truth = hsi[113:369, 128:384, :] # As MLP-AMDC dose.
    cassi = shift(T_CA*truth, args.shear_step, args.shear_dim).sum(dim=-1, keepdim=True)
    rgb = rgb[113:369, 128:384, :]
    return (cassi, rgb, truth)

def load_cassi(mask_path):
    mask_real = torch.from_numpy(scio.loadmat(mask_path)['mask']).float().to(args.device)
    T_CA = mask_real.unsqueeze(-1).expand([-1, -1, args.nbands])
    return T_CA

if __name__ == "__main__":
    import scipy.io as scio
    import os
    from utils.common import calculate_psnr, calculate_sam

    args = parse()
    optical_operators = load_cassi(args.dataset_dir + args.mask_path)


    scene_idx = ['0901', '0902', '0903', '0904', '0905', '0906', '0907', '0908', '0909', '0910']
    nSamples = len(scene_idx)
    preds = torch.zeros(nSamples, args.height, args.width, args.nbands)
    truths = torch.zeros(nSamples, args.height, args.width, args.nbands)
    for i in range(nSamples):
        data_path = os.path.join(args.dataset_dir, "Valid", "ARAD_1K_"+ scene_idx[i] + ".mat")
        (cassi, rgb, truth) = load_data(data_path, optical_operators)
        X_star = pipeline(rgb, cassi, optical_operators, args)
        print(f"{scene_idx[i]}: PSNR {calculate_psnr(truth, X_star)} dB, SAM {calculate_sam(truth, X_star)}.")
        preds[i] = X_star.cpu()
        truths[i] = truth.cpu()
    scio.savemat("./results/recon_ARAD_1K.mat", {"pred":preds, "truth":truths})