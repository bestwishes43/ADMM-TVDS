import torch
from algorithms import shift
from algorithms.pipeline import pipeline

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
    parser.add_argument('--rank', default=None, type=int, help="")
    parser.add_argument('--use_net_init', default=False, type=bool, help="Align the spatial weight to cassi with network")
    args = parser.parse_args()
    return args


def load_data(path, scene, T_CA):
    foo = scio.loadmat(path + 'HSI/' + scene + '.mat')
    truth = torch.from_numpy(foo['img']).float().to(args.device)

    foo = scio.loadmat(path + 'RGB/AMDC/' + scene + '.mat')
    # sRGB to linear
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

    scene_idx = ['scene01', 'scene02', 'scene03', 'scene04', 'scene05', 'scene06', 'scene07', 'scene08', 'scene09', 'scene10']
    nSamples = len(scene_idx)
    preds = torch.zeros(nSamples, args.height, args.width, args.nbands)
    truths = torch.zeros(nSamples, args.height, args.width, args.nbands)
    Metrics = torch.zeros(nSamples, 4)
    print("|Scene|PSNR|SSIM|SAM|Time|")
    print("|----|----|----|----|----|")
    for i in range(nSamples):
        (cassi, rgb, truth) = load_data(args.dataset_dir, scene_idx[i], coded_aperture_mask)
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

    # 保存结果
    from pathlib import Path
    save_path = Path("./result/TVDS")
    save_path.mkdir(parents=True, exist_ok=True)
    scio.savemat(save_path/"recon_KAIST.mat", {"preds":preds, "truths":truths, "Metrics":Metrics})
