import torch
from algorithms.pipeline import pipeline

def parse():
    import argparse
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])

    parser.add_argument('--dataset_dir', default="./datasets/ARAD-simu/")
    parser.add_argument('--height', default=256, type=int, help="")
    parser.add_argument('--width', default=256, type=int, help="")
    parser.add_argument('--nbands', default=31, type=int, help="")

    parser.add_argument('--mask_path', default="mask.mat")
    parser.add_argument('--shear_step', default=2, type=int, help="Shearing transformation step size.")
    parser.add_argument('--shear_dim', default=1, type=int, help="The dimension sheared.")

    parser.add_argument('--K', default=30, type=int, help="")
    parser.add_argument('--N', default=10, type=int, help="")
    parser.add_argument('--n_stage', default=30, type=int, help="")
    parser.add_argument('--mu', default=3e1, type=float, help="")
    parser.add_argument('--rho', default=3e1, type=float, help="")
    parser.add_argument('--alpha', default=1e-1, type=float, help="")
    parser.add_argument('--beta', default=1.2, type=float, help="")
    args = parser.parse_args()
    return args

def load_data(path, T_CA):
    foo = scio.loadmat(path)
    truth = torch.from_numpy(foo['truth']).float().to(args.device)
    rgb = torch.from_numpy(foo['rgb']).float().to(args.device)
    cassi = torch.from_numpy(foo['cassi']).float().to(args.device).unsqueeze(-1)
    return (cassi, rgb, truth)

def load_cassi(mask_path):
    mask_real = torch.from_numpy(scio.loadmat(mask_path)['mask']).float().to(args.device)
    T_CA = mask_real.unsqueeze(-1).expand([-1, -1, args.nbands])
    return T_CA

if __name__ == "__main__":
    import scipy.io as scio
    import os
    from utils.common import calculate_psnr, calculate_sam, calculate_ssim
    import time
    from pathlib import Path
    args = parse()
    coded_aperture_mask = load_cassi(args.dataset_dir + args.mask_path)
    
    scene_idx = ['0901', '0902', '0903', '0904', '0905', '0906', '0907', '0908', '0909', '0910']

    for case_idx, test_case in enumerate(["Poor", "Medium", "Good"]):
        for split in ["split0.3", "split0.5", "split0.7"]:
            nSamples = len(scene_idx)
            preds = torch.zeros(nSamples, args.height, args.width, args.nbands)
            truths = torch.zeros(nSamples, args.height, args.width, args.nbands)
            Metrics = torch.zeros(nSamples, 4)
            print(f"\nReconstruction on {test_case}-{split}:")
            print("|Scene|PSNR|SSIM|SAM|Time|")
            print("|----|----|----|----|----|")
            for i in range(nSamples):
                data_path = os.path.join(args.dataset_dir, test_case, split, "ARAD_1K_"+ scene_idx[i] + ".mat")
                (cassi, rgb, truth) = load_data(data_path, coded_aperture_mask)
                tic = time.time()
                X_star = pipeline(rgb, cassi, coded_aperture_mask, args)
                toc = time.time()
                Metrics[i, 0] = calculate_psnr(truth, X_star)
                Metrics[i, 1] = calculate_ssim(truth, X_star)
                Metrics[i, 2] = calculate_sam(truth, X_star)
                Metrics[i, 3] = toc - tic
                print(f"|{scene_idx[i]}|{Metrics[i, 0]:.3f}|{Metrics[i, 1]:.4f}|{Metrics[i, 2]:.4f}|{Metrics[i, 3]:.4f}|")
                preds[i] = X_star.cpu()
                truths[i] = truth.cpu()
            print("|AVG|{:.3f}|{:.4f}|{:.4f}|{:.4f}|".format(Metrics[:, 0].mean(), Metrics[:, 1].mean(), Metrics[:, 2].mean(), Metrics[:, 3].mean()))
            # 保存结果
            save_path = Path("./result/TVDS/ARAD")
            save_path.mkdir(parents=True, exist_ok=True)
            scio.savemat(save_path/("recon_"+test_case+"_"+split+".mat"), {"preds":preds, "truths":truths, "Metrics":Metrics})
    