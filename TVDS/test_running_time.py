import torch
from algorithms import shift
from algorithms.pipeline import pipeline
import scipy.io as scio
import time

def parse():
    import argparse
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    
    parser.add_argument('--dataset_dir', default="./datasets/cave_test/")
    parser.add_argument('--height', default=256, type=int, help="")
    parser.add_argument('--width', default=256, type=int, help="")
    parser.add_argument('--nbands', default=28, type=int, help="")

    parser.add_argument('--mask_path', default="mask_test.mat")
    parser.add_argument('--shear_step', default=2, type=int, help="Shearing transformation step size.")
    parser.add_argument('--shear_dim', default=1, type=int, help="The dimension sheared.")
    
    parser.add_argument('--K', default=30, type=int, help="Fixed Point Iteration for solving TV problem")
    parser.add_argument('--N', default=10, type=int, help="ADMM Iteration")
    parser.add_argument('--n_stage', default=30, type=int, help="")
    parser.add_argument('--mu', default=3e-2, type=float, help="TVDS regularization parameter")
    parser.add_argument('--rho', default=3e-2, type=float, help="ADMM penalty parameter")
    parser.add_argument('--alpha', default=1e-1, type=float, help="")
    parser.add_argument('--beta', default=1.2, type=float, help="rho decay parameter")
    parser.add_argument('--use_net_init', default=False, type=bool, help="use net init")
    args = parser.parse_args()
    return args

def load_cassi(mask_path, args):
    mask_real = torch.from_numpy(scio.loadmat(mask_path)['CASSI']).float().to(args.device)
    T_CA = mask_real.unsqueeze(-1).expand([-1, -1, args.nbands])
    return T_CA

def warmup_pipeline(rgb, cassi, mask, args):
    with torch.no_grad():  
        _ = pipeline(rgb, cassi, mask, args).clamp(0, 1)
    if args.device == "cuda":
        torch.cuda.synchronize()

def measure_time_pipeline(rgb, cassi, mask, args, n_runs=5):
    times = []
    with torch.no_grad():  
        if args.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            for _ in range(n_runs):
                
                torch.cuda.synchronize()
                start_event.record()
                _ = pipeline(rgb, cassi, mask, args).clamp(0, 1)
                
                end_event.record()
                torch.cuda.synchronize()
                
                elapsed = start_event.elapsed_time(end_event) / 1000.0
                times.append(elapsed)
        else:
            for _ in range(n_runs):
                tic = time.time()
                X_star = pipeline(rgb, cassi, mask, args).clamp(0, 1)
                toc = time.time()
                times.append(toc - tic)
    
    times = sorted(times)
    avg_time = sum(times[1:-1]) / (len(times)-2) if len(times) > 2 else sum(times)/len(times)
    return avg_time

if __name__ == "__main__":
    args = parse()
    
    height = list(range(16, 1024+1, 16))
    width = [256]*len(height)
    nSamples = len(height)
    print(nSamples)
    
    for i in range(nSamples):
        args.height = height[i]
        args.width = width[i]
        
        coded_aperture_mask = torch.randn(args.height, args.width, device=args.device).signbit().float().unsqueeze(-1).expand([-1, -1, args.nbands])
        mask_curr = coded_aperture_mask.contiguous()  
        
        truth = torch.randn(args.height, args.width, args.nbands, device=args.device, dtype=torch.float32)
        rgb = truth[:,:,:3].contiguous()
        cassi = shift(mask_curr * truth, args.shear_step, args.shear_dim).sum(dim=-1, keepdim=True).contiguous()
        
        warmup_pipeline(rgb, cassi, mask_curr, args)
        
        avg_time = measure_time_pipeline(rgb, cassi, mask_curr, args, n_runs=5)
        
        data_size_million = args.nbands * args.height * args.width / 1e6
        print(f"{data_size_million:.2f}, {avg_time:.4f}")