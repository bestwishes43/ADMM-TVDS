import torch
import math
import scipy.io as scio

from utils.common import calculate_psnr
from algorithms.operators import shift, shift_back

if __name__ == "__main__":
    shear_step = 1
    shear_dim = 0
    alpha_reg = 1e-2  
    
    for noise_std in [1, 0.3, 0.1, 0.03, 0.01]:
        noise_std_cassi = noise_std_pan = noise_std
        # =================================
        
        repeat_num = 1000
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 存储统计结果
        psnr_cassi_list = []
        psnr_pan_list = []
        nmse_alpha_list = []
        
        for scene in ["scene01", "scene02"]:
            foo = scio.loadmat(f"./datasets/RealData/{scene}.mat")
            
            coded_aperture = shift_back(
                torch.from_numpy(foo['Mask']).double().to(device), 
                shear_step, shear_dim
            )
            pan_srf = torch.from_numpy(foo['cameraSpectralResponse']).double().to(device)

            H, W = foo['panInput'].shape[:2]
            C = pan_srf.shape[0]
            
            for i in range(repeat_num):
                # ============================ Generate Random Scene ============================ #
                gt = torch.rand(H, W, C, device=device, dtype=coded_aperture.dtype)
                
                scale_gt = torch.rand(1, device=device, dtype=coded_aperture.dtype) * 9.9 + 0.1
                split_ratio_gt = 1 / (1 + scale_gt)

                with torch.no_grad():
                    cassi_clean = shift((split_ratio_gt * gt) * coded_aperture, 
                                    shear_step, shear_dim).sum(dim=-1, keepdim=True)
                    pan_clean = ((1 - split_ratio_gt) * gt) @ pan_srf
                
                cassi_noisy = cassi_clean + noise_std_cassi * torch.randn_like(cassi_clean)
                pan_noisy = pan_clean + noise_std_pan * torch.randn_like(pan_clean)
                
                psnr_cassi = calculate_psnr(cassi_clean, cassi_noisy, cassi_clean.max())
                psnr_pan = calculate_psnr(pan_clean, pan_noisy, pan_clean.max())
                psnr_cassi_list.append(psnr_cassi)
                psnr_pan_list.append(psnr_pan)
                
                # ============================ Supplementary S2: Estimate Radiometric Alignment Coefficient ============================ #
                Lambda = shift(coded_aperture * coded_aperture, shear_step, shear_dim).sum(dim=-1, keepdim=True)
                Weight = torch.svd_lowrank(pan_noisy.reshape(-1, pan_noisy.shape[-1]), q=1, niter=20)[0]
                
                w_kron_I = torch.kron(Weight, torch.eye(C, device=device)).T
                w_kron_I_tensor = w_kron_I.reshape(C, H, W, C)
                Phi_U = shift(coded_aperture[None] * w_kron_I_tensor, 
                            shear_step, shear_dim + 1).sum(dim=-1).reshape(C, -1).T
                
                DeLambda = (Lambda + alpha_reg) ** 0.5
                Phi_U_norm = Phi_U / DeLambda.reshape(-1, 1)
                
                ret = torch.linalg.lstsq(Phi_U_norm, 
                                    (cassi_noisy / DeLambda).reshape(-1, 1))
                spectral_bias = ret.solution.to(cassi_noisy.dtype)
                x_hat = (Weight @ spectral_bias.reshape(1, C)).reshape(H, W, C)
                
                pan_hat = (x_hat @ pan_srf).reshape(-1, 1)
                pan_meas = pan_noisy.reshape(-1, 1)
                scale_est = (pan_hat.T @ pan_meas) / (pan_hat.T @ pan_hat + 1e-8)
                
                # Calculate Relative Error of Estimation (NMSE)
                nmse_alpha = ((scale_gt - scale_est) ** 2) / (scale_gt ** 2 + 1e-8)
                nmse_alpha_list.append(nmse_alpha.item())
            
            avg_psnr_cassi = sum(psnr_cassi_list[-repeat_num:]) / repeat_num
            avg_psnr_pan = sum(psnr_pan_list[-repeat_num:]) / repeat_num
            avg_nmse_db = 10 * math.log10(sum(nmse_alpha_list[-repeat_num:]) / repeat_num + 1e-10)
            
            print(f"\n[{scene}] Results over {repeat_num} trials (noise_std={noise_std}):")
            print(f"  CASSI PSNR: {avg_psnr_cassi:.2f} dB (σ={noise_std_cassi})")
            print(f"  PAN   PSNR: {avg_psnr_pan:.2f} dB (σ={noise_std_pan})")
            print(f"  κ estimation NMSE: {avg_nmse_db:.2f} dB")