import torch
import torch.nn.functional as F

############################################################################
# # Operators
############################################################################
def grad(u:torch.Tensor, ndim=None):
    ndim = u.ndim if ndim is None else ndim
    assert ndim <= u.ndim
    slice_all = [0, slice(None, -1)]
    grad_u = torch.zeros([ndim, ] + list(u.shape), 
                        dtype=u.dtype, device=u.device)
    for d in range(ndim):
        grad_u[tuple(slice_all)] = torch.diff(u, dim=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return grad_u

def div(p):
    res = torch.zeros(p.shape[1:], device=p.device)
    for d in range(p.shape[0]):
        res += p[d] - torch.roll(p[d], 1, d)
    return res

def Diffusivity(x, dims=[0]):
    D = oslash(1.0, x.norm(p='fro', dim=dims, keepdim=True))
    return D

def oslash(a, b:torch.Tensor):
    mask = b != 0
    return torch.where(mask, a / b.clamp(1e-12), 0*a)

def shift(inputs, step, dim=1):
    """
        inputs : [..., H, W, C]
    """
    nC = inputs.shape[-1]
    n_order = len(inputs.shape)

    # 计算填充大小, 仅在指定维度后方填充 (nC-1)*step
    padsize = [0]*n_order*2
    padsize[n_order*2 - 2*dim - 1] = (nC - 1)*step 

    output = F.pad(inputs, padsize, 'constant', 0)
    for i in range(1, nC):
        output[:, :, i] = torch.roll(output[:, :, i], step*i, dim)
    return output

def shift_back(inputs, step, dim=1):
    nC = inputs.shape[-1]
    n_order = len(inputs.shape)
    
    output_shape = list(inputs.shape)
    output_shape[dim] = output_shape[dim] - step*(nC - 1)
    
    output = torch.zeros(output_shape, dtype=inputs.dtype, device=inputs.device)
    slices = [slice(None)] * n_order
    for i in range(nC):
        slices[-1] = i # type: ignore
        slices[dim] = slice(step*i, step*i + output_shape[dim])
        output[:, :, i] = inputs[tuple(slices)]
    return output

############################################################################
# # Algorithm I : Fixed Point Interation for TVDS-Fusion
############################################################################
def fixed_point_iteration(z:torch.Tensor, g:torch.Tensor, mu, n_iter=10, ndim=2):
    """
    Fixed Point Iteration for TVDS-Fusion
    """
    tau = 1/(8*mu)
    z_tmp = z + mu * g

    p = 0*grad(z_tmp, ndim)
    x = z_tmp
    for _ in range(n_iter):
        #  update p
        grad_x = grad(x, ndim)
        denorm = 1 + tau*grad_x.norm(p='fro', dim=0, keepdim=True)
        p = (p + tau*grad_x)/denorm
        # update x
        x = z_tmp + mu*div(p)
    return x, -div(p)


############################################################################
# # Algorithm II : ADMM-TVDS
############################################################################
def ADMM_TVDS(optical_oprators, Y:torch.Tensor, G_ref:torch.Tensor, mu, rho, n_iter, init_set, args):
    """
    ADMM-TVDS for DC-CASSI
    """
    (Phi, PhiT, Lambda) = optical_oprators
    Phi_dagger = lambda y: PhiT(oslash(y, Lambda + rho))

    X, U = (Phi_dagger(Y), 0) if init_set is None else init_set
    for n in tqdm.trange(n_iter):
        Z, _ = fixed_point_iteration(X + U, G_ref, 2*mu/rho, args.K)
        U = U + X - Z
        X_tmp = Z - U
        X = X_tmp + Phi_dagger(Y-Phi(X_tmp))
    return X, U

############################################################################
# # ADMM-TVDS for DC-CASSI : PipeLine
############################################################################
def ADMM_TVDS_for_DC_CASSI(rgb:torch.Tensor, cassi:torch.Tensor, T_CA:torch.Tensor, args):
    Phi = lambda x: shift(T_CA*x, args.shear_step, args.shear_dim).sum(dim=-1, keepdim=True)
    PhiT = lambda y: T_CA*shift_back(y.expand([-1, -1, args.nbands]), args.shear_step, args.shear_dim)
    Lambda = Phi(T_CA)
    optical_oprators = (Phi, PhiT, Lambda)
    ## Initial Stage
    bgr = rgb.flip(-1)
    
    # rgb branch
    # 1. Linear Interpolation
    X_rgb = F.interpolate(bgr, size=[args.nbands], mode='linear')
    # 2. Energy Match
    Y_rgb_hat = shift(X_rgb*T_CA, step=args.shear_step, dim=args.shear_dim)
    Phi_b = Y_rgb_hat.reshape(-1, args.nbands)
    alpha = (torch.linalg.pinv(Phi_b.T @ Phi_b) @ torch.sum(Y_rgb_hat*cassi, dim=[0, 1]))
    X_ref = X_rgb*alpha
    # 3. Compute Subgradient
    G_ref = compute_subgradient(X_ref, args.mu, args.rho, 30)

    # cassi branch
    X = X_ref + PhiT(oslash(cassi - Phi(X_ref), Lambda))
    U = 0*X

    ## loop
    W, _, _ = torch.svd_lowrank(bgr.reshape(args.height*args.width, 3), q=3)
    for stage in range(args.n_stage):
        X, U = ADMM_TVDS(optical_oprators, cassi, G_ref, args.mu*(1.2**(-stage)), args.rho, args.N, (X, U), args)
        print(f"Stage [{stage+1}/{args.n_stage}]: PSNR {calculate_psnr(truth, X)} dB, SAM {calculate_sam(truth, X)}.")
        # Update reference image
        X_ref = (W @ (W.T @ X.reshape(args.height*args.width, args.nbands))).reshape_as(X)
        G_ref = compute_subgradient(X_ref, args.mu*(1.2**(-stage-1)), args.rho)
    return X

def Sub_Theo(X:torch.Tensor, ndim=2):
    # If use this，you need use 
    grad_X = grad(X, ndim=ndim)
    return -div(grad_X*Diffusivity(grad_X))

def compute_subgradient(X:torch.Tensor, mu, rho, iter=20):
    # Employ the fixed-point iteration method to calculate the subgradient, 
    # aiming to mitigate the impact on the convergence speed when tau is low.
    G_ref = 0*X
    for _ in range(iter):
        _, G_ref = fixed_point_iteration(X, G_ref, 2*mu/rho, args.K)
    return G_ref

def parse():
    import argparse
    #-----------------------Opti. Configuration -----------------------#
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', default="./ARAD_1K_demosaic/")
    parser.add_argument('--mask_path', default="mask_256.mat")
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--shear_step', default=2, type=int, help="Shearing transformation step size.")
    parser.add_argument('--shear_dim', default=1, type=int, help="The dimension sheared.")
    parser.add_argument('--height', default=256, type=int, help="")
    parser.add_argument('--width', default=256, type=int, help="")
    parser.add_argument('--nbands', default=31, type=int, help="")
    parser.add_argument('--K', default=30, type=int, help="")
    parser.add_argument('--N', default=10, type=int, help="")
    parser.add_argument('--n_stage', default=30, type=int, help="")
    parser.add_argument('--mu', default=0.015, type=int, help="")
    parser.add_argument('--rho', default=0.03, type=int, help="")
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
    import tqdm
    import scipy.io as scio
    import os
    from utils.viz import implay
    from utils.common import calculate_psnr, calculate_sam

    args = parse()
    optical_operators = load_cassi(args.mask_path)


    scene_idx = ['0901', '0902', '0903', '0904', '0905', '0906', '0907', '0908', '0909', '0910']
    nSamples = len(scene_idx)
    preds = torch.zeros(nSamples, args.height, args.width, args.nbands)
    truths = torch.zeros(nSamples, args.height, args.width, args.nbands)
    for i in range(nSamples):
        data_path = os.path.join(args.dataset_dir, "Valid", "ARAD_1K_"+ scene_idx[i] + ".mat")
        (cassi, rgb, truth) = load_data(data_path, optical_operators)
        X_star = ADMM_TVDS_for_DC_CASSI(rgb, cassi, optical_operators, args)
        preds[i] = X_star.cpu()
        truths[i] = truth.cpu()
    scio.savemat("recon_ARAD_1K.mat", {"pred":preds, "truth":truths})