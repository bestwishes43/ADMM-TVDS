import torch
import torch.nn.functional as F
import tqdm
from .operators import *

__all__=["pipeline"]

############################################################################
# # Algorithm I : Fixed Point Interation for TVDS-Fusion
############################################################################
def fixed_point_iteration(z:torch.Tensor, g:torch.Tensor, mu:float, n_iter:int=10, ndim:int=2):
    """ Fixed Point Iteration for TVDS-Fusion
    Args:
        z (torch.Tensor): Input tensor.
        g (torch.Tensor): Gradient tensor.
        mu (float): Regularization parameter.
        n_iter (int): Number of iterations. Defaults to 10.
        ndim (int): Number of dimensions. Defaults to 2.
    Returns:    
        torch.Tensor: Updated tensor after fixed point iteration.
        torch.Tensor: Divergence of the gradient tensor.
    """
    tau = 1/(8*mu)
    z_tmp = z + mu * g
    p = torch.zeros_like(grad(z_tmp, ndim))
    for _ in range(n_iter):
        grad_x = grad(z_tmp + mu*div(p), ndim)
        grad_norm = grad_x.norm(p='fro', dim=0, keepdim=True)
        p = (p + tau*grad_x) / (1 + tau*grad_norm)
    divp = div(p)
    x = (z_tmp + mu*divp).clamp(0, 1)
    return x, -divp

def compute_subgradient(X:torch.Tensor, mu, rho, K, n_iter=20, G_init=None):
    # To minimize the effect on convergence speed at low tau values, 
    # the fixed-point iteration method is employed to calculate the subgradient.
    G_ref = torch.zeros_like(X) if G_init is None else G_init
    mu_rho = 2*mu/rho
    for _ in range(n_iter):
        _, G_ref = fixed_point_iteration(X, G_ref, mu_rho, K)
    return G_ref

############################################################################
# # Algorithm II : ADMM-TVDS
############################################################################
def ADMM_TVDS(optical_oprators, Y:torch.Tensor, G_ref:torch.Tensor, mu, rho, n_iter, init_set, args):
    """
    ADMM-TVDS for DC-CASSI
    """
    (Phi, PhiT, Lambda) = optical_oprators
    Phi_dagger = lambda y: PhiT(torch.div(y, Lambda + rho))

    if init_set is None:
        X = Phi_dagger(Y)
        U = 0
    else:
        X, U = init_set
    for n in range(n_iter):
        Z, _ = fixed_point_iteration(X + U, G_ref, 2*mu/rho, args.K)
        U = U + X - Z
        X_tmp = Z - U
        X = (X_tmp + Phi_dagger(Y - Phi(X_tmp))).clamp(0, 1)
    return X, U

############################################################################
# # ADMM-TVDS for DC-CASSI : PipeLine
############################################################################
def pipeline(rgb:torch.Tensor, cassi:torch.Tensor, T_CA:torch.Tensor, args):
    Phi = lambda x: shift(T_CA*x, args.shear_step, args.shear_dim).sum(dim=-1, keepdim=True)
    PhiT = lambda y: T_CA*shift_back(y.expand([-1, -1, args.nbands]), args.shear_step, args.shear_dim)
    Lambda = Phi(T_CA)
    optical_oprators = (Phi, PhiT, Lambda)
    # Initial Stage
    ## RGB branch
    bgr = rgb.flip(-1)
    ### 1. Linear Interpolation
    X_rgb = F.interpolate(bgr, size=[args.nbands], mode='linear')
    ### 2. Energy Match
    Y_rgb_hat = shift(X_rgb*T_CA, step=args.shear_step, dim=args.shear_dim)
    Phi_b = Y_rgb_hat.reshape(-1, args.nbands)
    alpha = (torch.linalg.pinv(Phi_b.T @ Phi_b) @ torch.sum(Y_rgb_hat*cassi, dim=[0, 1]))
    X_ref = X_rgb*alpha
    ### 3. Compute Initial Subgradient
    G_ref = compute_subgradient(X_ref, args.mu, args.rho, args.K, 60)

    ## CASSI branch
    X = X_ref + PhiT(oslash(cassi - Phi(X_ref), Lambda))
    
    # Iterative Stage
    U = 0
    W, _, _ = torch.svd_lowrank(bgr.reshape(-1, bgr.shape[-1]), q=bgr.shape[-1])
    for stage in tqdm.trange(args.n_stage):
        ## ADMM-TVDS
        X, U = ADMM_TVDS(optical_oprators, cassi, G_ref, args.mu*(args.beta**(-stage)), args.rho, args.N, (X, U/args.beta), args)
        if stage == args.n_stage-1 : continue
        ## LRDS fusion
        X_ref = (W @ (W.T @ X.reshape(args.height*args.width, args.nbands))).reshape_as(X)
        ## Compute Subgradient
        G_ref = compute_subgradient(X_ref, args.mu*(args.beta**(-stage-1)), args.rho, args.K, n_iter=20)
    return X


