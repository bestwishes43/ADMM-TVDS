import torch
from .operators import shift, shift_back, oslash, grad, div

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
        grad_norm = grad_x.norm(p=2, dim=0, keepdim=True)
        p = (p + tau*grad_x) / (1 + tau*grad_norm)
    divp = div(p)
    x = z_tmp + mu*divp
    return x, -divp

def compute_subgradient(X:torch.Tensor, mu, rho, K, n_iter=20, G_init=None) -> torch.Tensor:
    G_ref = torch.zeros_like(X) if G_init is None else G_init
    for _ in range(n_iter):
        _, G_ref = fixed_point_iteration(X, G_ref, mu/rho, K)
    return G_ref

############################################################################
# # Algorithm II : ADMM-TVDS for DC-CASSI
############################################################################
def pipeline(aux_observation:torch.Tensor, cassi:torch.Tensor, T_CA:torch.Tensor, args):
    Phi = lambda x: shift(T_CA*x, args.shear_step, args.shear_dim).sum(dim=-1, keepdim=True)
    PhiT = lambda y: T_CA*shift_back(y.expand([-1, -1, args.nbands]), args.shear_step, args.shear_dim)
    Lambda = Phi(T_CA)
    rank = aux_observation.shape[-1]
    H, W, C = args.height, args.width, args.nbands
    Weight, _, _ = torch.svd_lowrank(aux_observation.reshape(-1, aux_observation.shape[-1]).float(), q=rank)
    Weight = Weight.to(cassi.dtype)
    w_kron_I = torch.kron(Weight, torch.eye(args.nbands, device=cassi.device, dtype=cassi.dtype)).T # [rank*C, HWC]
    w_kron_I_tensor = w_kron_I.reshape(rank*C, H, W, C) # [rank*C, H, W, C]
    A = shift(T_CA[None]*w_kron_I_tensor, args.shear_step, args.shear_dim + 1).sum(dim=-1).reshape(rank*C, -1).T # [H(W+s(C-1)), rank*C]
    lambda_ = (Lambda + args.alpha).reshape(-1, 1)**0.5
    A = oslash(A, lambda_)
    ret = torch.linalg.lstsq(A.float(), oslash(cassi, (Lambda + args.alpha)**0.5).reshape(-1, 1).float()) 
    spectral_bias = ret.solution.to(cassi.dtype)
    X_ref = (Weight @ spectral_bias.reshape(rank, args.nbands)).reshape(H, W, C)
    X = X_ref + PhiT(oslash(cassi - Phi(X_ref), Lambda + args.alpha))
    # Iterative Stage
    U = torch.zeros_like(X)
    for stage in range(args.n_stage):
        rho = args.rho*(args.beta**(stage))
        Phi_dagger = lambda y: PhiT(torch.div(y, Lambda + rho))

        X_ref = (X_ref - X_ref.mean(dim=[0, 1], keepdim=True))/X_ref.std(dim=[0, 1], keepdim=True) * X.std(dim=[0, 1], keepdim=True)
        G_ref = compute_subgradient(X_ref, args.mu, rho, args.K, 2*args.N)
        for _ in range(args.N):
            Z, _ = fixed_point_iteration(X + U, G_ref, args.mu/rho, args.K)
            U = U + X - Z
            X_tmp = Z - U
            X = X_tmp + Phi_dagger(cassi - Phi(X_tmp))
        
        X_ref = (Weight @ (Weight.T @ (X+U).reshape(-1, args.nbands))).reshape_as(X)
    return X