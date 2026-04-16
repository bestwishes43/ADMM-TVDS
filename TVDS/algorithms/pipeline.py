import torch

from .net import SimpleNet
from .operators import *

############################################################################
# # Algorithm I : Fixed Point Interation for TVDS-Fusion
############################################################################
@torch.jit.script
def fixed_point_iteration(
    u: torch.Tensor, 
    eta: float, 
    n_iter: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perform fixed point iteration for TVDS-Fusion problem.

    Args:
        u (torch.Tensor): Input tensor with shape [H, W, C] to be processed.
        eta (float): Regularization parameter controlling the trade-off between data fidelity and regularization.
        n_iter (int, optional): Number of iteration steps. Default: 10.

    Returns:
        x (torch.Tensor): Optimized tensor with shape [H, W, C] after fixed point iteration.
        -divp (torch.Tensor): Negative divergence of the dual variable.
    """
    tau_mult_u = u / (8.0 * eta)
    p_x = torch.zeros((2, ) + u.shape, dtype=u.dtype, device=u.device)
    for _ in range(n_iter):
        # Update x
        tau_mult_x = tau_mult_u + div_2d(p_x) / 8.

        # Update p_x
        grad_x = diff_2d(tau_mult_x)
        grad_norm = grad_x.norm(p=2, dim=0, keepdim=True)
        p_x.add_(grad_x).div_(1 + grad_norm)
    divp = div_2d(p_x)
    x = u + eta * divp
    return x, -divp


def compute_subgradient(X:torch.Tensor, eta, K, n_iter=20, G_init=None) -> torch.Tensor:
    G_ref = torch.zeros_like(X) if G_init is None else G_init
    for _ in range(n_iter):
        _, G_ref = fixed_point_iteration(X + eta*G_ref, eta, K)
    return G_ref

############################################################################
# # Algorithm II : ADMM-TVDS for DC-CASSI
############################################################################
def pipeline(
    aux_observation: torch.Tensor, 
    cassi: torch.Tensor, 
    T_CA: torch.Tensor, 
    args
) -> torch.Tensor:
    """
    Execute the ADMM-TVDS pipeline for hyperspectral image reconstruction.

    Args:
        aux_observation (torch.Tensor): Auxiliary data tensor with shape [H, W, 3] or [H, W, 1].
        cassi (torch.Tensor): Target CASSI data tensor with shape [H', W', 1].
        T_CA (torch.Tensor): CASSI Coded Aperture with shape [H, W, C].
        args (object): Configuration object containing:
            - device: Computation device (CPU/GPU)
            - height, width, nbands: Spatial and spectral dimensions
            - rank: Rank for low-rank decomposition
            - use_net_init: Whether to use neural network initialization
            - alpha, rho, beta, mu: Algorithm parameters
            - n_stage, N, K: Iteration counts

    Returns:
        torch.Tensor: Reconstructed hyperspectral image with shape [H, W, C].
    """
    # Define forward and adjoint operators for the CASSI system
    Phi = lambda x: shift(
        T_CA*x, args.shear_step, args.shear_dim
    ).sum(dim=-1, keepdim=True)
    
    PhiT = lambda y: T_CA * shift_back(
        y.expand([-1, -1, args.nbands]), 
        args.shear_step, 
        args.shear_dim
    )
    Lambda = Phi(T_CA)
    
    
    # ============================= Initial Stage (Section IV in the paper) ============================= #
    rank = aux_observation.shape[-1] if args.rank is None else args.rank
    
    if args.use_net_init:
        Weight = generate_spatial_weight(
            aux_observation, cassi, Phi, rank, args
        )
    else:
        Weight = torch.svd_lowrank(
            aux_observation.reshape(-1, aux_observation.shape[-1]), q=rank
        )[0]
    
    w_kron_I_tensor = torch.kron(
        Weight, torch.eye(args.nbands, device=cassi.device, dtype=cassi.dtype)
    ).T.reshape(rank*args.nbands, args.height, args.width, args.nbands) # [rank*C, H, W, C]

    A = shift(
        T_CA[None]*w_kron_I_tensor, 
        args.shear_step, 
        args.shear_dim + 1
    ).sum(dim=-1).reshape(rank*C, -1).T # [H(W+s(C-1)), rank*C]
    
    # Solve weighted least squares problem
    sqrt_Lambda_plus_alpha = (Lambda + args.alpha) ** 0.5
    A = A / sqrt_Lambda_plus_alpha.reshape(-1, 1)
    ret = torch.linalg.lstsq(A, (cassi/ sqrt_Lambda_plus_alpha).reshape(-1, 1)) 
    spectral_bias = ret.solution

    # Clean up intermediate variables to free memory
    del ret, w_kron_I_tensor, A, sqrt_Lambda_plus_alpha
    torch.cuda.empty_cache()
    
    X_ref = (Weight @ spectral_bias.reshape(rank, args.nbands)).reshape(args.height, args.width, args.nbands)
    X = X_ref + PhiT((cassi - Phi(X_ref)) / (Lambda + args.alpha))
    
    # ============================= Iterative Stage (Section V in the paper) ============================= #
    U = torch.zeros_like(X)
    for stage in range(args.n_stage):
        rho = args.rho*(args.beta**(stage))
        eta = args.mu/rho

        # Normalize reference image
        X_ref = (X_ref - X_ref.mean(dim=[0, 1], keepdim=True))/X_ref.std(dim=[0, 1], keepdim=True) * X.std(dim=[0, 1], keepdim=True)
        
        # Compute guided subgradient
        neg_divP_X_ref = torch.zeros_like(X_ref)
        for _ in range(2*args.N):
            _, neg_divP_X_ref = fixed_point_iteration(X_ref + eta*neg_divP_X_ref, eta, args.K)
        
        # ADMM iteration
        for _ in range(args.N):
            Z_temp = X + U + eta*neg_divP_X_ref
            Z, _ = fixed_point_iteration(Z_temp, eta, args.K)

            U = U + X - Z
            
            X_temp = Z - U
            X = X_temp + PhiT((cassi - Phi(X_temp)) / (Lambda + rho))
        
        # Update reference image
        X_ref = (Weight @ (Weight.T @ (X+U).reshape(-1, args.nbands))).reshape_as(X)
    return X


############################################################################
# # Supplementary 1 : Spatial Alignment
############################################################################
def generate_spatial_weight(
    auxiliary_data: torch.Tensor,
    cassi: torch.Tensor,
    Phi: callable,
    rank: int,
    args: object,
    num_epochs: int = 1000,
    learning_rate: float = 1e-3
) -> torch.Tensor:
    """
    Generate a spatial weight matrix by optimizing an initial estimate via neural network.

    Args:
        auxiliary_data (torch.Tensor): Auxiliary data tensor with shape [H, W, C].
        cassi (torch.Tensor): Target CASSI data tensor with shape [H', W', 1].
        transformation_matrix (torch.Tensor): Transformation matrix T_CA with shape [H', W', C].
        args (object): Object containing device information and shear parameters.
        num_epochs (int, optional): Number of training epochs. Default: 1000.
        learning_rate (float, optional): Learning rate for the optimizer. Default: 1e-3.

    Returns:
        torch.Tensor: Weight matrix after low-rank decomposition with shape [rank, C].
    """
    device = args.device
    aux_tensor = auxiliary_data.permute(2, 0, 1).unsqueeze(0).to(device)


    im_net = SimpleNet(
        input_channels=aux_tensor.shape[1],
        output_channels=args.nbands
    ).to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(im_net.parameters(), lr=learning_rate)
    
    
    best_loss = float('inf')
    best_x_init = None

    for epoch in range(num_epochs):
        x_init_tensor = im_net(aux_tensor)
        x_init = x_init_tensor[0].permute(1, 2, 0)
        
        loss = loss_fn(Phi(x_init), cassi)
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_x_init = x_init.detach()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    Weight = torch.svd_lowrank(
        best_x_init.reshape(-1, args.nbands), q=rank
    )[0]
    return Weight