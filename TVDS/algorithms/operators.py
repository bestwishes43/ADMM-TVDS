import torch
import torch.nn.functional as F

@torch.jit.script
def diff_2d(u:torch.Tensor) -> torch.Tensor:
    """
    u:[H, W, C]
    返回：[2, H, W, C] 的差分张量
    """
    diff0 = torch.cat([u[1:, :] - u[:-1, :], torch.zeros_like(u[-1:, :])], dim=0)
    diff1 = torch.cat([u[:, 1:] - u[:, :-1], torch.zeros_like(u[:, -1:])], dim=1)
    diff_u = torch.stack([diff0, diff1], dim=0)
    return diff_u

@torch.jit.script
def div_2d(p:torch.Tensor) -> torch.Tensor:
    """
    p:[2, H, W, C]
    返回：[H, W, C] 的散度张量
    """
    p0, p1 = p[0], p[1]
    p0_rolled = torch.cat([p0[-1:, :], p0[:-1, :]], dim=0)
    p1_rolled = torch.cat([p1[:, -1:], p1[:, :-1]], dim=1)
    
    divp = (p0 - p0_rolled) + (p1 - p1_rolled)
    return divp

def shift(inputs:torch.Tensor, step:int, dim:int=1):
    """ Apply shear transformation to the input tensor along the specified dimension.
    Args:
        inputs (torch.Tensor): Input tensor of shape [..., H, W, C].
        step (int): Step size for the shear transformation.
        dim (int): Dimension along which to apply the shear transformation.
    Returns:
        torch.Tensor: Sheared tensor of shape [..., H, W+step*(C-1), C].
    """
    ndim = inputs.ndim
    if dim < 0:
        dim += ndim
    assert 0 <= dim < ndim, f"dim should be in [0, {ndim-1}]."

    nC = inputs.shape[-1]
    n_order = len(inputs.shape)

    # Calculate padding size, only pad after the specified dimension (nC-1)*step
    padsize = [0]*n_order*2
    padsize[n_order*2 - 2*dim - 1] = (nC - 1)*step 

    output = F.pad(inputs, padsize, 'constant', 0.0)
    for i in range(1, nC):
        output[..., i] = torch.roll(output[..., i], step*i, dim)
    return output

def shift_back(inputs:torch.Tensor, step:int, dim:int=1):
    """ Apply inverse shear transformation to the input tensor along the specified dimension.
    Args:
        inputs (torch.Tensor): Input tensor of shape [..., H, W+step*(C-1), C].
        step (int): Step size for the shear transformation.
        dim (int): Dimension along which to apply the shear transformation.
    Returns:
        torch.Tensor: Inverse sheared tensor of shape [..., H, W, C].
    """
    nC = inputs.shape[-1]
    orig_dim_size = inputs.shape[dim] - step * (nC - 1)
    
    output_shape = list(inputs.shape)
    output_shape[dim] = orig_dim_size
    output = torch.zeros(output_shape, dtype=inputs.dtype, device=inputs.device)
    for i in range(nC):
        start = step * i
        output[..., i] = inputs[..., i].narrow(dim, start, orig_dim_size)
    return output