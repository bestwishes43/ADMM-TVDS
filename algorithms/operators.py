import torch
import torch.nn.functional as F

def grad(u:torch.Tensor, ndim:int=-1):
    """Compute the gradient of u.
    Args:
        u (torch.Tensor): Input tensor.
        ndim (int): Number of dimensions to compute the gradient for. 
            If -1, it will use the number of dimensions of u.
    Returns:
        torch.Tensor: Gradient of u with shape (ndim, *u.shape).
    """
    ndim = u.ndim if ndim == -1 else ndim
    assert ndim <= u.ndim and ndim > 0, "ndim should be in [1, u.ndim] or -1."
    slice_all = [0, slice(None, -1)]
    grad_u = torch.zeros([ndim, ] + list(u.shape), 
                        dtype=u.dtype, device=u.device)
    for d in range(ndim):
        grad_u[tuple(slice_all)] = torch.diff(u, dim=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return grad_u

def div(p:torch.Tensor):
    """Compute the divergence of p.
    Args:
        p (torch.Tensor): Input tensor with shape (ndim, *).
    Returns:
        torch.Tensor: Divergence of p with shape (*).
    """
    return sum(p[d] - torch.roll(p[d], shifts=1, dims=d) for d in range(p.shape[0]))


def Diffusivity(x:torch.Tensor, dims:list[int]=[0]):
    """Compute the diffusivity of x.
    Args:
        x (torch.Tensor): Input tensor.
        dims (list[int]): Dimensions to compute the diffusivity for.
    Returns:
        torch.Tensor: Diffusivity of x.
    """
    # D = oslash(1.0, x.norm(p='fro', dim=dims, keepdim=True))
    # return D
    return torch.where((D_norm := x.norm(p='fro', dim=dims, keepdim=True)) > 1e-12, 1.0 / D_norm, 0)

def oslash(a, b:torch.Tensor):
    """Compute the oslash operation. If b is zero, return 0.
    Args:
        a (torch.Tensor): Numerator tensor.
        b (torch.Tensor): Denominator tensor.
    Returns:
        torch.Tensor: Result of $a \oslash b$.
    """
    return torch.where(b > 1e-12, a / b, 0)

@torch.jit.script
def shift(inputs:torch.Tensor, step:int, dim:int=1):
    """ Apply shear transformation to the input tensor along the specified dimension.
    Args:
        inputs (torch.Tensor): Input tensor of shape [..., H, W, C].
        step (int): Step size for the shear transformation.
        dim (int): Dimension along which to apply the shear transformation.
    Returns:
        torch.Tensor: Sheared tensor of shape [..., H, W+step*(C-1), C].
    """
    nC = inputs.shape[-1]
    n_order = len(inputs.shape)

    # Calculate padding size, only pad after the specified dimension (nC-1)*step
    padsize = [0]*n_order*2
    padsize[n_order*2 - 2*dim - 1] = (nC - 1)*step 

    output = F.pad(inputs, padsize, 'constant', 0.0)
    for i in range(1, nC):
        output[:, :, i] = torch.roll(output[:, :, i], step*i, dim)
    return output

@torch.jit.script
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