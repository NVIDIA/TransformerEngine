
import torch
import torch._prims_common as utils

import transformer_engine_torch as tex

def empty_like_cached(input, *, dtype=None, layout=None, device=None, requires_grad=False, 
        memory_format=torch.preserve_format):

    dtype = input.dtype if dtype is None else dtype
    # layout = input.layout if layout is None else layout
    device = input.device if device is None else device

    if isinstance(device, int):
        device = torch.device(device)
    if isinstance(device, str):
        device = torch.device(device, torch.cuda.current_device())

    copy = tex.empty_like_cached(
        input, 
        dtype=dtype,
        layout=layout,
        device=device,
        pin_memory=False,
        memory_format=None) #TODO
    wrapper = torch.Tensor()
    wrapper.data = copy
    wrapper.requires_grad = input.requires_grad
    
    return wrapper

def empty_cached(*size, out=None, dtype=None, layout=torch.strided, device=None, 
        requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format):

    size = utils.extract_shape_from_varargs(size)
    dtype = torch.get_default_dtype() if dtype is None else dtype
    device = torch.device("cpu") if device is None else device

    if isinstance(size, torch.Size):
        size = tuple(size)
    if isinstance(device, int):
        device = torch.device(device)
    if isinstance(device, str):
        device = torch.device(device, torch.cuda.current_device())

    copy = tex.empty_cached(
        size=size, 
        dtype=dtype, 
        device=device, 
        pin_memory=False, 
        memory_format=None)
    copy.requires_grad = requires_grad
    return copy