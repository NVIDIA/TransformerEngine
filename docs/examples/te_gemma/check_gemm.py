import functools
from typing import Optional, Tuple, Union, List
import torch
import transformer_engine as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.constants import TE_DType
from transformer_engine.pytorch.utils import assert_dim_for_fp8_exec
from transformer_engine.pytorch.module.base import get_workspace
import transformer_engine.pytorch.cpp_extensions as cpp_tex

@functools.lru_cache(maxsize=None)
def _empty_tensor() -> torch.Tensor:
    """Get tensor with no entries and no data"""
    return torch.Tensor()

def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    dtype: torch.dtype,
    workspace: torch.Tensor,
    gelu: bool = False,
    gelu_input: Optional[torch.Tensor] = None,
    grad: bool = False,
    accumulate: bool = False,
    layout: str = "TN",
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    ub_algo: tex.CommOverlapAlgo = None,
    ub: Union[tex.CommOverlap, tex.CommOverlapP2P] = None,
    extra_output_tensor: torch.Tensor = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Non FP8 GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    empty_tensor = _empty_tensor()
    fp8_index = -1  # dummy index

    if out is None:
        out = torch.empty(
            B.shape[1] if transb else B.shape[0],
            A.shape[0] if transa else A.shape[1],
            dtype=dtype,
            device="cuda",
        )
    else:
        if not out.is_contiguous():
            raise ValueError("Output tensor is not contiguous.")

    if gelu and not grad:
        gelu_input = torch.empty_like(out, dtype=dtype)
    elif not gelu:
        gelu_input = empty_tensor

    if grad and use_bias:
        grad_bias = torch.empty(B.shape[1], dtype=out.dtype, device="cuda")
    else:
        grad_bias = empty_tensor

    bias = bias if use_bias else empty_tensor

    assert (
        A.dtype == dtype and B.dtype == dtype
    ), f"Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}"
    input_dtype = TE_DType[dtype]
    output_dtype = TE_DType[out.dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias.dtype] if grad else TE_DType[bias.dtype]
    else:
        bias_dtype = output_dtype

    args = (
        A,
        empty_tensor,
        fp8_index,
        input_dtype,
        transa,
        B,
        empty_tensor,
        fp8_index,
        input_dtype,
        transb,
        out,
        empty_tensor,  # out_scale
        output_dtype,
        empty_tensor,  # out_amax
        grad_bias if grad else bias,
        bias_dtype,
        gelu_input,
        grad,
        workspace,
        workspace.shape[0],
        accumulate,
        False,  # use_split_accumulator
    )
    fn = torch.ops.tex_ts.te_gemm_ts
    if ub_algo is not None:
        assert ub is not None, "ub object is None!"
    _ = fn(*args)

    import pdb; pdb.set_trace()
    return out, grad_bias, gelu_input

if __name__ == "__main__":
    fc2_weight = torch.load("fc2_weight.pth").cuda()
    
    base_repo = "/perfhome/mnt/wkstn/work/repos/te_gemma_gen_support/TransformerEngine/docs/examples/te_gemma/"
    base_repo = ""
    gelu_out = torch.load(base_repo + "gelu_out.pth").cuda()
    
    activation_dtype = torch.bfloat16
    fc2_bias = _empty_tensor()
    use_fc2_bias = False
    
    dim_size = list(gelu_out.size())
    dim_size[1] = fc2_weight.size(0)
    fc2_out = torch.empty(dim_size, dtype=activation_dtype, device=gelu_out.device)

    _ = cpp_tex.gemm(
        fc2_weight,
        gelu_out,
        activation_dtype,
        get_workspace(),
        bias=fc2_bias,
        use_bias=use_fc2_bias,
        out=fc2_out,
        ub_algo=None,
        ub=None,
        extra_output_tensor=None,
    )