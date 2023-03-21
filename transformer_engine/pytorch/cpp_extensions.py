# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""TE FP8 extensions and GEMMs"""
from typing import Optional, Tuple, Union
import torch
import transformer_engine_extensions as tex
from .constants import TE_DType
import math

def check_tensor(x: torch.Tensor):
    assert (
            x.is_cuda and x.is_contiguous()
            ), f"Tensor needs to be on CUDA and contiguous."

def check_qkv(qkv: torch.Tensor):
    check_tensor(qkv)
    #print('qkv type ',qkv.dtype)
    assert (
            qkv.dtype is torch.uint8
            and qkv.dim() == 4
            and qkv.shape[1] == 3
            ), f"QKV needs to be in [total_seqs x 3 x num_heads x head_dim] and FP8."
    
def check_o(o: torch.Tensor):
    check_tensor(o)
    assert (
            o.dtype is torch.uint8
            and o.dim() == 3
            ), f"O needs to be a 3D FP8 tensor."

def check_stats(stats: torch.Tensor, b: int, h: int, s: int):
    check_tensor(stats)
    assert (
            stats.dtype is torch.float32
            and stats.dim() == 4
            and stats.shape == torch.Size([b, h, s, 1])
            ), f"Tensor needs to be in [b, h, s, 1] and float32."

def check_cu_seqlens(cu_seqlens: torch.Tensor):
    check_tensor(cu_seqlens)
    assert (
            cu_seqlens.dtype is torch.int32
            and cu_seqlens.dim() == 1
            ), f"cu_seqlens needs to be an int32 scalar."

def check_scalar(scalar: torch.Tensor):
    check_tensor(scalar)
    assert (
            scalar.dtype is torch.float32
            and scalar.dim() <= 1
            and scalar.numel() == 1
            ), f"Tensor needs to be a float32 scalar."

def check_seed(philox_unpacked: torch.Tensor):
    check_tensor(philox_unpacked)
    assert (
            philox_unpacked.dtype is torch.int64
            and philox_unpacked.numel() == 2
            ), f"Philox tensor should have two int64s."

def get_mha_layout(qkv_layout: str):
    qkv_layout = qkv_layout.lower()
    if qkv_layout == "not_interleaved":
        return 0
    elif qkv_layout == "qkv_interleaved":
        return 1 
    elif qkv_layout == "kv_interleaved":
        return 2 

def cudnn_flash_attn_fwd(
    qkv: torch.Tensor,
    qkv_dtype: tex.DType,
    cu_seqlens: torch.Tensor,
    d_scale_qkv: torch.Tensor,
    q_scale_s: torch.Tensor,
    q_scale_o: torch.Tensor,
    amax_s: torch.Tensor,
    amax_o: torch.Tensor,
    d_scale_s: torch.Tensor,
    d_scale_o: torch.Tensor,
    p_dropout: float,
    max_seq_len: int,
    is_training: bool,
    set_zero: bool,
    rng_gen: torch.Generator = None,
    qkv_layout: str = "qkv_interleaved",
) -> Tuple[Union[torch.Tensor, None], ...]:

    print("============== cpp_extension ============ ")
    #print("qkv_dtype ",qkv_dtype, qkv.shape)
    #print("entering fwd ")
    check_qkv(qkv)
    check_cu_seqlens(cu_seqlens)
    check_scalar(d_scale_qkv)
    check_scalar(q_scale_o)
    check_scalar(amax_s)
    check_scalar(amax_o)

    ##qkv_2d = qkv.view([512,3072]).contiguous()
    #descale_qkv = torch.Tensor([1.0])
    #qkv_float32 = torch.ops.tex_ts.cast_from_fp8_ts(qkv, descale_qkv, 0, qkv_dtype, tex.DType.kFloat32)
    #torch.cuda.synchronize()
    #print('----------- qkv float32 --------')
    #print(qkv_float32)

    assert max_seq_len <= 512, f"max_seq_len must be <= 512."
    b = cu_seqlens.numel() - 1
    assert b <= qkv.size(0), f"b must be <= qkv.size(0)."
    #actual_seqlens = (cu_seqlens[1:]-cu_seqlens[:-1]).to(dtype=torch.int32) 
    actual_seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    #print('actual seqlens ',actual_seqlens,cu_seqlens)
    #print('cu_seqlens ',cu_seqlens)

    total_seqs = qkv.size(0)
    h = qkv.size(2)
    d = qkv.size(3)
    scale_q_k = 1.0 / math.sqrt(d)

#    M = torch.empty([b, h, max_seq_len, 1], dtype = torch.float32, device = "cuda")
#    ZInv = torch.empty([b, h, max_seq_len, 1], dtype = torch.float32, device = "cuda")
#    O = torch.empty([total_seqs, h, d], dtype = torch.uint8, device = "cuda")
    QKVRaggedOffset = cu_seqlens * 3 * h * d
    ORaggedOffset = cu_seqlens * h * d

#    philox_unpacked = torch.empty([2], dtype = torch.int64, device="cuda")
#    if set_zero:
#        O.zero_()

#    qkv_type = TE_DType[qkv.dtype]
#    scale_amax_type = TE_DType[d_scale_qkv.dtype] 
#    seqlen_philox_type = TE_DType[cu_seqlens.dtype]
#             qkv_type,
#             scale_amax_type,
#             seqlen_philox_type,

    qkv_layout = get_mha_layout(qkv_layout)
    #rng_gen_new = torch.Generator(device="cuda") if not rng_gen else rng_gen
    #print("before calling ext fwd ")
    print('[P] b, max_seq_len, total_seqs, h, d, scale_q_k, p_dropout, qkv_layout, set_zero: ',
            b, max_seq_len, total_seqs, h, d, scale_q_k, p_dropout, qkv_layout, set_zero)
    print('[P] qkv: ', qkv.dtype, qkv_dtype)
    print('[P] d_scale_qkv, d_scale_s, d_scale_o, q_scale_s, q_scale_o, amax_s, amax_o: ',
            d_scale_qkv, d_scale_s, d_scale_o, q_scale_s, q_scale_o, amax_s, amax_o)
    print('[P] QKVRaggedOffset, ORaggedOffset, actual_seqlens: ')
    print(QKVRaggedOffset, ORaggedOffset, actual_seqlens)
    print('------- fwd',d_scale_qkv.shape,d_scale_qkv.dtype,d_scale_qkv)
    O, M, ZInv, philox_unpacked = tex.cudnn_flash_attn_fwd(
             b, max_seq_len, total_seqs, h, d, scale_q_k, p_dropout, qkv_layout, set_zero,
             qkv, qkv_dtype,
             d_scale_qkv,
             d_scale_s,
             d_scale_o,
             q_scale_s,
             q_scale_o,
             amax_s,
             amax_o,
             QKVRaggedOffset,
             ORaggedOffset,
             actual_seqlens,
             rng_gen,
    )
    #print('[P] O, M, ZInv, philox_unpacked: ')
    #print(O[0,:5,:5], M[0,0,:5,0], ZInv[0,0,:5,0], philox_unpacked)
    #print(O[1,:5,:5], M[1,0,:5,0], ZInv[1,0,:5,0], philox_unpacked)

    return O, M, ZInv, philox_unpacked 

def cudnn_flash_attn_bwd(
    dO: torch.Tensor,
    qkv: torch.Tensor,
    O: torch.Tensor,
    M: torch.Tensor,
    ZInv: torch.Tensor,
    qkv_dtype: tex.DType,
    cu_seqlens: torch.Tensor,
    d_scale_qkv: torch.Tensor,
    d_scale_s: torch.Tensor,
    d_scale_o: torch.Tensor,
    d_scale_do: torch.Tensor,
    q_scale_s: torch.Tensor,
    q_scale_ds: torch.Tensor,
    q_scale_dqkv: torch.Tensor,
    amax_ds: torch.Tensor,
    amax_dqkv: torch.Tensor,
    d_scale_ds: torch.Tensor,
    d_scale_dqkv: torch.Tensor,
    p_dropout: float,
    max_seq_len: int,
    set_zero: bool,
    all_e5m2: bool, # unused
    philox_unpacked: torch.Tensor,
    qkv_layout: str = "qkv_interleaved",
) -> Tuple[Union[torch.Tensor, None], ...]:

    check_o(O)
    check_o(dO)
    check_qkv(qkv)

    x = ['dO', 'qkv', 'O', 'M', 'ZInv']
    for i in x:
        print('[P] '+i+': ',eval(i).dtype,eval(i).shape)
    print('[P] qkv_dtype: ',qkv_dtype)
    print('[P] cu_seqlens: ',cu_seqlens)
    print('[P] d_scale_qkv, d_scale_s, d_scale_o, d_scale_do: ',d_scale_qkv, d_scale_s, d_scale_o, d_scale_do) 
    print('[P] q_scale_s, q_scale_ds, q_scale_dqkv: ',q_scale_s, q_scale_ds, q_scale_dqkv) 
    print('[P] amax_ds, amax_dqkv: ',amax_ds, amax_dqkv) 
    print('[P] d_scale_ds, d_scale_dqkv: ',d_scale_ds, d_scale_dqkv) 
    print('[P] p_dropout, max_seq_len, set_zero, all_e5m2, philox_unpacked, qkv_layout: ',p_dropout, max_seq_len, set_zero, all_e5m2, philox_unpacked, qkv_layout)
    print(' ')
    print('------',d_scale_qkv.shape,d_scale_qkv.dtype,d_scale_qkv)

    assert max_seq_len <= 512, f"max_seq_len must be <= 512."
    b = cu_seqlens.numel() - 1
    assert b <= qkv.size(0), f"b must be <= qkv.size(0)."
    actual_seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    #print('actual seqlens ',actual_seqlens,cu_seqlens)

    total_seqs = qkv.size(0)
    h = qkv.size(2)
    d = qkv.size(3)
    scale_q_k = 1.0 / math.sqrt(d)

    #print('M shape ',M.shape, ZInv.shape)
    check_stats(M, b, h, max_seq_len)
    check_stats(ZInv, b, h, max_seq_len)
    check_seed(philox_unpacked)

    #dQtmp = torch.empty([b, h, max_seq_len, d], dtype = torch.float32, device="cuda")
    #dQKV = torch.empty_like(qkv)

    #if set_zero:
    #    dQKV.zero_()

    QKVRaggedOffset = cu_seqlens * 3 * h * d
    ORaggedOffset = cu_seqlens * h * d
    qkv_layout = get_mha_layout(qkv_layout)
    print('[P] b, max_seq_len, total_seqs, h, d, scale_q_k, p_dropout, qkv_layout, set_zero',
            b, max_seq_len, total_seqs, h, d, scale_q_k, p_dropout, qkv_layout, set_zero)
    print('[P] QKVRaggedOffset, ORaggedOffset, actual_seqlens: ')
    print(QKVRaggedOffset, ORaggedOffset, actual_seqlens)
    dQKV = tex.cudnn_flash_attn_bwd(
             b, max_seq_len, total_seqs, h, d, scale_q_k, p_dropout, qkv_layout, set_zero,
             qkv,
             dO,
             O,
             M,
             ZInv,
             qkv_dtype,
             d_scale_qkv,
             d_scale_s,
             d_scale_o,
             d_scale_do,
             d_scale_ds,
             d_scale_dqkv,
             q_scale_s,
             q_scale_ds,
             q_scale_dqkv,
             amax_ds,
             amax_dqkv,
             QKVRaggedOffset,
             ORaggedOffset,
             actual_seqlens,
             philox_unpacked,
    )
             #dQtmp,

    return dQKV

def fp8_gemm(
    A: torch.Tensor,
    A_scale_inv: torch.Tensor,
    A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    A_dtype: tex.DType,
    B: torch.Tensor,
    B_scale_inv: torch.Tensor,
    B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    B_dtype: tex.DType,
    out_dtype: torch.dtype,
    workspace: torch.Tensor,
    accumulate: bool = False,
    out: Optional[torch.Tensor] = None,
    out_index = None,
    fp8_meta_tensor: tex.FP8TensorMeta = None,
    bias: Optional[torch.Tensor] = None,
    use_bias: bool = False,
    use_split_accumulator: bool = False,
    D_dtype: Optional[tex.DType] = None,
) -> torch.Tensor:
    """TN layout GEMM with fp8 inputs."""

    empty_tensor = torch.Tensor()
    if D_dtype is not None and D_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert fp8_meta_tensor is not None and out_index is not None

    return_output = False
    if out is None:
        out = torch.empty(
            B.shape[0],
            A.shape[0],
            dtype=out_dtype,
            device="cuda",
        )
        return_output = True

    out_dtype = TE_DType[out.dtype] if D_dtype is None else D_dtype
    # Use bfloat16 as default bias_dtype
    bias_dtype = tex.DType.kBFloat16 if bias is None else TE_DType[bias.dtype]

    _ = torch.ops.tex_ts.te_gemm_ts(
        A,
        A_scale_inv,
        A_fp8_tensor,
        A_dtype,
        True,  # transa
        B,
        B_scale_inv,
        B_fp8_tensor,
        B_dtype,
        False,  # transb
        out,
        empty_tensor if out_index is None else fp8_meta_tensor.scale[out_index],
        out_dtype,
        empty_tensor if out_index is None else fp8_meta_tensor.amax_history[0][out_index],
        bias if use_bias else empty_tensor,
        bias_dtype,
        empty_tensor,  # this is pre_gelu_out
        False,  # grad
        workspace,
        workspace.shape[0],
        accumulate,
        use_split_accumulator,
    )

    if return_output:
        return out
    return None


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
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Non FP8 GEMM."""

    assert layout in ("TN", "NN", "NT"), f"GEMM layout {layout} not supported."
    transa = layout[0] == "T"
    transb = layout[1] == "T"
    empty_tensor = torch.Tensor()
    fp8_index = -1 # dummy index

    return_output = False
    if out is None:
        out = torch.empty(
            B.shape[1] if transb else B.shape[0],
            A.shape[0] if transa else A.shape[1],
            dtype=dtype,
            device="cuda",
        )
        return_output = True

    if gelu and not grad:
        gelu_input = torch.empty_like(out, dtype=dtype)
    elif not gelu:
        gelu_input = empty_tensor

    if grad and use_bias:
        grad_bias = torch.empty(B.shape[1], dtype=out.dtype, device="cuda")
    else:
        grad_bias = empty_tensor

    bias = bias if use_bias else empty_tensor

    assert A.dtype == dtype and B.dtype == dtype, \
        f'Expected dtype={dtype}, but found A.dtype={A.dtype} and B.dtype={B.dtype}'
    input_dtype = TE_DType[dtype]
    output_dtype = TE_DType[out.dtype]
    if use_bias:
        bias_dtype = TE_DType[grad_bias.dtype] if grad else TE_DType[bias.dtype]
    else:
        bias_dtype = output_dtype

    _ = torch.ops.tex_ts.te_gemm_ts(
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
        empty_tensor, # out_scale
        output_dtype,
        empty_tensor, # out_amax
        grad_bias if grad else bias,
        bias_dtype,
        gelu_input,
        grad,
        workspace,
        workspace.shape[0],
        accumulate,
        False,  # use_split_accumulator
    )

    if return_output:
        return out, grad_bias, gelu_input
    return None, grad_bias, gelu_input


def fp8_cast_transpose_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    cast_out: Optional[torch.Tensor] = None,
    transpose_out: Optional[torch.Tensor] = None,
) -> Union[Tuple[torch.Tensor, torch.Tensor], None]:
    """Cast + Transpose with FP8 output"""

    return_outputs = False
    if cast_out is None or transpose_out is None:
        cast_out = torch.empty_like(inp, dtype=torch.uint8)
        transpose_out = torch.empty(
            inp.shape[1], inp.shape[0], device="cuda", dtype=torch.uint8
        )
        return_outputs = True

    tex.fused_cast_transpose(
        inp,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        cast_out,
        transpose_out,
        otype,
    )

    if return_outputs:
        return cast_out, transpose_out
    return None


def fp8_cast_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD with FP8 output"""
    print('fp8_cast_transpose_bgrad_fused otype',otype)
    return tex.fused_cast_transpose_bgrad(
        inp,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
    )


def fp8_transpose_bgrad_fused(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    grad_bias_type: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transpose + BGRAD with FP8 output"""
    return tex.fused_fp8_transpose_bgrad(
        inp,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
        TE_DType[grad_bias_type],
    )


def fp8_cast_transpose_bgrad_dgelu_fused(
    grad_output: torch.Tensor,
    gelu_input: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cast + Transpose + BGRAD + DGELU with FP8 output"""
    return tex.fused_cast_transpose_bgrad_dgelu(
        grad_output,
        gelu_input,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
    )


def fp8_gelu(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """GeLU with FP8 output"""
    return torch.ops.tex_ts.fp8_gelu_ts(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
    )


def layernorm_fwd_fp8(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    sm_margin: int,
    zero_centered_gamma: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LayerNorm with FP8 output"""
    return tex.layernorm_fwd_fp8(
        inp,
        weight,
        bias,
        eps,
        fp8_meta_tensor.scale[fp8_tensor],
        fp8_meta_tensor.amax_history[0][fp8_tensor],
        fp8_meta_tensor.scale_inv[fp8_tensor],
        otype,
        sm_margin,
        zero_centered_gamma
    )


def layernorm_fwd_fp8_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
    zero_centered_gamma,
) -> torch.Tensor:
    """LayerNorm with FP8 output.

    This version of layernorm_fwd_fp8 is specialized for inference, and returns
    only the normalized output.
    """
    ret = torch.ops.tex_ts.layernorm_fwd_fp8_inf_ts(
        inp,
        weight,
        bias,
        eps,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
        zero_centered_gamma)
    return ret


def layernorm_fwd_inf(
    inp: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    zero_centered_gamma: bool,
) -> torch.Tensor:
    """LayerNorm with FP8 output"""
    return torch.ops.tex_ts.layernorm_fwd_inf_ts(
        inp,
        weight,
        bias,
        eps,
        zero_centered_gamma,
    )


def cast_to_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    otype: tex.DType,
) -> torch.Tensor:
    """Cast input to FP8"""
    return torch.ops.tex_ts.cast_to_fp8_ts(
        inp,
        fp8_meta_tensor.scale,
        fp8_meta_tensor.amax_history,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        otype,
    )


def cast_from_fp8(
    inp: torch.Tensor,
    fp8_meta_tensor: tex.FP8TensorMeta,
    fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    itype: tex.DType,
    otype: tex.DType,
) -> torch.Tensor:
    """Cast input from FP8"""
    return torch.ops.tex_ts.cast_from_fp8_ts(
        inp,
        fp8_meta_tensor.scale_inv,
        fp8_tensor,
        itype,
        otype,
    )
