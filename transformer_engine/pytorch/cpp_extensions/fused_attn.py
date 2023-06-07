# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from .common import *
import torch

__all__ = ['fused_attn_fwd_qkvpacked',
           'fused_attn_bwd_qkvpacked',
           'fused_attn_fwd_kvpacked',
           'fused_attn_bwd_kvpacked']

def check_qkv(qkv: torch.Tensor, dtype: torch.dtype):
    """Check tensor properties."""
    check_tensor(qkv)
    assert (qkv.dtype is dtype
            and qkv.dim() == 4
            and qkv.shape[1] == 3
            ), """QKV should be in [total_seqs, 3, num_heads, head_dim] shape
    and {dtype} dtype."""

def check_q(q: torch.Tensor, dtype: torch.dtype):
    """Check tensor properties."""
    check_tensor(q)
    assert (q.dtype is dtype
            and q.dim() == 3
            ), """Q should be in [total_seqs, num_heads, head_dim] shape
    and {dtype} dtype."""

def check_kv(kv: torch.Tensor, dtype: torch.dtype):
    """Check tensor properties."""
    check_tensor(kv)
    assert (kv.dtype is dtype
            and kv.dim() == 4
            and kv.shape[1] == 2
            ), """KV should be in [total_seqs, 2, num_heads, head_dim] shape
    and {dtype} dtype."""

def check_o(o: torch.Tensor, dtype: torch.dtype):
    """Check tensor properties."""
    check_tensor(o)
    assert (o.dtype is dtype
            and o.dim() == 3
            ), """O and dO should be in [total_seqs, num_heads, head_dim] shape
    and {dtype} dtype."""

def check_stats(stats: torch.Tensor, b: int, h: int, s: int):
    """Check tensor properties."""
    check_tensor(stats)
    assert (stats.dtype is torch.float32
            and stats.dim() == 4
            and stats.shape == torch.Size([b, h, s, 1])
            ), """M and ZInv should be in [batch_size, num_heads, max_seqlen_q, 1]
    shape and float32 dtype."""

def check_cu_seqlens(cu_seqlens: torch.Tensor):
    """Check tensor properties."""
    check_tensor(cu_seqlens)
    assert (cu_seqlens.dtype is torch.int32
            and cu_seqlens.dim() == 1
            ), """cu_seqlens should be in [batch_size +1] shape and int32 dtype."""

def check_scalar(scalar: torch.Tensor):
    """Check tensor properties."""
    check_tensor(scalar)
    assert (scalar.dtype is torch.float32
            and scalar.dim() <= 1
            and scalar.numel() == 1
            ), "amax/scale/descale tensors should be scalars in float32 dtype."

def check_rng_state(rng_state: torch.Tensor):
    """Check tensor properties."""
    check_tensor(rng_state)
    assert (rng_state.dtype is torch.int64
            and rng_state.numel() == 2
            ), "rng_state should be [seed, offset] and in int64 dtype."

def fused_attn_fwd_qkvpacked(
    is_training: bool,
    max_seqlen: int,
    cu_seqlens: torch.Tensor,
    qkv: torch.Tensor,
    qkv_dtype: tex.DType,
    bias: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_o: torch.Tensor = None,
    amax_s: torch.Tensor = None,
    amax_o: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    set_zero: bool = True,
    qkv_layout: str = "qkv_interleaved",
    bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    rng_gen: torch.Generator = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention FWD for packed QKV input.

    Parameters
    ----------
    is_training: bool
                if True, runs training and produces auxiliary tensors aux_ctx_tensors
                for the backward; if False, runs inference and doesn't produce aux_ctx_tensors
    max_seqlen: int
                max sequence length for QKV, used for padding; may be larger than max(cu_seqlens)
    cu_seqlens: torch.Tensor
                accumulative sequence lengths for QKV; shape [batch_size + 1]
    qkv: torch.Tensor
                input tensor QKV;
                shape [total_seqs, 3, num_heads, head_dim], where total_seqs = cu_seqlens[-1]
    qkv_dtype: tex.DType
                data type of QKV; in tex.DType, not torch.dtype
    bias: torch.Tensor, default = None
                input tensor Bias when bias_type is "pre_scale_bias" or "post_scale_bias";
                shape [1, num_heads, max_seqlen, max_seqlen], same data type as qkv
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    q_scale_s: torch.Tensor, default = None
                input tensor for the quantization of S in FP8 computations, S = Softmax(Q * K.T)
    q_scale_o: torch.Tensor, default = None
                input tensor for the quantization of O in FP8 computations
    amax_s: torch.Tensor, default = None
                output tensor, amax of S, used by the next iteration in FP8 computations
    amax_o: torch.Tensor, default = None
                output tensor, amax of O, used by the next iteration in FP8 computations
    attn_scale: float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim) as the default
    dropout: float, default = 0.0
                dropout probability, 0.0 means no dropout, 1.0 means no output;
                dropout must be 0.0 if is_training is False
    set_zero: bool, default = True
                if True, initializes the output tensor O to zero using the mha_fill method;
                if False, doesn't initialize O after its allocation
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "no_mask"}
    rng_gen: torch.Generator, default = None
                random number generator;
                if None, uses the default CUDA generator from PyTorch; otherwise, uses rng_gen

    Returns
    ----------
    o: torch.Tensor
                output tensor O, of the attention calculation; same data type as QKV;
                shape [total_seqs, num_heads, head_dim], where total_seqs = cu_seqlens[-1]
    aux_ctx_tensors: List[torch.Tensor]
                auxiliary output tensors used for the backward;
                if is_training is True, aux_ctx_tensors = [M, ZInv, rng_state]
                if is_training is False, aux_ctx_tensors = [rng_state]
                M: torch.Tensor
                    max(Q*K.T)
                    shape [batch_size, num_heads, max_seqlen, 1], dtype float32
                ZInv: torch.Tensor
                    1/sum(e^(x - max(x))), where x=Q*K.T
                    shape [batch_size, num_heads, max_seqlen, 1], dtype float32
                rng_state: torch.Tensor
                    state of the random number generator;
                    [seed, offset], dtype uint64
    """

    check_cu_seqlens(cu_seqlens)
    b = cu_seqlens.numel() - 1
    qkv_type = TORCH_DType[qkv_dtype]
    check_qkv(qkv, qkv_type)

    total_seqs = qkv.size(0)
    h = qkv.size(2)
    d = qkv.size(3)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(d)

    if bias_type != "no_bias":
        assert bias is not None, "bias tensor cannot be None when bias_type is not no_bias."
        assert (bias.shape == [1, h, max_seqlen, max_seqlen]
               ), "bias tensor must be in [1, h, max_seqlen, max_seqlen] shape."
        assert (bias.dtype == qkv.dtype
               ), "bias tensor must be in the same dtype as qkv."

    # FP8 fused attention API
    if (qkv_type is torch.uint8) and (max_seqlen <= 512) and (d == 64):
        assert (qkv_layout == "qkv_interleaved"
                and bias_type == "no_bias"
                and attn_mask_type == "padding"
                ), """The FP8 fused attention API currently only supports qkv_interleaved layout,
                no_bias type, and padding attention mask type."""
        assert (d_scale_qkv is not None), "d_scale_qkv is required for the FP8 API."
        assert (q_scale_s is not None), "q_scale_s is required for the FP8 API."
        assert (q_scale_o is not None), "q_scale_o is required for the FP8 API."
        assert (amax_s is not None), "amax_s is required for the FP8 API."
        assert (amax_o is not None), "amax_o is required for the FP8 API."
        check_scalar(d_scale_qkv)
        check_scalar(q_scale_s)
        check_scalar(q_scale_o)
        check_scalar(amax_s)
        check_scalar(amax_o)

    # BF16/FP16 fused attention API from fmha_v2
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) and (max_seqlen > 512):
        # add BF/FP16 support for >512 sequence length
        assert False, "The BF16/FP16 support for >512 sequence length is coming!"

    # BF16/FP16 fused attention API from fmha_v1 apex
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) and (max_seqlen <= 512):
        # add BF/FP16 support for <=512 sequence length
        assert False, "The BF16/FP16 support for <=512 sequence length is coming!"

    else:
        assert False, "No support for this dtype and max_seqlen combination."

    # execute kernel
    output_tensors = tex.fused_attn_fwd_qkvpacked(
            b, max_seqlen, total_seqs, h, d,
            is_training, attn_scale, dropout, set_zero, qkv_layout, bias_type, attn_mask_type,
            cu_seqlens,
            qkv,
            qkv_dtype,
            d_scale_qkv,
            q_scale_s,
            q_scale_o,
            amax_s,
            amax_o,
            bias,
            rng_gen,
    )

    return output_tensors[0], output_tensors[1:]


def fused_attn_bwd_qkvpacked(
    max_seqlen: int,
    cu_seqlens: torch.Tensor,
    qkv: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    qkv_dtype: tex.DType,
    aux_ctx_tensors: List[torch.Tensor] = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    d_scale_o: torch.Tensor = None,
    d_scale_do: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_dp: torch.Tensor = None,
    q_scale_dqkv: torch.Tensor = None,
    amax_dp: torch.Tensor = None,
    amax_dqkv: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    set_zero: bool = True,
    qkv_layout: str = "qkv_interleaved",
    bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention BWD for packed QKV input.

    Parameters
    ----------
    max_seqlen: int
                max sequence length for QKV, used for padding; may be larger than max(cu_seqlens_q)
    cu_seqlens: torch.Tensor
                accumulative sequence lengths for QKV; shape [batch_size + 1]
    qkv: torch.Tensor
                input tensor QKV;
                shape [total_seqs, 3, num_heads, head_dim], where total_seqs = cu_seqlens[-1]
    o: torch.Tensor
                input tensor O (output of forward);
                shape [total_seqs, num_heads, head_dim], where total_seqs = cu_seqlens[-1]
    d_o: torch.Tensor
                input tensor dO (gradient of O);
                shape [total_seqs, num_heads, head_dim], where total_seqs = cu_seqlens[-1]
    qkv_dtype: tex.DType
                data type of QKV; in tex.DType, not torch.dtype
    aux_ctx_tensors: List[torch.Tensor]
                auxiliary output tensors of the forward pass when its is_training is True,
                e.g. aux_ctx_tensors = [M, ZInv, rng_state]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
    d_scale_o: torch.Tensor, default = None
                input tensor for the dequantization of O in FP8 computations
    d_scale_do: torch.Tensor, default = None
                input tensor for the dequantization of dO in FP8 computations
    q_scale_s: torch.Tensor, default = None
                input tensor for the quantization of S in FP8 computations
    q_scale_dp: torch.Tensor, default = None
                input tensor for the quantization of dP in FP8 computations, P = Q * K.T
    q_scale_dqkv: torch.Tensor, default = None
                input tensor for the quantization of dQKV in FP8 computations
    amax_dp: torch.Tensor, default = None
                output tensor, amax of dP, used by the next iteration in FP8 computations
    amax_dqkv: torch.Tensor, default = None
                output tensor, amax of dQKV, used by the next iteration in FP8 computations
    attn_scale: float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim) as the default
    dropout: float, default = 0.0
                dropout probability, 0.0 means no dropout, 1.0 means no output;
                dropout must be 0.0 if is_training is False
    set_zero: bool, default = True
                if True, initializes the output tensor O to zero using the mha_fill method;
                if False, doesn't initialize O after its allocation
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "no_mask"}

    Returns
    ----------
    d_qkv: torch.Tensor
                gradient tensor of QKV; same data type and shape as QKV
    d_bias: torch.Tensor, optional
                gradient tensor of Bias when bias_type is "pre_scale_bias" or "post_scale_bias";
                same data type and shape as Bias
    """

    check_cu_seqlens(cu_seqlens)
    b = cu_seqlens.numel() - 1
    qkv_type = TORCH_DType[qkv_dtype]
    check_qkv(qkv, qkv_type)
    check_o(o, qkv_type)
    check_o(d_o, qkv_type)

    total_seqs = qkv.size(0)
    h = qkv.size(2)
    d = qkv.size(3)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(d)

    assert (len(aux_ctx_tensors) >= 1
            ), "aux_ctx_tensors must contain rng_state as its last element."
    rng_state = aux_ctx_tensors[-1]
    check_rng_state(rng_state)

    # FP8 fused attention API
    if (qkv_type is torch.uint8) and (max_seqlen <= 512) and d == 64:
        assert (qkv_layout == "qkv_interleaved"
                and bias_type == "no_bias"
                and attn_mask_type == "padding"
                ), """The FP8 fused attention API currently only supports qkv_interleaved layout,
                no_bias type, and padding attention mask type."""
        assert (d_scale_qkv is not None), "d_scale_qkv is required for the FP8 API."
        assert (d_scale_s is not None), "d_scale_s is required for the FP8 API."
        assert (d_scale_o is not None), "d_scale_o is required for the FP8 API."
        assert (d_scale_do is not None), "d_scale_do is required for the FP8 API."
        assert (q_scale_s is not None), "q_scale_s is required for the FP8 API."
        assert (q_scale_dp is not None), "q_scale_dp is required for the FP8 API."
        assert (q_scale_dqkv is not None), "q_scale_dqkv is required for the FP8 API."
        assert (amax_dp is not None), "amax_dp is required for the FP8 API."
        assert (amax_dqkv is not None), "amax_dqkv is required for the FP8 API."
        assert (len(aux_ctx_tensors) == 3
                ), "aux_ctx_tensors is required to be [M, ZInv, rng_state] for the FP8 API."
        check_scalar(d_scale_qkv)
        check_scalar(d_scale_s)
        check_scalar(d_scale_o)
        check_scalar(d_scale_do)
        check_scalar(q_scale_s)
        check_scalar(q_scale_dp)
        check_scalar(q_scale_dqkv)
        check_scalar(amax_dp)
        check_scalar(amax_dqkv)
        m, z_inv = aux_ctx_tensors[:2]
        check_stats(m, b, h, max_seqlen)
        check_stats(z_inv, b, h, max_seqlen)

    # BF16/FP16 fused attention API from fmha_v2
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) and (max_seqlen > 512):
        # add BF/FP16 support for >512 sequence length
        assert False, "The BF16/FP16 support for >512 sequence length is coming!"

    # BF16/FP16 fused attention API from fmha_v1 apex
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) and (max_seqlen <= 512):
        # add BF/FP16 support for <=512 sequence length
        assert False, "The BF16/FP16 support for <=512 sequence length is coming!"

    else:
        assert False, "No support for this dtype and max_seqlen combination."

    # execute kernel
    output_tensors = tex.fused_attn_bwd_qkvpacked(
            b, max_seqlen, total_seqs, h, d,
            attn_scale, dropout, set_zero, qkv_layout, bias_type, attn_mask_type,
            cu_seqlens,
            qkv, o, d_o,
            qkv_dtype,
            aux_ctx_tensors,
            d_scale_qkv, d_scale_s, d_scale_o, d_scale_do,
            q_scale_s, q_scale_dp, q_scale_dqkv,
            amax_dp, amax_dqkv,
    )

    if bias_type == "no_bias":
        # return d_qkv when bias_type is no_bias
        return output_tensors[0]
    # otherwise return (d_qkv, d_bias)
    return output_tensors


def fused_attn_fwd_kvpacked(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    qkv_dtype: tex.DType,
    bias: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_o: torch.Tensor = None,
    amax_s: torch.Tensor = None,
    amax_o: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    set_zero: bool = True,
    qkv_layout: str = "qkv_interleaved",
    bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    rng_gen: torch.Generator = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention FWD for packed KV input.

    Parameters
    ----------
    is_training: bool
                if True, runs training and produces auxiliary tensors aux_ctx_tensors
                for the backward; if False, runs inference and doesn't produce aux_ctx_tensors
    max_seqlen_q: int
                max sequence length for Q, used for padding; may be larger than max(cu_seqlens_q)
    max_seqlen_kv: int
                max sequence length for KV, used for padding; may be larger than max(cu_seqlens_kv)
    cu_seqlens_q: torch.Tensor
                accumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv: torch.Tensor
                accumulative sequence lengths for KV; shape [batch_size + 1]
    q: torch.Tensor
                input tensor Q;
                shape [total_seqs_q, num_heads, head_dim], where total_seqs_q = cu_seqlens_q[-1]
    kv: torch.Tensor
                packed input tensor KV;
                shape [total_seqs_kv, 2, num_heads, head_dim],
                where total_seqs_kv = cu_seqlens_kv[-1]
    qkv_dtype: tex.DType
                data type of Q and KV; in tex.DType, not torch.dtype
    bias: torch.Tensor, default = None
                input tensor Bias when bias_type is "pre_scale_bias" or "post_scale_bias";
                shape [1, num_heads, max_seqlen_q, max_seqlen_kv], same data type as q and kv
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    q_scale_s: torch.Tensor, default = None
                input tensor for the quantization of S in FP8 computations, S = Softmax(Q * K.T)
    q_scale_o: torch.Tensor, default = None
                input tensor for the quantization of O in FP8 computations
    amax_s: torch.Tensor, default = None
                output tensor, amax of S, used by the next iteration in FP8 computations
    amax_o: torch.Tensor, default = None
                output tensor, amax of O, used by the next iteration in FP8 computations
    attn_scale: float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim) as the default
    dropout: float, default = 0.0
                dropout probability, 0.0 means no dropout, 1.0 means no output;
                dropout must be 0.0 if is_training is False
    set_zero: bool, default = True
                if True, initializes the output tensor O to zero using the mha_fill method;
                if False, doesn't initialize O after its allocation
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "no_mask"}
    rng_gen: torch.Generator, default = None
                random number generator;
                if None, uses the default CUDA generator from PyTorch; otherwise, uses rng_gen

    Returns
    ----------
    o: torch.Tensor
                output tensor O, of the attention calculation; same data type as QKV;
                shape [total_seqs, num_heads, head_dim], where total_seqs = cu_seqlens[-1]
    aux_ctx_tensors: List[torch.Tensor]
                auxiliary output tensors used for the backward;
                if is_training is True, aux_ctx_tensors = [M, ZInv, rng_state]
                if is_training is False, aux_ctx_tensors = [rng_state]
                M: torch.Tensor
                    max(Q*K.T)
                    shape [batch_size, num_heads, max_seqlen, 1], dtype float32
                ZInv: torch.Tensor
                    1/sum(e^(x - max(x))), where x=Q*K.T
                    shape [batch_size, num_heads, max_seqlen, 1], dtype float32
                rng_state: torch.Tensor
                    state of the random number generator;
                    [seed, offset], dtype uint64
    """

    check_cu_seqlens(cu_seqlens_q)
    check_cu_seqlens(cu_seqlens_kv)
    assert (cu_seqlens_q.numel() == cu_seqlens_kv.numel()
            ), "cu_seqlens_q and cu_seqlens_kv must have the same length."
    b = cu_seqlens_q.numel() - 1
    qkv_type = TORCH_DType[qkv_dtype]
    check_q(q, qkv_type)
    check_kv(kv, qkv_type)

    assert (q.size(1) == kv.size(2)
            and q.size(2) == kv.size(3)
            ), "Q and KV must have the same num_heads and head_dim."
    total_seqs_q = q.size(0)
    total_seqs_kv = kv.size(0)
    h = q.size(1)
    d = q.size(2)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(d)

    if bias_type != "no_bias":
        assert bias is not None, "bias tensor cannot be None when bias_type is not no_bias."
        assert (bias.shape == [1, h, max_seqlen_q, max_seqlen_kv]
               ), "bias tensor must be in [1, h, max_seqlen_q, max_seqlen_kv] shape."
        assert (bias.dtype == q.dtype
               ), "bias tensor must be in the same dtype as q and kv."

    # FP8 fused attention API
    if (qkv_type is torch.uint8) and (max_seqlen_q <= 512) and (max_seqlen_kv <= 512) \
            and (d == 64):
        assert False, "The FP8 fused attention API currently only supports packed QKV input."

    # BF16/FP16 fused attention API from fmha_v2
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) \
            and (max_seqlen_q > 512) and (max_seqlen_kv > 512):
        # add BF/FP16 support for >512 sequence length
        assert False, "The BF16/FP16 support for >512 sequence length is coming!"

    # BF16/FP16 fused attention API from fmha_v1 apex
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) \
            and (max_seqlen_q <= 512) and (max_seqlen_kv <= 512):
        # add BF/FP16 support for <=512 sequence length
        assert False, "The BF16/FP16 support for <=512 sequence length is coming!"

    else:
        assert False, "No support for this dtype and max_seqlen combination."

    # execute kernel
    output_tensors = tex.fused_attn_fwd_kvpacked(
            b, max_seqlen_q, max_seqlen_kv, total_seqs_q, total_seqs_kv, h, d,
            is_training, attn_scale, dropout, set_zero, qkv_layout, bias_type, attn_mask_type,
            cu_seqlens_q, cu_seqlens_kv,
            q, kv,
            qkv_dtype,
            d_scale_qkv,
            q_scale_s,
            q_scale_o,
            amax_s,
            amax_o,
            bias,
            rng_gen,
    )

    return output_tensors[0], output_tensors[1:]


def fused_attn_bwd_kvpacked(
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    qkv_dtype: tex.DType,
    aux_ctx_tensors: List[torch.Tensor] = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    d_scale_o: torch.Tensor = None,
    d_scale_do: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_dp: torch.Tensor = None,
    q_scale_dqkv: torch.Tensor = None,
    amax_dp: torch.Tensor = None,
    amax_dqkv: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    set_zero: bool = True,
    qkv_layout: str = "qkv_interleaved",
    bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention BWD for packed KV input.

    Parameters
    ----------
    max_seqlen_q: int
                max sequence length for Q, used for padding; may be larger than max(cu_seqlens_q)
    max_seqlen_kv: int
                max sequence length for KV, used for padding; may be larger than max(cu_seqlens_kv)
    cu_seqlens_q: torch.Tensor
                accumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv: torch.Tensor
                accumulative sequence lengths for KV; shape [batch_size + 1]
    q: torch.Tensor
                input tensor Q;
                shape [total_seqs_q, num_heads, head_dim], where total_seqs_q = cu_seqlens_q[-1]
    kv: torch.Tensor
                packed input tensor KV;
                shape [total_seqs_kv, 2, num_heads, head_dim],
                where total_seqs_kv = cu_seqlens_kv[-1]
    o: torch.Tensor
                input tensor O (output of forward);
                shape [total_seqs_q, num_heads, head_dim], where total_seqs_q = cu_seqlens_q[-1]
    d_o: torch.Tensor
                input tensor dO (gradient of O);
                shape [total_seqs_q, num_heads, head_dim], where total_seqs_q = cu_seqlens_q[-1]
    qkv_dtype: tex.DType
                data type of QKV; in tex.DType, not torch.dtype
    aux_ctx_tensors: List[torch.Tensor]
                auxiliary output tensors of the forward pass when its is_training is True,
                e.g. aux_ctx_tensors = [M, ZInv, rng_state]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
    d_scale_o: torch.Tensor, default = None
                input tensor for the dequantization of O in FP8 computations
    d_scale_do: torch.Tensor, default = None
                input tensor for the dequantization of dO in FP8 computations
    q_scale_s: torch.Tensor, default = None
                input tensor for the quantization of S in FP8 computations
    q_scale_dp: torch.Tensor, default = None
                input tensor for the quantization of dP in FP8 computations, P = Q * K.T
    q_scale_dqkv: torch.Tensor, default = None
                input tensor for the quantization of dQKV in FP8 computations
    amax_dp: torch.Tensor, default = None
                output tensor, amax of dP, used by the next iteration in FP8 computations,
                P = Q * K.T
    amax_dqkv: torch.Tensor, default = None
                output tensor, amax of dQKV, used by the next iteration in FP8 computations
    attn_scale: float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim) as the default
    dropout: float, default = 0.0
                dropout probability, 0.0 means no dropout, 1.0 means no output;
                dropout must be 0.0 if is_training is False
    set_zero: bool, default = True
                if True, initializes the output tensor O to zero using the mha_fill method;
                if False, doesn't initialize O after its allocation
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "no_mask"}

    Returns
    ----------
    d_q: torch.Tensor
                gradient tensor of Q; same data type and shape as Q
    d_kv: torch.Tensor
                gradient tensor of KV; same data type and shape as KV
    d_bias: torch.Tensor, optional
                gradient tensor of Bias when bias_type is "pre_scale_bias" or "post_scale_bias";
                same data type and shape as Bias
    """

    check_cu_seqlens(cu_seqlens_q)
    check_cu_seqlens(cu_seqlens_kv)
    assert (cu_seqlens_q.numel() == cu_seqlens_kv.numel()
            ), "cu_seqlens_q and cu_seqlens_kv must have the same length."
    b = cu_seqlens_q.numel() - 1
    qkv_type = TORCH_DType[qkv_dtype]
    check_q(q, qkv_type)
    check_kv(kv, qkv_type)
    check_o(o, qkv_type)
    check_o(d_o, qkv_type)

    assert (q.size(1) == kv.size(2)
            and q.size(2) == kv.size(3)
            ), "Q and KV must have the same num_heads and head_dim."
    total_seqs_q = q.size(0)
    total_seqs_kv = q.size(0)
    h = q.size(1)
    d = q.size(2)

    if attn_scale is None:
        attn_scale = 1.0 / math.sqrt(d)

    assert (len(aux_ctx_tensors) >= 1
            ), "aux_ctx_tensors must contain rng_state as its last element."
    rng_state = aux_ctx_tensors[-1]
    check_rng_state(rng_state)

    # FP8 fused attention API
    if (qkv_type is torch.uint8) and (max_seqlen_q <= 512) and (max_seqlen_kv <= 512) \
            and d == 64:
        assert False, "The FP8 fused attention API currently only supports packed QKV input."

    ############### BF16/FP16 fused attention API from fmha_v2 ################
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) \
            and (max_seqlen_q > 512) and (max_seqlen_kv > 512):
        # add BF/FP16 support for >512 sequence length
        assert False, "The BF16/FP16 support for >512 sequence length is coming!"

    ############### BF16/FP16 fused attention API from fmha_v1 apex ################
    elif (qkv_type is torch.bfloat16 or qkv_type is torch.float16) \
            and (max_seqlen_q <= 512) and (max_seqlen_kv <= 512):
        # add BF/FP16 support for <=512 sequence length
        assert False, "The BF16/FP16 support for <=512 sequence length is coming!"

    else:
        assert False, "No support for this dtype and max_seqlen combination."

    # execute kernel
    output_tensors = tex.fused_attn_bwd_kvpacked(
            b, max_seqlen_q, max_seqlen_kv, total_seqs_q, total_seqs_kv, h, d,
            attn_scale, dropout, set_zero, qkv_layout, bias_type, attn_mask_type,
            cu_seqlens_q, cu_seqlens_kv,
            q, kv, o, d_o,
            qkv_dtype,
            aux_ctx_tensors,
            d_scale_qkv, d_scale_s, d_scale_o, d_scale_do,
            q_scale_s, q_scale_dp, q_scale_dqkv,
            amax_dp, amax_dqkv,
    )

    # returns (d_q, d_kv) when bias_type is no_bias; otherwise returns (d_q, d_kv, d_bias)
    if bias_type == "no_bias":
        return output_tensors[:2]
    return output_tensors

