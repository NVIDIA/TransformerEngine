# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for fused attention extensions"""
import math
from typing import Tuple, List, Union
import torch
import transformer_engine_extensions as tex
from transformer_engine_extensions import (
    NVTE_QKV_Layout,
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_Fused_Attn_Backend
)


__all__ = ['fused_attn_fwd_qkvpacked',
           'fused_attn_bwd_qkvpacked',
           'fused_attn_fwd_kvpacked',
           'fused_attn_bwd_kvpacked']


TORCH_DType = {
    tex.DType.kFloat8E4M3: torch.uint8,
    tex.DType.kFloat8E5M2: torch.uint8,
    tex.DType.kFloat16: torch.half,
    tex.DType.kBFloat16: torch.bfloat16,
    tex.DType.kFloat32: torch.float32,
    tex.DType.kInt32: torch.int32,
}

QKVLayout = {
    "not_interleaved": NVTE_QKV_Layout.NVTE_NOT_INTERLEAVED,
    "qkv_interleaved": NVTE_QKV_Layout.NVTE_QKV_INTERLEAVED,
    "kv_interleaved": NVTE_QKV_Layout.NVTE_KV_INTERLEAVED,
    }

AttnBiasType = {
    "no_bias": NVTE_Bias_Type.NVTE_NO_BIAS,
    "pre_scale_bias": NVTE_Bias_Type.NVTE_PRE_SCALE_BIAS,
    "post_scale_bias": NVTE_Bias_Type.NVTE_POST_SCALE_BIAS,
    }

AttnMaskType = {
    "no_mask": NVTE_Mask_Type.NVTE_NO_MASK,
    "padding": NVTE_Mask_Type.NVTE_PADDING_MASK,
    "causal": NVTE_Mask_Type.NVTE_CAUSAL_MASK,
    }

FusedAttnBackend = {
    "F16_max512_seqlen": NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen,
    "F16_arbitrary_seqlen": NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
    "FP8": NVTE_Fused_Attn_Backend.NVTE_FP8,
    "No_Backend": NVTE_Fused_Attn_Backend.NVTE_No_Backend,
    }

BACKEND_F16m512_FP8_THREADS_PER_CTA = 128
BACKEND_F16arb_ELTS_PER_THREADS = 16


def check_tensor(x: torch.Tensor):
    """Check tensor properties."""
    assert (x.is_cuda and x.is_contiguous()
            ), "Tensor should be a GPU tensor and contiguous."


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
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    attn_bias: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_o: torch.Tensor = None,
    amax_s: torch.Tensor = None,
    amax_o: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "qkv_interleaved",
    attn_bias_type: str = "no_bias",
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
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    attn_bias: torch.Tensor, default = None
                input tensor Bias when attn_bias_type is "pre_scale_bias" or "post_scale_bias";
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
    fast_zero_fill: bool, default = True
                if True, initializes the output tensor O to zero using the fast filling method;
                if False, uses PyTorch's .fill_() method
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    attn_bias_type: str, default = "no_bias"
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
                if is_training is True, aux_ctx_tensors = [softmax-related tensors, rng_state]
                if is_training is False, aux_ctx_tensors = None

                softmax-related tensors:
                    1. if fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]
                       softmax: torch.Tensor
                           Softmax(Q*K.T)
                           shape [batch_size, num_heads, max_seqlen, max_seqlen], dtype float32
                    2. if fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]
                       softmaxStats: torch.Tensor
                           log(sum(e^(x - max(x)))), where x=Q*K.T
                           shape [batch_size, num_heads, max_seqlen, 1], dtype float32
                    3. if fused_attention_backend == FusedAttnBackend["FP8"]
                       M: torch.Tensor
                           max(Q*K.T)
                           shape [batch_size, num_heads, max_seqlen, 1], dtype float32
                       ZInv: torch.Tensor
                           1/sum(e^(x - max(x))), where x=Q*K.T
                           shape [batch_size, num_heads, max_seqlen, 1], dtype float32
                rng_state: torch.Tensor, optional, if backend is not F16_max512_seqlen
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

    if attn_bias_type != "no_bias":
        assert (attn_bias is not None
                ), "attn_bias tensor cannot be None when attn_bias_type is not no_bias."
        assert (attn_bias.shape == torch.Size([1, h, max_seqlen, max_seqlen])
                ), "attn_bias tensor must be in [1, h, max_seqlen, max_seqlen] shape."
        assert (attn_bias.dtype == qkv.dtype
                ), "attn_bias tensor must be in the same dtype as qkv."

    assert (fused_attention_backend != FusedAttnBackend["No_Backend"]
            ), "Fused attention does not support this input combination."

    # BF16/FP16 fused attention API from fmha_v1 apex
    if fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]:
        rng_elts_per_thread = (max_seqlen * max_seqlen
                + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1)//BACKEND_F16m512_FP8_THREADS_PER_CTA

    # BF16/FP16 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
        rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS

    # FP8 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["FP8"]:
        rng_elts_per_thread = (max_seqlen * max_seqlen
                + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1)//BACKEND_F16m512_FP8_THREADS_PER_CTA

        assert (d_scale_qkv is not None
                ), "d_scale_qkv is required as an input for FP8 fused attention."
        assert (q_scale_s is not None
                ), "q_scale_s is required as an input for FP8 fused attention."
        assert (q_scale_o is not None
                ), "q_scale_o is required as an input for FP8 fused attention."
        assert (amax_s is not None
                ), "amax_s is required as an input for FP8 fused attention."
        assert (amax_o is not None
                ), "amax_o is required as an input for FP8 fused attention."
        check_scalar(d_scale_qkv)
        check_scalar(q_scale_s)
        check_scalar(q_scale_o)
        check_scalar(amax_s)
        check_scalar(amax_o)

    # execute kernel
    output_tensors = tex.fused_attn_fwd_qkvpacked(
            b, max_seqlen, total_seqs, h, d,
            is_training, attn_scale, dropout, fast_zero_fill,
            QKVLayout[qkv_layout], AttnBiasType[attn_bias_type], AttnMaskType[attn_mask_type],
            cu_seqlens, qkv, qkv_dtype,
            d_scale_qkv, q_scale_s, q_scale_o, amax_s, amax_o, attn_bias,
            rng_gen, rng_elts_per_thread,
    )

    # out, aux_ctx_tensors
    return output_tensors[0], output_tensors[1:]


def fused_attn_bwd_qkvpacked(
    max_seqlen: int,
    cu_seqlens: torch.Tensor,
    qkv: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    qkv_dtype: tex.DType,
    aux_ctx_tensors: List[torch.Tensor],
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
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
    fast_zero_fill: bool = True,
    qkv_layout: str = "qkv_interleaved",
    attn_bias_type: str = "no_bias",
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
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
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
    fast_zero_fill: bool, default = True
                if True, initializes the output tensor O to zero using the fast filling method;
                if False, uses PyTorch's .fill_() method
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    attn_bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "no_mask"}

    Returns
    ----------
    d_qkv: torch.Tensor
                gradient tensor of QKV; same data type and shape as QKV
    d_bias: torch.Tensor, optional
                gradient tensor of Bias when attn_bias_type is "pre_scale_bias"
                or "post_scale_bias"; same data type and shape as Bias
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

    assert (fused_attention_backend != FusedAttnBackend["No_Backend"]
            ), "Fused attention does not support this input combination."

    if fused_attention_backend != FusedAttnBackend["F16_max512_seqlen"]:
        assert (len(aux_ctx_tensors) >= 1
                ), "aux_ctx_tensors must contain rng_state as its last element."
        rng_state = aux_ctx_tensors[-1]
        check_rng_state(rng_state)

    if fused_attention_backend == FusedAttnBackend["FP8"]:
        assert (d_scale_qkv is not None), "d_scale_qkv is required for FP8 fused attention."
        assert (d_scale_s is not None), "d_scale_s is required for FP8 fused attention."
        assert (d_scale_o is not None), "d_scale_o is required for FP8 fused attention."
        assert (d_scale_do is not None), "d_scale_do is required for FP8 fused attention."
        assert (q_scale_s is not None), "q_scale_s is required for FP8 fused attention."
        assert (q_scale_dp is not None), "q_scale_dp is required for FP8 fused attention."
        assert (q_scale_dqkv is not None), "q_scale_dqkv is required for FP8 fused attention."
        assert (amax_dp is not None), "amax_dp is required for FP8 fused attention."
        assert (amax_dqkv is not None), "amax_dqkv is required for FP8 fused attention."
        assert (len(aux_ctx_tensors) == 3
                ), "aux_ctx_tensors is required to be [M, ZInv, rng_state] for FP8 fused attention."
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

    # execute kernel
    output_tensors = tex.fused_attn_bwd_qkvpacked(
            b, max_seqlen, total_seqs, h, d,
            attn_scale, dropout, fast_zero_fill,
            QKVLayout[qkv_layout], AttnBiasType[attn_bias_type], AttnMaskType[attn_mask_type],
            cu_seqlens, qkv, o, d_o, qkv_dtype, aux_ctx_tensors,
            d_scale_qkv, d_scale_s, d_scale_o, d_scale_do,
            q_scale_s, q_scale_dp, q_scale_dqkv, amax_dp, amax_dqkv,
    )

    if attn_bias_type == "no_bias":
        # return d_qkv when attn_bias_type is no_bias
        return output_tensors
    # otherwise return (d_qkv, d_bias)
    return output_tensors[0], output_tensors[1]


def fused_attn_fwd_kvpacked(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    kv: torch.Tensor,
    qkv_dtype: tex.DType,
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    attn_bias: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_o: torch.Tensor = None,
    amax_s: torch.Tensor = None,
    amax_o: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "qkv_interleaved",
    attn_bias_type: str = "no_bias",
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
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    attn_bias: torch.Tensor, default = None
                input tensor Bias when attn_bias_type is "pre_scale_bias" or "post_scale_bias";
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
    fast_zero_fill: bool, default = True
                if True, initializes the output tensor O to zero using the fast filling method;
                if False, uses PyTorch's .fill_() method
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    attn_bias_type: str, default = "no_bias"
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
                if is_training is True, aux_ctx_tensors = [softmax-related tensors, rng_state]
                if is_training is False, aux_ctx_tensors = None

                softmax-related tensors:
                    1. if fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]
                       softmax: torch.Tensor
                           Softmax(Q*K.T)
                           shape [batch_size, num_heads, max_seqlen_q, max_seqlen_kv], dtype float32
                    2. if fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]
                       softmaxStats: torch.Tensor
                           log(sum(e^(x - max(x)))), where x=Q*K.T
                           shape [batch_size, num_heads, max_seqlen_q, 1], dtype float32
                    3. if fused_attention_backend == FusedAttnBackend["FP8"]
                       M: torch.Tensor
                           max(Q*K.T)
                           shape [batch_size, num_heads, max_seqlen_q, 1], dtype float32
                       ZInv: torch.Tensor
                           1/sum(e^(x - max(x))), where x=Q*K.T
                           shape [batch_size, num_heads, max_seqlen_q, 1], dtype float32
                rng_state: torch.Tensor, optional, if backend is not F16_max512_seqlen
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

    if attn_bias_type != "no_bias":
        assert (attn_bias is not None
                ), "attn_bias tensor cannot be None when attn_bias_type is not no_bias."
        assert (attn_bias.shape == torch.Size([1, h, max_seqlen_q, max_seqlen_kv])
                ), "attn_bias tensor must be in [1, h, max_seqlen_q, max_seqlen_kv] shape."
        assert (attn_bias.dtype == q.dtype
                ), "attn_bias tensor must be in the same dtype as q and kv."

    assert (fused_attention_backend != FusedAttnBackend["No_Backend"]
            ), "Fused attention does not support this input combination."

    # BF16/FP16 fused attention API from fmha_v1 apex
    if fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]:
        rng_elts_per_thread = (max_seqlen_q * max_seqlen_kv
                + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1)//BACKEND_F16m512_FP8_THREADS_PER_CTA

    # BF16/FP16 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
        rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS

    # FP8 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["FP8"]:
        rng_elts_per_thread = (max_seqlen_q * max_seqlen_q
                + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1)//BACKEND_F16m512_FP8_THREADS_PER_CTA

    # execute kernel
    output_tensors = tex.fused_attn_fwd_kvpacked(
            b, max_seqlen_q, max_seqlen_kv, total_seqs_q, total_seqs_kv, h, d,
            is_training, attn_scale, dropout, fast_zero_fill,
            QKVLayout[qkv_layout], AttnBiasType[attn_bias_type], AttnMaskType[attn_mask_type],
            cu_seqlens_q, cu_seqlens_kv, q, kv, qkv_dtype,
            d_scale_qkv, q_scale_s, q_scale_o, amax_s, amax_o,
            attn_bias, rng_gen, rng_elts_per_thread,
    )

    # out, aux_ctx_tensors
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
    aux_ctx_tensors: List[torch.Tensor],
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
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
    fast_zero_fill: bool = True,
    qkv_layout: str = "qkv_interleaved",
    attn_bias_type: str = "no_bias",
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
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
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
    fast_zero_fill: bool, default = True
                if True, initializes the output tensor O to zero using the fast filling method;
                if False, uses PyTorch's .fill_() method
    qkv_layout: str, default = "qkv_interleaved"
                layout of QKV; {"qkv_interleaved", "kv_interleaved", "not_interleaved"}
    attn_bias_type: str, default = "no_bias"
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
                gradient tensor of Bias when attn_bias_type is "pre_scale_bias"
                or "post_scale_bias"; same data type and shape as Bias
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

    assert (fused_attention_backend != FusedAttnBackend["No_Backend"]
            ), "Fused attention does not support this input combination."

    if fused_attention_backend != FusedAttnBackend["F16_max512_seqlen"]:
        assert (len(aux_ctx_tensors) >= 1
                ), "aux_ctx_tensors must contain rng_state as its last element."
        rng_state = aux_ctx_tensors[-1]
        check_rng_state(rng_state)

    if fused_attention_backend == FusedAttnBackend["FP8"]:
        assert (d_scale_qkv is not None), "d_scale_qkv is required for FP8 fused attention."
        assert (d_scale_s is not None), "d_scale_s is required for FP8 fused attention."
        assert (d_scale_o is not None), "d_scale_o is required for FP8 fused attention."
        assert (d_scale_do is not None), "d_scale_do is required for FP8 fused attention."
        assert (q_scale_s is not None), "q_scale_s is required for FP8 fused attention."
        assert (q_scale_dp is not None), "q_scale_dp is required for FP8 fused attention."
        assert (q_scale_dqkv is not None), "q_scale_dqkv is required for FP8 fused attention."
        assert (amax_dp is not None), "amax_dp is required for FP8 fused attention."
        assert (amax_dqkv is not None), "amax_dqkv is required for FP8 fused attention."
        assert (len(aux_ctx_tensors) == 3
                ), "aux_ctx_tensors is required to be [M, ZInv, rng_state] for FP8 fused attention."
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
        check_stats(m, b, h, max_seqlen_q)
        check_stats(z_inv, b, h, max_seqlen_q)

    # execute kernel
    output_tensors = tex.fused_attn_bwd_kvpacked(
            b, max_seqlen_q, max_seqlen_kv, total_seqs_q, total_seqs_kv, h, d,
            attn_scale, dropout, fast_zero_fill,
            QKVLayout[qkv_layout], AttnBiasType[attn_bias_type], AttnMaskType[attn_mask_type],
            cu_seqlens_q, cu_seqlens_kv, q, kv, o, d_o, qkv_dtype, aux_ctx_tensors,
            d_scale_qkv, d_scale_s, d_scale_o, d_scale_do,
            q_scale_s, q_scale_dp, q_scale_dqkv, amax_dp, amax_dqkv,
    )

    if attn_bias_type == "no_bias":
        # return (d_q, d_kv) when attn_bias_type is no_bias
        return output_tensors
    # otherwise return (d_q, d_kv), d_bias
    return output_tensors[:2], output_tensors[2]
