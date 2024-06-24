# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for fused attention extensions"""
import math
from typing import Tuple, List, Union
import torch
import transformer_engine_torch as tex
from transformer_engine_torch import (
    NVTE_QKV_Layout,
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_Fused_Attn_Backend,
)


__all__ = [
    "fused_attn_fwd_qkvpacked",
    "fused_attn_bwd_qkvpacked",
    "fused_attn_fwd_kvpacked",
    "fused_attn_bwd_kvpacked",
    "fused_attn_fwd",
    "fused_attn_bwd",
]


TORCH_DType = {
    tex.DType.kFloat8E4M3: torch.uint8,
    tex.DType.kFloat8E5M2: torch.uint8,
    tex.DType.kFloat16: torch.half,
    tex.DType.kBFloat16: torch.bfloat16,
    tex.DType.kFloat32: torch.float32,
    tex.DType.kInt32: torch.int32,
}

QKVLayout = {
    "sb3hd": NVTE_QKV_Layout.NVTE_SB3HD,
    "sbh3d": NVTE_QKV_Layout.NVTE_SBH3D,
    "sbhd_sb2hd": NVTE_QKV_Layout.NVTE_SBHD_SB2HD,
    "sbhd_sbh2d": NVTE_QKV_Layout.NVTE_SBHD_SBH2D,
    "sbhd_sbhd_sbhd": NVTE_QKV_Layout.NVTE_SBHD_SBHD_SBHD,
    "bs3hd": NVTE_QKV_Layout.NVTE_BS3HD,
    "bsh3d": NVTE_QKV_Layout.NVTE_BSH3D,
    "bshd_bs2hd": NVTE_QKV_Layout.NVTE_BSHD_BS2HD,
    "bshd_bsh2d": NVTE_QKV_Layout.NVTE_BSHD_BSH2D,
    "bshd_bshd_bshd": NVTE_QKV_Layout.NVTE_BSHD_BSHD_BSHD,
    "t3hd": NVTE_QKV_Layout.NVTE_T3HD,
    "th3d": NVTE_QKV_Layout.NVTE_TH3D,
    "thd_t2hd": NVTE_QKV_Layout.NVTE_THD_T2HD,
    "thd_th2d": NVTE_QKV_Layout.NVTE_THD_TH2D,
    "thd_thd_thd": NVTE_QKV_Layout.NVTE_THD_THD_THD,
}

AttnBiasType = {
    "no_bias": NVTE_Bias_Type.NVTE_NO_BIAS,
    "pre_scale_bias": NVTE_Bias_Type.NVTE_PRE_SCALE_BIAS,
    "post_scale_bias": NVTE_Bias_Type.NVTE_POST_SCALE_BIAS,
    "alibi": NVTE_Bias_Type.NVTE_ALIBI,
}

AttnMaskType = {
    "no_mask": NVTE_Mask_Type.NVTE_NO_MASK,
    "padding": NVTE_Mask_Type.NVTE_PADDING_MASK,
    "causal": NVTE_Mask_Type.NVTE_CAUSAL_MASK,
    "padding_causal": NVTE_Mask_Type.NVTE_PADDING_CAUSAL_MASK,
}

FusedAttnBackend = {
    "F16_max512_seqlen": NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen,
    "F16_arbitrary_seqlen": NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
    "FP8": NVTE_Fused_Attn_Backend.NVTE_FP8,
    "No_Backend": NVTE_Fused_Attn_Backend.NVTE_No_Backend,
}

BACKEND_F16m512_FP8_THREADS_PER_CTA = 128
BACKEND_F16arb_ELTS_PER_THREADS = 16


def fused_attn_fwd_qkvpacked(
    is_training: bool,
    max_seqlen: int,
    cu_seqlens: torch.Tensor,
    qkv: torch.Tensor,
    qkv_dtype: tex.DType,
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    attn_bias: torch.Tensor = None,
    cu_seqlens_padded: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_o: torch.Tensor = None,
    amax_s: torch.Tensor = None,
    amax_o: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
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
                max sequence length for QKV, used for padding; may be larger than max(seqlens),
                seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    cu_seqlens: torch.Tensor
                cumulative sequence lengths for QKV; shape [batch_size + 1]
    qkv: torch.Tensor
                input tensor QKV; shape 3hd or h3d (see `qkv_layout` for details)
    qkv_dtype: tex.DType
                data type of QKV; in tex.DType, not torch.dtype
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    attn_bias: torch.Tensor, default = None
                input tensor Bias when attn_bias_type is "pre_scale_bias" or "post_scale_bias";
                shape [1, num_heads, max_seqlen, max_seqlen], same data type as qkv
    cu_seqlens_padded: torch.Tensor, default = None
                cumulative sequence offsets for QKV; shape [batch_size + 1]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
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
    qkv_layout: str, default = "sbh3d"
                layout of QKV; {"sbh3d", "sb3hd", "bsh3d", "bs3hd", "th3d", "t3hd"}
    attn_bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}
    rng_gen: torch.Generator, default = None
                random number generator;
                if None, uses the default CUDA generator from PyTorch; otherwise, uses rng_gen

    Returns
    ----------
    o: torch.Tensor
                output tensor O, of the attention calculation; same data type as QKV;
                same shape as Q, i.e. thd, sbhd or bshd (see `qkv_layout` for details)
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

    if attn_scale is None:
        d = qkv.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    if attn_bias_type not in ["no_bias", "alibi"]:
        assert (
            attn_bias is not None
        ), "attn_bias tensor cannot be None when attn_bias_type is not no_bias or alibi."
        assert attn_bias.dtype == qkv.dtype, "attn_bias tensor must be in the same dtype as qkv."

    assert (
        fused_attention_backend != FusedAttnBackend["No_Backend"]
    ), "Fused attention does not support this input combination."

    # BF16/FP16 fused attention API from fmha_v1 apex
    if fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]:
        rng_elts_per_thread = (
            max_seqlen * max_seqlen + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1
        ) // BACKEND_F16m512_FP8_THREADS_PER_CTA

    # BF16/FP16 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
        rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS

    # FP8 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["FP8"]:
        rng_elts_per_thread = (
            max_seqlen * max_seqlen + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1
        ) // BACKEND_F16m512_FP8_THREADS_PER_CTA

        assert (
            d_scale_qkv is not None
        ), "d_scale_qkv is required as an input for FP8 fused attention."
        assert d_scale_s is not None, "q_scale_s is required as an input for FP8 fused attention."
        assert q_scale_s is not None, "q_scale_s is required as an input for FP8 fused attention."
        assert q_scale_o is not None, "q_scale_o is required as an input for FP8 fused attention."
        assert amax_s is not None, "amax_s is required as an input for FP8 fused attention."
        assert amax_o is not None, "amax_o is required as an input for FP8 fused attention."

    # execute kernel
    output_tensors = tex.fused_attn_fwd_qkvpacked(
        max_seqlen,
        is_training,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        cu_seqlens,
        qkv,
        qkv_dtype,
        cu_seqlens_padded,
        d_scale_qkv,
        d_scale_s,
        q_scale_s,
        q_scale_o,
        amax_s,
        amax_o,
        attn_bias,
        rng_gen,
        rng_elts_per_thread,
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
    dqkv_dtype: tex.DType,
    aux_ctx_tensors: List[torch.Tensor],
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    cu_seqlens_padded: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    d_scale_o: torch.Tensor = None,
    d_scale_do: torch.Tensor = None,
    d_scale_dp: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_dp: torch.Tensor = None,
    q_scale_dqkv: torch.Tensor = None,
    amax_dp: torch.Tensor = None,
    amax_dqkv: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention BWD for packed QKV input.

    Parameters
    ----------
    max_seqlen: int
                max sequence length for QKV, used for padding; may be larger than max(seqlens)
                seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    cu_seqlens: torch.Tensor
                cumulative sequence lengths for QKV; shape [batch_size + 1]
    qkv: torch.Tensor
                input tensor QKV; shape 3hd or h3d (see `qkv_layout` for details)
    o: torch.Tensor
                input tensor O (output of forward);
                same shape as Q, i.e. thd, sbhd or bshd (see `qkv_layout` for details)
    d_o: torch.Tensor
                input tensor dO (gradient of O);
                same shape as Q, i.e. thd, sbhd or bshd (see `qkv_layout` for details)
    qkv_dtype: tex.DType
                data type of QKV; in tex.DType, not torch.dtype
    dqkv_dtype: tex.DType
                data type of dQKV; in tex.DType, not torch.dtype
    aux_ctx_tensors: List[torch.Tensor]
                auxiliary output tensors of the forward pass when its is_training is True,
                e.g. aux_ctx_tensors = [M, ZInv, rng_state]
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    cu_seqlens_padded: torch.Tensor, default = None
                cumulative sequence offsets for QKV; shape [batch_size + 1]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
    d_scale_o: torch.Tensor, default = None
                input tensor for the dequantization of O in FP8 computations
    d_scale_do: torch.Tensor, default = None
                input tensor for the dequantization of dO in FP8 computations
    d_scale_dp: torch.Tensor, default = None
                input tensor for the dequantization of dP in FP8 computations
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
    qkv_layout: str, default = "sbh3d"
                layout of QKV; {"sbh3d", "sb3hd", "bsh3d", "bs3hd", "th3d", "t3hd"}
    attn_bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}

    Returns
    ----------
    d_qkv: torch.Tensor
                gradient tensor of QKV; same data type and shape as QKV
    d_bias: torch.Tensor, optional
                gradient tensor of Bias when attn_bias_type is "pre_scale_bias"
                or "post_scale_bias"; same data type and shape as Bias
    """

    if attn_scale is None:
        d = qkv.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    assert (
        fused_attention_backend != FusedAttnBackend["No_Backend"]
    ), "Fused attention does not support this input combination."

    if fused_attention_backend != FusedAttnBackend["F16_max512_seqlen"]:
        assert (
            len(aux_ctx_tensors) >= 1
        ), "aux_ctx_tensors must contain rng_state as its last element."

    if fused_attention_backend == FusedAttnBackend["FP8"]:
        assert d_scale_qkv is not None, "d_scale_qkv is required for FP8 fused attention."
        assert d_scale_s is not None, "d_scale_s is required for FP8 fused attention."
        assert d_scale_o is not None, "d_scale_o is required for FP8 fused attention."
        assert d_scale_do is not None, "d_scale_do is required for FP8 fused attention."
        assert d_scale_dp is not None, "d_scale_dp is required for FP8 fused attention."
        assert q_scale_s is not None, "q_scale_s is required for FP8 fused attention."
        assert q_scale_dp is not None, "q_scale_dp is required for FP8 fused attention."
        assert q_scale_dqkv is not None, "q_scale_dqkv is required for FP8 fused attention."
        assert amax_dp is not None, "amax_dp is required for FP8 fused attention."
        assert amax_dqkv is not None, "amax_dqkv is required for FP8 fused attention."
        assert (
            len(aux_ctx_tensors) == 3
        ), "aux_ctx_tensors is required to be [M, ZInv, rng_state] for FP8 fused attention."

    # execute kernel
    output_tensors = tex.fused_attn_bwd_qkvpacked(
        max_seqlen,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        cu_seqlens,
        qkv,
        o,
        d_o,
        qkv_dtype,
        dqkv_dtype,
        aux_ctx_tensors,
        cu_seqlens_padded,
        d_scale_qkv,
        d_scale_s,
        d_scale_o,
        d_scale_do,
        d_scale_dp,
        q_scale_s,
        q_scale_dp,
        q_scale_dqkv,
        amax_dp,
        amax_dqkv,
    )

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
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    attn_bias: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_o: torch.Tensor = None,
    amax_s: torch.Tensor = None,
    amax_o: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbhd_sbh2d",
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
                max sequence length for Q, used for padding; may be larger than max(seqlens_q),
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_kv: int
                max sequence length for KV, used for padding; may be larger than max(seqlens_kv),
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    cu_seqlens_q: torch.Tensor
                cumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv: torch.Tensor
                cumulative sequence lengths for KV; shape [batch_size + 1]
    q: torch.Tensor
                input tensor Q; shape thd, sbhd or bshd (see `qkv_layout` for details)
    kv: torch.Tensor
                packed input tensor KV; shape 2hd or h2d (see `qkv_layout` for details)
    qkv_dtype: tex.DType
                data type of Q and KV; in tex.DType, not torch.dtype
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    attn_bias: torch.Tensor, default = None
                input tensor Bias when attn_bias_type is "pre_scale_bias" or "post_scale_bias";
                shape [1, num_heads, max_seqlen_q, max_seqlen_kv], same data type as q and kv
    cu_seqlens_q_padded: torch.Tensor, default = None
                cumulative sequence offsets for Q; shape [batch_size + 1]
    cu_seqlens_kv_padded: torch.Tensor, default = None
                cumulative sequence offsets for KV; shape [batch_size + 1]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
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
    qkv_layout: str, default = "sbhd_sbh2d"
                layout of QKV;
                {"sbhd_sbh2d", "sbhd_sb2hd", "bshd_bsh2d", "bshd_bs2hd", "thd_th2d", "thd_t2hd"}
    attn_bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}
    rng_gen: torch.Generator, default = None
                random number generator;
                if None, uses the default CUDA generator from PyTorch; otherwise, uses rng_gen

    Returns
    ----------
    o: torch.Tensor
                output tensor O, of the attention calculation; same data type as QKV;
                same shape as Q, i.e. thd, sbhd or bshd (see `qkv_layout` for details)
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

    if attn_scale is None:
        d = q.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    if attn_bias_type not in ["no_bias", "alibi"]:
        assert (
            attn_bias is not None
        ), "attn_bias tensor cannot be None when attn_bias_type is not no_bias or alibi."
        assert attn_bias.dtype == q.dtype, "attn_bias tensor must be in the same dtype as q and kv."

    assert (
        fused_attention_backend != FusedAttnBackend["No_Backend"]
    ), "Fused attention does not support this input combination."

    # BF16/FP16 fused attention API from fmha_v1 apex
    if fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]:
        rng_elts_per_thread = (
            max_seqlen_q * max_seqlen_kv + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1
        ) // BACKEND_F16m512_FP8_THREADS_PER_CTA

    # BF16/FP16 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
        rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS

    # FP8 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["FP8"]:
        rng_elts_per_thread = (
            max_seqlen_q * max_seqlen_q + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1
        ) // BACKEND_F16m512_FP8_THREADS_PER_CTA

        assert (
            d_scale_qkv is not None
        ), "d_scale_qkv is required as an input for FP8 fused attention."
        assert d_scale_s is not None, "q_scale_s is required as an input for FP8 fused attention."
        assert q_scale_s is not None, "q_scale_s is required as an input for FP8 fused attention."
        assert q_scale_o is not None, "q_scale_o is required as an input for FP8 fused attention."
        assert amax_s is not None, "amax_s is required as an input for FP8 fused attention."
        assert amax_o is not None, "amax_o is required as an input for FP8 fused attention."

    # execute kernel
    output_tensors = tex.fused_attn_fwd_kvpacked(
        max_seqlen_q,
        max_seqlen_kv,
        is_training,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        kv,
        qkv_dtype,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        d_scale_qkv,
        d_scale_s,
        q_scale_s,
        q_scale_o,
        amax_s,
        amax_o,
        attn_bias,
        rng_gen,
        rng_elts_per_thread,
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
    dqkv_dtype: tex.DType,
    aux_ctx_tensors: List[torch.Tensor],
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    d_scale_o: torch.Tensor = None,
    d_scale_do: torch.Tensor = None,
    d_scale_dp: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_dp: torch.Tensor = None,
    q_scale_dqkv: torch.Tensor = None,
    amax_dp: torch.Tensor = None,
    amax_dqkv: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbhd_sbh2d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention BWD for packed KV input.

    Parameters
    ----------
    max_seqlen_q: int
                max sequence length for Q, used for padding; may be larger than max(seqlens_q),
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_kv: int
                max sequence length for KV, used for padding; may be larger than max(seqlens_kv),
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    cu_seqlens_q: torch.Tensor
                cumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv: torch.Tensor
                cumulative sequence lengths for KV; shape [batch_size + 1]
    q: torch.Tensor
                input tensor Q; shape thd, sbhd or bshd (see `qkv_layout` for details)
    kv: torch.Tensor
                packed input tensor KV; shape h2d or 2hd (see `qkv_layout` for details)
    o: torch.Tensor
                input tensor O (output of forward);
                same shape as Q, i.e. thd, sbhd or bshd (see `qkv_layout` for details)
    d_o: torch.Tensor
                input tensor dO (gradient of O);
                same shape as Q, i.e. thd, sbhd or bshd (see `qkv_layout` for details)
    qkv_dtype: tex.DType
                data type of Q and KV; in tex.DType, not torch.dtype
    dqkv_dtype: tex.DType
                data type of dQ and dKV; in tex.DType, not torch.dtype
    aux_ctx_tensors: List[torch.Tensor]
                auxiliary output tensors of the forward pass when its is_training is True,
                e.g. aux_ctx_tensors = [M, ZInv, rng_state]
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    cu_seqlens_q_padded: torch.Tensor, default = None
                cumulative sequence offsets for Q; shape [batch_size + 1]
    cu_seqlens_kv_padded: torch.Tensor, default = None
                cumulative sequence offsets for KV; shape [batch_size + 1]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of QKV in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
    d_scale_o: torch.Tensor, default = None
                input tensor for the dequantization of O in FP8 computations
    d_scale_do: torch.Tensor, default = None
                input tensor for the dequantization of dO in FP8 computations
    d_scale_dp: torch.Tensor, default = None
                input tensor for the dequantization of dP in FP8 computations
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
    qkv_layout: str, default = "sbhd_sbh2d"
                layout of QKV;
                {"sbhd_sbh2d", "sbhd_sb2hd", "bshd_bsh2d", "bshd_bs2hd", "thd_th2d", "thd_t2hd"}
    attn_bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}

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

    if attn_scale is None:
        d = q.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    assert (
        fused_attention_backend != FusedAttnBackend["No_Backend"]
    ), "Fused attention does not support this input combination."

    if fused_attention_backend != FusedAttnBackend["F16_max512_seqlen"]:
        assert (
            len(aux_ctx_tensors) >= 1
        ), "aux_ctx_tensors must contain rng_state as its last element."

    if fused_attention_backend == FusedAttnBackend["FP8"]:
        assert d_scale_qkv is not None, "d_scale_qkv is required for FP8 fused attention."
        assert d_scale_s is not None, "d_scale_s is required for FP8 fused attention."
        assert d_scale_o is not None, "d_scale_o is required for FP8 fused attention."
        assert d_scale_do is not None, "d_scale_do is required for FP8 fused attention."
        assert d_scale_dp is not None, "d_scale_dp is required for FP8 fused attention."
        assert q_scale_s is not None, "q_scale_s is required for FP8 fused attention."
        assert q_scale_dp is not None, "q_scale_dp is required for FP8 fused attention."
        assert q_scale_dqkv is not None, "q_scale_dqkv is required for FP8 fused attention."
        assert amax_dp is not None, "amax_dp is required for FP8 fused attention."
        assert amax_dqkv is not None, "amax_dqkv is required for FP8 fused attention."
        assert (
            len(aux_ctx_tensors) == 3
        ), "aux_ctx_tensors is required to be [M, ZInv, rng_state] for FP8 fused attention."

    # execute kernel
    output_tensors = tex.fused_attn_bwd_kvpacked(
        max_seqlen_q,
        max_seqlen_kv,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        kv,
        o,
        d_o,
        qkv_dtype,
        dqkv_dtype,
        aux_ctx_tensors,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        d_scale_qkv,
        d_scale_s,
        d_scale_o,
        d_scale_do,
        d_scale_dp,
        q_scale_s,
        q_scale_dp,
        q_scale_dqkv,
        amax_dp,
        amax_dqkv,
    )

    return output_tensors


def fused_attn_fwd(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qkv_dtype: tex.DType,
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    attn_bias: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_o: torch.Tensor = None,
    amax_s: torch.Tensor = None,
    amax_o: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    rng_gen: torch.Generator = None,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention FWD for separate QKV input.

    Parameters
    ----------
    is_training: bool
                if True, runs training and produces auxiliary tensors aux_ctx_tensors
                for the backward; if False, runs inference and doesn't produce aux_ctx_tensors
    max_seqlen_q: int
                max sequence length for Q, used for padding;
                may be larger than max(seqlens_q),
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_kv: int
                max sequence length for K and V, used for padding;
                may be larger than max(seqlens_kv),
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    cu_seqlens_q: torch.Tensor
                cumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv: torch.Tensor
                cumulative sequence lengths for K and V; shape [batch_size + 1]
    q: torch.Tensor
                input tensor Q; shape sbhd, bshd or thd (see `qkv_layout` for details)
    k: torch.Tensor
                input tensor K; shape sbhd, bshd or thd (see `qkv_layout` for details)
    v: torch.Tensor
                input tensor V; shape sbhd, bshd or thd (see `qkv_layout` for details)
    qkv_dtype: tex.DType
                data type of Q, K and V; in tex.DType, not torch.dtype
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    attn_bias: torch.Tensor, default = None
                input tensor Bias when attn_bias_type is "pre_scale_bias" or "post_scale_bias";
                shape [1, num_heads, max_seqlen_q, max_seqlen_kv], same data type as q, k and v
    cu_seqlens_q_padded: torch.Tensor, default = None
                cumulative sequence offsets for Q; shape [batch_size + 1]
    cu_seqlens_kv_padded: torch.Tensor, default = None
                cumulative sequence offsets for KV; shape [batch_size + 1]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of Q, K and V in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
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
    qkv_layout: str, default = "sbh3d"
                layout of Q, K and V;
                {"sb3hd", "sbh3d", "sbhd_sb2hd", "sbhd_sbh2d", "sbhd_sbhd_sbhd",
                "bs3hd", "bsh3d", "bshd_bs2hd", "bshd_bsh2d", "bshd_bshd_bshd",
                "t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd"}
    attn_bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}
    rng_gen: torch.Generator, default = None
                random number generator;
                if None, uses the default CUDA generator from PyTorch; otherwise, uses rng_gen

    Returns
    ----------
    o: torch.Tensor
                output tensor O, of the attention calculation; same data type as Q, K and V;
                same shape as Q
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

    if attn_scale is None:
        d = q.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    if attn_bias_type not in ["no_bias", "alibi"]:
        assert (
            attn_bias is not None
        ), "attn_bias tensor cannot be None when attn_bias_type is not no_bias or alibi."
        assert attn_bias.dtype == q.dtype, "attn_bias tensor must be in the same dtype as q and kv."

    assert (
        fused_attention_backend != FusedAttnBackend["No_Backend"]
    ), "Fused attention does not support this input combination."

    # BF16/FP16 fused attention API from fmha_v1 apex
    if fused_attention_backend == FusedAttnBackend["F16_max512_seqlen"]:
        rng_elts_per_thread = (
            max_seqlen_q * max_seqlen_kv + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1
        ) // BACKEND_F16m512_FP8_THREADS_PER_CTA

    # BF16/FP16 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
        rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS

    # FP8 fused attention API from fmha_v2
    if fused_attention_backend == FusedAttnBackend["FP8"]:
        rng_elts_per_thread = (
            max_seqlen_q * max_seqlen_q + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1
        ) // BACKEND_F16m512_FP8_THREADS_PER_CTA

        assert (
            d_scale_qkv is not None
        ), "d_scale_qkv is required as an input for FP8 fused attention."
        assert d_scale_s is not None, "q_scale_s is required as an input for FP8 fused attention."
        assert q_scale_s is not None, "q_scale_s is required as an input for FP8 fused attention."
        assert q_scale_o is not None, "q_scale_o is required as an input for FP8 fused attention."
        assert amax_s is not None, "amax_s is required as an input for FP8 fused attention."
        assert amax_o is not None, "amax_o is required as an input for FP8 fused attention."

    # execute kernel
    output_tensors = tex.fused_attn_fwd(
        max_seqlen_q,
        max_seqlen_kv,
        is_training,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        qkv_dtype,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        d_scale_qkv,
        d_scale_s,
        q_scale_s,
        q_scale_o,
        amax_s,
        amax_o,
        attn_bias,
        rng_gen,
        rng_elts_per_thread,
    )

    # out, aux_ctx_tensors
    return output_tensors[0], output_tensors[1:]


def fused_attn_bwd(
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    d_o: torch.Tensor,
    qkv_dtype: tex.DType,
    dqkv_dtype: tex.DType,
    aux_ctx_tensors: List[torch.Tensor],
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    d_scale_qkv: torch.Tensor = None,
    d_scale_s: torch.Tensor = None,
    d_scale_o: torch.Tensor = None,
    d_scale_do: torch.Tensor = None,
    d_scale_dp: torch.Tensor = None,
    q_scale_s: torch.Tensor = None,
    q_scale_dp: torch.Tensor = None,
    q_scale_dqkv: torch.Tensor = None,
    amax_dp: torch.Tensor = None,
    amax_dqkv: torch.Tensor = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention BWD for packed KV input.

    Parameters
    ----------
    max_seqlen_q: int
                max sequence length for Q, used for padding; may be larger than max(seqlens_q),
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_kv: int
                max sequence length for K and V, used for padding;
                may be larger than max(seqlens_kv),
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    cu_seqlens_q: torch.Tensor
                cumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv: torch.Tensor
                cumulative sequence lengths for K and V; shape [batch_size + 1]
    q: torch.Tensor
                input tensor Q; shape sbhd, bshd or thd (see `qkv_layout` for details)
    k: torch.Tensor
                input tensor K; shape sbhd, bshd or thd (see `qkv_layout` for details)
    v: torch.Tensor
                input tensor V; shape sbhd, bshd or thd (see `qkv_layout` for details)
    o: torch.Tensor
                input tensor O (output of forward); same data type as Q, K and V;
                same shape as Q
    d_o: torch.Tensor
                input tensor dO (gradient of O); same data type as Q, K and V;
                same shape as Q
    qkv_dtype: tex.DType
                data type of Q, K and V; in tex.DType, not torch.dtype
    dqkv_dtype: tex.DType
                data type of dQ, dK and dV; in tex.DType, not torch.dtype
    aux_ctx_tensors: List[torch.Tensor]
                auxiliary output tensors of the forward pass when its is_training is True,
                e.g. aux_ctx_tensors = [M, ZInv, rng_state]
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    cu_seqlens_q_padded: torch.Tensor, default = None
                cumulative sequence offsets for Q; shape [batch_size + 1]
    cu_seqlens_kv_padded: torch.Tensor, default = None
                cumulative sequence offsets for KV; shape [batch_size + 1]
    d_scale_qkv: torch.Tensor, default = None
                input tensor for the dequantization of Q, K and V in FP8 computations
    d_scale_s: torch.Tensor, default = None
                input tensor for the dequantization of S in FP8 computations, S = Softmax(Q * K.T)
    d_scale_o: torch.Tensor, default = None
                input tensor for the dequantization of O in FP8 computations
    d_scale_do: torch.Tensor, default = None
                input tensor for the dequantization of dO in FP8 computations
    d_scale_dp: torch.Tensor, default = None
                input tensor for the dequantization of dP in FP8 computations
    q_scale_s: torch.Tensor, default = None
                input tensor for the quantization of S in FP8 computations
    q_scale_dp: torch.Tensor, default = None
                input tensor for the quantization of dP in FP8 computations, P = Q * K.T
    q_scale_dqkv: torch.Tensor, default = None
                input tensor for the quantization of dQ, dK and dV in FP8 computations
    amax_dp: torch.Tensor, default = None
                output tensor, amax of dP, used by the next iteration in FP8 computations,
                P = Q * K.T
    amax_dqkv: torch.Tensor, default = None
                output tensor, amax of dQ, dK and dV, used by the next iteration in FP8 computations
    attn_scale: float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim) as the default
    dropout: float, default = 0.0
                dropout probability, 0.0 means no dropout, 1.0 means no output;
                dropout must be 0.0 if is_training is False
    fast_zero_fill: bool, default = True
                if True, initializes the output tensor O to zero using the fast filling method;
                if False, uses PyTorch's .fill_() method
    qkv_layout: str, default = "sbh3d"
                layout of Q, K and V;
                {"sb3hd", "sbh3d", "sbhd_sb2hd", "sbhd_sbh2d", "sbhd_sbhd_sbhd",
                "bs3hd", "bsh3d", "bshd_bs2hd", "bshd_bsh2d", "bshd_bshd_bshd",
                "t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd"}
    attn_bias_type: str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type: str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}

    Returns
    ----------
    d_q: torch.Tensor
                gradient tensor of Q; same data type and shape as Q
    d_k: torch.Tensor
                gradient tensor of K; same data type and shape as K
    d_v: torch.Tensor
                gradient tensor of V; same data type and shape as V
    d_bias: torch.Tensor, optional
                gradient tensor of Bias when attn_bias_type is "pre_scale_bias"
                or "post_scale_bias"; same data type and shape as Bias
    """

    if attn_scale is None:
        d = q.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    assert (
        fused_attention_backend != FusedAttnBackend["No_Backend"]
    ), "Fused attention does not support this input combination."

    if fused_attention_backend != FusedAttnBackend["F16_max512_seqlen"]:
        assert (
            len(aux_ctx_tensors) >= 1
        ), "aux_ctx_tensors must contain rng_state as its last element."

    if fused_attention_backend == FusedAttnBackend["FP8"]:
        assert d_scale_qkv is not None, "d_scale_qkv is required for FP8 fused attention."
        assert d_scale_s is not None, "d_scale_s is required for FP8 fused attention."
        assert d_scale_o is not None, "d_scale_o is required for FP8 fused attention."
        assert d_scale_do is not None, "d_scale_do is required for FP8 fused attention."
        assert d_scale_dp is not None, "d_scale_dp is required for FP8 fused attention."
        assert q_scale_s is not None, "q_scale_s is required for FP8 fused attention."
        assert q_scale_dp is not None, "q_scale_dp is required for FP8 fused attention."
        assert q_scale_dqkv is not None, "q_scale_dqkv is required for FP8 fused attention."
        assert amax_dp is not None, "amax_dp is required for FP8 fused attention."
        assert amax_dqkv is not None, "amax_dqkv is required for FP8 fused attention."
        assert (
            len(aux_ctx_tensors) == 3
        ), "aux_ctx_tensors is required to be [M, ZInv, rng_state] for FP8 fused attention."

    # execute kernel
    output_tensors = tex.fused_attn_bwd(
        max_seqlen_q,
        max_seqlen_kv,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        o,
        d_o,
        qkv_dtype,
        dqkv_dtype,
        aux_ctx_tensors,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        d_scale_qkv,
        d_scale_s,
        d_scale_o,
        d_scale_do,
        d_scale_dp,
        q_scale_s,
        q_scale_dp,
        q_scale_dqkv,
        amax_dp,
        amax_dqkv,
    )

    return output_tensors
