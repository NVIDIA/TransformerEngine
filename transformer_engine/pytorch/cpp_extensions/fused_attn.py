# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Python interface for fused attention extensions"""
import math
from typing import Tuple, List, Union, Optional
import torch
import transformer_engine_torch as tex
from transformer_engine_torch import (
    NVTE_QKV_Layout,
    NVTE_QKV_Format,
    NVTE_Bias_Type,
    NVTE_Mask_Type,
    NVTE_Softmax_Type,
    NVTE_Fused_Attn_Backend,
)
from ..quantized_tensor import Quantizer


__all__ = [
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

QKVFormat = {
    "bshd": NVTE_QKV_Format.NVTE_BSHD,
    "sbhd": NVTE_QKV_Format.NVTE_SBHD,
    "thd": NVTE_QKV_Format.NVTE_THD,
    "sbhd_2bshd": NVTE_QKV_Format.NVTE_SBHD_2BSHD,
    "bshd_2sbhd": NVTE_QKV_Format.NVTE_BSHD_2SBHD,
    "thd_2bshd": NVTE_QKV_Format.NVTE_THD_2BSHD,
    "thd_2sbhd": NVTE_QKV_Format.NVTE_THD_2SBHD,
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
    "sbhd_bshd_bshd": NVTE_QKV_Layout.NVTE_SBHD_BSHD_BSHD,
    "bshd_sbhd_sbhd": NVTE_QKV_Layout.NVTE_BSHD_SBHD_SBHD,
    "thd_bshd_bshd": NVTE_QKV_Layout.NVTE_THD_BSHD_BSHD,
    "thd_sbhd_sbhd": NVTE_QKV_Layout.NVTE_THD_SBHD_SBHD,
    "paged_kv_bshd_bshd_bshd": NVTE_QKV_Layout.NVTE_Paged_KV_BSHD_BSHD_BSHD,
    "paged_kv_bshd_sbhd_sbhd": NVTE_QKV_Layout.NVTE_Paged_KV_BSHD_SBHD_SBHD,
    "paged_kv_sbhd_bshd_bshd": NVTE_QKV_Layout.NVTE_Paged_KV_SBHD_BSHD_BSHD,
    "paged_kv_sbhd_sbhd_sbhd": NVTE_QKV_Layout.NVTE_Paged_KV_SBHD_SBHD_SBHD,
    "paged_kv_thd_bshd_bshd": NVTE_QKV_Layout.NVTE_Paged_KV_THD_BSHD_BSHD,
    "paged_kv_thd_sbhd_sbhd": NVTE_QKV_Layout.NVTE_Paged_KV_THD_SBHD_SBHD,
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
    "causal_bottom_right": NVTE_Mask_Type.NVTE_CAUSAL_BOTTOM_RIGHT_MASK,
    "padding_causal_bottom_right": NVTE_Mask_Type.NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK,
}

SoftmaxType = {
    "vanilla": NVTE_Softmax_Type.NVTE_VANILLA_SOFTMAX,
    "off-by-one": NVTE_Softmax_Type.NVTE_OFF_BY_ONE_SOFTMAX,
    "learnable": NVTE_Softmax_Type.NVTE_LEARNABLE_SOFTMAX,
}

FusedAttnBackend = {
    "F16_max512_seqlen": NVTE_Fused_Attn_Backend.NVTE_F16_max512_seqlen,
    "F16_arbitrary_seqlen": NVTE_Fused_Attn_Backend.NVTE_F16_arbitrary_seqlen,
    "FP8": NVTE_Fused_Attn_Backend.NVTE_FP8,
    "No_Backend": NVTE_Fused_Attn_Backend.NVTE_No_Backend,
}

BACKEND_F16m512_FP8_THREADS_PER_CTA = 128
BACKEND_F16arb_ELTS_PER_THREADS = 16

META_QKV = tex.FP8FwdTensors.GEMM1_OUTPUT
META_DQKV = tex.FP8BwdTensors.GRAD_OUTPUT1
META_O = tex.FP8FwdTensors.GEMM2_INPUT
META_DO = tex.FP8BwdTensors.GRAD_INPUT2
META_S = tex.FP8FwdTensors.GEMM3_OUTPUT
META_DP = tex.FP8BwdTensors.GRAD_INPUT3


def fused_attn_fwd(
    is_training: bool,
    max_seqlen_q: int,
    max_seqlen_kv: int,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_kv: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    fake_dtype: torch.dtype,
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    attn_bias: torch.Tensor = None,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    page_table_k: torch.Tensor = None,
    page_table_v: torch.Tensor = None,
    s_quantizer: Quantizer = None,
    o_quantizer: Quantizer = None,
    attn_scale: float = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    window_size: Tuple[int, int] = (-1, -1),
    rng_gen: torch.Generator = None,
    softmax_offset: torch.Tensor = None,
    return_max_logit: bool = False,
    cuda_graph: bool = False,
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
    fake_dtype: tex.DType
                data type of Q, K and V - in case of high precision, fake dtype in case of FP8;
                in torch.dtype
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    attn_bias: torch.Tensor, default = None
                input tensor Bias when attn_bias_type is "pre_scale_bias" or "post_scale_bias";
                shape [1, num_heads, max_seqlen_q, max_seqlen_kv], same data type as q, k and v
    cu_seqlens_q_padded: torch.Tensor, default = None
                cumulative sequence offsets for Q; shape [batch_size + 1]
    cu_seqlens_kv_padded: torch.Tensor, default = None
                cumulative sequence offsets for KV; shape [batch_size + 1]
    page_table_k: torch.Tensor, default = None
                page table for K cache; shape [batch_size, max_pages_per_seq_k]
    page_table_v: torch.Tensor, default = None
                page table for V cache; shape [batch_size, max_pages_per_seq_v]
    s_quantizer: Quantizer, default = None
                Quantizer object for the intermediate value S.
    o_quantizer: Quantizer, default = None
                Quantizer object for the output of the attention.
    attn_scale: float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim_qk) as the default
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
    softmax_type: str, default = "vanilla"
                type of the attention softmax; {"vanilla", "off-by-one", "learnable"}
    window_size: Tuple[int, int], default = (-1, -1)
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically.
    rng_gen: torch.Generator, default = None
                random number generator;
                if None, uses the default CUDA generator from PyTorch; otherwise, uses rng_gen
    softmax_offset: torch.Tensor, default = None
                softmax offset tensor in shape [1, h_q, 1, 1].
                See softmax_type in DotProductAttention for details.
    return_max_logit: bool, default = False
                      whether to return the maximum attention score
    cuda_graph: bool, default = False
                whether or not cuda graph capture is enabled.

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
    max_logit: if return_max_logit = True, shape [h] and same data type as O; otherwise None
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
    elif fused_attention_backend == FusedAttnBackend["F16_arbitrary_seqlen"]:
        rng_elts_per_thread = BACKEND_F16arb_ELTS_PER_THREADS
    # FP8 fused attention API from fmha_v2
    elif fused_attention_backend == FusedAttnBackend["FP8"]:
        rng_elts_per_thread = (
            max_seqlen_q * max_seqlen_q + BACKEND_F16m512_FP8_THREADS_PER_CTA - 1
        ) // BACKEND_F16m512_FP8_THREADS_PER_CTA

        assert (
            s_quantizer is not None
        ), "s_quantizer is required as an input for FP8 fused attention."
        assert (
            o_quantizer is not None
        ), "o_quantizer is required as an input for FP8 fused attention."
    else:
        raise ValueError(f"Unsupported backend {fused_attention_backend}")

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
        SoftmaxType[softmax_type],
        window_size,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        fake_dtype,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        page_table_k,
        page_table_v,
        s_quantizer,
        o_quantizer,
        attn_bias,
        softmax_offset,
        rng_gen,
        rng_elts_per_thread,
        return_max_logit,
        cuda_graph,
    )

    if return_max_logit:
        qkv_format = qkv_layout.replace("3", "").replace("2", "").split("_")[0]
        # thd:  output_tensors: out [tq, h, d],    Max [tq, h, 1],    Sum_Exp [tq, h, 1]
        # bshd: output_tensors: out [b, sq, h, d], Max [b, h, sq, 1], Sum_Exp [b, h, sq, 1]
        # sbhd: output_tensors: out [sq, b, h, d], Max [b, h, sq, 1], Sum_Exp [b, h, sq, 1]
        stats = output_tensors[1] + torch.log(output_tensors[2])
        amax_dims = (0, 2) if qkv_format == "thd" else (0, 2, 3)
        # Max -> max_logit [h]
        max_logit = torch.amax(output_tensors[1], dim=amax_dims).to(dtype=output_tensors[0].dtype)
        aux_ctx_tensors = [stats]
        aux_ctx_tensors.extend(output_tensors[3:])
        return output_tensors[0], aux_ctx_tensors, max_logit

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
    fake_dtype: torch.dtype,
    dqkv_dtype: tex.DType,
    aux_ctx_tensors: List[torch.Tensor],
    fused_attention_backend: tex.NVTE_Fused_Attn_Backend,
    cu_seqlens_q_padded: torch.Tensor = None,
    cu_seqlens_kv_padded: torch.Tensor = None,
    s_quantizer: Quantizer = None,
    dp_quantizer: Quantizer = None,
    dqkv_quantizer: Quantizer = None,
    attn_scale: Optional[float] = None,
    dropout: float = 0.0,
    fast_zero_fill: bool = True,
    qkv_layout: str = "sbh3d",
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    window_size: Tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    cuda_graph: bool = False,
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
    fake_dtype: tex.DType
                data type of Q, K and V - in case of high precision, fake dtype in case of FP8;
                in torch.dtype
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
    s_quantizer: Quantizer, default = None
                Quantizer object for the intermediate value S.
    dp_quantizer: Quantizer, default = None
                Quantizer object for the intermediate value dP.
    dqkv_quantizer: Quantizer, default = None
                Quantizer object for the output values of the fused_attn_bwd.
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
    softmax_type: str, default = "vanilla"
                type of the attention softmax; {"vanilla", "off-by-one", "learnable"}
    window_size: Tuple[int, int], default = (-1, -1)
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically.
    deterministic: bool, default = False
                whether to execute the backward pass with deterministic behaviours.
    cuda_graph: bool, default = False
                whether or not cuda graph capture is enabled.

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
    d_softmax_offset: torch.Tensor, optional
                gradient tensor of softmax offset in shape [1, h_q, 1, 1].
                See softmax_type in DotProductAttention for details.
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
        assert (
            s_quantizer is not None
        ), "s_quantizer is required as an input for FP8 fused attention backward."
        assert (
            dp_quantizer is not None
        ), "dp_quantizer is required as an input for FP8 fused attention backward."
        assert (
            dqkv_dtype is not None
        ), "dqkv_dtype is required as an input for FP8 fused attention backward."
        assert (
            len(aux_ctx_tensors) == 3
        ), "aux_ctx_tensors is required to be [M, ZInv, rng_state] for FP8 fused attention."

    output_tensors = tex.fused_attn_bwd(
        max_seqlen_q,
        max_seqlen_kv,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        SoftmaxType[softmax_type],
        window_size,
        deterministic,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        o,
        d_o,
        fake_dtype,
        dqkv_dtype,
        aux_ctx_tensors,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        s_quantizer,
        dp_quantizer,
        dqkv_quantizer,
        cuda_graph,
    )

    return output_tensors
