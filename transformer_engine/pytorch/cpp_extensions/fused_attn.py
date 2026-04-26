# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from ..constants import FP8BwdTensorIdx, FP8FwdTensorIdx


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
    None: NVTE_QKV_Format.NVTE_QKV_Format_NOT_SET,
    "bshd": NVTE_QKV_Format.NVTE_BSHD,
    "sbhd": NVTE_QKV_Format.NVTE_SBHD,
    "thd": NVTE_QKV_Format.NVTE_THD,
    "sbhd_2bshd": NVTE_QKV_Format.NVTE_SBHD_2BSHD,
    "bshd_2sbhd": NVTE_QKV_Format.NVTE_BSHD_2SBHD,
    "thd_2bshd": NVTE_QKV_Format.NVTE_THD_2BSHD,
    "thd_2sbhd": NVTE_QKV_Format.NVTE_THD_2SBHD,
    "bhsd": NVTE_QKV_Format.NVTE_BHSD,
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
    "bhsd_bhsd_bhsd": NVTE_QKV_Layout.NVTE_BHSD_BHSD_BHSD,
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

META_QKV = FP8FwdTensorIdx.GEMM1_OUTPUT
META_DQKV = FP8BwdTensorIdx.GRAD_OUTPUT1
META_O = FP8FwdTensorIdx.GEMM2_INPUT
META_DO = FP8BwdTensorIdx.GRAD_INPUT2
META_S = FP8FwdTensorIdx.GEMM3_OUTPUT
META_DP = FP8BwdTensorIdx.GRAD_INPUT3


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
    o_format: str = "sbhd",
    qkv_scale_inv_format: str = None,
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    window_size: Tuple[int, int] = (-1, -1),
    bottom_right_diagonal: bool = None,
    rng_gen: torch.Generator = None,
    softmax_offset: torch.Tensor = None,
    return_max_logit: bool = False,
    cuda_graph: bool = False,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention FWD for separate QKV input.

    Parameters
    ----------
    is_training : bool
                if True, runs training and produces auxiliary tensors aux_ctx_tensors
                for the backward; if False, runs inference and doesn't produce aux_ctx_tensors
    max_seqlen_q : int
                max sequence length for Q, used for padding;
                may be larger than max(seqlens_q),
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_kv : int
                max sequence length for K and V, used for padding;
                may be larger than max(seqlens_kv),
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    cu_seqlens_q : torch.Tensor
                cumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv : torch.Tensor
                cumulative sequence lengths for K and V; shape [batch_size + 1]
    q : torch.Tensor
                input tensor Q; shape sbhd, bshd or thd (see `qkv_layout` for details)
    k : torch.Tensor
                input tensor K; shape sbhd, bshd or thd (see `qkv_layout` for details)
    v : torch.Tensor
                input tensor V; shape sbhd, bshd or thd (see `qkv_layout` for details)
    fake_dtype : tex.DType
                data type of Q, K and V - in case of high precision, fake dtype in case of FP8;
                in torch.dtype
    fused_attention_backend : tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    attn_bias : torch.Tensor, default = None
                input tensor Bias when attn_bias_type is "pre_scale_bias" or "post_scale_bias";
                shape [1, num_heads, max_seqlen_q, max_seqlen_kv], same data type as q, k and v
    cu_seqlens_q_padded : torch.Tensor, default = None
                cumulative sequence offsets for Q; shape [batch_size + 1]
    cu_seqlens_kv_padded : torch.Tensor, default = None
                cumulative sequence offsets for KV; shape [batch_size + 1]
    page_table_k : torch.Tensor, default = None
                page table for K cache; shape [batch_size, max_pages_per_seq_k]
    page_table_v : torch.Tensor, default = None
                page table for V cache; shape [batch_size, max_pages_per_seq_v]
    s_quantizer : Quantizer, default = None
                Quantizer object for the intermediate value S.
    o_quantizer : Quantizer, default = None
                Quantizer object for the output of the attention.
    attn_scale : float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim_qk) as the default
    dropout : float, default = 0.0
                dropout probability, 0.0 means no dropout, 1.0 means no output;
                dropout must be 0.0 if is_training is False
    fast_zero_fill : bool, default = True
                if True, initializes the output tensor O to zero using the fast filling method;
                if False, uses PyTorch's .fill_() method
    qkv_layout : str, default = "sbh3d"
                layout of Q, K and V;
                {"sb3hd", "sbh3d", "sbhd_sb2hd", "sbhd_sbh2d", "sbhd_sbhd_sbhd",
                "bs3hd", "bsh3d", "bshd_bs2hd", "bshd_bsh2d", "bshd_bshd_bshd",
                "t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd"}
    o_format : str, default = "sbhd"
                format of O; {"sbhd", "bshd", "thd"}
    qkv_scale_inv_format : str, default = None
                format of the scale-inverse tensors for QKV; {"sbhd", "bshd", "thd", "bhsd"};
                if None, defaults to the format inferred from qkv_layout.
    attn_bias_type : str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type : str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}
    softmax_type : str, default = "vanilla"
                type of the attention softmax; {"vanilla", "off-by-one", "learnable"}
    window_size : Tuple[int, int], default = (-1, -1)
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically.
    bottom_right_diagonal: bool, default = None
                whether to align sliding window and ALiBi diagonal to the top left (False) or
                bottom right (True) corner of the softmax matrix.
    rng_gen : torch.Generator, default = None
                random number generator;
                if None, uses the default CUDA generator from PyTorch; otherwise, uses rng_gen
    softmax_offset : torch.Tensor, default = None
                softmax offset tensor of shape [1, h_q, 1, 1].
                See softmax_type in DotProductAttention for details.
    return_max_logit : bool, default = False
                      whether to return the maximum attention score
    cuda_graph : bool, default = False
                whether or not cuda graph capture is enabled.

    Returns
    ----------
    o : torch.Tensor
                output tensor O, of the attention calculation; same data type as Q, K and V;
                same shape as Q
    aux_ctx_tensors : List[torch.Tensor]
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
                       ZInv: torch.Tensor, only allocated for T3HD path
                           1/sum(e^(x - max(x))), where x=Q*K.T
                           shape [batch_size, num_heads, max_seqlen_q, 1], dtype float32
                rng_state: torch.Tensor, optional, if backend is not F16_max512_seqlen
                    state of the random number generator;
                    [seed, offset], dtype uint64
    max_logit : if return_max_logit = True, shape [h] and same data type as O; otherwise None
    """

    if bottom_right_diagonal is None:
        bottom_right_diagonal = attn_mask_type in {
            "causal_bottom_right",
            "padding_causal_bottom_right",
        }

    if attn_scale is None:
        d = q.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    if attn_bias_type not in ["no_bias", "alibi"]:
        if attn_bias is None:
            raise ValueError(
                f"attn_bias tensor cannot be None when attn_bias_type={attn_bias_type!r}."
            )
        if attn_bias.dtype != q.dtype:
            raise ValueError(
                "attn_bias tensor must have the same dtype as q and kv: "
                f"attn_bias.dtype={attn_bias.dtype} but q.dtype={q.dtype}."
            )

    if fused_attention_backend == FusedAttnBackend["No_Backend"]:
        raise ValueError(
            "Fused attention does not support this input combination:"
            f" qkv_layout={qkv_layout!r}, attn_bias_type={attn_bias_type!r},"
            f" attn_mask_type={attn_mask_type!r}, q.shape={list(q.shape)},"
            f" q.dtype={q.dtype}, backend={fused_attention_backend}."
        )

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
        QKVFormat[o_format],
        QKVFormat[qkv_scale_inv_format],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        SoftmaxType[softmax_type],
        window_size,
        bottom_right_diagonal,
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
        # thd (newer cuDNN runtimes, non-sm120): output_tensors: out [tq, h, d],    Stats [tq, h, 1],    Max [tq, h, 1]
        # thd (older cuDNN runtimes or sm120):   output_tensors: out [tq, h, d],    Stats [b, h, sq, 1], Max [b, h, sq, 1]
        # bshd:                                  output_tensors: out [b, sq, h, d], Stats [b, h, sq, 1], Max [b, h, sq, 1]
        # sbhd:                                  output_tensors: out [sq, b, h, d], Stats [b, h, sq, 1], Max [b, h, sq, 1]
        aux_ctx_tensors = [output_tensors[1]] + list(
            output_tensors[3:]
        )  # Stats + rng_state + optional tensors
        max_tensor = output_tensors[2]
        amax_dims = (0, 2) if max_tensor.ndim == 3 else (0, 2, 3)

        if qkv_format == "thd":
            if max_tensor.ndim == 4:
                # For THD on cuDNN <= 9.6 or THD on sm120, Max tensor can be [b, h, sq, 1]
                # with padded sequence positions. Exclude those padded positions when computing max_logit.
                seqlens_q = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(device=max_tensor.device)
                sq_idx = torch.arange(max_tensor.shape[2], device=max_tensor.device).view(
                    1, 1, -1, 1
                )
                valid = sq_idx < seqlens_q.view(-1, 1, 1, 1)
                max_tensor = max_tensor.masked_fill(~valid, float("-inf"))
            elif max_tensor.ndim == 3:
                if cu_seqlens_q_padded is not None:
                    # For THD + pad_between_seqs=True + non-sm120 + cuDNN>9.6, Max tensor is [tq, h, 1]
                    # and padding positions could be uninitialized. Exclude those padded positions when
                    # computing max_logit.
                    actual_seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).to(
                        device=max_tensor.device
                    )
                    padded_seqlens = (cu_seqlens_q_padded[1:] - cu_seqlens_q_padded[:-1]).to(
                        device=max_tensor.device
                    )
                    pad_lens = (padded_seqlens - actual_seqlens).to(device=max_tensor.device)
                    b = pad_lens.shape[0]

                    # Stack [actual, pad] per batch into counts: e.g. [3,1, 3,1, 2,2, 7,1]
                    counts = torch.stack([actual_seqlens, pad_lens], dim=1).flatten()
                    # Tile [T, F] per sequence: [T,F, T,F, T,F, T,F]
                    values = torch.tensor([True, False], device=max_tensor.device).repeat(b)
                    # Expand: T×3, F×1, T×3, F×1, T×2, F×2, T×7, F×1 → TTTF|TTTF|TTFF|TTTTTTTF
                    valid = torch.repeat_interleave(values, counts)
                    # Finally, replace invalid (F) positions with -inf
                    max_tensor = max_tensor.masked_fill(~valid.view(-1, 1, 1), float("-inf"))

        # Max -> max_logit [h]
        max_logit = torch.amax(max_tensor, dim=amax_dims).to(dtype=output_tensors[0].dtype)
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
    o_format: str = "sbhd",
    do_format: str = "sbhd",
    dqkv_layout: str = "sbh3d",
    qkv_scale_inv_format: str = None,
    do_scale_inv_format: str = None,
    attn_bias_type: str = "no_bias",
    attn_mask_type: str = "padding",
    softmax_type: str = "vanilla",
    window_size: Tuple[int, int] = (-1, -1),
    bottom_right_diagonal: bool = None,
    deterministic: bool = False,
    cuda_graph: bool = False,
) -> Tuple[Union[torch.Tensor, None], ...]:
    """Fused Attention BWD for packed KV input.

    Parameters
    ----------
    max_seqlen_q : int
                max sequence length for Q, used for padding; may be larger than max(seqlens_q),
                seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_kv : int
                max sequence length for K and V, used for padding;
                may be larger than max(seqlens_kv),
                seqlens_kv = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
    cu_seqlens_q : torch.Tensor
                cumulative sequence lengths for Q; shape [batch_size + 1]
    cu_seqlens_kv : torch.Tensor
                cumulative sequence lengths for K and V; shape [batch_size + 1]
    q : torch.Tensor
                input tensor Q; shape sbhd, bshd or thd (see `qkv_layout` for details)
    k : torch.Tensor
                input tensor K; shape sbhd, bshd or thd (see `qkv_layout` for details)
    v : torch.Tensor
                input tensor V; shape sbhd, bshd or thd (see `qkv_layout` for details)
    o : torch.Tensor
                input tensor O (output of forward); same data type as Q, K and V;
                same shape as Q
    d_o : torch.Tensor
                input tensor dO (gradient of O); same data type as Q, K and V;
                same shape as Q
    fake_dtype : tex.DType
                data type of Q, K and V - in case of high precision, fake dtype in case of FP8;
                in torch.dtype
    aux_ctx_tensors : List[torch.Tensor]
                auxiliary output tensors of the forward pass when its is_training is True,
                e.g. aux_ctx_tensors = [M, ZInv, rng_state]
    fused_attention_backend : tex.NVTE_Fused_Attn_Backend
                please see FusedAttention module for details on supported backends.
    cu_seqlens_q_padded : torch.Tensor, default = None
                cumulative sequence offsets for Q; shape [batch_size + 1]
    cu_seqlens_kv_padded : torch.Tensor, default = None
                cumulative sequence offsets for KV; shape [batch_size + 1]
    s_quantizer : Quantizer, default = None
                Quantizer object for the intermediate value S.
    dp_quantizer : Quantizer, default = None
                Quantizer object for the intermediate value dP.
    dqkv_quantizer : Quantizer, default = None
                Quantizer object for the output values of the fused_attn_bwd.
    attn_scale : float, default = None
                if not None, use attn_scale as the attention scale for Q*K.T BMM;
                if None, use 1.0/sqrt(head_dim_qk) as the default
    dropout : float, default = 0.0
                dropout probability, 0.0 means no dropout, 1.0 means no output;
                dropout must be 0.0 if is_training is False
    fast_zero_fill : bool, default = True
                if True, initializes the output tensor O to zero using the fast filling method;
                if False, uses PyTorch's .fill_() method
    qkv_layout : str, default = "sbh3d"
                layout of Q, K and V;
                {"sb3hd", "sbh3d", "sbhd_sb2hd", "sbhd_sbh2d", "sbhd_sbhd_sbhd",
                "bs3hd", "bsh3d", "bshd_bs2hd", "bshd_bsh2d", "bshd_bshd_bshd",
                "t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd"}
    o_format : str, default = "sbhd"
                format of O; {"sbhd", "bshd", "thd"}
    do_format : str, default = "sbhd"
                format of dO; {"sbhd", "bshd", "thd"}
    dqkv_layout : str, default = "sbh3d"
                layout of dQ, dK and dV;
                {"sb3hd", "sbh3d", "sbhd_sb2hd", "sbhd_sbh2d", "sbhd_sbhd_sbhd",
                "bs3hd", "bsh3d", "bshd_bs2hd", "bshd_bsh2d", "bshd_bshd_bshd",
                "t3hd", "th3d", "thd_t2hd", "thd_th2d", "thd_thd_thd"}
    qkv_scale_inv_format : str, default = None
                format of the scale-inverse tensors for QKV; {"sbhd", "bshd", "thd", "bhsd"};
                if None, defaults to the format inferred from qkv_layout.
    do_scale_inv_format : str, default = None
                format of the scale-inverse tensors for dO; {"sbhd", "bshd", "thd", "bhsd"};
                if None, defaults to the format inferred from the output layout.
    attn_bias_type : str, default = "no_bias"
                type of the bias; {"no_bias", "pre_scale_bias", "post_scale_bias", "alibi"}
    attn_mask_type : str, default = "padding"
                type of the attention mask; {"padding", "causal", "padding_causal", "no_mask"}
    softmax_type : str, default = "vanilla"
                type of the attention softmax; {"vanilla", "off-by-one", "learnable"}
    window_size : Tuple[int, int], default = (-1, -1)
                sliding window size for local attention, where query at position i attends to keys
                in [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q
                + window_size[1]] inclusive. Special cases (-1, -1) and (-1, 0) mean no sliding
                window and causal mask specifically.
    bottom_right_diagonal: bool, default = None
                whether to align sliding window and ALiBi diagonal to the top left (False) or
                bottom right (True) corner of the softmax matrix.
    deterministic : bool, default = False
                whether to execute the backward pass with deterministic behaviours.
    cuda_graph : bool, default = False
                whether or not cuda graph capture is enabled.

    Returns
    ----------
    d_q : torch.Tensor
                gradient tensor of Q; same data type and shape as Q
    d_k : torch.Tensor
                gradient tensor of K; same data type and shape as K
    d_v : torch.Tensor
                gradient tensor of V; same data type and shape as V
    d_bias : torch.Tensor, optional
                gradient tensor of Bias when attn_bias_type is "pre_scale_bias"
                or "post_scale_bias"; same data type and shape as Bias
    d_softmax_offset : torch.Tensor, optional
                gradient tensor of softmax offset of shape [1, h_q, 1, 1].
                See softmax_type in DotProductAttention for details.
    """
    if bottom_right_diagonal is None:
        bottom_right_diagonal = attn_mask_type in {
            "causal_bottom_right",
            "padding_causal_bottom_right",
        }

    if attn_scale is None:
        d = q.size(-1)
        attn_scale = 1.0 / math.sqrt(d)

    if fused_attention_backend == FusedAttnBackend["No_Backend"]:
        raise ValueError(
            "Fused attention backward does not support this input combination:"
            f" qkv_layout={qkv_layout!r}, attn_bias_type={attn_bias_type!r},"
            f" attn_mask_type={attn_mask_type!r}, q.shape={list(q.shape)},"
            f" q.dtype={q.dtype}, backend={fused_attention_backend}."
        )

    if fused_attention_backend != FusedAttnBackend["F16_max512_seqlen"]:
        if len(aux_ctx_tensors) < 1:
            raise ValueError(
                "aux_ctx_tensors must contain rng_state as its last element,"
                f" but got len(aux_ctx_tensors)={len(aux_ctx_tensors)}"
                f" for backend={fused_attention_backend}."
            )

    output_tensors = tex.fused_attn_bwd(
        max_seqlen_q,
        max_seqlen_kv,
        attn_scale,
        dropout,
        fast_zero_fill,
        QKVLayout[qkv_layout],
        QKVFormat[o_format],
        QKVFormat[do_format],
        QKVLayout[dqkv_layout],
        QKVFormat[qkv_scale_inv_format],
        QKVFormat[do_scale_inv_format],
        AttnBiasType[attn_bias_type],
        AttnMaskType[attn_mask_type],
        SoftmaxType[softmax_type],
        window_size,
        bottom_right_diagonal,
        deterministic,
        cu_seqlens_q,
        cu_seqlens_kv,
        q,
        k,
        v,
        o,
        d_o,
        fake_dtype,
        aux_ctx_tensors,
        cu_seqlens_q_padded,
        cu_seqlens_kv_padded,
        s_quantizer,
        dp_quantizer,
        dqkv_quantizer,
        cuda_graph,
    )

    return output_tensors
