"""Efficient multi-head attention"""
from typing import Any, Callable, Optional, Tuple, Union
import torch
import emha_C
import numpy as np

EMHA_MASK_MODE = emha_C.MaskMode

import scaled_upper_triang_masked_softmax_cuda
import scaled_upper_triang_masked_softmax_dropout_cuda
import math
from contextlib import nullcontext
from typing import Any, Callable, Optional, Tuple, Union

import torch

from transformer_engine.pytorch import LayerNormLinear, Linear, LayerNormMLP, LayerNorm
from transformer_engine.pytorch.jit import (
    set_jit_fusion_options,
    warmup_jit_bias_dropout_add_all_dtypes,
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from transformer_engine.pytorch.utils import (
    divide,
    attention_mask_func,
    split_tensor_along_last_dim,
    cast_if_needed,
)
from transformer_engine.pytorch.constants import (
    AttnMaskTypes,
    AttnTypes,
    LayerTypes,
    dist_group_type,
)
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax, ScaledUpperTriangMaskedSoftmax
from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    checkpoint,
)


# NOTE(ksivamani): Inference/validation optimization possibility:
# Set p_dropout=0.0 and remove ReLU prologs from BMMs
class CoreAttentionEMHA(torch.autograd.Function):
    """CoreAttentionEMHA"""

    @staticmethod
    def forward(
        ctx,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale_pre_softmax: float,
        p_dropout: float,
        is_training
    ):
        """eMHA as a drop-in replacement for CoreAttention.forward.

        Let `s`, `b`, `h`, and `d` denote `sequence`, `batch`, `number of
        attention heads`, and `feature dimension`.

        Args:
            Q: [s, b, h, d]
            K: [s, b, h, d]
            V: [s, b, h, d]
            scale_pre_softmax:
            p_dropout:
        """
        s, b, h, d = Q.shape
        Q = Q.view(s, -1, d).transpose(0, 1)
        K = K.view(s, -1, d).transpose(0, 1)
        V = V.view(s, -1, d).transpose(0, 1)

        P_v = torch.empty((b*h,s,s), dtype=Q.dtype, device=Q.device)
        # BMM1 (P = Q * K')
        #P_v = torch.baddbmm(P_v, Q, K.transpose(1, 2), alpha=scale_pre_softmax, beta=0.0)
        P_v = torch.bmm(Q, K.transpose(1, 2))
        if is_training:
            S_dmask_v = scaled_upper_triang_masked_softmax_dropout_cuda.forward(P_v, scale_pre_softmax, p_dropout, None)
            # Dropout_apply + BMM2 (C = P * V)
            C = torch.empty((s, b, h, d), dtype=Q.dtype, device=Q.device)
            Cv = C.view(s, -1, d).transpose(0, 1)
            #emha_C.relu_bmm_nn(S_dmask_v, V, Cv)
            torch.bmm(S_dmask_v.relu(), V, out=Cv)

            ctx.save_for_backward(Q, K, V, S_dmask_v)
            ctx.scale_pre_softmax = scale_pre_softmax
            ctx.p_dropout = p_dropout
            ctx.orig_shape = s, b, h, d
        else:
            S = scaled_upper_triang_masked_softmax_cuda.forward(P_v, 1.0)
            # Dropout_apply + BMM2 (C = P * V)
            C = torch.empty((s, b, h, d), dtype=Q.dtype, device=Q.device)
            Cv = C.view(s, -1, d).transpose(0, 1)
            torch.bmm(S, V, out=Cv)

        #P = P_v.view(b, h, s, s)

        ## Softmax + Dropout_draw
        #S_dmask = emha_C.softmax_fwd(
        #    P,
        #    cu_seqlens,
        #    scale_pre_softmax,
        #    p_dropout,
        #    attention_mask_mode,
        #    None,
        #)

        #S_dmask_v = S_dmask.view(-1, s, s)

        # `C` is supposed to be (s, b, h * d) by `CoreAttention`
        return C.view(s, b, h * d)

    @staticmethod
    def backward(ctx, grad_output):

        # `dC` has the shape of (s, b, h * d)
        dC = grad_output
        # Q, K, and V: (b * h, s, d)
        Q, K, V, S_dmask = ctx.saved_tensors
        s, b, h, d = ctx.orig_shape

        # dQ, dK, and dV: (b * h, s, d)
        dQ, dK, dV = torch.empty_like(Q), torch.empty_like(K), torch.empty_like(V)

        dC_v = dC.view(s, -1, d).transpose(0, 1)
        # dS = dC * V'
        dS_v = torch.bmm(dC_v, V.transpose(1, 2))
        # dV = (S * rp_keep * D)' * dC
        #emha_C.relu_bmm_nt(S_dmask.view(-1, s, s).transpose(1, 2), dC_v, dV)
        torch.bmm(S_dmask.relu().view(-1, s, s).transpose(1, 2), dC_v, out=dV)

        dS = dS_v.view_as(S_dmask)
        # d[Dropout + Softmax]
        #dP = emha_C.softmax_bwd(dS, S_dmask, ctx.scale_pre_softmax, ctx.p_dropout)
        dP = scaled_upper_triang_masked_softmax_dropout_cuda.backward(dS, S_dmask, ctx.scale_pre_softmax, ctx.p_dropout)
        dP_v = dP.view(-1, s, s)

        # dQ = dP * K
        torch.bmm(dP_v, K, out=dQ)
        # dK = dP' * Q
        torch.bmm(dP_v.transpose(1, 2), Q, out=dK)

        dQ, dK, dV = [t.transpose(0, 1).view(s, b, h, d) for t in (dQ, dK, dV)]

        return (
            dQ,
            dK,
            dV,
            None,  # cu_seqlens
            None,  # attention_mask_mode
            None,  # scale_pre_softmax
            None,  # p_dropout
            None,  # is_training
        )

fwd_count = 0
bwd_count = 0

class ScaledUpperTriangMaskedSoftmaxDropout(torch.autograd.Function):

    @staticmethod
    def forward(ctx, P_v, scale_pre_softmax, p_dropout, is_training):
        if not is_training:
            S = scaled_upper_triang_masked_softmax_cuda.forward(P_v, scale_pre_softmax)
            return S
        S_dmask = scaled_upper_triang_masked_softmax_dropout_cuda.forward(P_v, scale_pre_softmax, p_dropout, None)

        ctx.save_for_backward(S_dmask)
        ctx.scale_pre_softmax = scale_pre_softmax
        ctx.p_dropout = p_dropout
        return S_dmask

    @staticmethod
    def backward(ctx, grad_output):
        S_dmask = ctx.saved_tensors[0]

        dP = scaled_upper_triang_masked_softmax_dropout_cuda.backward(grad_output, S_dmask, ctx.scale_pre_softmax, ctx.p_dropout)

        return dP, None, None, None

class BMM2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S_dmask, V):
        # S: b*h, s, s
        # V: s, b, h, d

        s, b, h, d = V.shape
        Vv = V.view(s, -1, d).transpose(0, 1)
        C = torch.empty((s, b, h, d), dtype=V.dtype, device=V.device)
        Cv = C.view(s, -1, d).transpose(0, 1)
        #torch.bmm(S, Vv, out=Cv)
        emha_C.relu_bmm_nn(S_dmask, Vv, Cv)
        ctx.save_for_backward(S_dmask, V)

        return C.view(s, b, h*d)

    @staticmethod
    def backward(ctx, grad_output):
        S_dmask, V = ctx.saved_tensors
        # `dC` has the shape of (s, b, h * d)
        s, b, h, d = V.shape

        dC = grad_output
        dC_v = dC.view(s, b*h, d).transpose(0, 1)
        V_v = V.view(s, b*h, d).transpose(0, 1)

        # dS = dC * V'
        dS_v = torch.bmm(dC_v, V_v.transpose(1, 2))

        #s,b,h,d
        dV = torch.empty_like(V)
        #b*h, s, d
        dVv = dV.view(s, -1, d).transpose(0,1)
        #print(dV.shape, V.shape, dVv.shape)

        # dV = (S * rp_keep * D)' * dC
        #torch.bmm(S.view(-1, s, s).transpose(1, 2), dC_v, out=dVv)
        emha_C.relu_bmm_nt(S_dmask.view(-1, s, s).transpose(1, 2), dC_v, dVv)

        return dS_v, dV


class EMHA(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
        layer_number: Optional[int] = None,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        attn_mask_type: str = "causal",
        tp_size: int = 1,
        get_rng_state_tracker: Optional[Callable] = None,
        sequence_parallel: bool = False,
    ) -> None:
        super().__init__()

        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32

        if layer_number is None:
            self.apply_query_key_layer_scaling = False
        else:
            self.layer_number = max(1, layer_number)

        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True

        self.attn_mask_type = attn_mask_type
        projection_size = kv_channels * num_attention_heads
        assert (
            attn_mask_type in AttnMaskTypes
        ), f"attn_mask_type {attn_mask_type} not supported"

        # Per attention head and per partition values.
        self.hidden_size_per_partition = divide(projection_size, tp_size)
        self.hidden_size_per_attention_head = divide(
            projection_size, num_attention_heads
        )

        self.sequence_parallel = sequence_parallel
        if self.sequence_parallel or get_rng_state_tracker is None:
            self.attention_dropout_ctx = nullcontext
        else:
            self.attention_dropout_ctx = get_rng_state_tracker().fork

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.attn_mask_type,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff,
        )

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout)
        self.p_dropout = attention_dropout

    def forward(
        self,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """core attention fprop"""
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=1.0,
        )

        alpha = (1.0 / self.norm_factor)
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        b, np, sq, sk = attention_scores.size()

        assert sq == sk, "causal mask is only for self attention"

        # attention scores and attention mask [b, np, sq, sk]
        #attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = ScaledUpperTriangMaskedSoftmaxDropout.apply(attention_scores.view(-1, sq, sk), alpha, self.p_dropout, self.training)

        #if self.training:
        #    with self.attention_dropout_ctx():
        #        attention_probs = ScaledUpperTriangMaskedSoftmaxDropout.apply(attention_scores.view(-1, sq, sk), alpha, self.p_dropout)
        #else:
        #    attention_probs = ScaledUpperTriangMaskedSoftmax.apply(attention_scores.view(-1, sq, sk), alpha)

        context_layer = BMM2.apply(attention_probs, value_layer)

        return context_layer


