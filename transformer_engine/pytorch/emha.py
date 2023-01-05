# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Efficient multi-head attention"""

import math
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import emha_C
import scaled_upper_triang_masked_softmax_dropout_cuda
import transformer_engine_extensions as tex

from transformer_engine.pytorch.utils import (
    divide,
    attention_mask_func,
)
from transformer_engine.pytorch.constants import AttnMaskTypes
from transformer_engine.pytorch.softmax import FusedScaleMaskSoftmax

EMHA_MASK_MODE = emha_C.MaskMode


class ScaledUpperTriangMaskedSoftmaxDropout(torch.autograd.Function):
    """Fused softmax-dropout function"""

    @staticmethod
    def forward(ctx, P_v, scale_pre_softmax, p_dropout, is_training):
        if not is_training:
            S = tex.scaled_upper_triang_masked_softmax_forward(P_v, scale_pre_softmax)
            return S
        S_dmask = scaled_upper_triang_masked_softmax_dropout_cuda.forward(
            P_v, scale_pre_softmax, p_dropout, None
        )

        ctx.save_for_backward(S_dmask)
        ctx.scale_pre_softmax = scale_pre_softmax
        ctx.p_dropout = p_dropout
        return S_dmask

    @staticmethod
    def backward(ctx, grad_output):
        S_dmask = ctx.saved_tensors[0]

        dP = scaled_upper_triang_masked_softmax_dropout_cuda.backward(
            grad_output, S_dmask, ctx.scale_pre_softmax, ctx.p_dropout
        )

        return dP, None, None, None


class BMM2(torch.autograd.Function):
    """Implementation of BMM2 for MHA"""

    @staticmethod
    def forward(ctx, S_dmask, V, p_dropout):
        # S: b*h, s, s
        # V: s, b, h, d

        s, b, h, d = V.shape
        Vv = V.view(s, -1, d).transpose(0, 1)
        C = torch.empty((s, b, h, d), dtype=V.dtype, device=V.device)
        Cv = C.view(s, -1, d).transpose(0, 1)
        emha_C.relu_bmm_nn(S_dmask, Vv, Cv, 1.0 / (1.0 - p_dropout))
        ctx.save_for_backward(S_dmask, V)

        return C.view(s, b, h * d)

    @staticmethod
    def backward(ctx, grad_output):
        S_dmask, V = ctx.saved_tensors
        # `dC` has the shape of (s, b, h * d)
        s, b, h, d = V.shape

        dC = grad_output
        dC_v = dC.view(s, b * h, d).transpose(0, 1)
        V_v = V.view(s, b * h, d).transpose(0, 1)

        # dS = dC * V'
        dS_v = torch.bmm(dC_v, V_v.transpose(1, 2))

        # s,b,h,d
        dV = torch.empty_like(V)
        # b*h, s, d
        dVv = dV.view(s, -1, d).transpose(0, 1)

        emha_C.relu_bmm_nt(S_dmask.view(-1, s, s).transpose(1, 2), dC_v, dVv)

        return dS_v, dV, None


class EMHA(torch.nn.Module):
    """Efficient multihead attention with fused dropout"""

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

        alpha = 1.0 / self.norm_factor
        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        _, _, sq, sk = attention_scores.size()

        assert sq == sk, "causal mask is only for self attention"

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = ScaledUpperTriangMaskedSoftmaxDropout.apply(
                attention_scores.view(-1, sq, sk), alpha, self.p_dropout, self.training
            )

        context_layer = BMM2.apply(attention_probs, value_layer, self.p_dropout)

        return context_layer
