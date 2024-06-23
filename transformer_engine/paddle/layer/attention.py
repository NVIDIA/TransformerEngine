# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Attntion API"""

import math
import os
import warnings
from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F

try:
    from paddle.incubate.nn.functional import fused_rotary_position_embedding
except ImportError:
    fused_rotary_position_embedding = None
from transformer_engine import transformer_engine_paddle as tex

from .layernorm_linear import LayerNormLinear
from .linear import Linear
from .softmax import FusedScaleMaskSoftmax
from ..constants import (
    AttnTypes,
    TE_DType,
    AttnBiasType,
    AttnMaskType,
    FusedAttnBackend,
    dist_group_type,
)
from ..cpp_extensions import (
    fused_attn_fwd_qkvpacked,
    fused_attn_bwd_qkvpacked,
    fused_attn_fwd_kvpacked,
    fused_attn_bwd_kvpacked,
    fused_attn_fwd,
    fused_attn_bwd,
    mask_to_cu_seqlens,
)
from ..distributed import get_tp_group_and_world_size, track_rng_state
from ..utils import attention_mask_func, divide
from ..recompute import recompute

__all__ = ["DotProductAttention", "MultiHeadAttention", "RotaryPositionEmbedding"]


def repeat_kv(hidden_states: paddle.Tensor, n_rep: int) -> paddle.Tensor:
    """
    Used to repeat the key and value states for GQA.
    The hidden states go from (batch, seqlen, num_gqa_groups, head_size)
    to (batch, seqlen, num_heads, head_size)
    """
    batch, seqlen, num_gqa_groups, head_size = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states.unsqueeze(-2).tile([1, 1, 1, n_rep, 1])
    return hidden_states.reshape([batch, seqlen, num_gqa_groups * n_rep, head_size])


class RotaryPositionEmbedding(paddle.nn.Layer):
    """
    Implements Rotary Position Embedding from https://arxiv.org/abs/2104.09864.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
    ):
        """
        Parameters
        ----------
        dim: int
            rotary embedding dimension
        max_position_embeddings: int
            max_position_embeddings before position interpolation
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.inv_freq = 1.0 / (
            10000 ** (paddle.cast(paddle.arange(0, dim, 2), dtype="float32") / self.dim)
        )
        self._set_cos_sin_cache(seq_len=max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        # [seq_len]
        t = paddle.arange(seq_len, dtype="float32")
        # [seq_len, dim/2]
        freqs = paddle.einsum("i,j->ij", t, self.inv_freq)
        # [seq_len, dim]
        emb = paddle.concat([freqs, freqs], axis=-1)
        # [1, seqlen, 1, dim]
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def forward(self, max_seq_len: int):
        """
        Create rotary position embedding frequencies

        Parameters
        ----------
        max_seq_len: int
            sequence length of a sample
        """
        cos = self.cos_cached[:, :, :max_seq_len, ...]
        sin = self.sin_cached[:, :, :max_seq_len, ...]
        return (cos, sin)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return paddle.concat([-x2, x1], axis=-1)  # shape is the same as x


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    """Applies rotary positional embedding to the input."""

    if position_ids is None:
        # Note: Only for LlamaForCausalLMPipe model pretraining
        cos = cos[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
        sin = sin[:, : q.shape[1], :, :]  # [bs, seq_len, 1, dim]
    else:
        cos = cos.squeeze(axis=[0, 2])  # [seq_len, dim]
        sin = sin.squeeze(axis=[0, 2])  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
        sin = sin[position_ids].unsqueeze(2)  # [bs, seq_len, 1, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class FusedAttnFuncPackedQKV(paddle.autograd.PyLayer):
    """Function for FusedAttention with packed QKV input"""

    @staticmethod
    def forward(
        ctx,
        qkv,
        cu_seqlens,
        attn_bias,
        max_seqlen,
        attn_scale,
        qkv_dtype,
        dropout_p,
        set_zero,
        qkv_layout,
        attn_bias_type,
        attn_mask_type,
        is_training,
        fused_attention_backend,
    ):
        """Forward function for FusedAttention with packed QKV input"""
        out, softmax_aux, rng_state = fused_attn_fwd_qkvpacked(
            qkv,
            cu_seqlens,
            is_training,
            max_seqlen,
            qkv_dtype,
            fused_attention_backend,
            attn_bias,
            attn_scale,
            dropout_p,
            set_zero,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
        )

        ctx.save_for_backward(qkv, out, cu_seqlens, rng_state, softmax_aux)
        ctx.max_seqlen = max_seqlen
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.set_zero = set_zero
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.fused_attention_backend = fused_attention_backend

        return out

    @staticmethod
    def backward(ctx, d_out):
        """Backward function for FusedAttention with packed QKV input"""
        qkv, out, cu_seqlens, rng_state, softmax_aux = ctx.saved_tensor()
        dqkv, *rest = fused_attn_bwd_qkvpacked(
            qkv,
            cu_seqlens,
            rng_state,
            out,
            d_out,
            softmax_aux,
            ctx.fused_attention_backend,
            ctx.max_seqlen,
            ctx.qkv_dtype,
            ctx.attn_scale,
            ctx.dropout_p,
            ctx.set_zero,
            ctx.qkv_layout,
            ctx.attn_bias_type,
            ctx.attn_mask_type,
        )

        # if no_bias, return dqkv
        if ctx.attn_bias_type == "no_bias":
            return (dqkv, None)
        # else, return (dqkv, dbias)
        return (dqkv, None, rest[0])


class FusedAttnFuncPackedKV(paddle.autograd.PyLayer):
    """Function for FusedAttention with packed KV input"""

    @staticmethod
    def forward(
        ctx,
        q,
        kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_bias,
        max_seqlen_q,
        max_seqlen_kv,
        attn_scale,
        qkv_dtype,
        dropout_p,
        set_zero,
        qkv_layout,
        attn_bias_type,
        attn_mask_type,
        is_training,
        fused_attention_backend,
    ):
        """Forward function for FusedAttention with packed KV input"""
        out, softmax_aux, rng_state = fused_attn_fwd_kvpacked(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            is_training,
            max_seqlen_q,
            max_seqlen_kv,
            qkv_dtype,
            fused_attention_backend,
            attn_bias,
            attn_scale,
            dropout_p,
            set_zero,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
        )

        ctx.save_for_backward(q, kv, out, cu_seqlens_q, cu_seqlens_kv, rng_state, softmax_aux)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.set_zero = set_zero
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.fused_attention_backend = fused_attention_backend

        return out

    @staticmethod
    def backward(ctx, d_out):
        """Backward function for FusedAttention with packed KV input"""
        q, kv, out, cu_seqlens_q, cu_seqlens_kv, rng_state, softmax_aux = ctx.saved_tensor()
        dq, dkv, *rest = fused_attn_bwd_kvpacked(
            q,
            kv,
            cu_seqlens_q,
            cu_seqlens_kv,
            rng_state,
            out,
            d_out,
            softmax_aux,
            ctx.fused_attention_backend,
            ctx.max_seqlen_q,
            ctx.max_seqlen_kv,
            ctx.qkv_dtype,
            ctx.attn_scale,
            ctx.dropout_p,
            ctx.set_zero,
            ctx.qkv_layout,
            ctx.attn_bias_type,
            ctx.attn_mask_type,
        )

        # if no_bias, return dq, dkv
        if ctx.attn_bias_type == "no_bias":
            return (dq, dkv, None, None)
        # else, return (dq, dkv, dbias)
        return (dq, dkv, None, None, rest[0])


class FusedAttnFunc(paddle.autograd.PyLayer):
    """Function for FusedAttention with separate Q, K, V tensors"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        attn_bias,
        max_seqlen_q,
        max_seqlen_kv,
        attn_scale,
        qkv_dtype,
        dropout_p,
        set_zero,
        qkv_layout,
        attn_bias_type,
        attn_mask_type,
        is_training,
        fused_attention_backend,
    ):
        """Forward function for FusedAttention with separate Q, K, V tensors"""
        out, softmax_aux, rng_state = fused_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            is_training,
            max_seqlen_q,
            max_seqlen_kv,
            qkv_dtype,
            fused_attention_backend,
            attn_bias,
            attn_scale,
            dropout_p,
            set_zero,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
        )

        ctx.save_for_backward(q, k, v, out, cu_seqlens_q, cu_seqlens_kv, rng_state, softmax_aux)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.set_zero = set_zero
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type
        ctx.fused_attention_backend = fused_attention_backend

        return out

    @staticmethod
    def backward(ctx, d_out):
        """Backward function for FusedAttention with separate Q, K, V tensors"""
        q, k, v, out, cu_seqlens_q, cu_seqlens_kv, rng_state, softmax_aux = ctx.saved_tensor()
        dq, dk, dv, *rest = fused_attn_bwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            rng_state,
            out,
            d_out,
            softmax_aux,
            ctx.fused_attention_backend,
            ctx.max_seqlen_q,
            ctx.max_seqlen_kv,
            ctx.qkv_dtype,
            ctx.attn_scale,
            ctx.dropout_p,
            ctx.set_zero,
            ctx.qkv_layout,
            ctx.attn_bias_type,
            ctx.attn_mask_type,
        )
        # if no_bias, return dq, dk, dv
        if ctx.attn_bias_type == "no_bias":
            return (dq, dk, dv, None, None)
        # else, return (dq, dk, dv, dbias)
        return (dq, dk, dv, None, None, rest[0])


class DotProductAttention(paddle.nn.Layer):
    """
    Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`attn_mask_type` is set to `"causal"`.

    Parameters
    ----------
    num_attention_heads: int
            number of attention heads in the transformer layer.
    kv_channels: int
            number of channels in the key and value tensors.
    num_gqa_groups : Optional[int] = None
                    number of GQA groups in the transformer layer.
                    Grouped Query Attention is described in
                    `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                    This only affects the keys and values, not the queries.
                    GQA-1 is equivalent to Multi-Query Attention
                    (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                    is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: {'causal', 'padding', 'no_mask'}, default = `causal`
                   type of attention mask passed into softmax operation.
    attention_type: {'self', 'cross'}, default = `self`
                    type of attention operation.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    backend: {'transformer_engine', 'paddle'}, default = `transformer_engine`
             backend to use for attention operation.
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        num_gqa_groups: Optional[int] = None,
        attention_dropout: float = 0.1,
        attn_mask_type: str = "causal",
        attention_type: str = "self",
        tp_size: int = 1,
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()

        self.attn_mask_type = attn_mask_type
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.qkv_layout = "bshd_bshd_bshd"
        self.hidden_size_per_attention_head = kv_channels
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.tp_size = tp_size
        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // tp_size)
        self.num_queries_per_key_value = num_attention_heads // self.num_gqa_groups

        self.backend = backend

        self.use_fused_attention = bool(int(os.getenv("NVTE_FUSED_ATTN", "1")))

        if not self.use_fused_attention and backend == "transformer_engine":
            warnings.warn("Fused attention is not enabled, falling back to Paddle backend")
            self.backend = "paddle"

        if self.backend != "transformer_engine":
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                attn_mask_type, attention_mask_func, backend=self.backend
            )

    def forward(
        self,
        query_layer: paddle.Tensor,
        key_layer: paddle.Tensor,
        value_layer: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[paddle.Tensor] = None,
        set_zero: bool = True,
    ) -> paddle.Tensor:
        """
        Dot Product Attention Layer.

        .. note::

            Argument :attr:`attention_mask` will be ignored when :attr:`attn_mask_type`
            is set to `"causal"`.


        Parameters
        ----------
        query_layer : paddle.Tensor
                      Query tensor.
        key_layer : paddle.Tensor
                      Key tensor.
        value_layer : paddle.Tensor
                      Value tensor.
        attention_mask : Optional[paddle.Tensor], default = `None`
                         Boolean tensor used to mask out softmax input when not using attention.
        core_attention_bias_type: str, default = `no_bias`
                                  only support no_bias type currently, {`no_bias`}
        core_attention_bias: Optional[paddle.Tensor], default = `None`
                             Bias tensor for Q * K.T
        set_zero: bool, default = `True`
                  Whether to use the fast path to set output tensors to 0 or not.
        """

        backend = self.backend

        assert key_layer.shape == value_layer.shape, "Keys and values must have the same shape!"
        assert (
            key_layer.shape[-2] == self.num_gqa_groups_per_partition
        ), f"Keys and values must have num_gqa_group = {self.num_gqa_groups} heads!"

        if backend == "transformer_engine":
            max_s_q = query_layer.shape[1]
            max_s_kv = max_s_q if self.attention_type == "self" else key_layer.shape[1]
            self.fused_attention_backend = tex.get_fused_attn_backend(
                TE_DType[query_layer.dtype],
                TE_DType[query_layer.dtype],
                tex.get_nvte_qkv_layout(self.qkv_layout),
                AttnBiasType[core_attention_bias_type],
                AttnMaskType[self.attn_mask_type],
                self.attention_dropout,
                query_layer.shape[-2],
                key_layer.shape[-2] if key_layer is not None else query_layer.shape[-2],
                max_s_q,
                max_s_kv,
                query_layer.shape[-1],
            )

            is_backend_avail = self.fused_attention_backend in [
                FusedAttnBackend["F16_max512_seqlen"],
                FusedAttnBackend["F16_arbitrary_seqlen"],
            ]
            if is_backend_avail and self.use_fused_attention:
                return self._te_forward(
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask,
                    core_attention_bias_type,
                    core_attention_bias,
                    set_zero,
                )
            warnings.warn("Fused attention is not enabled, falling back to Paddle backend")
            backend = "paddle"
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                self.attn_mask_type, attention_mask_func, backend=backend
            )
        if backend == "paddle":
            if core_attention_bias_type != "no_bias":
                warnings.warn(
                    "Paddle backend dot product attention does not support bias yet. "
                    "Bias will be ignored."
                )
            return self._pd_forward(query_layer, key_layer, value_layer, attention_mask)
        raise AttributeError(f"Backend {backend} is not supported.")

    def _te_forward(
        self,
        query_layer: paddle.Tensor,
        key_layer: paddle.Tensor,
        value_layer: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[paddle.Tensor] = None,
        set_zero: bool = True,
    ) -> paddle.Tensor:

        if self.attention_type == "self":
            # self attention - q: [b, s, h, d]  kv: None
            assert (
                len(query_layer.shape) == 4
                and len(key_layer.shape) == 4
                and len(value_layer.shape) == 4
            ), "q,k,v shape must be [b, s, h, d] for dot product self attention"
            max_seqlen = query_layer.shape[1]
            if self.attn_mask_type == "causal" or attention_mask is None:
                cu_seqlens = paddle.arange(
                    0,
                    (query_layer.shape[0] + 1) * query_layer.shape[1],
                    step=query_layer.shape[1],
                    dtype="int32",
                )
            else:
                cu_seqlens, _ = mask_to_cu_seqlens(attention_mask, need_kv=False)
            qkv_dtype = TE_DType[query_layer.dtype]

            output = FusedAttnFunc.apply(
                query_layer,
                key_layer,
                value_layer,
                cu_seqlens,
                cu_seqlens,
                core_attention_bias,
                max_seqlen,
                max_seqlen,
                1.0 / self.norm_factor,
                qkv_dtype,
                self.attention_dropout if self.training else 0.0,
                set_zero,
                self.qkv_layout,
                core_attention_bias_type,
                self.attn_mask_type,
                self.training,
                self.fused_attention_backend,
            )
        elif self.attention_type == "cross":
            # cross attention - q: [b, s_q, h, d]  k,v: [b, s_kv, h, d]
            assert (
                len(query_layer.shape) == 4
                and len(key_layer.shape) == 4
                and len(value_layer.shape) == 4
            ), (
                "query shape must be [b, s_q, h, d] and key shape must be [b, s_kv, h, d]"
                "for dot product cross attention"
            )
            assert attention_mask is not None, "attention_mask must be provided for cross attention"
            max_seqlen_q = query_layer.shape[1]
            max_seqlen_kv = key_layer.shape[1]
            cu_seqlens_q, cu_seqlens_kv = mask_to_cu_seqlens(attention_mask, need_kv=True)
            qkv_dtype = TE_DType[query_layer.dtype]
            output = FusedAttnFunc.apply(
                query_layer,
                key_layer,
                value_layer,
                cu_seqlens_q,
                cu_seqlens_kv,
                core_attention_bias,
                max_seqlen_q,
                max_seqlen_kv,
                1.0 / self.norm_factor,
                qkv_dtype,
                self.attention_dropout if self.training else 0.0,
                set_zero,
                self.qkv_layout,
                core_attention_bias_type,
                self.attn_mask_type,
                self.training,
                self.fused_attention_backend,
            )
        else:
            raise ValueError("attention_type must be one of ['self', 'cross']")
        return output

    def _pd_forward(
        self,
        query_layer: paddle.Tensor,
        key_layer: paddle.Tensor,
        value_layer: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:

        q = query_layer
        k = repeat_kv(key_layer, self.num_queries_per_key_value)
        v = repeat_kv(value_layer, self.num_queries_per_key_value)

        q = paddle.transpose(x=q, perm=[0, 2, 1, 3])
        k = paddle.transpose(x=k, perm=[0, 2, 1, 3])
        v = paddle.transpose(x=v, perm=[0, 2, 1, 3])

        product = paddle.matmul(x=q * (1.0 / self.norm_factor), y=k, transpose_y=True)
        attention_probs = self.scale_mask_softmax(product, attention_mask, scale=None)

        if self.attention_dropout > 0:
            attention_probs = F.dropout(
                attention_probs,
                self.attention_dropout,
                training=self.training,
            )

        out = paddle.matmul(attention_probs, v)
        out = paddle.transpose(out, perm=[0, 2, 1, 3])  # [b, s, h, d]
        # out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        return out


class MultiHeadAttention(paddle.nn.Layer):
    """
    Multi-head Attention (MHA), including Query,
    Key, Value and Output projection.

    Parameters
    ----------
    hidden_size: int
                    hidden size of the model.
    num_attention_heads: int
                    number of attention heads.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    layernorm_epsilon: float, default = 1e-5
                          epsilon to use in the layer norm operations.
    weight_attr: Union[paddle.ParamAttr, None], default = `None`
                    paddle.ParamAttr object for the weight parameter.
    bias_attr: Union[paddle.ParamAttr, None, bool], default = `None`
                    paddle.ParamAttr object for the bias parameter.
    attn_mask_type: {'causal', 'padding', 'no_mask'}, default = `causal`
                   type of attention mask passed into softmax operation.
    params_dtype: Optional[paddle.dtype], default = `None`
                    data type for the weights and biases.
    return_layernorm_output: bool, default = `False`
                    whether to return the output of the layernorm operation.
    input_layernorm: bool, default = `False`
                    whether to apply layernorm to the input.
    attention_type: {'self', 'cross'}, default = `self`
                    type of attention operation.
    normalization : { 'LayerNorm', 'RMSNorm' }, default = 'LayerNorm'
                   type of normalization applied.
    zero_centered_gamma: bool, default = `False`
                    whether to zero initialize the gamma of the layernorm operation.
    backend: {'transformer_engine', 'paddle'}, default = `transformer_engine`
             backend to use for attention operation. If set to 'paddle', a framework
             only no-FP8 path is executed with limited optimization.

    Parallelism parameters
    ----------------------
    set_parallel_mode : bool, default = `False`
                      if set to `True`, QKV and FC1 layers are used as Column Parallel
                      whereas PROJ and FC2 is used as Row Parallel as described
                      `here <https://arxiv.org/pdf/1909.08053.pdf>`_.
    sequence_parallel : bool, default = `False`
                       if set to `True`, uses sequence parallelism.
    tp_group : ProcessGroup, default = `None`
              tensor parallel process group.
    num_gqa_groups : int, default = `None`
                     number of GQA groups in the transformer layer.
                     Grouped Query Attention is described in
                     `this paper <https://arxiv.org/pdf/2305.13245.pdf>`_.
                     This only affects the keys and values, not the querys.
                     GQA-1 is equivalent to Multi-Query Attention
                     (`MQA <https://arxiv.org/pdf/1911.02150.pdf>`_), while GQA-H
                     is equivalent to MHA, i.e. `num_gqa_groups = num_attention_heads`.
    rng_state_name : str, default = `local_seed`
                   Controls the rng state used for dropout on attention probs. The
                   specified rng should be set different seeds for different TP ranks.
                   It will be ignored if `set_parallel_mode` is False. The specified
                   name should be registered through
                   `paddle.distributed.fleet.meta_parallel.get_rng_state_tracker()
                   .add(rng_state_name, seed)`.

    Optimization parameters
    -----------------------
    fuse_wgrad_accumulation : bool, default = 'False'
                             if set to `True`, enables fusing of creation and accumulation of
                             the weight gradient. When enabled, it is assumed that the weights
                             have an additional `main_grad` attribute (used instead of the
                             regular `grad`) which is a pre-allocated buffer of the correct
                             size to accumulate gradients in.

    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        max_sequence_length: Optional[int] = None,
        attn_mask_type: str = "causal",
        params_dtype: Optional[paddle.dtype] = None,
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        normalization: str = "LayerNorm",
        zero_centered_gamma: bool = False,
        set_parallel_mode: bool = False,
        sequence_parallel: bool = False,
        tp_group: Optional[dist_group_type] = None,
        num_gqa_groups: Optional[int] = None,
        fuse_wgrad_accumulation: bool = False,
        rng_state_name: str = "local_seed",
        backend: str = "transformer_engine",
    ) -> None:
        super().__init__()
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = paddle.get_default_dtype() if params_dtype is None else params_dtype
        self.max_sequence_length = max_sequence_length
        self.weight_attr = weight_attr
        self.bias_attr = bias_attr
        self.attn_mask_type = attn_mask_type

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"

        self.tp_group, self.tp_size = get_tp_group_and_world_size(
            tp_group, enable_tp=set_parallel_mode
        )
        self.tensor_parallel = self.tp_size > 1
        self.sequence_parallel = self.tensor_parallel and sequence_parallel
        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        self.set_parallel_mode = set_parallel_mode
        self.rng_state_name = rng_state_name
        self.backend = backend

        self.num_attention_heads_per_partition = divide(self.num_attention_heads, self.tp_size)
        self.num_gqa_groups = num_attention_heads if num_gqa_groups is None else num_gqa_groups
        assert (
            self.num_attention_heads % self.num_gqa_groups == 0
        ), "The number of attention heads must be divisible by the number of GQA groups!"
        assert (
            self.num_gqa_groups % self.tp_size == 0
        ), "The number of GQA groups must be divisible by tensor parallel size!"
        self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)
        self.hidden_size_kv = int(hidden_size * self.num_gqa_groups // self.num_attention_heads)
        qkv_parallel_mode = "column" if set_parallel_mode else None

        if self.attention_type == "self":
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    hidden_size + 2 * self.hidden_size_kv,
                    eps=layernorm_epsilon,
                    weight_attr=self.weight_attr,
                    bias_attr=self.bias_attr,
                    return_layernorm_output=return_layernorm_output,
                    normalization=normalization,
                    zero_centered_gamma=zero_centered_gamma,
                    parallel_mode=qkv_parallel_mode,
                    sequence_parallel=self.sequence_parallel,
                    tp_group=self.tp_group,
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                    backend=self.backend,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    hidden_size + 2 * self.hidden_size_kv,
                    self.weight_attr,
                    self.bias_attr,
                    parallel_mode=qkv_parallel_mode,
                    sequence_parallel=self.sequence_parallel,
                    tp_group=self.tp_group,
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                    backend=self.backend,
                )

        else:  # cross attention
            if self.input_layernorm:
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    hidden_size,
                    eps=layernorm_epsilon,
                    weight_attr=self.weight_attr,
                    bias_attr=self.bias_attr,
                    return_layernorm_output=return_layernorm_output,
                    normalization=normalization,
                    zero_centered_gamma=zero_centered_gamma,
                    parallel_mode=qkv_parallel_mode,
                    sequence_parallel=self.sequence_parallel,
                    tp_group=self.tp_group,
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                    backend=self.backend,
                )
            else:
                self.query_layer = Linear(
                    hidden_size,
                    hidden_size,
                    self.weight_attr,
                    self.bias_attr,
                    parallel_mode=qkv_parallel_mode,
                    sequence_parallel=self.sequence_parallel,
                    tp_group=self.tp_group,
                    fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                    backend=self.backend,
                )
            self.key_value = Linear(
                hidden_size,
                2 * self.hidden_size_kv,
                self.weight_attr,
                self.bias_attr,
                parallel_mode=qkv_parallel_mode,
                sequence_parallel=self.sequence_parallel,
                tp_group=self.tp_group,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
                backend=self.backend,
            )

        # Attention.
        self.core_attention = DotProductAttention(
            self.num_attention_heads,
            self.hidden_size_per_attention_head,
            self.num_gqa_groups,
            attention_dropout,
            attn_mask_type=attn_mask_type,
            attention_type=self.attention_type,
            tp_size=self.tp_size,
            backend=self.backend,
        )

        # Linear
        self.proj = Linear(
            hidden_size,
            hidden_size,
            self.weight_attr,
            self.bias_attr,
            parallel_mode="row" if set_parallel_mode else None,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            backend=self.backend,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_output: Optional[paddle.Tensor] = None,
        rotary_pos_emb: Optional[Tuple[paddle.Tensor, paddle.Tensor]] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[paddle.Tensor] = None,
        set_zero: bool = True,
        recompute_core_attention: bool = False,
        is_first_microbatch: Optional[bool] = None,
    ) -> Tuple[Union[paddle.Tensor, None], ...]:
        """
        MultiHeadAttention Layer.

        Parameters
        ----------
        hidden_states : paddle.Tensor
                        Input tensor.
        attention_mask : Optional[paddle.Tensor], default = `None`
                        Boolean tensor used to mask out softmax input when not using attention.
        encoder_output : Optional[paddle.Tensor], default = `None`
                        Output of the encoder layer.
        rotary_pos_emb: Tuple[paddle.Tensor, paddle.Tensor], default = `None`
                       Embeddings for query and key tensors for applying rotary position
                       embedding. By default no input embedding is applied.
        core_attention_bias_type: str, default = `no_bias`
                                only support no_bias type currently, {`no_bias`}
        core_attention_bias: Optional[paddle.Tensor], default = `None`
                    Bias tensor for Q * K.T
        set_zero: bool, default = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        recompute_core_attention: bool, default = `False`
                                  If true, forward activations for core attention are recomputed
                                  during the backward pass in order to save memory that would
                                  otherwise be occupied to store the forward activations until
                                  backprop.
        is_first_microbatch : {True, False, None}, default = None
                             During training using either gradient accumulation or
                             pipeline parallelism a minibatch of data is further split
                             into microbatches. Between the microbatches of the same minibatch
                             the model weights are not updated. Setting this parameter indicates
                             whether the current microbatch is the first in a minibatch or not.
                             When set, this parameter enables additional optimizations:

                             * during FP8 training, it allows caching of the FP8 versions of
                               the weights
        """

        if self.attn_mask_type != "causal" and attention_mask is not None:
            assert attention_mask.dtype == paddle.bool, "Attention mask must be a boolean tensor"

        input_dim = len(hidden_states.shape)
        if input_dim == 2:
            # hidden_states: [b * s_q, hidden_size]
            # need to get max_seq_len from attention_mask
            assert self.max_sequence_length is not None, "max_sequence_length must be provided"
            max_seq_len = self.max_sequence_length
        elif input_dim == 3:
            # hidden_states: [b, s_q, hidden_size]
            max_seq_len = hidden_states.shape[1]
        else:
            raise ValueError(f"hidden_states should have 2 or 3 dimensions, got {input_dim}.")

        if self.attention_type == "self":
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    mixed_qkv_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_qkv_layer = layernorm_qkv_outputs
            else:
                mixed_qkv_layer = self.qkv(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )

            num_queries_per_key_value = (
                self.num_attention_heads_per_partition // self.num_gqa_groups_per_partition
            )

            # [b, s_q, hidden_size+2*hidden_size_kv] --> [b, s_q, (h/ng+2), ng, d]
            mixed_qkv_layer = mixed_qkv_layer.reshape(
                shape=[
                    -1,
                    max_seq_len,
                    (num_queries_per_key_value + 2),
                    self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                ]
            )

            # [b, s_q, (h/ng+2), ng, d]
            # --> [b, s_q, (h/ng), ng, d] [b, s_q, 1, ng, d] [b, s_q, 1, ng, d]
            query_layer, key_layer, value_layer = paddle.split(
                mixed_qkv_layer,
                num_or_sections=(num_queries_per_key_value, 1, 1),
                axis=2,
            )

            # query: -> [b, s, h, d]
            # key, value: -> [b, s, ng, d]
            query_layer, key_layer, value_layer = (
                x.reshape(shape=[x.shape[0], x.shape[1], -1, self.hidden_size_per_attention_head])
                for x in (query_layer, key_layer, value_layer)
            )

        else:  # cross attention
            mixed_kv_layer = self.key_value(
                encoder_output,
                is_first_microbatch=is_first_microbatch,
            )
            # [b, s_kv, 2 * hidden_size] --> [b, s_kv, 2, num_heads, head_size]
            mixed_kv_layer = mixed_kv_layer.reshape(
                shape=[
                    0,
                    0,
                    2 * self.num_gqa_groups_per_partition,
                    self.hidden_size_per_attention_head,
                ]
            )

            # [b, s_kv, 2 * ng, head_size]
            # --> 2 [b, s_kv, ng, head_size]
            key_layer, value_layer = paddle.split(
                mixed_kv_layer,
                num_or_sections=2,
                axis=2,
            )

            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query_layer(
                    hidden_states,
                    is_first_microbatch=is_first_microbatch,
                )

            # [b, s, hidden_size] --> [b, s, h, d]
            query_layer = query_layer.reshape(
                shape=[
                    -1,
                    max_seq_len,
                    self.num_attention_heads_per_partition,
                    self.hidden_size_per_attention_head,
                ]
            )

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            if fused_rotary_position_embedding is None:
                query_layer, key_layer = apply_rotary_pos_emb(
                    query_layer, key_layer, q_pos_emb, k_pos_emb
                )
            else:
                query_layer, key_layer, _ = fused_rotary_position_embedding(
                    query_layer,
                    key_layer,
                    v=None,
                    sin=k_pos_emb,
                    cos=q_pos_emb,
                    position_ids=None,
                    use_neox_rotary_style=False,
                )

        with track_rng_state(enable=self.tensor_parallel, name=self.rng_state_name):
            if recompute_core_attention:
                context_layer = recompute(
                    self.core_attention,
                    query_layer,
                    key_layer,
                    value_layer,
                    attention_mask,
                    core_attention_bias_type,
                    core_attention_bias,
                    set_zero,
                    use_reentrant=False,
                )
            else:
                context_layer = self.core_attention(
                    query_layer=query_layer,
                    key_layer=key_layer,
                    value_layer=value_layer,
                    attention_mask=attention_mask,
                    core_attention_bias_type=core_attention_bias_type,
                    core_attention_bias=core_attention_bias,
                    set_zero=set_zero,
                )

        if input_dim == 3:
            context_layer = paddle.reshape(
                context_layer, [-1, max_seq_len, context_layer.shape[2] * context_layer.shape[3]]
            )
        else:  # input_dim == 2
            context_layer = paddle.reshape(
                context_layer, [-1, context_layer.shape[2] * context_layer.shape[3]]
            )

        # Output. [b, s, hidden]
        attention_output = self.proj(context_layer, is_first_microbatch=is_first_microbatch)

        if self.input_layernorm and self.return_layernorm_output:
            return attention_output, layernorm_output
        return attention_output
