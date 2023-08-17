# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Attntion API"""

import math
import warnings
from typing import Optional, Tuple, Union

import paddle
import paddle.nn.functional as F

from transformer_engine.paddle.constants import (
    AttnTypes,
    TE_DType,
)
from transformer_engine.paddle.cpp_extensions import (
    fused_attn_fwd_qkvpacked,
    fused_attn_bwd_qkvpacked,
    fused_attn_fwd_kvpacked,
    fused_attn_bwd_kvpacked,
)
from transformer_engine.paddle.utils import (attention_mask_func, mask_to_cu_seqlens)
from .base import TransformerEngineBaseLayer
from .layernorm_linear import LayerNormLinear
from .linear import Linear
from .softmax import FusedScaleMaskSoftmax


class FusedAttnFuncPackedQKV(paddle.autograd.PyLayer):
    """Function for FusedAttention with packed QKV input"""

    @staticmethod
    def forward(ctx, qkv, cu_seqlens, attn_bias, rng_state, max_seqlen, attn_scale, qkv_dtype,
                dropout_p, set_zero, qkv_layout, attn_bias_type, attn_mask_type, is_training):
        """Forward function for FusedAttention with packed QKV input"""
        out, aux_ctx_tensors = fused_attn_fwd_qkvpacked(
            qkv,
            cu_seqlens,
            rng_state,
            is_training,
            max_seqlen,
            qkv_dtype,
            attn_bias,
            attn_scale,
            dropout_p,
            set_zero,
            qkv_layout,
            attn_bias_type,
            attn_mask_type,
        )

        ctx.save_for_backward(qkv, out, cu_seqlens, rng_state, aux_ctx_tensors)
        ctx.max_seqlen = max_seqlen
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.set_zero = set_zero
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type

        return out

    @staticmethod
    def backward(ctx, d_out):
        """Backward function for FusedAttention with packed QKV input"""
        qkv, out, cu_seqlens, rng_state, aux_ctx_tensors = ctx.saved_tensor()
        dqkv, *rest = fused_attn_bwd_qkvpacked(qkv, cu_seqlens, rng_state, out, d_out,
                                               aux_ctx_tensors, ctx.max_seqlen, ctx.qkv_dtype,
                                               ctx.attn_scale, ctx.dropout_p, ctx.set_zero,
                                               ctx.qkv_layout, ctx.attn_bias_type,
                                               ctx.attn_mask_type)

        # if no_bias, return dqkv
        if ctx.attn_bias_type == "no_bias":
            return (dqkv, None, None)
        # else, return (dqkv, dbias)
        return (dqkv, None, rest[0], None)


class FusedAttnFuncPackedKV(paddle.autograd.PyLayer):
    """Function for FusedAttention with packed KV input"""

    @staticmethod
    def forward(ctx, q, kv, cu_seqlens_q, cu_seqlens_kv, attn_bias, rng_state, max_seqlen_q,
                max_seqlen_kv, attn_scale, qkv_dtype, dropout_p, set_zero, qkv_layout,
                attn_bias_type, attn_mask_type, is_training):
        """Forward function for FusedAttention with packed KV input"""
        out, aux_ctx_tensors = fused_attn_fwd_kvpacked(q, kv, cu_seqlens_q, cu_seqlens_kv,
                                                       rng_state, is_training, max_seqlen_q,
                                                       max_seqlen_kv, qkv_dtype, attn_bias,
                                                       attn_scale, dropout_p, set_zero, qkv_layout,
                                                       attn_bias_type, attn_mask_type)

        ctx.save_for_backward(q, kv, out, cu_seqlens_q, cu_seqlens_kv, rng_state, aux_ctx_tensors)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_kv = max_seqlen_kv
        ctx.qkv_dtype = qkv_dtype
        ctx.attn_scale = attn_scale
        ctx.dropout_p = dropout_p
        ctx.set_zero = set_zero
        ctx.qkv_layout = qkv_layout
        ctx.attn_bias_type = attn_bias_type
        ctx.attn_mask_type = attn_mask_type

        return out

    @staticmethod
    def backward(ctx, d_out):
        """Backward function for FusedAttention with packed KV input"""
        q, kv, out, cu_seqlens_q, cu_seqlens_kv, rng_state, aux_ctx_tensors = ctx.saved_tensor()
        dq, dkv, *rest = fused_attn_bwd_kvpacked(q, kv, cu_seqlens_q, cu_seqlens_kv, rng_state, out,
                                                 d_out, aux_ctx_tensors, ctx.max_seqlen_q,
                                                 ctx.max_seqlen_kv, ctx.qkv_dtype, ctx.attn_scale,
                                                 ctx.dropout_p, ctx.set_zero, ctx.qkv_layout,
                                                 ctx.attn_bias_type, ctx.attn_mask_type)

        # if no_bias, return dq, dkv
        if ctx.attn_bias_type == "no_bias":
            return (dq, dkv, None, None, None)
        # else, return (dq, dkv, dbias)
        return (dq, dkv, None, None, rest[0], None)


class DotProductAttention(paddle.nn.Layer):
    """Dot Product Attention Layer
    Allows the model to jointly attend to information from different
    representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    .. note::

        Argument :attr:`attention_mask` will be ignored in the `forward` call when
        :attr:`attn_mask_type` is set to `"causal"`.

    Parameters
    ----------
    norm_factor : float
                    normalization factor for the attention scores.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    attn_mask_type: {'causal', 'padding', 'no_mask'}, default = `causal`
                   type of attention mask passed into softmax operation.
    attention_type: {'self', 'cross'}, default = `self`
                    type of attention operation.
    backend: {'transformer_engine', 'paddle'}, default = `transformer_engine`
                backend to use for attention operation.

    """

    def __init__(self,
                 norm_factor: float,
                 attention_dropout: float = 0.1,
                 attn_mask_type: str = "causal",
                 attention_type: str = "self",
                 backend: str = 'transformer_engine') -> None:
        super().__init__()

        self.norm_factor = norm_factor
        self.attn_mask_type = attn_mask_type
        self.attention_dropout = attention_dropout
        self.attention_type = attention_type
        self.backend = backend
        self.rng_state = paddle.zeros((2,), dtype='int64')
        self.rng_state.persistable = True
        if self.backend != 'transformer_engine':
            self.scale_mask_softmax = FusedScaleMaskSoftmax(attn_mask_type,
                                                            attention_mask_func,
                                                            backend=self.backend)

    def forward(
        self,
        query_layer: paddle.Tensor,
        key_value_layer: paddle.Tensor = None,
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

        .. note::

            For self attention, :attr:`query_layer` is the `[query, key, value]` tensor
            stacked along the 2nd dimension, which must be of shape (:attr:`batch_size`,
            :attr:`seq_length`, 3, :attr:`num_attention_heads`, :attr:`size_per_head`).
            And :attr:`key_value_layer` is `None`.
            For cross attention, :attr:`query_layer` is the `[query]` tensor, which must
            be of shape (:attr:`batch_size`, :attr:`seq_length`, :attr:`num_attention_heads`,
            :attr:`size_per_head`). And :attr:`key_value_layer` is the `[key, value]` tensor,
            which must be of shape (:attr:`batch_size`, :attr:`seq_length`, 2,
            :attr:`num_attention_heads`, :attr:`size_per_head`).



        Parameters
        ----------
        query_layer : paddle.Tensor
                     Query tensor.
        key_value_layer : paddle.Tensor
                   Key tensor.
        attention_mask : Optional[paddle.Tensor], default = `None`
                        Boolean tensor used to mask out softmax input when not using attention.
        core_attention_bias_type: str, default = `no_bias`
                                only support no_bias type currently, {`no_bias`}
        core_attention_bias: Optional[paddle.Tensor], default = `None`
                    Bias tensor for Q * K.T
        set_zero: bool, defautl = `True`
                    Whether to use the fast path to set output tensors to 0 or not.
        """

        if self.backend == 'transformer_engine':
            return self._te_forward(query_layer, key_value_layer, attention_mask,
                                    core_attention_bias_type, core_attention_bias, set_zero)
        if self.backend == 'paddle':
            if core_attention_bias_type != "no_bias":
                warnings.warn("Paddle backend dot product attention does not support bias yet. "
                              "Bias will be ignored.")
            return self._pd_forward(query_layer, key_value_layer, attention_mask)
        raise AttributeError(f"Backend {self.backend} is not supported.")

    def _te_forward(
        self,
        query_layer: paddle.Tensor,
        key_value_layer: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[paddle.Tensor] = None,
        set_zero: bool = True,
    ) -> paddle.Tensor:

        gen_state = paddle.get_rng_state()[0].__getstate__()
        self.rng_state[0], self.rng_state[1] = gen_state[1], gen_state[2]    # [seed, offset]
        if self.attention_type == "self":
            # self attention - q: [b, s, 3, h, d]  kv: None
            assert (len(query_layer.shape) == 5 and query_layer.shape[2] == 3
                    and key_value_layer is None
                   ), "query shape must be [b, s, 3, h, d] for dot product self attention"
            max_seqlen = query_layer.shape[1]
            cu_seqlens, _ = mask_to_cu_seqlens(attention_mask)
            qkv_dtype = TE_DType[query_layer.dtype]
            qkv_layout = "qkv_interleaved"

            output = FusedAttnFuncPackedQKV.apply(
                query_layer,
                cu_seqlens,
                core_attention_bias,
                self.rng_state,
                max_seqlen,
                1.0 / self.norm_factor,
                qkv_dtype,
                self.attention_dropout if self.training else 0.0,
                set_zero,
                qkv_layout,
                core_attention_bias_type,
                self.attn_mask_type,
                self.training,
            )
        elif self.attention_type == "cross":
            # cross attention - q: [b, s_q, h, d]  kv: [b, s_kv, 2, h, d]
            assert (
                len(query_layer.shape) == 4 and len(key_value_layer.shape) == 5
                and key_value_layer.shape[2] == 2
            ), "query shape must be [b, s, h, d] and key shape must be [b, s, 2, h, d]" \
                "for dot product cross attention"
            max_seqlen_q = query_layer.shape[1]
            max_seqlen_kv = key_value_layer.shape[1]
            cu_seqlens_q, cu_seqlens_kv = mask_to_cu_seqlens(attention_mask, need_kv=True)
            qkv_dtype = TE_DType[query_layer.dtype]
            qkv_layout = "kv_interleaved"
            output = FusedAttnFuncPackedKV.apply(
                query_layer,
                key_value_layer,
                cu_seqlens_q,
                cu_seqlens_kv,
                core_attention_bias,
                self.rng_state,
                max_seqlen_q,
                max_seqlen_kv,
                1.0 / self.norm_factor,
                qkv_dtype,
                self.attention_dropout if self.training else 0.0,
                set_zero,
                qkv_layout,
                core_attention_bias_type,
                self.attn_mask_type,
                self.training,
            )
        else:
            raise ValueError("attention_type must be one of ['self', 'cross']")
        return output

    def _pd_forward(
        self,
        query_layer: paddle.Tensor,
        key_value_layer: paddle.Tensor = None,
        attention_mask: Optional[paddle.Tensor] = None,
    ) -> paddle.Tensor:
        if self.attention_type == "self":
            # self attention - q: [b, s, 3, h, d]  k: None
            assert (len(query_layer.shape) == 5 and query_layer.shape[2] == 3
                    and key_value_layer is None
                   ), "query shape must be [b, s, 3, h, d] for dot product self attention"
            q = query_layer[:, :, 0]
            k = query_layer[:, :, 1]
            v = query_layer[:, :, 2]
        elif self.attention_type == "cross":
            # cross attention - q: [b, s, h, d]  kv: [b, s, 2, h, d]
            assert (
                len(query_layer.shape) == 4 and len(key_value_layer.shape) == 5
                and key_value_layer.shape[2] == 2
            ), f"query shape must be [b, s, h, d] and key_value shape must be [b, s, 2, h, d]" \
               f"for dot product cross attention. The actual shape is q: {query_layer.shape}" \
               f"kv: {key_value_layer.shape}"
            q = query_layer
            k = key_value_layer[:, :, 0]
            v = key_value_layer[:, :, 1]

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
        out = paddle.transpose(out, perm=[0, 2, 1, 3])    # [b, s, h, d]
        # out = paddle.reshape(x=out, shape=[0, 0, out.shape[2] * out.shape[3]])
        return out


class MultiHeadAttention(TransformerEngineBaseLayer):
    """Attention w/ QKV and Proj Gemms

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
    zero_centered_gamma: bool, default = `False`
                    whether to zero initialize the gamma of the layernorm operation.
    backend: {'transformer_engine', 'paddle'}, default = `transformer_engine`
                backend to use for attention operation.
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        layernorm_epsilon: float = 1e-5,
        weight_attr: Union[paddle.ParamAttr, None] = None,
        bias_attr: Union[paddle.ParamAttr, None, bool] = None,
        attn_mask_type: str = "causal",
        params_dtype: Optional[paddle.dtype] = None,
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        zero_centered_gamma: bool = False,
        backend: str = 'transformer_engine',
    ) -> None:
        super().__init__()
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.return_layernorm_output = return_layernorm_output
        self.params_dtype = paddle.get_default_dtype() if params_dtype is None else params_dtype
        self.weight_attr = weight_attr
        self.bias_attr = bias_attr
        self.attn_mask_type = attn_mask_type

        assert attention_type in AttnTypes, f"attention_type {attention_type} not supported"

        self.hidden_size_per_attention_head = hidden_size // num_attention_heads
        self.num_attention_heads = num_attention_heads
        norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        self.backend = backend

        if self.attention_type == "self":
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormLinear(
                    hidden_size,
                    3 * hidden_size,
                    eps=layernorm_epsilon,
                    weight_attr=self.weight_attr,
                    bias_attr=self.bias_attr,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    backend=self.backend,
                )
            else:
                self.qkv = Linear(
                    hidden_size,
                    3 * hidden_size,
                    self.weight_attr,
                    self.bias_attr,
                    backend=self.backend,
                )

        else:    # cross attention
            if self.input_layernorm:
                self.layernorm_query = LayerNormLinear(
                    hidden_size,
                    hidden_size,
                    eps=layernorm_epsilon,
                    weight_attr=self.weight_attr,
                    bias_attr=self.bias_attr,
                    return_layernorm_output=return_layernorm_output,
                    zero_centered_gamma=zero_centered_gamma,
                    backend=self.backend,
                )
            else:
                self.query_layer = Linear(
                    hidden_size,
                    hidden_size,
                    self.weight_attr,
                    self.bias_attr,
                    backend=self.backend,
                )
            self.key_value = Linear(
                hidden_size,
                2 * hidden_size,
                self.weight_attr,
                self.bias_attr,
                backend=self.backend,
            )

        # Attention.
        self.core_attention = DotProductAttention(
            norm_factor,
            attention_dropout,
            attn_mask_type=attn_mask_type,
            attention_type=self.attention_type,
            backend=self.backend,
        )

        # Linear
        self.proj = Linear(
            hidden_size,
            hidden_size,
            self.weight_attr,
            self.bias_attr,
            backend=self.backend,
        )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_output: Optional[paddle.Tensor] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[paddle.Tensor] = None,
        set_zero: bool = True,
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
        core_attention_bias_type: str, default = `no_bias`
                                only support no_bias type currently, {`no_bias`}
        core_attention_bias: Optional[paddle.Tensor], default = `None`
                    Bias tensor for Q * K.T
        set_zero: bool, defautl = `True`
                    Whether to use the fast path to set output tensors to 0 or not.

        """

        # hidden_states: [b, s_q, hidden_size]
        if self.attn_mask_type != "causal" and attention_mask is not None:
            assert (attention_mask.dtype == paddle.bool), "Attention mask must be a boolean tensor"

        if self.attention_type == "self":
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(hidden_states)
                if self.return_layernorm_output:
                    mixed_qkv_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_qkv_layer = layernorm_qkv_outputs
            else:
                mixed_qkv_layer = self.qkv(hidden_states)

            # [b, s_q, 3 * hidden_size] --> [b, s_q, 3, num_heads, head_size]
            mixed_qkv_layer = mixed_qkv_layer.reshape(
                shape=[0, 0, 3, self.num_attention_heads, self.hidden_size_per_attention_head])

            context_layer = self.core_attention(
                query_layer=mixed_qkv_layer,
                key_value_layer=None,
                attention_mask=attention_mask,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
                set_zero=set_zero,
            )

        else:    # cross attention
            mixed_kv_layer = self.key_value(encoder_output)
            # [b, s_kv, 2 * hidden_size] --> [b, s_kv, 2, num_heads, head_size]
            mixed_kv_layer = mixed_kv_layer.reshape(
                shape=[0, 0, 2, self.num_attention_heads, self.hidden_size_per_attention_head])

            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(hidden_states)
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query_layer(hidden_states)

            query_layer = query_layer.reshape(
                shape=[0, 0, self.num_attention_heads, self.hidden_size_per_attention_head])
            context_layer = self.core_attention(
                query_layer=query_layer,
                key_value_layer=mixed_kv_layer,
                attention_mask=attention_mask,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
                set_zero=set_zero,
            )

        context_layer = paddle.reshape(context_layer,
                                       [0, 0, context_layer.shape[2] * context_layer.shape[3]])
        # Output. [b, s, hidden]
        attention_output = self.proj(context_layer)

        if self.input_layernorm and self.return_layernorm_output:
            return attention_output, layernorm_output
        return attention_output
