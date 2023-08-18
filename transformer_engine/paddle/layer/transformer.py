# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Transformer"""

from typing import Optional, Union

import paddle

from transformer_engine.paddle.constants import (
    AttnMaskTypes,
    LayerTypes,
)
from transformer_engine.paddle.layer import (LayerNormMLP, LayerNorm, MultiHeadAttention)
from .base import TransformerEngineBaseLayer


class TransformerLayer(TransformerEngineBaseLayer):
    r"""
    TransformerLayer is made up of an attention block and a feedforward network (MLP).
    This standard layer is based on the paper "Attention Is All You Need".

    Parameters
    ----------
    hidden_size : int
                 size of each input sample.
    ffn_hidden_size : int
                     intermediate size to which input samples are projected.
    num_attention_heads : int
                         number of attention heads in the transformer layer.
    layernorm_epsilon : float, default = 1e-5
                       a value added to the denominator of layer normalization
                       for numerical stability.
    hidden_dropout: float, default = 0.1
                   dropout probability for the dropout op after FC2 layer.
    attention_dropout: float, default = 0.1
                      dropout probability for the dropout op during multi-head attention.
    self_attn_mask_type: {'causal', 'padding'}, default = `causal`
                        type of attention mask passed into softmax operation.
    apply_residual_connection_post_layernorm : bool, default = `False`
                                              if set to `True`, residual connections are taken
                                              from the output of layer norm (default is taken
                                              from input of layer norm)
    output_layernorm: bool, default = `False`
                     if set to `True`, layer normalization is applied on the output side,
                     after the final dropout-add. default behavior is to apply layer
                     normalization on the input side, before the QKV transformation.
    layer_type: {'encoder', 'decoder'}, default = `encoder`
               if set to `decoder`, an additional cross-attn block is added after self-attn.
               This can be used for structures like `T5` Transformer in conjunction with the
               `encoder` option.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    activation : str, default = 'gelu'
          Type of activation used in MLP block.
          Options are: 'gelu', 'relu', 'reglu', 'geglu' and 'swiglu'.

    params_dtype : paddle.dtype, default = `paddle.get_default_dtype()`
                  it controls the type used to allocate the initial parameters. Useful when
                  the model is trained with lower precision and the original FP32 parameters
                  would not fit in GPU memory.
    """

    def __init__(self,
                 hidden_size: int,
                 ffn_hidden_size: int,
                 num_attention_heads: int,
                 layernorm_epsilon: float = 1e-5,
                 hidden_dropout: float = 0.1,
                 attention_dropout: float = 0.1,
                 weight_attr: Union[paddle.ParamAttr, None] = None,
                 bias_attr: Union[paddle.ParamAttr, None, bool] = None,
                 self_attn_mask_type: str = "causal",
                 params_dtype: Optional[paddle.dtype] = None,
                 apply_residual_connection_post_layernorm: bool = False,
                 output_layernorm: bool = False,
                 layer_type: str = "encoder",
                 zero_centered_gamma: bool = False,
                 activation: str = 'gelu',
                 backend: str = 'transformer_engine') -> None:
        super().__init__()

        params_dtype = paddle.get_default_dtype() if params_dtype is None else params_dtype
        self.output_layernorm = output_layernorm
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.self_attn_mask_type = self_attn_mask_type

        assert (self_attn_mask_type
                in AttnMaskTypes), f"self_attn_mask_type {self_attn_mask_type} not supported"
        assert layer_type in LayerTypes, f"layer_type {layer_type} not supported"

        attention_args = (
            hidden_size,
            num_attention_heads,
            attention_dropout,
            layernorm_epsilon,
            weight_attr,
            bias_attr,
        )
        common_attention_kwargs = {
            "params_dtype": params_dtype,
            "return_layernorm_output": apply_residual_connection_post_layernorm,
            "zero_centered_gamma": zero_centered_gamma,
            "backend": backend,
        }

        self.self_attention = MultiHeadAttention(
            *attention_args,
            **common_attention_kwargs,
            attn_mask_type=self_attn_mask_type,
            input_layernorm=not output_layernorm,
            attention_type="self",
        )

        if layer_type == "decoder":
            self.inter_attention = MultiHeadAttention(
                *attention_args,
                **common_attention_kwargs,
                attn_mask_type="padding",
                input_layernorm=True,
                attention_type="cross",
            )

        self.layernorm_mlp = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            eps=layernorm_epsilon,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            activation=activation,
            return_layernorm_output=apply_residual_connection_post_layernorm,
            zero_centered_gamma=zero_centered_gamma,
            backend=backend,
        )

        self.hidden_dropout = hidden_dropout

        if self.output_layernorm:
            self.layernorm = LayerNorm(
                hidden_size,
                layernorm_epsilon,
                weight_attr,
                bias_attr,
                zero_centered_gamma=zero_centered_gamma,
                backend=backend,
            )

    def forward(
        self,
        hidden_states: paddle.Tensor,
        attention_mask: Optional[paddle.Tensor] = None,
        encoder_output: Optional[paddle.Tensor] = None,
        enc_dec_attn_mask: Optional[paddle.Tensor] = None,
        core_attention_bias_type: str = "no_bias",
        core_attention_bias: Optional[paddle.Tensor] = None,
        set_zero: bool = True,
    ) -> paddle.Tensor:
        """
        Transformer Layer: attention block and a feedforward network (MLP)

        .. note::

            Argument :attr:`attention_mask` will be ignored when :attr:`self_attn_mask_type`
            is set to `"causal"`.

        Parameters
        ----------
        hidden_states : paddle.Tensor
             Input tensor.
        attention_mask : Optional[paddle.Tensor], default = `None`
             Boolean tensor used to mask out self-attention softmax input.
        encoder_output : Optional[paddle.Tensor], default = `None`
             Output of the encoder block to be fed into the decoder block if using
             `layer_type="decoder"`.
        enc_dec_attn_mask : Optional[paddle.Tensor], default = `None`
             Boolean tensor used to mask out inter-attention softmax input if using
             `layer_type="decoder"`.
        core_attention_bias_type: str, default = `no_bias`
        core_attention_bias: Optional[paddle.Tensor], default = `None`
                    Bias tensor for Q * K.T
        set_zero: bool, default = `True`
                    Whether to set output tensors to 0 or not before use.
        """

        if self.self_attn_mask_type != "causal" and attention_mask is not None:
            assert (attention_mask.dtype == paddle.bool), "Attention mask must be a boolean tensor"

        assert core_attention_bias_type in ['no_bias'], f"Only no_bias is supported currently, " \
            f"but receive core_attention_bias_type = {core_attention_bias_type}"

        # Self attention.
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            core_attention_bias_type=core_attention_bias_type,
            core_attention_bias=core_attention_bias,
            set_zero=set_zero,
        )

        if self.apply_residual_connection_post_layernorm and not self.output_layernorm:
            attention_output, residual = self_attention_outputs
        else:
            attention_output = self_attention_outputs
            residual = hidden_states

        # dropoout add.
        out = paddle.nn.functional.dropout(
            attention_output,
            p=self.hidden_dropout,
            training=True,
        )
        bda_output = residual + out

        # Cross attention.
        if self.layer_type == "decoder":
            inter_attention_outputs = self.inter_attention(
                bda_output,
                enc_dec_attn_mask,
                encoder_output=encoder_output,
                core_attention_bias_type=core_attention_bias_type,
                core_attention_bias=core_attention_bias,
                set_zero=set_zero,
            )
            if self.apply_residual_connection_post_layernorm:
                attention_output, residual = inter_attention_outputs
            else:
                attention_output = inter_attention_outputs
                residual = bda_output

            out = paddle.nn.functional.dropout(
                attention_output,
                p=self.hidden_dropout,
                training=True,
            )
            bda_output = residual + out

        # MLP.
        mlp_outputs = self.layernorm_mlp(bda_output)
        if self.apply_residual_connection_post_layernorm:
            mlp_output, residual = mlp_outputs
        else:
            mlp_output = mlp_outputs
            residual = bda_output

        # dropoout add.
        out = paddle.nn.functional.dropout(mlp_output, p=self.hidden_dropout, training=True)
        output = residual + out

        # For BERT like architectures.
        if self.output_layernorm:
            output = self.layernorm(output)

        # output: [b, s, hidden]
        return output
