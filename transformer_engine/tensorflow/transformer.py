"""Transformer."""

from typing import Callable, Optional, Tuple, Union

import os
import tensorflow as tf

from contextlib import nullcontext
from keras import backend, layers, initializers
from keras.mixed_precision import autocast_variable
from tensorflow.tools.docs import doc_controls
from transformer_engine.tensorflow import (
    LayerNorm,
    LayerNormDense,
    LayerNormMLP,
    Dense,
)

from .softmax import FusedScaleMaskSoftmax
from .constants import (
    AttnMaskTypes,
    AttnTypes,
    LayerTypes,
)
from .utils import (
    divide,
    attention_mask_func,
)
from .jit import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)


class CoreAttention(tf.keras.Model): # pylint: disable=too-few-public-methods
    """Parallel attention w/o QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
        layer_number: Optional[int] = None,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        attn_mask_type: str = "causal",
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
        self.hidden_size_per_partition = divide(projection_size, 1)
        self.hidden_size_per_attention_head = divide(
            projection_size, num_attention_heads
        )

        self.attention_dropout_ctx = nullcontext

        coeff = None
        self.norm_factor = tf.math.sqrt(
            float(self.hidden_size_per_attention_head))
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
        self.attention_dropout = layers.Dropout(attention_dropout)

    def __call__(
        self,
        query_layer: tf.Tensor,
        key_layer: tf.Tensor,
        value_layer: tf.Tensor,
        attention_mask: tf.Tensor,
    ) -> tf.Tensor:
        """core attention fprop"""
        # [b, np, sq, sk]
        output_size = (
            query_layer.shape[1],
            query_layer.shape[2],
            query_layer.shape[0],
            key_layer.shape[0],
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        new_q_shape = (output_size[2], output_size[0] * output_size[1], -1)
        query_layer = tf.reshape(query_layer, new_q_shape)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        new_k_shape = (output_size[3], output_size[0] * output_size[1], -1)
        key_layer = tf.reshape(key_layer, new_k_shape)

        norm_factor = self._maybe_cast_inputs(self.norm_factor)
        # Raw attention scores. [b * np, sq, sk]
        matmul_result = (
            tf.matmul(
                tf.transpose(query_layer, perm=(1, 0, 2)),  # [b * np, sq, hn]
                tf.transpose(key_layer, perm=(1, 2, 0)),  # [b * np, hn, sk]
            )
            / norm_factor
        )

        # change view to [b, np, sq, sk]
        attention_scores = tf.reshape(matmul_result, output_size)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with self.attention_dropout_ctx():
            attention_probs = self.attention_dropout(attention_probs)

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        output_size = (
            value_layer.shape[1],
            value_layer.shape[2],
            query_layer.shape[0],
            value_layer.shape[3],
        )

        # change view [sk, b * np, hn]
        new_v_shape = (value_layer.shape[0], output_size[0] * output_size[1],
                       -1)
        value_layer = tf.reshape(value_layer, new_v_shape)

        # change view [b * np, sq, sk]
        new_attn_shape = (output_size[0] * output_size[1], output_size[2], -1)
        attention_probs = tf.reshape(attention_probs, new_attn_shape)

        # matmul: [b * np, sq, hn]
        context_layer = tf.matmul(
            attention_probs,  # [b * np, sq, sk]
            tf.transpose(value_layer, perm=(1, 0, 2)),  # [b * np, sk, hn]
        )

        # change view [b, np, sq, hn]
        context_layer = tf.reshape(context_layer, output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = tf.transpose(context_layer, perm=(2, 0, 1, 3))

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = (
            *context_layer.shape[:-2],
            self.hidden_size_per_partition,
        )
        context_layer = tf.reshape(context_layer, new_context_layer_shape)

        return context_layer


class MultiHeadAttention(layers.Layer):
    """Parallel attention w/ QKV and Proj Gemms
    BMM1 -> softmax + dropout -> BMM2
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        kv_channels: int,
        attention_dropout: float,
        layernorm_epsilon: float = 1e-3,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        attn_mask_type: str = "causal",
        return_layernorm_output: bool = False,
        input_layernorm: bool = False,
        attention_type: str = "self",
        fuse_qkv_params: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_number = (layer_number,)
        self.input_layernorm = input_layernorm
        self.attention_type = attention_type
        self.return_layernorm_output = return_layernorm_output
        self.init_method = init_method
        self.fuse_qkv_params = fuse_qkv_params
        # We only support zero-initializer for bias weights.
        self.bias_initializer = initializers.get("zeros")

        assert (
            attention_type in AttnTypes
        ), f"attention_type {attention_type} not supported"

        self.hidden_size_per_attention_head = kv_channels
        self.num_attention_heads_per_partition = divide(num_attention_heads, 1)

        if self.attention_type == "self":
            if self.input_layernorm:
                self.layernorm_qkv = LayerNormDense(
                    3 * hidden_size,
                    epsilon=layernorm_epsilon,
                    kernel_initializer=init_method,
                    use_bias=True,
                    return_bias=False,
                    return_layernorm_output=return_layernorm_output,
                    skip_weight_param_allocation=not fuse_qkv_params,
                )
            else:
                self.qkv = Dense(
                    3 * hidden_size,
                    kernel_initializer=init_method,
                    use_bias=True,
                    return_bias=False,
                    skip_weight_param_allocation=not fuse_qkv_params,
                )
        else:
            if self.input_layernorm:
                self.layernorm_query = LayerNormDense(
                    hidden_size,
                    epsilon=layernorm_epsilon,
                    kernel_initializer=init_method,
                    use_bias=True,
                    return_bias=False,
                    return_layernorm_output=return_layernorm_output,
                    skip_weight_param_allocation=not fuse_qkv_params,
                )
            else:
                self.query_layer = Dense(
                    hidden_size,
                    kernel_initializer=init_method,
                    use_bias=True,
                    return_bias=False,
                    skip_weight_param_allocation=not fuse_qkv_params,
                )
            self.key_value = Dense(
                2 * hidden_size,
                kernel_initializer=init_method,
                use_bias=True,
                return_bias=False,
                skip_weight_param_allocation=not fuse_qkv_params,
            )

        # Core Self attention.
        self.core_attention = CoreAttention(
            num_attention_heads,
            kv_channels,
            attention_dropout,
            layer_number=layer_number,
            apply_query_key_layer_scaling=apply_query_key_layer_scaling,
            attention_softmax_in_fp32=attention_softmax_in_fp32,
            attn_mask_type=attn_mask_type,
        )

        # Linear
        self.proj = Dense(
            hidden_size,
            kernel_initializer=output_layer_init_method,
            use_bias=False,
            return_bias=True,
        )

    @doc_controls.do_not_generate_docs
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        last_dim = tf.compat.dimension_value(input_shape[-1])
        if last_dim is None:
            raise ValueError(
                "The last dimension of the inputs to a Dense layer should be "
                f"defined. Found None. Full input shape received: {input_shape}"
            )

        if not self.fuse_qkv_params:
            self.set_qkv_params(
                last_dim,
                3 * self.hidden_size,
                use_bias=True,
            )

    def set_qkv_params(
        self,
        in_features,
        out_features,
        use_bias: bool = False,
    ) -> None:
        """Initialize separate Parameters for query, key, and value tensors."""

        assert (
            out_features % 3 == 0
        ), f"3 way QKV split with dimension {out_features} not possible."

        qkv_dim = out_features // 3
        if self.attention_type == "self":
            self.qkv_weight = self.add_weight(
                name="qkv_kernel",
                shape=(in_features, out_features),
                initializer=self.init_method,
                trainable=True,
            )

            self.qkv_bias = None
            if use_bias:
                self.qkv_bias = self.add_weight(
                    name="qkv_bias",
                    shape=(out_features,),
                    initializer=self.bias_initializer,
                    trainable=True,
                )
        else:
            self.q_weight = self.add_weight(
                name="q_kernel",
                shape=(in_features, qkv_dim),
                initializer=self.init_method,
                trainable=True,
            )
            self.kv_weight = self.add_weight(
                name="kv_kernel",
                shape=(in_features, 2 * qkv_dim),
                initializer=self.init_method,
                trainable=True,
            )

            self.q_bias = None
            self.kv_bias = None
            if use_bias:
                self.q_bias = self.add_weight(
                    name="q_bias",
                    shape=(qkv_dim,),
                    initializer=self.bias_initializer,
                    trainable=True,
                )
                self.kv_bias = self.add_weight(
                    name="kv_bias",
                    shape=(2 * qkv_dim,),
                    initializer=self.bias_initializer,
                    trainable=True,
                )

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed
            # from model.
            training = False
        return training

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        encoder_output: Optional[tf.Tensor] = None,
        training: bool = None,
    ) -> Tuple[Union[tf.Tensor, None], ...]:
        """MultiHeadAttention FWD"""
        training = self._get_training_value(training)
        # hidden_states: [sq, b, h]

        if attention_mask is not None:
            assert (
                attention_mask.dtype == tf.bool
            ), "Attention mask must be a boolean tensor"

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == "self":
            qkv_weight = self.qkv_weight if not self.fuse_qkv_params else None
            qkv_bias = self.qkv_bias if not self.fuse_qkv_params else None

            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            if self.input_layernorm:
                layernorm_qkv_outputs = self.layernorm_qkv(
                    hidden_states,
                    kernel=qkv_weight,
                    bias=qkv_bias,
                    training=training,
                )
                if self.return_layernorm_output:
                    mixed_x_layer, layernorm_output = layernorm_qkv_outputs
                else:
                    mixed_x_layer = layernorm_qkv_outputs
            else:
                mixed_x_layer = self.qkv(
                    hidden_states,
                    kernel=qkv_weight,
                    bias=qkv_bias,
                    training=training,
                )

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = (
                *mixed_x_layer.shape[:-1],
                self.num_attention_heads_per_partition,
                3 * self.hidden_size_per_attention_head,
            )
            mixed_x_layer = tf.reshape(mixed_x_layer, new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            query_layer, key_layer, value_layer = tf.split(
                mixed_x_layer, num_or_size_splits=3, axis=-1
            )
        else:
            kv_weight = self.kv_weight if not self.fuse_qkv_params else None
            kv_bias = self.kv_bias if not self.fuse_qkv_params else None

            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer = self.key_value(
                encoder_output,
                kernel=kv_weight,
                bias=kv_bias,
                training=training,
            )

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = (
                *mixed_kv_layer.shape[:-1],
                self.num_attention_heads_per_partition,
                2 * self.hidden_size_per_attention_head,
            )
            mixed_kv_layer = tf.reshape(mixed_kv_layer, new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            key_layer, value_layer = tf.split(
                mixed_kv_layer, num_or_size_splits=2, axis=-1
            )

            # Attention head [sq, b, h] --> [sq, b, hp]
            if self.input_layernorm:
                layernorm_query_outputs = self.layernorm_query(
                    hidden_states,
                    kernel=self.q_weight,
                    bias=self.q_bias,
                    training=training,
                )
                if self.return_layernorm_output:
                    query_layer, layernorm_output = layernorm_query_outputs
                else:
                    query_layer = layernorm_query_outputs
            else:
                query_layer = self.query_layer(
                    hidden_states,
                    kernel=self.q_weight,
                    bias=self.q_bias,
                    training=training,
                )

            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = (
                *query_layer.shape[:-1],
                self.num_attention_heads_per_partition,
                self.hidden_size_per_attention_head,
            )
            query_layer = tf.reshape(query_layer, new_tensor_shape)

        # ==================================
        # core attention computation
        # ==================================

        context_layer = self.core_attention(
            query_layer, key_layer, value_layer, attention_mask
        )

        # =================
        # Output. [sq, b, h]
        # =================

        attention_output, attention_bias = self.proj(
            context_layer,
            training=training,
        )

        if self.input_layernorm and self.return_layernorm_output:
            return attention_output, attention_bias, layernorm_output
        return attention_output, attention_bias


class DropPath(tf.keras.Model): # pylint: disable=too-few-public-methods
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def __call__(self, hidden_state: tf.Tensor, training: bool) -> tf.Tensor:
        """DropPath FWD"""
        if self.drop_prob == 0.0 or not training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (hidden_state.shape[0],) + (1,) * (len(hidden_state.shape) - 1)
        # TODO(kaixih): We set the seed mainly for debugging purpose. Should
        # allow users to turn it off.
        random_tensor = tf.random.stateless_uniform(shape, seed=[1, 0])
        random_mask = tf.cast(random_tensor <= keep_prob,
                              dtype=hidden_state.dtype)
        output = (hidden_state / keep_prob) * random_mask
        return output


class TransformerLayer(tf.keras.Model): # pylint: disable=too-few-public-methods
    """
    TransformerLayer is made up of an attention block and a feedforward network
    (MLP). This standard layer is based on the paper
    "Attention Is All You Need".

    Parameters
    ----------
    hidden_size : int
      size of each input sample.
    ffn_hidden_size : int
      intermediate size to which input samples are projected.
    num_attention_heads : int
      number of attention heads in the transformer layer.
    layernorm_epsilon : float, default = 1e-5
      a value added to the denominator of layer normalization for numerical
      stability.
    hidden_dropout: float, default = 0.1
      dropout probability for the dropout op after FC2 layer.
    attention_dropout: float, default = 0.1
      dropout probability for the dropout op during multi-head attention.
    init_method : Callable, default = `None`
      used for initializing weights of QKV and FC1 weights in the following way:
      `init_method(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    output_layer_init_method : Callable, default = `None`
      used for initializing weights of PROJ and FC2 in the following way:
      `output_layer_init_method(weight)`. When set to `None`, defaults to
      `tf.keras.initializers.RandomNormal(mean=0.0, std=0.023)`.
    apply_residual_connection_post_layernorm : bool, default = `False`
      if set to `True`, residual connections are taken from the output of layer
      norm (default is taken from input of layer norm)
    layer_number: int, default = `None`
      layer number of the current `TransformerLayer` when multiple such modules
      are concatenated to form a transformer block.
    apply_query_key_layer_scaling: bool, default = `True`
      apply query-key layer scaling during BMM1 by a factor of `layer_number`
    output_layernorm: bool, default = `False`
      if set to `True`, layer normalization is applied on the output side, after
      the final dropout-add. default behavior is to apply layer normalization on
      the input side, before the QKV transformation.
    attention_softmax_in_fp32: bool, default = `False`
      if set to `True`, softmax is executed in tf.float32 dtype (single
      precision)
    layer_type: {'encoder', 'decoder'}, default = `encoder`
      if set to `decoder`, an additional cross-attn block is added after
      self-attn. This can be used for structures like `T5` Transformer in
      conjunction with the `encoder` option.
    kv_channels: int, default = `None`
      number of key-value channels. defaults to
      `hidden_size / num_attention_heads` if `None`.
    self_attn_mask_type: {'causal', 'padding'}, default = `causal`
      type of attention mask passed into softmax operation.

    Optimization parameters
    -----------------------
    drop_path_rate: float, default = 0.0
      when > 0.0, applies stochastic depth per sample in the main path of the
      residual block.
    fuse_qkv_params: bool, default = 'False'
      if set to `True`, `TransformerLayer` module exposes a single fused
      parameter for query-key-value. This enables optimizations such as QKV
      fusion without concatentations/splits.
    """

    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        num_attention_heads: int,
        layernorm_epsilon: float = 1e-5,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        init_method: Optional[Callable] = None,
        output_layer_init_method: Optional[Callable] = None,
        layer_number: Optional[int] = None,
        kv_channels: Optional[int] = None,
        self_attn_mask_type: str = "causal",
        apply_query_key_layer_scaling: bool = True,
        attention_softmax_in_fp32: bool = False,
        apply_residual_connection_post_layernorm: bool = False,
        output_layernorm: bool = False,
        layer_type: str = "encoder",
        drop_path_rate: float = 0.0,
        fuse_qkv_params: bool = False,
    ) -> None:
        super().__init__()

        bias_dropout_fusion = \
            bool(int(os.getenv("NVTE_BIAS_DROPOUT_FUSION", "1")))
        self.layer_number = layer_number
        self.output_layernorm = output_layernorm
        self.layer_type = layer_type
        self.apply_residual_connection_post_layernorm = (
            apply_residual_connection_post_layernorm
        )
        assert (
            self_attn_mask_type in AttnMaskTypes
        ), f"self_attn_mask_type {self_attn_mask_type} not supported"
        assert layer_type in LayerTypes, \
            f"layer_type {layer_type} not supported"

        self.kv_channels = (
            kv_channels if kv_channels else (hidden_size // num_attention_heads)
        )

        if init_method is None:
            init_method = initializers.RandomNormal(mean=0.0, stddev=0.023)
        if output_layer_init_method is None:
            output_layer_init_method = initializers.RandomNormal(mean=0.0,
                                                                 stddev=0.023)

        attention_args = (
            hidden_size,
            num_attention_heads,
            self.kv_channels,
            attention_dropout,
            layernorm_epsilon,
            init_method,
            output_layer_init_method,
        )
        common_attention_kwargs = {
            "layer_number": layer_number,
            "apply_query_key_layer_scaling": apply_query_key_layer_scaling,
            "attention_softmax_in_fp32": attention_softmax_in_fp32,
            "return_layernorm_output": apply_residual_connection_post_layernorm,
            "fuse_qkv_params": fuse_qkv_params,
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

        # LayerNorm -> gelu(Linear + Bias) -> Linear
        self.layernorm_mlp = LayerNormMLP(
            hidden_size,
            ffn_hidden_size,
            epsilon=layernorm_epsilon,
            kernel_initializer=init_method,
            ffn_kernel_initializer=output_layer_init_method,
            use_bias=False,
            return_bias=True,
            return_layernorm_output=apply_residual_connection_post_layernorm,
        )

        self.hidden_dropout = hidden_dropout
        self.bias_dropout_fusion = bias_dropout_fusion
        self.drop_path = (DropPath(drop_path_rate) if drop_path_rate > 0.0 else
                          None)

        if self.output_layernorm:
            self.layernorm = LayerNorm(
                epsilon=layernorm_epsilon,
            )

    def _get_training_value(self, training=None):
        if training is None:
            training = backend.learning_phase()
        if isinstance(training, int):
            training = bool(training)
        if not self.trainable:
            # When the layer is not trainable, it overrides the value passed
            # from model.
            training = False
        return training

    def __call__(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        encoder_output: Optional[tf.Tensor] = None,
        enc_dec_attn_mask: Optional[tf.Tensor] = None,
        training: bool = None,
    ) -> tf.Tensor:
        """
        Transformer Layer: attention block and a feedforward network (MLP)

        Parameters
        ----------
        hidden_states : tf.Tensor
          Input tensor.
        attention_mask : tf.Tensor
          Boolean tensor used to mask out self-attention softmax input.
        encoder_output : tf.Tensor
          Output of the encoder block to be fed into the decoder block if using
          `layer_type="decoder"`.
        enc_dec_attn_mask : tf.Tensor
          Boolean tensor used to mask out inter-attention softmax input if using
          `layer_type="decoder"`.
        """
        if attention_mask is not None:
            assert (
                attention_mask.dtype == tf.bool
            ), "Attention mask must be a boolean tensor"

        # Theoretically, the input dtype can be handled by the autocast during
        # the layer call. However, we may use the input (hidden_states) in the
        # residual connection before the layer is called. So, we convert it
        # ahead of time. As for the other input (encoder_output), we can leave
        # the conversion to the inter_attention layer, since it won't be used in
        # the residual connection.
        hidden_states = self._maybe_cast_inputs(hidden_states)

        # Self attention.
        self_attention_outputs = self.self_attention(
            hidden_states,
            attention_mask,
            training=training,
        )

        if (self.apply_residual_connection_post_layernorm and
            not self.output_layernorm):
            attention_output, attention_bias, residual = self_attention_outputs
        else:
            attention_output, attention_bias = self_attention_outputs
            residual = hidden_states

        # Set BDA func.
        if self.bias_dropout_fusion:
            if training:
                bias_dropout_add_func = bias_dropout_add_fused_train
            else:
                bias_dropout_add_func = bias_dropout_add_fused_inference
        else:
            bias_dropout_add_func = get_bias_dropout_add(training)

        # Bias dropout add.
        # The autocast scope is used to enforce the correct dtype for the bias.
        with autocast_variable.enable_auto_cast_variables(
            self._compute_dtype_object):
            if self.drop_path is None:
                bda_output = bias_dropout_add_func(
                    attention_output,
                    attention_bias,
                    residual,
                    self.hidden_dropout,
                )
            else:
                # TODO(kaixih): Use stateless_dropout and specify the seed
                # mainly for debugging purpose. Should allow random seed.
                out = (
                    tf.nn.experimental.stateless_dropout(
                        attention_output + attention_bias,
                        rate=self.hidden_dropout,
                        seed=[1, 0],
                    )
                    if training
                    else attention_output + attention_bias
                )
                bda_output = residual + self.drop_path(out, training)

        # Cross attention.
        if self.layer_type == "decoder":
            inter_attention_outputs = self.inter_attention(
                bda_output,
                enc_dec_attn_mask,
                encoder_output=encoder_output,
                training=training,
            )
            if self.apply_residual_connection_post_layernorm:
                attention_output, attention_bias, residual = \
                    inter_attention_outputs
            else:
                attention_output, attention_bias = inter_attention_outputs
                residual = bda_output

            # The autocast scope is used to enforce the correct dtype for the
            # bias.
            with autocast_variable.enable_auto_cast_variables(
                self._compute_dtype_object
            ):
                bda_output = bias_dropout_add_func(
                    attention_output,
                    attention_bias,
                    residual,
                    self.hidden_dropout,
                )

        # MLP.
        mlp_outputs = self.layernorm_mlp(
            bda_output,
            training=training,
        )
        if self.apply_residual_connection_post_layernorm:
            mlp_output, mlp_bias, residual = mlp_outputs
        else:
            mlp_output, mlp_bias = mlp_outputs
            residual = bda_output

        # Bias dropout add.
        # The autocast scope is used to enforce the correct dtype for the bias.
        with autocast_variable.enable_auto_cast_variables(
            self._compute_dtype_object):
            if self.drop_path is None:
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias,
                    residual,
                    self.hidden_dropout,
                )
            else:
                # TODO(kaixih): Use stateless_dropout and specify the seed
                # mainly for debugging purpose. Should allow random seed.
                output = (
                    tf.nn.experimental.stateless_dropout(
                        mlp_output + mlp_bias,
                        rate=self.hidden_dropout,
                        seed=[1, 0],
                    )
                    if training
                    else mlp_output + mlp_bias
                )
                output = residual + self.drop_path(output, training)

        # For BERT like architectures.
        if self.output_layernorm:
            output = self.layernorm(output)

        # output: [b, s, h]
        return output
