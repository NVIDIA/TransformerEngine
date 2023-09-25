# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Praxis Modules related Transformer
"""
from functools import partial
from typing import Optional, Sequence, Tuple

from praxis import pax_fiddle
from praxis.base_layer import WeightInit
from praxis.pytypes import JTensor

from .module import TransformerEngineBaseLayer
from ..flax.transformer import TransformerLayerType
from ..flax.transformer import MultiHeadAttention as flax_MultiHeadAttention
from ..flax.transformer import RelativePositionBiases as flax_RelativePositionBiases
from ..flax.transformer import TransformerLayer as flax_TransformerLayer


class RelativePositionBiases(TransformerEngineBaseLayer):
    """RelativePositionBiases"""

    num_buckets: int = 32
    max_distance: int = 128
    num_attention_heads: int = 64
    embedding_init: WeightInit = None
    embedding_axes: Tuple[str, ...] = ()

    @staticmethod
    def generate_embedding_init(init, num_attention_heads, num_buckets):
        """generate_embedding_init"""
        embedding_init = init
        if embedding_init is None:
            rb_stddev = (num_attention_heads * num_buckets)**-0.5
            embedding_init = WeightInit.Gaussian(rb_stddev)
        return embedding_init

    def setup(self) -> None:
        """setup"""
        super().setup()

        embedding_init = RelativePositionBiases.generate_embedding_init(
            self.embedding_init, self.num_attention_heads, self.num_buckets)

        rpb_cls = partial(flax_RelativePositionBiases,
                          num_buckets=self.num_buckets,
                          max_distance=self.max_distance,
                          num_attention_heads=self.num_attention_heads,
                          embedding_init=TransformerEngineBaseLayer.generate_params_init(
                              "rel_embedding", embedding_init),
                          embedding_axes=self.embedding_axes,
                          dtype=self.dtype)

        self.create_layer("relative_position_bias", rpb_cls)

    def __call__(self, q_seqlen: JTensor, k_seqlen: JTensor, bidirectional: bool = True) -> JTensor:
        """__call__"""
        return self.relative_position_bias(q_seqlen, k_seqlen, bidirectional)


class MultiHeadAttention(TransformerEngineBaseLayer):
    """MultiHeadAttention"""

    head_dim: int = 64
    num_heads: int = 16
    dropout_rate: float = 0.
    dropout_rng_name: str = 'dropout'
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    use_bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    attn_mask_type: str = 'causal'
    fuse_qkv: bool = True
    transpose_batch_sequence: bool = True
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    float32_logits: bool = False

    def setup(self) -> None:
        """setup"""
        super().setup()

        mha_cls = partial(
            flax_MultiHeadAttention,
            dtype=self.dtype,
            head_dim=self.head_dim,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            dropout_rng_name=self.dropout_rng_name,
            layernorm_type=self.layernorm_type,
            layernorm_epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", self.params_init),
            use_bias=self.use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", self.bias_init),
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            output_layernorm=self.output_layernorm,
            attn_mask_type=self.attn_mask_type,
            fuse_qkv=self.fuse_qkv,
            transpose_batch_sequence=self.transpose_batch_sequence,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            float32_logits=self.float32_logits)

        self.create_layer("multi_head_attn", mha_cls)

    def __call__(self,
                 inputs_q: JTensor,
                 inputs_kv: JTensor,
                 mask: Optional[JTensor] = None,
                 bias: Optional[JTensor] = None,
                 *,
                 decode: bool = False,
                 deterministic: bool = False) -> JTensor:
        """__call__"""
        return self.multi_head_attn(inputs_q,
                                    inputs_kv,
                                    mask,
                                    bias,
                                    decode=decode,
                                    deterministic=deterministic)


class TransformerLayer(TransformerEngineBaseLayer):
    """TransformerLayer"""

    hidden_size: int = 512
    mlp_hidden_size: int = 2048
    num_attention_heads: int = 8
    layernorm_type: str = 'layernorm'
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    hidden_dropout: float = 0.1
    hidden_dropout_dims: Sequence[int] = ()
    attention_dropout: float = 0.1
    intermediate_dropout: float = 0.1
    intermediate_dropout_dims: Sequence[int] = ()
    dropout_rng_name: str = 'dropout'
    mlp_activations: Sequence[str] = ('relu',)
    use_bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    float32_attention_logits: bool = False
    layer_type: TransformerLayerType = TransformerLayerType.ENCODER
    self_attn_mask_type: str = 'causal'
    enable_relative_embedding: bool = True
    relative_embedding: pax_fiddle.Config[RelativePositionBiases] = pax_fiddle.template_field(None)
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True

    def setup(self) -> None:
        """setup"""
        super().setup()

        relative_embedding_flax_module = None
        if self.enable_relative_embedding and self.relative_embedding is not None:
            assert self.relative_embedding.num_attention_heads == \
                    self.num_attention_heads, \
                "TransformerLayer.relative_embedding.num_attention_heads shoule be" \
                "the same as TransformerLayer.num_attention_heads."

            embedding_init = RelativePositionBiases.generate_embedding_init(
                self.relative_embedding.embedding_init, self.relative_embedding.num_attention_heads,
                self.relative_embedding.num_buckets)

            relative_embedding_flax_module = flax_RelativePositionBiases(
                num_buckets=self.relative_embedding.num_buckets,
                max_distance=self.relative_embedding.max_distance,
                num_attention_heads=self.relative_embedding.num_attention_heads,
                embedding_init=TransformerEngineBaseLayer.generate_params_init(
                    "rel_embedding", embedding_init),
                embedding_axes=self.relative_embedding.embedding_axes,
                dtype=self.relative_embedding.dtype)

        transformerlayer_cls = partial(
            flax_TransformerLayer,
            dtype=self.dtype,
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            num_attention_heads=self.num_attention_heads,
            layernorm_type=self.layernorm_type,
            layernorm_epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            hidden_dropout=self.hidden_dropout,
            hidden_dropout_dims=self.hidden_dropout_dims,
            attention_dropout=self.attention_dropout,
            intermediate_dropout=self.intermediate_dropout,
            intermediate_dropout_dims=self.intermediate_dropout_dims,
            dropout_rng_name=self.dropout_rng_name,
            mha_kernel_init=TransformerEngineBaseLayer.generate_params_init(
                "mha_kernel", self.params_init),
            mlp_kernel_init=TransformerEngineBaseLayer.generate_params_init(
                "mlp_kernel", self.params_init),
            mlp_activations=self.mlp_activations,
            use_bias=self.use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", self.bias_init),
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            output_layernorm=self.output_layernorm,
            float32_attention_logits=self.float32_attention_logits,
            layer_type=self.layer_type,
            self_attn_mask_type=self.self_attn_mask_type,
            enable_relative_embedding=self.enable_relative_embedding,
            relative_embedding=relative_embedding_flax_module,
            drop_path=self.drop_path,
            fuse_qkv_params=self.fuse_qkv_params,
            transpose_batch_sequence=self.transpose_batch_sequence,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init)

        self.create_layer("transformerlayer", transformerlayer_cls)

    def __call__(self,
                 inputs: JTensor,
                 encoded: JTensor = None,
                 attention_mask: JTensor = None,
                 encoder_decoder_mask: JTensor = None,
                 deterministic: bool = False,
                 decode: bool = False,
                 max_decode_length: bool = None) -> JTensor:
        """__call__"""
        return self.transformerlayer(inputs, encoded, attention_mask, encoder_decoder_mask,
                                     deterministic, decode, max_decode_length)
