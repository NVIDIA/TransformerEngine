# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Praxis Modules related Transformer
"""
from functools import partial
from typing import Optional, Sequence, Tuple
import warnings

from praxis import pax_fiddle
from praxis.base_layer import WeightInit
from praxis.pytypes import JTensor

from .module import TransformerEngineBaseLayer
from ..flax.transformer import TransformerLayerType
from ..flax.transformer import DotProductAttention as flax_DotProductAttention
from ..flax.transformer import MultiHeadAttention as flax_MultiHeadAttention
from ..flax.transformer import RelativePositionBiases as flax_RelativePositionBiases
from ..flax.transformer import TransformerLayer as flax_TransformerLayer
from ..attention import AttnBiasType, AttnMaskType


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
            rb_stddev = (num_attention_heads * num_buckets) ** -0.5
            embedding_init = WeightInit.Gaussian(rb_stddev)
        return embedding_init

    def setup(self) -> None:
        """setup"""
        super().setup()

        embedding_init = RelativePositionBiases.generate_embedding_init(
            self.embedding_init, self.num_attention_heads, self.num_buckets
        )

        rpb_cls = partial(
            flax_RelativePositionBiases,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
            num_attention_heads=self.num_attention_heads,
            embedding_init=TransformerEngineBaseLayer.generate_params_init(
                "rel_embedding", embedding_init
            ),
            embedding_axes=self.embedding_axes,
            dtype=self.dtype,
        )

        self.create_layer("relative_position_bias", rpb_cls)

    def __call__(self, q_seqlen: JTensor, k_seqlen: JTensor, bidirectional: bool = True) -> JTensor:
        """__call__"""
        return self.relative_position_bias(q_seqlen, k_seqlen, bidirectional)


class DotProductAttention(TransformerEngineBaseLayer):
    """DotProductAttention"""

    head_dim: int = 0
    num_attention_heads: int = 0
    num_gqa_groups: Optional[int] = None
    attention_dropout: float = 0.0
    attn_mask_type: AttnMaskType = "causal"
    attn_bias_type: AttnBiasType = None
    dropout_rng_name: str = "dropout"
    float32_logits: bool = False
    qkv_layout: str = "bshd_bshd_bshd"
    scale_factor: Optional[float] = None
    transpose_batch_sequence: bool = True

    def setup(self) -> None:
        """setup"""
        super().setup()

        assert self.head_dim > 0, f"{self.head_dim=}"
        assert self.num_attention_heads > 0, f"{self.num_attention_heads=}"

        dpa_cls = partial(
            flax_DotProductAttention,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            attn_mask_type=self.attn_mask_type,
            attn_bias_type=self.attn_bias_type,
            attention_dropout=self.attention_dropout,
            dtype=self.dtype,
            dropout_rng_name=self.dropout_rng_name,
            float32_logits=self.float32_logits,
            qkv_layout=self.qkv_layout,
            scale_factor=self.scale_factor,
            transpose_batch_sequence=self.transpose_batch_sequence,
        )

        self.create_layer("dot_product_attention", dpa_cls)

    def __call__(
        self,
        query: JTensor,
        key: JTensor,
        value: JTensor,
        mask: Optional[JTensor] = None,
        bias: Optional[JTensor] = None,
        *,
        deterministic: bool = False,
    ) -> JTensor:
        """__call__"""
        return self.dot_product_attention(
            query, key, value, mask, bias, deterministic=deterministic
        )


class MultiHeadAttention(TransformerEngineBaseLayer):
    """MultiHeadAttention"""

    head_dim: int = 0
    num_attention_heads: int = 0
    num_gqa_groups: Optional[int] = None
    attention_dropout: float = 0.0
    dropout_rng_name: str = "dropout"
    input_layernorm: bool = True
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    return_layernorm_output: bool = False
    use_bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    attn_mask_type: str = "causal"
    attn_bias_type: Optional[str] = None
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_windows: Tuple[int, int] = (1, 10000)
    rotary_pos_emb_group_method: str = "consecutive"
    low_rank_adaptation_scope: str = "none"
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = True
    enable_sequence_parallel: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True
    float32_logits: bool = False

    # Deprecated parameters
    num_heads: Optional[int] = None
    dropout_rate: Optional[float] = None
    output_layernorm: Optional[bool] = None
    apply_residual_connection_post_layernorm: Optional[bool] = None
    fuse_qkv: Optional[bool] = None

    def __post_init__(self):
        # Deal with the deprecated parameters
        if self.num_heads is not None:
            self.num_attention_heads = self.num_heads
            warnings.warn(
                f"{__class__}.num_heads is deprecated. It will be removed recently. "
                f"Please uses {__class__}.num_attention_heads as the new API.",
                DeprecationWarning,
            )
        if self.dropout_rate is not None:
            self.attention_dropout = self.dropout_rate
            warnings.warn(
                f"{__class__}.dropout_rate is deprecated. It will be removed recently. "
                f"Please use {__class__}.attention_dropout as the new API.",
                DeprecationWarning,
            )
        if self.apply_residual_connection_post_layernorm is not None:
            warnings.warn(
                f"{__class__}.apply_residual_connection_post_layernorm is deprecated. "
                f"It will be removed recently, please use {__class__}.return_layernorm_output.",
                DeprecationWarning,
            )
        if self.fuse_qkv is not None:
            warnings.warn(
                f"{__class__}.fuse_qkv is deprecated. It will be removed recently. "
                f"Please use {__class__}.fuse_qkv_params as the new API.",
                DeprecationWarning,
            )
        assert self.output_layernorm is None, (
            f"{__class__}.output_layernorm is deprecated. It will be removed recently. "
            f"Please use {__class__}.input_layernorm for controlling whether to apply layernorm."
        )

        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_heads
        super().__post_init__()

    def setup(self) -> None:
        """setup"""
        super().setup()

        assert self.head_dim > 0, f"{self.head_dim=}"
        assert self.num_attention_heads > 0, f"{self.num_attention_heads=}"

        mha_cls = partial(
            flax_MultiHeadAttention,
            dtype=self.dtype,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
            attention_dropout=self.attention_dropout,
            dropout_rng_name=self.dropout_rng_name,
            input_layernorm=self.input_layernorm,
            layernorm_type=self.layernorm_type,
            layernorm_epsilon=self.layernorm_epsilon,
            zero_centered_gamma=self.zero_centered_gamma,
            return_layernorm_output=self.return_layernorm_output,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", self.params_init),
            use_bias=self.use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", self.bias_init),
            attn_mask_type=self.attn_mask_type,
            attn_bias_type=self.attn_bias_type,
            enable_rotary_pos_emb=self.enable_rotary_pos_emb,
            rotary_pos_emb_windows=self.rotary_pos_emb_windows,
            rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
            low_rank_adaptation_scope=self.low_rank_adaptation_scope,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            fuse_qkv_params=self.fuse_qkv_params,
            transpose_batch_sequence=self.transpose_batch_sequence,
            enable_sequence_parallel=self.enable_sequence_parallel,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
            float32_logits=self.float32_logits,
        )

        self.create_layer("multi_head_attn", mha_cls)

    def __call__(
        self,
        inputs_q: JTensor,
        inputs_kv: JTensor,
        mask: Optional[JTensor] = None,
        bias: Optional[JTensor] = None,
        *,
        decode: bool = False,
        deterministic: bool = False,
    ) -> JTensor:
        """__call__"""
        return self.multi_head_attn(
            inputs_q, inputs_kv, mask, bias, decode=decode, deterministic=deterministic
        )


class TransformerLayer(TransformerEngineBaseLayer):
    """TransformerLayer"""

    hidden_size: int = 512
    mlp_hidden_size: int = 2048
    num_attention_heads: int = 8
    num_gqa_groups: Optional[int] = None
    layernorm_type: str = "layernorm"
    layernorm_epsilon: float = 1e-6
    zero_centered_gamma: bool = False
    hidden_dropout: float = 0.1
    hidden_dropout_dims: Sequence[int] = ()
    attention_dropout: float = 0.1
    intermediate_dropout: float = 0.1
    intermediate_dropout_dims: Sequence[int] = ()
    dropout_rng_name: str = "dropout"
    mlp_activations: Sequence[str] = ("relu",)
    use_bias: bool = False
    bias_init: WeightInit = WeightInit.Constant(0.0)
    apply_residual_connection_post_layernorm: bool = False
    output_layernorm: bool = False
    float32_attention_logits: bool = False
    layer_type: TransformerLayerType = TransformerLayerType.ENCODER
    self_attn_mask_type: str = "causal"
    self_attn_bias_type: Optional[str] = None
    enable_rotary_pos_emb: bool = False
    rotary_pos_emb_windows: Tuple[int, int] = (1, 10000)
    rotary_pos_emb_group_method: str = "consecutive"
    low_rank_adaptation_scope: str = "none"
    low_rank_adaptation_dim: int = 32
    low_rank_adaptation_alpha: float = None
    enable_relative_embedding: bool = True
    relative_embedding: pax_fiddle.Config[RelativePositionBiases] = pax_fiddle.template_field(None)
    drop_path: float = 0.0
    fuse_qkv_params: bool = True
    transpose_batch_sequence: bool = False
    enable_sequence_parallel: bool = False
    scale_attn_logits: bool = False
    scaled_query_init: bool = True

    def __post_init__(self):
        if self.num_gqa_groups is None:
            self.num_gqa_groups = self.num_attention_heads
        super().__post_init__()

    def setup(self) -> None:
        """setup"""
        super().setup()

        relative_embedding_flax_module = None
        if self.enable_relative_embedding and self.relative_embedding is not None:
            assert self.relative_embedding.num_attention_heads == self.num_attention_heads, (
                "TransformerLayer.relative_embedding.num_attention_heads shoule be"
                "the same as TransformerLayer.num_attention_heads."
            )

            embedding_init = RelativePositionBiases.generate_embedding_init(
                self.relative_embedding.embedding_init,
                self.relative_embedding.num_attention_heads,
                self.relative_embedding.num_buckets,
            )

            relative_embedding_flax_module = flax_RelativePositionBiases(
                num_buckets=self.relative_embedding.num_buckets,
                max_distance=self.relative_embedding.max_distance,
                num_attention_heads=self.relative_embedding.num_attention_heads,
                embedding_init=TransformerEngineBaseLayer.generate_params_init(
                    "rel_embedding", embedding_init
                ),
                embedding_axes=self.relative_embedding.embedding_axes,
                dtype=self.relative_embedding.dtype,
            )

        transformerlayer_cls = partial(
            flax_TransformerLayer,
            dtype=self.dtype,
            hidden_size=self.hidden_size,
            mlp_hidden_size=self.mlp_hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_gqa_groups=self.num_gqa_groups,
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
                "mha_kernel", self.params_init
            ),
            mlp_kernel_init=TransformerEngineBaseLayer.generate_params_init(
                "mlp_kernel", self.params_init
            ),
            mlp_activations=self.mlp_activations,
            use_bias=self.use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", self.bias_init),
            apply_residual_connection_post_layernorm=self.apply_residual_connection_post_layernorm,
            output_layernorm=self.output_layernorm,
            float32_attention_logits=self.float32_attention_logits,
            layer_type=self.layer_type,
            self_attn_mask_type=self.self_attn_mask_type,
            self_attn_bias_type=self.self_attn_bias_type,
            enable_rotary_pos_emb=self.enable_rotary_pos_emb,
            rotary_pos_emb_windows=self.rotary_pos_emb_windows,
            rotary_pos_emb_group_method=self.rotary_pos_emb_group_method,
            low_rank_adaptation_scope=self.low_rank_adaptation_scope,
            low_rank_adaptation_dim=self.low_rank_adaptation_dim,
            low_rank_adaptation_alpha=self.low_rank_adaptation_alpha,
            enable_relative_embedding=self.enable_relative_embedding,
            relative_embedding=relative_embedding_flax_module,
            drop_path=self.drop_path,
            fuse_qkv_params=self.fuse_qkv_params,
            transpose_batch_sequence=self.transpose_batch_sequence,
            enable_sequence_parallel=self.enable_sequence_parallel,
            scale_attn_logits=self.scale_attn_logits,
            scaled_query_init=self.scaled_query_init,
        )

        self.create_layer("transformerlayer", transformerlayer_cls)

    def __call__(
        self,
        inputs: JTensor,
        encoded: JTensor = None,
        attention_mask: JTensor = None,
        encoder_decoder_mask: JTensor = None,
        deterministic: bool = False,
        decode: bool = False,
        max_decode_length: bool = None,
    ) -> JTensor:
        """__call__"""
        return self.transformerlayer(
            inputs,
            encoded,
            attention_mask,
            encoder_decoder_mask,
            deterministic,
            decode,
            max_decode_length,
        )
