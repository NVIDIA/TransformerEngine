# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test transformer_engine.jax.flax.TransformerLayer"""
import os
from functools import partial
from typing import Dict, Tuple, Optional

import flax
import jax
import jax.numpy as jnp
import pytest

from utils import (
    assert_allclose,
    assert_tree_like_allclose,
    dtype_tols,
    sync_params_values,
)
from utils import DecoderLayer as RefDecoderLayer
from utils import EncoderLayer as RefEncoderLayer

from transformer_engine.common import recipe
from transformer_engine.jax.flax import TransformerLayer, TransformerLayerType
from transformer_engine.jax.quantize import (
    get_quantize_config,
    ScalingMode,
    is_fp8_available,
    update_collections,
    TensorSource,
    fp8_autocast,
)
from transformer_engine.jax.sharding import MeshResource


@pytest.fixture(autouse=True, scope="function")
def enable_fused_attn():
    """Enable fused attention"""
    os.environ["NVTE_FUSED_ATTN"] = "1"
    yield
    del os.environ["NVTE_FUSED_ATTN"]


is_fp8_supported, reason = is_fp8_available()
is_mxfp8_supported, reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

QUANTIZE_RECIPES = []
""" Find supported scaling modes"""
if is_fp8_supported:
    QUANTIZE_RECIPES.append(pytest.param(recipe.DelayedScaling(), id="DelayedScaling"))
if is_mxfp8_supported:
    QUANTIZE_RECIPES.append(pytest.param(recipe.MXFP8BlockScaling(), id="MXFP8BlockScaling"))


DATA_SHAPE = [  # (batch, seqlen, emb_dim)
    pytest.param((32, 128, 1024), id="32-128-1024"),
]
DTYPE = [jnp.bfloat16]

_KEY_OF_RESIDUAL_POST_LAYERNORM = "apply_residual_connection_post_layernorm"
_KEY_OF_OUTPUT_LAYERNORM = "output_layernorm"
_KEY_OF_DROP_PATH = "drop_path"
_KEY_OF_FUSE_QKV_PARAMS = "fuse_qkv_params"
_KEY_OF_HIDDEN_DROPOUT = "hidden_dropout"
_KEY_OF_ATTENTION_DROPOUT = "attention_dropout"
_KEY_OF_INTERMEDIATE_DROPOUT = "intermediate_dropout"
_KEY_OF_HIDDEN_DROPOUT_DIMS = "hidden_dropout_dims"
_KEY_OF_INTERMEDIATE_DROPOUT_DIMS = "intermediate_dropout_dims"
_KEY_OF_MLP_ACTIVATIONS = "mlp_activations"
_KEY_OF_LAYERNORM_TYPE = "layernorm_type"
_KEY_OF_LAYERNORM_EPS = "layernorm_epsilon"
_KEY_OF_ZERO_CENTERED_GAMMA = "zero_centered_gamma"
_KEY_OF_TRANSPOSE_BS = "transpose_batch_sequence"
_KEY_OF_SCALE_ATTN_LOGITS = "scale_attn_logits"
_KEY_OF_NUM_HEADS = "num_attention_heads"
_KEY_OF_NUM_GQA_GROUPS = "num_gqa_groups"
_KEY_OF_ENABLE_ROPE = "enable_rotary_pos_emb"
_KEY_OF_ROPE_GROUP_METHOD = "rotary_pos_emb_group_method"
_KEY_OF_SELF_ATTN_BIAS_TYPE = "self_attn_bias_type"
_KEY_OF_SELF_ATTN_MASK_TYPE = "self_attn_mask_type"
_KEY_OF_FLOAT32_ATTENTION_LOGITS = "float32_attention_logits"
_KEY_OF_USE_BIAS = "use_bias"
_KEY_OF_RELATIVE_EMBEDDING = "enable_relative_embedding"
_KEY_OF_WINDOW_SIZE = "window_size"

BASE_ATTRS = {
    _KEY_OF_TRANSPOSE_BS: True,
    _KEY_OF_NUM_HEADS: 8,
    _KEY_OF_HIDDEN_DROPOUT: 0,
    _KEY_OF_ATTENTION_DROPOUT: 0.0,
    _KEY_OF_INTERMEDIATE_DROPOUT: 0,
    _KEY_OF_SELF_ATTN_MASK_TYPE: "padding_causal",
    _KEY_OF_LAYERNORM_TYPE: "layernorm",
    _KEY_OF_WINDOW_SIZE: (-1, -1),
}

ATTRS = [
    # attrs0
    {},
    # attrs1
    {
        _KEY_OF_LAYERNORM_TYPE: "rmsnorm",
    },
    # attrs2
    {
        _KEY_OF_ZERO_CENTERED_GAMMA: True,
        _KEY_OF_LAYERNORM_EPS: 1e-2,
    },
    # attrs3
    {_KEY_OF_LAYERNORM_TYPE: "rmsnorm", _KEY_OF_RESIDUAL_POST_LAYERNORM: True},
    # attrs4
    {_KEY_OF_LAYERNORM_TYPE: "rmsnorm", _KEY_OF_OUTPUT_LAYERNORM: True},
    # attrs5
    {
        _KEY_OF_LAYERNORM_TYPE: "rmsnorm",
        _KEY_OF_RESIDUAL_POST_LAYERNORM: True,
        _KEY_OF_OUTPUT_LAYERNORM: True,
    },
    # attrs6
    {_KEY_OF_LAYERNORM_TYPE: "rmsnorm", _KEY_OF_DROP_PATH: 0.1},
    # attrs7
    {_KEY_OF_LAYERNORM_TYPE: "rmsnorm", _KEY_OF_FUSE_QKV_PARAMS: False},
    # attrs8
    {
        _KEY_OF_LAYERNORM_TYPE: "rmsnorm",
        _KEY_OF_MLP_ACTIVATIONS: ("gelu", "linear"),
    },
    # attrs9
    {
        _KEY_OF_SCALE_ATTN_LOGITS: True,
        _KEY_OF_LAYERNORM_TYPE: "rmsnorm",
        _KEY_OF_HIDDEN_DROPOUT: 0.8,
        _KEY_OF_INTERMEDIATE_DROPOUT: 0.5,
        _KEY_OF_MLP_ACTIVATIONS: ("gelu", "linear"),
        _KEY_OF_USE_BIAS: True,
    },
    # attrs10
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_SCALE_ATTN_LOGITS: True,
        _KEY_OF_LAYERNORM_TYPE: "rmsnorm",
        _KEY_OF_MLP_ACTIVATIONS: ("gelu", "linear"),
    },
    # attrs11
    {
        _KEY_OF_NUM_HEADS: 8,
        _KEY_OF_NUM_GQA_GROUPS: 4,
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_SCALE_ATTN_LOGITS: True,
        _KEY_OF_MLP_ACTIVATIONS: ("gelu",),
        _KEY_OF_USE_BIAS: True,
    },
    # attrs12
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_LAYERNORM_TYPE: "rmsnorm",
        _KEY_OF_NUM_GQA_GROUPS: 1,
        _KEY_OF_ENABLE_ROPE: True,
        _KEY_OF_ROPE_GROUP_METHOD: "consecutive",
        _KEY_OF_FLOAT32_ATTENTION_LOGITS: True,
    },
    # attrs13
    {
        _KEY_OF_TRANSPOSE_BS: True,
        _KEY_OF_ENABLE_ROPE: True,
        _KEY_OF_ROPE_GROUP_METHOD: "consecutive",
        _KEY_OF_USE_BIAS: True,
    },
    # attrs14
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_LAYERNORM_TYPE: "layernorm",
        _KEY_OF_NUM_GQA_GROUPS: 2,
        _KEY_OF_ENABLE_ROPE: True,
        _KEY_OF_ROPE_GROUP_METHOD: "alternate",
        _KEY_OF_USE_BIAS: True,
        _KEY_OF_FLOAT32_ATTENTION_LOGITS: True,
    },
    # attrs15
    {
        _KEY_OF_TRANSPOSE_BS: True,
        _KEY_OF_LAYERNORM_TYPE: "rmsnorm",
        _KEY_OF_ENABLE_ROPE: True,
        _KEY_OF_ROPE_GROUP_METHOD: "alternate",
        _KEY_OF_USE_BIAS: True,
    },
    # attrs16
    {
        _KEY_OF_HIDDEN_DROPOUT: 0.3,
        _KEY_OF_HIDDEN_DROPOUT_DIMS: (0,),
        _KEY_OF_INTERMEDIATE_DROPOUT: 0.5,
        _KEY_OF_INTERMEDIATE_DROPOUT_DIMS: (1,),
    },
    # attrs17
    {
        _KEY_OF_SELF_ATTN_MASK_TYPE: "padding",
        _KEY_OF_USE_BIAS: True,
    },
    # attrs18
    {
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_BIAS_TYPE: "no_bias",
    },
    # attrs19
    {
        _KEY_OF_ATTENTION_DROPOUT: 0.3,
    },
    # attrs20
    {
        _KEY_OF_MLP_ACTIVATIONS: (("relu", "relu")),
    },
    # attrs21
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_MASK_TYPE: "causal",
        _KEY_OF_WINDOW_SIZE: (64, 0),  # Left size must < DATA_SHAPE seqlen
        _KEY_OF_FLOAT32_ATTENTION_LOGITS: True,
    },
    # attrs22
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_MASK_TYPE: "causal",
        _KEY_OF_WINDOW_SIZE: None,
        _KEY_OF_FLOAT32_ATTENTION_LOGITS: True,
    },
    # attrs23
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_MASK_TYPE: "causal",
        _KEY_OF_FLOAT32_ATTENTION_LOGITS: True,
    },
    # attrs24
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_MASK_TYPE: "no_mask",
    },
    # attrs25
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_MASK_TYPE: "no_mask",
        _KEY_OF_WINDOW_SIZE: (2, 2),
    },
    # attrs26
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_MASK_TYPE: "padding",
        _KEY_OF_WINDOW_SIZE: (2, 2),
    },
    # attrs27
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_SELF_ATTN_MASK_TYPE: "padding",
        _KEY_OF_WINDOW_SIZE: None,
    },
    # attrs28
    {
        _KEY_OF_TRANSPOSE_BS: False,
        _KEY_OF_RELATIVE_EMBEDDING: False,
        _KEY_OF_WINDOW_SIZE: (2, 2),
    },
    # attrs29
    {
        _KEY_OF_RELATIVE_EMBEDDING: True,
        _KEY_OF_SELF_ATTN_BIAS_TYPE: "pre_scale_bias",
    },
    # attrs30
    {
        _KEY_OF_RELATIVE_EMBEDDING: True,
        _KEY_OF_SELF_ATTN_BIAS_TYPE: "post_scale_bias",
    },
]

ATTRS = [{**BASE_ATTRS, **attr} for attr in ATTRS]


class BaseRunner:
    """Base runner to define forward and backward tests"""

    layer_type: TransformerLayerType = None
    reference_layer: flax.linen.Module = None
    transformations: Dict[str, str] = None

    def __init__(self, attrs):
        self.attrs = attrs
        self._generate_test_rngs()
        # Disable fused attention for attention dropout because the different dropout impl
        if attrs.get(_KEY_OF_ATTENTION_DROPOUT, False) and os.getenv("NVTE_FUSED_ATTN"):
            os.environ["NVTE_FUSED_ATTN"] = "0"

    def _generate_test_rngs(self):
        root_rng = jax.random.PRNGKey(0)
        params_rng, init_dropout_rng, apply_dropout_rng = jax.random.split(root_rng, 3)
        self.init_rng = {"params": params_rng, "dropout": init_dropout_rng}
        self.apply_rng = {"dropout": apply_dropout_rng}

    def _generate_layer(self, layer_cls, diff_inputs, no_diff_inputs):
        layer = layer_cls()
        variables = layer.init(self.init_rng, *diff_inputs, *no_diff_inputs)
        others, params = flax.core.pop(variables, "params")
        del variables
        return layer, params, others

    def _loss_fn(self, diff_xs, no_diff_xs, params, others, model):
        variables = {"params": params, **others}
        output = model.apply(variables, *diff_xs, *no_diff_xs, rngs=self.apply_rng)
        return jnp.mean(output, dtype=jnp.float32).astype(output.dtype)

    def _sync_params(self, ref, target):
        """Copy the reference params to target"""
        target = sync_params_values(target, ref, self.transformations)
        return ref, target

    def test_forward(
        self,
        data_shape: Tuple[int],
        dtype: jnp.dtype,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> None:
        """Test only the forward"""
        inputs, (ref_masks, test_masks) = self.generate_inputs(data_shape, dtype)

        ref_layer_cls = partial(self.reference_layer, **self.attrs)
        layer_cls = partial(TransformerLayer, layer_type=self.layer_type, **self.attrs)

        ref_layer, ref_params, ref_others = self._generate_layer(ref_layer_cls, inputs, ref_masks)
        test_layer, test_params, test_others = self._generate_layer(layer_cls, inputs, test_masks)
        ref_params, test_params = self._sync_params(ref_params, test_params)

        ref_out = self._loss_fn(inputs, ref_masks, ref_params, ref_others, ref_layer)
        test_out = self._loss_fn(inputs, test_masks, test_params, test_others, test_layer)

        tols = dtype_tols(dtype, rtol=rtol, atol=atol)
        assert_allclose(ref_out, test_out, **tols)

    def test_backward(
        self,
        data_shape: Tuple[int],
        dtype: jnp.dtype,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> None:
        """Test forward and backward through value_and_grad()"""
        inputs, (ref_masks, test_masks) = self.generate_inputs(data_shape, dtype)

        ref_layer_cls = partial(self.reference_layer, **self.attrs)
        layer_cls = partial(TransformerLayer, layer_type=self.layer_type, **self.attrs)

        ref_layer, ref_params, ref_others = self._generate_layer(ref_layer_cls, inputs, ref_masks)
        test_layer, test_params, test_others = self._generate_layer(layer_cls, inputs, test_masks)

        ref_params, test_params = self._sync_params(ref_params, test_params)

        if get_quantize_config().is_fp8_enabled():
            for _ in range(4):
                _, updated_state = jax.value_and_grad(self._loss_fn, argnums=(3,), has_aux=False)(
                    inputs,
                    test_masks,
                    test_params,
                    test_others,
                    test_layer,
                )
                if (
                    get_quantize_config().get_scaling_mode(TensorSource.X)
                    == ScalingMode.DELAYED_TENSOR_SCALING
                ):
                    _, updated_quantize_meta = flax.core.pop(
                        updated_state[0], get_quantize_config().COLLECTION_NAME
                    )
                    test_others = update_collections(
                        {get_quantize_config().COLLECTION_NAME: updated_quantize_meta}, test_others
                    )
                    del updated_quantize_meta
                del updated_state

        grad_fn = jax.value_and_grad(self._loss_fn, argnums=(0, 2), has_aux=False)

        ref_out, (ref_dgrads, ref_wgrads) = grad_fn(
            inputs, ref_masks, ref_params, ref_others, ref_layer
        )
        test_out, (test_dgrads, test_wgrads) = grad_fn(
            inputs, test_masks, test_params, test_others, test_layer
        )

        tols = dtype_tols(dtype, rtol=rtol, atol=atol)
        assert_allclose(ref_out, test_out, **tols)
        assert_tree_like_allclose(ref_dgrads, test_dgrads, **tols)

        _, restructed_ref_wgrads = self._sync_params(ref_wgrads, test_wgrads)
        assert_tree_like_allclose(restructed_ref_wgrads, test_wgrads, **tols)


class EncoderRunner(BaseRunner):
    """Encoder runner implementations"""

    layer_type = TransformerLayerType.ENCODER
    reference_layer = RefEncoderLayer
    transformations = {
        "attention/qkv/scale": "pre_attention_layer_norm/scale",
        "attention/qkv/ln_bias": "pre_attention_layer_norm/ln_bias",
        "attention/query/scale": "pre_attention_layer_norm/scale",
        "attention/query/ln_bias": "pre_attention_layer_norm/ln_bias",
        "mlp/wi_kernel": "mlp/wi/kernel",
        "mlp/wi_bias": "mlp/wi/bias",
        "mlp/wo_kernel": "mlp/wo/kernel",
        "mlp/wo_bias": "mlp/wo/bias",
        "mlp/scale": "pre_mlp_layer_norm/scale",
        "mlp/ln_bias": "pre_mlp_layer_norm/ln_bias",
    }

    def generate_inputs(self, data_shape, dtype):
        """
        Return inputs, (ref_masks, test_masks)
        """
        transpose_batch_sequence = self.attrs[_KEY_OF_TRANSPOSE_BS]
        batch, seqlen = data_shape[:2]
        if transpose_batch_sequence:
            data_shape = (data_shape[1], data_shape[0], *data_shape[2:])

        data_rng = jax.random.PRNGKey(2024)
        inputs = (jax.random.normal(data_rng, data_shape, dtype),)

        mask_shape = (batch, 1, seqlen, seqlen)
        padded_mask = jnp.zeros(mask_shape, dtype=jnp.uint8)
        causal_mask = jnp.triu(jnp.ones(mask_shape, dtype=jnp.uint8), k=1)
        if self.attrs[_KEY_OF_SELF_ATTN_MASK_TYPE] in ["causal", "padding_causal"]:
            mask = causal_mask
        else:
            mask = padded_mask
        ref_masks = (1 - mask,)
        test_masks = (None, mask)  # The second arg of Transformer is encoded tokens.

        return inputs, (ref_masks, test_masks)


class DecoderRunner(BaseRunner):
    """
    Decoder runner implementations
    """

    layer_type = TransformerLayerType.DECODER
    reference_layer = RefDecoderLayer
    transformations = {
        "encoder_decoder_attention/qkv/scale": "pre_cross_attention_layer_norm/scale",
        "encoder_decoder_attention/qkv/ln_bias": "pre_cross_attention_layer_norm/ln_bias",
        "encoder_decoder_attention/query/scale": "pre_cross_attention_layer_norm/scale",
        "encoder_decoder_attention/query/ln_bias": "pre_cross_attention_layer_norm/ln_bias",
        "self_attention/qkv/scale": "pre_self_attention_layer_norm/scale",
        "self_attention/qkv/ln_bias": "pre_self_attention_layer_norm/ln_bias",
        "self_attention/query/scale": "pre_self_attention_layer_norm/scale",
        "self_attention/query/ln_bias": "pre_self_attention_layer_norm/ln_bias",
        "mlp/wi_kernel": "mlp/wi/kernel",
        "mlp/wi_bias": "mlp/wi/bias",
        "mlp/wo_kernel": "mlp/wo/kernel",
        "mlp/wo_bias": "mlp/wo/bias",
        "mlp/scale": "pre_mlp_layer_norm/scale",
        "mlp/ln_bias": "pre_mlp_layer_norm/ln_bias",
    }

    def generate_inputs(self, data_shape, dtype):
        """
        Return inputs, (ref_masks, test_masks)
        """
        transpose_batch_sequence = self.attrs[_KEY_OF_TRANSPOSE_BS]
        batch, seqlen = data_shape[:2]
        if transpose_batch_sequence:
            data_shape = (data_shape[1], data_shape[0], *data_shape[2:])

        data_rng = jax.random.PRNGKey(0)
        data_rng_0, data_rng_1 = jax.random.split(data_rng, 2)
        inputs = (
            jax.random.normal(data_rng_0, data_shape, dtype),
            jax.random.normal(data_rng_1, data_shape, dtype),
        )

        padded_mask = jnp.zeros((batch, 1, seqlen, seqlen), dtype=jnp.uint8)
        causal_mask = jnp.triu(jnp.ones((batch, 1, seqlen, seqlen), dtype=jnp.uint8), k=1)
        if self.attrs[_KEY_OF_SELF_ATTN_MASK_TYPE] in ["causal", "padding_causal"]:
            self_mask = causal_mask
        else:
            self_mask = padded_mask

        ref_masks = (1 - self_mask, 1 - padded_mask)
        test_masks = (self_mask, padded_mask)

        return inputs, (ref_masks, test_masks)


@pytest.mark.parametrize("data_shape", DATA_SHAPE)
@pytest.mark.parametrize("dtype", DTYPE)
@pytest.mark.parametrize("attrs", ATTRS)
class BaseTester:
    """
    Pytest interface to invoke the runner
    """

    runner = BaseRunner

    def test_forward(self, data_shape, dtype, attrs):
        """Test normal datatype forward"""
        # Ensure FP8 disabled.
        # Empty MeshResource is used as we are running on a single device
        with fp8_autocast(enabled=False, mesh_resource=MeshResource()):
            self.runner(attrs).test_forward(data_shape, dtype)

    def test_backward(self, data_shape, dtype, attrs):
        """Test normal datatype backward"""
        # Ensure FP8 disabled.
        # Empty MeshResource is used as we are running on a single device
        with fp8_autocast(enabled=False, mesh_resource=MeshResource()):
            self.runner(attrs).test_backward(data_shape, dtype)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_recipe", QUANTIZE_RECIPES)
    def test_forward_with_fp8(self, data_shape, dtype, attrs, fp8_recipe):
        """Test forward with fp8 enabled"""
        # Empty MeshResource is used as we are running on a single device
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, mesh_resource=MeshResource()):
            self.runner(attrs).test_forward(data_shape, dtype, rtol=1e-4, atol=1e-3)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("fp8_recipe", QUANTIZE_RECIPES)
    def test_backward_with_fp8(self, data_shape, dtype, attrs, fp8_recipe):
        """Test backward with fp8 enabled"""
        # Empty MeshResource is used as we are running on a single device
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, mesh_resource=MeshResource()):
            self.runner(attrs).test_backward(data_shape, dtype, rtol=1e-4, atol=1e-3)


class TestEncoderLayer(BaseTester):
    """
    Test transformer_engine.jax.flax.TransformerLayer(layer_type=Encoder)
    """

    runner = EncoderRunner


class TestDecoderLayer(BaseTester):
    """
    Test transformer_engine.jax.flax.TransformerLayer(layer_type=Decoder)
    """

    runner = DecoderRunner
