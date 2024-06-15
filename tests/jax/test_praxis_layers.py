# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
from functools import partial
from typing import Dict

import flax
import jax
import jax.numpy as jnp
from praxis import pax_fiddle
from praxis.base_layer import WeightInit, DEFAULT_INIT_MUTABLE_LIST
import pytest

from utils import assert_allclose

from transformer_engine.transformer_engine_jax import get_device_compute_capability
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.jax import fp8_autocast, update_collections
from transformer_engine.jax.flax import DenseGeneral, LayerNormDenseGeneral
from transformer_engine.jax.flax import LayerNorm as flax_LayerNorm
from transformer_engine.jax.flax import LayerNormMLP as flax_LayerNormMLP
from transformer_engine.jax.flax import MultiHeadAttention as flax_MultiHeadAttention
from transformer_engine.jax.flax import DotProductAttention as flax_DotProductAttention
from transformer_engine.jax.flax import RelativePositionBiases as flax_RelativePositionBiases
from transformer_engine.jax.flax import TransformerLayer as flax_TransformerLayer
from transformer_engine.jax.flax.module import Softmax
from transformer_engine.jax.fp8 import FP8Helper, is_fp8_available
from transformer_engine.jax.praxis import LayerNorm
from transformer_engine.jax.praxis import FusedSoftmax
from transformer_engine.jax.praxis import LayerNormLinear, LayerNormMLP, Linear
from transformer_engine.jax.praxis import DotProductAttention, MultiHeadAttention
from transformer_engine.jax.praxis import RelativePositionBiases, TransformerEngineBaseLayer
from transformer_engine.jax.praxis import TransformerLayer, TransformerLayerType
from transformer_engine.jax.softmax import SoftmaxType

is_fp8_supported, reason = is_fp8_available()

DATA_SHAPE = [(32, 128, 512), (32, 512, 512)]  # (B, S, H)
DTYPE = [jnp.float32, jnp.bfloat16]
ENABLE_FP8 = [False, True]
FP8_FORMATS = [Format.E4M3, Format.HYBRID]


@pytest.fixture(autouse=True, scope="module")
def enable_fused_attn():
    """
    Enable fused attn for hopper+ arch.
    Fused attn kernels on pre-hopper arch are not deterministic.
    """
    if get_device_compute_capability(0) >= 90:
        os.environ["NVTE_FUSED_ATTN"] = "1"
    yield
    if "NVTE_FUSED_ATTN" in os.environ:
        del os.environ["NVTE_FUSED_ATTN"]


def compare_dict(ref_fd, test_fd, rtol=1e-05, atol=1e-08):
    for key in ref_fd:
        assert key in test_fd, f"{key} not found in test dict {test_fd}"
        assert isinstance(
            test_fd[key], type(ref_fd[key])
        ), f"The data type is not match between ref and test  Dict on {key=}"
        if isinstance(ref_fd[key], Dict):
            compare_dict(ref_fd[key], test_fd[key], rtol, atol)
        else:
            assert_allclose(
                ref_fd[key], test_fd[key], rtol=rtol, atol=atol, err_msg=f"{key=} is not close"
            )


class TestLayer:

    @staticmethod
    def loss(inner_variables, *inner_inputs, module, mean_out=True):
        outs = module.apply(inner_variables, *inner_inputs)
        out = outs
        if isinstance(outs, tuple):
            # The first place of outs is the real output, others
            # are auxiliary values.
            out = outs[0]
        return jnp.mean(out) if mean_out else out

    @staticmethod
    def loss_and_grads(module, variables, *inputs):
        grad_fn = jax.value_and_grad(TestLayer.loss, argnums=(0, 1))
        loss_val, (wgrads, dgrad) = grad_fn(variables, *inputs, module=module)
        return loss_val, wgrads, dgrad

    def input_getter(self, shape, dtype):
        raise NotImplementedError

    def get_layer_name(self):
        raise NotImplementedError

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        raise NotImplementedError

    def sync_variables(self, praxis_variables, flax_variables):
        synced_praxis_variables = praxis_variables

        lyr_name = self.get_layer_name()

        if "params" in flax_variables:
            synced_praxis_variables["params"][lyr_name]["cld"] = flax.core.unfreeze(
                flax_variables["params"]
            )

        return synced_praxis_variables, flax_variables

    def sync_wgrads(self, praxis_wgrads, flax_wgrads):
        synced_praxis_grads = praxis_wgrads

        lyr_name = self.get_layer_name()

        if "params" in synced_praxis_grads:
            synced_praxis_grads["params"] = synced_praxis_grads["params"][lyr_name]["cld"]

        if FP8Helper.is_fp8_enabled():
            synced_praxis_grads[FP8Helper.FP8_COLLECTION_NAME] = synced_praxis_grads[
                FP8Helper.FP8_COLLECTION_NAME
            ][lyr_name]["cld"]

        return synced_praxis_grads, flax.core.unfreeze(flax_wgrads)

    def forward_backward_runner(
        self, data_shape, dtype, praxis_p, flax_cls, rtol=1e-05, atol=1e-08
    ):
        init_key = jax.random.PRNGKey(seed=1234)

        test_inputs = self.input_getter(data_shape, dtype)

        praxis_layer = praxis_p.Instantiate()
        # This is a workaround to correctly enable FP8 meta generation for Praxis.
        # TODO (Ming Huang): To come out a better solution.
        mutable_list = DEFAULT_INIT_MUTABLE_LIST + [FP8Helper.FP8_COLLECTION_NAME]
        praxis_variables = praxis_layer.init(init_key, *test_inputs, mutable=mutable_list)

        flax_layer = flax_cls()
        flax_variables = flax_layer.init(init_key, *test_inputs)
        if "params_axes" in flax_variables:
            flax_variables, _ = flax.core.pop(flax_variables, "params_axes")
        if FP8Helper.is_fp8_enabled():
            flax_variables, _ = flax.core.pop(
                flax_variables, FP8Helper.FP8_COLLECTION_NAME + "_axes"
            )

        praxis_variables, flax_variables = self.sync_variables(praxis_variables, flax_variables)

        iter_times = 5 if FP8Helper.is_fp8_enabled() else 1

        for _ in range(iter_times):
            praxis_loss, praxis_wgrads, praxis_dgrad = TestLayer.loss_and_grads(
                praxis_layer, praxis_variables, *test_inputs
            )
            flax_loss, flax_wgrads, flax_dgrad = TestLayer.loss_and_grads(
                flax_layer, flax_variables, *test_inputs
            )
            if FP8Helper.is_fp8_enabled():
                praxis_wgrads.pop("params")
                praxis_variables = update_collections(praxis_wgrads, praxis_variables)
                flax_wgrads, _ = flax.core.pop(flax_wgrads, "params")
                flax_variables = update_collections(flax_wgrads, flax_variables)

        praxis_loss, praxis_wgrads, praxis_dgrad = TestLayer.loss_and_grads(
            praxis_layer, praxis_variables, *test_inputs
        )
        flax_loss, flax_wgrads, flax_dgrad = TestLayer.loss_and_grads(
            flax_layer, flax_variables, *test_inputs
        )

        assert_allclose(praxis_loss, flax_loss, rtol=rtol, atol=atol)
        assert_allclose(praxis_dgrad, flax_dgrad, rtol=rtol, atol=atol)

        praxis_wgrads, flax_wgrads = self.sync_wgrads(praxis_wgrads, flax_wgrads)
        compare_dict(praxis_wgrads, flax_wgrads, rtol=rtol, atol=atol)


class LayerNormAttr:
    LN_TYPE = "layernorm_type"
    ZERO_CEN = "zero_centered_gamma"
    ATTRS = [
        {LN_TYPE: "layernorm", ZERO_CEN: False},
        {LN_TYPE: "layernorm", ZERO_CEN: True},
        {LN_TYPE: "rmsnorm", ZERO_CEN: False},
    ]


class TestLayerNorm(TestLayer):

    def input_getter(self, shape, dtype):
        data_key = jax.random.PRNGKey(seed=1234)
        return (jax.random.normal(data_key, shape, dtype),)

    def get_layer_name(self):
        return "layer_norm"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        layernorm_type = attrs[LayerNormAttr.LN_TYPE]
        zero_centered_gamma = attrs[LayerNormAttr.ZERO_CEN]
        scale_init = None
        bias_init = WeightInit.Constant(0.0)
        transpose_batch_sequence = False

        praxis_p = pax_fiddle.Config(
            LayerNorm,
            name="layer_norm",
            dtype=dtype,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            scale_init=scale_init,
            bias_init=bias_init,
            transpose_batch_sequence=transpose_batch_sequence,
        )
        flax_cls = partial(
            flax_LayerNorm,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            scale_init=scale_init,
            bias_init=TransformerEngineBaseLayer.generate_params_init("ln_bias", bias_init),
            dtype=dtype,
            transpose_batch_sequence=transpose_batch_sequence,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", LayerNormAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
        self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)


class FusedSoftmaxAttr:
    SCALE_FACTOR = "scale_factor"
    ST_TYPE = "softmax_type"
    ATTRS = [
        {SCALE_FACTOR: 0.0, ST_TYPE: SoftmaxType.SCALED},
        {SCALE_FACTOR: 0.0, ST_TYPE: SoftmaxType.SCALED_MASKED},
        {SCALE_FACTOR: 0.0, ST_TYPE: SoftmaxType.SCALED_UPPER_TRIANG_MASKED},
    ]


class TestFusedSoftmax(TestLayer):

    def input_getter(self, shape, dtype):
        data_key = jax.random.PRNGKey(seed=1234)
        return jax.random.normal(data_key, shape, dtype), jnp.ones(shape, dtype=jnp.uint8)  # Masks

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        scale_factor = attrs[FusedSoftmaxAttr.SCALE_FACTOR]
        softmax_type = attrs[FusedSoftmaxAttr.ST_TYPE]

        praxis_p = pax_fiddle.Config(
            FusedSoftmax, name="fused_softmax", scale_factor=scale_factor, softmax_type=softmax_type
        )
        flax_cls = partial(Softmax, scale_factor=scale_factor, softmax_type=softmax_type)

        return praxis_p, flax_cls

    def sync_variables(self, praxis_variables, flax_variables):
        return praxis_variables, flax_variables

    def sync_wgrads(self, praxis_wgrads, flax_wgrads):
        return praxis_wgrads, flax_wgrads

    @pytest.mark.parametrize("data_shape", [(32, 1, 128, 128), (32, 1, 512, 128)])
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", FusedSoftmaxAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        if (attrs[FusedSoftmaxAttr.ST_TYPE] == SoftmaxType.SCALED_UPPER_TRIANG_MASKED) and (
            data_shape[-2] != data_shape[-1]
        ):
            pass  # Skip, due to not support
        else:
            praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
            self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)


class LinearAttr:
    FEATURE = "features"
    USE_BIAS = "use_bias"
    ATTRS = [
        {FEATURE: 512, USE_BIAS: False},
        {FEATURE: 512, USE_BIAS: True},
        {FEATURE: 1024, USE_BIAS: False},
        {FEATURE: 1024, USE_BIAS: True},
    ]


class TestLinear(TestLayer):

    def input_getter(self, shape, dtype):
        data_key = jax.random.PRNGKey(seed=1234)
        return (jax.random.normal(data_key, shape, dtype),)

    def get_layer_name(self):
        return "linear"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        out_features = attrs[LinearAttr.FEATURE]
        kernel_init = WeightInit.Gaussian(1.0)
        use_bias = attrs[LinearAttr.USE_BIAS]
        bias_init = WeightInit.Constant(0.0)
        axis = -1
        transpose_batch_sequence = False

        praxis_p = pax_fiddle.Config(
            Linear,
            name="linear",
            dtype=dtype,
            out_features=out_features,
            params_init=kernel_init,
            use_bias=use_bias,
            bias_init=bias_init,
            axis=axis,
            transpose_batch_sequence=transpose_batch_sequence,
        )
        flax_cls = partial(
            DenseGeneral,
            features=out_features,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", kernel_init),
            use_bias=use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", bias_init),
            axis=axis,
            dtype=dtype,
            transpose_batch_sequence=transpose_batch_sequence,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", LinearAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
        self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", LinearAttr.ATTRS)
    @pytest.mark.parametrize("fp8_format", FP8_FORMATS)
    def test_forward_backward_fp8(
        self, data_shape, dtype, attrs, fp8_format, rtol=1e-05, atol=1e-08
    ):

        ds = DelayedScaling(fp8_format=fp8_format)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
            self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)


class LayerNormLinearAttr:
    FEATURE = "features"
    USE_BIAS = "use_bias"
    ENABLE_LN = "enable_layernorm"
    LN_TYPE = "layernorm_type"
    ZERO_CEN = "zero_centered_gamma"
    ATTRS = [
        {FEATURE: 512, USE_BIAS: True, ENABLE_LN: True, LN_TYPE: "layernorm", ZERO_CEN: False},
        {FEATURE: 512, USE_BIAS: True, ENABLE_LN: True, LN_TYPE: "layernorm", ZERO_CEN: False},
        {FEATURE: 512, USE_BIAS: True, ENABLE_LN: True, LN_TYPE: "layernorm", ZERO_CEN: True},
        {FEATURE: 512, USE_BIAS: True, ENABLE_LN: True, LN_TYPE: "layernorm", ZERO_CEN: True},
        {FEATURE: 512, USE_BIAS: True, ENABLE_LN: True, LN_TYPE: "rmsnorm", ZERO_CEN: False},
        {FEATURE: 512, USE_BIAS: True, ENABLE_LN: True, LN_TYPE: "rmsnorm", ZERO_CEN: False},
        {FEATURE: 512, USE_BIAS: True, ENABLE_LN: False, LN_TYPE: "layernorm", ZERO_CEN: False},
    ]


class TestLayerNormLinear(TestLayer):

    def input_getter(self, shape, dtype):
        data_key = jax.random.PRNGKey(seed=1234)
        return (jax.random.normal(data_key, shape, dtype),)

    def get_layer_name(self):
        return "ln_linear"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        out_features = attrs[LayerNormLinearAttr.FEATURE]
        enable_layernorm = attrs[LayerNormLinearAttr.ENABLE_LN]
        layernorm_type = attrs[LayerNormLinearAttr.LN_TYPE]
        zero_centered_gamma = attrs[LayerNormLinearAttr.ZERO_CEN]
        kernel_init = WeightInit.Gaussian(1.0)
        use_bias = attrs[LayerNormLinearAttr.USE_BIAS]
        bias_init = WeightInit.Constant(0.0)
        axis = -1
        transpose_batch_sequence = False

        praxis_p = pax_fiddle.Config(
            LayerNormLinear,
            name="ln_linear",
            dtype=dtype,
            out_features=out_features,
            enable_layernorm=enable_layernorm,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            params_init=kernel_init,
            use_bias=use_bias,
            bias_init=bias_init,
            axis=axis,
            transpose_batch_sequence=transpose_batch_sequence,
        )
        flax_cls = partial(
            LayerNormDenseGeneral,
            features=out_features,
            enable_layernorm=enable_layernorm,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", kernel_init),
            use_bias=use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", bias_init),
            axis=axis,
            dtype=dtype,
            transpose_batch_sequence=transpose_batch_sequence,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", LayerNormLinearAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
        self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", LayerNormLinearAttr.ATTRS)
    @pytest.mark.parametrize("fp8_format", FP8_FORMATS)
    def test_forward_backward_fp8(
        self, data_shape, dtype, attrs, fp8_format, rtol=1e-05, atol=1e-08
    ):

        ds = DelayedScaling(fp8_format=fp8_format)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
            self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)


class LayerNormMLPAttr:
    INTERMEDIATE_DIM = "intermediate_dim"
    USE_BIAS = "use_bias"
    ENABLE_LN = "enable_layernorm"
    LN_TYPE = "layernorm_type"
    ZERO_CEN = "zero_centered_gamma"
    ACTIVATION = "activations"
    ATTRS = [
        {
            INTERMEDIATE_DIM: 2048,
            USE_BIAS: True,
            ENABLE_LN: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
        },
        {
            INTERMEDIATE_DIM: 2048,
            USE_BIAS: True,
            ENABLE_LN: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("relu",),
        },
        {
            INTERMEDIATE_DIM: 2048,
            USE_BIAS: True,
            ENABLE_LN: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
        },
        {
            INTERMEDIATE_DIM: 2048,
            USE_BIAS: True,
            ENABLE_LN: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
        },
        {
            INTERMEDIATE_DIM: 2048,
            USE_BIAS: False,
            ENABLE_LN: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
        },
        {
            INTERMEDIATE_DIM: 2048,
            USE_BIAS: True,
            ENABLE_LN: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("silu", "linear"),
        },
        {
            INTERMEDIATE_DIM: 2048,
            USE_BIAS: False,
            ENABLE_LN: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("silu", "linear"),
        },
    ]


class TestLayerNormMLP(TestLayer):

    def input_getter(self, shape, dtype):
        data_key = jax.random.PRNGKey(seed=1234)
        return (jax.random.normal(data_key, shape, dtype),)

    def get_layer_name(self):
        return "ln_mlp"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        intermediate_dim = attrs[LayerNormMLPAttr.INTERMEDIATE_DIM]
        enable_layernorm = attrs[LayerNormMLPAttr.ENABLE_LN]
        layernorm_type = attrs[LayerNormMLPAttr.LN_TYPE]
        zero_centered_gamma = attrs[LayerNormMLPAttr.ZERO_CEN]
        kernel_init = WeightInit.Gaussian(1.0)
        use_bias = attrs[LayerNormMLPAttr.USE_BIAS]
        bias_init = WeightInit.Constant(0.0)
        activations = attrs[LayerNormMLPAttr.ACTIVATION]
        axis = -1
        transpose_batch_sequence = False

        praxis_p = pax_fiddle.Config(
            LayerNormMLP,
            name="ln_mlp",
            dtype=dtype,
            intermediate_dim=intermediate_dim,
            enable_layernorm=enable_layernorm,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            params_init=kernel_init,
            use_bias=use_bias,
            bias_init=bias_init,
            activations=activations,
            intermediate_dropout_rate=0.0,
            axis=axis,
            transpose_batch_sequence=transpose_batch_sequence,
        )
        flax_cls = partial(
            flax_LayerNormMLP,
            intermediate_dim=intermediate_dim,
            enable_layernorm=enable_layernorm,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", kernel_init),
            use_bias=use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", bias_init),
            activations=activations,
            intermediate_dropout_rate=0.0,
            axis=axis,
            dtype=dtype,
            transpose_batch_sequence=transpose_batch_sequence,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", LayerNormMLPAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
        self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", LayerNormMLPAttr.ATTRS)
    @pytest.mark.parametrize("fp8_format", FP8_FORMATS)
    def test_forward_backward_fp8(
        self, data_shape, dtype, attrs, fp8_format, rtol=1e-05, atol=1e-08
    ):

        ds = DelayedScaling(fp8_format=fp8_format)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
            self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)


class TestRelativePositionBias(TestLayer):

    def get_layer_name(self):
        return "relative_position_bias"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        num_buckets = 32
        max_distance = 128
        num_attention_heads = 64
        rb_stddev = (num_attention_heads * num_buckets) ** -0.5
        embedding_init = WeightInit.Gaussian(rb_stddev)

        praxis_p = pax_fiddle.Config(
            RelativePositionBiases,
            name="relative_position_bias",
            dtype=dtype,
            num_buckets=num_buckets,
            max_distance=max_distance,
            num_attention_heads=num_attention_heads,
            embedding_init=embedding_init,
        )
        flax_cls = partial(
            flax_RelativePositionBiases,
            num_buckets=num_buckets,
            max_distance=max_distance,
            num_attention_heads=num_attention_heads,
            embedding_init=TransformerEngineBaseLayer.generate_params_init(
                "rel_embedding", embedding_init
            ),
            dtype=dtype,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", [{}])
    def test_forward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)

        init_key = jax.random.PRNGKey(seed=1234)

        test_inputs = [(128, 128, True), (128, 128, False)]
        for test_input in test_inputs:
            praxis_layer = praxis_p.Instantiate()
            praxis_variables = praxis_layer.init(init_key, *test_input)

            flax_layer = flax_cls()
            flax_variables = flax_layer.init(init_key, *test_input)
            if "params_axes" in flax_variables:
                flax_variables, _ = flax.core.pop(flax_variables, "params_axes")
            if FP8Helper.is_fp8_enabled():
                flax_variables, _ = flax.core.pop(
                    flax_variables, FP8Helper.FP8_COLLECTION_NAME + "_axes"
                )

            praxis_variables, flax_variables = self.sync_variables(praxis_variables, flax_variables)

            praxis_loss = TestLayer.loss(
                praxis_variables, *test_input, module=praxis_layer, mean_out=False
            )
            flax_loss = TestLayer.loss(
                flax_variables, *test_input, module=flax_layer, mean_out=False
            )

            assert_allclose(praxis_loss, flax_loss, rtol=rtol, atol=atol)


class DotProductAttnAttr:
    ATTN_MASK_TYPE = "attn_mask_type"
    NUM_GQA_GROUPS = "num_gqa_groups"
    TRANSPOSE_BS = "transpose_batch_sequence"
    SCALE_FACTOR = "scale_factor"
    ATTRS = [
        {
            ATTN_MASK_TYPE: "padding",
            TRANSPOSE_BS: True,
            SCALE_FACTOR: 0.125,
        },
        {
            ATTN_MASK_TYPE: "padding_causal",
            TRANSPOSE_BS: True,
            SCALE_FACTOR: 0.125,
        },
        {
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: True,
            SCALE_FACTOR: 0.125,
        },
        {
            ATTN_MASK_TYPE: "padding",
            TRANSPOSE_BS: False,
            SCALE_FACTOR: 0.125,
        },
        {
            ATTN_MASK_TYPE: "padding_causal",
            TRANSPOSE_BS: False,
            SCALE_FACTOR: 2.0,
        },
        {
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: False,
            SCALE_FACTOR: 1.0,
        },
        {
            ATTN_MASK_TYPE: "no_mask",
            TRANSPOSE_BS: False,
            SCALE_FACTOR: 1.0,
        },
    ]


class TestDotProductAttn(TestLayer):

    def input_getter(self, shape, dtype):
        key = jax.random.PRNGKey(seed=1234)
        q_key, k_key, v_key = jax.random.split(key, 3)
        b, s, *_ = shape
        if self.attrs[DotProductAttnAttr.TRANSPOSE_BS]:
            shape = (shape[1], shape[0]) + shape[2:]
        mask = jnp.zeros((b, 1, s, s), dtype=jnp.uint8)
        return [
            *map(partial(jax.random.normal, shape=shape, dtype=dtype), [q_key, k_key, v_key]),
            mask,
        ]

    def get_layer_name(self):
        return "dot_product_attn"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        head_dim = 64
        num_attention_heads = 16
        num_gqa_groups = num_attention_heads
        attn_mask_type = attrs[DotProductAttnAttr.ATTN_MASK_TYPE]
        transpose_batch_sequence = attrs[DotProductAttnAttr.TRANSPOSE_BS]

        praxis_p = pax_fiddle.Config(
            DotProductAttention,
            name="mha",
            dtype=dtype,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_gqa_groups=num_gqa_groups,
            attn_mask_type=attn_mask_type,
            transpose_batch_sequence=transpose_batch_sequence,
        )
        flax_cls = partial(
            flax_DotProductAttention,
            dtype=dtype,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_gqa_groups=num_gqa_groups,
            attn_mask_type=attn_mask_type,
            transpose_batch_sequence=transpose_batch_sequence,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", [(32, 128, 16, 64)])
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", DotProductAttnAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        self.attrs = attrs
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
        self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)


class MultiHeadAttnAttr:
    USE_BIAS = "use_bias"
    LN_TYPE = "layernorm_type"
    ATTN_MASK_TYPE = "attn_mask_type"
    ZERO_CEN = "zero_centered_gamma"
    NUM_ATTN_HEADS = "num_attention_heads"
    NUM_GQA_GROUPS = "num_gqa_groups"
    TRANSPOSE_BS = "transpose_batch_sequence"
    ENABLE_ROPE = "enable_rotary_pos_emb"
    ROPE_GROUP_METHOD = "rotary_pos_emb_group_method"
    LORA_SCOPE = "low_rank_adaptation_scope"
    ATTRS = [
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "padding",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "padding",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "padding",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            NUM_ATTN_HEADS: 8,
            NUM_GQA_GROUPS: 4,
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ENABLE_ROPE: True,
            ROPE_GROUP_METHOD: "consecutive",
            NUM_ATTN_HEADS: 8,
            NUM_GQA_GROUPS: 4,
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ENABLE_ROPE: True,
            ROPE_GROUP_METHOD: "alternate",
            NUM_ATTN_HEADS: 8,
            NUM_GQA_GROUPS: 4,
            ATTN_MASK_TYPE: "causal",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "padding",
            LORA_SCOPE: "all",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            ATTN_MASK_TYPE: "causal",
            LORA_SCOPE: "all",
            TRANSPOSE_BS: True,
        },
    ]


class TestMultiHeadAttn(TestLayer):

    def input_getter(self, shape, dtype):
        key = jax.random.PRNGKey(seed=1234)
        q_key, kv_key = jax.random.split(key, 2)
        b, s, *_ = shape
        if self.attrs[MultiHeadAttnAttr.TRANSPOSE_BS]:
            shape = (shape[1], shape[0]) + shape[2:]
        mask = jnp.zeros((b, 1, s, s), dtype=jnp.uint8)
        return [*map(partial(jax.random.normal, shape=shape, dtype=dtype), [q_key, kv_key]), mask]

    def get_layer_name(self):
        return "multi_head_attn"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        head_dim = 64
        num_attention_heads = 16
        num_gqa_groups = (
            attrs[MultiHeadAttnAttr.NUM_GQA_GROUPS]
            if MultiHeadAttnAttr.NUM_GQA_GROUPS in attrs
            else None
        )
        layernorm_type = attrs[MultiHeadAttnAttr.LN_TYPE]
        zero_centered_gamma = attrs[MultiHeadAttnAttr.ZERO_CEN]
        kernel_init = WeightInit.Gaussian(1.0)
        use_bias = attrs[MultiHeadAttnAttr.USE_BIAS]
        bias_init = WeightInit.Constant(0.0)
        input_layernorm = False
        return_layernorm_output = False
        attn_mask_type = attrs[MultiHeadAttnAttr.ATTN_MASK_TYPE]
        enable_rotary_pos_emb = attrs[MultiHeadAttnAttr.ENABLE_ROPE]
        rotary_pos_emb_group_method = attrs[MultiHeadAttnAttr.ROPE_GROUP_METHOD]
        low_rank_adaptation_scope = attrs.get(MultiHeadAttnAttr.LORA_SCOPE, "none")
        fuse_qkv_params = True
        transpose_batch_sequence = attrs[MultiHeadAttnAttr.TRANSPOSE_BS]
        scale_attn_logits = False
        scaled_query_init = True
        float32_logits = False

        praxis_p = pax_fiddle.Config(
            MultiHeadAttention,
            name="mha",
            dtype=dtype,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_gqa_groups=num_gqa_groups,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            params_init=kernel_init,
            use_bias=use_bias,
            bias_init=bias_init,
            return_layernorm_output=return_layernorm_output,
            input_layernorm=input_layernorm,
            attn_mask_type=attn_mask_type,
            enable_rotary_pos_emb=enable_rotary_pos_emb,
            rotary_pos_emb_group_method=rotary_pos_emb_group_method,
            low_rank_adaptation_scope=low_rank_adaptation_scope,
            fuse_qkv_params=fuse_qkv_params,
            transpose_batch_sequence=transpose_batch_sequence,
            scale_attn_logits=scale_attn_logits,
            scaled_query_init=scaled_query_init,
            float32_logits=float32_logits,
        )
        flax_cls = partial(
            flax_MultiHeadAttention,
            dtype=dtype,
            head_dim=head_dim,
            num_attention_heads=num_attention_heads,
            num_gqa_groups=num_gqa_groups,
            layernorm_type=layernorm_type,
            zero_centered_gamma=zero_centered_gamma,
            kernel_init=TransformerEngineBaseLayer.generate_params_init("kernel", kernel_init),
            use_bias=use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", bias_init),
            return_layernorm_output=return_layernorm_output,
            input_layernorm=input_layernorm,
            attn_mask_type=attn_mask_type,
            enable_rotary_pos_emb=enable_rotary_pos_emb,
            rotary_pos_emb_group_method=rotary_pos_emb_group_method,
            low_rank_adaptation_scope=low_rank_adaptation_scope,
            fuse_qkv_params=fuse_qkv_params,
            transpose_batch_sequence=transpose_batch_sequence,
            scale_attn_logits=scale_attn_logits,
            scaled_query_init=scaled_query_init,
            float32_logits=float32_logits,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", MultiHeadAttnAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        self.attrs = attrs
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
        self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", MultiHeadAttnAttr.ATTRS)
    @pytest.mark.parametrize("fp8_format", FP8_FORMATS)
    def test_forward_backward_fp8(
        self, data_shape, dtype, attrs, fp8_format, rtol=1e-05, atol=1e-08
    ):
        self.attrs = attrs
        ds = DelayedScaling(fp8_format=fp8_format)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
            self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)


class TransformerLayerAttr:
    USE_BIAS = "use_bias"
    LN_TYPE = "layernorm_type"
    ACTIVATION = "activations"
    LYR_TYPE = "layer_type"
    ZERO_CEN = "zero_centered_gamma"
    TRANSPOSE_BS = "transpose_batch_sequence"
    ENABLE_ROPE = "enable_rotary_pos_emb"
    ROPE_GROUP_METHOD = "rotary_pos_emb_group_method"
    LORA_SCOPE = "low_rank_adaptation_scope"
    ATTRS = [
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("relu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
            LORA_SCOPE: "all",
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: True,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "rmsnorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu", "linear"),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("gelu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: True,
            ROPE_GROUP_METHOD: "alternate",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("gelu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: True,
            ROPE_GROUP_METHOD: "alternate",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("gelu",),
            LYR_TYPE: TransformerLayerType.ENCODER,
            ENABLE_ROPE: True,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: True,
            ACTIVATION: ("gelu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: True,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
        },
        {
            USE_BIAS: True,
            LN_TYPE: "layernorm",
            ZERO_CEN: False,
            ACTIVATION: ("gelu",),
            LYR_TYPE: TransformerLayerType.DECODER,
            ENABLE_ROPE: False,
            ROPE_GROUP_METHOD: "consecutive",
            TRANSPOSE_BS: False,
            LORA_SCOPE: "all",
        },
    ]


class TestTransformer(TestLayer):

    def input_getter(self, shape, dtype):
        key = jax.random.PRNGKey(seed=1234)
        q_key, kv_key = jax.random.split(key, 2)
        b, s, *_ = shape
        if self.attrs[TransformerLayerAttr.TRANSPOSE_BS]:
            shape = (shape[1], shape[0]) + shape[2:]
        mask = jnp.zeros((b, 1, s, s), dtype=jnp.uint8)
        return [
            *map(partial(jax.random.normal, shape=shape, dtype=dtype), [q_key, kv_key]),
            mask,
            mask,
        ]

    def get_layer_name(self):
        return "transformerlayer"

    def generate_praxis_p_and_flax_cls(self, dtype, attrs):
        hidden_size = 512
        mlp_hidden_size = 2048
        num_attention_heads = 8
        layernorm_type = attrs[TransformerLayerAttr.LN_TYPE]
        hidden_dropout = 0.0
        attention_dropout = 0.0
        intermediate_dropout = 0.0
        mlp_activations = attrs[TransformerLayerAttr.ACTIVATION]
        kernel_init = WeightInit.Gaussian(1.0)
        use_bias = attrs[TransformerLayerAttr.USE_BIAS]
        bias_init = WeightInit.Constant(0.0)
        layer_type = attrs[TransformerLayerAttr.LYR_TYPE]
        enable_rotary_pos_emb = attrs[TransformerLayerAttr.ENABLE_ROPE]
        rotary_pos_emb_group_method = attrs[TransformerLayerAttr.ROPE_GROUP_METHOD]
        low_rank_adaptation_scope = attrs.get(TransformerLayerAttr.LORA_SCOPE, "none")
        enable_relative_embedding = True
        relative_embedding = pax_fiddle.Config(
            RelativePositionBiases, dtype=dtype, num_attention_heads=num_attention_heads
        )
        drop_path = 0.0
        transpose_batch_sequence = attrs[TransformerLayerAttr.TRANSPOSE_BS]

        rel_embedding_init = RelativePositionBiases.generate_embedding_init(
            relative_embedding.embedding_init,
            relative_embedding.num_attention_heads,
            relative_embedding.num_buckets,
        )

        relative_embedding_flax_module = flax_RelativePositionBiases(
            num_buckets=relative_embedding.num_buckets,
            max_distance=relative_embedding.max_distance,
            num_attention_heads=relative_embedding.num_attention_heads,
            embedding_init=TransformerEngineBaseLayer.generate_params_init(
                "rel_embedding", rel_embedding_init
            ),
            embedding_axes=relative_embedding.embedding_axes,
            dtype=relative_embedding.dtype,
        )

        praxis_p = pax_fiddle.Config(
            TransformerLayer,
            name="transformer_layer",
            params_init=kernel_init,
            dtype=dtype,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            num_attention_heads=num_attention_heads,
            layernorm_type=layernorm_type,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            intermediate_dropout=intermediate_dropout,
            mlp_activations=mlp_activations,
            use_bias=use_bias,
            bias_init=bias_init,
            layer_type=layer_type,
            enable_relative_embedding=enable_relative_embedding,
            enable_rotary_pos_emb=enable_rotary_pos_emb,
            rotary_pos_emb_group_method=rotary_pos_emb_group_method,
            low_rank_adaptation_scope=low_rank_adaptation_scope,
            relative_embedding=relative_embedding,
            drop_path=drop_path,
            transpose_batch_sequence=transpose_batch_sequence,
        )
        flax_cls = partial(
            flax_TransformerLayer,
            dtype=dtype,
            hidden_size=hidden_size,
            mlp_hidden_size=mlp_hidden_size,
            num_attention_heads=num_attention_heads,
            layernorm_type=layernorm_type,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            intermediate_dropout=intermediate_dropout,
            mlp_activations=mlp_activations,
            mha_kernel_init=TransformerEngineBaseLayer.generate_params_init(
                "mha_kernel", kernel_init
            ),
            mlp_kernel_init=TransformerEngineBaseLayer.generate_params_init(
                "mlp_kernel", kernel_init
            ),
            use_bias=use_bias,
            bias_init=TransformerEngineBaseLayer.generate_params_init("bias", bias_init),
            layer_type=layer_type,
            enable_rotary_pos_emb=enable_rotary_pos_emb,
            rotary_pos_emb_group_method=rotary_pos_emb_group_method,
            enable_relative_embedding=enable_relative_embedding,
            relative_embedding=relative_embedding_flax_module,
            low_rank_adaptation_scope=low_rank_adaptation_scope,
            drop_path=drop_path,
            transpose_batch_sequence=transpose_batch_sequence,
        )

        return praxis_p, flax_cls

    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", TransformerLayerAttr.ATTRS)
    def test_forward_backward(self, data_shape, dtype, attrs, rtol=1e-05, atol=1e-08):
        self.attrs = attrs
        praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
        self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest.mark.parametrize("data_shape", DATA_SHAPE)
    @pytest.mark.parametrize("dtype", DTYPE)
    @pytest.mark.parametrize("attrs", TransformerLayerAttr.ATTRS)
    @pytest.mark.parametrize("fp8_format", FP8_FORMATS)
    def test_forward_backward_fp8(
        self, data_shape, dtype, attrs, fp8_format, rtol=1e-05, atol=1e-08
    ):
        self.attrs = attrs
        ds = DelayedScaling(fp8_format=fp8_format)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            praxis_p, flax_cls = self.generate_praxis_p_and_flax_cls(dtype, attrs)
            self.forward_backward_runner(data_shape, dtype, praxis_p, flax_cls, rtol, atol)
