# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
from typing import Callable, Sequence, Union, Optional
import pytest

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from utils import (
    assert_allclose,
    assert_tree_like_allclose,
    is_devices_enough,
    pytest_parametrize_wrapper,
)

from transformer_engine.common import recipe
from transformer_engine.jax.quantize import is_fp8_available, ScalingMode
from transformer_engine.jax import fp8_autocast
from transformer_engine.jax.flax import LayerNormMLP
from transformer_engine.jax.layernorm_mlp import layernorm_mlp
from transformer_engine.jax.sharding import (
    HIDDEN_AXES,
    HIDDEN_TP_AXES,
    BATCH_AXES,
    SEQLEN_TP_AXES,
    SEQLEN_AXES,
    W_NO_SHARD_AXES,
    W_FSDP_AXES,
    W_TP_AXES,
    W_JOINED_AXES,
)
from transformer_engine.jax.sharding import MeshResource
from transformer_engine.jax.quantize import QuantizerFactory


is_fp8_supported, reason = is_fp8_available()
is_mxfp8_supported, reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

SUPPORTED_RECIPES = []
if is_fp8_supported:
    SUPPORTED_RECIPES.append(pytest.param(recipe.DelayedScaling(), id="DelayedScaling"))
    SUPPORTED_RECIPES.append(pytest.param(recipe.Float8CurrentScaling(), id="CurrentScaling"))
if is_mxfp8_supported:
    SUPPORTED_RECIPES.append(pytest.param(recipe.MXFP8BlockScaling(), id="MXFP8BlockScaling"))

DTYPES = [jnp.bfloat16, jnp.float16]
INPUT_SHAPE = [[4, 64, 128]]  # [batch, seqlen, hidden_in]

LAYERNORM_INPUT_AXES = (BATCH_AXES, SEQLEN_TP_AXES, HIDDEN_AXES)
DOT_1_INPUT_AXES = (BATCH_AXES, SEQLEN_AXES, HIDDEN_AXES)
DOT_2_INPUT_AXES = (BATCH_AXES, SEQLEN_AXES, HIDDEN_TP_AXES)
KERNEL_1_AXES = (W_FSDP_AXES, W_JOINED_AXES, W_TP_AXES)
KERNEL_2_AXES = (W_TP_AXES, W_FSDP_AXES)
LN_SCALE_AXES = (W_NO_SHARD_AXES,)
LN_BIAS_AXES = (W_NO_SHARD_AXES,)
BIAS_1_AXES = (W_JOINED_AXES, W_TP_AXES)
BIAS_2_AXES = (W_NO_SHARD_AXES,)
INTERMEDIATE = 64


# Only test with FSDP and TP as DP is not used
def generate_fsdp_and_tp_configs():
    configs = []
    if is_devices_enough(2):
        configs.append(
            [2, (1, 2), ("fsdp", "tp"), MeshResource(fsdp_resource="fsdp", tp_resource="tp")]
        )
    if is_devices_enough(4):
        configs.append(
            [4, (2, 2), ("fsdp", "tp"), MeshResource(fsdp_resource="fsdp", tp_resource="tp")]
        )
    return configs


class TestDistributedLayernormMLP:

    def generate_inputs(self, input_shape, activation_type, use_bias, dtype):
        batch, seqlen, hidden_in = input_shape
        hidden_out = hidden_in

        key = jax.random.PRNGKey(0)
        subkeys = jax.random.split(key, 6)

        x = jax.random.normal(subkeys[0], (batch, seqlen, hidden_in), dtype)
        gamma = jax.random.normal(subkeys[5], (hidden_in,), dtype=dtype)
        k1 = jax.random.normal(
            subkeys[1], (hidden_in, len(activation_type), INTERMEDIATE), dtype
        ) / jnp.sqrt(hidden_in)
        k2 = jax.random.normal(subkeys[2], (INTERMEDIATE, hidden_out), dtype) / jnp.sqrt(
            INTERMEDIATE
        )
        if use_bias:
            b1 = jax.random.normal(subkeys[3], (len(activation_type), INTERMEDIATE), dtype)
            b2 = jax.random.normal(subkeys[4], (hidden_out,), dtype)
        else:
            b1 = None
            b2 = None

        return (x, gamma, k1, k2, b1, b2)

    def layernorm_fp8_mlp_prim_func(
        self,
        x: jnp.ndarray,
        ln_scale: jnp.ndarray,
        kernel_1: jnp.ndarray,
        kernel_2: jnp.ndarray,
        bias_1: Optional[jnp.ndarray],
        bias_2: Optional[jnp.ndarray],
        layernorm_type: str = "rmsnorm",
        activation_type: Sequence[Union[str, Callable]] = ("gelu",),
        multi_gpus: bool = False,
    ) -> jnp.ndarray:

        if multi_gpus:
            layernorm_input_axes = LAYERNORM_INPUT_AXES
            dot_1_input_axes = DOT_1_INPUT_AXES
            dot_2_input_axes = DOT_2_INPUT_AXES
            kernel_1_axes = KERNEL_1_AXES
            kernel_2_axes = KERNEL_2_AXES
        else:
            layernorm_input_axes = None
            dot_1_input_axes = dot_2_input_axes = None
            kernel_1_axes = kernel_2_axes = None

        quantizer_sets = QuantizerFactory.create_set(n_quantizer_sets=2)

        # out = ((x * kernel_1) + bias_1) * kernel_2 + bias_2
        return jnp.mean(
            layernorm_mlp(
                x,
                ln_scale,
                None,
                [kernel_1, kernel_2],
                [bias_1, bias_2],
                layernorm_type,
                norm_input_axes=layernorm_input_axes,
                dot_1_input_axes=dot_1_input_axes,
                dot_2_input_axes=dot_2_input_axes,
                kernel_1_axes=kernel_1_axes,
                kernel_2_axes=kernel_2_axes,
                activation_type=activation_type,
                quantizer_sets=quantizer_sets,
            )
        )

    def _test_layernorm_mlp_grad(
        self, mesh_config, activation_type, use_bias, input_shape, dtype, fp8_recipe, use_shardy
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        device_count, mesh_shape, mesh_axes, mesh_resource = mesh_config
        layernorm_type = "rmsnorm"

        inputs = [x, gamma, k1, k2, b1, b2] = self.generate_inputs(
            input_shape, activation_type, use_bias, dtype
        )
        static_inputs = [layernorm_type, activation_type]
        value_and_grad_func = jax.value_and_grad(
            self.layernorm_fp8_mlp_prim_func, argnums=range(len(inputs))
        )

        # Single GPU
        with fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
            single_jitter = jax.jit(
                value_and_grad_func,
                static_argnums=range(len(inputs), len(static_inputs) + len(inputs)),
            )
            single_fwd, single_grads = single_jitter(*inputs, *static_inputs)

        # Multi GPUs
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(enabled=True, fp8_recipe=fp8_recipe, mesh_resource=mesh_resource):
            k1_sharding = NamedSharding(mesh, PartitionSpec("fsdp", None, "tp"))
            k2_sharding = NamedSharding(mesh, PartitionSpec("tp", "fsdp"))
            k1_ = jax.device_put(k1, k1_sharding)
            k2_ = jax.device_put(k2, k2_sharding)
            if use_bias:
                b1_sharding = NamedSharding(mesh, PartitionSpec(None, "tp"))
                b1_ = jax.device_put(b1, b1_sharding)
            else:
                b1_sharding = b1_ = None
            multi_inputs = [*inputs[:2], k1_, k2_, b1_, *inputs[5:]]

            # Position ref for sharding pspec lists
            #   x, gamma, k1, k2, b1,
            #   b2
            in_shardings = (
                None,
                None,
                k1_sharding,
                k2_sharding,
                b1_sharding,
                None,
            )
            out_shardings = (
                None,
                (None, None, k1_sharding, k2_sharding, b1_sharding, None),
            )

            multi_jitter = jax.jit(
                value_and_grad_func,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                static_argnums=range(len(multi_inputs), len(static_inputs) + len(multi_inputs) + 1),
            )  # +1 for multi_gpus

            multi_fwd, multi_grads = multi_jitter(*multi_inputs, *static_inputs, True)

        assert_allclose(multi_fwd, single_fwd, dtype=dtype)
        for i in range(len(inputs)):
            if multi_grads[i] is not None:
                if isinstance(multi_grads[i], list):
                    assert isinstance(single_grads[i], list)
                    for m_grad, s_grad in zip(multi_grads[i], single_grads[i]):
                        assert_allclose(
                            m_grad, s_grad, dtype=dtype, err_msg=f"multi_grads[{i}] is not close"
                        )
                else:
                    assert_allclose(
                        multi_grads[i],
                        single_grads[i],
                        dtype=dtype,
                        err_msg=f"multi_grads[{i}] is not close",
                    )

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("mesh_config", generate_fsdp_and_tp_configs())
    @pytest_parametrize_wrapper("input_shape", INPUT_SHAPE)
    @pytest_parametrize_wrapper("activation_type", [("gelu",), ("gelu", "linear")])
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("use_bias", [True, False])
    @pytest_parametrize_wrapper("fp8_recipe", SUPPORTED_RECIPES)
    def test_layernorm_mlp_grad(
        self, mesh_config, activation_type, use_bias, input_shape, dtype, fp8_recipe
    ):
        self._test_layernorm_mlp_grad(
            mesh_config,
            activation_type,
            use_bias,
            input_shape,
            dtype,
            fp8_recipe,
            use_shardy=False,
        )

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("mesh_config", generate_fsdp_and_tp_configs())
    @pytest_parametrize_wrapper("input_shape", INPUT_SHAPE)
    @pytest_parametrize_wrapper("activation_type", [("gelu",), ("gelu", "linear")])
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("use_bias", [True, False])
    def test_layernorm_mlp_grad_shardy(
        self, mesh_config, activation_type, use_bias, input_shape, dtype
    ):
        # We don't test block scaling with Shardy because at the time of writing,
        # it is not supported in JAX's scaled_matmul_stablehlo.
        self._test_layernorm_mlp_grad(
            mesh_config,
            activation_type,
            use_bias,
            input_shape,
            dtype,
            fp8_recipe=recipe.DelayedScaling(),
            use_shardy=True,
        )

    def _test_layernorm_mlp(
        self,
        mesh_config,
        activation_type,
        use_bias,
        input_shape,
        dtype,
        use_fp8,
        fp8_recipe,
        use_shardy,
    ):
        jax.config.update("jax_use_shardy_partitioner", use_shardy)
        batch, seqlen, hidden_in = input_shape
        layernorm_type = "rmsnorm"

        rng = jax.random.PRNGKey(0)
        subkeys = jax.random.split(rng, 2)

        x = jax.random.normal(subkeys[0], (batch, seqlen, hidden_in), dtype)
        init_rngs = {"params": subkeys[1]}

        # Single GPUs
        with fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
            ln_mlp_single = LayerNormMLP(
                layernorm_type=layernorm_type,
                transpose_batch_sequence=False,  # input: [batch, seqlen, hidden]
                intermediate_dim=INTERMEDIATE,
                activations=activation_type,
                use_bias=use_bias,
            )
            params_single = ln_mlp_single.init(init_rngs, x, deterministic=True)
            mlp_out_single, ln_out_single = ln_mlp_single.apply(
                params_single, x, deterministic=True
            )

        # Multi GPUs
        device_count, mesh_shape, mesh_axes, mesh_resource = mesh_config
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(
            enabled=use_fp8, fp8_recipe=fp8_recipe, mesh_resource=mesh_resource
        ):
            ln_mlp_sharded = LayerNormMLP(
                layernorm_type=layernorm_type,
                transpose_batch_sequence=False,
                intermediate_dim=INTERMEDIATE,
                activations=activation_type,
                scale_axes=LN_SCALE_AXES,
                ln_bias_axes=LN_BIAS_AXES,
                kernel_axes_1=KERNEL_1_AXES,
                kernel_axes_2=KERNEL_2_AXES,
                use_bias=use_bias,
                bias_axes_1=BIAS_1_AXES,
                bias_axes_2=BIAS_2_AXES,
                layernorm_input_axes=LAYERNORM_INPUT_AXES,
                dot_1_input_axes=DOT_1_INPUT_AXES,
                dot_2_input_axes=DOT_2_INPUT_AXES,
                name="mlp",
            )
            params_sharded = ln_mlp_sharded.init(init_rngs, x, deterministic=True)
            mlp_out_sharded, ln_out_sharded = ln_mlp_sharded.apply(
                params_sharded, x, deterministic=True
            )

        # Make sure params values are the same
        assert_tree_like_allclose(params_sharded["params"], params_single["params"])
        assert_allclose(ln_out_sharded, ln_out_single, dtype=dtype)
        assert_allclose(mlp_out_sharded, mlp_out_single, dtype=dtype)

    @pytest_parametrize_wrapper("input_shape", INPUT_SHAPE)
    @pytest_parametrize_wrapper("mesh_config", generate_fsdp_and_tp_configs())
    @pytest_parametrize_wrapper("activation_type", [("gelu",), ("silu", "linear")])
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("use_bias", [True, False])
    @pytest_parametrize_wrapper("use_shardy", [False, True])
    def test_layernorm_mlp_layer(
        self, mesh_config, activation_type, use_bias, input_shape, dtype, use_shardy
    ):
        self._test_layernorm_mlp(
            mesh_config,
            activation_type,
            use_bias,
            input_shape,
            dtype,
            use_fp8=False,
            fp8_recipe=None,
            use_shardy=use_shardy,
        )

    @pytest.mark.skipif(not is_fp8_supported, reason=reason)
    @pytest_parametrize_wrapper("mesh_config", generate_fsdp_and_tp_configs())
    @pytest_parametrize_wrapper("activation_type", [("gelu",), ("gelu", "linear")])
    @pytest_parametrize_wrapper("use_bias", [True, False])
    @pytest_parametrize_wrapper("input_shape", INPUT_SHAPE)
    @pytest_parametrize_wrapper("dtype", DTYPES)
    @pytest_parametrize_wrapper("fp8_recipe", SUPPORTED_RECIPES)
    def test_layernorm_mlp_layer_fp8(
        self, mesh_config, activation_type, use_bias, input_shape, dtype, fp8_recipe
    ):
        self._test_layernorm_mlp(
            mesh_config,
            activation_type,
            use_bias,
            input_shape,
            dtype,
            use_fp8=True,
            fp8_recipe=fp8_recipe,
            use_shardy=False,
        )
