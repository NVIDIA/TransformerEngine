# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest
from functools import partial

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from utils import assert_allclose, pytest_parametrize_wrapper
from transformer_engine.common.recipe import (
    DelayedScaling,
    MXFP8BlockScaling,
    Float8CurrentScaling,
    NVFP4BlockScaling,
)
from transformer_engine.common.recipe import Format as FP8Format
from transformer_engine.jax import autocast
from transformer_engine.jax.quantize import (
    get_quantize_config,
    get_supported_quantization_recipes,
    is_scaling_mode_supported,
    ScalingMode,
    update_collections,
    TensorSource,
    QuantizerFactory,
    QuantizeLayout,
)
from transformer_engine.jax.quantize.helper import _format2dtypes
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource
from transformer_engine.jax.flax.module import TransformerEngineBase
from transformer_engine.jax import flax as te_flax
import transformer_engine.jax as te

is_fp8_supported, reason = is_scaling_mode_supported(ScalingMode.DELAYED_TENSOR_SCALING)
is_mxfp8_supported, mxfp8_reason = is_scaling_mode_supported(ScalingMode.MXFP8_1D_SCALING)
is_nvfp4_supported, nvfp4_reason = is_scaling_mode_supported(ScalingMode.NVFP4_1D_SCALING)

SUPPORTED_RECIPES = get_supported_quantization_recipes()


def quantizer_check_vjp(outer_quantizer_set, assertion_func, x):
    """Check that the quantizers in the quantizer set are as expected and reconstructed correctly from flattened pytree representations across VJP boundaries."""

    # Define a function with a custom VJP (vector-Jacobian product)
    @partial(jax.custom_vjp, nondiff_argnums=(1,))
    def quantizer_check(inner_quantizer_set, assertion_func, x):
        return quantizer_check_fwd(inner_quantizer_set, assertion_func, x)

    def quantizer_check_fwd(inner_quantizer_set, assertion_func, x):
        assertion_func(inner_quantizer_set.x, TensorSource.X)
        assertion_func(inner_quantizer_set.kernel, TensorSource.KERNEL)
        assertion_func(inner_quantizer_set.dgrad, TensorSource.DGRAD)
        return x

    def quantizer_check_bwd(ctx, g):
        return (g,)

    quantizer_check.defvjp(quantizer_check_fwd, quantizer_check_bwd)
    return quantizer_check(outer_quantizer_set, assertion_func, x)


class TestModule(TransformerEngineBase):
    """A simple module to test quantizer creation and reconstruction across VJP boundaries."""

    # Signature: (quantizer: Quantizer, tensor_source: TensorSource) -> None
    assertion_func: callable

    @nn.compact
    def __call__(self, x):
        quantizer_set = self.generate_quantizer_set()
        return quantizer_check_vjp(quantizer_set, self.assertion_func, x)


class TestHelper(unittest.TestCase):

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_update_collections(self):
        original_val = 0.0
        updated_val = 10.0

        original_state = {
            "test1": original_val,
            "test2": original_val,
        }
        updated_state = update_collections({"test1": updated_val}, original_state)
        self.assertEqual(updated_state["test1"], updated_val)
        self.assertEqual(updated_state["test2"], original_val)

        original_state = flax.core.frozen_dict.FrozenDict(original_state)
        updated_state = update_collections({"test1": updated_val}, original_state)
        self.assertEqual(updated_state["test1"], updated_val)
        self.assertEqual(updated_state["test2"], original_val)


class TestFP8Functions(unittest.TestCase):

    def _check_default_state(self):
        self.assertFalse(get_quantize_config().is_fp8_enabled())

    def _compare_delay_scaling(self, test):
        self.assertEqual(get_quantize_config().MARGIN, test.margin)
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp8_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp8_format)[1])
        self.assertEqual(get_quantize_config().AMAX_HISTORY_LEN, test.amax_history_len)
        self.assertEqual(get_quantize_config().AMAX_COMPUTE_ALGO.value, test.amax_compute_algo)

    def _compare_current_scaling(self, test):
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp8_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp8_format)[1])
        for tensor_source in TensorSource:
            self.assertEqual(
                get_quantize_config().get_scaling_mode(tensor_source),
                ScalingMode.CURRENT_TENSOR_SCALING,
            )

    def _compare_mxfp8_scaling(self, test):
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp8_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp8_format)[1])
        for tensor_source in TensorSource:
            self.assertEqual(
                get_quantize_config().get_scaling_mode(tensor_source), ScalingMode.MXFP8_1D_SCALING
            )

    def _compare_nvfp4_scaling(self, test):
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp4_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp4_format)[1])
        for tensor_source in TensorSource:
            target_scaling_mode = (
                ScalingMode.NVFP4_2D_SCALING
                if (not test.disable_2d_quantization) and tensor_source == TensorSource.KERNEL
                else ScalingMode.NVFP4_1D_SCALING
            )
            self.assertEqual(
                get_quantize_config().get_scaling_mode(tensor_source), target_scaling_mode
            )
        self.assertEqual(
            get_quantize_config().DISABLE_STOCHASTIC_ROUNDING, test.disable_stochastic_rounding
        )
        self.assertEqual(get_quantize_config().DISABLE_RHT, test.disable_rht)
        self.assertEqual(
            get_quantize_config().DISABLE_2D_QUANTIZATION, test.disable_2d_quantization
        )

    def _compare_nvfp4_scaling_quantizers(self, test):
        """Check that the quantizers created have the expected stochastic rounding state and the state is preserved across VJP boundaries."""

        def assertion_func(quantizer, tensor_source):
            if test.disable_stochastic_rounding or tensor_source != TensorSource.DGRAD:
                self.assertIsNone(quantizer.stochastic_rounding_rng_state)
            else:
                self.assertIsNotNone(quantizer.stochastic_rounding_rng_state)

            expected_rht = (
                quantizer.scaling_mode == ScalingMode.NVFP4_1D_SCALING
                and quantizer.q_layout in {QuantizeLayout.ROWWISE_COLWISE, QuantizeLayout.COLWISE}
                and not test.disable_rht
            )
            self.assertEqual(quantizer.use_rht, expected_rht)

        x = jnp.ones((), dtype=jnp.float32)
        test_module = TestModule(assertion_func=assertion_func)
        param_key, sr_key = jax.random.split(jax.random.PRNGKey(0))
        rngs = {"params": param_key, "sr_rng": sr_key}
        variables = test_module.init(rngs, x)

        jax.jit(jax.value_and_grad(test_module.apply), static_argnums=(2,))(variables, x, rngs=rngs)

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_autocast_delayed_scaling(self):
        self._check_default_state()

        with autocast(enabled=False, recipe=DelayedScaling(), mesh_resource=MeshResource()):
            self._check_default_state()

        self._check_default_state()

        ds = DelayedScaling(margin=5.0, fp8_format=FP8Format.E4M3, amax_history_len=1)
        with autocast(enabled=True, recipe=ds, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_delay_scaling(ds)

        self._check_default_state()

        ds = DelayedScaling(margin=3.0, fp8_format=FP8Format.HYBRID, amax_history_len=1)
        with autocast(enabled=True, recipe=ds, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_delay_scaling(ds)

        self._check_default_state()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_autocast_current_scaling(self):
        self._check_default_state()

        with autocast(enabled=False, recipe=Float8CurrentScaling(), mesh_resource=MeshResource()):
            self._check_default_state()

        self._check_default_state()

        cs = Float8CurrentScaling(fp8_format=FP8Format.E4M3)
        with autocast(enabled=True, recipe=cs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_current_scaling(cs)

        self._check_default_state()

        cs = Float8CurrentScaling(fp8_format=FP8Format.HYBRID)
        with autocast(enabled=True, recipe=cs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_current_scaling(cs)

        self._check_default_state()

    @unittest.skipIf(not is_mxfp8_supported, reason=mxfp8_reason)
    def test_autocast_mxfp8_block_scaling(self):
        self._check_default_state()

        with autocast(enabled=False, recipe=MXFP8BlockScaling(), mesh_resource=MeshResource()):
            self._check_default_state()

        self._check_default_state()

        bs = MXFP8BlockScaling()
        with autocast(enabled=True, recipe=bs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_mxfp8_scaling(bs)

        self._check_default_state()

    @unittest.skipIf(not is_nvfp4_supported, reason=nvfp4_reason)
    def test_autocast_nvfp4_block_scaling(self):
        self._check_default_state()

        with autocast(enabled=False, recipe=NVFP4BlockScaling(), mesh_resource=MeshResource()):
            self._check_default_state()

        self._check_default_state()

        bs = NVFP4BlockScaling()
        with autocast(enabled=True, recipe=bs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_nvfp4_scaling(bs)
            self._compare_nvfp4_scaling_quantizers(bs)

        bs = NVFP4BlockScaling(
            disable_stochastic_rounding=True,
            disable_rht=True,
            disable_2d_quantization=True,
        )
        with autocast(enabled=True, recipe=bs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_nvfp4_scaling(bs)
            self._compare_nvfp4_scaling_quantizers(bs)

        self._check_default_state()


class TestJaxprAndHlo:
    """Tests to verify Jaxpr and/or HLO of compiled modules apply expected recipe functionality and optimizations."""

    def _generate_jaxpr_for_layernorm_mlp_fwd_bwd(self, quantization_recipe, ln_mlp_kwargs=None):
        """Generates the jaxpr for a forward and backward pass of LayerNormMLP under the given quantization recipe."""
        ln_mlp_kwargs = ln_mlp_kwargs or {}
        with te.autocast(enabled=True, recipe=quantization_recipe, mesh_resource=te.MeshResource()):
            model = te_flax.LayerNormMLP(
                layernorm_type="rmsnorm",
                return_layernorm_output=False,
                intermediate_dropout_rate=0.0,
                dtype=jnp.bfloat16,
                **ln_mlp_kwargs,
            )

            var_collect = model.init(
                jax.random.PRNGKey(0),
                jnp.ones((128, 128), dtype=jnp.bfloat16),
            )

            def loss_fn(x, rngs):
                return jnp.mean(model.apply(var_collect, x, rngs=rngs)[0])

            x = jax.random.normal(jax.random.PRNGKey(0), (128, 128), dtype=jnp.bfloat16)
            rngs = {"sr_rng": jax.random.PRNGKey(1), "dropout": jax.random.PRNGKey(2)}
            return jax.make_jaxpr(jax.value_and_grad(loss_fn))(x, rngs=rngs)

    @pytest_parametrize_wrapper(
        "quantization_recipe",
        [
            quantization_recipe
            for quantization_recipe in SUPPORTED_RECIPES
            if isinstance(quantization_recipe, NVFP4BlockScaling)
        ],
    )
    def test_layernorm_mlp_reuses_amax_nvfp4(self, quantization_recipe):
        """Tests that layernorm_mlp reuses the amax computed in layernorm and the activation and does not recompute it during quantizaton."""

        jaxpr = self._generate_jaxpr_for_layernorm_mlp_fwd_bwd(quantization_recipe)

        rht_amax_eqns = [
            eqn for eqn in jaxpr.jaxpr.eqns if eqn.primitive.name == "te_rht_amax_ffi_wrapper"
        ]

        assert len(rht_amax_eqns) == 4, f"Expected 4 rht_amax_eqns, got {len(rht_amax_eqns)}"

        def assert_param(index, tensor_name, expected_value: bool):
            if expected_value:
                assert rht_amax_eqns[index].params["produce_regular_amax"] == True, (
                    f"Expected produce_regular_amax for {tensor_name} to be True, indicating no"
                    " reuse of amax as this tensor does not have a previous operation to fuse"
                    " with"
                )
            else:
                assert rht_amax_eqns[index].params["produce_regular_amax"] == False, (
                    f"Expected produce_regular_amax for {tensor_name} to be False, indicating"
                    " reuse of amax"
                )

        assert_param(0, "fwd ln+q", False)
        assert_param(1, "fwd act+q", False)
        # No previous op before incoming dgrad in the backward so amax is not reused
        assert_param(2, "bwd dgrad", True)
        assert_param(3, "bwd dact+q", False)

    @pytest_parametrize_wrapper("quantization_recipe", SUPPORTED_RECIPES)
    @pytest_parametrize_wrapper(
        "quantization_checkpoint_name",
        [None, "quantization", "some_arbitrary_user_checkpoint_name"],
    )
    def test_recipe_supports_quantization_checkpointing(
        self, quantization_recipe, quantization_checkpoint_name
    ):
        """Tests that all supported quantization recipes correctly use checkpoint_name."""

        kwargs = {
            "quantization_checkpoint_name": quantization_checkpoint_name,
        }
        jaxpr = self._generate_jaxpr_for_layernorm_mlp_fwd_bwd(quantization_recipe, kwargs)

        checkpoint_name_eqns = [
            eqn
            for eqn in jaxpr.jaxpr.eqns
            if eqn.primitive.name == "name" and eqn.params["name"] == quantization_checkpoint_name
        ]

        if quantization_checkpoint_name is None:
            assert len(checkpoint_name_eqns) == 0, (
                "Expected 0 checkpoint_name eqns when quantization_checkpoint_name is None, got"
                f" {len(checkpoint_name_eqns)}"
            )
            return

        # 12 checkpointed values:
        # - Fwd pass:
        #   - Input RMSNorm+Q -> 3 possible output tensors that will be used in the backward
        #   - Kernel Q -> 3 possible output tensors that will be used in the backward
        #   - Input Activation+Q -> 3 possible output tensors that will be used in the backward
        #   - Kernel Q -> 3 possible output tensors that will be used in the backward
        expected_checkpoint_eqn_count = 12

        assert len(checkpoint_name_eqns) == expected_checkpoint_eqn_count, (
            f"Expected {expected_checkpoint_eqn_count} checkpoint_name eqns when"
            f" quantization_checkpoint_name is set, got {len(checkpoint_name_eqns)}"
        )
