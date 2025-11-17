# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest
from functools import partial
from abc import ABC, abstractmethod

import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from utils import assert_allclose, pytest_parametrize_wrapper
from transformer_engine.common.recipe import (
    Recipe,
    DelayedScaling,
    MXFP8BlockScaling,
    Float8CurrentScaling,
    NVFP4BlockScaling,
)
from transformer_engine.common.recipe import Format as FP8Format
from transformer_engine.jax import autocast
from transformer_engine.jax.quantize import (
    get_global_quantize_recipe,
    get_quantize_config_with_recipe,
    get_supported_quantization_recipes,
    is_scaling_mode_supported,
    ScalingMode,
    update_collections,
    TensorSource,
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
        return quantizer_check_fwd(inner_quantizer_set, assertion_func, x)[0]

    def quantizer_check_fwd(inner_quantizer_set, assertion_func, x):
        assertion_func(inner_quantizer_set.x, TensorSource.X)
        assertion_func(inner_quantizer_set.kernel, TensorSource.KERNEL)
        assertion_func(inner_quantizer_set.dgrad, TensorSource.DGRAD)
        return x, (inner_quantizer_set,)

    def quantizer_check_bwd(assertion_func, ctx, g):
        (inner_quantizer_set,) = ctx
        return (inner_quantizer_set, g)

    quantizer_check.defvjp(quantizer_check_fwd, quantizer_check_bwd)
    return quantizer_check(outer_quantizer_set, assertion_func, x)


class TestModule(TransformerEngineBase):
    """A simple module to test quantizer creation and reconstruction across VJP boundaries."""

    # Signature: (quantizer: Quantizer, tensor_source: TensorSource) -> None
    assertion_func: callable
    direct_recipe: Recipe

    @nn.compact
    def __call__(self, x):
        quantizer_set = self.generate_quantizer_set(fp8_recipe=self.direct_recipe)
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


def assert_fp8_format(quantizer, tensor_source, fp8_format):
    if fp8_format == FP8Format.HYBRID:
        if tensor_source == TensorSource.DGRAD:
            assert quantizer.q_dtype == jnp.float8_e5m2
        else:
            assert quantizer.q_dtype == jnp.float8_e4m3fn
    elif fp8_format == FP8Format.E4M3:
        assert quantizer.q_dtype == jnp.float8_e4m3fn
    else:
        raise ValueError(f"Unsupported FP8 format: {fp8_format}")


class RecipeAssertionBase(ABC):
    """Base class for defining recipe assertions."""

    @abstractmethod
    def assert_context(self, ref_recipe, quantize_config):
        """Asserts that the quantize_config matches the expected properties from the reference recipe when the recipe is used with an autocast context.

        Args:
            ref_recipe: The reference quantization recipe.
            quantize_config: The quantization configuration to be checked.
        """
        pass

    @abstractmethod
    def assert_quantizers(self, ref_recipe, quantizer, tensor_source):
        """Asserts that the quantizer matches the expected properties from the reference recipe. The quantizers are created in a small test Flax module TestModule and passed through a VJP boundary to ensure correct reconstruction.

        Args:
            ref_recipe: The reference quantization recipe.
            quantizer: The quantizer to be checked.
            tensor_source: The source of the tensor (e.g., KERNEL, X, DGRAD).
        """
        pass


class DelayedScalingRecipeAssertion(RecipeAssertionBase):

    def assert_context(self, ref_recipe, quantize_config):
        assert quantize_config.MARGIN == ref_recipe.margin
        assert quantize_config.FWD_DTYPE == _format2dtypes(ref_recipe.fp8_format)[0]
        assert quantize_config.BWD_DTYPE == _format2dtypes(ref_recipe.fp8_format)[1]
        assert quantize_config.AMAX_HISTORY_LEN == ref_recipe.amax_history_len
        assert quantize_config.AMAX_COMPUTE_ALGO.value == ref_recipe.amax_compute_algo
        for tensor_source in TensorSource:
            assert (
                quantize_config.get_scaling_mode(tensor_source)
                == ScalingMode.DELAYED_TENSOR_SCALING
            )

    def assert_quantizers(self, ref_recipe: DelayedScaling, quantizer, tensor_source):
        assert quantizer.scaling_mode == ScalingMode.DELAYED_TENSOR_SCALING
        assert quantizer.margin == ref_recipe.margin
        assert quantizer.amax_compute_algo.value == ref_recipe.amax_compute_algo
        assert quantizer.amax_history.shape == (ref_recipe.amax_history_len,)
        assert_fp8_format(quantizer, tensor_source, ref_recipe.fp8_format)


class CurrentScalingRecipeAssertion(RecipeAssertionBase):

    def assert_context(self, ref_recipe, quantize_config):
        assert quantize_config.FWD_DTYPE == _format2dtypes(ref_recipe.fp8_format)[0]
        assert quantize_config.BWD_DTYPE == _format2dtypes(ref_recipe.fp8_format)[1]
        for tensor_source in TensorSource:
            assert (
                quantize_config.get_scaling_mode(tensor_source)
                == ScalingMode.CURRENT_TENSOR_SCALING
            )

    def assert_quantizers(self, ref_recipe: Float8CurrentScaling, quantizer, tensor_source):
        assert quantizer.scaling_mode == ScalingMode.CURRENT_TENSOR_SCALING
        assert_fp8_format(quantizer, tensor_source, ref_recipe.fp8_format)


class MXFP8RecipeAssertion(RecipeAssertionBase):

    def assert_context(self, ref_recipe, quantize_config):
        assert quantize_config.FWD_DTYPE == _format2dtypes(ref_recipe.fp8_format)[0]
        assert quantize_config.BWD_DTYPE == _format2dtypes(ref_recipe.fp8_format)[1]
        for tensor_source in TensorSource:
            assert quantize_config.get_scaling_mode(tensor_source) == ScalingMode.MXFP8_1D_SCALING

    def assert_quantizers(self, ref_recipe: MXFP8BlockScaling, quantizer, tensor_source):
        assert quantizer.scaling_mode == ScalingMode.MXFP8_1D_SCALING
        assert_fp8_format(quantizer, tensor_source, ref_recipe.fp8_format)


class NVFP4RecipeAssertion(RecipeAssertionBase):

    def assert_context(self, ref_recipe, quantize_config):
        assert quantize_config.FWD_DTYPE == _format2dtypes(ref_recipe.fp4_format)[0]
        assert quantize_config.BWD_DTYPE == _format2dtypes(ref_recipe.fp4_format)[1]
        for tensor_source in TensorSource:
            target_scaling_mode = (
                ScalingMode.NVFP4_2D_SCALING
                if (not ref_recipe.disable_2d_quantization) and tensor_source == TensorSource.KERNEL
                else ScalingMode.NVFP4_1D_SCALING
            )
            assert quantize_config.get_scaling_mode(tensor_source) == target_scaling_mode
        assert quantize_config.DISABLE_STOCHASTIC_ROUNDING == ref_recipe.disable_stochastic_rounding
        assert quantize_config.DISABLE_RHT == ref_recipe.disable_rht
        assert quantize_config.DISABLE_2D_QUANTIZATION == ref_recipe.disable_2d_quantization

    def assert_quantizers(self, ref_recipe: NVFP4BlockScaling, quantizer, tensor_source):
        if tensor_source == TensorSource.KERNEL and not ref_recipe.disable_2d_quantization:
            assert quantizer.scaling_mode == ScalingMode.NVFP4_2D_SCALING
        else:
            assert quantizer.scaling_mode == ScalingMode.NVFP4_1D_SCALING

        if ref_recipe.disable_stochastic_rounding or tensor_source != TensorSource.DGRAD:
            assert quantizer.stochastic_rounding_rng_state is None
        else:
            assert quantizer.stochastic_rounding_rng_state is not None

        expected_rht = (
            quantizer.scaling_mode == ScalingMode.NVFP4_1D_SCALING
            and quantizer.q_layout in {QuantizeLayout.ROWWISE_COLWISE, QuantizeLayout.COLWISE}
            and not ref_recipe.disable_rht
        )
        assert quantizer.use_rht == expected_rht


class TestFP8Functions(unittest.TestCase):

    def _check_default_state(self):
        self.assertEqual(get_global_quantize_recipe(), None)

    def _test_recipe(self, quantization_recipe: Recipe, cls: RecipeAssertionBase):
        """Tests a quantization recipe by verifying its behavior in both autocast and direct application contexts."""
        assert_context_func = cls().assert_context
        assert_quantizer_func = partial(cls().assert_quantizers, quantization_recipe)
        self._test_recipe_autocast(quantization_recipe, assert_context_func, assert_quantizer_func)
        self._test_recipe_direct(quantization_recipe, assert_quantizer_func)

    def _test_recipe_autocast(
        self, quantization_recipe, assert_context_func, assert_quantizer_func
    ):
        """Tests a quantization recipe within an autocast context by verifying the quantize config and quantizers in a test module."""
        self._check_default_state()
        with autocast(enabled=False, recipe=quantization_recipe, mesh_resource=MeshResource()):
            self._check_default_state()
        with autocast(enabled=True, recipe=quantization_recipe, mesh_resource=MeshResource()):
            quantize_config = self._get_global_quantize_config()
            assert_context_func(quantization_recipe, quantize_config)
            self._test_quantizer_in_model(assert_quantizer_func)
        self._check_default_state()

    def _test_recipe_direct(self, quantization_recipe, assert_quantizer_func):
        """Tests a quantization recipe by directly passing it to a test module and verifying the quantizers."""
        self._check_default_state()
        self._test_quantizer_in_model(assert_quantizer_func, direct_recipe=quantization_recipe)
        self._check_default_state()

    def _test_quantizer_in_model(self, assert_quantizer_func, direct_recipe=None):
        """Tests that the quantizers created in a test module match the expected properties by passing them through a VJP boundary.

        Args:
            assert_quantizer_func: A function that asserts the properties of the quantizers. The function signature is (quantizer: Quantizer, tensor_source: TensorSource) -> None.
            direct_recipe: An optional quantization recipe to be passed directly to the test module. This is an alternative API to using autocast contexts.
        """
        x = jnp.ones((), dtype=jnp.float32)
        test_module = TestModule(assertion_func=assert_quantizer_func, direct_recipe=direct_recipe)
        param_key, sr_key = jax.random.split(jax.random.PRNGKey(0))
        rngs = {"params": param_key, "sr_rng": sr_key}
        variables = test_module.init(rngs, x)

        jax.jit(jax.value_and_grad(test_module.apply), static_argnums=(2,))(variables, x, rngs=rngs)

    def _get_global_quantize_config(self):
        quantization_recipe = get_global_quantize_recipe()
        assert quantization_recipe is not None, "No global quantization recipe set"
        quantize_config = get_quantize_config_with_recipe(quantization_recipe)
        assert (
            quantize_config.is_fp8_enabled()
        ), "Quantization not enabled in global quantize config"
        return quantize_config

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_autocast_delayed_scaling(self):
        self._test_recipe(
            quantization_recipe=DelayedScaling(),
            cls=DelayedScalingRecipeAssertion,
        )
        self._test_recipe(
            quantization_recipe=DelayedScaling(
                margin=5.0, fp8_format=FP8Format.E4M3, amax_history_len=1
            ),
            cls=DelayedScalingRecipeAssertion,
        )
        self._test_recipe(
            quantization_recipe=DelayedScaling(
                margin=3.0, fp8_format=FP8Format.HYBRID, amax_history_len=1
            ),
            cls=DelayedScalingRecipeAssertion,
        )

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_autocast_current_scaling(self):
        self._test_recipe(
            quantization_recipe=Float8CurrentScaling(),
            cls=CurrentScalingRecipeAssertion,
        )
        self._test_recipe(
            quantization_recipe=Float8CurrentScaling(margin=5.0, fp8_format=FP8Format.E4M3),
            cls=CurrentScalingRecipeAssertion,
        )
        self._test_recipe(
            quantization_recipe=Float8CurrentScaling(margin=3.0, fp8_format=FP8Format.HYBRID),
            cls=CurrentScalingRecipeAssertion,
        )

    @unittest.skipIf(not is_mxfp8_supported, reason=mxfp8_reason)
    def test_autocast_mxfp8_block_scaling(self):
        self._test_recipe(
            quantization_recipe=MXFP8BlockScaling(),
            cls=MXFP8RecipeAssertion,
        )

    @unittest.skipIf(not is_nvfp4_supported, reason=nvfp4_reason)
    def test_autocast_nvfp4_block_scaling(self):
        self._test_recipe(
            quantization_recipe=NVFP4BlockScaling(),
            cls=NVFP4RecipeAssertion,
        )
        self._test_recipe(
            quantization_recipe=NVFP4BlockScaling(
                disable_stochastic_rounding=True,
                disable_rht=True,
                disable_2d_quantization=True,
            ),
            cls=NVFP4RecipeAssertion,
        )


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
