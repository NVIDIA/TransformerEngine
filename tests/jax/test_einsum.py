# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for TE einsum operation with FP8 quantization."""

import jax
import jax.numpy as jnp
import pytest
from jax import value_and_grad

from utils import assert_allclose, pytest_parametrize_wrapper
from transformer_engine.jax.einsum import einsum
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    QuantizeMeta,
    QuantizeMetaSet,
)
from transformer_engine.jax.quantize import helper


# Test parameters
DTYPES = [jnp.bfloat16]
# (B, S, M, E, C, H)
# B: Batch size
# S: Sequence length (number of tokens)
# M: Model dimension (hidden size)
# E: Number of experts
# C: Capacity (max tokens per expert)
# H: Hidden dimension (MLP intermediate size)
MOE_CASES = [
    (2, 32, 128, 4, 32, 64),
]

# Get supported recipes
supported_recipes = helper.get_supported_quantization_recipes()
supported_recipes = [pytest.param(r, id=r.__class__.__name__) for r in supported_recipes]


@pytest.fixture(autouse=True, scope="module")
def init():
    """WAR for CUDA uninitialize error"""
    # Calling customcalls before jax may cause CUDA uninitialize error
    _ = jnp.zeros(0)
    yield


class TestMoEMLPWithRecipes:
    """Test MoE MLP operations with different FP8 recipes and gradients."""

    def _get_quantizer_sets(self, recipe, num_experts):
        return QuantizerFactory.create_set(
            n_quantizer_sets=num_experts,
            fp8_recipe=recipe,
            quantize_meta_set=QuantizeMetaSet(
                x=QuantizeMeta(), kernel=QuantizeMeta(), grad=QuantizeMeta()
            ),
        )

    def _einsum(self, equation, *operands, quantizer_sets=None, quantizer_dim=None, fallback=False):
        out = einsum(
            equation,
            *operands,
            quantizer_sets=quantizer_sets,
            quantizer_dim=quantizer_dim,
            fallback=fallback,
        )
        return jnp.mean(out)

    def _ref_einsum(self, equation, *operands):
        out = jnp.einsum(equation, *operands)
        return jnp.mean(out)

    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("recipe", supported_recipes)
    def test_mlp_up_grad(self, B, S, M, E, C, H, recipe):
        """Test MLP up: EBCM,EMH->EBCH with gradients and different recipes."""
        # Create per-expert quantizers
        quantizer_sets = self._get_quantizer_sets(recipe, E)
        dispatched = jax.random.normal(
            jax.random.PRNGKey(0), (E, B, C, M), dtype=jnp.bfloat16
        ) / jnp.sqrt(M)
        weights = jax.random.normal(jax.random.PRNGKey(1), (E, M, H), dtype=jnp.bfloat16)

        # Compute with TE einsum with quantization
        loss_te, grads_te = value_and_grad(self._einsum, argnums=(1, 2))(
            "EBCM,EMH->EBCH", dispatched, weights, quantizer_sets=quantizer_sets, quantizer_dim="E"
        )

        # Compute reference (BF16)
        loss_ref, grads_ref = value_and_grad(self._ref_einsum, argnums=(1, 2))(
            "EBCM,EMH->EBCH", dispatched, weights
        )

        # Verify shapes and no NaNs
        assert grads_te[0].shape == dispatched.shape
        assert grads_te[1].shape == weights.shape
        assert not jnp.isnan(loss_te)
        assert jnp.all(jnp.isfinite(grads_te[0]))
        assert jnp.all(jnp.isfinite(grads_te[1]))

        # Compare with reference (with FP8 tolerance)
        assert_allclose(loss_te, loss_ref, dtype=quantizer_sets[0].x.q_dtype)
        assert_allclose(grads_te[0], grads_ref[0], dtype=quantizer_sets[0].dgrad.q_dtype)
        assert_allclose(grads_te[1], grads_ref[1], dtype=quantizer_sets[0].dgrad.q_dtype)

    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("recipe", supported_recipes)
    def test_mlp_down_grad(self, B, S, M, E, C, H, recipe):
        """Test MLP down: EBCH,EHM->EBCM with gradients and different recipes."""
        # Create per-expert quantizers
        quantizer_sets = self._get_quantizer_sets(recipe, E)

        hidden = jax.random.normal(
            jax.random.PRNGKey(0), (E, B, C, H), dtype=jnp.bfloat16
        ) / jnp.sqrt(H)
        weights = jax.random.normal(jax.random.PRNGKey(1), (E, H, M), dtype=jnp.bfloat16)

        # Compute with TE einsum with quantization
        loss_te, grads_te = value_and_grad(self._einsum, argnums=(1, 2))(
            "EBCH,EHM->EBCM", hidden, weights, quantizer_sets=quantizer_sets, quantizer_dim="E"
        )

        # Compute reference (BF16)
        loss_ref, grads_ref = value_and_grad(self._ref_einsum, argnums=(1, 2))(
            "EBCH,EHM->EBCM", hidden, weights
        )

        # Verify shapes and no NaNs
        assert grads_te[0].shape == hidden.shape
        assert grads_te[1].shape == weights.shape
        assert not jnp.isnan(loss_te)
        assert jnp.all(jnp.isfinite(grads_te[0]))
        assert jnp.all(jnp.isfinite(grads_te[1]))

        # Compare with reference (with FP8 tolerance)
        assert_allclose(loss_te, loss_ref, dtype=quantizer_sets[0].x.q_dtype)
        assert_allclose(grads_te[0], grads_ref[0], dtype=quantizer_sets[0].dgrad.q_dtype)
        assert_allclose(grads_te[1], grads_ref[1], dtype=quantizer_sets[0].dgrad.q_dtype)

    @pytest_parametrize_wrapper("B,S,M,E,C,H", MOE_CASES)
    @pytest_parametrize_wrapper("recipe", supported_recipes)
    def test_full_moe_grad(self, B, S, M, E, C, H, recipe):
        """Test full MoE pipeline (all 4 einsums) with gradients and different recipes."""
        # Create per-expert quantizers for each einsum
        mlp_up_quantizer_sets = self._get_quantizer_sets(recipe, E)
        mlp_down_quantizer_sets = self._get_quantizer_sets(recipe, E)

        tokens = jax.random.normal(jax.random.PRNGKey(0), (B, S, M), dtype=jnp.bfloat16) / jnp.sqrt(
            M
        )
        routing = jax.random.normal(jax.random.PRNGKey(1), (B, S, E, C), dtype=jnp.bfloat16)
        routing = jax.nn.softmax(routing, axis=-1)  # Normalize routing weights
        up_weights = jax.random.normal(
            jax.random.PRNGKey(2), (E, M, H), dtype=jnp.bfloat16
        ) / jnp.sqrt(H)
        down_weights = jax.random.normal(
            jax.random.PRNGKey(3), (E, H, M), dtype=jnp.bfloat16
        ) / jnp.sqrt(M)

        # TE implementation with quantization
        def full_moe_te(tokens, routing, up_w, down_w):
            """Complete MoE pipeline with TE einsum."""
            dispatched = einsum("BSM,BSEC->EBCM", tokens, routing, fallback=True)
            hidden = einsum(
                "EBCM,EMH->EBCH",
                dispatched,
                up_w,
                quantizer_sets=mlp_up_quantizer_sets,
                quantizer_dim="E",
            )
            expert_out = einsum(
                "EBCH,EHM->EBCM",
                hidden,
                down_w,
                quantizer_sets=mlp_down_quantizer_sets,
                quantizer_dim="E",
            )
            output = einsum("EBCM,BSEC->BSM", expert_out, routing, fallback=True)
            return jnp.sum(output)

        # Reference implementation with jnp.einsum
        def full_moe_ref(tokens, routing, up_w, down_w):
            """Complete MoE pipeline with jnp.einsum."""
            dispatched = jnp.einsum("BSM,BSEC->EBCM", tokens, routing)
            hidden = jnp.einsum("EBCM,EMH->EBCH", dispatched, up_w)
            expert_out = jnp.einsum("EBCH,EHM->EBCM", hidden, down_w)
            output = jnp.einsum("EBCM,BSEC->BSM", expert_out, routing)
            return jnp.sum(output)

        loss_te, grads_te = value_and_grad(full_moe_te, argnums=(0, 1, 2, 3))(
            tokens, routing, up_weights, down_weights
        )

        loss_ref, grads_ref = value_and_grad(full_moe_ref, argnums=(0, 1, 2, 3))(
            tokens, routing, up_weights, down_weights
        )

        # Verify all gradient shapes
        assert grads_te[0].shape == tokens.shape, f"tokens grad shape mismatch"
        assert grads_te[1].shape == routing.shape, f"routing grad shape mismatch"
        assert grads_te[2].shape == up_weights.shape, f"up_weights grad shape mismatch"
        assert grads_te[3].shape == down_weights.shape, f"down_weights grad shape mismatch"

        # Verify no NaNs or Infs
        assert not jnp.isnan(loss_te), "Loss is NaN"
        assert jnp.isfinite(loss_te), "Loss is Inf"
        assert jnp.all(jnp.isfinite(grads_te[0])), "tokens grad has NaN/Inf"
        assert jnp.all(jnp.isfinite(grads_te[1])), "routing grad has NaN/Inf"
        assert jnp.all(jnp.isfinite(grads_te[2])), "up_weights grad has NaN/Inf"
        assert jnp.all(jnp.isfinite(grads_te[3])), "down_weights grad has NaN/Inf"

        # Compare with reference (with FP8 tolerance)
        assert_allclose(loss_te, loss_ref, dtype=mlp_up_quantizer_sets[0].x.q_dtype)
        assert_allclose(grads_te[0], grads_ref[0], dtype=mlp_up_quantizer_sets[0].dgrad.q_dtype)
        assert_allclose(grads_te[1], grads_ref[1], dtype=mlp_up_quantizer_sets[0].dgrad.q_dtype)
        assert_allclose(grads_te[2], grads_ref[2], dtype=mlp_down_quantizer_sets[0].x.q_dtype)
        assert_allclose(grads_te[3], grads_ref[3], dtype=mlp_down_quantizer_sets[0].dgrad.q_dtype)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
