# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Tests for Triton-based custom calls in TE JAX."""

import jax
import jax.numpy as jnp
import pytest

from utils import assert_allclose, pytest_parametrize_wrapper

import triton
import triton.language as tl

from transformer_engine.jax.cpp_extensions.base import BasePrimitive, register_primitive
from transformer_engine.jax.triton_extensions import triton_call_lowering


@pytest.fixture(autouse=True, scope="module")
def init():
    """WAR for CUDA uninitialize error"""
    _ = jnp.zeros(0)
    yield


class TestTritonBinding:
    """Test Triton binding primitive."""

    # Define autotuned Triton kernel
    @staticmethod
    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 256}),  # Uses defaults: num_warps=4, num_stages=3
            triton.Config({"BLOCK_SIZE": 512}, num_warps=8),  # Custom num_warps
        ],
        key=["n_elements"],  # Autotune based on input size
    )
    @triton.jit
    def amax_kernel(
        x_ptr,
        amax_ptr,
        n_elements: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Compute amax using Triton with autotuning."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        abs_x = tl.abs(x)
        block_max = tl.max(abs_x)

        tl.atomic_max(amax_ptr, block_max)

    # Define test primitive
    class AmaxTritonPrimitive(BasePrimitive):
        """Test primitive using Triton kernel."""

        name = "te_amax_triton_test"
        multiple_results = False
        impl_static_args = ()

        @staticmethod
        def abstract(x_aval):
            return jax.core.ShapedArray((1,), jnp.float32)

        @staticmethod
        def impl(x):
            assert TestTritonBinding.AmaxTritonPrimitive.inner_primitive is not None
            return TestTritonBinding.AmaxTritonPrimitive.inner_primitive.bind(x)

        @staticmethod
        def lowering(ctx, x):
            """MLIR lowering using Triton kernel."""
            n_elements = 1
            for dim in ctx.avals_in[0].shape:
                n_elements *= dim

            # For autotuned kernels, use the minimum BLOCK_SIZE from configs
            # to ensure all elements are processed by all configs
            block_size = min(
                config.kwargs.get("BLOCK_SIZE") for config in TestTritonBinding.amax_kernel.configs
            )
            grid = (triton.cdiv(n_elements, block_size),)

            return triton_call_lowering(
                ctx,
                TestTritonBinding.amax_kernel,  # Autotuned kernel
                x,
                grid=grid,
                constexprs={"n_elements": n_elements},
                # BLOCK_SIZE comes from autotuner config, not passed here
            )

    register_primitive(AmaxTritonPrimitive)

    @staticmethod
    def _triton_amax(x: jnp.ndarray) -> jnp.ndarray:
        """Compute amax using Triton kernel."""
        return TestTritonBinding.AmaxTritonPrimitive.outer_primitive.bind(x)

    @pytest_parametrize_wrapper("shape", [(1024, 1024)])
    @pytest_parametrize_wrapper("dtype", [jnp.bfloat16])
    def test_triton_amax(self, shape, dtype):
        """Test Triton amax with JIT."""
        key = jax.random.PRNGKey(0)
        x = jax.random.uniform(key, shape, dtype)

        expected = jnp.max(jnp.abs(x), keepdims=False).astype(jnp.float32)
        jitted_amax = jax.jit(self._triton_amax)
        result = jitted_amax(x)

        assert_allclose(result, expected, dtype=jnp.float32)
