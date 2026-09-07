# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest entry points for the JAX attention tutorials."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from transformer_engine.jax.attention import (
    AttnBiasType,
    AttnMaskType,
    AttnSoftmaxType,
    QKVLayout,
    is_fused_attn_kernel_available,
)
from transformer_engine_jax import get_device_compute_capability

# Imports from ``attention`` and ``attention_context_parallel`` are intentionally
# deferred into each test body. The tutorial modules create tensors and initialize
# models at module scope; deferring imports lets pytest apply skip marks before
# unsupported CI nodes allocate those examples.

requires_hopper_or_newer = pytest.mark.skipif(
    get_device_compute_capability(0) < 90,
    reason="the 4K native JAX baseline requires more than 48 GiB of device memory",
)


def test_bshd_gqa_swa_runs():
    import attention

    out = attention.te_model.apply(
        attention.te_vars,
        attention.qkv,
        sequence_descriptor=attention.sequence_descriptor,
        deterministic=False,
    )

    assert out.shape == attention.dout.shape
    assert out.dtype == attention.dtype


@requires_hopper_or_newer
def test_bshd_gqa_swa_matches_baseline():
    import attention

    attention.compare_te_to_baseline()


@requires_hopper_or_newer
def test_single_gpu_benchmark():
    import attention

    attention.run_single_gpu_bench()


def test_mla_variant_runs():
    import attention

    out = attention.mla_model.apply(
        attention.mla_vars,
        attention.mla_qkv,
        sequence_descriptor=attention.sequence_descriptor,
        deterministic=False,
    )
    loss, grads = attention.run_forward_backward(
        attention.mla_model,
        attention.mla_vars,
        attention.mla_qkv,
        attention.mla_dout,
        attention.sequence_descriptor,
    )
    jax.block_until_ready((out, loss, grads))

    assert out.shape == attention.mla_dout.shape
    assert out.dtype == attention.dtype
    assert loss.shape == ()
    assert [grad.shape for grad in grads] == [x.shape for x in attention.mla_qkv]


def _context_parallel_supported():
    cp_size = 4
    if len(jax.devices()) < cp_size:
        return False, f"needs {cp_size} GPUs"

    has_kernel = is_fused_attn_kernel_available(
        True,
        jnp.bfloat16,
        jnp.bfloat16,
        QKVLayout.THD_THD_THD,
        AttnBiasType.NO_BIAS,
        AttnMaskType.PADDING_CAUSAL_MASK,
        AttnSoftmaxType.VANILLA_SOFTMAX,
        0.0,
        128,
        8,
        65536,
        65536,
        128,
        128,
        (8192, 0),
    )
    if not has_kernel:
        return False, "no fused attention kernel for the THD SWA shape"
    return True, ""


_cp_supported, _cp_reason = _context_parallel_supported()
requires_cp = pytest.mark.skipif(
    not _cp_supported,
    reason=f"context-parallel attention tutorial skipped: {_cp_reason}",
)


def _assert_cp_result(cp_attention, strategy, stripe_size):
    result = cp_attention.run_context_parallel_case(strategy, stripe_size)
    reference = cp_attention.run_reference_attention()

    assert result["output"].shape == (
        cp_attention.batch,
        cp_attention.seq,
        cp_attention.num_query_heads,
        cp_attention.head_dim,
    )
    assert result["output"].dtype == cp_attention.dtype
    assert result["loss"].shape == ()
    assert [grad.shape for grad in result["grads"]] == [
        x.shape for x in cp_attention.create_qkv_inputs()[:3]
    ]

    valid_tokens = cp_attention.segment_ids.astype(bool)[..., None, None]
    valid_diff = jax.numpy.max(
        jax.numpy.where(
            valid_tokens,
            jax.numpy.abs(
                result["output"].astype(jax.numpy.float32) - reference.astype(jax.numpy.float32)
            ),
            0.0,
        )
    )
    padded_max = jax.numpy.max(
        jax.numpy.where(
            valid_tokens,
            0.0,
            jax.numpy.abs(result["output"].astype(jax.numpy.float32)),
        )
    )
    np.testing.assert_allclose(valid_diff, 0, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(padded_max, 0, rtol=5e-2, atol=5e-2)


@requires_cp
def test_multi_gpu_context_parallel_ring_case():
    import attention_context_parallel as cp_attention

    _assert_cp_result(
        cp_attention,
        cp_attention.CPStrategy.RING,
        cp_attention.ring_stripe_size,
    )


@requires_cp
def test_multi_gpu_context_parallel_allgather_case():
    import attention_context_parallel as cp_attention

    _assert_cp_result(
        cp_attention,
        cp_attention.CPStrategy.ALL_GATHER,
        cp_attention.ag_stripe_size,
    )


@requires_cp
def test_multi_gpu_context_parallel_benchmarks():
    import attention_context_parallel as cp_attention

    single_gpu_ms = cp_attention.run_single_gpu_bench()
    cp_attention.run_context_parallel_bench(
        cp_attention.CPStrategy.RING,
        cp_attention.ring_stripe_size,
        single_gpu_ms,
    )
    cp_attention.run_context_parallel_bench(
        cp_attention.CPStrategy.ALL_GATHER,
        cp_attention.ag_stripe_size,
        single_gpu_ms,
    )
