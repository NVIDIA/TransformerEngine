# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest entry points for the JAX attention tutorials."""

import jax
import numpy as np
import pytest

import attention
import attention_context_parallel as cp_attention


def test_bshd_gqa_swa_runs():
    out = attention.te_model.apply(
        attention.te_vars,
        attention.qkv,
        sequence_descriptor=attention.sequence_descriptor,
        deterministic=False,
    )

    assert out.shape == attention.dout.shape
    assert out.dtype == attention.dtype


def test_bshd_gqa_swa_matches_baseline():
    attention.compare_te_to_baseline()


def test_single_gpu_benchmark():
    attention.run_single_gpu_bench()


def test_mla_variant_runs():
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


_cp_supported, _cp_reason = cp_attention.context_parallel_supported()
requires_cp = pytest.mark.skipif(
    not _cp_supported,
    reason=f"context-parallel attention tutorial skipped: {_cp_reason}",
)


def _assert_cp_result(strategy, stripe_size):
    result = cp_attention.run_context_parallel_case(strategy, stripe_size)
    reference = cp_attention.run_reference_attention()

    assert result["output"].shape == (
        cp_attention.batch,
        cp_attention.seq,
        cp_attention.num_attention_heads,
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
    _assert_cp_result(cp_attention.CPStrategy.RING, cp_attention.ring_stripe_size)


@requires_cp
def test_multi_gpu_context_parallel_allgather_case():
    _assert_cp_result(cp_attention.CPStrategy.ALL_GATHER, cp_attention.ag_stripe_size)


@requires_cp
def test_multi_gpu_context_parallel_benchmarks():
    cp_attention.run_context_parallel_bench(
        cp_attention.CPStrategy.RING,
        cp_attention.ring_stripe_size,
    )
    cp_attention.run_context_parallel_bench(
        cp_attention.CPStrategy.ALL_GATHER,
        cp_attention.ag_stripe_size,
    )
