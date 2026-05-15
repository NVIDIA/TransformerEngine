# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest entry points for ``dense.py``.

These run the same code shown in ``dense.py`` and add numeric / smoke
assertions so CI catches regressions.

Run with:

    pytest -v docs/examples/jax/test_dense.py

The multi-GPU section auto-skips when fewer than 4 GPUs are visible.
"""

import jax
import jax.numpy as jnp
import pytest

import quickstart_jax_utils as utils
from transformer_engine.jax.quantize import is_scaling_mode_supported, ScalingMode

from dense import (
    baseline,
    baseline_vars,
    batch,
    dtype,
    dy,
    out_features,
    run_multi_gpu_bench,
    run_single_gpu_bench,
    seq,
    te_model,
    te_vars,
    x,
)

_mxfp8_supported, _mxfp8_reason = is_scaling_mode_supported(ScalingMode.MXFP8_1D_SCALING)
requires_mxfp8 = pytest.mark.skipif(
    not _mxfp8_supported, reason=f"MXFP8 not supported on this device: {_mxfp8_reason}"
)

# MXFP8 quantization noise is ~FP8 epsilon (~5%) of the per-tensor magnitude.
# ``atol`` covers near-zero ref values where the rtol fraction is too tight.
# y and dx are O(1) under Flax's lecun_normal init (in/out scaling cancels);
# dW has no init scaling and accumulates batch*seq products, so it grows as
# sqrt(batch*seq).
_FP8_REL_NOISE = float(jnp.finfo(jnp.float8_e4m3fn).eps)  # 0.125
_ATOL_FWD = 10.0 * _FP8_REL_NOISE                         # ~1.25; covers Gaussian-tail |y|, |dx|
_ATOL_DW = _ATOL_FWD * jnp.sqrt(batch * seq).item()       # ~113; covers Gaussian-tail |dW|


def test_baseline_runs():
    out = baseline.apply(baseline_vars, x)
    assert out.shape == (batch, seq, out_features)
    assert out.dtype == dtype


@requires_mxfp8
def test_te_dense_runs():
    out = te_model.apply(te_vars, x)
    assert out.shape == (batch, seq, out_features)


@requires_mxfp8
def test_te_matches_baseline():
    """TE quantized Dense should match the bf16 baseline within MXFP8 tolerance."""
    utils.compare_fwd_bwd(
        baseline.apply,
        baseline_vars,
        te_model.apply,
        te_vars,
        input=x,
        output_grad=dy,
        rtol=_FP8_REL_NOISE,
        atol=_ATOL_FWD,
        rtol_dW=_FP8_REL_NOISE,
        atol_dW=_ATOL_DW,
    )


@requires_mxfp8
def test_single_gpu_benchmark():
    run_single_gpu_bench()


@requires_mxfp8
@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 GPUs for DP=2/TP=2")
def test_multi_gpu_benchmark():
    run_multi_gpu_bench()
