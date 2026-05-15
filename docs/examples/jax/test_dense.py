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

# Imports from ``dense`` are intentionally deferred into each test body. dense.py
# runs ``te_vars = te_model.init(k_init, x)`` at module scope, which raises on
# devices without MXFP8 support (Hopper or older). A top-level import would fire
# that before pytest can apply the @requires_mxfp8 skip marks.

_mxfp8_supported, _mxfp8_reason = is_scaling_mode_supported(ScalingMode.MXFP8_1D_SCALING)
requires_mxfp8 = pytest.mark.skipif(
    not _mxfp8_supported, reason=f"MXFP8 not supported on this device: {_mxfp8_reason}"
)


def test_baseline_runs():
    from dense import baseline, baseline_vars, batch, dtype, out_features, seq, x

    out = baseline.apply(baseline_vars, x)
    assert out.shape == (batch, seq, out_features)
    assert out.dtype == dtype


@requires_mxfp8
def test_te_dense_runs():
    from dense import batch, out_features, seq, te_model, te_vars, x

    out = te_model.apply(te_vars, x)
    assert out.shape == (batch, seq, out_features)


@requires_mxfp8
def test_te_matches_baseline():
    """TE quantized Dense should match the bf16 baseline within MXFP8 tolerance."""
    from dense import baseline, baseline_vars, batch, dy, seq, te_model, te_vars, x

    fp8_rel_noise = float(jnp.finfo(jnp.float8_e4m3fn).eps)
    atol_fwd = 10.0 * fp8_rel_noise
    atol_dw = atol_fwd * jnp.sqrt(batch * seq).item()

    utils.compare_fwd_bwd(
        baseline.apply,
        baseline_vars,
        te_model.apply,
        te_vars,
        input=x,
        output_grad=dy,
        rtol=fp8_rel_noise,
        atol=atol_fwd,
        rtol_dW=fp8_rel_noise,
        atol_dW=atol_dw,
    )


@requires_mxfp8
def test_single_gpu_benchmark():
    from dense import run_single_gpu_bench

    run_single_gpu_bench()


@requires_mxfp8
@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 GPUs for DP=2/TP=2")
def test_multi_gpu_benchmark():
    from dense import run_multi_gpu_bench

    run_multi_gpu_bench()
