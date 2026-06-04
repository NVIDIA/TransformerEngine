# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest entry points for ``moe.py``.

Run with:

    pytest -v docs/examples/jax/test_moe.py

The tutorial uses a 2x2 EP/FSDP mesh, so tests skip when fewer than four GPUs
are visible. TransformerEngine MoE tests also skip when the installed TE build
does not expose the experimental ``_MoEBlock`` or when hardware support is
missing.
"""

import importlib
import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest


requires_4gpu = pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 GPUs")


os.environ.setdefault(
    "TRITON_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "transformer_engine_triton_cache"),
)


def _te_moe_available():
    try:
        import transformer_engine.jax  # noqa: F401

        mod = importlib.import_module("transformer_engine.jax.flax")
        getattr(mod, "_MoEBlock")
        transformer_engine_jax = sys.modules["transformer_engine_jax"]

        if transformer_engine_jax.get_device_compute_capability(0) < 100:
            return False, "TE MoE grouped GEMM requires Blackwell (sm_100+)"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, str(exc)
    return True, ""


_te_supported, _te_reason = _te_moe_available()
requires_te_moe = pytest.mark.skipif(not _te_supported, reason=_te_reason)


def _small_native_state():
    from jax.experimental import mesh_utils
    from jax.sharding import Mesh
    from moe import EP_AXIS, FSDP_AXIS, NativeMoEBlock

    mesh = Mesh(
        mesh_utils.create_device_mesh((2, 2), devices=jax.devices()[:4]),
        (EP_AXIS, FSDP_AXIS),
    )
    model = NativeMoEBlock(
        mesh=mesh,
        num_experts=8,
        num_experts_per_tok=2,
        intermediate_size=64,
        ep_axis=EP_AXIS,
        data_parallelism_axes=(FSDP_AXIS,),
        dtype=jnp.bfloat16,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (4, 16, 32), dtype=jnp.bfloat16)
    dy = jax.random.normal(jax.random.PRNGKey(2), x.shape, dtype=jnp.bfloat16)
    return mesh, model, x, dy


@requires_4gpu
def test_native_baseline_runs():
    mesh, model, x, _ = _small_native_state()
    with jax.set_mesh(mesh):
        variables = jax.jit(model.init)(jax.random.PRNGKey(0), x)
        out = jax.jit(model.apply)(variables, x)
        out.block_until_ready()

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert np.all(np.isfinite(np.asarray(out)))


@requires_4gpu
def test_native_baseline_grads_are_finite():
    mesh, model, x, dy = _small_native_state()

    def loss_fn(variables, x):
        return jnp.vdot(model.apply(variables, x), dy)

    with jax.set_mesh(mesh):
        variables = jax.jit(model.init)(jax.random.PRNGKey(0), x)
        grads = jax.jit(jax.grad(loss_fn))(variables, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads))

    for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
        grad = np.asarray(grads["params"][name])
        assert np.all(np.isfinite(grad)), f"{name} grad has NaN/Inf"
        assert np.any(grad != 0.0), f"{name} grad is identically zero"


@requires_4gpu
@requires_te_moe
def test_te_moe_matches_native_shape_and_dtype():
    import moe

    demo = moe.setup_demo(batch=4, seq=16, hidden=32, intermediate=64)
    native_out, te_out = moe.compare_forward(demo)

    assert native_out.shape == te_out.shape == demo.x.shape
    assert native_out.dtype == te_out.dtype == demo.x.dtype
    assert np.all(np.isfinite(np.asarray(te_out)))


@requires_4gpu
@requires_te_moe
def test_benchmark_entrypoint_runs():
    import moe

    demo = moe.setup_demo(batch=4, seq=16, hidden=32, intermediate=64)
    moe.run_benchmarks(demo, warmup_iters=1, timing_iters=1)
