# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX: Dense GEMMs with TransformerEngine.

Companion source for ``dense.rst``. Code blocks between ``# DENSE_*_START`` /
``# DENSE_*_END`` markers are pulled into the RST via ``literalinclude``.

Run as a pytest module to exercise the example end-to-end:

    pytest -v docs/examples/jax_examples/dense.py

The multi-GPU section auto-skips when fewer than 4 GPUs are visible.
"""

# DENSE_IMPORTS_START
import sys

sys.path.append("..")  # so we can import quickstart_jax_utils from docs/examples/

import jax
import jax.numpy as jnp
from flax import linen as nn

import quickstart_jax_utils as utils
# DENSE_IMPORTS_END


# DENSE_BASELINE_MODEL_START
class FlaxDenseBlock(nn.Module):
    """One linear layer. ``dot_general_cls`` lets us swap the GEMM impl."""

    features: int
    dot_general_cls: callable = lambda: None

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            features=self.features,
            use_bias=False,
            dot_general=self.dot_general_cls(),
        )(x)
# DENSE_BASELINE_MODEL_END


# DENSE_INPUTS_SETUP_START
batch, seq, hidden, out_features = 4, 2048, 4096, 16384
dtype = jnp.bfloat16

key = jax.random.PRNGKey(0)
k_init, k_x, k_dy = jax.random.split(key, 3)
x = jax.random.normal(k_x, (batch, seq, hidden)).astype(dtype)
dy = jax.random.normal(k_dy, (batch, seq, out_features)).astype(dtype)

baseline = FlaxDenseBlock(features=out_features)
baseline_vars = baseline.init(k_init, x)
# DENSE_INPUTS_SETUP_END


# DENSE_TE_SETUP_START
from transformer_engine.jax import flax as te_flax
from transformer_engine.common.recipe import MXFP8BlockScaling

recipe = MXFP8BlockScaling()
te_dot_general_cls = te_flax.make_dot_general_cls(recipe)

te_model = FlaxDenseBlock(features=out_features, dot_general_cls=te_dot_general_cls)
te_vars = te_model.init(k_init, x)

print("Variable collections:", list(te_vars.keys()))
print(jax.tree_util.tree_map(lambda a: (a.shape, a.dtype), te_vars))
# DENSE_TE_SETUP_END


# DENSE_SINGLE_GPU_BENCH_START
def run_single_gpu_bench():
    print("bf16 baseline:")
    utils.speedometer(
        model_apply_fn=baseline.apply,
        variables=baseline_vars,
        input=x,
        output_grad=dy,
    )

    print(f"\nTE {type(recipe).__name__}:")
    utils.speedometer(
        model_apply_fn=te_model.apply,
        variables=te_vars,
        input=x,
        output_grad=dy,
    )
# DENSE_SINGLE_GPU_BENCH_END


# DENSE_MULTI_GPU_MESH_SETUP_START
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental import mesh_utils
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


def build_dp_tp_mesh():
    # 2x2 mesh: DP on one axis, TP on the other.
    devices = mesh_utils.create_device_mesh((2, 2))
    mesh = Mesh(devices, axis_names=("dp", "tp"))

    # Tell TE which mesh axis is which. This is a *global* setting, established
    # outside JIT, so TE's GEMM primitives can plan comms accordingly.
    mesh_resource = MeshResource(dp_resource="dp", tp_resource="tp")
    return mesh, mesh_resource
# DENSE_MULTI_GPU_MESH_SETUP_END


# DENSE_MULTI_GPU_SHARD_SETUP_START
def shard_variables(mesh, variables_dict):
    kernel_sharding = NamedSharding(mesh, P(None, "tp"))

    def _shard(variables):
        params = variables["params"]
        sharded = jax.device_put(params["Dense_0"]["kernel"], kernel_sharding)
        return {
            **variables,
            "params": {
                **params,
                "Dense_0": {**params["Dense_0"], "kernel": sharded},
            },
        }

    input_sharding = NamedSharding(mesh, P("dp", None, None))
    output_grad_sharding = NamedSharding(mesh, P("dp", None, "tp"))

    return {
        "x": jax.device_put(x, input_sharding),
        "dy": jax.device_put(dy, output_grad_sharding),
        **{name: _shard(vars_) for name, vars_ in variables_dict.items()},
    }
# DENSE_MULTI_GPU_SHARD_SETUP_END


# DENSE_MULTI_GPU_BENCH_START
def run_multi_gpu_bench():
    mesh, mesh_resource = build_dp_tp_mesh()
    sharded = shard_variables(mesh, {"baseline": baseline_vars, "te": te_vars})

    with jax.set_mesh(mesh), global_shard_guard(mesh_resource):
        print("bf16 DP=2/TP=2:")
        utils.speedometer(
            model_apply_fn=baseline.apply,
            variables=sharded["baseline"],
            input=sharded["x"],
            output_grad=sharded["dy"],
        )

        print(f"\nTE {type(recipe).__name__} DP=2/TP=2:")
        utils.speedometer(
            model_apply_fn=te_model.apply,
            variables=sharded["te"],
            input=sharded["x"],
            output_grad=sharded["dy"],
        )
# DENSE_MULTI_GPU_BENCH_END


# -----------------------------------------------------------------------------
# Pytest entry points (not pulled into docs).
#
# These run the same code shown in the snippets above and add numeric / smoke
# assertions so CI catches regressions.
# -----------------------------------------------------------------------------

import pytest
from transformer_engine.jax.quantize import is_scaling_mode_supported, ScalingMode

_mxfp8_supported, _mxfp8_reason = is_scaling_mode_supported(ScalingMode.MXFP8_1D_SCALING)
requires_mxfp8 = pytest.mark.skipif(
    not _mxfp8_supported, reason=f"MXFP8 not supported on this device: {_mxfp8_reason}"
)


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
    diffs = utils.compare_fwd_bwd(
        baseline.apply,
        baseline_vars,
        te_model.apply,
        te_vars,
        input=x,
        output_grad=dy,
    )
    # MXFP8 quantizes activations / weights, so we accept noticeable rel diff vs bf16.
    # Tune these in follow-ups once we have real CI numbers.
    assert diffs["y"]["max_rel"] < 0.20, diffs
    assert diffs["dx"]["max_rel"] < 0.20, diffs
    assert diffs["dW"]["max_rel"] < 0.30, diffs


@requires_mxfp8
def test_single_gpu_benchmark():
    run_single_gpu_bench()


@requires_mxfp8
@pytest.mark.skipif(len(jax.devices()) < 4, reason="needs 4 GPUs for DP=2/TP=2")
def test_multi_gpu_benchmark():
    run_multi_gpu_bench()


if __name__ == "__main__":
    run_single_gpu_bench()
    if len(jax.devices()) >= 4:
        print()
        run_multi_gpu_bench()
    else:
        print("\n[skipped multi-GPU section: <4 devices visible]")
