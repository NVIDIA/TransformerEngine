# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX: Dense GEMMs with TransformerEngine.

Companion source for ``dense.rst``. Code blocks between ``# DENSE_*_START`` /
``# DENSE_*_END`` markers are pulled into the RST via ``literalinclude``.

Run as a script to exercise the example end-to-end:

    python docs/examples/jax/dense.py

Pytest tests live in ``test_dense.py``; the multi-GPU section auto-skips when
fewer than 4 GPUs are visible.
"""

# DENSE_IMPORTS_START
import jax
import jax.numpy as jnp
from flax import linen as nn

import quickstart_jax_utils as utils

# DENSE_IMPORTS_END


# DENSE_BASELINE_MODEL_START
class FlaxDenseBlock(nn.Module):
    """One linear layer. ``dot_general_cls`` lets us swap the GEMM impl."""

    features: int
    dtype: jnp.dtype = jnp.bfloat16
    dot_general_cls: callable = lambda: None

    @nn.compact
    def __call__(self, x):
        return nn.Dense(
            features=self.features,
            use_bias=False,
            dtype=self.dtype,
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


if __name__ == "__main__":
    run_single_gpu_bench()
    if len(jax.devices()) >= 4:
        print()
        run_multi_gpu_bench()
    else:
        print("\n[skipped multi-GPU section: <4 devices visible]")
