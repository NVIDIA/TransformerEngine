# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""JAX: BF16 Mixture-of-Experts with TransformerEngine.

Companion source for ``moe.rst``. Code blocks between ``# MOE_*_START`` /
``# MOE_*_END`` markers are pulled into the RST via ``literalinclude``.

Run as a script to exercise the example end-to-end:

    python docs/examples/jax/moe.py

The example uses a 2x2 expert-parallel/FSDP mesh and therefore requires four
visible GPUs. Both the native baseline and TransformerEngine path run in BF16;
the current ``_MoEBlock`` wrapper uses no-op quantizer sets.
"""

# MOE_IMPORTS_START
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from flax.linen import partitioning as nn_partitioning
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from moe_native import NativeMoEBlock

# MOE_IMPORTS_END


# MOE_CONFIG_START
EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
FSDP_SIZE = 2

NUM_EXPERTS = 8
TOPK = 2
BATCH = 8
SEQ = 2048
HIDDEN = 1024
INTERMEDIATE = 4096
DTYPE = jnp.bfloat16

LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
    ("batch", (EP_AXIS, FSDP_AXIS)),
)
# MOE_CONFIG_END


@dataclass
class DemoState:
    mesh: Mesh
    mesh_resource: Any
    native_model: NativeMoEBlock
    te_model: Any
    variables: Any
    x: jax.Array
    dy: jax.Array


def _ensure_writable_triton_cache():
    import os
    import tempfile

    os.environ.setdefault(
        "TRITON_CACHE_DIR",
        os.path.join(tempfile.gettempdir(), "transformer_engine_triton_cache"),
    )


# MOE_MESH_SETUP_START
def build_ep_fsdp_mesh():
    from transformer_engine.jax.sharding import MeshResource

    required_devices = EP_SIZE * FSDP_SIZE
    if len(jax.devices()) < required_devices:
        raise RuntimeError(
            f"MoE tutorial requires {required_devices} GPUs; "
            f"only {len(jax.devices())} visible"
        )

    devices = mesh_utils.create_device_mesh(
        (EP_SIZE, FSDP_SIZE),
        devices=jax.devices()[:required_devices],
    )
    mesh = Mesh(devices, axis_names=(EP_AXIS, FSDP_AXIS))
    mesh_resource = MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    return mesh, mesh_resource


# MOE_MESH_SETUP_END


# MOE_MODEL_SETUP_START
def build_models(mesh, *, hidden=HIDDEN, intermediate=INTERMEDIATE):
    _ensure_writable_triton_cache()

    from transformer_engine.jax.flax import _MoEBlock as TEMoEBlock
    from transformer_engine.jax.moe import PermutationBackend

    native_model = NativeMoEBlock(
        mesh=mesh,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=intermediate,
        ep_axis=EP_AXIS,
        data_parallelism_axes=(FSDP_AXIS,),
        dtype=DTYPE,
    )
    te_model = TEMoEBlock(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=intermediate,
        data_parallelism_axes=(FSDP_AXIS,),
        permutation_backend=PermutationBackend.PURE_JAX,
        dtype=DTYPE,
    )
    return native_model, te_model


# MOE_MODEL_SETUP_END


# MOE_INPUTS_SETUP_START
def make_inputs(*, batch=BATCH, seq=SEQ, hidden=HIDDEN):
    key = jax.random.PRNGKey(0)
    k_init, k_x, k_dy = jax.random.split(key, 3)
    x = jax.random.normal(k_x, (batch, seq, hidden), dtype=DTYPE)
    dy = jax.random.normal(k_dy, (batch, seq, hidden), dtype=DTYPE)
    return k_init, x, dy


def shard_inputs_and_variables(mesh, variables, x, dy):
    input_sharding = NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
    gate_sharding = NamedSharding(mesh, P())
    expert_sharding = NamedSharding(mesh, P(EP_AXIS, None, None))

    params = variables["params"]
    sharded_params = {
        "gate_kernel": jax.device_put(params["gate_kernel"], gate_sharding),
        "wi_0": jax.device_put(params["wi_0"], expert_sharding),
        "wi_1": jax.device_put(params["wi_1"], expert_sharding),
        "wo": jax.device_put(params["wo"], expert_sharding),
    }
    return {
        "variables": {**variables, "params": sharded_params},
        "x": jax.device_put(x, input_sharding),
        "dy": jax.device_put(dy, input_sharding),
    }


# MOE_INPUTS_SETUP_END


def _te_apply(te_model):
    def apply_fn(variables, x, **kwargs):
        out, _ = te_model.apply(variables, x, **kwargs)
        return out

    return apply_fn


def setup_demo(*, batch=BATCH, seq=SEQ, hidden=HIDDEN, intermediate=INTERMEDIATE):
    from transformer_engine.jax.sharding import global_shard_guard

    mesh, mesh_resource = build_ep_fsdp_mesh()
    native_model, te_model = build_models(mesh, hidden=hidden, intermediate=intermediate)
    k_init, x, dy = make_inputs(batch=batch, seq=seq, hidden=hidden)

    with jax.set_mesh(mesh), global_shard_guard(mesh_resource), nn_partitioning.axis_rules(
        LOGICAL_AXIS_RULES
    ):
        variables = jax.jit(native_model.init)(k_init, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(variables)[0])

    sharded = shard_inputs_and_variables(mesh, variables, x, dy)
    return DemoState(
        mesh=mesh,
        mesh_resource=mesh_resource,
        native_model=native_model,
        te_model=te_model,
        variables=sharded["variables"],
        x=sharded["x"],
        dy=sharded["dy"],
    )


def te_moe_supported():
    try:
        import importlib
        import sys

        _ensure_writable_triton_cache()

        import transformer_engine.jax  # noqa: F401

        transformer_engine_jax = sys.modules["transformer_engine_jax"]
        flax_mod = importlib.import_module("transformer_engine.jax.flax")
        getattr(flax_mod, "_MoEBlock")
        if transformer_engine_jax.get_device_compute_capability(0) < 100:
            return False, "TE MoE grouped GEMM currently requires Blackwell (sm_100+)"
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return False, str(exc)
    return True, ""


# MOE_CORRECTNESS_START
def compare_forward(demo):
    from transformer_engine.jax.sharding import global_shard_guard

    te_apply = _te_apply(demo.te_model)
    with jax.set_mesh(demo.mesh), global_shard_guard(
        demo.mesh_resource
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        native_out = jax.jit(demo.native_model.apply)(demo.variables, demo.x)
        te_out = jax.jit(te_apply)(demo.variables, demo.x)
        native_out, te_out = jax.block_until_ready((native_out, te_out))

    max_abs = jnp.max(
        jnp.abs(native_out.astype(jnp.float32) - te_out.astype(jnp.float32))
    )
    print(f"max |native BF16 - TE BF16|: {float(max_abs):.4f}")
    return native_out, te_out


# MOE_CORRECTNESS_END


# MOE_BENCH_START
def _block_until_ready_tree(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if leaves:
        jax.block_until_ready(leaves[0])


def _time_fwd_bwd(apply_fn, demo, *, warmup_iters, timing_iters):
    import time

    autocast_kwargs = {"enabled": False, "mesh_resource": demo.mesh_resource}

    def loss_fn(variables, inp, grad_target):
        import transformer_engine.jax as te

        with te.autocast(**autocast_kwargs):
            out = apply_fn(variables, inp)
        return jnp.vdot(out, grad_target)

    train_step = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1)))

    for _ in range(warmup_iters):
        _block_until_ready_tree(train_step(demo.variables, demo.x, demo.dy))

    start = time.perf_counter()
    for _ in range(timing_iters):
        _block_until_ready_tree(train_step(demo.variables, demo.x, demo.dy))
    return (time.perf_counter() - start) * 1000.0 / timing_iters


def run_benchmarks(demo, *, warmup_iters=5, timing_iters=10):
    from transformer_engine.jax.sharding import global_shard_guard

    te_apply = _te_apply(demo.te_model)
    with jax.set_mesh(demo.mesh), global_shard_guard(
        demo.mesh_resource
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        print("native JAX BF16:")
        native_ms = _time_fwd_bwd(
            demo.native_model.apply,
            demo,
            warmup_iters=warmup_iters,
            timing_iters=timing_iters,
        )
        print(f"Mean time: {native_ms:.3f} ms")

        print("\nTE _MoEBlock BF16:")
        te_ms = _time_fwd_bwd(
            te_apply,
            demo,
            warmup_iters=warmup_iters,
            timing_iters=timing_iters,
        )
        print(f"Mean time: {te_ms:.3f} ms")
    return native_ms, te_ms


# MOE_BENCH_END


def main():
    if len(jax.devices()) < EP_SIZE * FSDP_SIZE:
        print(f"[skipped: need {EP_SIZE * FSDP_SIZE} GPUs for EP=2/FSDP=2]")
        return

    te_supported, te_reason = te_moe_supported()
    if not te_supported:
        print(f"[skipped TE comparison: {te_reason}]")
        return

    demo = setup_demo()
    compare_forward(demo)
    run_benchmarks(demo)


if __name__ == "__main__":
    main()
