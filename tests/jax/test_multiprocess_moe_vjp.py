# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-process (one-GPU-per-process) tests for the unified MoE custom_vjp.

The launcher ``tests/jax/run_multiprocess_moe_vjp.sh`` forks one pytest
process per visible GPU (mirroring
``examples/jax/encoder/run_test_multiprocessing_encoder.sh``). Each
process binds to exactly one device via
``jax.distributed.initialize(..., local_device_ids=process_id)``; the
participating processes form a global mesh through JAX's distributed
runtime.

How to run
----------

You typically do NOT invoke pytest on this file directly -- use the
launcher, which passes ``--num-process=N --process-id=i`` to each
forked process. Driving it directly with only one process will skip
every test because :func:`jax.distributed.initialize` requires
multiple participants.

    bash tests/jax/run_multiprocess_moe_vjp.sh

CI invocation lives in ``qa/L0_jax_distributed_unittest/test.sh``.
"""

import os

# NCCL needs HBM headroom that JAX's default 90% preallocation does
# not leave. Set before any jax import below.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


# Per-process distributed bootstrap. Each pytest invocation initializes
# JAX with exactly one local device (its assigned GPU). Once
# initialized, the four processes form one global mesh of 4 devices.
def _init_distributed(num_process: int, process_id: int) -> bool:
    """Initialize jax.distributed for this pytest process.

    Returns True if initialization succeeded (i.e. this is a real
    multi-process launch), False if num_process == 0 / 1 meaning the
    file is being collected without a launcher and tests should be
    skipped at module level.
    """
    if num_process <= 1:
        return False
    coord = os.environ.get("MOE_VJP_COORDINATOR_ADDRESS", "127.0.0.1:1234")
    jax.distributed.initialize(
        coordinator_address=coord,
        num_processes=num_process,
        process_id=process_id,
        local_device_ids=process_id,
    )
    assert jax.local_device_count() == 1, "one GPU per process is the whole point"
    assert (
        jax.device_count() == num_process
    ), f"global device_count {jax.device_count()} != num_process {num_process}"
    return True


# Read --num-process / --process-id BEFORE pytest collects any tests so
# we can fast-skip the whole module when not in a multiprocess launch.
def _read_mp_options():
    # Use pytest's option lookup via the request fixture isn't available
    # at module top-level; parse argv ourselves the same way encoder
    # test does. CLI form is e.g. "pytest ... --num-process=4 --process-id=0".
    num = int(os.environ.get("MP_NUM_PROCESS", "0") or "0")
    pid = int(os.environ.get("MP_PROCESS_ID", "0") or "0")
    for i, a in enumerate(sys.argv):
        if a.startswith("--num-process="):
            num = int(a.split("=", 1)[1])
        elif a == "--num-process" and i + 1 < len(sys.argv):
            num = int(sys.argv[i + 1])
        elif a.startswith("--process-id="):
            pid = int(a.split("=", 1)[1])
        elif a == "--process-id" and i + 1 < len(sys.argv):
            pid = int(sys.argv[i + 1])
    return num, pid


_MP_NUM_PROCESS, _MP_PROCESS_ID = _read_mp_options()
_MP_ACTIVE = _init_distributed(_MP_NUM_PROCESS, _MP_PROCESS_ID)

if not _MP_ACTIVE:
    # Skip the entire module if not launched via the multiprocess
    # runner. Lets `pytest tests/jax/` collect this file harmlessly.
    pytest.skip(
        "test_multiprocess_moe_vjp.py requires the multiprocess launcher "
        "(run_multiprocess_moe_vjp.sh). Skipping.",
        allow_module_level=True,
    )


NUM_DEVICES_REQUIRED = 4
EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
FSDP_SIZE = 2

LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
    ("batch", (EP_AXIS, FSDP_AXIS)),
)


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < NUM_DEVICES_REQUIRED:
        pytest.skip(
            f"Need >={NUM_DEVICES_REQUIRED} devices for ep={EP_SIZE} x fsdp={FSDP_SIZE};"
            f" have {jax.device_count()}"
        )
    devices = mesh_utils.create_device_mesh((EP_SIZE, FSDP_SIZE))
    return Mesh(devices, axis_names=(EP_AXIS, FSDP_AXIS))


@pytest.fixture(autouse=True, scope="function")
def _inject_moe(request):
    if not request.node.get_closest_marker("triton"):
        yield
        return
    import transformer_engine.jax as te
    from transformer_engine.common import recipe as te_recipe
    from transformer_engine.jax.flax import _MoEBlock as MoEBlock
    from transformer_engine.jax.moe import PermutationBackend
    from transformer_engine.jax.sharding import MeshResource, global_shard_guard

    mod = sys.modules[__name__]
    mod.te = te
    mod.te_recipe = te_recipe
    mod.MoEBlock = MoEBlock
    mod.PermutationBackend = PermutationBackend
    mod.MeshResource = MeshResource
    mod.global_shard_guard = global_shard_guard
    yield


# ``recipe`` parametrize values used across all tests below. ``None``
# = plain bf16; the named recipes route through TE's autocast and
# exercise the FP8/MXFP8 quantization paths in _body_fwd/_body_bwd.
# Only recipes that work on TE Blackwell are included; older GPUs
# skip via the ``hardware_supports`` guard below.
RECIPE_NAMES = ("bf16", "MXFP8BlockScaling")


def _resolve_recipe(name):
    """Return ``(use_fp8, recipe_instance)`` for the parametrize id."""
    if name == "bf16":
        return False, None
    if name == "MXFP8BlockScaling":
        return True, te_recipe.MXFP8BlockScaling()  # noqa: F821
    raise ValueError(f"unknown recipe name: {name!r}")


def _hardware_supports(recipe_name):
    """Skip an FP8 recipe on GPUs that don't have the hw for it."""
    if recipe_name == "bf16":
        return True
    from transformer_engine_jax import get_device_compute_capability

    arch = get_device_compute_capability(0)
    if recipe_name == "MXFP8BlockScaling":
        return arch >= 100
    return False


def _autocast_ctx(recipe_name):
    """Context manager that turns FP8 on for non-bf16 recipes."""
    use_fp8, recipe_inst = _resolve_recipe(recipe_name)
    return te.autocast(enabled=use_fp8, recipe=recipe_inst)  # noqa: F821


def _tol_finite_grad(recipe_name):
    """Per-recipe absolute tolerance for parity grad comparison."""
    if recipe_name == "bf16":
        return 5e-2
    # MXFP8 grads carry block-scale quantization noise; loosen accordingly.
    return 3e-1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_block(
    *,
    num_experts,
    num_experts_per_tok,
    intermediate_size,
    permutation_backend,
    aux_loss_coeff=0.0,
    dtype=jnp.bfloat16,
    align_size=0,
):
    return MoEBlock(  # noqa: F821
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        intermediate_size=intermediate_size,
        permutation_backend=permutation_backend,
        data_parallelism_axes=(FSDP_AXIS,),
        aux_loss_coeff=aux_loss_coeff,
        dtype=dtype,
        _align_size=align_size,
    )


def _shard_inputs(x, mesh):
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P((EP_AXIS, FSDP_AXIS), None, None))
    )


def _init_apply(block, mesh, x, key):
    with mesh, global_shard_guard(  # noqa: F821
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x = _shard_inputs(x, mesh)
        variables = jax.jit(block.init)(key, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(variables)[0])
        output, aux = jax.jit(block.apply)(variables, x)
        jax.block_until_ready(output)
    return variables, output, aux


def _grad_step(block, variables, mesh, x):
    with mesh, global_shard_guard(  # noqa: F821
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
    ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
        x = _shard_inputs(x, mesh)

        def loss_fn(variables, x):
            output, aux = block.apply(variables, x)
            main = jnp.mean(output.astype(jnp.float32) ** 2)
            return main + (aux.astype(jnp.float32) if aux is not None else 0.0)

        grads = jax.jit(jax.grad(loss_fn))(variables, x)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
        return grads


def _unwrap(x):
    return x.value if hasattr(x, "value") else x


def _local_shard(x):
    """Return the local (this-process) shard of a global JAX Array as numpy.

    Every assertion in this file is structural (finite-ness, non-zero,
    parity within tolerance). For all of these, checking the local
    shard on each process is sufficient and avoids any cross-process
    collective in the test machinery. ``arr.addressable_data(0)``
    returns the local-device view of the sharded array -- with one
    GPU per process there is exactly one addressable shard.
    """
    return np.asarray(jax.device_get(x.addressable_data(0)))


# -----------------------------------------------------------------------------
# Mixtral-style shapes, sized to fit on a single 4-GPU bf16 box (a
# 4-way data-parallel shard of a Mixtral-8 block).
# -----------------------------------------------------------------------------

BATCH = EP_SIZE * FSDP_SIZE * 4  # 16
SEQ = 2048
HIDDEN = 1024
INTER = 4096
NUM_EXPERTS = 8
TOPK = 2


@pytest.mark.triton
class TestMoeVjpMultiprocess:
    """Multiprocess (one-GPU-per-process) correctness checks for the
    unified MoE custom_vjp.
    """

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    @pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
    def test_fwd_and_bwd(self, mesh, backend_name, recipe_name):
        if not _hardware_supports(recipe_name):
            pytest.skip(f"recipe {recipe_name} not supported on this GPU")
        backend = PermutationBackend(backend_name)  # noqa: F821
        block = _make_block(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            intermediate_size=INTER,
            permutation_backend=backend,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(0),
            (BATCH, SEQ, HIDDEN),
            dtype=jnp.bfloat16,
        )
        with _autocast_ctx(recipe_name):
            variables, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(1))
        # Local-shard checks (see _local_shard docstring for why).
        out_local = _local_shard(output)
        assert output.dtype == x.dtype
        assert np.all(np.isfinite(out_local)), "output has NaN/Inf"
        assert aux is None
        with _autocast_ctx(recipe_name):
            grads = _grad_step(block, variables, mesh, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_local = _local_shard(_unwrap(grads["params"][name]))
            assert np.all(np.isfinite(g_local)), f"{name} grad has NaN/Inf"
            assert np.any(g_local != 0.0), f"{name} grad is identically zero"

    @pytest.mark.parametrize("backend_name", ["pure_jax", "triton"])
    @pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
    def test_aux_loss(self, mesh, backend_name, recipe_name):
        if not _hardware_supports(recipe_name):
            pytest.skip(f"recipe {recipe_name} not supported on this GPU")
        backend = PermutationBackend(backend_name)  # noqa: F821
        block = _make_block(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            intermediate_size=INTER,
            permutation_backend=backend,
            aux_loss_coeff=1e-2,
        )
        x = jax.random.normal(
            jax.random.PRNGKey(4),
            (BATCH, SEQ, HIDDEN),
            dtype=jnp.bfloat16,
        )
        with _autocast_ctx(recipe_name):
            variables, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(5))
        out_local = _local_shard(output)
        assert np.all(np.isfinite(out_local)), "output has NaN/Inf under aux"
        assert aux is not None
        assert aux.shape == ()
        aux_local = _local_shard(aux)
        assert np.isfinite(aux_local), "aux is NaN/Inf"
        with _autocast_ctx(recipe_name):
            grads = _grad_step(block, variables, mesh, x)
        g_gate_local = _local_shard(_unwrap(grads["params"]["gate_kernel"]))
        assert np.all(np.isfinite(g_gate_local)), "gate grad NaN/Inf under aux"

    @pytest.mark.parametrize("recipe_name", RECIPE_NAMES)
    def test_pure_jax_triton_parity(self, mesh, recipe_name):
        if not _hardware_supports(recipe_name):
            pytest.skip(f"recipe {recipe_name} not supported on this GPU")
        block_pj = _make_block(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            intermediate_size=INTER,
            permutation_backend=PermutationBackend.PURE_JAX,  # noqa: F821
        )
        block_tr = _make_block(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            intermediate_size=INTER,
            permutation_backend=PermutationBackend.TRITON,  # noqa: F821
        )
        x = jax.random.normal(
            jax.random.PRNGKey(6),
            (BATCH, SEQ, HIDDEN),
            dtype=jnp.bfloat16,
        )
        tol = _tol_finite_grad(recipe_name)
        with _autocast_ctx(recipe_name):
            variables, out_pj, _ = _init_apply(block_pj, mesh, x, jax.random.PRNGKey(7))
            with mesh, global_shard_guard(  # noqa: F821
                MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)  # noqa: F821
            ), nn_partitioning.axis_rules(LOGICAL_AXIS_RULES):
                x_sh = _shard_inputs(x, mesh)
                out_tr, _ = jax.jit(block_tr.apply)(variables, x_sh)

        out_pj_local = _local_shard(out_pj)
        out_tr_local = _local_shard(out_tr)
        diff = float(np.max(np.abs(out_pj_local - out_tr_local)))
        assert diff < tol, f"forward parity breach: max_abs_diff={diff} (tol={tol})"

        with _autocast_ctx(recipe_name):
            grads_pj = _grad_step(block_pj, variables, mesh, x)
            grads_tr = _grad_step(block_tr, variables, mesh, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_pj = _local_shard(_unwrap(grads_pj["params"][name]))
            g_tr = _local_shard(_unwrap(grads_tr["params"][name]))
            d = float(np.max(np.abs(g_pj - g_tr)))
            assert d < tol, f"grad parity breach on {name}: max_abs_diff={d} (tol={tol})"
