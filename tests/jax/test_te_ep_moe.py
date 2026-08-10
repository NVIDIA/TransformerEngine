# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-process (one-GPU-per-process) tests for the TE-EP MoE custom_vjp.

The launcher ``tests/jax/run_te_ep_moe.sh`` forks one pytest process per
visible GPU. Each process binds to exactly one device via
``jax.distributed.initialize(..., local_device_ids=process_id)``; the
participating processes form a global ``(ep, fsdp)`` mesh through JAX's
distributed runtime.

How to run
----------

You typically do NOT invoke pytest on this file directly -- use the
launcher, which passes ``--num-process=N --process-id=i`` to each
forked process. Driving it directly with only one process will skip
every test because :func:`jax.distributed.initialize` requires
multiple participants, and the TE EP NCCL primitives require at
least four ranks.

    bash tests/jax/run_te_ep_moe.sh

What this suite covers
----------------------

Each test exercises one MoE-block run and bundles every check that
single run supports — shape, dtype,
finiteness AND numerical parity vs a pure-JAX reference. Variations
on the block are pytest parametrize values rather than separate test
classes:

* ``test_forward`` covers the forward across a curated set of
  configurations (softmax/sigmoid scoring, optional non-zero
  expert_bias). Each config asserts shape, dtype, finiteness and
  numerical parity vs the reference in one run.
* ``test_backward`` mirrors that for gradients.
* ``TestTeEpMoeAuxLoss`` covers the second return value end-to-end
  (returned + parity + aux-only grad propagates to gate + combined
  main+aux grads stay finite) in two consolidated tests.
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")

import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from flax.linen import partitioning as nn_partitioning


def _init_distributed(num_process: int, process_id: int) -> bool:
    """Initialize jax.distributed for this pytest process.

    Returns True on a real multi-process launch, False otherwise so
    the module can fast-skip when pytest collects it without the
    launcher.
    """
    if num_process <= 1:
        return False
    coord = os.environ.get("TE_EP_MOE_COORDINATOR_ADDRESS", "127.0.0.1:13457")
    jax.distributed.initialize(
        coordinator_address=coord,
        num_processes=num_process,
        process_id=process_id,
        local_device_ids=process_id,
    )
    assert jax.local_device_count() == 1, "one GPU per process is required for TE EP"
    assert (
        jax.device_count() == num_process
    ), f"global device_count {jax.device_count()} != num_process {num_process}"
    return True


def _read_mp_options():
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
    pytest.skip(
        "test_te_ep_moe.py requires the multiprocess launcher (run_te_ep_moe.sh). Skipping.",
        allow_module_level=True,
    )

from transformer_engine_jax import get_device_compute_capability

# Grouped GEMM in the MoE custom_vjp requires Blackwell (sm_100+). The
# TE EP NCCL primitives themselves need SM>=90, but the FFN body uses
# grouped_gemm, so the file as a whole gates on sm_100+.
if get_device_compute_capability(0) < 100:
    pytest.skip(
        "MoE TE EP tests require Blackwell (sm_100+) for grouped GEMM",
        allow_module_level=True,
    )

from transformer_engine.jax.flax import _MoEBlock as MoEBlock
from transformer_engine.jax.moe import (
    _ALIGN_SIZE,
    get_moe_recv_capacity_per_rank,
    moe,
    record_ep_bootstrap_signature_for_moe,
)
from transformer_engine.jax.ep import ep_bootstrap, ep_finalize
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


# -----------------------------------------------------------------------------
# Mesh / shape config
# -----------------------------------------------------------------------------

EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = int(os.environ.get("TE_EP_MOE_EP_SIZE", "2"))
assert EP_SIZE in (2, 4), f"TE_EP_MOE_EP_SIZE must be 2 or 4, got {EP_SIZE}"
assert (
    jax.device_count() % EP_SIZE == 0
), f"device_count {jax.device_count()} must be divisible by EP_SIZE={EP_SIZE}"
FSDP_SIZE = jax.device_count() // EP_SIZE
NUM_DEVICES_REQUIRED = EP_SIZE * FSDP_SIZE

LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    # Match MaxText's converging layout: FSDP is the outer component of
    # the compound expert dimension and EP is inner.
    ("exp_fsdp", (FSDP_AXIS, EP_AXIS)),
    ("embed", FSDP_AXIS),
    ("embed_replicated", None),
    ("mlp", None),
    ("batch", (FSDP_AXIS, EP_AXIS)),
)

# Small shapes so the parity tests stay tight on bf16. The block still
# has all four ranks participating in dispatch/combine.
DTYPE = jnp.bfloat16
BATCH = EP_SIZE * FSDP_SIZE * 2  # 8 on 4-GPU, 16 on 8-GPU
SEQ = 32
HIDDEN = 64
INTER = 128
NUM_EXPERTS = 8
TOPK = 2

# bf16 grouped_gemm + softmax-topk + ep all-to-all stack drifts ~1e-1 vs a
# fp32 numpy reference. Keep these tight enough to catch real bugs but
# loose enough to absorb expected bf16 rounding.
FWD_ATOL = 5e-2
FWD_RTOL = 5e-2
GRAD_FFN_ATOL = 1e-1
GRAD_FFN_RTOL = 1e-1
GRAD_GATE_ATOL = 5e-1
GRAD_GATE_RTOL = 5e-1

# Two TE EP runs that should be bitwise-equal modulo XLA fusion order
# (slot alignment rounding, etc.).
TE_TO_TE_ATOL = 5e-3
TE_TO_TE_RTOL = 5e-3
TE_TO_TE_GRAD_NORM_RATIO = (0.98, 1.02)
TE_TO_TE_GRAD_COSINE = 0.999


def _assert_gradient_direction_and_scale(actual, expected, *, name):
    """Catch permutations/reduction factors hidden by bf16 absolute tolerances."""
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual_norm = np.linalg.norm(actual)
    expected_norm = np.linalg.norm(expected)
    assert actual_norm > 0.0 and expected_norm > 0.0, f"{name}: zero gradient norm"
    norm_ratio = actual_norm / expected_norm
    cosine = np.dot(actual, expected) / (actual_norm * expected_norm)
    assert 0.8 <= norm_ratio <= 1.2, (
        f"{name}: gradient norm ratio {norm_ratio:.6f} outside [0.8, 1.2] "
        f"(cosine={cosine:.6f})"
    )
    assert cosine >= 0.98, f"{name}: gradient cosine similarity {cosine:.6f} < 0.98"


# Aux loss is computed in float32 from the SAME logits as the routing
# path. Numerical drift between TE-EP and the reference is dominated by
# the bf16-rounded softmax inside the topk kernel.
AUX_ATOL = 1e-3
AUX_RTOL = 1e-3


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh():
    if jax.device_count() < NUM_DEVICES_REQUIRED:
        pytest.skip(
            f"Need >={NUM_DEVICES_REQUIRED} devices for ep={EP_SIZE} x fsdp={FSDP_SIZE};"
            f" have {jax.device_count()}"
        )
    # ``ep`` must be the inner axis: ``ep_bootstrap`` forms NCCL EP groups
    # from consecutive global ranks via ``dp_color = rank // ep_size``, so
    # only an (outer_fsdp, inner_ep) device layout groups ranks correctly.
    devices = mesh_utils.create_device_mesh((FSDP_SIZE, EP_SIZE))
    mesh_obj = Mesh(devices, axis_names=(FSDP_AXIS, EP_AXIS))

    num_procs = jax.process_count()
    max_tokens_per_rank = (BATCH // num_procs) * SEQ
    recv_capacity_per_rank = get_moe_recv_capacity_per_rank(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        max_tokens_per_rank=max_tokens_per_rank,
        ep_size=EP_SIZE,
        recv_capacity_factor=2.0,
    )

    # Eager bootstrap: ep_bootstrap does a host-side NCCL UID allgather
    # and cannot run from inside jax.jit. Sized to the worst-case recv_pr
    # across _CONFIGS so every parametrized config is bootstrap-compatible.
    with mesh_obj, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ):
        ep_bootstrap(
            world_size=num_procs,
            rank=jax.process_index(),
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=HIDDEN,
            max_token_dtype=DTYPE,
            drop_on_overflow=True,
        )
    record_ep_bootstrap_signature_for_moe(
        num_experts=NUM_EXPERTS,
        max_tokens_per_rank=max_tokens_per_rank,
        recv_capacity_per_rank=recv_capacity_per_rank,
        hidden_dim=HIDDEN,
        ep_size=EP_SIZE,
    )
    return mesh_obj


# -----------------------------------------------------------------------------
# Pure-JAX reference MoE (no EP). Mirrors the exact math of TE's fused
# router primitive (see tests/jax/test_fused_router.py for the same
# reference applied to the standalone router kernel):
#
# softmax + post-softmax (use_pre_softmax=False, the default):
#   1. top_k by raw logits
#   2. softmax over just the K selected logits (so weights sum to 1)
#
# sigmoid + optional expert_bias:
#   1. scores = sigmoid(logits)
#   2. top_k by (scores + expert_bias)  [bias only steers selection]
#   3. weights = scores at top_k positions, normalized when K > 1
#
# Then for both:
#   * weights *= scaling_factor (we leave scaling_factor=1.0 in this
#     suite, matching _make_block's default).
#   * per-expert FFN: silu(layer_w0) * layer_w1 → wo.
# -----------------------------------------------------------------------------


@partial(
    jax.jit,
    static_argnames=(
        "num_experts",
        "num_experts_per_tok",
        "aux_loss_coeff",
        "score_function",
    ),
)
def _pure_jax_moe_reference(
    x,
    gate_kernel,
    wi_0,
    wi_1,
    wo,
    expert_bias=None,
    *,
    num_experts,
    num_experts_per_tok,
    aux_loss_coeff: float = 0.0,
    score_function: str = "softmax",
):
    B, S, H = x.shape
    T = B * S
    K = num_experts_per_tok
    x_2d = x.reshape(T, H)

    gate_kernel_cast = gate_kernel.astype(x.dtype)
    logits = (x_2d @ gate_kernel_cast).astype(jnp.float32)  # [T, E]

    if score_function == "softmax":
        # use_pre_softmax=False: topk on raw logits, then softmax over K.
        top_logits, top_indices = jax.lax.top_k(logits, k=K)
        weights = jax.nn.softmax(top_logits, axis=-1)  # [T, K], sums to 1
    elif score_function == "sigmoid":
        scores = jax.nn.sigmoid(logits)  # [T, E]
        if expert_bias is not None and expert_bias.shape != (0,):
            scores_for_routing = scores + expert_bias.astype(jnp.float32)[None, :]
            _, top_indices = jax.lax.top_k(scores_for_routing, k=K)
            weights = jnp.take_along_axis(scores, top_indices, axis=-1)
        else:
            weights, top_indices = jax.lax.top_k(scores, k=K)
        # Sigmoid weights are normalized when K > 1 (matches the kernel).
        if K > 1:
            weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
    else:
        raise ValueError(f"Unsupported score_function={score_function!r}")

    routing_weights_full = jnp.zeros((T, num_experts), dtype=jnp.float32)
    routing_weights_full = routing_weights_full.at[
        jnp.arange(T)[:, None], top_indices
    ].set(weights)

    # FFN. ``apply_topk_weights_early`` is a fusion knob that doesn't
    # change the math (wo is linear), so the reference is identical for
    # both placements.
    layer_w0 = jnp.einsum("th,ehm->tem", x_2d, wi_0)
    layer_w1 = jnp.einsum("th,ehm->tem", x_2d, wi_1)
    # Activation runs in x.dtype (typically bf16) to mirror the impl --
    # the impl keeps silu+multiply in the wi GEMM output dtype because
    # storing higher precision than the consumer (wo) GEMM buys nothing.
    intermediate = jax.nn.silu(layer_w0) * layer_w1
    expert_out = jnp.einsum("tem,emh->teh", intermediate, wo)  # [T, E, H]
    output_2d = jnp.einsum(
        "te,teh->th", routing_weights_full.astype(x.dtype), expert_out
    )
    output = output_2d.reshape(B, S, H).astype(x.dtype)

    if aux_loss_coeff > 0.0:
        # tex.fused_moe_aux_loss formula (matches the same
        # reference_aux_loss helper from test_fused_router.py). The
        # "aux scores" use the same score_function but always with
        # K-normalised sigmoid (when sigmoid) / plain softmax (when
        # softmax) — see tex.fused_topk_with_score_function_fwd with
        # compute_aux_scores=True.
        if score_function == "softmax":
            aux_scores = jax.nn.softmax(logits, axis=-1)
        else:  # sigmoid
            aux_scores = jax.nn.sigmoid(logits)
            if K > 1:
                aux_scores = aux_scores / (
                    aux_scores.sum(axis=-1, keepdims=True) + 1e-20
                )
        routing_map = (routing_weights_full > 0).astype(jnp.int32)
        tokens_per_expert = jnp.sum(routing_map, axis=0)  # [E]
        sum_probs_per_expert = jnp.sum(aux_scores, axis=0)  # [E]
        aux_loss = (num_experts * aux_loss_coeff / (K * (T**2))) * jnp.sum(
            sum_probs_per_expert * tokens_per_expert.astype(jnp.float32)
        )
        aux_loss = aux_loss.astype(x.dtype)
    else:
        aux_loss = jnp.zeros((), dtype=x.dtype)
    return output, aux_loss


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_block(
    *,
    apply_topk_weights_early=False,
    aux_loss_coeff=0.0,
    use_expert_routing_bias=False,
    score_function="softmax",
    expert_bias_init=None,
    compound_expert_sharding=False,
    input_axes=("batch", None, None),
    recv_capacity_per_rank=None,
):
    if recv_capacity_per_rank is None:
        recv_capacity_per_rank = get_moe_recv_capacity_per_rank(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            max_tokens_per_rank=(BATCH // jax.process_count()) * SEQ,
            ep_size=EP_SIZE,
            recv_capacity_factor=2.0,
        )
    kwargs = dict(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        data_parallelism_axes=(FSDP_AXIS,),
        apply_topk_weights_early=apply_topk_weights_early,
        aux_loss_coeff=aux_loss_coeff,
        use_expert_routing_bias=use_expert_routing_bias,
        score_function=score_function,
        dtype=DTYPE,
        input_axes=input_axes,
        recv_capacity_per_rank=recv_capacity_per_rank,
    )
    if compound_expert_sharding:
        # Match MaxText shard_exp_on_fsdp=True: FSDP and EP both shard the
        # expert group axis while the hidden dimensions remain replicated.
        kwargs["wi_kernel_axes"] = ("exp_fsdp", "embed_replicated", "mlp")
        kwargs["wo_kernel_axes"] = ("exp_fsdp", "mlp", "embed_replicated")
    # Custom expert_bias_init lets tests inject a non-zero expert_bias without
    # poking variables['params'] post-init.
    if expert_bias_init is not None:
        kwargs["expert_bias_init"] = expert_bias_init
    return MoEBlock(**kwargs)


def _strong_expert_bias_init(key, shape, dtype):
    """Half +5, half -5 — large enough to force topk onto the +ve half."""
    del key
    n = shape[0]
    return jnp.concatenate(
        [
            jnp.full((n // 2,), 5.0, dtype=dtype),
            jnp.full((n - n // 2,), -5.0, dtype=dtype),
        ]
    )


def _cross_rank_expert_bias_init(key, shape, dtype):
    """Select one expert on each EP rank for a balanced reduced-capacity test."""
    del key
    bias = jnp.full(shape, -5.0, dtype=dtype)
    return bias.at[jnp.asarray((0, shape[0] // EP_SIZE))].set(5.0)


def _shard_inputs(x, mesh):
    # Match the layout moe.py re-pins to: outer dp axes, then ep innermost.
    return jax.lax.with_sharding_constraint(
        x, NamedSharding(mesh, P((FSDP_AXIS, EP_AXIS), None, None))
    )


def _ctx(mesh):
    """Combined mesh + global_shard_guard + axis_rules context."""

    class _Combo:
        def __enter__(self_inner):
            self_inner._m = mesh.__enter__()
            self_inner._gs = global_shard_guard(
                MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
            )
            self_inner._gs.__enter__()
            self_inner._ar = nn_partitioning.axis_rules(LOGICAL_AXIS_RULES)
            self_inner._ar.__enter__()
            return self_inner._m

        def __exit__(self_inner, *args):
            self_inner._ar.__exit__(*args)
            self_inner._gs.__exit__(*args)
            mesh.__exit__(*args)

    return _Combo()


def _init_apply(block, mesh, x, key):
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)
        variables = jax.jit(block.init)(key, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(variables)[0])
        output, aux, _trt = jax.jit(block.apply)(variables, x_sh)
        jax.block_until_ready(output)
    return variables, output, aux


def _grad_step(block, variables, mesh, x, *, include_aux=False):
    """Run jax.grad of mean(out^2) [+ aux if include_aux] vs (params, x).

    Returns ``(grads_variables, grad_x)`` so callers can check both the
    weight gradients and the input-activation gradient that propagates
    back to the previous layer.
    """
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)

        def loss_fn(variables, x):
            output, aux, _trt = block.apply(variables, x)
            loss = jnp.mean(output.astype(jnp.float32) ** 2)
            if include_aux and aux is not None:
                loss = loss + aux.astype(jnp.float32)
            return loss

        grads_v, grad_x = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(variables, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads_v)[0])
        jax.block_until_ready(grad_x)
        return grads_v, grad_x


def _grad_aux_only(block, variables, mesh, x):
    """Jit'd grad of just the aux loss scalar — proves it reaches the
    gate even when no main-output contribution is present."""
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)

        def aux_only(variables, x):
            _, aux, _trt = block.apply(variables, x)
            return aux.astype(jnp.float32)

        grads = jax.jit(jax.grad(aux_only))(variables, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
        return grads


def _unwrap(x):
    return x.value if hasattr(x, "value") else x


def _to_global_numpy(arr, mesh):
    """Replicate a sharded JAX array onto every rank and return as numpy.

    Triggers an all-gather inside JIT. The resulting addressable_data(0)
    contains the full global array on every process, so we can run the
    pure-JAX reference and compare against it from any process.
    """
    rep = NamedSharding(mesh, P())
    with mesh:
        full = jax.jit(lambda a: jax.lax.with_sharding_constraint(a, rep))(arr)
        full.block_until_ready()
    return np.asarray(jax.device_get(full.addressable_data(0)))


def _params_global_numpy(variables, mesh):
    """Pull every entry of variables['params'] to a replicated numpy array."""
    params = variables["params"]
    return {name: _to_global_numpy(_unwrap(p), mesh) for name, p in params.items()}


def _make_inputs(key):
    """Generate a globally-identical input tensor on every process."""
    return jax.random.normal(key, (BATCH, SEQ, HIDDEN), dtype=DTYPE)


def _make_stacked_moe_params(mesh, num_layers):
    """Create distinct production-shaped MoE parameters with a scan axis.

    The expert parameters use the same compound ``(fsdp, ep)`` expert
    sharding as MaxText's ``shard_exp_on_fsdp=True`` integration. A rotating
    strong routing bias makes each layer's routing map distinct while also
    leaving half of the experts empty in every layer.
    """
    gate_key, wi_key, wo_key = jax.random.split(jax.random.PRNGKey(101), 3)

    with _ctx(mesh):

        @jax.jit
        def initialize():
            gate_kernel = jax.random.normal(
                gate_key,
                (num_layers, HIDDEN, NUM_EXPERTS),
                dtype=DTYPE,
            ) / np.sqrt(HIDDEN)
            wi = jax.random.normal(
                wi_key,
                (num_layers, NUM_EXPERTS, HIDDEN, 2 * INTER),
                dtype=DTYPE,
            ) / np.sqrt(HIDDEN)
            wo = jax.random.normal(
                wo_key,
                (num_layers, NUM_EXPERTS, INTER, HIDDEN),
                dtype=DTYPE,
            ) / np.sqrt(INTER)
            base_bias = jnp.concatenate(
                (
                    jnp.full((NUM_EXPERTS // 2,), 10.0, dtype=jnp.float32),
                    jnp.full(
                        (NUM_EXPERTS - NUM_EXPERTS // 2,),
                        -10.0,
                        dtype=jnp.float32,
                    ),
                )
            )
            expert_bias = jnp.stack(
                [
                    jnp.roll(base_bias, (2 * layer) % NUM_EXPERTS)
                    for layer in range(num_layers)
                ]
            )

            gate_kernel = jax.lax.with_sharding_constraint(
                gate_kernel, NamedSharding(mesh, P(None, None, None))
            )
            wi = jax.lax.with_sharding_constraint(
                wi,
                NamedSharding(
                    mesh,
                    P(None, (FSDP_AXIS, EP_AXIS), None, None),
                ),
            )
            wo = jax.lax.with_sharding_constraint(
                wo,
                NamedSharding(
                    mesh,
                    P(None, (FSDP_AXIS, EP_AXIS), None, None),
                ),
            )
            expert_bias = jax.lax.with_sharding_constraint(
                expert_bias, NamedSharding(mesh, P(None, None))
            )
            return {
                "gate_kernel": gate_kernel,
                "wi": wi,
                "wo": wo,
                "expert_bias": expert_bias,
            }

        params = initialize()
        jax.block_until_ready(params["wi"])
    return params


def _functional_production_moe_layer(params, x):
    """One residual TE MoE layer matching the failing integration knobs."""
    normed_x = (
        x.astype(jnp.float32)
        * jax.lax.rsqrt(
            jnp.mean(x.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + 1.0e-6
        )
    ).astype(x.dtype)
    branch, _ = moe(
        normed_x,
        params["gate_kernel"],
        params["wi"],
        params["wo"],
        expert_bias=params["expert_bias"],
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        score_function="sigmoid",
        apply_topk_weights_early=False,
        ep_axis=EP_AXIS,
        data_parallelism_axes=(FSDP_AXIS,),
        wi_kernel_axes=("exp_fsdp", "embed_replicated", "mlp"),
        wo_kernel_axes=("exp_fsdp", "mlp", "embed_replicated"),
        dtype=DTYPE,
    )
    return x + branch


def _run_stacked_te_moe(params, x, *, use_scan, remat):
    """Run identical per-layer parameters scanned or Python-unrolled."""
    layer_fn = _functional_production_moe_layer
    if remat:
        # MaxText applies nn.remat to its layer before passing the layer to
        # nn.scan. Checkpointing the functional body gives the same important
        # lowering: forward recomputation occurs inside reverse scan.
        layer_fn = jax.checkpoint(layer_fn, prevent_cse=True)

    if use_scan:

        def scan_body(value, layer_params):
            return layer_fn(layer_params, value), None

        return jax.lax.scan(scan_body, x, params)[0]

    value = x
    for layer in range(params["gate_kernel"].shape[0]):
        layer_params = jax.tree_util.tree_map(lambda p: p[layer], params)
        value = layer_fn(layer_params, value)
    return value


def _run_stacked_jax_reference(params, x):
    """Pure-JAX residual stack using the same distinct layer parameters."""
    value = x
    for layer in range(params["gate_kernel"].shape[0]):
        layer_params = jax.tree_util.tree_map(lambda p: p[layer], params)
        normed_value = (
            value.astype(jnp.float32)
            * jax.lax.rsqrt(
                jnp.mean(value.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
                + 1.0e-6
            )
        ).astype(value.dtype)
        branch, _ = _pure_jax_moe_reference(
            normed_value,
            layer_params["gate_kernel"],
            layer_params["wi"][..., :INTER],
            layer_params["wi"][..., INTER:],
            layer_params["wo"],
            layer_params["expert_bias"],
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            score_function="sigmoid",
        )
        value = value + branch
    return value


def _stack_value_and_grad(params, x, *, use_scan, remat):
    def loss_fn(p, value):
        output = _run_stacked_te_moe(
            p,
            value,
            use_scan=use_scan,
            remat=remat,
        )
        return jnp.mean(output.astype(jnp.float32) ** 2), output

    return jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(params, x)


def _gradient_similarity(actual, expected):
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    actual_norm = np.linalg.norm(actual)
    expected_norm = np.linalg.norm(expected)
    norm_ratio = actual_norm / expected_norm
    cosine = np.dot(actual, expected) / (actual_norm * expected_norm)
    return actual_norm, expected_norm, norm_ratio, cosine


def _assert_te_to_te_gradient(actual, expected, *, name):
    """Strict scan-vs-unrolled oracle with useful failure diagnostics."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert np.all(np.isfinite(actual)), f"{name}: scan gradient has NaN/Inf"
    assert np.all(np.isfinite(expected)), f"{name}: unrolled gradient has NaN/Inf"
    actual_norm, expected_norm, norm_ratio, cosine = _gradient_similarity(
        actual, expected
    )
    assert actual_norm > 0.0 and expected_norm > 0.0, (
        f"{name}: zero gradient norm "
        f"(scan={actual_norm:.9e}, unrolled={expected_norm:.9e})"
    )
    lo, hi = TE_TO_TE_GRAD_NORM_RATIO
    assert lo <= norm_ratio <= hi and cosine >= TE_TO_TE_GRAD_COSINE, (
        f"{name}: scan/unrolled gradient direction or scale mismatch: "
        f"scan_norm={actual_norm:.9e}, unrolled_norm={expected_norm:.9e}, "
        f"ratio={norm_ratio:.9f}, cosine={cosine:.9f}, "
        f"scan_mean={actual.astype(np.float64).mean():.9e}, "
        f"scan_std={actual.astype(np.float64).std():.9e}, "
        f"scan_absmax={np.abs(actual.astype(np.float64)).max():.9e}, "
        f"unrolled_mean={expected.astype(np.float64).mean():.9e}, "
        f"unrolled_std={expected.astype(np.float64).std():.9e}, "
        f"unrolled_absmax={np.abs(expected.astype(np.float64)).max():.9e}"
    )
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=TE_TO_TE_ATOL,
        rtol=TE_TO_TE_RTOL,
        err_msg=f"{name}: scan/unrolled elementwise mismatch",
    )


def _assert_stacked_reference_gradient(actual, expected, *, name):
    """Broad TE-vs-JAX stack control; strict parity is tested per block."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    assert np.all(np.isfinite(actual)), f"{name}: TE gradient has NaN/Inf"
    assert np.all(np.isfinite(expected)), f"{name}: reference gradient has NaN/Inf"
    actual_norm, expected_norm, norm_ratio, cosine = _gradient_similarity(
        actual, expected
    )
    assert actual_norm > 0.0 and expected_norm > 0.0, (
        f"{name}: zero gradient norm "
        f"(TE={actual_norm:.9e}, reference={expected_norm:.9e})"
    )
    assert 0.5 <= norm_ratio <= 1.5 and cosine >= 0.85, (
        f"{name}: gross TE/reference stack mismatch: "
        f"TE_norm={actual_norm:.9e}, reference_norm={expected_norm:.9e}, "
        f"ratio={norm_ratio:.9f}, cosine={cosine:.9f}"
    )


def _assert_distinct_layer_routes(params_np, x_np):
    """Prove the test does not accidentally reuse one routing map."""
    value = jnp.asarray(x_np)
    signatures = []
    for layer in range(params_np["gate_kernel"].shape[0]):
        gate = jnp.asarray(params_np["gate_kernel"][layer])
        bias = jnp.asarray(params_np["expert_bias"][layer])
        normed_value = (
            value.astype(jnp.float32)
            * jax.lax.rsqrt(
                jnp.mean(value.astype(jnp.float32) ** 2, axis=-1, keepdims=True)
                + 1.0e-6
            )
        ).astype(value.dtype)
        logits = jnp.einsum("bsh,he->bse", normed_value, gate).astype(jnp.float32)
        _, indices = jax.lax.top_k(jax.nn.sigmoid(logits) + bias, TOPK)
        indices_np = np.asarray(jax.device_get(indices), dtype=np.int64)
        position = np.arange(indices_np.size, dtype=np.int64).reshape(indices_np.shape)
        signatures.append(
            (
                int(indices_np.sum()),
                int((indices_np * (position + 1)).sum()),
            )
        )
        branch, _ = _pure_jax_moe_reference(
            normed_value,
            gate,
            jnp.asarray(params_np["wi"][layer])[..., :INTER],
            jnp.asarray(params_np["wi"][layer])[..., INTER:],
            jnp.asarray(params_np["wo"][layer]),
            bias,
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            score_function="sigmoid",
        )
        value = value + branch
    assert (
        len(set(signatures)) > 1
    ), f"all layers used one routing signature: {signatures}"


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# Parametrize variants exercised by both the forward and the backward
# parity tests. Each config is one MoE-block configuration the suite
# wants covered; the test body checks shape, dtype, finiteness AND
# numerical parity vs the same pure-JAX reference (which understands
# the same set of knobs).
# -----------------------------------------------------------------------------

_CONFIGS = [
    pytest.param(
        dict(score_function="softmax"),
        id="softmax",
    ),
    pytest.param(
        dict(score_function="softmax", apply_topk_weights_early=True),
        id="softmax-early-weighting",
    ),
    pytest.param(
        dict(score_function="softmax", compound_expert_sharding=True),
        id="softmax-compound-fsdp-expert",
    ),
    pytest.param(
        dict(
            score_function="sigmoid",
            apply_topk_weights_early=True,
            compound_expert_sharding=True,
        ),
        id="sigmoid-early-weighting-compound-fsdp-expert",
    ),
    pytest.param(
        dict(score_function="sigmoid"),
        id="sigmoid",
    ),
    # NOTE: a ``sigmoid-bias-zero`` config (use_expert_routing_bias=True
    # with a zero-initialised bias buffer) was previously exercised
    # here. It was dropped because the routing math collapses to the
    # no-bias case when the buffer is zero -- ``sigmoid`` already
    # covers that numerical path. The bias-aware codepath is still
    # exercised by ``sigmoid-bias-strong`` below, which uses a
    # non-zero bias.
    pytest.param(
        dict(
            score_function="sigmoid",
            use_expert_routing_bias=True,
            expert_bias_init=_strong_expert_bias_init,
        ),
        id="sigmoid-bias-strong",
    ),
]


def _reference_kwargs_from_config(config, params_np):
    """Pick out the reference-relevant pieces of a parametrize config."""
    return dict(
        score_function=config.get("score_function", "softmax"),
        expert_bias=(
            jnp.asarray(params_np["expert_bias"])
            if config.get("use_expert_routing_bias", False)
            else None
        ),
    )


class TestTeEpMoeReceiveCapacity:
    """Reduced receive buffers preserve valid results and report overflow."""

    def test_capacity_helper(self):
        # Use a production-like token count where alignment does not collapse
        # balanced and worst-case capacities to the same small test buffer.
        max_tpr = 256
        worst = get_moe_recv_capacity_per_rank(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            max_tokens_per_rank=max_tpr,
            ep_size=EP_SIZE,
        )
        balanced = get_moe_recv_capacity_per_rank(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            max_tokens_per_rank=max_tpr,
            ep_size=EP_SIZE,
            recv_capacity_factor=1.0,
        )
        headroom = get_moe_recv_capacity_per_rank(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            max_tokens_per_rank=max_tpr,
            ep_size=EP_SIZE,
            recv_capacity_factor=2.0,
        )
        assert balanced < headroom < worst
        assert balanced % _ALIGN_SIZE == 0
        with pytest.raises(ValueError, match="finite and >= 1.0"):
            get_moe_recv_capacity_per_rank(
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=TOPK,
                max_tokens_per_rank=max_tpr,
                ep_size=EP_SIZE,
                recv_capacity_factor=0.5,
            )


class TestTeEpMoeForward:
    """Per-config forward correctness in a single run: shape, dtype,
    finiteness AND numerical parity vs the pure-JAX reference."""

    @pytest.mark.parametrize("config", _CONFIGS)
    def test_forward(self, mesh, config):
        block = _make_block(**config)
        x = _make_inputs(jax.random.PRNGKey(0))
        variables, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(1))

        # Shape / dtype / finiteness (cheap; on the local shard).
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        out_local = np.asarray(jax.device_get(output.addressable_data(0)))
        assert np.all(np.isfinite(out_local)), "output has NaN/Inf"
        assert aux is None, "aux_loss should be None when aux_loss_coeff == 0"

        # Numerical parity (replicated global view -> single rank's numpy).
        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        out_te_np = _to_global_numpy(output, mesh)

        out_ref, _ = _pure_jax_moe_reference(
            jnp.asarray(x_np),
            jnp.asarray(params_np["gate_kernel"]),
            jnp.asarray(params_np["wi"])[..., :INTER],
            jnp.asarray(params_np["wi"])[..., INTER:],
            jnp.asarray(params_np["wo"]),
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            **_reference_kwargs_from_config(config, params_np),
        )
        np.testing.assert_allclose(
            out_te_np.astype(np.float32),
            np.asarray(jax.device_get(out_ref)).astype(np.float32),
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
            err_msg=f"forward parity breach for config={config}",
        )


class TestTeEpMoeBackward:
    """Per-config backward correctness in a single run: per-tensor
    grads finite, non-zero AND parity vs the pure-JAX reference."""

    @pytest.mark.parametrize("config", _CONFIGS)
    def test_backward(self, mesh, config):
        block = _make_block(**config)
        x = _make_inputs(jax.random.PRNGKey(2))
        variables, _, _ = _init_apply(block, mesh, x, jax.random.PRNGKey(3))
        grads_te, grad_x_te = _grad_step(block, variables, mesh, x)

        # Reference grads via jax.grad over the pure-JAX MoE with the
        # same config. argnums=(0, 1) so the reference also produces a
        # d_x for the propagated-gradient parity check below.
        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        ref_kwargs = _reference_kwargs_from_config(config, params_np)
        ref_expert_bias = ref_kwargs.pop("expert_bias")

        def loss_fn(params, x):
            out, _ = _pure_jax_moe_reference(
                x,
                params["gate_kernel"],
                params["wi"][..., :INTER],
                params["wi"][..., INTER:],
                params["wo"],
                ref_expert_bias,
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=TOPK,
                **ref_kwargs,
            )
            return jnp.mean(out.astype(jnp.float32) ** 2)

        grads_ref, grad_x_ref = jax.jit(jax.grad(loss_fn, argnums=(0, 1)))(
            {k: jnp.asarray(v) for k, v in params_np.items() if k != "expert_bias"},
            jnp.asarray(x_np),
        )
        grads_ref_np = {k: np.asarray(jax.device_get(v)) for k, v in grads_ref.items()}
        grad_x_ref_np = np.asarray(jax.device_get(grad_x_ref))

        for name in ("gate_kernel", "wi", "wo"):
            # Per-tensor: finite + non-zero + parity in one pass.
            g_te = _to_global_numpy(_unwrap(grads_te["params"][name]), mesh)
            assert np.all(
                np.isfinite(g_te)
            ), f"{name} grad has NaN/Inf [config={config}]"
            assert np.any(
                g_te != 0.0
            ), f"{name} grad identically zero [config={config}]"
            atol, rtol = (
                (GRAD_GATE_ATOL, GRAD_GATE_RTOL)
                if name == "gate_kernel"
                else (GRAD_FFN_ATOL, GRAD_FFN_RTOL)
            )
            np.testing.assert_allclose(
                g_te.astype(np.float32),
                grads_ref_np[name].astype(np.float32),
                atol=atol,
                rtol=rtol,
                err_msg=f"grad parity breach on {name} [config={config}]",
            )
            _assert_gradient_direction_and_scale(
                g_te,
                grads_ref_np[name],
                name=f"{name} [config={config}]",
            )

        # d_x: the gradient propagated back to the previous layer. Checks
        # shape, dtype (must match x.dtype — protects the
        # _with_sharding_constraint_cast_bwd wrapper that casts the
        # fp32-promoted gate path back to bf16), finiteness, non-zero
        # AND numerical parity vs the pure-JAX reference d_x.
        grad_x_te_np = _to_global_numpy(grad_x_te, mesh)
        assert (
            grad_x_te.shape == x.shape
        ), f"d_x shape {grad_x_te.shape} != x.shape {x.shape} [config={config}]"
        assert (
            grad_x_te.dtype == x.dtype
        ), f"d_x dtype {grad_x_te.dtype} != x.dtype {x.dtype} [config={config}]"
        assert np.all(np.isfinite(grad_x_te_np)), f"d_x has NaN/Inf [config={config}]"
        assert np.any(grad_x_te_np != 0.0), f"d_x identically zero [config={config}]"
        np.testing.assert_allclose(
            grad_x_te_np.astype(np.float32),
            grad_x_ref_np.astype(np.float32),
            atol=GRAD_FFN_ATOL,
            rtol=GRAD_FFN_RTOL,
            err_msg=f"d_x parity breach [config={config}]",
        )
        _assert_gradient_direction_and_scale(
            grad_x_te_np,
            grad_x_ref_np,
            name=f"d_x [config={config}]",
        )

    def test_repeated_production_block_backward(self, mesh):
        """Catch small d_x errors that amplify across a stack of MoE blocks."""
        repeats = 8
        config = dict(
            score_function="sigmoid",
            apply_topk_weights_early=True,
            compound_expert_sharding=True,
            use_expert_routing_bias=True,
        )
        block = _make_block(**config)
        x = _make_inputs(jax.random.PRNGKey(22))
        variables, _, _ = _init_apply(block, mesh, x, jax.random.PRNGKey(23))

        with _ctx(mesh):
            x_sh = _shard_inputs(x, mesh)

            def te_loss_fn(variables, value):
                for _ in range(repeats):
                    branch, _, _ = block.apply(variables, value)
                    value = value + branch
                return jnp.mean(value.astype(jnp.float32) ** 2)

            grads_te, grad_x_te = jax.jit(jax.grad(te_loss_fn, argnums=(0, 1)))(
                variables,
                x_sh,
            )
            jax.block_until_ready(grad_x_te)

        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        expert_bias = jnp.asarray(params_np["expert_bias"])

        def reference_loss_fn(params, value):
            for _ in range(repeats):
                branch, _ = _pure_jax_moe_reference(
                    value,
                    params["gate_kernel"],
                    params["wi"][..., :INTER],
                    params["wi"][..., INTER:],
                    params["wo"],
                    expert_bias,
                    num_experts=NUM_EXPERTS,
                    num_experts_per_tok=TOPK,
                    score_function="sigmoid",
                )
                value = value + branch
            return jnp.mean(value.astype(jnp.float32) ** 2)

        grads_ref, grad_x_ref = jax.jit(jax.grad(reference_loss_fn, argnums=(0, 1)))(
            {k: jnp.asarray(v) for k, v in params_np.items() if k != "expert_bias"},
            jnp.asarray(x_np),
        )

        for name in ("gate_kernel", "wi", "wo"):
            _assert_gradient_direction_and_scale(
                _to_global_numpy(_unwrap(grads_te["params"][name]), mesh),
                np.asarray(jax.device_get(grads_ref[name])),
                name=f"repeated {name}",
            )
        _assert_gradient_direction_and_scale(
            _to_global_numpy(grad_x_te, mesh),
            np.asarray(jax.device_get(grad_x_ref)),
            name="repeated d_x",
        )

    @pytest.mark.skip(reason="Experimental scan/remat parity coverage is currently disabled.")
    @pytest.mark.parametrize("num_layers", (4, 8))
    def test_scanned_production_block_backward(self, mesh, num_layers):
        """Full MoE scan/remat must match the identical unrolled stack.

        Unlike the primitive scan tests, this exercises top-2 routing,
        compound FSDP/EP expert parameters, grouped-GEMM weight gradients,
        the full ``moe`` custom VJP residual, and forward rematerialization
        inside the reverse scan.
        """
        params = _make_stacked_moe_params(mesh, num_layers)
        x = _make_inputs(jax.random.PRNGKey(102))

        with _ctx(mesh):
            x_sh = _shard_inputs(x, mesh)
            unrolled_run = jax.jit(
                partial(
                    _stack_value_and_grad,
                    use_scan=False,
                    remat=False,
                )
            )
            scan_run = jax.jit(
                partial(
                    _stack_value_and_grad,
                    use_scan=True,
                    remat=False,
                )
            )
            scan_remat_run = jax.jit(
                partial(
                    _stack_value_and_grad,
                    use_scan=True,
                    remat=True,
                )
            )

            unrolled = unrolled_run(params, x_sh)
            jax.block_until_ready(unrolled)
            scanned = scan_run(params, x_sh)
            jax.block_until_ready(scanned)
            scanned_remat = scan_remat_run(params, x_sh)
            jax.block_until_ready(scanned_remat)

        (unrolled_loss, unrolled_output), (unrolled_grads, unrolled_grad_x) = unrolled
        modes = (
            ("scan", scanned),
            ("scan-remat", scanned_remat),
        )
        unrolled_output_np = _to_global_numpy(unrolled_output, mesh)
        unrolled_grad_x_np = _to_global_numpy(unrolled_grad_x, mesh)
        unrolled_grads_np = {
            name: _to_global_numpy(grad, mesh) for name, grad in unrolled_grads.items()
        }

        params_np = {
            name: _to_global_numpy(param, mesh) for name, param in params.items()
        }
        x_np = np.asarray(jax.device_get(x))
        if jax.process_index() == 0:
            _assert_distinct_layer_routes(params_np, x_np)

        mismatches = []

        def record_check(check):
            try:
                check()
            except AssertionError as error:
                mismatches.append(str(error))

        for mode_name, mode_result in modes:
            (mode_loss, mode_output), (mode_grads, mode_grad_x) = mode_result
            record_check(
                lambda: np.testing.assert_allclose(
                    np.asarray(jax.device_get(mode_loss), dtype=np.float32),
                    np.asarray(jax.device_get(unrolled_loss), dtype=np.float32),
                    atol=TE_TO_TE_ATOL,
                    rtol=TE_TO_TE_RTOL,
                    err_msg=f"{mode_name}: loss mismatch",
                )
            )
            mode_output_np = _to_global_numpy(mode_output, mesh)
            record_check(
                lambda: np.testing.assert_allclose(
                    mode_output_np.astype(np.float32),
                    unrolled_output_np.astype(np.float32),
                    atol=TE_TO_TE_ATOL,
                    rtol=TE_TO_TE_RTOL,
                    err_msg=f"{mode_name}: output mismatch",
                )
            )
            mode_grad_x_np = _to_global_numpy(mode_grad_x, mesh)
            record_check(
                lambda: _assert_te_to_te_gradient(
                    mode_grad_x_np,
                    unrolled_grad_x_np,
                    name=f"{mode_name} d_x",
                )
            )
            for param_name in ("gate_kernel", "wi", "wo"):
                mode_grad_np = _to_global_numpy(mode_grads[param_name], mesh)
                record_check(
                    lambda mode_grad_np=mode_grad_np, param_name=param_name: (
                        _assert_te_to_te_gradient(
                            mode_grad_np,
                            unrolled_grads_np[param_name],
                            name=f"{mode_name} {param_name}",
                        )
                    )
                )
                for layer in range(num_layers):
                    record_check(
                        lambda layer=layer, mode_grad_np=mode_grad_np, param_name=param_name: (
                            _assert_te_to_te_gradient(
                                mode_grad_np[layer],
                                unrolled_grads_np[param_name][layer],
                                name=f"{mode_name} layer={layer} {param_name}",
                            )
                        )
                    )

        # The unrolled TE execution is the control. Check it against a
        # communication-free JAX reference so a scan/unrolled agreement cannot
        # hide a bug shared by both TE executions.
        reference_params = {
            name: jnp.asarray(value) for name, value in params_np.items()
        }

        def reference_loss_fn(p, value):
            output = _run_stacked_jax_reference(p, value)
            return jnp.mean(output.astype(jnp.float32) ** 2)

        reference_grads, reference_grad_x = jax.jit(
            jax.grad(reference_loss_fn, argnums=(0, 1))
        )(reference_params, jnp.asarray(x_np))
        for param_name in ("gate_kernel", "wi", "wo"):
            record_check(
                lambda param_name=param_name: _assert_stacked_reference_gradient(
                    unrolled_grads_np[param_name],
                    np.asarray(jax.device_get(reference_grads[param_name])),
                    name=f"unrolled-reference {param_name}",
                )
            )
        record_check(
            lambda: _assert_stacked_reference_gradient(
                unrolled_grad_x_np,
                np.asarray(jax.device_get(reference_grad_x)),
                name="unrolled-reference d_x",
            )
        )
        if mismatches:
            pytest.fail("\n\n".join(mismatches))

    @pytest.mark.skip(reason="Experimental scan/remat parity coverage is currently disabled.")
    def test_scanned_training_trajectory(self, mesh):
        """Three SGD steps must retain scan/remat vs unrolled parity."""
        num_layers = 4
        learning_rate = 1.0e-3
        params = _make_stacked_moe_params(mesh, num_layers)
        x = _make_inputs(jax.random.PRNGKey(103))

        def make_step(*, use_scan, remat):
            def loss_fn(p, value):
                output = _run_stacked_te_moe(
                    p,
                    value,
                    use_scan=use_scan,
                    remat=remat,
                )
                return jnp.mean(output.astype(jnp.float32) ** 2)

            def step(p, value):
                loss, grads = jax.value_and_grad(loss_fn)(p, value)
                updated = jax.tree_util.tree_map(
                    lambda weight, grad: weight - learning_rate * grad,
                    p,
                    grads,
                )
                return updated, loss

            return jax.jit(step)

        with _ctx(mesh):
            x_sh = _shard_inputs(x, mesh)
            unrolled_step = make_step(use_scan=False, remat=False)
            scan_remat_step = make_step(use_scan=True, remat=True)
            unrolled_params = params
            scan_remat_params = params
            unrolled_losses = []
            scan_remat_losses = []
            for _ in range(3):
                unrolled_params, unrolled_loss = unrolled_step(unrolled_params, x_sh)
                scan_remat_params, scan_remat_loss = scan_remat_step(
                    scan_remat_params, x_sh
                )
                jax.block_until_ready(unrolled_params)
                jax.block_until_ready(scan_remat_params)
                unrolled_losses.append(float(jax.device_get(unrolled_loss)))
                scan_remat_losses.append(float(jax.device_get(scan_remat_loss)))

        np.testing.assert_allclose(
            np.asarray(scan_remat_losses, dtype=np.float32),
            np.asarray(unrolled_losses, dtype=np.float32),
            atol=TE_TO_TE_ATOL,
            rtol=TE_TO_TE_RTOL,
            err_msg=(
                "scan-remat training trajectory differs from unrolled: "
                f"scan={scan_remat_losses}, unrolled={unrolled_losses}"
            ),
        )
        for param_name in ("gate_kernel", "wi", "wo"):
            _assert_te_to_te_gradient(
                _to_global_numpy(scan_remat_params[param_name], mesh),
                _to_global_numpy(unrolled_params[param_name], mesh),
                name=f"three-step parameter {param_name}",
            )


class TestTeEpMoeAuxLoss:
    """Aux-loss path. Consolidated into:
    * ``test_aux_loss``: one run that checks the returned scalar's
      shape / dtype / finiteness / magnitude AND numerical parity vs the
      reference AND that the aux-only bwd propagates to gate_kernel.
    * ``test_combined_loss_grads``: one run for joint main+aux bwd
      finite + non-zero per tensor.
    """

    def test_aux_loss(self, mesh):
        coeff = 1e-2
        block = _make_block(aux_loss_coeff=coeff)
        x = _make_inputs(jax.random.PRNGKey(20))
        variables, _, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(21))

        # Shape / dtype / finiteness / magnitude.
        assert aux is not None, "aux_loss should be returned when coeff > 0"
        assert aux.shape == (), f"aux_loss must be 0-d scalar, got {aux.shape}"
        assert aux.dtype == DTYPE, f"aux_loss dtype {aux.dtype} != {DTYPE}"
        aux_np = _to_global_numpy(aux, mesh)
        assert np.isfinite(aux_np), "aux_loss is NaN/Inf"
        assert abs(float(aux_np)) < 1e2, f"aux_loss looks unreasonable: {aux_np}"

        # Numerical parity vs the reference.
        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        _, aux_ref = _pure_jax_moe_reference(
            jnp.asarray(x_np),
            jnp.asarray(params_np["gate_kernel"]),
            jnp.asarray(params_np["wi"])[..., :INTER],
            jnp.asarray(params_np["wi"])[..., INTER:],
            jnp.asarray(params_np["wo"]),
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            aux_loss_coeff=coeff,
        )
        np.testing.assert_allclose(
            float(aux_np),
            float(jax.device_get(aux_ref)),
            atol=AUX_ATOL,
            rtol=AUX_RTOL,
        )

        # Aux-only bwd must propagate to gate_kernel — proves the
        # fused_moe_aux_loss_bwd → topk(compute_aux_scores)_bwd chain is
        # wired.
        aux_grads = _grad_aux_only(block, variables, mesh, x)
        g_gate = np.asarray(
            jax.device_get(
                _unwrap(aux_grads["params"]["gate_kernel"]).addressable_data(0)
            )
        )
        assert np.all(np.isfinite(g_gate)), "gate grad NaN/Inf under aux-only loss"
        assert np.any(g_gate != 0.0), "aux bwd should propagate to gate_kernel"

    def test_combined_loss_grads(self, mesh):
        """Joint main + aux loss bwd: per-tensor finite + non-zero in
        one pass."""
        block = _make_block(aux_loss_coeff=1e-2)
        x = _make_inputs(jax.random.PRNGKey(22))
        variables, _, _ = _init_apply(block, mesh, x, jax.random.PRNGKey(23))
        grads, _ = _grad_step(block, variables, mesh, x, include_aux=True)
        for name in ("gate_kernel", "wi", "wo"):
            g_local = np.asarray(
                jax.device_get(_unwrap(grads["params"][name]).addressable_data(0))
            )
            assert np.all(np.isfinite(g_local)), f"{name} grad NaN/Inf under main+aux"
            assert np.any(g_local != 0.0), f"{name} grad zero under main+aux"


class TestZZTeEpMoeOverflow:
    """Run last with a small exact capacity: valid routing then overflow."""

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def reduced_capacity_bootstrap(cls, mesh):
        del cls
        max_tpr = (BATCH // jax.process_count()) * SEQ
        capacity = _ALIGN_SIZE
        ep_finalize()
        with mesh, global_shard_guard(
            MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
        ):
            ep_bootstrap(
                world_size=jax.process_count(),
                rank=jax.process_index(),
                num_experts=NUM_EXPERTS,
                max_tokens_per_rank=max_tpr,
                recv_capacity_per_rank=capacity,
                hidden_dim=HIDDEN,
                max_token_dtype=DTYPE,
                drop_on_overflow=True,
            )
        record_ep_bootstrap_signature_for_moe(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tpr,
            recv_capacity_per_rank=capacity,
            hidden_dim=HIDDEN,
            ep_size=EP_SIZE,
        )
        yield capacity

    @pytest.mark.parametrize(
        ("expert_bias_init", "expect_overflow"),
        (
            pytest.param(_cross_rank_expert_bias_init, False, id="within-capacity"),
            pytest.param(_strong_expert_bias_init, True, id="overflow"),
        ),
    )
    def test_reduced_capacity_vjp(
        self, mesh, reduced_capacity_bootstrap, expert_bias_init, expect_overflow
    ):
        capacity = reduced_capacity_bootstrap

        x = jax.random.normal(jax.random.PRNGKey(72), (BATCH, SEQ, HIDDEN), dtype=DTYPE)
        block = _make_block(
            score_function="sigmoid",
            use_expert_routing_bias=True,
            expert_bias_init=expert_bias_init,
            recv_capacity_per_rank=capacity,
        )
        with _ctx(mesh):
            x_sh = _shard_inputs(x, mesh)
            variables = jax.jit(block.init)(jax.random.PRNGKey(73), x_sh)

            def loss_fn(variables, inputs):
                output, _aux, totals = block.apply(variables, inputs)
                return jnp.mean(output.astype(jnp.float32) ** 2), totals

            (loss, totals), grads = jax.jit(
                jax.value_and_grad(loss_fn, has_aux=True)
            )(variables, x_sh)
            jax.block_until_ready((loss, totals, grads))

        observed = int(_to_global_numpy(totals, mesh).max())
        assert (observed > capacity) is expect_overflow
        assert np.isfinite(float(jax.device_get(loss)))
        assert all(
            np.all(np.isfinite(np.asarray(jax.device_get(leaf.addressable_data(0)))))
            for leaf in jax.tree_util.tree_leaves(grads)
        )
