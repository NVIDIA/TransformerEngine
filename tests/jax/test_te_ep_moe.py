# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-process (one-GPU-per-process) tests for the TE-EP MoE custom_vjp.

The launcher ``tests/jax/run_te_ep_moe.sh`` forks one pytest process per
visible GPU (mirroring ``run_multiprocess_moe_vjp.sh``). Each process binds
to exactly one device via
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

This file is the TE-EP-only successor to ``test_moe_vjp.py`` and
``test_multiprocess_moe_vjp.py``. Each test exercises one MoE-block
run and bundles every check that single run supports — shape, dtype,
finiteness AND numerical parity vs a pure-JAX reference. Variations
on the block are pytest parametrize values rather than separate test
classes:

* ``test_forward`` covers the forward across a curated set of
  configurations (apply_topk_weights_early on/off, align_size=0/128,
  softmax/sigmoid scoring, optional expert_bias). Each config asserts
  shape, dtype, finiteness and numerical parity vs the reference in
  one run.
* ``test_backward`` mirrors that for gradients.
* ``TestTeEpMoeAuxLoss`` covers the second return value end-to-end
  (returned + parity + aux-only grad propagates to gate + combined
  main+aux grads stay finite) in two consolidated tests.
* ``TestTeEpMoEBlockFlax`` exercises the Flax wrapper with the same
  parity reference.
* ``TestZZZTeEpMoeBootstrap`` verifies the per-process NCCL bootstrap
  rejects a mismatched signature.

FP8 / MXFP8 recipes are deferred — the ``quantizer_sets`` plumbing
has not yet been re-wired across the TE-EP ``shard_map`` boundary
(see ``.pr3036-review/INTEGRATION_DESIGN.md``).
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
        "test_te_ep_moe.py requires the multiprocess launcher "
        "(run_te_ep_moe.sh). Skipping.",
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
from transformer_engine.jax.moe import moe, record_ep_bootstrap_signature_for_moe
from transformer_engine.jax.ep import ep_bootstrap
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


# -----------------------------------------------------------------------------
# Mesh / shape config
# -----------------------------------------------------------------------------

EP_AXIS = "ep"
FSDP_AXIS = "fsdp"
EP_SIZE = 2
assert (
    jax.device_count() % EP_SIZE == 0
), f"device_count {jax.device_count()} must be divisible by EP_SIZE={EP_SIZE}"
FSDP_SIZE = jax.device_count() // EP_SIZE
NUM_DEVICES_REQUIRED = EP_SIZE * FSDP_SIZE

LOGICAL_AXIS_RULES = (
    ("exp", EP_AXIS),
    ("embed", FSDP_AXIS),
    ("mlp", None),
    ("batch", (EP_AXIS, FSDP_AXIS)),
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
# (align_size rounding, etc.).
TE_TO_TE_ATOL = 5e-3
TE_TO_TE_RTOL = 5e-3

# Aux loss is computed in float32 from the SAME logits as the routing
# path. Numerical drift between TE-EP and the reference is dominated by
# the bf16-rounded softmax inside the topk kernel.
AUX_ATOL = 1e-3
AUX_RTOL = 1e-3


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def _compute_worst_case_recv_pr():
    """Worst-case per-rank recv buffer across every config in _CONFIGS.

    Bootstrap reserves NCCL EP buffers; per-call recv_pr <= bootstrap
    recv_pr is fine. We size with the largest align_size in _CONFIGS so
    the align128 config still fits the same singleton bootstrap.
    """
    num_procs = jax.device_count()
    dp_size = num_procs // EP_SIZE
    num_local_experts = NUM_EXPERTS // EP_SIZE
    natural_recv_pr = (BATCH // dp_size) * SEQ * TOPK
    natural_spe = (natural_recv_pr + num_local_experts - 1) // num_local_experts
    worst_align = 128
    worst_spe = ((natural_spe + worst_align - 1) // worst_align) * worst_align
    return num_local_experts * worst_spe


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
    recv_capacity_per_rank = _compute_worst_case_recv_pr()

    # Eager bootstrap: ep_bootstrap does a host-side NCCL UID allgather
    # and cannot run from inside jax.jit. Sized to the worst-case recv_pr
    # across _CONFIGS so every parametrized config is bootstrap-compatible.
    with mesh_obj, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS, fsdp_resource=FSDP_AXIS)
    ):
        ep_bootstrap(
            world_size=num_procs,
            rank=jax.process_index(),
            ep_size=EP_SIZE,
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=HIDDEN,
            allow_handle_mem_reloc=True,
            max_token_dtype=DTYPE,
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
    intermediate = jax.nn.silu(layer_w0.astype(jnp.float32)) * layer_w1.astype(jnp.float32)
    intermediate = intermediate.astype(x.dtype)
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
    align_size=0,
    aux_loss_coeff=0.0,
    use_expert_bias=False,
    score_function="softmax",
    bias_init=None,
):
    kwargs = dict(
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOPK,
        intermediate_size=INTER,
        data_parallelism_axes=(FSDP_AXIS,),
        apply_topk_weights_early=apply_topk_weights_early,
        align_size=align_size,
        aux_loss_coeff=aux_loss_coeff,
        use_expert_bias=use_expert_bias,
        score_function=score_function,
        dtype=DTYPE,
    )
    # Custom bias_init lets tests inject a non-zero expert_bias without
    # poking variables['params'] post-init.
    if bias_init is not None:
        kwargs["bias_init"] = bias_init
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
        output, aux = jax.jit(block.apply)(variables, x_sh)
        jax.block_until_ready(output)
    return variables, output, aux


def _grad_step(block, variables, mesh, x, *, include_aux=False):
    """Run jax.grad of mean(out^2) [+ aux if include_aux] vs params."""
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)

        def loss_fn(variables, x):
            output, aux = block.apply(variables, x)
            loss = jnp.mean(output.astype(jnp.float32) ** 2)
            if include_aux and aux is not None:
                loss = loss + aux.astype(jnp.float32)
            return loss

        grads = jax.jit(jax.grad(loss_fn))(variables, x_sh)
        jax.block_until_ready(jax.tree_util.tree_leaves(grads)[0])
        return grads


def _grad_aux_only(block, variables, mesh, x):
    """Jit'd grad of just the aux loss scalar — proves it reaches the
    gate even when no main-output contribution is present."""
    with _ctx(mesh):
        x_sh = _shard_inputs(x, mesh)

        def aux_only(variables, x):
            _, aux = block.apply(variables, x)
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
        id="softmax-topk-early",
    ),
    pytest.param(
        dict(score_function="softmax", align_size=128),
        id="softmax-align128",
    ),
    pytest.param(
        dict(score_function="sigmoid"),
        id="sigmoid",
    ),
    pytest.param(
        dict(score_function="sigmoid", use_expert_bias=True),
        id="sigmoid-bias-zero",
    ),
    pytest.param(
        dict(
            score_function="sigmoid",
            use_expert_bias=True,
            bias_init=_strong_expert_bias_init,
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
            if config.get("use_expert_bias", False)
            else None
        ),
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
            jnp.asarray(params_np["wi_0"]),
            jnp.asarray(params_np["wi_1"]),
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
        grads_te = _grad_step(block, variables, mesh, x)

        # Reference grads via jax.grad over the pure-JAX MoE with the
        # same config.
        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        ref_kwargs = _reference_kwargs_from_config(config, params_np)
        ref_expert_bias = ref_kwargs.pop("expert_bias")

        def loss_fn(params, x):
            out, _ = _pure_jax_moe_reference(
                x,
                params["gate_kernel"],
                params["wi_0"],
                params["wi_1"],
                params["wo"],
                ref_expert_bias,
                num_experts=NUM_EXPERTS,
                num_experts_per_tok=TOPK,
                **ref_kwargs,
            )
            return jnp.mean(out.astype(jnp.float32) ** 2)

        grads_ref = jax.jit(jax.grad(loss_fn))(
            {k: jnp.asarray(v) for k, v in params_np.items() if k != "expert_bias"},
            jnp.asarray(x_np),
        )
        grads_ref_np = {k: np.asarray(jax.device_get(v)) for k, v in grads_ref.items()}

        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            # Per-tensor: finite + non-zero + parity in one pass.
            g_te = _to_global_numpy(_unwrap(grads_te["params"][name]), mesh)
            assert np.all(np.isfinite(g_te)), f"{name} grad has NaN/Inf [config={config}]"
            assert np.any(g_te != 0.0), f"{name} grad identically zero [config={config}]"
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
            jnp.asarray(params_np["wi_0"]),
            jnp.asarray(params_np["wi_1"]),
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
        grads = _grad_step(block, variables, mesh, x, include_aux=True)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_local = np.asarray(
                jax.device_get(_unwrap(grads["params"][name]).addressable_data(0))
            )
            assert np.all(np.isfinite(g_local)), f"{name} grad NaN/Inf under main+aux"
            assert np.any(g_local != 0.0), f"{name} grad zero under main+aux"


class TestTeEpMoEBlockFlax:
    """Flax wrapper end-to-end in one run: shape/dtype/finiteness on the
    forward, numerical parity vs the same reference, and per-tensor
    grad finiteness + non-zeroness."""

    def test_init_apply_parity(self, mesh):
        block = _make_block()
        x = _make_inputs(jax.random.PRNGKey(12))
        variables, output, aux = _init_apply(block, mesh, x, jax.random.PRNGKey(13))

        assert aux is None
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        out_local = np.asarray(jax.device_get(output.addressable_data(0)))
        assert np.all(np.isfinite(out_local))

        params_np = _params_global_numpy(variables, mesh)
        x_np = np.asarray(jax.device_get(x))
        out_te_np = _to_global_numpy(output, mesh)
        out_ref, _ = _pure_jax_moe_reference(
            jnp.asarray(x_np),
            jnp.asarray(params_np["gate_kernel"]),
            jnp.asarray(params_np["wi_0"]),
            jnp.asarray(params_np["wi_1"]),
            jnp.asarray(params_np["wo"]),
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
        )
        np.testing.assert_allclose(
            out_te_np.astype(np.float32),
            np.asarray(jax.device_get(out_ref)).astype(np.float32),
            atol=FWD_ATOL,
            rtol=FWD_RTOL,
        )

        grads = _grad_step(block, variables, mesh, x)
        for name in ("gate_kernel", "wi_0", "wi_1", "wo"):
            g_local = np.asarray(
                jax.device_get(_unwrap(grads["params"][name]).addressable_data(0))
            )
            assert np.all(np.isfinite(g_local)), f"{name} grad NaN/Inf"
            assert np.any(g_local != 0.0), f"{name} grad zero"


# Keep the bootstrap-signature test last in the module (the "ZZZ" prefix
# ensures pytest's alphabetic class ordering picks it last): it
# intentionally mismatches the NCCL EP bootstrap signature, which
# permanently taints the per-process bootstrap cache for the rest of
# the file.
class TestZZZTeEpMoeBootstrap:
    """Per-process NCCL bootstrap re-bootstrap rejection."""

    def test_bootstrap_signature_mismatch_raises(self, mesh):
        block_a = _make_block()
        x_a = _make_inputs(jax.random.PRNGKey(14))
        _init_apply(block_a, mesh, x_a, jax.random.PRNGKey(15))

        # Different hidden dim → different bootstrap signature.
        bigger_hidden = HIDDEN * 2
        x_b = jax.random.normal(
            jax.random.PRNGKey(16), (BATCH, SEQ, bigger_hidden), dtype=DTYPE
        )
        block_b = MoEBlock(
            num_experts=NUM_EXPERTS,
            num_experts_per_tok=TOPK,
            intermediate_size=INTER,
            data_parallelism_axes=(FSDP_AXIS,),
            dtype=DTYPE,
        )
        with pytest.raises(ValueError, match="bootstrapped"):
            _init_apply(block_b, mesh, x_b, jax.random.PRNGKey(17))
