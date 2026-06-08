# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""End-to-end MoE example: dispatch -> batched expert linear -> combine, fwd + bwd.

One process per GPU. Run via run_test_ep.sh.
"""

import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from transformer_engine.jax.ep import EpLayerConfig, ep_bootstrap, ep_dispatch, ep_combine
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


# ── Setup ───────────────────────────────────────────────────────────────────


def _parse_args():
    p = argparse.ArgumentParser(description="TE-JAX EP MoE example (fwd + bwd)")
    p.add_argument("--coordinator-address", required=True)
    p.add_argument("--process-id", type=int, required=True)
    p.add_argument("--num-processes", type=int, required=True)
    p.add_argument("--num-tokens", type=int, default=8, help="Per-rank token count.")
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--hidden-out", type=int, default=32)
    p.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="Total experts across the EP group. Default: num_processes.",
    )
    p.add_argument("--dp-size", type=int, default=None, help="Default: num_procs // ep_size.")
    p.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Verify fwd+bwd against a single-rank numpy reference.",
    )
    return p.parse_args()


def _distributed_init(args):
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=[args.process_id],
    )
    assert (
        jax.local_device_count() == 1
    ), f"EP example requires 1 GPU per process; got {jax.local_device_count()}"


def _build_mesh_and_resource(args):
    """Pick a (2, 2) mesh by default. Override via --dp-size."""
    n = args.num_processes
    if n < 4:
        raise ValueError(f"num_processes ({n}) must be >= 4 for NCCL EP")
    if args.dp_size is None:
        if n != 4:
            raise ValueError(
                f"default mesh expects exactly 4 ranks (got {n}); pass --dp-size to override"
            )
        args.dp_size = 2
    assert n % args.dp_size == 0, f"num_processes={n} not divisible by dp_size={args.dp_size}"
    args.ep_size = n // args.dp_size
    if args.num_experts is None:
        args.num_experts = args.num_processes
    assert args.num_experts % args.ep_size == 0
    args.num_local_experts = args.num_experts // args.ep_size
    args.recv_capacity_per_rank = args.ep_size * args.num_tokens * args.top_k

    devs = np.asarray(jax.devices()).reshape(args.dp_size, args.ep_size)
    mesh = Mesh(devs, ("dp", "ep"))
    mr = MeshResource(dp_resource="dp", ep_resource="ep")
    return mesh, mr


def _make_routing(dp_color, num_tokens, top_k, num_experts, num_local_experts):
    """Deterministic routing: topk_idx[t, k] = (dp_color*NLE + t*K + k) % E."""
    topk_idx = np.empty((num_tokens, top_k), dtype=np.int32)
    for t in range(num_tokens):
        for k in range(top_k):
            topk_idx[t, k] = (dp_color * num_local_experts + t * top_k + k) % num_experts
    return topk_idx


def _make_inputs(args):
    """Build 3D ``[B, S, H]`` arrays sharded ``(("dp","ep"), None, None)``.

    B = num_processes (sharded across the compound (dp,ep) axis so each rank
    holds one slot); S = args.num_tokens. Global numpy views (rank-0
    reference) are kept 2D for the legacy reference implementation.
    """
    T, K, H, H_out = args.num_tokens, args.top_k, args.hidden, args.hidden_out
    E = args.num_experts
    dp_size = args.dp_size
    ep_size = args.ep_size
    num_procs = args.num_processes
    dp_color = args.process_id // ep_size

    rng_dp = np.random.default_rng(seed=42 + dp_color)
    tokens_np = (rng_dp.standard_normal((T, H), dtype=np.float32) * 0.5).astype(np.float32)
    topk_idx_np = _make_routing(dp_color, T, K, E, args.num_local_experts)
    w_np = np.full((T, K), 1.0 / K, dtype=np.float32)

    tokens_global_np = np.concatenate(
        [
            (
                np.random.default_rng(seed=42 + c).standard_normal((T, H), dtype=np.float32) * 0.5
            ).astype(np.float32)
            for c in range(dp_size)
        ],
        axis=0,
    )
    topk_idx_global_np = np.concatenate(
        [_make_routing(c, T, K, E, args.num_local_experts) for c in range(dp_size)], axis=0
    )
    w_global_np = np.full((dp_size * T, K), 1.0 / K, dtype=np.float32)

    # Same seed on every rank → identical kernel array everywhere.
    rng = np.random.default_rng(seed=42)
    kernels_np = (rng.standard_normal((E, H, H_out), dtype=np.float32) * (1.0 / np.sqrt(H))).astype(
        np.float32
    )

    # Each rank contributes one [1, T, ...] slab; the global shape is
    # [num_procs, T, ...] sharded on the first dim across (dp, ep).
    mesh = args.mesh
    dpep_spec = NamedSharding(mesh, PartitionSpec(("dp", "ep"), None, None))
    tokens = jax.make_array_from_process_local_data(
        dpep_spec, tokens_np[None, :, :].astype(np.float32), (num_procs, T, H)
    ).astype(jnp.bfloat16)
    topk_idx = jax.make_array_from_process_local_data(
        dpep_spec, topk_idx_np[None, :, :], (num_procs, T, K)
    )
    topk_w = jax.make_array_from_process_local_data(dpep_spec, w_np[None, :, :], (num_procs, T, K))
    kernels = jnp.asarray(kernels_np, dtype=jnp.bfloat16)
    return (
        tokens_global_np,
        topk_idx_global_np,
        w_global_np,
        kernels_np,
        tokens,
        topk_idx,
        topk_w,
        kernels,
    )


# ── MoE step ────────────────────────────────────────────────────────────────


def _batched_expert_linear(recv_tokens, kernels, num_local_experts, dp_size, ep_size):
    """Per-expert linear. ``recv_tokens`` is 3D ``[num_procs, recv_pr, H]``
    (compound (dp,ep) leading); ``kernels`` is 4D ``[ep_size, NLE, H, H_out]``,
    broadcast over the dp axis. Output matches ``recv_tokens``' 3D layout
    with ``H_out`` in place of ``H``."""
    num_procs, recv_pr, H = recv_tokens.shape
    H_out = kernels.shape[-1]
    slots_per_expert = recv_pr // num_local_experts
    # [num_procs, recv_pr, H] -> [dp, ep, NLE, slots, H]
    grouped = recv_tokens.reshape(dp_size, ep_size, num_local_experts, slots_per_expert, H)
    # Contract H; batch over (ep, NLE) which are present on both sides.
    out = jax.lax.dot_general(
        grouped,
        kernels.astype(grouped.dtype),
        dimension_numbers=(((4,), (2,)), ((1, 2), (0, 1))),
    )
    # Output dim order from dot_general: batch dims first, then remaining lhs, rhs.
    # batch=(ep,NLE), lhs_remaining=(dp,slots), rhs_remaining=(H_out,)
    # → shape [ep, NLE, dp, slots, H_out]. Permute to [dp, ep, NLE, slots, H_out].
    out = jnp.transpose(out, (2, 0, 1, 3, 4))
    return out.reshape(num_procs, recv_pr, H_out)


def _moe_step(args, topk_idx, tokens, topk_w, kernels):
    """Jit'd MoE step: dispatch -> batched per-expert linear -> combine.

    Inputs are 3D ``[B, S, H]`` with the first dim compound-sharded across
    ``("dp","ep")``. Combine returns the same 3D shape.
    """
    B = args.num_processes
    S = args.num_tokens
    NLE = args.num_local_experts
    dp_size, ep_size = args.dp_size, args.ep_size
    mesh = args.mesh
    in_spec = PartitionSpec(("dp", "ep"), None, None)  # [B, S, ...]
    ep3 = PartitionSpec(("dp", "ep"), None, None)  # [num_procs, recv_pr, H]
    ep2 = PartitionSpec(("dp", "ep"), None)  # [num_procs, recv_pr]
    # Kernels are EP-replicated across dp colors; shard only the ep-rank axis.
    kernel_spec = PartitionSpec("ep", None, None, None)

    kernels = kernels.reshape(ep_size, NLE, *kernels.shape[1:])
    ep_handle = EpLayerConfig(top_k=args.top_k, dispatch_output_per_expert_alignment=16)

    @jax.jit
    def step(topk_idx, tokens, topk_w, local_kernels):
        topk_idx = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(mesh, in_spec))
        tokens = jax.lax.with_sharding_constraint(tokens, NamedSharding(mesh, in_spec))
        topk_w = jax.lax.with_sharding_constraint(topk_w, NamedSharding(mesh, in_spec))
        local_kernels = jax.lax.with_sharding_constraint(
            local_kernels, NamedSharding(mesh, kernel_spec)
        )
        recv_tokens, recv_topk_w, handle_mem, _tc = ep_dispatch(
            ep_handle, topk_idx, tokens, topk_w, args.recv_capacity_per_rank
        )
        recv_tokens = jax.lax.with_sharding_constraint(recv_tokens, NamedSharding(mesh, ep3))
        recv_topk_w = jax.lax.with_sharding_constraint(recv_topk_w, NamedSharding(mesh, ep2))
        expert_out = _batched_expert_linear(recv_tokens, local_kernels, NLE, dp_size, ep_size)
        expert_out = jax.lax.with_sharding_constraint(expert_out, NamedSharding(mesh, ep3))
        # ep_combine is unweighted: pre-multiply by recv_topk_w and zero
        # padded slots (recv_topk_w == 0) before the scatter-sum.
        mask = (recv_topk_w != 0).astype(jnp.float32)[..., None]
        weighted = (
            expert_out.astype(jnp.float32) * recv_topk_w[..., None] * mask
        ).astype(expert_out.dtype)
        weighted = jax.lax.with_sharding_constraint(weighted, NamedSharding(mesh, ep3))
        return ep_combine(
            ep_handle,
            handle_mem,
            _tc,
            weighted,
            num_local_tokens=(B, S),
            out_sharding=(("dp", "ep"), None, None),
        )

    return step(topk_idx, tokens, topk_w, kernels)


# ── Reference (numerical check) ─────────────────────────────────────────────


def _reference_moe(tokens, topk_idx, topk_w, kernels):
    """Single-rank dense MoE reference. tokens [T, H], output [T, H_out]."""
    T, K = topk_idx.shape
    H_out = kernels.shape[-1]
    out = np.zeros((T, H_out), dtype=np.float32)
    for t in range(T):
        tok = tokens[t].astype(np.float32)
        for k in range(K):
            e = int(topk_idx[t, k])
            out[t] += float(topk_w[t, k]) * (tok @ kernels[e].astype(np.float32))
    return out


def _reference_grad(tokens, topk_idx, topk_w, kernels):
    """d/dtokens of 0.5 * sum(ref_out**2) — used by --check to validate bwd."""
    T, K = topk_idx.shape
    H = tokens.shape[-1]
    ref_out = _reference_moe(tokens, topk_idx, topk_w, kernels)
    grad = np.zeros((T, H), dtype=np.float32)
    for t in range(T):
        mixed = np.zeros_like(kernels[0])
        for k in range(K):
            mixed = mixed + float(topk_w[t, k]) * kernels[int(topk_idx[t, k])]
        grad[t] = ref_out[t] @ mixed.T
    return ref_out, grad


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    args = _parse_args()
    _distributed_init(args)

    dev = jax.local_devices()[0]
    cap = getattr(dev, "compute_capability", None)
    if cap is not None:
        major, minor = (int(x) for x in str(cap).split("."))
        if major * 10 + minor < 90:
            print(f"[ep_moe] SKIPPED: NCCL EP requires SM>=90 (got SM{major}{minor})")
            return

    args.mesh, args.mr = _build_mesh_and_resource(args)

    with args.mesh, global_shard_guard(args.mr):
        ep_bootstrap(
            world_size=args.num_processes,
            rank=args.process_id,
            ep_size=args.ep_size,
            num_experts=args.num_experts,
            max_tokens_per_rank=args.num_tokens,
            recv_capacity_per_rank=args.recv_capacity_per_rank,
            hidden_dim=args.hidden,
        )

        (
            tokens_global_np,
            topk_idx_global_np,
            w_global_np,
            kernels_np,
            tokens,
            topk_idx,
            topk_w,
            kernels,
        ) = _make_inputs(args)

        def loss_fn(toks, idx, w, kern):
            out = _moe_step(args, idx, toks, w, kern)
            return 0.5 * (out.astype(jnp.float32) ** 2).sum(), out

        (loss, out_fwd), grad_tokens = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(
            tokens, topk_idx, topk_w, kernels
        )
        grad_tokens.block_until_ready()
        out_fwd.block_until_ready()

        if args.process_id == 0:
            print(
                f"[ep_moe] loss={float(loss):.4f} grad_tokens.shape={grad_tokens.shape} "
                f"dp={args.dp_size} ep={args.ep_size} "
                f"num_experts={args.num_experts} recv_pr={args.recv_capacity_per_rank}"
            )

        if args.check:

            def _norm(spec, ndim):
                return tuple(spec) + (None,) * (ndim - len(spec))

            # JAX may collapse a size-1 mesh axis: when dp_size==1 the spec can
            # appear as ``(("dp","ep"),...)`` or ``("ep",...)``. Accept both.
            if args.dp_size > 1:
                acceptable_specs = ((("dp", "ep"), None, None),)
            else:
                acceptable_specs = ((("dp", "ep"), None, None), ("ep", None, None))
            assert (
                _norm(out_fwd.sharding.spec, out_fwd.ndim) in acceptable_specs
            ), f"out_fwd.sharding.spec={out_fwd.sharding.spec} (expected one of {acceptable_specs})"
            assert _norm(grad_tokens.sharding.spec, grad_tokens.ndim) in acceptable_specs, (
                f"grad_tokens.sharding.spec={grad_tokens.sharding.spec}"
                f" (expected one of {acceptable_specs})"
            )

            replicated = NamedSharding(args.mesh, jax.sharding.PartitionSpec())
            out_global = jax.jit(lambda x: jax.lax.with_sharding_constraint(x, replicated))(out_fwd)
            grad_global = jax.jit(lambda x: jax.lax.with_sharding_constraint(x, replicated))(
                grad_tokens
            )
            out_global.block_until_ready()
            grad_global.block_until_ready()

            ref_out, ref_grad = _reference_grad(
                tokens_global_np, topk_idx_global_np, w_global_np, kernels_np
            )
            ref_loss = 0.5 * float((ref_out.astype(np.float32) ** 2).sum())
            # 3D global ``[num_procs, S, H]`` with num_procs = dp * ep. Each EP
            # column in a DP color sees identical inputs (and produces identical
            # outputs), so collapse the ep dim to one replica before flattening
            # to 2D against the dp-only reference.
            dp_size, ep_size = args.dp_size, args.ep_size
            global_out = (
                np.asarray(out_global.addressable_shards[0].data.astype(jnp.float32))
                .reshape(dp_size, ep_size, -1, ref_out.shape[-1])[:, 0]
                .reshape(-1, ref_out.shape[-1])
            )
            global_grad = (
                np.asarray(grad_global.addressable_shards[0].data.astype(jnp.float32))
                .reshape(dp_size, ep_size, -1, ref_grad.shape[-1])[:, 0]
                .reshape(-1, ref_grad.shape[-1])
            )
            if args.process_id == 0:
                fwd_diff = np.abs(global_out - ref_out)
                grad_diff = np.abs(global_grad - ref_grad)
                print(
                    f"[ep_moe] DEBUG loss={float(loss):.4f} ref_loss(global)={ref_loss:.4f} "
                    f"ratio={float(loss) / max(ref_loss, 1e-9):.4f} (expected ~1.0)"
                )
                print(f"[ep_moe] DEBUG fwd  max-abs-diff per row: {fwd_diff.max(axis=1)}")
                print(f"[ep_moe] DEBUG grad max-abs-diff per row: {grad_diff.max(axis=1)}")
            np.testing.assert_allclose(
                global_out,
                ref_out,
                rtol=5e-2,
                atol=5e-2,
                err_msg=f"rank {args.process_id}: fwd mismatch",
            )
            np.testing.assert_allclose(
                global_grad,
                ref_grad,
                rtol=5e-2,
                atol=5e-2,
                err_msg=f"rank {args.process_id}: bwd mismatch",
            )
            if args.process_id == 0:
                print(f"[ep_moe] --check PASSED (ref_out.sum()={float(ref_out.sum()):.4f})")


if __name__ == "__main__":
    main()
    sys.exit(0)
