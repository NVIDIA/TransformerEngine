# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""JAX EP perf bench — dispatch/combine (raw fwd + custom_vjp wrapper) on a 1DP x EP mesh.

One process per GPU; launch via run_ep_bench.sh. Each stage is jitted and
timed separately with NVTX ranges (prepare runs once outside the loop).
Rank-0 prints mean wall in us; nsys / --xplane attribute kernels per stage.
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from transformer_engine.jax.cpp_extensions import ep as tex_ep
from transformer_engine.jax.ep import EpLayerConfig, ep_bootstrap, ep_dispatch, ep_combine
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


def _parse_args():
    p = argparse.ArgumentParser(description="TE-JAX EP perf bench (dispatch_fwd + combine_fwd)")
    p.add_argument("--coordinator-address", required=True)
    p.add_argument("--process-id", type=int, required=True)
    p.add_argument("--num-processes", type=int, required=True)
    p.add_argument("--tokens-per-rank", type=int, default=8192)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--num-experts", type=int, default=256)
    p.add_argument("--dp-size", type=int, default=1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument(
        "--max-num-sms",
        type=int,
        default=0,
        help="Max SMs for dispatch / combine / preprocess kernels (0 = auto).",
    )
    p.add_argument(
        "--mode-label",
        default=None,
        help="Optional label suffix for NVTX range names so nsys can partition kernels.",
    )
    p.add_argument(
        "--second-step",
        action="store_true",
        help=(
            "Time only the 2nd step (1 warmup iter, 1 timed iter). Use to isolate "
            "JIT-cache-warm-but-no-steady-state-batching overhead from steady-state perf."
        ),
    )
    p.add_argument(
        "--xplane",
        default=None,
        help="If set, jax.profiler dumps an XPlane trace into this dir (rank 0 only).",
    )
    return p.parse_args()


def _distributed_init(args):
    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_processes,
        process_id=args.process_id,
        local_device_ids=[args.process_id],
    )


def _build_mesh(args):
    n = args.num_processes
    assert n % args.dp_size == 0
    ep = n // args.dp_size
    devs = np.asarray(jax.devices()).reshape(args.dp_size, ep)
    return Mesh(devs, ("dp", "ep")), ep


def _make_inputs(args, ep_size):
    """Round-robin routing, uniform top-k weights; each rank sees ``args.tokens_per_rank`` tokens."""
    n = args.num_processes
    T = args.tokens_per_rank
    H = args.hidden
    K = args.top_k
    E = args.num_experts
    del ep_size

    topk_idx = np.empty((n * T, K), dtype=np.int32)
    for t in range(n * T):
        for k in range(K):
            topk_idx[t, k] = (t * K + k) % E
    topk_idx = jnp.asarray(topk_idx)
    topk_w = jnp.full((n * T, K), 1.0 / K, dtype=jnp.float32)
    tokens = jnp.asarray(
        np.random.default_rng(0).standard_normal((n * T, H), dtype=np.float32) * 0.5,
        dtype=jnp.bfloat16,
    )
    return tokens, topk_idx, topk_w


def main():
    args = _parse_args()
    _distributed_init(args)
    mesh, ep_size = _build_mesh(args)
    mr = MeshResource(dp_resource="dp", ep_resource="ep")
    rank = args.process_id

    local_experts = args.num_experts // ep_size
    recv_capacity_per_rank = args.num_processes * args.tokens_per_rank * args.top_k // 2

    if rank == 0:
        print(
            f"[ep_bench] world={args.num_processes} dp={args.dp_size} ep={ep_size}"
            f" T={args.tokens_per_rank} H={args.hidden} K={args.top_k}"
            f" E={args.num_experts} (local={local_experts}) recv_pr={recv_capacity_per_rank}"
            + (f" mode={args.mode_label}" if args.mode_label else ""),
            flush=True,
        )

    nvtx_suffix = f"[{args.mode_label}]" if args.mode_label else ""

    in_spec = PartitionSpec(("dp", "ep"), None)
    ep_spec_3d = PartitionSpec(("dp", "ep"), None, None)
    ep_spec_2d = PartitionSpec(("dp", "ep"), None)
    out_spec = (("dp", "ep"), None)
    T_global = args.num_processes * args.tokens_per_rank

    with mesh, global_shard_guard(mr):
        ep_bootstrap(
            world_size=args.num_processes,
            rank=rank,
            num_experts=args.num_experts,
            max_tokens_per_rank=args.tokens_per_rank,
            recv_capacity_per_rank=recv_capacity_per_rank,
            hidden_dim=args.hidden,
            max_num_sms=args.max_num_sms,
        )

        tokens, topk_idx, topk_w = _make_inputs(args, ep_size)
        idx_s = jax.lax.with_sharding_constraint(topk_idx, NamedSharding(mesh, in_spec))
        tok_s = jax.lax.with_sharding_constraint(tokens, NamedSharding(mesh, in_spec))
        w_s = jax.lax.with_sharding_constraint(topk_w, NamedSharding(mesh, in_spec))

        cfg = EpLayerConfig(top_k=args.top_k, dispatch_output_per_expert_alignment=16)

        @jax.jit
        def run_prepare(idx):
            tc, hm = tex_ep.ep_prepare(cfg, idx)
            return tc, hm

        @jax.jit
        def run_dispatch(hm, idx, toks, w):
            recv_t, recv_w = tex_ep.ep_dispatch_fwd(cfg, hm, idx, toks, w, recv_capacity_per_rank)
            recv_t = jax.lax.with_sharding_constraint(recv_t, NamedSharding(mesh, ep_spec_3d))
            recv_w = jax.lax.with_sharding_constraint(recv_w, NamedSharding(mesh, ep_spec_2d))
            return recv_t, recv_w

        @jax.jit
        def run_dispatch_vjp(idx, toks, w):
            recv_t, recv_w, _hm, _tc = ep_dispatch(cfg, idx, toks, w, recv_capacity_per_rank)
            recv_t = jax.lax.with_sharding_constraint(recv_t, NamedSharding(mesh, ep_spec_3d))
            recv_w = jax.lax.with_sharding_constraint(recv_w, NamedSharding(mesh, ep_spec_2d))
            return recv_t, recv_w

        @jax.jit
        def run_combine(hm, recv_t):
            out = tex_ep.ep_combine_fwd(
                cfg,
                hm,
                recv_t,
                T_global,
                out_partition_spec=out_spec,
            )
            return out

        @jax.jit
        def run_combine_vjp(hm, tc, recv_t):
            # ep_combine is unweighted; bench feeds expert_out directly (caller
            # would otherwise pre-multiply by recv_topk_weights + mask).
            out = ep_combine(cfg, hm, tc, recv_t, T_global, out_sharding=out_spec)
            return out

        tc, handle_mem = run_prepare(idx_s)
        tc.block_until_ready()
        handle_mem.block_until_ready()

        recv_t0, recv_w0 = run_dispatch(handle_mem, idx_s, tok_s, w_s)
        recv_t0.block_until_ready()
        recv_w0.block_until_ready()

        warmup_n = 1 if args.second_step else args.warmup
        iters_n = 1 if args.second_step else args.iters

        for _ in range(warmup_n):
            r, _rw = run_dispatch(handle_mem, idx_s, tok_s, w_s)
            r.block_until_ready()
            o = run_combine(handle_mem, r)
            o.block_until_ready()
        run_dispatch_vjp(idx_s, tok_s, w_s)[0].block_until_ready()
        run_combine_vjp(handle_mem, tc, recv_t0).block_until_ready()

        if args.xplane and rank == 0:
            os.makedirs(args.xplane, exist_ok=True)
            jax.profiler.start_trace(args.xplane)

        try:
            import nvtx as _nvtx

            def _push(name):
                _nvtx.push_range(message=name)

            def _pop():
                _nvtx.pop_range()

        except ImportError:

            def _push(name):
                pass

            def _pop():
                pass

        def _time_stage_wall_us(name, fn):
            # First timed iter still carries an autotune outlier even after JIT
            # warmup; run iters_n + 1, drop iter 0 from the average, and push
            # the NVTX range AFTER iter 0 so nsys' nvtx_kern_sum excludes the
            # outlier too.
            total_ns = 0
            counted = 0
            for i in range(iters_n + 1):
                if i == 1:
                    _push(f"{name}{nvtx_suffix}")
                t0 = time.perf_counter_ns()
                fn()
                dt = time.perf_counter_ns() - t0
                if i == 0:
                    continue
                total_ns += dt
                counted += 1
            _pop()
            return total_ns / 1e3 / counted

        def _do_dispatch():
            r, _ = run_dispatch(handle_mem, idx_s, tok_s, w_s)
            r.block_until_ready()

        def _do_dispatch_vjp():
            r, _ = run_dispatch_vjp(idx_s, tok_s, w_s)
            r.block_until_ready()

        def _do_combine():
            o = run_combine(handle_mem, recv_t0)
            o.block_until_ready()

        def _do_combine_vjp():
            o = run_combine_vjp(handle_mem, tc, recv_t0)
            o.block_until_ready()

        d_wall_us = _time_stage_wall_us("dispatch_fwd", _do_dispatch)
        dv_wall_us = _time_stage_wall_us("ep_dispatch_vjp", _do_dispatch_vjp)
        c_wall_us = _time_stage_wall_us("combine_fwd", _do_combine)
        cv_wall_us = _time_stage_wall_us("ep_combine_vjp", _do_combine_vjp)

        if args.xplane and rank == 0:
            jax.profiler.stop_trace()

    if rank == 0:
        label = f" [{args.mode_label}]" if args.mode_label else ""
        print("", flush=True)
        print(f"| stage             | mean wall (us){label} |", flush=True)
        print("|-------------------|---------------:|", flush=True)
        print(f"| dispatch_fwd      | {d_wall_us:14.1f} |", flush=True)
        print(f"| ep_dispatch_vjp   | {dv_wall_us:14.1f} |", flush=True)
        print(f"| combine_fwd       | {c_wall_us:14.1f} |", flush=True)
        print(f"| ep_combine_vjp    | {cv_wall_us:14.1f} |", flush=True)
        print(f"| (dispatch vjp-fwd)| {dv_wall_us - d_wall_us:14.1f} |", flush=True)
        print(f"| (combine  vjp-fwd)| {cv_wall_us - c_wall_us:14.1f} |", flush=True)
        print("", flush=True)
        print(
            "[ep_bench] kernel breakout: see nsys nvtx_kern_sum output below "
            "(produced by run_ep_bench.sh --nsys).",
            flush=True,
        )

    # Under nsys: force cudaDeviceReset() to drain CUPTI's in-process kernel
    # records into the .nsys-rep, then os._exit to skip JAX's coord-service
    # watchdog. The reset crashes during NCCL EP context teardown, so we only
    # take this path when the launcher opts in via EP_BENCH_FLUSH_CUPTI=1.
    if os.environ.get("EP_BENCH_FLUSH_CUPTI", "0") == "1":
        try:
            import ctypes

            cudart = ctypes.CDLL("libcudart.so")
            cudart.cudaDeviceSynchronize()
            cudart.cudaDeviceReset()
        except Exception:
            pass
        time.sleep(0.5)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
