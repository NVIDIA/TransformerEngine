# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""DeepSeekV3Layer with expert parallelism over all ranks: forward + backward timing.

One process per GPU, launched via run_deepseek_v3_layer_ep.sh (torchrun). Every rank
holds ``--num-local-experts`` routed experts; tokens are exchanged with NCCL EP.
Timed iterations run inside a ``torch.cuda.profiler`` window, so
``nsys profile -c cudaProfilerApi --capture-range-end=stop torchrun ...`` records
only them.
"""

import argparse
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe
from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize, release_symm_mem_pool
from transformer_engine.pytorch.models import DeepSeekV3Layer, DeepSeekV3MoE


def _parse_args():
    p = argparse.ArgumentParser(description="DeepSeekV3Layer EP example (fwd + bwd)")
    p.add_argument("--tokens-per-rank", type=int, default=4096)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--moe-ffn", type=int, default=1024)
    p.add_argument("--num-local-experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--q-lora-rank", type=int, default=512)
    p.add_argument("--kv-lora-rank", type=int, default=256)
    p.add_argument("--qk-nope-head-dim", type=int, default=64)
    p.add_argument("--qk-rope-head-dim", type=int, default=32)
    p.add_argument("--v-head-dim", type=int, default=64)
    p.add_argument(
        "--dsv3",
        action="store_true",
        help=(
            "DeepSeek-V3 layer dims (hidden 7168, 128 heads, MLA 1536/512/128/64/128, expert ffn"
            " 2048)."
        ),
    )
    p.add_argument("--recipe", choices=["none", "mxfp8"], default="none")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args()
    if args.dsv3:
        args.hidden, args.num_heads, args.moe_ffn = 7168, 128, 2048
        args.q_lora_rank, args.kv_lora_rank = 1536, 512
        args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim = 128, 64, 128
    return args


def _autocast(name):
    if name == "none":
        return nullcontext()
    return te.autocast(enabled=True, recipe=te_recipe.MXFP8BlockScaling())


def main():
    """Build the layer, run warmup + timed fwd/bwd iterations, print throughput on rank 0."""
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()

    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor < 90:
        if rank == 0:
            print(f"SKIPPED: NCCL EP requires SM>=90 (got SM{major}{minor})")
        dist.destroy_process_group()
        return 0

    ep_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    num_experts = args.num_local_experts * world_size
    ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.hidden,
        num_topk=args.topk,
        recv_capacity_per_rank=DeepSeekV3MoE.ep_recv_capacity(
            world_size, args.tokens_per_rank, args.topk, args.num_local_experts
        ),
    )

    torch.manual_seed(0)
    layer = DeepSeekV3Layer(
        args.hidden,
        args.num_heads,
        num_experts=num_experts,
        moe_ffn_hidden_size=args.moe_ffn,
        shared_expert_ffn_hidden_size=args.moe_ffn,
        topk=args.topk,
        params_dtype=torch.bfloat16,
        ep_group=ep_group,
        ep_max_tokens_per_rank=args.tokens_per_rank,
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
    )
    seq = args.tokens_per_rank // 4
    x = torch.randn(seq, 4, args.hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    def step():
        with _autocast(args.recipe):
            out = layer(x)
        out.backward(torch.ones_like(out))
        x.grad = None
        return out

    for _ in range(args.warmup):
        out = step()
    finite = bool(torch.isfinite(out).all())
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.profiler.start()
    start = time.perf_counter()
    for i in range(args.iters):
        with torch.cuda.nvtx.range(f"iter{i}"):
            step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / args.iters * 1e3
    torch.cuda.profiler.stop()
    dist.barrier()

    if rank == 0:
        tok_s = args.tokens_per_rank * world_size / (ms / 1e3)
        print(
            f"DeepSeekV3Layer EP: ranks={world_size} experts={num_experts} topk={args.topk} "
            f"tokens/rank={args.tokens_per_rank} hidden={args.hidden} recipe={args.recipe} "
            f"fused_mlp={os.environ.get('NVTE_CUTEDSL_FUSED_GROUPED_MLP', '0')} "
            f"fwd+bwd {ms:.3f} ms/iter ({tok_s / 1e6:.2f} Mtok/s) finite={finite}",
            flush=True,
        )
    ep_finalize()
    release_symm_mem_pool()
    dist.destroy_process_group()
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
