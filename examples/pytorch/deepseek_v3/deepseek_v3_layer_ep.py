# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""DeepSeekV3Layer with expert parallelism over all ranks: forward + backward timing.

One process per GPU, launched via run_deepseek_v3_layer_ep.sh (torchrun). Every rank
holds ``--num-local-experts`` routed experts. ``--impl te`` exchanges tokens with NCCL EP
and runs the experts as one grouped GEMM (DeepSeekV3MoE); ``--impl naive`` is a plain
PyTorch MoE (all_to_all_single + a Python loop over experts) dropped into the same layer;
``--impl naive_grouped`` keeps the all_to_all but runs the experts as one TE grouped GEMM.
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
import torch.nn.functional as F
from torch.distributed.nn.functional import all_to_all_single

import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe
from transformer_engine.pytorch.ep import ep_bootstrap, ep_finalize, release_symm_mem_pool
from transformer_engine.pytorch.models import DeepSeekV3Layer, DeepSeekV3MoE


def _parse_args(argv=None):
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
    p.add_argument("--impl", choices=["te", "naive", "naive_grouped"], default="te")
    p.add_argument("--recipe", choices=["none", "mxfp8"], default="none")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=10)
    args = p.parse_args(argv)
    if args.warmup < 0:
        p.error("--warmup must be non-negative")
    if args.iters <= 0:
        p.error("--iters must be positive")
    if args.tokens_per_rank <= 0 or args.tokens_per_rank % 4:
        p.error("--tokens-per-rank must be a positive multiple of 4")
    if args.impl != "te" and args.recipe != "none":
        p.error("--recipe mxfp8 is only supported with --impl te")
    if args.dsv3:
        args.hidden, args.num_heads, args.moe_ffn = 7168, 128, 2048
        args.q_lora_rank, args.kv_lora_rank = 1536, 512
        args.qk_nope_head_dim, args.qk_rope_head_dim, args.v_head_dim = 128, 64, 128
    return args


def _autocast(name):
    if name == "none":
        return nullcontext()
    return te.autocast(enabled=True, recipe=te_recipe.MXFP8BlockScaling())


def _check_finite(tensors, device):
    finite = torch.ones((), dtype=torch.int32, device=device)
    for tensor in tensors:
        if tensor is None:
            finite.zero_()
        else:
            finite.mul_(torch.isfinite(tensor).all())
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    return bool(finite.item())


class NaiveMoE(torch.nn.Module):
    """DeepSeek-style MoE with torch all_to_all dispatch/combine: sigmoid top-k router with
    expert bias, a shared expert, and experts either as a Python loop of dense SwiGLU MLPs or
    (``grouped=True``) as one TE grouped GEMM stack."""

    def __init__(self, hidden, ffn, num_experts, topk, ep_group, shared_ffn, dtype, grouped=False):
        super().__init__()
        self.grouped = grouped
        self.hidden, self.topk, self.group = hidden, topk, ep_group
        self.ws, self.rank = dist.get_world_size(ep_group), dist.get_rank(ep_group)
        self.num_experts, self.local = num_experts, num_experts // self.ws
        self.gate = torch.nn.Linear(hidden, num_experts, bias=False, dtype=dtype, device="cuda")
        self.register_buffer("expert_bias", torch.zeros(num_experts, device="cuda"))
        std = hidden**-0.5
        if grouped:
            self.experts = te.ops.Sequential(
                te.ops.GroupedLinear(self.local, hidden, 2 * ffn, bias=False, dtype=dtype),
                te.ops.ScaledSwiGLU(glu_interleave_size=32),
                te.ops.GroupedLinear(self.local, ffn, hidden, bias=False, dtype=dtype),
            )
        else:
            self.w1 = torch.nn.Parameter(
                torch.randn(self.local, 2 * ffn, hidden, dtype=dtype, device="cuda") * std
            )
            self.w2 = torch.nn.Parameter(
                torch.randn(self.local, hidden, ffn, dtype=dtype, device="cuda") * ffn**-0.5
            )
        self.shared_w1 = torch.nn.Linear(
            hidden, 2 * shared_ffn, bias=False, dtype=dtype, device="cuda"
        )
        self.shared_w2 = torch.nn.Linear(shared_ffn, hidden, bias=False, dtype=dtype, device="cuda")

    @staticmethod
    def _swiglu(h):
        a, g = h.chunk(2, dim=-1)
        return F.silu(a) * g

    def forward(self, hidden_states):
        x = hidden_states.reshape(-1, self.hidden)
        scores = torch.sigmoid(self.gate(x).float())
        _, idx = torch.topk(scores + self.expert_bias, self.topk, dim=-1)
        probs = scores.gather(1, idx)
        probs = probs / probs.sum(-1, keepdim=True) * 2.5
        # Dispatch: sort (token, expert) pairs by destination rank, exchange counts, all_to_all.
        flat_e, flat_p = idx.reshape(-1), probs.reshape(-1)
        tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.topk)
        order = torch.argsort(flat_e // self.local, stable=True)
        flat_e, flat_p, tok = flat_e[order], flat_p[order], tok[order]
        send = torch.bincount(flat_e // self.local, minlength=self.ws)
        recv = torch.empty_like(send)
        dist.all_to_all_single(recv, send, group=self.group)
        send, recv = send.tolist(), recv.tolist()
        n_recv = sum(recv)
        x_recv = all_to_all_single(
            torch.empty(n_recv, self.hidden, dtype=x.dtype, device=x.device),
            x[tok],
            recv,
            send,
            group=self.group,
        )
        e_recv = torch.empty(n_recv, dtype=flat_e.dtype, device=x.device)
        p_recv = torch.empty(n_recv, dtype=flat_p.dtype, device=x.device)
        dist.all_to_all_single(e_recv, flat_e.contiguous(), recv, send, group=self.group)
        p_recv = all_to_all_single(p_recv, flat_p.contiguous(), recv, send, group=self.group)
        local_e = e_recv - self.rank * self.local
        if self.grouped:
            # Experts: sort received rows by local expert, one grouped GEMM stack.
            by_expert = torch.argsort(local_e, stable=True)
            counts = torch.bincount(local_e, minlength=self.local)
            y_sorted = self.experts(
                x_recv[by_expert], counts, p_recv[by_expert].to(x.dtype), counts
            )
            y_recv = torch.empty_like(x_recv).index_copy(0, by_expert, y_sorted)
        else:
            # Experts: one dense SwiGLU MLP per local expert.
            y_recv = torch.zeros_like(x_recv)
            for e in range(self.local):
                sel = (local_e == e).nonzero().squeeze(1)
                if sel.numel() == 0:
                    continue
                h = self._swiglu(F.linear(x_recv[sel], self.w1[e])) * p_recv[sel, None].to(x.dtype)
                y_recv = y_recv.index_copy(0, sel, F.linear(h, self.w2[e]))
        # Combine: reverse all_to_all, sum the top-k contributions per token.
        y = all_to_all_single(torch.empty_like(x[tok]), y_recv, send, recv, group=self.group)
        out = torch.zeros_like(x).index_add(0, tok, y)
        out = out + self.shared_w2(self._swiglu(self.shared_w1(x)))
        return out.view_as(hidden_states)


def main():
    """Build the layer, run warmup + timed fwd/bwd iterations, print throughput on rank 0."""
    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    rank, world_size = dist.get_rank(), dist.get_world_size()

    major, minor = torch.cuda.get_device_capability()
    if args.impl == "te" and major * 10 + minor < 90:
        if rank == 0:
            print(f"SKIPPED: NCCL EP requires SM>=90 (got SM{major}{minor})")
        dist.destroy_process_group()
        return 0

    ep_group = dist.new_group(ranks=list(range(world_size)), backend="nccl")
    dist.all_reduce(torch.zeros(1, device="cuda"), group=ep_group)
    num_experts = args.num_local_experts * world_size
    if args.impl == "te":
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
    mlp_kwargs = dict(
        num_experts=num_experts,
        moe_ffn_hidden_size=args.moe_ffn,
        shared_expert_ffn_hidden_size=args.moe_ffn,
        topk=args.topk,
        ep_group=ep_group,
        ep_max_tokens_per_rank=args.tokens_per_rank,
    )
    if args.impl != "te":
        # Build the TE MoE without EP (replaced below); keeps the pre-MLP RMSNorm.
        mlp_kwargs.pop("ep_group"), mlp_kwargs.pop("ep_max_tokens_per_rank")
    layer = DeepSeekV3Layer(
        args.hidden,
        args.num_heads,
        params_dtype=torch.bfloat16,
        **mlp_kwargs,
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
    )
    if args.impl != "te":
        layer.mlp = NaiveMoE(
            args.hidden,
            args.moe_ffn,
            num_experts,
            args.topk,
            ep_group,
            args.moe_ffn,
            torch.bfloat16,
            grouped=args.impl == "naive_grouped",
        )
    seq = args.tokens_per_rank // 4
    x = torch.randn(seq, 4, args.hidden, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    def step():
        layer.zero_grad(set_to_none=True)
        x.grad = None
        with _autocast(args.recipe):
            out = layer(x)
        out.backward(torch.ones_like(out))
        return out

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    dist.barrier()

    torch.cuda.profiler.start()
    start = time.perf_counter()
    for i in range(args.iters):
        with torch.cuda.nvtx.range(f"iter{i}"):
            out = step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / args.iters * 1e3
    torch.cuda.profiler.stop()
    finite = _check_finite(
        [out, x.grad, layer.mlp.gate.weight.grad]
        + [p.grad for p in layer.parameters() if p.grad is not None],
        x.device,
    )
    dist.barrier()

    if rank == 0:
        tok_s = args.tokens_per_rank * world_size / (ms / 1e3)
        print(
            f"DeepSeekV3Layer impl={args.impl}:"
            f" ranks={world_size} experts={num_experts} topk={args.topk} tokens/rank={args.tokens_per_rank} hidden={args.hidden} recipe={args.recipe} fused_mlp={os.environ.get('NVTE_CUTEDSL_FUSED_GROUPED_MLP', '0')} fwd+bwd"
            f" {ms:.3f} ms/iter ({tok_s / 1e6:.2f} Mtok/s) finite={finite}",
            flush=True,
        )
    if args.impl == "te":
        ep_finalize()
        release_symm_mem_pool()
    dist.destroy_process_group()
    return 0 if finite else 1


if __name__ == "__main__":
    sys.exit(main())
