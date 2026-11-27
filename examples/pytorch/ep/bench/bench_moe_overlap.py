# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Timing harness for the overlapped MoE op (FusedMoE), forward and backward, in eager and
CUDA-graph modes. Launch with torchrun, one process per GPU:

    torchrun --standalone --nproc-per-node=4 bench_moe_overlap.py --precision bf16

The op fuses an MoE MLP (dispatch, GroupedLinear, ScaledSwiGLU, GroupedLinear, combine) so the
NCCL EP dispatch and combine overlap the expert GEMMs. Requires SM 10.0 or 10.3, an EP group
within one NVLink domain, NVTE_ENABLE_CUTEDSL_BACKEND=1 and NVTE_MOE_OVERLAP. The defaults match
the DeepSeek-V3 EP4 shape.
"""

import argparse
import os
import statistics

import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.pytorch import ops as te_ops
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch.ep import EpBuffer, EpConfig, ep_bootstrap, ep_finalize


def _parse_args():
    p = argparse.ArgumentParser(description="FusedMoE forward+backward perf bench")
    p.add_argument("--precision", choices=("bf16", "mxfp8"), default="bf16")
    p.add_argument("--mode", choices=("eager", "graph", "both"), default="both")
    p.add_argument("--tokens-per-rank", type=int, default=4096)
    p.add_argument("--hidden-dim", type=int, default=7168)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--num-local-experts", type=int, default=64)
    p.add_argument("--alignment", type=int, default=256)
    p.add_argument("--recv-capacity-per-rank", type=int, default=131072)
    p.add_argument(
        "--comm-sms",
        default="32,32,32,32",
        help="SMs the EP keeps for dispatch_fwd,combine_fwd,dispatch_bwd,combine_bwd",
    )
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    if args.iters < 1 or args.warmup < 0:
        p.error("--iters must be positive and --warmup nonnegative")
    return args


def _build(args):
    recipe = MXFP8BlockScaling() if args.precision == "mxfp8" else None
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.cuda.current_device())
    ep_size = dist.get_world_size()
    num_experts = args.num_local_experts * ep_size
    ep_group = dist.new_group(ranks=list(range(ep_size)), backend="nccl")

    sms = [int(s) for s in args.comm_sms.split(",")]
    ep_bootstrap(
        ep_group,
        num_experts=num_experts,
        max_tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.hidden_dim,
        num_topk=args.top_k,
        recv_capacity_per_rank=args.recv_capacity_per_rank,
        max_num_sms=max(sms),
        dispatch_fwd_num_sms=sms[0],
        combine_fwd_num_sms=sms[1],
        dispatch_bwd_num_sms=sms[2],
        combine_bwd_num_sms=sms[3],
    )

    config = EpConfig(
        top_k=args.top_k,
        max_tokens_per_rank=args.tokens_per_rank,
        hidden_dim=args.hidden_dim,
        num_local_experts=args.num_local_experts,
        recv_capacity_per_rank=args.recv_capacity_per_rank,
        ep_group=ep_group,
        alignment=args.alignment,
        dispatch_fwd_quant_recipe=recipe,
        combine_bwd_quant_recipe=recipe,
    )
    buffer = EpBuffer(
        top_k=config.top_k,
        max_tokens_per_rank=config.max_tokens_per_rank,
        hidden_dim=config.hidden_dim,
        num_local_experts=config.num_local_experts,
        recv_capacity_per_rank=config.recv_capacity_per_rank,
        alignment=config.alignment,
        payload_dtype=config.payload_dtype,
        dispatch_fwd_quant_recipe=config.dispatch_fwd_quant_recipe,
        combine_bwd_quant_recipe=config.combine_bwd_quant_recipe,
    )
    return recipe, num_experts, device, te_ops.Sequential(*_moe_ops(args, config, buffer, device))


def _moe_ops(args, config, buffer, device):
    """MoeDispatch, GroupedLinear, ScaledSwiGLU, GroupedLinear and MoeCombine with the routing
    channels connected: the dispatch forwards the expert counts and routing weights to the GEMMs
    and the activation, so the sequence fuses into one FusedMoE."""
    gen = torch.Generator(device=device).manual_seed(5)
    linear_kwargs = {
        "bias": False,
        "dtype": torch.bfloat16,
        "single_grouped_weight": True,
        "split_alignment": args.alignment,
    }
    dispatch = te_ops.MoeDispatch(config, buffer)
    fc1 = te_ops.GroupedLinear(
        args.num_local_experts, args.hidden_dim, 2 * args.ffn, device=device, **linear_kwargs
    )
    fc2 = te_ops.GroupedLinear(
        args.num_local_experts, args.ffn, args.hidden_dim, device=device, **linear_kwargs
    )
    act = te_ops.ScaledSwiGLU(glu_interleave_size=32)
    combine = te_ops.MoeCombine(config, buffer)
    with torch.no_grad():
        for fc in (fc1, fc2):
            for w in fc.parameters():
                w.copy_((torch.rand(w.shape, device=device, generator=gen) - 0.5) * 0.1)
    dispatch.set_extra_output_channel(0, "tokens_per_expert", output_to_caller=False)
    dispatch.set_extra_output_channel(1, "routing_weights", output_to_caller=False)
    fc1.set_extra_input_channel(0, "tokens_per_expert")
    act.set_extra_input_channel(0, "routing_weights")
    fc2.set_extra_input_channel(0, "tokens_per_expert")
    return dispatch, fc1, act, fc2, combine


def _inputs(args, num_experts, device):
    gen = torch.Generator(device=device).manual_seed(2026 + dist.get_rank())
    tokens = (
        torch.randn(args.tokens_per_rank, args.hidden_dim, generator=gen, device=device) * 0.25
    ).to(torch.bfloat16)
    logits = torch.randn(args.tokens_per_rank, num_experts, generator=gen, device=device)
    topk_logits, topk_idx = torch.topk(logits, args.top_k, dim=-1)
    topk_w = torch.softmax(topk_logits, dim=-1)
    grad_out = torch.randn(tokens.shape, generator=gen, device=device).to(torch.bfloat16)
    return topk_idx, tokens.requires_grad_(True), topk_w.requires_grad_(True), grad_out


def _autocast(recipe):
    return te.autocast(enabled=recipe is not None, recipe=recipe)


def _assert_fused(args, recipe, num_experts, device, model):
    from transformer_engine.pytorch.ops.fused import moe_overlap

    topk_idx, x, topk_w, _ = _inputs(args, num_experts, device)
    with _autocast(recipe):
        model(x, topk_idx, topk_w, topk_idx)
    torch.cuda.synchronize()
    fused = any(
        isinstance(op, moe_overlap.FusedMoE)
        for group in model._module_groups
        for op, _ in getattr(group, "_forward_ops", [])
    )
    if not fused:
        raise RuntimeError("FusedMoE did not fuse; the measurement would be the unfused path")


def _run_eager(args, recipe, num_experts, device, model):
    topk_idx, x, topk_w, grad_out = _inputs(args, num_experts, device)
    for _ in range(args.warmup):
        x.grad = topk_w.grad = None
        with _autocast(recipe):
            y = model(x, topk_idx, topk_w, topk_idx)
        y.backward(grad_out)
    torch.cuda.synchronize()

    fwd_t, bwd_t, tot_t = [], [], []
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    for _ in range(args.iters):
        x.grad = topk_w.grad = None
        ev[0].record()
        with _autocast(recipe):
            y = model(x, topk_idx, topk_w, topk_idx)
        ev[1].record()
        y.backward(grad_out)
        ev[2].record()
        torch.cuda.synchronize()
        fwd_t.append(ev[0].elapsed_time(ev[1]))
        bwd_t.append(ev[1].elapsed_time(ev[2]))
        tot_t.append(ev[0].elapsed_time(ev[2]))
    return statistics.median(fwd_t), statistics.median(bwd_t), statistics.median(tot_t)


def _run_graph(args, recipe, num_experts, device, model):
    topk_idx, x, topk_w, grad_out = _inputs(args, num_experts, device)
    # Warm up on a side stream before capture.
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(args.warmup):
            x.grad = topk_w.grad = None
            with _autocast(recipe):
                y = model(x, topk_idx, topk_w, topk_idx)
            y.backward(grad_out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Two graphs timed with events outside replay: full fwd+bwd, and fwd-only. bwd = total - fwd.
    # Events recorded inside a captured graph give cudaErrorInvalidValue on elapsed_time.
    g_full = torch.cuda.CUDAGraph()
    x.grad = topk_w.grad = None
    with torch.cuda.graph(g_full):
        with _autocast(recipe):
            y = model(x, topk_idx, topk_w, topk_idx)
        y.backward(grad_out)
    g_fwd = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_fwd):
        with _autocast(recipe):
            model(x, topk_idx, topk_w, topk_idx)
    torch.cuda.synchronize()

    def _time(graph):
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ts = []
        for _ in range(args.iters):
            ev[0].record()
            graph.replay()
            ev[1].record()
            torch.cuda.synchronize()
            ts.append(ev[0].elapsed_time(ev[1]))
        return statistics.median(ts)

    tot = _time(g_full)
    fwd = _time(g_fwd)
    return fwd, tot - fwd, tot


def main():
    """Build the FusedMoE sequence and report its forward/backward timings."""
    args = _parse_args()
    recipe, num_experts, device, model = _build(args)
    is_rank0 = dist.get_rank() == 0
    _assert_fused(args, recipe, num_experts, device, model)
    if is_rank0:
        print(
            f"\n=== FusedMoE fwd+bwd bench [{args.precision}] T/rank={args.tokens_per_rank} "
            f"H={args.hidden_dim} FFN={args.ffn} top_k={args.top_k} E={num_experts} "
            f"SMs={args.comm_sms} ===",
            flush=True,
        )

    if args.mode in ("eager", "both"):
        fwd, bwd, tot = _run_eager(args, recipe, num_experts, device, model)
        if is_rank0:
            print(f"EAGER  fwd {fwd:.3f} ms | bwd {bwd:.3f} ms | fwd+bwd {tot:.3f} ms", flush=True)

    if args.mode in ("graph", "both"):
        fwd, bwd, tot = _run_graph(args, recipe, num_experts, device, model)
        if is_rank0:
            print(
                f"GRAPH  fwd {fwd:.3f} ms | bwd {bwd:.3f} ms (total - fwd) | fwd+bwd {tot:.3f} ms",
                flush=True,
            )

    dist.barrier()
    ep_finalize()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
