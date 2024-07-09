#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import subprocess
import argparse
import warnings
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

torch_dtypes = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

atol_map = {
    torch.float32: 5e-3,
    torch.float16: 5e-2,
    torch.bfloat16: 5e-2,
}


def mapped_argtype(opt, typemap={}):
    if str(opt).lower() not in typemap.keys():
        raise TypeError(f"Unrecognized option! Please choose from: {typemap.keys()}")
    return typemap[str(opt).lower()]


def parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Test te.TransformerLayer with GEMM+comm overlap.")
    parser.add_argument("-s", "--seq-length", type=int, default=2048, help="Input sequence length.")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Input batch size.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=12, help="Number of attention heads."
    )
    parser.add_argument(
        "-d", "--head-dim", type=int, default=64, help="Dimension of each attention head."
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed.")
    parser.add_argument(
        "--fp8", action="store_true", default=False, help="Enables the te.fp8_autocast() context."
    )
    parser.add_argument(
        "--dtype",
        type=partial(mapped_argtype, typemap=torch_dtypes),
        default=torch.bfloat16,
    )
    parser.add_argument(
        "--no-mask", action="store_true", default=False, help="Disable the causal attention mask."
    )
    parser.add_argument(
        "--no-grad", action="store_true", default=False, help="Disable the backward pass."
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Print out debug info."
    )
    return parser.parse_args(argv, namespace)

@record
def train(opts):
    WORLD_RANK = int(os.getenv("RANK"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK"))

    # Seed RNG
    torch.cuda.set_device(LOCAL_RANK)
    torch.manual_seed(opts.seed + LOCAL_RANK)
    torch.cuda.manual_seed(opts.seed + LOCAL_RANK)

    # Initialize torch.distributed global process group and get TP group
    dist.init_process_group(
        backend="nccl",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        device_id=torch.device(f"cuda:{WORLD_RANK}"),
    )
    tp_group = dist.new_group(backend="nccl")
    tp_size = dist.get_world_size(tp_group)

    # Info printout
    def dist_print(msg, src=None, debug=False, section=False):
        if debug or opts.verbose:
            if section:
                dist.barrier()
                if WORLD_RANK == (0 if src is None else src):
                    print("\n", end="", flush=True)
            dist.barrier()
            if src is None or WORLD_RANK == src:
                prefix = "[GLOBAL] " if src is not None else f"[rank:{WORLD_RANK}] "
                lines = msg.splitlines()
                msg = "\n".join(
                    [prefix + lines[0]] + [(" " * len(prefix)) + line for line in lines[1:]]
                )
                print(msg + "\n", end="", flush=True)

    # Intialize userbuffers
    hidden_size = opts.num_heads * opts.head_dim
    batched_size = opts.seq_length * opts.batch_size
    te.initialize_ub(
        [batched_size, hidden_size],
        tp_size,
        use_fp8=opts.fp8,
        dtype=opts.dtype,
    )

    # Initialize test and reference model, and share parameters
    te_kwargs = {
        "layernorm_epsilon": 1e-5,
        "hidden_dropout": 0.0,
        "attention_dropout": 0.0,
        "fuse_qkv_params": True,
        "device": "cuda",
        "params_dtype": opts.dtype,
        "seq_length": opts.seq_length,
        "set_parallel_mode": WORLD_SIZE > 1,
        "tp_group": tp_group if WORLD_SIZE > 1 else None,
        "sequence_parallel": WORLD_SIZE > 1,
        "ub_tp_comm_overlap": True,
        "parallel_attention_mlp": False,
    }
    te_gpt = te.TransformerLayer(
        hidden_size,
        4 * hidden_size,
        opts.num_heads,
        **te_kwargs
    )

    # Create new TransformerLayer without comm overlap
    te_kwargs["ub_tp_comm_overlap"] = False
    te_gpt_no_overlap = te.TransformerLayer(
        hidden_size,
        4 * hidden_size,
        opts.num_heads,
        **te_kwargs,
    )

    # Clone parameters from original layer
    with torch.no_grad():
        for p1, p2 in zip(te_gpt.parameters(), te_gpt_no_overlap.parameters()):
            p2.copy_(p1)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Generate causal attention mask and random input in SBHD format
    causal_mask = (
        torch.triu(torch.ones(opts.seq_length, opts.seq_length, device="cuda"), diagonal=1).bool()
        if not opts.no_mask
        else None
    )
    x = torch.rand(
        (opts.seq_length // tp_size, opts.batch_size, hidden_size),
        dtype=opts.dtype,
        device="cuda",
        requires_grad=True,
    )
    x.retain_grad()
    dist_print(f"Distributed input: {x.size()}", section=True)

    # Forward + backward passes for overlapped layer
    torch.cuda.synchronize()
    dist.barrier()
    with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
        y = te_gpt(x, attention_mask=causal_mask)
        loss = y.flatten().sum()
    if not opts.no_grad:
        loss.backward()
    dist_print(f"Distributed output: {y.size()}", section=True)

    # Forward + backward passes for non-overlapped layer
    torch.cuda.synchronize()
    dist.barrier()
    xn = x.detach().clone().requires_grad_(True)
    xn.retain_grad()
    with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
        yn = te_gpt_no_overlap(xn, attention_mask=causal_mask)
        ln = yn.flatten().sum()
    if not opts.no_grad:
        ln.backward()

    # Assemble list of tensors to be compared
    torch.cuda.synchronize()
    dist.barrier()
    overlap = [y]
    no_overlap = [yn]
    names = ["output"]
    if not opts.no_grad:
        overlap.append(x.grad)
        names.append("input_grad")
        for name, param in te_gpt.named_parameters():
            if param.requires_grad:
                overlap.append(param)
                names.append(name)

        no_overlap.append(xn.grad)
        for param in te_gpt_no_overlap.parameters():
            if param.requires_grad:
                no_overlap.append(param)

    atol = atol_map[opts.dtype]
    max_diff = 0.0
    max_diff_name = None
    max_diff_idx = 0
    numerics_failed = False
    for i, (o, n) in enumerate(zip(overlap, no_overlap)):
        numerics_failed = not torch.allclose(o, n, atol=atol)
        diff = torch.abs(o - n).flatten()
        max_idx = torch.argmax(diff).item()
        if diff[max_idx].item() >= max_diff:
            max_diff = diff[max_idx].item()
            max_diff_name = names[i]
            max_diff_idx = max_idx
        if numerics_failed:
            failed_idx = diff > atol
            num_fails = failed_idx.int().sum()
            fail_ratio = failed_idx.float().mean()
            mismatch_info = (
                f"OVERLAP MISMATCH in '{names[i]}'!"
                + f"\nDiff above tolerance ({atol}) for {num_fails} elements "
                + f"({fail_ratio.item()*100.}%)."
                + f"\nMax diff {diff[max_idx].item()} at index {max_idx} "
                + f"({o.flatten()[max_idx].item()} vs {n.flatten()[max_idx].item()})."
            )
            dist_print(mismatch_info, section=True, debug=True)
            break

    numerics_failed_tensor = torch.tensor([int(numerics_failed)], dtype=torch.int64, device="cuda")
    dist.all_reduce(numerics_failed_tensor, dist.ReduceOp.MAX)
    numerics_failed = bool(numerics_failed_tensor[0].item())
    if not numerics_failed:
        max_diff_all = [ None for _ in range(WORLD_SIZE) ]
        dist.all_gather_object(max_diff_all, max_diff)
        max_diff_idx_all = [ None for _ in range(WORLD_SIZE) ]
        dist.all_gather_object(max_diff_idx_all, max_diff_idx)
        max_diff_name_all = [ None for _ in range(WORLD_SIZE) ]
        dist.all_gather_object(max_diff_name_all, max_diff_name)
        max_diff = max(max_diff_all)
        diff_idx = max_diff_all.index(max_diff)
        max_diff_idx = max_diff_idx_all[diff_idx]
        max_diff_name = max_diff_name_all[diff_idx]
        diff_info = (
            f"NUMERICAL CHECK PASSED: max error = {max_diff} "
            + f"in '{max_diff_name}' at idx {max_diff_idx}"
        )
        dist_print(diff_info, src=0, section=True, debug=True)

    torch.cuda.synchronize()
    dist.barrier()
    te.destroy_ub()

    dist.destroy_process_group()

    return int(numerics_failed)


if __name__ == "__main__":
    try:
        if "TORCHELASTIC_RUN_ID" in os.environ.keys():
            args = parse_args()
            os._exit(train(args))
        else:
            subprocess.run(
                ["torchrun", f"--nproc-per-node={torch.cuda.device_count()}", *sys.argv],
                env=os.environ,
                check=True,
            )
            os._exit(0)
    except Exception as err:  # pylint: disable=broad-exception-caught
        print(err)
        os._exit(1)
