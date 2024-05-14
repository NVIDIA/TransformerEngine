# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import argparse
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

def torch_dtype(opt):
    typemap = {
        'fp32' : torch.float32,
        'float32' : torch.float32,
        'fp16' : torch.float16,
        'float16' : torch.float16,
        'bf16' : torch.bfloat16,
        'bfloat16' : torch.bfloat16
    }
    if str(opt).lower() not in typemap.keys():
        raise TypeError
    return typemap[str(opt).lower()]

def dist_print(rank, msg, end='\n', all_ranks=False):
    if rank == 0 or all_ranks:
        print(f"[RANK-{rank}] {msg}", end=end)

def train(nprocs, config, rank):
    # Initialize torch.distributed global process group and get TP group
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl",
                            init_method=None if config.use_torchrun else "file:///tmp/rdzv",
                            rank=rank,
                            world_size=nprocs)
    tp_group = dist.new_group(backend="nccl")
    tp_size = dist.get_world_size(tp_group)
    torch.manual_seed(config.seed+rank)
    torch.cuda.manual_seed(config.seed+rank)

    # Intialize userbuffers
    ag_cfg = {  # Ring-exchange All-Gather overlap for fc1_fprop and fc2_dgrad
        'method': 'ring_exchange',
        'num_splits' : 8,
        'num_sm' : 1,
        'set_sm_margin' : False,
    }
    rs_cfg = {  # Reduce-scatter overlap for fc1_dgrad and fc2_fprop
        'num_splits' : 4,
        'num_sm' : 1,
        'set_sm_margin' : True,
    }
    hidden_size = config.num_heads * config.head_dim
    if not config.no_comm_overlap:
        te.initialize_ub(
            [config.seq_length * config.batch_size, hidden_size],
            tp_group,
            use_fp8 = config.fp8,
            dtype = config.dtype,
            ub_cfgs = {
                'fc1_fprop': ag_cfg,
                'fc1_dgrad': rs_cfg,
                'fc2_fprop': rs_cfg,
                'fc2_dgrad': ag_cfg,
            },
        )

    # Initialize TE model
    ln_mlp = te.LayerNormMLP(
        hidden_size, config.mlp_expansion_factor * hidden_size,
        params_dtype = config.dtype,
        device = 'cuda',
        tp_group = tp_group,
        tp_size = tp_size,
        set_parallel_mode = True,
        sequence_parallel = True,
        micro_batch_size = config.batch_size,
        ub_overlap_rs_dgrad = not config.no_comm_overlap,
        ub_overlap_rs = not config.no_comm_overlap,
        ub_overlap_ag = not config.no_comm_overlap,
    )

    # Initialize optimizer with model parameters
    optim = torch.optim.Adam(ln_mlp.parameters(), lr=0.0001)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32,
                                amax_compute_algo="max")

    # Start dummy "training" iterations
    for i in range(config.num_iters):
        dist_print(rank, f"Iter {i}", all_ranks=config.verbose)

        dist_print(rank, "|-- Generate random input batch", all_ranks=config.verbose)
        x = torch.rand((config.seq_length * config.batch_size // tp_size, hidden_size),
                       dtype=config.dtype, device='cuda')

        dist_print(rank, "|-- Forward pass", all_ranks=config.verbose)
        with te.fp8_autocast(enabled=config.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
            y = ln_mlp(x)
            loss = y.flatten().sum()

        dist_print(rank, "|-- Backward pass", all_ranks=config.verbose)
        loss.backward()

        dist_print(rank, "|-- Optimizer step", all_ranks=config.verbose)
        optim.step()

    te.destroy_ub()
    dist.destroy_process_group(tp_group)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a te.LayerNormMLP module with "
                                                 "GEMM+comm overlap via Userbuffers.")
    parser.add_argument('-i', "--num-iters", type=int, default=5,
                        help="Number of dummy 'training' iterations.")
    parser.add_argument('-b', "--batch-size", type=int, default=2,
                        help="Input batch size.")
    parser.add_argument('-s', "--seq-length", type=int, default=2048,
                        help="Input sequence length.")
    parser.add_argument('-n', "--num-heads", type=int, default=64,
                        help="Number of attention heads.")
    parser.add_argument('-d', "--head-dim", type=int, default=128,
                        help="Dimension of each attention head.")
    parser.add_argument('-m', "--mlp-expansion-factor", type=int, default=4,
                        help="MLP block intermediate size as a factor of hidden dimension.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="RNG seed.")
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Enables the te.fp8_autocast() context.")
    parser.add_argument("--no-comm-overlap", action="store_true", default=False,
                        help="Disable the comm+GEMM overlap.")
    parser.add_argument("--use-torchrun", action="store_true", default=False,
                        help="Disable `torch.multiprocessing.spawn` for launching with `torchrun`.")
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16,
                        help="Data type for input tensor and Transformer Engine module parameters.")
    parser.add_argument('-v', "--verbose", action="store_true", default=False)
    args = parser.parse_args()

    if args.use_torchrun:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        train(world_size, args, local_rank)
    else:
        ngpus = torch.cuda.device_count()
        worker = partial(train, ngpus, args)
        mp.spawn(worker, nprocs=ngpus)
