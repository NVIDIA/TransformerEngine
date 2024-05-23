#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import uuid
import faulthandler
import argparse

import torch
import torch.distributed as dist
from torch.distributed.run import get_args_parser, config_from_args, elastic_launch
from torch.distributed.launch import parse_args as parse_torch_args

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

def parse_train_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Test a te.LayerNormMLP module with GEMM+comm overlap via Userbuffers.")
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
    parser.add_argument("--mlp-expansion-factor", type=int, default=4,
                        help="MLP block intermediate size as a factor of hidden dimension.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="RNG seed.")
    parser.add_argument("--fp8", action="store_true", default=False,
                        help="Enables the te.fp8_autocast() context.")
    parser.add_argument("--no-comm-overlap", action="store_true", default=False,
                        help="Disable the comm+GEMM overlap.")
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16,
                        help="Data type for input tensor and Transformer Engine module parameters.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true", default=False)
    return parser.parse_args(argv)

def train(*argv):
    opts = parse_train_args(argv)

    WORLD_RANK = int(os.getenv("RANK"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE"))
    def dist_print(msg, end='\n', all_ranks=False):
        if WORLD_RANK == 0 or all_ranks:
            print(f"[RANK-{WORLD_RANK}] {msg}", end=end)

    # Debug log
    if opts.debug:
        with open(f'faulthandler_{WORLD_RANK}.log', 'w+') as dbg_log:
            faulthandler.enable(dbg_log)

    # Seed RNG
    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(opts.seed+WORLD_RANK)
    torch.cuda.manual_seed(opts.seed+WORLD_RANK)

    # Initialize torch.distributed global process group and get TP group
    dist.init_process_group(backend="nccl",
                            rank=WORLD_RANK,
                            world_size=WORLD_SIZE,
                            device_id=torch.device(f'cuda:{WORLD_RANK}'))
    tp_group = dist.new_group(backend="nccl")
    tp_size = dist.get_world_size(tp_group)

    # Intialize userbuffers
    ag_cfg = {  # Ring-exchange All-Gather overlap for fc1_fprop and fc2_dgrad
        'method': 'ring_exchange',
        'num_splits' : 8,
        'num_sm' : 1,
        'set_sm_margin' : False,
    }
    rs_cfg = {  # Reduce-scatter overlap for fc1_dgrad and fc2_fprop
        'method': 'ring_exchange',
        'num_splits' : 4,
        'num_sm' : 1,
        'set_sm_margin' : True,
    }
    hidden_size = opts.num_heads * opts.head_dim
    if not opts.no_comm_overlap:
        te.initialize_ub(
            [opts.seq_length * opts.batch_size, hidden_size],
            tp_group,
            use_fp8 = opts.fp8,
            dtype = opts.dtype,
            ub_cfgs = {
                'fc1_fprop': ag_cfg,
                'fc1_dgrad': rs_cfg,
                'fc2_fprop': rs_cfg,
                'fc2_dgrad': ag_cfg,
            },
        )

    #
    model = te.LayerNormMLP(
        hidden_size, opts.mlp_expansion_factor * hidden_size,
        params_dtype = opts.dtype,
        device = 'cuda',
        tp_group = tp_group,
        tp_size = tp_size,
        set_parallel_mode = True,
        sequence_parallel = True,  # this is required for comm+GEMM overlap
        seq_length = opts.seq_length,
        micro_batch_size = opts.batch_size,
        ub_overlap_rs_dgrad = not opts.no_comm_overlap,
        ub_overlap_rs = not opts.no_comm_overlap,
        ub_overlap_ag = not opts.no_comm_overlap,
    )

    # Initialize optimizer with model parameters
    optim = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Fp8 recipe setup
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32,
                                amax_compute_algo="max")

    # Start dummy "training" iterations
    torch.cuda.synchronize()
    dist.barrier()
    for i in range(opts.num_iters):
        dist_print(f"Iter {i+1}", all_ranks=opts.verbose)

        dist_print("|-- Generate random input batch", all_ranks=opts.verbose)
        x = torch.rand((opts.seq_length // tp_size, opts.batch_size, hidden_size),
                       dtype=opts.dtype, device='cuda', requires_grad=True)

        dist_print("|-- Forward pass", all_ranks=opts.verbose)
        with te.fp8_autocast(enabled=opts.fp8, fp8_recipe=fp8_recipe, fp8_group=tp_group):
            y = model(x)
            dist_print("|-- Compute loss", all_ranks=opts.verbose)
            loss = y.flatten().sum()

        dist_print("|-- Backward pass", all_ranks=opts.verbose)
        loss.backward()

        dist_print("|-- Optimizer step", all_ranks=opts.verbose)
        optim.step()

    te.destroy_ub()

def dist_launch(func) -> None:
    torch_parser = get_args_parser()
    torch_argv = []
    func_argv = []
    for argv in sys.argv:
        is_torch_argv = False
        for action in torch_parser._actions:
            if any(option in argv for option in action.option_strings):
                is_torch_argv = True
                break
        if is_torch_argv:
            torch_argv.append(argv)
        else:
            func_argv.append(argv)
    del torch_parser

    fake_sys_argv = [ torch.distributed.run.__file__ ] + torch_argv + func_argv
    torch_args = parse_torch_args(fake_sys_argv)
    setattr(torch_args, "standalone", "True")
    setattr(torch_args, "rdzv_backend", "c10d")
    setattr(torch_args, "rdzv_endpoint", "localhost:0")
    setattr(torch_args, "rdzv_id", str(uuid.uuid4()))
    setattr(torch_args, "nnodes", "1:1")
    setattr(torch_args, "nproc_per_node", torch.cuda.device_count())
    setattr(torch_args, "use_env", True)

    torch_config, *_ = config_from_args(torch_args)
    elastic_launch(config=torch_config, entrypoint=func)(*func_argv[1:])


if __name__ == "__main__":
    if "TORCHELASTIC_RUN_ID" in os.environ.keys():
        train(sys.argv)
    else:
        dist_launch(train)
    os._exit(0)
