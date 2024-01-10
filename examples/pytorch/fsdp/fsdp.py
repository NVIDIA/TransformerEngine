# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import argparse

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import always_wrap_policy

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

def lowercase(s):
    return str(s).lower()

def torch_dtype(d):
    typemap = {
        'fp32' : torch.float32,
        'float32' : torch.float32,
        'fp16' : torch.float16,
        'float16' : torch.float16,
        'bf16' : torch.bfloat16,
        'bfloat16' : torch.bfloat16
    }
    if lowercase(d) not in typemap.keys():
        raise TypeError
    return typemap[lowercase(d)]

def parse_fsdp_args():
    parser = argparse.ArgumentParser(description="Run Transformer Engine modules with the " +
                                    "torch.distributed.fsdp.FullyShardedDataParallel strategy.")
    parser.add_argument("--no-fp8", action="store_true", default=False,
                        help="Disables the te.fp8_autocast() context.")
    parser.add_argument('-i', "--num-iters", type=int, default=3,
                        help="Number of fake training iterations.")
    parser.add_argument('-b', "--batch-size", type=int, default=32,
                        help="Input batch size.")
    parser.add_argument('-s', "--seq-length", type=int, default=1048,
                        help="Input sequence length.")
    parser.add_argument('-n', "--num-heads", type=int, default=64,
                        help="Number of attention heads.")
    parser.add_argument('-d', "--head-dim", type=int, default=512,
                        help="Dimension of each attention head (number of KV channels).")
    parser.add_argument('-l', "--num-layers", type=int, default=1,
                        help="Number of modules chained together with nn.Sequential.")
    parser.add_argument("--seed", type=int, default=1234,
                        help="PyTorch RNG seed.")
    parser.add_argument("--defer-init", action="store_true",
                        help="Use 'meta' device to defer module initialization until FSDP sharding.")
    parser.add_argument("--dtype", type=torch_dtype, default=torch.bfloat16,
                        help="Data type for input tensor and Transformer Engine module parameters.")
    return parser.parse_args()

def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Initialize torch.distributed global process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    if local_rank == 0:
        print(f"[GPU-0] WORLD_SIZE = {world_size}\n\n", end='')

    dtype = args.dtype
    seq_len = args.seq_length
    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    hidden_size = num_heads * head_dim
    ffn_hidden_size = 4 * hidden_size
    torch.manual_seed(args.seed)

    # Construct a simple homogeneous model (only one layer type) with NO PARALLELISM
    # NOTE: Change the layer type, args and kwargs variables below to test different models.
    layer_type = te.Linear
    layer_args = (hidden_size, ffn_hidden_size)
    layer_kwargs = {
        'bias': True,
        'params_dtype': dtype,
        'device': 'meta' if args.defer_init else 'cuda'
    }

    if args.num_layers > 1:
        # Multi-layer model via torch.nn.Sequential
        if layer_type is te.Linear:
            # When layers are linear, input and output feature sizes need to be identical
            ffn_hidden_size = hidden_size
            layer_args = (hidden_size, hidden_size)
        te_layer_list = [ layer_type(*layer_args, **layer_kwargs) for i in range(args.num_layers) ]
        te_model = nn.Sequential(*te_layer_list)
    else:
        # Single layer model
        te_model = layer_type(*layer_args, **layer_kwargs)

    # Print out allocated device memory before the model parameters are sharded by FSDP
    pre_mem_use = torch.cuda.memory_allocated(device=f"cuda:{local_rank}") * 1e-6
    print(f"[GPU-{local_rank}] Pre-FSDP memory use = {pre_mem_use}MiB\n", end='')

    # Wrap the model with FSDP
    # NOTE: The TE model itself has no inherent parallelism. FSDP shards model parameters and
    #       controls all communication.
    all_gpus = dist.new_group(backend='nccl')
    te_model = FullyShardedDataParallel(te_model,
                                        process_group=all_gpus,
                                        use_orig_params=True,
                                        mixed_precision=MixedPrecision(
                                            param_dtype=dtype,
                                            reduce_dtype=torch.float32,
                                        ),
                                        sync_module_states=True,
                                        auto_wrap_policy=always_wrap_policy)

    # Print out allocated device memory after the model parameters are sharded
    post_mem_use = torch.cuda.memory_allocated(device=f"cuda:{local_rank}") * 1e-6
    print(f"[GPU-{local_rank}] Post-FSDP memory use = {post_mem_use}MiB\n", end='')

    # Fp8 setup for TE
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")

    # Optimizer must be created after the model is wrapped in FSDP and the parameters are sharded
    optim = torch.optim.Adam(te_model.parameters(), lr=0.0001)

    # Start and time dummy "training" iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for i in range(args.num_iters):
        # Generate a random input batch
        x = torch.rand(seq_len, batch_size, hidden_size).to(dtype=dtype).cuda()
        # fp8_autocast needs to be given the FSDP process group for amax reductions
        with te.fp8_autocast(enabled=not args.no_fp8, fp8_recipe=fp8_recipe, fp8_group=all_gpus):
            y = te_model(x)
            loss = y.sum()
        # calculate gradient and take training step outside the fp8_autocast context
        loss.backward()
        optim.step()
        del x
        if local_rank == 0:
            print(f"[GPU-0] Iter. {i+1}\n", end='')
    end.record()
    torch.cuda.synchronize()

    # Print out "training" time and peak memory use stats
    train_time = start.elapsed_time(end)/1000.
    max_memory_alloc = int(torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}") * 1e-6)
    print(f"\n[GPU-{local_rank}] Training Time: {train_time}s\n" +
            f"[GPU-{local_rank}] Avg. Iter. Time: {train_time /args.num_iters}s\n" +
            f"[GPU-{local_rank}] Peak memory use = {max_memory_alloc}MiB\n\n", end='')


if __name__ == "__main__":
    args = parse_fsdp_args()
    train(args)
