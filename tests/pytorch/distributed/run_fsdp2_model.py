#!/usr/bin/python3

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import argparse

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from contextlib import nullcontext

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        self.fc1 = te.Linear(input_size, hidden_size)
        self.fc2 = te.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items()}
    return custom_attrs

def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)

def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Toy example for debugging fully_shard()")
    parser.add_argument('--input-size', type=int, default=2048, help='Input size for the model')
    parser.add_argument('--hidden-size', type=int, default=2048, help='Hidden layer size')
    parser.add_argument('--output-size', type=int, default=2048, help='Output size for the model')
    parser.add_argument('--batch-size', type=int, default=2048, help='Output size for the model')
    parser.add_argument(
        "--fp8-init", action="store_true", default=False, help="Initialize primary weights in FP8."
    )
    parser.add_argument('--iter', type=int, default=10, help='Number of iterations for forward pass')
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    # Adding hsdp_dim as a list argument, comma-separated
    parser.add_argument('--sharding-dims', type=int, nargs='+', help='FSDP/HSDP sharding dimensions ("replicate", "shard")')
    args = parser.parse_args(argv, namespace)
    if args.sharding_dims:
        assert len(args.sharding_dims) <= 2
    return args

sub_modules_to_wrap=[te.Linear]

def _train(args):
    assert "TORCHELASTIC_RUN_ID" in os.environ
    WORLD_RANK = int(os.getenv("RANK", "0"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    assert LOCAL_SIZE == WORLD_SIZE

    # Set device and initialize RNG states
    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    # Initialize torch.distributed global process group and get DP/TP groups
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
    }
    assert dist.is_nccl_available()
    dist.init_process_group(**dist_init_kwargs)
    nccl_world = dist.new_group(backend="nccl")
    device = torch.device(f'cuda:{LOCAL_RANK}')

    # FP8 Configuration
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")

    if not args.fp8_init:
        # Build model context (FP8 init)
        build_model_context = nullcontext
        build_model_context_args = {}

        from transformer_engine.pytorch import fp8_model_init
        build_model_context = fp8_model_init
        build_model_context_args["enabled"] = True

        # Build the model with the specified context
        with build_model_context(**build_model_context_args):
            model = SimpleNet(args.input_size, args.hidden_size, args.output_size)
    else:
        model = SimpleNet(args.input_size, args.hidden_size, args.output_size)
    # Move the model to the correct device

    model.to(device)

    if LOCAL_RANK == 0:
        print(f"Rank {LOCAL_RANK}: Applying FSDP fully_shard() to the model...")
     # Creating a DeviceMesh for fully_shard
    world_size = int(WORLD_SIZE)
    device_ids = list(range(world_size))
    if LOCAL_RANK == 0:
        print(f"sharding-dims:{args.sharding_dims}")
    # Setup the sharding mesh for FSDP/HSDP
    if args.sharding_dims == None: # FSDP
        mesh = DeviceMesh("cuda", device_ids)
    elif len(args.sharding_dims) == 1:
        assert args.sharding_dims[0] == device_ids[-1] +1
        mesh = DeviceMesh("cuda", device_ids)
    elif len(args.sharding_dims) == 2: # HSDP
        assert args.sharding_dims[0] * args.sharding_dims[1] == device_ids[-1] +1
        mesh = init_device_mesh("cuda", (args.sharding_dims[0], args.sharding_dims[1]), mesh_dim_names=("replicate", "shard"))
    else:
        assert False

    # Apply FSDP/HSDP
    custom_attrs = save_custom_attrs(model)
    for sub_module in model.modules():
        if any(
            isinstance(sub_module, sub_module_to_wrap)
            for sub_module_to_wrap in sub_modules_to_wrap
        ):
            fully_shard(sub_module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    restore_custom_attrs(model, custom_attrs)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(args.iter):
        # Zero the parameter gradients
        optimizer.zero_grad()
        input_data = torch.randn(args.batch_size, args.input_size).to(device)
        output = model(input_data)
        target = torch.randn(args.batch_size, args.output_size).to(device)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if LOCAL_RANK == 0:
            print(f"Rank {LOCAL_RANK}: Iteration {iteration} completed.")
    
    dist.destroy_process_group()
    if LOCAL_RANK == 0:
        print(f"Rank {LOCAL_RANK}: Done...")
    return 0


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
