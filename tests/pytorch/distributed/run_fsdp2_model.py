#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import argparse
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import Format, DelayedScaling


def _save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items()}
    return custom_attrs


def _restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)


def _te_layer_type(layer_name):
    te_layer_types = [
        te.Linear,
        te.LayerNormLinear,
        te.LayerNormMLP,
        te.MultiheadAttention,
        te.TransformerLayer,
    ]
    te_layer_names = [layer.__name__ for layer in te_layer_types]
    te_layer_map = dict(zip([name.lower() for name in te_layer_names], te_layer_types))
    if layer_name.lower() not in te_layer_map.keys():
        raise argparse.ArgumentTypeError(
            f'"{layer_name}" is not a valid Transformer Engine layer, '
            f"please choose layer from {te_layer_names}."
        )
    return te_layer_map[layer_name.lower()]


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Toy example for debugging fully_shard()")
    parser.add_argument(
        "--layer-type",
        type=_te_layer_type,
        default=te.TransformerLayer,
        help="Transformer Engine layer type",
    )
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attn. heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Attention head size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size of input")
    parser.add_argument("--seq-length", type=int, default=1024, help="Sequence length of input")
    parser.add_argument(
        "--fp8-init", action="store_true", default=False, help="Initialize primary weights in FP8."
    )
    parser.add_argument("--iter", type=int, default=3, help="Number of iterations for forward pass")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    # Adding hsdp_dim as a list argument, comma-separated
    parser.add_argument(
        "--sharding-dims",
        type=int,
        nargs="+",
        help='FSDP/HSDP sharding dimensions ("replicate", "shard")',
    )
    args = parser.parse_args(argv, namespace)
    if args.sharding_dims:
        assert len(args.sharding_dims) <= 2
    return args


def _init_te_model(config):
    hidden_size = config.num_heads * config.head_dim
    args = [hidden_size, hidden_size]
    inp_shape = [config.seq_length, config.batch_size, hidden_size]
    out_shape = [config.seq_length, config.batch_size, hidden_size]
    kwargs = {
        "params_dtype": torch.bfloat16,
    }
    if config.layer_type == te.LayerNormLinear:
        args[1] *= 3  # QKV projection
        out_shape[-1] *= 3
    elif config.layer_type in [te.MultiheadAttention, te.TransformerLayer]:
        args[1] *= 4  # FFN hidden size
        args.append(config.num_heads)
        kwargs["fuse_qkv_params"] = True
        if config.layer_type is te.MultiheadAttention:
            kwargs["input_layernorm"] = True

    model = config.layer_type(*args, **kwargs)
    return model, inp_shape, out_shape


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
    device = torch.device(f"cuda:{LOCAL_RANK}")

    # Initialize TE model
    fp8_format = Format.HYBRID
    fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    with te.fp8_model_init(enabled=args.fp8_init, recipe=fp8_recipe):
        model, inp_shape, out_shape = _init_te_model(args)
    model.to(device)

    if LOCAL_RANK == 0:
        print(f"Rank {LOCAL_RANK}: Applying FSDP fully_shard() to the model...")
    # Creating a DeviceMesh for fully_shard
    world_size = int(WORLD_SIZE)
    device_ids = list(range(world_size))
    if LOCAL_RANK == 0:
        print(f"sharding-dims:{args.sharding_dims}")
    # Setup the sharding mesh for FSDP/HSDP
    if args.sharding_dims is None:  # FSDP
        mesh = DeviceMesh("cuda", device_ids)
    elif len(args.sharding_dims) == 1:
        assert args.sharding_dims[0] == world_size
        mesh = DeviceMesh("cuda", device_ids)
    elif len(args.sharding_dims) == 2:  # HSDP
        assert args.sharding_dims[0] * args.sharding_dims[1] == world_size
        mesh = init_device_mesh(
            "cuda",
            (args.sharding_dims[0], args.sharding_dims[1]),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        assert False

    # Apply FSDP/HSDP
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    custom_attrs = _save_custom_attrs(model)
    if args.layer_type in [te.MultiheadAttention, te.TransformerLayer]:
        # Composite modules require wrapping submodules bottom-up for the correct parameter grouping
        sub_modules_to_wrap = [
            te.Linear,
            te.LayerNormLinear,
            te.LayerNormMLP,
        ]
        for sub_module in model.modules():
            if any(
                isinstance(sub_module, sub_module_to_wrap)
                for sub_module_to_wrap in sub_modules_to_wrap
            ):
                fully_shard(sub_module, mesh=mesh, mp_policy=None if args.fp8_init else mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=None if args.fp8_init else mp_policy)
    _restore_custom_attrs(model, custom_attrs)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(args.iter):
        # Zero the parameter gradients
        optimizer.zero_grad()
        input_data = torch.randn(inp_shape).to(device)
        target = torch.randn(out_shape).to(device)
        with torch.autograd.detect_anomaly():
            with torch.amp.autocast(enabled=not args.fp8_init, device_type="cuda"):
                with te.fp8_autocast(enabled=args.fp8_init, fp8_recipe=fp8_recipe):
                    output = model(input_data)
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
