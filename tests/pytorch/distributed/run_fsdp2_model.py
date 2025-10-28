#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import sys
import argparse

import transformer_engine.pytorch as te
<<<<<<< HEAD
from transformer_engine.common.recipe import Format, DelayedScaling, Float8CurrentScaling, MXFP8BlockScaling
=======
from transformer_engine.common.recipe import (
    Format,
    DelayedScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
)

>>>>>>> 0a6e23c4cdd0111df4151ecec8ed5092fe850784
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from transformer_engine.pytorch import QuantizedTensor
from contextlib import nullcontext

LOCAL_RANK = None

    
def dist_print(msg):
    if LOCAL_RANK == 0:
        print(msg)


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Toy example for debugging fully_shard()")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attn. heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Attention head size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size of input")
    parser.add_argument("--seq-length", type=int, default=512, help="Sequence length of input")
    parser.add_argument("--params-dtype", type=str, default="float32", help="Parameter dtype.")
    parser.add_argument(
        "--fp8-init", action="store_true", default=False, help="Initialize primary weights in FP8."
    )
    parser.add_argument("--recipe", type=str, default="mx_fp8_block_scaling", help="Quantizer type.",
                        choices=["delayed_scaling", "current_scaling", "mx_fp8_block_scaling"])
    parser.add_argument(
        "--layer-type",
        type=str,
        default="TransformerLayer",
        choices=["Linear", "LayerNormLinear", "LayerNormMLP", "MultiheadAttention", "TransformerLayer"],
        help="Transformer Engine layer type",
    )
    parser.add_argument( 
        "--iter", type=int, default=10, help="Number of iterations for forward pass"
    )
    parser.add_argument("--device", type=str, default="meta", help="Device to run the model on.",
                        choices=["cuda", "meta"])
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



## Methods to help initialize the TE model in an FSDP2 setting
## with required configurations based on command line args
def get_te_layer_from_string(layer_name):
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

def get_recipe_from_string(recipe, fp8_format=Format.HYBRID):
    if recipe == "delayed_scaling":
        return DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    elif recipe == "current_scaling":
        return Float8CurrentScaling(fp8_format=fp8_format)
    elif recipe == "mx_fp8_block_scaling":
        return MXFP8BlockScaling(fp8_format=fp8_format)
    else:
        raise ValueError(f"Unknown quantizer type: {recipe}")

def init_te_model(config):
    hidden_size = config.num_heads * config.head_dim
    args = [hidden_size, hidden_size]
    inp_shape = [config.seq_length, config.batch_size, hidden_size]
    out_shape = [config.seq_length, config.batch_size, hidden_size]
    if config.params_dtype == "float16":
        params_dtype = torch.float16
    elif config.params_dtype == "bfloat16":
        params_dtype = torch.bfloat16
    else:
        params_dtype = torch.float32
    kwargs = {
        "params_dtype": params_dtype,
    }
    layer_type = get_te_layer_from_string(config.layer_type)
    if layer_type == te.LayerNormLinear:
        args[1] *= 3  # QKV projection
        out_shape[-1] *= 3
    elif layer_type in [te.MultiheadAttention, te.TransformerLayer]:
        args[1] *= 4  # FFN hidden size
        args.append(config.num_heads)
        kwargs["fuse_qkv_params"] = True
        if layer_type is te.MultiheadAttention:
            kwargs["input_layernorm"] = True
    kwargs["device"] = config.device
    model = layer_type(*args, **kwargs)
    return model, inp_shape, out_shape

def get_device_mesh(world_size, sharding_dims):
    dist_print(f"sharding-dims:{sharding_dims}")
    device_ids = list(range(world_size))
    if sharding_dims == None:  # FSDP
        mesh = DeviceMesh("cuda", device_ids)
    elif len(sharding_dims) == 1:
        assert sharding_dims[0] == world_size
        mesh = DeviceMesh("cuda", device_ids)
    elif len(sharding_dims) == 2:  # HSDP
        assert sharding_dims[0] * sharding_dims[1] == world_size
        mesh = init_device_mesh(
            "cuda",
            (sharding_dims[0], sharding_dims[1]),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        assert False
    return mesh

    
def shard_model_with_fsdp2(model, mesh):
    # fully_shard has compatibilty issue with TransformerLayer at the moment.
    # If we wrap the TransformerLayer as well as its submodule, there seems to be
    # some sort of parameter group conflict which needs to be fixed.
    # Workaround at the moment is to only shard the inner submodules that
    # have parameters
    # TODO Varun: Need to fix fully_shard to work with TransformerLayer.
    all_modules = list(model.modules())
    if isinstance(model, te.TransformerLayer):
        sub_modules_to_wrap = [
            te.Linear,
            te.LayerNormLinear,
            te.LayerNormMLP,
        ]
        for sub_module in all_modules[::-1]:  # Reverse order traversal
            if type(sub_module) in sub_modules_to_wrap:
                fully_shard(sub_module, mesh=mesh)
    else:
        for sub_module in all_modules[::-1]:
            fully_shard(sub_module, mesh=mesh)

    return model

#### Methods to save the custom attributes of QuantizedTensors before sharding
#### them with FSDP2, and restore them after sharding.
def save_custom_attrs(module):
    custom_attrs = {}
    for name, param in module.named_parameters():
        if isinstance(param, QuantizedTensor):
            # Ignore FP8 metadata attributes. Otherwise we will save duplicate copies
            # for data/transpose FP8 tensors on top of FP8 tensors that FSDP2 will save.
            ignore_keys = [key for key in param.__dict__.keys() if key.startswith("_")]
        else:
            ignore_keys = []
        attrs = vars(param)
        custom_attrs[name] = {k: v for k, v in attrs.items() if k not in ignore_keys}
    return custom_attrs

def restore_custom_attrs(module, custom_attrs):
    for name, param in module.named_parameters():
        if name in custom_attrs:
            for attr_name, attr_value in custom_attrs[name].items():
                setattr(param, attr_name, attr_value)

@torch.no_grad()
def test_fp8_fsdp2_allgather(model):
    # Do manual allgather in fp32 and match against fp8 allgather done 
    # with fsdp2
    # FP32 manual weight allgather
    fp32_allgathered_params = {}
    for name, param in model.named_parameters():
        assert(isinstance(param, DTensor))
        local_tensor = param._local_tensor
        # assert(isinstance(local_tensor, QuantizedTensor))
        device_mesh = param.device_mesh
        if device_mesh.ndim == 1:
            # For 1D mesh (FSDP), use the only available dimension
            dist_group = device_mesh.get_group()
        else:
            # For 2D mesh (HSDP), use the shard dimension (last dimension)
            # which corresponds to where weights are actually sharded
            shard_dim = device_mesh.ndim - 1
            dist_group = device_mesh.get_group(mesh_dim=shard_dim)
        # Perform manual allgather on local_tensor. zeros_like will create hp tensor since torch_dispatch
        # for local_tensor will go down the dequantization route.
        gathered_tensor = [torch.zeros_like(local_tensor) for _ in range(dist.get_world_size(group=dist_group))]
        dist.all_gather(gathered_tensor, local_tensor.dequantize(), group=dist_group)
        full_tensor = torch.cat(gathered_tensor, dim=0)
        fp32_allgathered_params[name] = full_tensor
    # FP8 allgather using FSDP2
    for module in model.modules():
        # In case of Transformerlayer, just root module is sharded
        # at the moment.
        if hasattr(module, 'unshard'):
            module.unshard()
    # Make sure allgathered parameters match exactly
    for name, param in model.named_parameters():
        assert(torch.allclose(param.dequantize(), fp32_allgathered_params[name]))
    # Revert model to original sharded state
    for module in model.modules():
        if hasattr(module, 'reshard'):
            module.reshard()


def _train(args):
    global LOCAL_RANK
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
    device = torch.device(f"cuda:{LOCAL_RANK}")

    # FP8 Configuration
    fp8_format = Format.HYBRID
<<<<<<< HEAD
    fp8_recipe = get_recipe_from_string(args.recipe, fp8_format)
=======
    # fp8_recipe = DelayedScaling(fp8_format=fp8_format, amax_history_len=16, amax_compute_algo="max")
    fp8_recipe = Float8CurrentScaling(fp8_format=fp8_format)
    # fp8_recipe = MXFP8BlockScaling(fp8_format=fp8_format)
>>>>>>> 0a6e23c4cdd0111df4151ecec8ed5092fe850784

    build_model_context_args = {}
    if not args.fp8_init:
        # Build model context (FP8 init)
        build_model_context = nullcontext
    else:
        from transformer_engine.pytorch import fp8_model_init

        build_model_context = fp8_model_init
        build_model_context_args["enabled"] = True
        build_model_context_args["recipe"] = fp8_recipe

<<<<<<< HEAD
    dist_print(f"Memory before model init: {torch.cuda.memory_allocated(device)/1e6} MB")
    # Create the model on the meta/cuda device as per args
    with build_model_context(**build_model_context_args):
        model, inp_shape, out_shape = init_te_model(args)
    dist_print(f"Memory after model init on device {args.device}: {torch.cuda.memory_allocated(device)/1e6} MB")

    # Creating a DeviceMesh for fully_shard
    world_size = int(WORLD_SIZE)
    # Setup the sharding mesh for FSDP/HSDP
    mesh = get_device_mesh(world_size, args.sharding_dims)
=======
    if LOCAL_RANK == 0:
        print("Memory before model init:", torch.cuda.memory_allocated(device) / 1e6, "MB")
    # Create the model on meta device for deferred initialization
    with build_model_context(**build_model_context_args):
        model = SimpleNet(args.input_size, args.hidden_size, args.output_size, device="meta")
    if LOCAL_RANK == 0:
        print("Memory before FSDP:", torch.cuda.memory_allocated(device) / 1e6, "MB")
    if LOCAL_RANK == 0:
        print(f"Rank {LOCAL_RANK}: Model created on meta device...")
        print(f"Rank {LOCAL_RANK}: Applying FSDP fully_shard() to the model...")
    # Creating a DeviceMesh for fully_shard
    world_size = int(WORLD_SIZE)
    device_ids = list(range(world_size))
    if LOCAL_RANK == 0:
        print(f"sharding-dims:{args.sharding_dims}")

    # Setup the sharding mesh for FSDP/HSDP
    if args.sharding_dims == None:  # FSDP
        mesh = DeviceMesh("cuda", device_ids)
    elif len(args.sharding_dims) == 1:
        assert args.sharding_dims[0] == device_ids[-1] + 1
        mesh = DeviceMesh("cuda", device_ids)
    elif len(args.sharding_dims) == 2:  # HSDP
        assert args.sharding_dims[0] * args.sharding_dims[1] == device_ids[-1] + 1
        mesh = init_device_mesh(
            "cuda",
            (args.sharding_dims[0], args.sharding_dims[1]),
            mesh_dim_names=("replicate", "shard"),
        )
    else:
        assert False

    # Apply FSDP/HSDP on meta device first
    # FSDP will create sharded parameters that are still on meta device
>>>>>>> 0a6e23c4cdd0111df4151ecec8ed5092fe850784
    custom_attrs = save_custom_attrs(model)
    model = shard_model_with_fsdp2(model, mesh)
    restore_custom_attrs(model, custom_attrs)
    # model now has DTensors as its parameters

<<<<<<< HEAD
    if args.device == "meta":
        # After FSDP2 has been applied, materialize and initialize the sharded parameters
        # TE base.py's reset_parameters() handles DTensors with FP8 initialization
        for module in model.modules():
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        dist_print(f" Sharded parameters materialized and initialized on cuda device.")
    if args.fp8_init:
        test_fp8_fsdp2_allgather(model)
    dist_print(f"FSDP2 model in cuda, memory allocated: {torch.cuda.memory_allocated(device)/1e6} MB")
=======
    if LOCAL_RANK == 0:
        print(f"Rank {LOCAL_RANK}: FSDP applied, now materializing sharded parameters...")
        print("Memory after FSDP:", torch.cuda.memory_allocated(device) / 1e6, "MB")

    # After FSDP has been applied, materialize and initialize the sharded parameters
    # TransformerEngine's reset_parameters() now properly handles DTensors and FP8 initialization
    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    if LOCAL_RANK == 0:
        print(f"Rank {LOCAL_RANK}: Sharded parameters materialized and initialized.")
        print("Memory after materialization:", torch.cuda.memory_allocated(device) / 1e6, "MB")
>>>>>>> 0a6e23c4cdd0111df4151ecec8ed5092fe850784

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(args.iter):
        # Zero the parameter gradients
        optimizer.zero_grad()
        input_data = torch.randn(inp_shape).to(device)
        with te.autocast(enabled=True, recipe=fp8_recipe):
            output = model(input_data)
        target = torch.randn(out_shape).to(device)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        dist_print(f"Iteration {iteration} completed with loss {loss.item()}")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
