#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Standalone test for FP8 FSDP2 all-gather correctness.

Verifies that FSDP2's internal all-gather of FP8 parameters produces the same
result as a manual all-gather of dequantized FP32 values.
"""

import argparse
import os
import sys
from contextlib import nullcontext

import transformer_engine.pytorch as te
import transformer_engine.common.recipe
from transformer_engine.pytorch import fp8_model_init
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import optim
from torch.distributed.tensor import DTensor
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
from torch import nn

LOCAL_RANK = None

# Fixed model dimensions — this test focuses on allgather correctness, not model flexibility.
_NUM_HEADS = 8
_HEAD_DIM = 128
_HIDDEN_SIZE = _NUM_HEADS * _HEAD_DIM
_FFN_SIZE = _HIDDEN_SIZE * 4
_NUM_LAYERS = 2
_BATCH_SIZE = 4
_SEQ_LEN = 32


def dist_print(msg):
    if LOCAL_RANK == 0:
        print(msg)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Test FP8 FSDP2 all-gather correctness with TransformerLayer."
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="DelayedScaling",
        choices=[
            "DelayedScaling",
            "Float8CurrentScaling",
            "Float8BlockScaling",
            "MXFP8BlockScaling",
            "NVFP4BlockScaling",
        ],
    )
    parser.add_argument(
        "--sharding-dims",
        type=int,
        nargs="+",
        required=True,
        help=(
            'Sharding mesh dimensions: ("dp_shard",), ("dp_replicate", "dp_shard"), '
            'or ("dp_replicate", "dp_shard", "tp")'
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    assert len(args.sharding_dims) <= 3
    args.tp_size = args.sharding_dims[2] if len(args.sharding_dims) >= 3 else 1
    return args


def _get_recipe(name):
    return getattr(transformer_engine.common.recipe, name)()


def _get_device_mesh(world_size, sharding_dims):
    dist_print(f"sharding-dims: {sharding_dims}")
    if len(sharding_dims) == 1:
        assert sharding_dims[0] == world_size
        return init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
    elif len(sharding_dims) == 2:
        assert sharding_dims[0] * sharding_dims[1] == world_size
        return init_device_mesh(
            "cuda",
            (sharding_dims[0], sharding_dims[1]),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
    else:
        assert sharding_dims[0] * sharding_dims[1] * sharding_dims[2] == world_size
        return init_device_mesh(
            "cuda",
            (sharding_dims[0], sharding_dims[1], sharding_dims[2]),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )


def _build_model(args):
    kwargs = {
        "params_dtype": torch.float32,
        "device": "meta",
        "tp_size": args.tp_size,
        "fuse_qkv_params": True,
    }
    if args.tp_size > 1:
        kwargs["tp_mesh"] = args.mesh["tp"]
        kwargs["weight_mesh"] = args.mesh["dp_shard", "tp"]._flatten("weight_mesh")
        kwargs["set_parallel_mode"] = True
    elif "dp_replicate" in args.mesh.mesh_dim_names:
        kwargs["weight_mesh"] = args.mesh["dp_shard"]

    model = nn.Sequential(
        *[
            te.TransformerLayer(_HIDDEN_SIZE, _FFN_SIZE, _NUM_HEADS, **kwargs)
            for _ in range(_NUM_LAYERS)
        ]
    )
    inp_shape = [_SEQ_LEN, _BATCH_SIZE, _HIDDEN_SIZE]
    return model, inp_shape


def _shard_model(model, mesh):
    dp_dims = (
        ("dp_replicate", "dp_shard") if "dp_replicate" in mesh.mesh_dim_names else ("dp_shard",)
    )
    for child in model.children():
        fully_shard(child, mesh=mesh[dp_dims])
    fully_shard(model, mesh=mesh[dp_dims])
    return model


@torch.no_grad()
def _test_fp8_fsdp2_allgather(model):
    """
    Compare the result of the FP8 AG by FSDP2 with a manual AG in FP32
    after dequantizing the FP8 values.
    """
    # FP32 manual weight allgather
    fp32_allgathered_params = {}
    for name, param in model.named_parameters():
        assert isinstance(
            param, DTensor
        ), f"[test_fp8_fsdp2_allgather] {param} should be a DTensor."
        local_tensor = param._local_tensor
        device_mesh = param.device_mesh
        dist_group = (
            device_mesh.get_group(mesh_dim="dp_shard")
            if device_mesh.ndim > 1
            else device_mesh.get_group()
        )
        # Perform manual allgather on local_tensor. zeros_like will create hp tensor since torch_dispatch
        # for local_tensor will go down the dequantization route.
        gathered_tensor = [
            torch.zeros_like(local_tensor) for _ in range(dist.get_world_size(group=dist_group))
        ]
        dist.all_gather(gathered_tensor, local_tensor.dequantize(), group=dist_group)
        full_tensor = torch.cat(gathered_tensor, dim=0)
        fp32_allgathered_params[name] = full_tensor
    # FP8 allgather using FSDP2
    for module in model.modules():
        # Not all modules are wrapped/sharded with FSDP2.
        if hasattr(module, "unshard"):
            module.unshard()
    # Make sure allgathered parameters match exactly
    for name, param in model.named_parameters():
        if isinstance(param, DTensor):
            # Will still be a DTensor in the case of TP, even after FSDP2 AG,
            # because we wrap our weights as DTensor shards over the TP group.
            param = param._local_tensor
        torch.testing.assert_close(param.dequantize(), fp32_allgathered_params[name])
    # Revert model to original sharded state
    for module in model.modules():
        # Not all modules are wrapped/sharded with FSDP2.
        if hasattr(module, "reshard"):
            module.reshard()


def _main(args):
    global LOCAL_RANK
    assert "TORCHELASTIC_RUN_ID" in os.environ
    WORLD_RANK = int(os.getenv("RANK", "0"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

    torch.cuda.set_device(WORLD_RANK)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dist.init_process_group(backend="nccl", rank=WORLD_RANK, world_size=WORLD_SIZE)
    device = torch.device(f"cuda:{LOCAL_RANK}")

    mesh = _get_device_mesh(WORLD_SIZE, args.sharding_dims)
    args.mesh = mesh

    fp8_recipe = _get_recipe(args.recipe)

    with fp8_model_init(enabled=True, recipe=fp8_recipe):
        model, inp_shape = _build_model(args)

    model = _shard_model(model, mesh)

    for module in model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()

    # Run a training step to initialize FSDP2 lazy state and update quantization
    # scales before testing the allgather. Block-scaling formats (Float8BlockScaling,
    # NVFP4BlockScaling) only exhibit allgather inconsistencies after weight updates.
    input_data = torch.randn(inp_shape, device=device)
    target = torch.randn(inp_shape, device=device)
    nvfp4_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.recipe == "NVFP4BlockScaling"
        else nullcontext()
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    with nvfp4_ctx, te.autocast(enabled=True, recipe=fp8_recipe):
        output = model(input_data)
        loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    _test_fp8_fsdp2_allgather(model)
    dist_print("test_fp8_fsdp2_allgather passed.")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(_main(_parse_args()))
