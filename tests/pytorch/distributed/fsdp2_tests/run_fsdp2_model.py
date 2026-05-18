#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 model sharding tests.

Run all tests (via torchrun + pytest):
  torchrun -m pytest <this_file> -v --tb=short

Run standalone (for debugging):
  torchrun <this_file> --recipe <recipe> [options]

Available --recipe values:
  DelayedScaling, Float8CurrentScaling, Float8BlockScaling,
  MXFP8BlockScaling, NVFP4BlockScaling

Other options:
  --fp8-init              Initialize weights in FP8
  --layer-type TYPE       Linear, LayerNormLinear, LayerNormMLP,
                          MultiheadAttention, TransformerLayer (default)
  --sharding-dims N [M]   FSDP dims, e.g. "2" or "2 2" for HSDP
  --num-layers N          Number of layers (default: 4)
  --iter N                Training iterations (default: 10)
  --device cuda|meta      Device for init (default: meta)
"""

import gc
import os
import sys
import argparse
from types import SimpleNamespace
from contextlib import nullcontext

import pytest

import transformer_engine.pytorch as te
import transformer_engine.common.recipe
from transformer_engine.pytorch.tensor import NVFP4Tensor

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

from fsdp2_utils import get_recipe_from_string, save_custom_attrs, restore_custom_attrs


def dist_print(msg):
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        print(msg)


def _parse_args(argv=None, namespace=None):
    parser = argparse.ArgumentParser(description="Toy example for debugging fully_shard()")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of attn. heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Attention head size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size of input")
    parser.add_argument("--seq-length", type=int, default=128, help="Sequence length of input")
    parser.add_argument("--params-dtype", type=str, default="float32", help="Parameter dtype.")
    parser.add_argument(
        "--fp8-init",
        action="store_true",
        default=False,
        help="Initialize primary weights in FP8.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default="MXFP8BlockScaling",
        help="Quantizer type.",
        choices=[
            "DelayedScaling",
            "Float8CurrentScaling",
            "Float8BlockScaling",
            "MXFP8BlockScaling",
            "NVFP4BlockScaling",
        ],
    )
    parser.add_argument(
        "--layer-type",
        type=str,
        default="TransformerLayer",
        choices=[
            "Linear",
            "LayerNormLinear",
            "LayerNormMLP",
            "MultiheadAttention",
            "TransformerLayer",
        ],
        help="Transformer Engine layer type",
    )
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers in the model")
    parser.add_argument(
        "--iter", type=int, default=10, help="Number of iterations for forward pass"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="meta",
        help="Device to run the model on.",
        choices=["cuda", "meta"],
    )
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
    kwargs["device"] = config.device

    layer_type = get_te_layer_from_string(config.layer_type)
    # We are creating model in a way so that we can test both reshard_after_forward=True/False cases.
    # more details below.
    if layer_type in [te.MultiheadAttention, te.TransformerLayer]:
        # For this case, we are creating a model that resemebles production use-cases
        # wherein there are mltiple TransformerLayers in the model. And we would need
        # to shard each transformer layer. Since each transformer layer is not a root module,
        # FSDP2's fully_shard assigns reshard_after_forward=False for all parameters of the model.
        args[1] *= 4  # FFN hidden size
        args.append(config.num_heads)
        kwargs["fuse_qkv_params"] = True
        if layer_type is te.MultiheadAttention:
            kwargs["input_layernorm"] = True
        model = nn.Sequential(*[layer_type(*args, **kwargs) for _ in range(config.num_layers)])
    elif layer_type == te.LayerNormLinear:
        # For this case, we are creating a model with just one LayerNormLinear layer
        # so that the model itself is a root module, and FSDP2's fully_shard assigns
        # reshard_after_forward=True for the parameters of these model.
        args[1] *= 3  # QKV projection
        out_shape[-1] *= 3
        model = layer_type(*args, **kwargs)
    else:
        model = layer_type(*args, **kwargs)

    return model, inp_shape, out_shape


def get_device_mesh(world_size, sharding_dims):
    dist_print(f"sharding-dims:{sharding_dims}")
    device_ids = list(range(world_size))
    if sharding_dims is None:  # FSDP
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
    for child in model.children():
        fully_shard(child, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


@torch.no_grad()
def _check_fp8_fsdp2_allgather(model):
    # Do manual allgather in fp32 and match against fp8 allgather done
    # with fsdp2
    # FP32 manual weight allgather
    fp32_allgathered_params = {}
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor)
        local_tensor = param._local_tensor
        device_mesh = param.device_mesh
        dist_group = (
            device_mesh.get_group(mesh_dim="shard")
            if device_mesh.ndim > 1
            else device_mesh.get_group()
        )
        # Perform manual allgather on local_tensor. zeros_like will create hp tensor since
        # torch_dispatch for local_tensor will go down the dequantization route.
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
        # NVFP4 scale unpad/repad through FSDP2 introduces small numerical
        # differences vs the manual dequantize-then-allgather path.
        if isinstance(param, NVFP4Tensor):
            tols = dict(atol=5e-4, rtol=5e-3)
        else:
            tols = {}
        torch.testing.assert_close(param.dequantize(), fp32_allgathered_params[name], **tols)
    # Revert model to original sharded state
    for module in model.modules():
        # Not all modules are wrapped/sharded with FSDP2.
        if hasattr(module, "reshard"):
            module.reshard()


def _run_training(args):
    """Core training logic. Assumes dist is already initialized."""
    device = torch.device(f"cuda:{int(os.getenv('LOCAL_RANK', '0'))}")
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    # FP8 Configuration
    fp8_recipe = get_recipe_from_string(args.recipe)

    build_model_context_args = {}
    if not args.fp8_init:
        # Build model context (FP8 init)
        build_model_context = nullcontext
    else:
        from transformer_engine.pytorch import fp8_model_init

        build_model_context = fp8_model_init
        build_model_context_args["enabled"] = True
        build_model_context_args["recipe"] = fp8_recipe

    dist_print(f"Memory before model init: {torch.cuda.memory_allocated(device) / 1e6} MB")
    # Create the model on the meta/cuda device as per args
    with build_model_context(**build_model_context_args):
        model, inp_shape, out_shape = init_te_model(args)
    dist_print(
        f"Memory after model init on device {args.device}:"
        f" {torch.cuda.memory_allocated(device) / 1e6} MB"
    )

    # Creating a DeviceMesh for fully_shard
    # Setup the sharding mesh for FSDP/HSDP
    mesh = get_device_mesh(world_size, args.sharding_dims)
    custom_attrs = save_custom_attrs(model)
    model = shard_model_with_fsdp2(model, mesh)
    restore_custom_attrs(model, custom_attrs)
    # model now has DTensors as its parameters

    if args.device == "meta":
        # After FSDP2 has been applied, materialize and initialize the sharded parameters
        # TE base.py's reset_parameters() handles DTensors with FP8 initialization
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        dist_print(f" Sharded parameters materialized and initialized on cuda device.")

    dist_print(
        f"FSDP2 model in cuda, memory allocated: {torch.cuda.memory_allocated(device) / 1e6} MB"
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for iteration in range(args.iter):
        # Zero the parameter gradients
        optimizer.zero_grad()

        input_data = torch.randn(inp_shape, device=device)
        target = torch.randn(out_shape, device=device)

        # NVFP4BlockScaling requires bfloat16 inputs in both the forward and backward passes.
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.recipe == "NVFP4BlockScaling"
            else nullcontext()
        ):
            with te.autocast(enabled=True, recipe=fp8_recipe):
                output = model(input_data)
                loss = F.mse_loss(output, target)

        loss.backward()
        optimizer.step()
        dist_print(f"Iteration {iteration} completed with loss {loss.item()}")

    # Some of the FSDP states are lazy initialized during FSDP forward pass
    # so testing fp8 allgather at the end of the training loop.
    if args.fp8_init:
        _check_fp8_fsdp2_allgather(model)


def _train(args):
    """Standalone entry point with full dist lifecycle."""
    assert "TORCHELASTIC_RUN_ID" in os.environ
    WORLD_RANK = int(os.getenv("RANK", "0"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    assert LOCAL_SIZE == WORLD_SIZE

    torch.cuda.set_device(LOCAL_RANK)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    assert dist.is_nccl_available()
    dist.init_process_group(
        backend="nccl",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
    )
    try:
        _run_training(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()
        gc.collect()

    return 0


# ── Pytest test function ─────────────────────────────────────────────

NUM_PROCS = int(os.environ.get("WORLD_SIZE", "1"))


@pytest.mark.parametrize("sharding_dims", [[NUM_PROCS], [2, NUM_PROCS // 2]])
@pytest.mark.parametrize("fp8_init", [False, True])
@pytest.mark.parametrize("layer_type", ["LayerNormLinear", "TransformerLayer"])
def test_distributed(recipe_name, fp8_init, sharding_dims, layer_type):
    if recipe_name == "MXFP8BlockScaling" and fp8_init and len(sharding_dims) == 2:
        pytest.xfail(
            "MXFP8BlockScaling + fp8_init + HSDP: fsdp_post_all_gather receives fewer "
            "all_gather_outputs than the number of tensors sent by fsdp_pre_all_gather "
            "when the HSDP shard dimension is trivial (size 1). MXFP8 sends 2 tensors "
            "(data + scale_inv, both uint8) but gets back 1. Float8Tensor avoids this by "
            "sending only 1 tensor (scale is per-tensor metadata). Fix: concatenate MXFP8 "
            "data and scale_inv into a single buffer in pre_all_gather, split in post."
        )

    if recipe_name == "Float8BlockScaling" and fp8_init:
        pytest.xfail(
            "Float8BlockScaling + fp8_init: scale inverse padding is not handled "
            "correctly during FSDP2 all-gather slice ops."
        )
    if recipe_name == "NVFP4BlockScaling" and fp8_init and layer_type == "TransformerLayer":
        pytest.xfail(
            "NVFP4BlockScaling + fp8_init + TransformerLayer: "
            "_check_fp8_fsdp2_allgather numerical error compounds across multiple "
            "linear layers in the transformer block (up to ~1e-2 max abs diff). "
            "LayerNormLinear passes with relaxed tolerances. "
            "NVFP4 + FSDP2 training is validated by run_fsdp2_fused_adam.py."
        )
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    args = SimpleNamespace(
        recipe=recipe_name,
        fp8_init=fp8_init,
        sharding_dims=list(sharding_dims),
        layer_type=layer_type,
        seed=42,
        num_heads=8,
        head_dim=64,
        batch_size=16,
        seq_length=128,
        params_dtype="float32",
        num_layers=4,
        iter=10,
        device="meta",
    )
    _run_training(args)


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
