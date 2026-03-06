#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import os
import sys
import shutil
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import transformer_engine.pytorch as te
import transformer_engine.common.recipe
from transformer_engine.pytorch import QuantizedTensor
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor
import torch.nn.functional as F
from torch import nn, optim
from torch.distributed import DeviceMesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh

LOCAL_RANK = None

# Needed for `torch.distributed.checkpoint.{save,load}` because
# multiple processes need to write to the same directory.
SHARED_TMP_DIR = "/tmp/pytest-shared-tmp"


@dataclass
class AppState(Stateful):
    """AppState for FSDP2 checkpoint via Torch DCP.

    Adapted from https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def state_dict(self):
        """
        Get the state dict for the model, optimizer, scheduler, and step.
        This factory both retrieves the model state dictionary when saving
        checkpoints and initializes a destination for the state read from
        DCP checkpoint files when loading checkpoints.
        """
        model_state_dict, optimizer_state_dict = get_state_dict(self.model, self.optimizer)
        for fqn in list(model_state_dict.keys()):
            # Get the model parameter.
            model_param = model_state_dict[fqn]
            if isinstance(model_param, DTensor):
                model_param = model_param.to_local()
            if model_param.numel() == 0 and fqn in optimizer_state_dict["state"]:
                # Empty model parameter. Clear the associated optimizer state
                # when initializing the optimizer state upon DCP load, because
                # empty optimizer state DTensors are not checkpointed with DCP,
                # yet get_state_dict / _init_optim_state produce empty Tensors.
                # TransformerEngine uses empty Tensors for dummy Parameters.
                optimizer_state_dict["state"][fqn] = {}
            if fqn.endswith("_extra_state"):
                # Evict `_extra_state` quantization data from model checkpoint.
                model_state_dict.pop(fqn)
        return {
            "model": model_state_dict,
            "optim": optimizer_state_dict,
        }

    def load_state_dict(self, state_dict: dict):
        """
        Load the state dict for the model, optimizer, scheduler, and step.
        Given the checkpoint-loaded state_dict, set the state of the model,
        optimizer, scheduler, step, and epoch to the values in state_dict.
        """
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optim_state_dict=state_dict["optim"],
            # Non-strict checkpoint loading ignores empty optimizer states,
            # skips loading non-FP8 checkpoint weights (e.g. _extra_state).
            options=StateDictOptions(strict=False),
        )


def dist_print(msg):
    if LOCAL_RANK == 0:
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
        help='FSDP/HSDP sharding dimensions ("dp_replicate", "dp_shard", "tp")',
    )
    args = parser.parse_args(argv, namespace)
    if args.sharding_dims:
        assert len(args.sharding_dims) <= 3
    if len(args.sharding_dims) >= 3:
        # Set the TP size in args.
        args.tp_size = args.sharding_dims[2]
    else:
        args.tp_size = 1
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


def get_recipe_from_string(recipe):
    return getattr(transformer_engine.common.recipe, recipe)()


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
    kwargs["tp_size"] = config.tp_size

    # DeviceMesh / DTensor-related model parameter operations!
    # NOTE(@cspades): `set_device_mesh` works, but needs to be called before reset_parameters.
    # If not using meta device initialization, reset_parameters is called during __init__.
    if config.tp_size > 1:  # (H/F)SDP-TP
        assert "dp_shard" in config.mesh.mesh_dim_names
        assert "tp" in config.mesh.mesh_dim_names
        dist_print(f"Tensor parallelism activated with size: {config.tp_size}")
        # For TP shards as DTensors.
        kwargs["tp_mesh"] = config.mesh["tp"]
        # For per-tensor quantization recipes with TP.
        kwargs["weight_mesh"] = config.mesh["dp_shard", "tp"]._flatten("weight_mesh")
    elif len(config.mesh.mesh_dim_names) > 1:  # HSDP
        assert "dp_shard" in config.mesh.mesh_dim_names
        # HSDP (DP-Repl, DP-Shard) requires a call to `set_device_mesh(weight_mesh)`.
        # Used for per-tensor quantization recipes like Float8CurrentScaling.
        kwargs["weight_mesh"] = config.mesh["dp_shard"]  # Only sharding with FSDP.

    layer_type = get_te_layer_from_string(config.layer_type)
    # We are creating model in a way so that we can test both reshard_after_forward=True/False cases.
    # more details below.
    if layer_type in [
        te.TransformerLayer,
        te.MultiheadAttention,
        te.LayerNormMLP,
    ]:
        # For this case, we are creating a model that resemebles production use-cases
        # wherein there are mltiple TransformerLayers in the model. And we would need
        # to shard each transformer layer. Since each transformer layer is not a root module,
        # FSDP2's fully_shard assigns reshard_after_forward=False for all parameters of the model.
        args[1] *= 4  # FFN hidden size
        args.append(config.num_heads)
        kwargs["fuse_qkv_params"] = True
        if layer_type is te.MultiheadAttention:
            kwargs["input_layernorm"] = True
        if config.tp_size > 1:
            # Activate TP in TE.
            kwargs["set_parallel_mode"] = True
        # Initialize model.
        model = nn.Sequential(*[layer_type(*args, **kwargs) for _ in range(config.num_layers)])
    elif layer_type in [te.LayerNormLinear, te.Linear]:
        # For this case, we are creating a model with just one LayerNormLinear layer
        # so that the model itself is a root module, and FSDP2's fully_shard assigns
        # reshard_after_forward=True for the parameters of these model.
        args[1] *= 3  # QKV projection
        out_shape[-1] *= 3
        if config.tp_size > 1:
            # Activate TP in TE.
            kwargs["parallel_mode"] = "column"
            # Modify output shape for column-parallel Linear.
            out_shape[-1] //= config.tp_size
        # Initialize model.
        model = layer_type(*args, **kwargs)
    else:
        # Other TE module. Just ambiguously initialize it.
        model = layer_type(*args, **kwargs)

    return model, inp_shape, out_shape


def get_device_mesh(world_size, sharding_dims):
    dist_print(f"sharding-dims: {sharding_dims}")
    device_ids = list(range(world_size))
    # FSDP
    if sharding_dims is None or len(sharding_dims) == 1:
        assert sharding_dims is None or sharding_dims[0] == world_size
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp_shard",),
        )
    # HSDP
    elif len(sharding_dims) == 2:
        assert sharding_dims[0] * sharding_dims[1] == world_size
        mesh = init_device_mesh(
            "cuda",
            (sharding_dims[0], sharding_dims[1]),
            mesh_dim_names=("dp_replicate", "dp_shard"),
        )
    # (H/F)SDP-TP
    elif len(sharding_dims) == 3:
        assert sharding_dims[0] * sharding_dims[1] * sharding_dims[2] == world_size
        mesh = init_device_mesh(
            "cuda",
            (sharding_dims[0], sharding_dims[1], sharding_dims[2]),
            mesh_dim_names=("dp_replicate", "dp_shard", "tp"),
        )
    else:
        # Unsupported topology.
        assert False
    return mesh


def shard_model_with_fsdp2(model, mesh):
    assert "dp_shard" in mesh.mesh_dim_names
    dp_dims = (
        ("dp_replicate", "dp_shard") if "dp_replicate" in mesh.mesh_dim_names else ("dp_shard",)
    )
    for child in model.children():
        fully_shard(child, mesh=mesh[dp_dims])
    fully_shard(model, mesh=mesh[dp_dims])
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


def _train(args):
    """
    Torch Distributed Initialization
    """
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

    # Create a DeviceMesh for fully_shard.
    world_size = int(WORLD_SIZE)
    # Setup the sharding mesh for FSDP/HSDP.
    mesh = get_device_mesh(world_size, args.sharding_dims)
    args.mesh = mesh

    """
    TransformerEngine Model Initialization
    """
    # FP8 Configuration
    fp8_recipe = get_recipe_from_string(args.recipe)

    # Model initialization context.
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

    # Avoid passing custom attributes to FSDP2.
    custom_attrs = save_custom_attrs(model)
    # Fully-shard the model. Will convert model parameters into DTensor
    # if not already converted by TP.
    model = shard_model_with_fsdp2(model, mesh)
    # Restore custom attributes on parameters.
    restore_custom_attrs(model, custom_attrs)

    if args.device == "meta":
        # After FSDP2 has been applied, materialize and initialize the sharded parameters
        # TE base.py's reset_parameters() handles DTensors with FP8 initialization.
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        dist_print(f" Sharded parameters materialized and initialized on cuda device.")

    dist_print(
        f"FSDP2 model in cuda, memory allocated: {torch.cuda.memory_allocated(device) / 1e6} MB"
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    """
    FSDP2 Training
    """
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
        test_fp8_fsdp2_allgather(model)

    """
    DCP Checkpoint Testing
    """
    # Compute the pre-save model loss to the last random input
    # with respect to the last random target.
    model.eval()
    with (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.recipe == "NVFP4BlockScaling"
        else nullcontext()
    ):
        with te.autocast(enabled=True, recipe=fp8_recipe):
            output = model(input_data)
            pre_save_loss = F.mse_loss(output, target)

    # Save deep copy of the model and optimizer state before checkpointing.
    # NOTE(@cspades): deepcopy has issues with DTensors. Just clone().
    s1 = {}
    for key, val in model.state_dict().items():
        s1[key] = val.clone()
    optim_state_dict = optimizer.state_dict()
    o1 = {"state": {}}
    for idx, state in optim_state_dict["state"].items():
        o1_state = o1["state"].setdefault(idx, {})
        for key, val in state.items():
            o1_state[key] = val.clone()
    o1["param_groups"] = deepcopy(optim_state_dict["param_groups"])

    # Write model to checkpoint.
    CKPT_DIR = (
        Path(SHARED_TMP_DIR)
        / "run_fsdp2_model"
        / f"dcp-{'_'.join(str(x) for x in args.sharding_dims)}-{args.layer_type}-{args.recipe}-fp8_init_{args.fp8_init}"
    )
    CKPT_DIR.mkdir(parents=True, exist_ok=True, mode=0o777)
    state_dict = {"app": AppState(model=model, optimizer=optimizer)}
    torch.distributed.checkpoint.save(state_dict, checkpoint_id=str(CKPT_DIR))

    # Perform an extra training step to change the weights such that
    # state parity tests will fail unless the checkpoint is loaded
    # without any errors or incongruities vs. the saved model state.
    model.train()
    for iteration in range(args.iter):
        optimizer.zero_grad()
        with (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if args.recipe == "NVFP4BlockScaling"
            else nullcontext()
        ):
            with te.autocast(enabled=True, recipe=fp8_recipe):
                output = model(torch.randn(inp_shape).to(device))
                loss = F.mse_loss(output, torch.randn(out_shape).to(device))
        loss.backward()
        optimizer.step()

    # Load the checkpoint.
    state_dict = {"app": AppState(model=model, optimizer=optimizer)}
    torch.distributed.checkpoint.load(state_dict=state_dict, checkpoint_id=str(CKPT_DIR))

    # FIXME(@cspades): DelayedScaling checkpointing has tiny uint8 parity issues
    # that affects the dequantized model state. Only test loss parity.
    if args.recipe != "DelayedScaling" and args.fp8_init:
        # Validate checkpoint parity with pre-save state dictionaries.
        # Compare pre-save and post-load model state dictionaries.
        s2 = model.state_dict()
        nonempty_model_state = False
        for key in s1.keys() | s2.keys():
            if key.endswith("_extra_state"):
                # Don't parity test _extra_state. Shape can change after reset_parameters().
                continue
            v1 = s1.get(key, None)
            if isinstance(v1, DTensor):
                v1 = v1.to_local()
            v2 = s2.get(key, None)
            if isinstance(v2, DTensor):
                v2 = v2.to_local()
            assert (
                v1 is not None and v2 is not None
            ), f"[{key} Not Found] Original Param: {v1} | Checkpoint Param: {v2}"
            assert (
                v1.shape == v2.shape
            ), f"[Checkpoint Param {key} Shape Mismatch] {v1.shape} != {v2.shape}"
            assert torch.allclose(v1, v2), f"[Checkpoint Param {key} Value Mismatch] {v1} != {v2}"
            nonempty_model_state = True
        assert nonempty_model_state, "Model state should not be empty for evenly-sharded DTensors!"

        # Compare pre-save and post-load optimizer state dictionaries.
        o2 = optimizer.state_dict()
        nonempty_optim_state = False
        for param_id in o1["state"].keys() | o2["state"].keys():
            param_state_1 = o1["state"].get(param_id, None)
            param_state_2 = o2["state"].get(param_id, None)
            assert param_state_1 is not None and param_state_2 is not None, (
                f"[{param_id} Not Found] Original Optim State: {param_state_1} | Checkpoint Optim"
                f" State: {param_state_2}"
            )
            for key in param_state_1.keys() | param_state_2.keys():
                v1 = param_state_1.get(key, None)
                if isinstance(v1, DTensor):
                    v1 = v1.to_local()
                v2 = param_state_2.get(key, None)
                if isinstance(v2, DTensor):
                    v2 = v2.to_local()
                assert v1 is not None and v2 is not None, (
                    f"[{param_id} {key} Not Found] Original Optim State: {v1} | Checkpoint Optim"
                    f" State: {v2}"
                )
                assert (
                    v1.shape == v2.shape
                ), f"[Optim State {param_id} {key} Shape Mismatch] {v1.shape} != {v2.shape}"
                assert torch.allclose(
                    v1, v2
                ), f"[Optim State {param_id} {key} Value Mismatch] {v1} != {v2}"
                nonempty_optim_state = True  # Optimizer state depends on wgrad, verify this!
        assert (
            nonempty_optim_state
        ), "Optimizer state should not be empty for evenly-sharded DTensors!"
        assert len(o1["param_groups"]) == len(o2["param_groups"]), (
            f"[Optim State Param Groups Length Mismatch] {o1['param_groups']} !="
            f" {o2['param_groups']}"
        )
        for i in range(len(o2["param_groups"])):
            for key in o1["param_groups"][i].keys():
                v1 = o1["param_groups"][i][key]
                v2 = o2["param_groups"][i][key]
                assert v1 == v2, f"[Optim State Param Group {i} {key} Value Mismatch] {v1} != {v2}"

    # Validate post-load model loss.
    model.eval()
    with (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.recipe == "NVFP4BlockScaling"
        else nullcontext()
    ):
        with te.autocast(enabled=True, recipe=fp8_recipe):
            output = model(input_data)
            post_load_loss = F.mse_loss(output, target)
    # Allow for 1% disparity due to _extra_state disparity.
    assert torch.allclose(
        pre_save_loss, post_load_loss, rtol=1e-2
    ), f"Pre-Save Loss: {pre_save_loss} != Post-Load Loss: {post_load_loss}"

    # Clean up temporary checkpoint directory.
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        shutil.rmtree(CKPT_DIR)
    torch.distributed.barrier()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(_train(_parse_args()))
