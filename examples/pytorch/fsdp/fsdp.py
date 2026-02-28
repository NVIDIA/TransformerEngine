# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import argparse

from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import always_wrap_policy, transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    Format,
    DelayedScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
)
from transformer_engine.pytorch.distributed import prepare_te_modules_for_fsdp

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))

# RNG state tracker for checkpointing
rng_seed = 1234
torch.manual_seed(rng_seed)
torch.cuda.manual_seed(rng_seed)
CUDA_RNG_STATES_TRACKER = te.distributed.CudaRNGStatesTracker()
CUDA_RNG_STATES_TRACKER.add("model-parallel-rng", rng_seed)


def get_cuda_rng_tracker():
    return CUDA_RNG_STATES_TRACKER


def apply_fsdp_checkpointing(model, blocks):
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    wrapper = lambda m: checkpoint_wrapper(
        m,
        checkpoint_fn=te.distributed.checkpoint,
        use_reentrant=False,
        get_rng_state_tracker=get_cuda_rng_tracker,
    )
    check_fn = lambda submodule: isinstance(submodule, blocks)
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=wrapper, check_fn=check_fn)


def lowercase(s):
    return str(s).lower()


def torch_dtype(d):
    typemap = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if lowercase(d) not in typemap.keys():
        raise TypeError
    return typemap[lowercase(d)]


def precision(d):
    typemap = ["fp32", "fp16", "fp8", "mxfp8", "nvfp4"]
    if lowercase(d) not in typemap:
        raise TypeError
    return lowercase(d)


te_layer_map = {
    "linear": te.Linear,
    "layernorm": te.LayerNorm,
    "rmsnorm": te.RMSNorm,
    "layernormlinear": te.LayerNormLinear,
    "layernormmlp": te.LayerNormMLP,
    "multiheadattention": te.MultiheadAttention,
    "transformerlayer": te.TransformerLayer,
}


def te_layer(l):
    if l is not None:
        if lowercase(l) not in te_layer_map.keys():
            raise TypeError
        return te_layer_map[lowercase(l)]
    return None


def get_layer_args(opts):
    hidden_size = opts.num_heads * opts.head_dim
    layer_args = (hidden_size,)
    layer_kwargs = {
        # "params_dtype": opts.dtype,
        "device": "cuda" if opts.no_defer_init else "meta",
        "get_rng_state_tracker": get_cuda_rng_tracker,
    }
    if opts.layer_type in [te.Linear, te.LayerNormLinear, te.LayerNormMLP]:
        ffn_hidden_size = 3 * hidden_size if opts.num_layers == 1 else hidden_size
        layer_args += (ffn_hidden_size,)
        layer_kwargs["bias"] = True
        if opts.layer_type == te.LayerNormMLP:
            layer_kwargs["seq_length"] = opts.seq_length
    elif opts.layer_type == te.MultiheadAttention:
        layer_args += (opts.num_heads,)
        layer_kwargs["fuse_qkv_params"] = True
        layer_kwargs["input_layernorm"] = True
    elif opts.layer_type == te.TransformerLayer:
        layer_args += (3 * hidden_size, opts.num_heads)
        layer_kwargs["fuse_qkv_params"] = True
        layer_kwargs["seq_length"] = opts.seq_length
    return layer_args, layer_kwargs


class StoreExplicitAction(argparse.Action):
    """Custom action that tracks whether an argument was explicitly set."""

    def __init__(self, option_strings, dest, type=None, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.type_converter = type  # Store the type converter

    def __call__(self, parser, namespace, values, option_string=None):
        # Apply the type converter if one was provided
        if self.type_converter is not None:
            try:
                values = self.type_converter(values)
            except (ValueError, TypeError) as e:
                raise argparse.ArgumentTypeError(f"invalid {self.dest} value: {values}") from e

        setattr(namespace, self.dest, values)
        setattr(namespace, f"{self.dest}_explicitly_set", True)


class StoreTrueExplicitAction(argparse.Action):
    """Custom action for store_true that tracks whether flag was explicitly set."""

    def __init__(self, option_strings, dest, default=False, required=False, help=None):
        super().__init__(
            option_strings, dest, nargs=0, const=True, default=default, required=required, help=help
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        setattr(namespace, f"{self.dest}_explicitly_set", True)


def parse_fsdp_args():
    parser = argparse.ArgumentParser(
        description="Run Transformer Engine modules with the "
        + "torch.distributed.fsdp.FullyShardedDataParallel strategy."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print out information from all GPUs instead of only the root GPU-0.",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Input batch size.")
    parser.add_argument("-s", "--seq-length", type=int, default=1048, help="Input sequence length.")
    parser.add_argument(
        "-n", "--num-heads", type=int, default=16, help="Number of attention heads."
    )
    parser.add_argument(
        "-d",
        "--head-dim",
        type=int,
        default=128,
        help="Dimension of each attention head (number of KV channels).",
    )
    parser.add_argument(
        "-i", "--num-iters", type=int, default=5, help="Number of dummy 'training' iterations."
    )
    parser.add_argument(
        "-k",
        "--num-layers",
        type=int,
        default=3,
        help="Number of modules chained together with nn.Sequential.",
    )
    parser.add_argument(
        "--layer-type",
        type=te_layer,
        default=te.TransformerLayer,
        choices=list(te_layer_map.values()),
        help="TE module type used to construct the test model.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="PyTorch RNG seed.")
    parser.add_argument(
        "--profile-memory",
        action="store_true",
        help="Enable memory profiling via torch.profiler.profile().",
    )
    parser.add_argument(
        "--profile-name", type=str, default=None, help="File path for memory profiling."
    )
    parser.add_argument(
        "--checkpoint-layer",
        type=te_layer,
        default=None,
        help="Recompute activations of the selected layer during the backward "
        + "pass instead of saving.",
    )
    parser.add_argument(
        "--no-fp8",
        action=StoreTrueExplicitAction,
        default=False,
        help=(
            "Disables the te.autocast() context. When set, FP8 training is disabled and the model"
            " trains in standard precision (as specified by --dtype). PRECEDENCE: This flag is"
            " incompatible with FP8-based --precision presets. BEHAVIOR: - Without --precision:"
            " Disables FP8 training (original behavior) - With --precision fp32/fp16: Redundant but"
            " harmless (already non-FP8) - With --precision fp8/mxfp8/nvfp4: RAISES ERROR"
            " (incompatible flags) RATIONALE: FP8 presets explicitly request FP8 training, so"
            " disabling FP8 would contradict the user's intent. Use --precision fp32/fp16 instead"
            " for non-FP8 training. EXAMPLES: '--no-fp8' disables FP8 (original behavior)."
            " '--precision fp8 --no-fp8' raises ValueError (incompatible). '--precision fp16' is"
            " the correct way to request non-FP8 training. Default: False (FP8 enabled based on"
            " configuration)."
        ),
    )
    parser.add_argument(
        "--no-defer-init",
        action="store_true",
        help="Defer module parameter initialization until after FSDP sharding.",
    )
    parser.add_argument(
        "--no-te-fsdp",
        action="store_true",
        help="Disable sharding of intermediate/activation tensors in TE modules.",
    )
    parser.add_argument(
        "--dtype",
        type=torch_dtype,
        default=torch.bfloat16,
        action=StoreExplicitAction,
        help=(
            "Data type for input tensor and Transformer Engine module parameters. Supported values:"
            " fp32/float32, fp16/float16, bf16/bfloat16. PRECEDENCE: When explicitly set, this flag"
            " overrides the dtype from --precision preset. BEHAVIOR: - Without --precision:"
            " Controls parameter dtype directly - With --precision: Overrides preset's dtype but"
            " preserves FP8 recipe selection EXAMPLES: '--dtype bf16' uses bfloat16 parameters"
            " (original behavior). '--precision mxfp8 --dtype fp16' uses fp16 parameters with"
            " MXFP8BlockScaling recipe. A warning is issued when overriding --precision dtype."
            " Default: bfloat16."
        ),
    )
    parser.add_argument(
        "--precision",
        type=precision,
        default=None,
        help=(
            "Precision preset for model training. Supported values: fp32, fp16, fp8, mxfp8, nvfp4."
            " This is a convenience flag that configures both dtype and FP8 settings automatically."
            " - fp32/fp16: Non-FP8 training with specified dtype - fp8: FP8 training with"
            " DelayedScaling recipe (bf16 parameters) - mxfp8: FP8 training with MXFP8BlockScaling"
            " recipe (bf16 parameters) - nvfp4: FP8 training with NVFP4BlockScaling recipe (bf16"
            " parameters) PRECEDENCE RULES: - If --dtype is explicitly set, it overrides the dtype"
            " from this preset (with warning) - If --no-fp8 is set with fp8/mxfp8/nvfp4 presets, an"
            " error is raised (incompatible) - If this flag is not set, original behavior is used"
            " (--dtype and --no-fp8 control training) EXAMPLES: '--precision mxfp8' enables MXFP8"
            " FP8 training with bf16 parameters. '--precision fp8 --dtype fp16' uses fp16"
            " parameters but keeps DelayedScaling recipe. Default: None (backward compatible mode)."
        ),
    )
    return parser.parse_args()


def dist_print(text, all_ranks=False, no_new_line=False):
    if LOCAL_RANK == 0 or all_ranks:
        end = "" if no_new_line else "\n"
        print(f"[GPU-{LOCAL_RANK}] " + text, end=end)


def get_precision_preset(precision_value):
    """Get dtype, no_fp8, and recipe based on precision preset.

    Returns:
        tuple: (dtype, no_fp8, recipe)
    """
    match precision_value:
        case "fp32":
            return torch.float32, True, None
        case "fp16":
            return torch.float16, True, None
        case "fp8":
            recipe = DelayedScaling(
                fp8_format=Format.HYBRID, amax_history_len=32, amax_compute_algo="max"
            )
            return torch.bfloat16, False, recipe
        case "mxfp8":
            recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)
            return torch.bfloat16, False, recipe
        case "nvfp4":
            recipe = NVFP4BlockScaling()
            return torch.bfloat16, False, recipe
        case _:
            # Fail loudly if validation is bypassed or new preset added without updating this function
            raise ValueError(
                f"Invalid precision preset: {precision_value}. "
                "Supported values: fp32, fp16, fp8, mxfp8, nvfp4"
            )


def get_recipe_for_precision(precision_value):
    """Get FP8 recipe based on precision preset (when FP8 is enabled).

    Args:
        precision_value: The precision preset string

    Returns:
        Recipe object for FP8 training
    """
    match precision_value:
        case "mxfp8":
            return MXFP8BlockScaling(fp8_format=Format.E4M3)
        case "nvfp4":
            return NVFP4BlockScaling()
        case _:
            # Default to DelayedScaling for fp8 or any other value
            return DelayedScaling(
                fp8_format=Format.HYBRID, amax_history_len=32, amax_compute_algo="max"
            )


def train(opts):
    # Check which flags were explicitly set
    dtype_explicitly_set = getattr(opts, "dtype_explicitly_set", False)
    no_fp8_explicitly_set = getattr(opts, "no_fp8_explicitly_set", False)

    # Initialize torch.distributed global process group
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(LOCAL_RANK)
    dist_print(f"WORLD_SIZE = {WORLD_SIZE}")
    torch.manual_seed(opts.seed)

    # Determine final configuration based on precedence rules
    if opts.precision is None:
        # Case 1: Backward compatibility - no precision preset specified
        # Use original behavior with dtype and no_fp8 flags
        dtype = opts.dtype
        no_fp8 = opts.no_fp8

        # Set up recipe if FP8 is enabled
        if not no_fp8:
            recipe = DelayedScaling(
                fp8_format=Format.HYBRID, amax_history_len=32, amax_compute_algo="max"
            )
        else:
            recipe = None
    else:
        # Case 2: Precision preset was explicitly specified
        # Start with precision preset values
        preset_dtype, preset_no_fp8, preset_recipe = get_precision_preset(opts.precision)

        # Check for incompatible flag combinations
        # Error if user requests FP8-based precision but also sets --no-fp8
        if opts.precision in ["fp8", "mxfp8", "nvfp4"] and no_fp8_explicitly_set and opts.no_fp8:
            raise ValueError(
                f"Cannot use --no-fp8 with --precision {opts.precision}. "
                "These flags are incompatible. "
                f"Either remove --no-fp8 to use {opts.precision} training, "
                "or use --precision fp32/fp16 for non-FP8 training."
            )

        dtype = preset_dtype
        no_fp8 = preset_no_fp8
        recipe = preset_recipe

        dist_print(f"Using precision preset: {opts.precision}")

        # Apply explicit dtype override with warning
        if dtype_explicitly_set:
            dtype = opts.dtype
            dist_print(
                f"Warning: --dtype {dtype} overrides --precision {opts.precision} dtype setting"
            )

            # If FP8 is still enabled, keep recipe based on precision
            # (dtype only affects parameter storage, not FP8 recipe)
            if not no_fp8:
                recipe = get_recipe_for_precision(opts.precision)

    # Always log the final configuration being used
    dist_print(f"Training configuration: dtype={dtype}, FP8={'disabled' if no_fp8 else 'enabled'}")

    # Construct a simple homogeneous model (only one layer type) with NO PARALLELISM
    layer_args, layer_kwargs = get_layer_args(opts)
    layer_kwargs["params_dtype"] = dtype

    if opts.num_layers > 1:
        te_layer_list = []
        for i in range(opts.num_layers):
            if opts.layer_type in [te.MultiheadAttention, te.TransformerLayer]:
                layer_kwargs["layer_number"] = i + 1
            te_layer_list.append(opts.layer_type(*layer_args, **layer_kwargs))
        te_model = nn.Sequential(*te_layer_list)
    else:
        # Single layer model
        te_model = opts.layer_type(*layer_args, **layer_kwargs)

    # Print out allocated device memory before the model parameters are sharded by FSDP
    pre_mem_use = torch.cuda.memory_allocated(device=f"cuda:{LOCAL_RANK}") * 1e-6
    dist_print(f"Pre-FSDP memory use = {pre_mem_use}MiB")

    # Wrap the model with FSDP
    # NOTE: The TE model itself has no inherent parallelism. FSDP shards model parameters and
    #       controls all communication.
    all_gpus = dist.new_group(backend="nccl")
    fsdp_wrap_policy = always_wrap_policy
    if opts.layer_type == te.TransformerLayer:
        # NOTE: FSDP causes illegal memory access without this special policy for Transformers
        fsdp_wrap_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={te.TransformerLayer}
        )
    te_model = FullyShardedDataParallel(
        te_model,
        process_group=all_gpus,
        use_orig_params=True,
        mixed_precision=MixedPrecision(
            param_dtype=dtype,
            reduce_dtype=torch.float32,
        ),
        auto_wrap_policy=fsdp_wrap_policy,
    )

    if opts.checkpoint_layer is not None:
        # Recompute the activations of the selected layer during the backward pass instead of
        # saving them during the forward pass
        apply_fsdp_checkpointing(te_model, blocks=opts.checkpoint_layer)
    elif not opts.no_te_fsdp:
        # Prepare TE modules to shard internal buffers that FSDP cannot shard on its own
        prepare_te_modules_for_fsdp(te_model)

    # Print out allocated device memory after the model parameters are sharded
    post_mem_use = torch.cuda.memory_allocated(device=f"cuda:{LOCAL_RANK}") * 1e-6
    dist_print(f"Post-FSDP memory use = {post_mem_use}MiB")
    dist_print(f"FSDP-Wrapped + Checkpointed TE Model:\n{te_model}")

    # Optimizer must be created after the model is wrapped in FSDP and the parameters are sharded
    optim = torch.optim.Adam(te_model.parameters(), lr=0.0001)

    # Profile memory use
    if opts.profile_memory:
        torch.cuda.memory._record_memory_history(max_entries=100000)
    else:
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()

    for i in range(opts.num_iters):
        # Generate a random input batch
        x = torch.rand(
            opts.seq_length,
            opts.batch_size,
            opts.num_heads * opts.head_dim,
            dtype=dtype,
            device="cuda",
        )
        # autocast needs to be given the FSDP process group for amax reductions
        with te.autocast(enabled=not no_fp8, recipe=recipe, amax_reduction_group=all_gpus):
            y = te_model(x)
            loss = y.sum()
        # calculate gradient and take training step outside the autocast context
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        del x

    if opts.profile_memory:
        torch.cuda.memory._dump_snapshot(f"gpu{LOCAL_RANK}_{opts.profile_name}.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
    else:
        end.record()
        torch.cuda.synchronize()
        peak_mem = torch.cuda.max_memory_allocated()
        train_time = start.elapsed_time(end) / 1000.0
        dist_print(f"Training Time: {train_time}s")
        dist_print(f"Avg. Iter. Time: {train_time / opts.num_iters}s")
        dist_print(f"Peak Memory Use: {peak_mem * 1e-6}MBs")


# Run with:
#   torchrun --nnodes=1 --nproc-per-node=$(nvidia-smi -L | wc -l) test_fsdp.py --defer-init
if __name__ == "__main__":
    args = parse_fsdp_args()
    train(args)
