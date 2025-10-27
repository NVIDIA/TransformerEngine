#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import datetime
import os
import sys
from functools import wraps
import warnings

import torch
from torch import nn
import torch.distributed as dist

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import (
    MXFP8BlockScaling,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    Format,
    Recipe,
)

SEQ_LEN, BATCH_SIZE = 16, 16
HIDDEN_SIZE = 64
NR_HEADS = 4
WORLD_RANK, WORLD_SIZE = None, None
NCCL_WORLD = None
LOSS_FN = nn.MSELoss()
QUANTIZATION = None

if os.environ.get("NVTE_TEST_NVINSPECT_ENABLED", False):
    # The numerics of all the layers should work the same,
    # when debug=True. I fed them with dummy feature
    # to prevent switching off debug, which can happen if
    # no feature is active.
    import nvdlfw_inspect.api as debug_api

    debug_api.initialize(
        os.environ["NVTE_TEST_NVINSPECT_CONFIG_FILE"],
        feature_dirs=os.environ["NVTE_TEST_NVINSPECT_FEATURE_DIRS"],
    )


def _compare_tensors(name, test, ref, rtol, atol):
    # Make sure tensors aren't zero and we don't pass trivially
    if test.count_nonzero() == 0:
        if ref.count_nonzero() == 0:
            warnings.warn(
                f"WARNING: {name} is a zero-tensor for both test and reference models!",
                category=RuntimeWarning,
            )
        else:
            numerics_info = (
                f"NUMERICAL CHECK FAILED: {name} is a zero-tensor but does not match reference!"
            )
            return 1, numerics_info

    diff = torch.abs(test.flatten() - ref.flatten())
    m = torch.argmax(diff)
    abs_err = diff[m].item()
    rel_err = abs_err / max(abs(ref.flatten()[m].item()), 1e-5)
    numerics_failed = 0
    if rel_err > rtol and abs_err > atol:
        numerics_failed = 1
        numerics_info = (
            "NUMERICAL CHECK FAILED: "
            + f"{name} not close enough at index {m.item()} "
            + f"with {test.flatten()[m].item()} vs {ref.flatten()[m].item()} | "
            + f"rel. error = {rel_err} (tol = {rtol}) | "
            + f"abs. error = {abs_err} (tol = {atol})"
        )
    else:
        numerics_info = f"NUMERICAL CHECK PASSED: {name} | "
        if rel_err <= rtol:
            numerics_info += f"rel. error = {rel_err} (tol = {rtol})" + (
                " | " if abs_err <= atol else "."
            )
        if abs_err <= atol:
            numerics_info += f" abs. error = {abs_err} (tol = {atol})"

    return numerics_failed, numerics_info

# Quantization recipe setup
def quantization_recipe() -> Recipe:
    if QUANTIZATION == "fp8":
        return DelayedScaling(
            fp8_format=Format.HYBRID, amax_history_len=32, amax_compute_algo="max"
        )
    if QUANTIZATION == "mxfp8":
        return MXFP8BlockScaling()
    if QUANTIZATION == "fp8_cs":
        return Float8CurrentScaling()
    if QUANTIZATION == "fp8_block_scaling":
        return Float8BlockScaling()
    return te.fp8.get_default_fp8_recipe()


def main(argv=None, namespace=None):
    global WORLD_RANK, WORLD_SIZE, NCCL_WORLD, QUANTIZATION

    WORLD_RANK = int(os.getenv("RANK", "0"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

    assert WORLD_SIZE == LOCAL_SIZE  # this test supports only 1 node
    assert LOCAL_SIZE <= torch.cuda.device_count()
    dist_init_kwargs = {
        "backend": "nccl",
        "rank": WORLD_RANK,
        "world_size": WORLD_SIZE,
        "timeout": datetime.timedelta(seconds=30),
    }
    dist_init_kwargs["init_method"] = "env://"
    dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    assert dist.is_nccl_available()
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(**dist_init_kwargs)

    NCCL_WORLD = dist.new_group(backend="nccl")

    WORLD_SIZE = dist.get_world_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--layer-type", type=str)
    parser.add_argument("--quantization", type=str, default=None)
    args = parser.parse_args(argv, namespace)

    # Quantization scheme
    QUANTIZATION = args.quantization
    global SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE
    if QUANTIZATION in ("fp8", "mxfp8"):
        SEQ_LEN = 32
        BATCH_SIZE = 32
        HIDDEN_SIZE = 128
    elif QUANTIZATION == "fp8_block_scaling":
        SEQ_LEN = 128
        BATCH_SIZE = 128
        HIDDEN_SIZE = 512

    test_dict = [test_layernorm_mlp]

    for test in test_dict:
        test()
    dist.destroy_process_group()
    return 0


def run_distributed_test(test_name=None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            name = test_name if test_name is not None else func.__name__

            dist_print(f"Starting test {name} with args {args} and {kwargs}")
            torch.cuda.set_device(WORLD_RANK)
            torch.manual_seed(12345)
            torch.cuda.manual_seed(12345)
            func(*args, **kwargs)

            dist.barrier()
            dist_print(f"Passed test {name}")

        return wrapper

    return decorator


def _gather(tensor, dim=0):
    """
    Gathers tensors and concats them. Since torch.distributed.nn.functional.all_gather
    multiplies gradients by WORLD_SIZE, those gradiedts are rescaled.
    """

    class HalfGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            return input  # forward pass (identity)

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output / WORLD_SIZE  # gradient division by WORLD_SIZE

    tensor = HalfGradient.apply(tensor)
    gathered = torch.distributed.nn.functional.all_gather(tensor, group=NCCL_WORLD)
    return torch.cat(gathered, dim=dim)


def _constant(tensor):
    return nn.init.constant_(tensor, 0.05)


def dist_print(msg, src=None, end="\n", error=False):
    stream = sys.stderr if error else sys.stdout
    if WORLD_RANK == (0 if src is None else src):
        stream.write(f"[rank{WORLD_RANK}] {msg}{end}\n")


def _get_tolerances(dtype):
    # loose tolerances for fp8_cs because of sequence parallel & amax reduction
    # so that each rank has a different scale_inv for computing Y when we have
    # row parallel & sequence parallel, because we do the all_gather in backward pass
    if QUANTIZATION == "fp8_cs":
        return {"rtol": 0.4, "atol": 0.25}
    elif QUANTIZATION is not None:
        return {"rtol": 0.125, "atol": 0.0625}

    if dtype == torch.float16:
        return {"rtol": 1e-3, "atol": 1e-5}
    if dtype == torch.bfloat16:
        return {"rtol": 1.6e-2, "atol": 1e-5}
    if dtype == torch.float32:
        # TF32 has same mantissa bits as FP16
        return {"rtol": 1e-3, "atol": 1e-5}
    raise ValueError(f"Unsupported dtype ({dtype})")


def _check_outputs(output_single_node, output_distributed):
    numerics_failed = torch.tensor([0], dtype=torch.uint8, device="cuda")

    output_failed, output_info = _compare_tensors(
        "outputs",
        output_distributed,
        output_single_node,
        **_get_tolerances(output_single_node.dtype),
    )
    if output_failed:
        dist_print(output_info, src=WORLD_RANK, error=output_failed)
    numerics_failed[0] = int(output_failed)
    dist.all_reduce(numerics_failed, dist.ReduceOp.MAX, NCCL_WORLD)
    assert not bool(numerics_failed.item())


def _match_param_sizes(dist_param, single_param):
    """
    Adjust single_param to match the shape of dist_param
    by slicing along dimensions where the shapes differ.
    This function is typically used in a distributed setting
    where single_param is a larger tensor that needs
    to be partitioned among multiple processes.

    Args:
        dist_param: Tensor representing the distributed output
        with the desired shape for the current process.
        single_param: Tensor representing the non-distributed output,
        possibly larger than dist_param.

    Returns:
        Tensor: Sliced version of single_param matching
        the shape of dist_param for the current process.
    """
    # Initialize indices for slicing with full slices for each dimension
    indices = [slice(None)] * len(single_param.shape)

    # Iterate over each dimension to identify where shapes differ
    for i in range(len(dist_param.shape)):
        if dist_param.shape[i] != single_param.shape[i]:
            # Calculate the start and end indices for slicing based on the world rank
            start = WORLD_RANK * dist_param.shape[i]
            end = (WORLD_RANK + 1) * dist_param.shape[i]
            src_slice = slice(start, end)

            # Update the slicing indices for the current dimension
            indices[i] = src_slice

    # Slice single_param to obtain the output matching dist_param's shape
    to_output = single_param[tuple(indices)]

    return to_output


def _check_gradients(model_distributed, model_single, main_grad_check=False):
    for i, ((name, param_d), param_s) in enumerate(
        zip(model_distributed.named_parameters(), model_single.parameters())
    ):
        numerics_failed = torch.tensor([0], dtype=torch.uint8, device="cuda")
        grad_failed, grad_info = None, None
        if main_grad_check:
            param_s_grad = _match_param_sizes(param_d.main_grad, param_s.main_grad)
            grad_failed, grad_info = _compare_tensors(
                str(i), param_d.main_grad, param_s_grad, **_get_tolerances(param_s_grad.dtype)
            )
        else:
            param_s_grad = _match_param_sizes(param_d.grad, param_s.grad)
            grad_failed, grad_info = _compare_tensors(
                str(i), param_d.grad, param_s_grad, **_get_tolerances(param_s_grad.dtype)
            )

        if grad_failed:
            dist_print(i, src=WORLD_RANK)
            dist_print(name, src=WORLD_RANK)
            dist_print(grad_info, src=WORLD_RANK, error=grad_failed)
        numerics_failed[0] = int(grad_failed)
        dist.all_reduce(numerics_failed, dist.ReduceOp.MAX, NCCL_WORLD)
        assert not bool(numerics_failed.item())


def _copy_params(model_distributed, model_single):
    for dist_param, single_param in zip(model_distributed.parameters(), model_single.parameters()):
        with torch.no_grad():
            to_copy = single_param
            for dim, _ in enumerate(dist_param.shape):
                if dist_param.shape[dim] != single_param.shape[dim]:
                    src_slice = slice(
                        WORLD_RANK * dist_param.shape[dim], (WORLD_RANK + 1) * dist_param.shape[dim]
                    )
                    indices = [slice(None)] * max(min(dim, len(dist_param.shape) - 1), 0)
                    indices.append(src_slice)
                    if dim < len(dist_param.shape) - 1:
                        indices.append(slice(None))
                    to_copy = single_param[tuple(indices)]
            dist_param.copy_(to_copy)


def _apply_models(
    model_single_node, model_distributed, input_single_node, input_distributed, **kwargs
):
    _alloc_main_grad(model_single_node, model_distributed)  # for fuse_wgrad_accumulation=True
    input_single_node.requires_grad_()
    input_distributed.requires_grad_()
    with te.fp8_autocast(
        enabled=QUANTIZATION is not None,
        fp8_recipe=quantization_recipe(),
    ):
        output_single_node = model_single_node(input_single_node, **kwargs)
    with te.fp8_autocast(
        enabled=QUANTIZATION is not None,
        fp8_recipe=quantization_recipe(),
        fp8_group=NCCL_WORLD,
    ):
        output_distributed = model_distributed(input_distributed, **kwargs)
    return output_single_node, output_distributed


def _loss_backward(output_single_node, output_distributed):
    target = torch.randn_like(output_single_node)
    LOSS_FN(output_single_node, target).backward()
    LOSS_FN(output_distributed, target).backward()


def _loss_backward_dw(model_single_node, model_distributed):
    model_single_node.backward_dw()
    model_distributed.backward_dw()


def _alloc_main_grad(model_single_node, model_distributed):
    for model in [model_single_node, model_distributed]:
        for param in model.parameters():
            param.main_grad = torch.zeros_like(param, dtype=torch.float32)


############################################
#               LayerNormMLP               #
############################################


@run_distributed_test()
def _test_layernorm_mlp(set_parallel_mode=None, sequence_parallel=False, **kwargs):
    """Test the LayerNormMLP with specified parallel mode and sequence parallelization.

    Args:
        set_parallel_mode (bool): Enable parallel mode.
        sequence_parallel (bool): Enable sequence parallelism if True.
        kwargs (dict): Additional arguments for the linear layer.
    """
    # Set parameter data type
    params_dtype = kwargs.get("params_dtype", torch.float32)
    FFN_HIDDEN_SIZE = 32 if QUANTIZATION is None else 128

    # Create models
    model_single_node = te.SelectiveLayerNormMLP(HIDDEN_SIZE, FFN_HIDDEN_SIZE, **kwargs)
    model_distributed = te.SelectiveLayerNormMLP(
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        set_parallel_mode=set_parallel_mode,
        sequence_parallel=sequence_parallel,
        **kwargs,
    )

    # Synchronize parameters between models
    _copy_params(model_distributed, model_single_node)

    # Prepare input tensors
    input_single_node = torch.randn((SEQ_LEN, HIDDEN_SIZE)).cuda().to(params_dtype)

    if sequence_parallel:
        # Duplicate input for sequence parallelism
        input_single_node = torch.empty((WORLD_SIZE * SEQ_LEN, HIDDEN_SIZE)).cuda().to(params_dtype)
        input_distributed = torch.randn((SEQ_LEN, HIDDEN_SIZE)).cuda().to(params_dtype)
        # make the last element of the input a large value to test the amax reduction on purpose
        # when quantization is fp8_cs, we need to trigger corner cases to see if amax reduction is working
        if QUANTIZATION == "fp8_cs":
            input_distributed = torch.clamp(input_distributed, min=-10, max=10)
            if WORLD_RANK == WORLD_SIZE - 1:
                input_distributed[SEQ_LEN - 1, HIDDEN_SIZE - 1] = 11
        input_single_node = _gather(input_distributed).detach()
    else:
        input_distributed = input_single_node.clone()
    # Apply models
    output_single_node, output_distributed = _apply_models(
        model_single_node, model_distributed, input_single_node, input_distributed
    )

    if "return_layernorm_output" in kwargs:
        output_single_node, norm_s = output_single_node
        output_distributed, norm_d = output_distributed
        if sequence_parallel and not kwargs.get("return_layernorm_output_gathered", False):
            norm_d = _gather(norm_d)
        _check_outputs(norm_s, norm_d)

    if "return_bias" in kwargs:
        output_single_node, bias_s = output_single_node
        output_distributed, bias_d = output_distributed
        _check_outputs(bias_s, bias_d)

    if sequence_parallel:
        output_distributed = _gather(output_distributed)

    # Compute loss and backpropagate
    _loss_backward(output_single_node, output_distributed)

    if "delay_wgrad_compute" in kwargs:
        _loss_backward_dw(model_single_node, model_distributed)

    # Validate outputs and gradients
    _check_outputs(output_single_node, output_distributed)

    # gradients in other cases need additional synchronization
    if not sequence_parallel and "return_bias" not in kwargs:
        _check_gradients(
            model_distributed,
            model_single_node,
            main_grad_check=("fuse_wgrad_accumulation" in kwargs),
        )


def test_layernorm_mlp():
    kwargs_list = [
        {},
        {"init_method": _constant},
        {"output_layer_init_method": _constant},
        {"normalization": "RMSNorm"},
        {"zero_centered_gamma": True},
        {"bias": False},
        {"params_dtype": torch.float16},
        {"activation": "relu"},
        {"fuse_wgrad_accumulation": True},
        {"return_bias": True},
        {"return_layernorm_output": True},
        {"delay_wgrad_compute": True},
    ]

    for kwargs in kwargs_list:
        for set_parallel_mode in [True]:
            for sequence_parallel in [False, True]:
                _test_layernorm_mlp(set_parallel_mode, sequence_parallel, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
