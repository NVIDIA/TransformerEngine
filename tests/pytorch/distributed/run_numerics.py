#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import datetime
import os
import sys
from functools import wraps

import transformer_engine.pytorch as te
import torch
from torch import nn
import torch.distributed as dist
import transformer_engine_torch as tex
from transformer_engine.common.recipe import (
    MXFP8BlockScaling,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    Format,
    Recipe,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
from run_layer_with_overlap import _compare_tensors

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


# Disable TF32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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
    if QUANTIZATION in ("fp8", "mxfp8", "fp8_block_scaling"):
        global SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE
        SEQ_LEN = 32
        BATCH_SIZE = 32
        HIDDEN_SIZE = 128

    test_dict = [
        test_quantizer,
        test_linear,
        test_layernorm,
        test_layernorm_linear,
        test_layernorm_mlp,
        test_transformer_layer,
    ]

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
    return nn.init.constant_(tensor, 0.5)


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
        return {"rtol": 1.3e-6, "atol": 4e-5}
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


###############################################
#                   Quantizer                 #
###############################################
def _construct_quantizer(quantizer_class, fp8_dtype, device, tp_group, tp_size):
    """
    quantizer is the reference quantizer on a single GPU.
    quantizer_dist is the distributed quantizer to be tested on multiple GPUs.
    """
    if quantizer_class == Float8CurrentScalingQuantizer:
        quantizer_dist = quantizer_class(
            fp8_dtype=fp8_dtype,
            device=device,
            with_amax_reduction=True,
            amax_reduction_group=tp_group,
        )
        quantizer = quantizer_class(
            fp8_dtype=fp8_dtype,
            device=device,
            with_amax_reduction=False,
        )
        return quantizer, quantizer_dist
    else:
        raise ValueError(f"Unsupported quantizer class: {quantizer_class}")


def _shard_tensor(x, world_size, axis):
    split_size = x.size()[axis] // world_size
    split_tensor = torch.split(x, split_size, axis)
    out = []
    for tensor in split_tensor:
        out.append(tensor.detach().clone().requires_grad_(x.requires_grad).cuda())
    return out


@run_distributed_test()
def _test_quantizer(input_dtype, fp8_dtype):
    """Test the quantizer under distributed settings.

    Args:
        input_dtype (torch.dtype): The data type of the input.
        fp8_dtype (tex.DType): The data type of the fp8.
    """

    M, N = WORLD_SIZE * BATCH_SIZE, HIDDEN_SIZE

    # high precision input
    x_hp_cpu = torch.randn((M, N), device="cpu").to(input_dtype)
    # set one element of the input to a very large value, which doesn't live in rank 0 after the split
    # to test the amax reduction on purpose
    x_hp_cpu[M - 1, N - 1] = 1e4
    # rank 0 takes the full copy and quantize with GPU 0 for verification
    if WORLD_RANK == 0:
        x_hp_rank0 = x_hp_cpu.clone().detach().requires_grad_(True).to("cuda")
    x_hp_local_rank = _shard_tensor(x_hp_cpu, WORLD_SIZE, 0)[WORLD_RANK]

    # Create quantizers
    quantizer, quantizer_dist = _construct_quantizer(
        Float8CurrentScalingQuantizer, fp8_dtype, x_hp_local_rank.device, NCCL_WORLD, WORLD_SIZE
    )

    # quantize the input
    if WORLD_RANK == 0:
        x_fp8_single = quantizer(x_hp_rank0)

    # multi-GPU quantizer
    x_fp8_dist = quantizer_dist(x_hp_local_rank)

    # check scale_inv with zero tolerance
    if WORLD_RANK == 0:
        torch.testing.assert_close(
            x_fp8_single._scale_inv, x_fp8_dist._scale_inv, rtol=0.0, atol=0.0
        )


def test_quantizer():
    """
    Run quantizer tests with various configurations.
    Currently only check fp8_cs because it needs to do amax reduction in the quantizer.
    """
    # skip this test for other quantization schemes
    if QUANTIZATION != "fp8_cs":
        return

    input_dtypes = [torch.float32, torch.bfloat16]
    fp8_dtypes = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]

    for input_dtype in input_dtypes:
        for fp8_dtype in fp8_dtypes:
            _test_quantizer(input_dtype, fp8_dtype)


############################################
#                   Linear                 #
############################################
@run_distributed_test()
def _test_linear(parallel_mode=None, sequence_parallel=False, **kwargs):
    """Test the linear layer with specified parallel mode and sequence parallelization.

    Args:
        parallel_mode (str): 'row' or 'column' parallelism.
        sequence_parallel (bool): Enable sequence parallelism if True.
        kwargs (dict): Additional arguments for the linear layer.
    """
    # Set parameter data type
    params_dtype = kwargs.get("params_dtype", torch.float32)

    # Create models
    model_single_node = te.Linear(HIDDEN_SIZE, HIDDEN_SIZE, **kwargs)
    model_distributed = te.Linear(
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        parallel_mode=parallel_mode,
        sequence_parallel=sequence_parallel,
        **kwargs,
    )

    # Synchronize parameters between models
    _copy_params(model_distributed, model_single_node)

    # Prepare input tensors
    input_single_node = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)

    if parallel_mode == "row":
        # Split input across GPUs for row parallelism
        split_size = HIDDEN_SIZE // WORLD_SIZE
        input_distributed = input_single_node[
            :, WORLD_RANK * split_size : (WORLD_RANK + 1) * split_size
        ].clone()
    elif parallel_mode == "column":
        if sequence_parallel:
            # Duplicate input for sequence parallelism
            input_single_node = (
                torch.empty((WORLD_SIZE * BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
            )
            input_distributed = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
            # when quantization is fp8_cs, we need to trigger corner cases to see if amax reduction is working
            if QUANTIZATION == "fp8_cs":
                input_distributed = torch.clamp(input_distributed, min=-10, max=10)
                if WORLD_RANK == WORLD_SIZE - 1:
                    input_distributed[BATCH_SIZE - 1, HIDDEN_SIZE - 1] = 11
            input_single_node = _gather(input_distributed, dim=0).detach()
        else:
            input_distributed = input_single_node.clone()
    else:
        raise ValueError(f"Invalid parallel_mode: {parallel_mode}")

    # Apply models
    output_single_node, output_distributed = _apply_models(
        model_single_node, model_distributed, input_single_node, input_distributed
    )

    if "return_bias" in kwargs:
        output_single_node, bias_s = output_single_node
        output_distributed, bias_d = output_distributed
        if parallel_mode == "column":
            bias_d = _gather(bias_d)
        _check_outputs(bias_s, bias_d)

    # Gather outputs if necessary
    if parallel_mode == "column" or (sequence_parallel and parallel_mode == "row"):
        output_distributed = _gather(output_distributed, dim=1 if parallel_mode == "column" else 0)

    # Compute loss and backpropagate
    _loss_backward(output_single_node, output_distributed)

    # Compute delayed weight gradient
    if "delay_wgrad_compute" in kwargs:
        _loss_backward_dw(model_single_node, model_distributed)

    # Validate outputs and gradients
    _check_outputs(output_single_node, output_distributed)

    # gradients in other cases need additional synchronization
    if (parallel_mode == "column" or not sequence_parallel) and "return_bias" not in kwargs:
        _check_gradients(
            model_distributed,
            model_single_node,
            main_grad_check=("fuse_wgrad_accumulation" in kwargs),
        )


def test_linear():
    """Run linear layer tests with various configurations."""
    kwargs_list = [
        {},
        {"bias": False},
        {"init_method": _constant},
        {"fuse_wgrad_accumulation": True},
        {"return_bias": True},
        {"params_dtype": torch.float16},
        {"delay_wgrad_compute": True},
    ]
    for kwargs in kwargs_list:
        for parallel_mode in ["column", "row"]:
            for sequence_parallel in [False, True]:
                _test_linear(parallel_mode, sequence_parallel, **kwargs)


############################################
#                 LayerNorm                #
############################################


@run_distributed_test()
def _test_layernorm(kwargs):
    """Test LayerNorm and RMSNorm with given arguments.

    Args:
        kwargs (dict): Contains 'norm', 'basic_args', and 'distributed_args'.
    """
    # Extract parameters
    norm = kwargs["norm"]
    basic_args = kwargs["basic_args"]
    distributed_args = kwargs["distributed_args"]
    params_dtype = basic_args.get("params_dtype", torch.float32)

    # Create models
    model_single_node = norm(HIDDEN_SIZE, **basic_args)
    model_distributed = norm(HIDDEN_SIZE, **{**basic_args, **distributed_args})

    # Synchronize parameters between models
    _copy_params(model_distributed, model_single_node)

    # Prepare input tensors
    input_single_node = torch.randn((BATCH_SIZE, HIDDEN_SIZE), dtype=params_dtype).cuda()
    input_distributed = input_single_node.clone()

    # Apply models
    output_single_node, output_distributed = _apply_models(
        model_single_node, model_distributed, input_single_node, input_distributed
    )

    # Compute loss and backpropagate
    _loss_backward(output_single_node, output_distributed)

    # Validate outputs and gradients
    _check_outputs(output_single_node, output_distributed)
    _check_gradients(model_distributed, model_single_node)


def test_layernorm():
    """Run LayerNorm and RMSNorm tests with various configurations."""
    norms = [te.LayerNorm, te.RMSNorm]

    # Define basic arguments for the models
    basic_args_list = [
        {"zero_centered_gamma": True},
        {"params_dtype": torch.float16},
    ]

    # Define distributed arguments
    distributed_args_list = [
        {},
        {"sequence_parallel": True},
    ]

    # Generate combinations of norms and arguments
    for norm in norms:
        for basic_args in basic_args_list:
            for distributed_args in distributed_args_list:
                kwargs = {
                    "norm": norm,
                    "basic_args": basic_args,
                    "distributed_args": distributed_args,
                }
                _test_layernorm(kwargs)


############################################
#              LayerNormLinear             #
############################################


@run_distributed_test()
def _test_layernorm_linear(parallel_mode=None, sequence_parallel=False, **kwargs):
    """Test the linear layer with specified parallel mode and sequence parallelization.

    Args:
        parallel_mode (str): 'row' or 'column' parallelism.
        sequence_parallel (bool): Enable sequence parallelism if True.
        kwargs (dict): Additional arguments for the linear layer.
    """
    # Set parameter data type
    params_dtype = kwargs.get("params_dtype", torch.float32)

    # Create models
    model_single_node = te.LayerNormLinear(HIDDEN_SIZE, HIDDEN_SIZE, **kwargs)
    model_distributed = te.LayerNormLinear(
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        parallel_mode=parallel_mode,
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
        if sequence_parallel:
            norm_d = _gather(norm_d)
        _check_outputs(norm_s, norm_d)

    if "return_bias" in kwargs:
        output_single_node, bias_s = output_single_node
        output_distributed, bias_d = output_distributed
        if parallel_mode == "column":
            bias_d = _gather(bias_d)
        _check_outputs(bias_s, bias_d)

    # Gather outputs if necessary
    if parallel_mode == "column" or (sequence_parallel and parallel_mode == "row"):
        output_distributed = _gather(output_distributed, dim=1 if parallel_mode == "column" else 0)

    # Compute loss and backpropagate
    _loss_backward(output_single_node, output_distributed)

    # Compute delayed weight gradient
    if "delay_wgrad_compute" in kwargs:
        _loss_backward_dw(model_single_node, model_distributed)

    # Validate outputs and gradients
    _check_outputs(output_single_node, output_distributed)

    # gradients in other cases need additional synchronization
    if parallel_mode == "column" and not sequence_parallel and "return_bias" not in kwargs:
        _check_gradients(
            model_distributed,
            model_single_node,
            main_grad_check=("fuse_wgrad_accumulation" in kwargs),
        )


def test_layernorm_linear():
    kwargs_list = [
        {},
        {"bias": False},
        {"init_method": _constant},
        {"fuse_wgrad_accumulation": True},
        {"return_bias": True},
        {"params_dtype": torch.float16},
        {"zero_centered_gamma": False},
        {"return_layernorm_output": True},
        {"delay_wgrad_compute": True},
    ]
    for kwargs in kwargs_list:
        for parallel_mode in ["column"]:
            for sequence_parallel in [False, True]:
                _test_layernorm_linear(parallel_mode, sequence_parallel, **kwargs)


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
    model_single_node = te.LayerNormMLP(HIDDEN_SIZE, FFN_HIDDEN_SIZE, **kwargs)
    model_distributed = te.LayerNormMLP(
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
        if sequence_parallel:
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


############################################
#             TransformerLayer             #
############################################


@run_distributed_test()
def _test_transformer_layer_parallel(sequence_parallel=False, **kwargs):
    params_dtype = kwargs.get("params_dtype", torch.float32)
    FFN_HIDDEN_SIZE = 32 if QUANTIZATION is None else 128

    model_single_node = te.TransformerLayer(
        HIDDEN_SIZE, FFN_HIDDEN_SIZE, NR_HEADS, attention_dropout=0, hidden_dropout=0, **kwargs
    )
    model_distributed = te.TransformerLayer(
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        NR_HEADS,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        set_parallel_mode=True,
        sequence_parallel=sequence_parallel,
        seq_length=WORLD_SIZE * SEQ_LEN if sequence_parallel else None,
        attention_dropout=0,
        hidden_dropout=0,
        **kwargs,
    )

    _copy_params(model_distributed, model_single_node)
    _alloc_main_grad(model_single_node, model_distributed)  # for fuse_wgrad_accumulation=True

    input_single_node = (
        torch.randn((WORLD_SIZE * SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
    )
    if sequence_parallel:
        input_distributed = input_single_node[
            WORLD_RANK * SEQ_LEN : (WORLD_RANK + 1) * SEQ_LEN, :, :
        ]
    else:
        input_distributed = input_single_node.clone().cuda()

    encoder_output = None
    if "layer_type" in kwargs:
        encoder_output = torch.randn((SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)).cuda()

    output_single_node, output_distributed = _apply_models(
        model_single_node,
        model_distributed,
        input_single_node,
        input_distributed,
        encoder_output=encoder_output,
    )

    if sequence_parallel:
        output_distributed = _gather(output_distributed)

    _loss_backward(output_single_node, output_distributed)
    _check_outputs(output_single_node, output_distributed)

    # gradients in other cases need additional synchronization
    if not sequence_parallel and "return_bias" not in kwargs:
        _check_gradients(
            model_distributed,
            model_single_node,
            main_grad_check=("fuse_wgrad_accumulation" in kwargs),
        )


def test_transformer_layer():
    kwargs_list = [
        {},
        {"num_gqa_groups": 4},
        {"init_method": _constant},
        {"output_layer_init_method": _constant},
        {"apply_residual_connection_post_layernorm": True},
        {"output_layernorm": True},
        {"parallel_attention_mlp": True},
        # {"layer_type": "decoder"},
        {"window_size": (2, 2)},
        {"normalization": "RMSNorm"},
        {"zero_centered_gamma": True},
        {"fuse_qkv_params": True},
        {"fuse_qkv_params": True, "fuse_wgrad_accumulation": True},
        {"qkv_weight_interleaved": False},
        {"bias": False},
        {"params_dtype": torch.float16},
        {"fuse_qkv_params": True},
        {"activation": "relu"},
    ]

    for kwargs in kwargs_list:
        for sequence_parallel in [False, True]:
            _test_transformer_layer_parallel(sequence_parallel, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
