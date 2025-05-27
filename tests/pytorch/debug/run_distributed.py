# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import tempfile
import functools
import os
import itertools
import random
import argparse
import re

import torch
import torch.distributed as dist
import transformer_engine
import transformer_engine_torch as tex
import nvdlfw_inspect.api as debug_api
from transformer_engine.debug import set_weight_tensor_tp_group_reduce


from test_numerics import (
    _emulate_linear,
    _init_debug,
    disable_fp8_gemms_create_config,
    DISABLE_FP8_LAYER_CONFIG,
    _cmp,
    IN_SIZE,
    OUT_SIZE,
    _init_model,
    SEED,
    SEQ_LEN,
    BATCH_SIZE,
    FP8_RECIPE,
    fake_quant_fp8_create_config,
    _get_current_scale,
    _prepare_per_tensor_scaling_config,
    AMAX_HISTORY_LEN,
    set_scaling_factors,
    set_current_scaling_factors,
)

WORLD_RANK, WORLD_SIZE = None, None
NCCL_WORLD = None
FEATURE_DIRS = None
all_boolean = [True, False]
TEST_NR = 0


def _get_tensors(parallel_mode, weight_seed=SEED, data_seed=SEED, tp_size=None, tp_rank=None):
    if tp_size is None:
        tp_size = WORLD_SIZE
        tp_rank = WORLD_RANK
    torch.manual_seed(weight_seed)
    weight = torch.randn((OUT_SIZE, IN_SIZE)).cuda()
    torch.manual_seed(data_seed)
    in_split_size = IN_SIZE // tp_size
    out_split_size = OUT_SIZE // tp_size
    x = torch.randn((SEQ_LEN * BATCH_SIZE, IN_SIZE), requires_grad=True).cuda()
    if parallel_mode == "row":
        x = x[:, tp_rank * in_split_size : (tp_rank + 1) * in_split_size]
    x.retain_grad()

    with torch.no_grad():
        if parallel_mode == "column":
            weight = weight[tp_rank * out_split_size : (tp_rank + 1) * out_split_size, :]
        else:
            weight = weight[:, tp_rank * in_split_size : (tp_rank + 1) * in_split_size]

    return x, weight.contiguous()


def _init_model(weight, parallel_mode=None, tp_group=None, name="linear"):
    model = transformer_engine.pytorch.Linear(
        IN_SIZE,
        OUT_SIZE,
        name=name,
        parallel_mode=parallel_mode,
        tp_group=(tp_group or NCCL_WORLD if parallel_mode else None),
    )
    with torch.no_grad():
        model.weight.copy_(weight)
    return model


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, dim, group=None):
        if group is None:
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            world_size = torch.distributed.get_world_size(group=group)
            rank = torch.distributed.get_rank(group=group)
            dist.barrier()

        # Create a list to gather tensors from all processes
        y_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        torch.distributed.all_gather(y_list, tensor, group=group)

        # Save the world size and rank for backward computation
        ctx.world_size = world_size
        ctx.rank = rank
        ctx.dim = dim

        # Concatenate the gathered tensors along the feature dimension
        y_full = torch.cat(y_list, dim=dim)

        return y_full

    @staticmethod
    def backward(ctx, grad_output):
        # Split the gradient output and return the portion corresponding to this rank
        grad_input = torch.chunk(grad_output, ctx.world_size, dim=ctx.dim)[ctx.rank]
        return grad_input, None, None


def _run_forward_backward(x, model, parallel_mode=None, group=None):
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
        y = model(x)

    y.requires_grad_(True)
    y.retain_grad()
    if parallel_mode == "column":
        y = AllGather.apply(y, -1, group)
        y.requires_grad_(True)
        y.retain_grad()
        l = y.sum()
        l.backward()
    elif parallel_mode == "row":
        l = y.sum()
        l.backward()
    debug_api.step()
    return y


def _emulate_linear_distributed(*args, parallel_mode=None, **kwargs):
    assert parallel_mode in ["column", "row"]

    def split(gradient):
        split_size = OUT_SIZE // WORLD_SIZE
        gradient = gradient[:, WORLD_RANK * split_size : (WORLD_RANK + 1) * split_size]
        return gradient

    activation_sync = None
    gradient_sync = None
    if parallel_mode == "column":
        activation_sync = lambda x: AllGather.apply(x, -1)
        gradient_sync = split
    else:
        activation_sync = (
            lambda activation: dist.all_reduce(activation, op=dist.ReduceOp.SUM) or activation
        )

    output = _emulate_linear(
        *args, activation_sync=activation_sync, gradient_sync=gradient_sync, **kwargs
    )

    if parallel_mode == "column":
        dist.all_reduce(output["dgrad"], op=dist.ReduceOp.SUM)

    return output


def check_debug_log(msg):
    with open(f"log/debug_logs/debug_log_globalrank-{WORLD_RANK}.log", "r") as f:
        for line in f.readlines():
            if msg in line:
                return True
    return False


def run_debug_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank = dist.get_rank()
        temp_file_name = None
        temp_logdir_name = None

        if rank == 0:
            with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
                temp_file_name = temp_file.name
            temp_dir_obj = tempfile.TemporaryDirectory()
            temp_logdir_name = temp_dir_obj.name

            # Store the TemporaryDirectory object to prevent it from being deleted
            wrapper.temp_dir_obj = temp_dir_obj

        temp_file_name_list = [temp_file_name]
        temp_logdir_name_list = [temp_logdir_name]

        # Broadcast the temporary file and directory names to all processes
        dist.broadcast_object_list(temp_file_name_list, src=0)
        dist.broadcast_object_list(temp_logdir_name_list, src=0)

        temp_file_name = temp_file_name_list[0]
        temp_logdir_name = temp_logdir_name_list[0]

        dist.barrier()

        config_file = open(temp_file_name, mode="r+", buffering=1)

        try:
            kwargs["config_file"] = config_file
            kwargs["log_dir"] = temp_logdir_name

            if rank == 0:
                global TEST_NR
                print(f"Running test {TEST_NR} {func.__name__} with args = {args}.")
                TEST_NR += 1

            func(*args, **kwargs)
        finally:
            if rank == 0 and temp_file_name is not None:
                os.unlink(temp_file_name)

            debug_api.end_debug()

            if rank == 0 and hasattr(wrapper, "temp_dir_obj"):
                wrapper.temp_dir_obj.cleanup()

    return wrapper


CONFIG_LOG_TEST_DISTRIBUTED = """log_distributed:
  layers:
    layer_types: [linear]
  enabled:
    True
  transformer_engine:
    LogTensorStats:
      enabled: True
      tensors: [activation, gradient, weight, output, wgrad, dgrad]
      stats: [min, max, mean, std, l1_norm, l2_norm, cur_amax, dynamic_range]
      start_step : 0
      end_step: 1
    LogFp8TensorStats:
      enabled: True
      tensors: [activation, gradient, weight]
      stats: [underflows%]
      start_step : 0
      end_step: 1
"""


def _prepare_config_test_log_distributed(config_file):
    if WORLD_RANK != 0:
        return
    config_file.write(CONFIG_LOG_TEST_DISTRIBUTED)
    config_file.flush()


def _compute_dynamic_range(tensor):
    tensor_abs = tensor.abs()
    tensor_abs = tensor_abs[tensor_abs != 0]
    if tensor_abs.any():
        amin = tensor_abs.min().float()
    else:
        amin = torch.tensor(1, device=tensor.device).to(torch.float)
    amax = tensor_abs.max().float()
    if not amax.all():
        amax = torch.tensor(1, device=tensor.device).to(torch.float)
    dynamic_range = torch.log2(amax) - torch.log2(amin)
    return dynamic_range


@run_debug_test
def test_log_distributed(parallel_mode, gather_weight, **kwargs):
    _prepare_config_test_log_distributed(kwargs["config_file"])
    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], FEATURE_DIRS)
    set_weight_tensor_tp_group_reduce(gather_weight)
    if WORLD_SIZE % 2 != 0:
        return  # skip
    TP_SIZE = WORLD_SIZE // 2
    DP_SIZE = 2
    TP_RANK = WORLD_RANK % TP_SIZE
    DP_RANK = (WORLD_RANK - TP_RANK) // TP_SIZE

    debug_api.set_tensor_reduction_group(NCCL_WORLD)

    x, weight = _get_tensors(
        parallel_mode,
        weight_seed=TP_RANK * 1234,
        data_seed=DP_RANK * 1234,
        tp_size=TP_SIZE,
        tp_rank=TP_RANK,
    )

    tp_group_ranks = [i for i in range(DP_RANK * TP_SIZE, (DP_RANK + 1) * TP_SIZE)]
    tp_group = dist.new_group(ranks=tp_group_ranks)

    dp_group_ranks = [i for i in range(TP_RANK, WORLD_SIZE, TP_SIZE)]
    dp_group = dist.new_group(ranks=dp_group_ranks)

    model = _init_model(weight, parallel_mode=parallel_mode, tp_group=tp_group)
    output = _run_forward_backward(x, model, parallel_mode=parallel_mode, group=tp_group)

    gathered_activation = AllGather.apply(x.contiguous(), 0)
    gathered_weight = AllGather.apply(weight.contiguous(), 0, tp_group)
    gathered_gradient = AllGather.apply(output.grad.contiguous(), 0, dp_group)
    if parallel_mode == "row":
        gathered_gradient = AllGather.apply(gathered_gradient, 0, tp_group)

    log_file = kwargs["log_dir"] + "/nvdlfw_inspect_statistics_logs/nvdlfw_inspect_globalrank-0.log"

    dist.barrier()
    if WORLD_RANK != 0:
        return  # stats are gathered on node 0
    with open(log_file) as f:
        content = f.read()

    def get_stat(tensor, stat):
        regex = r".*_{tensor}_{stat}\s+.*iteration=(\d+)\s+.*value=([-+]?\d*\.?\d+)".format(
            tensor=tensor, stat=stat
        )
        for line in content.splitlines():
            match = re.search(regex, line)
            if match:
                value = float(match.group(2))
                return value

    rf = lambda x: round(float(x), 4)
    stats = []
    tensors = {
        "activation": gathered_activation,
        "weight": gathered_weight if gather_weight else weight,
        "gradient": gathered_gradient,
    }
    stats = {
        "min": torch.min,
        "max": torch.max,
        "mean": torch.mean,
        "std": torch.std,
        "l1_norm": lambda x: torch.norm(x, p=1),
        "l2_norm": lambda x: torch.norm(x, p=2),
        "cur_amax": lambda x: x.abs().max(),
        "dynamic_range": _compute_dynamic_range,
    }
    for stat_key in stats.keys():
        for tensor_key in tensors.keys():
            torch.testing.assert_close(
                get_stat(tensor_key, stat_key),
                rf(stats[stat_key](tensors[tensor_key])),
                atol=0.0001,
                rtol=0.0001,
            )
    set_weight_tensor_tp_group_reduce(True)  # reset


@run_debug_test
def test_log_expert_parallel(**kwargs):
    """
    This test tests the scenario, when one of the node of data parallel does not invoke the debug layer.
    It naturally occurs in the expert parallelism, when one expert doesn't get input on one node,
    but gets it on other nodes. If there were all_gather inside forward(), this would result in deadlock.
    """
    _prepare_config_test_log_distributed(kwargs["config_file"])
    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], FEATURE_DIRS)
    debug_api.set_tensor_reduction_group(NCCL_WORLD)
    x, weight = _get_tensors(
        "row", weight_seed=WORLD_RANK * 1234, data_seed=WORLD_RANK * 1234, tp_size=1, tp_rank=0
    )  # data parallel
    model = _init_model(weight, parallel_mode=None, name="linear1")
    model1 = _init_model(weight, parallel_mode=None, name="linear2")
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
        y1 = model(x)
        y2 = model1(x)
        y = y1 + y2
    y.sum().backward()
    debug_api.step()
    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
        y = model(x)
        if WORLD_RANK != 0:
            y = y + model1(x)

    y.sum().backward()


@run_debug_test
def test_disable_fp8_gemms(fprop_fp8, dgrad_fp8, wgrad_fp8, parallel_mode, **kwargs):
    disable_fp8_gemms_create_config(fprop_fp8, dgrad_fp8, wgrad_fp8, kwargs["config_file"])
    fp8_kwargs = {
        "fprop_fp8": fprop_fp8,
        "dgrad_fp8": dgrad_fp8,
        "wgrad_fp8": wgrad_fp8,
    }

    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], FEATURE_DIRS)
    x, weight = _get_tensors(parallel_mode)
    model = _init_model(weight, parallel_mode=parallel_mode)
    y = _run_forward_backward(x, model, parallel_mode=parallel_mode)

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}

    x.grad.zero_()
    ground_truth = _emulate_linear_distributed(x, weight, parallel_mode=parallel_mode, **fp8_kwargs)
    _cmp(ground_truth, output)


@run_debug_test
def test_disable_fp8_layer(parallel_mode, **kwargs):
    if WORLD_RANK == 0:
        kwargs["config_file"].write(DISABLE_FP8_LAYER_CONFIG)
        kwargs["config_file"].flush()
    dist.barrier()

    x, weight = _get_tensors(parallel_mode)

    ground_truth = _emulate_linear_distributed(x, weight, parallel_mode=parallel_mode)
    x.grad.zero_()

    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], FEATURE_DIRS)

    model = _init_model(weight, parallel_mode)
    y = _run_forward_backward(x, model, parallel_mode)

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}
    _cmp(ground_truth, output)


@run_debug_test
def test_per_tensor_scaling(
    fprop_inp,
    fprop_weight,
    dgrad_weight,
    dgrad_grad,
    wgrad_input,
    wgrad_grad,
    parallel_mode,
    **kwargs,
):
    input_kwargs = {
        "fprop_inp": fprop_inp,
        "fprop_weight": fprop_weight,
        "dgrad_weight": dgrad_weight,
        "dgrad_grad": dgrad_grad,
        "wgrad_input": wgrad_input,
        "wgrad_grad": wgrad_grad,
    }
    fp8_kwargs = {
        "fprop_fp8": True,
        "dgrad_fp8": True,
        "wgrad_fp8": True,
    }
    """
        Runs a test to validate per-tensor (current) scaling in FP8 computations.
        The function performs warm-up iterations to populate the amax buffer of the model and compute scaling factors based on delayed scaling.
        Subsequently, weights and inputs are switched to ensure their current scaling factors differ from those based on delayed scaling;
        similarly, the loss is multiplied by a large factor to alter the gradient's magnitude,
        creating a discrepancy between the original (delayed) and per-tensor (current) scaling factors.
        Finally, a linear pass is emulated, and the results are compared.”
    """
    _prepare_per_tensor_scaling_config(
        fprop_inp,
        fprop_weight,
        dgrad_weight,
        dgrad_grad,
        wgrad_input,
        wgrad_grad,
        kwargs["config_file"],
    )
    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], FEATURE_DIRS)

    warmup_input, warmup_weight = _get_tensors(parallel_mode=parallel_mode)
    model = _init_model(warmup_weight, parallel_mode=parallel_mode)

    # Warmup run to setup amax and scaling factors.
    for _ in range(AMAX_HISTORY_LEN):
        _run_forward_backward(warmup_input, model, parallel_mode=parallel_mode)

    x, weight = _get_tensors(
        parallel_mode=parallel_mode, weight_seed=WORLD_RANK * 2137, data_seed=WORLD_RANK * 2137
    )
    model.weight.data = weight.data
    x.retain_grad()

    # delayed scaling factor
    # need to be collected before forward pass with test data,
    # because this forward pass changes scaling factors
    set_scaling_factors(model, input_kwargs, fp8_kwargs)

    LOSS_MULTIPLIER = 100

    with transformer_engine.pytorch.fp8_autocast(enabled=True, fp8_recipe=FP8_RECIPE):
        y = model(x)
        model.zero_grad()
        if parallel_mode == "column":
            y = AllGather.apply(y, -1)
        y.retain_grad()

        (
            LOSS_MULTIPLIER * y.sum()
        ).backward()  # Loss multiplication to change gradient's order of magintude

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}
    # per tensor - current - scaling factors
    # need to be collected after forward pass with test data,
    # because gradient(y.grad) cannot be accessed before forward,
    # but it needs to be collected.

    set_current_scaling_factors(x, weight, y, input_kwargs, fp8_kwargs)
    ground_truth = _emulate_linear_distributed(
        x, weight, parallel_mode=parallel_mode, loss_multiplier=LOSS_MULTIPLIER, **fp8_kwargs
    )

    _cmp(ground_truth, output)


@run_debug_test
def test_fake_quant_fp8(
    fprop_inp,
    fprop_weight,
    dgrad_weight,
    dgrad_grad,
    wgrad_input,
    wgrad_grad,
    parallel_mode,
    **kwargs,
):

    fp8_kwargs = {
        "fprop_input_fake_quant": fprop_inp,
        "fprop_weight_fake_quant": fprop_weight,
        "dgrad_gradient_fake_quant": dgrad_grad,
        "dgrad_weight_fake_quant": dgrad_weight,
        "wgrad_gradient_fake_quant": wgrad_grad,
        "wgrad_input_fake_quant": wgrad_input,
        "fprop_fp8": not (fprop_inp or fprop_weight),
        "dgrad_fp8": not (dgrad_weight or dgrad_grad),
        "wgrad_fp8": not (wgrad_grad or wgrad_input),
    }
    if WORLD_RANK == 0:
        fake_quant_fp8_create_config(
            fprop_inp,
            fprop_weight,
            dgrad_weight,
            dgrad_grad,
            wgrad_input,
            wgrad_grad,
            kwargs["config_file"],
        )
    dist.barrier()
    _init_debug(kwargs["config_file"].name, kwargs["log_dir"], FEATURE_DIRS)

    x, weight = _get_tensors(parallel_mode)
    model = _init_model(weight, parallel_mode)
    y = _run_forward_backward(x, model, parallel_mode)

    output = {"activation": y.clone(), "wgrad": model.weight.grad.clone(), "dgrad": x.grad.clone()}
    fp8_kwargs["fprop_input_scale"] = (
        _get_current_scale(x, fprop_inp) if not fp8_kwargs["fprop_fp8"] else None
    )
    fp8_kwargs["fprop_weight_scale"] = (
        _get_current_scale(weight, fprop_weight) if not fp8_kwargs["fprop_fp8"] else None
    )
    fp8_kwargs["dgrad_gradient_scale"] = (
        _get_current_scale(y.grad, dgrad_grad) if not fp8_kwargs["dgrad_fp8"] else None
    )
    fp8_kwargs["dgrad_weight_scale"] = (
        _get_current_scale(weight, dgrad_weight) if not fp8_kwargs["dgrad_fp8"] else None
    )
    fp8_kwargs["wgrad_gradient_scale"] = (
        _get_current_scale(y.grad, wgrad_grad) if not fp8_kwargs["wgrad_fp8"] else None
    )
    fp8_kwargs["wgrad_input_scale"] = (
        _get_current_scale(x, wgrad_input) if not fp8_kwargs["wgrad_fp8"] else None
    )
    ground_truth = _emulate_linear_distributed(x, weight, parallel_mode=parallel_mode, **fp8_kwargs)
    _cmp(ground_truth, output)


def _init_distributed():
    global WORLD_RANK, WORLD_SIZE, NCCL_WORLD, FP8

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
    }
    dist_init_kwargs["init_method"] = "env://"
    dist_init_kwargs["device_id"] = torch.device(f"cuda:{LOCAL_RANK}")
    assert dist.is_nccl_available()
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(**dist_init_kwargs)

    NCCL_WORLD = dist.new_group(backend="nccl")

    WORLD_SIZE = dist.get_world_size()


def _run_test_with_combinations(
    test_function, values_list, num_repeat, extra_args, sample_size=None
):
    combinations = itertools.product(values_list, repeat=num_repeat)
    total_combinations = itertools.product(combinations, extra_args)

    if sample_size is not None:
        total_combinations = random.sample(list(total_combinations), sample_size)

    for comb, arg in total_combinations:
        test_function(*comb, arg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dirs", type=str)
    args = parser.parse_args()
    FEATURE_DIRS = args.feature_dirs
    random.seed(SEED)
    _init_distributed()

    test_log_expert_parallel()
    for parallel_mode in ["column", "row"]:
        for gather_weight in [True, False]:
            test_log_distributed(parallel_mode, gather_weight)

    for parallel_mode in ["row", "column"]:
        test_disable_fp8_layer(parallel_mode)

    # test_disable_fp8_gemms
    _run_test_with_combinations(
        test_disable_fp8_gemms, all_boolean, num_repeat=3, extra_args=["column", "row"]
    )

    # test_fake_quant_fp8
    dtype_options = [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2, None]
    _run_test_with_combinations(
        test_fake_quant_fp8,
        dtype_options,
        num_repeat=6,
        extra_args=["column", "row"],
        sample_size=20,
    )

    _run_test_with_combinations(
        test_per_tensor_scaling,
        all_boolean,
        num_repeat=6,
        extra_args=["column"],
        sample_size=20,
    )
