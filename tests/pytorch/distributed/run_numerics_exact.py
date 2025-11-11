#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import datetime
import os
import sys
from functools import wraps
import math

import transformer_engine.pytorch as te
import torch
from torch import nn
import torch.distributed as dist
import transformer_engine_torch as tex
from transformer_engine.common.recipe import (
    NVFP4BlockScaling,
    Format,
    Recipe,
    QParams,
)
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.constants import NVFP4_BLOCK_SCALING_SIZE
from run_layer_with_overlap import _compare_tensors


BATCH_SIZE, HIDDEN_SIZE, OUT_SIZE = 128, 256, 128
WORLD_RANK, WORLD_SIZE = None, None
NCCL_WORLD = None
LOSS_FN = nn.MSELoss()
QUANTIZATION = None


def nvfp4_rht_and_2d_quantization():
    nvfp4_recipe = NVFP4BlockScaling()
    nvfp4_recipe.fp4_quant_fwd_inp = QParams(
        random_hadamard_transform=True, fp4_2d_quantization=False
    )
    nvfp4_recipe.fp4_quant_fwd_weight = QParams(
        random_hadamard_transform=False, fp4_2d_quantization=True
    )
    nvfp4_recipe.fp4_quant_bwd_grad = QParams(
        random_hadamard_transform=True, fp4_2d_quantization=False
    )
    return nvfp4_recipe


# Quantization recipe setup
def quantization_recipe() -> Recipe:
    if QUANTIZATION == "nvfp4":
        return nvfp4_rht_and_2d_quantization()
    raise ValueError(f"Unsupported quantization: {QUANTIZATION}")


def setup_environment_for_reference():
    if QUANTIZATION == "nvfp4":
        os.environ["QAT_PARAMS"] = "9003"
    else:
        raise ValueError(f"Unsupported quantization for reference: {QUANTIZATION}")


def cleanup_environment():
    if "QAT_PARAMS" in os.environ:
        del os.environ["QAT_PARAMS"]


def main(argv=None, namespace=None):
    global WORLD_RANK, WORLD_SIZE, NCCL_WORLD, QUANTIZATION, BATCH_SIZE, HIDDEN_SIZE, OUT_SIZE

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
    parser.add_argument("--quantization", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--out-size", type=int, default=128)
    args = parser.parse_args(argv, namespace)

    # Quantization scheme
    QUANTIZATION = args.quantization
    BATCH_SIZE = args.batch_size
    HIDDEN_SIZE = args.hidden_size
    OUT_SIZE = args.out_size

    test_dict = [
        test_linear,
        test_layernorm_linear,
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


def dist_print(msg, src=None, end="\n", error=False):
    stream = sys.stderr if error else sys.stdout
    if WORLD_RANK == (0 if src is None else src):
        stream.write(f"[rank{WORLD_RANK}] {msg}{end}\n")


############################################
#                   Linear                 #
############################################
class TestDistributedLinearBase:
    @staticmethod
    def _prepare_data(
        batch_size, hidden_size, out_size, use_bias=True, seed=0, dtype=torch.float32
    ):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        x = torch.randn((batch_size, hidden_size), dtype=dtype, device="cuda")
        w = torch.randn((out_size, hidden_size), dtype=dtype, device="cuda")
        bias = torch.randn((out_size), dtype=dtype, device="cuda") if use_bias else None
        gradient = torch.randn((batch_size, out_size), dtype=dtype, device="cuda")

        return x, w, bias, gradient

    @staticmethod
    def _shard_tensor(x, world_size, axis):
        split_size = x.size()[axis] // world_size
        split_tensor = torch.split(x, split_size, axis)
        out = []
        for tensor in split_tensor:
            out.append(tensor.detach().clone().requires_grad_(x.requires_grad))
        return out

    @staticmethod
    def _gather_tensor(local, world_size, tp_group, concat_dim):
        out_list = [torch.zeros_like(local) for _ in range(world_size)]
        torch.distributed.all_gather(out_list, local, tp_group)
        return torch.cat(out_list, dim=concat_dim)

    @staticmethod
    def _all_reduce_tensor(local, world_size, tp_group):
        if world_size == 1:
            return local
        handle = torch.distributed.all_reduce(local, group=tp_group, async_op=False)
        return local

    @staticmethod
    def _get_sum_abs_error(a, b):
        return torch.sum(torch.abs(a - b))

    @staticmethod
    def _get_mean_abs_relative_error(a, b):
        error = torch.where(b == 0, torch.ne(a, b), torch.abs((a - b) / b))
        return torch.mean(error)

    @classmethod
    def run_linear_preprocess_parallel(
        cls,
        x,
        w,
        bias,
        gradient,
        parallel_mode=None,
        sequence_parallel=False,
        tp_size=1,
        rank=0,
    ):
        if tp_size > 1:
            if parallel_mode == "column":
                # split w in N dim, which should be axis 0
                w = cls._shard_tensor(w, tp_size, 0)[rank]
                bias = cls._shard_tensor(bias, tp_size, 0)[rank] if bias is not None else None
                # split gradient in N dim, which should be axis 1
                gradient = cls._shard_tensor(gradient, tp_size, 1)[rank]
                if sequence_parallel:
                    # split x in M dim, which should be axis 0
                    x = cls._shard_tensor(x, tp_size, 0)[rank]
            # row parallel, split x in k dim, which should be axis 1, split w in k dim, should be axis 1
            if parallel_mode == "row":
                # split x in K dim, which should be axis 1
                x = cls._shard_tensor(x, tp_size, 1)[rank]
                # split w in K dim, which should be axis 1
                w = cls._shard_tensor(w, tp_size, 1)[rank]
                if sequence_parallel:
                    # split gradient in M dim, which should be axis 0
                    gradient = cls._shard_tensor(gradient, tp_size, 0)[rank]
        return x, w, bias, gradient

    @classmethod
    def run_linear_postprocess_parallel(
        cls,
        y_q,
        dgrad,
        wgrad,
        bgrad,
        parallel_mode,
        sequence_parallel,
        tp_size,
        tp_group,
    ):
        if tp_size > 1:
            if parallel_mode == "column":
                # gather y_q in N dim, which should be axis 1
                y_q = cls._gather_tensor(y_q, tp_size, tp_group, 1)
                # gather wgrad in N dim, which should be axis 0
                wgrad = cls._gather_tensor(wgrad, tp_size, tp_group, 0)
                # gather bgrad in N dim, which should be axis 0
                bgrad = (
                    cls._gather_tensor(bgrad, tp_size, tp_group, 0) if bgrad is not None else None
                )
                if sequence_parallel:
                    # gather dgrad in M dim, which should be axis 0
                    dgrad = cls._gather_tensor(dgrad, tp_size, tp_group, 0)
            if parallel_mode == "row":
                # gather dgrad in K dim, which should be axis 1
                dgrad = cls._gather_tensor(dgrad, tp_size, tp_group, 1)
                # gather wgrad in K dim, which should be axis 1
                wgrad = cls._gather_tensor(wgrad, tp_size, tp_group, 1)
                if sequence_parallel:
                    # gather y_q in M dim, which should be axis 0
                    y_q = cls._gather_tensor(y_q, tp_size, tp_group, 0)
                    # we need to sum bias gradient when using TP + SP
                    bgrad = (
                        cls._all_reduce_tensor(bgrad, tp_size, tp_group)
                        if bgrad is not None
                        else None
                    )

        return y_q, dgrad, wgrad, bgrad

    @classmethod
    def run_linear_one_step(
        cls, layer, x, gradient, is_first_microbatch=None, fuse_wgrad_accumulation=False
    ):
        # reset gradients
        layer.zero_grad()
        x.grad = None

        # Forward pass
        if isinstance(layer, te.Linear):
            # Kitchen Linear
            y_q = layer.forward(x, is_first_microbatch=is_first_microbatch)
        else:
            # the default torch.nn.Linear
            y_q = layer(x)

        # Backward pass
        y_q.backward(gradient)

        # Collect gradients
        dgrad = x.grad
        bgrad = (
            layer._parameters["bias"].grad
            if layer._parameters.get("bias", None) is not None
            else None
        )
        assert "weight" in layer._parameters
        if fuse_wgrad_accumulation:
            wgrad = layer._parameters["weight"].main_grad
            assert layer._parameters["weight"].grad is None
        else:
            wgrad = layer._parameters["weight"].grad

        return y_q, dgrad, wgrad, bgrad

    @classmethod
    def run_linear_multiple_steps(
        cls,
        layer,
        x,
        gradient,
        run_num_steps,
        enable_weight_cache,
        fuse_wgrad_accumulation=False,
    ):
        """
        Run multiple steps of linear layer and collect results.
        """

        y_q_list, dgrad_list, wgrad_list = [], [], []
        bgrad_list = [] if layer._parameters.get("bias", None) is not None else None

        for i in range(run_num_steps):
            x_i = (x + i).clone().detach().requires_grad_(True)
            # run_linear_one_step
            y_q, dgrad, wgrad, bgrad = cls.run_linear_one_step(
                layer,
                x_i,
                gradient,
                is_first_microbatch=(i == 0) if enable_weight_cache else None,
                fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            )

            # Collect results
            y_q_list.append(y_q.detach().clone())
            dgrad_list.append(dgrad.detach().clone())
            wgrad_list.append(wgrad.detach().clone())
            if bgrad_list is not None and bgrad is not None:
                bgrad_list.append(bgrad.detach().clone())

        # Stack the results
        return (
            torch.stack(y_q_list),
            torch.stack(dgrad_list),
            torch.stack(wgrad_list),
            torch.stack(bgrad_list) if bgrad_list is not None else None,
        )

    @classmethod
    def run_linear(
        cls,
        x,
        w,
        bias,
        gradient,
        parallel_mode=None,
        sequence_parallel=False,
        tp_group=None,
        tp_size=1,
        rank=0,
        run_num_steps=1,
        enable_weight_cache=False,
        fuse_wgrad_accumulation=False,
    ):
        """
        If Model parallel, split inputs for a given rank and return the gathered output and gradients, so that they can be compared with
        the reference single GPU run.
        """
        # clone inputs and move to current device
        # w has shape [N, K], x has shape [M, K], gradient has shape [M, N]
        x = x.clone().detach().requires_grad_(True).to("cuda")
        w = w.clone().detach().to("cuda")
        gradient = gradient.clone().detach().to("cuda")
        bias = bias.clone().detach().to("cuda") if bias is not None else None
        in_features = x.shape[1]
        out_features = w.shape[0]

        # If Model parallel: split inputs for a given rank
        x, w, bias, gradient = cls.run_linear_preprocess_parallel(
            x, w, bias, gradient, parallel_mode, sequence_parallel, tp_size, rank
        )

        # set data types
        params_dtype = x.dtype

        # Create linear layer and copy weights
        layer = te.Linear(
            in_features,
            out_features,
            bias=bias is not None,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            tp_size=tp_size,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )

        layer = layer.to("cuda")

        with torch.no_grad():
            layer.weight.copy_(w)
            if bias is not None:
                layer.bias.copy_(bias)

        if fuse_wgrad_accumulation:
            assert (
                run_num_steps > 1
            ), "Fused weight gradient accumulation requires run_num_steps > 1"
            layer.weight.main_grad = torch.zeros_like(layer.weight)

        # Run one step or multiple steps
        if run_num_steps == 1:
            y_q, dgrad, wgrad, bgrad = cls.run_linear_one_step(layer, x, gradient)
        else:
            y_q, dgrad, wgrad, bgrad = cls.run_linear_multiple_steps(
                layer,
                x,
                gradient,
                run_num_steps,
                enable_weight_cache,
                fuse_wgrad_accumulation,
            )

        # If Model parallel: gather output and gradients from all ranks
        y_q, dgrad, wgrad, bgrad = cls.run_linear_postprocess_parallel(
            y_q,
            dgrad,
            wgrad,
            bgrad,
            parallel_mode,
            sequence_parallel,
            tp_size,
            tp_group,
        )

        return y_q, dgrad, wgrad, bgrad


@run_distributed_test()
def _test_linear(parallel_mode=None, sequence_parallel=False, **kwargs):
    """Test the linear layer with specified parallel mode and sequence parallelization.

    Args:
        parallel_mode (str): 'row' or 'column' parallelism.
        sequence_parallel (bool): Enable sequence parallelism if True.
        kwargs (dict): Additional arguments for the linear layer.

        QUANTIZATION options: nvfp4 <=> experimental nvfp4 as a reference
    """
    params_dtype = torch.bfloat16
    use_bias = kwargs.get("bias", True)
    fuse_wgrad_accumulation = kwargs.get("fuse_wgrad_accumulation", False)
    seed = torch.initial_seed()
    recipe = quantization_recipe()

    # turn on weight quantization cache when fusing wgrad accumulation
    enable_weight_cache = fuse_wgrad_accumulation
    run_num_steps = 1 if not fuse_wgrad_accumulation else 5

    x, w, bias, gradient = TestDistributedLinearBase._prepare_data(
        BATCH_SIZE, HIDDEN_SIZE, OUT_SIZE, use_bias=use_bias, seed=seed, dtype=params_dtype
    )

    # run the recipe under test
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        y_q, dgrad, wgrad, bgrad = TestDistributedLinearBase.run_linear(
            x,
            w,
            bias,
            gradient,
            parallel_mode=parallel_mode,
            sequence_parallel=sequence_parallel,
            tp_group=NCCL_WORLD,
            tp_size=WORLD_SIZE,
            rank=WORLD_RANK,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            run_num_steps=1 if not fuse_wgrad_accumulation else 5,
            enable_weight_cache=fuse_wgrad_accumulation,
        )

    # run the reference
    setup_environment_for_reference()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        y_q_ref, dgrad_ref, wgrad_ref, bgrad_ref = TestDistributedLinearBase.run_linear(
            x,
            w,
            bias,
            gradient,
            parallel_mode=parallel_mode,
            sequence_parallel=sequence_parallel,
            tp_group=NCCL_WORLD,
            tp_size=WORLD_SIZE,
            rank=WORLD_RANK,
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
            run_num_steps=run_num_steps,
            enable_weight_cache=enable_weight_cache,
        )
    # Clean up env
    cleanup_environment()

    # compare results, zero tolerance
    if WORLD_RANK == 0:
        torch.testing.assert_close(y_q, y_q_ref, atol=0, rtol=0, msg="Output mismatch")
        torch.testing.assert_close(dgrad, dgrad_ref, atol=0, rtol=0, msg="Dgrad mismatch")
        torch.testing.assert_close(wgrad, wgrad_ref, atol=0, rtol=0, msg="Wgrad mismatch")
        if bgrad is not None and bgrad_ref is not None:
            torch.testing.assert_close(bgrad, bgrad_ref, atol=0, rtol=0, msg="Bgrad mismatch")


def test_linear():
    """Run linear layer tests with various configurations."""
    kwargs_list = [
        {"bias": False},
    ]

    for kwargs in kwargs_list:
        if kwargs.get("save_original_input", False) and QUANTIZATION == "fp8":
            continue
        for parallel_mode in ["column", "row"]:
            for sequence_parallel in [False, True]:
                _test_linear(parallel_mode, sequence_parallel, **kwargs)


############################################
#              LayerNormLinear             #
############################################
class TestDistributedLayerNormLinearBase(TestDistributedLinearBase):

    @classmethod
    def run_linear_one_step(cls, layer, x, gradient, is_first_microbatch=None):
        # reset gradients
        layer.zero_grad()
        x.grad = None

        # Forward pass
        y_q, ln_out = layer.forward(x, is_first_microbatch=is_first_microbatch)

        # Backward pass
        y_q.backward(gradient)

        # Collect gradients
        dgrad = x.grad

        parameters = layer._parameters

        # bias and weight gradients
        bgrad = parameters["bias"].grad if parameters.get("bias", None) is not None else None
        assert "weight" in parameters
        wgrad = parameters["weight"].grad

        return y_q, ln_out, dgrad, wgrad, bgrad

    @classmethod
    def run_linear_multiple_steps(
        cls, layer, x, gradient, run_num_steps, enable_weight_cache, fuse_wgrad_accumulation=False
    ):
        # raise error, no test case for multiple steps for now
        raise NotImplementedError("LayerNormLinear does not support test multiple steps for now")

    @classmethod
    def run_layernorm_linear(
        cls,
        x,
        w,
        bias,
        gradient,
        parallel_mode=None,
        sequence_parallel=False,
        tp_group=None,
        tp_size=1,
        rank=0,
        run_num_steps=1,
        enable_weight_cache=False,
        LayerNormLinearClass=te.LayerNormLinear,
        normalization="LayerNorm",
    ):
        """
        If Model parallel, split inputs for a given rank and return the gathered output and gradients, so that they can be compared with
        the reference single GPU run.
        """
        # clone inputs and move to current device
        # w has shape [N, K], x has shape [M, K], gradient has shape [M, N]
        x = x.clone().detach().requires_grad_(True).to("cuda")
        w = w.clone().detach().to("cuda")
        gradient = gradient.clone().detach().to("cuda")
        bias = bias.clone().detach().to("cuda") if bias is not None else None
        in_features = x.shape[1]
        out_features = w.shape[0]

        # If Model parallel: split inputs for a given rank
        x, w, bias, gradient = cls.run_linear_preprocess_parallel(
            x, w, bias, gradient, parallel_mode, sequence_parallel, tp_size, rank
        )

        # set data types
        params_dtype = x.dtype

        # Create linear layer and copy weights
        layer = LayerNormLinearClass(
            in_features,
            out_features,
            bias=bias is not None,
            params_dtype=params_dtype,
            parallel_mode=parallel_mode,
            sequence_parallel=sequence_parallel,
            tp_group=tp_group,
            tp_size=tp_size,
            normalization=normalization,
            return_layernorm_output=True,
        )

        layer = layer.to("cuda")

        # Copy weights
        # kitchen_linear has different parameter names
        with torch.no_grad():
            layer.weight.copy_(w)
            if bias is not None:
                layer.bias.copy_(bias)

        # Run one step
        y_q, ln_out, dgrad, wgrad, bgrad = cls.run_linear_one_step(layer, x, gradient)

        # If Model parallel: gather output and gradients from all ranks
        y_q, dgrad, wgrad, bgrad = cls.run_linear_postprocess_parallel(
            y_q,
            dgrad,
            wgrad,
            bgrad,
            parallel_mode,
            sequence_parallel,
            tp_size,
            tp_group,
        )

        return y_q, ln_out, dgrad, wgrad, bgrad


@run_distributed_test()
def _test_layernorm_linear(parallel_mode=None, sequence_parallel=False, **kwargs):
    """Test the linear layer with specified parallel mode and sequence parallelization.

    Args:
        parallel_mode (str): 'column' parallelism.
        sequence_parallel (bool): Enable sequence parallelism if True.
        kwargs (dict): Additional arguments for the linear layer.
    """
    params_dtype = torch.bfloat16
    use_bias = kwargs.get("bias", True)
    seed = torch.initial_seed()
    recipe = quantization_recipe()

    # run multiple steps currently not supported for LayerNormLinear
    run_num_steps = 1

    x, w, bias, gradient = TestDistributedLayerNormLinearBase._prepare_data(
        BATCH_SIZE, HIDDEN_SIZE, OUT_SIZE, use_bias=use_bias, seed=seed, dtype=params_dtype
    )

    # run the recipe under test
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        y_q, ln_out, dgrad, wgrad, bgrad = TestDistributedLayerNormLinearBase.run_layernorm_linear(
            x,
            w,
            bias,
            gradient,
            parallel_mode=parallel_mode,
            sequence_parallel=sequence_parallel,
            tp_group=NCCL_WORLD,
            tp_size=WORLD_SIZE,
            rank=WORLD_RANK,
            run_num_steps=run_num_steps,
            enable_weight_cache=False,
        )

    # run the reference
    setup_environment_for_reference()
    with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
        y_q_ref, ln_out_ref, dgrad_ref, wgrad_ref, bgrad_ref = (
            TestDistributedLayerNormLinearBase.run_layernorm_linear(
                x,
                w,
                bias,
                gradient,
                parallel_mode=parallel_mode,
                sequence_parallel=sequence_parallel,
                tp_group=NCCL_WORLD,
                tp_size=WORLD_SIZE,
                rank=WORLD_RANK,
                run_num_steps=run_num_steps,
                enable_weight_cache=False,
            )
        )
    # Clean up env
    cleanup_environment()

    # compare results, zero tolerance
    if WORLD_RANK == 0:
        torch.testing.assert_close(y_q, y_q_ref, atol=0, rtol=0, msg="Output mismatch")
        torch.testing.assert_close(ln_out, ln_out_ref, atol=0, rtol=0, msg="LN output mismatch")
        torch.testing.assert_close(dgrad, dgrad_ref, atol=0, rtol=0, msg="Dgrad mismatch")
        torch.testing.assert_close(wgrad, wgrad_ref, atol=0, rtol=0, msg="Wgrad mismatch")
        if bgrad is not None and bgrad_ref is not None:
            torch.testing.assert_close(bgrad, bgrad_ref, atol=0, rtol=0, msg="Bgrad mismatch")


def test_layernorm_linear():
    kwargs_list = [
        {"bias": False},
    ]

    for kwargs in kwargs_list:
        for parallel_mode in ["column"]:
            for sequence_parallel in [False, True]:
                _test_layernorm_linear(parallel_mode, sequence_parallel, **kwargs)


if __name__ == "__main__":
    sys.exit(main())
