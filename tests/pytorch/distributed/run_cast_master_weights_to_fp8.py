#!/usr/bin/python3

# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import datetime
import os
import sys

import torch
from torch import nn
import torch.distributed as dist

from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    Format,
    Recipe,
)
import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor import QuantizedTensor, cast_master_weights_to_fp8
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor


def _get_raw_data(quantized_tensor):
    """Get the underlying data of a quantized tensor, used in zero-1 optimizer"""
    if isinstance(quantized_tensor, Float8Tensor):
        assert hasattr(quantized_tensor, "_data"), "Float8Tensor does not have _data attribute"
        assert quantized_tensor._data.dtype == torch.uint8, "Float8Tensor _data must be uint8"
        return quantized_tensor._data
    else:
        raise ValueError(f"Unsupported quantized tensor type: {type(quantized_tensor)}")


class MiniZero_1:
    """A mini zero-1 optimizer implementation, just used for this test"""

    def __init__(self, weights, lr, dp_group):
        self.rank = dist.get_rank(dp_group)
        self.world_size = dist.get_world_size(dp_group)

        self.weights = weights
        self.lr = lr
        self.dp_group = dp_group

        # [self.offsets[i], self.offsets[i+1]) is the range of weights[i] in the global buffer
        self.offsets = [0]
        for weight in self.weights:
            self.offsets.append(self.offsets[-1] + weight.numel())

        # Padding to avoid global buffer cannot be divided by world size, so the offsets[-1] may
        # not be the end range of the last weight.
        if self.offsets[-1] % self.world_size != 0:
            self.offsets[-1] += self.world_size - self.offsets[-1] % self.world_size

        self.master_weights = []
        # The start offset of the master weight in the weight
        self.start_offsets = []
        # The overlapping area of the weight and this rank's local buffer
        self.overlapping_areas = []

        # The start and end of this rank's local buffer in the global buffer
        rank_start = self.offsets[-1] // self.world_size * self.rank
        rank_end = rank_start + self.offsets[-1] // self.world_size

        for weight, offset in zip(self.weights, self.offsets[:-1]):
            if offset >= rank_end or (offset + weight.numel()) <= rank_start:
                # This weight is not in this rank's local buffer
                master_weight = None
                start_offset = None
                overlapping_area = None
            else:
                overlapping_start = max(rank_start, offset)
                overlapping_end = min(rank_end, offset + weight.numel())
                length = overlapping_end - overlapping_start
                start_offset = overlapping_start - offset
                if isinstance(weight, QuantizedTensor):
                    # If weight is a FP8 tensor, we need to use the original high precision version
                    # to initialize the master weight.
                    high_precision_init_val = weight.get_high_precision_init_val().view(-1)
                    master_weight = high_precision_init_val.to(weight.device).float()[
                        start_offset : start_offset + length
                    ]
                else:
                    master_weight = (
                        weight.detach().view(-1).float()[start_offset : start_offset + length]
                    )
                overlapping_area = (overlapping_start, overlapping_end)
            self.master_weights.append(master_weight)
            self.start_offsets.append(start_offset)
            self.overlapping_areas.append(overlapping_area)

        # Create global buffer for grads reduce-scatter
        self.grad_buffer = torch.empty(
            [self.offsets[-1]], dtype=torch.float32, device=weights[0].device
        )
        self.grad_buffer_slice = self.grad_buffer[rank_start:rank_end]

        # Create global buffer for weights all-gather
        if isinstance(self.weights[0], QuantizedTensor):
            weight_buffer_dtype = torch.uint8
        else:
            weight_buffer_dtype = weights[0].dtype
        self.weight_buffer = torch.empty(
            [self.offsets[-1]], dtype=weight_buffer_dtype, device=weights[0].device
        )
        self.weight_buffer_slice = self.weight_buffer[rank_start:rank_end]

    def step(self):
        # -----------------------------------------------------------------------------------------
        # Step 1: Copy grads to the grad buffer
        # -----------------------------------------------------------------------------------------
        for weight, offset in zip(self.weights, self.offsets[:-1]):
            start = offset
            end = offset + weight.numel()
            self.grad_buffer[start:end].copy_(weight.main_grad.view(-1))

        # -----------------------------------------------------------------------------------------
        # Step 2: Grads reduce-scatter
        # -----------------------------------------------------------------------------------------
        # Don't use reduce_scatter directly to explicitly control the reduce order.
        # dist.reduce_scatter_tensor(self.grad_buffer_slice, self.grad_buffer, op=dist.ReduceOp.AVG,
        #                            group=self.dp_group)
        buffers = [torch.empty_like(self.grad_buffer) for _ in range(self.world_size)]
        dist.all_gather(buffers, self.grad_buffer, group=self.dp_group)
        for i in range(1, self.world_size):
            buffers[0] += buffers[i]
        rank_start = self.offsets[-1] // self.world_size * self.rank
        rank_end = rank_start + self.offsets[-1] // self.world_size
        self.grad_buffer_slice.copy_(buffers[0][rank_start:rank_end])
        self.grad_buffer_slice /= self.world_size

        # -----------------------------------------------------------------------------------------
        # Step 3: Update master weights
        # -----------------------------------------------------------------------------------------
        for master_weight, overlapping_area in zip(self.master_weights, self.overlapping_areas):
            if master_weight is None:
                # This weight's master weight is in other rank.
                continue
            grad = self.grad_buffer[overlapping_area[0] : overlapping_area[1]]
            master_weight -= grad * self.lr

        # -----------------------------------------------------------------------------------------
        # Step 4: Cast master weights to BF16 or FP8, depending on the type of the weight
        # -----------------------------------------------------------------------------------------
        if isinstance(self.weights[0], QuantizedTensor):
            # FP8 weights case
            for i in range(1, len(self.weights)):
                assert isinstance(self.weights[i], QuantizedTensor)
            cast_master_weights_to_fp8(
                self.weights, self.master_weights, self.start_offsets, self.dp_group
            )
        else:
            # BF16 weights case
            for weight, master_weight, start_offset in zip(
                self.weights, self.master_weights, self.start_offsets
            ):
                if master_weight is None:
                    continue
                start = start_offset
                end = start_offset + master_weight.numel()
                weight.data.view(-1)[start:end].copy_(master_weight)

        # -----------------------------------------------------------------------------------------
        # Step 5: Copy the updated weights (not all weights) to the weight buffer
        # -----------------------------------------------------------------------------------------
        for i in range(len(self.weights)):
            master_weight = self.master_weights[i]
            if master_weight is None:
                continue
            start_offset = self.start_offsets[i]
            if isinstance(self.weights[i], QuantizedTensor):
                weight = _get_raw_data(self.weights[i])
            else:
                weight = self.weights[i]
            weight_slice = weight.view(-1)[start_offset : start_offset + master_weight.numel()]
            overlapping_start, overlapping_end = self.overlapping_areas[i]
            self.weight_buffer[overlapping_start:overlapping_end].copy_(weight_slice)

        # -----------------------------------------------------------------------------------------
        # Step 6: Weight all-gather (FP8 or BF16)
        # -----------------------------------------------------------------------------------------
        dist.all_gather_into_tensor(
            self.weight_buffer, self.weight_buffer_slice, group=self.dp_group
        )

        # -----------------------------------------------------------------------------------------
        # Step 7: Copy the gathered weights from weight buffer to the actual weights
        # -----------------------------------------------------------------------------------------
        for weight, offset in zip(self.weights, self.offsets[:-1]):
            start = offset
            end = offset + weight.numel()
            if isinstance(weight, QuantizedTensor):
                weight = _get_raw_data(weight)
            weight.view(-1).data.copy_(self.weight_buffer[start:end])


class MiniOptimizer:

    def __init__(self, weights, lr, dp_group):
        self.world_size = dist.get_world_size(dp_group)

        self.weights = weights
        self.lr = lr
        self.dp_group = dp_group

        master_weights = []
        for weight in self.weights:
            master_weights.append(weight.detach().float())
        self.master_weights = master_weights

    def step(self):
        for weight, master_weight in zip(self.weights, self.master_weights):
            main_grad = weight.main_grad

            # Don't use all-reduce directly to explicitly control the reduce order.
            # dist.all_reduce(main_grad, op=dist.ReduceOp.AVG, group=self.dp_group)
            buffers = [torch.empty_like(main_grad) for _ in range(self.world_size)]
            dist.all_gather(buffers, main_grad, group=self.dp_group)
            for i in range(1, self.world_size):
                buffers[0] += buffers[i]
            main_grad.copy_(buffers[0])
            main_grad /= self.world_size

            master_weight -= main_grad * self.lr
            weight.data.copy_(master_weight)


def _test_zero_1(dp_group):
    """Make sure the implementation of zero-1 optimizer is correct"""
    rank = dist.get_rank(dp_group)
    world_size = dist.get_world_size(dp_group)

    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    weights = [
        torch.randn(256 * 256, dtype=torch.bfloat16, device="cuda"),
        torch.randn(256 * 256 * 3, dtype=torch.bfloat16, device="cuda"),
        torch.randn(256 * 256 * 2 - 1, dtype=torch.bfloat16, device="cuda"),
    ]

    weights_1 = weights
    weights_2 = [weight.clone() for weight in weights]

    lr = 1.0
    optimizer_1 = MiniZero_1(weights_1, lr, dp_group)
    optimizer_2 = MiniOptimizer(weights_2, lr, dp_group)

    for _ in range(100):
        for w1, w2 in zip(weights_1, weights_2):
            main_grads = [
                torch.randn_like(w1, dtype=torch.float32, device="cuda") for _ in range(world_size)
            ]
            # Choose based on rank to make sure the grads of different ranks are different.
            main_grad = main_grads[rank]
            w1.main_grad = main_grad
            w2.main_grad = main_grad

        optimizer_1.step()
        optimizer_2.step()

        for w1, w2 in zip(weights_1, weights_2):
            torch.testing.assert_close(w1, w2, atol=0, rtol=0)


def quantization_recipe(quantization) -> Recipe:
    """Quantization recipe setup"""
    if quantization == "fp8":
        return DelayedScaling(
            fp8_format=Format.HYBRID, amax_history_len=32, amax_compute_algo="max"
        )
    elif quantization == "fp8_cs":
        return Float8CurrentScaling()
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")


def _test_cast_master_weights_to_fp8(quantization, dp_group):
    rank = dist.get_rank(dp_group)
    world_size = dist.get_world_size(dp_group)

    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    linear_kwargs = {"params_dtype": torch.bfloat16, "bias": False, "fuse_wgrad_accumulation": True}

    # Create model with FP8 weights
    with te.fp8.fp8_model_init(
        enabled=quantization is not None,
        recipe=quantization_recipe(quantization),
        preserve_high_precision_init_val=True,
    ):
        model_fp8 = nn.Sequential(
            te.Linear(128, 256, **linear_kwargs),
            te.Linear(256, 256 * 3, **linear_kwargs),
            te.Linear(256 * 3, 128, **linear_kwargs),
        )

    # Create model with BF16 weights
    model = nn.Sequential(
        te.Linear(128, 256, **linear_kwargs),
        te.Linear(256, 256 * 3, **linear_kwargs),
        te.Linear(256 * 3, 128, **linear_kwargs),
    )

    # Make sure the BF16 model and FP8 model have the same initial weights
    for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
        high_precision_init_val = w_fp8.get_high_precision_init_val()
        w.data.copy_(high_precision_init_val)

    # Allocate main_grads for each weight
    for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
        w_fp8.main_grad = torch.zeros_like(w_fp8, dtype=torch.float32, device="cuda")
        w.main_grad = torch.zeros_like(w, dtype=torch.float32, device="cuda")

    optimizer_fp8 = MiniZero_1([w for w in model_fp8.parameters()], 10.0, dp_group)
    optimizer = MiniZero_1([w for w in model.parameters()], 10.0, dp_group)

    for _ in range(100):
        for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
            w_fp8.main_grad.zero_()
            w.main_grad.zero_()

        inputs = [
            torch.randn(16, 128, dtype=torch.bfloat16, device="cuda") for _ in range(world_size)
        ]
        # Choose based on rank to make sure the inputs of different ranks are different.
        x = inputs[rank]

        with te.fp8.fp8_autocast(
            enabled=quantization is not None,
            fp8_recipe=quantization_recipe(quantization),
        ):
            y_fp8 = model_fp8(x)

        with te.fp8_autocast(
            enabled=quantization is not None,
            fp8_recipe=quantization_recipe(quantization),
        ):
            y = model(x)

        targets = [torch.randn_like(y) for _ in range(world_size)]
        # Choose based on rank to make sure the targets of different ranks are different.
        target = targets[rank]
        loss_fp8 = nn.MSELoss()(y_fp8, target)
        loss = nn.MSELoss()(y, target)

        loss_fp8.backward()
        loss.backward()

        optimizer_fp8.step()
        optimizer.step()

        torch.testing.assert_close(loss_fp8, loss, atol=0, rtol=0)


def main(argv=None, namespace=None):
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", type=str, default=None, choices=["fp8", "fp8_cs"])
    args = parser.parse_args(argv, namespace)

    dp_group = dist.new_group(backend="nccl")
    _test_zero_1(dp_group)
    _test_cast_master_weights_to_fp8(args.quantization, dp_group)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":

    sys.exit(main())
