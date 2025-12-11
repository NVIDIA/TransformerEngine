# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import datetime
import os
import subprocess
import sys
import pathlib

import pytest
import torch
from torch import nn
import torch.distributed as dist

from transformer_engine.common.recipe import (
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    NVFP4BlockScaling,
    Format,
    Recipe,
)
import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_nvfp4_available,
    QuantizedTensor,
    Float8Tensor,
    Float8BlockwiseQTensor,
    NVFP4Tensor,
)
from transformer_engine.pytorch.tensor import (
    cast_master_weights_to_fp8,
    cast_master_weights_to_nvfp4,
)
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.tensor.utils import post_all_gather_processing, replace_raw_data


def _get_quantization_recipe(quantization) -> Recipe:
    """Quantization recipe setup"""
    fp8_format = Format.HYBRID
    if quantization == "fp8":
        return DelayedScaling(fp8_format=fp8_format, amax_history_len=32, amax_compute_algo="max")
    elif quantization == "fp8_cs":
        return Float8CurrentScaling(fp8_format=fp8_format)
    elif quantization == "fp8_block":
        return Float8BlockScaling(fp8_format=fp8_format)
    else:
        raise ValueError(f"Unsupported quantization: {quantization}")


def _get_raw_data(quantized_tensor):
    """Get the underlying data of a quantized tensor, used in zero-1 optimizer"""
    if isinstance(quantized_tensor, Float8Tensor):
        assert hasattr(quantized_tensor, "_data"), "Float8Tensor does not have _data attribute"
        assert quantized_tensor._data.dtype == torch.uint8, "Float8Tensor _data must be uint8"
        return quantized_tensor._data
    elif isinstance(quantized_tensor, Float8BlockwiseQTensor):
        assert hasattr(
            quantized_tensor, "_rowwise_data"
        ), "Float8BlockwiseQTensor does not have _rowwise_data attribute"
        assert (
            quantized_tensor._rowwise_data.dtype == torch.uint8
        ), "Float8BlockwiseQTensor _rowwise_data must be uint8"
        return quantized_tensor._rowwise_data
    elif isinstance(quantized_tensor, NVFP4Tensor):
        assert hasattr(quantized_tensor, "_rowwise_data"), "NVFP4Tensor missing _rowwise_data"
        assert (
            quantized_tensor._rowwise_data.dtype == torch.uint8
        ), "NVFP4Tensor _rowwise_data must be uint8"
        return quantized_tensor._rowwise_data
    else:
        raise ValueError(f"Unsupported quantized tensor type: {type(quantized_tensor)}")


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


class MiniZero_1:
    """A mini zero-1 optimizer implementation, just used for this test"""

    def __init__(self, weights, lr, dp_group, manual_post_all_gather_processing=False):
        self.rank = dist.get_rank(dp_group)
        self.world_size = dist.get_world_size(dp_group)

        self.weights = weights
        self.lr = lr
        self.dp_group = dp_group
        self.manual_post_all_gather_processing = manual_post_all_gather_processing

        # [self.offsets[i], self.offsets[i+1]) is the range of weights[i] in the global buffer
        self.offsets = [0]
        for weight in self.weights:
            self.offsets.append(self.offsets[-1] + weight.numel())
        print(f"offsets: {self.offsets}")
        # Padding to avoid global buffer cannot be divided by world size, so the offsets[-1] may
        # not be the end range of the last weight.
        if self.offsets[-1] % self.world_size != 0:
            self.offsets[-1] += self.world_size - self.offsets[-1] % self.world_size

        self.weights_are_nvfp4 = isinstance(self.weights[0], NVFP4Tensor)

        # Storage offsets operate on the packed representation (e.g., NVFP4 uint8 data).
        self.storage_offsets = None
        self.storage_sizes = None
        self.storage_total = None
        if self.weights_are_nvfp4:
            self.storage_offsets = [0]
            self.storage_sizes = []
            for weight in self.weights:
                storage_size = _get_raw_data(weight).view(-1).numel()
                self.storage_sizes.append(storage_size)
                self.storage_offsets.append(self.storage_offsets[-1] + storage_size)
            if self.storage_offsets[-1] % self.world_size != 0:
                self.storage_offsets[-1] += (
                    self.world_size - self.storage_offsets[-1] % self.world_size
                )
            self.storage_total = self.storage_offsets[-1]

        self.master_weights = []
        # The start offset of the master weight in the weight
        self.start_offsets = []
        # The overlapping area of the weight and this rank's local buffer
        self.overlapping_areas = []
        # Storage equivalents (only populated for NVFP4 tensors).
        self.storage_start_offsets = [None] * len(self.weights)
        self.storage_overlapping_areas = [None] * len(self.weights)

        # The start and end of this rank's local buffer in the global buffer
        rank_start = self.offsets[-1] // self.world_size * self.rank
        rank_end = rank_start + self.offsets[-1] // self.world_size
        print(f"current rank: {self.rank}, rank_start: {rank_start}, rank_end: {rank_end}")
        storage_rank_start = None
        storage_rank_end = None
        if self.weights_are_nvfp4:
            storage_rank_start = self.storage_total // self.world_size * self.rank
            storage_rank_end = storage_rank_start + self.storage_total // self.world_size
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

        if self.weights_are_nvfp4:
            for idx, (weight, storage_offset, storage_size) in enumerate(
                zip(self.weights, self.storage_offsets[:-1], self.storage_sizes)
            ):
                if storage_offset >= storage_rank_end or (storage_offset + storage_size) <= storage_rank_start:
                    continue
                overlap_start = max(storage_rank_start, storage_offset)
                overlap_end = min(storage_rank_end, storage_offset + storage_size)
                self.storage_start_offsets[idx] = overlap_start - storage_offset
                self.storage_overlapping_areas[idx] = (overlap_start, overlap_end)

        # Create global buffer for grads reduce-scatter
        self.grad_buffer = torch.empty(
            [self.offsets[-1]], dtype=torch.float32, device=weights[0].device
        )
        self.grad_buffer_slice = self.grad_buffer[rank_start:rank_end]

        # Create global buffer for weights all-gather
        if isinstance(self.weights[0], QuantizedTensor):
            weight_buffer_dtype = torch.uint8
            if self.weights_are_nvfp4:
                weight_buffer_length = self.storage_total
                buffer_rank_start = storage_rank_start
                buffer_rank_end = storage_rank_end
            else:
                weight_buffer_length = self.offsets[-1]
                buffer_rank_start = rank_start
                buffer_rank_end = rank_end
        else:
            weight_buffer_dtype = weights[0].dtype
            weight_buffer_length = self.offsets[-1]
            buffer_rank_start = rank_start
            buffer_rank_end = rank_end
        self.weight_buffer = torch.empty(
            [weight_buffer_length], dtype=weight_buffer_dtype, device=weights[0].device
        )
        self.weight_buffer_slice = self.weight_buffer[buffer_rank_start:buffer_rank_end]

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
        first_weight = self.weights[0]
        if isinstance(first_weight, NVFP4Tensor):
            for weight in self.weights:
                assert isinstance(weight, NVFP4Tensor)
            cast_master_weights_to_nvfp4(
                self.weights,
                self.master_weights,
                self.start_offsets,
                self.dp_group,
            )
        elif isinstance(first_weight, QuantizedTensor):
            for weight in self.weights:
                assert isinstance(weight, QuantizedTensor)
            cast_master_weights_to_fp8(
                self.weights,
                self.master_weights,
                self.start_offsets,
                self.dp_group,
                manual_post_all_gather_processing=self.manual_post_all_gather_processing,
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
            if isinstance(self.weights[i], NVFP4Tensor):
                storage_start = self.storage_start_offsets[i]
                storage_overlap = self.storage_overlapping_areas[i]
                if storage_start is None or storage_overlap is None:
                    continue
                weight = _get_raw_data(self.weights[i]).view(-1)
                storage_len = storage_overlap[1] - storage_overlap[0]
                weight_slice = weight[storage_start : storage_start + storage_len]
                overlapping_start, overlapping_end = storage_overlap
                buffer_len = overlapping_end - overlapping_start
                slice_len = weight_slice.numel()
                if buffer_len != slice_len:
                    print(
                        "[MiniZero_1] copy mismatch:",
                        f"idx={i}",
                        f"buffer_len={buffer_len}",
                        f"slice_len={slice_len}",
                        f"weight_shape={tuple(weight.shape)}",
                        f"storage_start={storage_start}",
                        f"storage_len={storage_len}",
                        f"overlap=({overlapping_start},{overlapping_end})",
                    )
                self.weight_buffer[overlapping_start:overlapping_end].copy_(weight_slice)
                continue
            elif isinstance(self.weights[i], QuantizedTensor):
                weight = _get_raw_data(self.weights[i])
            else:
                weight = self.weights[i]
            weight_slice = weight.view(-1)[start_offset : start_offset + master_weight.numel()]
            overlapping_start, overlapping_end = self.overlapping_areas[i]
            buffer_len = overlapping_end - overlapping_start
            slice_len = weight_slice.numel()
            if buffer_len != slice_len:
                print(
                    "[MiniZero_1] copy mismatch:",
                    f"idx={i}",
                    f"buffer_len={buffer_len}",
                    f"slice_len={slice_len}",
                    f"weight_shape={tuple(weight.shape)}",
                    f"start_offset={start_offset}",
                    f"master_numel={master_weight.numel()}",
                    f"overlap=({overlapping_start},{overlapping_end})",
                )
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
        for idx, weight in enumerate(self.weights):
            if isinstance(weight, NVFP4Tensor):
                start = self.storage_offsets[idx]
                end = start + self.storage_sizes[idx]
                weight_data = _get_raw_data(weight)
                buffer_len = end - start
                slice_len = weight_data.view(-1).numel()
                if slice_len != (end - start):
                    print(
                        "[MiniZero_1] gather mismatch:",
                        f"buffer_len={buffer_len}",
                        f"slice_len={slice_len}",
                        f"weight_shape={tuple(weight_data.shape)}",
                        f"offset=({start},{end})",
                    )
                weight_data.view(-1).data.copy_(self.weight_buffer[start:end])
                continue
            start = self.offsets[idx]
            end = start + weight.numel()
            if isinstance(weight, QuantizedTensor):
                weight_data = _get_raw_data(weight)
            else:
                weight_data = weight
            weight_data.view(-1).data.copy_(self.weight_buffer[start:end])

        if self.manual_post_all_gather_processing:
            quantized_weights = [
                weight for weight in self.weights if isinstance(weight, QuantizedTensor)
            ]
            post_all_gather_processing(quantized_weights)


class MiniFSDP:
    def __init__(self, weights, lr, dp_group, manual_post_all_gather_processing=False):
        rank = dist.get_rank(dp_group)
        world_size = dist.get_world_size(dp_group)

        self.weights = weights
        self.lr = lr
        self.dp_group = dp_group
        self.manual_post_all_gather_processing = manual_post_all_gather_processing

        # Flatten the weights and pad to align with world size
        if isinstance(weights[0], QuantizedTensor):
            raw_data_list = [_get_raw_data(w).view(-1) for w in weights]
        else:
            raw_data_list = [w.view(-1) for w in weights]
        self.flatten_weight, original_length = self._flatten_tensors_with_pad(raw_data_list)

        # Split flattened weights into shards
        self.local_weight_shard = torch.chunk(self.flatten_weight, world_size)[rank]
        self.local_main_grad_shard = torch.zeros_like(
            self.local_weight_shard, dtype=torch.float32, device="cuda"
        )
        shard_size = self.flatten_weight.size(0) // world_size

        # Map original tensors to flattened indices
        tensor_indices = []
        cumulative_length = 0
        for tensor in raw_data_list:
            length = tensor.size(0)
            tensor_indices.append((cumulative_length, cumulative_length + length))
            cumulative_length += length

        # Build shard index mappings
        self.weight_indices = []
        self.shard_indices = []
        for idx, (start, end) in enumerate(tensor_indices):
            shard_start = rank * shard_size
            shard_end = shard_start + shard_size
            adjusted_end = min(shard_end, original_length)

            if start <= adjusted_end and end >= shard_start:
                start_idx = max(start, shard_start)
                end_idx = min(end, adjusted_end)
                self.weight_indices.append((start_idx - start, end_idx - start))
                self.shard_indices.append((start_idx - shard_start, end_idx - shard_start))
            else:
                self.weight_indices.append((None, None))
                self.shard_indices.append((None, None))

            if isinstance(weights[idx], QuantizedTensor):
                replace_raw_data(
                    weights[idx], self.flatten_weight[start:end].view(weights[idx].shape)
                )
            else:
                weights[idx].data = self.flatten_weight[start:end].view(weights[idx].shape)

        # Initialize local model weights and high-precision master weights
        self.local_weights = []
        self.master_weights = []
        for i, weight in enumerate(self.weights):
            weight_start, weight_end = self.weight_indices[i]
            shard_start, shard_end = self.shard_indices[i]
            if shard_start is not None and shard_end is not None:
                local_weight_shard = self.local_weight_shard[shard_start:shard_end]
                self.local_weights.append(local_weight_shard)

                if isinstance(weight, QuantizedTensor):
                    high_precision_init_val = weight.get_high_precision_init_val().view(-1)
                    master_weight_shard = high_precision_init_val.to(weight.device).float()[
                        weight_start:weight_end
                    ]
                else:
                    master_weight_shard = weight.detach().view(-1).float()[weight_start:weight_end]
                self.master_weights.append(master_weight_shard)
            else:
                self.local_weights.append(None)
                self.master_weights.append(None)
            setattr(
                weight, "main_grad", torch.zeros_like(weight, dtype=torch.float32, device="cuda")
            )

    def _flatten_tensors_with_pad(self, tensors):
        """
        Flatten the list of tensors and pad them to align with the world size.

        Args:
            tensors (list): List of tensors to flatten.

        Returns:
            tuple: Flattened tensor and its original length before padding.
        """
        world_size = dist.get_world_size(self.dp_group)

        flatten_tensor = torch.cat(tensors)
        original_length = flatten_tensor.size(0)

        padding_needed = (world_size - original_length % world_size) % world_size
        if padding_needed > 0:
            zeros = torch.zeros(padding_needed, dtype=flatten_tensor.dtype, device="cuda")
            flatten_tensor = torch.cat([flatten_tensor, zeros])

        return flatten_tensor, original_length

    def zero_grad(self):
        for weight in self.weights:
            weight.grad = None
            weight.main_grad.zero_()

    def step(self):
        """
        Perform an optimization step for the distributed sharded model.

        This method includes:
        1. Gradient reduce-scatter: Synchronize gradients across all processes.
        2. Master weight update: Update high-precision master weights using local gradients.
        3. Precision casting: Cast updated master weights to FP8 or BF16 precision.
        4. Weight synchronization: All-gather updated weights across all processes.

        Returns:
            None
        """
        # Step 1: Reduce-scatter the gradients
        main_grad_buffer, _ = self._flatten_tensors_with_pad(
            [weight.main_grad.view(-1) for weight in self.weights]
        )
        dist.reduce_scatter_tensor(
            self.local_main_grad_shard, main_grad_buffer, group=self.dp_group
        )
        self.local_main_grad_shard /= dist.get_world_size(self.dp_group)

        # Step 2: Update the master weights
        for weight, master_weight, (shard_start, shard_end) in zip(
            self.weights, self.master_weights, self.shard_indices
        ):
            if master_weight is None:
                continue

            # Extract the local gradient shard for this weight
            grad = self.local_main_grad_shard[shard_start:shard_end]

            # Update the master weight using gradient descent
            master_weight -= grad * self.lr

        # Step 3: Cast master weights to quantized or BF16 precision
        first_weight = self.weights[0]
        if isinstance(first_weight, NVFP4Tensor):
            local_weights = []
            for local_weight in self.local_weights:
                if local_weight is None:
                    local_weights.append(None)
                    continue
                local_weights.append(local_weight)
            cast_master_weights_to_nvfp4(
                self.weights,
                self.master_weights,
                [idx[0] for idx in self.weight_indices],
                self.dp_group,
                local_weights,
            )
        elif isinstance(first_weight, QuantizedTensor):
            local_weights = []
            for local_weight in self.local_weights:
                if local_weight is None:
                    local_weights.append(None)
                    continue

                local_weights.append(local_weight)

            cast_master_weights_to_fp8(
                self.weights,
                self.master_weights,
                [idx[0] for idx in self.weight_indices],
                self.dp_group,
                local_weights,
                manual_post_all_gather_processing=self.manual_post_all_gather_processing,
            )
        else:
            for weight, master_weight in zip(self.local_weights, self.master_weights):
                if master_weight is None:
                    continue

                # Copy updated master weights to local weights
                weight.data.copy_(master_weight)

        # Step 4: All-gather updated weights across processes
        dist.all_gather_into_tensor(
            self.flatten_weight, self.local_weight_shard, group=self.dp_group
        )

        if self.manual_post_all_gather_processing:
            quantized_weights = [
                weight for weight in self.weights if isinstance(weight, QuantizedTensor)
            ]
            post_all_gather_processing(quantized_weights)


def _test_mini_optimizer(dp_group):
    """Make sure the implementation of MiniZero_1 and MiniFSDP is correct"""
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
    weights_3 = [weight.clone() for weight in weights]

    lr = 1.0
    optimizer_1 = MiniZero_1(weights_1, lr, dp_group)
    optimizer_2 = MiniOptimizer(weights_2, lr, dp_group)
    optimizer_3 = MiniFSDP(weights_3, lr, dp_group)

    for _ in range(100):
        for w1, w2, w3 in zip(weights_1, weights_2, weights_3):
            main_grads = [
                torch.randn_like(w1, dtype=torch.float32, device="cuda") for _ in range(world_size)
            ]
            # Choose based on rank to make sure the grads of different ranks are different.
            main_grad = main_grads[rank]
            w1.main_grad = main_grad
            w2.main_grad = main_grad
            w3.main_grad = main_grad

        optimizer_1.step()
        optimizer_2.step()
        optimizer_3.step()

        for w1, w2 in zip(weights_1, weights_2):
            torch.testing.assert_close(w1, w2, atol=0, rtol=0)
        for w1, w3 in zip(weights_1, weights_3):
            torch.testing.assert_close(w1, w3, atol=0, rtol=0)


def _test_cast_master_weights_to_fp8(quantization, dp_group, manual_post_all_gather_processing):
    rank = dist.get_rank(dp_group)
    world_size = dist.get_world_size(dp_group)

    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    mock_groups = [dist.new_group(ranks=[i]) for i in range(world_size)]
    mock_group = mock_groups[rank]

    linear_kwargs = {"params_dtype": torch.bfloat16, "bias": False, "fuse_wgrad_accumulation": True}

    # Create model with FP8 weights
    with te.quantized_model_init(
        enabled=quantization is not None,
        recipe=_get_quantization_recipe(quantization),
        preserve_high_precision_init_val=True,
    ):
        model_fp8 = nn.Sequential(
            te.Linear(128, 256 + 16, **linear_kwargs),
            te.Linear(256 + 16, 256 * 3, **linear_kwargs),
            te.Linear(256 * 3, 128, **linear_kwargs),
        )

    # Create model with BF16 weights
    model = nn.Sequential(
        te.Linear(128, 256 + 16, **linear_kwargs),
        te.Linear(256 + 16, 256 * 3, **linear_kwargs),
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

    optimizer_fp8 = MiniZero_1(
        [w for w in model_fp8.parameters()], 10.0, dp_group, manual_post_all_gather_processing
    )
    optimizer = MiniZero_1([w for w in model.parameters()], 10.0, dp_group)

    for i in range(100):
        for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
            w_fp8.main_grad.zero_()
            w.main_grad.zero_()

        inputs = [
            torch.randn(16, 128, dtype=torch.bfloat16, device="cuda") for _ in range(world_size)
        ]
        # Choose based on rank to make sure the inputs of different ranks are different.
        x = inputs[rank]

        with te.autocast(
            enabled=quantization is not None,
            recipe=_get_quantization_recipe(quantization),
            amax_reduction_group=mock_group,
        ):
            y_fp8 = model_fp8(x)

        with te.autocast(
            enabled=quantization is not None,
            recipe=_get_quantization_recipe(quantization),
            amax_reduction_group=mock_group,
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


def _test_fsdp_cast_master_weights_to_fp8(
    quantization, dp_group, manual_post_all_gather_processing
):
    rank = dist.get_rank(dp_group)
    world_size = dist.get_world_size(dp_group)

    # Configuration constants
    NUM_STEPS = 100
    SEED = 12345

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    mock_groups = [dist.new_group(ranks=[i]) for i in range(world_size)]
    mock_group = mock_groups[rank]

    linear_kwargs = {
        "params_dtype": torch.bfloat16,
        "bias": False,
        "fuse_wgrad_accumulation": True,
    }

    # Create model with FP8 weights
    with te.quantized_model_init(
        enabled=quantization is not None,
        recipe=_get_quantization_recipe(quantization),
        preserve_high_precision_init_val=True,
    ):
        model_fp8 = nn.Sequential(
            te.Linear(128, 256 + 16, **linear_kwargs),
            te.Linear(256 + 16, 256 * 3, **linear_kwargs),
            te.Linear(256 * 3, 128, **linear_kwargs),
        )

    # Create model with BF16 weights
    model = nn.Sequential(
        te.Linear(128, 256 + 16, **linear_kwargs),
        te.Linear(256 + 16, 256 * 3, **linear_kwargs),
        te.Linear(256 * 3, 128, **linear_kwargs),
    )

    # Make sure the BF16 model and FP8 model have the same initial weights
    for w_fp8, w in zip(model_fp8.parameters(), model.parameters()):
        high_precision_init_val = w_fp8.get_high_precision_init_val()
        w.data.copy_(high_precision_init_val)

    optimizer_fp8 = MiniFSDP(
        [w for w in model_fp8.parameters()], 10.0, dp_group, manual_post_all_gather_processing
    )
    optimizer = MiniFSDP([w for w in model.parameters()], 10.0, dp_group)

    for _ in range(100):
        optimizer_fp8.zero_grad()
        optimizer.zero_grad()

        inputs = [
            torch.randn(16, 128, dtype=torch.bfloat16, device="cuda") for _ in range(world_size)
        ]
        # Choose based on rank to make sure the inputs of different ranks are different.
        x = inputs[rank]

        with te.autocast(
            enabled=quantization is not None,
            recipe=_get_quantization_recipe(quantization),
            amax_reduction_group=mock_group,
        ):
            y_fp8 = model_fp8(x)

        with te.autocast(
            enabled=quantization is not None,
            recipe=_get_quantization_recipe(quantization),
            amax_reduction_group=mock_group,
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


def _test_cast_master_weights_to_nvfp4(dp_group, manual_post_all_gather_processing):
    available, reason = is_nvfp4_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    rank = dist.get_rank(dp_group)
    world_size = dist.get_world_size(dp_group)

    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    mock_groups = [dist.new_group(ranks=[i]) for i in range(world_size)]
    mock_group = mock_groups[rank]

    linear_kwargs = {"params_dtype": torch.bfloat16, "bias": False, "fuse_wgrad_accumulation": True}
    # Disable stochastic rounding for deterministic gradients
    nvfp4_recipe = NVFP4BlockScaling(disable_stochastic_rounding=True)

    # Original shapes (commented out for debugging padding issues):
    # with te.quantized_model_init(
    #     enabled=True, recipe=nvfp4_recipe, preserve_high_precision_init_val=True
    # ):
    #     model_nvfp4 = nn.Sequential(
    #         te.Linear(128, 256, **linear_kwargs),
    #         te.Linear(256, 256 * 3, **linear_kwargs),
    #         te.Linear(256 * 3, 128, **linear_kwargs),
    #     )
    # model = nn.Sequential(
    #     te.Linear(128, 256, **linear_kwargs),
    #     te.Linear(256, 256 * 3, **linear_kwargs),
    #     te.Linear(256 * 3, 128, **linear_kwargs),
    # )

    # Use 2048x2048 weights to avoid NVFP4 scale_inv padding issues
    with te.quantized_model_init(
        enabled=True, recipe=nvfp4_recipe, preserve_high_precision_init_val=True
    ):
        model_nvfp4 = nn.Sequential(
            te.Linear(2048, 2048, **linear_kwargs),
            te.Linear(2048, 2048, **linear_kwargs),
            te.Linear(2048, 2048, **linear_kwargs),
        )

    # BF16 model (created outside quantized_model_init)
    model = nn.Sequential(
        te.Linear(2048, 2048, **linear_kwargs),
        te.Linear(2048, 2048, **linear_kwargs),
        te.Linear(2048, 2048, **linear_kwargs),
    )

    for w_nvfp4, w in zip(model_nvfp4.parameters(), model.parameters()):
        high_precision_init_val = w_nvfp4.get_high_precision_init_val()
        w.data.copy_(high_precision_init_val)

    for w_nvfp4, w in zip(model_nvfp4.parameters(), model.parameters()):
        w_nvfp4.main_grad = torch.zeros_like(w_nvfp4, dtype=torch.float32, device="cuda")
        w.main_grad = torch.zeros_like(w, dtype=torch.float32, device="cuda")

    optimizer_nvfp4 = MiniZero_1(
        [w for w in model_nvfp4.parameters()], 10.0, dp_group, manual_post_all_gather_processing
    )
    optimizer = MiniZero_1([w for w in model.parameters()], 10.0, dp_group)

    # Add hooks to capture intermediate activations
    activations_nvfp4 = {}
    activations_bf16 = {}

    def make_hook(storage, name):
        def hook(module, input, output):
            storage[name] = (input[0].clone(), output.clone())
        return hook

    hooks = []
    for idx, (layer_nvfp4, layer_bf16) in enumerate(zip(model_nvfp4, model)):
        hooks.append(layer_nvfp4.register_forward_hook(make_hook(activations_nvfp4, f"layer_{idx}")))
        hooks.append(layer_bf16.register_forward_hook(make_hook(activations_bf16, f"layer_{idx}")))

    for i in range(100):
        for w_nvfp4, w in zip(model_nvfp4.parameters(), model.parameters()):
            w_nvfp4.main_grad.zero_()
            w.main_grad.zero_()

        # Original input shape: torch.randn(128, 128, ...)
        inputs = [
            torch.randn(2048, 2048, dtype=torch.bfloat16, device="cuda") for _ in range(world_size)
        ]
        x = inputs[rank]

        # Debug: compare master weights before forward
        if i in [0, 1, 7, 8] and rank == 0:
            print(f"\n=== Debug iteration {i} ===")
            for idx, (w_nvfp4, w) in enumerate(zip(model_nvfp4.parameters(), model.parameters())):
                # Compare master weights
                master_nvfp4 = optimizer_nvfp4.master_weights[idx]
                master_bf16 = optimizer.master_weights[idx]
                if master_nvfp4 is None or master_bf16 is None:
                    print(f"Layer {idx}: master weights = None (nvfp4={master_nvfp4 is not None}, bf16={master_bf16 is not None})")
                    continue
                print(f"Layer {idx}: master_nvfp4.dtype={master_nvfp4.dtype}, master_bf16.dtype={master_bf16.dtype}")
                print(f"Layer {idx}: w_nvfp4.dtype={w_nvfp4.dtype}, w.dtype={w.dtype}")
                master_match = torch.equal(master_nvfp4, master_bf16)
                print(f"Layer {idx}: master weights match = {master_match}")
                if not master_match:
                    diff = (master_nvfp4 - master_bf16).abs().max().item()
                    print(f"  max diff = {diff}")

        with te.autocast(
            enabled=True,
            recipe=nvfp4_recipe,
            amax_reduction_group=mock_group,
        ):
            y_nvfp4 = model_nvfp4(x)

        with te.autocast(
            enabled=True,
            recipe=nvfp4_recipe,
            amax_reduction_group=mock_group,
        ):
            y = model(x)

        # Debug: compare forward outputs and weight properties
        if i == 0 and rank == 0:
            print(f"\n=== Forward outputs iteration {i} ===")
            print(f"y_nvfp4 shape: {y_nvfp4.shape}, y shape: {y.shape}")
            y_match = torch.equal(y_nvfp4, y)
            print(f"Forward outputs match: {y_match}")
            if not y_match:
                diff = (y_nvfp4 - y).abs().max().item()
                print(f"  max diff: {diff}")

            # Compare intermediate activations
            print("\n=== Intermediate activations ===")
            for layer_name in activations_nvfp4.keys():
                inp_nvfp4, out_nvfp4 = activations_nvfp4[layer_name]
                inp_bf16, out_bf16 = activations_bf16[layer_name]
                inp_match = torch.equal(inp_nvfp4, inp_bf16)
                out_match = torch.equal(out_nvfp4, out_bf16)
                print(f"{layer_name}: input match = {inp_match}, output match = {out_match}")
                if not inp_match:
                    diff = (inp_nvfp4 - inp_bf16).abs().max().item()
                    print(f"  input max diff: {diff}")
                if not out_match:
                    diff = (out_nvfp4 - out_bf16).abs().max().item()
                    print(f"  output max diff: {diff}")
            
            # Compare quantizer states
            print("\n=== Quantizer comparison ===")
            for idx, (layer_nvfp4, layer_bf16) in enumerate(zip(model_nvfp4, model)):
                print(f"Layer {idx}:")
                print(f"  nvfp4.fp8: {layer_nvfp4.fp8}, bf16.fp8: {layer_bf16.fp8}")
                print(f"  nvfp4.fp8_initialized: {layer_nvfp4.fp8_initialized}, bf16.fp8_initialized: {layer_bf16.fp8_initialized}")
                if hasattr(layer_nvfp4, 'quantizers') and hasattr(layer_bf16, 'quantizers'):
                    q_nvfp4 = layer_nvfp4.quantizers.get('scaling_fwd', [])
                    q_bf16 = layer_bf16.quantizers.get('scaling_fwd', [])
                    if q_nvfp4:
                        print(f"  nvfp4 input quantizer type: {type(q_nvfp4[0])}")
                    if q_bf16:
                        print(f"  bf16 input quantizer type: {type(q_bf16[0])}")
            
            # Compare NVFP4 tensor properties
            print("\n=== NVFP4 weight properties ===")
            for idx, (w_nvfp4, w) in enumerate(zip(model_nvfp4.parameters(), model.parameters())):
                print(f"Layer {idx}:")
                print(f"  w_nvfp4 type: {type(w_nvfp4).__name__}, w type: {type(w).__name__}")
                if hasattr(w_nvfp4, '_amax_rowwise') and w_nvfp4._amax_rowwise is not None:
                    print(f"  w_nvfp4._amax_rowwise: {w_nvfp4._amax_rowwise.item()}")
                if hasattr(w_nvfp4, '_amax_columnwise') and w_nvfp4._amax_columnwise is not None:
                    print(f"  w_nvfp4._amax_columnwise: {w_nvfp4._amax_columnwise.item()}")
                # Compare dequantized values
                if hasattr(w_nvfp4, 'dequantize'):
                    w_nvfp4_dequant = w_nvfp4.dequantize(dtype=torch.bfloat16)
                    dequant_match = torch.equal(w_nvfp4_dequant, w)
                    print(f"  dequant(w_nvfp4) == w: {dequant_match}")
                    if not dequant_match:
                        diff = (w_nvfp4_dequant - w).abs().max().item()
                        print(f"    max diff: {diff}")

        targets = [torch.randn_like(y) for _ in range(world_size)]
        target = targets[rank]
        loss_nvfp4 = nn.MSELoss()(y_nvfp4, target)
        loss = nn.MSELoss()(y, target)

        # Debug: check if losses are identical
        if i == 0 and rank == 0:
            print(f"\n=== Loss comparison iteration {i} ===")
            print(f"loss_nvfp4: {loss_nvfp4.item()}, loss: {loss.item()}")
            print(f"Losses bitwise equal: {torch.equal(loss_nvfp4, loss)}")

        loss_nvfp4.backward()
        loss.backward()

        # Debug: compare gradients before optimizer step
        if i == 0 and rank == 0:
            print(f"\n=== Gradients before step iteration {i} ===")
            for idx, (w_nvfp4, w) in enumerate(zip(model_nvfp4.parameters(), model.parameters())):
                grad_nvfp4 = w_nvfp4.main_grad
                grad_bf16 = w.main_grad
                grad_match = torch.equal(grad_nvfp4, grad_bf16)
                print(f"Layer {idx}: gradients match = {grad_match}")
                if not grad_match:
                    diff = (grad_nvfp4 - grad_bf16).abs().max().item()
                    print(f"  max diff: {diff}")

            # Test: run same model twice to check for non-determinism
            print("\n=== Determinism test: run model_nvfp4 twice ===")
            for w_nvfp4 in model_nvfp4.parameters():
                w_nvfp4.main_grad.zero_()
            with te.autocast(enabled=True, recipe=nvfp4_recipe, amax_reduction_group=mock_group):
                y_test1 = model_nvfp4(x)
            loss_test1 = nn.MSELoss()(y_test1, target)
            loss_test1.backward()
            grads_run1 = [w.main_grad.clone() for w in model_nvfp4.parameters()]
            
            for w_nvfp4 in model_nvfp4.parameters():
                w_nvfp4.main_grad.zero_()
            with te.autocast(enabled=True, recipe=nvfp4_recipe, amax_reduction_group=mock_group):
                y_test2 = model_nvfp4(x)
            loss_test2 = nn.MSELoss()(y_test2, target)
            loss_test2.backward()
            grads_run2 = [w.main_grad.clone() for w in model_nvfp4.parameters()]
            
            for idx, (g1, g2) in enumerate(zip(grads_run1, grads_run2)):
                match = torch.equal(g1, g2)
                print(f"Layer {idx}: same model, 2 runs match = {match}")
                if not match:
                    diff = (g1 - g2).abs().max().item()
                    print(f"  max diff: {diff}")

        optimizer.step()
        optimizer_nvfp4.step()

        # Debug: compare weights after optimizer step (on all ranks)
        if i == 0:
            print(f"\n=== After optimizer step iteration {i} (rank {rank}) ===")
            for idx, (w_nvfp4, w) in enumerate(zip(model_nvfp4.parameters(), model.parameters())):
                # Compare master weights
                master_nvfp4 = optimizer_nvfp4.master_weights[idx]
                master_bf16 = optimizer.master_weights[idx]
                if master_nvfp4 is not None and master_bf16 is not None:
                    master_match = torch.equal(master_nvfp4, master_bf16)
                    print(f"Layer {idx}: master weights match = {master_match}")
                    if not master_match:
                        diff = (master_nvfp4 - master_bf16).abs().max().item()
                        print(f"  max diff: {diff}")
                else:
                    print(f"Layer {idx}: master weights = None")
                
                # Compare model weights: quantize BF16 and compare with NVFP4
                if hasattr(w_nvfp4, '_rowwise_data') and hasattr(w_nvfp4, '_quantizer'):
                    # Create a fresh quantizer with same config to avoid state issues
                    from transformer_engine.pytorch.tensor import NVFP4Quantizer
                    fresh_quantizer = NVFP4Quantizer(with_2d_quantization=True)
                    w_bf16_quantized = fresh_quantizer(w)
                    
                    # Debug: compare amax for layer 2
                    if idx == 2 and i == 0:
                        print(f"[Rank {rank}] Layer {idx} DEBUG:")
                        # Compare BF16 values: NVFP4 model's master weight vs BF16 model's weight
                        master_nvfp4 = optimizer_nvfp4.master_weights[idx]
                        master_bf16 = optimizer.master_weights[idx]
                        if master_nvfp4 is not None and master_bf16 is not None:
                            # Both ranks should have the master weight if they own this layer
                            bf16_from_master_nvfp4 = master_nvfp4.to(w_nvfp4.dtype).view(w.shape)
                            print(f"[Rank {rank}]   BF16 weight w == master_bf16.to(bf16): {torch.equal(w, master_bf16.to(w.dtype).view(w.shape))}")
                            print(f"[Rank {rank}]   BF16 weight w == master_nvfp4.to(bf16): {torch.equal(w, bf16_from_master_nvfp4)}")
                        else:
                            print(f"[Rank {rank}]   master_nvfp4={master_nvfp4 is not None}, master_bf16={master_bf16 is not None}")
                        print(f"[Rank {rank}]   w_nvfp4._amax_rowwise: {w_nvfp4._amax_rowwise}")
                        print(f"[Rank {rank}]   w_bf16_quantized._amax_rowwise: {w_bf16_quantized._amax_rowwise}")
                        # Sample of scales
                        print(f"[Rank {rank}]   w_nvfp4._rowwise_scale_inv[0,:8]: {w_nvfp4._rowwise_scale_inv[0,:8].tolist()}")
                        print(f"[Rank {rank}]   w_bf16_quantized._rowwise_scale_inv[0,:8]: {w_bf16_quantized._rowwise_scale_inv[0,:8].tolist()}")
                        # Check where scales differ
                        scale_diff = (w_nvfp4._rowwise_scale_inv != w_bf16_quantized._rowwise_scale_inv)
                        if scale_diff.any():
                            diff_indices = torch.nonzero(scale_diff, as_tuple=True)
                            print(f"[Rank {rank}]   First 5 scale diff positions: {list(zip(diff_indices[0][:5].tolist(), diff_indices[1][:5].tolist()))}")
                            for r, c in zip(diff_indices[0][:5].tolist(), diff_indices[1][:5].tolist()):
                                print(f"[Rank {rank}]     [{r},{c}]: nvfp4={w_nvfp4._rowwise_scale_inv[r,c].item()}, ref={w_bf16_quantized._rowwise_scale_inv[r,c].item()}")
                    
                    # Compare raw NVFP4 data
                    data_match = torch.equal(w_nvfp4._rowwise_data, w_bf16_quantized._rowwise_data)
                    print(f"Layer {idx}: _rowwise_data match = {data_match}")
                    if not data_match:
                        # Count mismatches
                        mismatches = (w_nvfp4._rowwise_data != w_bf16_quantized._rowwise_data).sum().item()
                        total = w_nvfp4._rowwise_data.numel()
                        print(f"  mismatches: {mismatches}/{total} ({100*mismatches/total:.2f}%)")
                    
                    # Compare scales
                    scale_match = torch.equal(w_nvfp4._rowwise_scale_inv, w_bf16_quantized._rowwise_scale_inv)
                    print(f"Layer {idx}: _rowwise_scale_inv match = {scale_match}")
                    if not scale_match:
                        mismatches = (w_nvfp4._rowwise_scale_inv != w_bf16_quantized._rowwise_scale_inv).sum().item()
                        total = w_nvfp4._rowwise_scale_inv.numel()
                        print(f"  mismatches: {mismatches}/{total} ({100*mismatches/total:.2f}%)")
        
        torch.testing.assert_close(loss_nvfp4, loss, atol=0, rtol=0)
        print("iter:", i, "loss matched")   


def run_parallel_tests() -> None:
    """Run parallel tests"""

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
    dp_group = dist.new_group(backend="nccl")

    quantizations = []
    if is_fp8_available():
        print("fp8 available")
        quantizations.extend(["fp8", "fp8_cs"])
    if is_fp8_block_scaling_available():
        quantizations.append("fp8_block")
    manual_post_all_gather_processings = [False, True]
    print("starting mini optimizer test")
    _test_mini_optimizer(dp_group)
    print("starting cast master weights to fp8 test")
    for quantization in quantizations:
        for post_ag_processing in manual_post_all_gather_processings:
            _test_cast_master_weights_to_fp8(quantization, dp_group, post_ag_processing)
            _test_fsdp_cast_master_weights_to_fp8(quantization, dp_group, post_ag_processing)
    print("starting cast master weights to nvfp4 test")
    nvfp4_available, _ = is_nvfp4_available(return_reason=True)
    if nvfp4_available:
        #for post_ag_processing in manual_post_all_gather_processings:
        _test_cast_master_weights_to_nvfp4(dp_group, False)

    dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="cast_master_weights_to_fp8 test needs at least 2 GPUs."
)
@pytest.mark.parametrize("world_size", [2])
def test_cast_master_weights_to_fp8(world_size: int) -> None:
    """Launch parallel job that runs parallel tests"""
    python_exe = pathlib.Path(sys.executable).resolve()
    current_file = pathlib.Path(__file__).resolve()
    command = [
        python_exe,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={world_size}",
        current_file,
        "--parallel",
    ]
    result = subprocess.run(
        command,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel", action="store_true", help="Run parallel tests")
    args = parser.parse_args()
    if args.parallel:
        run_parallel_tests()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVFP4 transpose test requires CUDA."
)
def test_nvfp4_transpose_kernel() -> None:
    """Test that nvfp4_transpose kernel produces bitwise identical results to reference."""
    available, reason = is_nvfp4_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    torch.manual_seed(1234)
    device = torch.device("cuda")
    shape = (2048, 64)
    master_weight = torch.randn(shape, dtype=torch.float32, device=device)

    print("\n=== Testing NVFP4 transpose kernel ===")

    # Create reference with both rowwise and columnwise data
    quantizer_with_colwise = NVFP4Quantizer(
        rowwise=True, columnwise=True, with_2d_quantization=True
    )
    reference_tensor = quantizer_with_colwise(master_weight.to(torch.bfloat16))
    assert reference_tensor._columnwise_data is not None, "Reference should have columnwise data"
    assert reference_tensor._columnwise_scale_inv is not None, "Reference should have columnwise scale_inv"
    reference_columnwise_data = reference_tensor._columnwise_data.detach().clone()
    reference_columnwise_scale_inv = reference_tensor._columnwise_scale_inv.detach().clone()
    reference_columnwise_amax = reference_tensor._amax_columnwise.detach().clone() if reference_tensor._amax_columnwise is not None else None
    print(
        "reference columnwise_data shape:",
        reference_columnwise_data.shape,
    )
    print(
        "reference columnwise_scale_inv shape:",
        reference_columnwise_scale_inv.shape,
    )

    # Create tensor with only rowwise data, then call _create_columnwise()
    quantizer_rowwise_only = NVFP4Quantizer(
        rowwise=True, columnwise=False, with_2d_quantization=True
    )
    test_tensor = quantizer_rowwise_only(master_weight.to(torch.bfloat16))
    assert test_tensor._columnwise_data is None, "Test tensor should not have columnwise data yet"

    # Now call _create_columnwise() which uses our nvfp4_transpose kernel
    test_tensor.update_usage(rowwise_usage=True, columnwise_usage=True)
    assert test_tensor._columnwise_data is not None, "Test tensor should have columnwise data after _create_columnwise()"
    assert test_tensor._columnwise_scale_inv is not None, "Test tensor should have columnwise scale_inv after _create_columnwise()"
    print(
        "test_tensor columnwise_data shape after transpose:",
        test_tensor._columnwise_data.shape,
    )
    print(
        "test_tensor columnwise_scale_inv shape after transpose:",
        test_tensor._columnwise_scale_inv.shape,
    )

    # Compare columnwise data - should be bitwise identical
    torch.testing.assert_close(
        test_tensor._columnwise_data,
        reference_columnwise_data,
        atol=0,
        rtol=0,
        msg="NVFP4 transpose kernel produced different columnwise data than reference!",
    )
    print("columnwise_data matches!")

    # Compare columnwise scale_inv - should be bitwise identical
    print("reference columnwise_scale_inv:\n", reference_columnwise_scale_inv)
    print("test columnwise_scale_inv:\n", test_tensor._columnwise_scale_inv)
    print("reference rowwise_scale_inv shape:", reference_tensor._rowwise_scale_inv.shape)
    print("test rowwise_scale_inv shape:", test_tensor._rowwise_scale_inv.shape)
    
    # Check if they match
    scale_match = torch.equal(test_tensor._columnwise_scale_inv, reference_columnwise_scale_inv)
    if not scale_match:
        diff_mask = test_tensor._columnwise_scale_inv != reference_columnwise_scale_inv
        print("Number of mismatches:", diff_mask.sum().item())
        print("Mismatch locations:", diff_mask.nonzero()[:10])
        print("Test values at mismatch:", test_tensor._columnwise_scale_inv[diff_mask][:10])
        print("Reference values at mismatch:", reference_columnwise_scale_inv[diff_mask][:10])
    
    torch.testing.assert_close(
        test_tensor._columnwise_scale_inv,
        reference_columnwise_scale_inv,
        atol=0,
        rtol=0,
        msg="NVFP4 _create_columnwise produced different columnwise scale_inv than reference!",
    )
    print("columnwise_scale_inv matches!")

    # Compare columnwise amax if available
    if reference_columnwise_amax is not None:
        torch.testing.assert_close(
            test_tensor._amax_columnwise,
            reference_columnwise_amax,
            atol=0,
            rtol=0,
            msg="NVFP4 _create_columnwise produced different columnwise amax than reference!",
        )
        print("columnwise_amax matches!")

    print("NVFP4 transpose kernel test PASSED!")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NVFP4 partial-cast test requires CUDA."
)
def test_nvfp4_partial_cast_matches_full() -> None:
    """Test multi-GPU partial cast: split master weight, partial cast on each rank, all-gather, compare."""
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
    dp_group = dist.new_group(backend="nccl")
    available, reason = is_nvfp4_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    torch.manual_seed(1234)
    device = torch.device("cuda")
    # Shape must be divisible by WORLD_SIZE for even splitting
    # Also ensure dimensions are multiples of 16 for NVFP4 tiles
    shape = (4096, 2048)
    total_elements = shape[0] * shape[1]
    assert total_elements % WORLD_SIZE == 0, "Total elements must be divisible by WORLD_SIZE"

    # Full master weight (same on all ranks due to same seed)
    full_master_weight = torch.randn(shape, dtype=torch.float32, device=device)

    # Create reference using full quantization
    quantizer = NVFP4Quantizer(rowwise=True, columnwise=False, with_2d_quantization=True)
    reference_tensor = quantizer(full_master_weight.to(torch.bfloat16))
    reference_data = reference_tensor._rowwise_data.detach().clone()
    reference_scale = reference_tensor._rowwise_scale_inv.detach().clone()
    reference_amax = reference_tensor._amax_rowwise.detach().clone()
    print(f"[Rank {WORLD_RANK}] reference_data shape: {reference_data.shape}")
    print(f"[Rank {WORLD_RANK}] reference_scale shape: {reference_scale.shape}")

    # Split master weight evenly across ranks
    shard_size = total_elements // WORLD_SIZE
    start_offset = WORLD_RANK * shard_size
    end_offset = start_offset + shard_size
    master_weight_shard = full_master_weight.view(-1)[start_offset:end_offset].clone()
    print(f"[Rank {WORLD_RANK}] shard: start_offset={start_offset}, end_offset={end_offset}, shard_size={shard_size}")

    # Create empty NVFP4 tensor for this rank (full shape, but we'll only fill our shard)
    nvfp4_tensor = quantizer.make_empty(shape, dtype=torch.bfloat16, device=device)
    nvfp4_tensor._rowwise_data.zero_()
    nvfp4_tensor._rowwise_scale_inv.zero_()
    if nvfp4_tensor._amax_rowwise is not None:
        nvfp4_tensor._amax_rowwise.zero_()

    # Partial cast on each rank's shard
    cast_master_weights_to_nvfp4(
        [nvfp4_tensor],
        [master_weight_shard],
        [start_offset],
        dp_group,
    )

    # All-gather the rowwise data (packed FP4 bytes)
    # Each rank has the full tensor but only its shard is filled
    # We need to all-gather the shards
    rowwise_data_flat = nvfp4_tensor._rowwise_data.view(-1)
    
    # For NVFP4, 2 elements are packed per byte, so byte shard size is shard_size // 2
    byte_shard_size = shard_size // 2
    byte_start = WORLD_RANK * byte_shard_size
    byte_end = byte_start + byte_shard_size
    my_shard_bytes = rowwise_data_flat[byte_start:byte_end].contiguous()
    
    # Gather all shards
    gathered_shards = [torch.empty_like(my_shard_bytes) for _ in range(WORLD_SIZE)]
    dist.all_gather(gathered_shards, my_shard_bytes, group=dp_group)
    
    # Reconstruct the full rowwise data
    gathered_data = torch.cat(gathered_shards, dim=0).view(reference_data.shape)
    print(f"[Rank {WORLD_RANK}] gathered_data shape: {gathered_data.shape}")

    # Compare with reference
    torch.testing.assert_close(
        gathered_data,
        reference_data,
        atol=0,
        rtol=0,
        msg=f"[Rank {WORLD_RANK}] Gathered rowwise data does not match reference!",
    )
    print(f"[Rank {WORLD_RANK}] rowwise_data matches reference!")

    # Also verify scale matches (scale should be identical on all ranks after all-reduce)
    torch.testing.assert_close(
        nvfp4_tensor._rowwise_scale_inv,
        reference_scale,
        atol=0,
        rtol=0,
        msg=f"[Rank {WORLD_RANK}] Scale does not match reference!",
    )
    print(f"[Rank {WORLD_RANK}] scale matches reference!")

    # Verify amax matches
    torch.testing.assert_close(
        nvfp4_tensor._amax_rowwise,
        reference_amax,
        atol=0,
        rtol=0,
        msg=f"[Rank {WORLD_RANK}] Amax does not match reference!",
    )
    print(f"[Rank {WORLD_RANK}] amax matches reference!")

    print(f"[Rank {WORLD_RANK}] Multi-GPU NVFP4 partial cast test PASSED!")


def test_single_gpu_partial_cast_vs_full():
    """
    Single GPU test: compare cast_master_weights_to_nvfp4 (offset=0) vs quantizer().
    This isolates whether the issue is in our manual Python scale computation or elsewhere.
    """
    import math
    from transformer_engine.pytorch.tensor import NVFP4Quantizer
    from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_nvfp4
    import transformer_engine_torch as tex
    
    torch.manual_seed(12345)
    device = torch.device("cuda")
    
    # Test with same shape as the optimizer test
    shape = (2048, 64)
    
    # Create BF16 master weight
    master_weight = torch.randn(shape, dtype=torch.bfloat16, device=device)
    
    # === Reference: Use NVFP4Quantizer directly ===
    quantizer = NVFP4Quantizer(rowwise=True, columnwise=False, with_2d_quantization=True)
    ref = quantizer(master_weight)
    ref_data = ref._rowwise_data.clone()
    ref_scale = ref._rowwise_scale_inv.clone()
    ref_amax = ref._amax_rowwise.clone()
    
    print(f"Reference:")
    print(f"  data shape: {ref_data.shape}")
    print(f"  scale shape: {ref_scale.shape}")
    print(f"  amax: {ref_amax}")
    
    # === Test: Use cast_master_weights_to_nvfp4 with offset=0 (full tensor) ===
    # Create empty NVFP4 tensor
    test_tensor = quantizer.make_empty(shape, dtype=torch.bfloat16, device=device)
    test_tensor._rowwise_data.zero_()
    test_tensor._rowwise_scale_inv.zero_()
    if test_tensor._amax_rowwise is not None:
        test_tensor._amax_rowwise.zero_()
    
    # Create a mock distributed group for single GPU
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1)
    mock_group = dist.new_group(ranks=[0])
    
    # Call cast_master_weights_to_nvfp4 with full tensor (offset=0)
    cast_master_weights_to_nvfp4(
        [test_tensor],
        [master_weight.view(-1)],  # Flatten as expected
        [0],  # offset=0 means full tensor
        mock_group,
    )
    
    print(f"\nTest (cast_master_weights_to_nvfp4 with offset=0):")
    print(f"  data shape: {test_tensor._rowwise_data.shape}")
    print(f"  scale shape: {test_tensor._rowwise_scale_inv.shape}")
    print(f"  amax: {test_tensor._amax_rowwise}")
    
    # === Compare ===
    print(f"\nComparison:")
    
    # Compare amax
    amax_match = torch.equal(test_tensor._amax_rowwise, ref_amax)
    print(f"  Amax match: {amax_match}")
    if not amax_match:
        print(f"    test: {test_tensor._amax_rowwise}")
        print(f"    ref:  {ref_amax}")
    
    # Compare scale
    scale_match = torch.equal(test_tensor._rowwise_scale_inv, ref_scale)
    print(f"  Scale match: {scale_match}")
    if not scale_match:
        mismatches = (test_tensor._rowwise_scale_inv != ref_scale).sum().item()
        total = ref_scale.numel()
        print(f"    Mismatches: {mismatches}/{total} ({100*mismatches/total:.4f}%)")
    
    # Compare data
    data_match = torch.equal(test_tensor._rowwise_data, ref_data)
    print(f"  Data match: {data_match}")
    if not data_match:
        mismatches = (test_tensor._rowwise_data != ref_data).sum().item()
        total = ref_data.numel()
        print(f"    Mismatches: {mismatches}/{total} ({100*mismatches/total:.4f}%)")
    
    if amax_match and scale_match and data_match:
        print("\nSUCCESS: cast_master_weights_to_nvfp4 (offset=0) matches quantizer!")
    else:
        print("\nFAILURE: Results don't match!")


def test_scale_computation_matches_quantizer():
    """
    Test that our Python scale computation in utils.py matches what NVFP4Quantizer produces.
    This isolates the scale computation issue outside of the optimizer.
    """
    import math
    from transformer_engine.pytorch.tensor import NVFP4Quantizer
    from transformer_engine.common.recipe import NVFP4BlockScaling
    import transformer_engine_torch as tex
    
    torch.manual_seed(12345)
    device = torch.device("cuda")
    
    # Test with 2048x2048 like in the optimizer test
    shape = (2048, 2048)
    block_len = 16
    
    # Create random BF16 tensor (simulating master weight converted to BF16)
    master_weight = torch.randn(shape, dtype=torch.bfloat16, device=device)
    
    # === Reference: Use NVFP4Quantizer ===
    quantizer = NVFP4Quantizer(with_2d_quantization=True)
    ref = quantizer(master_weight)
    ref_scale = ref._rowwise_scale_inv.clone()
    ref_data = ref._rowwise_data.clone()
    ref_amax = ref._amax_rowwise.clone()
    
    print(f"Reference scale shape: {ref_scale.shape}")
    print(f"Reference data shape: {ref_data.shape}")
    print(f"Reference amax: {ref_amax}")
    
    # === Our implementation: Replicate utils.py logic ===
    h, w = shape
    tile_h = math.ceil(h / block_len)
    tile_w = math.ceil(w / block_len)
    tile_shape = (tile_h, tile_w)
    
    print(f"Tile shape: {tile_shape}")
    
    # Step 1: Compute per-block amax using CUDA kernel
    amax_tensor = torch.zeros(tile_shape, dtype=torch.float32, device=device)
    global_amax = torch.zeros(1, dtype=torch.float32, device=device)
    
    tex.nvfp4_2d_compute_partial_amax(
        master_weight.view(-1), amax_tensor, h, w, 0, block_len
    )
    tex.compute_amax(master_weight.view(-1), global_amax)
    
    print(f"Computed global_amax: {global_amax.item()}")
    print(f"Reference global_amax: {ref_amax.item()}")
    print(f"Global amax match: {torch.equal(global_amax, ref_amax)}")
    
    # Step 2: Compute scales 
    fp4_max = 6.0
    fp8_max = 448.0
    finfo = torch.finfo(torch.float32)  
    tiny = finfo.tiny
    
    safe_global_amax = torch.clamp(global_amax, min=tiny)
    global_encode_scale = torch.clamp((fp8_max * fp4_max) / safe_global_amax, max=finfo.max)
    global_scale = global_encode_scale.item()
    
    print(f"global_encode_scale: {global_scale}")
    
    # per_block_decode_scale = amax / fp4_max * global_scale
    # CUDA computes: amax / 6.0 * global_scale (division first, then multiply)
    # Python was: amax * (1.0 / 6.0) * global_scale (multiply by reciprocal)
    # Try matching CUDA order:
    per_block_decode_scale_cuda_order = torch.clamp(
        (amax_tensor / fp4_max) * global_scale, max=finfo.max
    )
    per_block_decode_scale_python_order = torch.clamp(
        (amax_tensor * (1.0 / fp4_max)) * global_scale, max=finfo.max
    )
    print(f"global_scale (should be 2688/5 = 537.6): {global_scale}")
    print(f"\nComparing CUDA vs Python order:")
    print(f"  CUDA order [0,:5]: {per_block_decode_scale_cuda_order[0,:5].tolist()}")
    print(f"  Python order [0,:5]: {per_block_decode_scale_python_order[0,:5].tolist()}")
    print(f"  Difference: {(per_block_decode_scale_cuda_order - per_block_decode_scale_python_order).abs().max().item()}")
    
    # Use CUDA order for the rest of the test
    per_block_decode_scale = per_block_decode_scale_cuda_order
    
    print(f"per_block_decode_scale shape: {per_block_decode_scale.shape}")
    
    # Step 3: Expand to target_scale shape (replicate utils.py expansion)
    # Get the expected scale shape from quantizer (rowwise, not columnwise)
    target_scale_shape = quantizer.get_scale_shape(shape, columnwise=False)
    print(f"Expected target_scale shape: {target_scale_shape}")
    
    target_scale = torch.zeros(target_scale_shape, dtype=torch.uint8, device=device)
    expanded_scale = torch.zeros(target_scale_shape, dtype=torch.float32, device=device)
    
    tile_rows = tile_h
    tile_col_cnt = tile_w
    rows = h
    chunk = block_len
    
    for tile_row_idx in range(tile_rows):
        base_row = tile_row_idx * chunk
        row_end = min(base_row + chunk, rows)
        if base_row >= target_scale.shape[0]:
            break
        expanded_scale[base_row:row_end, :tile_col_cnt] = per_block_decode_scale[tile_row_idx]
    
    # Convert to FP8 and view as uint8 (this is the suspect operation)
    fp8_view = expanded_scale.to(dtype=torch.float8_e4m3fn).view(torch.uint8)
    target_scale.copy_(fp8_view)
    
    # === Compare ===
    print(f"\nComparing scales:")
    print(f"target_scale shape: {target_scale.shape}")
    print(f"ref_scale shape: {ref_scale.shape}")
    
    # Convert ref_scale back to FP32 to see what values it contains
    ref_scale_fp32 = ref_scale.view(torch.float8_e4m3fn).to(torch.float32)
    
    # Check if our amax matches what the quantizer computed internally
    # Reverse-engineer the reference amax from ref_scale
    # ref_scale (FP8) → ref_scale_fp32 → amax = ref_scale_fp32 * 6.0 / global_scale
    ref_amax_reverse = ref_scale_fp32 * fp4_max / global_scale
    print(f"\nReverse-engineered ref amax [0,:5]: {ref_amax_reverse[0,:5].tolist()}")
    print(f"Our computed amax [0,:5]: {amax_tensor[0,:5].tolist()}")
    
    # Check if amax values match
    amax_match = torch.allclose(amax_tensor, ref_amax_reverse[:tile_h, :tile_w], rtol=0.01)
    print(f"Amax roughly matches: {amax_match}")
    
    scale_match = torch.equal(target_scale, ref_scale)
    print(f"\nScales match exactly: {scale_match}")
    
    if not scale_match:
        mismatches = (target_scale != ref_scale).sum().item()
        total = target_scale.numel()
        print(f"Mismatches: {mismatches}/{total} ({100*mismatches/total:.4f}%)")


if __name__ == "__main__":
    #test_nvfp4_transpose_kernel()
    #test_nvfp4_partial_cast_matches_full()
    #test_scale_computation_matches_quantizer()
    test_single_gpu_partial_cast_vs_full()
    #main()
