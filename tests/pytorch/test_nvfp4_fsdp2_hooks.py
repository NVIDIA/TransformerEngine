# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for NVFP4Tensor FSDP2 all-gather hooks.

These tests verify the pre/post all-gather round-trip logic on a single GPU
without requiring torchrun or multi-GPU setup.
"""

import math
from typing import List, Tuple

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    NVFP4Quantizer,
    NVFP4Tensor,
)
from transformer_engine.pytorch.utils import round_up_to_nearest_multiple
from transformer_engine.pytorch.constants import NVFP4_BLOCK_SCALING_SIZE

nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# Shapes that exercise various M/K combinations:
# - (512, 256): both dims cleanly divisible by 128
# - (640, 128): M not a multiple of 128*2 but divisible by 16
# - (256, 1024): K > M
_test_shapes: List[Tuple[int, int]] = [
    (512, 256),
    (640, 128),
    (256, 1024),
]


def _make_nvfp4_tensor(shape: Tuple[int, int]) -> NVFP4Tensor:
    """Create an NVFP4Tensor from random BF16 data."""
    quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=True,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=True,
        stochastic_rounding=False,
        with_random_sign_mask=False,
    )
    src = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    return quantizer(src)


def _simulate_all_gather(
    sharded_tensors: Tuple[torch.Tensor, ...],
    world_size: int,
) -> Tuple[torch.Tensor, ...]:
    """Simulate FSDP2 all-gather by concatenating shards along dim0."""
    return tuple(torch.cat([t] * world_size, dim=0) for t in sharded_tensors)


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestNVFP4FSDP2Hooks:
    """Tests for fsdp_pre_all_gather / fsdp_post_all_gather round-trip."""

    @classmethod
    def setup_class(cls) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

    @pytest.mark.parametrize("shape", _test_shapes)
    @pytest.mark.parametrize("world_size", [2, 4])
    def test_round_trip_shapes(self, shape: Tuple[int, int], world_size: int):
        """Verify that pre_all_gather -> all_gather -> post_all_gather produces correct shapes."""
        M, K = shape
        shard_M = M // world_size
        shard_shape = (shard_M, K)

        qt = _make_nvfp4_tensor(shard_shape)

        # Pre all-gather
        sharded_tensors, metadata = qt.fsdp_pre_all_gather(
            mesh=None,
            orig_size=None,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )

        # Only rowwise tensors are all-gathered; columnwise is derived locally
        assert len(sharded_tensors) == 2, "Expected 2 tensors (rowwise data + scale only)"

        rowwise_data, rowwise_scale_inv = sharded_tensors

        # Rowwise data: (shard_M, K//2) — unmodified
        assert rowwise_data.shape == (shard_M, K // 2)
        # Rowwise scale: unpadded dim0 to shard_M
        assert rowwise_scale_inv.shape[0] == shard_M

        # Simulate all-gather
        all_gather_outputs = _simulate_all_gather(sharded_tensors, world_size)

        # Post all-gather
        result, _ = qt.fsdp_post_all_gather(
            all_gather_outputs,
            metadata,
            param_dtype=torch.bfloat16,
        )

        # Verify output is NVFP4Tensor with correct logical shape
        assert isinstance(result, NVFP4Tensor)
        assert tuple(result.shape) == (M, K)

        # Verify internal data shapes
        assert result._rowwise_data.shape == (M, K // 2)

        expected_rowwise_scale_shape = (
            round_up_to_nearest_multiple(M, 128),
            round_up_to_nearest_multiple(math.ceil(K / NVFP4_BLOCK_SCALING_SIZE), 4),
        )
        assert result._rowwise_scale_inv.shape == expected_rowwise_scale_shape

        # Columnwise data derived locally via _create_columnwise()
        assert result._columnwise_data.shape == (K, M // 2)

        expected_col_scale_shape = (
            round_up_to_nearest_multiple(K, 128),
            round_up_to_nearest_multiple(math.ceil(M / NVFP4_BLOCK_SCALING_SIZE), 4),
        )
        assert result._columnwise_scale_inv.shape == expected_col_scale_shape

    @pytest.mark.parametrize("shape", _test_shapes)
    def test_round_trip_data_integrity(self, shape: Tuple[int, int]):
        """Verify data and dequantized values survive the pre -> all_gather -> post round-trip."""
        world_size = 2
        M, K = shape
        shard_M = M // world_size
        shard_shape = (shard_M, K)

        qt = _make_nvfp4_tensor(shard_shape)

        # Save original internal tensors for comparison
        orig_rowwise_data = qt._rowwise_data.clone()
        orig_rowwise_scale = qt._rowwise_scale_inv.clone()
        orig_amax_row = qt._amax_rowwise.clone()
        orig_amax_col = qt._amax_columnwise.clone()
        orig_deq = qt.dequantize()

        # Pre all-gather
        sharded_tensors, metadata = qt.fsdp_pre_all_gather(
            mesh=None,
            orig_size=None,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )

        # Simulate all-gather (world_size copies — data from each "rank" is identical)
        all_gather_outputs = _simulate_all_gather(sharded_tensors, world_size)

        # Post all-gather
        result, _ = qt.fsdp_post_all_gather(
            all_gather_outputs,
            metadata,
            param_dtype=torch.bfloat16,
        )

        # Since each "rank" has the same data, the full rowwise_data should be
        # the original shard repeated world_size times
        expected_rowwise_data = torch.cat([orig_rowwise_data] * world_size, dim=0)
        assert torch.equal(result._rowwise_data, expected_rowwise_data)

        # Rowwise scale: each shard's unpadded scale is repeated, then repadded
        # Check that the first shard_M rows of the scale match the original (unpadded)
        assert torch.equal(
            result._rowwise_scale_inv[:shard_M, :],
            orig_rowwise_scale[:shard_M, :],
        )

        # Columnwise data is derived locally via _create_columnwise(), not all-gathered.
        # Verify it was created and has the correct shape.
        assert result._columnwise_data is not None
        assert result._columnwise_data.shape == (K, M // 2)
        assert result._columnwise_scale_inv is not None

        # Amax values passed through metadata — should be preserved
        assert torch.equal(result._amax_rowwise, orig_amax_row)
        assert torch.equal(result._amax_columnwise, orig_amax_col)

        # Dequantized values: the full tensor should dequantize to world_size copies of the shard
        result_deq = result.dequantize()
        expected_deq = torch.cat([orig_deq] * world_size, dim=0)
        torch.testing.assert_close(result_deq, expected_deq)

    @pytest.mark.parametrize("shape", _test_shapes)
    def test_in_place_update(self, shape: Tuple[int, int]):
        """Verify the out= path (in-place update on subsequent iterations)."""
        world_size = 2
        M, K = shape
        shard_M = M // world_size
        shard_shape = (shard_M, K)

        qt = _make_nvfp4_tensor(shard_shape)

        sharded_tensors, metadata = qt.fsdp_pre_all_gather(
            mesh=None,
            orig_size=None,
            contiguous_orig_stride=None,
            module=None,
            mp_policy=None,
        )
        all_gather_outputs = _simulate_all_gather(sharded_tensors, world_size)

        # First call: out=None -> creates new tensor
        result, _ = qt.fsdp_post_all_gather(
            all_gather_outputs,
            metadata,
            param_dtype=torch.bfloat16,
        )
        first_deq = result.dequantize().clone()

        # Second call: out=result -> in-place update
        result2, _ = qt.fsdp_post_all_gather(
            all_gather_outputs,
            metadata,
            param_dtype=torch.bfloat16,
            out=result,
        )
        assert result2 is result  # same object
        torch.testing.assert_close(result2.dequantize(), first_deq)

    def test_swizzled_scales_rejected(self):
        """Verify that GEMM-swizzled scales raise NotImplementedError."""
        shape = (512, 256)
        quantizer = NVFP4Quantizer(
            rowwise=True,
            columnwise=True,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            with_random_sign_mask=False,
        )
        quantizer.optimize_for_gemm = True
        src = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        qt = quantizer(src)

        if not qt._with_gemm_swizzled_scales:
            pytest.skip(
                "NVFP4Quantizer.optimize_for_gemm is not yet wired up in C++. "
                "Test will be unskipped once supported."
            )

        with pytest.raises(NotImplementedError, match="GEMM-swizzled"):
            qt.fsdp_pre_all_gather(
                mesh=None,
                orig_size=None,
                contiguous_orig_stride=None,
                module=None,
                mp_policy=None,
            )


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
class TestNVFP4DispatchHandlers:
    """Tests for as_strided, slice, and record_stream dispatch handlers."""

    def test_as_strided_noop(self):
        """as_strided with matching shape/strides returns NVFP4Tensor."""
        qt = _make_nvfp4_tensor((256, 128))
        M, K = qt.shape
        result = torch.ops.aten.as_strided.default(qt, [M, K], [K, 1], 0)
        assert isinstance(result, NVFP4Tensor)
        assert tuple(result.shape) == (M, K)

    def test_slice_noop(self):
        """slice covering full dimension returns NVFP4Tensor."""
        qt = _make_nvfp4_tensor((256, 128))
        M, K = qt.shape
        result = torch.ops.aten.slice.Tensor(qt, 0, 0, M)
        assert isinstance(result, NVFP4Tensor)
        assert tuple(result.shape) == (M, K)

    def test_record_stream(self):
        """record_stream completes without error."""
        qt = _make_nvfp4_tensor((256, 128))
        stream = torch.cuda.Stream()
        result = torch.ops.aten.record_stream.default(qt, stream)
        assert result is None
