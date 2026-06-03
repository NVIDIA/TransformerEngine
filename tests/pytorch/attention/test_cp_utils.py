# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for context parallel utils."""

import itertools
import torch
import unittest
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_batch_on_this_cp_rank,
    pad_thd_sequences_for_cp,
    generate_positional_ids_for_cp,
)

try:
    import transformer_engine_torch as tex
except ImportError:
    tex = None


class TestSequencePadding(unittest.TestCase):
    def test_padding_with_custom_padding_values_sequences_shorter_than_divisibility_factor(
        self,
    ):
        """Test with custom padding values for all tensors."""
        # Setup

        input_ids = torch.tensor([1, 1, 1, 2, 2, 3, 3, 3, 3])
        cu_seqlens = torch.tensor([0, 3, 5, 9])
        labels = torch.tensor([-100, -100, -100, -100, -100, -100, -100, 13, -100])
        positional_ids = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
        divisibility_factor = 8

        pid = 777
        label_pad = -200

        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            cu_seqlens,
            divisibility_factor,
            padding_token_id=pid,
            padding_label_id=label_pad,
        )

        positional_ids_padded = generate_positional_ids_for_cp(
            cu_seqlens,
            divisibility_factor,
        )

        # Sequence: [ a a a p p p p p b b pppppp ccccpppp]
        print("input_ids_padded: ", input_ids_padded)
        print("labels_padded: ", labels_padded)
        print("positional_ids_padded: ", positional_ids_padded)
        print("cu_seqlens_padded: ", cu_seqlens_padded)

        expected_input_ids = torch.tensor(
            [
                1,
                1,
                1,
                pid,
                pid,
                pid,
                pid,
                pid,
                2,
                2,
                pid,
                pid,
                pid,
                pid,
                pid,
                pid,
                3,
                3,
                3,
                3,
                pid,
                pid,
                pid,
                pid,
            ]
        )
        expected_cu_seqlens_padded = torch.tensor([0, 8, 16, 24])
        expected_labels_padded = torch.tensor(
            [
                -100,
                -100,
                -100,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                -100,
                -100,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                -100,
                -100,
                13,
                -100,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
            ]
        )
        expected_positional_ids = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
        )

        assert torch.equal(input_ids_padded, expected_input_ids)
        assert torch.equal(labels_padded, expected_labels_padded)
        assert torch.equal(positional_ids_padded, expected_positional_ids)
        assert torch.equal(cu_seqlens_padded, expected_cu_seqlens_padded)

    def test_mixed_sequence_lengths_with_divisibility_factor(self):
        """Test with sequences both shorter and longer than divisibility factor."""
        # Setup - divisibility factor 6
        # Seq 1: length 2 (shorter than 6, needs 4 padding)
        # Seq 2: length 7 (longer than 6, needs 5 padding to reach 12)
        # Seq 3: length 4 (shorter than 6, needs 2 padding)
        # Seq 4: length 10 (longer than 6, needs 2 padding to reach 12)

        input_ids = torch.tensor(
            [1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        )
        labels = torch.tensor(
            [
                10,
                11,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                30,
                31,
                32,
                33,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
            ]
        )
        positional_ids = torch.tensor(
            [0, 1, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        )
        cu_seqlens = torch.tensor([0, 2, 9, 13, 23])
        divisibility_factor = 6

        pid = 999
        label_pad = -300

        # Execute
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            cu_seqlens,
            divisibility_factor,
            padding_token_id=pid,
            padding_label_id=label_pad,
        )

        positional_ids_padded = generate_positional_ids_for_cp(
            cu_seqlens,
            divisibility_factor,
        )

        # Assert
        # Seq 1: [1,1] + 4 pads = 6 total
        # Seq 2: [2,2,2,2,2,2,2] + 5 pads = 12 total
        # Seq 3: [3,3,3,3] + 2 pads = 6 total
        # Seq 4: [4,4,4,4,4,4,4,4,4,4] + 2 pads = 12 total

        expected_input_ids = torch.tensor(
            [
                1,
                1,
                pid,
                pid,
                pid,
                pid,  # Seq 1: 2 + 4 padding
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                pid,
                pid,
                pid,
                pid,
                pid,  # Seq 2: 7 + 5 padding
                3,
                3,
                3,
                3,
                pid,
                pid,  # Seq 3: 4 + 2 padding
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                pid,
                pid,  # Seq 4: 10 + 2 padding
            ]
        )

        expected_labels = torch.tensor(
            [
                10,
                11,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                label_pad,
                30,
                31,
                32,
                33,
                label_pad,
                label_pad,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                label_pad,
                label_pad,
            ]
        )

        expected_positional_ids = torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,  # Seq 1 positions continue through padding
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,  # Seq 2 positions continue
                0,
                1,
                2,
                3,
                4,
                5,  # Seq 3 positions continue
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,  # Seq 4 positions continue
            ]
        )

        expected_cu_seqlens_padded = torch.tensor([0, 6, 18, 24, 36])

        self.assertTrue(torch.equal(input_ids_padded, expected_input_ids))
        self.assertTrue(torch.equal(labels_padded, expected_labels))
        self.assertTrue(torch.equal(positional_ids_padded, expected_positional_ids))
        self.assertTrue(torch.equal(cu_seqlens_padded, expected_cu_seqlens_padded))

    def test_sequences_longer_than_divisibility_factor(self):
        """Test with all sequences longer than the divisibility factor."""
        # Setup - divisibility factor 4, all sequences longer than 4
        # Seq 1: length 7 (needs 1 padding to reach 8)
        # Seq 2: length 11 (needs 1 padding to reach 12)
        # Seq 3: length 5 (needs 3 padding to reach 8)

        input_ids = torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,  # 7 tokens
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,  # 11 tokens
                3,
                3,
                3,
                3,
                3,  # 5 tokens
            ]
        )
        labels = torch.tensor(
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                200,
                201,
                202,
                203,
                204,
                205,
                206,
                207,
                208,
                209,
                210,
                300,
                301,
                302,
                303,
                304,
            ]
        )
        positional_ids = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4]
        )
        cu_seqlens = torch.tensor([0, 7, 18, 23])
        divisibility_factor = 4

        pid = 888
        label_pad = -400

        # Execute
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            cu_seqlens,
            divisibility_factor,
            padding_token_id=pid,
            padding_label_id=label_pad,
        )

        positional_ids_padded = generate_positional_ids_for_cp(
            cu_seqlens,
            divisibility_factor,
        )

        # Assert
        # Seq 1: 7 + 1 pad = 8 (divisible by 4)
        # Seq 2: 11 + 1 pad = 12 (divisible by 4)
        # Seq 3: 5 + 3 pads = 8 (divisible by 4)

        expected_input_ids = torch.tensor(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                pid,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                pid,
                3,
                3,
                3,
                3,
                3,
                pid,
                pid,
                pid,
            ]
        )

        expected_labels = torch.tensor(
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                label_pad,
                200,
                201,
                202,
                203,
                204,
                205,
                206,
                207,
                208,
                209,
                210,
                label_pad,
                300,
                301,
                302,
                303,
                304,
                label_pad,
                label_pad,
                label_pad,
            ]
        )

        expected_positional_ids = torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
            ]
        )

        expected_cu_seqlens_padded = torch.tensor([0, 8, 20, 28])

        self.assertTrue(torch.equal(input_ids_padded, expected_input_ids))
        self.assertTrue(torch.equal(labels_padded, expected_labels))
        self.assertTrue(torch.equal(positional_ids_padded, expected_positional_ids))
        self.assertTrue(torch.equal(cu_seqlens_padded, expected_cu_seqlens_padded))


class TestContextParallelUtils(unittest.TestCase):
    """Test utilities for context parallel functionality."""

    def setUp(self):
        """Set up mock distributed environment."""
        # Mock torch.distributed functions
        self.original_get_world_size = torch.distributed.get_world_size
        self.original_get_rank = torch.distributed.get_rank

    def tearDown(self):
        """Restore original torch.distributed functions."""
        torch.distributed.get_world_size = self.original_get_world_size
        torch.distributed.get_rank = self.original_get_rank

    def _mock_distributed_env(self, cp_size, cp_rank):
        """Mock the distributed environment for testing."""

        def mock_get_world_size(group=None):
            return cp_size

        def mock_get_rank(group=None):
            return cp_rank

        torch.distributed.get_world_size = mock_get_world_size
        torch.distributed.get_rank = mock_get_rank

    def test_cp_rank_slicing_simple_case(self):
        """Test CP rank slicing with a simple 2-rank, single sequence case."""
        # Setup: Single sequence of length 8, CP size = 2
        # Each sequence gets divided into 2*cp_size = 4 slices of size 2 each
        # Rank 0 gets slices [0,1] and [6,7] (first and last)
        # Rank 1 gets slices [2,3] and [4,5] (second and second-to-last)

        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8]]
        )  # Shape: (1, 8) - batch first
        labels = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        position_ids = torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7]
        )  # Shape: (8,) - 1D as expected
        cu_seqlens = torch.tensor([0, 8])

        # Test rank 0
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # Rank 0 should get indices [0,1] and [6,7]
        expected_input_ids_r0 = torch.tensor([[1, 2, 7, 8]])
        expected_labels_r0 = torch.tensor([[10, 20, 70, 80]])
        expected_pos_ids_r0 = torch.tensor([0, 1, 6, 7])

        self.assertTrue(torch.equal(input_ids_r0, expected_input_ids_r0))
        self.assertTrue(torch.equal(labels_r0, expected_labels_r0))
        self.assertTrue(torch.equal(pos_ids_r0, expected_pos_ids_r0))

        # Test rank 1
        self._mock_distributed_env(cp_size=2, cp_rank=1)
        input_ids_r1, labels_r1, pos_ids_r1 = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # Rank 1 should get indices [2,3] and [4,5]
        expected_input_ids_r1 = torch.tensor([[3, 4, 5, 6]])
        expected_labels_r1 = torch.tensor([[30, 40, 50, 60]])
        expected_pos_ids_r1 = torch.tensor([2, 3, 4, 5])

        self.assertTrue(torch.equal(input_ids_r1, expected_input_ids_r1))
        self.assertTrue(torch.equal(labels_r1, expected_labels_r1))
        self.assertTrue(torch.equal(pos_ids_r1, expected_pos_ids_r1))

    def test_cp_rank_slicing_multiple_sequences(self):
        """Test CP rank slicing with multiple sequences."""
        # Setup: Two sequences of length 8 each, CP size = 2
        # Total sequence length = 16, cu_seqlens = [0, 8, 16]

        input_ids = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]]
        )
        labels = torch.tensor(
            [[10, 20, 30, 40, 50, 60, 70, 80, 110, 120, 130, 140, 150, 160, 170, 180]]
        )
        position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7])
        cu_seqlens = torch.tensor([0, 8, 16])

        # Test rank 0
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # For each sequence, rank 0 gets first and last slices
        # Seq 1: indices [0,1] and [6,7] -> values [1,2] and [7,8]
        # Seq 2: indices [8,9] and [14,15] -> values [11,12] and [17,18]
        expected_input_ids_r0 = torch.tensor([[1, 2, 7, 8, 11, 12, 17, 18]])
        expected_labels_r0 = torch.tensor([[10, 20, 70, 80, 110, 120, 170, 180]])
        expected_pos_ids_r0 = torch.tensor([0, 1, 6, 7, 0, 1, 6, 7])

        self.assertTrue(torch.equal(input_ids_r0, expected_input_ids_r0))
        self.assertTrue(torch.equal(labels_r0, expected_labels_r0))
        self.assertTrue(torch.equal(pos_ids_r0, expected_pos_ids_r0))

    def test_cp_rank_slicing_with_cp_size_1(self):
        """Test that CP size = 1 returns original tensors unchanged."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
        labels = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        cu_seqlens = torch.tensor([0, 8])

        self._mock_distributed_env(cp_size=1, cp_rank=0)
        input_ids_result, labels_result, pos_ids_result = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # With CP size = 1, should return original tensors
        self.assertTrue(torch.equal(input_ids_result, input_ids))
        self.assertTrue(torch.equal(labels_result, labels))
        self.assertTrue(torch.equal(pos_ids_result, position_ids))

    def test_cp_rank_slicing_sequence_dim_detection(self):
        """Test that the function correctly detects sequence dimension."""
        # Test with sequence dimension = 0 (sequence_length, batch_size)
        input_ids = torch.tensor(
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]
        )  # (8, 2)
        labels = torch.tensor(
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80]]
        )
        position_ids = torch.tensor(
            [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]
        )
        cu_seqlens = torch.tensor([0, 8])

        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # Should get indices [0,1] and [6,7] along dimension 0
        expected_input_ids_r0 = torch.tensor([[1, 10], [2, 20], [7, 70], [8, 80]])
        expected_labels_r0 = torch.tensor([[1, 10], [2, 20], [7, 70], [8, 80]])
        expected_pos_ids_r0 = torch.tensor([[0, 0], [1, 1], [6, 6], [7, 7]])

        self.assertTrue(torch.equal(input_ids_r0, expected_input_ids_r0))
        self.assertTrue(torch.equal(labels_r0, expected_labels_r0))
        self.assertTrue(torch.equal(pos_ids_r0, expected_pos_ids_r0))

    def test_cp_rank_slicing_mixed_dimensions(self):
        """Test CP rank slicing where input_ids/labels are 1D but position_ids has batch dimension."""
        # Setup: Single sequence of length 8, CP size = 2
        # This tests the opposite case from the simple test:
        # - input_ids and labels: 1D (no batch dimension)
        # - position_ids: 2D (has batch dimension)

        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])  # Shape: (8,) - 1D
        labels = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])  # Shape: (8,) - 1D
        position_ids = torch.tensor(
            [[0, 1, 2, 3, 4, 5, 6, 7]]
        )  # Shape: (1, 8) - 2D with batch
        cu_seqlens = torch.tensor([0, 8])

        # Test rank 0
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # Rank 0 should get indices [0,1] and [6,7]
        expected_input_ids_r0 = torch.tensor([1, 2, 7, 8])  # 1D result
        expected_labels_r0 = torch.tensor([10, 20, 70, 80])  # 1D result
        expected_pos_ids_r0 = torch.tensor(
            [[0, 1, 6, 7]]
        )  # 2D result (preserves batch dim)

        self.assertTrue(torch.equal(input_ids_r0, expected_input_ids_r0))
        self.assertTrue(torch.equal(labels_r0, expected_labels_r0))
        self.assertTrue(torch.equal(pos_ids_r0, expected_pos_ids_r0))

        # Test rank 1
        self._mock_distributed_env(cp_size=2, cp_rank=1)
        input_ids_r1, labels_r1, pos_ids_r1 = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # Rank 1 should get indices [2,3] and [4,5]
        expected_input_ids_r1 = torch.tensor([3, 4, 5, 6])  # 1D result
        expected_labels_r1 = torch.tensor([30, 40, 50, 60])  # 1D result
        expected_pos_ids_r1 = torch.tensor(
            [[2, 3, 4, 5]]
        )  # 2D result (preserves batch dim)

        self.assertTrue(torch.equal(input_ids_r1, expected_input_ids_r1))
        self.assertTrue(torch.equal(labels_r1, expected_labels_r1))
        self.assertTrue(torch.equal(pos_ids_r1, expected_pos_ids_r1))

    def test_integration_with_padding_and_cp_slicing(self):
        """Integration test: pad sequences then slice for CP ranks."""
        # Start with unpadded sequences
        input_ids = torch.tensor([1, 1, 2, 2, 2])  # Two sequences: [1,1] and [2,2,2]
        labels = torch.tensor([10, 11, 20, 21, 22])
        positional_ids = torch.tensor([0, 1, 0, 1, 2])
        cu_seqlens = torch.tensor([0, 2, 5])
        divisibility_factor = 4  # Will pad to lengths 4 and 4

        # First, pad sequences
        input_ids_padded, labels_padded, cu_seqlens_padded = pad_thd_sequences_for_cp(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            cu_seqlens,
            divisibility_factor,
            padding_token_id=0,
            padding_label_id=-100,
        )

        positional_ids_padded = generate_positional_ids_for_cp(
            cu_seqlens,
            divisibility_factor,
        )

        # Expected after padding: [1,1,0,0,2,2,2,0] with cu_seqlens [0,4,8]
        expected_padded = torch.tensor([1, 1, 0, 0, 2, 2, 2, 0])
        self.assertTrue(torch.equal(input_ids_padded, expected_padded))

        # Now test CP slicing with cp_size=2

        # Test rank 0
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens_padded,
            input_ids_padded.unsqueeze(0),
            labels_padded.unsqueeze(0),
            positional_ids_padded,
        )

        # Each sequence of length 4 gets divided into 4 slices of size 1
        # Rank 0 gets slices [0] and [3] from each sequence
        # Seq 1: indices [0] and [3] -> values [1] and [0]
        # Seq 2: indices [4] and [7] -> values [2] and [0]
        expected_input_ids_r0 = torch.tensor([[1, 0, 2, 0]])

        self.assertTrue(torch.equal(input_ids_r0, expected_input_ids_r0))


def _legacy_reorder_thd_to_rank_sharded(x, cu_seqlens, cp_size, seq_dim=0):
    total_slices_of_any_sequence = 2 * cp_size
    slice_sizes = (cu_seqlens[1:] - cu_seqlens[:-1]) // total_slices_of_any_sequence

    indices = [
        (
            torch.arange(
                seq_start + (cp_rank * slice_size),
                seq_start + ((cp_rank + 1) * slice_size),
                device=cu_seqlens.device,
            ),
            torch.arange(
                seq_start + ((total_slices_of_any_sequence - cp_rank - 1) * slice_size),
                seq_start + ((total_slices_of_any_sequence - cp_rank) * slice_size),
                device=cu_seqlens.device,
            ),
        )
        for cp_rank in range(cp_size)
        for slice_size, seq_start in zip(slice_sizes, cu_seqlens[:-1])
    ]

    indices = list(itertools.chain(*indices))
    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


def _legacy_reorder_thd_to_contiguous(x, cu_seqlens, seq_chunk_ids, cp_size, seq_dim=0):
    max_cum_seqlen_per_cp_rank = cu_seqlens[-1] // cp_size
    cu_seqlens_on_any_cp_rank = cu_seqlens // cp_size

    indices = [
        torch.arange(
            (
                start + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            (
                (start + end) // 2 + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
                if loc < cp_size
                else end + max_cum_seqlen_per_cp_rank * (chunk_id // 2)
            ),
            device=cu_seqlens.device,
        )
        for start, end in zip(
            cu_seqlens_on_any_cp_rank[:-1], cu_seqlens_on_any_cp_rank[1:]
        )
        for loc, chunk_id in enumerate(seq_chunk_ids)
    ]

    indices = torch.cat(indices)
    return x.index_select(seq_dim, indices)


def _legacy_valid_copy(out, inp, cu_seqlens_padded, cu_seqlens):
    batch_size = cu_seqlens.shape[0] - 1
    for b in range(batch_size):
        s = cu_seqlens_padded[b].item()
        sz = (cu_seqlens[b + 1] - cu_seqlens[b]).item()
        if sz > 0:
            out[s : s + sz].copy_(inp[s : s + sz])


@unittest.skipIf(
    not torch.cuda.is_available() or tex is None,
    "THD kernel tests require CUDA and transformer_engine_torch",
)
class TestTHDKernels(unittest.TestCase):
    def test_thd_reorder_matches_legacy_python_reorder(self):
        cp_size = 4
        cu_seqlens = torch.tensor([0, 8, 24, 40], dtype=torch.int32, device="cuda")
        x = torch.arange(40 * 2 * 4, dtype=torch.float16, device="cuda").view(40, 2, 4)

        rank_sharded = tex.thd_reorder(x, cu_seqlens, cp_size, False, x.shape[0])
        ref_rank_sharded = _legacy_reorder_thd_to_rank_sharded(x, cu_seqlens, cp_size)
        self.assertTrue(torch.equal(rank_sharded, ref_rank_sharded))

        seq_chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device="cuda")
        for rank in range(cp_size):
            seq_chunk_ids[rank] = 2 * rank
            seq_chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
        contiguous = tex.thd_reorder(
            rank_sharded, cu_seqlens, cp_size, True, rank_sharded.shape[0]
        )
        ref_contiguous = _legacy_reorder_thd_to_contiguous(
            rank_sharded, cu_seqlens, seq_chunk_ids, cp_size
        )
        self.assertTrue(torch.equal(contiguous, ref_contiguous))
        self.assertTrue(torch.equal(contiguous, x))

    def test_thd_get_partitioned_indices_matches_dual_chunk_expected_indices(self):
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda")

        rank0 = tex.thd_get_partitioned_indices(cu_seqlens, 16, 2, 0)
        rank1 = tex.thd_get_partitioned_indices(cu_seqlens, 16, 2, 1)

        expected_rank0 = torch.tensor(
            [0, 1, 6, 7, 8, 9, 14, 15], dtype=torch.int32, device="cuda"
        )
        expected_rank1 = torch.tensor(
            [2, 3, 4, 5, 10, 11, 12, 13], dtype=torch.int32, device="cuda"
        )
        self.assertTrue(torch.equal(rank0, expected_rank0))
        self.assertTrue(torch.equal(rank1, expected_rank1))

    def test_thd_valid_copy_matches_legacy_slice_copy_loop(self):
        cu_seqlens_padded = torch.tensor([2, 6, 12], dtype=torch.int32, device="cuda")
        cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32, device="cuda")
        inp = torch.arange(12 * 2 * 4, dtype=torch.float16, device="cuda").view(
            12, 2, 4
        )
        out = torch.full_like(inp, -1)
        expected = torch.full_like(inp, -1)

        _legacy_valid_copy(expected, inp, cu_seqlens_padded, cu_seqlens)
        tex.thd_valid_copy(out, inp, cu_seqlens_padded, cu_seqlens)
        self.assertTrue(torch.equal(out, expected))

    def test_thd_read_half_tensor_reads_each_sequence_half(self):
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda")
        q = torch.arange(16 * 2 * 4, dtype=torch.float16, device="cuda").view(16, 2, 4)
        kv = torch.arange(2 * 16 * 2 * 4, dtype=torch.float16, device="cuda").view(
            2, 16, 2, 4
        )

        q_first = tex.thd_read_half_tensor(q, cu_seqlens, 0)
        q_second = tex.thd_read_half_tensor(q, cu_seqlens, 1)
        kv_first = tex.thd_read_half_tensor(kv, cu_seqlens, 0)
        kv_second = tex.thd_read_half_tensor(kv, cu_seqlens, 1)

        expected_first = torch.cat([q[0:4], q[8:12]], dim=0)
        expected_second = torch.cat([q[4:8], q[12:16]], dim=0)
        self.assertTrue(torch.equal(q_first, expected_first))
        self.assertTrue(torch.equal(q_second, expected_second))
        self.assertTrue(
            torch.equal(kv_first, torch.stack([expected_first, expected_first + 128]))
        )
        self.assertTrue(
            torch.equal(
                kv_second, torch.stack([expected_second, expected_second + 128])
            )
        )

    def test_thd_read_second_half_lse_handles_packed_and_batch_major_lse(self):
        cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32, device="cuda")
        lse = torch.arange(2 * 2 * 8, dtype=torch.float32, device="cuda").view(2, 2, 8)
        packed_lse = torch.arange(2 * 16, dtype=torch.float32, device="cuda").view(
            2, 16
        )

        second_half_lse = tex.thd_read_second_half_lse(lse, cu_seqlens, False, 4)
        packed_second_half_lse = tex.thd_read_second_half_lse(
            packed_lse, cu_seqlens, True, 8
        )

        expected = lse[:, :, 4:8]
        expected_packed = torch.cat([packed_lse[:, 4:8], packed_lse[:, 12:16]], dim=1)
        self.assertTrue(torch.equal(second_half_lse, expected))
        self.assertTrue(torch.equal(packed_second_half_lse, expected_packed))


if __name__ == "__main__":
    unittest.main()
