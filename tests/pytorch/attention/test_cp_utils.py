# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for context parallel utils."""
import torch
import unittest
from typing import Tuple
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_batch_on_this_cp_rank,
    pad_thd_sequences_for_cp,
    generate_positional_ids_for_cp,
)


class TestSequencePadding(unittest.TestCase):
    def test_padding_with_custom_padding_values_sequences_shorter_than_divisibility_factor(self):
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
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7]
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

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # Shape: (1, 8) - batch first
        labels = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]])
        position_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # Shape: (8,) - 1D as expected
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

        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18]])
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
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]])  # Shape: (1, 8) - 2D with batch
        cu_seqlens = torch.tensor([0, 8])

        # Test rank 0
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens, input_ids, labels, position_ids
        )

        # Rank 0 should get indices [0,1] and [6,7]
        expected_input_ids_r0 = torch.tensor([1, 2, 7, 8])  # 1D result
        expected_labels_r0 = torch.tensor([10, 20, 70, 80])  # 1D result
        expected_pos_ids_r0 = torch.tensor([[0, 1, 6, 7]])  # 2D result (preserves batch dim)

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
        expected_pos_ids_r1 = torch.tensor([[2, 3, 4, 5]])  # 2D result (preserves batch dim)

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

    def test_bshd_format_basic(self):
        """Test get_batch_on_this_cp_rank with bshd format."""
        # Setup: batch_size=2, seq_len=8, CP size = 2
        # For bshd format: (batch, sequence, heads, dim)
        batch_size = 2
        seq_len = 8  # Must be divisible by 2*cp_size = 4
        
        # Create test tensors in bshd format
        input_ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
        labels = input_ids * 10
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Test rank 0
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,  # Not used for bshd format
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="bshd"
        )
        
        # Rank 0 gets chunks 0 and 3 (indices 0,1 and 6,7 for each batch)
        expected_shape = (batch_size, seq_len // 2)
        self.assertEqual(input_ids_r0.shape, expected_shape)
        self.assertEqual(labels_r0.shape, expected_shape)
        self.assertEqual(pos_ids_r0.shape, expected_shape)
        
        # Verify the actual values for first batch
        # Should get elements at indices [0,1,6,7] from each batch
        expected_input_ids_batch0 = torch.tensor([0, 1, 6, 7])
        expected_input_ids_batch1 = torch.tensor([8, 9, 14, 15])
        expected_input_ids_r0 = torch.stack([expected_input_ids_batch0, expected_input_ids_batch1])
        self.assertTrue(torch.equal(input_ids_r0, expected_input_ids_r0))
        
        # Test rank 1
        self._mock_distributed_env(cp_size=2, cp_rank=1)
        input_ids_r1, labels_r1, pos_ids_r1 = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="bshd"
        )
        
        # Rank 1 gets chunks 1 and 2 (indices 2,3 and 4,5 for each batch)
        expected_input_ids_batch0 = torch.tensor([2, 3, 4, 5])
        expected_input_ids_batch1 = torch.tensor([10, 11, 12, 13])
        expected_input_ids_r1 = torch.stack([expected_input_ids_batch0, expected_input_ids_batch1])
        self.assertTrue(torch.equal(input_ids_r1, expected_input_ids_r1))

    def test_sbhd_format_basic(self):
        """Test get_batch_on_this_cp_rank with sbhd format."""
        # Setup: seq_len=8, batch_size=2, CP size = 2
        # For sbhd format: (sequence, batch, heads, dim)
        seq_len = 8  # Must be divisible by 2*cp_size = 4
        batch_size = 2
        
        # Create test tensors in sbhd format (seq first)
        input_ids = torch.arange(seq_len * batch_size).reshape(seq_len, batch_size)
        labels = input_ids * 10
        position_ids = torch.arange(seq_len).unsqueeze(1).expand(seq_len, batch_size)
        
        # Test rank 0
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_r0, labels_r0, pos_ids_r0 = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,  # Not used for sbhd format
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="sbhd"
        )
        
        # Rank 0 gets chunks 0 and 3 (indices 0,1 and 6,7)
        expected_shape = (seq_len // 2, batch_size)
        self.assertEqual(input_ids_r0.shape, expected_shape)
        self.assertEqual(labels_r0.shape, expected_shape)
        self.assertEqual(pos_ids_r0.shape, expected_shape)
        
        # Verify the actual values
        # Should get rows at indices [0,1,6,7]
        expected_indices = torch.tensor([0, 1, 6, 7])
        expected_input_ids_r0 = input_ids[expected_indices]
        self.assertTrue(torch.equal(input_ids_r0, expected_input_ids_r0))
        
        # Test rank 1
        self._mock_distributed_env(cp_size=2, cp_rank=1)
        input_ids_r1, labels_r1, pos_ids_r1 = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="sbhd"
        )
        
        # Rank 1 gets chunks 1 and 2 (indices 2,3 and 4,5)
        expected_indices = torch.tensor([2, 3, 4, 5])
        expected_input_ids_r1 = input_ids[expected_indices]
        self.assertTrue(torch.equal(input_ids_r1, expected_input_ids_r1))

    def test_bshd_format_device_placement(self):
        """Test that bshd format processing maintains correct device placement."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        batch_size = 2
        seq_len = 8
        device = torch.device("cuda:0")
        
        # Create tensors on GPU
        input_ids = torch.arange(batch_size * seq_len, device=device).reshape(batch_size, seq_len)
        labels = input_ids * 10
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_out, labels_out, pos_ids_out = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="bshd"
        )
        
        # Verify outputs are on the same device
        self.assertEqual(input_ids_out.device, device)
        self.assertEqual(labels_out.device, device)
        self.assertEqual(pos_ids_out.device, device)

    def test_sbhd_format_device_placement(self):
        """Test that sbhd format processing maintains correct device placement."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        
        seq_len = 8
        batch_size = 2
        device = torch.device("cuda:0")
        
        # Create tensors on GPU
        input_ids = torch.arange(seq_len * batch_size, device=device).reshape(seq_len, batch_size)
        labels = input_ids * 10
        position_ids = torch.arange(seq_len, device=device).unsqueeze(1).expand(seq_len, batch_size)
        
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        input_ids_out, labels_out, pos_ids_out = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="sbhd"
        )
        
        # Verify outputs are on the same device
        self.assertEqual(input_ids_out.device, device)
        self.assertEqual(labels_out.device, device)
        self.assertEqual(pos_ids_out.device, device)

    def test_bshd_format_error_handling(self):
        """Test error handling for invalid inputs in bshd format."""
        batch_size = 2
        invalid_seq_len = 7  # Not divisible by 2*cp_size = 4
        
        input_ids = torch.randn(batch_size, invalid_seq_len)
        labels = torch.randn(batch_size, invalid_seq_len)
        position_ids = torch.randn(batch_size, invalid_seq_len)
        
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        
        # Should raise ValueError for invalid sequence length
        with self.assertRaises(ValueError) as context:
            get_batch_on_this_cp_rank(
                cu_seqlens_padded=None,
                input_ids_padded=input_ids,
                labels_padded=labels,
                position_ids_padded=position_ids,
                cp_group=None,
                qvk_format="bshd"
            )
        
        self.assertIn("must be divisible by", str(context.exception))
        
        # Test with tensor that has insufficient dimensions
        invalid_1d = torch.tensor([1, 2, 3, 4])
        with self.assertRaises(ValueError) as context:
            get_batch_on_this_cp_rank(
                cu_seqlens_padded=None,
                input_ids_padded=invalid_1d,
                labels_padded=invalid_1d,
                position_ids_padded=invalid_1d,
                cp_group=None,
                qvk_format="bshd"
            )
        
        self.assertIn("at least 2 dimensions", str(context.exception))

    def test_sbhd_format_error_handling(self):
        """Test error handling for invalid inputs in sbhd format."""
        invalid_seq_len = 7  # Not divisible by 2*cp_size = 4
        batch_size = 2
        
        input_ids = torch.randn(invalid_seq_len, batch_size)
        labels = torch.randn(invalid_seq_len, batch_size)
        position_ids = torch.randn(invalid_seq_len, batch_size)
        
        self._mock_distributed_env(cp_size=2, cp_rank=0)
        
        # Should raise ValueError for invalid sequence length
        with self.assertRaises(ValueError) as context:
            get_batch_on_this_cp_rank(
                cu_seqlens_padded=None,
                input_ids_padded=input_ids,
                labels_padded=labels,
                position_ids_padded=position_ids,
                cp_group=None,
                qvk_format="sbhd"
            )
        
        self.assertIn("must be divisible by", str(context.exception))

    def test_bshd_format_with_different_cp_sizes(self):
        """Test bshd format with different CP sizes."""
        batch_size = 2
        seq_len = 16  # Divisible by 2*cp_size for cp_size in [1, 2, 4]
        
        input_ids = torch.arange(batch_size * seq_len).reshape(batch_size, seq_len)
        labels = input_ids * 10
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        
        # Test with CP size = 1 (no parallelism)
        self._mock_distributed_env(cp_size=1, cp_rank=0)
        input_ids_cp1, _, _ = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="bshd"
        )
        # Should return original tensor
        self.assertTrue(torch.equal(input_ids_cp1, input_ids))
        
        # Test with CP size = 4
        self._mock_distributed_env(cp_size=4, cp_rank=0)
        input_ids_cp4, _, _ = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="bshd"
        )
        # Should get 2 chunks of size 2 each (total 4 elements per batch)
        self.assertEqual(input_ids_cp4.shape, (batch_size, seq_len // 4))

    def test_sbhd_format_with_different_cp_sizes(self):
        """Test sbhd format with different CP sizes."""
        seq_len = 16  # Divisible by 2*cp_size for cp_size in [1, 2, 4]
        batch_size = 2
        
        input_ids = torch.arange(seq_len * batch_size).reshape(seq_len, batch_size)
        labels = input_ids * 10
        position_ids = torch.arange(seq_len).unsqueeze(1).expand(seq_len, batch_size)
        
        # Test with CP size = 1 (no parallelism)
        self._mock_distributed_env(cp_size=1, cp_rank=0)
        input_ids_cp1, _, _ = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="sbhd"
        )
        # Should return original tensor
        self.assertTrue(torch.equal(input_ids_cp1, input_ids))
        
        # Test with CP size = 4
        self._mock_distributed_env(cp_size=4, cp_rank=0)
        input_ids_cp4, _, _ = get_batch_on_this_cp_rank(
            cu_seqlens_padded=None,
            input_ids_padded=input_ids,
            labels_padded=labels,
            position_ids_padded=position_ids,
            cp_group=None,
            qvk_format="sbhd"
        )
        # Should get 2 chunks of size 2 each (total 4 elements)
        self.assertEqual(input_ids_cp4.shape, (seq_len // 4, batch_size))


if __name__ == "__main__":
    unittest.main()
