import torch
import unittest
from typing import Tuple
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import get_thd_batch_on_this_cp_rank, pad_sequences_to_divisibility


class TestSequencePadding(unittest.TestCase):
    def test_basic_padding_with_labels(self):
        """Test basic padding functionality including labels."""
        # Setup
        input_ids = torch.tensor([1, 1, 2, 2, 2])  # Two sequences: [1,1] and [2,2,2]
        labels = torch.tensor([10, 11, 20, 21, 22])  # Labels for each token
        positional_ids = torch.tensor([0, 1, 0, 1, 2])
        cu_seqlens_q = torch.tensor([0, 2, 5])
        divisibility_factor = 4

        # Execute
        input_ids_padded, labels_padded, _, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(
            input_ids.unsqueeze(0),  # Add batch dimension
            labels.unsqueeze(0),  # Add batch dimension
            positional_ids,
            cu_seqlens_q,
            divisibility_factor,
            padding_token_id=0,
            padding_label_id=-100,
        )

        # Assert
        # First sequence [1,1] needs 2 padding tokens to be divisible by 4
        # Second sequence [2,2,2] needs 1 padding token to be divisible by 4
        expected_input_ids = torch.tensor([1, 1, 0, 0, 2, 2, 2, 0])
        expected_labels = torch.tensor([10, 11, -100, -100, 20, 21, 22, -100])
        # Position IDs continue counting: [0,1] -> [0,1,2,3], [0,1,2] -> [0,1,2,3]
        expected_positional_ids_padded = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        expected_cu_seqlens_padded = torch.tensor([0, 4, 8])

        self.assertTrue(torch.equal(input_ids_padded, expected_input_ids))
        self.assertTrue(torch.equal(labels_padded, expected_labels))
        self.assertTrue(torch.equal(positional_ids_padded, expected_positional_ids_padded))
        self.assertTrue(torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded))

    def test_no_padding_needed_with_labels(self):
        """Test when sequences are already divisible (with labels)."""
        # Setup - sequences of length 4 and 8, divisibility factor 4
        input_ids = torch.tensor([1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
        labels = torch.tensor([10, 11, 12, 13, 20, 21, 22, 23, 24, 25, 26, 27])
        positional_ids = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7])
        cu_seqlens_q = torch.tensor([0, 4, 12])
        divisibility_factor = 4

        # Execute
        input_ids_padded, labels_padded, _, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(
            input_ids.unsqueeze(0), labels.unsqueeze(0), positional_ids, cu_seqlens_q, divisibility_factor
        )

        # Assert - no padding should be added
        self.assertTrue(torch.equal(input_ids_padded, input_ids))
        self.assertTrue(torch.equal(labels_padded, labels))
        self.assertTrue(torch.equal(positional_ids_padded, positional_ids))
        self.assertTrue(torch.equal(cu_seqlens_q_padded, cu_seqlens_q))

    def test_single_sequence_with_labels(self):
        """Test with a single sequence including labels."""
        # Setup
        input_ids = torch.tensor([3, 3, 3])
        labels = torch.tensor([30, 31, 32])
        positional_ids = torch.tensor([0, 1, 2])
        cu_seqlens_q = torch.tensor([0, 3])
        divisibility_factor = 5

        # Execute
        input_ids_padded, labels_padded, _, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            positional_ids,
            cu_seqlens_q,
            divisibility_factor,
            padding_token_id=99,
            padding_label_id=-100,
        )

        # Assert - need 2 padding tokens to make length 5
        expected_input_ids = torch.tensor([3, 3, 3, 99, 99])
        expected_labels = torch.tensor([30, 31, 32, -100, -100])
        # Position IDs continue: [0,1,2] -> [0,1,2,3,4]
        expected_positional_ids_padded = torch.tensor([0, 1, 2, 3, 4])
        expected_cu_seqlens_padded = torch.tensor([0, 5])

        self.assertTrue(torch.equal(input_ids_padded, expected_input_ids))
        self.assertTrue(torch.equal(labels_padded, expected_labels))
        self.assertTrue(torch.equal(positional_ids_padded, expected_positional_ids_padded))
        self.assertTrue(torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded))

    def test_multiple_sequences_varied_padding_with_labels(self):
        """Test with multiple sequences requiring different padding amounts (with labels)."""
        # Setup - 3 sequences: lengths 1, 3, 5
        input_ids = torch.tensor([1, 2, 2, 2, 3, 3, 3, 3, 3])
        labels = torch.tensor([100, 200, 201, 202, 300, 301, 302, 303, 304])
        positional_ids = torch.tensor([0, 0, 1, 2, 0, 1, 2, 3, 4])
        cu_seqlens_q = torch.tensor([0, 1, 4, 9])
        divisibility_factor = 3

        # Execute
        input_ids_padded, labels_padded, original_pos_ids, positional_ids_padded, cu_seqlens_q_padded = (
            pad_sequences_to_divisibility(
                input_ids.unsqueeze(0), labels.unsqueeze(0), positional_ids, cu_seqlens_q, divisibility_factor
            )
        )

        # Assert
        # Seq 1: length 1 -> needs 2 padding tokens, positions continue [0] -> [0,1,2]
        # Seq 2: length 3 -> needs 0 padding tokens, stays [0,1,2]
        # Seq 3: length 5 -> needs 1 padding token, positions continue [0,1,2,3,4] -> [0,1,2,3,4,5]
        expected_input_ids = torch.tensor([1, 0, 0, 2, 2, 2, 3, 3, 3, 3, 3, 0])
        expected_labels = torch.tensor([100, -100, -100, 200, 201, 202, 300, 301, 302, 303, 304, -100])
        expected_positional_ids_padded = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 3, 4, 5])
        expected_cu_seqlens_padded = torch.tensor([0, 3, 6, 12])

        self.assertTrue(torch.equal(input_ids_padded, expected_input_ids))
        self.assertTrue(torch.equal(labels_padded, expected_labels))
        self.assertTrue(torch.equal(original_pos_ids, positional_ids))  # Original unchanged
        self.assertTrue(torch.equal(positional_ids_padded, expected_positional_ids_padded))
        self.assertTrue(torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded))

    def test_divisibility_factor_one_with_labels(self):
        """Test with divisibility factor of 1 (no padding needed) with labels."""
        # Setup
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        labels = torch.tensor([10, 20, 30, 40, 50])
        positional_ids = torch.tensor([0, 1, 2, 3, 4])
        cu_seqlens_q = torch.tensor([0, 2, 5])
        divisibility_factor = 1

        # Execute
        input_ids_padded, labels_padded, _, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(
            input_ids.unsqueeze(0), labels.unsqueeze(0), positional_ids, cu_seqlens_q, divisibility_factor
        )

        # Assert - everything divisible by 1, no padding
        self.assertTrue(torch.equal(input_ids_padded, input_ids))
        self.assertTrue(torch.equal(labels_padded, labels))
        self.assertTrue(torch.equal(positional_ids_padded, positional_ids))
        self.assertTrue(torch.equal(cu_seqlens_q_padded, cu_seqlens_q))

    def test_custom_padding_values(self):
        """Test with custom padding values for input_ids and labels."""
        # Setup
        input_ids = torch.tensor([1, 1])
        labels = torch.tensor([10, 11])
        positional_ids = torch.tensor([0, 1])
        cu_seqlens_q = torch.tensor([0, 2])
        divisibility_factor = 3

        # Execute with custom padding values
        input_ids_padded, labels_padded, _, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            positional_ids,
            cu_seqlens_q,
            divisibility_factor,
            padding_token_id=777,
            padding_label_id=-999,
        )

        # Assert - need 1 padding token to make length 3
        expected_input_ids = torch.tensor([1, 1, 777])
        expected_labels = torch.tensor([10, 11, -999])
        # Position IDs continue: [0,1] -> [0,1,2]
        expected_positional_ids_padded = torch.tensor([0, 1, 2])
        expected_cu_seqlens_padded = torch.tensor([0, 3])

        self.assertTrue(torch.equal(input_ids_padded, expected_input_ids))
        self.assertTrue(torch.equal(labels_padded, expected_labels))
        self.assertTrue(torch.equal(positional_ids_padded, expected_positional_ids_padded))
        self.assertTrue(torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded))

    def test_padding_with_custom_padding_values_sequences_shorter_than_divisibility_factor(self):
        """Test with custom padding values for all tensors."""
        # Setup

        input_ids = torch.tensor([1, 1, 1, 2, 2, 3, 3, 3, 3])
        cu_seqlens_q = torch.tensor([0, 3, 5, 9])
        labels = torch.tensor([-100, -100, -100, -100, -100, -100, -100, 13, -100])
        positional_ids = torch.tensor([0, 1, 2, 0, 1, 0, 1, 2, 3])
        divisibility_factor = 8

        pid = 777
        label_pad = -200

        input_ids_padded, labels_padded, positional_ids, positional_ids_padded, cu_seqlens_q_padded = (
            pad_sequences_to_divisibility(
                input_ids.unsqueeze(0),
                labels.unsqueeze(0),
                positional_ids,
                cu_seqlens_q,
                divisibility_factor,
                padding_token_id=pid,
                padding_label_id=label_pad,
            )
        )

        # Sequence: [ a a a p p p p p b b pppppp ccccpppp]
        print("input_ids_padded: ", input_ids_padded)
        print("labels_padded: ", labels_padded)
        print("positional_ids_padded: ", positional_ids_padded)
        print("cu_seqlens_q_padded: ", cu_seqlens_q_padded)

        expected_input_ids = torch.tensor(
            [1, 1, 1, pid, pid, pid, pid, pid, 2, 2, pid, pid, pid, pid, pid, pid, 3, 3, 3, 3, pid, pid, pid, pid]
        )
        expected_cu_seqlens_q_padded = torch.tensor([0, 8, 16, 24])
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
        assert torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_q_padded)

    def test_mixed_sequence_lengths_with_divisibility_factor(self):
        """Test with sequences both shorter and longer than divisibility factor."""
        # Setup - divisibility factor 6
        # Seq 1: length 2 (shorter than 6, needs 4 padding)
        # Seq 2: length 7 (longer than 6, needs 5 padding to reach 12)
        # Seq 3: length 4 (shorter than 6, needs 2 padding)
        # Seq 4: length 10 (longer than 6, needs 2 padding to reach 12)

        input_ids = torch.tensor([1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        labels = torch.tensor(
            [10, 11, 20, 21, 22, 23, 24, 25, 26, 30, 31, 32, 33, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
        )
        positional_ids = torch.tensor([0, 1, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        cu_seqlens_q = torch.tensor([0, 2, 9, 13, 23])
        divisibility_factor = 6

        pid = 999
        label_pad = -300

        # Execute
        input_ids_padded, labels_padded, _, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            positional_ids,
            cu_seqlens_q,
            divisibility_factor,
            padding_token_id=pid,
            padding_label_id=label_pad,
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
        self.assertTrue(torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded))

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
        positional_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, 4])
        cu_seqlens_q = torch.tensor([0, 7, 18, 23])
        divisibility_factor = 4

        pid = 888
        label_pad = -400

        # Execute
        input_ids_padded, labels_padded, _, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(
            input_ids.unsqueeze(0),
            labels.unsqueeze(0),
            positional_ids,
            cu_seqlens_q,
            divisibility_factor,
            padding_token_id=pid,
            padding_label_id=label_pad,
        )

        # Assert
        # Seq 1: 7 + 1 pad = 8 (divisible by 4)
        # Seq 2: 11 + 1 pad = 12 (divisible by 4)
        # Seq 3: 5 + 3 pads = 8 (divisible by 4)

        expected_input_ids = torch.tensor(
            [1, 1, 1, 1, 1, 1, 1, pid, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, pid, 3, 3, 3, 3, 3, pid, pid, pid]
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
        self.assertTrue(torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded))

# Now let's write a test that utilizes the pad sequence to divisibility function
# and then shards that on the CP ranks.
def test_shard_on_cp_ranks():
    # Make some dummy data.
    input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    labels = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    positional_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    cu_seqlens_q = torch.tensor([0, 5])
    divisibility_factor = 2
    padding_token_id = 0
    padding_label_id = -100
    input_ids_padded, labels_padded, positional_ids_padded, cu_seqlens_q_padded = pad_sequences_to_divisibility(input_ids, labels, positional_ids, cu_seqlens_q, divisibility_factor, padding_token_id, padding_label_id)
    assert input_ids_padded.shape == (2, 10)
    
    # Ok now we are going to split the data on the CP ranks. I think we can mock the groups?



if __name__ == "__main__":
    unittest.main()