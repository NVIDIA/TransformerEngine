# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils for context parallel integration testing."""

import torch
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import pad_thd_sequences_for_cp, generate_positional_ids_for_cp
import torch.distributed as dist
import os
from dataclasses import dataclass, field

@dataclass
class DistributedConfig:
    """Class to track distributed ranks."""

    rank: int = field(default_factory=dist.get_rank)
    local_rank: int = field(default_factory=lambda: int(os.environ["LOCAL_RANK"]))
    world_size: int = field(default_factory=dist.get_world_size)

    def is_main_process(self) -> bool:
        """This is the global rank 0 process, to be used for wandb logging, etc."""
        return self.rank == 0

def get_dummy_data_bshd():
    """Generate dummy data in BSHD format for testing.
    
    BSHD format uses traditional padding where each sequence in the batch
    is padded to the same maximum length, unlike THD which packs sequences.
    """
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.
    
    # For BSHD, we create separate sequences and pad each to max_seq_length
    batch_size = 1  # 3 separate sequences instead of packed
    max_seq_length = 12  # Pad all sequences to this length (longest sequence is 11 tokens)
    
    # Create 3 separate sequences with different lengths, each padded to max_seq_length
    # Sequence 1: 7 tokens, padded to 12
    seq1_input_ids = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    seq1_labels = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
    seq1_position_ids = list(range(12))
    
    # Sequence 2: 11 tokens, padded to 12
    seq2_input_ids = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    seq2_labels = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    seq2_position_ids = list(range(12))
    
    # Sequence 3: 5 tokens, padded to 12
    seq3_input_ids = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    seq3_labels = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    seq3_position_ids = list(range(12))
    
    # Stack into batch tensors [batch_size, seq_length]
    input_ids_2d = torch.tensor([
        seq1_input_ids,
        seq2_input_ids, 
        seq3_input_ids
    ])  # Shape: [3, 12]
    
    labels_2d = torch.tensor([
        seq1_labels,
        seq2_labels,
        seq3_labels
    ])  # Shape: [3, 12]
    
    position_ids_2d = torch.tensor([
        seq1_position_ids,
        seq2_position_ids,
        seq3_position_ids
    ])  # Shape: [3, 12]
    
    # For BSHD, attention masks are typically used instead of cu_seqlens
    # But we'll provide cu_seqlens for compatibility with the existing model
    # Each sequence starts at multiples of max_seq_length
    cu_seqlens_q = torch.tensor([0, 12, 24, 36])  # 3 sequences of length 12 each
    
    # Create batch dictionary for BSHD format
    batch = {
        "input_ids": input_ids_2d.to(torch.int64),  # 2D format: [3, 12]
        "labels": labels_2d.to(torch.int64),  # 2D format: [3, 12]
        "position_ids": position_ids_2d.to(torch.int64),  # 2D format: [3, 12]
        "cu_seqlens_q": cu_seqlens_q.to(torch.int32),
        "cu_seqlens_kv": cu_seqlens_q.to(torch.int32),
        "pad_between_seqs": False, # TODO: Ensure this is false now for BSHD?
        "max_seqlen_q": max_seq_length,  # Fixed sequence length for BSHD
        "max_seqlen_kv": max_seq_length,
    }
    return batch

def get_dummy_data_thd():
    pid = 1 # The pad token id.
    label_pad = -100 # The label pad id.

    # Make some fake data.
    input_ids = torch.tensor([
                1, 1, 1, 1, 1, 1, 1,  # 7 tokens
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # 11 tokens
                3, 3, 3, 3, 3  # 5 tokens
            ])
    labels = torch.tensor([
        10, 11, 12, 13, 14, 15, 16,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        5, 6, 7, 8, 9
    ])
    positional_ids = torch.tensor([
        0, 1, 2, 3, 4, 5, 6,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        0, 1, 2, 3, 4
    ])
    cu_seqlens_q = torch.tensor([0, 7, 18, 23])
    cp_size=2
    divisibility_factor = 2 * cp_size

    input_ids_padded, labels_padded, cu_seqlens_q_padded = \
                pad_thd_sequences_for_cp(
                    input_ids.unsqueeze(0),
                    labels.unsqueeze(0),
                    cu_seqlens_q,
                    divisibility_factor,
                    padding_token_id=pid,
                    padding_label_id=label_pad
                )
    positional_ids_padded = generate_positional_ids_for_cp(cu_seqlens_q_padded, divisibility_factor=divisibility_factor)
    expected_input_ids = torch.tensor([
                1, 1, 1, 1, 1, 1, 1, pid,
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, pid,
                3, 3, 3, 3, 3, pid, pid, pid
            ])

    expected_labels = torch.tensor([
        10, 11, 12, 13, 14, 15, 16, label_pad,
        20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, label_pad,
        5, 6, 7, 8, 9, label_pad, label_pad, label_pad
    ])

    expected_positional_ids = torch.tensor([
        0, 1, 2, 3, 4, 5, 6, 7,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        0, 1, 2, 3, 4, 5, 6, 7
    ])

    expected_cu_seqlens_padded = torch.tensor([0, 8, 20, 28])

    torch.equal(input_ids_padded, expected_input_ids)
    torch.equal(labels_padded, expected_labels)
    torch.equal(positional_ids_padded, expected_positional_ids)
    torch.equal(cu_seqlens_q_padded, expected_cu_seqlens_padded)

    # Now we have our data ready to go.
    # IMPORTANT: get_batch_on_this_cp_rank only needs batch dim for the *_padded keys
    batch = {
        "input_ids": input_ids.to(torch.int64), # Keep 1D for now
        "input_ids_padded": input_ids_padded.unsqueeze(0).to(torch.int64), # Add batch dim: [1, seq_len]
        "labels": labels.to(torch.int64), # Keep 1D for now
        "labels_padded": labels_padded.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "position_ids": positional_ids.to(torch.int64), # Keep 1D for now
        "position_ids_padded": positional_ids_padded.unsqueeze(0).to(torch.int64), # [1, seq_len]
        "cu_seqlens_q": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seqlens_kv": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seqlens_q_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "cu_seqlens_kv_padded": cu_seqlens_q_padded.to(torch.int32), # Keep 1D - int32
        "pad_between_seqs": True,
        "max_seqlen_q": 8,
        "max_seqlen_kv": 8,
    }
    return batch


def collect_gradients(model, layer_patterns=None, max_params=10):
    """Collect gradients for specific model parameters.

    Args:
        model: The PyTorch model
        layer_patterns: List of strings to match parameter names (e.g., ['attention', 'layer.0'])
        max_params: Maximum number of parameters to collect (to avoid huge output)

    Returns:
        Dictionary mapping parameter names to gradient tensors
    """
    grads = {}
    param_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # If patterns specified, only include matching parameters
            if layer_patterns:
                if not any(pattern in name for pattern in layer_patterns):
                    continue

            # Clone the gradient to avoid issues with autograd
            grads[name] = param.grad.clone().detach()
            param_count += 1

            # Limit output size
            if param_count >= max_params:
                break

    return grads
