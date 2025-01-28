# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Non-Paged KV Cache Manager."""
from collections import OrderedDict
from typing import Optional
import torch


class NonPagedKVCacheManager:
    """
    The non-paged KV cache manager.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seqlen: int,
        num_heads: int,
        head_dim_k: int,
        dtype: torch.dtype,
        head_dim_v: Optional[int] = None,
        is_cuda_graph: bool = False,
    ):
        """Initialize the KV cache"""
        self.max_batch_size = max_batch_size
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k
        self.is_cuda_graph = is_cuda_graph

        # sequences contained in the kv cache, {seq_id: seq_len}
        self.sequences = OrderedDict()
        # KV cache tuple (k_cache, v_cache)
        self.cache = {}

    def allocate_memory(self, layer_number):
        """Allocate memory for the KV cache"""
        k_cache = torch.zeros(
            self.max_batch_size,
            self.max_seqlen,
            self.num_heads,
            self.head_dim_k,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
        )
        v_cache = torch.zeros(
            self.max_batch_size,
            self.max_seqlen,
            self.num_heads,
            self.head_dim_v,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
        )
        self.cache[layer_number] = (k_cache, v_cache)

    def step(
        self,
        layer_number,
        k: torch.Tensor,
        v: torch.Tensor,
        step_dict: OrderedDict,
        qkv_format: str,
    ):
        """
        Update the non-paged KV cache for a given inference iteration.
        For more details, please refer to InferenceParams.update_cache().

        Parameters
        ----------
        layer_number: int
            The layer number of kv cache to operate on
        k: torch.Tensor
            The new key tokens for the current iteration
        v: torch.Tensor
            The new value tokens for the current iteration
        step_dict: OrderedDict
            The {seq_id: step_len} information for the new inference step
        qkv_format: str
            The format of the new key/value tensors, {'bshd', 'sbhd', 'thd'}

        Returns
        -------
        k_cache: torch.Tensor
            The key cache tensor containing previous and the current tokens
        v_cache: torch.Tensor
            The value cache tensor containing previous and the current tokens
        """
        k_cache, v_cache = self.cache[layer_number]
        prev_batch_size = len(self.sequences)
        batch_size = len(step_dict)

        # Reorder cache
        unfinished_seqs = self.sequences.keys() & step_dict.keys()
        finished_seqs = self.sequences.keys() - unfinished_seqs
        unfinished_indices = [i for i, j in enumerate(self.sequences) if j in unfinished_seqs]
        finished_indices = [i for i, j in enumerate(self.sequences) if j in finished_seqs]
        batch_indices = (
            unfinished_indices
            + finished_indices
            + list(range(prev_batch_size, self.max_batch_size))
        )
        new_k_cache = k_cache[batch_indices, :]
        new_v_cache = v_cache[batch_indices, :]
        new_k_cache = new_k_cache.contiguous()
        new_v_cache = new_v_cache.contiguous()

        # Advance unfinished sequences
        for i in unfinished_seqs:
            self.sequences[i] += 1

        # Remove finished sequences
        for i in finished_seqs:
            self.sequences.pop(i)

        # Add new sequences
        new_seqs = step_dict.keys() - self.sequences.keys()
        for i in new_seqs:
            self.sequences[i] = step_dict[i]

        # Copy new key/value tokens to cache
        step_lens = list(step_dict.values())
        cu_seqlens = [0] + [sum(step_lens[:i]) for i in range(1, batch_size + 1)]
        for i, seq in enumerate(self.sequences):
            seq_s = self.sequences[seq] - step_dict[seq]
            seq_e = self.sequences[seq]
            if qkv_format == "bshd":
                new_k_cache[i, seq_s:seq_e, :, :] = k[i, : step_dict[seq], :, :]
                new_v_cache[i, seq_s:seq_e, :, :] = v[i, : step_dict[seq], :, :]
            if qkv_format == "sbhd":
                new_k_cache[i, seq_s:seq_e, :, :] = k[: step_dict[seq], i, :, :]
                new_v_cache[i, seq_s:seq_e, :, :] = v[: step_dict[seq], i, :, :]
            if qkv_format == "thd":
                new_k_cache[i, seq_s:seq_e, :, :] = k[cu_seqlens[i] : cu_seqlens[i + 1], :, :]
                new_v_cache[i, seq_s:seq_e, :, :] = v[cu_seqlens[i] : cu_seqlens[i + 1], :, :]
        self.cache[layer_number] = (new_k_cache, new_v_cache)

        # Return full key/value tensors for attention calculation
        if self.is_cuda_graph:
            # [max_batch_size, max_seqlen_kv, num_heads_kv, head_dim_kv]
            return new_k_cache, new_v_cache, None

        # [actual_batch_size, max_seqlen_kv, num_heads_kv, head_dim_kv]
        new_k_cache = new_k_cache[:batch_size].contiguous()
        new_v_cache = new_v_cache[:batch_size].contiguous()
        return new_k_cache, new_v_cache, None
