# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Non-Paged KV Cache Manager"""
from collections import OrderedDict
from typing import Optional
import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.kv_cache_manager import KVCacheManager
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat


class NonPagedKVCacheManager(KVCacheManager):
    """Non-paged KV cache manager"""

    def __init__(
        self,
        max_batch_size: int,
        max_seqlen: int,
        num_heads: int,
        head_dim_k: int,
        dtype: torch.dtype,
        head_dim_v: Optional[int] = None,
    ):
        """Initialize cache manager"""
        self.max_batch_size = max_batch_size
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k

        # track sequences in the cache, {seq_id: seq_len}
        self.sequences = OrderedDict()
        # cache tensors, cache[layer_number] = (k_cache, v_cache)
        self.cache = {}
        # track sequence indices in the batch in order to re-index k_cache and v_cache
        self.batch_indices = None

    def allocate_memory(self, layer_number):
        """Allocate memory for the cache"""
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

        self.batch_indices = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        # Track unfinished sequences' indices in the batch, e.g.
        # at t-1, seq_ids = [0, 1, 2, 3], and at t, seq_ids = [0, 2, 3], because seq_id 1 finished
        # batch_indices = [0, 2, 3, 1] is used in step() to re-index k_cache and v_cache so that
        # they are contiguous and match the sequence indexing in q.
        prev_batch_size = len(self.sequences)
        unfinished_seqs = self.sequences.keys() & step_dict.keys()
        finished_seqs = self.sequences.keys() - unfinished_seqs
        unfinished_indices = [i for i, j in enumerate(self.sequences) if j in unfinished_seqs]
        finished_indices = [i for i, j in enumerate(self.sequences) if j in finished_seqs]
        self.batch_indices.copy_(
            torch.Tensor(
                (
                    unfinished_indices
                    + finished_indices
                    + list(range(prev_batch_size, self.max_batch_size))
                )
            ).to(dtype=torch.int32, device="cpu")
        )

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

        return self.sequences

    def step(
        self,
        layer_number,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cu_new_seqlens,
        cu_cached_seqlens,
        qkv_format: str,
    ):
        """
        Copy the new tokens to the non-paged KV cache.

        Parameters
        ----------
        layer_number: int
            Layer number of attention in the model
        new_k: torch.Tensor
            New key tokens for layer_number in current inference iteration
        new_v: torch.Tensor
            New value tokens for layer_number in current inference iteration
        cu_new_seqlens: torch.Tensor
            Cumulative sequence lengths for new_k and new_v, in shape [batch_size + 1]
        cu_cached_seqlens: torch.Tensor
            Cumulative sequence lengths for k_cache and v_cache (after new tokens are copied in), in shape [batch_size + 1]
        qkv_format: str
            Format of new_k and new_v tensors, {'bshd', 'sbhd', 'thd'}

        Returns
        -------
        k_cache: torch.Tensor
            Full key tensor containing both previous and current key tokens
        v_cache: torch.Tensor
            Full value tensor containing both previous and current value tokens
        page_table: torch.Tensor
            None for non-paged KV cache
        """
        k_cache, v_cache = self.cache[layer_number]

        batch_size = self.max_batch_size
        ctx_len = 1
        if qkv_format == "bshd":
            batch_size = new_k.shape[0]
            ctx_len = new_k.shape[1]
        if qkv_format == "sbhd":
            batch_size = new_k.shape[1]
            ctx_len = new_k.shape[0]

        tex.copy_to_kv_cache(
            new_k,
            new_v,
            k_cache,
            v_cache,
            self.batch_indices,
            cu_new_seqlens,
            cu_cached_seqlens,
            QKVFormat[qkv_format],
            self.num_heads,
            self.head_dim_k,
            self.head_dim_v,
            batch_size,
            ctx_len,
            self.max_seqlen,
            1,
            True,
        )

        k_cache = k_cache[:batch_size]
        v_cache = v_cache[:batch_size]

        return k_cache, v_cache, None
