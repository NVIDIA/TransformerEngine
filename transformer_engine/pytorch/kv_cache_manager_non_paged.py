# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Non-Paged KV Cache Manager."""
from collections import OrderedDict
from typing import Optional, Dict, List
import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.kv_cache_manager import KVCacheManager
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat

class NonPagedKVCacheManager(KVCacheManager):
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
    ):
        """Initialize the KV cache"""
        self.max_batch_size = max_batch_size
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k

        self.cache = {}
        self.sequences = OrderedDict()
        self.batch_indices = None

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

        self.batch_indices = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )

    def pre_step(
        self,
        step_dict: Dict[List, List],
    ):
        # Reorder cache
        prev_batch_size = len(self.sequences)
        unfinished_seqs = self.sequences.keys() & step_dict.keys()
        finished_seqs = self.sequences.keys() - unfinished_seqs
        unfinished_indices = [i for i, j in enumerate(self.sequences) if j in unfinished_seqs]
        finished_indices = [i for i, j in enumerate(self.sequences) if j in finished_seqs]
        self.batch_indices.copy_(torch.Tensor((
            unfinished_indices
            + finished_indices
            + list(range(prev_batch_size, self.max_batch_size))
        )).to(dtype=torch.int32, device="cpu"))

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
        k: torch.Tensor,
        v: torch.Tensor,
        #step_dict: OrderedDict,
        cu_seqlens_q,
        cu_seqlens_kv,
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
        step_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seq_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
        batch_size = self.max_batch_size
        ctx_len=1
        if qkv_format == "bshd":
            batch_size = k.shape[0]
            ctx_len=k.shape[1]
        if qkv_format == "sbhd":
            batch_size = k.shape[1]
            ctx_len=k.shape[0]
        tex.copy_to_kv_cache(
            k, v, k_cache, v_cache,
            self.batch_indices, step_lens, seq_lens,
            QKVFormat[qkv_format],
            self.num_heads, self.head_dim_k, self.head_dim_v,
            batch_size, ctx_len, self.max_seqlen, 1, True)
        k_cache = k_cache[:batch_size, :ctx_len]
        v_cache = v_cache[:batch_size, :ctx_len]
        return k_cache, v_cache, None
