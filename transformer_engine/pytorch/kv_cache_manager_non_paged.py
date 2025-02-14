# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Non-Paged KV Cache Manager."""
from collections import OrderedDict
from typing import Optional, Dict, List
import torch
#from transformer_engine.pytorch.utils import StaticBufferAllocator
import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat

class KVCacheManager:
    """
    KV cache manager. This should be the base class for custom KV cache managers.
    """
    def __init__(self, *args, **kwargs):
        """Initialize the cache manager"""
        self.cache = {}
    def allocate_memory(self, layer_number: int):
        """Allocate memory for the cache"""
        self.cache[layer_number] = (None, None)
    def prepare(
        self,
        sequences: Dict[List, List],
        step_dict: Dict[List, List],
    ):
        """Prepare for step(). Update sequences with step_dict."""
        return sequences
    def step(
        self,
        layer_number: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        qkv_format: str,
    ):
        """Update the cache with new_k and new_v tokens"""
        return *self.cache[layer_number], None

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
        #is_cuda_graph: bool = False,
    ):
        """Initialize the KV cache"""
        self.max_batch_size = max_batch_size
        self.max_seqlen = max_seqlen
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k
        #self.is_cuda_graph = is_cuda_graph

        # sequences contained in the kv cache, {seq_id: seq_len}
        #self.sequences = OrderedDict()
        # KV cache tuple (k_cache, v_cache)
        self.cache = {}
        self.batch_indices = None
#        self._allocator = StaticBufferAllocator()
#
#    def alloc(self, size, dtype, device):
#        """
#            Allocated the buffer and works correctly with CUDA Graphs.
#        """
#        return self._allocator(size, dtype, device)

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

        #self.batch_indices = self.alloc(
        self.batch_indices = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
            )

    def prepare(
        self,
        sequences: Dict[List, List],
        step_dict: Dict[List, List],
    ):
        # TODO: remove
        self.sequences = sequences
        #self.step_dict = step_dict
        prev_batch_size = len(self.sequences)
        batch_size = len(step_dict)

        # Reorder cache
        unfinished_seqs = self.sequences.keys() & step_dict.keys()
        finished_seqs = self.sequences.keys() - unfinished_seqs
        unfinished_indices = [i for i, j in enumerate(self.sequences) if j in unfinished_seqs]
        finished_indices = [i for i, j in enumerate(self.sequences) if j in finished_seqs]
        self.batch_indices.copy_(torch.Tensor((
            unfinished_indices
            + finished_indices
            + list(range(prev_batch_size, self.max_batch_size))
        )).to(dtype=torch.int32, device="cpu"))
        print('self.batch_indices', self.batch_indices)

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
        #kk=k_cache.clone()
        #k_cache1 = kk[self.batch_indices].contiguous()
        #k_cache = k_cache[self.batch_indices].contiguous()
        #v_cache = v_cache[self.batch_indices].contiguous()
        step_lens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seq_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
        #h=self.num_heads #16
        #d=self.head_dim_k #64
        #b=self.max_batch_size #4
        max_ctx_len=1 #k.shape[1] if qkv_format in ["bshd", "sbhd"] else 1 #64
        if qkv_format == "bshd":
            max_ctx_len=k.shape[1]
        if qkv_format == "sbhd":
            max_ctx_len=k.shape[0]
        max_seq_len=self.max_seqlen #k_cache.shape[1] #64 #128
        max_ctx_tokens=k.shape[0]
        max_tokens=k_cache.shape[0]*k_cache.shape[1]
        print('kv shapes ', [x.shape for x in [k, v, k_cache, v_cache]])
        #print('step_lens ', step_lens)
        #print('seq_lens ', seq_lens)
        #print('self.batch_indices ', self.batch_indices)
        print('lensss ', max_ctx_len, max_seq_len, max_ctx_tokens, max_tokens)
        tex.copy_to_kv_cache_non_paged(
            k, v, k_cache, v_cache,
            self.batch_indices, step_lens, seq_lens,
            QKVFormat[qkv_format], self.num_heads, self.head_dim_k, self.head_dim_v, self.max_batch_size,
            max_ctx_len, max_seq_len)#, max_ctx_tokens, max_tokens)
        #print(k_cache1[0, :2, 0, :4])
        #print(k_cache1[1, :2, 0, :4])
        #print(k_cache[0, :2, 0, :4])
        #print(k_cache[1, :2, 0, :4])
        self.cache[layer_number] = k_cache, v_cache
        return k_cache, v_cache, None

#        #prev_batch_size = len(self.sequences)
#        #batch_size = len(step_dict)
#        batch_size = len(self.sequences)
#
#        ## Reorder cache
#        #unfinished_seqs = self.sequences.keys() & step_dict.keys()
#        #finished_seqs = self.sequences.keys() - unfinished_seqs
#        #unfinished_indices = [i for i, j in enumerate(self.sequences) if j in unfinished_seqs]
#        #finished_indices = [i for i, j in enumerate(self.sequences) if j in finished_seqs]
#        #batch_indices = (
#        #    unfinished_indices
#        #    + finished_indices
#        #    + list(range(prev_batch_size, self.max_batch_size))
#        #)
#        new_k_cache = k_cache[self.batch_indices, :]
#        new_v_cache = v_cache[self.batch_indices, :]
#        new_k_cache = new_k_cache.contiguous()
#        new_v_cache = new_v_cache.contiguous()
#
#        ## Advance unfinished sequences
#        #for i in unfinished_seqs:
#        #    self.sequences[i] += 1
#
#        ## Remove finished sequences
#        #for i in finished_seqs:
#        #    self.sequences.pop(i)
#
#        ## Add new sequences
#        #new_seqs = step_dict.keys() - self.sequences.keys()
#        #for i in new_seqs:
#        #    self.sequences[i] = step_dict[i]
#
#        # Copy new key/value tokens to cache
#        #step_lens = list(step_dict.values())
#        #cu_seqlens = [0] + [sum(step_lens[:i]) for i in range(1, batch_size + 1)]
#        cu_seqlens = cu_seqlens_q
#        step_lens = cu_seqlens[1:] - cu_seqlens[:-1]
#        seq_lens = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
#        #print('self.sequences', self.sequences)
#        #print('cu_seqlens_q', cu_seqlens_q)
#        #print('cu_seqlens_kv', cu_seqlens_kv)
#        #print('step_lens', step_lens)
#        for i, seq in enumerate(self.sequences.keys()):
#            print('kv cm non-paged i', i, 'seq', seq)
#            #seq_s = self.sequences[seq] - step_lens[i]
#            #seq_e = self.sequences[seq]
#            seq_s = seq_lens[i] - step_lens[i]
#            seq_e = seq_lens[i]
#            if qkv_format == "bshd":
#                print('bshd ', [x.device for x in [new_k_cache, step_lens]])
#                new_k_cache[i, seq_s:seq_e, :, :] = k[i, : step_lens[i], :, :]
#                new_v_cache[i, seq_s:seq_e, :, :] = v[i, : step_lens[i], :, :]
#            if qkv_format == "sbhd":
#                new_k_cache[i, seq_s:seq_e, :, :] = k[: step_lens[i], i, :, :]
#                new_v_cache[i, seq_s:seq_e, :, :] = v[: step_lens[i], i, :, :]
#            if qkv_format == "thd":
#                new_k_cache[i, seq_s:seq_e, :, :] = k[cu_seqlens[i] : cu_seqlens[i + 1], :, :]
#                new_v_cache[i, seq_s:seq_e, :, :] = v[cu_seqlens[i] : cu_seqlens[i + 1], :, :]
#        self.cache[layer_number] = (new_k_cache, new_v_cache)
#
#        # Return full key/value tensors for attention calculation
#        if self.is_cuda_graph:
#            # [max_batch_size, max_seqlen_kv, num_heads_kv, head_dim_kv]
#            return new_k_cache, new_v_cache, None
#
#        # [actual_batch_size, max_seqlen_kv, num_heads_kv, head_dim_kv]
#        new_k_cache = new_k_cache[:batch_size].contiguous()
#        new_v_cache = new_v_cache[:batch_size].contiguous()
#        return new_k_cache, new_v_cache, None
