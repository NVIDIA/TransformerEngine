# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Inference."""
import os
from collections import OrderedDict, defaultdict
from typing import Optional, Dict, List
from einops import rearrange
import logging

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat

__all__ = ["InferenceParams", "KVCacheManager", "NonPagedKVCacheManager", "PagedKVCacheManager"]


class KVCacheManager:
    """Base KV cache manager"""

    def __init__(self, *args, **kwargs):
        """Initialize cache manager"""
        self.cache = {}
        self.sequences = OrderedDict()

    def reset(self):
        """Reset cache manager state"""
        self.sequences = OrderedDict()

    def allocate_memory(self, layer_number: int):
        """Allocate memory for the cache"""
        self.cache[layer_number] = (None, None)

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        return self.sequences

    def step(
        self,
        layer_number: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cu_new_seqlens: torch.Tensor,
        cu_cached_seqlens: torch.Tensor,
        qkv_format: str,
    ):
        """Copy the new tokens to KV cache"""
        return *self.cache[layer_number], None


class InferenceParams:
    """
    Inference parameters that are passed to the main model in order
    to efficiently cache previous tokens and reuse them for the current
    inference iteration.

    Parameters
    ----------
    max_batch_size: int
        Maximum batch size in inference
    max_seqlen_kv: int
        Maximum sequence length in inference
    num_heads_kv: int
        Number of attention heads in keys and values
    head_dim_k: int
        Head size for keys
    dtype: torch.dtype
        Data type of the KV cache
    head_dim_v: int, default = None
        Head size for values. If None, initialized as head_dim_k.
    is_paged: bool, default = False
        Whether the KV cache is paged (True) or non-paged (False)
    total_num_pages: int, default = None
        Total number of pages in the KV cache. Required for is_paged = True.
    page_size: int, default = None
        Page size of the KV cache. Required for is_paged = True.
    num_heads_q: int, default = None
        Number of attention heads in queries
    head_dim_q: int, default = None
        Head size for queries. Required for qkv_format = 'thd'.
    max_ctx_len: int, default = None
        Maximum context length in inference. 1 <= max_ctx_len <= max_seqlen_kv.
    qkv_format: str, default = "bshd"
        Format of the incoming query/key/value tensors in current iteration
    cache_manager: KVCacheManager, default = None
        Custom cache manager, with KVCacheManager as the base class.
    allow_query_conversion: bool, default = True
        InferenceParams only supports cache_qkv_format = 'bshd'. When qkv_format = {'sbhd', 'thd'},
        output_qkv_format = {'sbhd_2bshd', 'thd_2bshd'}, which are supported by FusedAttention but
        not by FlashAttention or UnfusedDotProductAttention.

        For performance, try allow_query_conversion = False first. If it errors out with "No dot
        product attention support for the provided inputs!", set allow_query_conversion = True.

        For functionality, set allow_query_conversion = True. InferenceParams converts query from
        {'sbhd', 'thd'} to 'bshd', and converts the output back to {'sbhd', 'thd'}. The cost is
        two transposes for qkv_format = 'sbhd', and one memory buffer (q_buffer) and two conversion
        kernels (reshape_q and reshape_o) for qkv_format = 'thd'.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seqlen_kv: int,
        num_heads_kv: int,
        head_dim_k: int,
        dtype: torch.dtype,
        head_dim_v: int = None,
        is_paged: bool = False,
        total_num_pages: int = None,
        page_size: int = None,
        num_heads_q: int = None,
        head_dim_q: int = None,
        max_ctx_len: int = None,
        qkv_format: str = "bshd",
        cache_manager: KVCacheManager = None,
        allow_query_conversion: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.max_seqlen_kv = max_seqlen_kv
        self.num_heads_kv = num_heads_kv
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k
        self.is_paged = is_paged
        _NVTE_FLASH_ATTN = int(os.getenv("NVTE_FLASH_ATTN", "1"))
        _NVTE_FUSED_ATTN = int(os.getenv("NVTE_FUSED_ATTN", "1"))
        _NVTE_UNFUSED_ATTN = int(os.getenv("NVTE_UNFUSED_ATTN", "1"))
        self.allow_query_conversion = allow_query_conversion and (
            _NVTE_FLASH_ATTN or _NVTE_UNFUSED_ATTN or not _NVTE_FUSED_ATTN
        )

        if not self.is_paged:
            cls = cache_manager if cache_manager is not None else NonPagedKVCacheManager
            self.cache_manager = cls(
                max_batch_size=self.max_batch_size,
                max_seqlen=self.max_seqlen_kv,
                num_heads=self.num_heads_kv,
                head_dim_k=self.head_dim_k,
                dtype=self.dtype,
                head_dim_v=self.head_dim_v,
            )
        else:
            assert page_size is not None, "Paged KV cache requires page_size is not None."
            assert (
                max_seqlen_kv % page_size == 0
            ), "Paged KV cache requires max_seqlen_kv % page_size = 0."
            max_pages_per_seq = max_seqlen_kv // page_size
            assert (
                total_num_pages == self.max_batch_size * max_pages_per_seq
            ), "Paged KV cache requires total_num_pages = max_batch_size * max_pages_per_seq."
            self.page_size = page_size
            self.max_seqlen_kv = max_seqlen_kv
            self.total_num_pages = total_num_pages

            cls = cache_manager if cache_manager is not None else PagedKVCacheManager
            self.cache_manager = cls(
                total_num_pages=self.total_num_pages,
                page_size=self.page_size,
                num_heads=self.num_heads_kv,
                head_dim_k=self.head_dim_k,
                dtype=self.dtype,
                max_batch_size=self.max_batch_size,
                max_seqlen=self.max_seqlen_kv,
                head_dim_v=self.head_dim_v,
            )

        if qkv_format == "thd":
            assert max_ctx_len is not None, "max_ctx_len is required when qkv_format=thd!"
            self.max_ctx_len = max_ctx_len
            if self.allow_query_conversion:
                # query is converted to 'bshd' for certain backends
                assert num_heads_q is not None, "num_heads_q is required when qkv_format=thd!"
                assert head_dim_q is not None, "head_dim_q is required when qkv_format=thd!"
                self.num_heads_q = num_heads_q
                self.head_dim_q = head_dim_q
                self.max_seqlen_q = max_ctx_len
                self.q_orig = {}
                self.q_buffer = {}

        # NonPagedKVCacheManager and PagedKVCacheManager only support 'bshd' cache
        self.cache_qkv_format = "bshd"
        self.input_qkv_format = qkv_format
        if self.input_qkv_format == self.cache_qkv_format or self.allow_query_conversion:
            self.output_qkv_format = self.cache_qkv_format
        else:
            self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        self.sequences_pre = OrderedDict()
        self.sequences = OrderedDict()
        self.step_dict = OrderedDict()
        self.batch_size = 0

        self.cu_seqlens_q = None
        self.cu_seqlens_kv = None

        self.is_output_right_aligned = False

    def reset(self):
        """Reset InferenceParams state"""
        self.sequences = OrderedDict()
        self.cache_manager.reset()
        if self.input_qkv_format == "thd" and self.allow_query_conversion:
            for layer_number in self.q_buffer:
                self.q_buffer[layer_number].fill_(0)

    def __repr__(self) -> str:
        if self.is_paged:
            return (
                f"dtype={self.dtype}, "
                f"is_paged={self.is_paged}, "
                f"total_pages={self.total_num_pages}, "
                f"page_size={self.page_size}, "
                f"num_heads={self.num_heads_kv}, "
                f"head_dim_k={self.head_dim_k}, "
                f"head_dim_v={self.head_dim_v}"
            )
        return (
            f"dtype={self.dtype}, "
            f"is_paged={self.is_paged}, "
            f"max_batch_size={self.max_batch_size}, "
            f"max_seqlen={self.max_seqlen_kv}, "
            f"num_heads={self.num_heads_kv}, "
            f"head_dim_k={self.head_dim_k}, "
            f"head_dim_v={self.head_dim_v}"
        )

    def allocate_memory(self, layer_number: int, qkv_format: str):
        """
        Allocate memory for the cache. For layer layer_number,
        - NonPagedKVCacheManager:
          - K cache: [max_batch_size, max_seqlen_kv, num_heads_kv, head_dim_k]
          - V cache: [max_batch_size, max_seqlen_kv, num_heads_kv, head_dim_v]
        - PagedKVCacheManager:
          - K cache: [total_num_pages, page_size, num_heads_kv, head_dim_k]
          - V cache: [total_num_pages, page_size, num_heads_kv, head_dim_v]
        """
        self.cache_manager.allocate_memory(layer_number)

        self.cu_seqlens_q = torch.zeros(
            self.max_batch_size + 1,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        self.cu_seqlens_kv = torch.zeros(
            self.max_batch_size + 1,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )

        if qkv_format == "thd" and self.allow_query_conversion:
            self.q_buffer[layer_number] = torch.zeros(
                self.max_batch_size,
                self.max_ctx_len,
                self.num_heads_q,
                self.head_dim_q,
                dtype=self.dtype,
                device=torch.cuda.current_device(),
            )

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        self.step_dict = step_dict
        self.batch_size = len(step_dict)

        self.sequences = self.cache_manager.pre_step(step_dict)
        for k, v in enumerate(self.sequences):
            self.sequences_pre[k] = self.sequences[k] - self.step_dict[k]

        actual_batch_size = len(step_dict)
        seqlens_q = list(step_dict.values())
        cu_seqlens_q = [0] + [sum(seqlens_q[:i]) for i in range(1, actual_batch_size + 1)]
        cu_seqlens_q = cu_seqlens_q + [cu_seqlens_q[-1]] * (self.max_batch_size - actual_batch_size)
        self.cu_seqlens_q.copy_(torch.Tensor(cu_seqlens_q).to(dtype=torch.int32, device="cpu"))

        seq_lens = list(self.sequences.values())
        cu_seqlens_kv = [0] + [sum(seq_lens[:i]) for i in range(1, actual_batch_size + 1)]
        cu_seqlens_kv = cu_seqlens_kv + [cu_seqlens_kv[-1]] * (
            self.max_batch_size - actual_batch_size
        )
        self.cu_seqlens_kv.copy_(torch.Tensor(cu_seqlens_kv).to(dtype=torch.int32, device="cpu"))

    def get_seqlens_pre_step(self):
        """Get cached sequence lengths for current iteration before adding step_dict.values"""
        return self.sequences_pre

    def convert_paged_to_nonpaged(self, layer_number: int, qkv_format: str):
        """
        Convert k_cache and v_cache from paged to non-paged format. This is used by the
        UnfusedDotProductAttention backend. Both k_cache and v_cache are assumed to be
        in 'bshd' format.

        Parameters
        ----------
        layer_number: int
            Layer number of attention in the model
        qkv_format: str
            Format of new_q, new_k and new_v tensors, {'bshd', 'sbhd', 'thd'}

        Returns
        -------
        k_cache: torch.Tensor
            Non-paged key cache tensor
        v_cache: torch.Tensor
            Non-paged value cache tensor
        """
        k_cache, v_cache = self.cache_manager.cache[layer_number]
        page_table = self.cache_manager.page_table
        batch_size = page_table.shape[0]
        actual_batch_size = len(self.step_dict)
        new_k_cache = rearrange(
            k_cache[page_table.flatten()],
            "(b npages) page_size ... -> b (npages page_size) ...",
            b=batch_size,
        )
        new_v_cache = rearrange(
            v_cache[page_table.flatten()],
            "(b npages) page_size ... -> b (npages page_size) ...",
            b=batch_size,
        )

        new_k_cache = new_k_cache.contiguous()
        new_v_cache = new_v_cache.contiguous()
        if qkv_format != "thd":
            new_k_cache = new_k_cache[:actual_batch_size]
            new_v_cache = new_v_cache[:actual_batch_size]

        return new_k_cache, new_v_cache

    def step(
        self,
        layer_number: int,
        new_q: torch.Tensor,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        qkv_format: str,
    ):
        """
        Copy the new KV tokens to the KV cache and reshape Q if necessary.

        Parameters
        ----------
        layer_number: int
            Layer number of attention in the model
        new_q: torch.Tensor
            New query tokens for layer_number in current inference iteration
        new_k: torch.Tensor
            New key tokens for layer_number in current inference iteration
        new_v: torch.Tensor
            New value tokens for layer_number in current inference iteration
        qkv_format: str
            Format of new_q, new_k and new_v tensors, {'bshd', 'sbhd', 'thd'}

        Returns
        -------
        q_buffer: torch.Tensor
            new_q reshaped in order to allow certain backends to execute
        k_cache: torch.Tensor
            Full key tensor containing both previous and current key tokens
        v_cache: torch.Tensor
            Full value tensor containing both previous and current value tokens
        page_table: torch.Tensor
            Page table for paged KV cache, [batch_size, max_pages_per_seq]. None for non-paged KV cache
        cu_seqlens_q: torch.Tensor
            Updated cumulative sequence lengths for query, [batch_size + 1]
        cu_seqlens_kv: torch.Tensor
            Updated cumulative sequence lengths for key and value, [batch_size + 1]
        max_seqlen_q: int
            Update maximum sequence length for query
        max_seqlen_kv: int
            Update maximum sequence length for key and value
        qkv_format: str
            Updated qkv_format, e.g. the input 'thd' format may become 'thd_2bshd' after step()
        """
        self.input_qkv_format = qkv_format
        if self.input_qkv_format == self.cache_qkv_format or self.allow_query_conversion:
            self.output_qkv_format = self.cache_qkv_format
        else:
            self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        q_buffer = new_q
        if qkv_format == "bshd":
            self.max_seqlen_q = new_q.shape[1]
            q_buffer = new_q.contiguous()
        if qkv_format == "sbhd":
            self.max_seqlen_q = new_q.shape[0]
            if self.allow_query_conversion:
                q_buffer = new_q.transpose(0, 1).contiguous()
        if qkv_format == "thd":
            self.max_seqlen_q = self.max_ctx_len
            if self.allow_query_conversion:
                q_buffer = self.q_buffer[layer_number]
                tex.reshape_q(
                    new_q,
                    self.q_buffer[layer_number],
                    self.cu_seqlens_q,
                    self.num_heads_q,
                    self.head_dim_q,
                    self.max_batch_size,
                    self.max_ctx_len,
                )
                self.q_orig[layer_number] = new_q

        k_cache, v_cache, page_table = self.cache_manager.step(
            layer_number,
            new_k,
            new_v,
            self.cu_seqlens_q,
            self.cu_seqlens_kv,
            qkv_format,
        )

        return (
            q_buffer,
            k_cache,
            v_cache,
            page_table,
            self.cu_seqlens_q,
            self.cu_seqlens_kv,
            self.max_seqlen_q,
            self.max_seqlen_kv,
            self.output_qkv_format,
        )

    def post_step(
        self,
        layer_number: int,
        output: torch.Tensor,
    ):
        """
        Process the attention output in order to return it to the original qkv_format.
        """
        if self.input_qkv_format == "bshd":
            output = output[: self.batch_size, : self.max_seqlen_q].contiguous()
        if self.input_qkv_format == "sbhd" and self.allow_query_conversion:
            output = output[: self.batch_size, : self.max_seqlen_q].transpose(0, 1).contiguous()
        if self.input_qkv_format == "thd" and self.allow_query_conversion:
            output_buffer = self.q_orig[layer_number]
            tex.reshape_o(
                output,
                output_buffer,
                self.cu_seqlens_q,
                self.num_heads_q,
                self.head_dim_q,
                self.batch_size,
                self.max_ctx_len,
                self.is_output_right_aligned,
            )
            output = output_buffer.view(output_buffer.shape[0], -1)

        return output


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


class Page:
    """A single page"""

    def __init__(self, page_id: int):
        """Initialize a page"""
        self.page_id = page_id
        self.allocated = 0

    def allocate_page(self):
        """Allocate a page"""
        self.allocated = True

    def deallocate_page(self):
        """Deallocate a page"""
        self.allocated = False


class PagedKVCacheManager(KVCacheManager):
    """Paged KV cache manager"""

    def __init__(
        self,
        total_num_pages: int,
        page_size: int,
        num_heads: int,
        head_dim_k: int,
        dtype: torch.dtype,
        max_batch_size: int,
        max_seqlen: int,
        head_dim_v: Optional[int] = None,
    ):
        """Initialize cache manager"""
        self.total_num_pages = total_num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seqlen = max_seqlen
        self.max_pages_per_seq = max_seqlen // self.page_size
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k

        # track sequences in the cache, {seq_id: seq_len}
        self.sequences = OrderedDict()
        # cache tensors, cache[layer_number] = (k_cache, v_cache)
        self.cache = {}
        # available pages, [Page(),...]
        self.free_pages = []
        for i in range(self.total_num_pages):
            self.free_pages.append(Page(i))
        # allocated pages, {seq_id: [page_id,...]}
        self.allocated_pages = defaultdict(list)
        # page table, [batch_size, max_pages_per_seq]
        self.page_table = None

    def reset(self):
        """Reset cache manager state"""
        self.sequences = OrderedDict()
        self.free_pages = []
        for i in range(self.total_num_pages):
            self.free_pages.append(Page(i))
        self.allocated_pages = defaultdict(list)
        self.page_table.fill_(0)

    def allocate_memory(self, layer_number):
        """Allocate memory for the cache"""
        k_cache = torch.empty(
            self.total_num_pages,
            self.page_size,
            self.num_heads,
            self.head_dim_k,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
        )
        v_cache = torch.empty(
            self.total_num_pages,
            self.page_size,
            self.num_heads,
            self.head_dim_v,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
        )
        self.cache[layer_number] = (k_cache, v_cache)

        self.page_table = torch.zeros(
            self.max_batch_size, self.max_pages_per_seq, dtype=torch.int32, device="cuda"
        )

    def print_cache(self):
        """Print KV cache status"""
        used_pages = [self.get_page_count(seq) for seq in self.sequences]
        logger = logging.getLogger("PagedKVCacheManager")
        logger.debug("Cache status:")
        logger.debug(
            "  total pages:     %s (used %s, free %s)",
            self.total_num_pages,
            sum(used_pages),
            len(self.free_pages),
        )
        logger.debug("  total sequences: %s", self.get_sequence_count())
        for i, seq in enumerate(self.sequences):
            logger.debug(
                "  >> batch index %s: seq_id %s, num_tokens %s, num_pages %s, page_list %s",
                i,
                seq,
                self.get_sequence_lengths()[i],
                self.get_page_count(seq),
                self.get_page_list(seq),
            )

    def get_sequence_count(self):
        """Get the total number of sequences in the KV cache"""
        return len(self.sequences)

    def get_sequence_lengths(self):
        """Get the list of sequence lengths in the KV cache"""
        return list(self.sequences.values())

    def has_free_page(self) -> bool:
        """Whether the page pool has any free pages left"""
        return len(self.free_pages) > 0

    def get_page_count(self, seq: int):
        """Get the number of pages allocated to a sequence"""
        return len(self.allocated_pages[seq])

    def get_page_list(self, seq: int):
        """Get the list of pages allocated to a sequence"""
        return [x.page_id for x in self.allocated_pages[seq]]

    def get_page_table(self, sequences: List[int]):
        """Get the page table, in shape [batch_size, max_pages_per_seq]"""
        page_table = torch.Tensor(
            [
                self.get_page_list(seq) + [0] * (self.max_pages_per_seq - self.get_page_count(seq))
                for seq in sequences
            ]
        ).to(dtype=torch.int32, device="cpu")
        self.page_table[: self.get_sequence_count()].copy_(page_table)
        return self.page_table

    def allocate_page(self, seq: int):
        """Allocate a new page to a sequence"""
        if not self.has_free_page():
            raise RuntimeError("KV cache is full!")
        page = self.free_pages.pop(0)
        page.allocate_page()
        self.allocated_pages[seq].append(page)

    def allocate_sequence(self, seq: int, context_len: int):
        """Add a new sequence to the cache"""
        num_pages = context_len // self.page_size
        if context_len % self.page_size > 0:
            num_pages = num_pages + 1
        for _ in range(num_pages):
            self.allocate_page(seq)

    def deallocate_sequence(self, seq: int):
        """Deallocate all the pages for a sequence"""
        for page in self.allocated_pages[seq]:
            page.deallocate_page()
            if not page.allocated:
                self.free_pages.append(page)
        self.allocated_pages.pop(seq)

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        # Remove finished sequences and advance unfinished sequences
        unfinished_seqs = self.sequences.keys() & step_dict.keys()
        finished_seqs = self.sequences.keys() - unfinished_seqs
        for seq in finished_seqs:
            self.sequences.pop(seq)
            self.deallocate_sequence(seq)
        for seq in unfinished_seqs:
            if self.sequences[seq] % self.page_size == 0 and self.sequences[seq] < self.max_seqlen:
                self.allocate_page(seq)
            self.sequences[seq] += 1

        # Add new sequences
        new_seqs = step_dict.keys() - self.sequences.keys()
        for seq in new_seqs:
            self.sequences[seq] = step_dict[seq]
            self.allocate_sequence(seq, step_dict[seq])

        # Get page table
        self.page_table = self.get_page_table(list(self.sequences.keys()))

        return self.sequences

    def step(
        self,
        layer_number: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        cu_new_seqlens,
        cu_cached_seqlens,
        qkv_format: str,
    ):
        """
        Copy the new tokens to the paged KV cache.

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
            Page table for current iteration, in shape [batch_size, max_pages_per_seq]
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
            self.page_table,
            cu_new_seqlens,
            cu_cached_seqlens,
            QKVFormat[qkv_format],
            self.num_heads,
            self.head_dim_k,
            self.head_dim_v,
            batch_size,
            ctx_len,
            self.max_seqlen,
            self.max_pages_per_seq,
            False,
        )

        page_table = self.page_table[:batch_size]

        return k_cache, v_cache, page_table
