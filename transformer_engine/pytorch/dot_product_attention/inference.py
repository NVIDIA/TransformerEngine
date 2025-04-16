# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Inference"""
import logging
from collections import OrderedDict, defaultdict
from typing import Optional, List
from einops import rearrange

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat

__all__ = ["InferenceParams", "KVCacheManager", "NonPagedKVCacheManager", "PagedKVCacheManager"]


class KVCacheManager:
    """Base KV cache manager"""

    def __init__(self):
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
        step_dict: OrderedDict,  # pylint: disable=unused-argument
    ):
        """Update tracked sequences and prepare for step()"""
        return self.sequences

    def step(
        self,
        layer_number: int,
        new_k: torch.Tensor,  # pylint: disable=unused-argument
        new_v: torch.Tensor,  # pylint: disable=unused-argument
        cu_new_seqlens: torch.Tensor,  # pylint: disable=unused-argument
        cu_cached_seqlens: torch.Tensor,  # pylint: disable=unused-argument
        qkv_format: str,  # pylint: disable=unused-argument
    ):
        """Copy the new tokens to KV cache"""
        return self.cache[layer_number]


class InferenceParams:
    """
    KV caching for inference. The memory allocation of the caches and the copying of new tokens
    to the cache take place at the following locations.::

      class TransformerLayer:
          class MultiHeadAttention:
              if self.layer_number not in inference_params.cache_manager.cache:
                  inference_params.allocate_memory(self.layer_number)
              class DotProductAttention:
                  if inference_params is not None:
                      k_cache, v_cache, new_qkv_format = inference_params.step(
                          new_k, new_v, qkv_format)
                  output = attention(new_q, k_cache, v_cache, new_qkv_format)

    allocate_memory() can be called outside the model, independently. step() can take three formats,
    qkv_format = {'bshd', 'sbhd', 'thd'}. It converts new_k and new_v to 'bshd' in both
    NonPagedKVCacheManager and PagedKVCacheManager. The format of new_q may change depending on the
    backend. If it is unchanged, we would have new_qkv_format = {'bshd', 'sbhd_2bshd', 'thd_2bshd'}.
    A standard KV caching workflow for inference is as follows.::

      model = [TransformerLayer() for _ in range(num_layers)]
      # initialize InferenceParams, e.g. with PagedKVCacheManager
      inference_params = InferenceParams(..., is_paged=True)
      # inference loop
      for i in range(num_iters):
          # get info for iteration i, e.g. seq_ids = [0, 2, 3], step_lens = [10, 1, 1]
          step_dict = OrderedDict(zip(seq_ids, step_lens))
          # update inference_params' state
          inference_params.pre_step(step_dict)
          # run iteration
          output = model(
                ...,
                attn_mask_type="padding_causal",
                cu_seqlens_q=cu_seqlens_new_q,
                cu_seqlens_kv=cu_seqlens_new_kv,
                inference_params=inference_params,
                )
          # get output tokens based on qkv_format
          # 'bshd': output = output[:,step_dict.values()-1]
          # 'sbhd': output = output[step_dict.values()-1,:]
          # 'thd' : output = output[cu_seqlens_new_q[j+1]-1], j=0,...b-1


    Parameters
    ----------
    max_batch_size: int
        Maximum batch size in inference
    max_sequence_length: int
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
    max_ctx_len: int, default = None
        Maximum context length in inference. 1 <= max_ctx_len <= max_sequence_length.
    qkv_format: str, default = "bshd"
        Format of the incoming query/key/value tensors in current iteration
    custom_cache_manager: KVCacheManager, default = None
        Custom cache manager, with KVCacheManager as the base class.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_sequence_length: int,
        num_heads_kv: int = None,
        head_dim_k: int = None,
        dtype: torch.dtype = None,
        head_dim_v: int = None,
        is_paged: bool = False,
        total_num_pages: int = None,
        page_size: int = None,
        max_ctx_len: int = None,
        qkv_format: str = "bshd",
        custom_cache_manager: KVCacheManager = None,
    ):
        self.max_batch_size = max_batch_size
        self.max_sequence_length = max_sequence_length
        assert all(x is not None for x in [num_heads_kv, head_dim_k, dtype]), (
            "num_heads_kv, head_dim_k, and dtype are required for InferenceParams since Transformer"
            " Engine 2.2."
        )
        self.num_heads_kv = num_heads_kv
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k
        self.is_paged = is_paged

        if not self.is_paged:
            cache_manager = (
                custom_cache_manager if custom_cache_manager is not None else NonPagedKVCacheManager
            )
            self.cache_manager = cache_manager(
                max_batch_size=self.max_batch_size,
                max_seqlen=self.max_sequence_length,
                num_heads=self.num_heads_kv,
                head_dim_k=self.head_dim_k,
                dtype=self.dtype,
                head_dim_v=self.head_dim_v,
            )
        else:
            assert page_size is not None, "Paged KV cache requires page_size is not None."
            self.page_size = page_size
            assert (
                max_sequence_length % page_size == 0
            ), "Paged KV cache requires max_sequence_length % page_size = 0."
            max_pages_per_seq = max_sequence_length // page_size
            assert (
                total_num_pages == self.max_batch_size * max_pages_per_seq
            ), "Paged KV cache requires total_num_pages = max_batch_size * max_pages_per_seq."
            self.total_num_pages = total_num_pages

            cache_manager = (
                custom_cache_manager if custom_cache_manager is not None else PagedKVCacheManager
            )
            self.cache_manager = cache_manager(
                total_num_pages=self.total_num_pages,
                page_size=self.page_size,
                num_heads=self.num_heads_kv,
                head_dim_k=self.head_dim_k,
                dtype=self.dtype,
                max_batch_size=self.max_batch_size,
                max_seqlen=self.max_sequence_length,
                head_dim_v=self.head_dim_v,
            )

        if qkv_format == "thd":
            assert max_ctx_len is not None, "max_ctx_len is required when qkv_format=thd!"
            self.max_ctx_len = max_ctx_len

        self.cache_qkv_format = "bshd"
        self.input_qkv_format = qkv_format
        if self.input_qkv_format == self.cache_qkv_format:
            self.output_qkv_format = self.cache_qkv_format
        else:
            self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        self.sequences_pre_step = OrderedDict()
        self.sequences = OrderedDict()
        self.batch_size = 0

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

    def reset(self):
        """Reset InferenceParams state"""
        self.sequences = OrderedDict()
        self.cache_manager.reset()

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
            f"max_seqlen={self.max_sequence_length}, "
            f"num_heads={self.num_heads_kv}, "
            f"head_dim_k={self.head_dim_k}, "
            f"head_dim_v={self.head_dim_v}"
        )

    def allocate_memory(self, layer_number: int):
        """
        Allocate memory for the cache. For layer layer_number,
        - NonPagedKVCacheManager:
          - K cache: [max_batch_size, max_sequence_length, num_heads_kv, head_dim_k]
          - V cache: [max_batch_size, max_sequence_length, num_heads_kv, head_dim_v]
        - PagedKVCacheManager:
          - K cache: [total_num_pages, page_size, num_heads_kv, head_dim_k]
          - V cache: [total_num_pages, page_size, num_heads_kv, head_dim_v]
        """
        self.cache_manager.allocate_memory(layer_number)

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        self.batch_size = len(step_dict)

        self.sequences = self.cache_manager.pre_step(step_dict)
        # track the pre-step seqlens for the next layer in the model
        self.sequences_pre_step = OrderedDict()
        for k, v in self.sequences.items():
            self.sequences_pre_step[k] = v - step_dict[k]

        seqlens_q = list(step_dict.values())
        cu_seqlens_q = [0] + [sum(seqlens_q[:i]) for i in range(1, self.batch_size + 1)]
        cu_seqlens_q = cu_seqlens_q + [cu_seqlens_q[-1]] * (self.max_batch_size - self.batch_size)
        self.cu_seqlens_q.copy_(torch.Tensor(cu_seqlens_q).to(dtype=torch.int32, device="cpu"))

        seqlens_kv = list(self.sequences.values())
        cu_seqlens_kv = [0] + [sum(seqlens_kv[:i]) for i in range(1, self.batch_size + 1)]
        cu_seqlens_kv = cu_seqlens_kv + [cu_seqlens_kv[-1]] * (
            self.max_batch_size - self.batch_size
        )
        self.cu_seqlens_kv.copy_(torch.Tensor(cu_seqlens_kv).to(dtype=torch.int32, device="cpu"))

    def get_seqlens_pre_step(self):
        """Get cached sequence lengths before the stepping"""
        return torch.Tensor(list(self.sequences_pre_step.values())).to(
            dtype=torch.int32, device="cpu"
        )

    def convert_paged_to_nonpaged(self, layer_number: int):
        """
        Convert k_cache and v_cache from paged to non-paged format.

        Parameters
        ----------
        layer_number: int
            Layer number of attention in the model

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

        new_k_cache = new_k_cache[: self.batch_size].contiguous()
        new_v_cache = new_v_cache[: self.batch_size].contiguous()

        return new_k_cache, new_v_cache

    def step(
        self,
        layer_number: int,
        new_k: torch.Tensor,
        new_v: torch.Tensor,
        qkv_format: str,
    ):
        """
        Copy new KV tokens to the cache.

        Parameters
        ----------
        layer_number: int
            Layer number of attention in the model
        new_k: torch.Tensor
            New key tokens for layer_number in current inference iteration
        new_v: torch.Tensor
            New value tokens for layer_number in current inference iteration
        qkv_format: str
            Format of new_q, new_k and new_v tensors, {'bshd', 'sbhd', 'thd'}

        Returns
        -------
        k_cache: torch.Tensor
            Full key tensor containing both previous and current key tokens
        v_cache: torch.Tensor
            Full value tensor containing both previous and current value tokens
        cu_seqlens_q: torch.Tensor
            Updated cumulative sequence lengths for query, [batch_size + 1]
        cu_seqlens_kv: torch.Tensor
            Updated cumulative sequence lengths for key and value, [batch_size + 1]
        max_seqlen_q: int
            Update maximum sequence length for query
        max_sequence_length: int
            Update maximum sequence length for key and value
        qkv_format: str
            Updated qkv_format, e.g. 'thd' format becomes 'thd_2bshd' after step()
        """
        self.input_qkv_format = qkv_format
        if self.input_qkv_format == self.cache_qkv_format:
            self.output_qkv_format = self.cache_qkv_format
        else:
            self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        k_cache, v_cache = self.cache_manager.step(
            layer_number,
            new_k,
            new_v,
            self.cu_seqlens_q,
            self.cu_seqlens_kv,
            qkv_format,
        )

        return (
            k_cache,
            v_cache,
            self.cu_seqlens_q,
            self.cu_seqlens_kv,
            self.max_sequence_length,
            self.output_qkv_format,
        )


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
        super().__init__()
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
        self.batch_indices = torch.zeros(
            self.max_batch_size,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )
        # after re-indexing, batch indices are always [0, ..., b-1]
        self.batch_indices_post_step = torch.range(
            0,
            self.max_batch_size - 1,
            dtype=torch.int32,
            device=torch.cuda.current_device(),
        )

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

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        # Track unfinished sequences' indices in the batch, e.g.
        # at t-1, seq_ids = [0, 1, 2, 3]; at t, seq_ids = [0, 2, 3] since seq_id 1 is finished
        # step() re-indexes k_cache and v_cache using batch_indices = [0, 2, 3, 1] so that
        # they are contiguous and match the indexing in q
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
            batch_size,
            ctx_len,
            self.max_seqlen,
            1,
            True,
        )

        k_cache = k_cache[:batch_size]
        v_cache = v_cache[:batch_size]

        return k_cache, v_cache


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
        super().__init__()
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
        self.page_table = torch.zeros(
            self.max_batch_size, self.max_pages_per_seq, dtype=torch.int32, device="cuda"
        )

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
        k_cache = torch.zeros(
            self.total_num_pages,
            self.page_size,
            self.num_heads,
            self.head_dim_k,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
        )
        v_cache = torch.zeros(
            self.total_num_pages,
            self.page_size,
            self.num_heads,
            self.head_dim_v,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
        )
        self.cache[layer_number] = (k_cache, v_cache)

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
            batch_size,
            ctx_len,
            self.max_seqlen,
            self.max_pages_per_seq,
            False,
        )

        return k_cache, v_cache
