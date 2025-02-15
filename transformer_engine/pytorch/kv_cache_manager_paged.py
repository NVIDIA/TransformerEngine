# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Paged KV Cache Manager."""
from collections import defaultdict, OrderedDict
from typing import List, Optional, Dict
import logging

import torch
import transformer_engine_torch as tex
from transformer_engine.pytorch.kv_cache_manager import KVCacheManager
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat


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
    """
    Paged KV cache manager. It supports a set of utilities including adding and removing
    sequences, and copying new key/value tokens to the cache. Users can overwrite this class
    for more custom implementations.
    """

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
        #is_cuda_graph: bool = False,
    ):
        """Initialize the KV cache"""
        self.total_num_pages = total_num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.max_batch_size = max_batch_size
        self.max_seqlen = max_seqlen
        self.max_pages_per_seq = max_seqlen // self.page_size
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k
        #self.is_cuda_graph = is_cuda_graph

        # sequences contained in the kv cache, {seq_id: seq_len}
        self.sequences = OrderedDict()
        # kv cache, cache[layer_number] = (k_cache, v_cache)
        self.cache = {}
        # free pages allowed to allocate, [Page(),...]
        self.free_pages = []
        for i in range(self.total_num_pages):
            self.free_pages.append(Page(i))
        # allocated pages, {seq_id: [page_id,...]}
        self.allocated_pages = defaultdict(list)
        # page table, [batch_size, max_pages_per_seq]
        self.page_table = None

    def allocate_memory(self, layer_number):
        """Allocate memory for the KV cache"""
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
        logger = logging.getLogger("PagedAttention")
        logger.debug("cache status:")
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

    def get_page_token_offsets(self, seqlen: int):
        """Get the relevant page index and token index for a given sequence length"""
        page_offset = seqlen // self.page_size
        token_offset = seqlen % self.page_size
        return (page_offset, token_offset)

    def get_page_table(self, sequences: List[int]):
        """Get the page table, in shape [batch_size, max_pages_per_seq]"""
        page_table = torch.Tensor(
            [
                self.get_page_list(seq) + [0] * (self.max_pages_per_seq - self.get_page_count(seq))
                for seq in sequences
            ]
        ).to(dtype=torch.int32, device="cpu")
        self.page_table[: self.get_sequence_count()].copy_(page_table)
        #if self.is_cuda_graph:
        #    self.page_table[: self.get_sequence_count()].copy_(page_table)
        #else:
        #    self.page_table = page_table.to(device="cuda")
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
        step_dict: Dict[List, List],
        ):
        batch_size = len(step_dict)
        step_lens = list(step_dict.values())
        cu_seqlens = [0] + [sum(step_lens[:i]) for i in range(1, batch_size + 1)]

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
        k: torch.Tensor,
        v: torch.Tensor,
        #step_dict: OrderedDict,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_kv: torch.Tensor,
        qkv_format: str,
    ):
        """
        Update the paged KV cache for a given inference iteration.
        For more details, please refer to InferenceParams.update_cache().

        Parameters
        ----------
        layer_number: int
            The layer number of kv cache to operate on
        k: torch.Tensor
            A batch of new key tokens for the current iteration
        v: torch.Tensor
            A batch of new value tokens for the current iteration
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
            self.page_table, step_lens, seq_lens,
            QKVFormat[qkv_format],
            self.num_heads, self.head_dim_k, self.head_dim_v,
            batch_size, ctx_len, self.max_seqlen, self.max_pages_per_seq, False)
        page_table = self.page_table[:batch_size]

        return k_cache, v_cache, page_table
