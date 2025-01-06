# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Paged KV Cache Manager."""
from collections import defaultdict, OrderedDict
from typing import List, Optional
import logging

import torch


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


class PagedKVCacheManager:
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
        is_cuda_graph: bool = False,
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
        self.is_cuda_graph = is_cuda_graph

        # sequences contained in the kv cache, {seq_id: seq_len}
        self.sequences = OrderedDict()
        # kv cache, cache[layer_number] = (k_cache, v_cache)
        self.cache = {}
        # free pages allowed to allocate, [Page(),...]
        self.free_pages = []
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
        for i in range(self.total_num_pages):
            self.free_pages.append(Page(i))
        if self.is_cuda_graph:
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
        if self.is_cuda_graph:
            self.page_table[: self.get_sequence_count()].copy_(page_table)
        else:
            self.page_table = page_table.to(device="cuda")
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

    def step(
        self,
        layer_number: int,
        k: torch.Tensor,
        v: torch.Tensor,
        step_dict: OrderedDict,
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

        # Copy new key and value tenosrs to the cache
        seqlens = list(self.sequences.values())
        packed_k = torch.Tensor([]).to(dtype=k.dtype, device=k.device)
        packed_v = torch.Tensor([]).to(dtype=v.dtype, device=v.device)
        for i in range(batch_size):
            if qkv_format == "bshd":
                packed_k = torch.cat([packed_k, k[i, : step_lens[i], :, :]], dim=0)
                packed_v = torch.cat([packed_v, v[i, : step_lens[i], :, :]], dim=0)
            if qkv_format == "sbhd":
                packed_k = torch.cat([packed_k, k[: step_lens[i], i, :, :]], dim=0)
                packed_v = torch.cat([packed_v, v[: step_lens[i], i, :, :]], dim=0)
        if qkv_format == "thd":
            packed_k = k
            packed_v = v
        k_cache, v_cache = self.cache[layer_number]
        for i, seq in enumerate(step_dict.keys()):
            page_list = self.get_page_list(seq)
            start_page, start_token = self.get_page_token_offsets(seqlens[i] - step_lens[i])
            end_page, end_token = self.get_page_token_offsets(seqlens[i])
            if start_page == end_page:
                page_id = page_list[start_page]
                k_cache[page_id, start_token:end_token, :, :] = packed_k[
                    cu_seqlens[i] : cu_seqlens[i + 1], :, :
                ]
                v_cache[page_id, start_token:end_token, :, :] = packed_v[
                    cu_seqlens[i] : cu_seqlens[i + 1], :, :
                ]
            else:
                start_offset = 0
                end_offset = 0
                for j in range(start_page, end_page + 1):
                    if not (j == end_page and end_token == 0):
                        start_token_j = start_token if j == start_page else 0
                        end_token_j = end_token if j == end_page else self.page_size
                        page_id = page_list[start_page]
                        end_offset = end_token_j - start_token_j
                        k_cache[page_id, start_token_j:end_token_j, :, :] = packed_k[
                            cu_seqlens[i] + start_offset : cu_seqlens[i] + end_offset, :, :
                        ]
                        v_cache[page_id, start_token_j:end_token_j, :, :] = packed_v[
                            cu_seqlens[i] + start_offset : cu_seqlens[i] + end_offset, :, :
                        ]
                        start_offset = start_offset + end_offset

        # Get page table
        page_table = self.get_page_table(list(self.sequences.keys()))

        return k_cache, v_cache, page_table
