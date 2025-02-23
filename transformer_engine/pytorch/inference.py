# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Inference."""
from collections import OrderedDict
from typing import Dict, List
from einops import rearrange

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat
from transformer_engine.pytorch.kv_cache_manager import KVCacheManager
from transformer_engine.pytorch.kv_cache_manager_paged import PagedKVCacheManager
from transformer_engine.pytorch.kv_cache_manager_non_paged import NonPagedKVCacheManager


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
        Head size for queries. Required for qkv_format = thd.
    max_ctx_len: int, default = None
        Maximum context length in inference. 1 <= max_ctx_len <= max_seqlen_kv.
    qkv_format: str, default = "bshd"
        Format of the incoming query/key/value tensors in current iteration
    cache_manager: KVCacheManager, default = None
        Custom cache manager, with KVCacheManager as the base class.
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
    ):
        self.max_batch_size = max_batch_size
        self.max_seqlen_kv = max_seqlen_kv
        self.num_heads_kv = num_heads_kv
        self.head_dim_k = head_dim_k
        self.dtype = dtype
        self.head_dim_v = head_dim_v if head_dim_v is not None else head_dim_k
        self.is_paged = is_paged

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
            # query is converted to 'bshd' for certain backends
            assert num_heads_q is not None, "num_heads_q is required when qkv_format=thd!"
            assert head_dim_q is not None, "head_dim_q is required when qkv_format=thd!"
            assert max_ctx_len is not None, "max_ctx_len is required when qkv_format=thd!"
            self.num_heads_q = num_heads_q
            self.head_dim_q = head_dim_q
            self.max_ctx_len = max_ctx_len
            self.max_seqlen_q = max_ctx_len
            self.q_orig = {}
            self.q_buffer = {}

        # NonPagedKVCacheManager and PagedKVCacheManager only support 'bshd' cache
        self.cache_qkv_format = "bshd"
        self.input_qkv_format = qkv_format
        self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        self.sequences_prev = OrderedDict()
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
        if self.input_qkv_format == "thd":
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

    def pre_step(
        self,
        step_dict: OrderedDict,
    ):
        """Update tracked sequences and prepare for step()"""
        self.step_dict = step_dict
        self.batch_size = len(step_dict)
        self.sequences_prev = self.sequences

        self.sequences = self.cache_manager.pre_step(step_dict)

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

        new_k_cache = new_k_cache[:actual_batch_size].contiguous()
        new_v_cache = new_v_cache[:actual_batch_size].contiguous()
        if qkv_format == "sbhd":
            new_k_cache = new_k_cache.transpose(0,1)
            new_v_cache = new_v_cache.transpose(0,1)
        if qkv_format == "thd":
            assert False, "UnfusedDotProductAttention does not support qkv_format=thd."

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
        self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        if qkv_format == "thd" and layer_number not in self.q_buffer:
            self.q_buffer[layer_number] = torch.zeros(
                self.max_batch_size,
                self.max_ctx_len,
                self.num_heads_q,
                self.head_dim_q,
                dtype=self.dtype,
                device=torch.cuda.current_device(),
            )

        if qkv_format == "bshd":
            q_buffer = new_q.contiguous()
            self.max_seqlen_q = q_buffer.shape[1]
        if qkv_format == "sbhd":
            q_buffer = new_q.transpose(0, 1).contiguous()
            self.max_seqlen_q = q_buffer.shape[1]
        if qkv_format == "thd":
            q_buffer = new_q
        #    self.q_orig[layer_number] = q
        #    self.max_seqlen_q = self.max_ctx_len

        #    q_buffer = self.q_buffer[layer_number]
        #    step_lens = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
        #    ctx_len = 1
        #    if qkv_format == "bshd":
        #        ctx_len = q.shape[1]
        #    if qkv_format == "sbhd":
        #        ctx_len = q.shape[0]
        #    tex.reshape_q(
        #        q, q_buffer, step_lens,
        #        QKVFormat[qkv_format],
        #        self.num_heads_q, self.head_dim_q,
        #        self.max_batch_size, ctx_len, self.max_ctx_len)

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
        if self.input_qkv_format == "sbhd":
            output = output[: self.batch_size, : self.max_seqlen_q].transpose(0, 1).contiguous()
        if self.input_qkv_format == "thd":
            print("oooo ", output.shape)
            # print(output[:2, :4])
            # output_buffer = self.q_orig[layer_number]
            # step_lens = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
            # tex.reshape_o(output, output_buffer, step_lens,
            #    self.num_heads_q, self.head_dim_q, self.batch_size, self.max_ctx_len, self.is_output_right_aligned)
            # output = output_buffer.view(output_buffer.shape[0], -1)

        return output
