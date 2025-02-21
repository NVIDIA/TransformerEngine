# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Inference."""
import collections
from typing import Dict, List
from einops import rearrange

import torch

import transformer_engine_torch as tex
from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat
from transformer_engine.pytorch.kv_cache_manager import KVCacheManager
from transformer_engine.pytorch.kv_cache_manager_paged import PagedKVCacheManager
from transformer_engine.pytorch.kv_cache_manager_non_paged import NonPagedKVCacheManager

class InferenceParams:  # pylint: disable=too-few-public-methods
    """
    Inference parameters that are passed to the main model in order
    to efficiently calculate and store the context and previously generated tokens
    during inference.

    Parameters
    ----------
    max_batch_size : int
                    maximum batch size during inference.
    max_sequence_length : int
                         maximum sequence length during inference.
    num_heads: int
              number of attention heads in key/value tensor.
    head_dim_k: int
               head size for the key tensor.
    dtype: torch.dtype
          data type for the KV cache.
    head_dim_v: Optional[int], default = None
               head size for the value tensor. If None, it will be set to head_dim_k.
    is_paged: bool, default = False
             whether the KV cache is paged or non-paged (contiguous).
    total_num_pages: Optional[int], default = None
                    total number of pages in the K cache or V cache if is_paged = True.
    page_size: Optional[int], default = None
              page size in number of tokens if is_paged = True.
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
            # query will be converted to 'bshd' to be consistent with cache format
            assert num_heads_q is not None, "num_heads_q is required when qkv_format=thd!"
            assert head_dim_q is not None, "head_dim_q is required when qkv_format=thd!"
            assert max_ctx_len is not None, "max_ctx_len is required when qkv_format=thd!"
            self.num_heads_q = num_heads_q
            self.head_dim_q = head_dim_q
            self.max_ctx_len = max_ctx_len
            self.max_seqlen_q = max_ctx_len

        # NonPagedKVCacheManager and PagedKVCacheManager only support 'bshd' cache
        self.cache_qkv_format = "bshd"
        self.input_qkv_format = qkv_format
        self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        self.sequences_prev = collections.OrderedDict()
        self.sequences = collections.OrderedDict()
        self.step_dict = collections.OrderedDict()
        self.batch_size = 0

        self.cu_seqlens_q = None
        self.cu_seqlens_kv = None

        # original q will be used as the output buffer
        self.q_orig = {}
        # convert q to 'bshd' to be consistent with cache format
        self.q_buffer = {}

        self.is_output_right_aligned = False

    def reset(self):
        """
        Reset the state of InferenceParams.
        """
        self.sequences = collections.OrderedDict()
        self.cache_manager.reset()
        if self.input_qkv_format == 'thd':
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
        Allocate memory for the KV cache for the layer #layer_number.
        Both K cache and V cache are in 'bshd' format.
          - non-paged:
            - K cache: [max_batch_size, max_seqlen_kv, num_heads_kv, head_dim_k]
            - V cache: [max_batch_size, max_seqlen_kv, num_heads_kv, head_dim_v]
          - paged:
            - K cache: [total_num_pages, page_size, num_heads_kv, head_dim_k]
            - V cache: [total_num_pages, page_size, num_heads_kv, head_dim_v]
        If is_cuda_graph = True, several buffers are also allocated.
          - Q buffer: [max_batch_size, max_seqlen_kv, num_heads_q, head_dim_q]
          - cu_seqlens_q buffer: [max_batch_size + 1]
          - cu_seqlens_kv buffer: [max_batch_size + 1]
        """
        self.cache_manager.allocate_memory(layer_number)

        if qkv_format == 'thd':
            self.q_buffer[layer_number] = torch.zeros(
                self.max_batch_size,
                self.max_ctx_len,
                self.num_heads_q,
                self.head_dim_q,
                dtype=self.dtype,
                device=torch.cuda.current_device(),
            )

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
        step_dict: Dict[List, List],
    ):
        """
        Prepare for step().
        """
        self.step_dict = step_dict
        self.batch_size = len(step_dict)
        self.sequences_prev = self.sequences
        self.sequences = self.cache_manager.pre_step(step_dict)

        actual_batch_size = len(step_dict)
        seqlens_q = list(step_dict.values())
        cu_seqlens_q = [0] + [sum(seqlens_q[:i]) for i in range(1, actual_batch_size + 1)]
        cu_seqlens_q = cu_seqlens_q + [cu_seqlens_q[-1]] * (self.max_batch_size - actual_batch_size)
        self.cu_seqlens_q.copy_(
            torch.Tensor(cu_seqlens_q).to(dtype=torch.int32, device="cpu")
        )
        seq_lens = list(self.sequences.values())
        #seq_lens = [self.max_seqlen_kv] * self.batch_size
        cu_seqlens_kv = [0] + [sum(seq_lens[:i]) for i in range(1, actual_batch_size + 1)]
        cu_seqlens_kv = cu_seqlens_kv + [cu_seqlens_kv[-1]] * (self.max_batch_size - actual_batch_size)
        self.cu_seqlens_kv.copy_(
            torch.Tensor(cu_seqlens_kv).to(dtype=torch.int32, device="cpu")
        )

    def convert_paged_to_nonpaged(self, layer_number: int, qkv_format: str):
        """
        Convert the k cache and v cache from paged to non-paged format. This function
        can be used for debugging purposes or for backends that do not have paged attention
        support yet, for example, UnfusedDotProductAttention.

        It can be called after step(). Based on the page table, it re-indexes the cache
        tensors and returns the contiguous, non-paged, key and value tensors. The kv cache tensors
        are assumed to be in 'bshd' format (see self.allocate_memory), and the returned key and
        value tensors will be in :attr:`qkv_format` to be consistent with the original inputs.

        Parameters
        ----------
        layer_number: int
            The layer number of the kv cache
        qkv_format: str
            The format of the returned key and value tensors, {'bshd', 'sbhd', 'thd'}

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
        seqlens = list(self.sequences.values())
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
        if qkv_format == "thd":
            new_k_cache = new_k_cache.contiguous()
            new_v_cache = new_v_cache.contiguous()
        else:
            new_k_cache = new_k_cache[:actual_batch_size].contiguous()
            new_v_cache = new_v_cache[:actual_batch_size].contiguous()
        return new_k_cache, new_v_cache

    def step(
        self,
        layer_number: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        qkv_format: str,
    ):
        """
        Update KV cache with the new key/value tokens for a given inference iteration.

        NonPagedKVCacheManager and PagedKVCacheManager are two examples of the cache manager.
        Users can write their own cache manager with their own step() function.

        If the inference iteration has only generation sequences, :attr:`k` and :attr:`v` tensors
        should have shape:
          - [batch_size, 1, num_heads, head_dim] for :attr:`qkv_format` = 'bshd',
          - [1, batch_size, num_heads, head_dim] for :attr:`qkv_format` = 'sbhd', and
          - [batch_size, num_heads, head_dim] for :attr:`qkv_format` = 'thd'.

        If the inference iteration has both generation sequences and context sequences, :attr:`k`
        and :attr:`v` should be arranged in a way so that the sequences in generation phase come
        before the sequences in context phase, in the tensor. They should have the following shape.
          - [batch_size, max_seqlen, num_heads, head_dim] for :attr:`qkv_format` = 'bshd'
          - [max_seqlen, batch_size, num_heads, head_dim] for :attr:`qkv_format` = 'sbhd', and
          - [total_num_new_tokens, num_heads, head_dim] for :attr:`qkv_format` = 'thd'.
        Here, max_seqlen is the maximum sequence length for the new tokens in the batch, and it may
        be smaller than InferenceParams.max_seqlen_kv.

        Take a batch of 4, with seq_ids = [0, 1, 2, 3], as an example. At iteration t, all 4 sequences
        are processed, after which, sequence 2 is determined to be 'finished'. For iteration t+1, there
        may or may not be a new sequence added to the batch.

        If no new sequence is added, input tensors :attr:`k` and :attr:`v` should have shape
        [3, 1, num_heads, head_dim] for :attr:`qkv_format` = 'bshd', [1, 3, num_heads, head_dim] for
        :attr:`qkv_format` = 'sbhd', and [3, num_heads, head_dim] for :attr:`qkv_format` = 'thd'.

        If one new sequence is added, for example, sequence 8 with 10 context tokens, then input tensors
        :attr:`k` and :attr:`v` should be in [4, 10, num_heads, head_dim] shape if
        :attr:`qkv_format` = 'bshd', [10, 4, num_heads, head_dim] if :attr:`qkv_format` = 'sbhd',
        or [13, num_heads, head_dim] if :attr:`qkv_format` = 'thd'.

        Parameters
        ----------
        layer_number: int
            The layer number of the kv cache
        k: torch.Tensor
            The new key tokens for the current iteration
        v: torch.Tensor
            The new value tokens for the current iteration
        qkv_format: str
            The format of the new key/value tensors, {'bshd', 'sbhd', 'thd'}

        Returns
        -------
        k_cache: torch.Tensor
            The key cache tensor, containing tokens from both previous and current iterations
        v_cache: torch.Tensor
            The value cache tensor, containing tokens from both previous and current iterations
        page_table: torch.Tensor
            The page table if is_paged = True; else `None`
        """
        self.input_qkv_format = qkv_format
        self.output_qkv_format = self.input_qkv_format + "_2" + self.cache_qkv_format

        if qkv_format == "bshd":
            q_buffer = q.contiguous()
            self.max_seqlen_q = q_buffer.shape[1]
        if qkv_format == "sbhd":
            q_buffer = q.transpose(0, 1).contiguous()
            self.max_seqlen_q = q_buffer.shape[1]
        if qkv_format == "thd":
            q_buffer = q
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
            layer_number, k, v, self.cu_seqlens_q, self.cu_seqlens_kv, qkv_format,
        )

        return q_buffer, k_cache, v_cache, page_table, self.cu_seqlens_q, self.cu_seqlens_kv, self.max_seqlen_q, self.max_seqlen_kv, self.output_qkv_format

    def post_step(
        self,
        layer_number: int,
        output: torch.Tensor,
        ):
        """
        Process the attention output in order to return it in the original qkv_format.
        """
        if self.input_qkv_format == "bshd":
            output = output[:self.batch_size, :self.max_seqlen_q].contiguous()
        if self.input_qkv_format == "sbhd":
            output = output[:self.batch_size, :self.max_seqlen_q].transpose(0, 1).contiguous()
        if self.input_qkv_format == "thd":
            print('oooo ', output.shape)
            print(output[:2, :4])
            #output_buffer = self.q_orig[layer_number]
            #step_lens = self.cu_seqlens_q[1:] - self.cu_seqlens_q[:-1]
            #tex.reshape_o(output, output_buffer, step_lens,
            #    self.num_heads_q, self.head_dim_q, self.batch_size, self.max_ctx_len, self.is_output_right_aligned)
            #output = output_buffer.view(output_buffer.shape[0], -1)

        return output


