# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Attention kernels wrapped as custom ops, so they don't graph-break under torch.compile."""

import torch
import transformer_engine_torch as tex

from transformer_engine.pytorch.cpp_extensions.fused_attn import QKVFormat

# The ops take the format's value rather than the pybind enum: converting the
# enum inside a traced region makes dynamo recurse until it gives up.
QKV_FORMAT_VALUE = {name: int(fmt) for name, fmt in QKVFormat.items()}
_QKV_FORMAT_BY_VALUE = {int(fmt): fmt for fmt in QKVFormat.values()}


@torch.library.custom_op(
    "te_kv_cache::copy_to_kv_cache",
    mutates_args=("k_cache", "v_cache"),
    device_types="cuda",
)
def copy_to_kv_cache(
    new_k: torch.Tensor,
    new_v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    cu_new_lens: torch.Tensor,
    cu_cached_lens: torch.Tensor,
    qkv_format: int,
    b: int,
    max_ctx_len: int,
    max_seq_len: int,
    max_pages_per_seq: int,
    is_non_paged: bool,
) -> None:
    """Copy new key/value tokens into the KV cache."""
    tex.copy_to_kv_cache(
        new_k,
        new_v,
        k_cache,
        v_cache,
        page_table,
        cu_new_lens,
        cu_cached_lens,
        _QKV_FORMAT_BY_VALUE[qkv_format],
        b,
        max_ctx_len,
        max_seq_len,
        max_pages_per_seq,
        is_non_paged,
    )


@copy_to_kv_cache.register_fake
def _copy_to_kv_cache_fake(*_args, **_kwargs) -> None:
    return None


@torch.library.custom_op("te_kv_cache::convert_bshd_to_thd", mutates_args=(), device_types="cuda")
def convert_bshd_to_thd(tensor: torch.Tensor, cu_seqlens: torch.Tensor, t: int) -> torch.Tensor:
    """Convert a tensor from bshd to thd."""
    return tex.convert_bshd_to_thd(tensor, cu_seqlens, t)


@convert_bshd_to_thd.register_fake
def _convert_bshd_to_thd_fake(
    tensor: torch.Tensor, cu_seqlens: torch.Tensor, t: int
) -> torch.Tensor:
    del cu_seqlens
    return tensor.new_empty((t, *tensor.shape[2:]))


@torch.library.custom_op("te_kv_cache::convert_thd_to_bshd", mutates_args=(), device_types="cuda")
def convert_thd_to_bshd(
    tensor: torch.Tensor, cu_seqlens: torch.Tensor, b: int, max_seq_len: int
) -> torch.Tensor:
    """Convert a tensor from thd to bshd."""
    return tex.convert_thd_to_bshd(tensor, cu_seqlens, b, max_seq_len)


@convert_thd_to_bshd.register_fake
def _convert_thd_to_bshd_fake(
    tensor: torch.Tensor, cu_seqlens: torch.Tensor, b: int, max_seq_len: int
) -> torch.Tensor:
    del cu_seqlens
    return tensor.new_empty((b, max_seq_len, *tensor.shape[1:]))


@torch.library.custom_op("te_attention::fa_prepare_fwd", mutates_args=(), device_types="cuda")
def fa_prepare_fwd(qkvi: torch.Tensor) -> torch.Tensor:
    """Split interleaved sbh3d QKV into bshd q/k/v."""
    return tex.fa_prepare_fwd(qkvi)


@fa_prepare_fwd.register_fake
def _fa_prepare_fwd_fake(qkvi: torch.Tensor) -> torch.Tensor:
    # qkvi is the q view into the packed buffer, and its strides cover all of it.
    s, b, n, h = qkvi.shape
    return qkvi.new_empty((3, b, s, n, h))


@torch.library.custom_op("te_attention::fa_prepare_bwd", mutates_args=(), device_types="cuda")
def fa_prepare_bwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Pack bshd gradients back into an interleaved sbh3d buffer."""
    return tex.fa_prepare_bwd(q, k, v)


@fa_prepare_bwd.register_fake
def _fa_prepare_bwd_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    del k, v
    b, s, n, h = q.shape
    return q.new_empty((s, b, n, 3 * h))
