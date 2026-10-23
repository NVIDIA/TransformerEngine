# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""``total_num_pages`` is a physical pool budget, not the page-table size.

The paged KV cache used to require ``total_num_pages == max_batch_size *
max_pages_per_seq``. The left-hand side is how much KV memory the pool actually
owns; the right-hand side is only how many logical page slots the page table can
address. Tying them together stopped a custom cache manager from running a
deliberately smaller physical pool.
"""

import pytest
import torch

from transformer_engine.pytorch.attention.inference import InferenceParams

PAGE_SIZE = 8
MAX_SEQ = 64
MAX_BATCH = 4
PAGES_PER_SEQ = MAX_SEQ // PAGE_SIZE


def _params(total_num_pages: int) -> InferenceParams:
    return InferenceParams(
        max_batch_size=MAX_BATCH,
        max_sequence_length=MAX_SEQ,
        num_heads_kv=2,
        head_dim_k=64,
        dtype=torch.bfloat16,
        is_paged=True,
        total_num_pages=total_num_pages,
        page_size=PAGE_SIZE,
    )


def test_physical_pool_may_be_smaller_than_the_page_table():
    """One sequence worth of pages is enough to construct the cache."""
    params = _params(PAGES_PER_SEQ)
    assert params.total_num_pages == PAGES_PER_SEQ


def test_physical_pool_smaller_than_one_sequence_is_rejected():
    with pytest.raises(AssertionError, match="total_num_pages"):
        _params(PAGES_PER_SEQ - 1)
