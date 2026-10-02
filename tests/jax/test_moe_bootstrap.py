# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Host-side MoE bootstrap compatibility checks; no EP collective is required."""

import importlib

import pytest

moe_module = importlib.import_module("transformer_engine.jax.moe")


@pytest.fixture
def bootstrap_signature(monkeypatch):
    """Isolate the process-global signature from other EP tests."""
    monkeypatch.setattr(moe_module, "_te_ep_bootstrap_signature", (8, 128, 512, 128, 2))


@pytest.mark.parametrize("recv_capacity", [256, 1024])
def test_reject_mismatched_recv_capacity(bootstrap_signature, recv_capacity):
    """A smaller capacity must fail before NCCL EP dispatch, just like a larger one."""
    with pytest.raises(ValueError, match="dispatch capacity must exactly match bootstrap"):
        moe_module._te_ep_assert_compatible_bootstrap(8, 128, recv_capacity, 128, 2)


@pytest.mark.parametrize("tokens_per_rank", [64, 128])
def test_accept_smaller_token_count_with_matching_capacity(bootstrap_signature, tokens_per_rank):
    """Token counts remain bounded by, rather than equal to, the bootstrap maximum."""
    moe_module._te_ep_assert_compatible_bootstrap(8, tokens_per_rank, 512, 128, 2)


def test_reject_larger_token_count(bootstrap_signature):
    """The receive-capacity fix must not loosen the send-token bound."""
    with pytest.raises(ValueError, match="max_tokens_per_rank=256"):
        moe_module._te_ep_assert_compatible_bootstrap(8, 256, 512, 128, 2)
