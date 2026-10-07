# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Focused FA4 eligibility and backend-cache regressions."""

from unittest.mock import Mock

import pytest
import torch
from packaging.version import Version as PkgVersion

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import (
    backends as dpa_backends,
    dot_product_attention as dpa_module,
    utils as dpa_utils,
)


@pytest.fixture(autouse=True)
def isolated_backends(monkeypatch):
    for name, value in (
        ("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0"),
        ("NVTE_FLASH_ATTN", "1"),
        ("NVTE_FLASH_ATTN_V2", "0"),
        ("NVTE_FLASH_ATTN_V3", "0"),
        ("NVTE_FLASH_ATTN_V4", "1"),
        ("NVTE_FUSED_ATTN", "0"),
        ("NVTE_UNFUSED_ATTN", "1"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        dpa_module,
        "_attention_backends",
        {
            **dpa_module._attention_backends,
            "attention_params": None,
            "backend_selection_requires_update": True,
        },
    )
    monkeypatch.setattr(dpa_module, "_thd_policy_backend_cache", [])


@pytest.mark.parametrize(
    "capability,training,backward,deterministic,head_dim,unfused,expect_flash",
    [
        pytest.param((12, 0), True, True, True, 128, True, False, id="sm120-training"),
        pytest.param((12, 0), False, True, True, 128, True, False, id="sm120-eval-backward"),
        pytest.param((12, 1), False, True, True, 128, True, False, id="sm121-eval-backward"),
        pytest.param((12, 0), False, False, True, 128, True, True, id="forward-only"),
        pytest.param((12, 0), True, True, False, 128, True, True, id="nondeterministic"),
        pytest.param((12, 0), True, True, True, 128, False, False, id="no-fallback"),
        pytest.param((10, 0), True, True, True, 128, True, True, id="sm100-d128"),
        pytest.param((10, 0), True, True, True, 256, True, False, id="sm100-d256"),
    ],
)
def test_fa4_deterministic_backend_selection(
    monkeypatch, capability, training, backward, deterministic, head_dim, unfused, expect_flash
):
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", str(int(unfused)))
    monkeypatch.setattr(dpa_utils, "get_device_compute_capability", lambda device: capability)
    monkeypatch.setattr(dpa_utils, "get_cudnn_version", lambda: (9, 26, 0))
    monkeypatch.setattr(dpa_utils.FlashAttentionUtils, "v4_is_installed", True)
    monkeypatch.setattr(dpa_utils.FlashAttentionUtils, "fa4_version", PkgVersion("4.0.0b24"))
    monkeypatch.setattr(dpa_utils.FlashAttentionUtils, "v4_validate_head_dims", None)
    params = dpa_utils.AttentionParams(
        device=torch.device("cuda:0"),
        qkv_dtype=torch.bfloat16,
        qkv_layout="bshd_bshd_bshd",
        head_dim_qk=head_dim,
        head_dim_v=head_dim,
        attn_mask_type="causal",
        window_size=(-1, 0),
        deterministic=deterministic,
        is_training=training,
        requires_backward=backward,
    )
    flash, version, fused, _, selected_unfused, available = dpa_utils.get_attention_backend(params)
    assert bool(flash) is expect_flash
    assert version == (dpa_utils.FlashAttentionUtils.fa4_version if expect_flash else None)
    assert not fused
    assert bool(selected_unfused) is (unfused and not expect_flash)
    assert available == [expect_flash, False, unfused]


@pytest.mark.parametrize(
    "version,backward,expect_flash",
    [
        pytest.param("4.0.0b30", False, False, id="b30-forward"),
        pytest.param("4.0.0b31", False, True, id="b31-forward"),
        pytest.param("4.0.0b32", True, False, id="b32-backward"),
        pytest.param("4.0.0b33", True, True, id="b33-backward"),
    ],
)
@pytest.mark.parametrize("path", ["compact", "padding", "all-gather"])
def test_fa4_d256_seqused_backend_selection(monkeypatch, version, backward, expect_flash, path):
    """FA4 D=256 eligibility follows seqused_q/k forward/backward support."""
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")
    monkeypatch.setattr(dpa_utils, "get_device_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(dpa_utils, "get_cudnn_version", lambda: (9, 26, 0))
    monkeypatch.setattr(dpa_utils.FlashAttentionUtils, "v4_is_installed", True)
    monkeypatch.setattr(dpa_utils.FlashAttentionUtils, "fa4_version", PkgVersion(version))
    monkeypatch.setattr(dpa_utils.FlashAttentionUtils, "v4_validate_head_dims", None)
    params = dpa_utils.AttentionParams(
        device=torch.device("cuda:0"),
        qkv_dtype=torch.bfloat16,
        qkv_layout="thd_thd_thd",
        head_dim_qk=256,
        head_dim_v=256,
        attn_mask_type="padding_causal",
        pad_between_seqs=path == "padding",
        context_parallel=path == "all-gather",
        cp_comm_type="all_gather" if path == "all-gather" else "p2p",
        deterministic=False,
        requires_backward=backward,
    )
    flash, selected_version, _, _, _, available = dpa_utils.get_attention_backend(params)
    expected = path == "compact" or expect_flash
    assert bool(flash) is expected
    assert selected_version == (dpa_utils.FlashAttentionUtils.fa4_version if expected else None)
    assert bool(available[0]) is expected


@pytest.mark.parametrize("changed_field", ["device", "requires_backward"])
def test_thd_backend_cache_tracks_execution_context(monkeypatch, changed_field):
    selector = Mock(return_value=(False, None, False, None, True, [False, False, True]))
    monkeypatch.setattr(dpa_utils, "get_attention_backend", selector)
    policy = {
        "sequence_ids": torch.tensor([0]),
        "mask_type": "causal",
        "window_size": (-1, 0),
        "bottom_right_diagonal": True,
    }
    params = {"device": torch.device("cuda:0"), "requires_backward": False}
    select = dpa_module._get_thd_policy_attention_backend
    select(policy, params, False)
    select(policy, params, False)
    assert selector.call_count == 1
    changed = {
        **params,
        changed_field: torch.device("cuda:1") if changed_field == "device" else True,
    }
    select(policy, changed, False)
    assert selector.call_count == 2
    select(policy, params, False)
    assert selector.call_count == 2


def _sm12x_attention():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("Requires an SM12x GPU")
    if not dpa_utils.FlashAttentionUtils.v4_is_installed:
        pytest.skip("Requires FlashAttention 4")
    qkv = [
        torch.randn(2, 128, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    attention = DotProductAttention(
        num_attention_heads=8,
        kv_channels=128,
        qkv_format="bshd",
        attn_mask_type="causal",
        attention_dropout=0.0,
    ).cuda()
    return attention, qkv


def test_fa4_eval_backward_after_forward_only():
    """Changing grad eligibility in eval mode must invalidate the scalar cache."""
    attention, qkv = _sm12x_attention()
    attention.eval()
    with torch.no_grad():
        attention(*qkv)
    assert dpa_module._attention_backends["use_flash_attention"]
    output = attention(*qkv)
    output.float().square().mean().backward()
    assert dpa_module._attention_backends["use_unfused_attention"]
    assert torch.isfinite(output).all()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in qkv)
    attention(*(t.detach() for t in qkv))
    assert dpa_module._attention_backends["use_flash_attention"]


def test_fa4_cached_selection_from_other_device(monkeypatch):
    """A seeded other-device selection must not bypass the real SM12x backward guard."""
    attention, qkv = _sm12x_attention()
    # Obtain an FA4 selection as on SM100, running forward only on the available GPU.
    # This simulates the other architecture's cache entry; no second GPU is required.
    with monkeypatch.context() as context:
        context.setattr(dpa_utils, "get_device_compute_capability", lambda device=None: (10, 0))
        attention(*qkv)
    assert dpa_module._attention_backends["use_flash_attention"]
    cached = dpa_module._attention_backends["attention_params"]
    cached.device = torch.device("cuda", qkv[0].device.index + 1)
    selector = Mock(wraps=dpa_utils.get_attention_backend)
    monkeypatch.setattr(dpa_utils, "get_attention_backend", selector)
    output = attention(*qkv)
    output.float().square().mean().backward()
    assert selector.call_count == 1
    assert selector.call_args.args[0].device == qkv[0].device
    assert dpa_module._attention_backends["use_unfused_attention"]
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in qkv)


def _fa4_normalized_kwargs(kwargs):
    received = {}
    dpa_backends._fa4_with_none_window_sentinel(received.update)(**kwargs)
    return received


@pytest.mark.parametrize(
    "sent,expected",
    [
        ({"window_size": (-1, 0)}, {"window_size": (None, 0)}),
        ({"window_size": (-1, -1)}, {"window_size": (None, None)}),
        ({"window_size": (511, 0)}, {"window_size": (511, 0)}),
        ({"window_size": (511, -1)}, {"window_size": (511, None)}),
        # Other negative offsets are not the -1 sentinel.
        ({"window_size": (7, -8)}, {"window_size": (7, -8)}),
        (
            {"window_size_left": -1, "window_size_right": 0},
            {"window_size_left": None, "window_size_right": 0},
        ),
        (
            {"window_size_left": -1, "window_size_right": -1},
            {"window_size_left": None, "window_size_right": None},
        ),
        (
            {"window_size_left": 0, "window_size_right": -1},
            {"window_size_left": 0, "window_size_right": None},
        ),
        (
            {"window_size_left": 7, "window_size_right": -8},
            {"window_size_left": 7, "window_size_right": -8},
        ),
        (
            {"window_size_left": 256, "window_size_right": 0},
            {"window_size_left": 256, "window_size_right": 0},
        ),
        ({"window_size": (None, None)}, {"window_size": (None, None)}),
        ({"window_size": None}, {"window_size": None}),
        ({}, {}),
    ],
)
def test_fa4_window_sentinel_normalization(sent, expected):
    """Convert only TE's -1 sentinel for both FA4 window argument forms."""
    assert _fa4_normalized_kwargs(sent) == expected
