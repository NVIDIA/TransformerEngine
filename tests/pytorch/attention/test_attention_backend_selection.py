# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Focused FA4 eligibility and backend-cache regressions."""

from unittest.mock import Mock
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version as PkgVersion
from transformer_engine.common import recipe

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import (
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
    "capability,training,backward,deterministic,unfused,expect_flash",
    [
        pytest.param((12, 0), True, True, True, True, False, id="sm120-training"),
        pytest.param((12, 0), False, True, True, True, False, id="sm120-eval-backward"),
        pytest.param((12, 1), False, True, True, True, False, id="sm121-eval-backward"),
        pytest.param((12, 0), False, False, True, True, True, id="forward-only"),
        pytest.param((12, 0), True, True, False, True, True, id="nondeterministic"),
        pytest.param((12, 0), True, True, True, False, False, id="no-fallback"),
        pytest.param((10, 0), True, True, True, True, True, id="sm100-unaffected"),
    ],
)
def test_fa4_deterministic_backend_selection(
    monkeypatch, capability, training, backward, deterministic, unfused, expect_flash
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
        head_dim_qk=128,
        head_dim_v=128,
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


@pytest.mark.parametrize("capability", ((12, 0), (12, 1), (12, 9)))
@pytest.mark.parametrize("case", ("fp8", "kv_cache", "mla", "thd_version", "thd_packed"))
def test_sm12x_fused_backend_restrictions(monkeypatch, capability, case):
    """SM121 and other SM12x variants must retain the SM120 backend restrictions."""
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setattr(dpa_utils, "get_device_compute_capability", lambda _device: capability)
    version = (9, 18, 0) if case == "thd_version" else (9, 27, 0)
    monkeypatch.setattr(dpa_utils, "get_cudnn_version", lambda: version)
    probe = Mock(return_value=(dpa_utils.FusedAttnBackend.F16_arbitrary_seqlen.value, ""))
    monkeypatch.setattr(dpa_utils, "_get_fused_attn_backend", probe)
    options = dict(
        device=torch.device("cuda:0"),
        qkv_dtype=torch.bfloat16,
        qkv_layout="bshd_bshd_bshd",
        num_heads=4,
        num_gqa_groups=4,
        head_dim_qk=64,
        head_dim_v=64,
        deterministic=False,
    )
    if case == "fp8":
        options.update(fp8=True, fp8_meta={"recipe": recipe.DelayedScaling(fp8_dpa=True)})
    elif case == "kv_cache":
        options.update(inference_params=SimpleNamespace(is_paged=False), is_training=False)
    elif case == "mla":
        options.update(head_dim_qk=192, head_dim_v=128)
    else:
        options.update(
            qkv_layout="t3hd" if case == "thd_packed" else "thd_thd_thd",
            attn_mask_type="padding",
            num_tokens_q=128,
            num_tokens_kv=128,
        )
    result = dpa_utils.get_attention_backend(dpa_utils.AttentionParams(**options))
    assert not result[2]
    probe.assert_not_called()


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
