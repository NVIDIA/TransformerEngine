# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils
from transformer_engine.pytorch.utils import get_cudnn_version


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required.")


@pytest.fixture(autouse=True)
def clear_cu_seqlens_cache():
    dpa_utils._cu_seqlens_cache.clear()
    yield
    dpa_utils._cu_seqlens_cache.clear()


def _make_dpa(device: torch.device) -> DotProductAttention:
    return DotProductAttention(
        num_attention_heads=2,
        kv_channels=16,
        attention_dropout=0.0,
        qkv_format="bshd",
        attn_mask_type="no_mask",
        attention_type="self",
    ).to(device=device, dtype=torch.float16)


def _make_qkv(device: torch.device, requires_grad: bool = False):
    shape = (2, 8, 2, 16)
    q = torch.randn(*shape, device=device, dtype=torch.float16, requires_grad=requires_grad)
    k = torch.randn(*shape, device=device, dtype=torch.float16, requires_grad=requires_grad)
    v = torch.randn(*shape, device=device, dtype=torch.float16, requires_grad=requires_grad)
    return q, k, v


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
def test_cu_seqlens_cache_isolated_across_devices_for_forward():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires at least 2 CUDA devices.")

    dev0 = torch.device("cuda:0")
    dev1 = torch.device("cuda:1")

    dpa0 = _make_dpa(dev0).eval()
    dpa1 = _make_dpa(dev1).eval()

    with torch.no_grad():
        q0, k0, v0 = _make_qkv(dev0)
        out0 = dpa0(q0, k0, v0, attn_mask_type="no_mask")

        q1, k1, v1 = _make_qkv(dev1)
        out1 = dpa1(q1, k1, v1, attn_mask_type="no_mask")

    assert out0.device == dev0
    assert out1.device == dev1

    expected_key_0 = (2, 8, dev0, False)
    expected_key_1 = (2, 8, dev1, False)
    assert expected_key_0 in dpa_utils._cu_seqlens_cache
    assert expected_key_1 in dpa_utils._cu_seqlens_cache

    assert dpa_utils._cu_seqlens_cache[expected_key_0].device == dev0
    assert dpa_utils._cu_seqlens_cache[expected_key_1].device == dev1


@pytest.mark.skipif(get_cudnn_version() < (8, 9, 1), reason="cuDNN 8.9.1+ is required.")
def test_cu_seqlens_cache_isolated_between_inference_and_train_forward():
    dev = torch.device("cuda:0")
    dpa = _make_dpa(dev)

    dpa.eval()
    with torch.inference_mode():
        q_inf, k_inf, v_inf = _make_qkv(dev)
        out_inf = dpa(q_inf, k_inf, v_inf, attn_mask_type="no_mask")

    inf_key = (2, 8, dev, True)
    assert inf_key in dpa_utils._cu_seqlens_cache
    assert dpa_utils._cu_seqlens_cache[inf_key].device == dev

    dpa.train()
    q_tr, k_tr, v_tr = _make_qkv(dev, requires_grad=True)
    out_tr = dpa(q_tr, k_tr, v_tr, attn_mask_type="no_mask")
    out_tr.sum().backward()

    train_key = (2, 8, dev, False)
    assert train_key in dpa_utils._cu_seqlens_cache
    assert dpa_utils._cu_seqlens_cache[train_key].device == dev

    assert out_inf.device == dev
    assert out_tr.device == dev
    assert dpa_utils._cu_seqlens_cache[inf_key] is not dpa_utils._cu_seqlens_cache[train_key]
