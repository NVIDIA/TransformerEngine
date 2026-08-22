# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Isolation numerics test for tanh logit softcapping in DotProductAttention.

The reference implements softcapping in pure PyTorch:

    scores = (Q @ K^T) * scale
    scores = softcap * tanh(scores / softcap)     # only when softcap != 0.0
    scores = scores + mask
    attn   = softmax(scores)
    out    = attn @ V

and is compared against ``DotProductAttention(..., softcap=...)`` forced onto the
FlashAttention backend, for both the forward output and the input gradients
(dQ/dK/dV obtained via autograd).
"""

import sys
import pathlib

import pytest
import torch
from packaging.version import Version as PkgVersion

from transformer_engine.pytorch import DotProductAttention
from transformer_engine.pytorch.attention.dot_product_attention import _attention_backends

_current_file = pathlib.Path(__file__).resolve()
sys.path = [str(_current_file.parent.parent)] + sys.path
from utils import reset_rng_states  # pylint: disable=wrong-import-position


def _flash_attn_2_6_available() -> bool:
    """Whether flash-attn >= 2.6.0 (the first version exposing ``softcap``) is installed."""
    try:
        import flash_attn  # pylint: disable=import-outside-toplevel
    except ImportError:
        return False
    return PkgVersion(flash_attn.__version__) >= PkgVersion("2.6.0")


# Softcapping through DotProductAttention is only wired through the FlashAttention 2
# backend (>= 2.6.0), and requires CUDA tensors.
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required."),
    pytest.mark.skipif(not _flash_attn_2_6_available(), reason="flash-attn >= 2.6.0 is required."),
]


def _force_flash_backend() -> None:
    """Force DotProductAttention to select the FlashAttention backend."""
    import os  # pylint: disable=import-outside-toplevel

    os.environ["NVTE_FLASH_ATTN"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    _attention_backends["backend_selection_requires_update"] = True


def _reference_attention(q, k, v, scale, softcap, causal):
    """Pure-PyTorch reference for softcapped scaled dot product attention.

    q, k, v are in ``bshd`` layout. GQA is supported: ``k``/``v`` may have fewer
    heads than ``q``.
    """
    # bshd -> bhsd
    qt = q.transpose(1, 2).float()
    kt = k.transpose(1, 2).float()
    vt = v.transpose(1, 2).float()

    num_heads = qt.shape[1]
    num_gqa_groups = kt.shape[1]
    if num_heads != num_gqa_groups:
        assert num_heads % num_gqa_groups == 0
        repeats = num_heads // num_gqa_groups
        kt = kt.repeat_interleave(repeats, dim=1)
        vt = vt.repeat_interleave(repeats, dim=1)

    scores = torch.matmul(qt, kt.transpose(-2, -1)) * scale
    if softcap != 0.0:
        scores = softcap * torch.tanh(scores / softcap)
    if causal:
        sq, skv = scores.shape[-2], scores.shape[-1]
        mask = torch.triu(
            torch.ones(sq, skv, dtype=torch.bool, device=scores.device),
            diagonal=1 + skv - sq,
        )
        scores = scores.masked_fill(mask, float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vt)
    # bhsd -> bshd
    return out.transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("softcap", [0.0, 50.0])
@pytest.mark.parametrize("num_gqa_groups", [4, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_softcap_numerics(dtype, softcap, num_gqa_groups, causal):
    """FlashAttention softcap forward + grads match a pure-PyTorch reference.

    ``softcap == 0.0`` additionally proves that softcapping is a no-op relative to
    the plain (no-softcap) reference, i.e. today's behavior is reproduced exactly.
    """
    reset_rng_states()

    batch_size = 2
    max_seqlen = 32
    num_heads = 4
    head_dim = 64
    scale = 1.0 / (head_dim**0.5)

    q_shape = (batch_size, max_seqlen, num_heads, head_dim)
    kv_shape = (batch_size, max_seqlen, num_gqa_groups, head_dim)

    q = (0.5 * torch.randn(q_shape, dtype=dtype, device="cuda")).requires_grad_()
    k = (0.5 * torch.randn(kv_shape, dtype=dtype, device="cuda")).requires_grad_()
    v = (0.5 * torch.randn(kv_shape, dtype=dtype, device="cuda")).requires_grad_()
    q_ref, k_ref, v_ref = [x.detach().clone().requires_grad_() for x in (q, k, v)]

    grad_output = torch.randn(q_shape, dtype=dtype, device="cuda")

    _force_flash_backend()
    dpa = DotProductAttention(
        num_heads,
        head_dim,
        num_gqa_groups=num_gqa_groups,
        qkv_format="bshd",
        attn_mask_type="causal" if causal else "no_mask",
        softmax_scale=scale,
        softcap=softcap,
        layer_number=1,
    ).to(dtype=dtype, device="cuda")

    out = dpa(q, k, v)
    out.backward(grad_output)

    out_ref = _reference_attention(q_ref, k_ref, v_ref, scale, softcap, causal)
    out_ref.backward(grad_output.float())

    atol, rtol = (2e-2, 2e-2) if dtype == torch.float16 else (3.5e-2, 3.5e-2)

    torch.testing.assert_close(out.float(), out_ref.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(q.grad.float(), q_ref.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(k.grad.float(), k_ref.grad.float(), atol=atol, rtol=rtol)
    torch.testing.assert_close(v.grad.float(), v_ref.grad.float(), atol=atol, rtol=rtol)
