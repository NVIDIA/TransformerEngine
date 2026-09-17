# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FlashAttention 4 must never be handed TE's -1 window sentinel.

TE spells an unbounded window side as -1; FA4 spells it None. Since flash-attention #2490 a negative
bound is honoured arithmetically rather than widened to full attention, so the causal encoding
(-1, 0) describes the band [row + 1, row] -- empty. FA4 then returns an all-zero output and an
all -inf LSE and raises nothing, which is a silent wrong answer rather than a crash.

The numerical test below is anchored to a float64 reference rather than to another backend. That
matters here: the bug survived because tests/pytorch/attention/run_attention_with_cp.py grades a CP
run against a non-CP run *of the same backend*, so an error present on both sides cancels and the
comparison passes while measuring nothing.

float64 and not float32 -- torch computes fp32 matmuls in TF32 on Ampere and newer, whose
significand is 11 bits, the same as fp16, so an fp32 reference cannot judge a bf16 kernel.
"""

import pytest
import torch

from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    _fa4_normalized_window_kwargs,
)


@pytest.mark.parametrize(
    "sent,expected",
    [
        # TE's causal encoding: the pair that produced zeros.
        ({"window_size": (-1, 0)}, {"window_size": (None, 0)}),
        # TE's no-mask encoding. FA4 widened this one correctly on its own, since #2490 widens when
        # both bounds are negative -- it is normalized anyway so one rule covers every case.
        ({"window_size": (-1, -1)}, {"window_size": (None, None)}),
        # A genuine sliding window must survive untouched, bounds and all.
        ({"window_size": (511, 0)}, {"window_size": (511, 0)}),
        ({"window_size": (511, -1)}, {"window_size": (511, None)}),
        # The private entry points take the bounds separately.
        (
            {"window_size_left": -1, "window_size_right": 0},
            {"window_size_left": None, "window_size_right": 0},
        ),
        (
            {"window_size_left": -1, "window_size_right": -1},
            {"window_size_left": None, "window_size_right": None},
        ),
        (
            {"window_size_left": 256, "window_size_right": 0},
            {"window_size_left": 256, "window_size_right": 0},
        ),
        # Already idiomatic, and absent: both must be left alone.
        ({"window_size": (None, None)}, {"window_size": (None, None)}),
        ({"window_size": None}, {"window_size": None}),
        ({}, {}),
    ],
)
def test_fa4_window_sentinel_normalization(sent, expected):
    """Every negative bound becomes None; every other value is passed through unchanged."""
    assert _fa4_normalized_window_kwargs(dict(sent)) == expected


def test_fa4_window_normalization_preserves_other_kwargs():
    """The normalizer must not disturb anything else it is handed."""
    kwargs = {"causal": True, "window_size": (-1, 0), "softmax_scale": 0.125, "num_splits": 1}
    out = _fa4_normalized_window_kwargs(dict(kwargs))
    assert out["window_size"] == (None, 0)
    assert out["causal"] is True and out["softmax_scale"] == 0.125 and out["num_splits"] == 1


def _fa4_causal_unavailable():
    """Why this machine cannot exercise FA4 causal attention, or None if it can."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    from transformer_engine.pytorch.attention.dot_product_attention.backends import (
        flash_attn_func_v4,
    )

    if flash_attn_func_v4 is None:
        return "flash-attn-4 is not installed"
    if torch.cuda.get_device_capability() != (10, 0):
        return "the FA4 CuTe kernels under test are SM100"
    return None


_SKIP = _fa4_causal_unavailable()


@pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP))
def test_fa4_causal_attention_is_not_all_zeros():
    """End to end: FA4 causal through DotProductAttention must match a float64 reference.

    Pinned to FlashAttention because cuDNN FusedAttention wins backend selection for this shape and
    would mask the defect entirely -- the first attempt at reproducing this measured a correct
    result for exactly that reason.
    """
    import os

    from transformer_engine.pytorch import DotProductAttention
    from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
        _attention_backends,
    )

    b, h, s, d = 2, 8, 1024, 128
    dtype = torch.bfloat16
    torch.manual_seed(0)
    q, k, v = (torch.randn(b, s, h, d, device="cuda", dtype=dtype) for _ in range(3))

    saved = {key: os.environ.get(key) for key in ("NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN")}
    os.environ["NVTE_FUSED_ATTN"] = "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "0"
    _attention_backends["backend_selection_requires_update"] = True
    try:
        dpa = DotProductAttention(
            h, d, qkv_format="bshd", attn_mask_type="causal", attention_dropout=0.0
        ).to(dtype=dtype, device="cuda")
        out = dpa(q, k, v).view(b, s, h, d)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _attention_backends["backend_selection_requires_update"] = True

    # Stated separately from the tolerance check: an all-zero output is the specific failure this
    # test exists for, and it should be reported as such rather than as a large error.
    assert out.abs().max() > 0, "FA4 returned an all-zero output for causal attention"

    qs, ks, vs = (t[0, :, 0].double() for t in (q, k, v))
    scores = (qs @ ks.transpose(-1, -2)) * (d**-0.5)
    scores = scores.masked_fill(
        torch.ones(s, s, dtype=torch.bool, device=qs.device).triu(1), float("-inf")
    )
    reference = torch.softmax(scores, dim=-1) @ vs

    error = (out[0, :, 0].double() - reference).abs().max() / reference.abs().max()
    assert error < 2e-2, f"FA4 causal attention differs from the float64 reference: {error:.3e}"
