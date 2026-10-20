# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 attention preparation, numerical, and integration tests."""

import math

import pytest

from transformer_engine.common.cudnn import mxfp8
from transformer_engine.common.cudnn.graph import mxfp8_scale_spec


@pytest.mark.parametrize("columnwise", (False, True))
@pytest.mark.parametrize("dim", ((2, 3, 128, 64), (1, 4, 256, 96), (2, 1, 384, 160)))
def test_scale_reordering_matches_graph(dim, columnwise):
    """Preparation and cuDNN agree on padding each batch/head scale matrix."""
    batch, heads, seqlen, head_dim = dim
    scale_shape = (
        (batch, heads, seqlen // 32, head_dim)
        if columnwise
        else (batch, heads, seqlen, head_dim // 32)
    )
    padded, tiled, permutation = mxfp8.scale_reordering(scale_shape, columnwise)
    assert padded == mxfp8_scale_spec(dim, "bhsd", columnwise).dim
    assert math.prod(tiled) == math.prod(padded)
    assert sorted(permutation) == list(range(len(tiled)))


@pytest.mark.parametrize(
    "sm,version,fmt,error",
    (
        (90, 92600, "bshd", "SM100"),
        (100, 92000, "bshd", "9.21"),
        (100, 92300, "bshd", "defects"),
        (100, 92301, "bshd", "defects"),
        (100, 92600, "thd", "THD"),
        (100, 92100, "bshd", ""),
        (100, 92600, "sbhd", ""),
    ),
)
def test_shared_mxfp8_support(sm, version, fmt, error):
    """Both frontends retain the same hardware and known-defect exclusions."""
    reason = mxfp8.support_error(sm, version, fmt)
    assert error in reason if error else not reason


def test_quantization_directions():
    """Forward V contracts over sequence; backward Q/K need both directions."""
    assert mxfp8.quantization_layouts() == ((True, False), (True, False), (False, True))
    assert mxfp8.quantization_layouts(True) == ((True, True), (True, True), (True, False))
    assert not mxfp8.shape_error(128, 256, 64, 96)
    assert mxfp8.shape_error(129, 256, 64, 96)
    assert mxfp8.shape_error(128, 256, 64, 80)
