"""Correctness tests for grouped (multi-tensor) NVFP4 per-token cast.

The grouped kernel must be byte-equal to a for-loop of single-tensor
calls. Covers composite K1+K2, K1-only, single-split, and many-split.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import pytest
import torch

# Import transformer_engine first to dlopen libtransformer_engine.so so that
# transformer_engine_torch can resolve typeinfo / vtable symbols at load time.
import transformer_engine.pytorch as te  # noqa: F401
import transformer_engine_torch as tex  # type: ignore  # noqa: F401

from transformer_engine.pytorch.custom_recipes.quantization_nvfp4_per_token import (
    BLOCK_K,
    RefNVFP4TensorPerToken,
    nvfp4_per_token_quantize,
)


def _has_fp4() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


_GATED_FP4 = pytest.mark.skipif(
    not _has_fp4(),
    reason="NVFP4 per-token cast requires SM100 (Blackwell) + CUDA 12.8+",
)


# Helper: invoke the grouped binding.
def _alloc_per_token_buffers(
    M_i: int,
    K: int,
    rowwise: bool,
    columnwise: bool,
    device: torch.device,
) -> Tuple[
    Optional[torch.Tensor],  # q_row
    Optional[torch.Tensor],  # s_dec_row
    Optional[torch.Tensor],  # row_amax
    Optional[torch.Tensor],  # q_col
    Optional[torch.Tensor],  # s_dec_col
    Optional[torch.Tensor],  # col_amax
]:
    q_row = None
    s_dec_row = None
    row_amax = None
    q_col = None
    s_dec_col = None
    col_amax = None
    if rowwise:
        q_row = torch.empty((M_i, K // 2), dtype=torch.uint8, device=device)
        s_dec_row = torch.empty((M_i, K // BLOCK_K), dtype=torch.uint8, device=device)
        row_amax = torch.empty((M_i,), dtype=torch.float32, device=device)
    if columnwise:
        q_col = torch.empty((K, M_i // 2), dtype=torch.uint8, device=device)
        s_dec_col = torch.empty((K, M_i // BLOCK_K), dtype=torch.uint8, device=device)
        col_amax = torch.empty((K,), dtype=torch.float32, device=device)
    return q_row, s_dec_row, row_amax, q_col, s_dec_col, col_amax


def _group_quantize_py(
    x_concat: torch.Tensor,
    split_sections: List[int],
    rowwise: bool,
    columnwise: bool,
) -> List[RefNVFP4TensorPerToken]:
    """Pre-allocate per-split outputs, dispatch tex.nvfp4_per_token_group_quantize."""
    assert x_concat.dim() == 2
    sum_M, K = x_concat.shape
    assert sum(split_sections) == sum_M
    device = x_concat.device

    n = len(split_sections)
    q_row_list: List[torch.Tensor] = []
    s_dec_row_list: List[torch.Tensor] = []
    row_amax_list: List[torch.Tensor] = []
    q_col_list: List[torch.Tensor] = []
    s_dec_col_list: List[torch.Tensor] = []
    col_amax_list: List[torch.Tensor] = []

    for M_i in split_sections:
        qr, sr, ra, qc, sc, ca = _alloc_per_token_buffers(M_i, K, rowwise, columnwise, device)
        if rowwise:
            q_row_list.append(qr)
            s_dec_row_list.append(sr)
            row_amax_list.append(ra)
        if columnwise:
            q_col_list.append(qc)
            s_dec_col_list.append(sc)
            col_amax_list.append(ca)

    # Binding wants lists matching num_tensors; pass empty for skipped direction.
    empty: List[torch.Tensor] = []

    tex.nvfp4_per_token_group_quantize(
        x_concat,
        split_sections,
        q_row_list if rowwise else empty,
        s_dec_row_list if rowwise else empty,
        row_amax_list if rowwise else empty,
        q_col_list if columnwise else empty,
        s_dec_col_list if columnwise else empty,
        col_amax_list if columnwise else empty,
        rowwise,
        columnwise,
    )

    out: List[RefNVFP4TensorPerToken] = []
    for i in range(n):
        # Re-view e4m3 SF as torch.float8_e4m3fn (same bytes, expected dtype).
        tensor = RefNVFP4TensorPerToken(
            data=q_row_list[i] if rowwise else None,
            scale=(s_dec_row_list[i].view(torch.float8_e4m3fn) if rowwise else None),
            row_amax=row_amax_list[i] if rowwise else None,
            columnwise_data=q_col_list[i] if columnwise else None,
            columnwise_scale=(s_dec_col_list[i].view(torch.float8_e4m3fn) if columnwise else None),
            col_amax=col_amax_list[i] if columnwise else None,
        )
        out.append(tensor)
    return out


# Test fixtures. Per-token kernel requires M_i % 128 == 0 and K % 128 == 0.
_SHAPES: List[Tuple[List[int], int]] = [
    # (split_sections, K)
    ([128], 128),  # trivial: 1 split, smallest legal shape
    ([128, 128], 128),  # 2 equal splits
    ([128, 256], 128),  # 2 unequal splits
    ([128, 256, 128], 256),  # 3 splits, mixed sizes
    ([128, 128, 128, 128], 256),  # 4 equal splits
    ([256, 128, 384, 128, 128], 512),  # 5-way unequal split, typical MoE
    ([256, 256], 1024),  # larger K, 2 splits
]


# (1) Composite K1+K2: grouped == for-loop of single-tensor, byte-equal.
@_GATED_FP4
@pytest.mark.parametrize("split_sections,K", _SHAPES)
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_group_per_token_quantize_byte_equal(
    split_sections: List[int],
    K: int,
    rowwise: bool,
    columnwise: bool,
    dtype: torch.dtype,
) -> None:
    """Grouped == for-loop of single-tensor, byte-equal (FP4 + SF + amax)."""
    torch.manual_seed(0xCAFE * (sum(split_sections) + 7) + K)
    device = torch.device("cuda")
    sum_M = sum(split_sections)

    # Per-split inputs with sprinkled outliers to stress per-row outer.
    splits_in: List[torch.Tensor] = []
    for i, M_i in enumerate(split_sections):
        s = torch.randn((M_i, K), dtype=dtype, device=device) * (2.0 + 0.5 * i)
        if M_i >= 4:
            s[0, :] *= 8.0
            s[-1, :] *= 0.125
        splits_in.append(s)

    x_concat = torch.cat(splits_in, dim=0)
    assert x_concat.shape == (sum_M, K)

    oracle: List[RefNVFP4TensorPerToken] = [
        nvfp4_per_token_quantize(s, rowwise=rowwise, columnwise=columnwise) for s in splits_in
    ]

    sut: List[RefNVFP4TensorPerToken] = _group_quantize_py(
        x_concat, split_sections, rowwise=rowwise, columnwise=columnwise
    )

    assert len(sut) == len(oracle) == len(split_sections)

    for i in range(len(split_sections)):
        if rowwise:
            torch.testing.assert_close(
                sut[i].data.view(torch.uint8),
                oracle[i].data.view(torch.uint8),
                atol=0.0,
                rtol=0.0,
                msg=f"rowwise q[{i}] mismatch",
            )
            torch.testing.assert_close(
                sut[i].scale.view(torch.uint8),
                oracle[i].scale.view(torch.uint8),
                atol=0.0,
                rtol=0.0,
                msg=f"rowwise s_dec[{i}] mismatch",
            )
            torch.testing.assert_close(
                sut[i].row_amax,
                oracle[i].row_amax,
                atol=0.0,
                rtol=0.0,
                msg=f"row_amax[{i}] mismatch",
            )
        if columnwise:
            torch.testing.assert_close(
                sut[i].columnwise_data.view(torch.uint8),
                oracle[i].columnwise_data.view(torch.uint8),
                atol=0.0,
                rtol=0.0,
                msg=f"columnwise q[{i}] mismatch",
            )
            torch.testing.assert_close(
                sut[i].columnwise_scale.view(torch.uint8),
                oracle[i].columnwise_scale.view(torch.uint8),
                atol=0.0,
                rtol=0.0,
                msg=f"columnwise s_dec[{i}] mismatch",
            )
            torch.testing.assert_close(
                sut[i].col_amax,
                oracle[i].col_amax,
                atol=0.0,
                rtol=0.0,
                msg=f"col_amax[{i}] mismatch",
            )


# (2) K1-only (amax) entry == K1-only of single-tensor, byte-equal.
@_GATED_FP4
@pytest.mark.parametrize("split_sections,K", _SHAPES[:3])  # subset, K1 is simple
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
def test_group_per_token_amax_byte_equal(
    split_sections: List[int],
    K: int,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """tex.nvfp4_per_token_group_amax matches K1 of the for-loop variant."""
    torch.manual_seed(0xDEAD * sum(split_sections) + K)
    device = torch.device("cuda")
    sum_M = sum(split_sections)
    n = len(split_sections)

    splits_in: List[torch.Tensor] = []
    for i, M_i in enumerate(split_sections):
        splits_in.append(torch.randn((M_i, K), dtype=torch.bfloat16, device=device) * 3.0)
    x_concat = torch.cat(splits_in, dim=0)

    # Oracle row_amax / col_amax via single-tensor quantize (shared K1).
    oracle_row = []
    oracle_col = []
    for s in splits_in:
        o = nvfp4_per_token_quantize(s, rowwise=rowwise, columnwise=columnwise)
        oracle_row.append(o.row_amax if rowwise else None)
        oracle_col.append(o.col_amax if columnwise else None)

    row_amax_list = (
        [torch.empty((M_i,), dtype=torch.float32, device=device) for M_i in split_sections]
        if rowwise
        else []
    )
    col_amax_list = (
        [torch.empty((K,), dtype=torch.float32, device=device) for _ in range(n)]
        if columnwise
        else []
    )

    tex.nvfp4_per_token_group_amax(
        x_concat, split_sections, row_amax_list, col_amax_list, rowwise, columnwise
    )

    if rowwise:
        for i in range(n):
            torch.testing.assert_close(
                row_amax_list[i],
                oracle_row[i],
                atol=0.0,
                rtol=0.0,
                msg=f"row_amax[{i}] mismatch",
            )
    if columnwise:
        for i in range(n):
            torch.testing.assert_close(
                col_amax_list[i],
                oracle_col[i],
                atol=0.0,
                rtol=0.0,
                msg=f"col_amax[{i}] mismatch",
            )


# (3) Single-split call must equal the single-tensor kernel.
@_GATED_FP4
@pytest.mark.parametrize("M,K", [(128, 128), (128, 256), (256, 1024)])
@pytest.mark.parametrize("rowwise,columnwise", [(True, False), (False, True), (True, True)])
def test_group_single_split_matches_single_tensor(
    M: int, K: int, rowwise: bool, columnwise: bool
) -> None:
    """One-split grouped call == single-tensor call (boundary-advance no-op)."""
    torch.manual_seed(0xBABE * M + K)
    device = torch.device("cuda")
    x = torch.randn((M, K), dtype=torch.bfloat16, device=device) * 4.0

    oracle = nvfp4_per_token_quantize(x, rowwise=rowwise, columnwise=columnwise)
    sut_list = _group_quantize_py(x, [M], rowwise=rowwise, columnwise=columnwise)
    assert len(sut_list) == 1
    sut = sut_list[0]

    if rowwise:
        torch.testing.assert_close(sut.data, oracle.data, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            sut.scale.view(torch.uint8),
            oracle.scale.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(sut.row_amax, oracle.row_amax, atol=0.0, rtol=0.0)
    if columnwise:
        torch.testing.assert_close(sut.columnwise_data, oracle.columnwise_data, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            sut.columnwise_scale.view(torch.uint8),
            oracle.columnwise_scale.view(torch.uint8),
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(sut.col_amax, oracle.col_amax, atol=0.0, rtol=0.0)


# (4) Many-split scaling test (close to the 64-tensor cap).
@_GATED_FP4
@pytest.mark.parametrize("n_splits", [8, 16, 32, 64])
def test_group_many_splits_byte_equal(n_splits: int) -> None:
    """Many small splits (MoE expert layout) still byte-equal to oracle."""
    torch.manual_seed(0xFEED * n_splits)
    device = torch.device("cuda")
    K = 256
    split_sections = [128] * n_splits

    splits_in = [
        torch.randn((128, K), dtype=torch.bfloat16, device=device) * (1.0 + 0.1 * i)
        for i in range(n_splits)
    ]
    x_concat = torch.cat(splits_in, dim=0)

    oracle = [nvfp4_per_token_quantize(s, rowwise=True, columnwise=True) for s in splits_in]
    sut = _group_quantize_py(x_concat, split_sections, rowwise=True, columnwise=True)

    for i in range(n_splits):
        torch.testing.assert_close(sut[i].data, oracle[i].data, atol=0.0, rtol=0.0)
        torch.testing.assert_close(sut[i].row_amax, oracle[i].row_amax, atol=0.0, rtol=0.0)
        torch.testing.assert_close(
            sut[i].columnwise_data, oracle[i].columnwise_data, atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(sut[i].col_amax, oracle[i].col_amax, atol=0.0, rtol=0.0)
