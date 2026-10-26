# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device-side alignment check for grouped split offsets."""

import pytest
import torch

import transformer_engine.pytorch  # Load the common library before the extension.
import transformer_engine_torch as tex


def test_misaligned_splits_fail(capfd):
    """The multi-output scan rejects a split that is not 256-row aligned."""
    splits = torch.tensor([256, 128], device="cuda", dtype=torch.int64)

    with pytest.raises(RuntimeError):
        tex.splits_to_offsets_multi(
            splits,
            torch.device("cuda"),
            strides=[1] * 6,
            include_leading_zero=[True] * 6,
            dtypes=[torch.int64] * 6,
            split_alignment=256,
        )
        torch.cuda.synchronize()

    assert "Grouped split size is not divisible by split_alignment." in capfd.readouterr().out
