# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression test for makeTransformerEngineTensor(py::handle, py::handle).

On stock TE 2.14+ a few repeated BF16 forwards through a plain
``te.LayerNormLinear`` leave ``weight.data`` as a view without valid
storage. The next forward then aborts libtorch inside
``transformer_engine/pytorch/csrc/common.cpp`` at
``torch_tensor.data_ptr()`` with::

    RuntimeError: Cannot access data pointer of Tensor that doesn't have
    storage.

This test verifies that the shared helper no longer dereferences a
storage-less tensor (see PR #2481 for the sibling fix in the quantize /
activation kernels).

Empirically, 20 iterations reproduce the crash within the first 1-2
calls on stock TE 2.14 across 5 independent processes (probability
effectively 1). After the fix all 20 iterations complete without
raising the ``"Cannot access data pointer"`` error.
"""

import pytest
import torch


DATA_PTR_MARKER = "Cannot access data pointer"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for TE kernels")
def test_layernorm_linear_repeated_forward_bf16():
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    m = te.LayerNormLinear(1024, 1024, bias=False, params_dtype=torch.bfloat16).cuda()
    x = torch.randn(4, 1024, device="cuda", dtype=torch.bfloat16)

    for _ in range(20):
        try:
            m(x)
        except RuntimeError as exc:
            if DATA_PTR_MARKER in str(exc):
                raise AssertionError(
                    "makeTransformerEngineTensor still dereferences data_ptr() on a "
                    f"tensor without storage:\n  {exc}"
                ) from exc
            raise
