# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0
"""EP transport rejects scale layouts it cannot address without strides."""

import pytest
import torch
import transformer_engine.pytorch  # Load the common library before the binding.
import transformer_engine_torch as tex


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 9
    or not hasattr(tex, "ep_dispatch"),
    reason="EP block scaling requires Hopper or newer and an NCCL EP build",
)


def test_dispatch_rejects_padded_scale_rows():
    rows, capacity, hidden, top_k = 32, 128, 512, 2
    tokens = torch.empty(rows, hidden, dtype=torch.float8_e4m3fn, device="cuda")
    recv_tokens = torch.empty(capacity, hidden, dtype=torch.float8_e4m3fn, device="cuda")
    kwargs = dict(
        handle_mem=torch.empty(1, dtype=torch.uint8, device="cuda"),
        topk_idx=torch.zeros(rows, top_k, dtype=torch.int64, device="cuda"),
        tokens=tokens,
        topk_weights=torch.ones(rows, top_k, device="cuda"),
        recv_tokens=recv_tokens,
        recv_topk_weights=torch.empty(capacity, device="cuda"),
        top_k=top_k,
        dispatch_output_per_expert_alignment=0,
        tokens_scale_inv=torch.empty(rows, hidden // 128 + 1, device="cuda"),
        recv_scale_inv=torch.empty(capacity, hidden // 128, device="cuda"),
    )
    with pytest.raises(RuntimeError, match="non-zero divisor|must equal rows|must match send"):
        tex.ep_dispatch(**kwargs)
