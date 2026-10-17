# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""The fused grouped MLP must not save the NVFP4 rowwise input copies for backward.

Every wgrad path (the cuDNN grouped kernel, the single-group dense GEMM and the generic grouped
GEMM fallback) reads only the columnwise tiles (data and scales) of the saved FC1 and FC2 inputs,
so the rowwise copies are dead after forward. Saving them doubled the activation memory held (and
offloaded) for the quantized inputs.

Checked through ``saved_tensors_hooks``: the saved set must hold exactly one FP4 data tensor and
one scale tensor per quantized input. Numerics are covered by ``test_grouped_mlp.py``.

Weights require grad here: with frozen weights the input quantizers skip the columnwise copy and
the fused op's ``mark_grouped_tensor`` call rejects the input before the pruning is reached, on
this change and before it alike.
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.ops.fused.grouped_mlp import GroupedMLP_CuTeGEMMGLU

from utils import make_recipe

nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)


# (group_size, tokens_per_group): token counts chosen so the quantized inputs' byte sizes never
# coincide with a weight's, which the saved-tensor counting below relies on.
@pytest.mark.parametrize("group_size,tokens_per_group", ((1, 768), (4, 256)))
def test_fused_grouped_mlp_nvfp4_saves_columnwise_inputs_only(
    group_size,
    tokens_per_group,
    *,
    hidden_size: int = 256,
    glu_interleave_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not nvfp4_available:
        pytest.skip(reason_for_no_nvfp4)
    if not GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("Fused grouped MLP (CuTeDSL) is not supported on this system")
    torch.manual_seed(0)

    total_tokens = group_size * tokens_per_group
    split_sizes = torch.tensor([tokens_per_group] * group_size, dtype=torch.int, device=device)
    # The fused grouped MLP requires a split alignment divisible by 256 (the default is 128).
    fc1 = te.ops.GroupedLinear(
        group_size,
        hidden_size,
        2 * hidden_size,
        bias=False,
        device=device,
        dtype=dtype,
        split_alignment=256,
    )
    fc2 = te.ops.GroupedLinear(
        group_size,
        hidden_size,
        hidden_size,
        bias=False,
        device=device,
        dtype=dtype,
        split_alignment=256,
    )
    module = te.ops.Sequential(
        fc1, te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size), fc2
    )
    x = (torch.rand((total_tokens, hidden_size), dtype=dtype, device=device) - 0.5).requires_grad_()
    probs = torch.rand((total_tokens,), dtype=dtype, device=device)
    dy = torch.rand((total_tokens, hidden_size), dtype=dtype, device=device) - 0.5

    saved = []

    def pack(t):
        if isinstance(t, torch.Tensor):
            saved.append((t.dtype, t.numel()))
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        with te.autocast(enabled=True, recipe=make_recipe("nvfp4_rht")):
            y = module(x, split_sizes, probs, split_sizes)
    forward_ops = module._module_groups[0]._forward_ops
    assert isinstance(forward_ops[0][0], GroupedMLP_CuTeGEMMGLU)

    # Byte counts that identify the quantized FC1/FC2 inputs (both are [tokens, hidden_size] in
    # NVFP4: 2 elements per byte, one E4M3 scale per 16 elements). Weights have other sizes.
    fp4_data_numel = total_tokens * hidden_size // 2
    fp4_scale_numel = total_tokens * hidden_size // 16
    n_data = sum(1 for d, n in saved if d == torch.uint8 and n == fp4_data_numel)
    n_scale = sum(1 for d, n in saved if d == torch.uint8 and n == fp4_scale_numel)
    # One columnwise data + one columnwise scale tensor per quantized input (FC1 x, FC2 x).
    assert n_data == 2, f"expected 2 FP4 input data tensors saved (columnwise only), got {n_data}"
    assert n_scale == 2, f"expected 2 FP4 input scale tensors saved, got {n_scale}"

    # Backward still runs on what was saved.
    y.backward(dy)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for p in module.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
