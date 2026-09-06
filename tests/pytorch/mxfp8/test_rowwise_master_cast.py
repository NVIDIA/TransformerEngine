# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Run with pytest on one GPU or torchrun -m pytest on multiple GPUs."""

import os

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.utils import (
    cast_master_weights_to_fp8,
    quantize_master_weights,
)

available, reason = te.is_mxfp8_available(return_reason=True)
pytestmark = pytest.mark.skipif(not available, reason=reason)


@pytest.fixture
def group():
    owned = not torch.distributed.is_initialized()
    if owned:
        torch.cuda.set_device(int(os.getenv("LOCAL_RANK", "0")))
        if "RANK" in os.environ:
            torch.distributed.init_process_group("nccl")
        else:
            torch.distributed.init_process_group(
                "nccl", store=torch.distributed.HashStore(), rank=0, world_size=1
            )
    yield torch.distributed.group.WORLD
    if owned:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("fragments", [False, True])
def test_rowwise_master_cast(group, monkeypatch, mixed, fragments):
    rank = torch.distributed.get_rank(group)
    world = torch.distributed.get_world_size(group)
    shape = (160, 96)  # Both scale padding and non-tile-aligned logical shapes.
    full = torch.linspace(-4, 7, 160 * 96, device="cuda").reshape(shape)
    quantizers = [te.MXFP8Quantizer(te.DType.kFloat8E4M3, rowwise=True, columnwise=False)]
    if mixed:
        quantizers.append(te.MXFP8Quantizer(te.DType.kFloat8E4M3, rowwise=True, columnwise=True))
    weights = [q(full.to(torch.bfloat16)) for q in quantizers]
    pointers = [(w._rowwise_data.data_ptr(), w._rowwise_scale_inv.data_ptr()) for w in weights]
    if world == 1:
        lo, hi = 0, full.numel()
    elif rank == 0:
        lo, hi = 0, 17  # Split inside a 32-value block.
    elif rank == 1:
        lo, hi = 17, full.numel()
    else:
        lo, hi = 0, 0  # No master shard on tail ranks.
    real_reduce = torch.distributed.all_reduce
    reduced_sizes = []

    def record_reduce(tensor, *args, **kwargs):
        reduced_sizes.append(tensor.numel())
        return real_reduce(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.distributed, "all_reduce", record_reduce)
    for step in range(2):
        master = full + step * 0.25
        shard = master.flatten()[lo:hi] if lo < hi else None
        outputs = [
            (
                torch.empty(hi - lo, dtype=torch.uint8, device="cuda"),
                torch.empty(hi - lo, dtype=torch.uint8, device="cuda") if i else None,
            )
            for i in range(len(weights))
        ]
        reduced_sizes.clear()
        cast = quantize_master_weights if step == 0 else cast_master_weights_to_fp8
        cast(
            weights,
            [shard] * len(weights),
            [lo if shard is not None else None] * len(weights),
            group,
            fsdp_shard_model_weights=outputs if fragments else None,
        )
        expected_amax = sum(w._rowwise_scale_inv.numel() for w in weights)
        if mixed:
            expected_amax += weights[1]._columnwise_scale_inv.numel()
        assert reduced_sizes == [expected_amax]
        for idx, (weight, quantizer) in enumerate(zip(weights, quantizers)):
            expected = quantizer(master.to(torch.bfloat16))
            for direction in ["rowwise", "columnwise"] if idx else ["rowwise"]:
                data = getattr(weight, f"_{direction}_data")
                actual = torch.zeros_like(data).flatten()
                if lo < hi:
                    src = (
                        outputs[idx][direction == "columnwise"]
                        if fragments
                        else data.flatten()[lo:hi]
                    )
                    actual[lo:hi].copy_(src)
                real_reduce(actual, op=torch.distributed.ReduceOp.MAX, group=group)
                torch.testing.assert_close(
                    actual.view(shape), getattr(expected, f"_{direction}_data"), rtol=0, atol=0
                )
                scale = getattr(weight, f"_{direction}_scale_inv")
                ref = getattr(expected, f"_{direction}_scale_inv")
                rows, cols = (160, 3) if direction == "rowwise" else (5, 96)
                torch.testing.assert_close(scale[:rows, :cols], ref[:rows, :cols], rtol=0, atol=0)
            assert pointers[idx] == (
                weight._rowwise_data.data_ptr(),
                weight._rowwise_scale_inv.data_ptr(),
            )
        assert weights[0]._columnwise_data is None
        assert weights[0]._columnwise_scale_inv is None


@pytest.mark.parametrize("offset,length", [(0, 0), (31, 3), (1, 4094)])
def test_rowwise_partial_kernels(offset, length):
    inp = torch.randn(length, device="cuda", dtype=torch.bfloat16)
    row = torch.zeros((128, 4), device="cuda", dtype=inp.dtype)
    ref = torch.zeros_like(row)
    col = torch.zeros((4, 128), device="cuda", dtype=inp.dtype)
    # Empty view may still have backing storage. Omission is shape-based.
    omitted = col.flatten()[:0].view(0, 0)
    tex.mxfp8_scaling_compute_partial_amax(inp, row, omitted, 64, 64, offset)
    tex.mxfp8_scaling_compute_partial_amax(inp, ref, col, 64, 64, offset)
    torch.testing.assert_close(row, ref, rtol=0, atol=0)
    scales = torch.full((128, 4), 127, dtype=torch.uint8, device="cuda")
    out = torch.empty(length, dtype=torch.uint8, device="cuda")
    tex.mxfp8_scaling_partial_cast(
        inp, out, out[:0], scales, scales.flatten()[:0].view(0, 0), 64, 64, offset
    )
    torch.testing.assert_close(out, inp.to(torch.float8_e4m3fn).view(torch.uint8), rtol=0, atol=0)
