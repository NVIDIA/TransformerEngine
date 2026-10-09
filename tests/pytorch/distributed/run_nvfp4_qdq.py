# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-rank NVFP4 QDQ amax reduction and native reference parity."""

import os

import torch
import torch.distributed as dist
from transformer_engine.pytorch.custom_recipes.qdq import NVFP4QDQQuantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
import transformer_engine_torch as tex


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def run_case(options, source_dtype):
    rank = dist.get_rank()
    torch.manual_seed(123 + rank)
    x = torch.randn(128, 256, device="cuda", dtype=source_dtype) * (1 if rank == 0 else 17)
    output_dtype = torch.bfloat16
    local_amax = x.float().abs().amax().reshape(1)
    global_amax = local_amax.clone()
    dist.all_reduce(global_amax, op=dist.ReduceOp.MAX)
    assert dist.get_world_size() == 2
    if rank == 0:
        assert global_amax.item() > local_amax.item()
    native = NVFP4Quantizer(rowwise=True, columnwise=False, **options)
    native.internal = True
    q = NVFP4QDQQuantizer(native)
    q.amax_reduction_group = dist.group.WORLD
    q.with_amax_reduction = True
    assert q.selected_backend(x, output_dtype) == "reference"
    out = q.quantize(x, dtype=output_dtype)
    expected = tex.nvfp4_quantize_with_amax(x, native, global_amax, global_amax).dequantize(
        dtype=output_dtype
    )
    assert_bits(out.dequantize(), expected)
    saved = out._get_quantizer()
    assert not saved.with_amax_reduction and saved.amax_reduction_group is None
    assert q.with_amax_reduction and q.amax_reduction_group is dist.group.WORLD
    pointer = out._hp_data.data_ptr()
    x.mul_(2)
    global_amax.mul_(2)
    q.update_quantized(x, out)
    expected = tex.nvfp4_quantize_with_amax(x, native, global_amax, global_amax).dequantize(
        dtype=output_dtype
    )
    assert_bits(out.dequantize(), expected)
    assert out._hp_data.data_ptr() == pointer
    # Saved workspaces intentionally update locally, without persistent collectives.
    out.quantize_(x)
    assert_bits(out.dequantize(), native(x).dequantize(dtype=output_dtype))


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        count = 0
        for options in (
            {},
            {"with_2d_quantization": True},
            {
                "nvfp4_use_4over6": True,
                "nvfp4_e4m3_max": 448,
                "nvfp4_4over6_err_mode": "MSE",
            },
        ):
            for source_dtype in (torch.bfloat16, torch.float32):
                run_case(options, source_dtype)
                count += 1
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"NVFP4 QDQ distributed: {count} cases passed on 2 ranks", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
