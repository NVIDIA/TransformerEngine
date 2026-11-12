# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-rank QDQ dtype, amax-reduction, and TP/SP numerical coverage."""

import os

import torch
import torch.distributed as dist
import transformer_engine.pytorch as te
from transformer_engine.common.recipe import CustomRecipe
from transformer_engine.pytorch.custom_recipes.qdq import NVFP4QDQQuantizer
from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
    mxfp8_qdq_nvfp4_qdq_fwd_mxfp8_bwd_factory,
    nvfp4_qdq_weight_fwd_bf16_bwd_factory,
)
from transformer_engine.pytorch.distributed import gather_along_first_dim
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage
from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
import transformer_engine_torch as tex


def assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def run_case(options, source_dtype, output_dtype):
    rank = dist.get_rank()
    torch.manual_seed(123 + rank)
    x = torch.randn(128, 256, device="cuda", dtype=source_dtype) * (1 if rank == 0 else 17)
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


def run_high_precision_gather(mode, requantize):
    """Communication must preserve values held beyond a Hybrid's logical dtype."""
    row = IdentityQuantizer(dtype=torch.bfloat16)
    col = IdentityQuantizer(dtype=torch.bfloat16)
    if mode == "nested_hp_row":
        row = HybridQuantizer(rowwise_quantizer=row, columnwise_quantizer=IdentityQuantizer())
    if mode == "native_column":
        col = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    quantizer = HybridQuantizer(rowwise_quantizer=row, columnwise_quantizer=col)
    quantizer.internal = True
    quantizer.set_usage(rowwise=mode != "native_column", columnwise=mode == "native_column")
    source = torch.full((32, 128), 65504.0, device="cuda", dtype=torch.float16)
    local = quantizer(source)
    assert local._dtype == torch.float16
    gathered, _ = gather_along_first_dim(
        local, dist.group.WORLD, quantizer=quantizer if requantize else None
    )
    values = (
        gathered.dequantize(dtype=torch.float32)
        if isinstance(gathered, QuantizedTensorStorage)
        else gathered.float()
    )
    assert_bits(values, torch.full((64, 128), 65536.0, device="cuda"))


def run_tp_sp_amp(factory, parallel_mode, save_original_input):
    """FP16 inputs and BF16 AMP stay finite through TP/SP forward and backward."""
    layer = te.Linear(
        128,
        128,
        params_dtype=torch.bfloat16,
        bias=False,
        device="cuda",
        tp_group=dist.group.WORLD,
        tp_size=2,
        parallel_mode=parallel_mode,
        sequence_parallel=True,
        save_original_input=save_original_input,
    )
    with torch.no_grad():
        layer.weight.fill_(1.0 / 128)
    source = torch.full(
        (64, layer.weight.shape[1]), 65504.0, device="cuda", dtype=torch.float16, requires_grad=True
    )
    with te.autocast(recipe=CustomRecipe(qfactory=factory)), torch.autocast(
        "cuda", dtype=torch.bfloat16
    ):
        out = layer(source)
    out.backward(torch.ones_like(out))
    torch.testing.assert_close(out, torch.full_like(out, 65536.0), rtol=0, atol=0)
    torch.testing.assert_close(source.grad, torch.ones_like(source), rtol=0, atol=0)
    rows = 128 if parallel_mode == "column" else 64
    torch.testing.assert_close(
        layer.weight.grad, torch.full_like(layer.weight, rows * 65536.0), rtol=0, atol=0
    )


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
            for source_dtype in (torch.float16, torch.bfloat16, torch.float32):
                for output_dtype in (torch.float16, torch.bfloat16, torch.float32):
                    run_case(options, source_dtype, output_dtype)
                    count += 1
        modes = ["hp_row", "nested_hp_row"]
        factories = [nvfp4_qdq_weight_fwd_bf16_bwd_factory]
        if te.is_mxfp8_available():
            modes.append("native_column")
            factories.append(mxfp8_qdq_nvfp4_qdq_fwd_mxfp8_bwd_factory)
        for mode in modes:
            for requantize in (False, True):
                run_high_precision_gather(mode, requantize)
                count += 1
        for factory in factories:
            for parallel_mode in ("column", "row"):
                for save_original_input in (False, True):
                    run_tp_sp_amp(factory, parallel_mode, save_original_input)
                    count += 1
        dist.barrier()
        if dist.get_rank() == 0:
            print(f"NVFP4 QDQ distributed: {count} cases passed on 2 ranks", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
