# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Regression tests for the no-TMEM fused NVFP4 RHT path on SM120/SM121."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.custom_recipes.quantization_ref_nvfp4 import NVFP4QuantizerRef


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _is_sm12x(device: int = 0) -> bool:
    return torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))


def _reference_post_rht_amax(x: torch.Tensor, with_random_sign_mask: bool) -> torch.Tensor:
    quantizer = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=False,
        columnwise=True,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        with_rht=True,
        with_random_sign_mask=with_random_sign_mask,
    )
    transformed = quantizer._apply_rht(x.t().contiguous())
    return transformed.abs().amax().to(torch.float32).view(1)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("with_random_sign_mask", [False, True])
def test_sm12x_post_rht_amax_matches_aten_output(with_random_sign_mask: bool) -> None:
    """The fused post-RHT amax must describe the ATen-equivalent RHT tensor."""

    if not _is_sm12x():
        pytest.skip("Test targets the SM120/SM121 ATen RHT fallback")

    torch.manual_seed(1234)
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    expected_amax = _reference_post_rht_amax(x, with_random_sign_mask)

    quantizer = NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=False,
        columnwise=True,
        with_amax_reduction=False,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=with_random_sign_mask,
    )
    out = quantizer(x)

    torch.testing.assert_close(out._amax_columnwise, expected_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(128, 128), (256, 256)])
@pytest.mark.parametrize("rowwise", [False, True])
@pytest.mark.parametrize("with_random_sign_mask", [False, True])
def test_sm12x_fused_rht_codes_and_scales_match_aten_pipeline(
    shape: tuple[int, int], rowwise: bool, with_random_sign_mask: bool
) -> None:
    """Fused RHT codes/scales must match ATen RHT followed by the tuned quantizer."""

    if not _is_sm12x():
        pytest.skip("Test targets the SM120/SM121 no-TMEM fused RHT path")

    torch.manual_seed(2026)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    fused_quantizer = NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=rowwise,
        columnwise=True,
        with_amax_reduction=False,
        with_rht=True,
        with_post_rht_amax=True,
        with_random_sign_mask=with_random_sign_mask,
    )
    fused = fused_quantizer(x)

    reference_rht = NVFP4QuantizerRef(
        dtype=utils.Fp4Formats.E2M1,
        rowwise=False,
        columnwise=True,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        with_rht=True,
        with_random_sign_mask=with_random_sign_mask,
    )
    transformed = reference_rht._apply_rht(x.t().contiguous())
    plain_quantizer = NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        with_rht=False,
    )
    expected_columnwise = plain_quantizer(transformed)

    torch.testing.assert_close(
        fused._columnwise_data.view(torch.uint8),
        expected_columnwise._rowwise_data.view(torch.uint8),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        fused._columnwise_scale_inv,
        expected_columnwise._rowwise_scale_inv,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        fused._amax_columnwise, expected_columnwise._amax_rowwise, atol=0.0, rtol=0.0
    )


def _distributed_amax_worker(rank: int, world_size: int, init_file: str) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(4321 + rank)
        x = torch.randn((128, 128), device=f"cuda:{rank}", dtype=torch.bfloat16)
        x.mul_(rank + 1)
        expected_amax = _reference_post_rht_amax(x, with_random_sign_mask=True)

        quantizer = NVFP4Quantizer(
            fp4_dtype=te.DType.kFloat4E2M1,
            rowwise=False,
            columnwise=True,
            with_amax_reduction=True,
            amax_reduction_group=dist.group.WORLD,
            with_rht=True,
            with_post_rht_amax=True,
            with_random_sign_mask=True,
        )
        out = quantizer(x)

        dist.all_reduce(expected_amax, op=dist.ReduceOp.MAX)
        torch.testing.assert_close(out._amax_columnwise, expected_amax, atol=0.0, rtol=0.0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Two CUDA devices are required")
def test_sm12x_post_rht_amax_reduction_is_global(tmp_path) -> None:
    """ATen post-RHT amax must be computed before the distributed MAX reduction."""

    if not all(_is_sm12x(device) for device in range(2)):
        pytest.skip("Test targets the SM120/SM121 ATen RHT fallback")

    init_file = os.fspath(tmp_path / "nvfp4_rht_amax_init")
    mp.spawn(_distributed_amax_worker, args=(2, init_file), nprocs=2, join=True)
