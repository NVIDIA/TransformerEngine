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

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


def _is_sm12x(device: int = 0) -> bool:
    return torch.cuda.get_device_capability(device) in ((12, 0), (12, 1))


def _native_unfused_columnwise(x: torch.Tensor, with_random_sign_mask: bool):
    """Run TE's native K=16 RHT with cast fusion disabled."""

    env_name = "NVTE_NVFP4_DISABLE_RHT_CAST_FUSION"
    old_value = os.environ.get(env_name)
    os.environ[env_name] = "1"
    try:
        quantizer = NVFP4Quantizer(
            fp4_dtype=te.DType.kFloat4E2M1,
            rowwise=False,
            columnwise=True,
            with_rht=True,
            with_post_rht_amax=True,
            with_random_sign_mask=with_random_sign_mask,
        )
        return quantizer(x)
    finally:
        if old_value is None:
            os.environ.pop(env_name)
        else:
            os.environ[env_name] = old_value


def _unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    unpacked = x.view(torch.uint8).repeat_interleave(2, dim=-1)
    unpacked[..., 0::2] &= 0x0F
    unpacked[..., 1::2] >>= 4
    return unpacked


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("with_random_sign_mask", [False, True])
def test_sm12x_post_rht_amax_matches_native_k16(with_random_sign_mask: bool) -> None:
    """The fused post-RHT amax must match TE's native K=16 MMA RHT."""

    if not _is_sm12x():
        pytest.skip("Test targets the SM120/SM121 no-TMEM fused RHT path")

    torch.manual_seed(1234)
    x = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(5678)
    expected = _native_unfused_columnwise(x, with_random_sign_mask)

    torch.manual_seed(5678)
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

    torch.testing.assert_close(out._amax_columnwise, expected._amax_columnwise, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("shape", [(128, 128), (256, 256)])
@pytest.mark.parametrize("rowwise", [False, True])
@pytest.mark.parametrize("with_random_sign_mask", [False, True])
@pytest.mark.parametrize("seed", [1234, 2026])
def test_sm12x_fused_rht_codes_and_scales_match_native_k16(
    shape: tuple[int, int], rowwise: bool, with_random_sign_mask: bool, seed: int
) -> None:
    """Fused RHT codes/scales must match TE's native K=16 MMA RHT path."""

    if not _is_sm12x():
        pytest.skip("Test targets the SM120/SM121 no-TMEM fused RHT path")

    torch.manual_seed(seed)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(5678)
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

    torch.manual_seed(5678)
    expected = _native_unfused_columnwise(x, with_random_sign_mask)

    torch.testing.assert_close(
        _unpack_fp4(fused._columnwise_data),
        _unpack_fp4(expected._columnwise_data),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        fused._columnwise_scale_inv,
        expected._columnwise_scale_inv,
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        fused._amax_columnwise, expected._amax_columnwise, atol=0.0, rtol=0.0
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
        torch.manual_seed(5678)
        expected_amax = _native_unfused_columnwise(x, with_random_sign_mask=True)._amax_columnwise

        torch.manual_seed(5678)
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
        pytest.skip("Test targets the SM120/SM121 no-TMEM fused RHT path")

    init_file = os.fspath(tmp_path / "nvfp4_rht_amax_init")
    mp.spawn(_distributed_amax_worker, args=(2, init_file), nprocs=2, join=True)
