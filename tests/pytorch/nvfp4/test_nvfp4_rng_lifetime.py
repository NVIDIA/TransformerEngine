# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Stochastic quantization must use the restored RNG regardless of allocation history."""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import NVFP4Quantizer

recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("columns", [32, 64, 96, 128])
def test_nvfp4_stochastic_rounding_allocation_independence(columns):
    """Cover small unfused RHT shapes and a fused shape with logical-byte checks."""
    device = torch.cuda.current_device()
    cuda_device = torch.device("cuda", device)
    was_enabled = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    was_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        with torch.random.fork_rng(devices=[device]):
            generator = torch.Generator(device=cuda_device).manual_seed(2027)
            data = torch.randn(
                128, columns, device=cuda_device, dtype=torch.bfloat16, generator=generator
            )
            quantizer = NVFP4Quantizer(
                rowwise=True,
                columnwise=True,
                with_amax_reduction=False,
                with_rht=True,
                with_post_rht_amax=True,
                stochastic_rounding=True,
                with_2d_quantization=False,
            )
            quantizer.optimize_for_gemm = False

            def snapshot():
                output = quantizer(data)
                torch.cuda.synchronize()
                # Exclude padding from the ordinary scale-factor layout.
                tensors = {
                    "row_data": output._rowwise_data,
                    "column_data": output._columnwise_data,
                    "row_scale": output._rowwise_scale_inv[:128, : columns // 16],
                    "column_scale": output._columnwise_scale_inv[:columns, :8],
                    "row_amax": output._amax_rowwise,
                    "column_amax": output._amax_columnwise,
                }
                return {
                    name: tensor.contiguous().reshape(-1).view(torch.uint8).cpu().numpy().tobytes()
                    for name, tensor in tensors.items()
                }

            reference = {}
            for fragmented in (False, True, False):
                anchors = []
                if fragmented:
                    anchors = [
                        torch.empty(64, dtype=torch.int64, device=cuda_device) for _ in range(64)
                    ]
                    torch.cuda.synchronize()
                    for index in range(1, len(anchors), 2):
                        anchors[index] = None
                outputs = []
                for seed in (101, 102, 103):
                    torch.cuda.manual_seed(seed)
                    state = torch.cuda.get_rng_state()
                    actual = snapshot()
                    torch.cuda.set_rng_state(state)
                    replay = snapshot()
                    mismatches = [name for name in actual if actual[name] != replay[name]]
                    assert not mismatches, f"Restored-RNG replay differs: {mismatches}"
                    if seed not in reference:
                        reference[seed] = actual
                    mismatches = [name for name in actual if actual[name] != reference[seed][name]]
                    assert not mismatches, f"Allocation history changed seeded output: {mismatches}"
                    outputs.append(actual)
                for component in ("row_data", "column_data"):
                    assert (
                        len({output[component] for output in outputs}) > 1
                    ), f"Stochastic rounding did not respond to different seeds: {component}"
                del anchors
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=was_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = was_fill
