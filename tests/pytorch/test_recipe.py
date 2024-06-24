# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional

import pytest
import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    _amax_and_scale_update,
    get_default_fp8_recipe,
)

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
class TestFP8Recipe:

    @staticmethod
    def setup_class(cls) -> None:
        # Configure RNG
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    @pytest.mark.parametrize("amax_history_len", [31, 1024])
    @pytest.mark.parametrize("amax_compute_algo", ["max", "most_recent"])
    @pytest.mark.parametrize("is_first_microbatch", [None, True, False])
    def test_amax_and_scale_update(
        self,
        amax_history_len: int,
        amax_compute_algo: str,
        is_first_microbatch: Optional[bool],
        margin: int = 2,
    ):

        # Construct linear module
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
        recipe = transformer_engine.common.recipe.DelayedScaling(
            margin=margin,
            fp8_format=fp8_format,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            module = te.Linear(16, 16)
            y = module(
                torch.randn([16, 16], device="cuda"),
                is_first_microbatch=True,
            )
        y.backward(torch.zeros_like(y))

        # Get amax history and scaling factors
        fp8_meta = module.fp8_meta
        forward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
        amax_history_forward = fp8_meta[forward_key].amax_history
        scale_forward = fp8_meta[forward_key].scale
        scale_inv_forward = fp8_meta[forward_key].scale_inv
        backward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
        amax_history_backward = fp8_meta[backward_key].amax_history
        scale_backward = fp8_meta[backward_key].scale
        scale_inv_backward = fp8_meta[backward_key].scale_inv

        # Tweak amax history and scaling factors
        amax_history_forward.copy_(2 * torch.rand_like(amax_history_forward) + 0.5)
        amax_history_forward[0, :].zero_()
        scale_forward.copy_(2 * torch.rand_like(scale_forward) + 0.5)
        scale_inv_forward.copy_(torch.reciprocal(scale_forward))
        amax_history_backward[0, :].zero_()

        # Expected amax history after update
        # Note: amax history is only updated when amax is updated
        update_weight_amax = is_first_microbatch is None or is_first_microbatch
        ref_amax_history_forward = amax_history_forward.clone()
        ref_amax_history_forward[:, 0].copy_(torch.roll(amax_history_forward[:, 0], -1))
        if update_weight_amax:
            ref_amax_history_forward[:, 1].copy_(torch.roll(amax_history_forward[:, 1], -1))
        ref_amax_history_forward[0, :].zero_()
        ref_amax_history_backward = amax_history_backward.clone()
        ref_amax_history_backward[:, 0].copy_(torch.roll(amax_history_backward[:, 0], -1))
        ref_amax_history_backward[0, :].zero_()

        # Expected scale and scale inverse
        if amax_compute_algo == "max":
            ref_amax_forward = amax_history_forward.max(dim=0).values
            ref_amax_backward = amax_history_backward.max(dim=0).values
        elif amax_compute_algo == "most_recent":
            ref_amax_forward = amax_history_forward[-1]
            ref_amax_backward = amax_history_backward[-1]
        else:
            raise ValueError(f"{amax_compute_algo=} is not supported")
        ref_scale_forward = (fp8_format.value.max_fwd / ref_amax_forward) / (2**margin)
        ref_scale_backward = (fp8_format.value.max_bwd / ref_amax_backward) / (2**margin)
        ref_scale_inv_forward = torch.reciprocal(ref_scale_forward)
        update_weight_amax = is_first_microbatch is None or is_first_microbatch
        if not update_weight_amax:
            ref_scale_inv_forward[1].copy_(scale_inv_forward[1])
        ref_scale_inv_backward = torch.reciprocal(ref_scale_backward)

        # Perform forward, backward, and optimizer steps to update fp8_meta
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            x = torch.randn([16, 16], device="cuda")
            y = module(x, is_first_microbatch=is_first_microbatch)
        y.backward(torch.randn_like(y))

        # Check that amax history matches expected values
        torch.testing.assert_close(
            amax_history_forward[:-1],
            ref_amax_history_forward[:-1],
        )
        torch.testing.assert_close(
            amax_history_backward[:-1],
            ref_amax_history_backward[:-1],
        )

        # Expected scale and scale inverse
        if amax_compute_algo == "max":
            ref_amax_forward = amax_history_forward.max(dim=0).values
            ref_amax_backward = amax_history_backward.max(dim=0).values
        elif amax_compute_algo == "most_recent":
            ref_amax_forward = amax_history_forward[-1]
            ref_amax_backward = amax_history_backward[-1]
        else:
            raise ValueError(f"{amax_compute_algo=} is not supported")
        ref_scale_forward = (fp8_format.value.max_fwd / ref_amax_forward) / (2**margin)
        ref_scale_backward = (fp8_format.value.max_bwd / ref_amax_backward) / (2**margin)
        ref_scale_inv_forward = torch.reciprocal(ref_scale_forward)
        ref_scale_inv_backward = torch.reciprocal(ref_scale_backward)

        # Check that scale and scale inverse match expected values
        # Note: scale and scale inverse are only updated when amax is updated
        torch.testing.assert_close(
            scale_forward[0],
            ref_scale_forward[0],
        )
        torch.testing.assert_close(
            scale_inv_forward[0],
            ref_scale_inv_forward[0],
        )
        if update_weight_amax:
            torch.testing.assert_close(
                scale_forward[1],
                ref_scale_forward[1],
            )
            torch.testing.assert_close(
                scale_inv_forward[1],
                ref_scale_inv_forward[1],
            )
        torch.testing.assert_close(
            scale_backward[0],
            ref_scale_backward[0],
        )
        torch.testing.assert_close(
            scale_inv_backward[0],
            ref_scale_inv_backward[0],
        )

    @pytest.mark.parametrize("amax_case", ["zero", "tiny", "normal", "inf", "nan"])
    @pytest.mark.parametrize("fused_update", [True, False], ids=["fused", "non-fused"])
    @pytest.mark.parametrize(
        "fp8_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2], ids=["E4M3", "E5M2"]
    )
    def test_scale_update_numeric_scenarios(self, amax_case, fused_update, fp8_dtype):

        if fp8_dtype == tex.DType.kFloat8E4M3:
            fp8_format = transformer_engine.common.recipe.Format.E4M3
            fp8_max = fp8_format.value.max_fwd
        elif fp8_dtype == tex.DType.kFloat8E5M2:
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
            fp8_max = fp8_format.value.max_bwd
        else:
            raise ValueError(f"{fp8_dtype=} is not supported")

        scaling_factor_compute_algo = None
        if fused_update:
            scaling_factor_compute_algo = (
                lambda amax, scale, fp8_max, recipe: te.fp8._default_sf_compute(
                    amax, scale, fp8_max, recipe.margin
                )
            )
        recipe = transformer_engine.common.recipe.DelayedScaling(
            fp8_format=fp8_format, scaling_factor_compute_algo=scaling_factor_compute_algo
        )

        # Setup fp8_meta dictionary
        def setup_fp8_meta():
            with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
                module = te.Linear(16, 16)
                y = module(torch.zeros([16, 16], device="cuda"))
            y.backward(torch.zeros_like(y))
            return module.fp8_meta

        fp8_meta = setup_fp8_meta()
        forward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)

        # Replace the fp8_meta[forward_key] with a new TensorMeta for test purpose
        fp8_meta[forward_key] = tex.FP8TensorMeta()
        fp8_meta[forward_key].scale = torch.ones(1, dtype=torch.float32, device="cuda")
        fp8_meta[forward_key].scale_inv = torch.ones(1, dtype=torch.float32, device="cuda")

        # test different scenarios
        if amax_case == "zero":
            fp8_meta[forward_key].amax_history = torch.tensor(
                [[0]], dtype=torch.float32, device="cuda"
            )
            expected_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        elif amax_case == "tiny":
            # calculate the minimum amax value that results in a FP32 maximum scale
            fp32_max = torch.tensor(torch.finfo(torch.float32).max)
            tiny_amax = fp8_max / fp32_max
            # make the amax less than the minimum amax so that the scale will be infinite
            amax_value = tiny_amax / 2
            fp8_meta[forward_key].amax_history = torch.tensor(
                [[amax_value]], dtype=torch.float32, device="cuda"
            )
            # expected scale is FP32_max
            expected_scale = fp32_max.view(1).cuda()
        elif amax_case == "normal":
            # plus a small epsilon to avoid zero amax
            amax_value = torch.rand(1, dtype=torch.float32, device="cuda") + 1e-5
            fp8_meta[forward_key].amax_history = amax_value.view(1, 1)
            expected_scale = fp8_max / amax_value
        elif amax_case == "inf":
            fp8_meta[forward_key].amax_history = torch.tensor(
                [[torch.inf]], dtype=torch.float32, device="cuda"
            )
            expected_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
        elif amax_case == "nan":
            fp8_meta[forward_key].amax_history = torch.tensor(
                [[torch.nan]], dtype=torch.float32, device="cuda"
            )
            expected_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        if fused_update:
            tex.fused_amax_and_scale_update_after_reduction(
                fp8_meta[forward_key].amax_history.clone().view(-1),
                [fp8_meta[forward_key].amax_history],
                [fp8_meta[forward_key].scale],
                [fp8_meta[forward_key].scale_inv],
                recipe.amax_compute_algo,
                fp8_dtype,
                recipe.margin,
            )
        else:
            _amax_and_scale_update(
                fp8_meta[forward_key].amax_history,
                fp8_meta[forward_key].scale,
                fp8_meta[forward_key].scale_inv,
                fp8_max,
                recipe,
            )

        torch.testing.assert_close(fp8_meta[forward_key].scale, expected_scale)
        torch.testing.assert_close(
            fp8_meta[forward_key].scale_inv, torch.reciprocal(expected_scale)
        )
