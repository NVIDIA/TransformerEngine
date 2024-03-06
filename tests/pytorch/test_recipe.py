# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Optional

import pytest
import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    amax_and_scale_update,
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

    @pytest.mark.parametrize("amax_history_len", [1, 31, 1024])
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
            interval=1,
            fp8_format=fp8_format,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            module = te.Linear(16, 16)
            y = module(torch.zeros([16, 16], device="cuda"))
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
        if amax_history_len > 1:
            amax_history_forward[1, 0].fill_(3)
        scale_forward.copy_(2 * torch.rand_like(scale_forward) + 0.5)
        scale_inv_forward.copy_(torch.reciprocal(scale_forward))
        amax_history_backward.copy_(2 * torch.rand_like(amax_history_backward) + 0.5)
        scale_backward.copy_(2 * torch.rand_like(scale_backward) + 0.5)
        scale_inv_backward.copy_(torch.reciprocal(scale_backward))

        # Expected amax history after update
        ref_amax_history_forward = torch.roll(amax_history_forward, -1, dims=0)
        ref_amax_history_forward[0].zero_()
        ref_amax_history_backward = torch.roll(amax_history_backward, -1, dims=0)
        ref_amax_history_backward[0].zero_()

        # Expected scale and scale inverse
        if amax_compute_algo == "max":
            ref_amax_forward = amax_history_forward.max(dim=0).values
            ref_amax_backward = amax_history_backward.max(dim=0).values
        elif amax_compute_algo == "most_recent":
            ref_amax_forward = amax_history_forward[0]
            ref_amax_backward = amax_history_backward[0]
        else:
            raise ValueError(f"{amax_compute_algo=} is not supported")
        ref_scale_forward = (fp8_format.value.max_fwd / ref_amax_forward) / (2 ** margin)
        ref_scale_backward = (fp8_format.value.max_bwd / ref_amax_backward) / (2 ** margin)
        ref_scale_inv_forward = torch.reciprocal(ref_scale_forward)
        update_weight_scale_inv = is_first_microbatch is None or is_first_microbatch
        if not update_weight_scale_inv:
            ref_scale_inv_forward[1].copy_(scale_inv_forward[1])
        ref_scale_inv_backward = torch.reciprocal(ref_scale_backward)

        # Make sure we are not trivially passing tests
        if amax_history_len > 1:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    amax_history_forward[1:],
                    ref_amax_history_forward[1:],
                )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                scale_forward,
                ref_scale_forward,
            )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                scale_inv_forward,
                ref_scale_inv_forward,
            )
        if amax_history_len > 1:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(
                    fp8_meta[backward_key].amax_history[1:],
                    ref_amax_history_backward[1:],
                )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                fp8_meta[backward_key].scale,
                ref_scale_backward,
            )
        with pytest.raises(AssertionError):
            torch.testing.assert_close(
                fp8_meta[backward_key].scale_inv,
                ref_scale_inv_backward,
            )

        # Perform forward and backward pass to update fp8_meta
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            x = torch.zeros([16, 16], device="cuda")
            y = module(x, is_first_microbatch=is_first_microbatch)
        y.backward(torch.zeros_like(y))

        # Check that fp8_meta matches expected values
        torch.testing.assert_close(
            fp8_meta[forward_key].amax_history[1:],
            ref_amax_history_forward[1:],
        )
        torch.testing.assert_close(
            fp8_meta[forward_key].scale,
            ref_scale_forward,
        )
        torch.testing.assert_close(
            fp8_meta[forward_key].scale_inv,
            ref_scale_inv_forward,
        )
        torch.testing.assert_close(
            fp8_meta[backward_key].amax_history[1:],
            ref_amax_history_backward[1:],
        )
        torch.testing.assert_close(
            fp8_meta[backward_key].scale,
            ref_scale_backward,
        )
        torch.testing.assert_close(
            fp8_meta[backward_key].scale_inv,
            ref_scale_inv_backward,
        )
