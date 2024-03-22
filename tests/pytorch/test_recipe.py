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
import transformer_engine.pytorch.fuser
import transformer_engine_extensions as tex

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
    def test_fp8_scale_update_with_linear_module(
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

    def test_fp8_scale_update_with_linear_fuser_op(
        self,
        num_steps: int = 4,
        in_shape: tuple[int] = (16, 16),
        dtype: torch.dtype = torch.float32,
    ):
        device = torch.device("cuda")

        ### TODO Non-default recipe
        amax_history_len: int = 1024
        amax_compute_algo: str = "max"
        margin: float = 0

        # Construct linear op
        op = te.fuser.ops.UnfusedLinear(in_shape[-1], in_shape[-1])

        # Get FP8 meta tensors
        forward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
        backward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
        x_fp8_meta = op.get_fp8_meta("input")[forward_key]
        w_fp8_meta = op.get_fp8_meta("param")[forward_key]
        dy_fp8_meta = op.get_fp8_meta("grad_output")[backward_key]

        # Perform training steps
        x_history = []
        w_history = []
        dy_history = []
        for step in range(num_steps):

            # Fill tensors with known values
            x_history.append(step + 0.25)
            w_history.append(step + 0.5)
            dy_history.append(step + 0.75)
            x = torch.full(
                in_shape,
                x_history[-1],
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            dy = torch.full(
                in_shape,
                dy_history[-1],
                dtype=dtype,
                device=device,
            )
            with torch.no_grad():
                op.weight.fill_(w_history[-1])

            # Forward and backward pass
            with te.fp8_autocast(enabled=True):
                y = op(x)
            y.backward(dy)

            def check_amax_history(fp8_meta, ref_amax_history):
                """Check that amax history matches expected values"""
                if len(ref_amax_history) > amax_history_len:
                    ref_amax_history = ref_amax_history[-amax_history_len:]
                ref_amax_history = torch.tensor(
                    ref_amax_history,
                    dtype=torch.float32,
                    device=device,
                )
                test_amax_history = fp8_meta.amax_history[:, 0]
                tols = dict(rtol=0, atol=0)
                torch.testing.assert_close(
                    test_amax_history[0],
                    ref_amax_history[-1],
                    **tols,
                )
                if step > 0:
                    torch.testing.assert_close(
                        test_amax_history[-step:],
                        ref_amax_history[:step],
                        **tols,
                    )

            def check_scale(
                fp8_meta,
                ref_amax_history,
                stage,
            ):
                """Check that scale and scale reciprocal match expected values"""

                # Initial scale
                if step == 0:
                    torch.testing.assert_close(fp8_meta.scale.item(), 1.0)
                    torch.testing.assert_close(fp8_meta.scale_inv.item(), 1.0)
                    return

                # Compute amax
                if len(ref_amax_history) > amax_history_len:
                    ref_amax_history = ref_amax_history[-amax_history_len:]
                if amax_compute_algo == "max":
                    ref_amax = max(ref_amax_history[:-1])
                elif amax_compute_algo == "most_recent":
                    ref_amax = ref_amax_history[-2]
                else:
                    raise RuntimeError(f"{amax_compute_algo=} is not supported")

                # Compute scale
                max_val ={
                    "forward": 448.0,
                    "backward": 57344.0,
                }[stage]
                ref_scale = (max_val / ref_amax) / (2 ** margin)

                # Check values in FP8 meta tensors
                torch.testing.assert_close(
                    fp8_meta.scale.item(),
                    ref_scale,
                )
                torch.testing.assert_close(
                    fp8_meta.scale_inv.item(),
                    1 / ref_scale,
                )

            # Check that results match expected values
            check_amax_history(x_fp8_meta, x_history)
            check_amax_history(w_fp8_meta, w_history)
            check_amax_history(dy_fp8_meta, dy_history)
            check_scale(x_fp8_meta, x_history, "forward")
            check_scale(w_fp8_meta, w_history, "forward")
            check_scale(dy_fp8_meta, dy_history, "backward")
