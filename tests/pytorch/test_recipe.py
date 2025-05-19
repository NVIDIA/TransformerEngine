# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from typing import Iterable, Optional

import pytest
import torch
import warnings

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine_torch as tex
from transformer_engine.pytorch.fp8 import (
    FP8GlobalStateManager,
    _amax_and_scale_update,
    fp8_model_init,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.pytorch import Linear
from transformer_engine.pytorch.distributed import fp8_autocast
from transformer_engine.common.recipe import DelayedScaling, Float8BlockScaling, MXFP8BlockScaling
import transformer_engine_torch as tex

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = FP8GlobalStateManager.is_fp8_available()
mxfp8_available, reason_for_no_mxfp8 = FP8GlobalStateManager.is_mxfp8_available()
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = (
    FP8GlobalStateManager.is_fp8_block_scaling_available()
)


# FP8 per tensor delayed scaling
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
            fp8_format=fp8_format,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )
        with te.fp8_autocast(fp8_recipe=recipe):
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
        # scale_inv_forward = fp8_meta[forward_key].scale_inv
        backward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
        amax_history_backward = fp8_meta[backward_key].amax_history
        scale_backward = fp8_meta[backward_key].scale
        # scale_inv_backward = fp8_meta[backward_key].scale_inv

        # Tweak amax history and scaling factors
        amax_history_forward.copy_(2 * torch.rand_like(amax_history_forward) + 0.5)
        amax_history_forward[0, :].zero_()
        scale_forward.copy_(2 * torch.rand_like(scale_forward) + 0.5)
        # scale_inv_forward.copy_(torch.reciprocal(scale_forward))
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
        # ref_scale_inv_forward = torch.reciprocal(ref_scale_forward)
        update_weight_amax = is_first_microbatch is None or is_first_microbatch
        # if not update_weight_amax:
        #    ref_scale_inv_forward[1].copy_(scale_inv_forward[1])
        # ref_scale_inv_backward = torch.reciprocal(ref_scale_backward)

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
        # ref_scale_inv_forward = torch.reciprocal(ref_scale_forward)
        # ref_scale_inv_backward = torch.reciprocal(ref_scale_backward)

        # Check that scale and scale inverse match expected values
        # Note: scale and scale inverse are only updated when amax is updated
        torch.testing.assert_close(
            scale_forward[0],
            ref_scale_forward[0],
        )
        if update_weight_amax:
            torch.testing.assert_close(
                scale_forward[1],
                ref_scale_forward[1],
            )
        torch.testing.assert_close(
            scale_backward[0],
            ref_scale_backward[0],
        )

    @pytest.mark.parametrize("amax_history_len", [31, 1024])
    @pytest.mark.parametrize("amax_compute_algo", ["max", "most_recent"])
    def test_fp8_scale_update_with_linear_fuser_op(
        self,
        amax_history_len: int,
        amax_compute_algo: str,
        margin: float = 2,
        num_steps: int = 4,
        in_shape: tuple[int] = (16, 16),
        dtype: torch.dtype = torch.float32,
        device: torch.device = "cuda",
    ):

        # Construct linear op
        op = te_ops.BasicLinear(in_shape[-1], in_shape[-1])

        # FP8 recipe
        forward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
        backward_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
        recipe = transformer_engine.common.recipe.DelayedScaling(
            margin=margin,
            fp8_format=fp8_format,
            amax_history_len=amax_history_len,
            amax_compute_algo=amax_compute_algo,
        )

        # Get FP8 meta tensors
        with te.fp8_autocast(fp8_recipe=recipe):
            x_fp8_meta = op.get_quantizer("forward", 0)
            w_fp8_meta = op.get_quantizer("forward", 1)
            dy_fp8_meta = op.get_quantizer("backward", 0)

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
            with te.fp8_autocast(fp8_recipe=recipe):
                y = op(x)
            y.backward(dy)

            def check_amax_history(
                fp8_meta: dict,
                ref_amax_history: Iterable[float],
            ) -> None:
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
                    test_amax_history[-(step + 1) :],
                    ref_amax_history[: (step + 1)],
                    **tols,
                )

            def check_scale(
                quantizer: Float8Quantizer,
                ref_amax_history: Iterable[float],
                stage: str,
            ):
                """Check that scale and scale reciprocal match expected values"""

                # Compute amax
                if len(ref_amax_history) > amax_history_len:
                    ref_amax_history = ref_amax_history[-(amax_history_len + 1) :]
                if amax_compute_algo == "max":
                    ref_amax = max(ref_amax_history)
                elif amax_compute_algo == "most_recent":
                    ref_amax = ref_amax_history[-1]
                else:
                    raise RuntimeError(f"{amax_compute_algo=} is not supported")

                # Compute scale
                max_val = {
                    "forward": 448.0,
                    "backward": 57344.0,
                }[stage]
                ref_scale = (max_val / ref_amax) / (2**margin)

                # Check values in FP8 meta tensors
                torch.testing.assert_close(
                    quantizer.scale.item(),
                    ref_scale,
                )

            # Check that results match expected values
            check_scale(x_fp8_meta, x_history, "forward")
            check_scale(w_fp8_meta, w_history, "forward")
            check_scale(dy_fp8_meta, dy_history, "backward")

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
            with te.fp8_autocast(fp8_recipe=recipe):
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
                recipe.amax_compute_algo,
                fp8_dtype,
                recipe.margin,
            )
        else:
            _amax_and_scale_update(
                fp8_meta[forward_key].amax_history,
                fp8_meta[forward_key].scale,
                fp8_max,
                recipe,
            )

        torch.testing.assert_close(fp8_meta[forward_key].scale, expected_scale)

    @pytest.mark.parametrize(
        "model_init_recipe",
        [
            pytest.param(
                MXFP8BlockScaling(),
                marks=pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8),
            ),
            pytest.param(
                Float8BlockScaling(),
                marks=pytest.mark.skipif(
                    not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling
                ),
            ),
        ],
    )
    def test_check_for_weight_tensor_and_recipe_correspondence(self, model_init_recipe):
        with fp8_model_init(enabled=True, recipe=model_init_recipe):
            linear = Linear(32, 32).cuda()

        x = torch.randn(32, 32, device="cuda")
        with fp8_autocast(enabled=True, fp8_recipe=DelayedScaling()):
            with pytest.raises(RuntimeError) as excinfo:
                _ = linear(x)
            assert "Recipe mismatch for " in str(excinfo.value)

    @pytest.mark.parametrize(
        "target_recipe_class, expected_quantizer_type, available_flag, reason",
        [
            pytest.param(
                MXFP8BlockScaling,
                MXFP8Quantizer,
                mxfp8_available,
                reason_for_no_mxfp8,
                id="DelayedScaling->MXFP8BlockScaling",
            ),
            pytest.param(
                Float8BlockScaling,
                Float8BlockQuantizer,
                fp8_block_scaling_available,
                reason_for_no_fp8_block_scaling,
                id="DelayedScaling->Float8BlockScaling",
            ),
        ],
    )
    def test_dynamic_recipe_update(
        self, target_recipe_class, expected_quantizer_type, available_flag, reason
    ):
        if not available_flag:
            pytest.skip(reason)

        in_features = 32
        out_features = 32
        batch_size = 32
        linear = Linear(in_features, out_features).cuda()
        initial_recipe = DelayedScaling()

        # Run initial iterations with DelayedScaling
        for _ in range(3):
            x = torch.randn(batch_size, in_features, device="cuda")
            with fp8_autocast(enabled=True, fp8_recipe=initial_recipe):
                y = linear(x)
            loss = y.mean()
            loss.backward()

        for quantizer in linear.quantizers["scaling_fwd"]:
            assert isinstance(quantizer, Float8Quantizer)

        # Change recipe
        target_recipe = target_recipe_class()

        # Run subsequent iterations with the target recipe
        for i in range(3):
            x = torch.randn(batch_size, in_features, device="cuda")
            if i == 0:
                # Expect a warning on the first iteration with the new recipe
                with pytest.warns(UserWarning, match="Recipe type changed"):
                    with fp8_autocast(enabled=True, fp8_recipe=target_recipe):
                        y = linear(x)
                for quantizer in linear.quantizers["scaling_fwd"]:
                    assert isinstance(quantizer, expected_quantizer_type)
            else:
                # No warning expected on subsequent iterations
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # Raise error if unexpected warning occurs
                    with fp8_autocast(enabled=True, fp8_recipe=target_recipe):
                        y = linear(x)
            loss = y.mean()
            loss.backward()

        # Final check
        for quantizer in linear.quantizers["scaling_fwd"]:
            assert isinstance(quantizer, expected_quantizer_type)
