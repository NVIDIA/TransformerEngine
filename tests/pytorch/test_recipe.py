# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import fields
from typing import Optional

import pickle

import pytest
import torch
import warnings

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    MXFP8Quantizer,
    Float8Quantizer,
    NVFP4Quantizer,
    quantized_model_init,
    Linear,
    LayerNormLinear,
    LayerNormMLP,
    GroupedLinear,
)

import transformer_engine_torch as tex
from transformer_engine.pytorch.quantization import (
    FP8GlobalStateManager,
    NVFP4BlockScalingRecipeState,
    QuantizerRole,
    _amax_and_scale_update,
)
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import (
    CustomRecipe,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    Recipe,
    quantizer_policy,
)
from transformer_engine.pytorch._extra_state import (
    CheckpointExtraStatePolicy,
    UNSAFE_PICKLE_EXTRA_STATE_ENV,
    _RECIPE_POLICIES,
    should_load_extra_state_pickle,
)

# Check if FP8 is supported
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
fp4_available, reason_for_no_fp4 = te.is_nvfp4_available(return_reason=True)


def test_recipe_quantizer_config_cache_and_invalidation():
    """Semantic configuration is built once and rebuilt after recipe mutation."""

    config_builds = []

    class CountingRecipe(Recipe):
        def __init__(self, value: int) -> None:
            self.value = value

        def _make_quantizer_config(self):
            config_builds.append(self.value)
            return ("counting_recipe", self.value)

        def _make_repr(self) -> str:
            return f"counting_recipe={self.value}"

    recipe = CountingRecipe(1)
    assert recipe.quantizer_config() == ("counting_recipe", 1)
    assert recipe.quantizer_config() == ("counting_recipe", 1)
    assert config_builds == [1]

    # Populating the independent repr cache must not invalidate the config cache.
    assert repr(recipe) == "counting_recipe=1"
    assert recipe.quantizer_config() == ("counting_recipe", 1)
    assert config_builds == [1]

    recipe.value = 2
    assert recipe.quantizer_config() == ("counting_recipe", 2)
    assert config_builds == [1, 2]


def test_recipe_quantizer_config_must_be_hashable():
    """Reject mutable containers that cannot serve as runtime-key components."""

    class UnhashableConfigRecipe(Recipe):
        def _make_quantizer_config(self):
            return ["mutable"]

        def _make_repr(self) -> str:
            return "unhashable_config_recipe"

    with pytest.raises(
        TypeError,
        match=r"UnhashableConfigRecipe\._make_quantizer_config\(\) must return a hashable value",
    ):
        UnhashableConfigRecipe().quantizer_config()


@pytest.mark.parametrize(
    "recipe_type,derived_fields",
    [
        (DelayedScaling, ()),
        (
            Float8CurrentScaling,
            ("fp8_quant_fwd_inp", "fp8_quant_fwd_weight", "fp8_quant_bwd_grad"),
        ),
        (MXFP8BlockScaling, ()),
        (
            Float8BlockScaling,
            ("fp8_quant_fwd_inp", "fp8_quant_fwd_weight", "fp8_quant_bwd_grad"),
        ),
        (
            NVFP4BlockScaling,
            ("fp4_quant_fwd_inp", "fp4_quant_fwd_weight", "fp4_quant_bwd_grad"),
        ),
    ],
)
def test_builtin_recipe_quantizer_configs_are_exhaustive(
    recipe_type,
    derived_fields,
):
    """Built-in configs include every declared field and derived parameter bundle."""
    recipe = recipe_type()
    config = recipe.quantizer_config()
    config_labels = tuple(name for name, _ in config)
    expected_labels = {
        "recipe_type",
        *(field.name for field in fields(recipe_type)),
        *derived_fields,
    }

    assert len(config_labels) == len(set(config_labels))
    assert set(config_labels) == expected_labels

    config_by_name = dict(config)
    for name in derived_fields:
        params = getattr(recipe, name)
        expected = tuple(
            (field.name, getattr(params, field.name)) for field in fields(type(params))
        )
        assert config_by_name[name] == expected

    for name in ("fp8_gemm_fprop", "fp8_gemm_dgrad", "fp8_gemm_wgrad"):
        if hasattr(recipe, name):
            params = getattr(recipe, name)
            expected = tuple(
                (field.name, getattr(params, field.name)) for field in fields(type(params))
            )
            assert config_by_name[name] == expected


@pytest.mark.parametrize(
    "recipe_type",
    [
        DelayedScaling,
        Float8CurrentScaling,
        MXFP8BlockScaling,
        Float8BlockScaling,
        NVFP4BlockScaling,
    ],
)
def test_equal_builtin_recipes_have_equal_quantizer_configs(recipe_type):
    """Independent built-in recipes with the same ingredients are semantically equal."""
    first = recipe_type().quantizer_config()
    second = recipe_type().quantizer_config()
    assert first == second


@pytest.mark.parametrize(
    "algorithm_field",
    ["amax_compute_algo", "scaling_factor_compute_algo"],
)
def test_delayed_scaling_callable_config_uses_explicit_key(algorithm_field):
    """Delayed-scaling callable identity is replaced by an explicit semantic key."""
    calls = []

    def first_algorithm(*args):
        calls.append(("first", args))

    def second_algorithm(*args):
        calls.append(("second", args))

    key_field = f"{algorithm_field}_key"
    shared_key = ("test_algorithm", 1)
    first = DelayedScaling(**{algorithm_field: first_algorithm, key_field: shared_key})
    second = DelayedScaling(**{algorithm_field: second_algorithm, key_field: shared_key})

    assert first.quantizer_config() == second.quantizer_config()
    assert calls == []

    second = DelayedScaling(**{algorithm_field: second_algorithm, key_field: ("test_algorithm", 2)})
    assert first.quantizer_config() != second.quantizer_config()

    missing_key = DelayedScaling(**{algorithm_field: first_algorithm})
    with pytest.raises(
        ValueError,
        match=rf"{algorithm_field} is callable, so {key_field} must provide",
    ):
        missing_key.quantizer_config()


def test_quantizer_policy_contract():
    """The decorator validates and attaches metadata without wrapping or calling."""
    calls = []

    def factory(role):
        calls.append(role)

    decorated = quantizer_policy(key=("test_policy", 1))(factory)

    assert decorated is factory
    assert getattr(decorated, "policy_key") == ("test_policy", 1)
    assert calls == []
    assert te.quantizer_policy is quantizer_policy

    with pytest.raises(TypeError, match="quantizer_policy key must be hashable"):
        quantizer_policy(key=["mutable"])


def test_custom_recipe_policy_key_contract():
    """Custom recipes resolve semantic keys without calling their factories."""
    calls = []

    def first_factory(role):
        calls.append(("first", role))

    def second_factory(role):
        calls.append(("second", role))

    policy_key = ("hybrid_policy", 2, ("double_quantization", True))
    first = CustomRecipe(qfactory=first_factory, policy_key=policy_key)
    second = CustomRecipe(qfactory=second_factory, policy_key=policy_key)

    assert first.quantizer_config() == second.quantizer_config()
    assert set(dict(first.quantizer_config())) == {
        "recipe_type",
        "policy_key",
        "fp8_format",
        "fp8_dpa",
        "fp8_mha",
        "backward_override",
        "quantization_alignment",
    }
    assert calls == []

    @quantizer_policy(key=("attached_policy", 1))
    def factory(role):  # pylint: disable=unused-argument
        calls.append(("attached", role))

    attached = CustomRecipe(qfactory=factory)
    assert attached.policy_key == ("attached_policy", 1)
    assert dict(attached.quantizer_config())["policy_key"] == ("attached_policy", 1)

    explicit = CustomRecipe(qfactory=factory, policy_key=("explicit_policy", 2))
    assert explicit.policy_key == ("explicit_policy", 2)
    assert dict(explicit.quantizer_config())["policy_key"] == ("explicit_policy", 2)

    def unkeyed_factory(role):
        calls.append(("unkeyed", role))

    with pytest.raises(
        ValueError,
        match=r"Pass policy_key=.*@quantizer_policy",
    ):
        CustomRecipe(qfactory=unkeyed_factory).quantizer_config()
    assert calls == []


def test_global_state_caches_active_quantizer_config():
    """Recipe activation publishes the recipe and its semantic configuration together."""
    FP8GlobalStateManager.reset()
    active_recipe = Float8CurrentScaling()

    FP8GlobalStateManager.activate_recipe(active_recipe)
    original_config = active_recipe.quantizer_config()
    assert FP8GlobalStateManager.get_fp8_recipe() is active_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is original_config

    # Recipe mutation is requested state until the recipe is activated again.
    active_recipe.fp8_dpa = True
    assert FP8GlobalStateManager.get_quantizer_config() is original_config

    FP8GlobalStateManager.activate_recipe(active_recipe)
    assert FP8GlobalStateManager.get_quantizer_config() == active_recipe.quantizer_config()
    assert FP8GlobalStateManager.get_quantizer_config() != original_config
    FP8GlobalStateManager.reset()


def test_recipe_activation_does_not_call_qfactory_and_is_atomic_on_config_error():
    """Configuration is resolved without probing the factory and before changing active state."""
    FP8GlobalStateManager.reset()
    calls = []

    @quantizer_policy(key=("activation_test", 1))
    def keyed_factory(role):
        calls.append(role)

    keyed_recipe = CustomRecipe(qfactory=keyed_factory)
    FP8GlobalStateManager.activate_recipe(keyed_recipe)
    keyed_config = keyed_recipe.quantizer_config()
    assert FP8GlobalStateManager.get_fp8_recipe() is keyed_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is keyed_config
    assert calls == []

    def unkeyed_factory(role):
        calls.append(role)

    with pytest.raises(ValueError, match="requires a semantic policy key"):
        FP8GlobalStateManager.activate_recipe(CustomRecipe(qfactory=unkeyed_factory))
    assert FP8GlobalStateManager.get_fp8_recipe() is keyed_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is keyed_config
    assert calls == []
    FP8GlobalStateManager.reset()


def test_autocast_restores_recipe_and_quantizer_config_together():
    """Leaving autocast restores the previously active recipe/configuration pair."""
    FP8GlobalStateManager.reset()
    outer_recipe = Float8CurrentScaling()
    inner_recipe = MXFP8BlockScaling()
    FP8GlobalStateManager.activate_recipe(outer_recipe)
    outer_config = FP8GlobalStateManager.get_quantizer_config()

    with te.autocast(enabled=False, recipe=inner_recipe):
        assert FP8GlobalStateManager.get_fp8_recipe() is inner_recipe
        assert FP8GlobalStateManager.get_quantizer_config() == inner_recipe.quantizer_config()

    assert FP8GlobalStateManager.get_fp8_recipe() is outer_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is outer_config
    FP8GlobalStateManager.reset()


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
        with te.autocast(recipe=recipe):
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
        with te.autocast(enabled=True, recipe=recipe):
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
            with te.autocast(recipe=recipe):
                y = op(x)
            y.backward(dy)

            def check_metas(
                test_scale: float,
                test_amax_history: torch.Tensor,
                ref_amax_history_list: list[float],
                stage: str,
            ):
                """Check that meta tensors match expected values"""

                # Compute amax
                if len(ref_amax_history_list) > amax_history_len:
                    ref_amax_history_list = ref_amax_history_list[-(amax_history_len + 1) :]
                ref_amax_history = torch.tensor(
                    ref_amax_history_list,
                    dtype=torch.float32,
                    device=device,
                )
                if amax_compute_algo == "max":
                    ref_amax = max(ref_amax_history_list)
                elif amax_compute_algo == "most_recent":
                    ref_amax = ref_amax_history_list[-1]
                else:
                    raise RuntimeError(f"{amax_compute_algo=} is not supported")

                # Compare amax history
                tols = dict(rtol=0, atol=0)
                torch.testing.assert_close(
                    test_amax_history[-(step + 1) :],
                    ref_amax_history[: (step + 1)],
                    **tols,
                )

                # Compute scale
                max_val = {
                    "forward": 448.0,
                    "backward": 57344.0,
                }[stage]
                ref_scale = (max_val / ref_amax) / (2**margin)

                # Compare scale
                torch.testing.assert_close(
                    test_scale,
                    ref_scale,
                )

            # Get scaling factors
            x_test_scale = op.get_quantizer("forward", 0).scale.item()
            w_test_scale = op.get_quantizer("forward", 1).scale.item()
            dy_test_scale = op.get_quantizer("backward", 0).scale.item()

            # Get amax histories
            x_test_history = op._fp8_metas["forward"][forward_key].amax_history[:, 0]
            w_test_history = op._fp8_metas["forward"][forward_key].amax_history[:, 1]
            dy_test_history = op._fp8_metas["backward"][backward_key].amax_history[:, 0]

            # Check that results match expected values
            check_metas(x_test_scale, x_test_history, x_history, "forward")
            check_metas(w_test_scale, w_test_history, w_history, "forward")
            check_metas(dy_test_scale, dy_test_history, dy_history, "backward")

    @pytest.mark.parametrize("amax_case", ["zero", "tiny", "normal", "inf", "nan"])
    @pytest.mark.parametrize("fused_update", [True, False], ids=["fused", "non-fused"])
    @pytest.mark.parametrize(
        "fp8_dtype",
        [te.DType.kFloat8E4M3, te.DType.kFloat8E5M2],
        ids=["E4M3", "E5M2"],
    )
    def test_scale_update_numeric_scenarios(self, amax_case, fused_update, fp8_dtype):

        if fp8_dtype == te.DType.kFloat8E4M3:
            fp8_format = transformer_engine.common.recipe.Format.E4M3
            fp8_max = fp8_format.value.max_fwd
        elif fp8_dtype == te.DType.kFloat8E5M2:
            fp8_format = transformer_engine.common.recipe.Format.HYBRID
            fp8_max = fp8_format.value.max_bwd
        else:
            raise ValueError(f"{fp8_dtype=} is not supported")

        scaling_factor_compute_algo = None
        if fused_update:
            scaling_factor_compute_algo = (
                lambda amax, scale, fp8_max, recipe: te.quantization._default_sf_compute(
                    amax, scale, fp8_max, recipe.margin
                )
            )
        recipe = transformer_engine.common.recipe.DelayedScaling(
            fp8_format=fp8_format,
            scaling_factor_compute_algo=scaling_factor_compute_algo,
            scaling_factor_compute_algo_key=("test_fused_update", 1) if fused_update else None,
        )

        # Setup fp8_meta dictionary
        def setup_fp8_meta():
            with te.autocast(recipe=recipe):
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
        with quantized_model_init(enabled=True, recipe=model_init_recipe):
            linear = Linear(32, 32).cuda()

        x = torch.randn(32, 32, device="cuda")
        with te.autocast(enabled=True, recipe=DelayedScaling()):
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
            with te.autocast(enabled=True, recipe=initial_recipe):
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
                    with te.autocast(enabled=True, recipe=target_recipe):
                        y = linear(x)
                for quantizer in linear.quantizers["scaling_fwd"]:
                    assert isinstance(quantizer, expected_quantizer_type)
            else:
                # No warning expected on subsequent iterations
                with warnings.catch_warnings():
                    warnings.simplefilter("error")  # Raise error if unexpected warning occurs
                    with te.autocast(enabled=True, recipe=target_recipe):
                        y = linear(x)
            loss = y.mean()
            loss.backward()

        # Final check
        for quantizer in linear.quantizers["scaling_fwd"]:
            assert isinstance(quantizer, expected_quantizer_type)

    @pytest.mark.parametrize(
        "module_class",
        [
            Linear,
            LayerNormLinear,
            LayerNormMLP,
            GroupedLinear,
        ],
    )
    def test_quantizer_update(self, module_class):
        in_features = 32
        out_features = 32
        batch_size = 32

        recipe = DelayedScaling(amax_history_len=1024)
        with quantized_model_init(recipe=recipe):
            if module_class == GroupedLinear:
                module = module_class(1, in_features, out_features).cuda()
            else:
                module = module_class(in_features, out_features).cuda()

        x = torch.randn(batch_size, in_features, device="cuda")
        recipe = DelayedScaling(amax_history_len=1)
        with te.autocast(enabled=True, recipe=recipe):
            warn_msg = "Quantizer is being updated, this may affect model behavior"
            with pytest.warns(UserWarning, match=warn_msg):
                if module_class == GroupedLinear:
                    y = module(x, [batch_size])
                else:
                    y = module(x)


@pytest.mark.skipif(not fp4_available, reason=reason_for_no_fp4)
@pytest.mark.parametrize(
    "nvfp4_4over6",
    ["none", "weights", "activations", "all"],
    ids=["disabled", "weights", "activations", "all"],
)
@pytest.mark.parametrize(
    "nvfp4_4over6_e4m3_use_256",
    ["none", "weights", "activations", "all"],
    ids=["e4m3_448", "e4m3_256_weights", "e4m3_256_activations", "e4m3_256_all"],
)
@pytest.mark.parametrize("nvfp4_4over6_err_mode", ["MAE", "MSE"], ids=["mae_err", "mse_err"])
def test_nvfp4_row_scaled_quantizer_roles(
    nvfp4_4over6, nvfp4_4over6_e4m3_use_256, nvfp4_4over6_err_mode
):
    recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_2d_quantization=True,
        nvfp4_4over6=nvfp4_4over6,
        nvfp4_4over6_e4m3_use_256=nvfp4_4over6_e4m3_use_256,
        nvfp4_4over6_err_mode=nvfp4_4over6_err_mode,
        row_scaled_activation=True,
    )

    def expected_use_4over6(tensor_type):
        if tensor_type in ("grad_output", "grad_input"):
            return False
        if nvfp4_4over6 == "all":
            return True
        if nvfp4_4over6 == "weights":
            return tensor_type == "weight"
        if nvfp4_4over6 == "activations":
            return tensor_type != "weight"
        return False

    def expected_e4m3_max(tensor_type):
        if not expected_use_4over6(tensor_type):
            return 448
        if nvfp4_4over6_e4m3_use_256 == "all":
            return 256
        if nvfp4_4over6_e4m3_use_256 == "weights":
            if tensor_type == "weight":
                return 256
        if nvfp4_4over6_e4m3_use_256 == "activations":
            if tensor_type != "weight":
                return 256
        return 448

    forward_quantizers = NVFP4BlockScalingRecipeState(
        recipe,
        mode="forward",
        num_quantizers=3,
    ).make_quantizers()
    assert [q.row_scaled_nvfp4 for q in forward_quantizers] == [True, False, True]
    assert [q.stochastic_rounding for q in forward_quantizers] == [False, False, False]
    assert [q.with_rht for q in forward_quantizers] == [False, False, False]
    assert [q.nvfp4_use_4over6 for q in forward_quantizers] == [
        expected_use_4over6(tensor_type) for tensor_type in ("input", "weight", "output")
    ]
    assert [q.nvfp4_e4m3_max for q in forward_quantizers] == [
        expected_e4m3_max(tensor_type) for tensor_type in ("input", "weight", "output")
    ]
    assert [q.nvfp4_4over6_err_mode for q in forward_quantizers] == [nvfp4_4over6_err_mode] * 3
    assert not forward_quantizers[0].is_quantizable(torch.empty(16, 16))
    assert forward_quantizers[1].is_quantizable(torch.empty(16, 16))

    role_quantizers = NVFP4BlockScalingRecipeState(
        recipe,
        mode="forward",
        num_quantizers=4,
        roles=[
            QuantizerRole(module_type="linear", tensor_type="weight"),
            QuantizerRole(module_type="linear", tensor_type="input"),
            QuantizerRole(module_type="linear", tensor_type="output"),
            None,
        ],
    ).make_quantizers()
    assert [q.row_scaled_nvfp4 for q in role_quantizers] == [False, True, True, True]
    assert [q.nvfp4_use_4over6 for q in role_quantizers] == [
        expected_use_4over6(tensor_type) for tensor_type in ("weight", "input", "output", "input")
    ]
    assert [q.nvfp4_e4m3_max for q in role_quantizers] == [
        expected_e4m3_max(tensor_type) for tensor_type in ("weight", "input", "output", "input")
    ]
    assert [q.nvfp4_4over6_err_mode for q in role_quantizers] == [nvfp4_4over6_err_mode] * 4

    backward_quantizers = NVFP4BlockScalingRecipeState(
        recipe,
        mode="backward",
        num_quantizers=2,
        roles=[
            QuantizerRole(module_type="linear", tensor_type="grad_output"),
            QuantizerRole(module_type="linear", tensor_type="grad_input"),
        ],
    ).make_quantizers()
    assert [q.row_scaled_nvfp4 for q in backward_quantizers] == [False, False]
    assert [q.nvfp4_use_4over6 for q in backward_quantizers] == [False, False]
    assert [q.nvfp4_e4m3_max for q in backward_quantizers] == [448, 448]
    assert [q.nvfp4_4over6_err_mode for q in backward_quantizers] == [nvfp4_4over6_err_mode] * 2
    assert [q.stochastic_rounding for q in backward_quantizers] == [True, True]
    assert [q.with_rht for q in backward_quantizers] == [False, False]

    backward_operand_quantizers = NVFP4BlockScalingRecipeState(
        recipe,
        mode="backward",
        num_quantizers=4,
        roles=[
            QuantizerRole(module_type="linear", tensor_type="input"),
            QuantizerRole(module_type="linear", tensor_type="weight"),
            QuantizerRole(module_type="linear", tensor_type="grad_output"),
            QuantizerRole(module_type="linear", tensor_type="grad_input"),
        ],
    ).make_quantizers()
    assert [q.nvfp4_use_4over6 for q in backward_operand_quantizers] == [
        expected_use_4over6(tensor_type)
        for tensor_type in ("input", "weight", "grad_output", "grad_input")
    ]
    assert [q.nvfp4_e4m3_max for q in backward_operand_quantizers] == [
        expected_e4m3_max(tensor_type)
        for tensor_type in ("input", "weight", "grad_output", "grad_input")
    ]
    assert [q.stochastic_rounding for q in backward_operand_quantizers] == [
        False,
        False,
        True,
        True,
    ]


@pytest.mark.skipif(not fp4_available, reason=reason_for_no_fp4)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("row_scaled_nvfp4", [False, True], ids=["nvfp4", "nvfp4_row_scaled"])
@pytest.mark.parametrize("use_4over6", [False, True], ids=["default", "4over6"])
@pytest.mark.parametrize(
    "M, N",
    [
        # full tile cases
        (128, 128),
        (256, 1024),
        (1024, 256),
        # Padding required cases
        (256, 272),
        (304, 304),
        (320, 256),
        # # largest tile
        (8192, 8192),
    ],
)
def test_fp4_dequantize(dtype, row_scaled_nvfp4, use_4over6, M, N):
    q = NVFP4Quantizer(
        columnwise=not row_scaled_nvfp4,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=use_4over6,
    )
    a = torch.rand((M, N)).cuda().to(dtype=dtype)
    starting_tensor = q(a)
    assert starting_tensor._row_scaled_nvfp4 == row_scaled_nvfp4
    assert starting_tensor._nvfp4_use_4over6 == use_4over6
    assert starting_tensor._amax_rowwise.numel() == (M if row_scaled_nvfp4 else 1)
    dequantized_tensor = starting_tensor.dequantize()
    new_tensor = q(dequantized_tensor)
    assert new_tensor._row_scaled_nvfp4 == row_scaled_nvfp4
    assert new_tensor._nvfp4_use_4over6 == use_4over6
    assert new_tensor._amax_rowwise.numel() == (M if row_scaled_nvfp4 else 1)
    # 4over6 can re-encode a dequantized block with the alternate 4/6 scale
    # choice while preserving the dequantized values.
    if not use_4over6:
        torch.testing.assert_close(
            new_tensor._rowwise_data,
            starting_tensor._rowwise_data,
            rtol=0,
            atol=0,
        )
    new_dequantized_tensor = new_tensor.dequantize()
    torch.testing.assert_close(dequantized_tensor, new_dequantized_tensor)


def _custom_recipe_qfactory(_role):
    return None


def _recipe_subclasses(cls):
    for subcls in cls.__subclasses__():
        yield subcls
        yield from _recipe_subclasses(subcls)


def _pickled_extra_state_payload(recipe_obj, *, include_delayed_state=False):
    state = {"recipe": recipe_obj, "extra_fp8_variables": {}}
    if include_delayed_state:
        state.update(
            {
                "scale_fwd": torch.ones(1),
                "amax_history_fwd": torch.zeros(1, 1),
                "scale_bwd": torch.ones(1),
                "amax_history_bwd": torch.zeros(1, 1),
            }
        )
    return pickle.dumps(state)


def test_checkpoint_extra_state_policy_classifier_map_covers_all_recipes():
    for cls in _recipe_subclasses(Recipe):
        key = ("transformer_engine.common.recipe", cls.__name__)
        assert key in _RECIPE_POLICIES
        assert _RECIPE_POLICIES[key] in CheckpointExtraStatePolicy


@pytest.mark.parametrize(
    "recipe_obj",
    [
        Float8CurrentScaling(),
        MXFP8BlockScaling(),
        Float8BlockScaling(),
        NVFP4BlockScaling(),
    ],
)
def test_stateless_pickled_extra_state_is_ignored(recipe_obj):
    payload = _pickled_extra_state_payload(recipe_obj)
    assert not should_load_extra_state_pickle(payload, "test")


def test_stateless_custom_pickled_extra_state_is_ignored():
    payload = _pickled_extra_state_payload(CustomRecipe(qfactory=_custom_recipe_qfactory))
    assert not should_load_extra_state_pickle(payload, "test")


@pytest.mark.parametrize("payload", [pickle.dumps({}), pickle.dumps({"extra_fp8_variables": {}})])
def test_global_free_pickled_extra_state_is_ignored(payload):
    # Older stateless checkpoints serialized an empty dict. Such a payload
    # resolves no globals and cannot execute code, so it must load without the
    # unsafe opt-in.
    assert not should_load_extra_state_pickle(payload, "test")


@pytest.mark.parametrize(
    "payload",
    [
        _pickled_extra_state_payload(DelayedScaling(), include_delayed_state=True),
        _pickled_extra_state_payload(
            CustomRecipe(qfactory=_custom_recipe_qfactory), include_delayed_state=True
        ),
        pickle.dumps({"scale_inv_fwd": torch.ones(1), "extra_fp8_variables": {}}),
        pickle.dumps({"recipe": object(), "extra_fp8_variables": {}}),
        b"not a pickle",
    ],
)
def test_stateful_unknown_or_malformed_pickled_extra_state_requires_opt_in(payload, monkeypatch):
    with pytest.raises(RuntimeError, match=UNSAFE_PICKLE_EXTRA_STATE_ENV):
        should_load_extra_state_pickle(payload, "test")

    monkeypatch.setenv(UNSAFE_PICKLE_EXTRA_STATE_ENV, "1")
    assert should_load_extra_state_pickle(payload, "test")
