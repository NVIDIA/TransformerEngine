# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from dataclasses import fields, replace
from types import SimpleNamespace
from typing import Optional

import pickle

import pytest
import torch
import warnings

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch import (
    Float8BlockQuantizer,
    Float8CurrentScalingQuantizer,
    MXFP8Quantizer,
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
    Float8BlockScalingRecipeState,
    Float8CurrentScalingRecipeState,
    NVFP4BlockScalingRecipeState,
    QuantizerRole,
    RecipeState,
    _QuantizationRuntimeKey,
    _QuantizationRuntime,
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
    quantizer_factory,
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


def test_quantization_runtime_key_is_semantic_and_normalizes_slot_roles():
    """Runtime-key equality includes recipe config and each ordered role slot."""
    input_role = QuantizerRole(module_type="linear", tensor_type="input", name="qkv")
    weight_role = QuantizerRole(module_type="linear", tensor_type="weight", name="qkv")
    grad_role = QuantizerRole(module_type="linear", tensor_type="grad_output", name="qkv")

    key = _QuantizationRuntimeKey(
        recipe_config=("recipe", "fp8"),
        forward_roles=[input_role, None, weight_role],
        backward_roles=[grad_role],
    )
    same_request = _QuantizationRuntimeKey(
        recipe_config=("recipe", "fp8"),
        forward_roles=(input_role, None, weight_role),
        backward_roles=(grad_role,),
    )

    assert key == same_request
    assert hash(key) == hash(same_request)
    assert key.forward_roles == (input_role, None, weight_role)
    assert key != _QuantizationRuntimeKey(
        recipe_config=("recipe", "fp8"),
        forward_roles=(input_role, weight_role),
        backward_roles=(grad_role,),
    )
    assert key != _QuantizationRuntimeKey(
        recipe_config=("recipe", "new-fp8"),
        forward_roles=(input_role, None, weight_role),
        backward_roles=(grad_role,),
    )


def test_quantization_runtime_bundles_both_quantization_directions():
    """A runtime retains the complete active-or-candidate quantizer bundle."""
    runtime_key = _QuantizationRuntimeKey(
        recipe_config=("recipe", "fp8"),
        forward_roles=(),
        backward_roles=(),
    )
    forward_quantizers = []
    backward_quantizers = []
    runtime_recipe = Float8CurrentScaling()
    runtime = _QuantizationRuntime(
        key=runtime_key,
        recipe=runtime_recipe,
        num_gemms=1,
        recipe_config_revision=3,
        role_revision=2,
        forward_states=(),
        backward_states=(),
        forward_quantizers=forward_quantizers,
        backward_quantizers=backward_quantizers,
    )

    assert runtime.key is runtime_key
    assert runtime.recipe is runtime_recipe
    assert runtime.forward_quantizers is forward_quantizers
    assert runtime.backward_quantizers is backward_quantizers


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


def test_same_recipe_mutation_invalidates_config_for_direct_and_nested_parameters():
    """Built-in recipe config changes for same-object and nested-qparam updates."""
    recipe = Float8CurrentScaling()
    original_config = recipe.quantizer_config()

    recipe.fp8_dpa = True
    direct_mutation_config = recipe.quantizer_config()
    assert direct_mutation_config != original_config
    assert dict(direct_mutation_config)["fp8_dpa"] is True

    recipe.fp8_quant_fwd_inp = replace(recipe.fp8_quant_fwd_inp, amax_epsilon=0.25)
    nested_mutation_config = recipe.quantizer_config()
    assert nested_mutation_config != direct_mutation_config
    assert dict(nested_mutation_config)["fp8_quant_fwd_inp"] == (
        ("power_2_scale", recipe.fp8_quant_fwd_inp.power_2_scale),
        ("amax_epsilon", 0.25),
        ("random_hadamard_transform", recipe.fp8_quant_fwd_inp.random_hadamard_transform),
        ("stochastic_rounding", recipe.fp8_quant_fwd_inp.stochastic_rounding),
        ("fp4_2d_quantization", recipe.fp8_quant_fwd_inp.fp4_2d_quantization),
    )


@pytest.mark.parametrize(
    "make_recipe",
    (
        pytest.param(DelayedScaling, id="delayed-scaling"),
        pytest.param(Float8CurrentScaling, id="current-scaling"),
        pytest.param(MXFP8BlockScaling, id="mxfp8"),
        pytest.param(NVFP4BlockScaling, id="nvfp4"),
        pytest.param(
            lambda **kwargs: CustomRecipe(
                qfactory=lambda _role: None,
                qfactory_key=("canonical-attention-flags", 1),
                **kwargs,
            ),
            id="custom",
        ),
    ),
)
def test_fp8_mha_canonicalizes_fp8_dpa_during_construction_and_mutation(make_recipe):
    """FP8 MHA must never produce a semantic configuration with DPA disabled."""
    constructed = make_recipe(fp8_mha=True)
    assert constructed.fp8_mha is True
    assert constructed.fp8_dpa is True
    assert dict(constructed.quantizer_config())["fp8_dpa"] is True

    mutated = make_recipe()
    inactive_config = mutated.quantizer_config()
    mutated.fp8_mha = True
    canonical_config = mutated.quantizer_config()
    assert mutated.fp8_dpa is True
    assert canonical_config != inactive_config
    assert dict(canonical_config)["fp8_dpa"] is True
    assert dict(canonical_config)["fp8_mha"] is True

    # DPA cannot be disabled while MHA still depends on it.
    mutated.fp8_dpa = False
    assert mutated.fp8_dpa is True
    assert mutated.quantizer_config() == canonical_config

    # Disable MHA first when disabling both attention modes.
    mutated.fp8_mha = False
    mutated.fp8_dpa = False
    assert mutated.quantizer_config() == inactive_config


def test_high_level_recipe_flags_configure_concrete_quantizers_at_construction():
    """Constructor flags must reach derived QParams and concrete quantizers."""
    current = Float8CurrentScaling(use_power_2_scales=True)
    current_quantizers = Float8CurrentScalingRecipeState(
        current,
        mode="forward",
        num_quantizers=2,
        device=torch.device("cpu"),
        roles=[QuantizerRole(tensor_type="input"), QuantizerRole(tensor_type="weight")],
    ).make_quantizers()
    assert all(quantizer.force_pow_2_scales for quantizer in current_quantizers)

    block = Float8BlockScaling(use_f32_scales=True)
    block_quantizers = Float8BlockScalingRecipeState(
        block,
        mode="forward",
        num_quantizers=2,
        device=torch.device("cpu"),
        roles=[QuantizerRole(tensor_type="input"), QuantizerRole(tensor_type="weight")],
    ).make_quantizers()
    assert all(not quantizer.force_pow_2_scales for quantizer in block_quantizers)

    nvfp4 = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
    )
    nvfp4_forward = NVFP4BlockScalingRecipeState(
        nvfp4,
        mode="forward",
        num_quantizers=2,
        device=torch.device("cpu"),
        roles=[QuantizerRole(tensor_type="input"), QuantizerRole(tensor_type="weight")],
    ).make_quantizers()
    nvfp4_backward = NVFP4BlockScalingRecipeState(
        nvfp4,
        mode="backward",
        num_quantizers=1,
        device=torch.device("cpu"),
        roles=[QuantizerRole(tensor_type="grad_output")],
    ).make_quantizers()
    assert not nvfp4_forward[0].with_rht
    assert not nvfp4_forward[1].with_2d_quantization
    assert not nvfp4_backward[0].with_rht
    assert not nvfp4_backward[0].stochastic_rounding


def test_high_level_recipe_mutation_preserves_unowned_qparams_fields():
    """Convenience fields update their traits without discarding nested customization."""
    current = Float8CurrentScaling()
    current.fp8_quant_fwd_inp = replace(current.fp8_quant_fwd_inp, amax_epsilon=0.125)
    current.use_power_2_scales = True
    assert current.fp8_quant_fwd_inp.power_2_scale
    assert current.fp8_quant_fwd_inp.amax_epsilon == 0.125
    assert current.fp8_quant_fwd_weight.power_2_scale
    assert current.fp8_quant_bwd_grad.power_2_scale

    block = Float8BlockScaling()
    block.fp8_quant_fwd_weight = replace(block.fp8_quant_fwd_weight, amax_epsilon=0.25)
    block.use_f32_scales = True
    assert not block.fp8_quant_fwd_inp.power_2_scale
    assert not block.fp8_quant_fwd_weight.power_2_scale
    assert block.fp8_quant_fwd_weight.amax_epsilon == 0.25
    assert not block.fp8_quant_bwd_grad.power_2_scale
    block_quantizer = Float8BlockScalingRecipeState(
        block,
        mode="forward",
        num_quantizers=1,
        device=torch.device("cpu"),
        roles=[QuantizerRole(tensor_type="weight")],
    ).make_quantizers()[0]
    assert not block_quantizer.force_pow_2_scales

    nvfp4 = NVFP4BlockScaling()
    nvfp4.fp4_quant_fwd_inp = replace(nvfp4.fp4_quant_fwd_inp, amax_epsilon=0.375)
    nvfp4.fp4_quant_fwd_weight = replace(
        nvfp4.fp4_quant_fwd_weight,
        stochastic_rounding=True,
    )
    nvfp4.fp4_quant_bwd_grad = replace(nvfp4.fp4_quant_bwd_grad, amax_epsilon=0.5)
    nvfp4.disable_rht = True
    nvfp4.disable_stochastic_rounding = True
    nvfp4.disable_2d_quantization = True
    assert not nvfp4.fp4_quant_fwd_inp.random_hadamard_transform
    assert nvfp4.fp4_quant_fwd_inp.amax_epsilon == 0.375
    assert not nvfp4.fp4_quant_fwd_weight.fp4_2d_quantization
    assert nvfp4.fp4_quant_fwd_weight.stochastic_rounding
    assert not nvfp4.fp4_quant_bwd_grad.random_hadamard_transform
    assert not nvfp4.fp4_quant_bwd_grad.stochastic_rounding
    assert nvfp4.fp4_quant_bwd_grad.amax_epsilon == 0.5


def test_current_scaling_recipe_state_configures_roles_and_preserves_boundary_defaults():
    """Current-scaling construction follows role dispatch and legacy boundary defaults."""
    recipe = Float8CurrentScaling()
    recipe.fp8_quant_fwd_inp = replace(
        recipe.fp8_quant_fwd_inp, power_2_scale=True, amax_epsilon=0.1
    )
    recipe.fp8_quant_fwd_weight = replace(
        recipe.fp8_quant_fwd_weight, power_2_scale=False, amax_epsilon=0.2
    )
    recipe.fp8_quant_bwd_grad = replace(
        recipe.fp8_quant_bwd_grad, power_2_scale=True, amax_epsilon=0.3
    )

    forward_quantizers = Float8CurrentScalingRecipeState(
        recipe,
        mode="forward",
        num_quantizers=4,
        device=torch.device("cpu"),
        roles=[
            QuantizerRole(tensor_type="input"),
            QuantizerRole(tensor_type="weight"),
            None,
            QuantizerRole(tensor_type="unknown"),
        ],
    ).make_quantizers()
    backward_quantizers = Float8CurrentScalingRecipeState(
        recipe,
        mode="backward",
        num_quantizers=3,
        device=torch.device("cpu"),
        roles=[
            QuantizerRole(tensor_type="grad_output"),
            None,
            QuantizerRole(tensor_type="grad_input"),
        ],
    ).make_quantizers()

    assert [(q.force_pow_2_scales, q.amax_epsilon) for q in forward_quantizers] == [
        (True, 0.1),
        (False, 0.2),
        (recipe.use_power_2_scales, 0.0),
        (True, 0.1),
    ]
    assert [(q.force_pow_2_scales, q.amax_epsilon) for q in backward_quantizers] == [
        (True, 0.3),
        (recipe.use_power_2_scales, 0.0),
        (recipe.use_power_2_scales, 0.0),
    ]


@pytest.mark.parametrize(
    "recipe",
    [
        DelayedScaling(),
        Float8CurrentScaling(),
        MXFP8BlockScaling(),
        Float8BlockScaling(),
        NVFP4BlockScaling(),
        CustomRecipe(qfactory=lambda role: role, qfactory_key=("invalid-mode-test", 1)),
    ],
)
def test_recipe_state_rejects_invalid_mode(recipe):
    """Every RecipeState implementation uses the same mode validation."""
    with pytest.raises(ValueError, match=r"Unexpected recipe mode \(invalid\)"):
        RecipeState.create(recipe, mode="invalid", device=torch.device("cpu"))


def test_current_scaling_role_layouts_cover_module_and_basic_op_families():
    """Every Phase 2 owner supplies roles that select the expected per-slot qparams."""
    recipe = Float8CurrentScaling()
    recipe.fp8_quant_fwd_inp = replace(
        recipe.fp8_quant_fwd_inp, power_2_scale=True, amax_epsilon=0.1
    )
    recipe.fp8_quant_fwd_weight = replace(
        recipe.fp8_quant_fwd_weight, power_2_scale=False, amax_epsilon=0.2
    )
    recipe.fp8_quant_bwd_grad = replace(
        recipe.fp8_quant_bwd_grad, power_2_scale=True, amax_epsilon=0.3
    )

    boundary = (recipe.use_power_2_scales, 0.0)
    inp = (True, 0.1)
    weight = (False, 0.2)
    grad = (True, 0.3)

    def settings(mode, roles):
        quantizers = Float8CurrentScalingRecipeState(
            recipe,
            mode=mode,
            num_quantizers=len(roles),
            device=torch.device("cpu"),
            roles=roles,
        ).make_quantizers()
        return [(q.force_pow_2_scales, q.amax_epsilon) for q in quantizers]

    module_owner = SimpleNamespace(name="test")
    for module_type in (Linear, LayerNormLinear):
        forward_roles = module_type.get_quantizer_roles(
            module_owner,
            fwd=True,
            num_quantizers=3,
            boundary_role=None,
        )
        backward_roles = module_type.get_quantizer_roles(
            module_owner,
            fwd=False,
            num_quantizers=2,
            boundary_role=None,
        )
        assert settings("forward", forward_roles) == [inp, weight, boundary]
        assert settings("backward", backward_roles) == [grad, boundary]

    mlp_forward_roles = LayerNormMLP.get_quantizer_roles(
        module_owner,
        fwd=True,
        num_quantizers=6,
        boundary_role=None,
    )
    mlp_backward_roles = LayerNormMLP.get_quantizer_roles(
        module_owner,
        fwd=False,
        num_quantizers=4,
        boundary_role=None,
    )
    assert settings("forward", mlp_forward_roles) == [
        inp,
        weight,
        inp,
        inp,
        weight,
        boundary,
    ]
    assert settings("backward", mlp_backward_roles) == [grad, boundary, grad, grad]

    grouped_forward_roles = GroupedLinear.get_quantizer_roles(
        module_owner,
        fwd=True,
        num_quantizers=6,
        boundary_role=None,
    )
    grouped_backward_roles = GroupedLinear.get_quantizer_roles(
        module_owner,
        fwd=False,
        num_quantizers=4,
        boundary_role=None,
    )
    assert settings("forward", grouped_forward_roles) == [
        inp,
        weight,
        boundary,
        inp,
        weight,
        boundary,
    ]
    assert settings("backward", grouped_backward_roles) == [grad, boundary, grad, boundary]

    basic_owner = SimpleNamespace(name="test", num_groups=2)
    basic_forward_roles = te_ops.BasicLinear.get_quantizer_roles(basic_owner, "forward")
    basic_backward_roles = te_ops.BasicLinear.get_quantizer_roles(basic_owner, "backward")
    assert settings("forward", basic_forward_roles) == [inp, weight]
    assert settings("backward", basic_backward_roles) == [grad]

    basic_grouped_forward_roles = te_ops.GroupedLinear.get_quantizer_roles(basic_owner, "forward")
    basic_grouped_backward_roles = te_ops.GroupedLinear.get_quantizer_roles(basic_owner, "backward")
    assert settings("forward", basic_grouped_forward_roles) == [inp, weight, inp, weight]
    assert settings("backward", basic_grouped_backward_roles) == [grad, grad]


def test_current_scaling_owner_configuration_paths_preserve_numerics_and_traits():
    """Owner initialization paths publish identical qparams and retain non-numerical traits."""
    recipe = Float8CurrentScaling()
    recipe.fp8_quant_fwd_inp = replace(
        recipe.fp8_quant_fwd_inp, power_2_scale=True, amax_epsilon=0.1
    )
    recipe.fp8_quant_fwd_weight = replace(
        recipe.fp8_quant_fwd_weight, power_2_scale=False, amax_epsilon=0.2
    )
    recipe.fp8_quant_bwd_grad = replace(
        recipe.fp8_quant_bwd_grad, power_2_scale=True, amax_epsilon=0.3
    )

    boundary = (recipe.use_power_2_scales, 0.0)
    inp = (True, 0.1)
    weight = (False, 0.2)
    grad = (True, 0.3)

    def settings(quantizers):
        return [(q.force_pow_2_scales, q.amax_epsilon) for q in quantizers]

    def make_module_owner(module_type, num_gemms):
        owner = object.__new__(module_type)
        owner.__dict__.update(
            name="test",
            num_gemms=num_gemms,
            fp8_meta={"num_gemms": num_gemms},
            fp8_meta_tensors_initialized=False,
            quantizers={"scaling_fwd": [], "scaling_bwd": []},
            _output_quantizer_role=None,
            _grad_input_quantizer_role=None,
        )
        return owner

    FP8GlobalStateManager.reset()
    FP8GlobalStateManager.activate_recipe(recipe)
    try:
        for module_type in (Linear, LayerNormLinear):
            owner = make_module_owner(module_type, 1)
            module_type.set_meta_tensor(owner, True, recipe)
            module_type.set_meta_tensor(owner, False, recipe)
            assert settings(owner.quantizers["scaling_fwd"]) == [inp, weight, boundary]
            assert settings(owner.quantizers["scaling_bwd"]) == [grad, boundary]

        mlp = make_module_owner(LayerNormMLP, 2)
        LayerNormMLP.set_meta_tensor(mlp, True, recipe)
        LayerNormMLP.set_meta_tensor(mlp, False, recipe)
        assert settings(mlp.quantizers["scaling_fwd"]) == [
            inp,
            weight,
            inp,
            inp,
            weight,
            boundary,
        ]
        assert settings(mlp.quantizers["scaling_bwd"]) == [grad, boundary, grad, grad]

        grouped = make_module_owner(GroupedLinear, 2)
        grouped.__dict__.update(
            tp_size=1,
            _offsets={
                "input": 0,
                "weight": 1,
                "output": 2,
                "grad_output": 0,
                "grad_input": 1,
            },
            _num_fp8_tensors_per_gemm={"fwd": 3, "bwd": 2},
            _validated_quantizer_generations={},
            _delayed_scaling_input_quantizer=None,
            _unsafe_requantization_input_quantizer=None,
        )
        GroupedLinear.set_meta_tensor(grouped, True, recipe)
        GroupedLinear.set_meta_tensor(grouped, False, recipe)
        assert settings(grouped.quantizers["scaling_fwd"]) == [
            inp,
            weight,
            boundary,
            inp,
            weight,
            boundary,
        ]
        assert settings(grouped.quantizers["scaling_bwd"]) == [
            grad,
            boundary,
            grad,
            boundary,
        ]

        grouped_tp = make_module_owner(GroupedLinear, 2)
        grouped_tp.__dict__["tp_size"] = 2
        with pytest.raises(
            ValueError,
            match="GroupedLinear doesn't support TP > 1 with Float8 current scaling",
        ):
            GroupedLinear.set_meta_tensor(grouped_tp, True, recipe)

        basic_linear = te_ops.BasicLinear(4, 4, device="meta")
        basic_linear.reset_recipe_state(recipe=recipe)
        assert settings(basic_linear._quantizers["forward"]) == [inp, weight]
        assert settings(basic_linear._quantizers["backward"]) == [grad]
        assert basic_linear.get_quantizer("forward", 0).internal
        assert basic_linear.get_quantizer("forward", 0).optimize_for_gemm
        assert basic_linear.get_quantizer("forward", 1).internal
        assert basic_linear.get_quantizer("backward", 0).internal
        assert basic_linear.get_quantizer("backward", 0).optimize_for_gemm

        basic_grouped = te_ops.GroupedLinear(2, 4, 4, device="meta")
        basic_grouped.reset_recipe_state(recipe=recipe)
        assert settings(basic_grouped._quantizers["forward"]) == [inp, weight, inp, weight]
        assert settings(basic_grouped._quantizers["backward"]) == [grad, grad]
        for quantizer in basic_grouped._quantizers["forward"]:
            assert quantizer.internal
        for quantizer in basic_grouped._quantizers["backward"]:
            assert quantizer.internal
    finally:
        FP8GlobalStateManager.reset()


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


def test_quantizer_factory_contract():
    """The decorator validates and attaches metadata without wrapping or calling."""
    calls = []

    def factory(role):
        calls.append(role)

    decorated = quantizer_factory(key=("test_policy", 1))(factory)

    assert decorated is factory
    assert getattr(decorated, "qfactory_key") == ("test_policy", 1)
    assert calls == []
    assert te.quantizer_factory is quantizer_factory

    with pytest.raises(TypeError, match="quantizer_factory key must be hashable"):
        quantizer_factory(key=["mutable"])


def test_custom_recipe_qfactory_key_contract():
    """Custom recipes resolve semantic keys without calling their factories."""
    calls = []

    def first_factory(role):
        calls.append(("first", role))

    def second_factory(role):
        calls.append(("second", role))

    qfactory_key = ("hybrid_policy", 2, ("double_quantization", True))
    first = CustomRecipe(qfactory=first_factory, qfactory_key=qfactory_key)
    second = CustomRecipe(qfactory=second_factory, qfactory_key=qfactory_key)

    assert first.quantizer_config() == second.quantizer_config()
    assert set(dict(first.quantizer_config())) == {
        "recipe_type",
        "qfactory_key",
        "fp8_format",
        "fp8_dpa",
        "fp8_mha",
        "backward_override",
        "quantization_alignment",
    }
    assert calls == []

    @quantizer_factory(key=("attached_policy", 1))
    def factory(role):  # pylint: disable=unused-argument
        calls.append(("attached", role))

    attached = CustomRecipe(qfactory=factory)
    assert attached.qfactory_key == ("attached_policy", 1)
    assert dict(attached.quantizer_config())["qfactory_key"] == ("attached_policy", 1)

    explicit = CustomRecipe(qfactory=factory, qfactory_key=("explicit_policy", 2))
    assert explicit.qfactory_key == ("explicit_policy", 2)
    assert dict(explicit.quantizer_config())["qfactory_key"] == ("explicit_policy", 2)

    def unkeyed_factory(role):
        calls.append(("unkeyed", role))

    with pytest.raises(
        ValueError,
        match=r"Pass qfactory_key=.*@quantizer_factory",
    ):
        CustomRecipe(qfactory=unkeyed_factory).quantizer_config()
    assert calls == []


def test_custom_recipe_qfactory_key_mutation_changes_semantic_configuration():
    """Changing a CustomRecipe policy key invalidates its cached config without probing."""
    calls = []

    def factory(role):
        calls.append(role)

    recipe = CustomRecipe(qfactory=factory, qfactory_key=("test-policy", 1))
    original_config = recipe.quantizer_config()
    recipe.qfactory_key = ("test-policy", 2)

    assert recipe.quantizer_config() != original_config
    assert dict(recipe.quantizer_config())["qfactory_key"] == ("test-policy", 2)
    assert calls == []


def test_global_state_caches_active_quantizer_config_and_revision():
    """Recipe activation publishes semantic configuration with a manager-owned revision."""
    FP8GlobalStateManager.reset()
    active_recipe = Float8CurrentScaling()
    assert FP8GlobalStateManager.get_quantizer_config_revision() == 0

    FP8GlobalStateManager.activate_recipe(active_recipe)
    original_config = active_recipe.quantizer_config()
    assert FP8GlobalStateManager.get_fp8_recipe() is active_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is original_config
    assert FP8GlobalStateManager.get_quantizer_config_revision() == 1

    # Equivalent recipe objects update the requested recipe without advancing the revision.
    equivalent_recipe = Float8CurrentScaling()
    FP8GlobalStateManager.activate_recipe(equivalent_recipe)
    assert FP8GlobalStateManager.get_fp8_recipe() is equivalent_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is original_config
    assert FP8GlobalStateManager.get_quantizer_config_revision() == 1

    # Recipe mutation is requested state until the recipe is activated again.
    active_recipe.fp8_dpa = True
    assert FP8GlobalStateManager.get_quantizer_config() is original_config
    assert FP8GlobalStateManager.get_quantizer_config_revision() == 1

    FP8GlobalStateManager.activate_recipe(active_recipe)
    assert FP8GlobalStateManager.get_quantizer_config() == active_recipe.quantizer_config()
    assert FP8GlobalStateManager.get_quantizer_config() != original_config
    assert FP8GlobalStateManager.get_quantizer_config_revision() == 2
    FP8GlobalStateManager.reset()


def test_recipe_activation_does_not_call_qfactory_and_is_atomic_on_config_error():
    """Configuration is resolved without probing the factory and before changing active state."""
    FP8GlobalStateManager.reset()
    calls = []

    @quantizer_factory(key=("activation_test", 1))
    def keyed_factory(role):
        calls.append(role)

    keyed_recipe = CustomRecipe(qfactory=keyed_factory)
    FP8GlobalStateManager.activate_recipe(keyed_recipe)
    keyed_config = keyed_recipe.quantizer_config()
    keyed_revision = FP8GlobalStateManager.get_quantizer_config_revision()
    assert FP8GlobalStateManager.get_fp8_recipe() is keyed_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is keyed_config
    assert calls == []

    def unkeyed_factory(role):
        calls.append(role)

    with pytest.raises(ValueError, match="requires a semantic qfactory key"):
        FP8GlobalStateManager.activate_recipe(CustomRecipe(qfactory=unkeyed_factory))
    assert FP8GlobalStateManager.get_fp8_recipe() is keyed_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is keyed_config
    assert FP8GlobalStateManager.get_quantizer_config_revision() == keyed_revision
    assert calls == []
    FP8GlobalStateManager.reset()


def _autocast_activation_state():
    """Capture identity-bearing state that autocast entry may publish."""
    qstate = FP8GlobalStateManager.quantization_state
    return (
        qstate.fp8_enabled,
        qstate.fp8_calibration,
        id(qstate.fp8_recipe),
        id(qstate.quantizer_config),
        qstate.quantizer_config_revision,
        id(qstate.fp8_distributed_group),
        qstate.is_first_fp8_module,
        qstate.fp8_graph_capturing,
        qstate.autocast_depth,
        qstate.abort_amax_reduction,
        tuple(
            (key, id(recipe), id(group))
            for key, (recipe, group) in qstate.autocast_arguments.items()
        ),
    )


def test_failed_autocast_recipe_activation_is_atomic_and_context_is_reusable():
    """A semantic-config error must not publish state or poison the context instance."""
    FP8GlobalStateManager.reset()
    active_recipe = Float8CurrentScaling()
    FP8GlobalStateManager.activate_recipe(active_recipe)
    original_state = _autocast_activation_state()

    def unkeyed_factory(role):
        del role

    context = te.autocast(
        enabled=True,
        calibrating=True,
        recipe=CustomRecipe(qfactory=unkeyed_factory),
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="requires a semantic qfactory key"):
            context.__enter__()
        assert context._fp8_state is None  # pylint: disable=protected-access
        assert _autocast_activation_state() == original_state

    FP8GlobalStateManager.reset()


def test_failed_autocast_support_check_is_atomic(monkeypatch):
    """Platform validation must complete before autocast state is published."""
    FP8GlobalStateManager.reset()
    active_recipe = Float8CurrentScaling()
    FP8GlobalStateManager.activate_recipe(active_recipe)
    original_state = _autocast_activation_state()

    @quantizer_factory(key=("autocast_support_failure", 1))
    def keyed_factory(role):
        del role

    monkeypatch.setattr(
        FP8GlobalStateManager,
        "is_fp8_available",
        classmethod(lambda _cls: (False, "injected FP8 support failure")),
    )
    context = te.autocast(enabled=True, recipe=CustomRecipe(qfactory=keyed_factory))
    with pytest.raises(AssertionError, match="injected FP8 support failure"):
        context.__enter__()

    assert context._fp8_state is None  # pylint: disable=protected-access
    assert _autocast_activation_state() == original_state
    FP8GlobalStateManager.reset()


def test_failed_nested_autocast_activation_preserves_outer_state():
    """A rejected inner recipe must leave its active outer autocast unchanged."""
    FP8GlobalStateManager.reset()

    def unkeyed_factory(role):
        del role

    invalid_context = te.autocast(
        enabled=True,
        calibrating=True,
        recipe=CustomRecipe(qfactory=unkeyed_factory),
    )
    outer_recipe = Float8CurrentScaling()
    with te.autocast(enabled=False, calibrating=True, recipe=outer_recipe):
        outer_state = _autocast_activation_state()
        with pytest.raises(ValueError, match="requires a semantic qfactory key"):
            invalid_context.__enter__()
        assert invalid_context._fp8_state is None  # pylint: disable=protected-access
        assert _autocast_activation_state() == outer_state

    assert FP8GlobalStateManager.quantization_state.autocast_depth == 0
    FP8GlobalStateManager.reset()


def test_autocast_restores_recipe_and_quantizer_config_together():
    """Leaving autocast restores the previously active recipe/configuration pair."""
    FP8GlobalStateManager.reset()
    outer_recipe = Float8CurrentScaling()
    inner_recipe = MXFP8BlockScaling()
    FP8GlobalStateManager.activate_recipe(outer_recipe)
    outer_config = FP8GlobalStateManager.get_quantizer_config()
    outer_revision = FP8GlobalStateManager.get_quantizer_config_revision()

    # Equal independent configurations do not advance the revision on entry or restoration.
    with te.autocast(enabled=False, recipe=Float8CurrentScaling()):
        assert FP8GlobalStateManager.get_quantizer_config_revision() == outer_revision
    assert FP8GlobalStateManager.get_quantizer_config_revision() == outer_revision

    with te.autocast(enabled=False, recipe=inner_recipe):
        assert FP8GlobalStateManager.get_fp8_recipe() is inner_recipe
        assert FP8GlobalStateManager.get_quantizer_config() == inner_recipe.quantizer_config()
        assert FP8GlobalStateManager.get_quantizer_config_revision() == outer_revision + 1

    assert FP8GlobalStateManager.get_fp8_recipe() is outer_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is outer_config
    assert FP8GlobalStateManager.get_quantizer_config_revision() == outer_revision + 2
    FP8GlobalStateManager.reset()


def test_quantized_model_init_restores_recipe_config_with_monotonic_revision():
    """Model-init recipe restoration republishes config rather than reusing an old revision."""
    FP8GlobalStateManager.reset()
    outer_recipe = Float8CurrentScaling()
    inner_recipe = MXFP8BlockScaling()
    FP8GlobalStateManager.activate_recipe(outer_recipe)
    outer_config = FP8GlobalStateManager.get_quantizer_config()
    outer_revision = FP8GlobalStateManager.get_quantizer_config_revision()

    with te.quantized_model_init(enabled=False, recipe=inner_recipe):
        assert FP8GlobalStateManager.get_fp8_recipe() is inner_recipe
        assert FP8GlobalStateManager.get_quantizer_config_revision() == outer_revision + 1

    assert FP8GlobalStateManager.get_fp8_recipe() is outer_recipe
    assert FP8GlobalStateManager.get_quantizer_config() is outer_config
    assert FP8GlobalStateManager.get_quantizer_config_revision() == outer_revision + 2
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
                id="Float8CurrentScaling->MXFP8BlockScaling",
            ),
            pytest.param(
                Float8BlockScaling,
                Float8BlockQuantizer,
                fp8_block_scaling_available,
                reason_for_no_fp8_block_scaling,
                id="Float8CurrentScaling->Float8BlockScaling",
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
        initial_recipe = Float8CurrentScaling()

        # Run initial iterations with a stateless recipe. Delayed-scaling
        # enter/leave transitions are intentionally rejected by the lazy
        # module-local update path and covered separately.
        for _ in range(3):
            x = torch.randn(batch_size, in_features, device="cuda")
            with te.autocast(enabled=True, recipe=initial_recipe):
                y = linear(x)
            loss = y.mean()
            loss.backward()

        for quantizer in linear.quantizers["scaling_fwd"]:
            assert isinstance(quantizer, Float8CurrentScalingQuantizer)

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
    def test_quantized_primary_recipe_update_is_rejected(self, module_class):
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
            with pytest.raises(RuntimeError, match="Recipe mismatch for quantized primary weights"):
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
