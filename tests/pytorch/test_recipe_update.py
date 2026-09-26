# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Module-local runtime lifecycle tests for mid-training recipe updates."""

import pytest
import torch
import transformer_engine.pytorch.ops as te_ops

from transformer_engine.common.recipe import (
    CustomRecipe,
    DelayedScaling,
    Float8BlockScaling,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    QParams,
)
from transformer_engine.pytorch import (
    DotProductAttention,
    GroupedLinear,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    MultiheadAttention,
    TransformerLayer,
    apply_recipe,
    autocast,
    is_fp8_available,
    is_fp8_block_scaling_available,
    is_mxfp8_available,
    is_nvfp4_available,
    quantized_model_init,
)
from transformer_engine.pytorch._extra_state import UNSAFE_PICKLE_EXTRA_STATE_ENV
from transformer_engine.pytorch.custom_recipes.quantizer_factories import (
    current_scaling_factory,
    delayed_scaling_factory,
)
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.quantization import (
    DelayedScalingRequest,
    FP8GlobalStateManager,
    QuantizerRole,
)
from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
requires_fp8 = pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)


def _make_counting_recipe(key, calls, *, fail_on_grad_output=False):
    """Build an identity recipe whose factory calls are externally observable."""

    def qfactory(role):
        calls.append(role)
        if fail_on_grad_output and role is not None and role.tensor_type == "grad_output":
            raise RuntimeError("backward factory failure")
        return IdentityQuantizer()

    return CustomRecipe(qfactory=qfactory, qfactory_key=key)


def _ensure_runtime(module, recipe, *, num_gemms=1):
    return module._ensure_quantization_runtime(  # pylint: disable=protected-access
        recipe=recipe,
        num_gemms=num_gemms,
    )


def _prepare_runtime_update(module, recipe, *, num_gemms=1):
    return module._plan_quantization_update(  # pylint: disable=protected-access
        recipe=recipe,
        num_gemms=num_gemms,
    )


def _mixed_delayed_factory(role):
    """Mix delayed/plain slots while keeping delayed state in both directions."""
    if role is not None and role.tensor_type in ("input", "weight", "grad_output"):
        return DelayedScalingRequest(amax_history_len=4)
    return IdentityQuantizer()


def _forward_only_delayed_factory(role):
    """Request delayed state only in the forward direction."""
    if role is not None and role.tensor_type in ("input", "weight"):
        return DelayedScalingRequest(amax_history_len=4)
    return IdentityQuantizer()


def _runtime_views(module):
    """Return identity-bearing views that a rejected update must preserve."""
    runtime = module._quantization_runtime  # pylint: disable=protected-access
    return (
        runtime,
        module.fp8_meta["recipe"],
        module.fp8_meta["scaling_fwd"],
        module.fp8_meta["scaling_bwd"],
        module.quantizers["scaling_fwd"],
        module.quantizers["scaling_bwd"],
    )


def _assert_runtime_recipes_match_keys(module):
    """Check every committed recipe snapshot in a module hierarchy."""
    for module_name, owner in module.named_modules():
        if not isinstance(owner, TransformerEngineBaseModule):
            continue
        runtime = owner._quantization_runtime  # pylint: disable=protected-access
        if runtime is None:
            continue
        active_recipe = owner.fp8_meta.get("recipe")
        assert (
            active_recipe is runtime.recipe
        ), f"Runtime recipe view changed after commit for {module_name or '<root>'}"
        recipe_config = active_recipe.quantizer_config()
        assert recipe_config == runtime.key.recipe_config, (
            f"Runtime recipe mutated after commit for {module_name or '<root>'}: "
            f"{recipe_config} != {runtime.key.recipe_config}"
        )


def _run_update_step(module, recipe, inp, *args, **kwargs):
    """Run a real forward/backward step and return the output."""
    inputs = inp if isinstance(inp, tuple) else (inp,)
    _assert_runtime_recipes_match_keys(module)
    module.zero_grad(set_to_none=True)
    with autocast(enabled=True, recipe=recipe):
        output = module(*inputs, *args, **kwargs)
    _assert_runtime_recipes_match_keys(module)
    if isinstance(output, tuple):
        output = output[0]
    output.float().sum().backward()
    _assert_runtime_recipes_match_keys(module)
    assert all(input_tensor.grad is not None for input_tensor in inputs)
    assert all(param.grad is not None for param in module.parameters() if param.requires_grad)
    return output


def _clone_inputs(inp):
    """Clone one tensor or a tuple of tensors while preserving grad requirements."""
    inputs = inp if isinstance(inp, tuple) else (inp,)
    clones = tuple(
        input_tensor.detach().clone().requires_grad_(input_tensor.requires_grad)
        for input_tensor in inputs
    )
    return clones if isinstance(inp, tuple) else clones[0]


def _run_numerical_step(module, recipe, inp, *args, output_grad=None, **kwargs):
    """Run forward/backward and capture values used by exact numerical oracles."""
    inputs = inp if isinstance(inp, tuple) else (inp,)
    _assert_runtime_recipes_match_keys(module)
    module.zero_grad(set_to_none=True)
    with autocast(enabled=True, recipe=recipe):
        output = module(*inputs, *args, **kwargs)
    _assert_runtime_recipes_match_keys(module)
    if isinstance(output, tuple):
        output = output[0]
    if output_grad is None:
        output_grad = torch.linspace(
            -1,
            1,
            output.numel(),
            device=output.device,
            dtype=output.dtype,
        ).reshape(output.shape)
    output.backward(output_grad)
    _assert_runtime_recipes_match_keys(module)
    return (
        (
            output.detach().clone(),
            tuple(input_tensor.grad.detach().clone() for input_tensor in inputs),
            {
                name: param.grad.detach().clone()
                for name, param in module.named_parameters()
                if param.requires_grad
            },
        ),
        output_grad,
    )


def _assert_numerical_traces_match(actual, expected):
    """Require bitwise-identical output, input gradients, and parameter gradients."""
    actual_output, actual_input_grads, actual_param_grads = actual
    expected_output, expected_input_grads, expected_param_grads = expected
    torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
    assert len(actual_input_grads) == len(expected_input_grads)
    for actual_grad, expected_grad in zip(actual_input_grads, expected_input_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert actual_param_grads.keys() == expected_param_grads.keys()
    for name, actual_grad in actual_param_grads.items():
        torch.testing.assert_close(actual_grad, expected_param_grads[name], rtol=0, atol=0)


def _make_identical_linears():
    """Construct identical BF16 Linear modules for numerical comparisons."""
    torch.manual_seed(2026)
    first = Linear(
        128,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        name="linear",
    )
    second = Linear(
        128,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        name="linear",
    )
    second.load_state_dict(first.state_dict())
    return first, second


def _check_same_object_mutation_against_fresh_recipe(
    active_recipe,
    fresh_recipe,
    mutations,
    *,
    seed,
    prepare_models=None,
):
    """Compare a warmed, mutated runtime with a fresh target runtime."""
    switching, oracle = _make_identical_linears()
    if prepare_models is not None:
        prepare_models(switching, oracle)
    torch.manual_seed(seed)
    warmup_inp = torch.randn(
        32,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    _run_numerical_step(switching, active_recipe, warmup_inp)
    old_runtime = switching._quantization_runtime  # pylint: disable=protected-access

    for name, value in mutations.items():
        setattr(active_recipe, name, value)
    apply_recipe(switching, active_recipe)
    runtime = switching._quantization_runtime  # pylint: disable=protected-access
    apply_recipe(oracle, fresh_recipe)

    base_inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
    switching_trace, output_grad = _run_numerical_step(
        switching,
        active_recipe,
        base_inp.detach().clone().requires_grad_(True),
    )
    oracle_trace, _ = _run_numerical_step(
        oracle,
        fresh_recipe,
        base_inp.detach().clone().requires_grad_(True),
        output_grad=output_grad,
    )
    _assert_numerical_traces_match(switching_trace, oracle_trace)
    assert runtime is not old_runtime


def _check_recipe_transition_sequence(recipes, *, seed):
    """Compare each runtime in a warmed recipe sequence with a fresh target."""
    switching, _ = _make_identical_linears()
    previous_runtime = None
    previous_config = None
    for step, recipe in enumerate(recipes):
        _, oracle = _make_identical_linears()
        if previous_runtime is not None:
            switching._fp8_workspaces["sentinel"] = object()  # pylint: disable=protected-access
        apply_recipe(switching, recipe)
        runtime = switching._quantization_runtime  # pylint: disable=protected-access
        recipe_config = recipe.quantizer_config()
        if previous_config is not None and recipe_config != previous_config:
            assert runtime is not previous_runtime
            assert not switching._fp8_workspaces  # pylint: disable=protected-access

        apply_recipe(oracle, recipe)
        oracle_runtime = oracle._quantization_runtime  # pylint: disable=protected-access
        assert tuple(type(quantizer) for quantizer in runtime.forward_quantizers) == tuple(
            type(quantizer) for quantizer in oracle_runtime.forward_quantizers
        )
        assert tuple(type(quantizer) for quantizer in runtime.backward_quantizers) == tuple(
            type(quantizer) for quantizer in oracle_runtime.backward_quantizers
        )

        torch.manual_seed(seed + step)
        base_inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
        switching_trace, output_grad = _run_numerical_step(
            switching,
            recipe,
            base_inp.detach().clone().requires_grad_(True),
        )
        oracle_trace, _ = _run_numerical_step(
            oracle,
            recipe,
            base_inp.detach().clone().requires_grad_(True),
            output_grad=output_grad,
        )
        _assert_numerical_traces_match(switching_trace, oracle_trace)
        previous_runtime = runtime
        previous_config = recipe_config


def _assert_modules_and_sgd_state_match(first, second, first_optimizer, second_optimizer):
    """Require exact parameters and per-parameter SGD state."""
    first_params = dict(first.named_parameters())
    second_params = dict(second.named_parameters())
    assert first_params.keys() == second_params.keys()
    for name, first_param in first_params.items():
        second_param = second_params[name]
        torch.testing.assert_close(first_param, second_param, rtol=0, atol=0)
        first_state = first_optimizer.state[first_param]
        second_state = second_optimizer.state[second_param]
        assert first_state.keys() == second_state.keys()
        for state_name, first_value in first_state.items():
            second_value = second_state[state_name]
            if isinstance(first_value, torch.Tensor):
                torch.testing.assert_close(first_value, second_value, rtol=0, atol=0)
            else:
                assert first_value == second_value


def _active_runtime_owners(module):
    """Return executed TE runtime owners in module traversal order."""
    return [
        submodule
        for submodule in module.modules()
        if isinstance(submodule, TransformerEngineBaseModule)
        and submodule._quantization_runtime is not None  # pylint: disable=protected-access
    ]


def _global_recipe_state():
    """Return the identity-bearing global recipe state."""
    state = FP8GlobalStateManager.quantization_state
    return (state.fp8_recipe,)


def _make_compatible_fp8_mha_recipe(key):
    """Build an executable FP8-MHA recipe with compatible boundary formats."""
    from transformer_engine.pytorch.custom_recipes.quantizer_factories import (
        delayed_scaling_factory,
    )
    from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
        nvfp4_linear_fp8_dpa_factory,
    )
    from transformer_engine.pytorch.utils import get_device_compute_capability

    compute_capability = get_device_compute_capability()

    def qfactory(role):
        # Hopper supports the coherent all-delayed configuration. On Blackwell,
        # use the shipped native FP8-attention mix for DPA-owned slots and
        # current scaling for linears and the two DPA/linear boundaries.
        if compute_capability < (10, 0):
            return delayed_scaling_factory(role)
        if role is not None and role.module_type == "dpa":
            return nvfp4_linear_fp8_dpa_factory(role)
        return current_scaling_factory(role)

    recipe = CustomRecipe(
        qfactory=qfactory,
        qfactory_key=key,
        fp8_mha=True,
    )
    assert recipe.fp8_dpa
    return recipe


@pytest.mark.parametrize(
    "replacement",
    (
        pytest.param(DelayedScaling(margin=1), id="margin"),
        pytest.param(DelayedScaling(amax_history_len=8), id="history-length"),
        pytest.param(Float8CurrentScaling(), id="leave-delayed-scaling"),
    ),
)
def test_delayed_runtime_rejects_effective_recipe_updates_atomically(replacement):
    """Delayed state is frozen once a module runtime has been initialized."""
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    recipe = DelayedScaling(amax_history_len=4)
    assert _ensure_runtime(module, recipe)
    old_views = _runtime_views(module)

    with pytest.raises(
        RuntimeError,
        match="Mid-training recipe updates do not support delayed scaling",
    ):
        _ensure_runtime(module, replacement)

    assert all(current is old for current, old in zip(_runtime_views(module), old_views))


def test_delayed_runtime_rejects_role_and_slot_layout_updates():
    """Role and GEMM-layout changes cannot rebuild a delayed runtime."""
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    recipe = DelayedScaling(amax_history_len=4)
    assert _ensure_runtime(module, recipe)
    old_views = _runtime_views(module)

    module.output_quantizer_role = QuantizerRole(
        module_type="linear",
        tensor_type="input",
        name="consumer",
    )
    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, recipe)
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))

    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, recipe, num_gemms=2)
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))


def test_stateless_runtime_cannot_acquire_delayed_state():
    """An active stateless runtime cannot acquire delayed state mid-training."""
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    assert _ensure_runtime(module, Float8CurrentScaling())
    old_views = _runtime_views(module)

    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, DelayedScaling(amax_history_len=4))
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))


def test_custom_recipe_cannot_introduce_delayed_state():
    """A delayed request discovered in a CustomRecipe candidate cannot commit."""
    calls = []
    active_recipe = _make_counting_recipe(("custom-enter-delayed", 1), calls)
    replacement = CustomRecipe(
        qfactory=_mixed_delayed_factory,
        qfactory_key=("custom-enter-delayed", 2),
    )
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    assert _ensure_runtime(module, active_recipe)
    old_views = _runtime_views(module)

    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, replacement)
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))


def test_asymmetric_custom_delayed_scaling_is_rejected_clearly():
    """This PR does not add a delayed topology unsupported on main."""
    recipe = CustomRecipe(
        qfactory=_forward_only_delayed_factory,
        qfactory_key=("forward-only-delayed", 1),
    )
    module = Linear(16, 16, bias=False, device="cuda", name="linear")

    with pytest.raises(
        RuntimeError,
        match="This hybrid quantization configuration with delayed scaling is not supported",
    ):
        _ensure_runtime(module, recipe)
    assert module._quantization_runtime is None  # pylint: disable=protected-access


def test_mixed_custom_recipe_is_frozen_when_it_contains_delayed_state():
    """Even a nominally non-delayed CustomRecipe edit is outside the contract."""
    calls = []

    def active_factory(role):
        calls.append(role)
        return _mixed_delayed_factory(role)

    active_recipe = CustomRecipe(qfactory=active_factory, qfactory_key=("mixed", 1))
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    assert _ensure_runtime(module, active_recipe)
    old_views = _runtime_views(module)
    active_call_count = len(calls)

    def replacement_factory(_role):
        raise AssertionError("frozen delayed runtime invoked the replacement factory")

    replacement = CustomRecipe(qfactory=replacement_factory, qfactory_key=("mixed", 2))
    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, replacement)

    assert len(calls) == active_call_count
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))


def test_same_delayed_recipe_object_mutation_keeps_committed_snapshot():
    """Rejecting caller mutation must not mutate the active reduction recipe."""
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    recipe = DelayedScaling(amax_history_len=4, margin=0)
    assert _ensure_runtime(module, recipe)
    committed_recipe = module.fp8_meta["recipe"]
    assert committed_recipe is not recipe

    recipe.margin = 1
    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, recipe)

    assert module.fp8_meta["recipe"] is committed_recipe
    assert committed_recipe.margin == 0


def test_region_failure_does_not_skip_autocast_reduction(monkeypatch):
    """All ranks retain the DelayedScaling collective schedule after an exception."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    try:
        reductions = []
        monkeypatch.setattr(
            FP8GlobalStateManager,
            "reduce_and_update_fp8_tensors",
            classmethod(lambda _cls, forward=True: reductions.append(forward)),
        )
        with pytest.raises(RuntimeError, match="deliberate region failure"):
            with autocast(enabled=True, recipe=DelayedScaling()):
                raise RuntimeError("deliberate region failure")
        assert reductions == [True]
    finally:
        FP8GlobalStateManager.reset()


def test_apply_recipe_preflight_rejects_unsupported_platform_before_commit(monkeypatch):
    """A recipe this platform cannot run is rejected before any module is planned."""
    monkeypatch.setattr(
        "transformer_engine.pytorch.quantization._NVFP4_SUPPORT",
        (False, "injected NVFP4 support failure"),
    )
    calls = []
    active_recipe = _make_counting_recipe(("apply-preflight", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    assert _ensure_runtime(module, active_recipe)
    active_runtime = module._quantization_runtime  # pylint: disable=protected-access
    active_recipe_pointer = FP8GlobalStateManager.quantization_state.fp8_recipe
    call_count = len(calls)

    try:
        with pytest.raises(RuntimeError, match="injected NVFP4 support failure"):
            apply_recipe(module, NVFP4BlockScaling())

        # pylint: disable-next=protected-access
        assert module._quantization_runtime is active_runtime
        assert FP8GlobalStateManager.quantization_state.fp8_recipe is active_recipe_pointer
        assert len(calls) == call_count
    finally:
        FP8GlobalStateManager.reset()


def test_equal_recipe_objects_and_missed_updates_reuse_the_runtime():
    """Equal independent recipes and missed A -> B -> A updates reuse the runtime."""
    calls = []
    first_recipe = _make_counting_recipe(("runtime-reuse", 1), calls)
    equal_recipe = _make_counting_recipe(("runtime-reuse", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")

    assert _ensure_runtime(module, first_recipe)
    active = module._quantization_runtime  # pylint: disable=protected-access
    assert active is not None
    assert len(calls) == 5

    # A module that missed an intervening region still recognizes an equal recipe.
    assert not _ensure_runtime(module, equal_recipe)
    assert module._quantization_runtime is active  # pylint: disable=protected-access
    assert len(calls) == 5


@pytest.mark.parametrize("reuse_recipe_object", (True, False), ids=("same-object", "equal-object"))
def test_unchanged_runtime_uses_the_steady_state_path(monkeypatch, reuse_recipe_object):
    """The steady-state check resolves no roles and invokes no factory.

    A reused recipe object hits on identity; an equivalent one built per region
    falls through to the configuration comparison and must agree.
    """
    calls = []
    recipe = _make_counting_recipe(("runtime-hot-path", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, recipe)
    active = module._quantization_runtime  # pylint: disable=protected-access
    factory_call_count = len(calls)

    def unexpected_role_resolution(**_kwargs):
        raise AssertionError("unchanged runtime resolved quantizer roles")

    monkeypatch.setattr(module, "get_quantizer_roles", unexpected_role_resolution)

    for _ in range(3):
        requested = (
            recipe if reuse_recipe_object else _make_counting_recipe(("runtime-hot-path", 1), calls)
        )
        assert not _ensure_runtime(module, requested)
        assert module._quantization_runtime is active  # pylint: disable=protected-access
        assert len(calls) == factory_call_count


def test_equal_semantic_update_is_not_published_during_planning():
    """An equal semantic update commits nothing and calls no factory."""
    calls = []
    recipe = _make_counting_recipe(("equal-plan", 1), calls)
    equal_recipe = _make_counting_recipe(("equal-plan", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, recipe)
    active = module._quantization_runtime  # pylint: disable=protected-access
    factory_call_count = len(calls)

    update = _prepare_runtime_update(module, equal_recipe)

    assert update.candidate is None
    assert module._quantization_runtime is active  # pylint: disable=protected-access
    assert len(calls) == factory_call_count

    assert not module._apply_quantization_update(update)  # pylint: disable=protected-access
    assert module._quantization_runtime is active  # pylint: disable=protected-access
    assert len(calls) == factory_call_count


def test_candidate_update_is_not_published_during_planning():
    """Candidate state, views, and workspaces change only during commit."""
    calls = []
    active_recipe = _make_counting_recipe(("candidate-plan", 1), calls)
    replacement_recipe = _make_counting_recipe(("candidate-plan", 2), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, active_recipe)
    old_views = _runtime_views(module)
    workspace = object()
    module._fp8_workspaces["weight"] = workspace  # pylint: disable=protected-access

    update = _prepare_runtime_update(module, replacement_recipe)

    assert update.candidate is not None
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))
    assert module._fp8_workspaces["weight"] is workspace  # pylint: disable=protected-access

    assert module._apply_quantization_update(update)  # pylint: disable=protected-access
    assert module._quantization_runtime is update.candidate  # pylint: disable=protected-access
    assert not module._fp8_workspaces  # pylint: disable=protected-access


def test_runtime_snapshot_keeps_caches_but_not_in_checkpoint_bytes():
    """A snapshot keeps the caller's caches; only its payload excludes them."""
    recipe = _make_counting_recipe(("runtime-repr-cache", 1), [])
    str(recipe)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")

    update = _prepare_runtime_update(module, recipe)
    snapshot = update.candidate.recipe

    assert recipe.__dict__["_cached_repr"] is not None
    assert snapshot is not recipe
    assert snapshot.quantizer_config() is recipe.quantizer_config()

    # The serialized payload is where display history would otherwise leak.
    payload = snapshot.__getstate__()
    assert "_cached_repr" not in payload
    assert "_cached_quantizer_config" not in payload


def test_uninitialized_runtime_can_be_prepared_without_publication():
    """Planning supports modules that have not yet executed a quantized forward."""
    calls = []
    recipe = _make_counting_recipe(("initial-plan", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")

    update = _prepare_runtime_update(module, recipe)

    assert update.candidate is not None
    assert module._quantization_runtime is None  # pylint: disable=protected-access
    assert "scaling_fwd" not in module.fp8_meta
    assert "scaling_bwd" not in module.fp8_meta
    assert module.quantizers == {"scaling_fwd": [], "scaling_bwd": []}

    assert module._apply_quantization_update(update)  # pylint: disable=protected-access
    assert module._quantization_runtime is update.candidate  # pylint: disable=protected-access


def test_later_planning_failure_leaves_earlier_module_unchanged(monkeypatch):
    """Prepared candidates can be discarded without changing any module."""
    calls = []
    active_recipe = _make_counting_recipe(("multi-plan", 1), calls)
    replacement_recipe = _make_counting_recipe(("multi-plan", 2), calls)
    first = Linear(16, 16, bias=False, device="cuda", name="first")
    second = Linear(16, 16, bias=False, device="cuda", name="second")
    _ensure_runtime(first, active_recipe)
    _ensure_runtime(second, active_recipe)
    old_first_views = _runtime_views(first)
    old_second_views = _runtime_views(second)

    first_update = _prepare_runtime_update(first, replacement_recipe)
    assert first_update.candidate is not None

    def reject_candidate(_candidate):
        raise RuntimeError("later module validation failure")

    monkeypatch.setattr(second, "_validate_quantization_runtime", reject_candidate)
    with pytest.raises(RuntimeError, match="later module validation failure"):
        _prepare_runtime_update(second, replacement_recipe)

    assert all(current is old for current, old in zip(_runtime_views(first), old_first_views))
    assert all(current is old for current, old in zip(_runtime_views(second), old_second_views))


@pytest.mark.parametrize(
    ("module_factory", "expected_num_gemms"),
    (
        pytest.param(
            lambda: Linear(16, 16, bias=False, device="cuda"),
            1,
            id="linear",
        ),
        pytest.param(
            lambda: LayerNormLinear(16, 16, bias=False, device="cuda"),
            1,
            id="layernorm-linear",
        ),
        pytest.param(
            lambda: LayerNormMLP(16, 32, bias=False, device="cuda"),
            2,
            id="layernorm-mlp",
        ),
        pytest.param(
            lambda: GroupedLinear(2, 16, 16, bias=False, device="cuda"),
            2,
            id="grouped-linear",
        ),
    ),
)
def test_module_owns_runtime_slot_layout(module_factory, expected_num_gemms):
    """Model-wide orchestration does not need a module-type dispatch table."""
    module = module_factory()
    assert (
        module._get_quantization_runtime_num_gemms()  # pylint: disable=protected-access
        == expected_num_gemms
    )


@requires_fp8
def test_unchanged_forward_uses_revision_hot_path(monkeypatch):
    """Repeated forwards do not enter any runtime-construction cold path."""
    calls = []
    recipe = _make_counting_recipe(("forward-hot-path", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    inp = torch.randn(8, 16, device="cuda")

    with autocast(enabled=True, recipe=recipe):
        module(inp)
        active = module._quantization_runtime  # pylint: disable=protected-access
        factory_call_count = len(calls)
        assert active is not None
        assert factory_call_count > 0

        def unexpected_cold_path(*_args, **_kwargs):
            raise AssertionError("unchanged forward entered the runtime cold path")

        monkeypatch.setattr(module, "get_quantizer_roles", unexpected_cold_path)
        monkeypatch.setattr(module, "_plan_quantization_update", unexpected_cold_path)
        monkeypatch.setattr(module, "_build_quantization_runtime", unexpected_cold_path)
        monkeypatch.setattr(module, "_validate_quantization_runtime", unexpected_cold_path)
        monkeypatch.setattr(module, "_activate_quantization_runtime", unexpected_cold_path)

        workspace_sentinel = object()
        module._fp8_workspaces["sentinel"] = workspace_sentinel  # pylint: disable=protected-access
        module(inp)
        module(inp)

        assert module._quantization_runtime is active  # pylint: disable=protected-access
        assert len(calls) == factory_call_count
        assert (
            module._fp8_workspaces["sentinel"] is workspace_sentinel
        )  # pylint: disable=protected-access


def test_same_recipe_object_semantic_mutation_rebuilds_runtime():
    """Runtime matching observes a new semantic config on the same recipe object."""
    calls = []
    recipe = _make_counting_recipe(("same-object-update", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, recipe)
    old_runtime = module._quantization_runtime  # pylint: disable=protected-access

    recipe.qfactory_key = ("same-object-update", 2)
    assert _ensure_runtime(module, recipe)
    assert module._quantization_runtime is not old_runtime  # pylint: disable=protected-access
    assert len(calls) == 10


def test_float8_block_same_object_layout_mutation_matches_fresh_recipe():
    """A warmed Float8BlockScaling layout update matches a fresh target."""
    available, reason = is_fp8_block_scaling_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    try:
        _check_same_object_mutation_against_fresh_recipe(
            Float8BlockScaling(x_block_scaling_dim=1, w_block_scaling_dim=2),
            Float8BlockScaling(x_block_scaling_dim=2, w_block_scaling_dim=1),
            {"x_block_scaling_dim": 2, "w_block_scaling_dim": 1},
            seed=6250,
        )
    finally:
        FP8GlobalStateManager.reset()


def test_apply_recipe_rejects_unsupported_block_scales_before_commit():
    """Blackwell scale restrictions are checked before an existing runtime is replaced."""
    available, reason = is_fp8_block_scaling_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("FP32 block scales are supported before Blackwell")

    FP8GlobalStateManager.reset()
    module = Linear(128, 128, bias=False, device="cuda", name="block")
    initial_recipe = Float8BlockScaling()
    try:
        apply_recipe(module, initial_recipe)
        old_runtime = module._quantization_runtime  # pylint: disable=protected-access
        old_global_state = _global_recipe_state()

        unsupported_recipe = Float8BlockScaling()
        unsupported_recipe.fp8_quant_fwd_inp = QParams(power_2_scale=False)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="requires power-of-two scales"):
                apply_recipe(module, unsupported_recipe)
            assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
            assert _global_recipe_state() == old_global_state
    finally:
        FP8GlobalStateManager.reset()


def test_mxfp8_same_object_2d_mutation_matches_fresh_recipe():
    """A warmed MXFP8 2D-weight update matches a fresh target."""
    available, reason = is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    def prepare_models(switching, oracle):
        torch.manual_seed(6261)
        weight = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        row_scales = torch.pow(
            2.0,
            torch.arange(128, device="cuda", dtype=torch.float32).remainder(16) - 8,
        ).to(torch.bfloat16)
        weight *= row_scales[:, None]
        with torch.no_grad():
            switching.weight.copy_(weight)
            oracle.weight.copy_(weight)

    FP8GlobalStateManager.reset()
    try:
        _check_same_object_mutation_against_fresh_recipe(
            MXFP8BlockScaling(enable_2d_quantization=False),
            MXFP8BlockScaling(enable_2d_quantization=True),
            {"enable_2d_quantization": True},
            seed=6260,
            prepare_models=prepare_models,
        )
    finally:
        FP8GlobalStateManager.reset()


def test_nvfp4_same_object_2d_mutation_matches_fresh_recipe():
    """A warmed deterministic NVFP4 weight-layout update matches a fresh target."""
    if not is_nvfp4_available():
        pytest.skip("NVFP4 is not available")

    FP8GlobalStateManager.reset()
    try:
        _check_same_object_mutation_against_fresh_recipe(
            NVFP4BlockScaling(
                disable_rht=True,
                disable_stochastic_rounding=True,
                disable_2d_quantization=False,
            ),
            NVFP4BlockScaling(
                disable_rht=True,
                disable_stochastic_rounding=True,
                disable_2d_quantization=True,
            ),
            {"disable_2d_quantization": True},
            seed=6270,
        )
    finally:
        FP8GlobalStateManager.reset()


def test_native_recipe_transition_matches_fresh_target():
    """A warmed stateless runtime matches a fresh target across A -> B -> A."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    available, reason = is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    try:
        _check_recipe_transition_sequence(
            (Float8CurrentScaling(), MXFP8BlockScaling(), Float8CurrentScaling()),
            seed=6300,
        )
    finally:
        FP8GlobalStateManager.reset()


def test_nvfp4_flag_mutations_rebuild_concrete_quantizers():
    """Every mutable NVFP4 convenience flag must change its concrete trait."""
    if not is_nvfp4_available():
        pytest.skip("NVFP4 is not available")

    FP8GlobalStateManager.reset()
    recipe = NVFP4BlockScaling(
        disable_rht=False,
        disable_stochastic_rounding=False,
        disable_2d_quantization=False,
    )
    module = Linear(
        128,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        name="nvfp4",
    )
    try:
        apply_recipe(module, recipe)
        runtime = module._quantization_runtime  # pylint: disable=protected-access
        assert runtime.forward_quantizers[0].with_rht
        assert runtime.forward_quantizers[1].with_2d_quantization
        assert runtime.backward_quantizers[0].with_rht
        assert runtime.backward_quantizers[0].stochastic_rounding

        recipe.disable_rht = True
        apply_recipe(module, recipe)
        rht_runtime = module._quantization_runtime  # pylint: disable=protected-access
        assert rht_runtime is not runtime
        assert not rht_runtime.forward_quantizers[0].with_rht
        assert not rht_runtime.backward_quantizers[0].with_rht

        recipe.disable_stochastic_rounding = True
        apply_recipe(module, recipe)
        rounding_runtime = module._quantization_runtime  # pylint: disable=protected-access
        assert rounding_runtime is not rht_runtime
        assert not rounding_runtime.backward_quantizers[0].stochastic_rounding

        recipe.disable_2d_quantization = True
        apply_recipe(module, recipe)
        block_runtime = module._quantization_runtime  # pylint: disable=protected-access
        assert block_runtime is not rounding_runtime
        assert not block_runtime.forward_quantizers[1].with_2d_quantization

        inp = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        _run_update_step(module, recipe, inp)
        assert module._quantization_runtime is block_runtime  # pylint: disable=protected-access
    finally:
        FP8GlobalStateManager.reset()


def test_role_revision_is_requested_until_atomic_runtime_commit():
    """Changing a boundary role does not mutate live state before the next update."""
    calls = []
    recipe = _make_counting_recipe(("role-update", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, recipe)

    old_runtime = module._quantization_runtime  # pylint: disable=protected-access
    old_forward_state = module.fp8_meta["scaling_fwd"]
    old_forward_quantizers = module.quantizers["scaling_fwd"]
    old_role_revision = module._role_revision  # pylint: disable=protected-access
    role = QuantizerRole(module_type="linear", tensor_type="input", name="consumer")

    module.output_quantizer_role = role
    assert module._role_revision == old_role_revision + 1  # pylint: disable=protected-access
    assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
    assert module.fp8_meta["scaling_fwd"] is old_forward_state
    assert module.quantizers["scaling_fwd"] is old_forward_quantizers

    assert _ensure_runtime(module, recipe)
    new_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert new_runtime is not old_runtime
    assert new_runtime.key.forward_roles[-1] == role
    assert module.fp8_meta["scaling_fwd"] is new_runtime.forward_state
    assert module.quantizers["scaling_fwd"] is new_runtime.forward_quantizers
    assert len(calls) == 10

    # Reassigning an equal immutable role is a hot-path no-op.
    module.output_quantizer_role = QuantizerRole(
        module_type="linear", tensor_type="input", name="consumer"
    )
    assert module._role_revision == old_role_revision + 1  # pylint: disable=protected-access
    assert not _ensure_runtime(module, recipe)
    assert len(calls) == 10


def test_backward_factory_failure_keeps_complete_active_runtime():
    """A backward preparation failure must not publish candidate forward state."""
    calls = []
    active_recipe = _make_counting_recipe(("atomic-update", 1), calls)
    failing_recipe = _make_counting_recipe(
        ("atomic-update", 2),
        calls,
        fail_on_grad_output=True,
    )
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, active_recipe)

    old_runtime = module._quantization_runtime  # pylint: disable=protected-access
    old_recipe = module.fp8_meta["recipe"]
    old_forward_state = module.fp8_meta["scaling_fwd"]
    old_backward_state = module.fp8_meta["scaling_bwd"]
    old_forward_quantizers = module.quantizers["scaling_fwd"]
    old_backward_quantizers = module.quantizers["scaling_bwd"]

    with pytest.raises(RuntimeError, match="backward factory failure"):
        _ensure_runtime(module, failing_recipe)

    assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
    assert module.fp8_meta["recipe"] is old_recipe
    assert module.fp8_meta["scaling_fwd"] is old_forward_state
    assert module.fp8_meta["scaling_bwd"] is old_backward_state
    assert module.quantizers["scaling_fwd"] is old_forward_quantizers
    assert module.quantizers["scaling_bwd"] is old_backward_quantizers


@pytest.mark.parametrize(
    ("module_factory", "num_gemms", "cache_names"),
    (
        pytest.param(
            lambda: Linear(16, 16, bias=False, device="cuda", name="linear"),
            1,
            ("weight",),
            id="linear",
        ),
        pytest.param(
            lambda: LayerNormLinear(
                16,
                16,
                bias=False,
                device="cuda",
                name="layernorm_linear",
            ),
            1,
            ("weight",),
            id="layernorm_linear",
        ),
        pytest.param(
            lambda: LayerNormMLP(
                16,
                32,
                bias=False,
                device="cuda",
                name="layernorm_mlp",
            ),
            2,
            ("fc1_weight", "fc2_weight"),
            id="layernorm_mlp",
        ),
        pytest.param(
            lambda: GroupedLinear(
                2,
                16,
                16,
                bias=False,
                device="cuda",
                name="grouped_linear",
            ),
            2,
            ("weight0", "weight1"),
            id="grouped_linear",
        ),
    ),
)
def test_runtime_update_workspace_lifecycle(
    module_factory,
    num_gemms,
    cache_names,
):
    """Workspace ownership stays atomic across updates in every module family."""
    calls = []
    active_recipe = _make_counting_recipe(("workspace-update", 1), calls)
    equal_recipe = _make_counting_recipe(("workspace-update", 1), calls)
    failing_recipe = _make_counting_recipe(
        ("workspace-update", 2),
        calls,
        fail_on_grad_output=True,
    )
    replacement_recipe = _make_counting_recipe(("workspace-update", 3), calls)
    module = module_factory()
    assert _ensure_runtime(module, active_recipe, num_gemms=num_gemms)

    active_runtime = module._quantization_runtime  # pylint: disable=protected-access
    active_views = (
        module.fp8_meta["recipe"],
        module.fp8_meta["scaling_fwd"],
        module.fp8_meta["scaling_bwd"],
        module.quantizers["scaling_fwd"],
        module.quantizers["scaling_bwd"],
    )
    workspaces = {cache_name: object() for cache_name in cache_names}
    for cache_name, workspace in workspaces.items():
        module._fp8_workspaces[cache_name] = workspace  # pylint: disable=protected-access

    # A global revision change with an equal semantic runtime preserves caches.
    assert not _ensure_runtime(module, equal_recipe, num_gemms=num_gemms)
    assert module._quantization_runtime is active_runtime  # pylint: disable=protected-access
    for cache_name, workspace in workspaces.items():
        assert module._fp8_workspaces[cache_name] is workspace  # pylint: disable=protected-access

    # Candidate construction failure preserves both compatibility views and caches.
    with pytest.raises(RuntimeError, match="backward factory failure"):
        _ensure_runtime(module, failing_recipe, num_gemms=num_gemms)
    assert module._quantization_runtime is active_runtime  # pylint: disable=protected-access
    current_views = (
        module.fp8_meta["recipe"],
        module.fp8_meta["scaling_fwd"],
        module.fp8_meta["scaling_bwd"],
        module.quantizers["scaling_fwd"],
        module.quantizers["scaling_bwd"],
    )
    assert all(current is active for current, active in zip(current_views, active_views))
    for cache_name, workspace in workspaces.items():
        assert module._fp8_workspaces[cache_name] is workspace  # pylint: disable=protected-access

    # Only a fully committed replacement clears cached workspaces.
    assert _ensure_runtime(module, replacement_recipe, num_gemms=num_gemms)
    replacement_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert replacement_runtime is not active_runtime
    assert not module._fp8_workspaces  # pylint: disable=protected-access


@requires_fp8
def test_forward_recipe_mutation_breaks_committed_key_invariant():
    """A forward path must not change its committed recipe snapshot."""

    class MutatingLinear(Linear):
        def forward(self, inp, *args, **kwargs):
            output = super().forward(inp, *args, **kwargs)
            self.fp8_meta["recipe"].fp8_dpa = True
            return output

    FP8GlobalStateManager.reset()
    calls = []
    recipe = _make_counting_recipe(("forward-recipe-mutation", 1), calls)
    module = MutatingLinear(16, 16, bias=False, device="cuda")
    inp = torch.randn(8, 16, device="cuda", requires_grad=True)

    try:
        with pytest.raises(AssertionError, match="Runtime recipe mutated after commit"):
            _run_update_step(module, recipe, inp)
    finally:
        FP8GlobalStateManager.reset()


@pytest.mark.parametrize(
    ("module_factory", "input_factory", "forward_args"),
    (
        pytest.param(
            lambda: Linear(16, 16, bias=False, device="cuda"),
            lambda: torch.randn(8, 16, device="cuda", requires_grad=True),
            (),
            id="linear",
        ),
        pytest.param(
            lambda: LayerNormLinear(16, 16, bias=False, device="cuda"),
            lambda: torch.randn(8, 16, device="cuda", requires_grad=True),
            (),
            id="layernorm-linear",
        ),
        pytest.param(
            lambda: LayerNormMLP(16, 32, bias=False, device="cuda"),
            lambda: torch.randn(8, 16, device="cuda", requires_grad=True),
            (),
            id="layernorm-mlp",
        ),
        pytest.param(
            lambda: GroupedLinear(2, 16, 16, bias=False, device="cuda"),
            lambda: torch.randn(8, 16, device="cuda", requires_grad=True),
            ([4, 4],),
            id="grouped-linear",
        ),
        pytest.param(
            lambda: DotProductAttention(
                num_attention_heads=2,
                kv_channels=16,
                attention_dropout=0.0,
                qkv_format="bshd",
                name="dpa",
            ).cuda(),
            lambda: tuple(
                torch.randn(
                    2,
                    8,
                    2,
                    16,
                    device="cuda",
                    requires_grad=True,
                )
                for _ in range(3)
            ),
            (),
            id="dot-product-attention",
        ),
    ),
)
@requires_fp8
def test_module_family_executes_after_recipe_update(
    module_factory,
    input_factory,
    forward_args,
):
    """A semantic-only runtime replacement is numerically transparent."""
    calls = []
    first_recipe = _make_counting_recipe(("module-execution", 1), calls)
    second_recipe = _make_counting_recipe(("module-execution", 2), calls)
    module = module_factory()

    first_inp = input_factory()
    second_inp = _clone_inputs(first_inp)
    first_trace, output_grad = _run_numerical_step(
        module,
        first_recipe,
        first_inp,
        *forward_args,
    )
    first_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert first_runtime is not None

    second_trace, _ = _run_numerical_step(
        module,
        second_recipe,
        second_inp,
        *forward_args,
        output_grad=output_grad,
    )
    second_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert second_runtime is not first_runtime
    assert second_runtime.key.recipe_config == second_recipe.quantizer_config()
    _assert_numerical_traces_match(second_trace, first_trace)


def test_equivalent_recipe_round_trip_matches_uninterrupted_training():
    """A -> equivalent B -> A training is bitwise identical to uninterrupted A."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    reference, switching = _make_identical_linears()
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9)
    switching_optimizer = torch.optim.SGD(switching.parameters(), lr=0.01, momentum=0.9)
    reference_recipe = Float8CurrentScaling()
    native_recipe = Float8CurrentScaling()
    equivalent_recipe = CustomRecipe(qfactory=current_scaling_factory)
    schedule = (native_recipe, equivalent_recipe, native_recipe, native_recipe)
    previous_runtime = None
    torch.manual_seed(6100)

    try:
        for step, switching_recipe in enumerate(schedule):
            base_inp = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
            reference_inp = base_inp.detach().clone().requires_grad_(True)
            switching_inp = base_inp.detach().clone().requires_grad_(True)
            reference_trace, output_grad = _run_numerical_step(
                reference,
                reference_recipe,
                reference_inp,
            )
            switching_trace, _ = _run_numerical_step(
                switching,
                switching_recipe,
                switching_inp,
                output_grad=output_grad,
            )
            _assert_numerical_traces_match(switching_trace, reference_trace)

            runtime = switching._quantization_runtime  # pylint: disable=protected-access
            if step in (1, 2):
                assert runtime is not previous_runtime
            if step == 3:
                assert runtime is previous_runtime
            previous_runtime = runtime

            reference_optimizer.step()
            switching_optimizer.step()
            _assert_modules_and_sgd_state_match(
                reference,
                switching,
                reference_optimizer,
                switching_optimizer,
            )
    finally:
        FP8GlobalStateManager.reset()


@pytest.mark.parametrize(
    "module_factory",
    (
        pytest.param(
            lambda: MultiheadAttention(
                hidden_size=32,
                num_attention_heads=2,
                attention_dropout=0.0,
                attn_mask_type="no_mask",
                bias=False,
                device="cuda",
                name="mha",
            ),
            id="multihead-attention",
        ),
        pytest.param(
            lambda: TransformerLayer(
                hidden_size=32,
                ffn_hidden_size=64,
                num_attention_heads=2,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                self_attn_mask_type="no_mask",
                bias=False,
                device="cuda",
            ),
            id="transformer-layer",
        ),
    ),
)
def test_composed_module_executes_after_recipe_update(module_factory):
    """A composed module's semantic-only runtime replacements are bitwise exact."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    calls = []
    first_recipe = _make_counting_recipe(("composed-execution", 1), calls)
    second_recipe = _make_counting_recipe(("composed-execution", 2), calls)
    module = module_factory()

    first_inp = torch.randn(8, 2, 32, device="cuda", requires_grad=True)
    second_inp = _clone_inputs(first_inp)
    first_trace, output_grad = _run_numerical_step(module, first_recipe, first_inp)
    owners = _active_runtime_owners(module)
    assert len(owners) >= 2
    assert any(isinstance(owner, DotProductAttention) for owner in owners)
    first_runtimes = {
        id(owner): owner._quantization_runtime  # pylint: disable=protected-access
        for owner in owners
    }

    second_trace, _ = _run_numerical_step(
        module,
        second_recipe,
        second_inp,
        output_grad=output_grad,
    )
    assert _active_runtime_owners(module) == owners
    for owner in owners:
        runtime = owner._quantization_runtime  # pylint: disable=protected-access
        assert runtime is not first_runtimes[id(owner)]
        assert runtime.key.recipe_config == second_recipe.quantizer_config()
    _assert_numerical_traces_match(second_trace, first_trace)


@pytest.mark.parametrize("eager_apply", (False, True), ids=("lazy", "apply-recipe"))
def test_inert_builtin_dpa_recipe_update_matches_fresh_target(eager_apply):
    """A warmed BF16-attention MHA accepts an outer built-in recipe update."""
    if not is_nvfp4_available():
        pytest.skip("NVFP4 is not available")
    available, reason = is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    def make_mha():
        return MultiheadAttention(
            hidden_size=128,
            num_attention_heads=2,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            name="mha",
        )

    FP8GlobalStateManager.reset()
    torch.manual_seed(2026)
    switching = make_mha()
    reference = make_mha()
    reference.load_state_dict(switching.state_dict())
    old_recipe = NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
    )
    target_recipe = MXFP8BlockScaling()

    try:
        warmup_inp = torch.randn(
            128,
            2,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        _run_numerical_step(switching, old_recipe, warmup_inp)
        assert switching.core_attention.fp8_initialized

        if eager_apply:
            apply_recipe(switching, target_recipe)

        base_inp = torch.randn(128, 2, 128, device="cuda", dtype=torch.bfloat16)
        actual, output_grad = _run_numerical_step(
            switching,
            target_recipe,
            base_inp.detach().clone().requires_grad_(True),
        )
        expected, _ = _run_numerical_step(
            reference,
            target_recipe,
            base_inp.detach().clone().requires_grad_(True),
            output_grad=output_grad,
        )
        _assert_numerical_traces_match(actual, expected)
    finally:
        FP8GlobalStateManager.reset()


@pytest.mark.parametrize(
    "module_factory",
    (
        pytest.param(
            lambda: MultiheadAttention(
                hidden_size=128,
                num_attention_heads=2,
                attention_dropout=0.0,
                attn_mask_type="no_mask",
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                name="mha",
            ),
            id="multihead-attention",
        ),
        pytest.param(
            lambda: TransformerLayer(
                hidden_size=128,
                ffn_hidden_size=256,
                num_attention_heads=2,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                self_attn_mask_type="no_mask",
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
            ),
            id="transformer-layer",
        ),
    ),
)
def test_apply_recipe_before_first_mha_forward_has_stable_boundary_topology(
    module_factory,
    monkeypatch,
):
    """E2E TE-1 regression: first forward must not migrate unwarmed MHA roles."""
    from transformer_engine.pytorch.utils import get_device_compute_capability

    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    compute_capability = get_device_compute_capability()
    if compute_capability < (9, 0) or compute_capability >= (12, 0):
        pytest.skip("FP8 attention is not supported on this compute capability")
    # This test exercises runtime topology, not backend availability. Keep an
    # executable fallback on CI configurations without FP8 fused attention.
    monkeypatch.setenv("NVTE_UnfusedDPA_Emulate_FP8", "1")

    FP8GlobalStateManager.reset()
    module = module_factory()
    recipe = _make_compatible_fp8_mha_recipe(("apply-unwarmed-mha", type(module).__name__, 1))

    try:
        assert not _active_runtime_owners(module)
        expected_recipe_config = recipe.quantizer_config()
        apply_recipe(module, recipe)
        owners = _active_runtime_owners(module)
        assert owners
        before = [
            (
                owner._quantization_runtime,  # pylint: disable=protected-access
                owner._quantization_runtime.key,  # pylint: disable=protected-access
                owner._role_revision,  # pylint: disable=protected-access
                owner._quantization_runtime.role_revision,  # pylint: disable=protected-access
            )
            for owner in owners
        ]
        assert all(requested == active for _, _, requested, active in before)
        assert all(
            owner._quantization_runtime.key.recipe_config == expected_recipe_config
            and owner._quantization_runtime.recipe.fp8_dpa
            for owner in owners
        )

        mha = next(child for child in module.modules() if isinstance(child, MultiheadAttention))
        qkv = mha.layernorm_qkv if mha.input_layernorm else mha.qkv
        assert qkv._quantization_runtime.key.forward_roles[
            -1
        ] == QuantizerRole(  # pylint: disable=protected-access
            module_type="dpa",
            tensor_type="qkv",
            name=mha.core_attention.name or "",
        )
        assert mha.proj._quantization_runtime.key.backward_roles[
            -1
        ] == QuantizerRole(  # pylint: disable=protected-access
            module_type="dpa",
            tensor_type="do",
            name=mha.core_attention.name or "",
        )

        for _ in range(2):
            inp = torch.randn(
                128,
                2,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            _run_update_step(module, recipe, inp)
            after = [
                (
                    owner._quantization_runtime,  # pylint: disable=protected-access
                    owner._quantization_runtime.key,  # pylint: disable=protected-access
                    owner._role_revision,  # pylint: disable=protected-access
                    owner._quantization_runtime.role_revision,  # pylint: disable=protected-access
                )
                for owner in owners
            ]
            assert after == before
            assert recipe.quantizer_config() == expected_recipe_config
    finally:
        FP8GlobalStateManager.reset()


def test_mha_warmed_and_unwarmed_topology_matches_and_rope_does_not_migrate_runtime(monkeypatch):
    """Cover the TE-1 invariants not asserted by existing attention numerics tests."""
    from transformer_engine.pytorch import RotaryPositionEmbedding
    from transformer_engine.pytorch.utils import get_device_compute_capability

    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    compute_capability = get_device_compute_capability()
    if compute_capability < (9, 0) or compute_capability >= (12, 0):
        pytest.skip("FP8 attention is not supported on this compute capability")
    # This test exercises runtime topology, not backend availability. Keep an
    # executable fallback on CI configurations without FP8 fused attention.
    monkeypatch.setenv("NVTE_UnfusedDPA_Emulate_FP8", "1")

    FP8GlobalStateManager.reset()
    recipe = _make_compatible_fp8_mha_recipe(("mha-warm-rope-topology", 1))

    def make_mha():
        return MultiheadAttention(
            hidden_size=128,
            num_attention_heads=2,
            attention_dropout=0.0,
            attn_mask_type="no_mask",
            bias=False,
            params_dtype=torch.bfloat16,
            device="cuda",
            name="mha",
        )

    def runtime_signature(module):
        signature = []
        for fqn, owner in module.named_modules():
            if not isinstance(owner, TransformerEngineBaseModule):
                continue
            runtime = owner._quantization_runtime  # pylint: disable=protected-access
            assert runtime is not None
            signature.append(
                (
                    fqn,
                    runtime.key.forward_roles,
                    runtime.key.backward_roles,
                    owner._role_revision,  # pylint: disable=protected-access
                    runtime.role_revision,
                    tuple(type(quantizer) for quantizer in runtime.forward_quantizers),
                    tuple(type(quantizer) for quantizer in runtime.backward_quantizers),
                )
            )
        return signature

    try:
        warmed = make_mha()
        unwarmed = make_mha()

        warm_inp = torch.randn(
            128,
            2,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        _run_update_step(warmed, recipe, warm_inp)
        assert not _active_runtime_owners(unwarmed)

        apply_recipe(unwarmed, recipe)
        apply_recipe(warmed, recipe)
        assert runtime_signature(unwarmed) == runtime_signature(warmed)

        # Toggle RoPE on the already-warmed instance. The old forward-time
        # wiring changed QKV boundary roles between these two call shapes.
        owners = _active_runtime_owners(warmed)
        before_rope = [
            (
                owner._quantization_runtime,  # pylint: disable=protected-access
                owner._quantization_runtime.key,  # pylint: disable=protected-access
                owner._role_revision,  # pylint: disable=protected-access
                owner._quantization_runtime.role_revision,  # pylint: disable=protected-access
            )
            for owner in owners
        ]
        rotary_pos_emb = RotaryPositionEmbedding(dim=64)(128).to(device="cuda")
        rope_inp = torch.randn(
            128,
            2,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        _run_update_step(
            warmed,
            recipe,
            rope_inp,
            rotary_pos_emb=rotary_pos_emb,
        )
        after_rope = [
            (
                owner._quantization_runtime,  # pylint: disable=protected-access
                owner._quantization_runtime.key,  # pylint: disable=protected-access
                owner._role_revision,  # pylint: disable=protected-access
                owner._quantization_runtime.role_revision,  # pylint: disable=protected-access
            )
            for owner in owners
        ]
        assert after_rope == before_rope
    finally:
        FP8GlobalStateManager.reset()


def test_apply_recipe_rejects_e2e_mha_reproducer_during_planning():
    """The shipped factory's incompatible FP8-MHA boundaries fail before commit."""
    from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
        nvfp4_linear_fp8_dpa_factory,
    )

    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    if not is_nvfp4_available():
        pytest.skip("NVFP4 is required for the E2E TE-1 reproducer")

    FP8GlobalStateManager.reset()
    module = MultiheadAttention(
        hidden_size=32,
        num_attention_heads=2,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        bias=False,
        device="cuda",
        name="mha",
    )
    recipe = CustomRecipe(
        qfactory=nvfp4_linear_fp8_dpa_factory,
        fp8_dpa=True,
        fp8_mha=True,
    )
    owners = [child for child in module.modules() if isinstance(child, TransformerEngineBaseModule)]
    old_global_state = _global_recipe_state()

    try:
        # The owner raises TypeError; apply_recipe annotates but does not rewrite it.
        with pytest.raises(TypeError, match="O quantizer is NVFP4Quantizer"):
            apply_recipe(module, recipe)

        assert _global_recipe_state() == old_global_state
        assert all(
            owner._quantization_runtime is None for owner in owners
        )  # pylint: disable=protected-access
        assert all(
            owner._role_revision == 0 for owner in owners
        )  # pylint: disable=protected-access
        assert all("scaling_fwd" not in owner.fp8_meta for owner in owners)
    finally:
        FP8GlobalStateManager.reset()


@pytest.mark.parametrize("attention_type", ("self", "cross"))
@pytest.mark.parametrize("input_layernorm", (False, True))
@pytest.mark.parametrize(
    "name",
    (pytest.param(None, id="unnamed"), pytest.param("mha", id="named")),
)
def test_mha_constructor_uses_consistent_dpa_qkv_role(attention_type, input_layernorm, name):
    """Every MHA variant uses one effective name on both sides of its boundaries.

    Construction alone is the regression guard: ``dpa_name`` used to be
    referenced before assignment, so every one of these variants raised
    ``NameError`` from ``__init__``.
    """
    module = MultiheadAttention(
        hidden_size=32,
        num_attention_heads=2,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        attention_type=attention_type,
        input_layernorm=input_layernorm,
        bias=False,
        device="cuda",
        name=name,
    )

    expected_qkv = QuantizerRole(
        module_type="dpa",
        tensor_type="qkv",
        name=module.core_attention.name,
    )
    if attention_type == "self":
        qkv_producers = [module.layernorm_qkv if input_layernorm else module.qkv]
    else:
        qkv_producers = [
            module.layernorm_query if input_layernorm else module.query_layer,
            module.key_value,
        ]
    for producer in qkv_producers:
        assert (
            producer._declared_output_quantizer_role  # pylint: disable=protected-access
            == expected_qkv
        )

    dpa_own_qkv_role = module.core_attention.get_quantizer_roles(
        fwd=True,
        num_quantizers=9,
        boundary_role=None,
    )[0]
    assert dpa_own_qkv_role == expected_qkv
    assert module.core_attention.name == f"{module.name}.core_attention"
    assert module.proj.name == f"{module.name}.proj"


@pytest.mark.parametrize(
    "name",
    (pytest.param(None, id="unnamed"), pytest.param("layer", id="named")),
)
def test_transformer_layer_constructor_declares_dpa_qkv_role(name):
    """TransformerLayer builds its MHA and both boundary sides agree.

    A TransformerLayer always has a name (its own auto name when unnamed) and
    passes it down, so the unnamed case composes a name here rather than "".
    """
    layer = TransformerLayer(
        hidden_size=32,
        ffn_hidden_size=64,
        num_attention_heads=2,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        self_attn_mask_type="no_mask",
        bias=False,
        device="cuda",
        name=name,
    )

    mha = layer.self_attention
    dpa = mha.core_attention
    assert dpa.name == f"{mha.name}.core_attention"
    if name is not None:
        assert dpa.name == "layer.self_attention.core_attention"

    expected_qkv = QuantizerRole(module_type="dpa", tensor_type="qkv", name=dpa.name)
    assert (
        mha.layernorm_qkv._declared_output_quantizer_role  # pylint: disable=protected-access
        == expected_qkv
    )
    assert (
        dpa.get_quantizer_roles(fwd=True, num_quantizers=9, boundary_role=None)[0] == expected_qkv
    )


@pytest.mark.parametrize("attention_type", ("self", "cross"))
@pytest.mark.parametrize("input_layernorm", (False, True))
def test_mha_declares_all_boundary_topology_variants(attention_type, input_layernorm):
    """Self/cross-attention QKV producers expose stable recipe-resolved roles."""
    module = MultiheadAttention(
        hidden_size=32,
        num_attention_heads=2,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        attention_type=attention_type,
        input_layernorm=input_layernorm,
        bias=False,
        device="cuda",
        name="mha",
    )
    recipe = CustomRecipe(
        qfactory=lambda _role: IdentityQuantizer(),
        qfactory_key=("mha-topology-variants", attention_type, input_layernorm),
        fp8_mha=True,
    )
    expected_qkv = QuantizerRole(
        module_type="dpa",
        tensor_type="qkv",
        name=module.core_attention.name or "",
    )
    if attention_type == "self":
        qkv_producers = [module.layernorm_qkv if input_layernorm else module.qkv]
    else:
        qkv_producers = [
            module.layernorm_query if input_layernorm else module.query_layer,
            module.key_value,
        ]

    def resolved_roles(owner, *, fwd, num_quantizers):
        _, roles = owner._resolve_quantizer_roles(  # pylint: disable=protected-access
            recipe=recipe,
            fwd=fwd,
            num_quantizers=num_quantizers,
        )
        assert roles is not None
        return roles

    for producer in qkv_producers:
        assert producer.output_quantizer_role is None
        assert (
            producer._declared_output_quantizer_role  # pylint: disable=protected-access
            == expected_qkv
        )
        assert resolved_roles(producer, fwd=True, num_quantizers=3)[-1] == expected_qkv
        assert producer._role_revision == 0  # pylint: disable=protected-access

    assert resolved_roles(module.proj, fwd=False, num_quantizers=2)[-1] == QuantizerRole(
        module_type="dpa",
        tensor_type="do",
        name=module.core_attention.name or "",
    )
    assert resolved_roles(module.core_attention, fwd=True, num_quantizers=9)[3] == QuantizerRole(
        module_type="linear",
        tensor_type="input",
        name=module.proj.name or "",
    )
    assert resolved_roles(module.core_attention, fwd=False, num_quantizers=6)[0] == QuantizerRole(
        module_type="linear",
        tensor_type="grad_output",
        name=qkv_producers[0].name or "",
    )


@pytest.mark.parametrize(
    "name",
    (pytest.param(None, id="unnamed"), pytest.param("mha", id="named")),
)
@pytest.mark.parametrize("input_layernorm", (False, True))
def test_delayed_qmi_mha_first_forward_preserves_initialized_runtimes(
    name,
    input_layernorm,
):
    """Initial composed roles are not mistaken for a delayed runtime update."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    def runtime_object_ids(owner):
        runtime = owner._quantization_runtime  # pylint: disable=protected-access
        assert runtime is not None
        forward_state = runtime.forward_state
        backward_state = runtime.backward_state
        return tuple(
            id(obj)
            for obj in (
                runtime,
                forward_state,
                backward_state,
                forward_state.scale,
                forward_state.amax_history,
                backward_state.scale,
                backward_state.amax_history,
                *runtime.forward_quantizers,
                *runtime.backward_quantizers,
            )
        )

    FP8GlobalStateManager.reset()
    recipe = DelayedScaling(amax_history_len=4)

    try:
        with quantized_model_init(recipe=recipe):
            module = MultiheadAttention(
                hidden_size=32,
                num_attention_heads=2,
                attention_dropout=0.0,
                attn_mask_type="no_mask",
                input_layernorm=input_layernorm,
                fuse_qkv_params=True,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                name=name,
            )

        qkv = module.layernorm_qkv if input_layernorm else module.qkv
        initialized_owners = _active_runtime_owners(module)
        assert initialized_owners == [qkv, module.proj]
        assert all(owner.primary_weights_in_fp8 for owner in initialized_owners)
        semantic_dpa_name = module.core_attention.name
        semantic_qkv_name = qkv.name
        semantic_proj_name = module.proj.name
        assert all(
            owner._role_revision == owner._quantization_runtime.role_revision == 0
            for owner in initialized_owners
        )  # pylint: disable=protected-access
        assert qkv._quantization_runtime.key.forward_roles[-1] == QuantizerRole(
            module_type="dpa",
            tensor_type="qkv",
            name=semantic_dpa_name,
        )  # pylint: disable=protected-access
        assert module.proj._quantization_runtime.key.backward_roles[-1] == QuantizerRole(
            module_type="dpa",
            tensor_type="do",
            name=semantic_dpa_name,
        )  # pylint: disable=protected-access
        assert module.core_attention._declared_output_quantizer_role == QuantizerRole(
            module_type="linear",
            tensor_type="input",
            name=semantic_proj_name,
        )  # pylint: disable=protected-access
        assert module.core_attention._declared_grad_input_quantizer_role == QuantizerRole(
            module_type="linear",
            tensor_type="grad_output",
            name=semantic_qkv_name,
        )  # pylint: disable=protected-access

        before = {owner: runtime_object_ids(owner) for owner in initialized_owners}
        inp = torch.randn(
            8,
            2,
            32,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        _run_update_step(module, recipe, inp)

        assert all(runtime_object_ids(owner) == before[owner] for owner in initialized_owners)
        assert all(
            owner._role_revision == owner._quantization_runtime.role_revision == 0
            for owner in initialized_owners
        )  # pylint: disable=protected-access
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_explicit_mha_boundary_override_wins_and_can_restore_declared_role():
    """Public role overrides retain revision semantics above declared topology."""
    FP8GlobalStateManager.reset()
    module = MultiheadAttention(
        hidden_size=32,
        num_attention_heads=2,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        bias=False,
        device="cuda",
        name="mha",
    )
    recipe = CustomRecipe(
        qfactory=lambda _role: IdentityQuantizer(),
        qfactory_key=("mha-explicit-boundary-override", 1),
        fp8_mha=True,
    )
    qkv = module.qkv

    try:
        apply_recipe(qkv, recipe)
        declared_runtime = qkv._quantization_runtime  # pylint: disable=protected-access
        declared_role = declared_runtime.key.forward_roles[-1]

        override = QuantizerRole(
            module_type="linear",
            tensor_type="input",
            name="explicit-consumer",
        )
        qkv.output_quantizer_role = override
        assert qkv._role_revision == 1  # pylint: disable=protected-access
        apply_recipe(qkv, recipe)
        override_runtime = qkv._quantization_runtime  # pylint: disable=protected-access
        assert override_runtime is not declared_runtime
        assert override_runtime.key.forward_roles[-1] == override

        qkv.output_quantizer_role = None
        assert qkv._role_revision == 2  # pylint: disable=protected-access
        apply_recipe(qkv, recipe)
        restored_runtime = qkv._quantization_runtime  # pylint: disable=protected-access
        assert restored_runtime is not override_runtime
        assert restored_runtime.key.forward_roles[-1] == declared_role
    finally:
        FP8GlobalStateManager.reset()


def test_shipped_dpa_factory_unwarmed_mha_keeps_bf16_boundaries_stable():
    """The documented fp8_dpa=True/fp8_mha=False mode remains lazy-forward compatible."""
    from transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo import (
        nvfp4_linear_fp8_dpa_factory,
    )
    from transformer_engine.pytorch.utils import get_device_compute_capability

    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    if not is_nvfp4_available():
        pytest.skip("NVFP4 is required for the shipped DPA factory")
    compute_capability = get_device_compute_capability()
    if compute_capability < (9, 0) or compute_capability >= (12, 0):
        pytest.skip("FP8 attention is not supported on this compute capability")

    FP8GlobalStateManager.reset()
    module = MultiheadAttention(
        hidden_size=128,
        num_attention_heads=2,
        attention_dropout=0.0,
        attn_mask_type="no_mask",
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        name="mha",
    )
    recipe = CustomRecipe(
        qfactory=nvfp4_linear_fp8_dpa_factory,
        fp8_dpa=True,
        fp8_mha=False,
    )

    try:
        apply_recipe(module, recipe)
        owners = _active_runtime_owners(module)
        before = [
            (
                owner._quantization_runtime,  # pylint: disable=protected-access
                owner._quantization_runtime.key,  # pylint: disable=protected-access
                owner._role_revision,  # pylint: disable=protected-access
            )
            for owner in owners
        ]
        inp = torch.randn(
            128,
            2,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        _run_update_step(module, recipe, inp)
        after = [
            (
                owner._quantization_runtime,  # pylint: disable=protected-access
                owner._quantization_runtime.key,  # pylint: disable=protected-access
                owner._role_revision,  # pylint: disable=protected-access
            )
            for owner in owners
        ]
        assert after == before
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_rejects_quantized_primary_update_before_candidate_construction():
    """The model-wide API preserves the quantized-model-init support boundary."""
    FP8GlobalStateManager.reset()
    initial_recipe = CustomRecipe(
        qfactory=lambda _role: IdentityQuantizer(),
        qfactory_key=("quantized-primary-apply", 1),
    )
    replacement_calls = []
    replacement_recipe = _make_counting_recipe(
        ("quantized-primary-apply", 2),
        replacement_calls,
    )

    try:
        with quantized_model_init(recipe=initial_recipe):
            module = Linear(
                16,
                16,
                bias=False,
                params_dtype=torch.bfloat16,
                device="cuda",
                name="linear",
            )
        assert module.primary_weights_in_fp8
        old_views = _runtime_views(module)
        old_global_state = _global_recipe_state()

        with pytest.raises(
            RuntimeError,
            match="Recipe mismatch for quantized primary weights",
        ):
            apply_recipe(module, replacement_recipe)

        assert all(current is old for current, old in zip(_runtime_views(module), old_views))
        assert _global_recipe_state() == old_global_state
        assert not replacement_calls
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_success_noop_mutation_and_forward_fast_path():
    """Explicit application publishes once and matching forwards remain lazy-path no-ops."""
    FP8GlobalStateManager.reset()
    calls = []
    first_recipe = _make_counting_recipe(("apply-success", 1), calls)
    equal_recipe = _make_counting_recipe(("apply-success", 1), calls)
    model = torch.nn.Sequential(
        Linear(16, 16, bias=False, device="cuda", name="first"),
        Linear(16, 16, bias=False, device="cuda", name="second"),
    )

    try:
        apply_recipe(model, first_recipe)
        first_runtimes = [module._quantization_runtime for module in model]
        first_config = FP8GlobalStateManager.get_quantizer_config()
        first_factory_calls = len(calls)
        assert first_factory_calls == 10
        assert FP8GlobalStateManager.get_fp8_recipe() is first_recipe

        # An independent equal recipe updates only the manager's requested
        # recipe object. Runtime identity, configuration, and factories stay fixed.
        apply_recipe(model, equal_recipe)
        assert [module._quantization_runtime for module in model] == first_runtimes
        assert FP8GlobalStateManager.get_quantizer_config() == first_config
        assert FP8GlobalStateManager.get_fp8_recipe() is equal_recipe
        assert len(calls) == first_factory_calls

        # The first matching autocast forward after explicit application uses
        # each module's steady-state path and performs no factory work.
        inp = torch.randn(8, 16, device="cuda", requires_grad=True)
        _run_update_step(model, equal_recipe, inp)
        assert [module._quantization_runtime for module in model] == first_runtimes
        assert len(calls) == first_factory_calls

        # Mutating and reusing the same recipe object produces a real model-wide
        # replacement and a different active configuration.
        equal_recipe.qfactory_key = ("apply-success", 2)
        apply_recipe(model, equal_recipe)
        assert all(
            module._quantization_runtime is not old_runtime
            for module, old_runtime in zip(model, first_runtimes)
        )
        assert FP8GlobalStateManager.get_quantizer_config() != first_config
        assert len(calls) == first_factory_calls + 10
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_computes_config_once(monkeypatch):
    """All participant plans receive one shared semantic configuration."""
    FP8GlobalStateManager.reset()
    config_calls = []
    factory_calls = []
    make_config = CustomRecipe._make_quantizer_config

    def counted_make_config(recipe):
        config_calls.append(recipe)
        return make_config(recipe)

    monkeypatch.setattr(CustomRecipe, "_make_quantizer_config", counted_make_config)
    recipe = _make_counting_recipe(("apply-config-once", 1), factory_calls)
    model = torch.nn.Sequential(
        Linear(16, 16, bias=False, device="cuda", name="first"),
        Linear(16, 16, bias=False, device="cuda", name="second"),
    )

    try:
        apply_recipe(model, recipe)
        assert config_calls == [recipe]
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_reconstructs_stateless_runtime_after_checkpoint_restore():
    """A restored model can rebuild its committed recipe before its first forward."""
    FP8GlobalStateManager.reset()
    source_calls = []
    source = Linear(16, 16, bias=False, device="cuda", name="linear")
    initial_recipe = _make_counting_recipe(("checkpoint-resume", 1), source_calls)
    committed_recipe = _make_counting_recipe(("checkpoint-resume", 2), source_calls)
    inp = torch.randn(8, 16, device="cuda", requires_grad=True)

    try:
        apply_recipe(source, initial_recipe)
        apply_recipe(source, committed_recipe)
        checkpoint = source.state_dict()
        expected, output_grad = _run_numerical_step(source, committed_recipe, inp)

        # Model/checkpoint restoration happens in a fresh runtime context. The
        # intended recipe is reconstructed and explicitly applied before the
        # restored model executes its first forward.
        FP8GlobalStateManager.reset()
        restored = Linear(16, 16, bias=False, device="cuda", name="linear")
        restored.load_state_dict(checkpoint)
        assert restored._quantization_runtime is None  # pylint: disable=protected-access

        resumed_calls = []
        resumed_recipe = _make_counting_recipe(("checkpoint-resume", 2), resumed_calls)
        apply_recipe(restored, resumed_recipe)
        runtime = restored._quantization_runtime  # pylint: disable=protected-access
        assert runtime is not None
        assert runtime.key.recipe_config == committed_recipe.quantizer_config()

        restored_inp = _clone_inputs(inp)
        actual, _ = _run_numerical_step(
            restored,
            resumed_recipe,
            restored_inp,
            output_grad=output_grad,
        )
        _assert_numerical_traces_match(actual, expected)
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_includes_unexecuted_conditional_branch():
    """Traversal updates owners independently of the branch executed by forward."""

    class ConditionalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.executed = Linear(16, 16, bias=False, device="cuda", name="executed")
            self.unexecuted = Linear(16, 16, bias=False, device="cuda", name="unexecuted")

        def forward(self, inp, *, use_unexecuted=False):
            module = self.unexecuted if use_unexecuted else self.executed
            return module(inp)

    FP8GlobalStateManager.reset()
    calls = []
    first_recipe = _make_counting_recipe(("apply-conditional", 1), calls)
    replacement_recipe = _make_counting_recipe(("apply-conditional", 2), calls)
    model = ConditionalModel()

    try:
        # Lazy execution initializes only one branch.
        inp = torch.randn(8, 16, device="cuda", requires_grad=True)
        with autocast(enabled=True, recipe=first_recipe):
            model(inp).sum().backward()
        old_runtime = model.executed._quantization_runtime
        assert old_runtime is not None
        assert model.unexecuted._quantization_runtime is None

        apply_recipe(model, replacement_recipe)
        assert model.executed._quantization_runtime is not old_runtime
        assert model.unexecuted._quantization_runtime is not None
        assert model.executed._quantization_runtime.key.recipe_config == (
            replacement_recipe.quantizer_config()
        )
        assert model.unexecuted._quantization_runtime.key.recipe_config == (
            replacement_recipe.quantizer_config()
        )
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_supports_custom_recipe_dpa():
    """Runtime-managed CustomRecipe DPA participates in model-wide application."""
    FP8GlobalStateManager.reset()
    calls = []
    recipe = _make_counting_recipe(("apply-custom-dpa", 1), calls)
    dpa = DotProductAttention(
        num_attention_heads=2,
        kv_channels=16,
        attention_dropout=0.0,
        name="dpa",
    ).cuda()

    try:
        apply_recipe(dpa, recipe)
        assert dpa._quantization_runtime is not None
        assert dpa._quantization_runtime.key.recipe_config == recipe.quantizer_config()
        assert len(calls) == 15
    finally:
        FP8GlobalStateManager.reset()


@pytest.mark.parametrize("failure_index", (0, 1, 2))
@requires_fp8
def test_apply_recipe_planning_failure_is_model_wide_atomic(failure_index):
    """A factory failure in any participant leaves modules and manager unchanged."""
    FP8GlobalStateManager.reset()
    calls = []
    active_recipe = _make_counting_recipe(("apply-failure", 1), calls)
    modules = [Linear(16, 16, bias=False, device="cuda", name=f"line{index}") for index in range(3)]
    model = torch.nn.Sequential(*modules)
    control = torch.nn.Sequential(
        *[Linear(16, 16, bias=False, device="cuda", name=f"line{index}") for index in range(3)]
    )
    control.load_state_dict(model.state_dict())

    try:
        apply_recipe(model, active_recipe)
        apply_recipe(control, active_recipe)
        warmup_inp = torch.randn(8, 16, device="cuda", requires_grad=True)
        expected, output_grad = _run_numerical_step(control, active_recipe, warmup_inp)
        actual, _ = _run_numerical_step(
            model,
            active_recipe,
            _clone_inputs(warmup_inp),
            output_grad=output_grad,
        )
        _assert_numerical_traces_match(actual, expected)
        old_global_state = _global_recipe_state()
        old_views = [_runtime_views(module) for module in modules]
        workspaces = []
        for module in modules:
            workspace = object()
            module._fp8_workspaces["weight"] = workspace
            workspaces.append(workspace)

        def failing_factory(role):
            if role is not None and role.name == f"line{failure_index}":
                raise RuntimeError("model-wide factory failure")
            return IdentityQuantizer()

        replacement_recipe = CustomRecipe(
            qfactory=failing_factory,
            qfactory_key=("apply-failure", 2, failure_index),
        )
        with pytest.raises(RuntimeError, match="model-wide factory failure") as failure:
            apply_recipe(model, replacement_recipe)
        # The owner's own exception type survives; the module is named in a note.
        assert type(failure.value) is RuntimeError
        assert any(
            f"while planning module '{failure_index}'" in note
            for note in getattr(failure.value, "__notes__", ())
        )

        assert _global_recipe_state() == old_global_state
        for module, expected_views, workspace in zip(modules, old_views, workspaces):
            assert all(
                current is expected
                for current, expected in zip(_runtime_views(module), expected_views)
            )
            assert module._fp8_workspaces["weight"] is workspace

        continuation_inp = torch.randn(8, 16, device="cuda", requires_grad=True)
        expected, output_grad = _run_numerical_step(
            control,
            active_recipe,
            continuation_inp,
        )
        actual, _ = _run_numerical_step(
            model,
            active_recipe,
            _clone_inputs(continuation_inp),
            output_grad=output_grad,
        )
        _assert_numerical_traces_match(actual, expected)
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_deduplicates_shared_runtime_owner():
    """A shared module is planned and applied exactly once."""
    FP8GlobalStateManager.reset()
    calls = []
    recipe = _make_counting_recipe(("apply-shared", 1), calls)
    shared = Linear(16, 16, bias=False, device="cuda", name="shared")
    model = torch.nn.Module()
    model.add_module("first", shared)
    model.add_module("second", shared)

    try:
        apply_recipe(model, recipe)
        assert shared._quantization_runtime is not None
        assert len(calls) == 5
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_rejects_fusible_owner_without_committing_anything():
    """A fusible owner rejects the model, and nothing is committed on the way there.

    Owners are planned as they are discovered, so a participant ahead of the
    fusible owner may have its factory invoked. Planning publishes no state, so
    the documented guarantee -- no module and no global recipe changes -- holds.
    """
    FP8GlobalStateManager.reset()

    calls = []

    def counting_factory(role):
        calls.append(role)
        return IdentityQuantizer()

    recipe = CustomRecipe(
        qfactory=counting_factory,
        qfactory_key=("apply-fusible", 1),
    )
    model = torch.nn.ModuleList(
        [
            Linear(16, 16, bias=False, device="cuda", name="linear"),
            te_ops.Quantize(),
        ]
    )
    old_global_state = _global_recipe_state()

    try:
        with pytest.raises(RuntimeError, match="does not support fusible operations"):
            apply_recipe(model, recipe)
        assert model[0]._quantization_runtime is None
        assert _global_recipe_state() == old_global_state
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_skips_zero_quantizer_fusible_owner():
    """An operation that builds no quantizers is inert, not an excluded owner."""
    FP8GlobalStateManager.reset()

    calls = []

    def counting_factory(role):
        calls.append(role)
        return IdentityQuantizer()

    recipe = CustomRecipe(
        qfactory=counting_factory,
        qfactory_key=("apply-zero-quantizer", 1),
    )
    model = torch.nn.ModuleList(
        [
            Linear(16, 16, bias=False, device="cuda", name="linear"),
            te_ops.LayerNorm(16, device="cuda"),
        ]
    )
    try:
        apply_recipe(model, recipe)
        assert model[0]._quantization_runtime is not None
        assert FP8GlobalStateManager.quantization_state.fp8_recipe is recipe

        # A fusible owner that does build quantizers still rejects.
        with pytest.raises(RuntimeError, match="does not support fusible operations"):
            apply_recipe(
                torch.nn.ModuleList([te_ops.Quantize()]),
                CustomRecipe(qfactory=counting_factory, qfactory_key=("apply-zero-q", 2)),
            )
    finally:
        FP8GlobalStateManager.reset()


def test_apply_recipe_rejects_legacy_dpa_path():
    """Model-wide application does not alter the built-in NVTE_DPA_* mechanism."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    dpa = DotProductAttention(
        num_attention_heads=2,
        kv_channels=16,
        attention_dropout=0.0,
        name="dpa",
    ).cuda()
    linear = Linear(16, 16, bias=False, device="cuda", name="linear")
    model = torch.nn.ModuleList([linear, dpa])
    old_global_state = _global_recipe_state()
    try:
        # Quantization-inert for attention: the DPA is skipped, not rejected, so a
        # stock TransformerLayer-shaped model can still be updated model-wide.
        inert = Float8CurrentScaling()
        apply_recipe(model, inert)
        assert linear._quantization_runtime is not None
        assert dpa._quantization_runtime is None
        assert FP8GlobalStateManager.quantization_state.fp8_recipe is inert

        # Asking the built-in path to quantize attention is still rejected.
        for attention_recipe in (
            Float8CurrentScaling(fp8_dpa=True),
            Float8CurrentScaling(fp8_dpa=True, fp8_mha=True),
        ):
            with pytest.raises(RuntimeError, match="only through CustomRecipe"):
                apply_recipe(model, attention_recipe)
            assert dpa._quantization_runtime is None

        # A bare inert DPA has nothing to update at all.
        with pytest.raises(ValueError, match="found no Transformer Engine runtime owners"):
            apply_recipe(dpa, Float8CurrentScaling())
    finally:
        FP8GlobalStateManager.reset()


def test_rejected_builtin_dpa_update_keeps_planning_and_forward_state_atomic(monkeypatch):
    """Unsupported built-in FP8-attention changes never publish partial state."""
    import importlib

    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    dpa_module = importlib.import_module(
        "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention"
    )
    monkeypatch.setattr(dpa_module, "_dpa_fp8_recipe", "Float8CurrentScaling")

    FP8GlobalStateManager.reset()
    dpa = DotProductAttention(
        num_attention_heads=2,
        kv_channels=16,
        attention_dropout=0.0,
        name="dpa",
    ).cuda()
    linear = Linear(16, 16, bias=False, device="cuda", name="linear")
    active_recipe = Float8CurrentScaling(fp8_dpa=True)

    try:
        with autocast(enabled=True, recipe=active_recipe):
            dpa.init_fp8_metadata()
        assert dpa.fp8_initialized

        metadata_keys = (
            "global_recipe",
            "local_recipes",
            "recipe",
            "scaling_fwd",
            "scaling_bwd",
        )
        metadata_before = {key: dpa.fp8_meta.get(key) for key in metadata_keys}

        def assert_metadata_unchanged():
            assert all(dpa.fp8_meta.get(key) is value for key, value in metadata_before.items())

        global_before = _global_recipe_state()

        # Planning an active-attention -> BF16-attention transition rejects before
        # the sibling Linear candidate or global recipe can be committed.
        with pytest.raises(RuntimeError, match="only through CustomRecipe"):
            apply_recipe(torch.nn.ModuleList([linear, dpa]), Float8CurrentScaling())
        assert linear._quantization_runtime is None  # pylint: disable=protected-access
        assert _global_recipe_state() == global_before
        assert_metadata_unchanged()

        changed_recipe = Float8CurrentScaling(fp8_dpa=True)
        changed_recipe.fp8_quant_fwd_inp = QParams(amax_epsilon=0.25)
        for _ in range(2):
            with pytest.raises(RuntimeError, match="built-in DotProductAttention recipe path"):
                with autocast(enabled=True, recipe=changed_recipe):
                    dpa.init_fp8_metadata()
            assert_metadata_unchanged()
    finally:
        FP8GlobalStateManager.reset()


def test_graph_capture_resizes_amax_history_and_leaves_a_consistent_runtime():
    """A direct resize bypasses the planner, so it must re-key the runtime itself."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    try:
        with quantized_model_init(enabled=True, recipe=DelayedScaling(amax_history_len=1024)):
            module = Linear(
                16, 16, bias=False, params_dtype=torch.bfloat16, device="cuda", name="linear"
            )
        inp = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)

        import transformer_engine.pytorch as te

        te.make_graphed_callables(
            module,
            (inp,),
            fp8_enabled=True,
            fp8_recipe=DelayedScaling(amax_history_len=16),
        )

        assert module.fp8_meta["scaling_fwd"].amax_history.shape[0] == 16
        runtime = module._quantization_runtime  # pylint: disable=protected-access
        assert runtime.recipe.amax_history_len == 16
        assert runtime.forward_state.recipe is runtime.recipe
        assert runtime.backward_state.recipe is runtime.recipe
        assert runtime.key.recipe_config == runtime.recipe.quantizer_config()
        assert module.fp8_meta["recipe"] is runtime.recipe
    finally:
        FP8GlobalStateManager.reset()


def test_live_cuda_graph_rejects_a_recipe_update_instead_of_ignoring_it():
    """A captured module replays captured kernels, so an update must not be accepted."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    import transformer_engine.pytorch as te

    FP8GlobalStateManager.reset()
    try:
        module = Linear(
            64, 64, bias=False, params_dtype=torch.bfloat16, device="cuda", name="linear"
        )
        inp = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        captured = Float8CurrentScaling()
        graphed = te.make_graphed_callables(module, (inp,), fp8_enabled=True, fp8_recipe=captured)
        assert module._graph_lease_count == 1  # pylint: disable=protected-access
        runtime = module._quantization_runtime  # pylint: disable=protected-access

        changed = Float8CurrentScaling()
        changed.fp8_quant_fwd_inp = QParams(power_2_scale=True)

        # apply_recipe used to be accepted and then silently ignored by replay.
        with pytest.raises(RuntimeError, match="captured by a live CUDA graph"):
            apply_recipe(module, changed)
        # Replaying under a different recipe was already refused by the graph's own
        # capture check; that path is unchanged.
        with pytest.raises(RuntimeError, match="differs from the CUDA graph capture recipe"):
            with autocast(enabled=True, recipe=changed):
                module(inp)
        assert module._quantization_runtime is runtime  # pylint: disable=protected-access

        # Re-entering the captured recipe is not a change and stays allowed.
        with autocast(enabled=True, recipe=captured):
            graphed(inp)

        # Releasing the graph releases the lease.
        graphed.reset()
        assert module._graph_lease_count == 0  # pylint: disable=protected-access
        apply_recipe(module, changed)
        # pylint: disable-next=protected-access
        assert module._quantization_runtime is not runtime
    finally:
        FP8GlobalStateManager.reset()


@pytest.mark.parametrize(
    "live_recipe",
    [
        pytest.param(DelayedScaling(amax_history_len=8), id="different-history"),
        pytest.param(DelayedScaling(amax_history_len=4, margin=1), id="different-margin"),
        pytest.param(MXFP8BlockScaling(), id="different-recipe"),
    ],
)
def test_checkpoint_restore_adopts_the_checkpoints_recipe(monkeypatch, live_recipe):
    """Loading a checkpoint replaces state; it is not an update from the live recipe."""
    # Delayed-scaling extra state is pickled, which is gated off by default.
    monkeypatch.setenv(UNSAFE_PICKLE_EXTRA_STATE_ENV, "1")
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    if live_recipe.mxfp8() and not is_mxfp8_available():
        pytest.skip("MXFP8 is required for this variant")

    FP8GlobalStateManager.reset()
    try:

        def warmed(active):
            module = Linear(
                64, 64, bias=False, params_dtype=torch.bfloat16, device="cuda", name="linear"
            )
            with autocast(enabled=True, recipe=active):
                module(torch.randn(64, 64, device="cuda", dtype=torch.bfloat16))
            return module

        saved = warmed(DelayedScaling(amax_history_len=4))
        state_dict = saved.state_dict()
        expected_history = saved.fp8_meta["scaling_fwd"].amax_history.clone()

        destination = warmed(live_recipe)
        destination.load_state_dict(state_dict)

        state = destination.fp8_meta["scaling_fwd"]
        assert state.amax_history.shape[0] == 4
        assert torch.equal(state.amax_history, expected_history)
        assert destination.fp8_meta["recipe"].delayed()
    finally:
        FP8GlobalStateManager.reset()


@requires_fp8
def test_apply_recipe_dispatches_on_the_owner_protocol_not_concrete_classes():
    """Any owner implementing the private plan/apply pair participates."""
    FP8GlobalStateManager.reset()

    class _StubOwner(torch.nn.Module):
        """A non-TE owner that plugs into te.apply_recipe()."""

        def __init__(self):
            super().__init__()
            self.planned = None
            self.applied = None

        def _plan_recipe_update(self, recipe, *, diagnostic_name):
            self.planned = (recipe, diagnostic_name)
            return "stub-update"

        def _apply_recipe_update(self, update):
            # Record the call itself, so a skipped owner is distinguishable from
            # one committed with a None update.
            self.applied = ("applied", update)

    class _InertOwner(_StubOwner):
        def _plan_recipe_update(self, recipe, *, diagnostic_name):
            super()._plan_recipe_update(recipe, diagnostic_name=diagnostic_name)
            return None

    stub, inert = _StubOwner(), _InertOwner()
    model = torch.nn.ModuleList([stub, inert])
    recipe = _make_counting_recipe(("apply-duck-typing", 1), [])
    try:
        apply_recipe(model, recipe)
        assert stub.planned == (recipe, "0")
        assert stub.applied == ("applied", "stub-update")
        assert inert.planned == (recipe, "1")
        assert inert.applied is None
        assert FP8GlobalStateManager.quantization_state.fp8_recipe is recipe
    finally:
        FP8GlobalStateManager.reset()


def test_apply_recipe_rejects_custom_delayed_dpa_crossing_before_commit():
    """A delayed custom DPA runtime cannot be handed to the built-in path."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    dpa = DotProductAttention(
        num_attention_heads=2,
        kv_channels=16,
        attention_dropout=0.0,
        name="dpa",
    ).cuda()
    linear = Linear(16, 16, bias=False, device="cuda", name="linear")
    model = torch.nn.ModuleList([linear, dpa])
    try:
        with autocast(
            enabled=True,
            recipe=CustomRecipe(
                qfactory=delayed_scaling_factory,
                qfactory_key=("apply-crossing-warm", 1),
                fp8_dpa=True,
            ),
        ):
            dpa.get_qkv_quantization_capabilities()
        runtime = dpa._quantization_runtime
        assert runtime is not None

        if not TransformerEngineBaseModule._runtime_has_delayed_scaling(runtime):
            pytest.skip("this factory produced no delayed DPA state")

        old_global_state = _global_recipe_state()
        with pytest.raises(RuntimeError, match="frozen after initialization"):
            apply_recipe(model, Float8CurrentScaling())
        assert dpa._quantization_runtime is runtime
        assert _global_recipe_state() == old_global_state
    finally:
        FP8GlobalStateManager.reset()


def test_zero_quantizer_norm_survives_same_class_change_on_lazy_path():
    """te.ops norms build no quantizers, so a same-class change cannot invalidate them."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    FP8GlobalStateManager.reset()
    model = te_ops.Sequential(te_ops.RMSNorm(16, device="cuda"))
    inp = torch.randn(16, 16, device="cuda", dtype=torch.bfloat16)
    try:
        first = Float8CurrentScaling()
        second = Float8CurrentScaling()
        second.fp8_quant_fwd_inp = QParams(power_2_scale=True)
        for active in (first, second, first):
            with torch.no_grad(), autocast(enabled=True, recipe=active):
                model(inp)
    finally:
        FP8GlobalStateManager.reset()


def test_apply_recipe_rejects_active_autocast_and_graph_capture(monkeypatch):
    """Explicit model-wide application is limited to the documented safe boundary."""
    FP8GlobalStateManager.reset()
    calls = []
    recipe = _make_counting_recipe(("apply-boundary", 1), calls)
    model = Linear(16, 16, bias=False, device="cuda", name="linear")

    try:
        with autocast(enabled=False):
            with pytest.raises(RuntimeError, match="outside te.autocast"):
                apply_recipe(model, recipe)
        assert not calls
        assert model._quantization_runtime is None

        monkeypatch.setattr(
            FP8GlobalStateManager,
            "fp8_graph_capturing",
            classmethod(lambda _cls: True),
        )
        with pytest.raises(RuntimeError, match="outside CUDA graph capture"):
            apply_recipe(model, recipe)
        assert not calls
        assert model._quantization_runtime is None
    finally:
        FP8GlobalStateManager.reset()


def test_candidate_validation_failure_keeps_complete_active_runtime(monkeypatch):
    """Validation runs before any candidate state is published."""
    calls = []
    active_recipe = _make_counting_recipe(("validation-update", 1), calls)
    candidate_recipe = _make_counting_recipe(("validation-update", 2), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, active_recipe)

    old_runtime = module._quantization_runtime  # pylint: disable=protected-access
    old_forward_state = module.fp8_meta["scaling_fwd"]
    old_backward_state = module.fp8_meta["scaling_bwd"]
    old_forward_quantizers = module.quantizers["scaling_fwd"]
    old_backward_quantizers = module.quantizers["scaling_bwd"]

    def reject_candidate(_candidate):
        assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
        raise RuntimeError("candidate validation failure")

    monkeypatch.setattr(module, "_validate_quantization_runtime", reject_candidate)
    with pytest.raises(RuntimeError, match="candidate validation failure"):
        _ensure_runtime(module, candidate_recipe)

    assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
    assert module.fp8_meta["scaling_fwd"] is old_forward_state
    assert module.fp8_meta["scaling_bwd"] is old_backward_state
    assert module.quantizers["scaling_fwd"] is old_forward_quantizers
    assert module.quantizers["scaling_bwd"] is old_backward_quantizers


@pytest.mark.parametrize("mismatched_tensor_type", ("input", "grad_output"))
def test_grouped_candidate_validation_is_atomic(mismatched_tensor_type):
    """Forward and backward grouped validation failures preserve all live state."""

    class UnsafeIdentityQuantizer(IdentityQuantizer):
        def is_requantization_safe(self):
            return False

    def make_recipe(
        key,
        *,
        mismatched_role=None,
        dtype=torch.bfloat16,
        unsafe_inputs=False,
    ):
        matching_role_count = 0

        def qfactory(role):
            nonlocal matching_role_count
            quantizer_dtype = dtype
            if role is not None and role.tensor_type == mismatched_role:
                quantizer_dtype = torch.bfloat16 if matching_role_count % 2 == 0 else torch.float16
                matching_role_count += 1
            quantizer_type = (
                UnsafeIdentityQuantizer
                if unsafe_inputs and role is not None and role.tensor_type == "input"
                else IdentityQuantizer
            )
            return quantizer_type(dtype=quantizer_dtype)

        return CustomRecipe(qfactory=qfactory, qfactory_key=key)

    module = GroupedLinear(2, 16, 16, bias=False, device="cuda", name="grouped")
    active_recipe = make_recipe(("grouped-atomic", "active", mismatched_tensor_type))
    _ensure_runtime(module, active_recipe, num_gemms=2)

    old_runtime = module._quantization_runtime  # pylint: disable=protected-access
    old_forward_state = module.fp8_meta["scaling_fwd"]
    old_backward_state = module.fp8_meta["scaling_bwd"]
    old_forward_quantizers = module.quantizers["scaling_fwd"]
    old_backward_quantizers = module.quantizers["scaling_bwd"]
    old_traits = old_runtime.owner_traits

    invalid_recipe = make_recipe(
        ("grouped-atomic", "invalid", mismatched_tensor_type),
        mismatched_role=mismatched_tensor_type,
    )
    with pytest.raises(ValueError, match="incompatible plain backend configurations"):
        _ensure_runtime(module, invalid_recipe, num_gemms=2)

    assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
    assert module.fp8_meta["scaling_fwd"] is old_forward_state
    assert module.fp8_meta["scaling_bwd"] is old_backward_state
    assert module.quantizers["scaling_fwd"] is old_forward_quantizers
    assert module.quantizers["scaling_bwd"] is old_backward_quantizers
    assert old_runtime.owner_traits is old_traits

    replacement_recipe = make_recipe(
        ("grouped-atomic", "replacement", mismatched_tensor_type),
        dtype=torch.float16,
        unsafe_inputs=True,
    )
    update = _prepare_runtime_update(module, replacement_recipe, num_gemms=2)
    # Planning derives traits for the candidate without touching the active runtime.
    assert old_runtime.owner_traits is old_traits
    assert module._apply_quantization_update(update)  # pylint: disable=protected-access
    replacement_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert replacement_runtime is not old_runtime
    traits = replacement_runtime.owner_traits
    assert traits is not old_traits
    assert traits.delayed_scaling_input_quantizer is None
    assert traits.unsafe_requantization_input_quantizer is replacement_runtime.forward_quantizers[0]
