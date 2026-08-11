# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Module-local runtime lifecycle tests for mid-training recipe updates."""

import pytest
import torch

from transformer_engine.common.recipe import CustomRecipe, DelayedScaling, Float8CurrentScaling
from transformer_engine.pytorch import (
    GroupedLinear,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    autocast,
    is_fp8_available,
)
from transformer_engine.pytorch.quantization import (
    DelayedScalingRequest,
    QuantizerRole,
)
from transformer_engine.pytorch.tensor.identity_tensor import IdentityQuantizer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _make_counting_recipe(key, calls, *, fail_on_grad_output=False):
    """Build an identity recipe whose factory calls are externally observable."""

    def qfactory(role):
        calls.append(role)
        if fail_on_grad_output and role is not None and role.tensor_type == "grad_output":
            raise RuntimeError("backward factory failure")
        return IdentityQuantizer()

    return CustomRecipe(qfactory=qfactory, qfactory_key=key)


def _ensure_runtime(module, recipe, revision, *, num_gemms=1):
    return module._ensure_active_quantization_runtime(  # pylint: disable=protected-access
        recipe=recipe,
        recipe_config=recipe.quantizer_config(),
        recipe_config_revision=revision,
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
    assert _ensure_runtime(module, recipe, revision=1)
    old_views = _runtime_views(module)

    with pytest.raises(
        RuntimeError,
        match="Mid-training recipe updates do not support delayed scaling",
    ):
        _ensure_runtime(module, replacement, revision=2)

    assert all(current is old for current, old in zip(_runtime_views(module), old_views))
    assert old_views[0].recipe_config_revision == 1


def test_delayed_runtime_rejects_role_and_slot_layout_updates():
    """Role and GEMM-layout changes cannot rebuild a delayed runtime."""
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    recipe = DelayedScaling(amax_history_len=4)
    assert _ensure_runtime(module, recipe, revision=1)
    old_views = _runtime_views(module)

    module.output_quantizer_role = QuantizerRole(
        module_type="linear",
        tensor_type="input",
        name="consumer",
    )
    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, recipe, revision=1)
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))

    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, recipe, revision=1, num_gemms=2)
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))

    """An active stateless runtime cannot acquire delayed state mid-training."""
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    assert _ensure_runtime(module, Float8CurrentScaling(), revision=1)
    old_views = _runtime_views(module)

    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, DelayedScaling(amax_history_len=4), revision=2)
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
    assert _ensure_runtime(module, active_recipe, revision=1)
    old_views = _runtime_views(module)

    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, replacement, revision=2)
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
        _ensure_runtime(module, recipe, revision=1)
    assert module._quantization_runtime is None  # pylint: disable=protected-access


def test_mixed_custom_recipe_is_frozen_when_it_contains_delayed_state():
    """Even a nominally non-delayed CustomRecipe edit is outside the contract."""
    calls = []

    def active_factory(role):
        calls.append(role)
        return _mixed_delayed_factory(role)

    active_recipe = CustomRecipe(qfactory=active_factory, qfactory_key=("mixed", 1))
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    assert _ensure_runtime(module, active_recipe, revision=1)
    old_views = _runtime_views(module)
    active_call_count = len(calls)

    def replacement_factory(_role):
        raise AssertionError("frozen delayed runtime invoked the replacement factory")

    replacement = CustomRecipe(qfactory=replacement_factory, qfactory_key=("mixed", 2))
    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, replacement, revision=2)

    assert len(calls) == active_call_count
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))


def test_same_delayed_recipe_object_mutation_keeps_committed_snapshot():
    """Rejecting caller mutation must not mutate the active reduction recipe."""
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    recipe = DelayedScaling(amax_history_len=4, margin=0)
    assert _ensure_runtime(module, recipe, revision=1)
    committed_recipe = module.fp8_meta["recipe"]
    assert committed_recipe is not recipe

    recipe.margin = 1
    with pytest.raises(RuntimeError, match="do not support delayed scaling"):
        _ensure_runtime(module, recipe, revision=2)

    assert module.fp8_meta["recipe"] is committed_recipe
    assert committed_recipe.margin == 0


def test_rejected_delayed_update_aborts_autocast_reduction():
    """A caught activation failure cannot update the old delayed tensors on exit."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    module = Linear(
        16,
        16,
        bias=False,
        params_dtype=torch.bfloat16,
        device="cuda",
        name="linear",
    )
    inp = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
    recipe = DelayedScaling(amax_history_len=4, margin=0)
    with autocast(enabled=True, recipe=recipe):
        module(inp)

    state = module.fp8_meta["scaling_fwd"]
    state.scale.fill_(3)
    state.amax_history.fill_(7)
    expected_scale = state.scale.clone()
    expected_history = state.amax_history.clone()

    recipe.margin = 1
    with autocast(enabled=True, recipe=recipe):
        with pytest.raises(RuntimeError, match="do not support delayed scaling"):
            module(inp)

    assert torch.equal(state.scale, expected_scale)
    assert torch.equal(state.amax_history, expected_history)

    """Equal independent recipes and missed A -> B -> A updates reuse the runtime."""
    calls = []
    first_recipe = _make_counting_recipe(("runtime-reuse", 1), calls)
    equal_recipe = _make_counting_recipe(("runtime-reuse", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")

    assert _ensure_runtime(module, first_recipe, revision=1)
    active = module._quantization_runtime  # pylint: disable=protected-access
    assert active is not None
    assert len(calls) == 5

    # Revision 3 models a module that did not run while revision 2 was active.
    assert not _ensure_runtime(module, equal_recipe, revision=3)
    assert module._quantization_runtime is active  # pylint: disable=protected-access
    assert active.recipe_config_revision == 3
    assert len(calls) == 5


def test_unchanged_runtime_uses_revision_hot_path(monkeypatch):
    """The steady-state check does not resolve roles or invoke the factory."""
    calls = []
    recipe = _make_counting_recipe(("runtime-hot-path", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, recipe, revision=1)

    def unexpected_role_resolution(**_kwargs):
        raise AssertionError("unchanged runtime resolved quantizer roles")

    monkeypatch.setattr(module, "get_quantizer_roles", unexpected_role_resolution)
    assert not _ensure_runtime(module, recipe, revision=1)
    assert len(calls) == 5


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
        monkeypatch.setattr(module, "_prepare_quantization_runtime", unexpected_cold_path)
        monkeypatch.setattr(module, "_validate_quantization_runtime", unexpected_cold_path)
        monkeypatch.setattr(module, "_commit_quantization_runtime", unexpected_cold_path)

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
    _ensure_runtime(module, recipe, revision=1)
    old_runtime = module._quantization_runtime  # pylint: disable=protected-access

    recipe.qfactory_key = ("same-object-update", 2)
    assert _ensure_runtime(module, recipe, revision=2)
    assert module._quantization_runtime is not old_runtime  # pylint: disable=protected-access
    assert len(calls) == 10


def test_role_revision_is_requested_until_atomic_runtime_commit():
    """Changing a boundary role does not mutate live state before the next update."""
    calls = []
    recipe = _make_counting_recipe(("role-update", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, recipe, revision=1)

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

    assert _ensure_runtime(module, recipe, revision=1)
    new_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert new_runtime is not old_runtime
    assert new_runtime.key.forward_roles[-1] == role
    assert module.fp8_meta["scaling_fwd"] is new_runtime.forward_states[0]
    assert module.quantizers["scaling_fwd"] is new_runtime.forward_quantizers
    assert len(calls) == 10

    # Reassigning an equal immutable role is a hot-path no-op.
    module.output_quantizer_role = QuantizerRole(
        module_type="linear", tensor_type="input", name="consumer"
    )
    assert module._role_revision == old_role_revision + 1  # pylint: disable=protected-access
    assert not _ensure_runtime(module, recipe, revision=1)
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
    _ensure_runtime(module, active_recipe, revision=1)

    old_runtime = module._quantization_runtime  # pylint: disable=protected-access
    old_recipe = module.fp8_meta["recipe"]
    old_forward_state = module.fp8_meta["scaling_fwd"]
    old_backward_state = module.fp8_meta["scaling_bwd"]
    old_forward_quantizers = module.quantizers["scaling_fwd"]
    old_backward_quantizers = module.quantizers["scaling_bwd"]

    with pytest.raises(RuntimeError, match="backward factory failure"):
        _ensure_runtime(module, failing_recipe, revision=2)

    assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
    assert module.fp8_meta["recipe"] is old_recipe
    assert module.fp8_meta["scaling_fwd"] is old_forward_state
    assert module.fp8_meta["scaling_bwd"] is old_backward_state
    assert module.quantizers["scaling_fwd"] is old_forward_quantizers
    assert module.quantizers["scaling_bwd"] is old_backward_quantizers
    assert old_runtime.recipe_config_revision == 1


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
    assert _ensure_runtime(module, active_recipe, revision=1, num_gemms=num_gemms)

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
    assert not _ensure_runtime(module, equal_recipe, revision=2, num_gemms=num_gemms)
    assert module._quantization_runtime is active_runtime  # pylint: disable=protected-access
    for cache_name, workspace in workspaces.items():
        assert module._fp8_workspaces[cache_name] is workspace  # pylint: disable=protected-access

    # Candidate construction failure preserves both compatibility views and caches.
    with pytest.raises(RuntimeError, match="backward factory failure"):
        _ensure_runtime(module, failing_recipe, revision=3, num_gemms=num_gemms)
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
    assert _ensure_runtime(module, replacement_recipe, revision=4, num_gemms=num_gemms)
    replacement_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert replacement_runtime is not active_runtime
    assert not module._fp8_workspaces  # pylint: disable=protected-access


def test_candidate_validation_failure_keeps_complete_active_runtime(monkeypatch):
    """Validation runs before any candidate state is published."""
    calls = []
    active_recipe = _make_counting_recipe(("validation-update", 1), calls)
    candidate_recipe = _make_counting_recipe(("validation-update", 2), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, active_recipe, revision=1)

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
        _ensure_runtime(module, candidate_recipe, revision=2)

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
    _ensure_runtime(module, active_recipe, revision=1, num_gemms=2)

    old_runtime = module._quantization_runtime  # pylint: disable=protected-access
    old_forward_state = module.fp8_meta["scaling_fwd"]
    old_backward_state = module.fp8_meta["scaling_bwd"]
    old_forward_quantizers = module.quantizers["scaling_fwd"]
    old_backward_quantizers = module.quantizers["scaling_bwd"]
    old_validated_generations = module._validated_quantizer_generations
    old_delayed_quantizer = module._delayed_scaling_input_quantizer
    old_unsafe_quantizer = module._unsafe_requantization_input_quantizer

    invalid_recipe = make_recipe(
        ("grouped-atomic", "invalid", mismatched_tensor_type),
        mismatched_role=mismatched_tensor_type,
    )
    with pytest.raises(ValueError, match="incompatible plain backend configurations"):
        _ensure_runtime(module, invalid_recipe, revision=2, num_gemms=2)

    assert module._quantization_runtime is old_runtime  # pylint: disable=protected-access
    assert module.fp8_meta["scaling_fwd"] is old_forward_state
    assert module.fp8_meta["scaling_bwd"] is old_backward_state
    assert module.quantizers["scaling_fwd"] is old_forward_quantizers
    assert module.quantizers["scaling_bwd"] is old_backward_quantizers
    assert module._validated_quantizer_generations is old_validated_generations
    assert module._delayed_scaling_input_quantizer is old_delayed_quantizer
    assert module._unsafe_requantization_input_quantizer is old_unsafe_quantizer

    replacement_recipe = make_recipe(
        ("grouped-atomic", "replacement", mismatched_tensor_type),
        dtype=torch.float16,
        unsafe_inputs=True,
    )
    assert _ensure_runtime(module, replacement_recipe, revision=3, num_gemms=2)
    replacement_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert replacement_runtime is not old_runtime
    assert (
        module._validated_quantizer_generations["scaling_fwd"]
        is replacement_runtime.forward_quantizers
    )
    assert (
        module._validated_quantizer_generations["scaling_bwd"]
        is replacement_runtime.backward_quantizers
    )
    assert module._delayed_scaling_input_quantizer is None
    assert (
        module._unsafe_requantization_input_quantizer is replacement_runtime.forward_quantizers[0]
    )
