# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Module-local runtime lifecycle tests for mid-training recipe updates."""

import pytest
import torch
import transformer_engine.pytorch.ops as te_ops

from transformer_engine.common.recipe import CustomRecipe, DelayedScaling, Float8CurrentScaling
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
)
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.quantization import (
    DelayedScalingRequest,
    FP8GlobalStateManager,
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
    return module._ensure_quantization_runtime(  # pylint: disable=protected-access
        recipe=recipe,
        recipe_config=recipe.quantizer_config(),
        recipe_config_revision=revision,
        num_gemms=num_gemms,
    )


def _prepare_runtime_update(module, recipe, revision, *, num_gemms=1):
    return module._plan_quantization_update(  # pylint: disable=protected-access
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


def _run_update_step(module, recipe, inp, *args):
    """Run a real forward/backward step and return the output."""
    module.zero_grad(set_to_none=True)
    with autocast(enabled=True, recipe=recipe):
        output = module(inp, *args)
    if isinstance(output, tuple):
        output = output[0]
    output.float().sum().backward()
    assert inp.grad is not None
    assert all(param.grad is not None for param in module.parameters() if param.requires_grad)
    return output


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
    return (
        state.fp8_recipe,
        state.quantizer_config,
        state.quantizer_config_revision,
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


def test_revision_only_update_is_not_published_during_planning():
    """An equal semantic update synchronizes revisions only when committed."""
    calls = []
    recipe = _make_counting_recipe(("revision-plan", 1), calls)
    equal_recipe = _make_counting_recipe(("revision-plan", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, recipe, revision=1)
    active = module._quantization_runtime  # pylint: disable=protected-access
    factory_call_count = len(calls)

    update = _prepare_runtime_update(module, equal_recipe, revision=3)

    assert update.candidate is None
    assert module._quantization_runtime is active  # pylint: disable=protected-access
    assert active.recipe_config_revision == 1
    assert len(calls) == factory_call_count

    assert not module._apply_quantization_update(update)  # pylint: disable=protected-access
    assert active.recipe_config_revision == 3
    assert len(calls) == factory_call_count


def test_candidate_update_is_not_published_during_planning():
    """Candidate state, views, and workspaces change only during commit."""
    calls = []
    active_recipe = _make_counting_recipe(("candidate-plan", 1), calls)
    replacement_recipe = _make_counting_recipe(("candidate-plan", 2), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")
    _ensure_runtime(module, active_recipe, revision=1)
    old_views = _runtime_views(module)
    workspace = object()
    module._fp8_workspaces["weight"] = workspace  # pylint: disable=protected-access

    update = _prepare_runtime_update(module, replacement_recipe, revision=2)

    assert update.candidate is not None
    assert all(current is old for current, old in zip(_runtime_views(module), old_views))
    assert old_views[0].recipe_config_revision == 1
    assert module._fp8_workspaces["weight"] is workspace  # pylint: disable=protected-access

    assert module._apply_quantization_update(update)  # pylint: disable=protected-access
    assert module._quantization_runtime is update.candidate  # pylint: disable=protected-access
    assert not module._fp8_workspaces  # pylint: disable=protected-access


def test_runtime_snapshot_does_not_copy_caller_repr_cache():
    """Caller display history must not affect a runtime snapshot or its checkpoint bytes."""
    recipe = _make_counting_recipe(("runtime-repr-cache", 1), [])
    str(recipe)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")

    update = _prepare_runtime_update(module, recipe, revision=1)

    assert recipe.__dict__["_cached_repr"] is not None
    assert update.candidate.recipe.__dict__["_cached_repr"] is None


def test_uninitialized_runtime_can_be_prepared_without_publication():
    """Planning supports modules that have not yet executed a quantized forward."""
    calls = []
    recipe = _make_counting_recipe(("initial-plan", 1), calls)
    module = Linear(16, 16, bias=False, device="cuda", name="linear")

    update = _prepare_runtime_update(module, recipe, revision=1)

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
    _ensure_runtime(first, active_recipe, revision=1)
    _ensure_runtime(second, active_recipe, revision=1)
    old_first_views = _runtime_views(first)
    old_second_views = _runtime_views(second)

    first_update = _prepare_runtime_update(first, replacement_recipe, revision=2)
    assert first_update.candidate is not None

    def reject_candidate(_candidate):
        raise RuntimeError("later module validation failure")

    monkeypatch.setattr(second, "_validate_quantization_runtime", reject_candidate)
    with pytest.raises(RuntimeError, match="later module validation failure"):
        _prepare_runtime_update(second, replacement_recipe, revision=2)

    assert all(current is old for current, old in zip(_runtime_views(first), old_first_views))
    assert all(current is old for current, old in zip(_runtime_views(second), old_second_views))
    assert old_first_views[0].recipe_config_revision == 1
    assert old_second_views[0].recipe_config_revision == 1


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


@pytest.mark.parametrize(
    ("module_factory", "input_factory", "forward_args"),
    (
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
    ),
)
def test_module_family_executes_after_recipe_update(
    module_factory,
    input_factory,
    forward_args,
):
    """Every base module family executes forward/backward across a runtime replacement."""
    calls = []
    first_recipe = _make_counting_recipe(("module-execution", 1), calls)
    second_recipe = _make_counting_recipe(("module-execution", 2), calls)
    module = module_factory()

    _run_update_step(module, first_recipe, input_factory(), *forward_args)
    first_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert first_runtime is not None

    _run_update_step(module, second_recipe, input_factory(), *forward_args)
    second_runtime = module._quantization_runtime  # pylint: disable=protected-access
    assert second_runtime is not first_runtime
    assert second_runtime.key.recipe_config == second_recipe.quantizer_config()


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
    """A composed module updates every runtime owner reached by real execution."""
    available, reason = is_fp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)

    calls = []
    first_recipe = _make_counting_recipe(("composed-execution", 1), calls)
    second_recipe = _make_counting_recipe(("composed-execution", 2), calls)
    module = module_factory()

    first_inp = torch.randn(8, 2, 32, device="cuda", requires_grad=True)
    _run_update_step(module, first_recipe, first_inp)
    owners = _active_runtime_owners(module)
    assert len(owners) >= 2
    first_runtimes = {
        id(owner): owner._quantization_runtime  # pylint: disable=protected-access
        for owner in owners
    }

    second_inp = torch.randn(8, 2, 32, device="cuda", requires_grad=True)
    _run_update_step(module, second_recipe, second_inp)
    assert _active_runtime_owners(module) == owners
    for owner in owners:
        runtime = owner._quantization_runtime  # pylint: disable=protected-access
        assert runtime is not first_runtimes[id(owner)]
        assert runtime.key.recipe_config == second_recipe.quantizer_config()


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
        first_revision = FP8GlobalStateManager.get_quantizer_config_revision()
        first_factory_calls = len(calls)
        assert first_factory_calls == 10
        assert FP8GlobalStateManager.get_fp8_recipe() is first_recipe

        # An independent equal recipe updates only the manager's requested
        # recipe object. Runtime identity, revision, and factories stay fixed.
        apply_recipe(model, equal_recipe)
        assert [module._quantization_runtime for module in model] == first_runtimes
        assert FP8GlobalStateManager.get_quantizer_config_revision() == first_revision
        assert FP8GlobalStateManager.get_fp8_recipe() is equal_recipe
        assert len(calls) == first_factory_calls

        # The first matching autocast forward after explicit application uses
        # each module's revision fast path and performs no factory work.
        inp = torch.randn(8, 16, device="cuda", requires_grad=True)
        _run_update_step(model, equal_recipe, inp)
        assert [module._quantization_runtime for module in model] == first_runtimes
        assert len(calls) == first_factory_calls

        # Mutating and reusing the same recipe object produces a real model-wide
        # replacement and one new global revision.
        equal_recipe.qfactory_key = ("apply-success", 2)
        apply_recipe(model, equal_recipe)
        assert all(
            module._quantization_runtime is not old_runtime
            for module, old_runtime in zip(model, first_runtimes)
        )
        assert FP8GlobalStateManager.get_quantizer_config_revision() == first_revision + 1
        assert len(calls) == first_factory_calls + 10
    finally:
        FP8GlobalStateManager.reset()


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


def test_apply_recipe_reconstructs_stateless_runtime_after_checkpoint_restore():
    """A restored model can rebuild its committed recipe before its first forward."""
    FP8GlobalStateManager.reset()
    source_calls = []
    source = Linear(16, 16, bias=False, device="cuda", name="linear")
    initial_recipe = _make_counting_recipe(("checkpoint-resume", 1), source_calls)
    committed_recipe = _make_counting_recipe(("checkpoint-resume", 2), source_calls)
    inp = torch.randn(8, 16, device="cuda")

    try:
        apply_recipe(source, initial_recipe)
        apply_recipe(source, committed_recipe)
        checkpoint = source.state_dict()
        with torch.no_grad(), autocast(enabled=True, recipe=committed_recipe):
            expected = source(inp)

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

        with torch.no_grad(), autocast(enabled=True, recipe=resumed_recipe):
            actual = restored(inp)
        torch.testing.assert_close(actual, expected)
    finally:
        FP8GlobalStateManager.reset()


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
def test_apply_recipe_planning_failure_is_model_wide_atomic(failure_index):
    """A factory failure in any participant leaves modules and manager unchanged."""
    FP8GlobalStateManager.reset()
    calls = []
    active_recipe = _make_counting_recipe(("apply-failure", 1), calls)
    modules = [Linear(16, 16, bias=False, device="cuda", name=f"line{index}") for index in range(3)]
    model = torch.nn.Sequential(*modules)

    try:
        apply_recipe(model, active_recipe)
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
        with pytest.raises(
            RuntimeError,
            match=rf"planning module '{failure_index}': model-wide factory failure",
        ):
            apply_recipe(model, replacement_recipe)

        assert _global_recipe_state() == old_global_state
        for module, expected_views, workspace in zip(modules, old_views, workspaces):
            assert all(
                current is expected
                for current, expected in zip(_runtime_views(module), expected_views)
            )
            assert module._fp8_workspaces["weight"] is workspace
    finally:
        FP8GlobalStateManager.reset()


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


def test_apply_recipe_rejects_fusible_owner_before_factory():
    """Excluded fusible owners are discovered before any participant planning."""
    FP8GlobalStateManager.reset()

    def unexpected_factory(_role):
        raise AssertionError("excluded model invoked qfactory")

    recipe = CustomRecipe(
        qfactory=unexpected_factory,
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
    old_global_state = _global_recipe_state()
    try:
        with pytest.raises(RuntimeError, match="only through CustomRecipe"):
            apply_recipe(dpa, Float8CurrentScaling())
        assert dpa._quantization_runtime is None
        assert _global_recipe_state() == old_global_state
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
    update = _prepare_runtime_update(module, replacement_recipe, revision=3, num_gemms=2)
    assert module._validated_quantizer_generations is old_validated_generations
    assert module._delayed_scaling_input_quantizer is old_delayed_quantizer
    assert module._unsafe_requantization_input_quantizer is old_unsafe_quantizer
    assert module._apply_quantization_update(update)  # pylint: disable=protected-access
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
