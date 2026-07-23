# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE.

Frozen (``requires_grad=False``) quantized weights only need their
columnwise (transposed) copy transiently, as the dgrad GEMM operand.
With ``NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE=1`` the copy is released
right after the dgrad GEMM and rebuilt on demand from rowwise data,
saving up to 1 byte/param of resident GPU memory for PEFT-style
fine-tuning of large frozen base models.
"""

from __future__ import annotations

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common import recipe
from transformer_engine.pytorch import (
    GroupedLinear,
    LayerNormLinear,
    LayerNormMLP,
    Linear,
    autocast,
    quantized_model_init,
)
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage

from utils import assert_close, reset_rng_states

fp8_block_scaling_available, reason_for_no_fp8_block_scaling = te.is_fp8_block_scaling_available(
    return_reason=True
)
fp8_available, reason_for_no_fp8 = te.is_fp8_available(return_reason=True)

SEED = 1234
IN_FEATURES = 256
OUT_FEATURES = 256
NUM_GEMMS = 4
TOKENS_PER_GEMM = 64

_MODULE_IDS = ("linear", "grouped_linear", "layernorm_linear", "layernorm_mlp")
_MODULE_CLASSES = (Linear, GroupedLinear, LayerNormLinear, LayerNormMLP)


@pytest.fixture(autouse=True)
def reset_global_state():
    reset_rng_states()
    yield


def _make_block_scaling_recipe(backward_override=None) -> recipe.Float8BlockScaling:
    kwargs = {"fp8_format": recipe.Format.E4M3}
    if backward_override is not None:
        kwargs["backward_override"] = backward_override
    return recipe.Float8BlockScaling(**kwargs)


def _make_module(module_cls, fp8_recipe, frozen: bool):
    """Construct a quantized module, optionally freezing all its weights."""
    ctx = torch.no_grad() if frozen else torch.enable_grad()
    with ctx, quantized_model_init(enabled=True, recipe=fp8_recipe):
        if module_cls is GroupedLinear:
            module = GroupedLinear(
                NUM_GEMMS, IN_FEATURES, OUT_FEATURES, bias=False, params_dtype=torch.bfloat16
            )
        elif module_cls is LayerNormMLP:
            module = LayerNormMLP(
                IN_FEATURES, 4 * IN_FEATURES, bias=False, params_dtype=torch.bfloat16
            )
        else:
            module = module_cls(IN_FEATURES, OUT_FEATURES, bias=False, params_dtype=torch.bfloat16)
    if frozen:
        for param in module.parameters():
            param.requires_grad_(False)
    return module


def _quantized_weights(module):
    return [p for p in module.parameters() if isinstance(p, QuantizedTensorStorage)]


def _run_fwd_bwd(module, fp8_recipe):
    """One fwd+bwd step where dgrad is required."""
    inp = torch.randn(
        NUM_GEMMS * TOKENS_PER_GEMM,
        IN_FEATURES,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    with autocast(enabled=True, recipe=fp8_recipe):
        if isinstance(module, GroupedLinear):
            out = module(inp, [TOKENS_PER_GEMM] * NUM_GEMMS)
        else:
            out = module(inp)
    out.float().pow(2).mean().backward()
    return inp.grad.detach().clone()


def _columnwise_present(weight) -> bool:
    """Whether the tensor currently satisfies columnwise usage."""
    return weight.get_usages()["columnwise"]


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("module_cls", _MODULE_CLASSES, ids=_MODULE_IDS)
def test_frozen_columnwise_released_after_dgrad(module_cls, monkeypatch):
    """Columnwise data of frozen 2D-block-scaled weights is released after backward."""
    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    fp8_recipe = _make_block_scaling_recipe()
    module = _make_module(module_cls, fp8_recipe, frozen=True)
    weights = _quantized_weights(module)
    assert weights, "expected quantized weights"

    for _ in range(3):  # release -> rebuild -> release must be stable
        _run_fwd_bwd(module, fp8_recipe)
        for weight in weights:
            assert weight._columnwise_data is None, "columnwise copy should be released"


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("module_cls", _MODULE_CLASSES, ids=_MODULE_IDS)
def test_frozen_columnwise_kept_when_disabled(module_cls, monkeypatch):
    """Default behavior (env unset/0) keeps the columnwise copy resident."""
    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "0")
    fp8_recipe = _make_block_scaling_recipe()
    module = _make_module(module_cls, fp8_recipe, frozen=True)
    weights = _quantized_weights(module)

    _run_fwd_bwd(module, fp8_recipe)
    for weight in weights:
        assert _columnwise_present(weight), "columnwise copy should be kept by default"


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("module_cls", _MODULE_CLASSES, ids=_MODULE_IDS)
def test_frozen_columnwise_release_preserves_numerics(module_cls, monkeypatch):
    """dgrad is identical with and without the release flag."""
    fp8_recipe = _make_block_scaling_recipe()

    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "0")
    reset_rng_states()
    module_ref = _make_module(module_cls, fp8_recipe, frozen=True)
    dgrad_ref = [_run_fwd_bwd(module_ref, fp8_recipe) for _ in range(2)]

    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    reset_rng_states()
    module_rel = _make_module(module_cls, fp8_recipe, frozen=True)
    dgrad_rel = [_run_fwd_bwd(module_rel, fp8_recipe) for _ in range(2)]

    for ref, rel in zip(dgrad_ref, dgrad_rel):
        assert_close(rel, ref, rtol=0, atol=0)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_grouped_linear_release_with_backward_override(monkeypatch):
    """Release also works when dgrad runs on dequantized weight copies."""
    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    fp8_recipe = _make_block_scaling_recipe(backward_override="dequantized")
    module = _make_module(GroupedLinear, fp8_recipe, frozen=True)
    weights = _quantized_weights(module)

    _run_fwd_bwd(module, fp8_recipe)
    for weight in weights:
        assert not _columnwise_present(weight), (
            "columnwise copy of the original quantized weight should be released"
            " even when dgrad uses dequantized copies"
        )


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("module_cls", (Linear, GroupedLinear), ids=("linear", "grouped_linear"))
def test_trainable_weights_unaffected(module_cls, monkeypatch):
    """Trainable weights keep their columnwise copy and receive wgrad."""
    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    fp8_recipe = _make_block_scaling_recipe()
    module = _make_module(module_cls, fp8_recipe, frozen=False)
    assert all(p.requires_grad for p in module.parameters())

    _run_fwd_bwd(module, fp8_recipe)
    for param in module.parameters():
        assert param.grad is not None or getattr(param, "main_grad", None) is not None
    for weight in _quantized_weights(module):
        assert _columnwise_present(weight), "trainable columnwise copy must not be released"


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_columnwise_only_tensor_skipped(monkeypatch):
    """Columnwise-only tensors (FSDP2 backward all-gather shape) are skipped, not crashed.

    Upstream FSDP2 + Float8BlockScaling + quantized_model_init is currently
    broken (scale-inv padding in all-gather slice ops), so the guard is
    exercised here at unit level with a hand-built columnwise-only tensor.
    """
    from transformer_engine.pytorch.module.base import release_frozen_weight_columnwise

    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    fp8_recipe = _make_block_scaling_recipe()
    module = _make_module(Linear, fp8_recipe, frozen=True)
    (weight,) = _quantized_weights(module)

    # Mimic the FSDP2 backward all-gather product: columnwise-only.
    # (Materialize columnwise from rowwise first, then drop rowwise —
    # the columnwise-only update_usage path requires the copy to exist.)
    weight.update_usage(rowwise_usage=True, columnwise_usage=True)
    weight.update_usage(rowwise_usage=False, columnwise_usage=True)
    assert weight._rowwise_data is None
    assert _columnwise_present(weight)

    release_frozen_weight_columnwise((weight,))  # must be a safe no-op
    assert _columnwise_present(weight), "columnwise-only tensor must not be touched"


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_non_blockwise_never_receives_release_call(monkeypatch):
    """Spy on Float8TensorStorage.update_usage: no columnwise release is ever requested.

    Architecture-independent proof (works even where releasing would be a
    physical no-op, e.g. non-TN-capable GEMM hardware).
    """
    from transformer_engine.pytorch.tensor.storage.float8_tensor_storage import (
        Float8TensorStorage,
    )

    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    calls = []
    original = Float8TensorStorage.update_usage

    def spy(self, *args, **kwargs):
        calls.append((args, kwargs))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Float8TensorStorage, "update_usage", spy)

    fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.E4M3)
    module = _make_module(Linear, fp8_recipe, frozen=True)
    _run_fwd_bwd(module, fp8_recipe)

    for args, kwargs in calls:
        columnwise = kwargs.get("columnwise_usage", args[1] if len(args) > 1 else None)
        assert columnwise is not False, "non-blockwise tensor received a columnwise release"


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
@pytest.mark.parametrize("frozen", (True, False), ids=("frozen", "trainable"))
def test_workspace_path_with_microbatch_cache(frozen, monkeypatch):
    """bf16 Parameters + FP8 workspaces (no quantized_model_init), with microbatch cache.

    Trainable workspaces must keep their columnwise copy; frozen workspaces
    are released and rebuilt with numerics identical to flag-off.
    """
    fp8_recipe = _make_block_scaling_recipe()

    def build_and_run(flag: str):
        monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", flag)
        reset_rng_states()
        module = Linear(IN_FEATURES, OUT_FEATURES, bias=False, params_dtype=torch.bfloat16)
        if frozen:
            for param in module.parameters():
                param.requires_grad_(False)
        grads = []
        for step in range(3):
            inp = torch.randn(
                NUM_GEMMS * TOKENS_PER_GEMM,
                IN_FEATURES,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            with autocast(enabled=True, recipe=fp8_recipe):
                out = module(inp, is_first_microbatch=(step == 0))
            out.float().pow(2).mean().backward()
            grads.append(inp.grad.detach().clone())
        workspaces = [
            ws for ws in module._fp8_workspaces.values() if isinstance(ws, QuantizedTensorStorage)
        ]
        return grads, workspaces

    grads_on, workspaces_on = build_and_run("1")
    assert workspaces_on, "expected FP8 weight workspaces"
    for workspace in workspaces_on:
        if frozen:
            assert not _columnwise_present(workspace), "frozen workspace should be released"
        else:
            assert _columnwise_present(workspace), "trainable workspace must be kept"

    grads_off, _ = build_and_run("0")
    for ref, rel in zip(grads_off, grads_on):
        assert_close(rel, ref, rtol=0, atol=0)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_fp8_activation_recompute(monkeypatch):
    """te.checkpoint (activation recompute) + release flag: stable and numerics match."""
    from transformer_engine.pytorch import checkpoint

    fp8_recipe = _make_block_scaling_recipe()

    def run(flag: str):
        monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", flag)
        reset_rng_states()
        module = _make_module(Linear, fp8_recipe, frozen=True)
        grads = []
        for _ in range(2):
            inp = torch.randn(
                NUM_GEMMS * TOKENS_PER_GEMM,
                IN_FEATURES,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            with autocast(enabled=True, recipe=fp8_recipe):
                out = checkpoint(module, inp)
            out.float().pow(2).mean().backward()
            grads.append(inp.grad.detach().clone())
        return grads, _quantized_weights(module)

    grads_on, weights_on = run("1")
    for weight in weights_on:
        assert not _columnwise_present(weight), "frozen weight should be released after recompute"
    grads_off, _ = run("0")
    for ref, rel in zip(grads_off, grads_on):
        assert_close(rel, ref, rtol=0, atol=0)


def _eager_reference(inp_data, flag: str, monkeypatch):
    """Same-seed frozen module, one eager fwd+bwd; returns (output, dgrad)."""
    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", flag)
    fp8_recipe = _make_block_scaling_recipe()
    reset_rng_states()
    module = _make_module(Linear, fp8_recipe, frozen=True)
    inp = inp_data.clone().requires_grad_(True)
    with autocast(enabled=True, recipe=fp8_recipe):
        out = module(inp)
    out.float().pow(2).mean().backward()
    return out.detach().clone(), inp.grad.detach().clone()


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_cuda_graph_make_graphed_callables(monkeypatch):
    """make_graphed_callables: guard keeps columnwise, numerics match eager flag-off."""
    from transformer_engine.pytorch import make_graphed_callables

    torch.manual_seed(SEED)
    inp_data = torch.randn(
        NUM_GEMMS * TOKENS_PER_GEMM, IN_FEATURES, device="cuda", dtype=torch.bfloat16
    )
    ref_out, ref_dgrad = _eager_reference(inp_data, "0", monkeypatch)

    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    fp8_recipe = _make_block_scaling_recipe()
    reset_rng_states()
    module = _make_module(Linear, fp8_recipe, frozen=True)
    weights = _quantized_weights(module)

    sample = inp_data.clone().requires_grad_(True)
    graphed = make_graphed_callables(
        module, (sample,), num_warmup_iters=3, enabled=True, recipe=fp8_recipe
    )
    for _ in range(3):  # capture + replays must be stable
        inp = inp_data.clone().requires_grad_(True)
        out = graphed(inp)
        out.float().pow(2).mean().backward()

    # Guard is the only thing standing between capture-time dgrad and the
    # release: columnwise must still be present after capture + replays
    # (replay executes no Python, so the state must be stable).
    for weight in weights:
        assert _columnwise_present(weight), "columnwise must survive graph capture/replay"
    assert_close(out.detach(), ref_out, rtol=0, atol=0)
    assert_close(inp.grad, ref_dgrad, rtol=0, atol=0)


@pytest.mark.skipif(not fp8_block_scaling_available, reason=reason_for_no_fp8_block_scaling)
def test_cuda_graph_direct_capture(monkeypatch):
    """Direct torch.cuda.graph capture/replay: fixed grad buffer updated, numerics match."""
    torch.manual_seed(SEED)
    inp_data = torch.randn(
        NUM_GEMMS * TOKENS_PER_GEMM, IN_FEATURES, device="cuda", dtype=torch.bfloat16
    )
    ref_out, ref_dgrad = _eager_reference(inp_data, "0", monkeypatch)

    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    fp8_recipe = _make_block_scaling_recipe()
    reset_rng_states()
    module = _make_module(Linear, fp8_recipe, frozen=True)
    weights = _quantized_weights(module)

    static_inp = inp_data.clone().requires_grad_(True)
    # Warmup on a side stream (per torch.cuda.graph docs). Do NOT reset
    # .grad to None afterwards: replay writes into the captured buffers,
    # so the grad buffer address must stay fixed to observe the update.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            with autocast(enabled=True, recipe=fp8_recipe):
                out = module(static_inp)
            out.float().pow(2).mean().backward()
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with autocast(enabled=True, recipe=fp8_recipe):
            static_out = module(static_inp)
        static_loss = static_out.float().pow(2).mean()
        static_loss.backward()

    for _ in range(3):  # replays must be stable and re-write the fixed buffer
        static_inp.grad.zero_()  # in place: keep the captured address
        graph.replay()
        torch.cuda.synchronize()
        assert not torch.all(static_inp.grad == 0), "replay must update the fixed grad buffer"

    for weight in weights:
        assert _columnwise_present(weight), "columnwise must survive graph capture/replay"
    assert_close(static_out.detach(), ref_out, rtol=0, atol=0)
    assert_close(static_inp.grad.clone(), ref_dgrad, rtol=0, atol=0)


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_non_2d_scaled_weights_not_released(monkeypatch):
    """Delayed-scaling weights (not rebuildable from rowwise) keep columnwise usage."""
    monkeypatch.setenv("NVTE_RELEASE_FROZEN_WEIGHT_COLUMNWISE", "1")
    fp8_recipe = recipe.DelayedScaling(fp8_format=recipe.Format.E4M3)
    module = _make_module(Linear, fp8_recipe, frozen=True)
    weights = _quantized_weights(module)

    for _ in range(2):  # would fail on step 2 if the transpose had been dropped
        _run_fwd_bwd(module, fp8_recipe)
    for weight in weights:
        assert not getattr(weight, "_is_2D_scaled", False)
        assert _columnwise_present(weight), "columnwise usage must not be released"
