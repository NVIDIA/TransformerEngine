# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear


# ---------------------------------------------------------------------------
# ToyLinear – minimal TE module backed by BasicLinear functional ops
# ---------------------------------------------------------------------------

# Global list of ToyLinear instances.  Each module registers itself here on
# construction; the custom op identifies which module to use via an integer
# index so that no Python object ever enters the compiled graph.
_toy_modules: list["ToyLinear"] = []


class ToyLinear(TransformerEngineBaseModule):
    """Minimal TE-compatible linear module used for torch.compile tests.

    Inherits TransformerEngineBaseModule so that prepare_forward / end_forward
    and the FP8 metadata machinery work exactly as in production modules.
    The actual compute is delegated to BasicLinear._functional_forward /
    _functional_backward via the opaque custom op below.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device=device)
        )
        torch.nn.init.normal_(self.weight)

        # Register in the global list and remember the index.
        self._module_idx = len(_toy_modules)
        _toy_modules.append(self)

    # -- required abstract overrides -----------------------------------------

    def _get_weight_tensors(self):
        return [self.weight]

    def _get_weight_quantizers(self):
        # Weight quantizer: use FP8 scaling when FP8 is enabled.
        if not self.fp8 and not self.fp8_calibration:
            return [None]
        weight_q = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_WEIGHT]
        weight_q.internal = True
        return [weight_q]

    # -- quantizer helpers (mirrors what Linear._get_quantizers does) ---------

    def get_forward_quantizers(self):
        """Return (input_q, weight_q) for the forward GEMM."""
        if not self.fp8:
            return None, None
        input_q = self.quantizers["scaling_fwd"][tex.FP8FwdTensors.GEMM1_INPUT]
        input_q.internal = True
        input_q.optimize_for_gemm = True
        (weight_q,) = self._get_weight_quantizers()
        return input_q, weight_q

    def get_backward_quantizers(self):
        """Return (grad_output_q, grad_input_q) for the backward GEMMs."""
        if not self.fp8:
            return None, None
        grad_output_q = self.quantizers["scaling_bwd"][tex.FP8BwdTensors.GRAD_OUTPUT1]
        grad_output_q.internal = True
        grad_output_q.optimize_for_gemm = True
        return grad_output_q, None

    # -- forward -------------------------------------------------------------

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        inp = self.prepare_forward(inp, num_gemms=1)
        try:
            return _toy_linear_fwd_op(inp, self.weight, self._module_idx)
        finally:
            self.end_forward()


# ---------------------------------------------------------------------------
# Opaque custom op  (torch.library)
# ---------------------------------------------------------------------------


@torch.library.custom_op("test_te::toy_linear", mutates_args=[])
def _toy_linear_fwd_op(inp: torch.Tensor, weight: torch.Tensor, module_idx: int) -> torch.Tensor:
    """Forward GEMM wrapped as an opaque custom op."""
    module = _toy_modules[module_idx]
    input_q, weight_q = module.get_forward_quantizers()
    out, _, _ = BasicLinear._functional_forward(
        input=inp,
        weight=weight,
        dtype=inp.dtype,
        input_quantizer=input_q,
        weight_quantizer=weight_q,
    )
    return out


@_toy_linear_fwd_op.register_fake
def _(inp: torch.Tensor, weight: torch.Tensor, module_idx: int) -> torch.Tensor:
    """Abstract implementation for shape inference under torch.compile."""
    return inp @ weight.T


def _toy_linear_setup_context(ctx, inputs, output):
    inp, weight, module_idx = inputs
    ctx.save_for_backward(inp, weight)
    ctx.module_idx = module_idx


@torch.library.custom_op("test_te::toy_linear_backward", mutates_args=[])
def _toy_linear_bwd_op(
    grad_output: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor, module_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque backward op wrapping BasicLinear._functional_backward."""
    module = _toy_modules[module_idx]
    grad_output_q, grad_input_q = module.get_backward_quantizers()
    dx, dw = BasicLinear._functional_backward(
        grad_output=grad_output,
        input=inp,
        weight=weight,
        grad_output_quantizer=grad_output_q,
        grad_input_quantizer=grad_input_q,
    )
    return dx, dw


@_toy_linear_bwd_op.register_fake
def _(grad_output: torch.Tensor, inp: torch.Tensor, weight: torch.Tensor, module_idx: int):
    """Abstract backward implementation for shape inference under torch.compile."""
    return torch.empty_like(inp), torch.empty_like(weight)


def _toy_linear_backward(ctx, grad_output: torch.Tensor):
    """Backward: dispatch to opaque custom op so TE backward is not traced."""
    inp, weight = ctx.saved_tensors
    dx, dw = _toy_linear_bwd_op(grad_output, inp, weight, ctx.module_idx)
    return dx, dw, None  # None for module_idx gradient


torch.library.register_autograd(
    "test_te::toy_linear",
    _toy_linear_backward,
    setup_context=_toy_linear_setup_context,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

_fp8_available = te.is_fp8_available()
_mxfp8_available = te.is_mxfp8_available()
_fp8_block_scaling_available = te.is_fp8_block_scaling_available()

# Each entry is (fp8_recipe, fp8_enabled).
# For the "no_fp8" variant, enabled=False so autocast is a no-op, but we still
# pass a real pre-created recipe object so that get_default_fp8_recipe() is
# never called during compilation (which would assert-fail inside torch.compile).
_recipes = [
    pytest.param(recipe.DelayedScaling(), False, id="no_fp8"),
    pytest.param(
        recipe.Float8CurrentScaling(),
        True,
        id="float8_current_scaling",
        marks=pytest.mark.skipif(not _fp8_available, reason="FP8 not supported"),
    ),
    pytest.param(
        recipe.MXFP8BlockScaling(),
        True,
        id="mxfp8_block_scaling",
        marks=pytest.mark.skipif(not _mxfp8_available, reason="MXFP8 not supported"),
    ),
    pytest.param(
        recipe.Float8BlockScaling(),
        True,
        id="float8_block_scaling",
        marks=pytest.mark.skipif(
            not _fp8_block_scaling_available, reason="FP8 block scaling not supported"
        ),
    ),
]


@pytest.mark.parametrize("fp8_recipe,fp8_enabled", _recipes)
def test_autocast(fp8_recipe, fp8_enabled):
    """Test that ToyLinear inside te.autocast compiles without graph breaks.

    fullgraph=True makes torch.compile raise an error if any graph break occurs.
    Parametrized over all supported recipes.  The no_fp8 variant uses
    enabled=False so the autocast is a no-op, but still passes a real
    pre-created recipe object to avoid calling get_default_fp8_recipe() during
    compilation.
    """
    global _toy_modules
    _toy_modules = []

    dtype = torch.bfloat16
    device = "cuda"

    model = ToyLinear(32, 64, device=device, dtype=dtype)
    inp = torch.randn(8, 32, dtype=dtype, device=device, requires_grad=True)

    def fn(inp):
        with te.autocast(recipe=fp8_recipe, enabled=fp8_enabled):
            return model(inp)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    out = compiled(inp)
    out.sum().backward()


@pytest.mark.skipif(not _fp8_available, reason="FP8 not supported")
def test_autocast_delayed_scaling_unsupported():
    """DelayedScaling should fail with a clear error under torch.compile."""
    global _toy_modules
    _toy_modules = []

    dtype = torch.bfloat16
    device = "cuda"

    model = ToyLinear(32, 64, device=device, dtype=dtype)
    inp = torch.randn(8, 32, dtype=dtype, device=device, requires_grad=True)
    fp8_recipe = recipe.DelayedScaling()

    def fn(inp):
        with te.autocast(recipe=fp8_recipe, enabled=True):
            return model(inp)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    with pytest.raises(RuntimeError, match="DelayedScaling is not supported under torch.compile"):
        compiled(inp)


@pytest.mark.skipif(not te.is_fp8_available(), reason="FP8 not supported on this GPU")
def test_autocast_nested():
    """Test sequential model with different FP8 recipes and nested te.autocast.

    Layout:
        with autocast(Float8CurrentScaling):       # outer
            out = m0(inp)
            with autocast(Float8CurrentScaling):   # nested inside outer
                out = m1(out)
        with autocast(Float8CurrentScaling):       # separate, after the nested pair
            out = m2(out)

    fullgraph=True makes torch.compile raise an error if any graph break occurs.
    """
    global _toy_modules
    _toy_modules = []

    dtype = torch.bfloat16
    device = "cuda"

    m0 = ToyLinear(32, 32, device=device, dtype=dtype)
    m1 = ToyLinear(32, 32, device=device, dtype=dtype)
    m2 = ToyLinear(32, 32, device=device, dtype=dtype)

    # Use distinct recipe objects so nested/separate autocast contexts use
    # different identities under torch.compile.
    recipe_current0 = recipe.Float8CurrentScaling()
    recipe_current1 = recipe.Float8CurrentScaling()
    recipe_current2 = recipe.Float8CurrentScaling()

    inp = torch.randn(8, 32, dtype=dtype, device=device, requires_grad=True)

    def fn(inp):
        with te.autocast(recipe=recipe_current0):  # outer
            out = m0(inp)
            with te.autocast(recipe=recipe_current1):  # nested inside outer
                out = m1(out)

        with te.autocast(recipe=recipe_current2):  # separate, after nested pair
            out = m2(out)
        return out

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    out = compiled(inp)
    out.sum().backward()
