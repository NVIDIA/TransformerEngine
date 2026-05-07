# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import abc

import pytest
import torch

try:
    from torch._opaque_base import OpaqueBaseMeta
    from torch._library.opaque_object import (
        get_opaque_type_name,
        register_opaque_type,
        MemberType,
    )

    _opaque_available = True
except ImportError:
    _opaque_available = False

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe
from transformer_engine.pytorch.constants import FP8FwdTensorIdx, FP8BwdTensorIdx
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
from transformer_engine.pytorch import (
    is_fp8_available,
    is_mxfp8_available,
    is_fp8_block_scaling_available,
    is_nvfp4_available,
)

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_block_scaling_available = is_fp8_block_scaling_available()
nvfp4_available = is_nvfp4_available()

_all_recipes: list = []
if fp8_available:
    _all_recipes.append(recipe.Float8CurrentScaling())
if fp8_block_scaling_available:
    _all_recipes.append(recipe.Float8BlockScaling())
if mxfp8_available:
    _all_recipes.append(recipe.MXFP8BlockScaling())
if nvfp4_available:
    _all_recipes.append(recipe.NVFP4BlockScaling())


# ---------------------------------------------------------------------------
# ToyQuantizer – opaque value-type quantizer for torch.compile
# (requires torch opaque object support, not available in older PyTorch)
# ---------------------------------------------------------------------------

if _opaque_available:

    class _ToyQuantizerMeta(OpaqueBaseMeta, abc.ABCMeta):
        pass

    class ToyQuantizer(Float8CurrentScalingQuantizer, metaclass=_ToyQuantizerMeta):
        """Quantizer with a string tag, registered as an
        opaque value type so torch.compile can treat it as a baked-in constant."""

        def __init__(self, tag: str):
            super().__init__(fp8_dtype=tex.DType.kFloat8E4M3, device=torch.device("cuda"))
            self.tag = tag

        def __eq__(self, other):
            if not isinstance(other, ToyQuantizer):
                return NotImplemented
            return self.tag == other.tag and self.dtype == other.dtype

        def __hash__(self):
            return hash((type(self), self.tag, self.dtype))

        def __fx_repr__(self):
            return (
                f"ToyQuantizer(tag={self.tag!r})",
                {"ToyQuantizer": ToyQuantizer},
            )

    register_opaque_type(
        ToyQuantizer,
        typ="value",
        members={
            "__setattr__": MemberType.USE_REAL,
            "set_usage": MemberType.USE_REAL,
        },
    )

    _Q = get_opaque_type_name(ToyQuantizer)

    def _make_qfactory(tag: str):
        """Return a qfactory that produces ToyQuantizer instances tagged with *tag*."""

        def qfactory(role: str):
            return ToyQuantizer(tag=f"{tag}:{role}")

        return qfactory

    # ---------------------------------------------------------------------------
    # ToyLinear – minimal TE module backed by BasicLinear functional ops
    # ---------------------------------------------------------------------------

    class ToyLinear(TransformerEngineBaseModule):
        """Minimal TE-compatible linear module used for torch.compile tests."""

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

        def _get_weight_tensors(self):
            return [self.weight]

        def _get_weight_quantizers(self):
            if not self.fp8 and not self.fp8_calibration:
                return [None]
            weight_q = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
            weight_q.internal = True
            return [weight_q]

        def forward(self, inp: torch.Tensor) -> torch.Tensor:
            inp = self.prepare_forward(inp, num_gemms=1)
            try:
                input_q = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
                input_q.internal = True
                input_q.optimize_for_gemm = True
                (weight_q,) = self._get_weight_quantizers()
                grad_output_q = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
                grad_output_q.internal = True
                grad_output_q.optimize_for_gemm = True

                return torch.ops.test_te.toy_linear(
                    inp,
                    self.weight,
                    input_q,
                    weight_q,
                    grad_output_q,
                )
            finally:
                self.end_forward()

    # ---------------------------------------------------------------------------
    # Opaque custom ops (torch.library)
    # ---------------------------------------------------------------------------

    _lib = torch.library.Library("test_te", "DEF")

    _lib.define(
        f"toy_linear(Tensor inp, Tensor weight, {_Q} input_q, {_Q} weight_q, {_Q} grad_output_q)"
        " -> Tensor"
    )

    _lib.define(
        "toy_linear_backward(Tensor grad_output, Tensor inp, Tensor weight,"
        f" {_Q} grad_output_q) -> (Tensor, Tensor)"
    )

    last_fwd_quantizers: list[dict[str, "ToyQuantizer"]] = []
    last_bwd_quantizers: list[dict[str, "ToyQuantizer"]] = []

    @torch.library.impl("test_te::toy_linear", "CompositeExplicitAutograd", lib=_lib)
    def _toy_linear_fwd_impl(inp, weight, input_q, weight_q, grad_output_q):
        last_fwd_quantizers.append(
            {
                "input_q": input_q,
                "weight_q": weight_q,
                "grad_output_q": grad_output_q,
            }
        )
        out, _, _ = BasicLinear._functional_forward(
            input=inp,
            weight=weight,
            dtype=inp.dtype,
            input_quantizer=input_q,
            weight_quantizer=weight_q,
        )
        return out

    @torch.library.register_fake("test_te::toy_linear", lib=_lib)
    def _toy_linear_fwd_fake(inp, weight, input_q, weight_q, grad_output_q):
        return inp @ weight.T

    def _toy_linear_setup_context(ctx, inputs, output):
        inp, weight, _input_q, _weight_q, grad_output_q = inputs
        ctx.save_for_backward(inp, weight)
        ctx.grad_output_q = grad_output_q

    @torch.library.impl("test_te::toy_linear_backward", "CompositeExplicitAutograd", lib=_lib)
    def _toy_linear_bwd_impl(grad_output, inp, weight, grad_output_q):
        last_bwd_quantizers.append({"grad_output_q": grad_output_q})
        dx, dw = BasicLinear._functional_backward(
            grad_output=grad_output,
            input=inp,
            weight=weight,
            grad_output_quantizer=grad_output_q,
            grad_input_quantizer=None,
        )
        return dx, dw

    @torch.library.register_fake("test_te::toy_linear_backward", lib=_lib)
    def _toy_linear_bwd_fake(grad_output, inp, weight, grad_output_q):
        return torch.empty_like(inp), torch.empty_like(weight)

    def _toy_linear_backward(ctx, grad_output):
        inp, weight = ctx.saved_tensors
        dx, dw = torch.ops.test_te.toy_linear_backward(
            grad_output,
            inp,
            weight,
            ctx.grad_output_q,
        )
        return dx, dw, None, None, None

    torch.library.register_autograd(
        "test_te::toy_linear",
        _toy_linear_backward,
        setup_context=_toy_linear_setup_context,
        lib=_lib,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_autocast_nested_custom():
    """One ToyLinear model used under nested te.autocast with 3 distinct
    CustomRecipe instances (each producing differently-tagged ToyQuantizers).

    Layout:
        with autocast(recipe0):           # outer
            out = model(inp)
            with autocast(recipe1):       # nested inside outer
                out = model(out)
        with autocast(recipe2):           # separate, after the nested pair
            out = model(out)

    fullgraph=True makes torch.compile raise if any graph break occurs.
    """
    dtype = torch.bfloat16
    device = "cuda"

    model = ToyLinear(32, 32, device=device, dtype=dtype)

    recipe0 = recipe.CustomRecipe(qfactory=_make_qfactory("R0"))
    recipe1 = recipe.CustomRecipe(qfactory=_make_qfactory("R1"))
    recipe2 = recipe.CustomRecipe(qfactory=_make_qfactory("R2"))

    inp = torch.randn(8, 32, dtype=dtype, device=device, requires_grad=True)

    def fn(inp):
        with te.autocast(recipe=recipe0):
            out = model(inp)
            with te.autocast(recipe=recipe1):
                out = model(out)
        with te.autocast(recipe=recipe2):
            out = model(out)
        return out

    torch._dynamo.reset()

    compiled = torch.compile(fn, fullgraph=True)
    last_fwd_quantizers.clear()
    last_bwd_quantizers.clear()

    out = compiled(inp)
    out.sum().backward()

    # Forward: 3 calls — R0, R1, R2
    assert len(last_fwd_quantizers) == 3, f"Expected 3 fwd calls, got {len(last_fwd_quantizers)}"
    for i, tag in enumerate(["R0", "R1", "R2"]):
        fq = last_fwd_quantizers[i]
        assert fq["input_q"].tag.startswith(f"{tag}:"), f"fwd[{i}] input_q: {fq['input_q'].tag}"
        assert fq["weight_q"].tag.startswith(f"{tag}:"), f"fwd[{i}] weight_q: {fq['weight_q'].tag}"
        assert fq["grad_output_q"].tag.startswith(
            f"{tag}:"
        ), f"fwd[{i}] grad_output_q: {fq['grad_output_q'].tag}"

    # Backward: 3 calls — reverse order R2, R1, R0
    assert len(last_bwd_quantizers) == 3, f"Expected 3 bwd calls, got {len(last_bwd_quantizers)}"
    for i, tag in enumerate(["R2", "R1", "R0"]):
        bq = last_bwd_quantizers[i]
        assert bq["grad_output_q"].tag.startswith(
            f"{tag}:"
        ), f"bwd[{i}] grad_output_q: {bq['grad_output_q'].tag}"


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("fp8_recipe", _all_recipes, ids=lambda r: type(r).__name__)
def test_autocast_sanity(fp8_recipe):
    """Smoke test: torch.nn.Linear inside a single te.autocast with each
    built-in recipe. Forward + backward under torch.compile(fullgraph=True)."""
    dtype = torch.bfloat16
    device = "cuda"

    model = torch.nn.Linear(32, 32, dtype=dtype, device=device)
    inp = torch.randn(8, 32, dtype=dtype, device=device, requires_grad=True)

    def fn(inp):
        with te.autocast(recipe=fp8_recipe):
            return model(inp)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    out = compiled(inp)
    out.sum().backward()
