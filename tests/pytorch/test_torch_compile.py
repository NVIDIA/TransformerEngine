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
from transformer_engine.pytorch.quantization import QuantizerRole
from transformer_engine.pytorch import (
    is_fp8_available,
    is_mxfp8_available,
    is_fp8_block_scaling_available,
    is_nvfp4_available,
)
from utils import recipe_id

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_block_scaling_available = is_fp8_block_scaling_available()
nvfp4_available = is_nvfp4_available()


def nvfp4_row_scaled():
    nvfp4_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        disable_2d_quantization=True,
        row_scaled_activation=True,
        backward_override="dequantized",
    )
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams()
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
    return nvfp4_recipe


def nvfp4_4over6():
    nvfp4_recipe = recipe.NVFP4BlockScaling(
        disable_rht=True,
        disable_stochastic_rounding=True,
        nvfp4_4over6="all",
    )
    nvfp4_recipe.fp4_quant_fwd_inp = recipe.QParams()
    nvfp4_recipe.fp4_quant_fwd_weight = recipe.QParams(fp4_2d_quantization=True)
    nvfp4_recipe.fp4_quant_bwd_grad = recipe.QParams()
    return nvfp4_recipe


_all_recipes: list = []
if fp8_available:
    _all_recipes.append(recipe.Float8CurrentScaling())
if fp8_block_scaling_available:
    _all_recipes.append(recipe.Float8BlockScaling())
if mxfp8_available:
    _all_recipes.append(recipe.MXFP8BlockScaling())
if nvfp4_available:
    _all_recipes.append(recipe.NVFP4BlockScaling())
    _all_recipes.append(nvfp4_4over6())
    _all_recipes.append(nvfp4_row_scaled())


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
            super().__init__(fp8_dtype=te.DType.kFloat8E4M3, device=torch.device("cuda"))
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
        """Return a qfactory that produces ToyQuantizer instances tagged with *tag*.

        The factory dispatches on ``QuantizerRole.tensor_type``; the roles are
        supplied by :meth:`ToyLinear.get_quantizer_roles`.
        """

        quantizers = {
            tensor_type: ToyQuantizer(tag=f"{tag}:{tensor_type}")
            for tensor_type in (
                "input",
                "weight",
                "output",
                "grad_output",
                "grad_input",
            )
        }

        def qfactory(role: QuantizerRole):
            return quantizers[role.tensor_type]

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

        def get_quantizer_roles(self, *, fwd: bool, num_quantizers: int):
            # Supplying explicit roles keeps CustomRecipeState from emitting a
            # warning (which would graph-break under fullgraph=True) and lets the
            # qfactory dispatch per tensor slot. Order must match the module's
            # quantizer array (FP8FwdTensorIdx / FP8BwdTensorIdx).
            if fwd:
                return [
                    QuantizerRole(module_type="linear", tensor_type="input"),
                    QuantizerRole(module_type="linear", tensor_type="weight"),
                    QuantizerRole(module_type="linear", tensor_type="output"),
                ]
            return [
                QuantizerRole(module_type="linear", tensor_type="grad_output"),
                QuantizerRole(module_type="linear", tensor_type="grad_input"),
            ]

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
@pytest.mark.parametrize("fp8_recipe", _all_recipes, ids=recipe_id)
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


# ---------------------------------------------------------------------------
# get_attention_backend under torch.compile
# ---------------------------------------------------------------------------


def test_get_attention_backend_traceable(monkeypatch):
    """get_attention_backend must trace under torch.compile(fullgraph=True),
    i.e. without any graph break, and NVTE_* env var reads must be guarded so
    that changing an env var triggers recompilation instead of silently
    reusing a stale backend selection."""
    from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils

    attention_params = dpa_utils.AttentionParams()

    def fn(x):
        (
            use_flash_attention,
            _,
            use_fused_attention,
            _,
            use_unfused_attention,
            _,
        ) = dpa_utils.get_attention_backend(attention_params)
        return (
            x
            + (1 if use_flash_attention else 0)
            + (2 if use_fused_attention else 0)
            + (4 if use_unfused_attention else 0)
        )

    # Dynamo only guards os.environ entries that exist at trace time (reads
    # of absent keys are not guarded yet), so set the vars explicitly.
    for env_var in ("NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"):
        monkeypatch.setenv(env_var, "1")

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    x = torch.zeros(8, device="cuda")
    torch.testing.assert_close(compiled(x), fn(x))

    # Flip env vars one by one: the compiled function must recompile (guards
    # on os.environ) and keep matching eager.
    for env_var in ("NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN", "NVTE_FLASH_ATTN"):
        monkeypatch.setenv(env_var, "0")
        torch.testing.assert_close(compiled(x), fn(x))
        monkeypatch.setenv(env_var, "1")


def test_get_attention_backend_fused_backend_constant(monkeypatch):
    """The tex.get_fused_attn_backend call inside get_attention_backend is
    wrapped with assume_constant_result. Check that under torch.compile it is
    (1) invoked at trace time only (not on every run), (2) still consulted
    (with correct results) when attention params change and dynamo turns the
    changed ints into symbolic scalars via automatic dynamic, and (3) its
    return value actually drives the backend selection (verified by
    monkeypatching it to report no available sub-backend)."""
    from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils

    # Restrict candidates to FusedAttention vs UnfusedDotProductAttention so
    # the selection outcome encodes the tex.get_fused_attn_backend result.
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "1")

    def fn(x, params):
        (
            use_flash_attention,
            _,
            use_fused_attention,
            _,
            use_unfused_attention,
            _,
        ) = dpa_utils.get_attention_backend(params)
        return (
            x
            + (1 if use_flash_attention else 0)
            + (2 if use_fused_attention else 0)
            + (4 if use_unfused_attention else 0)
        )

    x = torch.zeros(8, device="cuda")
    params = dpa_utils.AttentionParams()
    if fn(x, params)[0].item() != 2.0:
        pytest.skip("FusedAttention not available for the default attention params")

    calls = []
    real_get_backend = tex.get_fused_attn_backend

    def counting_get_backend(*args):
        calls.append(args)
        return real_get_backend(*args)

    monkeypatch.setattr(dpa_utils.tex, "get_fused_attn_backend", counting_get_backend)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    out = compiled(x, params)
    torch.testing.assert_close(out, x + 2.0)
    calls_after_trace = len(calls)
    assert calls_after_trace >= 1, "tex.get_fused_attn_backend not invoked during tracing"

    # Baked as a constant: running the compiled function again must not call it.
    compiled(x, params)
    assert len(calls) == calls_after_trace

    # Changing attention params: the changed ints (head_dim, seqlens) become
    # symbolic scalars on recompilation (dynamo's automatic dynamic).
    # assume_constant_result cannot convert symbolic scalars to constants and
    # graph breaks on them, so this phase compiles without fullgraph and only
    # requires that the selection stays correct (tex consulted again, eagerly).
    compiled_dyn = torch.compile(fn)
    for changed_params in (
        dpa_utils.AttentionParams(head_dim_qk=128, head_dim_v=128),
        dpa_utils.AttentionParams(max_seqlen_q=512, max_seqlen_kv=512),
        dpa_utils.AttentionParams(qkv_layout="bshd_bshd_bshd"),
        dpa_utils.AttentionParams(qkv_dtype=torch.float16),
    ):
        num_calls = len(calls)
        out = compiled_dyn(x, changed_params)
        assert len(calls) > num_calls, f"tex not consulted for {changed_params}"
        torch.testing.assert_close(out, fn(x, changed_params))

    # The baked result must drive the selection: report no fused sub-backend
    # and expect UnfusedDotProductAttention instead of FusedAttention. Use a
    # fresh frame: already-compiled frames keep the previously baked constant
    # (assume_constant_result installs no guard on the wrapped function).
    monkeypatch.setattr(
        dpa_utils.tex,
        "get_fused_attn_backend",
        lambda *args: dpa_utils.FusedAttnBackend["No_Backend"],
    )

    def fn_no_backend(x, params):
        return fn(x, params)

    compiled_no_backend = torch.compile(fn_no_backend, fullgraph=True)
    torch.testing.assert_close(compiled_no_backend(x, params), x + 4.0)


def test_get_attention_backend_traceable_fp8(monkeypatch):
    """Backend selection for FP8 attention (fp8_dpa recipes) must also trace
    under torch.compile(fullgraph=True), including the FP8-only env var reads
    and recipe filters, and the FP8 env vars must be guarded."""
    from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils

    for env_var, value in (
        ("NVTE_FLASH_ATTN", "1"),
        ("NVTE_FUSED_ATTN", "1"),
        ("NVTE_UNFUSED_ATTN", "1"),
        ("NVTE_FP8_DPA_BWD", "1"),
        ("NVTE_DPA_FP8CS_O_in_F16", "1"),
        ("NVTE_DPA_FP8_RECIPE", ""),
        ("NVTE_UnfusedDPA_Emulate_FP8", "0"),
    ):
        monkeypatch.setenv(env_var, value)

    attention_params = dpa_utils.AttentionParams(
        fp8=True,
        fp8_meta={"recipe": recipe.DelayedScaling(fp8_dpa=True)},
    )

    def fn(x):
        (
            use_flash_attention,
            _,
            use_fused_attention,
            _,
            use_unfused_attention,
            _,
        ) = dpa_utils.get_attention_backend(attention_params)
        return (
            x
            + (1 if use_flash_attention else 0)
            + (2 if use_fused_attention else 0)
            + (4 if use_unfused_attention else 0)
        )

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    x = torch.zeros(8, device="cuda")
    torch.testing.assert_close(compiled(x), fn(x))

    # FP8-only env var: allowing FP8 emulation enables UnfusedDotProductAttention,
    # which must trigger recompilation (guard on os.environ) and match eager.
    monkeypatch.setenv("NVTE_UnfusedDPA_Emulate_FP8", "1")
    torch.testing.assert_close(compiled(x), fn(x))
