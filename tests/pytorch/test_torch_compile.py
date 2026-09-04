# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import abc
import contextlib
import os
import re
import sys
import warnings

import pytest
import torch

try:
    from torch._dynamo.utils import counters
except ImportError:  # pragma: no cover
    counters = None
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

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
from transformer_engine.pytorch.quantization import FP8GlobalStateManager, QuantizerRole
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor.float8_blockwise_tensor import Float8BlockQuantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, Quantizer
from transformer_engine.pytorch.dynamo import TensorSpec, to_tensor_spec
from transformer_engine.pytorch import (
    is_fp8_available,
    is_mxfp8_available,
    is_fp8_block_scaling_available,
    is_nvfp4_available,
)

# Import from the local utils.py by explicit path: importing cutedsl makes a
# top-level ``utils`` package visible that would shadow it.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import ModelConfig, dtype_tols, get_available_attention_backends, recipe_id

sys.path.pop(0)
from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    UnfusedDotProductAttention,
)

fp8_available, reason_for_no_fp8 = is_fp8_available(return_reason=True)
mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
fp8_block_scaling_available, reason_for_no_fp8_block_scaling = is_fp8_block_scaling_available(
    return_reason=True
)
nvfp4_available, reason_for_no_nvfp4 = is_nvfp4_available(return_reason=True)


@pytest.fixture(autouse=True)
def _reset_fp8_global_state():
    """Pending FP8 global state (e.g. delayed-scaling amax reductions) must not
    leak between tests: a leftover buffer makes a later autocast __exit__ call
    raw tex bindings, which graph-breaks fullgraph=True tests."""
    yield
    FP8GlobalStateManager.reset()


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


# Modes exercised by the te.Linear tests; "reduce-overhead" = CUDA-graph trees.
_compile_modes = ["default", "reduce-overhead"]


def _cudagraph_warmup(fn, inp, *, backward: bool) -> None:
    """One eager iteration so lazily-initialized TE state (fp8 meta, workspaces)
    is allocated before any CUDA-graph capture."""
    out = fn(inp)
    if backward:
        out.sum().backward()


def _dynamo_counter(group: str, key: str):
    """Read a torch._dynamo counter; None (with a warning) if the private
    counters API is gone, so CI degrades instead of failing."""
    try:
        return counters[group][key]
    except Exception:  # pylint: disable=broad-except
        warnings.warn(f"torch._dynamo.utils.counters[{group!r}][{key!r}] unavailable")
        return None


@contextlib.contextmanager
def _assert_no_cudagraph_skips(enabled: bool):
    """Assert reduce-overhead really captured CUDA graphs: inductor may skip
    capture and silently fall back to eager, which ``fullgraph=True`` does not
    catch. No-op when ``enabled`` is False."""
    before = _dynamo_counter("inductor", "cudagraph_skips")
    yield
    if enabled and before is not None:
        skipped = _dynamo_counter("inductor", "cudagraph_skips") - before
        assert skipped == 0, (
            f"reduce-overhead fell back to eager: {skipped} cudagraph skip(s); "
            "see the 'skipping cudagraphs due to ...' log for the reason"
        )


# All compute runs inside the op and the loss grad is ones, so bit-exact.
_EAGER_ATOL, _EAGER_RTOL = 0.0, 0.0


def _assert_close_eager_compiled(fn, compiled, model, base):
    """Run ``fn`` eagerly and ``compiled`` on identical inputs; assert the
    forward output and the input / weight / bias gradients match."""
    inp_eager = base.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    out_eager = fn(inp_eager)
    out_eager.sum().backward()
    ref_out = out_eager.detach().clone()
    ref_wgrad = model.weight.grad.detach().clone()
    ref_igrad = inp_eager.grad.detach().clone()
    # bias=False keeps a 0-element ``bias`` around; grad is None then.
    ref_bgrad = model.bias.grad.detach().clone() if model.bias.grad is not None else None

    inp_compiled = base.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    # Clone before a later cuda-graph replay overwrites the static output buffer.
    out_compiled = compiled(inp_compiled).clone()
    out_compiled.sum().backward()

    torch.testing.assert_close(out_compiled, ref_out, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)
    torch.testing.assert_close(inp_compiled.grad, ref_igrad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)
    torch.testing.assert_close(model.weight.grad, ref_wgrad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)
    if ref_bgrad is not None:
        torch.testing.assert_close(model.bias.grad, ref_bgrad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)


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


_UNFUSED_DPA_CONFIG = dict(
    batch_size=2,
    num_heads=4,
    head_dim=64,
    max_seqlen_q=128,
    max_seqlen_kv=128,
)


def _make_unfused_attention(dtype: torch.dtype) -> UnfusedDotProductAttention:
    cfg = _UNFUSED_DPA_CONFIG
    softmax_scale = cfg["head_dim"] ** -0.5
    module = UnfusedDotProductAttention(
        softmax_scale=softmax_scale,
        attention_type="self",
        attention_dropout=0.0,
        layer_number=1,
        softmax_type="vanilla",
        return_max_logit=False,
    )
    return module.to(dtype=dtype, device="cuda")


_EMPTY_ALIBI_CACHE = {
    "_num_heads": None,
    "_alibi_slopes": None,
    "_max_seqlen_q": None,
    "_max_seqlen_kv": None,
    "_bottom_right_alignment": True,
    "_alibi_bias": None,
    "_alibi_slopes_require_update": False,
    "_alibi_bias_require_update": False,
}


def _make_unfused_qkv(qkv_layout: str, dtype: torch.dtype, requires_grad: bool = True):
    """Build (q, k, v) tensors matching `qkv_layout`. Returns also the
    extra kwargs (`cu_seqlens_*`, `max_seqlen_*`) that the unfused module
    needs for `thd` layouts (empty dict otherwise)."""
    cfg = _UNFUSED_DPA_CONFIG
    b, s_q, s_kv = cfg["batch_size"], cfg["max_seqlen_q"], cfg["max_seqlen_kv"]
    h, d = cfg["num_heads"], cfg["head_dim"]
    qkv_format = "".join(c for c in qkv_layout.split("_")[0] if c.isalpha())

    extra: dict = {}

    def _separate(shape):
        return tuple(
            torch.randn(shape, dtype=dtype, device="cuda", requires_grad=requires_grad)
            for _ in range(3)
        )

    if qkv_layout == "bshd_bshd_bshd":
        q, k, v = _separate((b, s_q, h, d))
    elif qkv_layout == "sbhd_sbhd_sbhd":
        q, k, v = _separate((s_q, b, h, d))
    elif qkv_layout == "thd_thd_thd":
        # All sequences in the batch have the maximum length; no padding.
        cu = torch.arange(0, (b + 1) * s_q, step=s_q, dtype=torch.int32, device="cuda")
        q, k, v = _separate((b * s_q, h, d))
        extra = dict(
            cu_seqlens_q=cu,
            cu_seqlens_kv=cu,
            max_seqlen_q=s_q,
            max_seqlen_kv=s_kv,
        )
    elif qkv_layout == "bs3hd":
        # Packed: shape (b, s, 3, h, d), q/k/v are views along dim=-3.
        qkv = torch.randn(
            (b, s_q, 3, h, d),
            dtype=dtype,
            device="cuda",
            requires_grad=requires_grad,
        )
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        # q/k/v are non-leaf views; retain their grads so the assertions in
        # the test (`q.grad is not None` etc.) work for packed layouts.
        if requires_grad:
            for t in (q, k, v):
                t.retain_grad()
    elif qkv_layout == "sbh3d":
        # Packed: shape (s, b, h, 3, d), q/k/v are views along dim=-2.
        qkv = torch.randn(
            (s_q, b, h, 3, d),
            dtype=dtype,
            device="cuda",
            requires_grad=requires_grad,
        )
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        if requires_grad:
            for t in (q, k, v):
                t.retain_grad()
    else:
        raise ValueError(f"Unsupported qkv_layout in test: {qkv_layout}")

    return q, k, v, extra, qkv_format


def _call_unfused(
    module: UnfusedDotProductAttention,
    qkv_layout: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    extra: dict,
) -> torch.Tensor:
    return module(
        _EMPTY_ALIBI_CACHE,
        q,
        k,
        v,
        qkv_layout=qkv_layout,
        attn_mask_type="causal",
        **extra,
    )


@pytest.mark.parametrize(
    "qkv_layout",
    [
        "bshd_bshd_bshd",
        "sbhd_sbhd_sbhd",
        "thd_thd_thd",
        "bs3hd",
        "sbh3d",
    ],
)
def test_unfused_dpa_torch_compile(qkv_layout):
    """Compile UnfusedDotProductAttention.forward with
    `torch.compile(fullgraph=True, mode="reduce-overhead")` for several
    qkv layouts.

    - `fullgraph=True` makes the test fail on any graph break inside the
      unfused attention path.
    - `mode="reduce-overhead"` uses the inductor cudagraphs backend, so
      forward+backward are captured into CUDA graphs and replayed on
      subsequent iterations."""
    dtype = torch.bfloat16

    module = _make_unfused_attention(dtype)

    def fn(q, k, v, extra):
        return _call_unfused(module, qkv_layout, q, k, v, extra)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, mode="reduce-overhead")

    for _ in range(3):
        # Compared against eager rather than only checked for finiteness: a
        # replay reusing stale buffers would pass the latter.
        q, k, v, extra, _ = _make_unfused_qkv(qkv_layout, dtype, requires_grad=True)
        inputs = (q, k, v, extra)
        replayed = _run_and_capture(compiled, inputs, {}, [q, k, v])
        eager = _run_and_capture(fn, inputs, {}, [q, k, v])
        _assert_run_matches(replayed, eager, "unfused", dtype)


# ---------------------------------------------------------------------------
# DotProductAttention under torch.compile
# ---------------------------------------------------------------------------


# Model configurations, described with the same ModelConfig the eager attention
# tests use. Which backends can run each of them is not hardcoded here --
# get_available_attention_backends() answers that, so configurations only one
# backend supports (arbitrary masks, biases, MLA head dims, ...) are covered
# rather than avoided.
def _cfg(
    model_config,
    qkv_format="bshd",
    packed=None,
    interleave_dim=-3,
    share_cu_seqlens=False,
    kv_cache=False,
):
    """A model configuration plus what DotProductAttention is handed it as.

    `packed` is "qkv" or "kv" for the declarative packed inputs, interleaved at
    `interleave_dim`, or None for separate q/k/v. `kv_cache` runs the call as a
    decoding step against an InferenceParams KV cache.
    """
    return dict(
        model_config=model_config,
        qkv_format=qkv_format,
        packed=packed,
        interleave_dim=interleave_dim,
        share_cu_seqlens=share_cu_seqlens,
        kv_cache=kv_cache,
    )


def _packed_layout(qkv_format: str, packed_dim: int, interleave_dim: int) -> str:
    """bshd + 3 @ -3 -> bs3hd; bshd + 2 @ -2 -> bsh2d; as DotProductAttention
    derives it from the declaration."""
    position = len(qkv_format) + interleave_dim + 1
    return qkv_format[:position] + str(packed_dim) + qkv_format[position:]


_DPA_COMPILE_CONFIGS = {
    "self_bshd_causal": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="causal")),
    "self_sbhd_no_mask": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="no_mask"), "sbhd"),
    "self_bshd_swa": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="causal", window_size=(16, 0))),
    "gqa_bshd_causal": _cfg(ModelConfig(2, 128, 8, 64, num_gqa_groups=2, attn_mask_type="causal")),
    "cross_bshd_no_mask": _cfg(
        ModelConfig(2, 128, 4, 64, max_seqlen_kv=256, attn_mask_type="no_mask")
    ),
    "self_bshd_padding_causal": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="padding_causal")),
    "self_thd_padding_causal": _cfg(
        ModelConfig(2, 128, 4, 64, attn_mask_type="padding_causal"), "thd"
    ),
    # Self attention naturally passes one cu_seqlens tensor for both q and kv,
    # which flash-attn hands to two inputs of the same autograd.Function.
    "self_thd_shared_cu_seqlens": _cfg(
        ModelConfig(2, 128, 4, 64, attn_mask_type="padding_causal"), "thd", share_cu_seqlens=True
    ),
    "packed_qkv_bs3hd": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="causal"), packed="qkv"),
    "packed_qkv_bsh3d": _cfg(
        ModelConfig(2, 128, 4, 64, attn_mask_type="causal"), packed="qkv", interleave_dim=-2
    ),
    "packed_kv_bshd_bs2hd": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="causal"), packed="kv"),
    "packed_kv_thd_th2d": _cfg(
        ModelConfig(2, 128, 4, 64, attn_mask_type="padding_causal"),
        "thd",
        packed="kv",
        interleave_dim=-2,
    ),
    # Configurations below are supported by one backend only, or take a code
    # path of their own inside DotProductAttention.
    "alibi_bshd_causal": _cfg(
        ModelConfig(2, 128, 4, 64, attn_mask_type="causal", attn_bias_type="alibi")
    ),
    "post_scale_bias_bshd": _cfg(
        ModelConfig(2, 128, 4, 64, attn_bias_type="post_scale_bias", bias_shape="1hss")
    ),
    "arbitrary_mask_bshd": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="arbitrary")),
    "mla_bshd_causal": _cfg(ModelConfig(2, 128, 4, 128, head_dim_v=64, attn_mask_type="causal")),
    "sink_softmax_bshd": _cfg(
        ModelConfig(2, 128, 4, 64, attn_mask_type="causal", softmax_type="off-by-one")
    ),
    # FlashAttention takes a fused sbhd->bshd split for sbh3d with a head
    # dimension of 128 and at least 512 tokens, and a plain transpose otherwise.
    "packed_qkv_sbh3d_fused_split": _cfg(
        ModelConfig(4, 128, 4, 128, attn_mask_type="no_mask"),
        "sbhd",
        packed="qkv",
        interleave_dim=-2,
    ),
    # A decoding step against a KV cache. FlashAttention 2 wants the non-paged
    # cache length divisible by 256.
    "kv_cache_bshd": _cfg(
        ModelConfig(2, 8, 4, 64, max_seqlen_kv=256, attn_mask_type="padding_causal_bottom_right"),
        kv_cache=True,
    ),
}


def _qkv_layout(spec: dict) -> str:
    qkv_format, packed = spec["qkv_format"], spec["packed"]
    if packed is None:
        return "_".join([qkv_format] * 3)
    if packed == "qkv":
        return _packed_layout(qkv_format, 3, spec["interleave_dim"])
    return f"{qkv_format}_{_packed_layout(qkv_format, 2, spec['interleave_dim'])}"


def _make_dpa(spec: dict, dtype: torch.dtype) -> te.DotProductAttention:
    config = spec["model_config"]
    if spec["kv_cache"]:
        # KV caching addresses the cache by layer number, and a decoding step
        # runs in inference mode.
        return (
            te.DotProductAttention(
                num_attention_heads=config.num_heads,
                kv_channels=config.kv_channels,
                num_gqa_groups=config.num_gqa_groups,
                qkv_format=spec["qkv_format"],
                attn_mask_type=config.attn_mask_type,
                layer_number=1,
            )
            .to(dtype=dtype, device="cuda")
            .eval()
        )
    return te.DotProductAttention(
        num_attention_heads=config.num_heads,
        kv_channels=config.kv_channels,
        num_gqa_groups=config.num_gqa_groups,
        attention_dropout=config.dropout_p,
        qkv_format=spec["qkv_format"],
        attn_mask_type=config.attn_mask_type,
        window_size=config.window_size,
        attention_type=config.attn_type,
        softmax_type=config.softmax_type,
    ).to(dtype=dtype, device="cuda")


def _cu_seqlens(seqlens: torch.Tensor) -> torch.Tensor:
    cu = torch.zeros(seqlens.numel() + 1, dtype=torch.int32, device="cuda")
    cu[1:] = torch.cumsum(seqlens, dim=0)
    return cu


def _make_dpa_inputs(spec: dict, dtype: torch.dtype):
    """Build the (args, kwargs) that `DotProductAttention.forward` is called
    with, plus the list of tensors whose gradients the test compares."""
    config = spec["model_config"]
    qkv_format, packed = spec["qkv_format"], spec["packed"]
    b = config.batch_size
    s_q, s_kv = config.max_seqlen_q, config.max_seqlen_kv
    h, g = config.num_heads, config.num_gqa_groups
    d_qk, d_v = config.head_dim_qk, config.head_dim_v
    padded = "padding" in config.attn_mask_type

    kwargs = {}
    if padded:
        # Sequences shorter than the maximum, so the padding mask is not
        # degenerate: with all sequences full it would be all-False, and any
        # difference in how masking is compiled would be invisible.
        seqlens_q = torch.randint(1, s_q, [b], dtype=torch.int32, device="cuda")
        seqlens_kv = (
            seqlens_q
            if config.attn_type == "self"
            else (torch.randint(1, s_kv, [b], dtype=torch.int32, device="cuda"))
        )
        cu_q = _cu_seqlens(seqlens_q)
        cu_kv = cu_q if spec["share_cu_seqlens"] else _cu_seqlens(seqlens_kv)
        kwargs.update(cu_seqlens_q=cu_q, cu_seqlens_kv=cu_kv, max_seqlen_q=s_q, max_seqlen_kv=s_kv)
        t_q, t_kv = int(cu_q[-1]), int(cu_kv[-1])
    else:
        t_q, t_kv = b * s_q, b * s_kv

    def _shape(s, t, heads, head_dim):
        return {
            "bshd": (b, s, heads, head_dim),
            "sbhd": (s, b, heads, head_dim),
            "thd": (t, heads, head_dim),
        }[qkv_format]

    def _randn(shape):
        return torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

    if spec["kv_cache"]:
        from collections import OrderedDict
        from transformer_engine.pytorch.attention import InferenceParams

        inference_params = InferenceParams(
            max_batch_size=b,
            max_sequence_length=s_kv,
            num_heads_kv=g,
            head_dim_k=d_qk,
            dtype=dtype,
            qkv_format=qkv_format,
        )
        inference_params.allocate_memory(1)
        inference_params.pre_step(OrderedDict((i, s_q) for i in range(b)))
        qkv = [
            torch.randn(_shape(s_q, t_q, heads, d_qk), dtype=dtype, device="cuda")
            for heads in (h, g, g)
        ]
        # The sequence lengths come from the cache, not from cu_seqlens, and a
        # decoding step has no gradients to compare.
        return tuple(qkv), {"inference_params": inference_params}, []

    if packed is not None:
        # Declarative packed inputs: q/k/v are derived from one buffer by DPA
        # itself, and the layout comes from the declaration -- the only packed
        # layout that torch.compile supports.
        assert h == g and d_qk == d_v, "packed inputs require uniform heads and head dims"
        interleave_dim = spec["interleave_dim"]

        def _packed(seqlen, tokens, packed_dim):
            leading = {
                "bshd": (b, seqlen),
                "sbhd": (seqlen, b),
                "thd": (tokens,),
            }[qkv_format]
            trailing = (packed_dim, h, d_qk) if interleave_dim == -3 else (h, packed_dim, d_qk)
            return _randn(leading + trailing)

        kwargs["qkv_interleave_dim"] = interleave_dim
        if packed == "qkv":
            qkv = _packed(s_q, t_q, 3)
            kwargs["qkv_layer"] = qkv
            grad_tensors, args = [qkv], ()
        else:
            q = _randn(_shape(s_q, t_q, h, d_qk))
            kv = _packed(s_kv, t_kv, 2)
            kwargs["kv_layer"] = kv
            grad_tensors, args = [q, kv], (q,)
    else:
        q = _randn(_shape(s_q, t_q, h, d_qk))
        k = _randn(_shape(s_kv, t_kv, g, d_qk))
        v = _randn(_shape(s_kv, t_kv, g, d_v))
        grad_tensors, args = [q, k, v], (q, k, v)

    if config.attn_mask_type == "arbitrary":
        kwargs["attention_mask"] = torch.zeros(b, 1, s_q, s_kv, dtype=torch.bool, device="cuda")
    if config.attn_bias_type != "no_bias":
        kwargs["core_attention_bias_type"] = config.attn_bias_type
    if config.attn_bias_type == "post_scale_bias":
        kwargs["core_attention_bias"] = torch.randn(1, h, s_q, s_kv, dtype=dtype, device="cuda")

    return args, kwargs, grad_tensors


def _skip_unsupported(
    spec: dict, backend: str, dtype, compiled: bool = True, inference_params=None
) -> None:
    """Skip what the backend under test cannot run, or -- for a test that
    compiles it -- cannot be compiled."""
    if compiled and backend == "fused":
        # FusedAttention's forward carries @no_torch_dynamo, so there is nothing
        # to compile: it runs as an eager island. Drop this skip once it traces,
        # and the tests below cover it as they do the others.
        pytest.skip("FusedAttention is an eager island and does not compile")
    available, _, _ = get_available_attention_backends(
        spec["model_config"],
        dtype,
        _qkv_layout(spec),
        inference_params=inference_params,
        is_training=inference_params is None,
    )
    flash_supported, fused_supported, unfused_supported = available
    supported = {
        "flash": flash_supported,
        "fused": fused_supported,
        "unfused": unfused_supported,
    }[backend]
    if not supported:
        pytest.skip(f"the {backend} backend does not support this configuration")


def _force_dpa_backend(monkeypatch, backend: str) -> None:
    """Restrict DotProductAttention to a single backend."""
    from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
        _attention_backends,
    )
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        FlashAttentionUtils,
    )

    if backend == "flash" and not FlashAttentionUtils.is_installed:
        pytest.skip("flash-attn is not installed")

    for name, var in (
        ("flash", "NVTE_FLASH_ATTN"),
        ("fused", "NVTE_FUSED_ATTN"),
        ("unfused", "NVTE_UNFUSED_ATTN"),
    ):
        monkeypatch.setenv(var, "1" if name == backend else "0")
    # Backend selection is cached on the attention params only, so the env vars
    # above are not enough to invalidate it.
    _attention_backends["backend_selection_requires_update"] = True


def _assert_dpa_backend(backend: str) -> None:
    from transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention import (
        _attention_backends,
    )

    assert _attention_backends[f"use_{backend}_attention"], (
        f"expected the {backend} backend to run, selected:"
        f" flash={_attention_backends['use_flash_attention']},"
        f" fused={_attention_backends['use_fused_attention']},"
        f" unfused={_attention_backends['use_unfused_attention']}"
    )


def _assert_matches_eager(actual, expected, backend: str, dtype: torch.dtype) -> None:
    """Assert a compiled result matches the eager one.

    A backend that calls the same kernel either way has to match exactly:
    compiling changes what surrounds it, not its arithmetic. The unfused
    backend is instead built from PyTorch ops, which inductor fuses and
    reassociates; that error scales with the softmax sums rather than with each
    output element, hence an absolute tolerance taken from the tensor's scale.
    """
    if backend != "unfused":
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        return
    tols = dtype_tols(dtype)
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tols["rtol"],
        atol=max(tols["atol"], 1e-3 * expected.abs().max().item()),
    )


def _run_and_capture(fn, args, kwargs, grads):
    """Run `fn` and backward through its output, returning the output and a copy
    of every input gradient.

    The output is cloned because CUDA graphs hand back tensors owned by their
    memory pool, which the next replay overwrites, and the gradients are cleared
    so that the same inputs can be reused for the other run.
    """
    out = fn(*args, **kwargs).clone()
    if grads:
        out.sum().backward()
    torch.cuda.synchronize()
    captured = []
    for tensor in grads:
        assert tensor.grad is not None
        captured.append(tensor.grad.clone())
        tensor.grad = None
    return out, captured


def _assert_run_matches(actual, expected, backend: str, dtype: torch.dtype) -> None:
    """Compare an (output, gradients) pair produced by `_run_and_capture`."""
    (out, out_grads), (ref, ref_grads) = actual, expected
    _assert_matches_eager(out, ref, backend, dtype)
    for out_grad, ref_grad in zip(out_grads, ref_grads):
        _assert_matches_eager(out_grad, ref_grad, backend, dtype)


def _compare_compiled_to_eager(
    module, args, kwargs, grads, monkeypatch, backend: str, dtype: torch.dtype, **compile_kwargs
) -> None:
    """Run the module eagerly and compiled on the same inputs, and compare both
    the output and every input gradient."""
    eager = _run_and_capture(module, args, kwargs, grads)

    torch._dynamo.reset()
    # Force backend selection to be re-run (and traced) inside the compiled
    # region instead of being served from the cache the eager call populated.
    _force_dpa_backend(monkeypatch, backend)
    compiled = _run_and_capture(torch.compile(module, **compile_kwargs), args, kwargs, grads)

    _assert_dpa_backend(backend)
    _assert_run_matches(compiled, eager, backend, dtype)


@pytest.mark.parametrize("backend", ["flash", "fused", "unfused"])
@pytest.mark.parametrize("config", _DPA_COMPILE_CONFIGS.keys())
def test_dpa_torch_compile(monkeypatch, backend, config):
    """`DotProductAttention` under `torch.compile(fullgraph=True)` must match
    eager in forward and backward, for every backend that supports the
    configuration.

    `fullgraph=True` makes the test fail on any graph break, so it covers the
    whole module: input unpacking, qkv layout, backend selection and the backend
    itself.
    """
    dtype = torch.bfloat16
    spec = _DPA_COMPILE_CONFIGS[config]
    module = _make_dpa(spec, dtype)
    args, kwargs, grads = _make_dpa_inputs(spec, dtype)
    _skip_unsupported(spec, backend, dtype, inference_params=kwargs.get("inference_params"))
    _force_dpa_backend(monkeypatch, backend)

    _compare_compiled_to_eager(
        module, args, kwargs, grads, monkeypatch, backend, dtype, fullgraph=True
    )


def test_dpa_torch_compile_around_fused(monkeypatch):
    """FusedAttention itself is an eager island, but everything around it is
    compiled: DotProductAttention traces up to the backend call, breaks the
    graph there and resumes afterwards. What crosses that break has to survive
    it -- the sub-backend enum did not, and reached cuDNN as the function that
    produced it."""
    dtype = torch.bfloat16
    spec = _DPA_COMPILE_CONFIGS["self_bshd_causal"]
    _skip_unsupported(spec, "fused", dtype, compiled=False)
    _force_dpa_backend(monkeypatch, "fused")

    module = _make_dpa(spec, dtype)
    args, kwargs, grads = _make_dpa_inputs(spec, dtype)
    # No fullgraph: the graph break at the eager island is the point here.
    _compare_compiled_to_eager(module, args, kwargs, grads, monkeypatch, "fused", dtype)


@pytest.mark.parametrize("backend", ["flash", "unfused"])
@pytest.mark.parametrize("config", ["self_bshd_causal", "kv_cache_bshd"])
def test_dpa_torch_compile_cudagraphs(monkeypatch, backend, config):
    """`mode="reduce-overhead"`: forward and backward of DotProductAttention
    are captured into CUDA graphs and replayed on subsequent iterations."""
    dtype = torch.bfloat16
    spec = _DPA_COMPILE_CONFIGS[config]
    _force_dpa_backend(monkeypatch, backend)

    module = _make_dpa(spec, dtype)

    torch._dynamo.reset()
    counters.clear()
    compiled = torch.compile(module, fullgraph=True, mode="reduce-overhead")

    for _ in range(3):
        # Fresh inputs every iteration: a replay that reuses stale buffers would
        # still produce finite values and non-None gradients, so only comparing
        # against eager catches it.
        args, kwargs, grads = _make_dpa_inputs(spec, dtype)
        replayed = _run_and_capture(compiled, args, kwargs, grads)
        eager = _run_and_capture(module, args, kwargs, grads)
        _assert_run_matches(replayed, eager, backend, dtype)
    _assert_dpa_backend(backend)
    # Without this, inductor declining to capture -- a mutated input, a CPU
    # scalar -- would leave the test passing while measuring nothing.
    assert not counters["inductor"]["cudagraph_skips"], "inductor skipped CUDA graphs"


@pytest.mark.parametrize("backend", ["flash", "unfused"])
@pytest.mark.parametrize("paged", [False, True], ids=["non_paged", "paged"])
@pytest.mark.parametrize("cuda_graphs", [False, True], ids=["default", "cudagraphs"])
def test_dpa_torch_compile_kv_cache_decoding(monkeypatch, backend, paged, cuda_graphs):
    """Generation against a KV cache, one prefill and three single-token steps.

    The cache carries state from one step to the next, so a step that updates it
    wrongly is only visible in the step after -- hence comparing every step, and
    a cache of its own for each of the two runs.
    """
    from collections import OrderedDict
    from transformer_engine.pytorch.attention import InferenceParams

    dtype = torch.bfloat16
    spec = _DPA_COMPILE_CONFIGS["kv_cache_bshd"]
    config = spec["model_config"]
    b, ctx_len = config.batch_size, config.max_seqlen_q
    h, g, d = config.num_heads, config.num_gqa_groups, config.head_dim_qk

    def make_cache():
        kwargs = dict(
            max_batch_size=b,
            max_sequence_length=config.max_seqlen_kv,
            num_heads_kv=g,
            head_dim_k=d,
            dtype=dtype,
            qkv_format=spec["qkv_format"],
        )
        if paged:
            # One page per sequence, and FlashAttention 2 wants it divisible by 256.
            kwargs.update(is_paged=True, page_size=config.max_seqlen_kv, total_num_pages=b)
        inference_params = InferenceParams(**kwargs)
        inference_params.allocate_memory(1)
        return inference_params

    _skip_unsupported(spec, backend, dtype, inference_params=make_cache())

    gen = torch.Generator(device="cuda").manual_seed(1234)
    steps = [
        [
            torch.randn(b, seqlen, heads, d, device="cuda", dtype=dtype, generator=gen)
            for heads in (h, g, g)
        ]
        for seqlen in (ctx_len, 1, 1, 1)
    ]

    _force_dpa_backend(monkeypatch, backend)
    module = _make_dpa(spec, dtype)

    def generate(fn):
        inference_params = make_cache()
        outputs = []
        with torch.no_grad():
            for args in steps:
                inference_params.pre_step(OrderedDict((i, args[0].shape[1]) for i in range(b)))
                _force_dpa_backend(monkeypatch, backend)
                # Cloned because a CUDA graph replay overwrites what it returned.
                outputs.append(fn(*args, inference_params=inference_params).clone())
        return outputs

    eager = generate(module)

    torch._dynamo.reset()
    counters.clear()
    compile_kwargs = {"mode": "reduce-overhead"} if cuda_graphs else {}
    compiled = generate(torch.compile(module, fullgraph=True, **compile_kwargs))

    _assert_dpa_backend(backend)
    for out, ref in zip(compiled, eager):
        _assert_matches_eager(out, ref, backend, dtype)
    if cuda_graphs:
        assert not counters["inductor"]["cudagraph_skips"], "inductor skipped CUDA graphs"


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("fp8_attention", [False, True], ids=["fp8_gemms_only", "fp8_attention"])
def test_dpa_torch_compile_fp8(monkeypatch, fp8_attention):
    """FP8 attention is not supported on the compiled path and falls back to
    eager. FP8 elsewhere in the model with attention in high precision -- the
    common training setup -- must stay compiled."""
    dtype = torch.bfloat16
    spec = _DPA_COMPILE_CONFIGS["self_bshd_causal"]
    _force_dpa_backend(monkeypatch, "unfused")

    module = _make_dpa(spec, dtype)
    args, kwargs, _ = _make_dpa_inputs(spec, dtype)
    fp8_recipe = recipe.DelayedScaling(fp8_dpa=fp8_attention)

    def fn(*args, **kwargs):
        with te.autocast(enabled=True, recipe=fp8_recipe):
            return module(*args, **kwargs)

    torch._dynamo.reset()
    _force_dpa_backend(monkeypatch, "unfused")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            torch.compile(fn)(*args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            # FP8 attention is not available on every device; what matters here
            # is which path was taken, which the warning below reports.
            pass
        fell_back = any("Falling back to eager" in str(w.message) for w in caught)

    assert fell_back == fp8_attention


def _packed_views_inputs(spec, dtype, interleave_dim=-3):
    """Packed q/k/v handed over as plain views, without declaring the packing."""
    config = spec["model_config"]
    b, s = config.batch_size, config.max_seqlen_q
    h, d = config.num_heads, config.head_dim_qk
    shape = (b, s, 3, h, d) if interleave_dim == -3 else (b, s, h, 3, d)
    qkv = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    q, k, v = [qkv.select(interleave_dim, i) for i in range(3)]
    return (q, k, v), {}, [qkv]


def _thd_without_max_seqlen_inputs(spec, dtype):
    """thd inputs that leave max_seqlen to be derived from cu_seqlens."""
    args, kwargs, grads = _make_dpa_inputs(spec, dtype)
    del kwargs["max_seqlen_q"], kwargs["max_seqlen_kv"]
    return args, kwargs, grads


_EAGER_FALLBACK_CASES = {
    # name: (config name, input builder)
    "packed_views_bs3hd": (
        "self_bshd_causal",
        lambda spec, dtype: _packed_views_inputs(spec, dtype, -3),
    ),
    "packed_views_bsh3d": (
        "self_bshd_causal",
        lambda spec, dtype: _packed_views_inputs(spec, dtype, -2),
    ),
    "thd_without_max_seqlen": ("self_thd_padding_causal", _thd_without_max_seqlen_inputs),
}


@pytest.mark.parametrize("backend", ["flash", "unfused"])
@pytest.mark.parametrize("case", _EAGER_FALLBACK_CASES.keys())
def test_dpa_torch_compile_eager_fallback(monkeypatch, backend, case):
    """Calls that cannot be traced run as an eager island instead, with a
    warning: recognizing packed q/k/v takes the data pointers dynamo cannot
    read, and deriving max_seqlen off cu_seqlens is a device synchronization.
    Both keep working, and both are avoidable -- by declaring the packing via
    qkv_layer/kv_layer, or by passing max_seqlen."""
    dtype = torch.bfloat16
    config_name, make_inputs = _EAGER_FALLBACK_CASES[case]
    spec = _DPA_COMPILE_CONFIGS[config_name]
    _force_dpa_backend(monkeypatch, backend)

    module = _make_dpa(spec, dtype)
    args, kwargs, grads = make_inputs(spec, dtype)

    eager = _run_and_capture(module, args, kwargs, grads)

    torch._dynamo.reset()
    _force_dpa_backend(monkeypatch, backend)
    with pytest.warns(UserWarning, match="Falling back to eager execution"):
        fell_back = _run_and_capture(torch.compile(module), args, kwargs, grads)
    _assert_run_matches(fell_back, eager, backend, dtype)

    # The same eager island is an error when the user asked for a full graph.
    torch._dynamo.reset()
    _force_dpa_backend(monkeypatch, backend)
    with pytest.raises(Exception, match="torch.compiler.disable"):
        torch.compile(module, fullgraph=True)(*args, **kwargs)


# ---------------------------------------------------------------------------
# get_attention_backend under torch.compile
# ---------------------------------------------------------------------------


# Scalars in AttentionParams must stay concrete: assume_constant_result cannot
# convert symbolic scalars (dynamo's automatic dynamic would make changed ints
# symbolic on recompilation), so pin them static explicitly.
@torch._dynamo.config.patch(specialize_int=True, specialize_float=True, recompile_limit=32)
def test_get_attention_backend_traceable(monkeypatch):
    """get_attention_backend must trace under torch.compile(fullgraph=True)
    without graph breaks. The compiled selection must stay consistent with
    eager when NVTE_* env vars flip (dynamo guards on os.environ) and when
    attention params change, and the baked tex.get_fused_attn_backend result
    must drive the selection."""
    from transformer_engine.pytorch.attention.dot_product_attention import utils as dpa_utils

    def fn(x, params):
        (
            use_flash_attention,
            _,
            use_fused_attention,
            fused_attention_backend,
            use_unfused_attention,
            _,
        ) = dpa_utils.get_attention_backend(params)
        # Encode the full selection (enabled backends + fused sub-backend) in
        # the tensor value: without a tensor op dynamo skips the frame entirely
        # (nothing gets compiled or guarded), and the output makes compiled vs
        # eager selection directly comparable.
        return (
            x
            + (1 if use_flash_attention else 0)
            + (2 if use_fused_attention else 0)
            + (4 if use_unfused_attention else 0)
            + (8 * int(fused_attention_backend) if fused_attention_backend is not None else 0)
        )

    # Dynamo only guards os.environ entries that exist at trace time (reads of
    # absent keys are not guarded yet), so set the vars explicitly.
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

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)
    x = torch.zeros(8, device="cuda")
    params = dpa_utils.AttentionParams()

    torch.testing.assert_close(compiled(x, params), fn(x, params))

    # Flip env vars one by one: the compiled function must recompile (guards
    # on os.environ) and keep matching eager.
    for env_var in ("NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN", "NVTE_FLASH_ATTN"):
        monkeypatch.setenv(env_var, "0")
        torch.testing.assert_close(compiled(x, params), fn(x, params))
        monkeypatch.setenv(env_var, "1")

    # FP8 attention (fp8_dpa recipe): covers the FP8-only branch (run_config
    # env reads, recipe filters, get_fp8_te_dtype). Flipping an FP8-only env
    # var (emulation enables UnfusedDotProductAttention) must recompile too.
    fp8_params = dpa_utils.AttentionParams(
        fp8=True, fp8_meta={"recipe": recipe.DelayedScaling(fp8_dpa=True)}
    )
    torch.testing.assert_close(compiled(x, fp8_params), fn(x, fp8_params))
    monkeypatch.setenv("NVTE_UnfusedDPA_Emulate_FP8", "1")
    torch.testing.assert_close(compiled(x, fp8_params), fn(x, fp8_params))
    monkeypatch.setenv("NVTE_UnfusedDPA_Emulate_FP8", "0")

    # Changing attention params (ints, layout string, dtype) must recompile
    # and keep matching eager, still with no graph break.
    for changed_params in (
        dpa_utils.AttentionParams(head_dim_qk=128, head_dim_v=128),
        dpa_utils.AttentionParams(max_seqlen_q=512, max_seqlen_kv=512),
        dpa_utils.AttentionParams(qkv_layout="bshd_bshd_bshd"),
        dpa_utils.AttentionParams(qkv_dtype=torch.float16),
    ):
        torch.testing.assert_close(compiled(x, changed_params), fn(x, changed_params))

    # The baked probe result must drive the selection: report no fused
    # sub-backend and expect UnfusedDotProductAttention (flash disabled, so the
    # outcome is deterministic). Use a fresh frame: already-compiled frames
    # keep the previously baked constant (assume_constant_result installs no
    # guard on the wrapped function).
    monkeypatch.setenv("NVTE_FLASH_ATTN", "0")
    monkeypatch.setattr(
        dpa_utils.tex,
        "get_fused_attn_backend",
        lambda *args: dpa_utils.FusedAttnBackend["No_Backend"],
    )

    def fn_no_backend(x, params):
        return fn(x, params)

    compiled_no_backend = torch.compile(fn_no_backend, fullgraph=True)
    torch.testing.assert_close(compiled_no_backend(x, params), x + 4.0)


# ---------------------------------------------------------------------------
# Value-opaque quantizers
# ---------------------------------------------------------------------------


def _mxfp8(dtype=tex.DType.kFloat8E4M3):
    return MXFP8Quantizer(fp8_dtype=dtype)


def _blockwise(force_pow_2_scales=True):
    return Float8BlockQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
        force_pow_2_scales=force_pow_2_scales,
    )


def _current_scaling(amax_epsilon=0.0):
    return Float8CurrentScalingQuantizer(
        fp8_dtype=tex.DType.kFloat8E4M3,
        device=torch.device("cpu"),
        amax_epsilon=amax_epsilon,
    )


def _nvfp4(with_rht=True):
    # Default with_rht=True so the quantize round-trip below exercises the
    # derived ``rht_matrix`` tensor (the field most likely to be dropped on
    # value-key reconstruction). Post-RHT amax is required by the kernel
    # whenever RHT is on (pre-RHT amax is unsupported).
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=with_rht,
        with_post_rht_amax=with_rht,
    )


def _hw_available(quantizer):
    """Whether this HW can actually run the quantize kernel for *quantizer*."""
    if isinstance(quantizer, MXFP8Quantizer):
        return mxfp8_available
    if isinstance(quantizer, NVFP4Quantizer):
        return nvfp4_available
    if isinstance(quantizer, Float8BlockQuantizer):
        return fp8_block_scaling_available
    return fp8_available  # Float8CurrentScalingQuantizer


_VALUE_QUANTIZERS = [
    pytest.param(_mxfp8, id="mxfp8"),
    pytest.param(_blockwise, id="float8_blockwise"),
    pytest.param(_current_scaling, id="float8_current_scaling"),
    pytest.param(
        _nvfp4,
        id="nvfp4",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="NVFP4Quantizer requires CUDA to construct",
        ),
    ),
]


@pytest.mark.parametrize("factory", _VALUE_QUANTIZERS)
def test_quantizer_value_object(factory):
    """Value semantics + ``__fx_repr__`` round-trip via the production FX path."""
    a = factory()

    # ``__fx_repr__`` (used by torch.compile codegen) rebuilds an equal object.
    repr_str, globals_ = a.__fx_repr__()
    rebuilt = eval(repr_str, dict(globals_))  # pylint: disable=eval-used
    assert rebuilt == a and rebuilt is not a
    assert hash(rebuilt) == hash(a)
    # The deprecated amax-reduction group is never part of the value.
    assert getattr(rebuilt, "amax_reduction_group", None) is None

    # The rebuilt quantizer must also *behave* identically, not just compare
    # equal: equality only looks at the value key, so a field the kernel needs
    # but that is absent from the key (e.g. NVFP4's derived ``rht_matrix``) would
    # slip through the checks above and only blow up at quantize time. Run the
    # real quantize kernel on both and require bit-exact results.
    if _hw_available(a):
        x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")
        torch.testing.assert_close(rebuilt(x).dequantize(), a(x).dequantize(), rtol=0.0, atol=0.0)


def test_value_quantizer_rejects_process_group():
    """A value quantizer holding a live ProcessGroup must refuse to be turned
    into a value key / FX constant (raise), not silently drop the group."""
    import torch.distributed as dist  # pylint: disable=import-outside-toplevel

    created = not dist.is_initialized()
    if created:
        dist.init_process_group(backend="gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        q = MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3)
        q.amax_reduction_group = dist.group.WORLD
        # Every value-materialization path must reject it (hash, eq, __fx_repr__).
        with pytest.raises(TypeError):
            hash(q)
        with pytest.raises(TypeError):
            q.__fx_repr__()
    finally:
        if created:
            dist.destroy_process_group()


if _opaque_available:
    # A minimal custom op taking a tensor and a value-opaque quantizer that
    # quantizes + dequantizes inside it, one per production quantizer class.
    # ``test_quantizer_value_object_fullgraph`` drives this under
    # ``torch.compile(fullgraph=True)`` so the quantizer is used *inside* the
    # graph -- proving the opaque-type registration took effect (a graph break
    # would make ``fullgraph=True`` raise).
    _qdq_lib = torch.library.Library("test_te_qdq", "DEF")
    _QDQ_OPS = {}
    for _qcls in (
        MXFP8Quantizer,
        Float8BlockQuantizer,
        Float8CurrentScalingQuantizer,
        NVFP4Quantizer,
    ):
        _op = f"qdq_{_qcls.__name__}"
        _qdq_lib.define(f"{_op}(Tensor x, {get_opaque_type_name(_qcls)} q) -> Tensor")

        @torch.library.impl(f"test_te_qdq::{_op}", "CompositeExplicitAutograd", lib=_qdq_lib)
        def _qdq_impl(x, q):
            return q(x).dequantize()

        @torch.library.register_fake(f"test_te_qdq::{_op}", lib=_qdq_lib)
        def _qdq_fake(x, q):
            return torch.empty_like(x)

        _QDQ_OPS[_qcls] = getattr(torch.ops.test_te_qdq, _op)


@pytest.mark.skipif(
    not _opaque_available,
    reason="torch.compile opaque-object support requires PyTorch >= 2.11",
)
@pytest.mark.parametrize("factory", _VALUE_QUANTIZERS)
def test_quantizer_value_object_fullgraph(factory):
    """Quantizer is usable *inside* a torch.compile(fullgraph=True) graph.

    A custom op quantizes+dequantizes with the (opaque value) quantizer; the
    compiled result must match eager. ``fullgraph=True`` raises on any graph
    break, so this proves the opaque-type registration actually took effect --
    unlike merely passing the quantizer through.
    """
    q = factory()
    if not _hw_available(q):
        pytest.skip("format not supported on this HW")

    op = _QDQ_OPS[type(q)]
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    def fn(inp):
        return op(inp, q)

    ref = fn(x)
    torch._dynamo.reset()
    out = torch.compile(fn, fullgraph=True)(x)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)


# ---------------------------------------------------------------------------
# torch.compile-traceable allocation primitives + TensorSpec
# ---------------------------------------------------------------------------


# (factory, logical shape) -- shapes respect MXFP8 (mult. of 32) / blockwise (128)
# / NVFP4 (mult. of 16) constraints.
# Format support is gated at runtime, in the tests that run a kernel; the rest is
# pure Python and works on any HW.
_SPEC_QUANTIZERS = [
    pytest.param(_current_scaling, (4, 8), id="fp8_current_scaling"),
    pytest.param(_mxfp8, (64, 128), id="mxfp8"),
    pytest.param(_blockwise, (128, 256), id="fp8_blockwise"),
    pytest.param(_nvfp4, (64, 128), id="nvfp4"),
]


def _build_from_primitives(quantizer, shape, dtype, device="cpu"):
    """Assemble a quantized tensor straight from the quantizer primitives:
    ``alloc_tensors`` (inner tensors) + ``create_metadata`` (ctx) + the storage's
    ``__tensor_unflatten__`` -- i.e. exactly what ``TensorSpec.create_tensor``
    does, but without going through :class:`TensorSpec`.
    """
    names = tuple(quantizer.inner_tensor_specs(shape))
    ctx = quantizer.create_metadata(shape, dtype=dtype)
    allocated = quantizer.alloc_tensors(shape, device=device)
    inner_tensors = {name: allocated[name] for name in names}
    storage_cls = ctx["cls"]
    # Row-major (contiguous) outer stride for ``__tensor_unflatten__``; ``meta``
    # device computes it without allocating storage.
    outer_stride = torch.empty(tuple(shape), device="meta").stride()
    return storage_cls.__tensor_unflatten__(inner_tensors, ctx, tuple(shape), outer_stride)


def _signature(tensor, names):
    """Comparable shape/dtype fingerprint of a tensor and its inner tensors."""
    sig = {"__shape__": tuple(tensor.shape), "__dtype__": tensor.dtype}
    for name in names:
        inner = getattr(tensor, name)
        sig[name] = (tuple(inner.shape), inner.dtype)
    return sig


def _skip_if_dequantize_unsupported(q):
    """Skip when this HW can't run ``dequantize()`` for the quantizer's format.

    ``dequantize()`` runs the real kernel on CUDA, so each format has its own
    availability gate (mirrors the ``is_*_available`` checks in test_numerics).
    """
    if isinstance(q, MXFP8Quantizer):
        if not mxfp8_available:
            pytest.skip(reason_for_no_mxfp8)
    elif isinstance(q, NVFP4Quantizer):
        if not nvfp4_available:
            pytest.skip(reason_for_no_nvfp4)
    elif isinstance(q, Float8BlockQuantizer):
        if not fp8_block_scaling_available:
            pytest.skip(reason_for_no_fp8_block_scaling)
    elif not fp8_available:  # Float8 current scaling
        pytest.skip(reason_for_no_fp8)


# ----- Quantizer primitives -----


@pytest.mark.parametrize("factory, shape", _SPEC_QUANTIZERS)
def test_alloc_tensors_fake(factory, shape):
    """``alloc_tensors`` produces FakeTensors with the described shapes/dtypes."""
    q = factory()
    specs = q.inner_tensor_specs(shape)
    with FakeTensorMode():
        alloc = q.alloc_tensors(shape, device="cpu")
    assert set(alloc) == set(specs)
    for name, (spec_shape, spec_dtype) in specs.items():
        assert isinstance(alloc[name], FakeTensor)
        assert tuple(alloc[name].shape) == tuple(spec_shape)
        assert alloc[name].dtype == spec_dtype


@pytest.mark.parametrize("factory, shape", _SPEC_QUANTIZERS)
def test_storage_flatten_unflatten_roundtrip(factory, shape):
    """Storage ``__tensor_flatten__`` / ``__tensor_unflatten__`` round-trips.

    Build a tensor from ``alloc_tensors`` + ``create_metadata``, flatten it, then
    unflatten and verify shape/dtype and every inner buffer match before vs after.
    """
    q = factory()
    _skip_if_dequantize_unsupported(q)

    tensor = _build_from_primitives(q, shape, torch.bfloat16)
    names = tuple(q.inner_tensor_specs(shape))
    # Fill inner tensors with deterministic data (empty() may contain NaNs) so
    # the round-trip can be checked by value via dequantize().
    for name in names:
        inner = getattr(tensor, name)
        inner.copy_(torch.arange(inner.numel(), device=inner.device).reshape(inner.shape))
    before = _signature(tensor, names)
    expected = tensor.dequantize()

    flat_names, flat_ctx = tensor.__tensor_flatten__()
    assert set(flat_names) == set(names)
    inner = {name: getattr(tensor, name) for name in flat_names}
    rebuilt = type(tensor).__tensor_unflatten__(
        inner, flat_ctx, tuple(tensor.shape), tensor.stride()
    )

    assert isinstance(rebuilt, QuantizedTensor)
    assert _signature(rebuilt, flat_names) == before
    # The reconstructed tensor dequantizes to the same values.
    torch.testing.assert_close(rebuilt.dequantize(), expected, atol=0, rtol=0, equal_nan=True)


_USAGE_COMBOS = [
    pytest.param(True, True, id="rowwise_columnwise"),
    pytest.param(True, False, id="rowwise_only"),
    pytest.param(False, True, id="columnwise_only"),
]


@pytest.mark.parametrize("factory, shape", _SPEC_QUANTIZERS)
@pytest.mark.parametrize("rowwise, columnwise", _USAGE_COMBOS)
@pytest.mark.parametrize("internal", [False, True], ids=["wrapper", "internal"])
def test_python_alloc_matches_cpp_make_empty(factory, shape, rowwise, columnwise, internal):
    """Pure-Python allocation is interchangeable with the C++ allocation.

    Builds the same quantized tensor twice: via ``Quantizer.make_empty``
    (``tex.create_empty_quantized_tensor``, the C++ path) and via the Python
    primitives ``inner_tensor_specs`` + ``create_metadata`` + ``alloc_tensors``
    + ``__tensor_unflatten__`` (exactly what ``TensorSpec.create_tensor``
    does). Checks:

    * structural parity -- same concrete class, buffer set, per-buffer
      shape/dtype/device, logical shape/dtype and flatten context;
    * functional parity -- the real C++ quantize kernel writes bit-identical
      results into the Python-allocated inner tensors as into the C++-allocated
      ones, proving the Python buffer description matches the layout
      (padding/alignment) the kernels expect.
    """

    # Two independent, identically-configured quantizers so no state can leak
    # between the two allocation paths.
    def make_quantizer():
        q = factory()
        q.set_usage(rowwise=rowwise, columnwise=columnwise)
        q.internal = internal
        return q

    q_ref = make_quantizer()
    if not _hw_available(q_ref):
        pytest.skip("format not supported on this HW")
    q_py = make_quantizer()

    ref = q_ref.make_empty(shape, dtype=torch.bfloat16, device="cuda")
    py = _build_from_primitives(q_py, shape, torch.bfloat16, device="cuda")

    # --- Structural parity ---
    assert type(py) is type(ref)
    ref_names, ref_ctx = ref.__tensor_flatten__()
    py_names, py_ctx = py.__tensor_flatten__()
    assert set(py_names) == set(ref_names)
    for name in ref_names:
        ref_inner, py_inner = getattr(ref, name), getattr(py, name)
        assert tuple(py_inner.shape) == tuple(ref_inner.shape), name
        assert py_inner.dtype == ref_inner.dtype, name
        assert py_inner.device == ref_inner.device, name

    # Logical shape / dtype (bare storages are not torch.Tensors: they expose
    # size() and _dtype instead of .shape / .dtype).
    if isinstance(ref, QuantizedTensor):
        assert tuple(py.shape) == tuple(ref.shape) == tuple(shape)
        assert py.dtype == ref.dtype == torch.bfloat16
    else:
        assert tuple(py.size()) == tuple(ref.size()) == tuple(shape)
        # pylint: disable=protected-access
        assert py._dtype == ref._dtype == torch.bfloat16

    # Flatten context. The quantizer entry needs special handling: production
    # quantizers get a value-based __eq__ from register_value_opaque_quantizer,
    # but fall back to field-wise comparison for classes that don't define one
    # (plain object.__eq__ is identity, which would spuriously fail).
    assert set(py_ctx) == set(ref_ctx)
    for key in ("cls", "is_tensor", "requires_grad"):
        assert py_ctx[key] == ref_ctx[key], key
    ref_kwargs, py_kwargs = ref_ctx["nontensor_kwargs"], py_ctx["nontensor_kwargs"]
    assert set(py_kwargs) == set(ref_kwargs)
    for key in ref_kwargs:
        rv, pv = ref_kwargs[key], py_kwargs[key]
        if isinstance(rv, Quantizer) or isinstance(pv, Quantizer):
            assert type(pv) is type(rv), key
            assert (pv.rowwise_usage, pv.columnwise_usage, pv.internal) == (
                rv.rowwise_usage,
                rv.columnwise_usage,
                rv.internal,
            ), key
            if type(rv).__eq__ is not object.__eq__:
                assert pv == rv, key
        else:
            assert pv == rv, key

    # --- Functional parity: run the real C++ quantize kernel into both ---
    x = torch.randn(*shape, dtype=torch.bfloat16, device="cuda")

    def _quantize_into(quantizer, dst):
        if internal:
            # update_quantized() only accepts the wrapper classes; internal
            # (bare storage) tensors are filled through the same underlying
            # kernel binding directly.
            tex.quantize(x, quantizer, dst, None)
        else:
            quantizer.update_quantized(x, dst)

    # Scale-inv padding is never written by the kernel and both paths allocate it
    # uninitialized; zero it so the comparison below covers only kernel output.
    for name in ref_names:
        getattr(ref, name).zero_()
        getattr(py, name).zero_()

    # Some combos are rejected by the quantize kernel itself regardless of who
    # allocated the tensor (e.g. FP8 current-scaling columnwise-only on
    # TN-capable archs: there is no rowwise data buffer and
    # nvte_compute_scale_from_amax asserts on it). Parity then means the
    # Python-allocated tensor is rejected the same way -- not a silent skip.
    try:
        _quantize_into(q_ref, ref)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            _quantize_into(q_py, py)
        return
    _quantize_into(q_py, py)
    for name in ref_names:
        torch.testing.assert_close(
            getattr(py, name), getattr(ref, name), rtol=0.0, atol=0.0, equal_nan=True
        )

    # Value check through dequantize(). Some layouts cannot dequantize at all
    # (e.g. FP8 columnwise-only raises NotImplementedError) -- the C++-allocated
    # reference defines what is supported, and when it raises, the bitwise
    # buffer equality above already proves value parity.
    try:
        expected = ref.dequantize()
    except NotImplementedError:
        expected = None
    if expected is not None:
        torch.testing.assert_close(py.dequantize(), expected, rtol=0.0, atol=0.0)


# ----- TensorSpec -----


@pytest.mark.parametrize("factory, shape", _SPEC_QUANTIZERS)
def test_tensor_spec_matches_primitives(factory, shape):
    """TensorSpec is a thin wrapper: its ``create_metadata`` /
    ``create_inner_tensors`` / ``create_tensor`` match building everything
    directly from the quantizer primitives."""
    q = factory()
    spec = TensorSpec(shape=shape, dtype=torch.bfloat16, quantizer=q, device=torch.device("cpu"))
    assert spec.is_quantized

    # Metadata matches the quantizer's.
    assert spec.create_metadata() == q.create_metadata(shape, dtype=torch.bfloat16)

    # inner_names follows the storage's canonical __tensor_flatten__ order (the
    # order the real op flattens its outputs to), while create_inner_tensors
    # matches the inner_tensor_specs geometry (a name->shape/dtype mapping).
    specs = q.inner_tensor_specs(shape)
    direct = _build_from_primitives(q, shape, torch.bfloat16)
    names = tuple(direct.__tensor_flatten__()[0])
    assert set(names) == set(specs)
    assert spec.inner_names() == names
    inner_tensors = spec.create_inner_tensors()
    assert len(inner_tensors) == len(names)
    for name, inner in zip(names, inner_tensors):
        exp_shape, exp_dtype = specs[name]
        assert tuple(inner.shape) == tuple(exp_shape)
        assert inner.dtype == exp_dtype

    # The assembled tensor matches one built directly from the primitives.
    assert _signature(spec.create_tensor(), names) == _signature(direct, names)


@pytest.mark.parametrize("factory, shape", _SPEC_QUANTIZERS)
@pytest.mark.parametrize("fake", [False, True], ids=["eager", "fake"])
def test_tensor_spec_create_tensor(factory, shape, fake):
    """``create_tensor`` yields a quantized tensor with the right shape/dtype;
    its inner tensors are fake exactly under ``FakeTensorMode``."""
    q = factory()
    spec = TensorSpec(shape=shape, dtype=torch.bfloat16, quantizer=q, device=torch.device("cpu"))
    with FakeTensorMode() if fake else contextlib.nullcontext():
        out = spec.create_tensor()
    assert isinstance(out, QuantizedTensor)
    assert tuple(out.shape) == tuple(shape)
    assert out.dtype == torch.bfloat16
    for name in spec.inner_names():
        assert isinstance(getattr(out, name), FakeTensor) == fake


@pytest.mark.parametrize("factory, shape", _SPEC_QUANTIZERS)
def test_tensor_spec_create_tensor_compiles(factory, shape):
    """``TensorSpec.create_tensor`` traces under ``fullgraph=True`` (CPU)."""
    q = factory()

    def fn(x):
        spec = TensorSpec(shape=tuple(x.shape), dtype=x.dtype, quantizer=q, device=x.device)
        t = spec.create_tensor()
        acc = x.new_zeros(())
        for name in spec.inner_names():
            acc = acc + getattr(t, name).float().sum()
        return acc

    x = torch.zeros(*shape, dtype=torch.bfloat16)
    torch._dynamo.reset()
    out = torch.compile(fn, fullgraph=True)(x)
    assert out.shape == ()


def test_to_tensor_spec_plain():
    """``to_tensor_spec`` describes a plain tensor."""
    t = torch.empty(2, 3, dtype=torch.float32)
    spec = to_tensor_spec(t)
    assert not spec.is_quantized
    assert spec.shape == (2, 3)
    assert spec.dtype == torch.float32
    assert spec.inner_names() == ("data",)


@pytest.mark.parametrize("factory, shape", _SPEC_QUANTIZERS)
def test_to_tensor_spec_quantized(factory, shape):
    """``to_tensor_spec`` round-trips a quantized tensor back into a spec."""
    q = factory()
    tensor = TensorSpec(
        shape=shape, dtype=torch.bfloat16, quantizer=q, device=torch.device("cpu")
    ).create_tensor()

    spec = to_tensor_spec(tensor)
    assert spec.is_quantized
    assert spec.shape == tuple(shape)
    assert spec.dtype == torch.bfloat16
    # Same buffer layout as the original tensor.
    assert spec.inner_names() == tuple(q.inner_tensor_specs(shape))
    # Rebuilding from the derived spec matches the original tensor's structure.
    assert _signature(spec.create_tensor(), spec.inner_names()) == _signature(
        tensor, spec.inner_names()
    )


# ---------------------------------------------------------------------------
# te.Linear
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
@pytest.mark.parametrize("compile_mode", _compile_modes)
@pytest.mark.parametrize(
    "fp8_recipe",
    [None, *_all_recipes],
    ids=lambda r: "bf16" if r is None else type(r).__name__,
)
def test_te_linear_compiles(fp8_recipe, compile_mode):
    """
    torch.compile(fullgraph=True) of ``te.Linear`` under every built-in
    recipe (plus the bf16-only baseline with no autocast), for both the default
    backend and ``mode="reduce-overhead"`` (CUDA-graph trees).
    """
    dtype = torch.bfloat16
    device = "cuda"

    # FP8 GEMMs require leading dimensions divisible by 16.
    model = te.Linear(64, 32, params_dtype=dtype, device=device)

    def fn(inp):
        if fp8_recipe is None:
            return model(inp)
        with te.autocast(recipe=fp8_recipe):
            return model(inp)

    torch._dynamo.reset()
    if compile_mode == "reduce-overhead":
        _cudagraph_warmup(
            fn,
            torch.randn(32, 64, dtype=dtype, device=device, requires_grad=True),
            backward=True,
        )
        model.zero_grad(set_to_none=True)
    compiled = torch.compile(fn, fullgraph=True, mode=compile_mode)

    # Iterate a few times so reduce-overhead actually replays a captured graph.
    n_iters = 3 if compile_mode == "reduce-overhead" else 1
    with _assert_no_cudagraph_skips(compile_mode == "reduce-overhead"):
        for _ in range(n_iters):
            base = torch.randn(32, 64, dtype=dtype, device=device)
            _assert_close_eager_compiled(fn, compiled, model, base)


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("compile_mode", _compile_modes)
def test_te_linear_compile_with_quantized_fp8_weight(compile_mode):
    """torch.compile of Linear with the weight initialized as an FP8 tensor
    (exercises the wrapper op's ``register_torch_dispatch`` input flattening)."""
    dtype = torch.bfloat16
    device = "cuda"
    fp8_recipe = recipe.Float8CurrentScaling()

    with te.quantized_model_init(enabled=True, recipe=fp8_recipe):
        model = te.Linear(64, 32, params_dtype=dtype, device=device)

    assert isinstance(model.weight, te.Float8Tensor)

    def fn(inp):
        with te.autocast(recipe=fp8_recipe):
            return model(inp)

    torch._dynamo.reset()
    if compile_mode == "reduce-overhead":
        _cudagraph_warmup(
            fn,
            torch.randn(32, 64, dtype=dtype, device=device, requires_grad=True),
            backward=True,
        )
        model.zero_grad(set_to_none=True)
    compiled = torch.compile(fn, fullgraph=True, mode=compile_mode)

    n_iters = 3 if compile_mode == "reduce-overhead" else 1
    with _assert_no_cudagraph_skips(compile_mode == "reduce-overhead"):
        for _ in range(n_iters):
            base = torch.randn(32, 64, dtype=dtype, device=device)
            _assert_close_eager_compiled(fn, compiled, model, base)


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("compile_mode", _compile_modes)
def test_te_linear_compile_with_fp8_output(compile_mode):
    """torch.compile of ``te.Linear(..., fp8_output=True)`` under no_grad:
    forward must return a working :class:`Float8Tensor` (exercises the output
    rewrap path). The differentiable case falls back to eager, so it is not
    covered here."""
    dtype = torch.bfloat16
    device = "cuda"
    fp8_recipe = recipe.Float8CurrentScaling()

    model = te.Linear(64, 32, params_dtype=dtype, device=device)

    def fn(inp):
        with te.autocast(recipe=fp8_recipe):
            return model(inp, fp8_output=True)

    torch._dynamo.reset()
    if compile_mode == "reduce-overhead":
        with torch.no_grad():
            _cudagraph_warmup(fn, torch.randn(32, 64, dtype=dtype, device=device), backward=False)
    compiled = torch.compile(fn, fullgraph=True, mode=compile_mode)

    n_iters = 3 if compile_mode == "reduce-overhead" else 1
    with _assert_no_cudagraph_skips(compile_mode == "reduce-overhead"):
        for _ in range(n_iters):
            inp = torch.randn(32, 64, dtype=dtype, device=device)
            with torch.no_grad():
                out_eager = fn(inp)
                out = compiled(inp)
            assert isinstance(
                out, te.Float8Tensor
            ), f"expected Float8Tensor output, got {type(out).__name__}"
            assert out.shape == (32, 32)
            assert (
                out._quantizer is not None
            ), "FP8 output lost its quantizer on the torch.compile path"
            deq = out.dequantize()
            assert deq.shape == (32, 32)
            assert deq.dtype == dtype
            torch.testing.assert_close(
                deq, out_eager.dequantize(), atol=_EAGER_ATOL, rtol=_EAGER_RTOL
            )


# Configs rejected by LinearFwdArgs.compile_unsupported_reason() that a
# single-GPU unit test can construct. Distributed-only reasons (fsdp_group,
# DistributedWeight) and CPU offloading need machinery this file doesn't have;
# delayed scaling is a hard error (check_recipe_support), tested separately.
# Modes: "bwd" = fwd+bwd vs eager; "fwd_grad" = grad-enabled forward only
# (differentiable fp8_output backward hits a PyTorch limitation: the Float8
# output crossing the graph-break boundary gets a plain-tensor tangent);
# "no_grad" = forward under no_grad.
_FALLBACK_CASES = [
    "fp8_output_differentiable",
    "fuse_wgrad_accumulation",
    "delayed_wgrad",
    "quantized_input",
]


def _fallback_case(case, dtype, device):
    """Build ``(model, fn, mode, post_backward, reason)`` for one case."""
    model_kwargs = {}
    if case == "fuse_wgrad_accumulation":
        model_kwargs["fuse_wgrad_accumulation"] = True
    elif case == "delayed_wgrad":
        model_kwargs["delay_wgrad_compute"] = True
    model = te.Linear(64, 32, params_dtype=dtype, device=device, **model_kwargs)

    if case == "fp8_output_differentiable":
        fp8_recipe = recipe.Float8CurrentScaling()

        def fn(inp):
            with te.autocast(recipe=fp8_recipe):
                return model(inp, fp8_output=True).dequantize()

        return model, fn, "fwd_grad", None, "differentiable fp8_output=True"
    if case == "fuse_wgrad_accumulation":
        model.weight.main_grad = torch.zeros_like(model.weight, dtype=torch.float32)
        return model, model, "bwd", None, "fuse_wgrad_accumulation"
    if case == "delayed_wgrad":
        return model, model, "bwd", model.backward_dw, "delayed wgrad compute"
    if case == "quantized_input":
        fp8_recipe = recipe.Float8CurrentScaling()

        def fn(inp):
            with te.autocast(recipe=fp8_recipe):
                return model(inp)

        return model, fn, "no_grad", None, "a quantized input tensor"
    raise ValueError(case)


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.parametrize("case", _FALLBACK_CASES)
def test_te_linear_compile_eager_fallback(case):
    """Configs unsupported on the compiled custom-op path must fall back to
    eager under ``torch.compile`` -- warning + numerics identical to eager --
    and graph-break with the explicit reason under ``fullgraph=True``."""
    dtype, device = torch.bfloat16, "cuda"
    torch.manual_seed(0)
    model_ref, fn_ref, mode, post_bwd_ref, _ = _fallback_case(case, dtype, device)
    torch.manual_seed(0)
    model, fn, _, post_bwd, reason = _fallback_case(case, dtype, device)

    def make_inp():
        torch.manual_seed(1)
        x = torch.randn(32, 64, dtype=dtype, device=device)
        if case == "quantized_input":
            quantizer = Float8CurrentScalingQuantizer(
                fp8_dtype=tex.DType.kFloat8E4M3, device=device
            )
            return quantizer(x)
        return x.requires_grad_(mode != "no_grad")

    torch._dynamo.reset()
    compiled = torch.compile(fn)
    grad_ctx = torch.no_grad() if mode == "no_grad" else contextlib.nullcontext()

    inp_ref, inp = make_inp(), make_inp()
    with grad_ctx:
        out_ref = fn_ref(inp_ref)
        with pytest.warns(UserWarning, match="Falling back to eager execution under torch.compile"):
            out = compiled(inp)
    if mode == "bwd":
        out_ref.sum().backward()
        if post_bwd_ref is not None:
            post_bwd_ref()
        out.sum().backward()
        if post_bwd is not None:
            post_bwd()
        torch.testing.assert_close(inp.grad, inp_ref.grad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)
        # The fallback must also preserve the stateful side effects that define
        # these cases (main_grad accumulation, delayed wgrad), not just inp.grad.
        if getattr(model_ref.weight, "main_grad", None) is not None:
            assert model.weight.grad is None
            torch.testing.assert_close(
                model.weight.main_grad,
                model_ref.weight.main_grad,
                atol=_EAGER_ATOL,
                rtol=_EAGER_RTOL,
            )
        else:
            torch.testing.assert_close(
                model.weight.grad, model_ref.weight.grad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL
            )
    torch.testing.assert_close(out.detach(), out_ref.detach(), atol=_EAGER_ATOL, rtol=_EAGER_RTOL)

    torch._dynamo.reset()
    compiled_fg = torch.compile(fn, fullgraph=True)
    with pytest.raises(Exception, match=re.escape(reason)):
        with grad_ctx:
            compiled_fg(make_inp())


@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_te_linear_compile_delayed_scaling_raises():
    """Delayed scaling is rejected under torch.compile with a hard error
    (``check_recipe_support`` in ``te.autocast.__enter__``), not a fallback.
    Without fullgraph the raising frame is skipped and re-run eagerly (where
    the guard passes), so only ``fullgraph=True`` surfaces the error."""
    dtype, device = torch.bfloat16, "cuda"
    model = te.Linear(64, 32, params_dtype=dtype, device=device)
    fp8_recipe = recipe.DelayedScaling()

    def fn(inp):
        with te.autocast(recipe=fp8_recipe):
            return model(inp)

    torch._dynamo.reset()
    inp = torch.randn(32, 64, dtype=dtype, device=device, requires_grad=True)
    with pytest.raises(Exception, match="DelayedScaling is not supported under torch.compile"):
        torch.compile(fn, fullgraph=True)(inp)


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
def test_te_linear_compile_is_first_microbatch():
    """te.Linear with ``is_first_microbatch`` under torch.compile: FP8 weight
    caching updates the cached workspace in place, which the functional custom
    op can't express, so the schedule must fall back to eager -- warning +
    numerics identical to eager, cache reused in place across steps. The eager
    reference runs on a separate module so it cannot mask a corrupted or
    rebuilt cache."""
    dtype = torch.bfloat16
    device = "cuda"
    fp8_recipe = recipe.Float8CurrentScaling()
    model = te.Linear(64, 32, params_dtype=dtype, device=device)
    ref_model = te.Linear(64, 32, params_dtype=dtype, device=device)
    with torch.no_grad():
        ref_model.weight.copy_(model.weight)
        ref_model.bias.copy_(model.bias)

    schedule = [True, False, False]
    is_first = schedule[0]  # rebound each step; closed over by the fns.

    def fn(inp):
        with te.autocast(recipe=fp8_recipe):
            return model(inp, is_first_microbatch=is_first)

    def ref_fn(inp):
        with te.autocast(recipe=fp8_recipe):
            return ref_model(inp, is_first_microbatch=is_first)

    # Eager priming: FP8 state must exist before tracing (creating quantizers
    # in-graph breaks later recompiles; upstream Dynamo bug).
    is_first = None
    fn(torch.randn(32, 64, dtype=dtype, device=device, requires_grad=True))
    is_first = schedule[0]

    torch._dynamo.reset()
    compiled = torch.compile(fn)

    cached_workspace = None
    for step, is_first in enumerate(schedule):
        base = torch.randn(32, 64, dtype=dtype, device=device)

        inp_ref = base.detach().clone().requires_grad_(True)
        ref_model.zero_grad(set_to_none=True)
        out_ref = ref_fn(inp_ref)
        out_ref.sum().backward()

        inp = base.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        if step == 0:
            with pytest.warns(
                UserWarning, match="Falling back to eager execution under torch.compile"
            ):
                out = compiled(inp).clone()
        else:
            out = compiled(inp).clone()
        out.sum().backward()

        torch.testing.assert_close(out, out_ref.detach(), atol=_EAGER_ATOL, rtol=_EAGER_RTOL)
        torch.testing.assert_close(inp.grad, inp_ref.grad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)
        torch.testing.assert_close(
            model.weight.grad, ref_model.weight.grad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL
        )

        workspace = model._fp8_workspaces.get("weight")
        assert workspace is not None, f"no cached FP8 weight after step {step}"
        if step == 0:
            cached_workspace = workspace
        else:
            assert workspace is cached_workspace, f"cache rebuilt at step {step}"

    torch._dynamo.reset()
    compiled_fg = torch.compile(fn, fullgraph=True)
    is_first = True
    with pytest.raises(Exception, match=re.escape("FP8 weight caching")):
        compiled_fg(torch.randn(32, 64, dtype=dtype, device=device, requires_grad=True))


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
@pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
@pytest.mark.xfail(
    reason=(
        "value-opaque module state comes back as None on recompile"
        " (pytorch/pytorch#187041; fixed by #187057 (cold compile, merged)"
        " + #193190 (FX-graph-cache hit, in review))"
    ),
    strict=False,
)
def test_te_linear_compile_train_eval_switch():
    """train -> eval -> train on the same compiled ``te.Linear``, vs eager."""
    dtype = torch.bfloat16
    device = "cuda"
    fp8_recipe = recipe.Float8CurrentScaling()
    model = te.Linear(64, 32, params_dtype=dtype, device=device)

    def fn(inp):
        with te.autocast(recipe=fp8_recipe):
            return model(inp)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    def train_step():
        inp = torch.randn(16, 64, dtype=dtype, device=device, requires_grad=True)
        out = compiled(inp)
        out.sum().backward()
        inp_ref = inp.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out_ref = fn(inp_ref)
        out_ref.sum().backward()
        torch.testing.assert_close(
            out.detach(), out_ref.detach(), atol=_EAGER_ATOL, rtol=_EAGER_RTOL
        )
        torch.testing.assert_close(inp.grad, inp_ref.grad, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)
        model.zero_grad(set_to_none=True)

    train_step()

    model.eval()
    x = torch.randn(16, 64, dtype=dtype, device=device)
    with torch.no_grad():
        out_eval = compiled(x)
        ref_eval = fn(x)
    torch.testing.assert_close(out_eval, ref_eval, atol=_EAGER_ATOL, rtol=_EAGER_RTOL)

    model.train()
    train_step()


@pytest.mark.skipif(not _opaque_available, reason="torch opaque object API not available")
def test_te_linear_dynamic_shapes():
    """torch.compile of ``te.Linear`` with a ``mark_dynamic`` batch dimension:
    one graph must serve all batch sizes -- no recompiles -- and match eager
    numerically.

    Only the leading (batch/sequence) dims may be dynamic; the last dim is
    fixed by the weight's ``in_features``.
    """
    dtype = torch.bfloat16
    device = "cuda"
    in_features, out_features = 64, 32
    model = te.Linear(in_features, out_features, params_dtype=dtype, device=device)

    def fn(inp):
        return model(inp)

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True)

    batch_sizes = [16, 32, 48]

    # Two warmup calls: the second absorbs the one-time recompile from module
    # attributes lazily created during call one (e.g. the cached ``is_fsdp2``).
    for _ in range(2):
        warm = torch.randn(batch_sizes[0], in_features, dtype=dtype, device=device)
        torch._dynamo.mark_dynamic(warm, 0)
        compiled(warm.requires_grad_(True)).sum().backward()
    model.zero_grad(set_to_none=True)
    unique_graphs_baseline = _dynamo_counter("stats", "unique_graphs")
    if not unique_graphs_baseline:
        warnings.warn("unique_graphs counter is stale; skipping the recompile check")

    for batch in batch_sizes:
        inp = torch.randn(batch, in_features, dtype=dtype, device=device, requires_grad=True)
        # Mark batch dim as dynamic so Dynamo traces once and reuses across batch sizes.
        torch._dynamo.mark_dynamic(inp, 0)
        out = compiled(inp)
        assert out.shape == (batch, out_features), f"wrong output shape for batch={batch}"
        out.sum().backward()
        assert inp.grad is not None, f"no input gradient for batch={batch}"
        assert inp.grad.shape == inp.shape, f"wrong grad shape for batch={batch}"

        # Verify numerics against eager on each distinct batch size.
        inp_eager = inp.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        out_eager = model(inp_eager)
        out_eager.sum().backward()
        torch.testing.assert_close(
            out.detach(),
            out_eager.detach(),
            atol=_EAGER_ATOL,
            rtol=_EAGER_RTOL,
            msg=f"forward mismatch at batch={batch}",
        )
        torch.testing.assert_close(
            inp.grad,
            inp_eager.grad,
            atol=_EAGER_ATOL,
            rtol=_EAGER_RTOL,
            msg=f"dgrad mismatch at batch={batch}",
        )

    if unique_graphs_baseline:
        unique_graphs_after = _dynamo_counter("stats", "unique_graphs")
        assert unique_graphs_after == unique_graphs_baseline, (
            "Unexpected recompilation(s) across different batch sizes: "
            f"{unique_graphs_after - unique_graphs_baseline} extra graph(s) compiled"
        )
