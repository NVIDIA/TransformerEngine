# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import abc
import warnings
from typing import Optional

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
    Float8BlockQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
)
from utils import ModelConfig, dtype_tols, get_available_attention_backends, recipe_id
from transformer_engine.pytorch.attention.dot_product_attention.backends import (
    UnfusedDotProductAttention,
)

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
        q, k, v, extra, _ = _make_unfused_qkv(qkv_layout, dtype, requires_grad=True)
        out = compiled(q, k, v, extra)
        out.sum().backward()
        torch.cuda.synchronize()
        assert torch.isfinite(out).all()
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


# ---------------------------------------------------------------------------
# DotProductAttention under torch.compile
# ---------------------------------------------------------------------------


# Model configurations, described with the same ModelConfig the eager attention
# tests use. Which backends can run each of them is not hardcoded here --
# get_available_attention_backends() answers that, so configurations only one
# backend supports (arbitrary masks, biases, MLA head dims, ...) are covered
# rather than avoided.
def _cfg(model_config, qkv_format="bshd", packed=None, share_cu_seqlens=False):
    return dict(
        model_config=model_config,
        qkv_format=qkv_format,
        packed=packed,
        share_cu_seqlens=share_cu_seqlens,
    )


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
    "packed_qkv_bs3hd": _cfg(ModelConfig(2, 128, 4, 64, attn_mask_type="causal"), packed="bs3hd"),
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
}


def _qkv_layout(spec: dict) -> str:
    return spec["packed"] or "_".join([spec["qkv_format"]] * 3)


def _make_dpa(spec: dict, dtype: torch.dtype) -> te.DotProductAttention:
    config = spec["model_config"]
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

    if packed is not None:
        # Declarative packed QKV: q/k/v are derived from one buffer by DPA
        # itself, and the layout comes from the declaration -- the only packed
        # layout that torch.compile supports.
        assert packed == "bs3hd" and qkv_format == "bshd" and h == g and d_qk == d_v
        qkv = _randn((b, s_q, 3, h, d_qk))
        kwargs["qkv_layer"] = qkv
        kwargs["qkv_interleave_dim"] = -3
        grad_tensors, args = [qkv], ()
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


def _skip_unsupported(spec: dict, backend: str, dtype) -> None:
    """Skip configurations the backend under test cannot run at all."""
    available, _, _ = get_available_attention_backends(
        spec["model_config"], dtype, _qkv_layout(spec)
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

    FlashAttention and FusedAttention run the same kernel either way, so they
    have to match exactly. Inductor reassociates the unfused backend's softmax
    sums, and that error scales with the magnitude of those sums rather than
    with each output element -- so the absolute tolerance comes from the
    tensor's own scale, an elementwise relative tolerance being meaningless for
    elements near zero.
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
    """`DotProductAttention` under `torch.compile` must match eager in forward
    and backward, for every backend that supports the configuration.

    FlashAttention and UnfusedDotProductAttention are compiled with
    `fullgraph=True`, so the test fails on any graph break -- covering the whole
    module: input unpacking, qkv layout, backend selection and the backend
    itself. FusedAttention is an eager island (`@no_torch_dynamo` on its
    forward), so it is expected to graph-break and only has to stay correct.
    """
    dtype = torch.bfloat16
    spec = _DPA_COMPILE_CONFIGS[config]
    _skip_unsupported(spec, backend, dtype)
    _force_dpa_backend(monkeypatch, backend)

    module = _make_dpa(spec, dtype)
    args, kwargs, grads = _make_dpa_inputs(spec, dtype)
    _compare_compiled_to_eager(
        module, args, kwargs, grads, monkeypatch, backend, dtype, fullgraph=backend != "fused"
    )


@pytest.mark.parametrize("backend", ["flash", "unfused"])
def test_dpa_torch_compile_cudagraphs(monkeypatch, backend):
    """`mode="reduce-overhead"`: forward and backward of DotProductAttention
    are captured into CUDA graphs and replayed on subsequent iterations."""
    dtype = torch.bfloat16
    spec = _DPA_COMPILE_CONFIGS["self_bshd_causal"]
    _force_dpa_backend(monkeypatch, backend)

    module = _make_dpa(spec, dtype)

    torch._dynamo.reset()
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

    ref = module(*args, **kwargs)
    ref.sum().backward()
    ref_grads = [t.grad.clone() for t in grads]
    for tensor in grads:
        tensor.grad = None

    torch._dynamo.reset()
    _force_dpa_backend(monkeypatch, backend)
    with pytest.warns(UserWarning, match="Falling back to eager execution"):
        out = torch.compile(module)(*args, **kwargs)
    out.sum().backward()
    torch.cuda.synchronize()

    _assert_matches_eager(out, ref, backend, dtype)
    for tensor, ref_grad in zip(grads, ref_grads):
        _assert_matches_eager(tensor.grad, ref_grad, backend, dtype)

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


# (factory, kwargs producing a different-but-valid config)
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

    # The rebuilt quantizer must also *behave* identically, not just compare
    # equal: equality only looks at the value key, so a field the kernel needs
    # but that is absent from the key (e.g. NVFP4's derived ``rht_matrix``) would
    # slip through the checks above and only blow up at quantize time. Run the
    # real quantize kernel on both and require bit-exact results.
    if torch.cuda.is_available() and _hw_available(a):
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
    if not (torch.cuda.is_available() and _hw_available(q)):
        pytest.skip("format not supported on this HW")

    op = _QDQ_OPS[type(q)]
    x = torch.randn(128, 256, dtype=torch.bfloat16, device="cuda")

    def fn(inp):
        return op(inp, q)

    ref = fn(x)
    torch._dynamo.reset()
    out = torch.compile(fn, fullgraph=True)(x)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
