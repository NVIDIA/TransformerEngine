# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import abc

import pytest
import torch
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
from transformer_engine.pytorch.quantization import QuantizerRole
from transformer_engine.pytorch.ops.basic.basic_linear import BasicLinear
from transformer_engine.pytorch.tensor.float8_tensor import Float8CurrentScalingQuantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor, _STORAGE_REGISTRY
from transformer_engine.pytorch.dynamo import TensorProto, to_tensor_proto
from transformer_engine.pytorch import (
    is_fp8_available,
    is_mxfp8_available,
    is_fp8_block_scaling_available,
    is_nvfp4_available,
    Float8Quantizer,
    Float8BlockQuantizer,
    MXFP8Quantizer,
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
    # value-key reconstruction).
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_rht=with_rht,
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
    pytest.param(_mxfp8, {"dtype": tex.DType.kFloat8E5M2}, id="mxfp8"),
    pytest.param(_blockwise, {"force_pow_2_scales": False}, id="float8_blockwise"),
    pytest.param(_current_scaling, {"amax_epsilon": 1e-4}, id="float8_current_scaling"),
    pytest.param(
        _nvfp4,
        {"with_rht": False},
        id="nvfp4",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(),
            reason="NVFP4Quantizer requires CUDA to construct",
        ),
    ),
]


@pytest.mark.parametrize("factory, other_kwargs", _VALUE_QUANTIZERS)
def test_quantizer_value_object(factory, other_kwargs):
    """Value semantics + ``__fx_repr__`` round-trip via the production FX path."""
    a, b = factory(), factory()
    # Same config -> equal, same hash, interchangeable as a dict/set key.
    assert a is not b
    assert a == b
    assert hash(a) == hash(b)
    assert {a: "x"}[b] == "x"
    # Different config -> not equal.
    assert a != factory(**other_kwargs)

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
@pytest.mark.parametrize("factory, other_kwargs", _VALUE_QUANTIZERS)
def test_quantizer_value_object_fullgraph(factory, other_kwargs):
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


# ---------------------------------------------------------------------------
# torch.compile-traceable allocation primitives + TensorProto
# ---------------------------------------------------------------------------


# (factory, logical shape) -- shapes respect MXFP8 (mult. of 32) / blockwise (128)
# / NVFP4 (mult. of 16) constraints.
_PROTO_QUANTIZERS = [
    pytest.param(_current_scaling, (4, 8), id="fp8_current_scaling"),
    pytest.param(_mxfp8, (64, 128), id="mxfp8"),
    pytest.param(_blockwise, (128, 256), id="fp8_blockwise"),
    pytest.param(
        _nvfp4,
        (64, 128),
        id="nvfp4",
        marks=pytest.mark.skipif(
            not nvfp4_available,
            reason="NVFP4 is not available",
        ),
    ),
]


def _build_from_primitives(quantizer, shape, dtype, device="cpu"):
    """Assemble a quantized tensor straight from the quantizer primitives:
    ``alloc_tensors`` (buffers) + ``create_metadata`` (ctx) + the storage's
    ``__tensor_unflatten__`` -- i.e. exactly what ``TensorProto.create_tensor``
    does, but without going through :class:`TensorProto`.
    """
    names = tuple(quantizer._describe_buffers(shape))  # pylint: disable=protected-access
    ctx = quantizer.create_metadata(shape, dtype=dtype)
    buffers = quantizer.alloc_tensors(shape, device=device)
    inner = {name: buffers[name] for name in names}
    storage_cls = _STORAGE_REGISTRY[ctx["cls"]]
    # Row-major (contiguous) outer stride for ``__tensor_unflatten__``; ``meta``
    # device computes it without allocating storage.
    outer_stride = torch.empty(tuple(shape), device="meta").stride()
    return storage_cls.__tensor_unflatten__(inner, ctx, tuple(shape), outer_stride)


def _signature(tensor, names):
    """Comparable shape/dtype fingerprint of a tensor and its inner buffers."""
    sig = {"__shape__": tuple(tensor.shape), "__dtype__": tensor.dtype}
    for name in names:
        buf = getattr(tensor, name)
        sig[name] = (tuple(buf.shape), buf.dtype)
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
            pytest.skip("NVFP4 is not available")
    elif isinstance(q, Float8BlockQuantizer):
        if not fp8_block_scaling_available:
            pytest.skip("FP8 block scaling is not available")
    elif not fp8_available:  # Float8 current scaling
        pytest.skip(reason_for_no_fp8)


# ----- Quantizer primitives -----


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_primitives_unflatten_compiles(factory, shape):
    """create_metadata + alloc_tensors + __tensor_unflatten__ compose and trace
    under ``fullgraph=True`` (CPU), without TensorProto."""
    q = factory()
    names = tuple(q._describe_buffers(shape))  # pylint: disable=protected-access

    def fn(x):
        t = _build_from_primitives(q, shape, x.dtype, device=x.device)
        # Read every buffer into the result so the alloc + unflatten can't be
        # eliminated as dead code -- forces the whole build path into the graph.
        acc = x.new_zeros(())
        for name in names:
            acc = acc + getattr(t, name).float().sum()
        return acc

    x = torch.zeros(*shape, dtype=torch.bfloat16)
    torch._dynamo.reset()
    out = torch.compile(fn, fullgraph=True)(x)
    assert out.shape == ()


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_alloc_tensors_fake(factory, shape):
    """``alloc_tensors`` produces FakeTensors with the described shapes/dtypes."""
    q = factory()
    bufs = q._describe_buffers(shape)  # pylint: disable=protected-access
    with FakeTensorMode():
        alloc = q.alloc_tensors(shape, device="cpu")
    assert set(alloc) == set(bufs)
    for name, (buf_shape, buf_dtype) in bufs.items():
        assert isinstance(alloc[name], FakeTensor)
        assert tuple(alloc[name].shape) == tuple(buf_shape)
        assert alloc[name].dtype == buf_dtype


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_storage_flatten_unflatten_roundtrip(factory, shape):
    """Storage ``__tensor_flatten__`` / ``__tensor_unflatten__`` round-trips.

    Build a tensor from ``alloc_tensors`` + ``create_metadata``, flatten it, then
    unflatten and verify shape/dtype and every inner buffer match before vs after.
    """
    q = factory()
    _skip_if_dequantize_unsupported(q)

    tensor = _build_from_primitives(q, shape, torch.bfloat16)
    names = tuple(q._describe_buffers(shape))  # pylint: disable=protected-access
    # Fill buffers with deterministic data (empty() may contain NaNs) so the
    # round-trip can be checked by value via dequantize().
    for name in names:
        buf = getattr(tensor, name)
        buf.copy_(torch.arange(buf.numel(), device=buf.device).reshape(buf.shape))
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


# ----- TensorProto -----


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_tensor_proto_matches_primitives(factory, shape):
    """TensorProto is a thin wrapper: its ``create_metadata`` /
    ``create_inner_tensors`` / ``create_tensor`` match building everything
    directly from the quantizer primitives."""
    q = factory()
    proto = TensorProto(shape=shape, dtype=torch.bfloat16, quantizer=q, device=torch.device("cpu"))
    assert proto.is_quantized

    # Metadata matches the quantizer's.
    assert proto.create_metadata() == q.create_metadata(shape, dtype=torch.bfloat16)

    # inner_names + create_inner_tensors match _describe_buffers.
    bufs = q._describe_buffers(shape)  # pylint: disable=protected-access
    names = tuple(bufs)
    assert proto.inner_names() == names
    inner = proto.create_inner_tensors()
    assert len(inner) == len(names)
    for name, buf in zip(names, inner):
        exp_shape, exp_dtype = bufs[name]
        assert tuple(buf.shape) == tuple(exp_shape)
        assert buf.dtype == exp_dtype

    # The assembled tensor matches one built directly from the primitives.
    direct = _build_from_primitives(q, shape, torch.bfloat16)
    assert _signature(proto.create_tensor(), names) == _signature(direct, names)


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_tensor_proto_create_tensor_eager(factory, shape):
    """``create_tensor`` (no fake) yields a real quantized tensor."""
    q = factory()
    proto = TensorProto(shape=shape, dtype=torch.bfloat16, quantizer=q, device=torch.device("cpu"))
    out = proto.create_tensor()
    assert isinstance(out, QuantizedTensor)
    assert tuple(out.shape) == tuple(shape)
    assert out.dtype == torch.bfloat16
    for name in proto.inner_names():
        assert not isinstance(getattr(out, name), FakeTensor)


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_tensor_proto_create_tensor_fake(factory, shape):
    """``create_tensor`` under ``FakeTensorMode`` yields a fake-backed quantized
    tensor with the right shape/dtype and fake inner buffers."""
    q = factory()
    proto = TensorProto(shape=shape, dtype=torch.bfloat16, quantizer=q, device=torch.device("cpu"))
    with FakeTensorMode():
        out = proto.create_tensor()
    assert isinstance(out, QuantizedTensor)
    assert tuple(out.shape) == tuple(shape)
    assert out.dtype == torch.bfloat16
    for name in proto.inner_names():
        assert isinstance(getattr(out, name), FakeTensor)


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_tensor_proto_create_tensor_compiles(factory, shape):
    """``TensorProto.create_tensor`` traces under ``fullgraph=True`` (CPU)."""
    q = factory()

    def fn(x):
        proto = TensorProto(shape=tuple(x.shape), dtype=x.dtype, quantizer=q, device=x.device)
        t = proto.create_tensor()
        acc = x.new_zeros(())
        for name in proto.inner_names():
            acc = acc + getattr(t, name).float().sum()
        return acc

    x = torch.zeros(*shape, dtype=torch.bfloat16)
    torch._dynamo.reset()
    out = torch.compile(fn, fullgraph=True)(x)
    assert out.shape == ()


def test_to_tensor_proto_plain():
    """``to_tensor_proto`` describes a plain tensor."""
    t = torch.empty(2, 3, dtype=torch.float32)
    proto = to_tensor_proto(t)
    assert not proto.is_quantized
    assert proto.shape == (2, 3)
    assert proto.dtype == torch.float32
    assert proto.inner_names() == ("data",)


@pytest.mark.parametrize("factory, shape", _PROTO_QUANTIZERS)
def test_to_tensor_proto_quantized(factory, shape):
    """``to_tensor_proto`` round-trips a quantized tensor back into a proto."""
    q = factory()
    tensor = TensorProto(
        shape=shape, dtype=torch.bfloat16, quantizer=q, device=torch.device("cpu")
    ).create_tensor()

    proto = to_tensor_proto(tensor)
    assert proto.is_quantized
    assert proto.shape == tuple(shape)
    assert proto.dtype == torch.bfloat16
    # Same buffer layout as the original tensor.
    assert proto.inner_names() == tuple(
        q._describe_buffers(shape)
    )  # pylint: disable=protected-access
    # Rebuilding from the derived proto matches the original tensor's structure.
    assert _signature(proto.create_tensor(), proto.inner_names()) == _signature(
        tensor, proto.inner_names()
    )
