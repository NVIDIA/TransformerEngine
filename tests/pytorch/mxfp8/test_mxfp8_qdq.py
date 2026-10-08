# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""MXFP8 QDQ native parity, dispatch, storage lifecycle, and API validation."""

import pytest
import torch
from transformer_engine.pytorch.custom_recipes.qdq import MXFP8QDQQuantizer
import transformer_engine_torch as tex

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="MXFP8 QDQ fused implementation targets SM100",
)


@pytest.fixture(autouse=True)
def ordinary_math(monkeypatch):
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "0")


def assert_bits(actual, expected):
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert torch.equal(
        actual.contiguous().view(torch.int16), expected.contiguous().view(torch.int16)
    ), (
        (actual != expected).sum().item(),
        (actual.float() - expected.float()).abs().max().item(),
    )


def operand(backend):
    return MXFP8QDQQuantizer(backend=backend)


def data(shape, dtype, distribution):
    torch.manual_seed(41)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    if distribution == "zeros":
        x.zero_()
        x[:, ::2] = -0.0
    elif distribution == "outliers":
        x[0, 0] = 4096
        x[1::4] *= 1e-5
    elif distribution == "boundaries":
        v = torch.tensor(
            [
                0,
                -0.0,
                0.25,
                0.75,
                1.25,
                1.75,
                2.5,
                3.5,
                5,
                6,
                -6,
                -5,
                -2.5,
                1e-7,
                1e-4,
                448,
            ],
            device="cuda",
            dtype=dtype,
        )
        x.copy_(v.repeat(shape[0], shape[1] // 16))
    return x


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(32, 32), (96, 160), (128, 256), (1536, 1024)])
@pytest.mark.parametrize("distribution", ["normal", "zeros", "outliers", "boundaries"])
def test_fused_matches_materialized_q_plus_dq(dtype, shape, distribution):
    x = data(shape, dtype, distribution)
    fused, reference = operand("fused"), operand("reference")
    assert fused.selected_backend(x) == "fused"
    expected = reference(x).dequantize()
    for _ in range(2):  # Exact replay, no stochastic forward state.
        actual = fused(x).dequantize()
        if dtype == torch.float16:
            # Native's specialized FP16 MXFP8 FMA canonicalizes input -0.
            # Require exact numerical equality; zero signs are immaterial.
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert_bits(actual, expected)


def test_inplace_graph_noop_and_ste():
    x = data((128, 256), torch.bfloat16, "normal").requires_grad_()
    q = operand("fused")
    ref = operand("reference")
    y = q(x)
    # Differentiable operations use the tensor wrapper. dequantize() exposes
    # TE IdentityTensor's detached storage, not its autograd edge.
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))
    ptr = y._hp_data.data_ptr()
    noop = torch.ones(1, device="cuda", dtype=torch.float32)
    q.update_quantized(x * 2, y, noop_flag=noop)
    assert_bits(y.dequantize(), ref(x).dequantize())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        q.update_quantized(x, y, noop_flag=noop)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        q.update_quantized(x, y, noop_flag=noop)
    with torch.no_grad():
        x.mul_(2)
    noop.zero_()
    graph.replay()
    assert y._hp_data.data_ptr() == ptr
    assert_bits(y.dequantize(), ref(x).dequantize())


def test_noncontiguous_and_output_dtype_fallback():
    x = data((128, 256), torch.bfloat16, "normal").t()
    q = operand("auto")
    assert q.selected_backend(x) == "reference"
    assert_bits(q(x).dequantize(), operand("reference")(x).dequantize())
    with pytest.raises(ValueError):
        operand("fused")(x)
    source = x.contiguous().float()
    out = q.quantize(source, dtype=torch.bfloat16)
    assert out.dequantize().dtype == torch.bfloat16
    q.update_quantized(source * 2, out)
    assert_bits(out.dequantize(), q._reference(source * 2, torch.bfloat16))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_scale_boundaries_and_extremes(dtype):
    # Adjacent representable values straddling E8M0 scale transitions.
    exponents = range(-120, 119, 13) if dtype == torch.bfloat16 else range(-20, 7, 2)
    thresholds = torch.tensor(
        [448.0 * 2.0**e for e in exponents], device="cuda", dtype=dtype
    )
    maxima = torch.cat(
        (
            torch.nextafter(thresholds, torch.zeros_like(thresholds)),
            thresholds,
            torch.nextafter(thresholds, torch.full_like(thresholds, float("inf"))),
            torch.tensor(
                [torch.finfo(dtype).max, torch.finfo(dtype).tiny],
                device="cuda",
                dtype=dtype,
            ),
            torch.nextafter(
                torch.zeros(1, device="cuda", dtype=dtype),
                torch.ones(1, device="cuda", dtype=dtype),
            ),
        )
    )
    # Native materialized MXFP8 requires both logical dimensions aligned.
    maxima = torch.nn.functional.pad(maxima, (0, (-maxima.numel()) % 32))
    x = maxima[:, None] * torch.linspace(-1, 1, 32, device="cuda", dtype=dtype)
    actual = operand("fused")(x).dequantize()
    expected = operand("reference")(x).dequantize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_exceptional_values(value, dtype):
    x = torch.randn(32, 64, device="cuda", dtype=dtype)
    x[0, 0] = value
    actual = operand("fused")(x).dequantize()
    expected = operand("reference")(x).dequantize()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("backend", ["reference", "fused"])
@pytest.mark.parametrize("internal", [False, True])
def test_copy_workspace_dtype_and_noop(backend, internal):
    q = operand(backend)
    q.internal = internal
    copied = q.copy()
    assert copied.mxfp8_quantizer is not q.mxfp8_quantizer
    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    y = q.make_empty(x.shape, dtype=x.dtype, device=x.device)
    q.update_quantized(x, y)
    before = y.dequantize().clone()
    flag = torch.ones(1, device=x.device)
    q.update_quantized(2 * x, y, noop_flag=flag)
    torch.testing.assert_close(y.dequantize(), before, rtol=0, atol=0)
    flag.zero_()
    q.update_quantized(2 * x, y, noop_flag=flag)
    torch.testing.assert_close(
        y.dequantize(), operand("reference")(2 * x).dequantize(), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "bad",
    [
        "cpu",
        "shape",
        "dtype",
        "strided",
        "alignment",
        "columns",
        "rank",
        "noop_dtype",
        "noop_shape",
        "noop_cpu",
    ],
)
def test_direct_api_rejects_invalid_buffers(bad):
    x = torch.ones(32, 64, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)
    flag = None
    if bad == "cpu":
        y = y.cpu()
    elif bad == "shape":
        y = y[:16]
    elif bad == "dtype":
        y = y.float()
    elif bad == "strided":
        y = y.T
    elif bad == "alignment":
        y = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view_as(x)
    elif bad == "columns":
        x = x[:, :16].contiguous()
        y = torch.empty_like(x)
    elif bad == "rank":
        x = x.flatten()
        y = torch.empty_like(x)
    elif bad == "noop_dtype":
        flag = torch.ones(1, device="cuda", dtype=torch.int32)
    elif bad == "noop_shape":
        flag = torch.ones(2, device="cuda")
    elif bad == "noop_cpu":
        flag = torch.ones(1)
    with pytest.raises((RuntimeError, ValueError)):
        tex.mxfp8_qdq(x, y, flag)


def test_wrapper_errors_and_fast_math(monkeypatch):
    with pytest.raises(ValueError, match="backend"):
        MXFP8QDQQuantizer(backend="unknown")
    x = torch.ones(32, 64, device="cuda", dtype=torch.bfloat16)
    q = operand("auto")
    with pytest.raises(TypeError, match="MXFP8QDQQuantizer"):
        q.update_quantized(x, torch.empty_like(x))
    with pytest.raises(ValueError, match="MXFP8QDQQuantizer"):
        q.update_quantized(x[:16], q(x))
    monkeypatch.setenv("NVTE_USE_FAST_MATH", "1")
    assert q.selected_backend(x) == "reference"
    with pytest.raises(ValueError, match="Forced fused"):
        operand("fused")(x)


@pytest.mark.parametrize("shape", [(0, 128), (128, 0)])
def test_empty_fallback(shape):
    x = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    q = operand("auto")
    assert q.selected_backend(x) == "reference"
    assert q(x).dequantize().shape == x.shape


def test_current_stream_and_device():
    device = 1 if torch.cuda.device_count() > 1 else 0
    with torch.cuda.device(device), torch.cuda.stream(torch.cuda.Stream(device=device)):
        x = torch.randn(32, 64, device=f"cuda:{device}", dtype=torch.bfloat16)
        torch.testing.assert_close(
            operand("fused")(x).dequantize(),
            operand("reference")(x).dequantize(),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("source", ["original", "rowwise_dequantized"])
def test_hybrid_backward_source(source):
    from transformer_engine.pytorch.tensor.hybrid_tensor import HybridQuantizer
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    q = HybridQuantizer(
        rowwise_quantizer=operand("fused"),
        columnwise_quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3),
        columnwise_source=source,
    )
    y = q(x)
    decoded = operand("reference")(x).dequantize()
    torch.testing.assert_close(y._rowwise_storage.dequantize(), decoded, rtol=0, atol=0)
    native = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=False, columnwise=True)
    expected = native(x if source == "original" else decoded)
    for name in ("_columnwise_data", "_columnwise_scale_inv"):
        assert torch.equal(
            getattr(y._columnwise_storage, name), getattr(expected, name)
        )


@pytest.mark.parametrize("backend", ["reference", "fused"])
def test_linear_forward_and_backward(backend):
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import CustomRecipe

    # A standalone custom factory: every operand uses the QDQ wrapper, so
    # forward and backward can both be checked against materialized Q+DQ.
    def factory(role):
        return operand(backend)

    layer = te.Linear(128, 64, bias=False, params_dtype=torch.bfloat16, device="cuda")
    x = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)
    with te.autocast(recipe=CustomRecipe(qfactory=factory)):
        y = layer(x)
    y.backward(grad)
    q = operand("reference")
    xq, wq, gq = [q(t.detach()).dequantize() for t in (x, layer.weight, grad)]
    torch.testing.assert_close(y, xq @ wq.T, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(x.grad, gq @ wq, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(layer.weight.grad, gq.T @ xq, rtol=1e-2, atol=2e-3)


def test_custom_qdq_base_lifecycle():
    """A user-defined format inherits storage and autograd without TE kernels."""
    import copy
    from transformer_engine.pytorch.custom_recipes.qdq import QDQQuantizer

    class RoundedQDQQuantizer(QDQQuantizer):
        def selected_backend(self, tensor, dtype=None):
            if self.backend == "fused":
                raise ValueError("This example only supports reference execution")
            return "reference"

        def _reference(self, tensor, dtype):
            return tensor.round().to(dtype)

        @torch.no_grad()
        def _compute(self, tensor, output, noop_flag=None, *, backend=None):
            data = self._reference(tensor, output.dtype)
            if noop_flag is None:
                output.copy_(data)
            else:
                torch.where(noop_flag != 1, data, output, out=output)

        def copy(self):
            return copy.copy(self)

        def is_requantization_safe(self):
            return True

    with pytest.raises(TypeError):
        QDQQuantizer()
    x = torch.tensor([[0.25, 1.75]], device="cuda", requires_grad=True)
    q = RoundedQDQQuantizer(dtype=torch.bfloat16)
    y = q(x)
    assert y.dtype == torch.bfloat16
    torch.testing.assert_close(y.dequantize(), x.detach().round().bfloat16())
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
    workspace = q.make_empty(x.shape, dtype=x.dtype, device=x.device)
    q.update_quantized(x, workspace)
    flag = torch.ones(1, device=x.device)
    q.update_quantized(x + 2, workspace, noop_flag=flag)
    torch.testing.assert_close(workspace.dequantize(), y.dequantize())
    flag.zero_()
    q.update_quantized(x + 2, workspace, noop_flag=flag)
    torch.testing.assert_close(
        workspace.dequantize(), (x.detach() + 2).round().bfloat16()
    )
    q.internal = True
    assert q(x)._hp_data.dtype == torch.bfloat16
