# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Two-rank NCCL tests for tensor-parallel reduction dtype selection."""

import os
from importlib import import_module

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as ops
from transformer_engine.pytorch.distributed import (
    _AsyncHandle,
    allreduce,
    reduce_scatter_along_first_dim,
)


@pytest.fixture(scope="module")
def group():
    if "WORLD_SIZE" not in os.environ:
        pytest.skip("Launch two ranks with validation_reduction/launch.py")
    assert int(os.environ["WORLD_SIZE"]) == 2
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{os.environ['NVTE_TEST_RDZV_PATH']}", rank=rank, world_size=2
    )
    yield dist.group.WORLD
    dist.destroy_process_group()


DTYPES = [torch.float16, torch.bfloat16, torch.float32]
REDUCTION_DTYPES = [None, torch.float16, torch.bfloat16, torch.float32, torch.float64]


def rounded_sum(inp, group, dtype):
    inp = inp.contiguous()
    parts = [torch.empty_like(inp) for _ in range(2)]
    dist.all_gather(parts, inp, group=group)
    dtype = inp.dtype if dtype is None else dtype
    return torch.stack([part.to(dtype).double() for part in parts]).sum(0).to(dtype)


@pytest.mark.parametrize("input_dtype", DTYPES)
@pytest.mark.parametrize("reduction_dtype", REDUCTION_DTYPES)
@pytest.mark.parametrize("scatter", [False, True])
@pytest.mark.parametrize("async_op", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_collective_dtype(group, input_dtype, reduction_dtype, scatter, async_op, contiguous):
    torch.manual_seed(2026 + dist.get_rank(group))
    inp = torch.randn(16, 8, device="cuda", dtype=input_dtype)
    if not contiguous:
        inp = inp.t().contiguous().t()
    original = inp.clone()
    if not contiguous and not scatter and reduction_dtype is None:
        with pytest.raises(ValueError, match="contiguous"):
            allreduce(inp, group, async_op=async_op)
        return
    expected = rounded_sum(inp, group, reduction_dtype).to(input_dtype)
    if scatter:
        expected = expected.chunk(2)[dist.get_rank(group)]
        buffer = torch.empty_like(expected)
        out, handle = reduce_scatter_along_first_dim(
            inp, group, async_op=async_op, output=buffer, reduction_dtype=reduction_dtype
        )
        assert out is buffer
    else:
        out, handle = allreduce(inp, group, async_op=async_op, reduction_dtype=reduction_dtype)
        assert out is inp
    if async_op:
        assert handle is not None
        handle.wait()
        handle.wait()
    else:
        assert handle is None
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert out.dtype == input_dtype and out.device == inp.device
    if scatter:
        torch.testing.assert_close(inp, original, rtol=0, atol=0)


def make_layer(kind, dtype, sequence_parallel, group, reduction_dtype=None):
    if kind == "ops.Linear":
        return ops.Linear(
            64,
            64,
            bias=False,
            dtype=dtype,
            tensor_parallel_mode="row",
            tensor_parallel_group=group,
            sequence_parallel=sequence_parallel,
            reduction_dtype=reduction_dtype,
        )
    kwargs = dict(
        bias=False,
        params_dtype=dtype,
        tp_group=group,
        tp_size=2,
        sequence_parallel=sequence_parallel,
    )
    if kind == "Linear":
        return te.Linear(64, 64, parallel_mode="row", **kwargs)
    if kind == "LayerNormLinear":
        return te.LayerNormLinear(64, 64, parallel_mode="column", **kwargs)
    return te.LayerNormMLP(64, 128, set_parallel_mode=True, **kwargs)


@pytest.mark.parametrize("kind", ["Linear", "LayerNormLinear", "LayerNormMLP", "ops.Linear"])
@pytest.mark.parametrize("input_dtype", DTYPES)
@pytest.mark.parametrize("reduction_dtype", REDUCTION_DTYPES)
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_module_forward_backward(
    group, monkeypatch, kind, input_dtype, reduction_dtype, sequence_parallel
):
    torch.manual_seed(42 + dist.get_rank(group))
    layer = make_layer(kind, input_dtype, sequence_parallel, group, reduction_dtype)
    width = 64 if kind in ("LayerNormLinear", "LayerNormMLP") else 32
    rows = 8 if kind in ("LayerNormLinear", "LayerNormMLP") and sequence_parallel else 16
    inp = torch.randn(rows, width, device="cuda", dtype=input_dtype, requires_grad=True)
    if kind == "ops.Linear":
        layer.basic_ops[-1].reduction_dtype = None
    baseline = layer(inp)
    torch.manual_seed(7)
    dout = torch.randn_like(baseline)
    parameters = tuple(layer.parameters())
    baseline_grads = torch.autograd.grad(baseline, (inp, *parameters), dout)
    actual_input = inp.detach().clone().requires_grad_(True)
    calls = []
    reference_outputs = []
    original_allreduce = dist.all_reduce
    original_scatter = dist.reduce_scatter_tensor

    def checked_allreduce(tensor, *, group=None, async_op=False):
        calls.append(tensor.dtype)
        expected = rounded_sum(tensor, group, None)
        reference_outputs.append(expected.to(input_dtype))
        handle = original_allreduce(tensor, group=group, async_op=async_op)

        def check_result():
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        if async_op:
            return _AsyncHandle(handle, check_result, ())
        check_result()
        return handle

    def checked_scatter(output, tensor, *, group=None, async_op=False):
        calls.append(tensor.dtype)
        expected = rounded_sum(tensor, group, None).chunk(2)[dist.get_rank(group)]
        reference_outputs.append(expected.to(input_dtype))
        handle = original_scatter(output, tensor, group=group, async_op=async_op)

        def check_result():
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

        if async_op:
            return _AsyncHandle(handle, check_result, ())
        check_result()
        return handle

    with monkeypatch.context() as patch:
        patch.setattr(dist, "all_reduce", checked_allreduce)
        patch.setattr(dist, "reduce_scatter_tensor", checked_scatter)
        if kind == "ops.Linear":
            layer.basic_ops[-1].reduction_dtype = reduction_dtype
            actual = layer(actual_input)
        else:
            actual = layer(actual_input, reduction_dtype=reduction_dtype)
        if kind == "LayerNormLinear":
            actual_grads = torch.autograd.grad(actual, (actual_input, *parameters), dout)
    assert calls == [input_dtype if reduction_dtype is None else reduction_dtype]
    assert actual.shape == baseline.shape and actual.dtype == input_dtype
    if kind == "LayerNormLinear":
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
    else:
        torch.testing.assert_close(
            actual, reference_outputs[0].reshape(actual.shape), rtol=0, atol=0
        )
    assert torch.isfinite(actual).all()
    if kind != "LayerNormLinear":
        actual_grads = torch.autograd.grad(actual, (actual_input, *parameters), dout)
    for actual_grad, baseline_grad in zip(actual_grads, baseline_grads):
        assert actual_grad.shape == baseline_grad.shape and torch.isfinite(actual_grad).all()
        if kind != "LayerNormLinear" or reduction_dtype in (None, input_dtype):
            torch.testing.assert_close(actual_grad, baseline_grad, rtol=0, atol=0)
    if kind == "LayerNormLinear":
        ref_input = inp.detach().double().requires_grad_(True)
        ref_gamma = layer.layer_norm_weight.detach().double().requires_grad_(True)
        ref_beta = layer.layer_norm_bias.detach().double().requires_grad_(True)
        normalized = torch.nn.functional.layer_norm(
            ref_input, (64,), ref_gamma, ref_beta, layer.eps
        )
        ref_grads = torch.autograd.grad(
            normalized, (ref_input, ref_gamma, ref_beta), reference_outputs[0].double()
        )
        named_grads = dict(zip(dict(layer.named_parameters()), actual_grads[1:]))
        norm_grads = (
            actual_grads[0],
            named_grads["layer_norm_weight"],
            named_grads["layer_norm_bias"],
        )
        # Same dtype tolerances as tests/pytorch/utils.py; FP64 reference is independent.
        rtol = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}[input_dtype]
        for actual_grad, ref_grad in zip(norm_grads, ref_grads):
            torch.testing.assert_close(actual_grad.double(), ref_grad, rtol=rtol, atol=1e-5)
    if reduction_dtype is None or reduction_dtype == input_dtype:
        torch.testing.assert_close(actual, baseline, rtol=0, atol=0)


@pytest.mark.parametrize("scatter", [False, True])
def test_single_rank_validation(group, scatter):
    groups = [dist.new_group([rank]) for rank in range(2)]
    own_group = groups[dist.get_rank(group)]
    inp = torch.ones(8, device="cuda", dtype=torch.bfloat16)
    collective = reduce_scatter_along_first_dim if scatter else allreduce
    with pytest.raises(ValueError, match="reduction_dtype"):
        collective(inp, own_group, reduction_dtype=torch.int32)
    out, handle = collective(inp, own_group, reduction_dtype=torch.float32)
    assert out is inp and handle is None


@pytest.mark.parametrize("scatter", [False, True])
def test_narrowing_changes_sum(group, scatter):
    inp = torch.full((16, 8), 1.003 + 0.01 * dist.get_rank(group), device="cuda")
    expected = rounded_sum(inp, group, torch.bfloat16).float()
    native = rounded_sum(inp, group, torch.float32)
    assert not torch.equal(expected, native)
    if scatter:
        expected = expected.chunk(2)[dist.get_rank(group)]
        actual, _ = reduce_scatter_along_first_dim(inp, group, reduction_dtype=torch.bfloat16)
    else:
        actual, _ = allreduce(inp, group, reduction_dtype=torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("reduction_dtype", [None, torch.float32])
def test_userbuffers_fusion_preserves_dtype(group, reduction_dtype):
    from transformer_engine.pytorch.ops.fused.userbuffers_forward_linear import (
        UserbuffersForwardLinear,
    )

    layer = make_layer("ops.Linear", torch.bfloat16, True, group, reduction_dtype)
    linear, scatter = layer.basic_ops
    linear._userbuffers_options = {"comm_name": "proj"}
    fused = UserbuffersForwardLinear.fuse_forward_ops([linear, scatter])
    if reduction_dtype is None:
        assert len(fused) == 1 and isinstance(fused[0], UserbuffersForwardLinear)
    else:
        assert fused == [linear, scatter]


@pytest.mark.parametrize("kind", ["Linear", "LayerNormLinear", "LayerNormMLP"])
@pytest.mark.parametrize("unsupported", ["dtype", "userbuffers", "symmetric"])
def test_module_validation(group, kind, unsupported):
    layer = make_layer(kind, torch.bfloat16, False, group)
    dtype = torch.int32 if unsupported == "dtype" else torch.float32
    if unsupported == "userbuffers":
        if kind == "LayerNormMLP":
            layer.ub_overlap_rs = True
        else:
            layer.ub_overlap_rs_fprop = True
    if unsupported == "symmetric":
        layer.symmetric_ar_type = "one_shot"
    inp = torch.ones(
        16,
        64 if kind in ("LayerNormLinear", "LayerNormMLP") else 32,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="reduction_dtype"):
        layer(inp, reduction_dtype=dtype)


def test_compiled_eager_fallback(group, monkeypatch):
    linear_module = import_module("transformer_engine.pytorch.module.linear")
    layer = make_layer("Linear", torch.bfloat16, False, group)
    monkeypatch.setattr(linear_module, "_linear_op", lambda args: None)
    monkeypatch.setattr(
        layer, "_compile_eager_fallback_reason", lambda *args: "dtype test fallback"
    )
    seen = []
    original = dist.all_reduce

    def record_dtype(tensor, *, group=None, async_op=False):
        seen.append(tensor.dtype)
        return original(tensor, group=group, async_op=async_op)

    monkeypatch.setattr(dist, "all_reduce", record_dtype)
    inp = torch.ones(16, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad(), pytest.warns(UserWarning, match="Falling back to eager execution"):
        out = torch.compile(layer, backend="eager")(inp, reduction_dtype=torch.float32)
    assert seen == [torch.float32]
    assert out.dtype == inp.dtype and torch.isfinite(out).all()
