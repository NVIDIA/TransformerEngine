# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Check fake output contracts against native operations."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch.dynamo import to_tensor_spec
from transformer_engine.pytorch.dynamo.fake import fake_bgrad
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)


def _assert_matches_native(spec, native):
    if spec.is_quantized:
        names, native_metadata = native.__tensor_flatten__()
        assert spec.inner_names() == tuple(names)
        fake_metadata = spec.create_metadata()
        for metadata in (fake_metadata, native_metadata):
            metadata["nontensor_kwargs"].pop("quantizer")
        assert fake_metadata == native_metadata
        native_inner = [getattr(native, name) for name in names]
    else:
        native_inner = [native]
    assert spec.shape == native.shape
    with FakeTensorMode():
        fake_inner = spec.create_inner_tensors()
    assert len(fake_inner) == len(native_inner)
    for fake, actual in zip(fake_inner, native_inner):
        assert isinstance(fake, FakeTensor)
        assert fake.shape == actual.shape
        assert fake.stride() == actual.stride()
        assert fake.dtype == actual.dtype
        assert fake.device == actual.device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(32,), (64, 128), (2, 32, 128), (0, 128)])
def test_fake_bgrad_unquantized_matches_native(dtype, shape):
    inp = torch.randn(shape, dtype=dtype, device="cuda")
    native_outputs = tex.bgrad_quantize(inp, None)
    with FakeTensorMode() as mode:
        specs = fake_bgrad(to_tensor_spec(mode.from_tensor(inp)))
    for spec, native in zip(specs, native_outputs):
        _assert_matches_native(spec, native)


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
@pytest.mark.parametrize("mode", ["rht", "1d", "2d", "row-scaled", "4over6"])
@pytest.mark.parametrize("optimize_for_gemm", [False, True])
@pytest.mark.parametrize("columnwise", [False, True])
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("disable_rht_fusion", [False, True])
@pytest.mark.parametrize(
    "shape",
    [
        (32, 32),
        (64, 128),
        (96, 128),
        (128, 64),
        (128, 128),
        (128, 160),
        (2, 32, 128),
        (2, 64, 128),
        (2, 1, 32, 128),
        (0, 128),
    ],
)
def test_fake_bgrad_nvfp4_matches_native(
    mode, optimize_for_gemm, columnwise, internal, disable_rht_fusion, shape, monkeypatch
):
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT_CAST_FUSION", str(int(disable_rht_fusion)))
    quantizer = NVFP4Quantizer(
        columnwise=columnwise,
        with_rht=mode == "rht",
        with_post_rht_amax=mode == "rht",
        with_2d_quantization=mode in ("2d", "4over6"),
        row_scaled_nvfp4=mode == "row-scaled",
        nvfp4_use_4over6=mode == "4over6",
        nvfp4_e4m3_max=448 if mode == "4over6" else 0,
        with_random_sign_mask=False,
    )
    quantizer.optimize_for_gemm = optimize_for_gemm
    quantizer.internal = internal
    inp = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    native_outputs = tex.bgrad_quantize(inp, quantizer)
    with FakeTensorMode() as fake_mode:
        specs = fake_bgrad(to_tensor_spec(fake_mode.from_tensor(inp)), quantizer)
    for spec, native in zip(specs, native_outputs):
        _assert_matches_native(spec, native)


@pytest.mark.skipif(not nvfp4_available, reason=reason_for_no_nvfp4)
@pytest.mark.parametrize("with_rht", [False, True])
def test_fake_bgrad_dynamic_alignment(with_rht, monkeypatch):
    monkeypatch.setenv("NVTE_NVFP4_DISABLE_RHT_CAST_FUSION", "0")
    quantizer = NVFP4Quantizer(
        with_rht=with_rht,
        with_post_rht_amax=with_rht,
        with_2d_quantization=not with_rht,
        with_random_sign_mask=False,
    )
    quantizer.optimize_for_gemm = True
    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    def describe(inp):
        _, spec = fake_bgrad(to_tensor_spec(inp), quantizer)
        return inp + spec.with_gemm_swizzled_scales

    torch._dynamo.reset()
    compiled = torch.compile(describe, backend=backend, fullgraph=True)
    # Pairs preserve alignment, including distinct nonzero remainders.
    for i, shape in enumerate(
        (
            (64, 128),
            (192, 384),
            (96, 160),
            (160, 224),
            (160, 160),
            (288, 288),
            (128, 64),
            (256, 192),
            (128, 128),
            (256, 256),
        )
    ):
        inp = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
        torch._dynamo.mark_dynamic(inp, (0, 1))
        previous_graph_count = len(graphs)
        actual = compiled(inp)
        _, native = tex.bgrad_quantize(inp, quantizer)
        torch.testing.assert_close(actual, inp + native._with_gemm_swizzled_scales, atol=0, rtol=0)
        if i % 2 == 1:
            assert len(graphs) == previous_graph_count
