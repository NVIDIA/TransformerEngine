# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
from contextlib import contextmanager
from dataclasses import dataclass

import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.pytorch import NVFP4Quantizer
from transformer_engine.pytorch.custom_recipes.reference_nvfp4 import NVFP4QuantizerRef
from transformer_engine.pytorch.custom_recipes import reference_utils
from transformer_engine.common.recipe import NVFP4BlockScaling
from transformer_engine.pytorch.module import _split_quantization


recipe_available, reason_for_no_recipe = te.is_nvfp4_available(return_reason=True)
ue5m3_available, reason_for_no_ue5m3 = te.is_fp8_ue5m3_available(return_reason=True)
NVFP4_E4M3_AMAX_FOR_UNIT_GLOBAL_SCALE = 448.0 * 6.0
NVFP4_SUPPORTED_SCALE_MAX_VALUES = (0, 256, 448, 114688)


def _batch_rowwise_quantizers(count, *, scale_dtype=tex.DType.kFloat8E4M3, optimize=True):
    quantizers = []
    for _ in range(count):
        quantizer = NVFP4Quantizer(
            rowwise=True, columnwise=False, row_scaled_nvfp4=True, scale_dtype=scale_dtype
        )
        quantizer.internal = True
        quantizer.optimize_for_gemm = optimize
        quantizers.append(quantizer)
    return quantizers


def _assert_batch_rowwise_equal(actual, expected, rows, columns):
    assert type(actual) is type(expected)
    assert actual.get_metadata()["fake_dtype"] == expected.get_metadata()["fake_dtype"]
    assert actual._row_scaled_nvfp4 == expected._row_scaled_nvfp4
    assert actual._with_gemm_swizzled_scales == expected._with_gemm_swizzled_scales
    assert actual._rowwise_scale_inv.shape == expected._rowwise_scale_inv.shape
    assert actual._columnwise_data is expected._columnwise_data is None
    torch.testing.assert_close(actual._rowwise_data, expected._rowwise_data, atol=0, rtol=0)
    torch.testing.assert_close(actual._amax_rowwise, expected._amax_rowwise, atol=0, rtol=0)
    # Only live scale bytes are defined by the quantization kernel.
    torch.testing.assert_close(
        actual._rowwise_scale_inv[:rows, : columns // 16],
        expected._rowwise_scale_inv[:rows, : columns // 16],
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        actual.dequantize(dtype=torch.bfloat16),
        expected.dequantize(dtype=torch.bfloat16),
        atol=0,
        rtol=0,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("splits", [(128, 256), (32, 0, 96, 160), (0, 128, 0), (32, 32)])
@pytest.mark.parametrize("columns", [64, 192, 512])
@pytest.mark.parametrize("optimize", [False, True])
@pytest.mark.parametrize("scale_dtype", [tex.DType.kFloat8E4M3, tex.DType.kFloat8UE5M3])
def test_batch_rowwise_matches_per_expert(splits, columns, optimize, scale_dtype):
    """Batching must preserve payload, live scales, amax, and dequantized values."""
    if scale_dtype == tex.DType.kFloat8UE5M3 and not ue5m3_available:
        pytest.skip(reason_for_no_ue5m3)
    torch.manual_seed(307)
    quantizers = _batch_rowwise_quantizers(len(splits), scale_dtype=scale_dtype, optimize=optimize)
    x = torch.randn(sum(splits), columns, device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    x[1].fill_(2.0**-20)
    x[-1].mul_(1024)
    assert _split_quantization._can_batch_nvfp4_rowwise(x, splits, quantizers)
    for factor in (1.0, 0.25, 8.0):
        x.mul_(factor)
        actual, _ = _split_quantization._split_quantize(x, splits, quantizers, x.dtype)
        assert all(quantizer.optimize_for_gemm == optimize for quantizer in quantizers)
        expected = tex.split_quantize(x, splits, quantizers)
        for rows, output, reference in zip(splits, actual, expected):
            _assert_batch_rowwise_equal(output, reference, rows, columns)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "case",
    [
        "noncontiguous",
        "fp32",
        "unaligned_rows",
        "unaligned_columns",
        "single",
        "all_empty",
        "offload",
        "external",
        "columnwise",
        "rht",
        "post_rht",
        "stochastic",
        "two_dimensional",
        "amax_reduction",
        "4over6",
        "mixed_scale",
        "tensor_scaled",
        "zero_columns",
        "disabled_second_level_scale",
        "mixed_scale_bound",
    ],
)
def test_batch_rowwise_preserves_native_fallback(monkeypatch, case):
    """Unsupported settings and offload must continue through native split quantization."""
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    splits = (64, 64)
    quantizers = _batch_rowwise_quantizers(2)
    if case == "noncontiguous":
        x = x.t()
    elif case == "fp32":
        x = x.float()
    elif case == "unaligned_rows":
        splits = (16, 112)
    elif case == "unaligned_columns":
        x = x[:, :96].contiguous()
    elif case == "single":
        splits, quantizers = (128,), quantizers[:1]
    elif case == "all_empty":
        x, splits = x[:0], (0, 0)
    elif case == "external":
        quantizers[0].internal = False
    elif case == "columnwise":
        quantizers[0].columnwise_usage = True
    elif case == "mixed_scale":
        quantizers[1].scale_dtype = tex.DType.kFloat8UE5M3
    elif case == "tensor_scaled":
        quantizers[0].row_scaled_nvfp4 = False
    elif case == "zero_columns":
        x = x[:, :0]
    elif case == "disabled_second_level_scale":
        quantizers[0].disable_second_level_scale = True
    elif case == "mixed_scale_bound":
        quantizers[1].nvfp4_e4m3_max = 256
    elif case != "offload":
        field = {
            "rht": "with_rht",
            "post_rht": "with_post_rht_amax",
            "stochastic": "stochastic_rounding",
            "two_dimensional": "with_2d_quantization",
            "amax_reduction": "with_amax_reduction",
            "4over6": "nvfp4_use_4over6",
        }[case]
        setattr(quantizers[0], field, True)
    sentinel = object()
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(tex, "split_quantize", native)
    outputs, _ = _split_quantization._split_quantize(
        x, splits, quantizers, x.dtype, disable_bulk_allocation=case == "offload"
    )
    assert outputs is sentinel
    assert len(calls) == 1
    assert calls[0][0][0] is x
    assert calls[0][1]["disable_bulk_allocation"] == (case == "offload")


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_batch_rowwise_preserves_custom_quantizer():
    """A subclass's Python implementation must not be replaced by native batching."""
    calls = []

    class CustomNVFP4(NVFP4Quantizer):
        def quantize_impl(self, tensor):
            calls.append(tensor.shape)
            return tensor + 1

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    quantizers = [CustomNVFP4(row_scaled_nvfp4=True) for _ in range(2)]
    for quantizer in quantizers:
        quantizer.internal = True
        quantizer.set_usage(rowwise=True, columnwise=False)
    outputs, _ = _split_quantization._split_quantize(x, (64, 64), quantizers, x.dtype)
    assert len(calls) == 2
    torch.testing.assert_close(torch.cat(outputs), x + 1, atol=0, rtol=0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("columnwise", [False, True])
def test_batch_rowwise_mxfp8_unchanged(columnwise):
    """MXFP8 must retain the native path and identical payloads and scales."""
    splits = (128, 256)
    x = torch.randn(sum(splits), 128, device="cuda", dtype=torch.bfloat16)
    quantizers = [
        te.MXFP8Quantizer(fp8_dtype=tex.DType.kFloat8E4M3, rowwise=True, columnwise=columnwise)
        for _ in splits
    ]
    for quantizer in quantizers:
        quantizer.internal = True
    assert not _split_quantization._can_batch_nvfp4_rowwise(x, splits, quantizers)
    actual, _ = _split_quantization._split_quantize(x, splits, quantizers, x.dtype)
    expected = tex.split_quantize(x, splits, quantizers)
    for output, reference in zip(actual, expected):
        for name in (
            "_rowwise_data",
            "_rowwise_scale_inv",
            "_columnwise_data",
            "_columnwise_scale_inv",
        ):
            got, ref = getattr(output, name), getattr(reference, name)
            if ref is None:
                assert got is None
            else:
                torch.testing.assert_close(got, ref, atol=0, rtol=0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_batch_rowwise_padding_is_disjoint():
    """Writing an expert's padded scale extent must not corrupt a neighboring expert."""
    splits = (32, 32, 96)
    quantizers = _batch_rowwise_quantizers(len(splits))
    x = torch.randn(sum(splits), 128, device="cuda", dtype=torch.bfloat16)
    outputs, _ = _split_quantization._split_quantize(x, splits, quantizers, x.dtype)
    snapshots = [
        (
            result._rowwise_data.clone(),
            result._rowwise_scale_inv.clone(),
            result._amax_rowwise.clone(),
        )
        for result in outputs[1:]
    ]
    outputs[0]._rowwise_data.zero_()
    outputs[0]._rowwise_scale_inv.zero_()
    outputs[0]._amax_rowwise.zero_()
    for result, (data, scales, amax) in zip(outputs[1:], snapshots):
        torch.testing.assert_close(result._rowwise_data, data, atol=0, rtol=0)
        torch.testing.assert_close(result._rowwise_scale_inv, scales, atol=0, rtol=0)
        torch.testing.assert_close(result._amax_rowwise, amax, atol=0, rtol=0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("splits", [(128, 256), (32, 0, 96, 160)])
@pytest.mark.parametrize("scale_max", [0, 448])
def test_batch_rowwise_requantize_expert(splits, scale_max):
    """Requantizing one expert must preserve its layout and every neighbor's data."""
    torch.manual_seed(319)
    quantizers = _batch_rowwise_quantizers(len(splits), optimize=False)
    for quantizer in quantizers:
        quantizer.nvfp4_e4m3_max = scale_max
    x = torch.randn(sum(splits), 192, device="cuda", dtype=torch.bfloat16)
    actual, _ = _split_quantization._split_quantize(x, splits, quantizers, x.dtype)
    expected = tex.split_quantize(x, splits, quantizers)
    for index, rows in enumerate(splits):
        if not rows:
            continue
        replacement = 8 * torch.randn(rows, x.shape[1], device=x.device, dtype=x.dtype)
        snapshots = [
            (
                result._rowwise_data.clone(),
                result._rowwise_scale_inv.clone(),
                result._amax_rowwise.clone(),
            )
            for result in actual
        ]
        tex.quantize(replacement, quantizers[index], actual[index])
        tex.quantize(replacement, quantizers[index], expected[index])
        for other, (output, reference) in enumerate(zip(actual, expected)):
            _assert_batch_rowwise_equal(output, reference, splits[other], x.shape[1])
            if other != index:
                for tensor, snapshot in zip(
                    (output._rowwise_data, output._rowwise_scale_inv, output._amax_rowwise),
                    snapshots[other],
                ):
                    torch.testing.assert_close(tensor, snapshot, atol=0, rtol=0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_batch_rowwise_preserves_invalid_scale_bound():
    """A nondefault scale bound without 4over6 must retain the native validation error."""
    quantizers = _batch_rowwise_quantizers(2)
    for quantizer in quantizers:
        quantizer.nvfp4_e4m3_max = 256
    x = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    message = "non-default scale max is only supported with 4over6"
    with pytest.raises(RuntimeError, match=message):
        _split_quantization._split_quantize(x, (128, 128), quantizers, x.dtype)
    with pytest.raises(RuntimeError, match=message):
        tex.split_quantize(x, (128, 128), quantizers)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("splits", [(128, 256), (32, 0, 96)])
def test_batch_rowwise_graph_replay(splits):
    """Captured quantization must consume fresh inputs and refresh every expert's scales."""
    quantizers = _batch_rowwise_quantizers(len(splits))
    x = torch.randn(sum(splits), 128, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            tex.split_quantize(x, splits, quantizers)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual, _ = _split_quantization._split_quantize(x, splits, quantizers, x.dtype)
    for factor in (1.0, 0.25, 8.0):
        with torch.cuda.stream(stream):
            x.copy_(factor * torch.randn_like(x))
            graph.replay()
            expected = tex.split_quantize(x, splits, quantizers)
        stream.synchronize()
        for rows, output, reference in zip(splits, actual, expected):
            _assert_batch_rowwise_equal(output, reference, rows, x.shape[1])


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
def test_batch_rowwise_independent_devices_and_streams():
    """All allocations and launches must follow the input device and active stream."""
    for device in (0, 1):
        with torch.cuda.device(device):
            splits = (32, 96)
            quantizers = _batch_rowwise_quantizers(len(splits))
            x = torch.randn(sum(splits), 128, device="cuda", dtype=torch.bfloat16)
            streams = [torch.cuda.Stream(), torch.cuda.Stream()]
            results = []
            for stream in streams:
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(100_000)
                    results.append(
                        _split_quantization._split_quantize(x, splits, quantizers, x.dtype)[0]
                    )
            expected = tex.split_quantize(x, splits, quantizers)
            for stream, outputs in zip(streams, results):
                stream.synchronize()
                for rows, output, reference in zip(splits, outputs, expected):
                    assert output.device == x.device
                    _assert_batch_rowwise_equal(output, reference, rows, x.shape[1])


@dataclass(frozen=True)
class NVFP44Over6TestConfig:
    id: str
    use_4over6: bool = True
    e4m3_max: int = 0
    err_mode: str = "MAE"
    err_use_fast_math: bool = False


NVFP4_4OVER6_CONFIGS = [
    NVFP44Over6TestConfig(id="nvfp4", use_4over6=False),
    NVFP44Over6TestConfig(id="4over6-mae-e4m3-448-exact", e4m3_max=448, err_mode="MAE"),
    NVFP44Over6TestConfig(
        id="4over6-mae-e4m3-448-err-fast",
        e4m3_max=448,
        err_mode="MAE",
        err_use_fast_math=True,
    ),
    NVFP44Over6TestConfig(id="4over6-mae-e4m3-256-exact", e4m3_max=256, err_mode="MAE"),
    NVFP44Over6TestConfig(
        id="4over6-mae-e4m3-256-err-fast",
        e4m3_max=256,
        err_mode="MAE",
        err_use_fast_math=True,
    ),
    NVFP44Over6TestConfig(id="4over6-mse-e4m3-448-exact", e4m3_max=448, err_mode="MSE"),
    NVFP44Over6TestConfig(
        id="4over6-mse-e4m3-448-err-fast",
        e4m3_max=448,
        err_mode="MSE",
        err_use_fast_math=True,
    ),
    NVFP44Over6TestConfig(id="4over6-mse-e4m3-256-exact", e4m3_max=256, err_mode="MSE"),
    NVFP44Over6TestConfig(
        id="4over6-mse-e4m3-256-err-fast",
        e4m3_max=256,
        err_mode="MSE",
        err_use_fast_math=True,
    ),
]


@contextmanager
def nvfp4_4over6_err_fast_math(enabled: bool):
    old_value = os.environ.get("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH")
    os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = "1" if enabled else "0"
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop("NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH", None)
        else:
            os.environ["NVTE_NVFP4_4OVER6_ERR_USE_FAST_MATH"] = old_value


def maybe_skip_row_scaled_unsupported_quantization(
    row_scaled_nvfp4: bool,
    return_transpose: bool,
    with_2d_quantization: bool = False,
    use_4over6: bool = False,
    x_dtype: torch.dtype | None = None,
    M: int | None = None,
    N: int | None = None,
) -> None:
    if use_4over6 and with_2d_quantization:
        if x_dtype != torch.bfloat16 or M is None or N is None or M % 32 != 0 or N % 32 != 0:
            pytest.skip("NVFP4 2D 4over6 exact tests require the optimized BF16 kernel path")
    if not row_scaled_nvfp4:
        return
    if return_transpose:
        if use_4over6:
            pytest.skip("Row-scaled NVFP4 transpose does not support 4over6 mode")
        if x_dtype != torch.bfloat16 or M is None or N is None or M % 32 != 0 or N % 32 != 0:
            pytest.skip(
                "Row-scaled NVFP4 transpose requires BF16 input and dimensions divisible by 32"
            )
    if with_2d_quantization:
        pytest.skip("Row-scaled NVFP4 does not support 2D quantization")


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


def check_quantization_nvfp4_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    swizzled_scale: bool,
    use_cpp_allocator: bool,
    with_2d_quantization: bool,
    row_scaled_nvfp4: bool = False,
    use_4over6: bool = False,
    nvfp4_e4m3_max: int = 0,
    nvfp4_4over6_err_mode: str = "MAE",
    nvfp4_4over6_err_use_fast_math: bool = False,
    scale_dtype: te.DType = te.DType.kFloat8E4M3,
    check_amax_reset: bool = False,
) -> None:
    scale_dtype = te.DType.cast(scale_dtype)
    if nvfp4_e4m3_max not in NVFP4_SUPPORTED_SCALE_MAX_VALUES:
        raise ValueError("nvfp4_e4m3_max must be 0, 256, 448, or 114688.")
    if use_4over6 and scale_dtype == te.DType.kFloat8UE5M3:
        pytest.skip("NVFP4 4over6 is incompatible with UE5M3 scales")
    maybe_skip_row_scaled_unsupported_quantization(
        row_scaled_nvfp4, return_transpose, with_2d_quantization, use_4over6, x_dtype, M, N
    )

    te_dtype = te.DType.kFloat4E2M1

    # Setup device and random seed
    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Input
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    # Quantize
    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        with_2d_quantization=with_2d_quantization,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_err_mode,
        scale_dtype=scale_dtype,
    )

    if use_4over6:
        with nvfp4_4over6_err_fast_math(nvfp4_4over6_err_use_fast_math):
            if use_cpp_allocator:
                x_nvfp4_sut = nvfp4_quantizer(x)
            else:
                x_nvfp4_sut = nvfp4_quantizer.make_empty(
                    (M, N), dtype=x_dtype, device=device, requires_grad=False
                )
                x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
    else:
        if use_cpp_allocator:
            x_nvfp4_sut = nvfp4_quantizer(x)
        else:
            x_nvfp4_sut = nvfp4_quantizer.make_empty(
                (M, N), dtype=x_dtype, device=device, requires_grad=False
            )
            x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    assert x_nvfp4_sut._scale_dtype == scale_dtype

    # Extract data from NVFP4Tensor
    assert x_nvfp4_sut._rowwise_data is not None
    qx: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise
    qx_amax_t = x_nvfp4_sut._amax_columnwise

    # Reference quantization
    quant_tile_shape = (1, 16) if not with_2d_quantization else (16, 16)
    ref_quantizer = NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=quant_tile_shape,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=use_4over6,
        nvfp4_e4m3_max=nvfp4_e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_err_mode,
        nvfp4_4over6_err_use_fast_math=nvfp4_4over6_err_use_fast_math,
        scale_dtype=scale_dtype,
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)

    # Extract data from RefNVFP4Tensor
    qx_ref = (
        unpack_fp4(x_nvfp4_ref.data.view(dtype=torch.uint8))
        if x_nvfp4_ref.data is not None
        else None
    )
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        unpack_fp4(x_nvfp4_ref.data_t.view(dtype=torch.uint8))
        if x_nvfp4_ref.data_t is not None
        else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row
    ref_amax_t = x_nvfp4_ref.global_amax_col

    qx = unpack_fp4(qx)
    qx_t = unpack_fp4(qx_t) if qx_t is not None else None

    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    # Compare only the valid portion of scale tensors (reference may not have padding)
    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]

    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)

        # Compare only the valid portion of transpose scale tensors
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(qx_amax_t, ref_amax_t, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)

    if check_amax_reset:
        assert not row_scaled_nvfp4
        # Reusing an output whose maximum decreases verifies that every amax buffer is reset.
        for magnitude in (16.0, 0.5, 0.0, 2.0):
            x = torch.full((M, N), -magnitude, dtype=x_dtype, device=device)
            nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
            expected_amax = x.float().abs().amax().reshape(1)
            torch.testing.assert_close(x_nvfp4_sut._amax_rowwise, expected_amax, atol=0.0, rtol=0.0)
            if return_transpose:
                torch.testing.assert_close(
                    x_nvfp4_sut._amax_columnwise, expected_amax, atol=0.0, rtol=0.0
                )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize("return_transpose", [False, True], ids=["rowwise", "with_columnwise"])
@pytest.mark.parametrize("use_4over6", [False, True], ids=["standard", "4over6"])
def test_disable_second_level_scale_uses_only_block_scale(
    return_transpose: bool,
    use_4over6: bool,
) -> None:
    """A missing amax makes the global NVFP4 encode scale exactly one."""
    torch.manual_seed(0)
    x = torch.randn((128, 128), dtype=torch.bfloat16, device="cuda")
    x[0, 0] = NVFP4_E4M3_AMAX_FOR_UNIT_GLOBAL_SCALE

    common_kwargs = {
        "rowwise": True,
        "columnwise": return_transpose,
        "with_rht": False,
        "nvfp4_use_4over6": use_4over6,
    }
    expected = NVFP4Quantizer(**common_kwargs)(x)
    actual = NVFP4Quantizer(**common_kwargs, disable_second_level_scale=True)(x)

    torch.testing.assert_close(actual._rowwise_data, expected._rowwise_data, atol=0, rtol=0)
    torch.testing.assert_close(
        actual._rowwise_scale_inv, expected._rowwise_scale_inv, atol=0, rtol=0
    )
    torch.testing.assert_close(actual.dequantize(), expected.dequantize(), atol=0, rtol=0)
    assert actual._amax_rowwise is None
    if return_transpose:
        torch.testing.assert_close(
            actual._columnwise_data, expected._columnwise_data, atol=0, rtol=0
        )
        torch.testing.assert_close(
            actual._columnwise_scale_inv, expected._columnwise_scale_inv, atol=0, rtol=0
        )
        assert actual._amax_columnwise is None

    # Reusing an output previously populated by two-level scaling must remove
    # its amax buffers so common kernels select the unit-global-scale path.
    NVFP4Quantizer(**common_kwargs, disable_second_level_scale=True).update_quantized(
        x / 2, expected
    )
    assert expected._amax_rowwise is None
    if return_transpose:
        assert expected._amax_columnwise is None


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
def test_disable_second_level_scale_disables_row_scaled_nvfp4() -> None:
    """Row-scaled NVFP4 is incompatible with omitting second-level scales."""
    with pytest.warns(UserWarning, match="Row-scaled NVFP4 requires second-level scaling"):
        quantizer = NVFP4Quantizer(
            row_scaled_nvfp4=True,
            disable_second_level_scale=True,
        )

    assert not quantizer.row_scaled_nvfp4
    assert quantizer.disable_second_level_scale


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        # full tile cases
        (128, 128),
        (256, 256),
        (256, 1024),
        (1024, 256),
        # Padding required cases
        (256, 272),
        (304, 304),
        (320, 256),
        # Some larger tiles
        (2048, 2048),
        (1024, 2048),
        (2048, 1024),
        # # largest tile
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("return_transpose", [True, False], ids=["both_directions", "rowwise_only"])
@pytest.mark.parametrize("swizzled_scale", [False], ids=["linear_scale"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize(
    "with_2d_quantization", [True, False], ids=["2d_quantization", "1d_quantization"]
)
@pytest.mark.parametrize("row_scaled_nvfp4", [False, True], ids=["nvfp4", "nvfp4_row_scaled"])
@pytest.mark.parametrize(
    "nvfp4_4over6_config",
    NVFP4_4OVER6_CONFIGS,
    ids=lambda config: config.id,
)
def test_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    swizzled_scale: bool,
    use_cpp_allocator: bool,
    with_2d_quantization: bool,
    row_scaled_nvfp4: bool,
    nvfp4_4over6_config: NVFP44Over6TestConfig,
) -> None:
    check_amax_reset = not row_scaled_nvfp4 and not nvfp4_4over6_config.use_4over6
    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
        return_transpose=return_transpose,
        swizzled_scale=swizzled_scale,
        use_cpp_allocator=use_cpp_allocator,
        with_2d_quantization=with_2d_quantization,
        row_scaled_nvfp4=row_scaled_nvfp4,
        use_4over6=nvfp4_4over6_config.use_4over6,
        nvfp4_e4m3_max=nvfp4_4over6_config.e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_config.err_mode,
        nvfp4_4over6_err_use_fast_math=nvfp4_4over6_config.err_use_fast_math,
        check_amax_reset=check_amax_reset,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (128, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("extrema_high", [False, True], ids=["zeros", "maxes"])
@pytest.mark.parametrize("return_transpose", [True, False], ids=["both_directions", "rowwise_only"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize("row_scaled_nvfp4", [False, True], ids=["nvfp4", "nvfp4_row_scaled"])
@pytest.mark.parametrize(
    "nvfp4_4over6_config",
    NVFP4_4OVER6_CONFIGS,
    ids=lambda config: config.id,
)
def test_nvfp4_quantization_extrema_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    extrema_high: bool,
    return_transpose: bool,
    use_cpp_allocator: bool,
    row_scaled_nvfp4: bool,
    nvfp4_4over6_config: NVFP44Over6TestConfig,
):
    maybe_skip_row_scaled_unsupported_quantization(
        row_scaled_nvfp4, return_transpose, use_4over6=nvfp4_4over6_config.use_4over6
    )

    te_dtype = te.DType.kFloat4E2M1

    device = "cuda"
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if extrema_high:
        x = torch.full((M, N), torch.finfo(x_dtype).max, dtype=x_dtype, device=device)
    else:
        x = torch.zeros((M, N), dtype=x_dtype, device=device)

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=nvfp4_4over6_config.use_4over6,
        nvfp4_e4m3_max=nvfp4_4over6_config.e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_config.err_mode,
    )

    if nvfp4_4over6_config.use_4over6:
        with nvfp4_4over6_err_fast_math(nvfp4_4over6_config.err_use_fast_math):
            if use_cpp_allocator:
                x_nvfp4_sut = nvfp4_quantizer(x)
            else:
                x_nvfp4_sut = nvfp4_quantizer.make_empty(
                    (M, N), dtype=x_dtype, device=device, requires_grad=False
                )
                x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
    else:
        if use_cpp_allocator:
            x_nvfp4_sut = nvfp4_quantizer(x)
        else:
            x_nvfp4_sut = nvfp4_quantizer.make_empty(
                (M, N), dtype=x_dtype, device=device, requires_grad=False
            )
            x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    assert x_nvfp4_sut._rowwise_data is not None
    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise
    qx_amax_t = x_nvfp4_sut._amax_columnwise

    ref_quantizer = NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=nvfp4_4over6_config.use_4over6,
        nvfp4_e4m3_max=nvfp4_4over6_config.e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_config.err_mode,
        nvfp4_4over6_err_use_fast_math=nvfp4_4over6_config.err_use_fast_math,
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)

    qx_ref = x_nvfp4_ref.data.view(dtype=torch.uint8) if x_nvfp4_ref.data is not None else None
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        x_nvfp4_ref.data_t.view(dtype=torch.uint8) if x_nvfp4_ref.data_t is not None else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row
    ref_amax_t = x_nvfp4_ref.global_amax_col

    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(qx_amax_t, ref_amax_t, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.skipif(not ue5m3_available, reason=reason_for_no_ue5m3)
@pytest.mark.parametrize(
    "with_2d_quantization",
    [False, True],
    ids=["1d_quantization", "2d_quantization"],
)
def test_nvfp4_ue5m3_quantization_versus_reference(with_2d_quantization: bool) -> None:
    check_quantization_nvfp4_versus_reference(
        x_dtype=torch.bfloat16,
        M=128,
        N=128,
        return_transpose=True,
        swizzled_scale=False,
        use_cpp_allocator=False,
        with_2d_quantization=with_2d_quantization,
        scale_dtype=te.DType.kFloat8UE5M3,
    )


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (16, 128),
        (32, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("return_transpose", [True, False], ids=["both_directions", "rowwise_only"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize("row_scaled_nvfp4", [False, True], ids=["nvfp4", "nvfp4_row_scaled"])
@pytest.mark.parametrize(
    "nvfp4_4over6_config",
    NVFP4_4OVER6_CONFIGS,
    ids=lambda config: config.id,
)
def test_nvfp4_quantization_boundary_values(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
    row_scaled_nvfp4: bool,
    nvfp4_4over6_config: NVFP44Over6TestConfig,
):
    """
    Stress rounding/threshold behavior by placing values just below/above
    many potential bin edges within each 16-element microblock.
    Validates native vs reference byte-for-byte and scale parity.
    """
    maybe_skip_row_scaled_unsupported_quantization(
        row_scaled_nvfp4, return_transpose, use_4over6=nvfp4_4over6_config.use_4over6
    )

    te_dtype = te.DType.kFloat4E2M1

    device = "cuda"
    seed = 123
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Construct a single row with paired boundary values: v-eps, v+eps
    # spanning a wide dynamic range to exercise clipping and multiple bins.
    # Ensure even N and N is multiple of 16 for microblocks, which holds for 128.
    base = torch.linspace(-12.0, 12.0, steps=N // 2, dtype=torch.float32, device=device)
    eps = torch.full_like(base, 1e-3)
    # Avoid zero eps for very small magnitudes
    eps = torch.maximum(eps, 1e-4 * torch.ones_like(base))
    lower = base - eps
    upper = base + eps
    row = torch.empty(N, dtype=torch.float32, device=device)
    row[0::2] = lower
    row[1::2] = upper
    x = row.unsqueeze(0).repeat(M, 1).to(dtype=x_dtype)

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=nvfp4_4over6_config.use_4over6,
        nvfp4_e4m3_max=nvfp4_4over6_config.e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_config.err_mode,
    )

    if nvfp4_4over6_config.use_4over6:
        with nvfp4_4over6_err_fast_math(nvfp4_4over6_config.err_use_fast_math):
            if use_cpp_allocator:
                x_nvfp4_sut = nvfp4_quantizer(x)
            else:
                x_nvfp4_sut = nvfp4_quantizer.make_empty(
                    (M, N), dtype=x_dtype, device=device, requires_grad=False
                )
                x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)
    else:
        if use_cpp_allocator:
            x_nvfp4_sut = nvfp4_quantizer(x)
        else:
            x_nvfp4_sut = nvfp4_quantizer.make_empty(
                (M, N), dtype=x_dtype, device=device, requires_grad=False
            )
            x_nvfp4_sut = nvfp4_quantizer.update_quantized(x, x_nvfp4_sut)

    assert x_nvfp4_sut._rowwise_data is not None
    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise
    qx_amax_t = x_nvfp4_sut._amax_columnwise

    ref_quantizer = NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=nvfp4_4over6_config.use_4over6,
        nvfp4_e4m3_max=nvfp4_4over6_config.e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_config.err_mode,
        nvfp4_4over6_err_use_fast_math=nvfp4_4over6_config.err_use_fast_math,
    )
    x_nvfp4_ref = ref_quantizer.quantize(x)

    qx_ref = x_nvfp4_ref.data.view(dtype=torch.uint8) if x_nvfp4_ref.data is not None else None
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        x_nvfp4_ref.data_t.view(dtype=torch.uint8) if x_nvfp4_ref.data_t is not None else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row
    ref_amax_t = x_nvfp4_ref.global_amax_col

    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    # Compare only valid portion of scales (trim any padding)
    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(qx_amax_t, ref_amax_t, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (32, 128),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
@pytest.mark.parametrize("return_transpose", [True, False], ids=["both_directions", "rowwise_only"])
@pytest.mark.parametrize(
    "use_cpp_allocator", [True, False], ids=["cpp_allocator", "python_allocator"]
)
@pytest.mark.parametrize("row_scaled_nvfp4", [False, True], ids=["nvfp4", "nvfp4_row_scaled"])
@pytest.mark.parametrize(
    "nvfp4_4over6_config",
    NVFP4_4OVER6_CONFIGS,
    ids=lambda config: config.id,
)
def test_nvfp4_quantization_noncontiguous_inputs(
    x_dtype: torch.dtype,
    M: int,
    N: int,
    return_transpose: bool,
    use_cpp_allocator: bool,
    row_scaled_nvfp4: bool,
    nvfp4_4over6_config: NVFP44Over6TestConfig,
):
    maybe_skip_row_scaled_unsupported_quantization(
        row_scaled_nvfp4, return_transpose, use_4over6=nvfp4_4over6_config.use_4over6
    )

    te_dtype = te.DType.kFloat4E2M1

    device = "cuda"
    seed = 17
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Start from a contiguous tensor, then make a non-contiguous view by transpose
    x_base = torch.randn((M, N), dtype=x_dtype, device=device)
    x_nc = x_base.t()  # shape (N, M), non-contiguous
    assert not x_nc.is_contiguous()

    nvfp4_quantizer = NVFP4Quantizer(
        fp4_dtype=te_dtype,
        rowwise=True,
        columnwise=return_transpose,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=nvfp4_4over6_config.use_4over6,
        nvfp4_e4m3_max=nvfp4_4over6_config.e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_config.err_mode,
    )

    if nvfp4_4over6_config.use_4over6:
        with nvfp4_4over6_err_fast_math(nvfp4_4over6_config.err_use_fast_math):
            if use_cpp_allocator:
                x_nvfp4_sut = nvfp4_quantizer(x_nc)
            else:
                x_nvfp4_sut = nvfp4_quantizer.make_empty(
                    x_nc.shape, dtype=x_dtype, device=device, requires_grad=False
                )
                x_nvfp4_sut = nvfp4_quantizer.update_quantized(x_nc, x_nvfp4_sut)
    else:
        if use_cpp_allocator:
            x_nvfp4_sut = nvfp4_quantizer(x_nc)
        else:
            x_nvfp4_sut = nvfp4_quantizer.make_empty(
                x_nc.shape, dtype=x_dtype, device=device, requires_grad=False
            )
            x_nvfp4_sut = nvfp4_quantizer.update_quantized(x_nc, x_nvfp4_sut)

    assert x_nvfp4_sut._rowwise_data is not None
    qx = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx = x_nvfp4_sut._rowwise_scale_inv
    qx_t = (
        x_nvfp4_sut._columnwise_data.view(dtype=torch.uint8)
        if x_nvfp4_sut._columnwise_data is not None
        else None
    )
    sx_t = x_nvfp4_sut._columnwise_scale_inv
    qx_amax = x_nvfp4_sut._amax_rowwise
    qx_amax_t = x_nvfp4_sut._amax_columnwise

    ref_quantizer = NVFP4QuantizerRef(
        dtype=reference_utils.Fp4Formats.E2M1,
        rowwise=True,
        columnwise=return_transpose,
        pow_2_scales=False,
        eps=0.0,
        quant_tile_shape=(1, 16),
        row_scaled_nvfp4=row_scaled_nvfp4,
        nvfp4_use_4over6=nvfp4_4over6_config.use_4over6,
        nvfp4_e4m3_max=nvfp4_4over6_config.e4m3_max,
        nvfp4_4over6_err_mode=nvfp4_4over6_config.err_mode,
        nvfp4_4over6_err_use_fast_math=nvfp4_4over6_config.err_use_fast_math,
    )
    x_nvfp4_ref = ref_quantizer.quantize(x_nc)

    qx_ref = x_nvfp4_ref.data.view(dtype=torch.uint8) if x_nvfp4_ref.data is not None else None
    sx_ref = x_nvfp4_ref.scale.view(dtype=torch.uint8) if x_nvfp4_ref.scale is not None else None
    qx_t_ref = (
        x_nvfp4_ref.data_t.view(dtype=torch.uint8) if x_nvfp4_ref.data_t is not None else None
    )
    sx_t_ref = (
        x_nvfp4_ref.scale_t.view(dtype=torch.uint8) if x_nvfp4_ref.scale_t is not None else None
    )
    ref_amax = x_nvfp4_ref.global_amax_row
    ref_amax_t = x_nvfp4_ref.global_amax_col

    # Quantized must match
    torch.testing.assert_close(qx, qx_ref, atol=0.0, rtol=0.0)

    # Compare only valid portion of scales (trim padding)
    ref_sx_shape = sx_ref.shape
    sx_valid = sx[: ref_sx_shape[0], : ref_sx_shape[1]]
    torch.testing.assert_close(sx_valid, sx_ref, atol=0.0, rtol=0.0)

    if return_transpose:
        torch.testing.assert_close(qx_t, qx_t_ref, atol=0.0, rtol=0.0)
        ref_sx_t_shape = sx_t_ref.shape
        sx_t_valid = sx_t[: ref_sx_t_shape[0], : ref_sx_t_shape[1]]
        torch.testing.assert_close(sx_t_valid, sx_t_ref, atol=0.0, rtol=0.0)
        torch.testing.assert_close(qx_amax_t, ref_amax_t, atol=0.0, rtol=0.0)

    torch.testing.assert_close(qx_amax, ref_amax, atol=0.0, rtol=0.0)


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "with_2d_quantization, M, N",
    [
        # Existing 2D coverage: aligned and padded tiles.
        (True, 128, 128),
        (True, 256, 256),
        (True, 512, 512),
        (True, 2048, 2048),
        (True, 256, 272),
        (True, 304, 304),
        (True, 320, 256),
        # Focused 1D coverage for aligned and padded output reuse.
        (False, 128, 128),
        (False, 256, 272),
    ],
    ids=[
        "2d-128x128",
        "2d-256x256",
        "2d-512x512",
        "2d-2048x2048",
        "2d-256x272",
        "2d-304x304",
        "2d-320x256",
        "1d-128x128",
        "1d-256x272",
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_nvfp4_columnwise_only_matches_both_directions(
    x_dtype: torch.dtype,
    with_2d_quantization: bool,
    M: int,
    N: int,
):
    """Columnwise-only NVFP4 must match the columnwise half of both-directions.

    The 2D cases cover both kernels depending on the (dtype, shape) routing:
    - bf16 with rows % 32 == 0 and cols % 32 == 0 routes to the optimized
      ``quantize_transpose_nvfp4_2D_kernel`` (instantiated with RETURN_ROWWISE=false),
      validating that gating the rowwise pass/store leaves the shared
      ``block_amax_matrix`` and columnwise output bitwise-identical to both-directions.
    - non-bf16, or cols % 32 != 0, falls back to the columnwise-only 2D-amax-only
      pass in ``quantize_transpose_vector_blockwise_fp4.cu``.
    """
    te_dtype = tex.DType.kFloat4E2M1
    device = "cuda"

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((M, N), dtype=x_dtype, device=device)

    def _make_quantizer(*, rowwise: bool, columnwise: bool) -> NVFP4Quantizer:
        return NVFP4Quantizer(
            fp4_dtype=te_dtype,
            rowwise=rowwise,
            columnwise=columnwise,
            with_amax_reduction=False,
            amax_reduction_group=None,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=with_2d_quantization,
            row_scaled_nvfp4=False,
        )

    # Reference: produce both directions in a single kernel call.
    q_both = _make_quantizer(rowwise=True, columnwise=True)
    out_both = q_both(x)

    # SUT: produce columnwise only.
    q_col_only = _make_quantizer(rowwise=False, columnwise=True)
    out_col_only = q_col_only(x)

    # Columnwise data/scales/amax must be bitwise identical between the two paths.
    # If amax_smem is populated differently in the column-only path, scales diverge,
    # and the FP4 cast (which divides by encode_scale) produces different bytes.
    assert out_both._columnwise_data is not None
    assert out_col_only._columnwise_data is not None
    torch.testing.assert_close(
        out_col_only._columnwise_data.view(dtype=torch.uint8),
        out_both._columnwise_data.view(dtype=torch.uint8),
        atol=0,
        rtol=0,
    )

    # Compare only the valid (in-bounds) region of the columnwise scale tensor.
    # The padded tail (rows K..round_up(K, 128), cols ceil(M/16)..round_up(.., 4))
    # exists for cuBLAS alignment and is NEVER written by the kernel — its bytes
    # are whatever ``at::empty`` returned, which differs between two allocations.
    NVFP4_BLOCK = 16
    valid_outer = N  # cols of input == rows of columnwise scale tensor
    valid_inner = (M + NVFP4_BLOCK - 1) // NVFP4_BLOCK
    assert out_both._columnwise_scale_inv is not None
    assert out_col_only._columnwise_scale_inv is not None
    col_sx_both = out_both._columnwise_scale_inv.view(dtype=torch.uint8)
    col_sx_col_only = out_col_only._columnwise_scale_inv.view(dtype=torch.uint8)
    torch.testing.assert_close(
        col_sx_col_only[:valid_outer, :valid_inner],
        col_sx_both[:valid_outer, :valid_inner],
        atol=0,
        rtol=0,
    )

    assert out_both._amax_columnwise is not None
    assert out_col_only._amax_columnwise is not None
    torch.testing.assert_close(
        out_col_only._amax_columnwise, out_both._amax_columnwise, atol=0, rtol=0
    )

    if (M, N) in ((128, 128), (256, 272)):
        # Reusing a columnwise-only output exercises the path where its amax is primary.
        for magnitude in (16.0, 0.5, 0.0, 2.0):
            x = torch.full((M, N), -magnitude, dtype=x_dtype, device=device)
            q_col_only.update_quantized(x, out_col_only)
            expected_amax = x.float().abs().amax().reshape(1)
            torch.testing.assert_close(
                out_col_only._amax_columnwise, expected_amax, atol=0.0, rtol=0.0
            )

    # Sanity: column-only path must not allocate a rowwise output.
    assert out_col_only._rowwise_data is None
