# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for experimental per-locality-domain MXFP8 quantization."""

import os

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch.cpp_extensions import general_gemm


def _benchmark_ms(function, warmup: int = 20, iterations: int = 100) -> float:
    """Measure average CUDA execution time, including joined side-stream work."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _capture_cuda_graph(function):
    """Capture a callable, including any joined side-stream branches."""
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        function()
    torch.cuda.synchronize()
    return graph


def _localization_available() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.cuda.current_device()
    if os.getenv("NVTE_FORCE_DRIVER_LOCALIZATION", "0") != "1":
        try:
            from torch.cuda.green_contexts import is_localization_supported
            from torch.cuda.memory import get_num_locality_domains

            try:
                supported = is_localization_supported(device)
            except TypeError:
                supported = is_localization_supported()
            if supported and get_num_locality_domains(device) == 2:
                return True
        except (ImportError, TypeError):
            pass

    from transformer_engine.pytorch.tensor.driver_localization import (
        is_driver_localization_supported,
    )

    return is_driver_localization_supported(device)


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_rowwise_localized_pair() -> None:
    """Localized halves must match a full-tensor rowwise quantization."""
    tensor = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    quantizer.optimize_for_gemm = True

    reference = quantizer(tensor)
    localized = te.localize_mxfp8_tensor(tensor, quantizer)
    outputs = localized.quantize()
    torch.cuda.synchronize()

    assert len(outputs) == 2
    assert tuple(outputs[0].shape) == (128, 128)
    assert tuple(outputs[1].shape) == (128, 128)
    torch.testing.assert_close(
        localized.dequantize(),
        reference.dequantize(),
        atol=0.0,
        rtol=0.0,
    )


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_bidirectional_localized_pair() -> None:
    """Localized compact rowwise and columnwise outputs must match full-tensor output."""
    tensor = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )

    reference = quantizer(tensor)
    localized = te.localize_mxfp8_tensor(tensor, quantizer)
    outputs = localized.quantize()
    torch.cuda.synchronize()

    rows_per_domain = tensor.shape[0] // 2
    col_scale_rows_per_domain = rows_per_domain // 32
    for domain, output in enumerate(outputs):
        row_start = domain * rows_per_domain
        row_end = row_start + rows_per_domain
        scale_start = domain * col_scale_rows_per_domain
        scale_end = scale_start + col_scale_rows_per_domain
        torch.testing.assert_close(
            output._rowwise_data,
            reference._rowwise_data[row_start:row_end],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._rowwise_scale_inv,
            reference._rowwise_scale_inv[row_start:row_end],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._columnwise_data,
            reference._columnwise_data[row_start:row_end],
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._columnwise_scale_inv,
            reference._columnwise_scale_inv[scale_start:scale_end],
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_bidirectional_swizzled_localized_pair() -> None:
    """Localized fused-swizzle outputs must match independent half-tensor outputs."""
    tensor = torch.randn((256, 128), dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = True

    localized = te.localize_mxfp8_tensor(tensor, quantizer)
    outputs = localized.quantize()
    torch.cuda.synchronize()

    rows_per_domain = tensor.shape[0] // 2
    for domain, output in enumerate(outputs):
        row_start = domain * rows_per_domain
        row_end = row_start + rows_per_domain
        reference_half = quantizer(tensor[row_start:row_end])
        torch.testing.assert_close(
            output._rowwise_data,
            reference_half._rowwise_data,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._rowwise_scale_inv,
            reference_half._rowwise_scale_inv,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._columnwise_data,
            reference_half._columnwise_data,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._columnwise_scale_inv,
            reference_half._columnwise_scale_inv,
            atol=0.0,
            rtol=0.0,
        )


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_bidirectional_compact_vmm() -> None:
    """Two row-partition launches must produce one compact MXFP8 tensor."""
    shape = (256, 32768)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = False
    reference = quantizer(tensor)

    try:
        workspace = te.localize_mxfp8_output_vmm(tensor, quantizer)
    except (ImportError, RuntimeError, ValueError) as exc:
        pytest.skip(f"VMM localization is unavailable: {exc}")
    output = workspace.quantize()
    torch.cuda.synchronize()

    for name in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        torch.testing.assert_close(
            getattr(output, name),
            getattr(reference, name),
            atol=0.0,
            rtol=0.0,
        )
    assert not output._with_gemm_swizzled_scales
    workspace.close()


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_bidirectional_swizzled_vmm() -> None:
    """Two row-partition launches must produce one GEMM-swizzled MXFP8 tensor."""
    shape = (256, 32768)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = True
    reference = quantizer(tensor)

    try:
        workspace = te.localize_mxfp8_output_vmm(tensor, quantizer)
    except (ImportError, RuntimeError, ValueError) as exc:
        pytest.skip(f"VMM localization is unavailable: {exc}")
    output = workspace.quantize()
    torch.cuda.synchronize()

    for name in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        torch.testing.assert_close(
            getattr(output, name),
            getattr(reference, name),
            atol=0.0,
            rtol=0.0,
        )

    weight = torch.randn((128, shape[1]), dtype=tensor.dtype, device=tensor.device)
    quantized_weight = quantizer(weight)
    reference_gemm, *_ = general_gemm(
        quantized_weight,
        reference,
        out_dtype=tensor.dtype,
    )
    vmm_gemm, *_ = general_gemm(
        quantized_weight,
        output,
        out_dtype=tensor.dtype,
    )
    torch.testing.assert_close(vmm_gemm, reference_gemm, atol=0.0, rtol=0.0)
    workspace.close()


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_vmm_add_producer() -> None:
    """A BF16 add can write directly into the VMM input consumed by quantization."""
    shape = (256, 32768)
    lhs = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    rhs = torch.randn_like(lhs)
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = True
    reference = quantizer(lhs + rhs)

    try:
        workspace = te.MXFP8VMMWorkspace.empty(
            shape,
            dtype=lhs.dtype,
            device=lhs.device,
            quantizer=quantizer,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        pytest.skip(f"VMM localization is unavailable: {exc}")

    torch.add(lhs, rhs, out=workspace.input)
    output = workspace.quantize()
    torch.cuda.synchronize()
    for name in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        torch.testing.assert_close(
            getattr(output, name),
            getattr(reference, name),
            atol=0.0,
            rtol=0.0,
        )
    workspace.close()


def _run_localized_performance_comparison(
    quantizer: te.MXFP8Quantizer,
    mode: str,
) -> None:
    """Compare one full-chip launch with ordinary and localized green launches."""
    shape = (4096, 32768)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")

    baseline_output = quantizer.make_empty(
        shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    localized = te.localize_mxfp8_tensor(tensor, quantizer)

    # Control: use the same two half-sized green-stream launches as the
    # localized path, but keep input, output, and scales in ordinary allocations.
    # Comparing this with localized_ms isolates memory placement from launch
    # geometry and SM partitioning.
    rows_per_domain = shape[0] // 2
    green_unlocalized_inputs = (
        tensor[:rows_per_domain],
        tensor[rows_per_domain:],
    )
    green_unlocalized_outputs = tuple(
        quantizer.make_empty(
            (rows_per_domain, shape[1]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(2)
    )
    green_fork = torch.cuda.Event(enable_timing=False)
    green_joins = (
        torch.cuda.Event(enable_timing=False),
        torch.cuda.Event(enable_timing=False),
    )

    def green_unlocalized_quantize() -> None:
        parent_stream = torch.cuda.current_stream(tensor.device)
        green_fork.record(parent_stream)
        for domain, (input_half, output_half, stream) in enumerate(
            zip(
                green_unlocalized_inputs,
                green_unlocalized_outputs,
                localized.streams,
            )
        ):
            stream.wait_event(green_fork)
            with torch.cuda.stream(stream):
                quantizer.update_quantized(input_half, output_half)
            green_joins[domain].record(stream)
        for event in green_joins:
            parent_stream.wait_event(event)

    def baseline_quantize() -> None:
        quantizer.update_quantized(tensor, baseline_output)

    use_cuda_graph = os.getenv("MXFP8_LOCALIZATION_USE_CUDA_GRAPH") == "1"
    if use_cuda_graph:
        baseline_function = _capture_cuda_graph(baseline_quantize).replay
        green_unlocalized_function = _capture_cuda_graph(green_unlocalized_quantize).replay
        localized_function = _capture_cuda_graph(localized.quantize).replay
    else:
        baseline_function = baseline_quantize
        green_unlocalized_function = green_unlocalized_quantize
        localized_function = localized.quantize

    baseline_ms = _benchmark_ms(baseline_function)
    green_unlocalized_ms = _benchmark_ms(green_unlocalized_function)
    localized_ms = _benchmark_ms(localized_function)

    assert baseline_ms > 0.0
    assert green_unlocalized_ms > 0.0
    assert localized_ms > 0.0
    execution = "CUDA Graph" if use_cuda_graph else "eager"
    print(
        f"\nMXFP8 {mode} localization {shape} ({execution}):"
        f"\n  full-chip single launch:       {baseline_ms:.3f} ms"
        f"\n  two green, ordinary memory:    {green_unlocalized_ms:.3f} ms"
        f"\n  two green, localized memory:   {localized_ms:.3f} ms"
        f"\n  launch/partition contribution: {baseline_ms / green_unlocalized_ms:.3f}x"
        f"\n  memory-locality contribution:  {green_unlocalized_ms / localized_ms:.3f}x"
        f"\n  overall speedup:               {baseline_ms / localized_ms:.3f}x"
    )


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
def test_mxfp8_rowwise_localized_performance() -> None:
    """Compare rowwise full-chip and two-domain quantization."""
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=False,
    )
    quantizer.optimize_for_gemm = True
    _run_localized_performance_comparison(quantizer, "rowwise")


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
def test_mxfp8_bidirectional_localized_performance() -> None:
    """Compare compact bidirectional full-chip and two-domain quantization."""
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    _run_localized_performance_comparison(quantizer, "bidirectional compact")


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
def test_mxfp8_bidirectional_swizzled_localized_performance() -> None:
    """Compare fused-swizzle bidirectional full-chip and two-domain quantization."""
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = True
    _run_localized_performance_comparison(quantizer, "bidirectional fused-swizzle")


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
def test_mxfp8_bidirectional_swizzled_vmm_performance() -> None:
    """Compare quant and quant+small-GEMM with VMM-localized MXFP8 output."""
    shape = (4096, 32768)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = True
    baseline_output = quantizer.make_empty(
        shape,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    try:
        workspace = te.localize_mxfp8_output_vmm(tensor, quantizer)
    except (ImportError, RuntimeError, ValueError) as exc:
        pytest.skip(f"VMM localization is unavailable: {exc}")

    gemm_n = int(os.getenv("MXFP8_LOCALIZATION_GEMM_N", "256"))
    if gemm_n % 128 != 0:
        raise ValueError(f"MXFP8_LOCALIZATION_GEMM_N must be 128-aligned, got {gemm_n}")
    weight = torch.randn(
        (gemm_n, shape[1]),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    quantized_weight = quantizer(weight)
    baseline_gemm_output = torch.empty(
        (shape[0], gemm_n),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    localized_gemm_output = torch.empty_like(baseline_gemm_output)

    def baseline_quantize() -> None:
        quantizer.update_quantized(tensor, baseline_output)

    def baseline_pipeline() -> None:
        baseline_quantize()
        general_gemm(
            quantized_weight,
            baseline_output,
            out_dtype=tensor.dtype,
            out=baseline_gemm_output,
        )

    def localized_pipeline() -> None:
        workspace.quantize()
        general_gemm(
            quantized_weight,
            workspace.output,
            out_dtype=tensor.dtype,
            out=localized_gemm_output,
        )

    use_cuda_graph = os.getenv("MXFP8_LOCALIZATION_USE_CUDA_GRAPH") == "1"
    if use_cuda_graph:
        baseline_function = _capture_cuda_graph(baseline_quantize).replay
        localized_function = _capture_cuda_graph(workspace.quantize).replay
        baseline_pipeline_function = _capture_cuda_graph(baseline_pipeline).replay
        localized_pipeline_function = _capture_cuda_graph(localized_pipeline).replay
    else:
        baseline_function = baseline_quantize
        localized_function = workspace.quantize
        baseline_pipeline_function = baseline_pipeline
        localized_pipeline_function = localized_pipeline

    baseline_ms = _benchmark_ms(baseline_function)
    localized_ms = _benchmark_ms(localized_function)
    baseline_pipeline_ms = _benchmark_ms(baseline_pipeline_function)
    localized_pipeline_ms = _benchmark_ms(localized_pipeline_function)
    baseline_pipeline()
    localized_pipeline()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        localized_gemm_output,
        baseline_gemm_output,
        atol=0.0,
        rtol=0.0,
    )
    execution = "CUDA Graph" if use_cuda_graph else "eager"
    print(
        f"\nMXFP8 ordinary-input/VMM-output {shape} ({execution}):"
        f"\n  GEMM N:                       {gemm_n}"
        f"\n  full-chip quant:              {baseline_ms:.3f} ms"
        f"\n  localized-output quant:       {localized_ms:.3f} ms"
        f"\n  quant speedup:                {baseline_ms / localized_ms:.3f}x"
        f"\n  full-chip quant + GEMM:       {baseline_pipeline_ms:.3f} ms"
        f"\n  localized output + GEMM:      {localized_pipeline_ms:.3f} ms"
        "\n  quant + GEMM speedup:         "
        f"{baseline_pipeline_ms / localized_pipeline_ms:.3f}x"
    )
    workspace.close()


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
def test_mxfp8_vmm_add_producer_performance() -> None:
    """Measure BF16 add plus quant with ordinary and VMM producer outputs."""
    shape = (4096, 32768)
    lhs = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    rhs = torch.randn_like(lhs)
    ordinary_input = torch.empty_like(lhs)
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = True
    ordinary_output = quantizer.make_empty(
        shape,
        dtype=lhs.dtype,
        device=lhs.device,
    )
    try:
        workspace = te.MXFP8VMMWorkspace.empty(
            shape,
            dtype=lhs.dtype,
            device=lhs.device,
            quantizer=quantizer,
        )
    except (ImportError, RuntimeError, ValueError) as exc:
        pytest.skip(f"VMM localization is unavailable: {exc}")

    def ordinary_pipeline() -> None:
        torch.add(lhs, rhs, out=ordinary_input)
        quantizer.update_quantized(ordinary_input, ordinary_output)

    def localized_pipeline() -> None:
        torch.add(lhs, rhs, out=workspace.input)
        workspace.quantize()

    use_cuda_graph = os.getenv("MXFP8_LOCALIZATION_USE_CUDA_GRAPH") == "1"
    if use_cuda_graph:
        ordinary_function = _capture_cuda_graph(ordinary_pipeline).replay
        localized_function = _capture_cuda_graph(localized_pipeline).replay
    else:
        ordinary_function = ordinary_pipeline
        localized_function = localized_pipeline

    ordinary_ms = _benchmark_ms(ordinary_function)
    localized_ms = _benchmark_ms(localized_function)

    lhs.normal_()
    ordinary_function()
    localized_function()
    torch.cuda.synchronize()
    for name in (
        "_rowwise_data",
        "_rowwise_scale_inv",
        "_columnwise_data",
        "_columnwise_scale_inv",
    ):
        torch.testing.assert_close(
            getattr(workspace.output, name),
            getattr(ordinary_output, name),
            atol=0.0,
            rtol=0.0,
        )

    execution = "CUDA Graph" if use_cuda_graph else "eager"
    print(
        f"\nMXFP8 BF16-add producer {shape} ({execution}):"
        f"\n  ordinary add output + quant: {ordinary_ms:.3f} ms"
        f"\n  VMM add output + quant:      {localized_ms:.3f} ms"
        f"\n  speedup:                     {ordinary_ms / localized_ms:.3f}x"
    )
    workspace.close()
