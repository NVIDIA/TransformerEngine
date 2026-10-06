# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for experimental per-locality-domain MXFP8 quantization."""

import os

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import cpp_extensions as tex
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
@pytest.mark.parametrize("localized_data_layout", ["rowwise", "columnwise"])
def test_mxfp8_vmm_single_localized_data_layout(localized_data_layout: str) -> None:
    """Attention localizes its forward layout and graph-pools its backward layout."""
    from transformer_engine.pytorch.tensor.localized_mxfp8 import MXFP8VMMWorkspace
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator, is_vmm_tensor

    shape = (256, 32768)
    input_allocator = VMMRowSplitAllocator("cuda")
    tensor = input_allocator.allocate(shape, torch.bfloat16)
    tensor.normal_()
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = False
    reference = quantizer(tensor)
    workspace = MXFP8VMMWorkspace.from_vmm_input(
        tensor,
        quantizer,
        localized_data_layout=localized_data_layout,
    )
    try:
        output = workspace.quantize()
        torch.cuda.synchronize()
        assert is_vmm_tensor(output._rowwise_data) == (localized_data_layout == "rowwise")
        assert is_vmm_tensor(output._columnwise_data) == (localized_data_layout == "columnwise")
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
    finally:
        workspace.close()
        input_allocator.close()


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_vmm_columnwise_only_quantize_impl(monkeypatch) -> None:
    """Linear backward can localize its columnwise-only dKV quantization."""
    from transformer_engine.pytorch.tensor.localized_mxfp8 import (
        begin_mxfp8_vmm_workspace_iteration,
        clear_mxfp8_vmm_workspace_pools,
        end_mxfp8_vmm_workspace_iteration,
        release_mxfp8_vmm_tensor_workspaces,
    )
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator, is_vmm_tensor

    shape = (256, 32768)
    input_allocator = VMMRowSplitAllocator("cuda")
    tensor = input_allocator.allocate(shape, torch.bfloat16)
    tensor.normal_()
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=False,
        columnwise=True,
    )
    # TE LayerNormLinear owns this quantizer and marks it internal.
    quantizer.internal = True
    quantizer.optimize_for_gemm = True
    reference = quantizer.make_empty(shape, dtype=tensor.dtype, device=tensor.device)
    quantizer.update_quantized(tensor, reference)

    monkeypatch.setenv("NVTE_MXFP8_VMM_LOCALIZATION", "1")
    try:
        begin_mxfp8_vmm_workspace_iteration("test-columnwise")
        output = quantizer(tensor)
        torch.cuda.synchronize()
        assert output._rowwise_data is None
        assert output._rowwise_scale_inv is None
        assert is_vmm_tensor(output._columnwise_data)
        torch.testing.assert_close(
            output._columnwise_data,
            reference._columnwise_data,
            atol=0.0,
            rtol=0.0,
        )
        torch.testing.assert_close(
            output._columnwise_scale_inv,
            reference._columnwise_scale_inv,
            atol=0.0,
            rtol=0.0,
        )
        release_mxfp8_vmm_tensor_workspaces(output)
        end_mxfp8_vmm_workspace_iteration()
    finally:
        end_mxfp8_vmm_workspace_iteration(validate=False)
        clear_mxfp8_vmm_workspace_pools()
        input_allocator.close()


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.parametrize("shape", [(5760, 2880), (24576, 2880)])
def test_mxfp8_vmm_localizes_ordinary_cast_only_inputs(shape, monkeypatch) -> None:
    """GPT-OSS weight and activation casts use localized outputs."""
    from transformer_engine.pytorch.tensor.localized_mxfp8 import (
        begin_mxfp8_vmm_workspace_iteration,
        clear_mxfp8_vmm_workspace_pools,
        end_mxfp8_vmm_workspace_iteration,
        release_mxfp8_vmm_tensor_workspaces,
    )
    from transformer_engine.pytorch.tensor.vmm import is_vmm_tensor

    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.internal = True
    quantizer.optimize_for_gemm = False
    reference = quantizer(tensor)

    monkeypatch.setenv("NVTE_MXFP8_VMM_LOCALIZATION", "1")
    monkeypatch.setenv("NVTE_MXFP8_VMM_LOCALIZE_ALL_CASTS", "1")
    try:
        begin_mxfp8_vmm_workspace_iteration("test-ordinary-cast")
        output = quantizer(tensor)
        torch.cuda.synchronize()
        assert is_vmm_tensor(output._rowwise_data)
        assert is_vmm_tensor(output._columnwise_data)
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
        release_mxfp8_vmm_tensor_workspaces(output)
        end_mxfp8_vmm_workspace_iteration()
    finally:
        end_mxfp8_vmm_workspace_iteration(validate=False)
        clear_mxfp8_vmm_workspace_pools()


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_vmm_splits_cached_weight_updates(monkeypatch) -> None:
    """Cached GPT-OSS weight casts preserve their persistent output buffers."""
    from transformer_engine.pytorch.tensor.localized_mxfp8 import (
        begin_mxfp8_vmm_workspace_iteration,
        end_mxfp8_vmm_workspace_iteration,
    )
    from transformer_engine.pytorch.tensor.vmm import is_vmm_tensor

    shape = (5760, 2880)
    tensor = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    reference = quantizer(tensor)
    output = quantizer.make_empty(shape, dtype=tensor.dtype, device=tensor.device)
    rowwise_ptr = output._rowwise_data.data_ptr()
    columnwise_ptr = output._columnwise_data.data_ptr()

    monkeypatch.setenv("NVTE_MXFP8_VMM_LOCALIZATION", "1")
    monkeypatch.setenv("NVTE_MXFP8_VMM_LOCALIZE_ALL_CASTS", "1")
    try:
        begin_mxfp8_vmm_workspace_iteration("test-cached-weight")
        quantizer.update_quantized(tensor, output)
        torch.cuda.synchronize()
        assert output._rowwise_data.data_ptr() == rowwise_ptr
        assert output._columnwise_data.data_ptr() == columnwise_ptr
        assert not is_vmm_tensor(output._rowwise_data)
        assert not is_vmm_tensor(output._columnwise_data)
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
        end_mxfp8_vmm_workspace_iteration()
    finally:
        end_mxfp8_vmm_workspace_iteration(validate=False)


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
def test_mxfp8_vmm_workspace_pool_reuses_warmup_storage() -> None:
    """Full-iteration capture reuses VMM outputs allocated during eager warmup."""
    from transformer_engine.pytorch.tensor.localized_mxfp8 import (
        acquire_mxfp8_vmm_workspace,
        begin_mxfp8_vmm_workspace_iteration,
        clear_mxfp8_vmm_workspace_pools,
        end_mxfp8_vmm_workspace_iteration,
        release_mxfp8_vmm_workspace,
    )
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator

    shape = (256, 32768)
    input_allocator = VMMRowSplitAllocator("cuda")
    tensor = input_allocator.allocate(shape, torch.bfloat16)
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    quantizer.optimize_for_gemm = False
    try:
        begin_mxfp8_vmm_workspace_iteration("test")
        first = acquire_mxfp8_vmm_workspace(
            tensor,
            quantizer,
            localized_data_layout="rowwise",
        )
        second = acquire_mxfp8_vmm_workspace(
            tensor.view(shape),
            quantizer,
            localized_data_layout="rowwise",
        )
        assert second is not first

        release_mxfp8_vmm_workspace(first)
        reused = acquire_mxfp8_vmm_workspace(
            tensor,
            quantizer,
            localized_data_layout="rowwise",
        )
        assert reused is first
        release_mxfp8_vmm_workspace(reused)
        release_mxfp8_vmm_workspace(second)
        end_mxfp8_vmm_workspace_iteration()

        begin_mxfp8_vmm_workspace_iteration("test")
        next_iteration = acquire_mxfp8_vmm_workspace(
            tensor.view(shape),
            quantizer,
            localized_data_layout="rowwise",
        )
        release_mxfp8_vmm_workspace(next_iteration)
        end_mxfp8_vmm_workspace_iteration()

        assert next_iteration is first
        assert (
            next_iteration.output._rowwise_data.data_ptr()
            == first.output._rowwise_data.data_ptr()
        )
    finally:
        clear_mxfp8_vmm_workspace_pools()
        input_allocator.close()


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

    # Controls separate split-launch geometry, green-context SM partitioning,
    # and VMM memory placement.
    rows_per_domain = shape[0] // 2
    unlocalized_inputs = (
        tensor[:rows_per_domain],
        tensor[rows_per_domain:],
    )
    split_unlocalized_outputs = tuple(
        quantizer.make_empty(
            (rows_per_domain, shape[1]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(2)
    )
    green_unlocalized_outputs = tuple(
        quantizer.make_empty(
            (rows_per_domain, shape[1]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for _ in range(2)
    )
    ordinary_streams = tuple(torch.cuda.Stream(device=tensor.device) for _ in range(2))
    split_fork = torch.cuda.Event(enable_timing=False)
    split_joins = (
        torch.cuda.Event(enable_timing=False),
        torch.cuda.Event(enable_timing=False),
    )
    green_fork = torch.cuda.Event(enable_timing=False)
    green_joins = (
        torch.cuda.Event(enable_timing=False),
        torch.cuda.Event(enable_timing=False),
    )

    def split_unlocalized_quantize() -> None:
        parent_stream = torch.cuda.current_stream(tensor.device)
        split_fork.record(parent_stream)
        for domain, (input_half, output_half, stream) in enumerate(
            zip(
                unlocalized_inputs,
                split_unlocalized_outputs,
                ordinary_streams,
            )
        ):
            stream.wait_event(split_fork)
            with torch.cuda.stream(stream):
                quantizer.update_quantized(input_half, output_half)
            split_joins[domain].record(stream)
        for event in split_joins:
            parent_stream.wait_event(event)

    def green_unlocalized_quantize() -> None:
        parent_stream = torch.cuda.current_stream(tensor.device)
        green_fork.record(parent_stream)
        for domain, (input_half, output_half, stream) in enumerate(
            zip(
                unlocalized_inputs,
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
        split_unlocalized_function = _capture_cuda_graph(split_unlocalized_quantize).replay
        green_unlocalized_function = _capture_cuda_graph(green_unlocalized_quantize).replay
        localized_function = _capture_cuda_graph(localized.quantize).replay
    else:
        baseline_function = baseline_quantize
        split_unlocalized_function = split_unlocalized_quantize
        green_unlocalized_function = green_unlocalized_quantize
        localized_function = localized.quantize

    baseline_ms = _benchmark_ms(baseline_function)
    split_unlocalized_ms = _benchmark_ms(split_unlocalized_function)
    green_unlocalized_ms = _benchmark_ms(green_unlocalized_function)
    localized_ms = _benchmark_ms(localized_function)

    assert baseline_ms > 0.0
    assert split_unlocalized_ms > 0.0
    assert green_unlocalized_ms > 0.0
    assert localized_ms > 0.0
    execution = "CUDA Graph" if use_cuda_graph else "eager"
    print(
        f"\nMXFP8 {mode} localization {shape} ({execution}):"
        f"\n  full-chip single launch:       {baseline_ms:.3f} ms"
        f"\n  split ordinary streams/memory: {split_unlocalized_ms:.3f} ms"
        f"\n  two green, ordinary memory:    {green_unlocalized_ms:.3f} ms"
        f"\n  two green, localized memory:   {localized_ms:.3f} ms"
        f"\n  split-launch contribution:     {baseline_ms / split_unlocalized_ms:.3f}x"
        "\n  green-context contribution:    "
        f"{split_unlocalized_ms / green_unlocalized_ms:.3f}x"
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

    rows_per_domain = shape[0] // 2
    partition_shape = (rows_per_domain, shape[1])
    partition_inputs = tuple(tensor.chunk(2, dim=0))
    split_outputs = tuple(
        quantizer.make_empty(partition_shape, dtype=tensor.dtype, device=tensor.device)
        for _ in range(2)
    )
    green_unlocalized_outputs = tuple(
        quantizer.make_empty(partition_shape, dtype=tensor.dtype, device=tensor.device)
        for _ in range(2)
    )
    ordinary_streams = tuple(torch.cuda.Stream(device=tensor.device) for _ in range(2))
    eager_events = {
        key: (
            torch.cuda.Event(enable_timing=False),
            tuple(torch.cuda.Event(enable_timing=False) for _ in range(2)),
        )
        for key in ("split", "green")
    }
    capture_events = []

    def partitioned_quantize(outputs, streams, event_key: str) -> None:
        parent_stream = torch.cuda.current_stream(tensor.device)
        if torch.cuda.is_current_stream_capturing():
            fork_event = torch.cuda.Event(enable_timing=False)
            join_events = tuple(torch.cuda.Event(enable_timing=False) for _ in range(2))
            capture_events.extend((fork_event, *join_events))
        else:
            fork_event, join_events = eager_events[event_key]
        fork_event.record(parent_stream)
        for input_half, output_half, stream, join_event in zip(
            partition_inputs, outputs, streams, join_events
        ):
            stream.wait_event(fork_event)
            with torch.cuda.stream(stream):
                quantizer.update_quantized(input_half, output_half)
            join_event.record(stream)
        for event in join_events:
            parent_stream.wait_event(event)

    def split_unlocalized_quantize() -> None:
        partitioned_quantize(split_outputs, ordinary_streams, "split")

    def green_unlocalized_quantize() -> None:
        partitioned_quantize(green_unlocalized_outputs, workspace.streams, "green")

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
        split_unlocalized_function = _capture_cuda_graph(split_unlocalized_quantize).replay
        green_unlocalized_function = _capture_cuda_graph(green_unlocalized_quantize).replay
        localized_function = _capture_cuda_graph(workspace.quantize).replay
        baseline_pipeline_function = _capture_cuda_graph(baseline_pipeline).replay
        localized_pipeline_function = _capture_cuda_graph(localized_pipeline).replay
    else:
        baseline_function = baseline_quantize
        split_unlocalized_function = split_unlocalized_quantize
        green_unlocalized_function = green_unlocalized_quantize
        localized_function = workspace.quantize
        baseline_pipeline_function = baseline_pipeline
        localized_pipeline_function = localized_pipeline

    baseline_ms = _benchmark_ms(baseline_function)
    split_unlocalized_ms = _benchmark_ms(split_unlocalized_function)
    green_unlocalized_ms = _benchmark_ms(green_unlocalized_function)
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
        f"\n  split ordinary quant:         {split_unlocalized_ms:.3f} ms"
        f"\n  green ordinary-memory quant:  {green_unlocalized_ms:.3f} ms"
        f"\n  localized-output quant:       {localized_ms:.3f} ms"
        f"\n  split-launch contribution:    {baseline_ms / split_unlocalized_ms:.3f}x"
        "\n  green-context contribution:   "
        f"{split_unlocalized_ms / green_unlocalized_ms:.3f}x"
        "\n  memory-locality contribution: "
        f"{green_unlocalized_ms / localized_ms:.3f}x"
        f"\n  overall quant speedup:        {baseline_ms / localized_ms:.3f}x"
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


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
@pytest.mark.parametrize("hidden_size", [1536, 512], ids=["q_up", "kv_up"])
def test_mxfp8_vmm_layernorm_quant_localization_performance(
    hidden_size: int, monkeypatch
) -> None:
    """Measure the fused MLA LayerNorm+MXFP8 producer on two locality domains."""
    from transformer_engine.pytorch.tensor.localized_mxfp8 import _get_localization_context
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator

    monkeypatch.setenv("NVTE_NORM_FWD_USE_CUDNN", "1")
    shape = (4096, hidden_size)
    dtype = torch.bfloat16
    device = torch.device("cuda")
    eps = 1e-6
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((hidden_size,), dtype=dtype, device=device)
    bias = torch.randn((hidden_size,), dtype=dtype, device=device)
    quantizer = te.MXFP8Quantizer(
        fp8_dtype=te.DType.kFloat8E4M3,
        rowwise=True,
        columnwise=True,
    )
    # cuDNN normalization produces compact MXFP8 scales. LayerNormLinear
    # swizzles the scales for GEMM in a subsequent small kernel.
    quantizer.optimize_for_gemm = False
    baseline_output = quantizer.make_empty(shape, dtype=dtype, device=device)
    allocator = VMMRowSplitAllocator(device)
    rows_per_domain = shape[0] // 2
    partition_shape = (rows_per_domain, hidden_size)

    def make_partition_output(domain: int) -> MXFP8Tensor:
        return MXFP8Tensor(
            shape=partition_shape,
            dtype=dtype,
            rowwise_data=allocator.allocate_in_domain(
                partition_shape, torch.uint8, domain
            ),
            rowwise_scale_inv=torch.empty(
                quantizer.get_scale_shape(partition_shape, columnwise=False),
                dtype=torch.uint8,
                device=device,
            ),
            columnwise_data=allocator.allocate_in_domain(
                partition_shape, torch.uint8, domain
            ),
            columnwise_scale_inv=torch.empty(
                quantizer.get_scale_shape(partition_shape, columnwise=True),
                dtype=torch.uint8,
                device=device,
            ),
            fp8_dtype=quantizer.dtype,
            quantizer=quantizer,
            with_gemm_swizzled_scales=False,
            device=device,
        )

    ordinary_input_outputs = tuple(make_partition_output(domain) for domain in range(2))
    localized_input_outputs = tuple(make_partition_output(domain) for domain in range(2))
    split_ordinary_outputs = tuple(
        quantizer.make_empty(partition_shape, dtype=dtype, device=device) for _ in range(2)
    )
    ordinary_inputs = tuple(input_tensor.chunk(2, dim=0))
    localized_inputs = tuple(
        allocator.allocate_in_domain(partition_shape, dtype, domain) for domain in range(2)
    )
    for domain, local_input in enumerate(localized_inputs):
        row_start = domain * rows_per_domain
        local_input.copy_(input_tensor[row_start : row_start + rows_per_domain])

    _, _, streams = _get_localization_context(torch.cuda.current_device())
    assert len(streams) == 2
    ordinary_streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))

    total_sms = torch.cuda.get_device_properties(device).multi_processor_count
    sms_per_domain = total_sms // 2
    localized_sm_margin = total_sms - sms_per_domain

    def baseline() -> None:
        tex.layernorm_fwd(
            input_tensor,
            weight,
            bias,
            eps,
            baseline_output,
            quantizer,
            tex.DType.kBFloat16,
            0,
            False,
        )

    eager_events = {
        "ordinary": (
            torch.cuda.Event(enable_timing=False),
            tuple(torch.cuda.Event(enable_timing=False) for _ in range(2)),
        ),
        "localized": (
            torch.cuda.Event(enable_timing=False),
            tuple(torch.cuda.Event(enable_timing=False) for _ in range(2)),
        ),
        "split": (
            torch.cuda.Event(enable_timing=False),
            tuple(torch.cuda.Event(enable_timing=False) for _ in range(2)),
        ),
    }
    capture_events = []

    def partitioned(inputs, outputs, launch_streams, event_key: str) -> None:
        parent_stream = torch.cuda.current_stream(device)
        if torch.cuda.is_current_stream_capturing():
            fork_event = torch.cuda.Event(enable_timing=False)
            join_events = tuple(torch.cuda.Event(enable_timing=False) for _ in range(2))
            capture_events.extend((fork_event, *join_events))
        else:
            fork_event, join_events = eager_events[event_key]
        fork_event.record(parent_stream)
        for domain, (local_input, output, stream) in enumerate(
            zip(inputs, outputs, launch_streams)
        ):
            stream.wait_event(fork_event)
            with torch.cuda.stream(stream):
                tex.layernorm_fwd(
                    local_input,
                    weight,
                    bias,
                    eps,
                    output,
                    quantizer,
                    tex.DType.kBFloat16,
                    localized_sm_margin,
                    False,
                )
            join_events[domain].record(stream)
        for event in join_events:
            parent_stream.wait_event(event)

    def ordinary_input_localized_output() -> None:
        partitioned(
            ordinary_inputs,
            ordinary_input_outputs,
            streams,
            "ordinary",
        )

    def localized_input_and_output() -> None:
        partitioned(localized_inputs, localized_input_outputs, streams, "localized")

    def split_ordinary() -> None:
        partitioned(
            ordinary_inputs,
            split_ordinary_outputs,
            ordinary_streams,
            "split",
        )

    use_cuda_graph = os.getenv("MXFP8_LOCALIZATION_USE_CUDA_GRAPH") == "1"
    if use_cuda_graph:
        baseline_fn = _capture_cuda_graph(baseline).replay
        split_ordinary_fn = _capture_cuda_graph(split_ordinary).replay
        ordinary_input_fn = _capture_cuda_graph(ordinary_input_localized_output).replay
        localized_input_fn = _capture_cuda_graph(localized_input_and_output).replay
    else:
        baseline_fn = baseline
        split_ordinary_fn = split_ordinary
        ordinary_input_fn = ordinary_input_localized_output
        localized_input_fn = localized_input_and_output

    baseline_ms = _benchmark_ms(baseline_fn)
    split_ordinary_ms = _benchmark_ms(split_ordinary_fn)
    ordinary_input_ms = _benchmark_ms(ordinary_input_fn)
    localized_input_ms = _benchmark_ms(localized_input_fn)

    baseline_fn()
    split_ordinary_fn()
    ordinary_input_fn()
    localized_input_fn()
    torch.cuda.synchronize()
    for outputs in (
        split_ordinary_outputs,
        ordinary_input_outputs,
        localized_input_outputs,
    ):
        for domain, output in enumerate(outputs):
            for name in (
                "_rowwise_data",
                "_rowwise_scale_inv",
                "_columnwise_data",
                "_columnwise_scale_inv",
            ):
                partition = getattr(output, name)
                reference = getattr(baseline_output, name)
                partition_rows = partition.shape[0]
                row_start = domain * partition_rows
                reference_partition = reference[row_start : row_start + partition_rows]
                if name.endswith("_data"):
                    # Splitting the cuDNN normalization launch changes its grid
                    # and SM count. Values exactly on an FP8 rounding boundary
                    # may therefore differ by one adjacent encoding.
                    delta = partition.to(torch.int16) - reference_partition.to(torch.int16)
                    mismatches = torch.count_nonzero(delta).item()
                    assert mismatches <= 16, (
                        f"{name} domain {domain} has {mismatches} mismatches "
                        f"out of {partition.numel()} elements"
                    )
                    assert delta.abs().max().item() <= 1
                else:
                    torch.testing.assert_close(
                        partition,
                        reference_partition,
                        atol=0.0,
                        rtol=0.0,
                        msg=f"{name} mismatch in domain {domain}",
                    )

    execution = "CUDA Graph" if use_cuda_graph else "eager"
    print(
        f"\nMXFP8 fused LayerNorm+quant {shape} ({execution}):"
        f"\n  full-chip ordinary input/output:       {baseline_ms:.3f} ms"
        f"\n  split ordinary input/output:           {split_ordinary_ms:.3f} ms"
        f"\n  green ordinary input/VMM output:       {ordinary_input_ms:.3f} ms"
        f"\n  green VMM input/output:                {localized_input_ms:.3f} ms"
        f"\n  split-launch speedup:                  "
        f"{baseline_ms / split_ordinary_ms:.3f}x"
        f"\n  green vs split-ordinary speedup:       "
        f"{split_ordinary_ms / ordinary_input_ms:.3f}x"
        f"\n  output-only localization speedup:      "
        f"{baseline_ms / ordinary_input_ms:.3f}x"
        f"\n  input+output localization speedup:     "
        f"{baseline_ms / localized_input_ms:.3f}x"
    )

    allocator.close()


@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS") != "1",
    reason="Benchmark test - run with RUN_BENCHMARK_TESTS=1",
)
@pytest.mark.parametrize(
    ("layer_name", "output_size", "input_size"),
    [
        ("FC1", 2048, 4096),
        ("FC2", 4096, 1024),
    ],
    ids=["fc1", "fc2"],
)
def test_mxfp8_qwen35_grouped_gemm_localization_performance(
    layer_name: str,
    output_size: int,
    input_size: int,
    monkeypatch,
) -> None:
    """Measure standalone Qwen3.5 EP8 MXFP8 expert GEMMs with split-batch localization."""
    import transformer_engine_torch as tex_cpp
    from transformer_engine.pytorch.cpp_extensions import (
        general_grouped_gemm_for_grouped_tensor,
    )
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.localized_mxfp8 import (
        _get_localization_context,
    )
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator

    if tex_cpp.get_cublasLt_version() < 130300:
        pytest.skip("MXFP8 GroupedTensor GEMM requires cuBLASLt 13.3+.")
    if torch.cuda.get_device_capability() < (10, 0):
        pytest.skip("MXFP8 GroupedTensor GEMM requires SM100 or newer.")

    num_experts = 64
    experts_per_domain = num_experts // 2
    dtype = torch.bfloat16
    device = torch.device("cuda")
    m_values_string = os.getenv(
        "NVTE_GROUPED_GEMM_LOCALIZATION_M_SWEEP",
        "1,2,3,5,10,20,30,40",
    )
    try:
        m_values = [int(value.strip()) for value in m_values_string.split(",") if value.strip()]
    except ValueError as exc:
        pytest.fail(
            "NVTE_GROUPED_GEMM_LOCALIZATION_M_SWEEP must be a comma-separated "
            f"list of integers, got {m_values_string!r}: {exc}"
        )
    if not m_values or any(value <= 0 for value in m_values):
        pytest.fail("NVTE_GROUPED_GEMM_LOCALIZATION_M_SWEEP must contain positive integers")

    def quantize_uniform_group(num_groups: int, rows: int, cols: int) -> GroupedTensor:
        quantizer = te.MXFP8Quantizer(
            fp8_dtype=te.DType.kFloat8E4M3,
            rowwise=True,
            columnwise=False,
        )
        quantizer.optimize_for_gemm = True
        source = torch.full(
            (rows, cols),
            0.015625,
            dtype=dtype,
            device=device,
        )
        quantized = quantizer(source)
        del source
        assert quantized._with_gemm_swizzled_scales

        elements_per_group = rows * cols
        scales_per_group = quantized._rowwise_scale_inv.numel()
        first_dims = torch.full((num_groups,), rows, dtype=torch.int64, device=device)
        last_dims = torch.full((num_groups,), cols, dtype=torch.int64, device=device)
        tensor_offsets = (
            torch.arange(num_groups + 1, dtype=torch.int64, device=device)
            * elements_per_group
        )
        return GroupedTensor(
            shape=(1, num_groups * elements_per_group),
            dtype=dtype,
            num_tensors=num_groups,
            shapes=[(rows, cols)] * num_groups,
            quantizer=quantizer,
            data=quantized._rowwise_data.view(-1).repeat(num_groups),
            columnwise_data=None,
            scale_inv=quantized._rowwise_scale_inv.view(-1).repeat(num_groups),
            columnwise_scale_inv=None,
            first_dims=first_dims,
            last_dims=last_dims,
            tensor_offsets=tensor_offsets,
            offsets=[index * elements_per_group for index in range(num_groups + 1)],
            scale_inv_offsets=[
                index * scales_per_group for index in range(num_groups + 1)
            ],
            with_gemm_swizzled_scales=True,
        )

    def grouped_mxfp8_alias(
        source: GroupedTensor,
        num_groups: int,
        member_shape: tuple[int, int],
        rowwise_data: torch.Tensor,
        scale_inv: torch.Tensor,
    ) -> GroupedTensor:
        rows, cols = member_shape
        elements_per_group = rows * cols
        scales_per_group = scale_inv.numel() // num_groups
        return GroupedTensor(
            shape=(1, num_groups * elements_per_group),
            dtype=source.fake_dtype,
            num_tensors=num_groups,
            shapes=[member_shape] * num_groups,
            quantizer=source.quantizer,
            data=rowwise_data.view(-1),
            columnwise_data=None,
            scale_inv=scale_inv.view(-1),
            columnwise_scale_inv=None,
            first_dims=torch.full(
                (num_groups,),
                rows,
                dtype=torch.int64,
                device=rowwise_data.device,
            ),
            last_dims=torch.full(
                (num_groups,),
                cols,
                dtype=torch.int64,
                device=rowwise_data.device,
            ),
            tensor_offsets=(
                torch.arange(
                    num_groups + 1,
                    dtype=torch.int64,
                    device=rowwise_data.device,
                )
                * elements_per_group
            ),
            offsets=[
                index * elements_per_group for index in range(num_groups + 1)
            ],
            scale_inv_offsets=[
                index * scales_per_group for index in range(num_groups + 1)
            ],
            with_gemm_swizzled_scales=True,
        )

    def split_grouped_mxfp8(
        source: GroupedTensor,
        member_shape: tuple[int, int],
    ) -> tuple[GroupedTensor, GroupedTensor]:
        data_partitions = source.rowwise_data.view(-1).chunk(2)
        scale_partitions = source.scale_inv.view(-1).chunk(2)
        return tuple(
            grouped_mxfp8_alias(
                source,
                experts_per_domain,
                member_shape,
                data_partitions[domain],
                scale_partitions[domain],
            )
            for domain in range(2)
        )

    def localize_grouped_mxfp8(
        source_domains: tuple[GroupedTensor, GroupedTensor],
        member_shape: tuple[int, int],
        allocator: VMMRowSplitAllocator,
    ) -> tuple[GroupedTensor, GroupedTensor]:
        outputs = []
        for domain, source in enumerate(source_domains):
            data = allocator.allocate_in_domain(
                tuple(source.rowwise_data.shape),
                source.rowwise_data.dtype,
                domain,
            )
            scale_inv = allocator.allocate_in_domain(
                tuple(source.scale_inv.shape),
                source.scale_inv.dtype,
                domain,
            )
            data.copy_(source.rowwise_data)
            scale_inv.copy_(source.scale_inv)
            outputs.append(
                grouped_mxfp8_alias(
                    source,
                    experts_per_domain,
                    member_shape,
                    data,
                    scale_inv,
                )
            )
        return outputs[0], outputs[1]

    def grouped_output(
        num_groups: int,
        member_shape: tuple[int, int],
        data: torch.Tensor,
    ) -> GroupedTensor:
        return GroupedTensor.make_grouped_tensor_from_rowwise_data(
            num_tensors=num_groups,
            tensor_shape=member_shape,
            rowwise_data=data,
            dtype=dtype,
        )

    weight = quantize_uniform_group(num_experts, output_size, input_size)
    ordinary_weight_domains = split_grouped_mxfp8(
        weight,
        (output_size, input_size),
    )
    weight_allocator = VMMRowSplitAllocator(device)
    localized_weight_domains = localize_grouped_mxfp8(
        ordinary_weight_domains,
        (output_size, input_size),
        weight_allocator,
    )

    _, _, green_streams = _get_localization_context(torch.cuda.current_device())
    ordinary_streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))
    total_sms = torch.cuda.get_device_properties(device).multi_processor_count
    sms_per_domain = total_sms // 2
    localized_sm_margin = total_sms - sms_per_domain
    use_cuda_graph = os.getenv("MXFP8_LOCALIZATION_USE_CUDA_GRAPH") == "1"

    for m_exp in m_values:
        iteration_allocator = VMMRowSplitAllocator(device)
        grouped_input = quantize_uniform_group(num_experts, m_exp, input_size)
        ordinary_input_domains = split_grouped_mxfp8(
            grouped_input,
            (m_exp, input_size),
        )
        localized_input_domains = localize_grouped_mxfp8(
            ordinary_input_domains,
            (m_exp, input_size),
            iteration_allocator,
        )

        output_shape = (num_experts, m_exp, output_size)
        baseline_output = grouped_output(
            num_experts,
            (m_exp, output_size),
            torch.empty(output_shape, dtype=dtype, device=device),
        )
        split_output_buffer = torch.empty(output_shape, dtype=dtype, device=device)
        green_output_buffer = torch.empty(output_shape, dtype=dtype, device=device)
        split_outputs = tuple(
            grouped_output(
                experts_per_domain,
                (m_exp, output_size),
                partition,
            )
            for partition in split_output_buffer.chunk(2, dim=0)
        )
        green_outputs = tuple(
            grouped_output(
                experts_per_domain,
                (m_exp, output_size),
                partition,
            )
            for partition in green_output_buffer.chunk(2, dim=0)
        )
        localized_outputs = tuple(
            grouped_output(
                experts_per_domain,
                (m_exp, output_size),
                iteration_allocator.allocate_in_domain(
                    (experts_per_domain, m_exp, output_size),
                    dtype,
                    domain,
                ),
            )
            for domain in range(2)
        )

        def launch_grouped_gemm(
            local_weight: GroupedTensor,
            local_input: GroupedTensor,
            output: GroupedTensor,
            workspace_slot: int,
        ) -> None:
            general_grouped_gemm_for_grouped_tensor(
                local_weight,
                local_input,
                output,
                layout="TN",
                workspace_slot=workspace_slot,
            )

        def baseline() -> None:
            launch_grouped_gemm(weight, grouped_input, baseline_output, 0)

        eager_events = {
            key: (
                torch.cuda.Event(enable_timing=False),
                tuple(torch.cuda.Event(enable_timing=False) for _ in range(2)),
            )
            for key in ("split", "green", "localized")
        }
        capture_events = []

        def partitioned(
            weights,
            inputs,
            outputs,
            launch_streams,
            workspace_slots,
            event_key: str,
        ) -> None:
            parent_stream = torch.cuda.current_stream(device)
            if torch.cuda.is_current_stream_capturing():
                fork_event = torch.cuda.Event(enable_timing=False)
                join_events = tuple(torch.cuda.Event(enable_timing=False) for _ in range(2))
                capture_events.extend((fork_event, *join_events))
            else:
                fork_event, join_events = eager_events[event_key]
            fork_event.record(parent_stream)
            for domain, stream in enumerate(launch_streams):
                stream.wait_event(fork_event)
                with torch.cuda.stream(stream):
                    launch_grouped_gemm(
                        weights[domain],
                        inputs[domain],
                        outputs[domain],
                        workspace_slots[domain],
                    )
                join_events[domain].record(stream)
            for event in join_events:
                parent_stream.wait_event(event)

        def split_ordinary() -> None:
            partitioned(
                ordinary_weight_domains,
                ordinary_input_domains,
                split_outputs,
                ordinary_streams,
                (1, 2),
                "split",
            )

        def green_ordinary() -> None:
            partitioned(
                ordinary_weight_domains,
                ordinary_input_domains,
                green_outputs,
                green_streams,
                (3, 4),
                "green",
            )

        def green_localized() -> None:
            partitioned(
                localized_weight_domains,
                localized_input_domains,
                localized_outputs,
                green_streams,
                (5, 6),
                "localized",
            )

        monkeypatch.delenv("NVTE_EXT_MARGIN_SM", raising=False)
        if use_cuda_graph:
            baseline_fn = _capture_cuda_graph(baseline).replay
        else:
            baseline_fn = baseline
        baseline_ms = _benchmark_ms(baseline_fn)

        monkeypatch.setenv("NVTE_EXT_MARGIN_SM", str(localized_sm_margin))
        if use_cuda_graph:
            split_fn = _capture_cuda_graph(split_ordinary).replay
            green_fn = _capture_cuda_graph(green_ordinary).replay
            localized_fn = _capture_cuda_graph(green_localized).replay
        else:
            split_fn = split_ordinary
            green_fn = green_ordinary
            localized_fn = green_localized
        split_ms = _benchmark_ms(split_fn)
        green_ms = _benchmark_ms(green_fn)
        localized_ms = _benchmark_ms(localized_fn)

        split_fn()
        green_fn()
        localized_fn()
        torch.cuda.synchronize()
        reference = baseline_output.rowwise_data.view(output_shape)
        for name, outputs in (
            ("split ordinary", split_outputs),
            ("green ordinary", green_outputs),
            ("green localized", localized_outputs),
        ):
            candidate = torch.cat(
                [
                    output.rowwise_data.view(
                        experts_per_domain,
                        m_exp,
                        output_size,
                    )
                    for output in outputs
                ],
                dim=0,
            )
            torch.testing.assert_close(
                candidate,
                reference,
                atol=0.25,
                rtol=0.05,
                msg=f"{layer_name} M_exp={m_exp} {name} mismatch",
            )

        execution = "CUDA Graph" if use_cuda_graph else "eager"
        print(
            f"\nQwen3.5 EP8 MXFP8 grouped GEMM {layer_name} "
            f"E=512/64, M_exp={m_exp}, N={output_size}, K={input_size} "
            f"({execution}):"
            f"\n  full ordinary:                 {baseline_ms:.3f} ms"
            f"\n  split ordinary experts:        {split_ms:.3f} ms"
            f"\n  green ordinary memory:         {green_ms:.3f} ms"
            f"\n  green localized A/B/output:    {localized_ms:.3f} ms"
            f"\n  split-launch speedup:          {baseline_ms / split_ms:.3f}x"
            f"\n  green-context contribution:    {split_ms / green_ms:.3f}x"
            f"\n  memory-locality contribution:  {green_ms / localized_ms:.3f}x"
            f"\n  overall speedup:               {baseline_ms / localized_ms:.3f}x"
        )

        del baseline_fn, split_fn, green_fn, localized_fn
        torch.cuda.synchronize()
        iteration_allocator.close()

    weight_allocator.close()
