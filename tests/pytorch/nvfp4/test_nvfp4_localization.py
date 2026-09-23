# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Localization tests for NVFP4 RHT with stochastic rounding."""

import os

import pytest
import torch

import transformer_engine.pytorch as te


NVFP4_AVAILABLE, NO_NVFP4_REASON = te.is_nvfp4_available(return_reason=True)


def _localization_available() -> bool:
    if not torch.cuda.is_available():
        return False
    if os.getenv("NVTE_FORCE_DRIVER_LOCALIZATION", "0") != "1":
        try:
            from torch.cuda.green_contexts import is_localization_supported

            if is_localization_supported is not None:
                try:
                    if is_localization_supported(torch.cuda.current_device()):
                        return True
                except TypeError:
                    if is_localization_supported():
                        return True
        except (ImportError, RuntimeError):
            pass
    try:
        from transformer_engine.pytorch.tensor.driver_localization import (
            is_driver_localization_supported,
        )

        return is_driver_localization_supported(torch.cuda.current_device())
    except (ImportError, RuntimeError):
        return False


def _capture_cuda_graph(function):
    function()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        function()
    torch.cuda.synchronize()
    return graph


def _benchmark_ms(function, warmup: int = 20, iterations: int = 100) -> float:
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


def _make_quantizer(*, stochastic_rounding: bool) -> te.NVFP4Quantizer:
    quantizer = te.NVFP4Quantizer(
        fp4_dtype=te.DType.kFloat4E2M1,
        rowwise=True,
        columnwise=True,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=True,
        with_post_rht_amax=True,
        stochastic_rounding=stochastic_rounding,
    )
    quantizer.optimize_for_gemm = False
    return quantizer


def _localize_output_data(output, allocator, domain: int) -> None:
    """Replace the large packed data buffers with domain-local VMM storage."""
    for name in ("_rowwise_data", "_columnwise_data"):
        tensor = getattr(output, name)
        setattr(
            output,
            name,
            allocator.allocate_in_domain(tuple(tensor.shape), tensor.dtype, domain),
        )


@pytest.mark.skipif(not NVFP4_AVAILABLE, reason=NO_NVFP4_REASON)
@pytest.mark.skipif(not _localization_available(), reason="CUDA localization is unavailable")
@pytest.mark.skipif(
    os.getenv("RUN_BENCHMARK_TESTS", "0") != "1",
    reason="Benchmark test - set RUN_BENCHMARK_TESTS=1",
)
@pytest.mark.parametrize("shape", [(16384, 4096)])
def test_nvfp4_rht_sr_localization_performance(shape) -> None:
    """Benchmark the localized RHT-amax and row/column RHT+SR quant kernels."""
    from transformer_engine.pytorch.tensor.localized_mxfp8 import _get_localization_context
    from transformer_engine.pytorch.tensor.vmm import VMMRowSplitAllocator

    dtype = torch.bfloat16
    device = torch.device("cuda")
    rows_per_domain = shape[0] // 2
    partition_shape = (rows_per_domain, shape[1])
    input_tensor = torch.randn(shape, dtype=dtype, device=device)
    ordinary_inputs = tuple(input_tensor.chunk(2, dim=0))
    quantizer = _make_quantizer(stochastic_rounding=True)
    allocator = VMMRowSplitAllocator(device)

    full_output = quantizer.make_empty(shape, dtype=dtype, device=device)
    split_ordinary_outputs = tuple(
        quantizer.make_empty(partition_shape, dtype=dtype, device=device) for _ in range(2)
    )
    localized_output_only = tuple(
        quantizer.make_empty(partition_shape, dtype=dtype, device=device) for _ in range(2)
    )
    localized_input_output = tuple(
        quantizer.make_empty(partition_shape, dtype=dtype, device=device) for _ in range(2)
    )
    for domain in range(2):
        _localize_output_data(localized_output_only[domain], allocator, domain)
        _localize_output_data(localized_input_output[domain], allocator, domain)

    localized_inputs = tuple(
        allocator.allocate_in_domain(partition_shape, dtype, domain) for domain in range(2)
    )
    for domain, local_input in enumerate(localized_inputs):
        local_input.copy_(ordinary_inputs[domain])

    _, _, green_streams = _get_localization_context(torch.cuda.current_device())
    ordinary_streams = tuple(torch.cuda.Stream(device=device) for _ in range(2))
    eager_events = {
        key: (
            torch.cuda.Event(enable_timing=False),
            tuple(torch.cuda.Event(enable_timing=False) for _ in range(2)),
        )
        for key in ("ordinary", "output", "input_output")
    }
    capture_events = []

    def partitioned(inputs, outputs, streams, event_key: str) -> None:
        parent_stream = torch.cuda.current_stream(device)
        if torch.cuda.is_current_stream_capturing():
            fork_event = torch.cuda.Event(enable_timing=False)
            join_events = tuple(torch.cuda.Event(enable_timing=False) for _ in range(2))
            capture_events.extend((fork_event, *join_events))
        else:
            fork_event, join_events = eager_events[event_key]
        fork_event.record(parent_stream)
        for input_, output, stream, join_event in zip(inputs, outputs, streams, join_events):
            stream.wait_event(fork_event)
            with torch.cuda.stream(stream):
                quantizer.update_quantized(input_, output)
            join_event.record(stream)
        for event in join_events:
            parent_stream.wait_event(event)

    def full_ordinary() -> None:
        quantizer.update_quantized(input_tensor, full_output)

    def split_ordinary() -> None:
        partitioned(
            ordinary_inputs,
            split_ordinary_outputs,
            ordinary_streams,
            "ordinary",
        )

    def green_output_only() -> None:
        partitioned(
            ordinary_inputs,
            localized_output_only,
            green_streams,
            "output",
        )

    def green_input_output() -> None:
        partitioned(
            localized_inputs,
            localized_input_output,
            green_streams,
            "input_output",
        )

    use_cuda_graph = os.getenv("NVFP4_LOCALIZATION_USE_CUDA_GRAPH", "1") == "1"
    functions = {
        "full": full_ordinary,
        "split": split_ordinary,
        "output": green_output_only,
        "input_output": green_input_output,
    }
    if use_cuda_graph:
        functions = {
            name: _capture_cuda_graph(function).replay for name, function in functions.items()
        }

    timings = {name: _benchmark_ms(function) for name, function in functions.items()}
    for function in functions.values():
        function()
    torch.cuda.synchronize()

    # Stochastic rounding changes packed data, but amax and scale selection
    # must not depend on stream type or VMM placement for the same partition.
    for reference_outputs, localized_outputs in (
        (split_ordinary_outputs, localized_output_only),
        (split_ordinary_outputs, localized_input_output),
    ):
        for reference, localized in zip(reference_outputs, localized_outputs):
            for name in (
                "_rowwise_scale_inv",
                "_columnwise_scale_inv",
                "_amax_rowwise",
                "_amax_columnwise",
            ):
                torch.testing.assert_close(
                    getattr(localized, name),
                    getattr(reference, name),
                    atol=0.0,
                    rtol=0.0,
                    msg=f"{name} changed with localization",
                )

    execution = "CUDA Graph" if use_cuda_graph else "eager"
    print(
        f"\nNVFP4 RHT+SR quant {shape} ({execution}):"
        f"\n  full ordinary:                 {timings['full']:.3f} ms"
        f"\n  split ordinary:                {timings['split']:.3f} ms"
        f"\n  green ordinary input/VMM out:  {timings['output']:.3f} ms"
        f"\n  green VMM input/output:        {timings['input_output']:.3f} ms"
        f"\n  split-launch speedup:          {timings['full'] / timings['split']:.3f}x"
        f"\n  green vs split speedup:        {timings['split'] / timings['output']:.3f}x"
        f"\n  input+output speedup:          "
        f"{timings['split'] / timings['input_output']:.3f}x"
    )

    allocator.close()
