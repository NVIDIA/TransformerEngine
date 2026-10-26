# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""cuBLAS workspace ownership and numerical checks for independent CUDA streams."""

import gc
import weakref

import pytest
import torch
from transformer_engine.pytorch.cpp_extensions import gemm

from transformer_engine.pytorch.cpp_extensions.gemm import (
    general_gemm,
    general_grouped_gemm,
    get_cublas_workspace,
)


@pytest.mark.parametrize("ub,grouped_gemm", [(False, False), (True, False), (False, True)])
def test_live_workspaces_do_not_alias(ub, grouped_gemm):
    """Live invocations own distinct scratch, even on the same capture stream."""
    device = torch.cuda.current_device()
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    workspaces = []
    for stream in streams:
        with torch.cuda.stream(stream):
            first = get_cublas_workspace(device, ub, grouped_gemm)
            workspaces.append(first if grouped_gemm else [first])
            second = get_cublas_workspace(device, ub, grouped_gemm)
            workspaces.append(second if grouped_gemm else [second])
    assert len(workspaces[0]) == len(workspaces[1])
    pointers = [w.data_ptr() for invocation in workspaces for w in invocation]
    assert len(pointers) == len(set(pointers))


@pytest.mark.parametrize("ub,grouped_gemm", [(False, False), (True, False), (False, True)])
def test_temporary_streams_release_workspaces(ub, grouped_gemm):
    """Scratch is returned to the allocator when the invocation goes out of scope."""
    device = torch.cuda.current_device()
    torch.cuda.synchronize()
    gc.collect()
    allocated = torch.cuda.memory_allocated()
    for _ in range(8):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            workspace = get_cublas_workspace(device, ub, grouped_gemm)
            tensors = workspace if grouped_gemm else [workspace]
            refs = [weakref.ref(tensor) for tensor in tensors]
            for tensor in tensors:
                tensor.fill_(1)
            del tensor, tensors, workspace
        stream.synchronize()
        del stream
        assert all(ref() is None for ref in refs)
    assert torch.cuda.memory_allocated() == allocated


@pytest.mark.parametrize("ub,grouped_gemm", [(False, False), (True, False), (False, True)])
def test_graph_workspaces_survive_replay_and_are_released(ub, grouped_gemm):
    """Different captures on one stream must not borrow an earlier graph's scratch."""
    device = torch.cuda.current_device()
    capture_stream = torch.cuda.Stream()
    replay_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    # Initialize the allocator/library state outside the graph, on a different stream.
    workspace = get_cublas_workspace(device, ub, grouped_gemm)
    count = len(workspace) if grouped_gemm else 1
    del workspace
    outputs = [torch.empty(count, dtype=torch.uint8, device=device) for _ in range(2)]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    for _ in range(3):
        graphs, pointers = [], []
        for value, output in enumerate(outputs, start=1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                workspace = get_cublas_workspace(device, ub, grouped_gemm)
                tensors = workspace if grouped_gemm else [workspace]
                pointers.append({tensor.data_ptr() for tensor in tensors})
                refs = [weakref.ref(tensor) for tensor in tensors]
                for index, tensor in enumerate(tensors):
                    tensor.fill_(value)
                    output[index : index + 1].copy_(tensor[:1])
                del tensor, tensors, workspace
            assert all(ref() is None for ref in refs)
            graphs.append(graph)
        assert pointers[0].isdisjoint(pointers[1])
        for _ in range(8):
            for graph, stream in zip(graphs, replay_streams):
                with torch.cuda.stream(stream):
                    graph.replay()
        torch.cuda.synchronize()
        for value, output in enumerate(outputs, start=1):
            torch.testing.assert_close(output, torch.full_like(output, value), rtol=0, atol=0)
        for graph in graphs:
            graph.reset()
        del graph, graphs
        gc.collect()
        torch.cuda.empty_cache()
        assert torch.cuda.memory_allocated() == allocated
        assert torch.cuda.memory_reserved() == reserved


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("execution", ["eager", "graph"])
@pytest.mark.parametrize("layout", ["NT", "NN"])
def test_concurrent_gemms_match_exact_reference(dtype, execution, layout):
    """Concurrent router wgrad and dgrad must own scratch in eager and graph execution."""
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    hidden, experts = 4096, 128
    if layout == "NT":
        # wgrad = grad_output.T @ input; reduce over tokens.
        tokens = reduction = 16384
        input_shape, output_shape = (tokens, hidden), (experts, hidden)
    else:
        # dgrad = grad_output @ weight; match the router's NN [8192, 128, 4096] GEMM.
        tokens, reduction = 8192, experts
        input_shape, output_shape = (experts, hidden), (tokens, hidden)
    inputs = [torch.full(input_shape, value, dtype=dtype, device="cuda") for value in (1.0, 2.0)]
    gradients = [
        torch.full((tokens, experts), value, dtype=dtype, device="cuda") for value in (1.0, 3.0)
    ]
    # Both operands and these analytic results are exactly representable in either dtype.
    expected = [reduction, reduction * 6]
    ready = torch.cuda.Event()
    ready.record()
    for stream in streams:
        stream.wait_event(ready)

    outputs, graphs, checks = [], [], []
    if execution == "graph":
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_event(ready)
        for inp, grad in zip(inputs, gradients):
            with torch.cuda.stream(capture_stream):
                general_gemm(inp, grad, dtype, layout=layout, grad=True)
            capture_stream.synchronize()
            # Separate private pools, captured on one stream and replayed on two others.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture_stream):
                out, *_ = general_gemm(inp, grad, dtype, layout=layout, grad=True)
            assert out.shape == output_shape
            assert out.dtype == dtype
            graphs.append(graph)
            outputs.append(out)
            del out
    for _ in range(8):
        for index, (stream, inp, grad) in enumerate(zip(streams, inputs, gradients)):
            with torch.cuda.stream(stream):
                if execution == "graph":
                    graphs[index].replay()
                else:
                    out, *_ = general_gemm(inp, grad, dtype, layout=layout, grad=True)
                    assert out.shape == output_shape
                    assert out.dtype == dtype
                    outputs.append(out)
                    del out
        if execution == "eager":
            # Submit both GEMMs before the checks. Keep only this pair of large NN outputs.
            for index, (stream, out) in enumerate(zip(streams, outputs)):
                with torch.cuda.stream(stream):
                    checks.append(torch.all(out == expected[index]))
            del out
            outputs.clear()
    for stream in streams:
        stream.synchronize()
    for index, out in enumerate(outputs):
        checks.append(torch.all(out == expected[index]))
    torch.testing.assert_close(
        torch.stack(checks),
        torch.ones(len(checks), dtype=torch.bool, device="cuda"),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("execution", ["eager", "graph"])
def test_grouped_gemms_match_exact_reference(dtype, execution):
    """Internal GEMM streams must join the caller before its scratch is recycled."""
    reduction, hidden, experts, count = 8192, 512, 64, 4
    inputs = [torch.ones((reduction, hidden), dtype=dtype, device="cuda") for _ in range(count)]
    gradients = [
        torch.full((reduction, experts), index + 1, dtype=dtype, device="cuda")
        for index in range(count)
    ]
    outputs = [torch.empty((experts, hidden), dtype=dtype, device="cuda") for _ in range(count)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        general_grouped_gemm(
            inputs,
            gradients,
            outputs,
            [None] * count,
            dtype,
            layout="NT",
            m_splits=[experts] * count,
            grad=True,
        )

    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    if execution == "graph":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    for _ in range(8):
        with torch.cuda.stream(stream):
            if execution == "graph":
                graph.replay()
            else:
                run()
            # Recycle same-size allocations immediately after the internal-stream join.
            scratch = get_cublas_workspace(torch.cuda.current_device(), False, True)
            for tensor in scratch:
                tensor.fill_(0)
            del tensor, scratch
    stream.synchronize()
    for index, output in enumerate(outputs):
        expected = torch.full_like(output, reduction * (index + 1))
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_grouped_gemm_exception_joins_helper_streams(monkeypatch):
    """A later native rejection must join earlier work before releasing scratch."""
    monkeypatch.setenv("NVTE_USE_CUTLASS_GROUPED_GEMM", "0")
    dtype = torch.bfloat16
    reduction, hidden, experts = 8192, 512, 64
    inputs = [torch.ones((reduction, hidden), dtype=dtype, device="cuda") for _ in range(2)]
    gradients = [
        torch.full((reduction, experts), value, dtype=dtype, device="cuda") for value in (1, 2)
    ]
    outputs = [torch.empty((experts, hidden), dtype=dtype, device="cuda") for _ in range(2)]
    biases = [
        torch.empty(0, dtype=dtype, device="cuda"),
        torch.zeros(hidden, dtype=dtype, device="cuda"),
    ]
    invalid_biases = [biases[0], torch.zeros(hidden, dtype=torch.float32, device="cuda")]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run(bias):
        general_grouped_gemm(
            inputs,
            gradients,
            outputs,
            [None, None],
            dtype,
            layout="NT",
            m_splits=[experts, experts],
            bias=bias,
            use_bias=True,
        )

    def reject_and_churn():
        for output in outputs:
            output.fill_(-1)
        # Incompatible input shapes are rejected by getGemmOutputShape for ALL
        # groups before any launch. A mismatched second bias instead reaches
        # cublas_gemm's native check AFTER the first GEMM has been submitted.
        with pytest.raises(
            RuntimeError, match="Non-FP8/FP4 GEMM bias dtype must match output dtype"
        ):
            run(invalid_biases)
        scratch = get_cublas_workspace(torch.cuda.current_device(), False, True)
        for tensor in scratch:
            tensor.fill_(0)
        del tensor, scratch
        # These caller-stream consumers must follow the first helper's GEMM.
        return [output.clone() for output in outputs]

    # Initialize cuBLAS and all used helper streams before capture.
    with torch.cuda.stream(stream):
        run(biases)
    stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        # The outer context restores the stream even if capture_end raises.
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph, stream=stream):
                observed = reject_and_churn()
        # Without exception-path joins, capture_end fails with
        # cudaErrorStreamCaptureUnjoined. No sleep or allocator reuse luck is
        # needed to detect the missing dependency. Resetting outputs in the
        # graph and changing inputs also prevents warmup results from passing.
        for value in (1, 2):
            with torch.cuda.stream(stream):
                inputs[0].fill_(value)
                graph.replay()
            stream.synchronize()
            torch.testing.assert_close(
                observed[0], torch.full_like(observed[0], reduction * value), rtol=0, atol=0
            )
            torch.testing.assert_close(
                observed[1], torch.full_like(observed[1], -1), rtol=0, atol=0
            )

        # Exercise eager unwinding too, then recover with a valid grouped call.
        with torch.cuda.stream(stream):
            inputs[0].fill_(3)
            observed = reject_and_churn()
        stream.synchronize()
        torch.testing.assert_close(
            observed[0], torch.full_like(observed[0], reduction * 3), rtol=0, atol=0
        )
        torch.testing.assert_close(observed[1], torch.full_like(observed[1], -1), rtol=0, atol=0)
        with torch.cuda.stream(stream):
            run(biases)
        stream.synchronize()
        for output, value in zip(outputs, (3, 2)):
            torch.testing.assert_close(
                output, torch.full_like(output, reduction * value), rtol=0, atol=0
            )
    finally:
        graph.reset()


@pytest.mark.skipif(
    not hasattr(torch.library, "custom_op")
    or not hasattr(torch.compiler, "cudagraph_mark_step_begin"),
    reason="Custom operators and CUDA graph trees require newer PyTorch",
)
def test_compiled_gemm_graph_generations(monkeypatch):
    """An opaque GEMM's scratch must remain valid when graph trees are rerecorded."""
    original = gemm.tex.generic_gemm
    captured_workspace_requests = []

    def observe_native_gemm(*args, **kwargs):
        if torch.cuda.is_current_stream_capturing():
            # Record only scalar metadata, never tensors or capture-pool owners.
            captured_workspace_requests.append((args[12] is None, args[13]))
        return original(*args, **kwargs)

    monkeypatch.setattr(gemm.tex, "generic_gemm", observe_native_gemm)

    @torch.library.custom_op("te_workspace_test::gemm", mutates_args=())
    def run(inp: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        return general_gemm(inp, grad, inp.dtype, layout="NT", grad=True)[0]

    @run.register_fake
    def fake(inp, grad):
        return inp.new_empty((grad.shape[1], inp.shape[1]))

    torch._dynamo.reset()
    compiled = torch.compile(run, fullgraph=True, mode="reduce-overhead", dynamic=False)
    for reduction, value in [(8192, 1), (16384, 2), (8192, 3)]:
        inp = torch.full((reduction, 512), value, dtype=torch.bfloat16, device="cuda")
        grad = torch.ones((reduction, 64), dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            out = compiled(inp, grad)
            torch.testing.assert_close(out, torch.full_like(out, reduction * value), rtol=0, atol=0)
            del out
    assert (
        len(captured_workspace_requests) >= 2
    ), "Expected CUDA graph captures for different shapes"
    assert all(
        native and size == 0 for native, size in captured_workspace_requests
    ), "Captured GEMMs must request invocation-local native scratch with workspace=None, size=0"
    del compiled
    torch._dynamo.reset()
