# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Output, gradient and graph coverage for fused chunk sorting and padding."""

import pytest
import torch
import torch.nn.functional as F

from transformer_engine.pytorch.chunk_padding import (
    moe_sort_chunks_and_pad,
    moe_unpad_and_restore_chunks,
)


def reference(inp, probs, sizes, order, output_sizes):
    """Independently split, reorder and pad tokens and probabilities."""
    chunks = inp.split(sizes)
    prob_chunks = probs.split(sizes) if probs is not None else None
    tokens, probabilities = [], []
    for index, size in zip(order, output_sizes):
        padding = size - sizes[index]
        tokens.append(F.pad(chunks[index], (0, 0, 0, padding)))
        if probs is not None:
            probabilities.append(F.pad(prob_chunks[index], (0, padding)))
    return torch.cat(tokens), torch.cat(probabilities) if probs is not None else None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("prob_dtype", [None, torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "sizes,order,output_sizes",
    [
        ([0, 3, 5], [2, 0, 1], [5, 3, 8]),
        ([2, 0, 129, 1], [3, 1, 2, 0], [1, 7, 129, 127]),
        ([0, 0, 0], [2, 0, 1], [0, 0, 0]),
        ([0, 0, 0], [2, 0, 1], [4, 0, 4]),
        ([7], [0], [7]),
    ],
)
def test_outputs_and_gradients(dtype, prob_dtype, sizes, order, output_sizes):
    """Check every element, including arbitrary gradients at padded rows."""
    count = sum(sizes)
    with_probs = prob_dtype is not None
    inp = torch.randn(count, 130, device="cuda", dtype=dtype)[:, ::2].requires_grad_()
    probs = (
        torch.rand(count * 2, device="cuda", dtype=prob_dtype)[::2].requires_grad_()
        if with_probs
        else None
    )
    expected_inp = inp.detach().clone().requires_grad_()
    expected_probs = probs.detach().clone().requires_grad_() if with_probs else None
    metadata = [
        torch.tensor(values, device="cuda", dtype=torch.int64)
        for values in (sizes, order, output_sizes)
    ]
    actual, actual_probs, row_map = moe_sort_chunks_and_pad(
        inp, *metadata, sum(output_sizes), probs
    )
    expected, expected_p = reference(expected_inp, expected_probs, sizes, order, output_sizes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    restored = moe_unpad_and_restore_chunks(actual, row_map, count)
    torch.testing.assert_close(restored, inp, rtol=0, atol=0)
    outputs, references = [actual], [expected]
    if with_probs:
        torch.testing.assert_close(actual_probs, expected_p, rtol=0, atol=0)
        outputs.append(actual_probs)
        references.append(expected_p)
    grads = [torch.randn_like(output) for output in outputs]
    torch.autograd.backward(outputs, grads)
    torch.autograd.backward(references, grads)
    torch.testing.assert_close(inp.grad, expected_inp.grad, rtol=0, atol=0)
    if with_probs:
        torch.testing.assert_close(probs.grad, expected_probs.grad, rtol=0, atol=0)
    expert_output = torch.randn(
        sum(output_sizes), 37, device="cuda", dtype=dtype, requires_grad=True
    )
    restored = moe_unpad_and_restore_chunks(expert_output, row_map, count)
    dy = torch.randn_like(restored)
    restored.backward(dy)
    valid = row_map >= 0
    torch.testing.assert_close(expert_output.grad[valid], dy[row_map[valid].long()], rtol=0, atol=0)
    assert torch.count_nonzero(expert_output.grad[~valid]).item() == 0


def test_probability_only_gradient():
    """An unused token output must not prevent probability differentiation."""
    inp = torch.randn(5, 7, device="cuda", requires_grad=True)
    probs = torch.rand(5, device="cuda", requires_grad=True)
    meta = [torch.tensor(v, device="cuda") for v in ([2, 3], [1, 0], [4, 4])]
    _, output_probs, _ = moe_sort_chunks_and_pad(inp, *meta, 8, probs)
    output_probs.sum().backward()
    torch.testing.assert_close(probs.grad, torch.ones_like(probs), rtol=0, atol=0)
    torch.testing.assert_close(inp.grad, torch.zeros_like(inp), rtol=0, atol=0)


@pytest.mark.parametrize("compiled", [False, True])
def test_dynamic_metadata_graph_replay(compiled):
    """Replay with changed routing and values at fixed input/output sizes."""
    inp = torch.randn(8, 65, device="cuda", requires_grad=True)
    probs = torch.rand(8, device="cuda", requires_grad=True)
    sizes = torch.tensor([2, 3, 3], device="cuda")
    order = torch.tensor([2, 0, 1], device="cuda")
    output_sizes = torch.tensor([4, 4, 4], device="cuda")

    def run():
        output, output_probs, row_map = moe_sort_chunks_and_pad(
            inp, sizes, order, output_sizes, 12, probs
        )
        restored = moe_unpad_and_restore_chunks(output, row_map, 8)
        return output, output_probs, restored

    fn = torch.compile(run, fullgraph=True) if compiled else run
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            inp.grad = probs.grad = None
            output, output_probs, restored = fn()
            (output.sum() + output_probs.sum() + restored.sum()).backward()
            del output, output_probs, restored
    torch.cuda.current_stream().wait_stream(stream)
    inp.grad = probs.grad = None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output, output_probs, restored = fn()
        (output.sum() + output_probs.sum() + restored.sum()).backward()
    for values, permutation in (([1, 4, 3], [1, 0, 2]), ([4, 0, 4], [2, 1, 0])):
        with torch.no_grad():
            inp.normal_()
            probs.uniform_()
            sizes.copy_(torch.tensor(values, device="cuda"))
            order.copy_(torch.tensor(permutation, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected, expected_p = reference(inp, probs, values, permutation, [4, 4, 4])
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(output_probs, expected_p, rtol=0, atol=0)
        torch.testing.assert_close(restored, inp, rtol=0, atol=0)
        torch.testing.assert_close(inp.grad, torch.full_like(inp, 2), rtol=0, atol=0)
        torch.testing.assert_close(probs.grad, torch.ones_like(probs), rtol=0, atol=0)
    graph.reset()
