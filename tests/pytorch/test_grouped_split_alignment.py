# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Device-side alignment checks in graph-safe grouped offset scans."""

import multiprocessing

import pytest
import torch

import transformer_engine.pytorch  # Load the common library before the extension.
import transformer_engine_torch as tex


STRIDES = [1, 1, 128, 256, 64, 512, 2, 4, 8]


def _replay_invalid_splits(num_outputs, alignment, num_splits, bad_index, bad_value, connection):
    """Run an invalid replay in its own CUDA context, then report where it failed."""
    try:
        splits = torch.full((num_splits,), 256, device="cuda", dtype=torch.int64)

        def scan():
            return tex.splits_to_offsets_multi(
                splits,
                torch.device("cuda"),
                strides=STRIDES[:num_outputs],
                include_leading_zero=[True] * num_outputs,
                dtypes=[torch.int64] * num_outputs,
                bulk_allocate=num_outputs > 1,
                split_alignment=alignment,
            )

        scan()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            scan()
    except Exception as error:
        connection.send(("setup_error", str(error)))
        connection.close()
        return

    connection.send(("captured", ""))
    try:
        splits[bad_index] = bad_value
        graph.replay()
        torch.cuda.synchronize()
    except RuntimeError as error:
        connection.send(("cuda_error", str(error)))
    else:
        connection.send(("unexpected_success", ""))
    finally:
        connection.close()


@pytest.mark.parametrize(
    "num_outputs,alignment,num_splits,bad_index,bad_value",
    [
        (1, 128, 2, 0, 64),
        (4, 128, 2, 0, 64),
        (6, 256, 2, 0, 128),
        (6, 256, 2, 0, -256),
        (9, 256, 257, 256, 128),
    ],
)
def test_multi_offsets_alignment_graph_replay(
    num_outputs, alignment, num_splits, bad_index, bad_value, capfd
):
    """Aligned splits produce correct offsets; invalid replays fail on device."""
    splits = torch.full((num_splits,), 256, device="cuda", dtype=torch.int64)
    strides = STRIDES[:num_outputs]
    leading_zeros = [False] + [True] * (num_outputs - 1)
    dtypes = [torch.int32] + [torch.int64] * (num_outputs - 1)

    def scan():
        return tex.splits_to_offsets_multi(
            splits,
            torch.device("cuda"),
            strides=strides,
            include_leading_zero=leading_zeros,
            dtypes=dtypes,
            bulk_allocate=num_outputs > 1,
            split_alignment=alignment,
        )

    scan()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, outputs = scan()

    replay_sizes = [(256, 256), (0, 512)]
    if alignment == 128:
        replay_sizes.append((128, 384))
    for first, last in replay_sizes:
        sizes = torch.full((num_splits,), 256, dtype=torch.int64)
        sizes[0], sizes[-1] = first, last
        splits.copy_(sizes)
        graph.replay()
        prefix = torch.cumsum(sizes, dim=0)
        for output, stride, leading_zero, dtype in zip(outputs, strides, leading_zeros, dtypes):
            expected = (
                torch.cat((torch.zeros(1, dtype=torch.int64), prefix)) if leading_zero else prefix
            )
            torch.testing.assert_close(output, expected.to(device="cuda", dtype=dtype) * stride)

    # A device trap invalidates its CUDA context, so use a fresh process for this replay.
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_replay_invalid_splits,
        args=(num_outputs, alignment, num_splits, bad_index, bad_value, child),
    )
    process.start()
    child.close()
    try:
        process.join(timeout=60)
        assert not process.is_alive(), "Invalid CUDA replay timed out"
        messages = []
        while parent.poll():
            try:
                messages.append(parent.recv())
            except EOFError:
                break
        assert (
            messages and messages[0][0] == "captured"
        ), f"Child did not capture a graph: exit={process.exitcode}, messages={messages}"
        if len(messages) > 1:
            assert messages[1][0] == "cuda_error", messages[1]
        else:
            assert process.exitcode != 0, "Invalid replay completed without a CUDA error"
        assert "Grouped split size is not divisible by split_alignment." in capfd.readouterr().out
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
