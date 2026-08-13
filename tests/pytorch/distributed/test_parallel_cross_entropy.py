# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import pathlib
import sys
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from transformer_engine.pytorch import parallel_cross_entropy

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from utils import dtype_tols


def _run_tensor_parallel(rank, world_size, init_file):
    """Two-rank correctness worker for tensor-parallel cross entropy."""

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        generator = torch.Generator().manual_seed(2025)
        shape = (2, 3, 22)
        local_vocab = shape[-1] // world_size
        target = torch.randint(
            0,
            shape[-1],
            shape[:-1],
            generator=generator,
        ).to(device)
        target[0, 2] = -100
        external_grad = torch.randn(shape[:-1], generator=generator).to(device)

        for dtype in (torch.float32, torch.bfloat16):
            global_values = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
            vocab_start = rank * local_vocab
            local_values = global_values[..., vocab_start : vocab_start + local_vocab]
            for reduce_loss in (False, True):
                for overwrite_input in (False, True):
                    local_logits = local_values.clone().requires_grad_()
                    local_before = local_logits.detach().clone()
                    version_before = local_logits._version
                    ref_logits = global_values.float().clone().requires_grad_()

                    loss = parallel_cross_entropy(
                        local_logits,
                        target,
                        label_smoothing=0.1,
                        reduce_loss=reduce_loss,
                        dist_process_group=dist.group.WORLD,
                        overwrite_input=overwrite_input,
                    )
                    ref_loss = torch.nn.functional.cross_entropy(
                        ref_logits.reshape(-1, shape[-1]),
                        target.reshape(-1),
                        label_smoothing=0.1,
                        reduction="mean" if reduce_loss else "none",
                    ).reshape_as(loss)
                    loss_grad = (
                        torch.full_like(loss, 0.37) if reduce_loss else external_grad
                    )
                    loss.backward(loss_grad)
                    ref_loss.backward(loss_grad)
                    assert local_logits._version == version_before + int(overwrite_input)
                    if overwrite_input:
                        assert not torch.equal(local_logits, local_before)
                    else:
                        torch.testing.assert_close(
                            local_logits, local_before, rtol=0.0, atol=0.0
                        )

                    torch.testing.assert_close(
                        loss, ref_loss, **dtype_tols(torch.float32)
                    )
                    expected_grad = ref_logits.grad[
                        ..., vocab_start : vocab_start + local_vocab
                    ].to(dtype)
                    torch.testing.assert_close(
                        local_logits.grad, expected_grad, **dtype_tols(dtype)
                    )
    finally:
        dist.destroy_process_group()


def test_parallel_cross_entropy_tensor_parallel():
    """Validate tensor-parallel loss and gradients on two ranks."""

    if torch.cuda.device_count() < 2:
        pytest.skip("tensor-parallel cross entropy test requires two CUDA devices")
    world_size = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, "distributed_init")
        mp.spawn(
            _run_tensor_parallel,
            args=(world_size, init_file),
            nprocs=world_size,
            join=True,
        )
