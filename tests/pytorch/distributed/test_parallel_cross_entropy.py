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

import transformer_engine_torch as tex
from transformer_engine.pytorch import is_mxfp8_available, parallel_cross_entropy
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
import transformer_engine.pytorch.triton.cross_entropy as triton_cross_entropy


# Prepend so installed packages with a top-level utils module cannot shadow the test helpers.
sys.path = [str(pathlib.Path(__file__).resolve().parent.parent)] + sys.path
from utils import dtype_tols


def _run_tensor_parallel(rank, world_size, init_file, label_smoothing):
    """Two-rank correctness worker for tensor-parallel cross entropy."""

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
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
                        label_smoothing=label_smoothing,
                        reduce_loss=reduce_loss,
                        dist_process_group=dist.group.WORLD,
                        overwrite_input=overwrite_input,
                    )
                    ref_loss = torch.nn.functional.cross_entropy(
                        ref_logits.reshape(-1, shape[-1]),
                        target.reshape(-1),
                        label_smoothing=label_smoothing,
                        reduction="mean" if reduce_loss else "none",
                    ).reshape_as(loss)
                    loss_grad = torch.full_like(loss, 0.37) if reduce_loss else external_grad
                    loss.backward(loss_grad)
                    ref_loss.backward(loss_grad)
                    assert local_logits._version == version_before + int(overwrite_input)
                    if overwrite_input:
                        assert not torch.equal(local_logits, local_before)
                    else:
                        torch.testing.assert_close(local_logits, local_before, rtol=0.0, atol=0.0)

                    torch.testing.assert_close(loss, ref_loss, **dtype_tols(torch.float32))
                    expected_grad = ref_logits.grad[
                        ..., vocab_start : vocab_start + local_vocab
                    ].to(dtype)
                    torch.testing.assert_close(
                        local_logits.grad, expected_grad, **dtype_tols(dtype)
                    )
    finally:
        dist.destroy_process_group()


def _run_tensor_parallel_argmax(rank, world_size, init_file, label_smoothing):
    """Two-rank worker: the argmax over vocabulary shards, with and without multiple blocks."""

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        generator = torch.Generator().manual_seed(2026)
        for shape in ((2, 3, 22), (1, 4, 80000)):
            vocab = shape[-1]
            local_vocab = vocab // world_size
            target = torch.randint(0, vocab, shape[:-1], generator=generator).to(device)
            target[0, 2] = -100
            values = torch.randn(shape, generator=generator).to(device)
            peak = 8.0
            values[0, 0, [3, local_vocab + 4]] = peak  # equal maxima on both ranks
            values[0, 1, [local_vocab + 2, vocab - 2]] = peak  # equal maxima on rank 1
            values[0, 2, vocab - 1] = peak  # the maximum on the last rank only
            if shape[0] > 1:
                # NaN counts as the maximum, as in torch.argmax: the first NaN wins.
                values[1, 0, [5, local_vocab + 1]] = float("nan")  # NaN on both ranks
                values[1, 1, 2] = peak
                values[1, 1, local_vocab + 3] = float("nan")  # NaN after a larger rank-0 maximum
            values[-1, -1] = 0.5  # a constant row
            vocab_start = rank * local_vocab
            for dtype in (torch.float32, torch.bfloat16):
                global_values = values.to(dtype)
                expected_argmax = torch.argmax(global_values, dim=-1)
                local_values = global_values[..., vocab_start : vocab_start + local_vocab]
                for reduce_loss in (False, True):
                    for overwrite_input in (False, True):
                        results = []
                        for return_argmax in (False, True):
                            local_logits = local_values.clone().requires_grad_()
                            out = parallel_cross_entropy(
                                local_logits,
                                target,
                                label_smoothing=label_smoothing,
                                reduce_loss=reduce_loss,
                                dist_process_group=dist.group.WORLD,
                                overwrite_input=overwrite_input,
                                return_argmax=return_argmax,
                            )
                            loss, argmax = out if return_argmax else (out, None)
                            loss.backward(torch.full_like(loss, 0.37))
                            results.append((loss, local_logits.grad, argmax))
                        (loss, grad, _), (argmax_loss, argmax_grad, argmax) = results
                        torch.testing.assert_close(argmax, expected_argmax, rtol=0, atol=0)
                        torch.testing.assert_close(
                            argmax_loss, loss, rtol=0, atol=0, equal_nan=True
                        )
                        torch.testing.assert_close(
                            argmax_grad, grad, rtol=0, atol=0, equal_nan=True
                        )
    finally:
        dist.destroy_process_group()


def _run_tensor_parallel_masked(rank, world_size, init_file, label_smoothing):
    """Two-rank worker: masked (-inf) logits, including shards and blocks masked entirely."""

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        generator = torch.Generator().manual_seed(2027)
        for shape, block in (((1, 5, 22), 8), ((1, 5, 80000), 32768)):
            vocab = shape[-1]
            local_vocab = vocab // world_size
            values = torch.randn(shape, generator=generator)
            values[0, 0, :local_vocab] = -float("inf")  # the shard of rank 0 is masked
            values[0, 1, local_vocab:] = -float("inf")  # the shard of rank 1 is masked
            for start in range(0, vocab, local_vocab):
                values[0, 2, start : start + block] = -float("inf")  # each first block
            values[0, 3, :-1] = -float("inf")  # only the last logit is unmasked
            target = torch.multinomial(
                values.isfinite().float().reshape(-1, vocab), 1, generator=generator
            ).reshape(shape[:-1])
            values, target = values.to(device), target.to(device)
            vocab_start = rank * local_vocab
            for dtype in (torch.float32, torch.bfloat16):
                global_values = values.to(dtype)
                local_values = global_values[..., vocab_start : vocab_start + local_vocab]
                for reduce_loss in (False, True):
                    for overwrite_input in (False, True):
                        local_logits = local_values.clone().requires_grad_()
                        ref_logits = global_values.double().clone().requires_grad_()
                        loss, argmax = parallel_cross_entropy(
                            local_logits,
                            target,
                            label_smoothing=label_smoothing,
                            reduce_loss=reduce_loss,
                            dist_process_group=dist.group.WORLD,
                            overwrite_input=overwrite_input,
                            return_argmax=True,
                        )
                        ref_loss = torch.nn.functional.cross_entropy(
                            ref_logits.reshape(-1, vocab),
                            target.reshape(-1),
                            label_smoothing=label_smoothing,
                            reduction="mean" if reduce_loss else "none",
                        ).reshape_as(loss)
                        loss_grad = torch.full_like(loss, 0.37)
                        loss.backward(loss_grad)
                        ref_loss.backward(loss_grad.double())
                        torch.testing.assert_close(
                            loss, ref_loss.float(), **dtype_tols(torch.float32)
                        )
                        expected_grad = ref_logits.grad[
                            ..., vocab_start : vocab_start + local_vocab
                        ].to(dtype)
                        assert local_logits.grad.isfinite().all()
                        torch.testing.assert_close(
                            local_logits.grad, expected_grad, **dtype_tols(dtype)
                        )
                        torch.testing.assert_close(
                            argmax, torch.argmax(global_values, dim=-1), rtol=0, atol=0
                        )
    finally:
        dist.destroy_process_group()


class _KernelLaunchCounter:
    """Stands in for a Triton kernel, counting its launches and forwarding them."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.launches = 0

    def __getitem__(self, grid):
        launch = self.kernel[grid]

        def counted_launch(*args, **kwargs):
            self.launches += 1
            return launch(*args, **kwargs)

        return counted_launch


def _run_tensor_parallel_mxfp8_grad(rank, world_size, init_file, label_smoothing):
    """Two-rank worker: each rank's MXFP8 gradient is written by the fused kernel and has the
    bytes of its BF16 gradient shard quantized by the quantizer."""

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    counter = _KernelLaunchCounter(triton_cross_entropy.cross_entropy_backward_mxfp8_kernel)
    triton_cross_entropy.cross_entropy_backward_mxfp8_kernel = counter
    try:
        generator = torch.Generator().manual_seed(2027)
        # Shards of 384 and 4096 columns: 128- and 256-column tiles.
        for local_vocab in (384, 4096):
            vocab = local_vocab * world_size
            shape = (2, 64, vocab)
            target = torch.randint(0, vocab, shape[:-1], generator=generator).to(device)
            target[0, :3] = -100
            values = (torch.randn(shape, generator=generator) * 3).to(
                device=device, dtype=torch.bfloat16
            )
            vocab_start = rank * local_vocab
            local_values = values[..., vocab_start : vocab_start + local_vocab]
            for fp8_dtype in (tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2):
                for reduce_loss in (False, True):
                    quantizer = MXFP8Quantizer(fp8_dtype)
                    grads = []
                    for grad_quantizer in (None, quantizer):
                        logits = local_values.clone().requires_grad_()
                        loss = parallel_cross_entropy(
                            logits,
                            target,
                            label_smoothing=label_smoothing,
                            reduce_loss=reduce_loss,
                            dist_process_group=dist.group.WORLD,
                            grad_quantizer=grad_quantizer,
                        )
                        launches = counter.launches
                        (grad,) = torch.autograd.grad(loss, logits, torch.full_like(loss, 0.37))
                        assert counter.launches - launches == int(grad_quantizer is not None)
                        grads.append(grad)
                    ref_grad, grad = grads
                    ref = quantizer(ref_grad)
                    # 128 rows and a shard that is a multiple of 128: no scale padding.
                    for name in (
                        "_rowwise_data",
                        "_rowwise_scale_inv",
                        "_columnwise_data",
                        "_columnwise_scale_inv",
                    ):
                        torch.testing.assert_close(
                            getattr(grad, name).view(torch.uint8),
                            getattr(ref, name).view(torch.uint8),
                            rtol=0,
                            atol=0,
                        )
    finally:
        triton_cross_entropy.cross_entropy_backward_mxfp8_kernel = counter.kernel
        dist.destroy_process_group()


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1], ids=["plain", "smoothed"])
def test_parallel_cross_entropy_tensor_parallel(label_smoothing):
    """Validate tensor-parallel loss and gradients on two ranks."""

    if torch.cuda.device_count() < 2:
        pytest.skip("tensor-parallel cross entropy test requires two CUDA devices")
    world_size = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, "distributed_init")
        mp.spawn(
            _run_tensor_parallel,
            args=(world_size, init_file, label_smoothing),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1], ids=["plain", "smoothed"])
def test_parallel_cross_entropy_tensor_parallel_argmax(label_smoothing):
    """Validate the global argmax and the unchanged loss and gradients on two ranks."""

    if torch.cuda.device_count() < 2:
        pytest.skip("tensor-parallel cross entropy test requires two CUDA devices")
    world_size = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, "distributed_init")
        mp.spawn(
            _run_tensor_parallel_argmax,
            args=(world_size, init_file, label_smoothing),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1], ids=["plain", "smoothed"])
def test_parallel_cross_entropy_tensor_parallel_masked(label_smoothing):
    """Validate tensor-parallel loss and gradients with masked logits on two ranks."""

    if torch.cuda.device_count() < 2:
        pytest.skip("tensor-parallel cross entropy test requires two CUDA devices")
    world_size = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, "distributed_init")
        mp.spawn(
            _run_tensor_parallel_masked,
            args=(world_size, init_file, label_smoothing),
            nprocs=world_size,
            join=True,
        )


@pytest.mark.parametrize("label_smoothing", [0.0, 0.1], ids=["plain", "smoothed"])
def test_parallel_cross_entropy_tensor_parallel_mxfp8_grad(label_smoothing):
    """Validate the fused MXFP8 gradient of each vocabulary shard on two ranks."""

    if torch.cuda.device_count() < 2:
        pytest.skip("tensor-parallel cross entropy test requires two CUDA devices")
    mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
    if not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    world_size = 2
    with tempfile.TemporaryDirectory() as temp_dir:
        init_file = os.path.join(temp_dir, "distributed_init")
        mp.spawn(
            _run_tensor_parallel_mxfp8_grad,
            args=(world_size, init_file, label_smoothing),
            nprocs=world_size,
            join=True,
        )
