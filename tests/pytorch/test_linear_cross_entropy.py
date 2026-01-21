# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import contextlib
import os
import typing
from contextlib import ExitStack

import numpy as np
import pytest
import torch
import torch.distributed as dist

from transformer_engine.pytorch import linear_cross_entropy

@pytest.mark.skipif(
    "WORLD_SIZE" in os.environ and os.environ["WORLD_SIZE"] != "1", reason="Requires single GPU"
)
class TestFusedLinearCrossEntropyDataParallel:
    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @staticmethod
    def torch_linear_cross_entropy(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: str,
        ignore_index: int,
    ):
        # NOTE: need to convert to fp32 to fp32 accumulation,
        # thus assure accuracy
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        logprobs = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            reduction=reduction,
            ignore_index=ignore_index,
        )
        return logprobs.to(torch.float32)

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            (4096, 152063, 8192),
            ((1, 4096), 152064, 8192),
            ((2, 4096), 152064, 8192),
        ]

    @staticmethod
    def get_ignore_index():
        return [-100, 4]

    def test_kernel_launch(self):
        """
        Check if the compiled kernel can be
        launched with different problem sizes
        """
        self.cleanup()

        num_tokens = [15, 26, 128, 513, 2048, 8192]
        vocab_size = 152064
        dim = 4096
        dtype = torch.bfloat16
        reduction = "mean"
        ignore_index = -100

        weight = torch.randn(vocab_size, dim, dtype=dtype, device="cuda").requires_grad_()
        for num_token in num_tokens:
            hidden = torch.randn(num_token, dim, dtype=dtype, device="cuda").requires_grad_()
            labels = torch.randint(0, vocab_size, (num_token,), dtype=torch.long, device="cuda")

            logprobs = linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index
            )
            assert not torch.isnan(logprobs).any()

            gLogprobs = torch.randn_like(logprobs)
            (d_hidden, d_weight) = torch.autograd.grad(
                (logprobs,), (hidden, weight), (gLogprobs,), retain_graph=False
            )
            assert not torch.isnan(d_hidden).any()
            assert not torch.isnan(d_weight).any()

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("problem", get_problems())
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    @pytest.mark.parametrize("ignore_index", get_ignore_index())
    def test_correctness(self, dtype, problem, reduction, ignore_index):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        hidden = (
            torch.empty(hidden_shape, dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
        if ignore_index >= 0 and ignore_index < vocabsize:
            pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
            labels = pad_labels[..., 1:].contiguous()

        # forward
        torch_logprobs = self.torch_linear_cross_entropy(
            hidden, weight, labels, reduction=reduction, ignore_index=ignore_index
        )

        custom_logprobs = linear_cross_entropy(
            hidden, weight, labels, reduction=reduction, ignore_index=ignore_index
        )

        torch.testing.assert_close(torch_logprobs, custom_logprobs)

        # backward
        g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)

        (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
            (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
        )

        (d_custom_hidden, d_custom_weight) = torch.autograd.grad(
            (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
        )

        torch.testing.assert_close(d_torch_hidden, d_custom_hidden, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(d_torch_weight, d_custom_weight, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_performance(self, problem, dtype, reduction, ignore_index):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch_fwd_latency = list()
        torch_bwd_latency = list()
        custom_fwd_latency = list()
        custom_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
            if ignore_index >= 0 and ignore_index < vocabsize:
                pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                labels = pad_labels[..., 1:].contiguous()

            # -------- forward -------- #
            start_event.record()
            torch_logprobs = self.torch_linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_fwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            custom_logprobs = linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_fwd_latency.append(start_event.elapsed_time(end_event))

            # -------- backward -------- #
            g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)

            start_event.record()
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
                (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_bwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_custom_hidden, d_custom_weight) = torch.autograd.grad(
                (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_bwd_latency.append(start_event.elapsed_time(end_event))

        # --- remove first latency due to warmup --- #
        torch_fwd_latency = torch_fwd_latency[1:]
        torch_bwd_latency = torch_bwd_latency[1:]
        custom_fwd_latency = custom_fwd_latency[1:]
        custom_bwd_latency = custom_bwd_latency[1:]

        print()
        print(f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}:")
        print(
            f"[INFO]: Torch forward latency: {sum(torch_fwd_latency) / len(torch_fwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Custom forward latency: {sum(custom_fwd_latency) / len(custom_fwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Torch backward latency: {sum(torch_bwd_latency) / len(torch_bwd_latency):.2f} ms"
        )
        print(
            f"[INFO]: Custom backward latency: {sum(custom_bwd_latency) / len(custom_bwd_latency):.2f} ms"
        )

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    @pytest.mark.parametrize("ignore_index", [-100])
    def test_storage(self, problem, dtype, reduction, ignore_index):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens
        print()
        print(f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}:")

        def torch_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
            if ignore_index >= 0 and ignore_index < vocabsize:
                pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                labels = pad_labels[..., 1:].contiguous()

            torch.cuda.reset_peak_memory_stats()
            torch_logprobs = self.torch_linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index
            )
            torch.cuda.synchronize()
            torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Torch Forward pass peak memory: {torch_max_memory:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad(
                (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            torch_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Torch Backward pass peak memory: {torch_backward_max_memory:.2f} MB")

        def custom_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")
            if ignore_index >= 0 and ignore_index < vocabsize:
                pad_labels = torch.nn.functional.pad(labels, (0, 1), value=ignore_index)
                labels = pad_labels[..., 1:].contiguous()

            torch.cuda.reset_peak_memory_stats()
            custom_logprobs = linear_cross_entropy(
                hidden, weight, labels, reduction=reduction, ignore_index=ignore_index
            )
            torch.cuda.synchronize()
            custom_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Custom Forward pass peak memory: {custom_max_memory:.2f} MB")

            torch.cuda.reset_peak_memory_stats()
            g_logprobs = torch.empty_like(custom_logprobs).uniform_(-0.1, 0.1)
            (d_custom_hidden, d_custom_weight) = torch.autograd.grad(
                (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            custom_backward_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            print(f"[INFO]: Custom Backward pass peak memory: {custom_backward_max_memory:.2f} MB")

        self.cleanup()
        torch_storage()
        self.cleanup()
        custom_storage()


@pytest.mark.skipif(
    ("WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 2),  # or True,
    reason="Requires torchrun with multiple GPUs",
)
class TestFusedLinearCrossEntropyTensorParallel:
    @classmethod
    def setup_class(cls):
        if dist.is_initialized():
            cls.must_teardown = False
        else:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=int(os.environ["RANK"]),
            )
            cls.must_teardown = True
        cls.tp_group = dist.group.WORLD

        cls.tp_rank = dist.get_rank(cls.tp_group)
        cls.tp_world_size = dist.get_world_size(cls.tp_group)
        cls.is_chief = cls.tp_rank == 0
        device = torch.device(f"cuda:{cls.tp_rank}")
        torch.cuda.set_device(device)
        print(f"[INFO]: TP rank: {cls.tp_rank}, TP world size: {cls.tp_world_size}")

    @classmethod
    def teardown_class(cls):
        if cls.must_teardown:
            dist.destroy_process_group()

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @staticmethod
    def torch_linear_cross_entropy_single_gpu(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: typing.Optional[str] = "mean",
    ):
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        logprobs = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction=reduction
        )
        return logprobs.to(torch.float32)

    class TorchLinearCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden: torch.Tensor,
            weight: torch.Tensor,
            labels: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            reduction: typing.Optional[str] = "mean",
        ):
            tp_rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
            tp_world_size = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)

            logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)

            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            logprobs = torch.nn.functional.cross_entropy(
                whole_logits.view(-1, whole_logits.shape[-1]), labels.view(-1), reduction=reduction
            )

            # If we don't preserve whole_logits,
            # we need to re-compute it in the backward pass
            ctx.save_for_backward(hidden, weight, labels)
            ctx.tp_group = tp_group
            ctx.reduction = reduction
            ctx.tp_rank = tp_rank
            ctx.tp_world_size = tp_world_size

            return logprobs.to(torch.float32)

        @staticmethod
        def backward(ctx, g_logprobs: torch.Tensor):
            hidden, weight, labels = ctx.saved_tensors
            tp_group = ctx.tp_group
            reduction = ctx.reduction
            tp_rank = ctx.tp_rank
            tp_world_size = ctx.tp_world_size

            num_tokens, dim = hidden.shape

            if reduction == "mean":
                _g_logprobs = torch.broadcast_to(g_logprobs / num_tokens, (num_tokens,))
            elif reduction == "sum":
                _g_logprobs = torch.broadcast_to(g_logprobs, (num_tokens,))
            else:
                _g_logprobs = g_logprobs

            # re-compute whole_logits
            logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            one_hot = torch.zeros_like(whole_logits)
            one_hot.scatter_(1, labels.view(-1).unsqueeze(-1), 1)

            pd = torch.nn.functional.softmax(whole_logits, dim=-1)
            d_logits = (pd - one_hot) * _g_logprobs.unsqueeze(-1)
            d_logits = d_logits.to(hidden.dtype)

            local_size = weight.size(0)
            local_d_logits = d_logits[:, tp_rank * local_size : (tp_rank + 1) * local_size]

            local_d_hidden = local_d_logits @ weight
            local_d_weight = local_d_logits.T @ hidden

            dist.all_reduce(local_d_hidden, op=dist.ReduceOp.SUM, group=tp_group)

            return local_d_hidden, local_d_weight, None, None, None

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", [(4096, 129280, 8192)])
    def test_torch_tp_vs_single_gpu(self, dtype, reduction, problem):
        num_tokens, vocabsize, dim = problem

        hidden = (
            torch.empty((num_tokens, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, (num_tokens,), dtype=torch.long, device="cuda")

        # ------------ forward pass ------------ #
        dist.broadcast(hidden, src=0, group=self.tp_group)
        dist.broadcast(labels, src=0, group=self.tp_group)

        # single GPU
        whole_weight = torch.empty(
            (vocabsize * self.tp_world_size, dim), dtype=dtype, device="cuda"
        )
        whole_weight_view = [
            whole_weight[i * vocabsize : (i + 1) * vocabsize, :] for i in range(self.tp_world_size)
        ]
        dist.all_gather(whole_weight_view, weight, group=self.tp_group)
        whole_weight = whole_weight.clone().requires_grad_()
        logprobs_single_gpu = self.torch_linear_cross_entropy_single_gpu(
            hidden, whole_weight, labels, reduction=reduction
        )

        # TP
        logprobs_tp = self.TorchLinearCrossEntropy.apply(
            hidden, weight, labels, self.tp_group, reduction
        )
        torch.testing.assert_close(logprobs_single_gpu, logprobs_tp)

        # ------------ backward pass ------------ #
        g_logprobs = torch.empty_like(logprobs_single_gpu).uniform_(-0.1, 0.1)
        dist.broadcast(g_logprobs, src=0, group=self.tp_group)

        # single GPU
        (d_hidden_single_gpu, d_weight_single_gpu) = torch.autograd.grad(
            (logprobs_single_gpu,), (hidden, whole_weight), (g_logprobs,), retain_graph=False
        )

        # TP
        (d_hidden_tp, d_weight_tp) = torch.autograd.grad(
            (logprobs_tp,), (hidden, weight), (g_logprobs,), retain_graph=False
        )
        torch.testing.assert_close(d_hidden_single_gpu, d_hidden_tp, atol=1e-3, rtol=1e-3)
        local_d_weight_single_gpu = d_weight_single_gpu[
            self.tp_rank * weight.shape[0] : (self.tp_rank + 1) * weight.shape[0], :
        ]
        torch.testing.assert_close(local_d_weight_single_gpu, d_weight_tp, atol=1e-3, rtol=1e-3)

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            (4096, 152063, 8192),
            ((1, 4096), 152064, 8192),
            ((2, 4096), 152064, 8192),
        ]

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", get_problems())
    def test_correctness(self, dtype, reduction, problem):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        hidden = (
            torch.empty(hidden_shape, dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

        # ------ forward pass ------ #
        dist.broadcast(hidden, src=0, group=self.tp_group)
        dist.broadcast(labels, src=0, group=self.tp_group)

        torch_logprobs = self.TorchLinearCrossEntropy.apply(
            hidden.view(-1, dim), weight, labels, self.tp_group, reduction
        )

        custom_logprobs = linear_cross_entropy(
            hidden, weight, labels, tp_group=self.tp_group, reduction=reduction
        )

        torch.testing.assert_close(torch_logprobs, custom_logprobs)

        # ------- backward pass ------- #
        g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
        dist.broadcast(g_logprobs, src=0, group=self.tp_group)

        (d_hidden_torch, d_weight_torch) = torch.autograd.grad(
            (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
        )
        (d_hidden_custom, d_weight_custom) = torch.autograd.grad(
            (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
        )
        torch.testing.assert_close(d_hidden_torch, d_hidden_custom, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(d_weight_torch, d_weight_custom, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_performance(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch_fwd_latency = list()
        torch_bwd_latency = list()
        custom_fwd_latency = list()
        custom_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            # ------ forward pass ------ #
            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            start_event.record()
            torch_logprobs = self.TorchLinearCrossEntropy.apply(
                hidden.view(-1, dim), weight, labels, self.tp_group, reduction
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_fwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            custom_logprobs = linear_cross_entropy(
                hidden, weight, labels, tp_group=self.tp_group, reduction=reduction
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_fwd_latency.append(start_event.elapsed_time(end_event))

            # ------- backward pass ------- #
            g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            start_event.record()
            (d_hidden_torch, d_weight_torch) = torch.autograd.grad(
                (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_bwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_hidden_custom, d_weight_custom) = torch.autograd.grad(
                (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_bwd_latency.append(start_event.elapsed_time(end_event))

        # --- remove first latency due to warmup --- #
        torch_fwd_latency = torch_fwd_latency[1:]
        torch_bwd_latency = torch_bwd_latency[1:]
        custom_fwd_latency = custom_fwd_latency[1:]
        custom_bwd_latency = custom_bwd_latency[1:]

        if self.is_chief:
            print()
            print(
                f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}, TP size {self.tp_world_size}:"
            )
            print(
                f"[INFO]: Torch forward latency: {sum(torch_fwd_latency) / len(torch_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Custom forward latency: {sum(custom_fwd_latency) / len(custom_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Torch backward latency: {sum(torch_bwd_latency) / len(torch_bwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Custom backward latency: {sum(custom_bwd_latency) / len(custom_bwd_latency):.2f} ms"
            )

    @pytest.mark.parametrize("problem", [((1, 4096), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_storage(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (num_tokens,) if isinstance(num_tokens, int) else num_tokens

        if self.is_chief:
            print()
            print(
                f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}, TP size {self.tp_world_size}:"
            )

        def torch_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            torch_logprobs = self.TorchLinearCrossEntropy.apply(
                hidden.view(-1, dim), weight, labels, self.tp_group, reduction
            )
            torch.cuda.synchronize()
            torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Torch Forward pass peak memory: {torch_max_memory:.2f} MB"
                )

            g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            (d_hidden_torch, d_weight_torch) = torch.autograd.grad(
                (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Torch Backward pass peak memory: {torch_max_memory:.2f} MB"
                )

        def custom_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            custom_logprobs = linear_cross_entropy(
                hidden, weight, labels, tp_group=self.tp_group, reduction=reduction
            )
            torch.cuda.synchronize()
            custom_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Custom Forward pass peak memory: {custom_max_memory:.2f} MB"
                )

            g_logprobs = torch.empty_like(custom_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            (d_hidden_custom, d_weight_custom) = torch.autograd.grad(
                (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            custom_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Custom Backward pass peak memory: {custom_max_memory:.2f} MB"
                )

        self.cleanup()
        torch_storage()
        self.cleanup()
        custom_storage()


@pytest.mark.skipif(
    "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 2,
    reason="Requires torchrun with multiple GPUs",
)
class TestFusedLinearCrossEntropySequenceParallel:
    @classmethod
    def setup_class(cls):
        if dist.is_initialized():
            cls.must_teardown = False
        else:
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=int(os.environ["WORLD_SIZE"]),
                rank=int(os.environ["RANK"]),
            )
            cls.must_teardown = True
        cls.tp_group = dist.group.WORLD

        cls.tp_rank = dist.get_rank(cls.tp_group)
        cls.tp_world_size = dist.get_world_size(cls.tp_group)
        cls.is_chief = cls.tp_rank == 0
        device = torch.device(f"cuda:{cls.tp_rank}")
        torch.cuda.set_device(device)
        print(f"[INFO]: TP rank: {cls.tp_rank}, TP world size: {cls.tp_world_size}")

    @classmethod
    def teardown_class(cls):
        if cls.must_teardown:
            dist.destroy_process_group()

    @staticmethod
    def timed_barrier(timeout_s=10):
        import time

        work = torch.distributed.barrier(async_op=True)
        t0 = time.time()
        while not work.is_completed():
            if time.time() - t0 > timeout_s:
                exit(1)
            time.sleep(0.05)
        work.wait()

    def cleanup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc

        gc.collect()
        torch.cuda.synchronize()

    @staticmethod
    def torch_linear_cross_entropy_single_gpu(
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        reduction: typing.Optional[str] = "mean",
    ):
        logits = hidden.to(torch.float32) @ weight.T.to(torch.float32)
        logprobs = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction=reduction
        )
        return logprobs.to(torch.float32)

    class TorchLinearCrossEntropy(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden: torch.Tensor,
            weight: torch.Tensor,
            labels: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            reduction: typing.Optional[str] = "mean",
        ):
            tp_rank = 0 if tp_group is None else torch.distributed.get_rank(tp_group)
            tp_world_size = 1 if tp_group is None else torch.distributed.get_world_size(tp_group)

            whole_hidden = torch.empty(
                (hidden.shape[0] * tp_world_size, hidden.shape[-1]),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            dist.all_gather_into_tensor(whole_hidden, hidden, group=tp_group)

            logits = whole_hidden.to(torch.float32) @ weight.T.to(torch.float32)

            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            logprobs = torch.nn.functional.cross_entropy(
                whole_logits.view(-1, whole_logits.shape[-1]), labels.view(-1), reduction=reduction
            )

            # If we don't preserve whole_logits,
            # we need to re-compute it in the backward pass
            ctx.save_for_backward(whole_hidden, weight, labels)
            ctx.tp_group = tp_group
            ctx.reduction = reduction
            ctx.tp_rank = tp_rank
            ctx.tp_world_size = tp_world_size

            return logprobs.to(torch.float32)

        @staticmethod
        def backward(ctx, g_logprobs: torch.Tensor):
            whole_hidden, weight, labels = ctx.saved_tensors
            tp_group = ctx.tp_group
            reduction = ctx.reduction
            tp_rank = ctx.tp_rank
            tp_world_size = ctx.tp_world_size

            num_tokens, dim = whole_hidden.shape

            if reduction == "mean":
                _g_logprobs = torch.broadcast_to(g_logprobs / num_tokens, (num_tokens,))
            elif reduction == "sum":
                _g_logprobs = torch.broadcast_to(g_logprobs, (num_tokens,))
            else:
                _g_logprobs = g_logprobs

            # re-compute whole_logits
            logits = whole_hidden.to(torch.float32) @ weight.T.to(torch.float32)
            whole_logits = torch.empty(
                (logits.shape[0], logits.shape[-1] * tp_world_size),
                dtype=logits.dtype,
                device=logits.device,
            )
            whole_logits_ref = [
                whole_logits[..., i * logits.shape[-1] : (i + 1) * logits.shape[-1]]
                for i in range(tp_world_size)
            ]
            dist.all_gather(whole_logits_ref, logits, group=tp_group)

            one_hot = torch.zeros_like(whole_logits)
            one_hot.scatter_(1, labels.view(-1).unsqueeze(-1), 1)

            pd = torch.nn.functional.softmax(whole_logits, dim=-1)
            d_logits = (pd - one_hot) * _g_logprobs.unsqueeze(-1)
            d_logits = d_logits.to(whole_hidden.dtype)

            local_size = weight.size(0)
            local_d_logits = d_logits[:, tp_rank * local_size : (tp_rank + 1) * local_size]

            d_hidden = local_d_logits @ weight
            local_d_weight = local_d_logits.T @ whole_hidden

            # dist.all_reduce(
            #     local_d_hidden,
            #     op=dist.ReduceOp.SUM,
            #     group=tp_group
            # )

            # split the local_d_hidden along the sequence length dimension
            local_num_tokens = num_tokens // tp_world_size
            # local_d_hidden = local_d_hidden[tp_rank * local_num_tokens : (tp_rank + 1) * local_num_tokens, :]

            local_d_hidden = torch.empty(
                (local_num_tokens, dim), dtype=weight.dtype, device=weight.device
            )
            dist.reduce_scatter_tensor(
                local_d_hidden, d_hidden, op=dist.ReduceOp.SUM, group=tp_group
            )
            return local_d_hidden, local_d_weight, None, None, None

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", [(256, 12928, 8192)])
    def test_torch_tp_vs_single_gpu(self, dtype, reduction, problem):
        num_tokens, vocabsize, dim = problem

        hidden = (
            torch.empty((num_tokens, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(
            0, vocabsize, (num_tokens * self.tp_world_size,), dtype=torch.long, device="cuda"
        )

        # ------------ forward pass ------------ #
        dist.broadcast(labels, src=0, group=self.tp_group)

        # single GPU
        whole_hidden = torch.empty(
            (num_tokens * self.tp_world_size, dim), dtype=dtype, device="cuda"
        )
        dist.all_gather_into_tensor(whole_hidden, hidden, group=self.tp_group)
        whole_hidden = whole_hidden.clone().requires_grad_()

        whole_weight = torch.empty(
            (vocabsize * self.tp_world_size, dim), dtype=dtype, device="cuda"
        )
        whole_weight_view = [
            whole_weight[i * vocabsize : (i + 1) * vocabsize, :] for i in range(self.tp_world_size)
        ]
        dist.all_gather(whole_weight_view, weight, group=self.tp_group)
        whole_weight = whole_weight.clone().requires_grad_()
        logprobs_single_gpu = self.torch_linear_cross_entropy_single_gpu(
            whole_hidden, whole_weight, labels, reduction=reduction
        )

        # TP
        logprobs_tp = self.TorchLinearCrossEntropy.apply(
            hidden, weight, labels, self.tp_group, reduction
        )
        torch.testing.assert_close(logprobs_single_gpu, logprobs_tp)

        # ------------ backward pass ------------ #
        g_logprobs = torch.empty_like(logprobs_single_gpu).uniform_(-0.1, 0.1)
        dist.broadcast(g_logprobs, src=0, group=self.tp_group)

        # single GPU
        (d_hidden_single_gpu, d_weight_single_gpu) = torch.autograd.grad(
            (logprobs_single_gpu,), (whole_hidden, whole_weight), (g_logprobs,), retain_graph=False
        )

        # TP
        (d_hidden_tp, d_weight_tp) = torch.autograd.grad(
            (logprobs_tp,), (hidden, weight), (g_logprobs,), retain_graph=False
        )

        local_d_hidden_single_gpu = d_hidden_single_gpu[
            self.tp_rank * hidden.shape[0] : (self.tp_rank + 1) * hidden.shape[0], :
        ]
        torch.testing.assert_close(local_d_hidden_single_gpu, d_hidden_tp, atol=1e-3, rtol=1e-3)
        local_d_weight_single_gpu = d_weight_single_gpu[
            self.tp_rank * weight.shape[0] : (self.tp_rank + 1) * weight.shape[0], :
        ]
        torch.testing.assert_close(local_d_weight_single_gpu, d_weight_tp, atol=1e-3, rtol=1e-3)

        self.cleanup()

    @staticmethod
    def get_problems():
        return [
            (80, 125, 64),
            (80, 152064, 64),
            (1024, 152064, 4096),
            (4096, 15206, 1024),
            ((1, 4096), 15206, 1024),
            ((4, 1024), 15206, 1024),
        ]

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    @pytest.mark.parametrize("problem", get_problems())
    def test_correctness(self, dtype, reduction, problem):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (
            (num_tokens * self.tp_world_size,)
            if isinstance(num_tokens, int)
            else (num_tokens[0] * self.tp_world_size, *num_tokens[1:])
        )

        hidden = (
            torch.empty(hidden_shape, dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        weight = (
            torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

        # ------ forward pass ------ #
        dist.broadcast(labels, src=0, group=self.tp_group)

        torch_logprobs = self.TorchLinearCrossEntropy.apply(
            hidden.view(-1, dim), weight, labels, self.tp_group, reduction
        )

        custom_logprobs = linear_cross_entropy(
            hidden,
            weight,
            labels,
            tp_group=self.tp_group,
            reduction=reduction,
            sequence_parallel=True,
        )

        torch.testing.assert_close(torch_logprobs, custom_logprobs)

        # ------- backward pass ------- #
        g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
        dist.broadcast(g_logprobs, src=0, group=self.tp_group)

        (d_hidden_torch, d_weight_torch) = torch.autograd.grad(
            (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
        )
        (d_hidden_custom, d_weight_custom) = torch.autograd.grad(
            (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
        )

        # in case one GPU failed, and leading to hang
        torch.testing.assert_close(d_hidden_torch, d_hidden_custom, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(d_weight_torch, d_weight_custom, atol=1e-3, rtol=1e-3)
        self.timed_barrier()

        self.cleanup()

    @pytest.mark.parametrize("problem", [((1, 1024), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_performance(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (
            (num_tokens * self.tp_world_size,)
            if isinstance(num_tokens, int)
            else (num_tokens[0] * self.tp_world_size, *num_tokens[1:])
        )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch_fwd_latency = list()
        torch_bwd_latency = list()
        custom_fwd_latency = list()
        custom_bwd_latency = list()

        iterations = 5
        for i in range(iterations):
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            # ------ forward pass ------ #
            dist.broadcast(labels, src=0, group=self.tp_group)

            start_event.record()
            torch_logprobs = self.TorchLinearCrossEntropy.apply(
                hidden.view(-1, dim), weight, labels, self.tp_group, reduction
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_fwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            custom_logprobs = linear_cross_entropy(
                hidden,
                weight,
                labels,
                tp_group=self.tp_group,
                reduction=reduction,
                sequence_parallel=True,
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_fwd_latency.append(start_event.elapsed_time(end_event))

            # ------- backward pass ------- #
            g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            start_event.record()
            (d_hidden_torch, d_weight_torch) = torch.autograd.grad(
                (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            torch_bwd_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_hidden_custom, d_weight_custom) = torch.autograd.grad(
                (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            end_event.record()
            torch.cuda.synchronize()
            custom_bwd_latency.append(start_event.elapsed_time(end_event))

        # --- remove first latency due to warmup --- #
        torch_fwd_latency = torch_fwd_latency[1:]
        torch_bwd_latency = torch_bwd_latency[1:]
        custom_fwd_latency = custom_fwd_latency[1:]
        custom_bwd_latency = custom_bwd_latency[1:]

        if self.is_chief:
            print()
            print(
                f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}, TP size {self.tp_world_size}, Sequence Parallel: True:"
            )
            print(
                f"[INFO]: Torch forward latency: {sum(torch_fwd_latency) / len(torch_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Custom forward latency: {sum(custom_fwd_latency) / len(custom_fwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Torch backward latency: {sum(torch_bwd_latency) / len(torch_bwd_latency):.2f} ms"
            )
            print(
                f"[INFO]: Custom backward latency: {sum(custom_bwd_latency) / len(custom_bwd_latency):.2f} ms"
            )

    @pytest.mark.parametrize("problem", [((1, 1024), 129280, 7168)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("reduction", ["mean"])
    def test_storage(self, problem, dtype, reduction):
        num_tokens, vocabsize, dim = problem
        hidden_shape = (num_tokens, dim) if isinstance(num_tokens, int) else (*num_tokens, dim)
        labels_shape = (
            (num_tokens * self.tp_world_size,)
            if isinstance(num_tokens, int)
            else (num_tokens[0] * self.tp_world_size, *num_tokens[1:])
        )

        if self.is_chief:
            print()
            print(
                f"[INFO]: On problem {problem}, dtype {dtype}, reduction {reduction}, TP size {self.tp_world_size}, Sequence Parallel: True:"
            )

        def torch_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            torch_logprobs = self.TorchLinearCrossEntropy.apply(
                hidden.view(-1, dim), weight, labels, self.tp_group, reduction
            )
            torch.cuda.synchronize()
            torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Torch Forward pass peak memory: {torch_max_memory:.2f} MB"
                )

            g_logprobs = torch.empty_like(torch_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            (d_hidden_torch, d_weight_torch) = torch.autograd.grad(
                (torch_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            torch_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Torch Backward pass peak memory: {torch_max_memory:.2f} MB"
                )

        def custom_storage():
            hidden = (
                torch.empty(hidden_shape, dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            weight = (
                torch.empty((vocabsize, dim), dtype=dtype, device="cuda")
                .uniform_(-0.1, 0.1)
                .requires_grad_()
            )
            labels = torch.randint(0, vocabsize, labels_shape, dtype=torch.long, device="cuda")

            dist.broadcast(hidden, src=0, group=self.tp_group)
            dist.broadcast(labels, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            custom_logprobs = linear_cross_entropy(
                hidden,
                weight,
                labels,
                tp_group=self.tp_group,
                reduction=reduction,
                sequence_parallel=True,
            )
            torch.cuda.synchronize()
            custom_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Custom Forward pass peak memory: {custom_max_memory:.2f} MB"
                )

            g_logprobs = torch.empty_like(custom_logprobs).uniform_(-0.1, 0.1)
            dist.broadcast(g_logprobs, src=0, group=self.tp_group)

            torch.cuda.reset_peak_memory_stats()
            (d_hidden_custom, d_weight_custom) = torch.autograd.grad(
                (custom_logprobs,), (hidden, weight), (g_logprobs,), retain_graph=False
            )
            torch.cuda.synchronize()
            custom_max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            if self.is_chief:
                print(
                    f"[INFO]: On GPU {self.tp_rank}, Custom Backward pass peak memory: {custom_max_memory:.2f} MB"
                )

        self.cleanup()
        torch_storage()
        self.cleanup()
        custom_storage()
