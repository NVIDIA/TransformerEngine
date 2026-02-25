# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import torch
import torch.nn.functional as F
from transformer_engine.pytorch import parallel_cross_entropy

from utils import dtype_tols


class TestParallelCrossEntropy:

    def generate_iters(self, iters: int):
        self.iters = iters

    def generate_infra(self, reduce_loss: bool, label_smoothing: float):
        self.test_loss_func = parallel_cross_entropy
        self.ref_loss_func = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="mean" if reduce_loss else "none"
        )

    def generate_input(
        self,
        dtype: torch.dtype,
        swap_dim: bool,
        ignore_idx: bool,
        device: torch.device = "cuda",
    ):
        SQ = random.choice([64, 128])
        batch = random.choice([1, 2])
        vocab = random.choice([64000, 128000])
        ignore = random.sample(range(0, SQ - 1), 5)

        # Generate random data
        if swap_dim:
            self.input_test = torch.rand((SQ, batch, vocab), dtype=dtype, device=device)
            self.tar_test = torch.randint(0, vocab, (SQ, batch), device=device)
        else:
            self.input_test = torch.rand((batch, SQ, vocab), dtype=dtype, device=device)
            self.tar_test = torch.randint(0, vocab, (batch, SQ), device=device)

        if ignore_idx:
            for i in ignore:
                # Ignore 5 indices
                if swap_dim:
                    self.tar_test[i][0] = -100
                else:
                    self.tar_test[0][i] = -100

        # Make copy of data for reference implementation
        self.input_ref = torch.reshape(self.input_test.clone().detach(), (batch * SQ, vocab))
        self.tar_ref = torch.reshape(self.tar_test.clone().detach(), (batch * SQ,))

        # Enable autograd
        self.input_test.requires_grad_()
        self.input_ref.requires_grad_()

    def one_iteration_test(
        self,
        dtype: torch.dtype,
        swap_dim: bool,
        label_smoothing: float,
        reduce_loss: bool,
        ignore_idx: bool = False,
    ):

        # Random data
        self.generate_input(dtype, swap_dim, ignore_idx)

        # Forward pass — default return is a single tensor (backward compatible)
        test_loss = self.test_loss_func(
            self.input_test, self.tar_test, label_smoothing, reduce_loss, None
        )
        ref_loss = self.ref_loss_func(self.input_ref, self.tar_ref)

        # Compute square to avoid trivial backward pass
        test_loss = torch.square(test_loss)
        ref_loss = torch.square(ref_loss)

        # Backward pass
        if reduce_loss:
            test_loss.backward()
            ref_loss.backward()
        else:
            test_loss.sum().backward()
            ref_loss.sum().backward()

        # Check that loss and grad input match
        tols = dtype_tols(dtype)
        test_loss = test_loss.to(dtype=torch.float64, device="cpu")
        ref_loss = ref_loss.to(dtype=torch.float64, device="cpu")
        ref_loss = ref_loss.reshape(test_loss.size())
        test_grad_input = self.input_test.grad.to(dtype=torch.float64, device="cpu")
        ref_grad_input = self.input_ref.grad.to(dtype=torch.float64, device="cpu")
        ref_grad_input = ref_grad_input.reshape(test_grad_input.size())
        torch.testing.assert_close(test_loss, ref_loss, **tols)
        torch.testing.assert_close(test_grad_input, ref_grad_input, **tols)

        # Reset data
        self.input_test = None
        self.input_ref = None
        self.tar_test = None
        self.tar_ref = None

    def test_float32_input(self):
        self.generate_iters(5)
        self.generate_infra(True, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=False, label_smoothing=0, reduce_loss=True
            )

    def test_bfloat16_input(self):
        self.generate_iters(5)
        self.generate_infra(True, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.bfloat16, swap_dim=False, label_smoothing=0, reduce_loss=True
            )

    def test_swapped_input(self):
        self.generate_iters(5)
        self.generate_infra(True, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=True, label_smoothing=0, reduce_loss=True
            )

    def test_label_smoothing(self):
        self.generate_iters(3)
        self.generate_infra(True, 0.1)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=False, label_smoothing=0.1, reduce_loss=True
            )

    def test_non_reduced_loss(self):
        self.generate_iters(1)
        self.generate_infra(False, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32, swap_dim=False, label_smoothing=0, reduce_loss=False
            )

    def test_ignore_idx(self):
        self.generate_iters(5)
        self.generate_infra(False, 0)
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32,
                swap_dim=random.choice([True, False]),
                label_smoothing=0,
                reduce_loss=False,
                ignore_idx=True,
            )

    def test_ignore_idx_reduced_loss(self):
        """Test ignore_idx with reduce_loss=True"""
        self.generate_iters(5)
        self.generate_infra(True, 0)  # reduce_loss=True
        for i in range(self.iters):
            self.one_iteration_test(
                dtype=torch.float32,
                swap_dim=random.choice([True, False]),
                label_smoothing=0,
                reduce_loss=True,
                ignore_idx=True,
            )

    def test_z_loss(self):
        """Z-loss: loss and gradients must match a manual PyTorch reference."""
        batch, SQ, vocab = 2, 64, 8192
        z_loss_weight = 0.001

        inp_test = torch.randn(batch, SQ, vocab, dtype=torch.float32, device="cuda", requires_grad=True)
        inp_ref = inp_test.detach().clone().requires_grad_(True)
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")

        loss_te, log_sum_exp_te = parallel_cross_entropy(
            inp_test, tar, z_loss_weight=z_loss_weight, return_log_sum_exp=True
        )

        ref_ce = F.cross_entropy(inp_ref.view(-1, vocab), tar.view(-1), reduction="none").view(batch, SQ)
        log_sum_exp_ref = torch.logsumexp(inp_ref, dim=-1)
        ref_loss = ref_ce + z_loss_weight * torch.square(log_sum_exp_ref)

        tols = dtype_tols(torch.float32)
        torch.testing.assert_close(loss_te, ref_loss, **tols)
        torch.testing.assert_close(log_sum_exp_te, log_sum_exp_ref, **tols)

        loss_te.sum().backward()
        ref_loss.sum().backward()
        torch.testing.assert_close(inp_test.grad, inp_ref.grad, **tols)

    def test_z_loss_zero_weight(self):
        """z_loss_weight=0.0 must produce bit-identical results to the baseline."""
        batch, SQ, vocab = 2, 32, 4096
        inp = torch.randn(batch, SQ, vocab, dtype=torch.float32, device="cuda")
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")

        loss_base = parallel_cross_entropy(inp.clone(), tar)
        loss_zero = parallel_cross_entropy(inp.clone(), tar, z_loss_weight=0.0)
        assert torch.equal(loss_base, loss_zero), "z_loss_weight=0.0 must be bit-identical to the default"

    def test_z_loss_with_label_smoothing(self):
        """Z-loss and label smoothing must compose correctly."""
        batch, SQ, vocab = 2, 32, 4096
        z_loss_weight = 0.001
        label_smoothing = 0.1

        inp_test = torch.randn(batch, SQ, vocab, dtype=torch.float32, device="cuda", requires_grad=True)
        inp_ref = inp_test.detach().clone().requires_grad_(True)
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")

        loss_te = parallel_cross_entropy(inp_test, tar, label_smoothing=label_smoothing, z_loss_weight=z_loss_weight)

        ref_ce = F.cross_entropy(inp_ref.view(-1, vocab), tar.view(-1), label_smoothing=label_smoothing, reduction="none").view(batch, SQ)
        log_sum_exp_ref = torch.logsumexp(inp_ref, dim=-1)
        ref_loss = ref_ce + z_loss_weight * torch.square(log_sum_exp_ref)

        # Higher tolerance due to label-smoothing implementation differences
        torch.testing.assert_close(loss_te, ref_loss, rtol=2e-2, atol=0.1)

        loss_te.sum().backward()
        ref_loss.sum().backward()
        torch.testing.assert_close(inp_test.grad, inp_ref.grad, rtol=2e-2, atol=0.1)

    def test_z_loss_with_ignore_idx(self):
        """Ignored tokens must receive zero gradients even with z-loss enabled."""
        batch, SQ, vocab = 2, 32, 4096
        z_loss_weight = 0.001

        inp_test = torch.randn(batch, SQ, vocab, dtype=torch.float32, device="cuda", requires_grad=True)
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")
        tar[0, :5] = -100  # ignore first 5 positions in batch 0

        loss_te = parallel_cross_entropy(inp_test, tar, z_loss_weight=z_loss_weight)
        loss_te.sum().backward()

        assert torch.all(inp_test.grad[0, :5] == 0.0), "Ignored tokens must have zero gradients"

    def test_non_uniform_gradient_backward(self):
        """Non-uniform grad_output (loss masking) must produce correct input gradients.

        The original TE backward bug (PR #2139) always read grad_output[0] for all rows.
        With uniform grad_output (all-ones from .sum().backward()), this was invisible.
        This test explicitly uses non-uniform grad_output to catch that class of bug.
        """
        batch, SQ, vocab = 2, 32, 4096

        inp_test = torch.randn(batch, SQ, vocab, dtype=torch.float32, device="cuda", requires_grad=True)
        inp_ref = inp_test.detach().clone().requires_grad_(True)
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")

        # Non-uniform grad_output simulating loss masking (some tokens have zero weight)
        grad_output = torch.rand(batch, SQ, device="cuda")
        grad_output[0, :5] = 0.0   # mask first 5 positions in batch 0
        grad_output[1, -3:] = 0.0  # mask last 3 positions in batch 1

        loss_te = parallel_cross_entropy(inp_test, tar, 0.0, False, None)
        loss_te.backward(grad_output)

        loss_ref = F.cross_entropy(inp_ref.view(-1, vocab), tar.view(-1), reduction="none").view(batch, SQ)
        loss_ref.backward(grad_output)

        tols = dtype_tols(torch.float32)
        torch.testing.assert_close(inp_test.grad, inp_ref.grad, **tols)

    def test_log_sum_exp_zero_for_ignored(self):
        """Ignored positions must have log_sum_exp=0.0.

        The kernel returns early for y==ignore_idx before storing lse,
        leaving the tensor at its zero-initialized value.
        """
        batch, SQ, vocab = 2, 32, 4096

        inp = torch.randn(batch, SQ, vocab, dtype=torch.float32, device="cuda")
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")

        ignored = [(0, 3), (0, 7), (1, 0), (1, 15)]
        for b, s in ignored:
            tar[b, s] = -100

        _, log_sum_exp = parallel_cross_entropy(inp, tar, 0.0, False, None, return_log_sum_exp=True)

        for b, s in ignored:
            assert log_sum_exp[b, s].item() == 0.0, \
                f"log_sum_exp[{b},{s}] must be 0.0 for ignored token, got {log_sum_exp[b, s].item()}"

        # Non-ignored positions must have non-zero log_sum_exp
        assert log_sum_exp[0, 0].item() != 0.0, "Non-ignored token must have non-zero log_sum_exp"

    def test_log_sum_exp_non_differentiable(self):
        """log_sum_exp must be non-differentiable (ctx.mark_non_differentiable must have taken effect)."""
        batch, SQ, vocab = 2, 16, 1024

        inp = torch.randn(batch, SQ, vocab, dtype=torch.float32, device="cuda", requires_grad=True)
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")

        loss, log_sum_exp = parallel_cross_entropy(inp, tar, 0.0, False, None, return_log_sum_exp=True)

        assert not log_sum_exp.requires_grad, "log_sum_exp must not require gradients"
        assert log_sum_exp.grad_fn is None, "log_sum_exp must have no grad_fn"
        assert loss.requires_grad, "loss must still require gradients"

    def test_z_loss_bfloat16(self):
        """Z-loss must work correctly with BF16 input (the main production dtype in Megatron)."""
        batch, SQ, vocab = 2, 64, 8192
        z_loss_weight = 0.001

        inp_bf16 = torch.randn(batch, SQ, vocab, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        inp_ref = inp_bf16.detach().float().requires_grad_(True)
        tar = torch.randint(0, vocab, (batch, SQ), device="cuda")

        loss_te, log_sum_exp_te = parallel_cross_entropy(
            inp_bf16, tar, 0.0, False, None, z_loss_weight=z_loss_weight, return_log_sum_exp=True
        )

        ref_ce = F.cross_entropy(inp_ref.view(-1, vocab), tar.view(-1), reduction="none").view(batch, SQ)
        log_sum_exp_ref = torch.logsumexp(inp_ref, dim=-1)
        ref_loss = ref_ce + z_loss_weight * torch.square(log_sum_exp_ref)

        tols = dtype_tols(torch.bfloat16)
        torch.testing.assert_close(loss_te, ref_loss, **tols)
        torch.testing.assert_close(log_sum_exp_te, log_sum_exp_ref, **tols)

        loss_te.sum().backward()
        ref_loss.sum().backward()
        torch.testing.assert_close(inp_bf16.grad.float(), inp_ref.grad, **tols)
