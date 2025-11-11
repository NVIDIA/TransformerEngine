# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import torch
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

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

        # Forward pass
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
        ref_loss = test_loss.to(dtype=torch.float64, device="cpu")
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
