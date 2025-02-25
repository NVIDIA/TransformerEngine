# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import pytest
import torch
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy


class TestParallelCrossEntropy:

    def generate_iters(self, iters: int):
        self.iters = iters

    def generate_infra(self, reduce_loss: bool, label_smoothing: float):
        self.test_loss_func = parallel_cross_entropy
        self.ref_loss_func = torch.nn.CrossEntropyLoss(
            label_smoothing=label_smoothing, reduction="mean" if reduce_loss else "none"
        )

    def generate_input(self, dtype: torch.dtype, swap_dim: bool):

        SQ = random.choice([64, 128])
        batch = random.choice([1, 2])
        vocab = random.choice([64000, 128000])

        if swap_dim:
            self.input_test = torch.rand((SQ, batch, vocab), dtype=dtype).cuda()
            self.tar_test = torch.randint(0, vocab, (SQ, batch)).cuda()
        else:
            self.input_test = torch.rand((batch, SQ, vocab), dtype=dtype).cuda()
            self.tar_test = torch.randint(0, vocab, (batch, SQ)).cuda()

        self.input_ref = torch.reshape(self.input_test.clone().detach(), (batch * SQ, vocab))
        self.tar_ref = torch.reshape(self.tar_test.clone().detach(), (batch * SQ,))

    def one_iteration_test(
        self, dtype: torch.dtype, swap_dim: bool, label_smoothing: float, reduce_loss: bool
    ):

        self.generate_input(dtype, swap_dim)

        self.input_test.requires_grad_(True)
        self.input_ref.requires_grad_(True)

        test_loss = self.test_loss_func(
            self.input_test, self.tar_test, label_smoothing, reduce_loss, None
        )
        if reduce_loss:
            test_loss.backward()

        ref_loss = self.ref_loss_func(self.input_ref, self.tar_ref)
        if reduce_loss:
            ref_loss.backward()

        test_loss = torch.flatten(test_loss) if not reduce_loss else test_loss

        torch.testing.assert_close(test_loss, ref_loss, check_dtype=False)
        if reduce_loss:
            torch.testing.assert_close(
                torch.flatten(self.input_test.grad, start_dim=0, end_dim=1), self.input_ref.grad
            )

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
