# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import random
import pytest
import torch
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


def test_non_contiguous_transposed_input():
    """Regression test: stride(-2) != shape[-1] should not produce wrong results."""
    s, b, v = 4, 2, 8
    torch.manual_seed(42)
    logits = torch.randn(s, b, v, device="cuda")
    target = torch.randint(0, v, (b, s), device="cuda")

    logits_transposed = logits.transpose(0, 1)  # stride(-2) != shape[-1]
    logits_contiguous = logits_transposed.contiguous()

    assert logits_transposed.stride(-1) == 1
    assert logits_transposed.stride(-2) != logits_transposed.shape[-1]

    loss_t = parallel_cross_entropy(logits_transposed, target, 0.0, False, None)
    loss_c = parallel_cross_entropy(logits_contiguous, target, 0.0, False, None)

    assert torch.allclose(
        loss_t, loss_c
    ), f"Non-contiguous transposed input gave wrong results: {loss_t} vs {loss_c}"


def test_bfloat16_unreduced_external_grad():
    """Apply per-token loss scaling before rounding the BF16 input gradient."""
    logits = torch.zeros(2, 2, 3, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.tensor([[0, 1], [2, -100]], device="cuda")
    logits_before = logits.detach().clone()
    external_grad = torch.tensor([[0.01, 0.03], [0.1, 0.7]], device="cuda")
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        loss = parallel_cross_entropy(logits, target, 0.0, False, None)

    assert len(saved_tensors) == 4
    assert saved_tensors[0].dtype == torch.bfloat16
    torch.testing.assert_close(logits, logits_before, rtol=0.0, atol=0.0)
    loss.backward(external_grad)

    ref_logits = logits_before.float().requires_grad_()
    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits.reshape(-1, ref_logits.size(-1)), target.reshape(-1), reduction="none"
    ).reshape_as(target)
    ref_loss.backward(external_grad)
    expected_grad = ref_logits.grad.to(torch.bfloat16)

    torch.testing.assert_close(logits.grad, expected_grad, rtol=0.0, atol=0.0)


def test_bfloat16_loss_matches_float32_input():
    """BF16 and exactly equivalent FP32 logits should produce nearly identical losses."""
    torch.manual_seed(42)
    logits = torch.randn(1, 64, 64000, dtype=torch.bfloat16, device="cuda")
    target = torch.randint(0, logits.size(-1), logits.shape[:-1], device="cuda")

    bf16_loss = parallel_cross_entropy(logits, target, 0.0, False, None)
    fp32_loss = parallel_cross_entropy(logits.float(), target, 0.0, False, None)

    # The input values are identical, but dtype-specialized Triton reductions may
    # differ by a small number of FP32 rounding steps.
    fp32_eps = torch.finfo(torch.float32).eps
    torch.testing.assert_close(bf16_loss, fp32_loss, rtol=2 * fp32_eps, atol=2 * fp32_eps)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("reduce_loss", [False, True], ids=["none", "mean"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1], ids=["plain", "smoothed"])
@pytest.mark.parametrize("overwrite_input", [False, True], ids=["safe", "destructive"])
def test_parallel_cross_entropy_matches_pytorch(
    dtype,
    reduce_loss,
    label_smoothing,
    overwrite_input,
):
    """Check loss and externally-scaled gradients against PyTorch."""

    torch.manual_seed(1234)
    shape = (2, 5, 37)
    target = torch.randint(0, shape[-1], shape[:-1], device="cuda")
    target[0, 1] = -100
    # Use the same BF16-representable inputs for both dtypes so that the FP32
    # loss comparison measures only the implementation's arithmetic.
    values = torch.randn(shape, dtype=torch.bfloat16, device="cuda").to(dtype)

    logits = values.clone().requires_grad_()
    ref_logits = values.float().clone().requires_grad_()

    loss = parallel_cross_entropy(
        logits,
        target,
        label_smoothing,
        reduce_loss,
        overwrite_input=overwrite_input,
    )
    ref_loss = torch.nn.functional.cross_entropy(
        ref_logits.reshape(-1, shape[-1]),
        target.reshape(-1),
        label_smoothing=label_smoothing,
        reduction="mean" if reduce_loss else "none",
    )
    if reduce_loss:
        ref_loss = ref_loss.reshape_as(loss)
    else:
        ref_loss = ref_loss.reshape_as(target)

    external_grad = (
        torch.full_like(loss, 0.37) if reduce_loss else torch.randn_like(loss, dtype=torch.float32)
    )
    loss.backward(external_grad)
    ref_loss.backward(external_grad)

    assert loss.dtype == ref_loss.dtype == torch.float32
    torch.testing.assert_close(loss, ref_loss, **dtype_tols(torch.float32))
    expected_grad = ref_logits.grad.to(dtype)
    torch.testing.assert_close(logits.grad, expected_grad, **dtype_tols(dtype))


@pytest.mark.parametrize("overwrite_input", [False, True], ids=["safe", "destructive"])
def test_parallel_cross_entropy_saved_state_and_buffer_reuse(overwrite_input):
    """The saved input buffer is input-typed and becomes the returned derivative."""

    torch.manual_seed(42)
    logits = torch.randn(2, 3, 11, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    target = torch.tensor([[0, -100, 4], [7, 2, 10]], device="cuda")
    before = logits.detach().clone()
    version_before = logits._version
    other_consumer = logits.square().sum()
    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        loss = parallel_cross_entropy(
            logits,
            target,
            label_smoothing=0.1,
            overwrite_input=overwrite_input,
        )

    assert len(saved_tensors) == 4
    saved_input, stats, saved_target, n_non_ignore = saved_tensors
    assert saved_input.shape == logits.shape
    assert saved_input.dtype == logits.dtype
    assert stats.shape == (target.numel(), 2)
    assert stats.dtype == torch.float32
    assert saved_target.numel() == target.numel()
    assert saved_target.dtype == torch.int64
    assert n_non_ignore.shape == (1,)
    assert n_non_ignore.dtype == torch.int64
    assert n_non_ignore.item() == target.numel() - 1

    if overwrite_input:
        assert saved_input.data_ptr() == logits.data_ptr()
    else:
        assert saved_input.data_ptr() != logits.data_ptr()
    torch.testing.assert_close(logits, before, rtol=0.0, atol=0.0)
    assert logits._version == version_before

    expected_max = before.float().amax(dim=-1).reshape(-1)
    expected_denominator = (
        torch.exp(before.float() - expected_max.reshape(logits.shape[0], logits.shape[1], 1))
        .sum(dim=-1)
        .reshape(-1)
    )
    torch.testing.assert_close(stats[:, 0], expected_max, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(stats[:, 1], expected_denominator, rtol=1e-6, atol=1e-6)

    external_grad = torch.randn_like(loss)
    loss.backward(external_grad)
    assert logits._version == version_before + int(overwrite_input)

    # Backward writes directly into the tensor saved by forward.
    torch.testing.assert_close(saved_input, logits.grad, rtol=0.0, atol=0.0)
    if overwrite_input:
        assert not torch.equal(logits, before)
    else:
        torch.testing.assert_close(logits, before, rtol=0.0, atol=0.0)

    if overwrite_input:
        with pytest.raises(RuntimeError, match="modified by an inplace operation"):
            other_consumer.backward()
    else:
        other_consumer.backward()


@pytest.mark.parametrize("layout", ["transpose", "strided_vocab"])
def test_parallel_cross_entropy_safe_mode_supported_layouts(layout):
    """Safe mode copies non-contiguous logical layouts directly into its work buffer."""

    torch.manual_seed(17)
    if layout == "transpose":
        values = torch.randn(3, 2, 13, device="cuda").transpose(0, 1)
    else:
        values = torch.randn(2, 3, 26, device="cuda")[..., ::2]
    logits = values.detach().requires_grad_()
    reference = values.detach().clone().requires_grad_()
    target = torch.randint(0, values.shape[-1], values.shape[:-1], device="cuda")
    external_grad = torch.randn(values.shape[:-1], device="cuda")

    loss = parallel_cross_entropy(logits, target)
    ref_loss = torch.nn.functional.cross_entropy(
        reference.reshape(-1, reference.shape[-1]),
        target.reshape(-1),
        reduction="none",
    ).reshape_as(target)
    loss.backward(external_grad)
    ref_loss.backward(external_grad)

    torch.testing.assert_close(loss, ref_loss, **dtype_tols(torch.float32))
    torch.testing.assert_close(logits.grad, reference.grad, **dtype_tols(torch.float32))


def test_parallel_cross_entropy_validation_errors():
    """Reject inputs that violate dtype, shape, device, or overwrite assumptions."""

    target = torch.zeros((2, 3), dtype=torch.int64, device="cuda")
    logits = torch.randn(2, 3, 5, device="cuda")

    with pytest.raises(ValueError, match="3D"):
        parallel_cross_entropy(logits.reshape(6, 5), target)
    with pytest.raises(TypeError, match="BF16 or FP32"):
        parallel_cross_entropy(logits.half(), target)
    with pytest.raises(TypeError, match="int64"):
        parallel_cross_entropy(logits, target.int())
    with pytest.raises(ValueError, match="one target"):
        parallel_cross_entropy(logits, target[:, :2])
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        parallel_cross_entropy(logits, target, label_smoothing=1.1)
    with pytest.raises(ValueError, match="contiguous"):
        parallel_cross_entropy(
            logits.transpose(0, 1),
            target.transpose(0, 1),
            overwrite_input=True,
        )
    with pytest.raises(ValueError, match="CUDA"):
        parallel_cross_entropy(logits.cpu(), target.cpu())


def test_parallel_cross_entropy_deprecated_input_alias():
    """The deprecated _input keyword remains compatible with the replaced path."""

    logits = torch.randn(2, 3, 5, device="cuda")
    target = torch.zeros((2, 3), dtype=torch.int64, device="cuda")
    with pytest.warns(FutureWarning, match="_input"):
        alias_loss = parallel_cross_entropy(logits, target, _input=logits)
    direct_loss = parallel_cross_entropy(logits, target)
    torch.testing.assert_close(alias_loss, direct_loss)
