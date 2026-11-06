# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import math
import random
import pytest
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common.recipe import Format, MXFP8BlockScaling
from transformer_engine.pytorch import is_mxfp8_available, parallel_cross_entropy
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer, MXFP8Tensor

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

    assert [tensor.dtype for tensor in saved_tensors] == [
        torch.bfloat16,
        torch.float32,
        torch.int64,
    ]
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


@pytest.mark.parametrize("reduce_loss", [False, True], ids=["none", "mean"])
@pytest.mark.parametrize("overwrite_input", [False, True], ids=["safe", "destructive"])
def test_parallel_cross_entropy_saved_state_and_buffer_reuse(reduce_loss, overwrite_input):
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
            reduce_loss=reduce_loss,
            overwrite_input=overwrite_input,
        )

    assert len(saved_tensors) == 3 + int(reduce_loss)
    saved_input, stats, saved_target, *optional_tensors = saved_tensors
    assert saved_input.shape == logits.shape
    assert saved_input.dtype == logits.dtype
    assert stats.shape == (target.numel(), 2)
    assert stats.dtype == torch.float32
    assert saved_target.numel() == target.numel()
    assert saved_target.dtype == torch.int64
    if reduce_loss:
        n_non_ignore = optional_tensors[0]
        assert n_non_ignore.shape == (1,)
        assert n_non_ignore.dtype == torch.int64
        assert n_non_ignore.item() == target.numel() - 1
    else:
        assert not optional_tensors

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
    with pytest.raises(ValueError, match="cuda"):
        parallel_cross_entropy(logits.cpu(), target.cpu())


def test_parallel_cross_entropy_deprecated_input_alias():
    """The deprecated _input keyword remains compatible with the replaced path."""

    logits = torch.randn(2, 3, 5, device="cuda")
    target = torch.zeros((2, 3), dtype=torch.int64, device="cuda")
    with pytest.warns(FutureWarning, match="_input"):
        alias_loss = parallel_cross_entropy(logits, target, _input=logits)
    direct_loss = parallel_cross_entropy(logits, target)
    torch.testing.assert_close(alias_loss, direct_loss)


def _logits_with_argmax_cases(shape, dtype):
    """BF16-representable logits whose rows exercise ties within and across 32K-wide blocks."""

    torch.manual_seed(7)
    values = torch.randn(shape, dtype=torch.bfloat16, device="cuda").to(dtype)
    vocab = shape[-1]
    peak = 8.0
    values[0, 0, [5, vocab - 30000]] = peak  # equal maxima in different blocks
    values[0, 1, [vocab - 30000, vocab - 4]] = peak  # equal maxima in later blocks
    values[0, 2, [100, 7]] = peak  # equal maxima in the same block
    values[0, 3, vocab - 1] = peak  # a strictly larger maximum in the last block
    values[0, 4] = 0.5  # a constant row
    return values


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("reduce_loss", [False, True], ids=["none", "mean"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1], ids=["plain", "smoothed"])
@pytest.mark.parametrize("overwrite_input", [False, True], ids=["safe", "destructive"])
def test_parallel_cross_entropy_return_argmax(dtype, reduce_loss, label_smoothing, overwrite_input):
    """The argmax matches torch.argmax, and loss and gradient do not change with it."""

    shape = (2, 5, 70000)
    values = _logits_with_argmax_cases(shape, dtype)
    target = torch.randint(0, shape[-1], shape[:-1], device="cuda")
    target[1, 1] = -100
    expected_argmax = torch.argmax(values, dim=-1)

    logits = values.clone().requires_grad_()
    loss = parallel_cross_entropy(
        logits, target, label_smoothing, reduce_loss, overwrite_input=overwrite_input
    )
    argmax_logits = values.clone().requires_grad_()
    argmax_loss, argmax = parallel_cross_entropy(
        argmax_logits,
        target,
        label_smoothing,
        reduce_loss,
        overwrite_input=overwrite_input,
        return_argmax=True,
    )

    assert argmax.dtype == torch.int64
    assert not argmax.requires_grad
    torch.testing.assert_close(argmax, expected_argmax, rtol=0, atol=0)

    external_grad = torch.randn_like(loss)
    loss.backward(external_grad)
    argmax_loss.backward(external_grad)
    torch.testing.assert_close(argmax_loss, loss, rtol=0, atol=0)
    torch.testing.assert_close(argmax_logits.grad, logits.grad, rtol=0, atol=0)


def test_parallel_cross_entropy_return_argmax_transposed_input():
    """Each argmax belongs to the row of the loss next to it."""

    values = _logits_with_argmax_cases((2, 5, 70000), torch.float32).transpose(0, 1)
    target = torch.randint(0, values.shape[-1], values.shape[:-1], device="cuda")
    loss, argmax = parallel_cross_entropy(values, target, return_argmax=True)
    assert argmax.shape == loss.shape == values.shape[:-1]
    torch.testing.assert_close(argmax, torch.argmax(values, dim=-1), rtol=0, atol=0)


def test_parallel_cross_entropy_return_argmax_nan():
    """As with torch.argmax, a NaN logit counts as the maximum and the first NaN wins, within a
    block and across blocks."""

    nan = float("nan")
    small = torch.tensor(
        [
            [8.0, -10.0, nan, -10.0],
            [nan, 1.0, 5.0, 2.0],
            [1.0, 5.0, 2.0, nan],
            [3.0, nan, 3.0, nan],
            [nan, nan, nan, nan],
            [float("inf"), 1.0, nan, 2.0],
        ],
        device="cuda",
    )
    large = torch.randn(4, 70000, device="cuda")
    pair = torch.tensor([nan, 50.0], device="cuda")
    large[0, [40000, 10]] = pair  # NaN in a later block than the maximum
    large[1, [10, 40000]] = pair  # NaN in an earlier block than the maximum
    large[2, [40001, 40000]] = pair  # NaN after the maximum in the same block
    large[3, [65000, 35000]] = nan  # NaN in two blocks
    for values in (small, large):
        for dtype in (torch.float32, torch.bfloat16):
            logits = values.to(dtype).unsqueeze(0)
            target = torch.zeros(logits.shape[:-1], dtype=torch.long, device="cuda")
            _, argmax = parallel_cross_entropy(logits, target, return_argmax=True)
            torch.testing.assert_close(argmax, torch.argmax(logits, dim=-1), rtol=0, atol=0)


mxfp8_available, reason_for_no_mxfp8 = is_mxfp8_available(return_reason=True)
_MXFP8_DTYPES = {"e4m3": tex.DType.kFloat8E4M3, "e5m2": tex.DType.kFloat8E5M2}
_MXFP8_BUFFERS = (
    "_rowwise_data",
    "_rowwise_scale_inv",
    "_columnwise_data",
    "_columnwise_scale_inv",
)


def _heavy_tailed_logits(shape, seed):
    """BF16 logits whose MXFP8 block scales vary across the row."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn(shape, device="cuda", generator=generator) * 3
    logits[..., :64] *= 8
    return logits.to(torch.bfloat16)


def _assert_same_mxfp8_bytes(grad, ref):
    """Same data and scale inverses. Only the valid part of the scale inverses is compared,
    since the quantizer leaves the padding of its scale layout uninitialized."""
    assert isinstance(grad, MXFP8Tensor), type(grad)
    rows = math.prod(grad.shape[:-1])
    cols = grad.shape[-1]
    valid = {
        "_rowwise_scale_inv": (rows, math.ceil(cols / 32)),
        "_columnwise_scale_inv": (math.ceil(rows / 32), cols),
    }
    for name in _MXFP8_BUFFERS:
        a, b = getattr(grad, name), getattr(ref, name)
        assert (a is None) == (b is None), name
        if a is None:
            continue
        assert a.shape == b.shape, (name, a.shape, b.shape)
        if name in valid:
            a, b = a[: valid[name][0], : valid[name][1]], b[: valid[name][0], : valid[name][1]]
        torch.testing.assert_close(a.view(torch.uint8), b.view(torch.uint8), rtol=0, atol=0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize(
    "shape",
    [(2, 64, 4096), (512, 1, 129280), (128, 2, 32000), (96, 1, 4096), (128, 1, 4000)],
    ids=["v4096", "v129280", "v32000", "rows96_fallback", "v4000_fallback"],
)
@pytest.mark.parametrize("fp8_dtype", ["e4m3", "e5m2"])
@pytest.mark.parametrize("usage", ["both", "rowwise", "columnwise"])
@pytest.mark.parametrize("reduce_loss", [False, True], ids=["none", "mean"])
@pytest.mark.parametrize("label_smoothing", [0.0, 0.1], ids=["plain", "smoothed"])
@pytest.mark.parametrize("overwrite_input", [False, True], ids=["safe", "destructive"])
def test_parallel_cross_entropy_mxfp8_grad(
    shape, fp8_dtype, usage, reduce_loss, label_smoothing, overwrite_input
):
    """The MXFP8 gradient has the bytes of the BF16 gradient quantized by the quantizer, and
    the loss is unchanged."""

    values = _heavy_tailed_logits(shape, seed=sum(shape))
    target = torch.randint(0, shape[-1], shape[:-1], device="cuda")
    target.view(-1)[:3] = -100

    def run(grad_quantizer):
        logits = values.clone().requires_grad_()
        loss = parallel_cross_entropy(
            logits,
            target,
            label_smoothing,
            reduce_loss,
            overwrite_input=overwrite_input,
            grad_quantizer=grad_quantizer,
        )
        external_grad = (
            torch.full_like(loss, 0.37)
            if reduce_loss
            else torch.linspace(0.5, 1.5, loss.numel(), device="cuda").reshape(loss.shape)
        )
        loss.backward(external_grad)
        return loss.detach(), logits.grad

    quantizer = MXFP8Quantizer(
        _MXFP8_DTYPES[fp8_dtype],
        rowwise=usage != "columnwise",
        columnwise=usage != "rowwise",
    )
    ref_loss, ref_grad = run(None)
    loss, grad = run(quantizer)

    torch.testing.assert_close(loss, ref_loss, rtol=0, atol=0)
    _assert_same_mxfp8_bytes(grad, quantizer(ref_grad))
    # The dequantized gradient is the BF16 gradient up to the MXFP8 rounding error: relative
    # for normal FP8 values, and absolute (below 2^-16 of the largest magnitude) for values
    # that are subnormal or flushed to zero in FP8.
    rel_err = 2.0**-3 if fp8_dtype == "e4m3" else 2.0**-2
    abs_err = ref_grad.abs().max().item() * 2.0**-16
    torch.testing.assert_close(grad.dequantize(), ref_grad, rtol=rel_err, atol=abs_err)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_parallel_cross_entropy_mxfp8_grad_unfused_cases():
    """Inputs the fused kernel does not write still return the quantized gradient."""

    values = torch.randn(2, 64, 4096, device="cuda")
    target = torch.randint(0, 4096, (2, 64), device="cuda")
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    # FP32 input, and a non-contiguous BF16 input. torch.autograd.grad returns the gradient as
    # the loss returns it (AccumulateGrad would re-layout it for the non-contiguous leaf).
    for logits_values in (values, values.to(torch.bfloat16).transpose(0, 1)):
        logits_target = target.reshape(logits_values.shape[:-1])
        logits = logits_values.clone().requires_grad_()
        (ref_grad,) = torch.autograd.grad(
            parallel_cross_entropy(logits, logits_target).sum(), logits
        )
        (grad,) = torch.autograd.grad(
            parallel_cross_entropy(logits, logits_target, grad_quantizer=quantizer).sum(), logits
        )
        _assert_same_mxfp8_bytes(grad, quantizer(ref_grad))

    swizzled = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    swizzled.optimize_for_gemm = True
    logits = values.to(torch.bfloat16).requires_grad_()
    parallel_cross_entropy(logits, target, grad_quantizer=swizzled).sum().backward()
    assert isinstance(logits.grad, MXFP8Tensor) and logits.grad._with_gemm_swizzled_scales

    internal = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    internal.internal = True
    with pytest.raises(ValueError, match="internal"):
        parallel_cross_entropy(values, target, grad_quantizer=internal)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
@pytest.mark.parametrize("overwrite_input", [False, True], ids=["safe", "destructive"])
def test_parallel_cross_entropy_mxfp8_grad_lm_head(overwrite_input):
    """An MXFP8 linear layer followed by the loss gets the same input and weight gradients when
    the loss returns the MXFP8 gradient of the logits."""

    tokens, hidden, vocab = 512, 1024, 32768
    torch.manual_seed(0)
    linear = te.Linear(hidden, vocab, bias=False, params_dtype=torch.bfloat16, device="cuda")
    inp = torch.randn(tokens, 1, hidden, device="cuda", dtype=torch.bfloat16)
    target = torch.randint(0, vocab, (tokens, 1), device="cuda")
    recipe = MXFP8BlockScaling(fp8_format=Format.E4M3)

    def run(grad_quantizer):
        linear.weight.grad = None
        x = inp.clone().requires_grad_()
        with te.autocast(enabled=True, recipe=recipe):
            logits = linear(x)
        loss = parallel_cross_entropy(
            logits,
            target,
            overwrite_input=overwrite_input,
            grad_quantizer=grad_quantizer,
        )
        loss.sum().backward()
        return loss.detach(), x.grad, linear.weight.grad.clone()

    ref = run(None)
    out = run(MXFP8Quantizer(tex.DType.kFloat8E4M3))
    for a, b in zip(out, ref):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
def test_parallel_cross_entropy_mxfp8_grad_cuda_graph():
    """The MXFP8 backward is CUDA graph capturable."""

    shape = (256, 1, 8192)
    static_values = _heavy_tailed_logits(shape, seed=4)
    static_target = torch.randint(0, shape[-1], shape[:-1], device="cuda")
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    buffer = torch.empty_like(static_values)
    outputs = {}

    def step():
        buffer.copy_(static_values)
        logits = buffer.detach().requires_grad_()
        loss = parallel_cross_entropy(
            logits, static_target, overwrite_input=True, grad_quantizer=quantizer
        )
        outputs["grad"] = torch.autograd.grad(loss.sum(), logits)[0]

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    static_values.copy_(_heavy_tailed_logits(shape, seed=5))
    static_target.copy_(torch.randint(0, shape[-1], shape[:-1], device="cuda"))
    graph.replay()
    logits = static_values.clone().requires_grad_()
    parallel_cross_entropy(logits, static_target).sum().backward()
    _assert_same_mxfp8_bytes(outputs["grad"], quantizer(logits.grad))
