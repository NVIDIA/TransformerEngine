# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for the GTP-agnostic DistributedWeight protocol and dispatchers.

These tests exercise TE's weight-parallelism extension point in isolation, with
a tiny in-repo ``FakeDistributedWeight`` stub standing in for a real implementer
(e.g. Megatron's GTPShardedParam). No GPU, process group, or Megatron import is
required: the whole contract lives in TE and is verified against the fake.
"""

import pytest
import torch

from transformer_engine.pytorch.distributed_weight import (
    DistributedWeight,
    is_distributed_weight,
    materialize_weight_for_forward,
    materialize_weight_for_backward,
    finalize_weight_grads,
    weight_grad_buffers,
    weight_grad_dtype,
)


class FakeDistributedWeight(torch.Tensor):
    """Minimal DistributedWeight implementer for dispatcher tests.

    Records how it was called and returns marker tensors so the dispatcher's
    behavior (delegation, list normalization, no-op fallback) is observable.
    """

    is_distributed_weight = True

    def __new__(cls, group_size=1, marker=-1.0):
        t = torch.zeros(1).as_subclass(cls)
        t.group_size = group_size
        # Distinguishes this member's grad_buffer from its peers'.
        t.marker = marker
        t.calls = []
        return t

    def materialize_group_for_forward(self):
        self.calls.append("fwd")
        out = [torch.full((2, 2), float(i)) for i in range(self.group_size)]
        # Match the real GTP contract: single weight returns a bare tensor.
        return out if self.group_size > 1 else out[0]

    def materialize_group_for_backward(self, **kwargs):
        self.calls.append(("bwd", kwargs))
        out = [torch.full((2, 2), float(10 + i)) for i in range(self.group_size)]
        return out if self.group_size > 1 else out[0]

    def finalize_group_grads(self, wgrads, **kwargs):
        self.calls.append(("finalize", wgrads))
        wl = wgrads if isinstance(wgrads, (list, tuple)) else [wgrads]
        out = [w + 100 for w in wl]
        return out if self.group_size > 1 else out[0]

    def grad_buffer(self):
        return torch.full((2, 2), self.marker)


class FakeNonTensorWeight:
    """DistributedWeight-shaped object that is NOT a torch.Tensor (contract violation)."""

    is_distributed_weight = True

    def materialize_group_for_forward(self):
        return torch.zeros(2, 2)

    def materialize_group_for_backward(self, **kwargs):
        return torch.zeros(2, 2)

    def finalize_group_grads(self, wgrads, **kwargs):
        return wgrads

    def grad_buffer(self):
        return torch.zeros(2, 2)


def test_protocol_runtime_checkable():
    """A conforming object passes isinstance; a plain tensor does not."""
    assert isinstance(FakeDistributedWeight(), DistributedWeight)
    assert not isinstance(torch.zeros(2), DistributedWeight)


def test_is_distributed_weight():
    assert is_distributed_weight(FakeDistributedWeight())
    assert not is_distributed_weight(torch.zeros(2))
    assert not is_distributed_weight(torch.nn.Parameter(torch.zeros(2)))


def test_non_tensor_implementer_rejected():
    """Implementers must be torch.Tensor subclasses; a non-Tensor fails loudly."""
    with pytest.raises(TypeError, match="torch.Tensor subclass"):
        is_distributed_weight(FakeNonTensorWeight())


def test_forward_noop_on_plain_tensor():
    """Plain weights pass through unchanged — the critical non-regression."""
    w = torch.nn.Parameter(torch.randn(4, 4))
    out = materialize_weight_for_forward(w)
    assert out == [w]
    assert out[0] is w


@pytest.mark.parametrize("group_size", [1, 3])
def test_forward_dispatches(group_size):
    """Linear (N=1) and GroupedLinear (N=k): one coalesced call, full list returned."""
    w = FakeDistributedWeight(group_size=group_size)
    out = materialize_weight_for_forward(w)
    assert isinstance(out, list) and len(out) == group_size
    # Leader is delegated to exactly once (coalesced), not once per weight.
    assert w.calls == ["fwd"]


def test_forward_accepts_weight_list():
    """The dispatcher accepts the full per-expert list; the leader (index 0) coalesces it."""
    leader = FakeDistributedWeight(group_size=3)
    followers = [torch.zeros(2, 2), torch.zeros(2, 2)]
    out = materialize_weight_for_forward([leader, *followers])
    assert isinstance(out, list) and len(out) == 3
    # Leader delegated exactly once; the follower entries are not materialized separately.
    assert leader.calls == ["fwd"]


def test_forward_noop_on_plain_weight_list():
    """A non-distributed weight list passes through unchanged (all N returned)."""
    ws = [torch.nn.Parameter(torch.randn(2, 2)) for _ in range(3)]
    out = materialize_weight_for_forward(ws)
    assert out == ws


@pytest.mark.parametrize("group_size", [1, 2])
def test_backward_dispatches(group_size):
    w = FakeDistributedWeight(group_size=group_size)
    out = materialize_weight_for_backward(w)
    assert len(out) == group_size
    assert torch.equal(out[0], torch.full((2, 2), 10.0))


def test_backward_accepts_weight_list():
    """Backward dispatcher also accepts the full per-expert list; the leader coalesces it."""
    leader = FakeDistributedWeight(group_size=2)
    out = materialize_weight_for_backward([leader, torch.zeros(2, 2)])
    assert isinstance(out, list) and len(out) == 2
    assert torch.equal(out[0], torch.full((2, 2), 10.0))


def test_backward_noop_on_plain_tensor():
    plain = torch.zeros(2)
    assert materialize_weight_for_backward(plain) == [plain]


def test_backward_noop_on_plain_weight_list():
    """A non-distributed weight list passes through unchanged (all N returned)."""
    ws = [torch.nn.Parameter(torch.randn(2, 2)) for _ in range(3)]
    assert materialize_weight_for_backward(ws) == ws


@pytest.mark.parametrize("group_size", [1, 2])
def test_finalize_grads_dispatches(group_size):
    w = FakeDistributedWeight(group_size=group_size)
    wgrads = [torch.zeros(2, 2) for _ in range(group_size)]
    out = finalize_weight_grads(w, wgrads)
    assert len(out) == group_size
    assert torch.equal(out[0], torch.full((2, 2), 100.0))


def test_finalize_grads_noop_on_plain_tensor():
    """No-op path leaves the grads untouched."""
    plain_w = torch.nn.Parameter(torch.zeros(2))
    g = [torch.ones(2)]
    assert finalize_weight_grads(plain_w, g) == g


@pytest.mark.parametrize("group_size", [1, 2])
def test_weight_grad_buffers_dispatches(group_size):
    """EVERY member supplies its own buffer -- the leader's is not reused for the group.

    Distinct markers catch a ``[weights[0].grad_buffer()] * N`` implementation, which would hand
    the whole group one buffer and let the per-expert GEMMs clobber each other.
    """
    group = [
        FakeDistributedWeight(group_size=group_size, marker=float(i)) for i in range(group_size)
    ]
    out = weight_grad_buffers(group, (2, 2), torch.bfloat16, "cpu")
    assert [b[0, 0].item() for b in out] == [float(i) for i in range(group_size)]


def test_weight_grad_buffers_rejects_mismatched_buffer():
    """A shard-shaped buffer would let the GEMM write past the end -- fail loudly instead."""
    w = FakeDistributedWeight()
    with pytest.raises(RuntimeError, match="wgrad GEMM needs"):
        weight_grad_buffers([w], (4, 5), torch.bfloat16, "cpu")


def test_weight_grad_buffers_allocates_for_plain_weights():
    """Plain weights have no buffer of their own, so they get fresh scratch instead."""
    plain = [torch.nn.Parameter(torch.zeros(2)) for _ in range(3)]
    out = weight_grad_buffers(plain, (4, 5), torch.bfloat16, "cpu")
    assert len(out) == 3
    assert all(tuple(b.shape) == (4, 5) and b.dtype is torch.bfloat16 for b in out)


@pytest.mark.parametrize("group_size", [1, 2])
def test_weight_grad_dtype_dispatches(group_size):
    """A distributed weight types its wgrad from main_grad, not from the compute dtype."""
    w = FakeDistributedWeight(group_size=group_size)
    w.main_grad = torch.zeros(2, 2, dtype=torch.float32)
    assert weight_grad_dtype(w, torch.bfloat16) == torch.float32
    assert weight_grad_dtype([w], torch.bfloat16) == torch.float32


def test_weight_grad_dtype_without_main_grad():
    """No main_grad yet (pre-DDP): fall back to the compute dtype rather than raising."""
    w = FakeDistributedWeight()
    assert weight_grad_dtype(w, torch.bfloat16) == torch.bfloat16


def test_weight_grad_dtype_noop_on_plain_tensor():
    """A plain weight keeps the compute dtype even when it carries an fp32 main_grad.

    Without weight sharding the wgrad is either accumulated into main_grad by the GEMM or
    returned as an ordinary .grad, so widening here would only cost memory.
    """
    plain_w = torch.nn.Parameter(torch.zeros(2))
    plain_w.main_grad = torch.zeros(2, dtype=torch.float32)
    assert weight_grad_dtype(plain_w, torch.bfloat16) == torch.bfloat16
