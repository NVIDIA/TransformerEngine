# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Unit tests for the GTP-agnostic DistributedWeight protocol and dispatchers.

These tests exercise TE's weight-parallelism extension point in isolation, with
a tiny in-repo ``FakeDistributedWeight`` stub standing in for a real implementer
(e.g. Megatron's GTPShardedParam). No GPU, process group, or Megatron import is
required: the whole contract lives in TE and is verified against the fake.
"""

import torch

from transformer_engine.pytorch.distributed_weight import (
    DistributedWeight,
    is_distributed_weight,
    materialize_weights_for_forward,
    materialize_weights_for_backward,
    finalize_weight_grads,
)


class FakeDistributedWeight(torch.Tensor):
    """Minimal DistributedWeight implementer for dispatcher tests.

    Records how it was called and returns marker tensors so the dispatcher's
    behavior (delegation, list normalization, no-op fallback) is observable.
    """

    is_distributed_weight = True

    def __new__(cls, group_size=1):
        t = torch.zeros(1).as_subclass(cls)
        t.group_size = group_size
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
        return torch.full((2, 2), -1.0)


def test_protocol_runtime_checkable():
    """A conforming object passes isinstance; a plain tensor does not."""
    assert isinstance(FakeDistributedWeight(), DistributedWeight)
    assert not isinstance(torch.zeros(2), DistributedWeight)


def test_is_distributed_weight():
    assert is_distributed_weight(FakeDistributedWeight())
    assert not is_distributed_weight(torch.zeros(2))
    assert not is_distributed_weight(torch.nn.Parameter(torch.zeros(2)))


def test_forward_noop_on_plain_tensor():
    """Plain weights pass through unchanged — the critical non-regression."""
    w = torch.nn.Parameter(torch.randn(4, 4))
    out = materialize_weights_for_forward([w])
    assert out == [w]
    assert out[0] is w


def test_forward_dispatches_single_weight():
    """Linear (N=1): dispatcher delegates and returns a length-1 list."""
    w = FakeDistributedWeight(group_size=1)
    out = materialize_weights_for_forward([w])
    assert isinstance(out, list) and len(out) == 1
    assert torch.equal(out[0], torch.zeros(2, 2))
    # The forward dispatcher owns the forward semantic; the leader is delegated to once.
    assert w.calls == ["fwd"]


def test_forward_dispatches_grouped_weights():
    """GroupedLinear (N=k): one coalesced call, full list returned."""
    w = FakeDistributedWeight(group_size=3)
    out = materialize_weights_for_forward([w, w, w])
    assert isinstance(out, list) and len(out) == 3
    # Leader was called exactly once (coalesced), not once per weight.
    assert len(w.calls) == 1


def test_backward_dispatch_and_noop():
    w = FakeDistributedWeight(group_size=2)
    out = materialize_weights_for_backward([w, w])
    assert len(out) == 2 and torch.equal(out[0], torch.full((2, 2), 10.0))

    plain = [torch.zeros(2), torch.ones(2)]
    assert materialize_weights_for_backward(plain) == plain


def test_finalize_grads_dispatch_and_noop():
    w = FakeDistributedWeight(group_size=1)
    wgrads = [torch.zeros(2, 2)]
    out = finalize_weight_grads([w], wgrads)
    assert len(out) == 1 and torch.equal(out[0], torch.full((2, 2), 100.0))

    # No-op path leaves the grads untouched.
    plain_w = [torch.nn.Parameter(torch.zeros(2))]
    g = [torch.ones(2)]
    assert finalize_weight_grads(plain_w, g) == g


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: PASS")
