# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DistributedWeight dispatch in the fusible ``ops.GroupedLinear`` (unfused fallback path).

TE ships no DistributedWeight implementer (GTP etc. live in the caller), so this validates the
*dispatch wiring* with an in-repo fake. The fake applies a DISTINCT, observable scale in each
materialize hook so every plumbing point is independently checked -- a wiring bug fails a specific
assertion:

  * ``materialize_group_for_forward`` scales the weights by ``FWD_SCALE``  -> the fwd GEMM must use
    the materialized weights, so ``out == FWD_SCALE * plain_out``.
  * ``materialize_group_for_backward`` scales by ``BWD_SCALE``            -> dgrad must use the
    re-materialized weights, so ``dgrad == BWD_SCALE * plain_dgrad``.
  * ``finalize_group_grads`` reduce-scatters the wgrads into ``main_grad`` in-place and returns a
    dummy -> the real wgrad lands in ``main_grad`` and the ops path returns a throwaway ``.grad``
    (it discards finalize's return; see DistributedWeight.finalize_group_grads).

Real all-gather / reduce-scatter math is exercised by the caller's distributed tests; here the fake
is single-process and only proves the ops.GroupedLinear integration routes weights/grads through
the hooks (and does not bypass them).
"""

import pytest
import torch

import transformer_engine.pytorch as te

# Distinct powers of two so each scale commutes exactly through the (possibly TF32) GEMM rounding,
# making the linear scale relations bit-exact under a tight tolerance.
FWD_SCALE = 2.0
BWD_SCALE = 4.0


class _FakeDistWeight(torch.nn.Parameter):
    """Single-process fake DistributedWeight leader for dispatch testing.

    Each materialize hook applies its own scale (see module docstring) so fwd/bwd routing is
    observable; ``finalize_group_grads`` models the real main-grad contract -- reduce-scatter (here
    an identity accumulate) the wgrads into each shard's ``main_grad`` in-place, flag
    ``grad_added_to_main_grad``, and return a dummy that the ops path discards.
    """

    is_distributed_weight = True

    def materialize_group_for_forward(self):
        self.calls["fwd"] += 1
        return [w * FWD_SCALE for w in self._group]

    def materialize_group_for_backward(self, **kwargs):
        self.calls["bwd"] += 1
        return [w * BWD_SCALE for w in self._group]

    def finalize_group_grads(self, wgrads, **kwargs):
        self.calls["finalize"] += 1
        wl = list(wgrads) if isinstance(wgrads, (list, tuple)) else [wgrads]
        for w, g in zip(self._group, wl):
            w.main_grad.add_(g.to(w.main_grad.dtype))  # in-place accumulate into main_grad
            w.grad_added_to_main_grad = True
        return [torch.zeros_like(g) for g in wl]  # dummy grads (real value is now in main_grad)

    def grad_buffer(self):
        return self.data


def _make_fake_dist_leader(op, num_gemms):
    """Replace ``op.weight0`` with a fake distributed leader referencing the whole group."""
    w0 = op.weight0
    leader = _FakeDistWeight(w0.data)
    leader.calls = {"fwd": 0, "bwd": 0, "finalize": 0}
    leader._group = [leader] + [getattr(op, f"weight{i}") for i in range(1, num_gemms)]
    op.weight0 = leader
    return leader


@pytest.mark.parametrize("num_gemms", [2, 4])
def test_ops_grouped_linear_distributed_weight_dispatch(num_gemms):
    """Every DistributedWeight hook must be routed through the GEMM flow (and not bypassed).

    fwd/bwd use the scaled materialized weights; finalize reduce-scatters the wgrad into
    ``main_grad`` in-place, and the ops path returns a throwaway dummy ``.grad``.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(0)
    # fp32 with power-of-two scales: each scale commutes exactly through the GEMM rounding (even
    # TF32), so the linear scale relations below are bit-exact under a tight tolerance.
    in_f, out_f, total_tokens = 32, 64, num_gemms * 8
    dtype, device = torch.float32, "cuda"

    op = te.ops.GroupedLinear(num_gemms, in_f, out_f, bias=False, device=device, dtype=dtype)
    reference = te.ops.GroupedLinear(num_gemms, in_f, out_f, bias=False, device=device, dtype=dtype)
    reference.load_state_dict(op.state_dict())

    leader = _make_fake_dist_leader(op, num_gemms)
    for i in range(num_gemms):
        w = getattr(op, f"weight{i}")
        w.main_grad = torch.zeros((out_f, in_f), dtype=torch.float32, device=device)
        w.grad_added_to_main_grad = False  # DDP initializes this on every param

    m_splits = [total_tokens // num_gemms] * num_gemms
    m_splits[-1] += total_tokens - sum(m_splits)
    split_sizes = torch.tensor(m_splits, dtype=torch.int64, device=device)

    x = torch.randn(total_tokens, in_f, dtype=dtype, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)

    out = op(x, split_sizes)
    out.sum().backward()
    ref_out = reference(ref_x, split_sizes)
    ref_out.sum().backward()

    # All three dispatch hooks actually fired.
    assert leader.calls["fwd"] > 0 and leader.calls["bwd"] > 0 and leader.calls["finalize"] > 0

    tols = dict(rtol=1e-5, atol=1e-5)
    # fwd used the materialized (FWD_SCALE) weights.
    torch.testing.assert_close(out, FWD_SCALE * ref_out, **tols)
    # dgrad used the re-materialized (BWD_SCALE) weights.
    torch.testing.assert_close(x.grad, BWD_SCALE * ref_x.grad, **tols)
    for i in range(num_gemms):
        w = getattr(op, f"weight{i}")
        ref_w = getattr(reference, f"weight{i}")
        # The distributed op's grad of record is main_grad (finalize reduce-scattered the identity
        # wgrad there); compare it to the plain reference module's ordinary autograd .grad.
        torch.testing.assert_close(w.main_grad.to(dtype), ref_w.grad, **tols)
        # main_grad is flagged, and op's own .grad is a discarded dummy (not the real grad, which
        # lives in main_grad). Nobody writes the real grad into op.weight.grad by design.
        assert w.grad_added_to_main_grad is True
        assert w.grad is not None, f"weight{i} should receive a dummy .grad"
