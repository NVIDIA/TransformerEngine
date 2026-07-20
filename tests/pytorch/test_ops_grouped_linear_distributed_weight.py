# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DistributedWeight dispatch in the fusible ``ops.GroupedLinear`` (unfused fallback path).

TE ships no DistributedWeight implementer (GTP etc. live in the caller), so this validates the
*dispatch wiring* with an in-repo fake. The fake applies a DISTINCT, observable scale in each hook
so every plumbing point is independently checked -- a wiring bug fails a specific assertion:

  * ``materialize_group_for_forward`` scales the weights by ``FWD_SCALE``  -> the fwd GEMM must use
    the materialized weights, so ``out == FWD_SCALE * plain_out``.
  * ``materialize_group_for_backward`` scales by ``BWD_SCALE``            -> dgrad must use the
    re-materialized weights, so ``dgrad == BWD_SCALE * plain_dgrad``.
  * ``finalize_group_grads`` scales the wgrads by ``FINALIZE_SCALE``      -> the returned param grad
    must be finalize's output, so ``weight.grad == FINALIZE_SCALE * plain_wgrad``.

Real all-gather / reduce-scatter math is exercised by the caller's distributed tests; here the fake
is single-process and only proves the ops.GroupedLinear integration routes weights/grads through
the hooks (and does not bypass them).
"""

import pytest
import torch

import transformer_engine.pytorch as te

FWD_SCALE = 2.0
BWD_SCALE = 3.0
FINALIZE_SCALE = 5.0


class _ScaledDistWeight(torch.nn.Parameter):
    """Non-identity DistributedWeight leader: each hook applies its own scale (see module docstring),
    coalescing its sibling group without any real gather."""

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
        return [g * FINALIZE_SCALE for g in wl]

    def grad_buffer(self):
        return self.data


def _make_scaled_dist_leader(op, num_gemms):
    """Replace ``op.weight0`` with a scaled distributed leader referencing the whole group."""
    w0 = op.weight0
    leader = _ScaledDistWeight(w0.data)
    leader.calls = {"fwd": 0, "bwd": 0, "finalize": 0}
    leader._group = [leader] + [getattr(op, f"weight{i}") for i in range(1, num_gemms)]
    op.weight0 = leader
    return leader


@pytest.mark.parametrize("num_gemms", [2, 4])
def test_ops_grouped_linear_distributed_weight_dispatch(num_gemms):
    """Each DistributedWeight hook's output must be routed through the GEMM flow (not bypassed)."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    torch.manual_seed(0)
    in_f, out_f, total_tokens = 32, 64, num_gemms * 8
    dtype, device = torch.bfloat16, "cuda"

    op = te.ops.GroupedLinear(num_gemms, in_f, out_f, bias=False, device=device, dtype=dtype)
    reference = te.ops.GroupedLinear(num_gemms, in_f, out_f, bias=False, device=device, dtype=dtype)
    reference.load_state_dict(op.state_dict())

    leader = _make_scaled_dist_leader(op, num_gemms)

    m_splits = [total_tokens // num_gemms] * num_gemms
    m_splits[-1] += total_tokens - sum(m_splits)
    split_sizes = torch.tensor(m_splits, dtype=torch.int64, device=device)

    x = torch.randn(total_tokens, in_f, dtype=dtype, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)

    out = op(x, split_sizes)
    out.sum().backward()
    ref_out = reference(ref_x, split_sizes)
    ref_out.sum().backward()

    # The dispatch hooks actually fired.
    assert leader.calls["fwd"] > 0 and leader.calls["bwd"] > 0 and leader.calls["finalize"] > 0

    tols = dict(rtol=2e-2, atol=2e-2)
    # fwd used the materialized (FWD_SCALE) weights.
    torch.testing.assert_close(out, FWD_SCALE * ref_out, **tols)
    # dgrad used the re-materialized (BWD_SCALE) weights.
    torch.testing.assert_close(x.grad, BWD_SCALE * ref_x.grad, **tols)
    # returned param grads are finalize's (FINALIZE_SCALE) output (wgrad GEMM itself is scale-free).
    for i in range(num_gemms):
        got = getattr(op, f"weight{i}").grad
        exp = getattr(reference, f"weight{i}").grad
        assert got is not None, f"weight{i} received no grad"
        torch.testing.assert_close(got, FINALIZE_SCALE * exp, **tols)
