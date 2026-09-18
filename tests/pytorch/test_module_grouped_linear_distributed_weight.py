# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DistributedWeight dispatch in ``module.GroupedLinear``, on both of its execution paths.

Companion to ``test_ops_grouped_linear_distributed_weight.py``, which covers the fusible
``ops.GroupedLinear``. The module has two internal paths, and they were *not* equivalent for a
distributed weight:

  * ``use_grouped_tensor=False`` -- the split-quantize fall-through, wired up in #3005.
  * ``use_grouped_tensor=True``  -- the native GroupedTensor grouped GEMM. ``forward``
    materializes the weight before dispatching here, so this path closed its
    wgrad-accumulation callables over the *gathered copies* and backward then read
    ``.main_grad`` off a plain tensor (AttributeError).

TE ships no DistributedWeight implementer -- they live in the caller (e.g. Megatron-LM's
generalized tensor parallelism) -- so this validates the *dispatch wiring* with an in-repo
fake that applies a distinct, observable scale in each materialize hook: a wiring bug fails
a specific assertion rather than drifting numerically.
"""

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import MXFP8BlockScaling
from transformer_engine.pytorch import is_mxfp8_available

# Powers of two, so the scales survive GEMM rounding and MXFP8 block scaling exactly.
FWD_SCALE = 2.0
BWD_SCALE = 4.0

# bf16 because the grouped GEMM rejects fp32; 128-token segments because MXFP8 requires it.
IN_F, OUT_F, TOKENS = 256, 512, 128
DTYPE, DEVICE = torch.bfloat16, "cuda"

PATHS = pytest.mark.parametrize(
    "use_grouped_tensor", [False, True], ids=["split-quantize", "grouped"]
)


class _FakeDistWeight(torch.nn.Parameter):
    """Fake distributed weight: weight0 leads, and holds the whole group in ``_group``."""

    is_distributed_weight = True
    # Whether the gathered copies carry requires_grad. The protocol does not require it, so
    # both settings are valid; see the test that pins the False case.
    propagate_requires_grad = True
    # Mirrors the module's fuse_wgrad_accumulation: it selects which buffer the wgrad GEMM
    # writes into (grad_buffer scratch vs a fresh allocation) and what finalize returns.
    fuse_wgrad_accumulation = True

    def materialize_group_for_forward(self):
        self.calls["fwd"] += 1
        out = [w.detach() * FWD_SCALE for w in self._group]
        return [w.requires_grad_(True) for w in out] if self.propagate_requires_grad else out

    def materialize_group_for_backward(self, **kwargs):
        self.calls["bwd"] += 1
        # High precision even under fp8, as a real implementer gathering bf16 shards would be.
        return [w.detach() * BWD_SCALE for w in self._group]

    def finalize_group_grads(self, wgrads, **kwargs):
        self.calls["finalize"] += 1
        wl = list(wgrads) if isinstance(wgrads, (list, tuple)) else [wgrads]
        if not self.fuse_wgrad_accumulation:
            # No main_grad to accumulate into: the reduce-scattered wgrad is the return value,
            # which autograd then puts on .grad.
            return [g.clone() for g in wl]
        for w, g in zip(self._group, wl):
            w.main_grad.add_(g.to(w.main_grad.dtype))  # stands in for the reduce-scatter
            w.grad_added_to_main_grad = True
        return [torch.zeros_like(g) for g in wl]  # real grad is in main_grad; these are discarded

    def grad_buffer(self):
        # Scratch for the wgrad GEMM. Must not alias main_grad, which finalize adds into after.
        return self.wgrad_scratch


def _build(num_gemms, use_grouped_tensor, fuse_wgrad=True):
    return te.GroupedLinear(
        num_gemms,
        IN_F,
        OUT_F,
        bias=False,
        device=DEVICE,
        params_dtype=DTYPE,
        fuse_wgrad_accumulation=fuse_wgrad,
        use_grouped_tensor=use_grouped_tensor,
    )


def _init_grad_state(module, num_gemms):
    """Give every weight the DDP-style grad state TE expects, fake or plain."""
    for i in range(num_gemms):
        w = getattr(module, f"weight{i}")
        w.main_grad = torch.zeros((OUT_F, IN_F), dtype=torch.float32, device=DEVICE)
        w.grad_added_to_main_grad = False
        if isinstance(w, _FakeDistWeight):
            w.wgrad_scratch = torch.zeros_like(w.main_grad)


def _make_fake_dist_leader(module, num_gemms, **attrs):
    """Replace every ``weightN`` with a fake distributed weight and return the leader.

    Every member is an implementer, not just the leader: this path reads ``grad_buffer``
    off each weight in the group. ``attrs`` override fake behaviour per test.
    """
    fakes = []
    for i in range(num_gemms):
        fake = _FakeDistWeight(getattr(module, f"weight{i}").data)
        fake.calls = {"fwd": 0, "bwd": 0, "finalize": 0}
        for name, value in attrs.items():
            setattr(fake, name, value)
        setattr(module, f"weight{i}", fake)
        fakes.append(fake)
    for fake in fakes:
        fake._group = fakes
    _init_grad_state(module, num_gemms)
    return fakes[0]


def _inputs(num_gemms):
    m_splits = torch.full((num_gemms,), TOKENS, dtype=torch.int64, device=DEVICE)
    x = torch.randn(num_gemms * TOKENS, IN_F, dtype=DTYPE, device=DEVICE, requires_grad=True)
    return m_splits, x


@PATHS
@pytest.mark.parametrize("quantized", [False, True], ids=["bf16", "mxfp8"])
@pytest.mark.parametrize("num_gemms", [2, 4])
def test_module_grouped_linear_distributed_weight_dispatch(
    num_gemms, quantized, use_grouped_tensor
):
    """Both module paths must route a distributed weight through every DistributedWeight hook."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if quantized:
        available, reason = is_mxfp8_available(return_reason=True)
        if not available:
            pytest.skip(f"MXFP8 unavailable: {reason}")

    torch.manual_seed(0)
    module, reference = (_build(num_gemms, use_grouped_tensor) for _ in range(2))
    reference.load_state_dict(module.state_dict())
    leader = _make_fake_dist_leader(module, num_gemms)
    _init_grad_state(reference, num_gemms)

    m_splits, x = _inputs(num_gemms)
    ref_x = x.detach().clone().requires_grad_(True)

    def run(m, inp):
        if quantized:
            with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                out = m(inp, m_splits)
        else:
            out = m(inp, m_splits)
        out.sum().backward()
        return out

    out, ref_out = run(module, x), run(reference, ref_x)

    # Every hook fired, so the path did not bypass the protocol.
    assert leader.calls["fwd"] > 0, "materialize_group_for_forward never called"
    assert leader.calls["bwd"] > 0, "materialize_group_for_backward never called"
    assert leader.calls["finalize"] > 0, "finalize_group_grads never called -- wgrad path skipped"

    tols = dict(rtol=2e-2, atol=2e-2) if quantized else dict(rtol=1e-5, atol=1e-5)
    # The scales prove which weights were used: FWD_SCALE in forward, BWD_SCALE in dgrad.
    torch.testing.assert_close(out.float(), FWD_SCALE * ref_out.float(), **tols)
    torch.testing.assert_close(x.grad.float(), BWD_SCALE * ref_x.grad.float(), **tols)
    for i in range(num_gemms):
        w, ref_w = getattr(module, f"weight{i}"), getattr(reference, f"weight{i}")
        # Both modules use fuse_wgrad_accumulation, so main_grad is the grad of record.
        torch.testing.assert_close(w.main_grad, ref_w.main_grad, **tols)
        assert w.grad_added_to_main_grad is True


@PATHS
def test_wgrad_runs_when_materialized_copies_drop_requires_grad(use_grouped_tensor):
    """Trainability must be read from the parameters, not from the materialized copies.

    The protocol lets an implementer return plain gathered buffers, which carry
    requires_grad=False. Reading trainability off those skips the whole wgrad path silently.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(0)
    num_gemms = 2
    module = _build(num_gemms, use_grouped_tensor)
    leader = _make_fake_dist_leader(module, num_gemms, propagate_requires_grad=False)

    m_splits, x = _inputs(num_gemms)
    module(x, m_splits).sum().backward()

    assert leader.calls["finalize"] > 0, "wgrad path was skipped"
    for i in range(num_gemms):
        w = getattr(module, f"weight{i}")
        assert w.grad_added_to_main_grad is True
        assert torch.count_nonzero(w.main_grad) > 0, "main_grad was never written"


@PATHS
def test_unfused_wgrad_returns_grads_through_finalize(use_grouped_tensor):
    """Without fuse_wgrad_accumulation the wgrad GEMM writes a fresh buffer, not grad_buffer(),
    and what finalize_group_grads returns becomes .grad. main_grad must stay untouched."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    torch.manual_seed(0)
    num_gemms = 2
    module, reference = (_build(num_gemms, use_grouped_tensor, fuse_wgrad=False) for _ in range(2))
    reference.load_state_dict(module.state_dict())
    leader = _make_fake_dist_leader(module, num_gemms, fuse_wgrad_accumulation=False)

    m_splits, x = _inputs(num_gemms)
    ref_x = x.detach().clone().requires_grad_(True)
    module(x, m_splits).sum().backward()
    reference(ref_x, m_splits).sum().backward()

    assert leader.calls["finalize"] > 0, "finalize_group_grads never called"
    for i in range(num_gemms):
        w, ref_w = getattr(module, f"weight{i}"), getattr(reference, f"weight{i}")
        assert w.grad is not None, "finalize's return value never reached .grad"
        assert torch.count_nonzero(w.grad) > 0, "grad is all zeros -- the dummy, not the wgrad"
        # wgrad is x summed per group, independent of the weight values, so it matches exactly.
        torch.testing.assert_close(w.grad.float(), ref_w.grad.float(), rtol=1e-5, atol=1e-5)
        assert torch.count_nonzero(w.main_grad) == 0, "main_grad was written without fusion"


def test_single_grouped_weight_dispatches(monkeypatch):
    """A distributed weight must also work when the group is one packed GroupedTensor.

    single_grouped_weight requires use_grouped_tensor, and makes the group a single ``weight``
    instead of weight0..N -- so finalize_group_grads receives a bare tensor, not a list.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    torch.manual_seed(0)
    num_gemms = 2
    module = te.GroupedLinear(
        num_gemms,
        IN_F,
        OUT_F,
        bias=False,
        device=DEVICE,
        params_dtype=DTYPE,
        fuse_wgrad_accumulation=True,
        use_grouped_tensor=True,
        single_grouped_weight=True,
    )
    weight = getattr(module, "weight", None)
    assert weight is not None, "single_grouped_weight did not take effect"

    # Duck-type the protocol onto the GroupedTensor: this path needs the GEMM operand to stay
    # a GroupedTensor, so the materialize hooks hand back the same object.
    calls = []
    # grad_buffer is shaped like the unsharded weight, not flattened.
    weight.wgrad_scratch = torch.zeros((num_gemms, OUT_F, IN_F), dtype=torch.float32, device=DEVICE)
    weight.is_distributed_weight = True
    weight.grad_buffer = lambda: weight.wgrad_scratch
    weight.materialize_group_for_forward = lambda: (calls.append("fwd"), [weight])[1]
    weight.materialize_group_for_backward = lambda **kw: (calls.append("bwd"), [weight])[1]

    def _finalize(wgrads, **kwargs):
        assert not isinstance(wgrads, (list, tuple)), "a one-member group should pass a bare tensor"
        calls.append("finalize")
        return [torch.zeros_like(wgrads)]

    weight.finalize_group_grads = _finalize

    m_splits, x = _inputs(num_gemms)
    module(x, m_splits).sum().backward()

    assert calls == ["fwd", "bwd", "finalize"], f"unexpected hook dispatch: {calls}"
    assert torch.count_nonzero(weight.wgrad_scratch) > 0, "wgrad never reached grad_buffer"
