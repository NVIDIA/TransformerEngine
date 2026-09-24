# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DistributedWeight dispatch in ``Linear`` / ``GroupedLinear`` under saved-tensor hooks.

Both modules pass the DistributedWeight parameter itself through ``save_for_backward`` and
used to gate their backward on ``is_distributed_weight(saved_weight)``. That only holds while
no ``torch.autograd.graph.saved_tensors_hooks`` are installed: with hooks active, autograd
unpacks a saved leaf as a *fresh plain tensor* (whatever the unpack hook returns), dropping the
Python subclass and its ``is_distributed_weight`` marker. Backward then took the plain-parameter
branch, whose weakrefs point at the transient all-gathered weights, and failed with
"weight was removed while fuse_wgrad_accumulation=True" (or silently used the wrong weights).

Megatron-LM's fine-grained activation offloading installs exactly such hooks around expert
layers, which is how this surfaced. The fix keeps the DistributedWeight objects on the autograd
context and prefers them in backward. These tests pin that: with identity hooks installed, the
distributed path must still be taken and produce the same results as without hooks.

TE ships no DistributedWeight implementer (they live in the caller, e.g. Megatron-LM's GTP), so
an in-repo fake is used. It applies distinct scales in the forward / backward materialize hooks,
so taking the wrong branch fails a specific assertion rather than drifting numerically.
"""

import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks

import transformer_engine.pytorch as te

FWD_SCALE = 2.0
BWD_SCALE = 4.0
IN_F, OUT_F, TOKENS = 256, 512, 128
DTYPE, DEVICE = torch.bfloat16, "cuda"

FUSE_WGRAD = pytest.mark.parametrize("fuse_wgrad", [False, True], ids=["unfused", "fused-wgrad"])


class _FakeDistWeight(torch.nn.Parameter):
    """Fake distributed weight; the group leader holds every member in ``_group``."""

    is_distributed_weight = True

    def materialize_group_for_forward(self):
        self.calls["fwd"] += 1
        return [(w.detach() * FWD_SCALE).requires_grad_(True) for w in self._group]

    def materialize_group_for_backward(self, **kwargs):
        self.calls["bwd"] += 1
        return [w.detach() * BWD_SCALE for w in self._group]

    def finalize_group_grads(self, wgrads, **kwargs):
        self.calls["finalize"] += 1
        wl = list(wgrads) if isinstance(wgrads, (list, tuple)) else [wgrads]
        if not self.fuse_wgrad_accumulation:
            return [g.clone() for g in wl]
        for w, g in zip(self._group, wl):
            w.main_grad.add_(g.to(w.main_grad.dtype))  # stands in for the reduce-scatter
            w.grad_added_to_main_grad = True
        return [torch.zeros_like(g) for g in wl]  # real grad is in main_grad

    def grad_buffer(self):
        return self.wgrad_scratch


def _identity_hooks():
    """Hooks that change nothing; their mere presence makes autograd re-wrap saved leaves."""
    return saved_tensors_hooks(lambda t: t, lambda t: t)


def _install_fakes(module, weight_names, fuse_wgrad):
    """Replace the module's weights with fakes sharing one group; return the leader."""
    fakes = []
    for name in weight_names:
        fake = _FakeDistWeight(getattr(module, name).data)
        fake.calls = {"fwd": 0, "bwd": 0, "finalize": 0}
        fake.fuse_wgrad_accumulation = fuse_wgrad
        fake.main_grad = torch.zeros((OUT_F, IN_F), dtype=torch.float32, device=DEVICE)
        fake.wgrad_scratch = torch.zeros_like(fake.main_grad)
        fake.grad_added_to_main_grad = False
        setattr(module, name, fake)
        fakes.append(fake)
    for fake in fakes:
        fake._group = fakes
    return fakes[0]


def _check_dist_path_kept(
    module, weight_names, leader, reference, ref_leader, out, ref_out, x, ref_x, fuse_wgrad
):
    """Hooked run must have used every DistributedWeight hook and match the unhooked run."""
    assert leader.calls["fwd"] > 0, "materialize_group_for_forward never called"
    assert leader.calls["bwd"] > 0, (
        "materialize_group_for_backward never called: backward lost the DistributedWeight "
        "after saved_tensors_hooks unpacked the saved weight as a plain tensor"
    )
    assert leader.calls["finalize"] > 0, "finalize_group_grads never called"
    assert leader.calls == ref_leader.calls
    # Same kernels on the same data: results are bitwise equal to the unhooked run.
    torch.testing.assert_close(out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
    for name in weight_names:
        w, ref_w = getattr(module, name), getattr(reference, name)
        if fuse_wgrad:
            assert w.grad_added_to_main_grad is True
            torch.testing.assert_close(w.main_grad, ref_w.main_grad, rtol=0, atol=0)
            assert torch.count_nonzero(w.main_grad) > 0
        else:
            assert w.grad is not None and ref_w.grad is not None
            torch.testing.assert_close(w.grad, ref_w.grad, rtol=0, atol=0)


def _skip_without_cuda():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")


def test_saved_tensor_hooks_unpack_subclass_as_plain_tensor():
    """Premise: with hooks installed, a saved DistributedWeight comes back as a plain tensor."""
    _skip_without_cuda()
    seen = {}

    class _Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            ctx.save_for_backward(w)
            return x * 2

        @staticmethod
        def backward(ctx, g):
            (w,) = ctx.saved_tensors
            seen["type"] = type(w)
            seen["marked"] = bool(getattr(w, "is_distributed_weight", False))
            return g * 2, None

    w = _FakeDistWeight(torch.ones(4, device=DEVICE))
    x = torch.ones(4, device=DEVICE, requires_grad=True)

    _Probe.apply(x, w).sum().backward()
    assert seen["type"] is _FakeDistWeight and seen["marked"]

    with _identity_hooks():
        _Probe.apply(x, w).sum().backward()
    assert seen["type"] is torch.Tensor and not seen["marked"]


@FUSE_WGRAD
def test_linear_distributed_weight_under_saved_tensor_hooks(fuse_wgrad):
    """``Linear`` must keep the distributed path in backward when saved-tensor hooks are on."""
    _skip_without_cuda()
    torch.manual_seed(0)
    module, reference = (
        te.Linear(
            IN_F,
            OUT_F,
            bias=False,
            device=DEVICE,
            params_dtype=DTYPE,
            fuse_wgrad_accumulation=fuse_wgrad,
        )
        for _ in range(2)
    )
    reference.load_state_dict(module.state_dict())
    leader = _install_fakes(module, ["weight"], fuse_wgrad)
    ref_leader = _install_fakes(reference, ["weight"], fuse_wgrad)

    x = torch.randn(TOKENS, IN_F, dtype=DTYPE, device=DEVICE, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)

    ref_out = reference(ref_x)
    ref_out.sum().backward()
    with _identity_hooks():
        out = module(x)
        out.sum().backward()

    _check_dist_path_kept(
        module, ["weight"], leader, reference, ref_leader, out, ref_out, x, ref_x, fuse_wgrad
    )
    # The scales prove which weights were used: FWD_SCALE in forward, BWD_SCALE in dgrad.
    plain = te.Linear(IN_F, OUT_F, bias=False, device=DEVICE, params_dtype=DTYPE)
    plain.load_state_dict({"weight": leader.data}, strict=False)
    px = x.detach().clone().requires_grad_(True)
    pout = plain(px)
    pout.sum().backward()
    torch.testing.assert_close(out.float(), FWD_SCALE * pout.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad.float(), BWD_SCALE * px.grad.float(), rtol=1e-5, atol=1e-5)


@FUSE_WGRAD
@pytest.mark.parametrize("num_gemms", [2, 4])
def test_grouped_linear_distributed_weight_under_saved_tensor_hooks(num_gemms, fuse_wgrad):
    """``GroupedLinear`` (split-quantize path) must keep the distributed path under hooks."""
    _skip_without_cuda()
    torch.manual_seed(0)
    names = [f"weight{i}" for i in range(num_gemms)]
    module, reference = (
        te.GroupedLinear(
            num_gemms,
            IN_F,
            OUT_F,
            bias=False,
            device=DEVICE,
            params_dtype=DTYPE,
            fuse_wgrad_accumulation=fuse_wgrad,
            use_grouped_tensor=False,
        )
        for _ in range(2)
    )
    reference.load_state_dict(module.state_dict())
    leader = _install_fakes(module, names, fuse_wgrad)
    ref_leader = _install_fakes(reference, names, fuse_wgrad)

    m_splits = torch.full((num_gemms,), TOKENS, dtype=torch.int64, device=DEVICE)
    x = torch.randn(num_gemms * TOKENS, IN_F, dtype=DTYPE, device=DEVICE, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)

    ref_out = reference(ref_x, m_splits)
    ref_out.sum().backward()
    with _identity_hooks():
        out = module(x, m_splits)
        out.sum().backward()

    _check_dist_path_kept(
        module, names, leader, reference, ref_leader, out, ref_out, x, ref_x, fuse_wgrad
    )
