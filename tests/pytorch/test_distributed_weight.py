# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tests for the GTP-agnostic DistributedWeight protocol, dispatchers, and module dispatch.

The first half exercises TE's weight-parallelism extension point in isolation, with
a tiny in-repo ``FakeDistributedWeight`` stub standing in for a real implementer
(e.g. Megatron's GTPShardedParam). No GPU, process group, or Megatron import is
required: the whole contract lives in TE and is verified against the fake.

The second half checks that ``Linear`` / ``LayerNormLinear`` / ``GroupedLinear`` keep the
distributed-weight path in backward when saved-tensor hooks are installed (directly, or via
non-reentrant activation checkpointing); see the notes above those tests.
"""

import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks
from torch.utils.checkpoint import checkpoint as torch_checkpoint

import transformer_engine.pytorch as te
from transformer_engine.pytorch.distributed_weight import (
    DistributedWeight,
    is_distributed_weight,
    materialize_weight_for_forward,
    materialize_weight_for_backward,
    finalize_weight_grads,
    weight_grad_dtype,
)
from transformer_engine.pytorch.module.grouped_linear import (
    is_module_grouped_tensor_path_supported,
)
from transformer_engine.pytorch.quantized_tensor import (
    prepare_for_saving,
    restore_from_func_ctx,
    restore_from_saved,
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


# ---------------------------------------------------------------------------------------------
# DistributedWeight dispatch under saved-tensor hooks.
#
# The modules pass the DistributedWeight parameter to ``prepare_for_saving`` and gate their
# backward on ``is_distributed_weight(saved_weight)``. Saving it through ``save_for_backward``
# only round-trips the subclass while no ``torch.autograd.graph.saved_tensors_hooks`` are
# installed: with hooks active, autograd unpacks a saved leaf as a *fresh plain tensor*
# (whatever the unpack hook returns), dropping the Python subclass and its
# ``is_distributed_weight`` marker. Backward then took the plain-parameter branch and failed
# (e.g. "weight was removed while fuse_wgrad_accumulation=True") or used the wrong weights.
#
# Activation offloading (TE's or Megatron-LM's fine-grained offloading) and non-reentrant
# activation checkpointing both install such hooks. ``prepare_for_saving`` therefore also keeps
# a DistributedWeight by reference and restores that object in place of the unpacked saved
# tensor (which is still saved, so autograd's in-place check is kept). These tests pin that:
# with hooks installed, the distributed path must still be taken and produce the same results
# as without hooks.
#
# TE ships no DistributedWeight implementer (they live in the caller, e.g. Megatron-LM's GTP), so
# an in-repo fake is used. It applies distinct scales in the forward / backward materialize hooks,
# so taking the wrong branch fails a specific assertion rather than drifting numerically.
# ---------------------------------------------------------------------------------------------

FWD_SCALE = 2.0
BWD_SCALE = 4.0
IN_F, OUT_F, TOKENS = 256, 512, 128
DTYPE, DEVICE = torch.bfloat16, "cuda"

FUSE_WGRAD = pytest.mark.parametrize("fuse_wgrad", [False, True], ids=["unfused", "fused-wgrad"])
HOOK_MODES = pytest.mark.parametrize("mode", ["saved-tensor-hooks", "checkpoint"])


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


def _run_hooked(mode, module, *args):
    """Run forward and backward with saved-tensor hooks active; return the output."""
    if mode == "saved-tensor-hooks":
        with _identity_hooks():
            out = module(*args)
    else:
        # Non-reentrant TE checkpointing discards saved tensors in forward and hands backward
        # the recomputed ones through its unpack hook.
        out = te.checkpoint(module, *args, use_reentrant=False)
    out.sum().backward()
    return out


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
    module, weight_names, leader, reference, ref_leader, out, ref_out, x, ref_x, fuse_wgrad, mode
):
    """Hooked run must have used every DistributedWeight hook and match the unhooked run."""
    assert leader.calls["fwd"] > 0, "materialize_group_for_forward never called"
    assert leader.calls["bwd"] > 0, (
        "materialize_group_for_backward never called: backward lost the DistributedWeight "
        "after saved_tensors_hooks unpacked the saved weight as a plain tensor"
    )
    assert leader.calls["finalize"] > 0, "finalize_group_grads never called"
    # Checkpointing runs the forward a second time to recompute the saved tensors.
    fwd_calls = ref_leader.calls["fwd"] * (2 if mode == "checkpoint" else 1)
    assert leader.calls == {**ref_leader.calls, "fwd": fwd_calls}
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


def test_prepare_for_saving_keeps_distributed_weight_by_reference():
    """A DistributedWeight is still saved, but restored as the original object."""
    w = FakeDistributedWeight()
    plain = torch.zeros(2)
    tensors, tensor_objects = prepare_for_saving(plain, w, None)
    assert tensors[0] is plain and tensors[1] is w and tensors[2] is None
    # Under saved-tensor hooks autograd hands back a plain tensor in the weight's slot.
    unpacked = [tensors[0], torch.Tensor(w), None]
    restored = restore_from_saved(tensor_objects, unpacked)
    assert restored[0] is plain and restored[1] is w and restored[2] is None


@pytest.mark.parametrize("mode", ["no-hooks", "saved-tensor-hooks", "checkpoint"])
def test_prepare_for_saving_round_trips_distributed_weight(mode):
    """Through autograd, backward gets the DistributedWeight object back under any hooks."""
    seen = {}

    class _SaveWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            tensors_to_save, tensor_objects = prepare_for_saving(x, w)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            return x * 2

        @staticmethod
        def backward(ctx, g):
            _, seen["w"] = restore_from_func_ctx(ctx)
            return g * 2, None

    w = _FakeDistWeight(torch.ones(4))
    x = torch.ones(4, requires_grad=True)
    if mode == "no-hooks":
        out = _SaveWeight.apply(x, w)
    elif mode == "saved-tensor-hooks":
        with _identity_hooks():
            out = _SaveWeight.apply(x, w)
    else:
        out = torch_checkpoint(_SaveWeight.apply, x, w, use_reentrant=False)
    out.sum().backward()
    assert seen["w"] is w


def test_prepare_for_saving_keeps_inplace_check_for_distributed_weight():
    """Restoring the original object must not bypass autograd's in-place modification check."""

    class _SaveWeight(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            tensors_to_save, tensor_objects = prepare_for_saving(x, w)
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            return x * 2

        @staticmethod
        def backward(ctx, g):
            restore_from_func_ctx(ctx)
            return g * 2, None

    w = _FakeDistWeight(torch.ones(4))
    x = torch.ones(4, requires_grad=True)
    out = _SaveWeight.apply(x, w)
    with torch.no_grad():
        w.add_(1)  # e.g. an optimizer step between forward and backward
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()


@HOOK_MODES
@FUSE_WGRAD
def test_linear_distributed_weight_under_saved_tensor_hooks(fuse_wgrad, mode):
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
    out = _run_hooked(mode, module, x)

    _check_dist_path_kept(
        module, ["weight"], leader, reference, ref_leader, out, ref_out, x, ref_x, fuse_wgrad, mode
    )
    # The scales prove which weights were used: FWD_SCALE in forward, BWD_SCALE in dgrad.
    plain = te.Linear(IN_F, OUT_F, bias=False, device=DEVICE, params_dtype=DTYPE)
    plain.load_state_dict({"weight": leader.data}, strict=False)
    px = x.detach().clone().requires_grad_(True)
    pout = plain(px)
    pout.sum().backward()
    torch.testing.assert_close(out.float(), FWD_SCALE * pout.float(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(x.grad.float(), BWD_SCALE * px.grad.float(), rtol=1e-5, atol=1e-5)


@HOOK_MODES
@FUSE_WGRAD
def test_layernorm_linear_distributed_weight_under_saved_tensor_hooks(fuse_wgrad, mode):
    """``LayerNormLinear`` must keep the distributed path in backward when hooks are on."""
    _skip_without_cuda()
    torch.manual_seed(0)
    module, reference = (
        te.LayerNormLinear(
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
    out = _run_hooked(mode, module, x)

    _check_dist_path_kept(
        module, ["weight"], leader, reference, ref_leader, out, ref_out, x, ref_x, fuse_wgrad, mode
    )
    torch.testing.assert_close(
        module.layer_norm_weight.grad, reference.layer_norm_weight.grad, rtol=0, atol=0
    )


@HOOK_MODES
@FUSE_WGRAD
@pytest.mark.parametrize("num_gemms", [2, 4])
@pytest.mark.parametrize(
    "use_grouped_tensor", [False, True], ids=["split-quantize", "grouped-tensor"]
)
def test_grouped_linear_distributed_weight_under_saved_tensor_hooks(
    num_gemms, fuse_wgrad, use_grouped_tensor, mode
):
    """``GroupedLinear`` (both GEMM paths) must keep the distributed path under hooks."""
    _skip_without_cuda()
    if use_grouped_tensor and not is_module_grouped_tensor_path_supported(None, DTYPE):
        # GroupedLinear would silently fall back to split-quantize, which the other case
        # already covers; skip rather than report coverage of the native backward.
        pytest.skip("native grouped-tensor path unsupported on this device / cuBLASLt")
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
            use_grouped_tensor=use_grouped_tensor,
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
    out = _run_hooked(mode, module, x, m_splits)

    _check_dist_path_kept(
        module, names, leader, reference, ref_leader, out, ref_out, x, ref_x, fuse_wgrad, mode
    )
