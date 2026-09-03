# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Single-process multi-device FP8 tests (issue #3124).

These tests deliberately never call ``torch.cuda.set_device`` and never
enable peer access: the ambient current device stays ``cuda:0`` while the
modules under test live on other devices, which is the situation produced
by ``accelerate.dispatch_model`` and plain ``device_map`` placement.
"""

import pytest
import torch

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >= 2 CUDA devices in one process"
)

DT = torch.bfloat16


@pytest.fixture(autouse=True)
def _fresh_fp8_state():
    FP8GlobalStateManager.reset()
    yield
    FP8GlobalStateManager.reset()


def _state_devices(module):
    out = {}
    for key in ("scaling_fwd", "scaling_bwd"):
        state = module.fp8_meta[key]
        out[key] = (state.scale.device, state.amax_history.device)
    return out


def _assert_global_buffer_invariants(modules):
    """Device consistency and object identity of the global amax buffers.

    For every registered position ``i``, the amax, amax history and scale
    entries must live on one device, the amax entry must remain the row-0
    view of that position's history tensor, and each module must still find
    its own scale/history objects at its registered position after the
    reduction/update pass.
    """
    qstate = FP8GlobalStateManager.quantization_state
    for key, amax_buffer in qstate.global_amax_buffer.items():
        histories = qstate.global_amax_history_buffer[key]
        scales = qstate.global_scale_buffer[key]
        assert len(amax_buffer) == len(histories) == len(scales)
        for i in range(len(amax_buffer)):
            assert amax_buffer[i].device == histories[i].device == scales[i].device
            assert amax_buffer[i].data_ptr() == histories[i][0].data_ptr()

    for module, expected_device in modules:
        fwd_pos, fwd_key, bwd_pos, bwd_key = module.fp8_meta[
            FP8GlobalStateManager.get_buffer_info()
        ]
        for pos, buffer_key, meta_key in (
            (fwd_pos, fwd_key, "scaling_fwd"),
            (bwd_pos, bwd_key, "scaling_bwd"),
        ):
            assert qstate.global_scale_buffer[buffer_key][pos] is module.fp8_meta[meta_key].scale
            assert (
                qstate.global_amax_history_buffer[buffer_key][pos]
                is module.fp8_meta[meta_key].amax_history
            )
            amax = qstate.global_amax_buffer[buffer_key][pos]
            assert amax.device == expected_device
            assert module.fp8_meta[meta_key].scale.device == expected_device
            assert module.fp8_meta[meta_key].amax_history.device == expected_device


def _assert_state_evolved(module):
    """The delayed-scaling update must have run, not been skipped.

    The fused update rolls the history so that the newest amax lands in the
    last row and row 0 is zeroed (the quantizer writes the *next* iteration's
    amax there), so "evolved" means: last row nonzero, scale off its init
    value of 1.
    """
    for key in ("scaling_fwd", "scaling_bwd"):
        state = module.fp8_meta[key]
        assert not torch.all(state.amax_history[-1] == 0), f"{key} amax never updated"
        assert not torch.all(state.scale == 1.0), f"{key} scale never updated"


def test_module_off_current_device():
    """One module on cuda:1 while the current device is cuda:0 (#3124 case A)."""
    torch.manual_seed(1234)
    module = te.Linear(512, 1024, bias=False, params_dtype=DT, device="cuda:1")
    inp = torch.randn(128, 512, device="cuda:1", dtype=DT, requires_grad=True)
    with te.autocast(enabled=True, recipe=DelayedScaling(fp8_format=Format.HYBRID)):
        out = module(inp)
    out.sum().backward()
    torch.cuda.synchronize()

    assert out.device == torch.device("cuda:1")
    assert inp.grad.device == torch.device("cuda:1")
    for key in ("scaling_fwd", "scaling_bwd"):
        scale_dev, hist_dev = _state_devices(module)[key]
        assert scale_dev == torch.device("cuda:1")
        assert hist_dev == torch.device("cuda:1")
    _assert_global_buffer_invariants([(module, torch.device("cuda:1"))])
    _assert_state_evolved(module)


def test_prepare_forward_exception_restores_current_device(monkeypatch):
    """A failed prepare must not leak the temporary CUDA device guard."""
    assert torch.cuda.current_device() == 0
    module = te.Linear(16, 16, bias=False, params_dtype=DT, device="cuda:1")
    inp = torch.randn(2, 16, device="cuda:1", dtype=DT)

    def fail_init(*args, **kwargs):
        raise RuntimeError("injected prepare_forward failure")

    monkeypatch.setattr(module, "init_fp8_metadata", fail_init)
    with pytest.raises(RuntimeError, match="injected prepare_forward failure"):
        module.prepare_forward(inp)

    assert torch.cuda.current_device() == 0
    assert module._forward_device_guards == []


def test_basic_operation_device_awareness():
    """The standalone ops API must use the op's device for state and execution."""
    recipe = DelayedScaling(fp8_format=Format.HYBRID)
    assert torch.cuda.current_device() == 0
    with te.quantized_model_init(enabled=True, recipe=recipe):
        op = te_ops.basic.BasicLinear(512, 1024, device="cuda:1", dtype=DT)
    inp = torch.randn(128, 512, device="cuda:1", dtype=DT, requires_grad=True)
    with te.autocast(enabled=True, recipe=DelayedScaling(fp8_format=Format.HYBRID)):
        out = op(inp)
    out.sum().backward()
    torch.cuda.synchronize()
    assert out.device == torch.device("cuda:1")
    assert inp.grad is not None and inp.grad.device == torch.device("cuda:1")
    assert torch.cuda.current_device() == 0
    for mode in ("forward", "backward"):
        state = op._fp8_metas[mode][FP8GlobalStateManager.get_meta_tensor_key(mode == "forward")]
        assert state.scale.device == torch.device("cuda:1")
        assert state.amax_history.device == torch.device("cuda:1")


def test_two_modules_on_different_devices_one_autocast():
    """Two modules on different devices inside one autocast (#3124 case C)."""
    torch.manual_seed(1234)
    a = te.Linear(512, 1024, bias=False, params_dtype=DT, device="cuda:0")
    torch.manual_seed(5678)
    b = te.Linear(1024, 512, bias=False, params_dtype=DT, device="cuda:1")

    for _ in range(3):
        inp = torch.randn(128, 512, device="cuda:0", dtype=DT, requires_grad=True)
        with te.autocast(enabled=True, recipe=DelayedScaling(fp8_format=Format.HYBRID)):
            hidden = a(inp)
            out = b(hidden.to("cuda:1"))
        out.sum().backward()
    torch.cuda.synchronize()

    assert out.device == torch.device("cuda:1")
    assert hidden.device == torch.device("cuda:0")
    _assert_global_buffer_invariants([(a, torch.device("cuda:0")), (b, torch.device("cuda:1"))])
    _assert_state_evolved(a)
    _assert_state_evolved(b)


def test_split_model_matches_single_device_bitwise():
    """Same weights and inputs: 2-GPU split must match the 1-GPU run exactly."""

    def build(device_pair):
        torch.manual_seed(1234)
        a = te.Linear(512, 1024, bias=False, params_dtype=DT, device=device_pair[0])
        torch.manual_seed(5678)
        b = te.Linear(1024, 512, bias=False, params_dtype=DT, device=device_pair[1])
        return a, b

    ref_a, ref_b = build(("cuda:0", "cuda:0"))
    spl_a, spl_b = build(("cuda:0", "cuda:1"))

    def run(a, b, split):
        outs = []
        for it in range(4):
            torch.manual_seed(100 + it)
            inp = torch.randn(128, 512, device="cuda:0", dtype=DT)
            with te.autocast(enabled=True, recipe=DelayedScaling(fp8_format=Format.HYBRID)):
                hidden = a(inp)
                out = b(hidden.to("cuda:1") if split else hidden)
            torch.cuda.synchronize()
            outs.append(out.detach().float().cpu())
        return outs

    ref_outs = run(ref_a, ref_b, split=False)
    spl_outs = run(spl_a, spl_b, split=True)
    for ref, spl in zip(ref_outs, spl_outs):
        torch.testing.assert_close(ref, spl, rtol=0, atol=0)
    torch.testing.assert_close(
        ref_b.fp8_meta["scaling_fwd"].scale.cpu(),
        spl_b.fp8_meta["scaling_fwd"].scale.cpu(),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        ref_b.fp8_meta["scaling_fwd"].amax_history.cpu(),
        spl_b.fp8_meta["scaling_fwd"].amax_history.cpu(),
        rtol=0,
        atol=0,
    )


def _run_three_module_chain(iters=4):
    """Chain a(cuda:0) -> b(cuda:1) -> c(cuda:0): registration order has an
    interleaved device layout, so the gather/scatter position bookkeeping
    handles non-contiguous per-device runs."""
    torch.manual_seed(1234)
    a = te.Linear(512, 1024, bias=False, params_dtype=DT, device="cuda:0")
    torch.manual_seed(2345)
    b = te.Linear(1024, 512, bias=False, params_dtype=DT, device="cuda:1")
    torch.manual_seed(3456)
    c = te.Linear(512, 256, bias=False, params_dtype=DT, device="cuda:0")

    outs = []
    for it in range(iters):
        torch.manual_seed(100 + it)
        inp = torch.randn(128, 512, device="cuda:0", dtype=DT, requires_grad=True)
        with te.autocast(enabled=True, recipe=DelayedScaling(fp8_format=Format.HYBRID)):
            h1 = a(inp)
            h2 = b(h1.to("cuda:1"))
            out = c(h2.to("cuda:0"))
        out.sum().backward()
        torch.cuda.synchronize()
        outs.append(out.detach().float().cpu())

    states = []
    for m in (a, b, c):
        for key in ("scaling_fwd", "scaling_bwd"):
            states.append(m.fp8_meta[key].scale.cpu().clone())
            states.append(m.fp8_meta[key].amax_history.cpu().clone())
    return outs, states


def test_multi_device_gather_reduce_matches_local(monkeypatch):
    """Multi-device + distributed-reduction path vs fully local path.

    A fake two-rank world routes the buffer through the gather -> collective
    -> scatter path. The collective is mocked as the identity (both "ranks"
    hold identical values), so results must be bit-identical to the no-dist
    local path, and the collective itself must keep its stock shape: one call
    per direction per iteration over the registration order, on the
    first-registered module's device.
    """
    fake_dist = {"on": True}
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: fake_dist["on"])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 2)
    monkeypatch.setattr(
        FP8GlobalStateManager,
        "reduce_tensor_across_group_op_max",
        staticmethod(
            lambda tensor, group: calls.append((tensor.device, tensor.numel(), tensor.dtype))
        ),
    )

    gather_outs, gather_states = _run_three_module_chain()
    fake_dist["on"] = False
    local_outs, local_states = _run_three_module_chain()

    for g, l in zip(gather_outs, local_outs):
        torch.testing.assert_close(g, l, rtol=0, atol=0)
    for g, l in zip(gather_states, local_states):
        torch.testing.assert_close(g, l, rtol=0, atol=0)

    # 3 fwd + 2 bwd amax values per module per iteration, one call each way.
    iters = 4
    assert len(calls) == 2 * iters
    fwd_numel = 3 * 3  # 3 modules x 3 forward fp8 tensors
    bwd_numel = 3 * 2  # 3 modules x 2 backward fp8 tensors
    for i, (dev, numel, dtype) in enumerate(calls):
        assert dev == torch.device("cuda:0")  # first-registered module's device
        assert dtype == torch.float32
        assert numel == (fwd_numel if i % 2 == 0 else bwd_numel)
