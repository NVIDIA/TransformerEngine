# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Single-process multi-device FP8 tests (issue #3124).

These tests never call ``torch.cuda.set_device`` and never enable peer access
before the first forward: the current device stays ``cuda:0`` while the modules
under test live on other devices, as with ``accelerate.dispatch_model`` and plain
``device_map`` placement. Placement is asserted explicitly because, once peer
access is enabled, the unfixed code silently runs on the wrong device instead of
raising.
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
RECIPE = DelayedScaling(fp8_format=Format.HYBRID)


@pytest.fixture(autouse=True)
def _fresh_fp8_state():
    FP8GlobalStateManager.reset()
    yield
    FP8GlobalStateManager.reset()


def _assert_state_on(module, device):
    """Recipe state lives on ``device`` and the delayed-scaling update ran there.

    The fused update rolls the history so the newest amax lands in the last row
    (row 0 is zeroed for the next iteration) and moves the scale off its init
    value of 1.
    """
    for key in ("scaling_fwd", "scaling_bwd"):
        state = module.fp8_meta[key]
        assert state.scale.device == device
        assert state.amax_history.device == device
        assert not torch.all(state.amax_history[-1] == 0), f"{key} amax never updated"
        assert not torch.all(state.scale == 1.0), f"{key} scale never updated"


@pytest.mark.parametrize("quantized_weight", (False, True))
def test_module_off_current_device(quantized_weight):
    """One module on cuda:1 while the current device is cuda:0 (#3124 case A)."""
    assert torch.cuda.current_device() == 0
    with te.quantized_model_init(enabled=quantized_weight, recipe=RECIPE):
        module = te.Linear(512, 1024, bias=False, params_dtype=DT, device="cuda:1")
    inp = torch.randn(128, 512, device="cuda:1", dtype=DT, requires_grad=True)
    with te.autocast(enabled=True, recipe=RECIPE):
        out = module(inp)
    out.sum().backward()
    torch.cuda.synchronize()

    assert out.device == torch.device("cuda:1")
    assert inp.grad.device == torch.device("cuda:1")
    assert torch.cuda.current_device() == 0
    _assert_state_on(module, torch.device("cuda:1"))


def test_prepare_forward_exception_restores_current_device(monkeypatch):
    """A failed prepare must not leak the temporary CUDA device guard."""
    module = te.Linear(16, 16, bias=False, params_dtype=DT, device="cuda:1")
    inp = torch.randn(2, 16, device="cuda:1", dtype=DT)

    def fail_init(*args, **kwargs):
        raise RuntimeError("injected prepare_forward failure")

    monkeypatch.setattr(module, "init_fp8_metadata", fail_init)
    with pytest.raises(RuntimeError, match="injected prepare_forward failure"):
        module.prepare_forward(inp)
    assert torch.cuda.current_device() == 0


def test_basic_operation_off_current_device():
    """The fusible-ops API must allocate state and execute on the op's device."""
    with te.quantized_model_init(enabled=True, recipe=RECIPE):
        op = te_ops.basic.BasicLinear(512, 1024, device="cuda:1", dtype=DT)
    inp = torch.randn(128, 512, device="cuda:1", dtype=DT, requires_grad=True)
    with te.autocast(enabled=True, recipe=RECIPE):
        out = op(inp)
    out.sum().backward()
    torch.cuda.synchronize()

    assert out.device == torch.device("cuda:1")
    assert inp.grad.device == torch.device("cuda:1")
    assert torch.cuda.current_device() == 0
    for mode in ("forward", "backward"):
        state = op._fp8_metas[mode][FP8GlobalStateManager.get_meta_tensor_key(mode == "forward")]
        assert state.scale.device == torch.device("cuda:1")
        assert state.amax_history.device == torch.device("cuda:1")


def test_two_modules_on_different_devices_one_autocast():
    """Two modules on different devices inside one autocast (#3124 case C)."""
    a = te.Linear(512, 1024, bias=False, params_dtype=DT, device="cuda:0")
    b = te.Linear(1024, 512, bias=False, params_dtype=DT, device="cuda:1")
    for _ in range(3):
        inp = torch.randn(128, 512, device="cuda:0", dtype=DT, requires_grad=True)
        with te.autocast(enabled=True, recipe=RECIPE):
            hidden = a(inp)
            out = b(hidden.to("cuda:1"))
        out.sum().backward()
    torch.cuda.synchronize()

    assert hidden.device == torch.device("cuda:0")
    assert out.device == torch.device("cuda:1")
    _assert_state_on(a, torch.device("cuda:0"))
    _assert_state_on(b, torch.device("cuda:1"))


def _run_chain(devices, iters=4):
    """Run a chain of Linears placed on ``devices``; return outputs and FP8 state."""
    modules = []
    for i, (device, (in_f, out_f)) in enumerate(
        zip(devices, ((512, 1024), (1024, 512), (512, 256)))
    ):
        torch.manual_seed(1234 + i)
        modules.append(te.Linear(in_f, out_f, bias=False, params_dtype=DT, device=device))

    outs, states = [], []
    for it in range(iters):
        torch.manual_seed(100 + it)
        x = torch.randn(128, 512, device="cuda:0", dtype=DT, requires_grad=True)
        with te.autocast(enabled=True, recipe=RECIPE):
            for module, device in zip(modules, devices):
                x = module(x.to(device))
        x.sum().backward()
        torch.cuda.synchronize()
        outs.append(x.detach().float().cpu())
    for module in modules:
        for key in ("scaling_fwd", "scaling_bwd"):
            states.append(module.fp8_meta[key].scale.cpu())
            states.append(module.fp8_meta[key].amax_history.cpu())
    return outs, states


def test_split_model_matches_single_device_bitwise():
    """Same weights and inputs: the 2-GPU split must match the 1-GPU run exactly."""
    ref = _run_chain(("cuda:0", "cuda:0", "cuda:0"))
    split = _run_chain(("cuda:0", "cuda:1", "cuda:0"))
    for r, s in zip(ref[0] + ref[1], split[0] + split[1]):
        torch.testing.assert_close(r, s, rtol=0, atol=0)


def test_multi_device_amax_reduction(monkeypatch):
    """Multi-device buffer with a distributed amax reduction.

    A fake two-rank world routes the interleaved (cuda:0, cuda:1, cuda:0) buffer
    through the gather -> collective -> scatter path. The collective must see the
    whole buffer in registration order on the first-registered device, and with the
    collective mocked as the identity the results must be bit-identical to the
    fully local path.
    """
    fake_dist = {"on": True}
    calls = []
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: fake_dist["on"])
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda *a, **k: 2)

    def fake_reduce(tensor, _group):
        qstate = FP8GlobalStateManager.quantization_state
        entries = [
            entries
            for entries in qstate.global_amax_buffer.values()
            if sum(entry.numel() for entry in entries) == tensor.numel()
        ]
        assert len(entries) == 1
        expected = torch.cat([entry.to(tensor.device) for entry in entries[0]])
        torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
        calls.append(tensor.device)

    monkeypatch.setattr(
        FP8GlobalStateManager, "reduce_tensor_across_group_op_max", staticmethod(fake_reduce)
    )
    devices = ("cuda:0", "cuda:1", "cuda:0")
    gathered = _run_chain(devices)
    fake_dist["on"] = False
    local = _run_chain(devices)

    for g, l in zip(gathered[0] + gathered[1], local[0] + local[1]):
        torch.testing.assert_close(g, l, rtol=0, atol=0)
    # One collective per direction per iteration, on the first-registered device.
    assert calls == [torch.device("cuda:0")] * 8
