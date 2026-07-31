#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FSDP2 coverage for MXFP4-QAT fusible operations.

Run with:
  torchrun --nproc_per_node=2 -m pytest <this_file> -v -s --tb=short

Covers only configurations that pure MXFP8 already supports:

* BasicLinear/Linear: all backward override modes.
* GroupedLinear, independent high-precision weights: ``None``.
* Fused GroupedMLP, independent high-precision weights: ``None``, when the CuTe
  DSL fusion is available.
"""

from collections.abc import Callable, Sequence

import pytest
import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed._composable.fsdp import fully_shard

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import (
    MXFP4QATMXFP8BlockScaling,
    MXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8
from transformer_engine.pytorch.mxfp4_qat import mxfp4_fake_quantize

_DTYPE = torch.bfloat16


def _skip_if_unsupported() -> None:
    supported, reason = fp8.check_mxfp8_support()
    if not supported:
        pytest.skip(reason)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


def _named_weights(module: torch.nn.Module) -> list[tuple[str, torch.nn.Parameter]]:
    weights = [
        (name, param)
        for name, param in module.named_parameters()
        if name.rsplit(".", 1)[-1].startswith("weight")
    ]
    assert weights, f"{type(module).__name__} has no weight parameters"
    return weights


def _make_master_weight(
    out_features: int,
    in_features: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct BF16 values that MXFP8 preserves but MXFP4 changes."""
    pattern = torch.tensor(
        [
            1.0,
            0.3125,
            -0.3125,
            0.6875,
            -0.6875,
            0.15625,
            -0.15625,
            0.8125,
        ],
        dtype=_DTYPE,
        device=device,
    )
    repeats = (in_features + pattern.numel() - 1) // pattern.numel()
    row = pattern.repeat(repeats)[:in_features]
    return row.unsqueeze(0).expand(out_features, -1).contiguous()


def _initialize_master_weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    master_weights = {}
    with torch.no_grad():
        for name, param in _named_weights(module):
            weight = _make_master_weight(
                param.shape[0],
                param.shape[1],
                device=param.device,
            )
            param.copy_(weight)
            master_weights[name] = weight
    return master_weights


def _project_weights(
    master_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    projected_weights = {
        name: mxfp4_fake_quantize(weight).detach() for name, weight in master_weights.items()
    }
    for name in master_weights:
        assert not torch.equal(
            master_weights[name], projected_weights[name]
        ), f"Test weight {name} unexpectedly lies entirely on the MXFP4 grid"
    return projected_weights


@torch.no_grad()
def _load_weights(module: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    named_weights = _named_weights(module)
    assert [name for name, _ in named_weights] == list(weights)
    for name, param in named_weights:
        param.copy_(weights[name])


def _fsdp2_shard(module: torch.nn.Module) -> torch.nn.Module:
    world_size = torch.distributed.get_world_size()
    mesh = DeviceMesh("cuda", list(range(world_size)))
    fully_shard(module, mesh=mesh)
    return module


def _run_reference(
    factory: Callable[[], torch.nn.Module],
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    dy: torch.Tensor,
    extra_args: Sequence[torch.Tensor],
    *,
    override: str | None,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    module = factory()
    _load_weights(module, weights)
    x_ref = x.detach().clone().requires_grad_(True)
    recipe = MXFP8BlockScaling(backward_override=override)
    with te.autocast(enabled=True, recipe=recipe):
        y_ref = module(x_ref, *extra_args)
    y_ref.backward(dy)
    assert x_ref.grad is not None
    return module, y_ref.detach(), x_ref.grad.detach()


def _assert_projected_forward(
    y_qat: torch.Tensor,
    y_projected: torch.Tensor,
    y_master: torch.Tensor,
) -> None:
    torch.testing.assert_close(y_qat, y_projected, rtol=0, atol=0)
    assert not torch.equal(
        y_qat, y_master
    ), "QAT forward matched the unprojected master-weight path"


def _assert_fsdp_weight_grads(
    fsdp_module: torch.nn.Module,
    reference_module: torch.nn.Module,
    *,
    rtol: float = 0,
    atol: float = 0,
) -> None:
    fsdp_weights = _named_weights(fsdp_module)
    reference_weights = _named_weights(reference_module)
    assert [name for name, _ in fsdp_weights] == [name for name, _ in reference_weights]
    for (name, fsdp_weight), (_, reference_weight) in zip(fsdp_weights, reference_weights):
        assert isinstance(fsdp_weight, DTensor), f"{name} was not FSDP2-sharded"
        assert fsdp_weight.grad is not None, f"STE did not return a gradient for {name}"
        assert isinstance(fsdp_weight.grad, DTensor), f"{name}.grad is not sharded"
        assert reference_weight.grad is not None
        grad = fsdp_weight.grad.full_tensor()
        assert torch.isfinite(grad).all(), f"{name}.grad contains non-finite values"
        torch.testing.assert_close(
            grad,
            reference_weight.grad,
            rtol=rtol,
            atol=atol,
            msg=lambda msg: (f"{name}: FSDP2 QAT wgrad did not reach the master shard\n{msg}"),
        )


@pytest.mark.parametrize("op_kind", ("BasicLinear", "Linear"))
@pytest.mark.parametrize(
    "override",
    (
        pytest.param(None, id="none"),
        pytest.param("dequantized", id="dequantized"),
        pytest.param("high_precision", id="high_precision"),
    ),
)
def test_fsdp2_linear_qat_backward_overrides(op_kind: str, override: str | None) -> None:
    """BasicLinear/Linear preserve all pure-MXFP8 override modes under FSDP2."""
    _skip_if_unsupported()
    device = _device()
    in_features = out_features = 128

    def factory() -> torch.nn.Module:
        op_cls = getattr(te_ops, op_kind)
        kwargs = {"device": device, "dtype": _DTYPE}
        if op_kind == "Linear":
            kwargs["bias"] = False
        return op_cls(in_features, out_features, **kwargs)

    qat_module = factory()
    master_weights = _initialize_master_weights(qat_module)
    projected_weights = _project_weights(master_weights)
    _fsdp2_shard(qat_module)

    # Identity inputs make dgrad and wgrad comparisons strict and expose
    # per-element differences between master and projected weights.
    x = torch.eye(in_features, dtype=_DTYPE, device=device).requires_grad_(True)
    dy = torch.eye(out_features, dtype=_DTYPE, device=device)
    qat_recipe = MXFP4QATMXFP8BlockScaling(backward_override=override)
    with te.autocast(enabled=True, recipe=qat_recipe):
        y_qat = qat_module(x)
    y_qat.backward(dy)
    assert x.grad is not None

    projected_ref, y_projected, dx_projected = _run_reference(
        factory,
        projected_weights,
        x,
        dy,
        (),
        override=override,
    )
    master_ref, y_master, dx_master = _run_reference(
        factory,
        master_weights,
        x,
        dy,
        (),
        override=override,
    )

    _assert_projected_forward(y_qat.detach(), y_projected, y_master)
    expected_dx = dx_master if override == "high_precision" else dx_projected
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)
    expected_wgrad_ref = master_ref if override == "high_precision" else projected_ref
    _assert_fsdp_weight_grads(qat_module, expected_wgrad_ref)


def test_fsdp2_grouped_linear_qat_none() -> None:
    """Independent GroupedLinear weights support QAT mode None under FSDP2."""
    _skip_if_unsupported()
    device = _device()
    group_size = 2
    in_features = out_features = 128

    def factory() -> torch.nn.Module:
        return te_ops.GroupedLinear(
            group_size,
            in_features,
            out_features,
            bias=False,
            device=device,
            dtype=_DTYPE,
            single_grouped_weight=False,
        )

    qat_module = factory()
    master_weights = _initialize_master_weights(qat_module)
    projected_weights = _project_weights(master_weights)
    _fsdp2_shard(qat_module)

    split_sizes = torch.tensor([128, 128], dtype=torch.int32, device=device)
    eye = torch.eye(in_features, dtype=_DTYPE, device=device)
    x = torch.cat((eye, eye)).requires_grad_(True)
    dy = torch.cat((eye, eye))
    qat_recipe = MXFP4QATMXFP8BlockScaling(backward_override=None)
    with te.autocast(enabled=True, recipe=qat_recipe):
        y_qat = qat_module(x, split_sizes)
    y_qat.backward(dy)
    assert x.grad is not None

    projected_ref, y_projected, dx_projected = _run_reference(
        factory,
        projected_weights,
        x,
        dy,
        (split_sizes,),
        override=None,
    )
    _, y_master, _ = _run_reference(
        factory,
        master_weights,
        x,
        dy,
        (split_sizes,),
        override=None,
    )

    _assert_projected_forward(y_qat.detach(), y_projected, y_master)
    torch.testing.assert_close(x.grad, dx_projected, rtol=0, atol=0)
    _assert_fsdp_weight_grads(qat_module, projected_ref)


def test_fsdp2_fused_grouped_mlp_qat_none() -> None:
    """Fused GroupedMLP mode None preserves QAT weights through FSDP2 backward."""
    _skip_if_unsupported()
    fused_cls = te_ops.fused.GroupedMLP_CuTeGEMMGLU
    if not fused_cls.is_supported():
        pytest.skip("MXFP8 fused GroupedMLP is not supported on this system")

    device = _device()
    group_size = 2
    hidden_size = 128

    def factory() -> torch.nn.Module:
        fc1 = te_ops.GroupedLinear(
            group_size,
            hidden_size,
            2 * hidden_size,
            bias=False,
            device=device,
            dtype=_DTYPE,
            single_grouped_weight=False,
        )
        fc2 = te_ops.GroupedLinear(
            group_size,
            hidden_size,
            hidden_size,
            bias=False,
            device=device,
            dtype=_DTYPE,
            single_grouped_weight=False,
        )
        return te_ops.Sequential(
            fc1,
            te_ops.ScaledSwiGLU(glu_interleave_size=32),
            fc2,
        )

    qat_module = factory()
    master_weights = _initialize_master_weights(qat_module)
    projected_weights = _project_weights(master_weights)
    _fsdp2_shard(qat_module)

    split_sizes = torch.tensor([128, 128], dtype=torch.int32, device=device)
    num_tokens = int(split_sizes.sum().item())
    values = torch.linspace(
        -0.25,
        0.25,
        num_tokens * hidden_size,
        dtype=torch.float32,
        device=device,
    )
    x = values.reshape(num_tokens, hidden_size).to(_DTYPE).requires_grad_(True)
    probs = torch.linspace(0.5, 1.0, num_tokens, dtype=torch.float32, device=device).to(_DTYPE)
    dy = values.flip(0).reshape(num_tokens, hidden_size).to(_DTYPE)
    extra_args = (split_sizes, probs, split_sizes)

    qat_recipe = MXFP4QATMXFP8BlockScaling(backward_override=None)
    with te.autocast(enabled=True, recipe=qat_recipe):
        y_qat = qat_module(x, *extra_args)
    forward_ops = qat_module._module_groups[0]._forward_ops
    assert len(forward_ops) == 1 and isinstance(forward_ops[0][0], fused_cls)
    y_qat.backward(dy)
    assert x.grad is not None

    projected_ref, y_projected, dx_projected = _run_reference(
        factory,
        projected_weights,
        x,
        dy,
        extra_args,
        override=None,
    )
    _, y_master, _ = _run_reference(
        factory,
        master_weights,
        x,
        dy,
        extra_args,
        override=None,
    )

    _assert_projected_forward(y_qat.detach(), y_projected, y_master)
    torch.testing.assert_close(x.grad, dx_projected, rtol=0, atol=0)
    # NCCL reduction of nontrivial BF16 grouped-MLP wgrads need not be
    # bitwise identical to the single-rank reference.
    _assert_fsdp_weight_grads(qat_module, projected_ref, rtol=2e-2, atol=2e-3)
