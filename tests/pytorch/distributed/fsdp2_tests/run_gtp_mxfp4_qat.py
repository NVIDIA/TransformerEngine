#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Multi-rank DistributedWeight (GTP) coverage for MXFP4-QAT.

Run with: torchrun --nproc_per_node=2 -m pytest <this_file> -v -s --tb=short

A real DistributedWeight implementer (``_GTPShardedWeight``) drives te.Linear /
LayerNormLinear / GroupedLinear through all backward override modes; unsupported
GTP surfaces must fail fast. Master vs MXFP4-projected full weights discriminate
which operand each override used (high_precision reads the master, others the
projection).
"""

from collections.abc import Callable, Sequence

import pytest
import torch
import torch.distributed as dist

import transformer_engine.pytorch as te
import transformer_engine.pytorch.ops as te_ops
from transformer_engine.common.recipe import (
    Float8BlockScaling,
    MXFP4QATFloat8BlockScaling,
    MXFP4QATMXFP8BlockScaling,
    MXFP8BlockScaling,
)
from transformer_engine.pytorch import fp8
from transformer_engine.pytorch.mxfp4_qat import mxfp4_fake_quantize

_DTYPE = torch.bfloat16

_RECIPES = [
    pytest.param(
        (MXFP4QATMXFP8BlockScaling, MXFP8BlockScaling, fp8.check_mxfp8_support),
        id="mxfp8",
    ),
    pytest.param(
        (MXFP4QATFloat8BlockScaling, Float8BlockScaling, fp8.check_fp8_block_scaling_support),
        id="blockwise",
    ),
]

_OVERRIDES = (
    pytest.param(None, id="none"),
    pytest.param("dequantized", id="dequantized"),
    pytest.param("high_precision", id="high_precision"),
)


def _skip_if_unsupported(check_fn) -> None:
    supported, reason = check_fn()
    if not supported:
        pytest.skip(reason)


def _device() -> torch.device:
    return torch.device("cuda", torch.cuda.current_device())


class _GTPShardedWeight(torch.nn.Parameter):
    """Dim-0 sharded DistributedWeight: all-gather materialize, reduce-scatter finalize.

    AVG reduce keeps single-rank references bit-comparable (every rank feeds the same batch).
    """

    is_distributed_weight = True

    @staticmethod
    def wrap(shard: torch.Tensor, full_shape: torch.Size) -> "_GTPShardedWeight":
        weight = _GTPShardedWeight(shard.detach().clone())
        weight._full_shape = tuple(full_shape)
        weight._group = None  # set for grouped members; None -> single weight
        weight.calls = {"fwd": 0, "bwd": 0, "finalize": 0}
        return weight

    def _members(self) -> list["_GTPShardedWeight"]:
        return self._group if self._group is not None else [self]

    def _materialize(self) -> list[torch.Tensor]:
        full_weights = []
        for member in self._members():
            full = torch.empty(member._full_shape, dtype=member.dtype, device=member.device)
            dist.all_gather_into_tensor(full, member.data.contiguous())
            full_weights.append(full)
        return full_weights

    def materialize_group_for_forward(self):
        self.calls["fwd"] += 1
        out = self._materialize()
        return out if self._group is not None else out[0]

    def materialize_group_for_backward(self, **kwargs):
        self.calls["bwd"] += 1
        out = self._materialize()
        return out if self._group is not None else out[0]

    def finalize_group_grads(self, wgrads, **kwargs):
        self.calls["finalize"] += 1
        wgrad_list = list(wgrads) if isinstance(wgrads, (list, tuple)) else [wgrads]
        shard_grads = []
        for member, wgrad in zip(self._members(), wgrad_list):
            shard_grad = torch.empty_like(member.data, dtype=wgrad.dtype)
            dist.reduce_scatter_tensor(shard_grad, wgrad.contiguous(), op=dist.ReduceOp.AVG)
            shard_grads.append(shard_grad)
        return shard_grads if self._group is not None else shard_grads[0]

    def grad_buffer(self) -> torch.Tensor:
        if not hasattr(self, "_grad_buffer"):
            self._grad_buffer = torch.zeros(
                self._full_shape, dtype=torch.float32, device=self.device
            )
        return self._grad_buffer


def _shard(full: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    assert full.shape[0] % world_size == 0
    return full.chunk(world_size, dim=0)[rank].contiguous()


def _make_master_weight(
    out_features: int,
    in_features: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Construct BF16 values that MXFP8/blockwise FP8 preserve but MXFP4 changes."""
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


def _gtp_wrap_weights(
    module: torch.nn.Module, weight_names: Sequence[str]
) -> list[_GTPShardedWeight]:
    """Replace each named full-weight Parameter with its rank's dim-0 shard wrapper."""
    wrappers = []
    for name in weight_names:
        full = getattr(module, name)
        wrapper = _GTPShardedWeight.wrap(_shard(full.data), full.shape)
        setattr(module, name, wrapper)
        wrappers.append(wrapper)
    if len(wrappers) > 1:
        wrappers[0]._group = wrappers
    return wrappers


def _load_weights(module: torch.nn.Module, weights: dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, value in weights.items():
            getattr(module, name).copy_(value)


def _make_input(tokens: int, features: int, device: torch.device) -> torch.Tensor:
    values = torch.linspace(-0.25, 0.25, tokens * features, dtype=torch.float32, device=device)
    return values.reshape(tokens, features).to(_DTYPE)


def _run_reference(
    factory: Callable[[], torch.nn.Module],
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    dy: torch.Tensor,
    extra_args: Sequence,
    *,
    pure_cls: type,
    override: str | None,
) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    module = factory()
    _load_weights(module, weights)
    x_ref = x.detach().clone().requires_grad_(True)
    recipe = pure_cls(backward_override=override)
    with te.autocast(enabled=True, recipe=recipe):
        y_ref = module(x_ref, *extra_args)
    y_ref.backward(dy)
    assert x_ref.grad is not None
    return module, y_ref.detach(), x_ref.grad.detach()


def _run_gtp_case(
    factory: Callable[[], torch.nn.Module],
    weight_names: Sequence[str],
    x: torch.Tensor,
    extra_args: Sequence,
    *,
    qat_cls: type,
    pure_cls: type,
    override: str | None,
) -> None:
    """Shared body: GTP-sharded QAT module vs full-weight projected/master references."""
    device = _device()
    gtp_module = factory()
    master_weights = {
        name: _make_master_weight(
            getattr(gtp_module, name).shape[0],
            getattr(gtp_module, name).shape[1],
            device=device,
        )
        for name in weight_names
    }
    _load_weights(gtp_module, master_weights)
    projected_weights = {
        name: mxfp4_fake_quantize(weight).detach() for name, weight in master_weights.items()
    }
    for name in weight_names:
        assert not torch.equal(master_weights[name], projected_weights[name])
    wrappers = _gtp_wrap_weights(gtp_module, weight_names)

    x = x.detach().clone().requires_grad_(True)
    qat_recipe = qat_cls(backward_override=override)
    with te.autocast(enabled=True, recipe=qat_recipe):
        y_qat = gtp_module(x, *extra_args)
    dy = _make_input(y_qat.shape[0], y_qat.shape[1], device).flip(0)
    y_qat.backward(dy)
    assert x.grad is not None

    leader = wrappers[0]
    assert leader.calls["fwd"] > 0, "materialize_group_for_forward was bypassed"
    assert leader.calls["bwd"] > 0, "materialize_group_for_backward was bypassed"
    assert leader.calls["finalize"] > 0, "finalize_group_grads was bypassed"

    projected_ref, y_projected, dx_projected = _run_reference(
        factory, projected_weights, x, dy, extra_args, pure_cls=pure_cls, override=override
    )
    master_ref, y_master, dx_master = _run_reference(
        factory, master_weights, x, dy, extra_args, pure_cls=pure_cls, override=override
    )

    # Forward used the projected full weight, not the raw master.
    torch.testing.assert_close(y_qat.detach(), y_projected, rtol=0, atol=0)
    assert not torch.equal(y_qat.detach(), y_master)

    expected_dx = dx_master if override == "high_precision" else dx_projected
    torch.testing.assert_close(x.grad, expected_dx, rtol=0, atol=0)

    # Wgrad: finalize reduce-scattered (AVG of identical per-rank batches) the
    # full wgrad back to this rank's shard of the master parameter.
    expected_ref = master_ref if override == "high_precision" else projected_ref
    for name, wrapper in zip(weight_names, wrappers):
        assert wrapper.grad is not None, f"{name}: STE did not deliver a shard gradient"
        assert wrapper.grad.shape == wrapper.shape, f"{name}: gradient is not shard-shaped"
        ref_grad = getattr(expected_ref, name).grad
        assert ref_grad is not None
        torch.testing.assert_close(
            wrapper.grad,
            _shard(ref_grad),
            rtol=0,
            atol=0,
            msg=lambda msg: f"{name}: GTP QAT wgrad did not reach the master shard\n{msg}",
        )


@pytest.mark.parametrize("recipes", _RECIPES)
@pytest.mark.parametrize("override", _OVERRIDES)
def test_gtp_linear_qat_backward_overrides(recipes, override) -> None:
    """te.Linear + GTP preserves all pure override modes."""
    qat_cls, pure_cls, check_fn = recipes
    _skip_if_unsupported(check_fn)
    device = _device()
    in_features = out_features = 128

    def factory() -> torch.nn.Module:
        return te.Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=_DTYPE,
            device=device,
        )

    x = _make_input(256, in_features, device)
    _run_gtp_case(
        factory, ("weight",), x, (), qat_cls=qat_cls, pure_cls=pure_cls, override=override
    )


@pytest.mark.parametrize("recipes", _RECIPES)
@pytest.mark.parametrize("override", _OVERRIDES)
def test_gtp_layernorm_linear_qat_backward_overrides(recipes, override) -> None:
    """te.LayerNormLinear + GTP preserves all pure override modes."""
    qat_cls, pure_cls, check_fn = recipes
    _skip_if_unsupported(check_fn)
    device = _device()
    in_features = out_features = 128

    def factory() -> torch.nn.Module:
        return te.LayerNormLinear(
            in_features,
            out_features,
            bias=False,
            params_dtype=_DTYPE,
            device=device,
        )

    x = _make_input(256, in_features, device)
    _run_gtp_case(
        factory, ("weight",), x, (), qat_cls=qat_cls, pure_cls=pure_cls, override=override
    )


@pytest.mark.parametrize("recipes", _RECIPES)
@pytest.mark.parametrize("override", _OVERRIDES)
def test_gtp_grouped_linear_qat_backward_overrides(recipes, override) -> None:
    """te.GroupedLinear + GTP replays projection per expert in every override mode."""
    qat_cls, pure_cls, check_fn = recipes
    _skip_if_unsupported(check_fn)
    device = _device()
    num_gemms = 2
    in_features = out_features = 128
    m_splits = [128, 128]

    def factory() -> torch.nn.Module:
        return te.GroupedLinear(
            num_gemms,
            in_features,
            out_features,
            bias=False,
            params_dtype=_DTYPE,
            device=device,
        )

    x = _make_input(sum(m_splits), in_features, device)
    _run_gtp_case(
        factory,
        ("weight0", "weight1"),
        x,
        (m_splits,),
        qat_cls=qat_cls,
        pure_cls=pure_cls,
        override=override,
    )


# ── Loud rejection of unsupported GTP surfaces ───────────────────────


def _qat_autocast():
    return te.autocast(enabled=True, recipe=MXFP4QATMXFP8BlockScaling())


def test_gtp_ops_grouped_linear_qat_rejected() -> None:
    """ops.GroupedLinear rejects QAT + DistributedWeight instead of computing on shards."""
    _skip_if_unsupported(fp8.check_mxfp8_support)
    device = _device()
    op = te_ops.GroupedLinear(2, 128, 128, bias=False, device=device, dtype=_DTYPE)
    _gtp_wrap_weights(op, ("weight0", "weight1"))
    split_sizes = torch.tensor([128, 128], dtype=torch.int64, device=device)
    x = _make_input(256, 128, device).requires_grad_(True)
    with pytest.raises(NotImplementedError, match="does not support DistributedWeight"):
        with _qat_autocast():
            op(x, split_sizes)


def test_gtp_ops_linear_rejected() -> None:
    """te.ops.Linear has no GTP protocol and must fail fast (pure and QAT alike)."""
    _skip_if_unsupported(fp8.check_mxfp8_support)
    device = _device()
    op = te_ops.Linear(128, 128, bias=False, device=device, dtype=_DTYPE)
    _gtp_wrap_weights(op, ("weight",))
    x = _make_input(256, 128, device).requires_grad_(True)
    with pytest.raises(NotImplementedError, match="does not support distributed"):
        with _qat_autocast():
            op(x)


def test_gtp_layernorm_mlp_rejected() -> None:
    """LayerNormMLP has no GTP wiring and must fail fast."""
    _skip_if_unsupported(fp8.check_mxfp8_support)
    device = _device()
    module = te.LayerNormMLP(128, 512, bias=False, params_dtype=_DTYPE, device=device)
    _gtp_wrap_weights(module, ("fc1_weight",))
    x = _make_input(256, 128, device).requires_grad_(True)
    with pytest.raises(NotImplementedError, match="LayerNormMLP does not support distributed"):
        with _qat_autocast():
            module(x)


def test_gtp_single_grouped_weight_rejected() -> None:
    """single_grouped_weight splits members and must reject distributed weights."""
    _skip_if_unsupported(fp8.check_mxfp8_support)
    device = _device()
    module = te.GroupedLinear(
        2,
        128,
        128,
        bias=False,
        params_dtype=_DTYPE,
        device=device,
        single_grouped_weight=True,
    )
    module.make_grouped_weights()
    module.weight.is_distributed_weight = True
    x = _make_input(256, 128, device).requires_grad_(True)
    with pytest.raises(
        NotImplementedError, match="single_grouped_weight does not support distributed"
    ):
        with _qat_autocast():
            module(x, [128, 128])


def test_gtp_delayed_wgrad_rejected() -> None:
    """delay_wgrad_compute would skip finalize_weight_grads and must fail fast."""
    _skip_if_unsupported(fp8.check_mxfp8_support)
    device = _device()
    module = te.Linear(
        128,
        128,
        bias=False,
        params_dtype=_DTYPE,
        device=device,
        delay_wgrad_compute=True,
    )
    _gtp_wrap_weights(module, ("weight",))
    x = _make_input(256, 128, device).requires_grad_(True)
    with pytest.raises(
        NotImplementedError, match="Delayed weight grad computation is not supported"
    ):
        with _qat_autocast():
            module(x)
