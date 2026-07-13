#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Distributed TP/SP coverage for hybrid quantization."""

import argparse
import datetime
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from transformer_engine.common import recipe as te_recipe
from transformer_engine.pytorch.custom_recipes.quantization_factory_base import (
    nvfp4_quantizer_factory,
)
from transformer_engine.pytorch import (
    Float8CurrentScalingQuantizer,
    HybridQuantizer,
    IdentityQuantizer,
    MXFP8Quantizer,
)

# Sibling helper shared with distributed numerics tests.
TEST_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(TEST_ROOT))
from run_layer_with_overlap import _compare_tensors  # noqa: E402

# ── Global state ─────────────────────────────────────────────────────

SEQ_LEN = 32
BATCH_SIZE = 32
HIDDEN_SIZE = 128
FFN_HIDDEN_SIZE = 128
NR_HEADS = 4

WORLD_RANK = None
WORLD_SIZE = None
NCCL_WORLD = None
QUANTIZATION = None

LOSS_FN = nn.MSELoss()


# ── Hybrid recipe factories ──────────────────────────────────────────
#
# Factories stay small; these tests target TP/SP plumbing.


def _make_fp8_current_quantizer(*, fp8_dtype=tex.DType.kFloat8E4M3):
    return Float8CurrentScalingQuantizer(fp8_dtype=fp8_dtype, device="cuda")


def _make_mxfp8_quantizer(*, fp8_dtype=tex.DType.kFloat8E4M3):
    return MXFP8Quantizer(fp8_dtype=fp8_dtype)


def _hybrid_fp8_qfactory(role):
    """FP8 current scaling; backward operands use E5M2."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=_make_fp8_current_quantizer(),
            columnwise_quantizer=_make_fp8_current_quantizer(),
        )
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return _make_fp8_current_quantizer(fp8_dtype=tex.DType.kFloat8E5M2)
    return _make_fp8_current_quantizer()


def _hybrid_mxfp8_qfactory(role):
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=_make_mxfp8_quantizer(),
            columnwise_quantizer=_make_mxfp8_quantizer(),
        )
    return _make_mxfp8_quantizer()


def _hybrid_fp8_identity_qfactory(role):
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=_make_fp8_current_quantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return IdentityQuantizer()
    return _make_fp8_current_quantizer()


def _hybrid_mxfp8_identity_qfactory(role):
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=_make_mxfp8_quantizer(),
            columnwise_quantizer=IdentityQuantizer(),
        )
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return IdentityQuantizer()
    return _make_mxfp8_quantizer()


def _identity_qfactory(role):  # pylint: disable=unused-argument
    return IdentityQuantizer()


def _hybrid_nvfp4_qfactory(role):
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        return HybridQuantizer(
            rowwise_quantizer=nvfp4_quantizer_factory(role),
            columnwise_quantizer=nvfp4_quantizer_factory(role),
        )
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return nvfp4_quantizer_factory(role)
    return nvfp4_quantizer_factory(role)


def _hybrid_mxfp8_nvfp4_qfactory(role):
    """Cross-format recipe: MXFP8 rowwise, NVFP4 columnwise."""
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return nvfp4_quantizer_factory(role)
    # Forward/boundary roles keep both formats available.
    return HybridQuantizer(
        rowwise_quantizer=_make_mxfp8_quantizer(),
        columnwise_quantizer=nvfp4_quantizer_factory(role),
    )


def hybrid_recipe():
    """Return a fresh CustomRecipe for the selected test recipe."""
    if QUANTIZATION == "hybrid_fp8":
        return te_recipe.CustomRecipe(qfactory=_hybrid_fp8_qfactory)
    if QUANTIZATION == "hybrid_mxfp8":
        return te_recipe.CustomRecipe(qfactory=_hybrid_mxfp8_qfactory)
    if QUANTIZATION == "hybrid_fp8_identity":
        return te_recipe.CustomRecipe(qfactory=_hybrid_fp8_identity_qfactory)
    if QUANTIZATION == "hybrid_mxfp8_identity":
        return te_recipe.CustomRecipe(qfactory=_hybrid_mxfp8_identity_qfactory)
    if QUANTIZATION == "identity":
        return te_recipe.CustomRecipe(qfactory=_identity_qfactory)
    if QUANTIZATION == "hybrid_nvfp4":
        return te_recipe.CustomRecipe(qfactory=_hybrid_nvfp4_qfactory)
    if QUANTIZATION == "hybrid_mxfp8_nvfp4":
        return te_recipe.CustomRecipe(qfactory=_hybrid_mxfp8_nvfp4_qfactory)
    raise ValueError(f"Unknown hybrid QUANTIZATION={QUANTIZATION!r}")


# ── Tolerances ───────────────────────────────────────────────────────
#
# Mostly upstream ``run_numerics.py`` tolerances. NVFP4 needs a slightly
# larger atol for the measured column+SP Linear output.


def _get_tolerances():
    if QUANTIZATION == "identity":
        # BF16 TP reductions accumulate in a different order.
        return {"rtol": 1.6e-2, "atol": 1.0e-5}
    if QUANTIZATION in ("hybrid_fp8", "hybrid_fp8_identity"):
        # Loose because of sequence parallel & amax reduction (fp8_cs).
        return {"rtol": 0.4, "atol": 0.25}
    if QUANTIZATION in ("hybrid_mxfp8", "hybrid_mxfp8_identity"):
        return {"rtol": 0.125, "atol": 0.0625}
    if QUANTIZATION == "hybrid_nvfp4":
        # Measured column+SP Linear output max abs is ~0.144.
        return {"rtol": 0.125, "atol": 0.15}
    if QUANTIZATION == "hybrid_mxfp8_nvfp4":
        # Backward GEMMs run in NVFP4 -> inherit the (looser) NVFP4 bounds.
        return {"rtol": 0.125, "atol": 0.12}
    raise ValueError(f"No tolerances for QUANTIZATION={QUANTIZATION!r}")


# ── Distributed helpers ──────────────────────────────────────────────


def dist_print(msg, src=None, error=False):
    stream = sys.stderr if error else sys.stdout
    if WORLD_RANK == (0 if src is None else src):
        stream.write(f"[rank{WORLD_RANK}] {msg}\n")
        stream.flush()


def _gather(tensor, dim=0):
    """All-gather with ``run_numerics.py`` gradient scaling."""

    class HalfGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, inp):
            return inp

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output / WORLD_SIZE

    tensor = HalfGradient.apply(tensor)
    gathered = torch.distributed.nn.functional.all_gather(tensor, group=NCCL_WORLD)
    return torch.cat(gathered, dim=dim)


def _copy_params(model_distributed, model_single):
    """Copy single-node parameters into the local TP shard."""
    for dp, sp in zip(model_distributed.parameters(), model_single.parameters()):
        with torch.no_grad():
            to_copy = sp
            for dim, _ in enumerate(dp.shape):
                if dp.shape[dim] != sp.shape[dim]:
                    start = WORLD_RANK * dp.shape[dim]
                    end = (WORLD_RANK + 1) * dp.shape[dim]
                    indices = [slice(None)] * max(min(dim, len(dp.shape) - 1), 0)
                    indices.append(slice(start, end))
                    if dim < len(dp.shape) - 1:
                        indices.append(slice(None))
                    to_copy = sp[tuple(indices)]
            dp.copy_(to_copy)


def _match_param_sizes(dist_param, single_param):
    indices = [slice(None)] * len(single_param.shape)
    for i in range(len(dist_param.shape)):
        if dist_param.shape[i] != single_param.shape[i]:
            start = WORLD_RANK * dist_param.shape[i]
            end = (WORLD_RANK + 1) * dist_param.shape[i]
            indices[i] = slice(start, end)
    return single_param[tuple(indices)]


def _check_outputs(output_single, output_dist, label="outputs", *, tolerances=None):
    failed = torch.tensor([0], dtype=torch.uint8, device="cuda")
    f, info = _compare_tensors(
        label, output_dist, output_single, **(tolerances or _get_tolerances())
    )
    if f:
        dist_print(info, src=WORLD_RANK, error=True)
    failed[0] = int(f)
    dist.all_reduce(failed, dist.ReduceOp.MAX, NCCL_WORLD)
    assert not bool(failed.item()), f"{label}: numerical check failed on at least one rank"


def _collective_assert(condition, label):
    """Raise on every rank only after all ranks report structural status."""
    failed = torch.tensor([not condition], dtype=torch.uint8, device="cuda")
    if not condition:
        dist_print(label, src=WORLD_RANK, error=True)
    dist.all_reduce(failed, dist.ReduceOp.MAX, NCCL_WORLD)
    assert not bool(failed.item()), f"{label}: failed on at least one rank"


def _check_gradients(
    model_dist, model_single, *, bitwise=False, reduce_replicated=False, tolerances=None
):
    """Compare every parameter gradient, including presence and parameter count.

    Sequence-parallel replicated parameters accumulate a partial gradient on
    each rank. ``reduce_replicated`` reconstructs their full reference gradient
    without mutating the gradient held by the model.
    """
    dist_params = list(model_dist.named_parameters())
    single_params = list(model_single.named_parameters())
    local_counts = torch.tensor(
        [len(dist_params), len(single_params)], dtype=torch.int64, device="cuda"
    )
    min_counts = local_counts.clone()
    max_counts = local_counts.clone()
    dist.all_reduce(min_counts, dist.ReduceOp.MIN, NCCL_WORLD)
    dist.all_reduce(max_counts, dist.ReduceOp.MAX, NCCL_WORLD)
    _collective_assert(
        torch.equal(min_counts, max_counts) and len(dist_params) == len(single_params),
        "parameter count mismatch: "
        f"local distributed={len(dist_params)}, local single={len(single_params)}, "
        f"global min={min_counts.tolist()}, global max={max_counts.tolist()}",
    )
    for i, ((name, pd), (single_name, ps)) in enumerate(zip(dist_params, single_params)):
        _collective_assert(
            name == single_name,
            f"parameter {i} name mismatch: {name!r} != {single_name!r}",
        )
        local_presence = torch.tensor(
            [pd.grad is not None, ps.grad is not None], dtype=torch.uint8, device="cuda"
        )
        min_presence = local_presence.clone()
        max_presence = local_presence.clone()
        dist.all_reduce(min_presence, dist.ReduceOp.MIN, NCCL_WORLD)
        dist.all_reduce(max_presence, dist.ReduceOp.MAX, NCCL_WORLD)
        _collective_assert(
            torch.equal(min_presence, max_presence)
            and local_presence[0].item() == local_presence[1].item(),
            f"grad[{i}].{name}: gradient presence differs locally or across ranks: "
            f"local={local_presence.tolist()}, global min={min_presence.tolist()}, "
            f"global max={max_presence.tolist()}",
        )
        if pd.grad is None:
            continue
        pd_grad = pd.grad
        local_reduce = torch.tensor(
            [reduce_replicated and pd_grad.shape == ps.grad.shape],
            dtype=torch.uint8,
            device="cuda",
        )
        min_reduce = local_reduce.clone()
        max_reduce = local_reduce.clone()
        dist.all_reduce(min_reduce, dist.ReduceOp.MIN, NCCL_WORLD)
        dist.all_reduce(max_reduce, dist.ReduceOp.MAX, NCCL_WORLD)
        _collective_assert(
            torch.equal(min_reduce, max_reduce),
            f"grad[{i}].{name}: replicated-gradient reduction branch differs across ranks",
        )
        if local_reduce.item():
            pd_grad = pd_grad.detach().clone()
            dist.all_reduce(pd_grad, dist.ReduceOp.SUM, NCCL_WORLD)
        ps_grad = _match_param_sizes(pd_grad, ps.grad)
        label = f"grad[{i}].{name}"
        if bitwise:
            _check_bitwise(pd_grad, ps_grad, label)
        else:
            _check_outputs(ps_grad, pd_grad, label=label, tolerances=tolerances)


def _check_input_gradient(inp_dist, inp_single, label, *, bitwise=False):
    """Compare the local TP/SP input-gradient shard with its full reference."""
    local_presence = torch.tensor(
        [inp_dist.grad is not None, inp_single.grad is not None],
        dtype=torch.uint8,
        device="cuda",
    )
    min_presence = local_presence.clone()
    max_presence = local_presence.clone()
    dist.all_reduce(min_presence, dist.ReduceOp.MIN, NCCL_WORLD)
    dist.all_reduce(max_presence, dist.ReduceOp.MAX, NCCL_WORLD)
    _collective_assert(
        torch.equal(min_presence, max_presence)
        and local_presence[0].item() == local_presence[1].item(),
        f"{label}: input-gradient presence differs locally or across ranks: "
        f"local={local_presence.tolist()}, global min={min_presence.tolist()}, "
        f"global max={max_presence.tolist()}",
    )
    _collective_assert(
        inp_dist.grad is not None,
        f"{label}: both input gradients are unexpectedly absent",
    )
    expected = _match_param_sizes(inp_dist.grad, inp_single.grad)
    if bitwise:
        _check_bitwise(inp_dist.grad, expected, label)
    else:
        _check_outputs(expected, inp_dist.grad, label=label)


def _apply_models(model_single, model_dist, inp_single, inp_dist, **kwargs):
    """Run both models with fresh CustomRecipe instances."""
    inp_single.requires_grad_()
    inp_dist.requires_grad_()
    with te.autocast(enabled=True, recipe=hybrid_recipe()):
        out_single = model_single(inp_single, **kwargs)
    with te.autocast(enabled=True, recipe=hybrid_recipe()):
        out_dist = model_dist(inp_dist, **kwargs)
    return out_single, out_dist


def _loss_backward(out_single, out_dist):
    target = torch.randn_like(out_single)
    LOSS_FN(out_single, target).backward()
    LOSS_FN(out_dist, target).backward()


# ── Test 1: te.Linear TP (column + row) × SP (on/off) ────────────────


def _test_linear(parallel_mode, sequence_parallel, params_dtype=torch.bfloat16, amax_stress=False):
    dist_print(
        f"linear: parallel_mode={parallel_mode} sequence_parallel={sequence_parallel}"
        f" dtype={params_dtype} amax_stress={amax_stress}"
    )

    torch.manual_seed(12345)
    torch.cuda.manual_seed(12345)

    model_single = te.Linear(HIDDEN_SIZE, HIDDEN_SIZE, params_dtype=params_dtype).cuda()
    model_dist = te.Linear(
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        parallel_mode=parallel_mode,
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
    ).cuda()

    _copy_params(model_dist, model_single)

    # Match run_numerics._test_linear input layouts.
    inp_single = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
    if parallel_mode == "row":
        split = HIDDEN_SIZE // WORLD_SIZE
        inp_dist = inp_single[:, WORLD_RANK * split : (WORLD_RANK + 1) * split].clone()
    elif parallel_mode == "column":
        if sequence_parallel:
            # SP column: input is sharded along batch/sequence dim 0.
            inp_single = torch.empty((WORLD_SIZE * BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
            inp_dist = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
            if amax_stress and WORLD_RANK == WORLD_SIZE - 1:
                # One-rank outlier before SP gather.
                inp_dist[-1, -1] = 1.0e3
            inp_single = _gather(inp_dist, dim=0).detach()
        else:
            inp_dist = inp_single.clone()
    else:
        raise ValueError(parallel_mode)

    out_single, out_dist = _apply_models(model_single, model_dist, inp_single, inp_dist)

    # For column-parallel: output is split along feature dim 1; gather.
    # For row-parallel + SP: output is split along seq dim 0; gather.
    if parallel_mode == "column" or (sequence_parallel and parallel_mode == "row"):
        gather_dim = 1 if parallel_mode == "column" else 0
        out_dist = _gather(out_dist, dim=gather_dim)

    _loss_backward(out_single, out_dist)
    _check_outputs(
        out_single,
        out_dist,
        label=f"linear[{parallel_mode},sp={sequence_parallel},amax_stress={amax_stress}]",
    )

    _check_input_gradient(
        inp_dist,
        inp_single,
        label=f"linear[{parallel_mode},sp={sequence_parallel},amax_stress={amax_stress}] dgrad",
    )
    _check_gradients(model_dist, model_single, reduce_replicated=sequence_parallel)


def test_linear():
    for parallel_mode in ["column", "row"]:
        for sequence_parallel in [False, True]:
            _test_linear(parallel_mode, sequence_parallel)
    # Current-scaling amax stress: one rank owns an outlier before SP gather.
    if QUANTIZATION in ("hybrid_fp8", "hybrid_fp8_identity"):
        _test_linear("column", True, amax_stress=True)


# ── Test 1b: te.Linear hybrid-vs-vanilla bitwise operand equivalence ─


def vanilla_recipe():
    """Built-in recipe for same-format hybrid-vs-vanilla checks."""
    if QUANTIZATION == "hybrid_fp8":
        return te_recipe.Float8CurrentScaling()
    if QUANTIZATION == "hybrid_mxfp8":
        return te_recipe.MXFP8BlockScaling()
    if QUANTIZATION == "hybrid_nvfp4":
        return te_recipe.NVFP4BlockScaling()
    if QUANTIZATION == "identity":
        return None
    raise ValueError(f"No vanilla recipe for QUANTIZATION={QUANTIZATION!r}")


def _backward_not_bitwise_comparable():
    """NVFP4 backward consumes SR RNG differently in hybrid vs vanilla."""
    return QUANTIZATION == "hybrid_nvfp4"


def _check_bitwise(actual, expected, label):
    """Assert bitwise equality (rtol=0, atol=0), all-reduced across ranks."""
    failed = torch.tensor([0], dtype=torch.uint8, device="cuda")
    try:
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    except AssertionError as exc:
        dist_print(f"{label}: {exc}", src=WORLD_RANK, error=True)
        failed[0] = 1
    dist.all_reduce(failed, dist.ReduceOp.MAX, NCCL_WORLD)
    assert not bool(failed.item()), f"{label}: not bitwise-identical on at least one rank"


def _test_linear_vs_vanilla(parallel_mode, sequence_parallel, params_dtype=torch.bfloat16):
    """Same-topology Linear check: forward bitwise, backward where comparable."""
    dist_print(
        f"linear_vs_vanilla: parallel_mode={parallel_mode} sequence_parallel={sequence_parallel}"
    )

    def run(recipe):
        # Fresh model per recipe (re-seeded for identical weights): TE caches a
        # quantized weight workspace on the module, so reusing one model would
        # let the first recipe's cached weight contaminate the second.
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)
        model = te.Linear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            tp_size=WORLD_SIZE,
            tp_group=NCCL_WORLD,
            parallel_mode=parallel_mode,
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
        ).cuda()

        torch.manual_seed(34567)
        torch.cuda.manual_seed(34567)
        inp = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
        if parallel_mode == "row":
            split = HIDDEN_SIZE // WORLD_SIZE
            inp = inp[:, WORLD_RANK * split : (WORLD_RANK + 1) * split].clone()
        inp.requires_grad_()

        with te.autocast(enabled=recipe is not None, recipe=recipe):
            out = model(inp)
        # Fixed, recipe-independent target so both backward graphs match.
        torch.manual_seed(54321)
        torch.cuda.manual_seed(54321)
        target = torch.randn_like(out)
        LOSS_FN(out, target).backward()
        weight_grads = [p.grad.detach().clone() for p in model.parameters() if p.grad is not None]
        return out.detach().clone(), inp.grad.detach().clone(), weight_grads

    out_h, dinp_h, wgrads_h = run(hybrid_recipe())
    out_v, dinp_v, wgrads_v = run(vanilla_recipe())

    tag = f"linear_vs_vanilla[{parallel_mode},sp={sequence_parallel}]"

    # Same-topology fprop operands should match bitwise.
    _check_bitwise(out_h, out_v, f"{tag} forward")

    # NVFP4 backward consumes columnwise stochastic-rounding RNG differently.
    if not _backward_not_bitwise_comparable():
        _check_bitwise(dinp_h, dinp_v, f"{tag} dgrad")
        assert len(wgrads_h) == len(wgrads_v), f"{tag}: weight-grad count mismatch"
        for i, (gh, gv) in enumerate(zip(wgrads_h, wgrads_v)):
            _check_bitwise(gh, gv, f"{tag} wgrad[{i}]")


def test_linear_vs_vanilla():
    # These recipes have no same-format vanilla bitwise target.
    if QUANTIZATION in (
        "hybrid_mxfp8_nvfp4",
        "hybrid_fp8_identity",
        "hybrid_mxfp8_identity",
    ):
        dist_print("linear_vs_vanilla: skipped for hybrid without a vanilla equivalent")
        return
    for parallel_mode in ["column", "row"]:
        for sequence_parallel in [False, True]:
            _test_linear_vs_vanilla(parallel_mode, sequence_parallel)


def _same_format_parity_supported():
    return QUANTIZATION in ("hybrid_fp8", "hybrid_mxfp8", "identity")


def _check_same_topology_parity(
    out_h, dinp_h, model_h, out_v, dinp_v, model_v, tag, *, tolerances=None
):
    check = (
        _check_bitwise
        if tolerances is None
        else lambda a, e, label: _check_outputs(e, a, label=label, tolerances=tolerances)
    )
    check(out_h, out_v, f"{tag} forward")
    check(dinp_h, dinp_v, f"{tag} dgrad")
    _check_gradients(model_h, model_v, bitwise=tolerances is None, tolerances=tolerances)


def _test_layernorm_linear_vs_vanilla(sequence_parallel, params_dtype=torch.bfloat16):
    if not _same_format_parity_supported():
        dist_print("layernorm_linear_vs_vanilla: skipped for recipe without vanilla equivalent")
        return
    dist_print(f"layernorm_linear_vs_vanilla: sequence_parallel={sequence_parallel}")

    def run(recipe_obj):
        torch.manual_seed(23456)
        torch.cuda.manual_seed(23456)
        model = te.LayerNormLinear(
            HIDDEN_SIZE,
            HIDDEN_SIZE,
            tp_size=WORLD_SIZE,
            tp_group=NCCL_WORLD,
            parallel_mode="column",
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
        ).cuda()
        torch.manual_seed(45670)
        torch.cuda.manual_seed(45670)
        inp = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
        inp.requires_grad_()
        with te.autocast(enabled=recipe_obj is not None, recipe=recipe_obj):
            out = model(inp)
        torch.manual_seed(45671)
        torch.cuda.manual_seed(45671)
        LOSS_FN(out, torch.randn_like(out)).backward()
        return model, out.detach().clone(), inp.grad.detach().clone()

    model_h, out_h, dinp_h = run(hybrid_recipe())
    model_v, out_v, dinp_v = run(vanilla_recipe())
    _check_same_topology_parity(
        out_h,
        dinp_h,
        model_h,
        out_v,
        dinp_v,
        model_v,
        f"layernorm_linear_vs_vanilla[sp={sequence_parallel}]",
    )


def test_layernorm_linear_vs_vanilla():
    for sequence_parallel in [False, True]:
        _test_layernorm_linear_vs_vanilla(sequence_parallel)


def _test_layernorm_mlp_vs_vanilla(sequence_parallel, params_dtype=torch.bfloat16):
    if not _same_format_parity_supported():
        dist_print("layernorm_mlp_vs_vanilla: skipped for recipe without vanilla equivalent")
        return
    dist_print(f"layernorm_mlp_vs_vanilla: sequence_parallel={sequence_parallel}")

    def run(recipe_obj):
        torch.manual_seed(45678)
        torch.cuda.manual_seed(45678)
        model = te.LayerNormMLP(
            HIDDEN_SIZE,
            FFN_HIDDEN_SIZE,
            tp_size=WORLD_SIZE,
            tp_group=NCCL_WORLD,
            set_parallel_mode=True,
            sequence_parallel=sequence_parallel,
            params_dtype=params_dtype,
        ).cuda()
        torch.manual_seed(56780)
        torch.cuda.manual_seed(56780)
        inp = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
        inp.requires_grad_()
        with te.autocast(enabled=recipe_obj is not None, recipe=recipe_obj):
            out = model(inp)
        torch.manual_seed(56781)
        torch.cuda.manual_seed(56781)
        LOSS_FN(out, torch.randn_like(out)).backward()
        return model, out.detach().clone(), inp.grad.detach().clone()

    model_h, out_h, dinp_h = run(hybrid_recipe())
    model_v, out_v, dinp_v = run(vanilla_recipe())
    _check_same_topology_parity(
        out_h,
        dinp_h,
        model_h,
        out_v,
        dinp_v,
        model_v,
        f"layernorm_mlp_vs_vanilla[sp={sequence_parallel}]",
        # Hybrid CustomRecipe disables quantized-norm fusion while the native
        # recipe uses it. Both paths consume identical quantized GEMM operands,
        # but the BF16 norm boundary can differ by one rounding step.
        tolerances=(None if QUANTIZATION == "identity" else {"rtol": 2**-7, "atol": 2**-10}),
    )


def test_layernorm_mlp_vs_vanilla():
    for sequence_parallel in [False, True]:
        _test_layernorm_mlp_vs_vanilla(sequence_parallel)


# ── Test 2: te.LayerNormLinear column + SP ──────────────────────────


def _test_layernorm_linear(sequence_parallel, params_dtype=torch.bfloat16):
    """Column-parallel LayerNormLinear with optional SP."""
    dist_print(f"layernorm_linear: parallel_mode=column sequence_parallel={sequence_parallel}")

    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)

    model_single = te.LayerNormLinear(HIDDEN_SIZE, HIDDEN_SIZE, params_dtype=params_dtype).cuda()
    model_dist = te.LayerNormLinear(
        HIDDEN_SIZE,
        HIDDEN_SIZE,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        parallel_mode="column",
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
    ).cuda()

    _copy_params(model_dist, model_single)

    if sequence_parallel:
        inp_dist = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
        inp_single = _gather(inp_dist, dim=0).detach()
    else:
        inp_single = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
        inp_dist = inp_single.clone()

    out_single, out_dist = _apply_models(model_single, model_dist, inp_single, inp_dist)

    # Column-parallel output: gather along dim 1.
    out_dist = _gather(out_dist, dim=1)

    _loss_backward(out_single, out_dist)
    _check_outputs(out_single, out_dist, label=f"layernorm_linear[sp={sequence_parallel}]")

    _check_input_gradient(
        inp_dist, inp_single, label=f"layernorm_linear[sp={sequence_parallel}] dgrad"
    )
    _check_gradients(model_dist, model_single, reduce_replicated=sequence_parallel)


def test_layernorm_linear():
    for sequence_parallel in [False, True]:
        _test_layernorm_linear(sequence_parallel)


# ── Test 3: te.LayerNormMLP + TP + SP ───────────────────────────────


def _test_layernorm_mlp(sequence_parallel, params_dtype=torch.bfloat16):
    """LayerNormMLP with set_parallel_mode=True and optional SP."""
    dist_print(f"layernorm_mlp: parallel_mode=set sequence_parallel={sequence_parallel}")

    torch.manual_seed(45678)
    torch.cuda.manual_seed(45678)

    model_single = te.LayerNormMLP(HIDDEN_SIZE, FFN_HIDDEN_SIZE, params_dtype=params_dtype).cuda()
    model_dist = te.LayerNormMLP(
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        set_parallel_mode=True,
        sequence_parallel=sequence_parallel,
        params_dtype=params_dtype,
    ).cuda()

    _copy_params(model_dist, model_single)

    if sequence_parallel:
        inp_dist = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
        inp_single = _gather(inp_dist, dim=0).detach()
    else:
        inp_single = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
        inp_dist = inp_single.clone()

    out_single, out_dist = _apply_models(model_single, model_dist, inp_single, inp_dist)

    # Row-parallel FC2 output is in the full hidden space; with SP it is
    # reduce-scattered along the token dim 0, so gather it back.
    if sequence_parallel:
        out_dist = _gather(out_dist, dim=0)

    _loss_backward(out_single, out_dist)
    _check_outputs(out_single, out_dist, label=f"layernorm_mlp[sp={sequence_parallel}]")

    _check_input_gradient(
        inp_dist, inp_single, label=f"layernorm_mlp[sp={sequence_parallel}] dgrad"
    )
    _check_gradients(model_dist, model_single, reduce_replicated=sequence_parallel)


def test_layernorm_mlp():
    for sequence_parallel in [False, True]:
        _test_layernorm_mlp(sequence_parallel)


# ── Test 4: te.TransformerLayer + TP + SP ───────────────────────────


def _test_transformer_layer(sequence_parallel, params_dtype=torch.bfloat16):
    """TransformerLayer integration with TP and optional SP."""
    dist_print(f"transformer_layer: parallel_mode=set sequence_parallel={sequence_parallel}")

    torch.manual_seed(34567)
    torch.cuda.manual_seed(34567)

    model_single = te.TransformerLayer(
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        NR_HEADS,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        fuse_qkv_params=True,
        params_dtype=params_dtype,
    ).cuda()
    model_dist = te.TransformerLayer(
        HIDDEN_SIZE,
        FFN_HIDDEN_SIZE,
        NR_HEADS,
        tp_size=WORLD_SIZE,
        tp_group=NCCL_WORLD,
        set_parallel_mode=True,
        sequence_parallel=sequence_parallel,
        seq_length=WORLD_SIZE * SEQ_LEN if sequence_parallel else None,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        fuse_qkv_params=True,
        params_dtype=params_dtype,
    ).cuda()

    _copy_params(model_dist, model_single)

    inp_single = (
        torch.randn((WORLD_SIZE * SEQ_LEN, BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
    )
    if sequence_parallel:
        inp_dist = inp_single[WORLD_RANK * SEQ_LEN : (WORLD_RANK + 1) * SEQ_LEN, :, :].contiguous()
    else:
        inp_dist = inp_single.clone()

    out_single, out_dist = _apply_models(model_single, model_dist, inp_single, inp_dist)

    if sequence_parallel:
        out_dist = _gather(out_dist, dim=0)

    _loss_backward(out_single, out_dist)
    _check_outputs(out_single, out_dist, label=f"transformer_layer[sp={sequence_parallel}]")

    _check_input_gradient(
        inp_dist, inp_single, label=f"transformer_layer[sp={sequence_parallel}] dgrad"
    )
    _check_gradients(model_dist, model_single, reduce_replicated=sequence_parallel)


def test_transformer_layer():
    for sequence_parallel in [False, True]:
        _test_transformer_layer(sequence_parallel)


# ── Driver ───────────────────────────────────────────────────────────


def main(argv=None):
    global WORLD_RANK, WORLD_SIZE, NCCL_WORLD, QUANTIZATION

    WORLD_RANK = int(os.getenv("RANK", "0"))
    WORLD_SIZE = int(os.getenv("WORLD_SIZE", "1"))
    LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))
    LOCAL_SIZE = int(os.getenv("LOCAL_WORLD_SIZE", "1"))

    assert WORLD_SIZE == LOCAL_SIZE, "This test is single-node only"
    assert LOCAL_SIZE <= torch.cuda.device_count()

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend="nccl",
        rank=WORLD_RANK,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(seconds=60),
        init_method="env://",
        device_id=torch.device(f"cuda:{LOCAL_RANK}"),
    )
    NCCL_WORLD = dist.new_group(backend="nccl")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quantization",
        type=str,
        required=True,
        choices=[
            "hybrid_fp8",
            "hybrid_mxfp8",
            "hybrid_fp8_identity",
            "hybrid_mxfp8_identity",
            "identity",
            "hybrid_nvfp4",
            "hybrid_mxfp8_nvfp4",
        ],
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=[
            "all",
            "linear",
            "linear_vs_vanilla",
            "layernorm_linear_vs_vanilla",
            "layernorm_mlp_vs_vanilla",
            "layernorm_linear",
            "layernorm_mlp",
            "transformer_layer",
        ],
        help="Run only the named test (speeds up iterative debugging)",
    )
    args = parser.parse_args(argv)
    QUANTIZATION = args.quantization

    test_map = {
        "linear": test_linear,
        "linear_vs_vanilla": test_linear_vs_vanilla,
        "layernorm_linear_vs_vanilla": test_layernorm_linear_vs_vanilla,
        "layernorm_mlp_vs_vanilla": test_layernorm_mlp_vs_vanilla,
        "layernorm_linear": test_layernorm_linear,
        "layernorm_mlp": test_layernorm_mlp,
        "transformer_layer": test_transformer_layer,
    }
    if args.test == "all":
        tests_to_run = list(test_map.values())
    else:
        tests_to_run = [test_map[args.test]]

    for test_fn in tests_to_run:
        dist_print(f"=== Starting {test_fn.__name__} ===")
        test_fn()
        dist.barrier()
        dist_print(f"=== Passed {test_fn.__name__} ===")

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
