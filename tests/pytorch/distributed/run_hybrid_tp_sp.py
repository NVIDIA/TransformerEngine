#!/usr/bin/python3

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Distributed TP/SP numerics tests for hybrid quantization.

Launched via torchrun from ``test_hybrid_tp_sp.py``. Each test compares a
tensor-parallel (optionally sequence-parallel) TE module running a hybrid
``CustomRecipe`` against a single-node reference module running the same
recipe. Weights are synchronized via ``_copy_params`` (shared with
``run_numerics.py``), so any drift between the two paths is a hybrid-
specific TP/SP issue rather than an initialization artifact.

Test surface: ``te.Linear`` (column/row x SP on/off, plus a bitwise
hybrid-vs-vanilla operand-equivalence check), ``te.LayerNormLinear``,
``te.LayerNormMLP``, and ``te.TransformerLayer`` (all with SP on/off). The
non-attention tests also compare per-parameter gradients in the no-SP configs,
where grads align directly with the single-node reference.

Recipes: same-format (FP8-current, MXFP8, NVFP4) for a clean signal and the
bitwise check, plus a cross-format one (MXFP8 fwd / NVFP4 bwd) that exercises
the forward-vs-backward all-gather format asymmetry (fwd gathers rowwise, bwd
columnwise) -- which same-format recipes cannot surface.

Two comparison kinds with different tolerances:
  * distributed-vs-single-node (``_test_*``): inherently loose -- the sharded
    side quantizes per-shard and reduces across ranks, so it is never bitwise.
    ``_get_tolerances`` matches upstream ``run_numerics.py`` per format.
  * hybrid-vs-vanilla (``_test_linear_vs_vanilla``): same topology, so bitwise
    (``rtol=0, atol=0``) for forward (all configs) and backward (non-SP).
"""

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
from transformer_engine.pytorch import (
    Float8CurrentScalingQuantizer,
    HybridQuantizer,
    IdentityQuantizer,
    MXFP8Quantizer,
    NVFP4Quantizer,
)

# Reuse helpers from run_numerics.py (sibling import — same pattern as
# run_numerics.py's own `from run_layer_with_overlap import _compare_tensors`).
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
# Both rowwise and columnwise sub-quantizers use the same format so the
# observed distributed numerics only reflect TP/SP interactions and not
# cross-format composition noise. For comparison against vanilla built-in
# recipes that have well-understood TP/SP tolerances, see upstream
# ``run_numerics.py``.


def _make_fp8_current_quantizer(*, fp8_dtype=tex.DType.kFloat8E4M3):
    return Float8CurrentScalingQuantizer(fp8_dtype=fp8_dtype, device="cuda")


def _make_mxfp8_quantizer(*, fp8_dtype=tex.DType.kFloat8E4M3):
    return MXFP8Quantizer(fp8_dtype=fp8_dtype)


def _hybrid_fp8_qfactory(role):
    """FP8 current scaling in both directions for fwd roles; E5M2 for
    grad roles (standard Hybrid:HYBRID format pairing)."""
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
    # MXFP8 uses E4M3 for every pass (its canonical Format.E4M3)
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


def _make_nvfp4_bare():
    """Bare NVFP4Quantizer (1D, no RHT/SR/2D), used by the cross-format recipe
    to avoid cross-operand RHT-consistency concerns in the mixed MXFP8/NVFP4
    GEMMs."""
    return NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1)


def _make_nvfp4_quantizer(role=None):
    """Role-based NVFP4Quantizer mirroring ``nvfp4_quantizer_factory`` /
    :class:`NVFP4BlockScaling`, but with 2D quantization disabled.

    Per role: weight = no RHT/SR, input = RHT, grad = RHT + SR.

    ``with_2d_quantization`` is forced off everywhere: the 2D quantize-transpose
    kernel has no columnwise-only path, so a hybrid columnwise sub-quantizer
    cannot use it.
    TODO(negvet): enable 2D for the rowwise direction once
    https://github.com/NVIDIA/TransformerEngine/pull/3027 lands.
    """
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    is_weight = is_linear and role.tensor_type == "weight"
    is_grad = is_linear and role.tensor_type in ("grad_output", "grad_input")
    return NVFP4Quantizer(
        fp4_dtype=tex.DType.kFloat4E2M1,
        with_rht=not is_weight,
        with_post_rht_amax=not is_weight,
        with_2d_quantization=False,  # TODO(negvet): enable via PR #3027
        stochastic_rounding=is_grad,
        with_random_sign_mask=True,
    )


def _hybrid_nvfp4_qfactory(role):
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("input", "weight", "output"):
        # Same per-role config for both directions (RHT/SR are per role).
        return HybridQuantizer(
            rowwise_quantizer=_make_nvfp4_quantizer(role),
            columnwise_quantizer=_make_nvfp4_quantizer(role),
        )
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return _make_nvfp4_quantizer(role)
    return _make_nvfp4_quantizer(role)


def _hybrid_mxfp8_nvfp4_qfactory(role):
    """Cross-format: MXFP8 forward (rowwise) + NVFP4 backward (columnwise).

      fprop TN: weight.row(MXFP8) x input.row(MXFP8)       -> MXFP8 x MXFP8
      dgrad NN: weight.col(NVFP4) x grad_output.row(NVFP4) -> NVFP4 x NVFP4
      wgrad NT: input.col(NVFP4)  x grad_output.col(NVFP4) -> NVFP4 x NVFP4

    So weight/input = Hybrid(row=MXFP8, col=NVFP4), grad_output = plain NVFP4.
    The forward all-gather consumes the MXFP8 rowwise sub-storage and the
    backward all-gather the NVFP4 columnwise one -- the fwd-vs-bwd format
    asymmetry that same-format recipes cannot surface.
    """
    is_linear = role is not None and role.module_type in ("linear", "grouped_linear")
    if is_linear and role.tensor_type in ("grad_output", "grad_input"):
        return _make_nvfp4_bare()
    # input / weight / output / unknown-None: MXFP8 rowwise + NVFP4 columnwise.
    return HybridQuantizer(
        rowwise_quantizer=_make_mxfp8_quantizer(),
        columnwise_quantizer=_make_nvfp4_bare(),
    )


def hybrid_recipe():
    """Fresh CustomRecipe instance per call (mirrors
    ``run_numerics.quantization_recipe`` lifetime contract)."""
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
# These match upstream ``run_numerics.py::_get_tolerances`` exactly, for
# the same TP/SP-distributed-vs-single-node comparison. A same-format
# hybrid inherits the underlying format's distributed behaviour: both the
# distributed and single-node models run the *same* two-pass hybrid recipe,
# so the two-pass quantization cancels in the comparison and the only
# remaining difference is the TP/SP path (per-shard quantization,
# all-gather/reduce-scatter order, and -- for fp8_cs only -- cross-rank
# amax reduction). There is therefore no reason for hybrid to need looser
# bounds than the vanilla format.


def _get_tolerances():
    if QUANTIZATION == "identity":
        # Same tolerance as upstream distributed BF16 numerics: TP row
        # reductions can accumulate in a different order from the single-node ref.
        return {"rtol": 1.6e-2, "atol": 1.0e-5}
    if QUANTIZATION in ("hybrid_fp8", "hybrid_fp8_identity"):
        # Loose because of sequence parallel & amax reduction (fp8_cs).
        return {"rtol": 0.4, "atol": 0.25}
    if QUANTIZATION in ("hybrid_mxfp8", "hybrid_mxfp8_identity"):
        return {"rtol": 0.125, "atol": 0.0625}
    if QUANTIZATION == "hybrid_nvfp4":
        # Upstream ``run_numerics.py`` uses (0.125, 0.12) for vanilla NVFP4
        # (with an open TODO to investigate why the tolerance is so large).
        return {"rtol": 0.125, "atol": 0.12}
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
    """All-gather with gradient scaling, matching
    ``run_numerics.py::_gather``. Required because
    ``torch.distributed.nn.functional.all_gather`` multiplies gradients
    by WORLD_SIZE on the backward pass — so gradients in the
    ``output_distributed`` backward would be WORLD_SIZE× too large
    compared to ``output_single_node``."""

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
    """Shard the single-node parameters into the TP-split distributed
    model. Same algorithm as ``run_numerics.py::_copy_params``: for each
    dim where shapes differ between the two params, slice the single-
    node param along that dim using ``WORLD_RANK``."""
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


def _check_outputs(output_single, output_dist, label="outputs"):
    failed = torch.tensor([0], dtype=torch.uint8, device="cuda")
    f, info = _compare_tensors(label, output_dist, output_single, **_get_tolerances())
    if f:
        dist_print(info, src=WORLD_RANK, error=True)
    failed[0] = int(f)
    dist.all_reduce(failed, dist.ReduceOp.MAX, NCCL_WORLD)
    assert not bool(failed.item()), f"{label}: numerical check failed on at least one rank"


def _check_gradients(model_dist, model_single):
    for i, ((name, pd), ps) in enumerate(
        zip(model_dist.named_parameters(), model_single.parameters())
    ):
        if pd.grad is None or ps.grad is None:
            continue
        failed = torch.tensor([0], dtype=torch.uint8, device="cuda")
        ps_grad = _match_param_sizes(pd.grad, ps.grad)
        f, info = _compare_tensors(f"grad[{i}].{name}", pd.grad, ps_grad, **_get_tolerances())
        if f:
            dist_print(info, src=WORLD_RANK, error=True)
        failed[0] = int(f)
        dist.all_reduce(failed, dist.ReduceOp.MAX, NCCL_WORLD)
        assert not bool(failed.item()), f"grad[{i}].{name}: failed on at least one rank"


def _apply_models(model_single, model_dist, inp_single, inp_dist, **kwargs):
    """Run both models under te.autocast with a fresh hybrid recipe each
    time. Both models see the same recipe instance-shape (CustomRecipe
    with the same qfactory), but get independently-constructed
    quantizers — matching how real training would instantiate them."""
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

    # Prepare inputs matching run_numerics._test_linear's conventions.
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
                # Large element on one rank: its local amax diverges from the
                # global one. Hybrid gathers the SP activation whole before
                # quantizing, so the output must still match the single-node ref.
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

    # Gradient check is only well-defined in these configurations (the
    # others need cross-rank synchronization that the test doesn't
    # perform — see run_numerics.py::_test_linear line 725 for the
    # matching gate).
    if parallel_mode == "column" or not sequence_parallel:
        _check_gradients(model_dist, model_single)


def test_linear():
    for parallel_mode in ["column", "row"]:
        for sequence_parallel in [False, True]:
            _test_linear(parallel_mode, sequence_parallel)
    # Amax corner-case (current scaling only): a large element on one rank makes
    # its local amax diverge from the global one. Hybrid gathers SP activations in
    # high precision before quantizing, so the SP output must still match
    # single-node -- guards against a future regression to quantize-then-gather
    # without cross-rank amax reduction.
    if QUANTIZATION in ("hybrid_fp8", "hybrid_fp8_identity"):
        _test_linear("column", True, amax_stress=True)


# ── Test 1b: te.Linear hybrid-vs-vanilla bitwise operand equivalence ─


def vanilla_recipe():
    """Built-in single-format recipe matching the same-format hybrid recipe
    for the bitwise ``_test_linear_vs_vanilla`` check: FP8 current scaling and
    MXFP8 use their defaults (E4M3 fwd / E5M2 bwd, and E4M3 everywhere); NVFP4
    uses the full recipe with 2D disabled to match the role-based 1D
    ``_make_nvfp4_quantizer``."""
    if QUANTIZATION == "hybrid_fp8":
        return te_recipe.Float8CurrentScaling()
    if QUANTIZATION == "hybrid_mxfp8":
        return te_recipe.MXFP8BlockScaling()
    if QUANTIZATION == "hybrid_nvfp4":
        return te_recipe.NVFP4BlockScaling(disable_2d_quantization=True)
    raise ValueError(f"No vanilla recipe for QUANTIZATION={QUANTIZATION!r}")


def _backward_not_bitwise_comparable():
    """Whether the recipe's backward can't be compared bitwise to vanilla.

    True only for NVFP4's full recipe, which combines RHT with stochastic
    rounding. That pair triggers NVFP4's separate columnwise RNG state
    (``need_separate_columnwise_rng`` in the cast backend), and the hybrid
    (two-pass) vs vanilla (fused) executions then consume that columnwise
    random stream differently. Verified by isolation: neither RHT nor SR alone
    diverges -- only the combination, and only on the columnwise gradient
    (wgrad); the rowwise gradient (dgrad) stays bitwise.
    """
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
    """Same-format hybrid must match its built-in vanilla recipe **bitwise**
    through the *same* TP/SP-distributed ``te.Linear`` (forward in all configs;
    backward in the non-SP, non-SR configs).

    Unlike ``_test_linear`` (distributed vs single-node, inherently loose),
    this compares hybrid vs vanilla at the same topology, so any non-bitwise
    difference is a real hybrid-plumbing bug. Complements the FSDP2 parity test
    in ``run_fsdp2_fused_adam.py`` by locking the TP/SP comm path.
    """
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

        with te.autocast(enabled=True, recipe=recipe):
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

    # Forward is bitwise-identical in every config (the fprop operand-
    # equivalence check: hybrid weight.rowwise/input.rowwise == vanilla).
    _check_bitwise(out_h, out_v, f"{tag} forward")

    # Backward is bitwise only without SP and without stochastic rounding;
    # both are within training tolerance and covered by the loose
    # distributed-vs-single-node check:
    #  * Under SP, hybrid has no native quantized all-gather, so it routes
    #    through the BF16 dequant/requant fallback while vanilla gathers native
    #    per-shard bytes. For per-tensor-scaled formats (FP8 current, NVFP4) the
    #    requantized scale then differs; MXFP8 (per-block only) is immune.
    #    TODO(negvet): extend to SP once native hybrid AG lands (tracked in
    #    HybridQuantizer.supports_only_rowwise_all_gather).
    #  * NVFP4's full recipe combines RHT + stochastic rounding, which triggers
    #    a separate columnwise RNG (need_separate_columnwise_rng); the hybrid
    #    two-pass and vanilla fused paths then consume that columnwise random
    #    stream differently, so the columnwise gradient (wgrad) rounds
    #    differently. Neither RHT nor SR alone diverges -- only the pair.
    if not sequence_parallel and not _backward_not_bitwise_comparable():
        _check_bitwise(dinp_h, dinp_v, f"{tag} dgrad")
        assert len(wgrads_h) == len(wgrads_v), f"{tag}: weight-grad count mismatch"
        for i, (gh, gv) in enumerate(zip(wgrads_h, wgrads_v)):
            _check_bitwise(gh, gv, f"{tag} wgrad[{i}]")


def test_linear_vs_vanilla():
    # Cross-format hybrid has no single built-in vanilla recipe to compare
    # against bitwise; it is covered by the distributed-vs-single-node checks.
    if QUANTIZATION in (
        "identity",
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
    return QUANTIZATION in ("hybrid_fp8", "hybrid_mxfp8")


def _check_same_topology_parity(
    out_h, dinp_h, model_h, out_v, dinp_v, model_v, tag, *, check_grads
):
    # Larger modules use different fused/unfused norm paths between hybrid and
    # vanilla, so numerical parity is the meaningful contract here. Linear keeps
    # the stricter bitwise check above.
    _check_outputs(out_v, out_h, label=f"{tag} forward")
    if check_grads:
        _check_outputs(dinp_v, dinp_h, label=f"{tag} dgrad")
        _check_gradients(model_h, model_v)


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
        with te.autocast(enabled=True, recipe=recipe_obj):
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
        check_grads=not sequence_parallel,
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
        with te.autocast(enabled=True, recipe=recipe_obj):
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
        check_grads=not sequence_parallel,
    )


def test_layernorm_mlp_vs_vanilla():
    for sequence_parallel in [False, True]:
        _test_layernorm_mlp_vs_vanilla(sequence_parallel)


# ── Test 2: te.LayerNormLinear column + SP ──────────────────────────


def _test_layernorm_linear(sequence_parallel, params_dtype=torch.bfloat16):
    """Column-parallel LayerNormLinear. Exercises the SP all-gather path
    that runs BEFORE quantization for hybrid (since
    ``with_quantized_norm=False`` for HybridQuantizer — see
    ``layernorm_linear.py:220``)."""
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


def test_layernorm_linear():
    for sequence_parallel in [False, True]:
        _test_layernorm_linear(sequence_parallel)


# ── Test 3: te.LayerNormMLP + TP + SP ───────────────────────────────


def _test_layernorm_mlp(sequence_parallel, params_dtype=torch.bfloat16):
    """``te.LayerNormMLP`` with ``set_parallel_mode=True`` and optional SP:
    column-parallel FC1 → activation → row-parallel FC2. Isolates the FC1
    hybrid unfused-norm path and the row-parallel FC2 + SP reduce-scatter,
    otherwise only exercised transitively inside ``_test_transformer_layer``.
    """
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

    # Without SP, grads align with the single-node ref (SP needs cross-rank
    # grad sync the test doesn't do -- matches run_numerics.py).
    if not sequence_parallel:
        _check_gradients(model_dist, model_single)


def test_layernorm_mlp():
    for sequence_parallel in [False, True]:
        _test_layernorm_mlp(sequence_parallel)


# ── Test 4: te.TransformerLayer + TP + SP ───────────────────────────


def _test_transformer_layer(sequence_parallel, params_dtype=torch.bfloat16):
    """Integration test: full TransformerLayer with TP and optional SP.
    Hits LayerNormLinear(QKV), DPA, and LayerNormMLP all with hybrid
    quantizers. If any of the unfused/hybrid code paths break something
    visible to the backward graph, this catches it with a concrete
    forward-output mismatch."""
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

    # Without SP, verify the integration path at the gradient level too (SP
    # needs cross-rank grad sync the test doesn't do -- matches run_numerics.py).
    if not sequence_parallel:
        _check_gradients(model_dist, model_single)


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
