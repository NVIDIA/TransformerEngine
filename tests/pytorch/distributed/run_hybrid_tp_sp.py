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

Test surface:
  * ``te.Linear`` column-parallel and row-parallel, with and without
    sequence parallelism.
  * ``te.LayerNormLinear`` column-parallel with sequence parallelism —
    the quantized-AG path that currently unfuses LN+quantize for
    ``HybridQuantizer``.
  * ``te.TransformerLayer`` with ``set_parallel_mode=True`` and SP on —
    integration test hitting LayerNormLinear + DPA + LayerNormMLP + row-
    parallel output projection in one shot.

Only same-format hybrid recipes (FP8 current rowwise + FP8 current
columnwise; MXFP8 rowwise + MXFP8 columnwise) are exercised here so the
numerical signal is clean. Cross-format hybrid adds independent
numerical variation unrelated to TP/SP and is covered by single-GPU
tests already.

Tolerances match upstream ``run_numerics.py`` per-format settings (see
``_get_tolerances``); they're loose enough to absorb the amax-reduction
and stochastic numerical asymmetries inherent to distributed FP8, tight
enough to catch a silent BF16 fallback on the hybrid sub-storage path.
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
    if role in ("linear_input", "linear_weight", "linear_output"):
        return HybridQuantizer(
            rowwise_quantizer=_make_fp8_current_quantizer(),
            columnwise_quantizer=_make_fp8_current_quantizer(),
        )
    if role in ("linear_grad_output", "linear_grad_input"):
        return _make_fp8_current_quantizer(fp8_dtype=tex.DType.kFloat8E5M2)
    return _make_fp8_current_quantizer()


def _hybrid_mxfp8_qfactory(role):
    if role in ("linear_input", "linear_weight", "linear_output"):
        return HybridQuantizer(
            rowwise_quantizer=_make_mxfp8_quantizer(),
            columnwise_quantizer=_make_mxfp8_quantizer(),
        )
    if role in ("linear_grad_output", "linear_grad_input"):
        return _make_mxfp8_quantizer(fp8_dtype=tex.DType.kFloat8E5M2)
    return _make_mxfp8_quantizer()


def _make_nvfp4_quantizer():
    """Default NVFP4Quantizer: no RHT, no stochastic rounding, no 2D
    scaling — matches upstream ``run_numerics.py::nvfp4_vanilla()`` which
    strips the recipe to bare 1D block scaling for distributed TP
    fairness. Same-format hybrid NVFP4 has no E5M2 variant (NVFP4 is a
    single format family — E2M1 only), so grad roles reuse the same
    NVFP4 quantizer."""
    return NVFP4Quantizer(fp4_dtype=tex.DType.kFloat4E2M1)


def _hybrid_nvfp4_qfactory(role):
    if role in ("linear_input", "linear_weight", "linear_output"):
        return HybridQuantizer(
            rowwise_quantizer=_make_nvfp4_quantizer(),
            columnwise_quantizer=_make_nvfp4_quantizer(),
        )
    if role in ("linear_grad_output", "linear_grad_input"):
        return _make_nvfp4_quantizer()
    return _make_nvfp4_quantizer()


def hybrid_recipe():
    """Fresh CustomRecipe instance per call (mirrors
    ``run_numerics.quantization_recipe`` lifetime contract)."""
    if QUANTIZATION == "hybrid_fp8":
        return te_recipe.CustomRecipe(qfactory=_hybrid_fp8_qfactory)
    if QUANTIZATION == "hybrid_mxfp8":
        return te_recipe.CustomRecipe(qfactory=_hybrid_mxfp8_qfactory)
    if QUANTIZATION == "hybrid_nvfp4":
        return te_recipe.CustomRecipe(qfactory=_hybrid_nvfp4_qfactory)
    raise ValueError(f"Unknown hybrid QUANTIZATION={QUANTIZATION!r}")


# ── Tolerances ───────────────────────────────────────────────────────
#
# Upstream ``run_numerics.py::_get_tolerances`` uses (0.4, 0.25) for
# fp8_cs (loose because of sequence parallel & amax reduction) and
# (0.125, 0.0625) for other FP8 recipes. Hybrid with same-format
# sub-quantizers should inherit the underlying format's distributed
# behaviour — with slightly looser bounds to absorb the two-pass
# quantization (rowwise and columnwise quantizers run independently, so
# their outputs may differ by ~1 ULP from a single fused-quantize path
# in edge cases).


def _get_tolerances():
    if QUANTIZATION == "hybrid_fp8":
        return {"rtol": 0.4, "atol": 0.25}
    if QUANTIZATION == "hybrid_mxfp8":
        return {"rtol": 0.2, "atol": 0.1}
    if QUANTIZATION == "hybrid_nvfp4":
        # Upstream ``run_numerics.py`` uses (0.125, 0.12) for vanilla
        # NVFP4 with an open TODO to investigate why the tolerance is so
        # large. Hybrid NVFP4 runs the same block-scaled kernel in each
        # direction independently; bump atol modestly to absorb the
        # two-pass asymmetry without hiding a real regression.
        return {"rtol": 0.2, "atol": 0.15}
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
    f, info = _compare_tensors(
        label, output_dist, output_single, **_get_tolerances()
    )
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
        f, info = _compare_tensors(
            f"grad[{i}].{name}", pd.grad, ps_grad, **_get_tolerances()
        )
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


def _test_linear(parallel_mode, sequence_parallel, params_dtype=torch.bfloat16):
    dist_print(
        f"linear: parallel_mode={parallel_mode} sequence_parallel={sequence_parallel}"
        f" dtype={params_dtype}"
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
            inp_single = torch.empty(
                (WORLD_SIZE * BATCH_SIZE, HIDDEN_SIZE)
            ).cuda().to(params_dtype)
            inp_dist = torch.randn((BATCH_SIZE, HIDDEN_SIZE)).cuda().to(params_dtype)
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
    _check_outputs(out_single, out_dist, label=f"linear[{parallel_mode},sp={sequence_parallel}]")

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


# ── Test 2: te.LayerNormLinear column + SP ──────────────────────────


def _test_layernorm_linear(sequence_parallel, params_dtype=torch.bfloat16):
    """Column-parallel LayerNormLinear. Exercises the SP all-gather path
    that runs BEFORE quantization for hybrid (since
    ``with_quantized_norm=False`` for HybridQuantizer — see
    ``layernorm_linear.py:220``)."""
    dist_print(
        f"layernorm_linear: parallel_mode=column sequence_parallel={sequence_parallel}"
    )

    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)

    model_single = te.LayerNormLinear(
        HIDDEN_SIZE, HIDDEN_SIZE, params_dtype=params_dtype
    ).cuda()
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


# ── Test 3: te.TransformerLayer + TP + SP ───────────────────────────


def _test_transformer_layer(sequence_parallel, params_dtype=torch.bfloat16):
    """Integration test: full TransformerLayer with TP and optional SP.
    Hits LayerNormLinear(QKV), DPA, and LayerNormMLP all with hybrid
    quantizers. If any of the unfused/hybrid code paths break something
    visible to the backward graph, this catches it with a concrete
    forward-output mismatch."""
    dist_print(
        f"transformer_layer: parallel_mode=set sequence_parallel={sequence_parallel}"
    )

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
        inp_dist = inp_single[
            WORLD_RANK * SEQ_LEN : (WORLD_RANK + 1) * SEQ_LEN, :, :
        ].contiguous()
    else:
        inp_dist = inp_single.clone()

    out_single, out_dist = _apply_models(model_single, model_dist, inp_single, inp_dist)

    if sequence_parallel:
        out_dist = _gather(out_dist, dim=0)

    _loss_backward(out_single, out_dist)
    _check_outputs(out_single, out_dist, label=f"transformer_layer[sp={sequence_parallel}]")


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
        choices=["hybrid_fp8", "hybrid_mxfp8", "hybrid_nvfp4"],
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "linear", "layernorm_linear", "transformer_layer"],
        help="Run only the named test (speeds up iterative debugging)",
    )
    args = parser.parse_args(argv)
    QUANTIZATION = args.quantization

    test_map = {
        "linear": test_linear,
        "layernorm_linear": test_layernorm_linear,
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
