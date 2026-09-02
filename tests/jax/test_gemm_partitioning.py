# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unit tests for the GEMM custom-partitioning spec inference.

These tests exercise ``GemmPrimitive._parse_operand_output_specs`` directly on
sharding specs, covering the FWD/DGRAD/WGRAD GEMMs of an MoE FFN block. They run
on CPU and do not require GPUs.

Cases come in two self-consistent parallelism configs where every backward output
sharding matches its forward input (dX~X, dW~W, dH~H):

  Megatron TP (tp shards feature dims, token dim is (fsdp, expert)):
    X=((fsdp,expert),None), W1=(None,tp), H=Y1=((fsdp,expert),tp), W2=(tp,None).
    - FWD/DGRAD contracting the tp-sharded ffn dim reduce over tp.
    - WGRAD contracting the token dim reduces over (fsdp, expert).

  TPSP (tpsp shards the token dim, weights replicated):
    X=H=Y=((fsdp,tpsp,expert),None), W1=W2=(None,None).
    - FWD/DGRAD contract replicated dims, no reduction.
    - WGRAD contracting the nested token dim reduces over (fsdp, tpsp, expert).
"""

import os

if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (
        os.environ.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8"
    ).strip()

from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pytest
import jax
from jax.sharding import Mesh, PartitionSpec

from transformer_engine.jax.cpp_extensions.gemm import CollectiveOp, GemmPrimitive
from transformer_engine.jax.quantize import ScalingMode
from transformer_engine.jax.sharding import MeshResource, global_shard_guard

pytestmark = pytest.mark.skipif(len(jax.devices("cpu")) < 8, reason="requires 8 CPU devices")


def _mesh(axes):
    """Build a CPU mesh with a size-2 device grid over the named axes."""
    devices = np.asarray(jax.devices("cpu")[: 2 ** len(axes)]).reshape((2,) * len(axes))
    return Mesh(devices, axes)


def _arg_info(shape, spec):
    """Minimal arg_info stub exposing ndim, size and sharding.spec."""
    sharding = None if spec is None else SimpleNamespace(spec=PartitionSpec(*spec))
    return SimpleNamespace(ndim=len(shape), size=int(np.prod(shape)), sharding=sharding)


def _parse(lhs_shape, lhs_spec, rhs_shape, rhs_spec, contracting_dims):
    """Run the partition spec inference for a plain (non-collective) GEMM."""
    scalar = _arg_info((0,), None)  # scales / bias / alpha / beta (unused for NO_SCALING)
    arg_infos = (
        _arg_info(lhs_shape, lhs_spec),
        scalar,
        _arg_info(rhs_shape, rhs_spec),
        scalar,
        scalar,
        scalar,
        scalar,
    )
    operand_specs, out_specs, reduce_spec, _ = GemmPrimitive._parse_operand_output_specs(
        arg_infos,
        contracting_dims,
        transpose_batch_sequence=False,
        collective_op=CollectiveOp.NONE,
        scaling_mode=ScalingMode.NO_SCALING,
    )
    lhs_specs, _, rhs_specs, *_ = operand_specs
    return lhs_specs, rhs_specs, tuple(out_specs), reduce_spec


# Representative MoE FFN GEMMs (FFN1: hidden->ffn, FFN2: ffn->hidden) for the two
# parallelism configs described in the module docstring. Expected: reduce over the mesh
# axes that shard the contracting dim of both operands, gather any axis that shards it on
# only one operand.
Case = namedtuple(
    "Case",
    "axes, resource, lhs_shape, lhs_spec, rhs_shape, rhs_spec, cdims,"
    " exp_lhs, exp_rhs, exp_out, exp_reduce",
)

_MR_TP = dict(fsdp_resource="fsdp", tp_resource="tp", ep_resource="expert")
_MR_TPSP = dict(fsdp_resource="fsdp", tpsp_resource="tpsp", ep_resource="expert")

_TP_AXES = ("fsdp", "tp", "expert")
_TPSP_AXES = ("fsdp", "tpsp", "expert")

CASES = {
    # --- Megatron TP: tp shards feature dims, token dim is (fsdp, expert). ---
    # FFN1 FWD: Y1 = X @ W1, contract the replicated hidden dim -> no reduction.
    "megatron_ffn1_fwd": Case(
        _TP_AXES,
        _MR_TP,
        (524288, 7168),
        (("fsdp", "expert"), None),
        (7168, 2048),
        (None, "tp"),
        ((1,), (0,)),
        (("fsdp", "expert"), None),
        (None, "tp"),
        (("fsdp", "expert"), "tp"),
        None,
    ),
    # FFN1 DGRAD: dX = dY1 @ W1^T, contract the tp-sharded ffn dim -> reduce over tp.
    "megatron_ffn1_dgrad": Case(
        _TP_AXES,
        _MR_TP,
        (524288, 2048),
        (("fsdp", "expert"), "tp"),
        (7168, 2048),
        (None, "tp"),
        ((1,), (1,)),
        (("fsdp", "expert"), "tp"),
        (None, "tp"),
        (("fsdp", "expert"), None),
        "tp",
    ),
    # FFN1 WGRAD: dW1 = X^T @ dY1, contract the token dim -> reduce over (fsdp, expert).
    "megatron_ffn1_wgrad": Case(
        _TP_AXES,
        _MR_TP,
        (7168, 524288),
        (None, ("fsdp", "expert")),
        (2048, 524288),
        ("tp", ("fsdp", "expert")),
        ((1,), (1,)),
        (None, ("fsdp", "expert")),
        ("tp", ("fsdp", "expert")),
        (None, "tp"),
        ("fsdp", "expert"),
    ),
    # FFN2 FWD: Y2 = H @ W2, contract the tp-sharded ffn dim -> reduce over tp.
    "megatron_ffn2_fwd": Case(
        _TP_AXES,
        _MR_TP,
        (524288, 2048),
        (("fsdp", "expert"), "tp"),
        (2048, 7168),
        ("tp", None),
        ((1,), (0,)),
        (("fsdp", "expert"), "tp"),
        ("tp", None),
        (("fsdp", "expert"), None),
        "tp",
    ),
    # FFN2 DGRAD: dH = dY2 @ W2^T, contract the replicated hidden dim -> no reduction.
    "megatron_ffn2_dgrad": Case(
        _TP_AXES,
        _MR_TP,
        (524288, 7168),
        (("fsdp", "expert"), None),
        (2048, 7168),
        ("tp", None),
        ((1,), (1,)),
        (("fsdp", "expert"), None),
        ("tp", None),
        (("fsdp", "expert"), "tp"),
        None,
    ),
    # FFN2 WGRAD: dW2 = H^T @ dY2, contract the token dim -> reduce over (fsdp, expert).
    "megatron_ffn2_wgrad": Case(
        _TP_AXES,
        _MR_TP,
        (2048, 524288),
        ("tp", ("fsdp", "expert")),
        (7168, 524288),
        (None, ("fsdp", "expert")),
        ((1,), (1,)),
        ("tp", ("fsdp", "expert")),
        (None, ("fsdp", "expert")),
        ("tp", None),
        ("fsdp", "expert"),
    ),
    # --- Sequence-parallel TP: tpsp shards the token dim, weights replicated. ---
    # FFN1 FWD: Y1 = X @ W1, contract the replicated hidden dim -> no reduction.
    "tpsp_ffn1_fwd": Case(
        _TPSP_AXES,
        _MR_TPSP,
        (524288, 7168),
        (("fsdp", "tpsp", "expert"), None),
        (7168, 2048),
        (None, None),
        ((1,), (0,)),
        (("fsdp", "tpsp", "expert"), None),
        (None, None),
        (("fsdp", "tpsp", "expert"), None),
        None,
    ),
    # FFN1 DGRAD: dX = dY1 @ W1^T, contract the replicated ffn dim -> no reduction.
    "tpsp_ffn1_dgrad": Case(
        _TPSP_AXES,
        _MR_TPSP,
        (524288, 2048),
        (("fsdp", "tpsp", "expert"), None),
        (7168, 2048),
        (None, None),
        ((1,), (1,)),
        (("fsdp", "tpsp", "expert"), None),
        (None, None),
        (("fsdp", "tpsp", "expert"), None),
        None,
    ),
    # FFN1 WGRAD: dW1 = X^T @ dY1, contract the nested token dim -> reduce (fsdp, tpsp, expert).
    "tpsp_ffn1_wgrad": Case(
        _TPSP_AXES,
        _MR_TPSP,
        (7168, 524288),
        (None, ("fsdp", "tpsp", "expert")),
        (2048, 524288),
        (None, ("fsdp", "tpsp", "expert")),
        ((1,), (1,)),
        (None, ("fsdp", "tpsp", "expert")),
        (None, ("fsdp", "tpsp", "expert")),
        (None, None),
        ("fsdp", "tpsp", "expert"),
    ),
    # FFN2 FWD: Y2 = H @ W2, contract the replicated ffn dim -> no reduction.
    "tpsp_ffn2_fwd": Case(
        _TPSP_AXES,
        _MR_TPSP,
        (524288, 2048),
        (("fsdp", "tpsp", "expert"), None),
        (2048, 7168),
        (None, None),
        ((1,), (0,)),
        (("fsdp", "tpsp", "expert"), None),
        (None, None),
        (("fsdp", "tpsp", "expert"), None),
        None,
    ),
    # FFN2 DGRAD: dH = dY2 @ W2^T, contract the replicated hidden dim -> no reduction.
    "tpsp_ffn2_dgrad": Case(
        _TPSP_AXES,
        _MR_TPSP,
        (524288, 7168),
        (("fsdp", "tpsp", "expert"), None),
        (2048, 7168),
        (None, None),
        ((1,), (1,)),
        (("fsdp", "tpsp", "expert"), None),
        (None, None),
        (("fsdp", "tpsp", "expert"), None),
        None,
    ),
    # FFN2 WGRAD: dW2 = H^T @ dY2, contract the nested token dim -> reduce (fsdp, tpsp, expert).
    "tpsp_ffn2_wgrad": Case(
        _TPSP_AXES,
        _MR_TPSP,
        (2048, 524288),
        (None, ("fsdp", "tpsp", "expert")),
        (7168, 524288),
        (None, ("fsdp", "tpsp", "expert")),
        ((1,), (1,)),
        (None, ("fsdp", "tpsp", "expert")),
        (None, ("fsdp", "tpsp", "expert")),
        (None, None),
        ("fsdp", "tpsp", "expert"),
    ),
    # Single-axis contracting shared by both operands (backward compat).
    "single_axis": Case(
        ("fsdp",),
        dict(fsdp_resource="fsdp"),
        (128, 512),
        (None, "fsdp"),
        (256, 512),
        (None, "fsdp"),
        ((1,), (1,)),
        (None, "fsdp"),
        (None, "fsdp"),
        (None, None),
        "fsdp",
    ),
    # No shared contracting axis -> gather both, no reduction.
    "no_shared_axis": Case(
        ("fsdp", "tp"),
        dict(fsdp_resource="fsdp", tp_resource="tp"),
        (128, 512),
        ("fsdp", None),
        (256, 512),
        ("tp", None),
        ((1,), (1,)),
        ("fsdp", None),
        ("tp", None),
        ("fsdp", "tp"),
        None,
    ),
}


class TestGemmPartitioning:
    """Spec inference for the plain GEMM partition rule."""

    @pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
    def test_partition_specs(self, case):
        with _mesh(case.axes), global_shard_guard(MeshResource(**case.resource)):
            lhs_specs, rhs_specs, out_specs, reduce_spec = _parse(
                case.lhs_shape, case.lhs_spec, case.rhs_shape, case.rhs_spec, case.cdims
            )
        assert lhs_specs == case.exp_lhs
        assert rhs_specs == case.exp_rhs
        assert out_specs == case.exp_out
        assert reduce_spec == case.exp_reduce


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
