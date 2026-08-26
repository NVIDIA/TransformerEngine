# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unit tests for the GEMM custom-partitioning spec inference.

These tests exercise ``GemmPrimitive._parse_operand_output_specs`` directly on
sharding specs, covering the FWD/DGRAD/WGRAD GEMMs of an MoE FFN block. They run
on CPU and do not require GPUs.
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
    (operand_specs, out_specs, reduce_spec, _) = GemmPrimitive._parse_operand_output_specs(
        arg_infos,
        contracting_dims,
        transpose_batch_sequence=False,
        collective_op=CollectiveOp.NONE,
        scaling_mode=ScalingMode.NO_SCALING,
    )
    lhs_specs, _, rhs_specs, *_ = operand_specs
    return lhs_specs, rhs_specs, tuple(out_specs), reduce_spec


# Representative MoE FFN GEMMs on a fsdp x tp x expert mesh. Hidden dims are sharded
# over tp; the token dim is sharded over (fsdp, expert) and may additionally carry tp
# on only one operand (the two WGrad orientations seen in a real 256-GPU HLO). Expected:
# gather the odd tp axis on whichever operand carries it, then reduce over the axes that
# shard the contracting dim of both operands.
Case = namedtuple(
    "Case",
    "axes, resource, lhs_shape, lhs_spec, rhs_shape, rhs_spec, cdims,"
    " exp_lhs, exp_rhs, exp_out, exp_reduce",
)

_MR_TP = dict(fsdp_resource="fsdp", tp_resource="tp", ep_resource="expert")
_MR_TPSP = dict(fsdp_resource="fsdp", tpsp_resource="tp", ep_resource="expert")

CASES = {
    # WGrad, tp leaked onto X's token dim only -> gather tp on X, reduce (fsdp, expert).
    "wgrad_nested_tp_on_x": Case(
        ("fsdp", "tp", "expert"),
        _MR_TP,
        (7168, 524288),
        (None, ("fsdp", "tp", "expert")),
        (256, 524288),
        ("tp", ("fsdp", "expert")),
        ((1,), (1,)),
        (None, ("fsdp", "expert")),
        ("tp", ("fsdp", "expert")),
        (None, "tp"),
        ("fsdp", "expert"),
    ),
    # Mirror orientation: tp leaked onto dY's token dim only -> gather tp on dY.
    "wgrad_mirror_tp_on_dy": Case(
        ("fsdp", "tp", "expert"),
        _MR_TP,
        (7168, 524288),
        (None, ("fsdp", "expert")),
        (256, 524288),
        ("tp", ("fsdp", "tp", "expert")),
        ((1,), (1,)),
        (None, ("fsdp", "expert")),
        ("tp", ("fsdp", "expert")),
        (None, "tp"),
        ("fsdp", "expert"),
    ),
    # Forward: contract the tp-sharded hidden dim -> reduce over tp only.
    "fwd_reduce_over_tp": Case(
        ("fsdp", "tp", "expert"),
        _MR_TPSP,
        (524288, 7168),
        (("fsdp", "expert"), "tp"),
        (7168, 2048),
        ("tp", None),
        ((1,), (0,)),
        (("fsdp", "expert"), "tp"),
        ("tp", None),
        (("fsdp", "expert"), None),
        "tp",
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
