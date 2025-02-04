# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pytest
from functools import partial
from collections.abc import Iterable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.experimental import mesh_utils

import transformer_engine.jax as te
from transformer_engine.jax.gemm import gemm

from utils import assert_allclose


jax.config.update("jax_enable_compilation_cache", False)


# AG+GEMM: (4, 32/P, 128) ----(AG)----> (4, 32, 128) x (128, 256/P) ----------> (4, 32, 256/P)
# - DGRAD:                            (4, 32, 256/P) x (128, 256/P)^T --(AR)--> (4, 32, 128)
# - WGRAD: (4, 32/P, 128)^T --(AG)--> (4, 32, 128)^T x (4, 32, 256/P) --------> (128, 256/P)

# GEMM+AR:                            (4, 32, 256/P) x (256/P, 128) --(AR)--> (4, 32, 128)
# - DGRAD:                              (4, 32, 128) x (256/P, 128)^T ------> (4, 32, 256/P)
# - WGRAD: (4, 32, 256/P)^T --(AG)--> (4, 32, 256)^T x (4, 32, 128) --------> (256, 128)

BATCH = 4
BASE_SIZE = 16
SEQ_LEN = BASE_SIZE * 8
HIDDEN_SIZE = BASE_SIZE * 6
FFN_HIDDEN_SIZE = BASE_SIZE * 16

COMM_TYPES = ["ALL_GATHER", "ALL_REDUCE"]
MESH_TYPES = ["FSDP_TP", "DP_TP", "TP"]
NUM_DEVICES = 4

is_fp8_supported, no_fp8_reason = te.fp8.is_fp8_available()


def _get_mesh(parallel_dist):
    jax.clear_caches()

    batched = False
    fsdp = False
    mesh_shape = dict(tp=NUM_DEVICES)
    resources = dict(cp_resource="tp", tp_resource="tp")
    if parallel_dist in ["DP_TP", "FSDP_TP"]:
        batched = True
        mesh_shape.update(dict(tp=NUM_DEVICES // 2, dp=NUM_DEVICES // 2))
        resources.update(dict(dp_resource="dp"))
        if parallel_dist == "FSDP_TP":
            fsdp = True
            mesh_shape.update(dict(tp=NUM_DEVICES // 2, dp=1, zp=NUM_DEVICES // 2))
            resources.update(dict(fsdp_resource="zp"))
    mesh_resource = te.MeshResource(**resources)

    devices = mesh_utils.create_device_mesh((NUM_DEVICES,), devices=jax.devices()[:NUM_DEVICES])

    mesh = Mesh(np.array(devices).reshape(tuple(mesh_shape.values())), tuple(mesh_shape.keys()))

    return mesh, mesh_resource, batched, fsdp


def _get_inputs(mesh, mesh_resource, dtype, fwd_comm_type, batched, fsdp, fwd_bwd=False):
    fp8_gemm = dtype in [jnp.float8_e4m3fn, jnp.float8_e5m2]

    # Operand and output shapes
    lhs_shape = (
        [SEQ_LEN, HIDDEN_SIZE] if fwd_comm_type == "ALL_GATHER" else [SEQ_LEN, FFN_HIDDEN_SIZE]
    )
    rhs_shape = (
        [HIDDEN_SIZE, FFN_HIDDEN_SIZE]
        if fwd_comm_type == "ALL_GATHER"
        else [FFN_HIDDEN_SIZE, HIDDEN_SIZE]
    )
    out_shape = [lhs_shape[0], rhs_shape[1]]

    if batched:
        lhs_shape = [BATCH] + lhs_shape
        out_shape = [BATCH] + out_shape

    # Operand and output partition specs
    lhs_spec = (
        [mesh_resource.tp_resource, None]
        if fwd_comm_type == "ALL_GATHER"
        else [None, mesh_resource.tp_resource]
    )
    rhs_spec = (
        [None, mesh_resource.tp_resource]
        if fwd_comm_type == "ALL_GATHER"
        else [mesh_resource.tp_resource, None]
    )
    out_spec = [None, rhs_spec[-1]]

    # Modify RHS operand for FP8
    fsdp_gathered_rhs_spec = rhs_spec.copy()
    if fp8_gemm:
        rhs_shape = list(reversed(rhs_shape))
        rhs_spec = list(reversed(rhs_spec))
        fsdp_gathered_rhs_spec = list(reversed(fsdp_gathered_rhs_spec))

    # Add batch dimensions and specs
    if batched:
        if fsdp:
            lhs_spec = [(mesh_resource.dp_resource, mesh_resource.fsdp_resource)] + lhs_spec
            rhs_spec = [mesh_resource.fsdp_resource if spec is None else spec for spec in rhs_spec]
            out_spec = [(mesh_resource.dp_resource, mesh_resource.fsdp_resource)] + out_spec
        else:
            lhs_spec = [mesh_resource.dp_resource] + lhs_spec
            out_spec = [mesh_resource.dp_resource] + out_spec

    # Allocate global operands on device
    key = jax.random.PRNGKey(42)
    split_keys = jax.random.split(key, 3 if fwd_bwd else 2)
    mu = 0.0
    sigma = 0.023
    shapes = (lhs_shape, rhs_shape)
    if fwd_bwd:
        shapes += (out_shape,)
    global_operands = list(
        map(
            lambda key, shape: jax.device_put(
                mu + (sigma * jax.random.normal(key, shape, dtype=dtype)),
                NamedSharding(mesh, PartitionSpec(None)),
            ),
            split_keys,
            shapes,
        )
    )

    # Allocate sharded operands on device
    partition_axes = (lhs_spec, rhs_spec)
    if fwd_bwd:
        partition_axes += (out_spec,)
    local_operands = list(
        map(
            lambda x, spec: jax.device_put(x, NamedSharding(mesh, PartitionSpec(*spec))),
            global_operands,
            partition_axes,
        )
    )

    # Tranpose global RHS back to non-transpoosed orientation if it was originally allocated
    # for FP8 GEMM
    if fp8_gemm:
        rhs_global = jnp.matrix_transpose(global_operands[1])
        global_operands = (global_operands[0], rhs_global, *global_operands[2:])

    return (
        local_operands,
        global_operands,
        (out_shape, out_spec),
        fsdp_gathered_rhs_spec,
    )


def _check_output(mesh, expected_out_shape, expected_out_specs, *tensors, fwd_bwd=False):
    num_operands = 3 if fwd_bwd else 2
    ref_operands = tensors[:num_operands]
    test_outputs = tensors[num_operands:]

    # Check number of dimensions
    assert test_outputs[0].ndim == len(expected_out_shape), (
        f"Output has different number of dimensions ({test_outputs[0].ndim}) than expected "
        + f"({len(expected_out_shape)})"
    )

    # Pad test output spec for unsharded dimensions
    test_spec = te.sharding.get_padded_spec(test_outputs[0].sharding.spec, test_outputs[0].ndim)

    for i in range(test_outputs[0].ndim):
        # Check shape
        assert test_outputs[0].shape[i] == expected_out_shape[i], (
            f"Output with shape {test_outputs[0].shape} does not match expected shape "
            + f"{expected_out_shape} in dimension index {i}."
        )

        # Check shardings (with padded output spec)
        spec_mismatch = False
        if isinstance(expected_out_specs[i], str):
            if test_spec[i] != expected_out_specs[i]:
                spec_mismatch = True
        elif isinstance(expected_out_specs[i], Iterable):
            if not isinstance(test_spec[i], type(expected_out_specs[i])):
                if test_spec[i] not in expected_out_specs[i]:
                    spec_mismatch = True
            elif len(test_spec[i]) != len(expected_out_specs[i]):
                spec_mismatch = True
            else:
                for j in range(len(expected_out_specs[i])):
                    if test_spec[i][j] != expected_out_specs[i][j]:
                        spec_mismatch = True
                        break
        elif expected_out_specs[i] == None:
            if test_spec[i] != None:
                spec_mismatch = True
        else:
            raise RuntimeError("Internal TE error: Unrecognized reference partition spec type.")
        if spec_mismatch:
            raise AssertionError(
                f"Output sharding {test_spec} does not match expected sharding "
                + f"{expected_out_specs} in dimension index {i}."
            )

    def _native_gemm_fwd_bwd(lhs, rhs, grad):
        fwd_out, vjp_fn = jax.vjp(jnp.dot, lhs, rhs)
        lhs_grad, rhs_grad = vjp_fn(grad)
        return fwd_out, lhs_grad, rhs_grad

    ref_fn = jax.jit(_native_gemm_fwd_bwd if fwd_bwd else jnp.dot)

    out_names = ["output"]
    ref_outputs = ref_fn(*ref_operands)
    if not fwd_bwd:
        ref_outputs = [ref_outputs]
    else:
        out_names += ["dgrad", "wgrad"]

    for i, (test_out, ref_out) in enumerate(zip(test_outputs, ref_outputs)):
        test_out_global = jax.lax.with_sharding_constraint(
            test_out, NamedSharding(mesh, PartitionSpec(None))
        )
        try:
            assert_allclose(ref_out, test_out_global)
        except AssertionError as err:
            raise AssertionError(f"Numerical mismatch in {out_names[i]}:\n" + str(err))


@pytest.mark.parametrize("comm_type", COMM_TYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_gemm_impl(comm_type, mesh_type):
    mesh, mesh_resource, batched, fsdp = _get_mesh(mesh_type)

    (
        local_operands,
        global_operands,
        output_info,
        fsdp_gathered_rhs_spec,
    ) = _get_inputs(mesh, mesh_resource, jnp.bfloat16, comm_type, batched, fsdp)

    @jax.jit
    def _test_fn(lhs, rhs):
        rhs_no_fsdp = jax.lax.with_sharding_constraint(
            rhs, NamedSharding(mesh, PartitionSpec(*fsdp_gathered_rhs_spec))
        )
        return te.cpp_extensions.gemm_impl(lhs, rhs_no_fsdp, batched_output=batched)

    with te.sharding.global_shard_guard(mesh_resource):
        output, *_ = _test_fn(*local_operands)

    _check_output(mesh, *output_info, *global_operands, output)


@pytest.mark.parametrize("comm_type", COMM_TYPES)
@pytest.mark.parametrize("mesh_type", MESH_TYPES)
def test_gemm_fwd_bwd(comm_type, mesh_type):
    mesh, mesh_resource, batched, fsdp = _get_mesh(mesh_type)

    (
        local_operands,
        global_operands,
        output_info,
        fsdp_gathered_rhs_spec,
    ) = _get_inputs(mesh, mesh_resource, jnp.bfloat16, comm_type, batched, fsdp, fwd_bwd=True)

    @jax.jit
    def _test_fn(lhs, rhs, grad):
        # Gather weights in FSDP axis
        rhs_no_fsdp = jax.lax.with_sharding_constraint(
            rhs, NamedSharding(mesh, PartitionSpec(*fsdp_gathered_rhs_spec))
        )

        # FWD pass
        fwd_out, vjp_fn = jax.vjp(gemm, lhs, rhs_no_fsdp)

        # BWD pass
        lhs_grad, rhs_grad = vjp_fn(grad)

        return fwd_out, lhs_grad, rhs_grad

    print(
        f"INPUTS: {local_operands[0].shape} x {local_operands[1].shape}\n"
        + f"    LHS sharding: {local_operands[0].sharding.spec}\n"
        + f"    RHS sharding: {local_operands[1].sharding.spec}\n"
    )

    with te.sharding.global_shard_guard(mesh_resource):
        output, dgrad, wgrad = _test_fn(*local_operands)

    print(
        f"{'AG + GEMM' if comm_type == 'AG' else 'GEMM + AR'} output: "
        + f"{output.shape} | {output.sharding.spec}\n"
        + f"DGRAD: {dgrad.shape} | {dgrad.sharding.spec}\n"
        + f"WGRAD: {wgrad.shape} | {wgrad.sharding.spec}\n"
    )

    _check_output(mesh, *output_info, *global_operands, output, dgrad, wgrad, fwd_bwd=True)
