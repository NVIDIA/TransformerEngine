# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Sharding Meta for xmap with CustomCall
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable
from jax.interpreters import pxla
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

_PXLA_THREAD_RESOURCES = pxla.thread_resources

# Axis Names
BATCH_AXES = "nvte_batch"
SEQLEN_AXES = "nvte_seqlen"
SEQLEN_TP_AXES = "nvte_seqlen_tp"
HEAD_AXES = "nvte_head"
HIDDEN_AXES = "nvte_hidden"
HIDDEN_TP_AXES = "nvte_hidden_tp"
JOINED_AXES = "nvte_joined"
W_NO_SHARD_AXES = "nvte_w_no_shard"
W_FSDP_AXES = "nvte_w_fsdp"
W_TP_AXES = "nvte_w_tp"
W_JOINED_AXES = "nvte_w_joined"


def _get_mesh_info(resource: str):
    mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
    assert resource in mesh.axis_names, f"{resource} is not in the axis_names of Mesh {mesh}."
    return mesh.shape[resource], resource


def get_sharding_map_logic_axis_to_mesh_axis():
    """
    Generate a dict to map logical axes to mesh axes.
    """
    gsr = global_mesh_resource()

    IS_FSDP_OUTER = bool(int(os.environ.get("NVTE_OUTER_BATCH_FSDP_DIM", False)))

    batch_resources = (
        [gsr.fsdp_resource, gsr.dp_resource]
        if IS_FSDP_OUTER
        else [gsr.dp_resource, gsr.fsdp_resource]
    )

    batch_dim_rule = []
    for resource in batch_resources:
        if resource is not None and resource not in batch_dim_rule:
            batch_dim_rule.append(resource)

    if len(batch_dim_rule) <= 0:
        batch_dim_rule = None
    elif len(batch_dim_rule) == 1:
        batch_dim_rule = batch_dim_rule[0]
    else:
        batch_dim_rule = tuple(batch_dim_rule)

    te_logical_axis_to_mesh_axis = {
        BATCH_AXES: batch_dim_rule,
        SEQLEN_AXES: None,
        SEQLEN_TP_AXES: gsr.tp_resource,
        HEAD_AXES: gsr.tp_resource,
        HIDDEN_AXES: None,
        HIDDEN_TP_AXES: gsr.tp_resource,
        JOINED_AXES: None,
        W_NO_SHARD_AXES: None,
        W_FSDP_AXES: gsr.fsdp_resource,
        W_TP_AXES: gsr.tp_resource,
        W_JOINED_AXES: None,
    }
    return te_logical_axis_to_mesh_axis


def generate_pspec(logical_axis_names):
    """
    Convert logical axes to PartitionSpec
    """
    rules = get_sharding_map_logic_axis_to_mesh_axis()
    mesh_axis_names = [rules[name] for name in logical_axis_names]
    pspec = jax.sharding.PartitionSpec(*mesh_axis_names)
    return pspec


def with_sharding_constraint(x: jnp.array, pspec: PartitionSpec):
    """
    A wrapper function to jax.lax.with_sharding_constraint to
    support the case that Mesh is empty.
    """
    if pspec is None:
        return x

    mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
    if mesh.empty:
        return x
    return jax.lax.with_sharding_constraint(x, pspec)


def with_sharding_constraint_by_logical_axes(x: jnp.array, logical_axis_names: tuple | list):
    """
    A wrapper function to jax.lax.with_sharding_constraint to accept logical axes.
    """
    if logical_axis_names is None:
        return x

    assert len(x.shape) == len(logical_axis_names)
    pspec = generate_pspec(logical_axis_names)
    return with_sharding_constraint(x, pspec)


def get_all_mesh_axes():
    """
    Get all name of mesh axes
    """
    mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
    return mesh.axis_names


def get_padded_spec(spec, ndim):
    """
    Get padded spec for partitioning from arguments' information
    """
    if spec is None:
        return (None,) * ndim
    assert len(spec) <= ndim
    return spec + (None,) * (ndim - len(spec))


def lax_paral_op(x: jnp.array, ops: Callable, mesh_resource: str):
    """
    A wrapper function to invoke lax.p* operations, like psum.
    """
    if mesh_resource is not None:
        _, resource = _get_mesh_info(mesh_resource)
        return ops(x, resource)
    return x


def num_of_devices():
    """
    Get total number of detected devices
    """
    return len(jax.devices())


@dataclass
class MeshResource:
    """
    A data container to indicate which axis in Mesh for data parallelism and
    which for tensor parallelism.

    Parameters
    ----------
    dp_resource : str, default = None
        The axis name in Mesh used to shard batches along.
        If it is None, then data parallelism is disabled.
    tp_resource : str, default = None
        The axis name in Mesh used to split the hidden dimensions along.
        If it is None, then tensor parallelism is disabled.
    fsdp_resource : str, default = None
        The axis name in Mesh used to split the batch and weights along.
        If it is None, then full-sharded data parallelism is disabled.
    pp_resource : str, default = None
        The axis name in Mesh used to split model layers. along.
        If it is None, then pipeline parallelism is disabled.
    """

    dp_resource: str = None
    tp_resource: str = None
    fsdp_resource: str = None
    pp_resource: str = None


_GLOBAL_MESH_RESOURCE = MeshResource()


@contextmanager
def global_shard_guard(resource: MeshResource):
    """
    A context manager to switch the global MeshResource
    """
    global _GLOBAL_MESH_RESOURCE
    prev_gmr = _GLOBAL_MESH_RESOURCE
    try:
        _GLOBAL_MESH_RESOURCE = resource
        yield
    finally:
        _GLOBAL_MESH_RESOURCE = prev_gmr


def global_mesh_resource() -> MeshResource:
    """
    A getter of the global MeshResource
    """
    return _GLOBAL_MESH_RESOURCE


def all_reduce_sum_along_dp_fsdp(x: jnp.array):
    """
    All-Reduce (Sum) along DP and FSDP mesh axes.
    """
    x = lax_paral_op(x, jax.lax.psum, global_mesh_resource().dp_resource)
    return lax_paral_op(x, jax.lax.psum, global_mesh_resource().fsdp_resource)


def all_reduce_max_along_all_axes_except_PP(x: jnp.array):
    """
    All-Reduce (Max) along all mesh axes.
    """
    all_axes = get_all_mesh_axes()
    for axis in all_axes:
        if axis != global_mesh_resource().pp_resource:
            x = lax_paral_op(x, jax.lax.pmax, axis)
    return x


# Deprecating Items ---------------------------------------------------------------
ShardingResource = MeshResource

global_shard_resource = global_mesh_resource


class MajorShardingType(Enum):
    r"""
    The major sharding type to indicate sharding pattern.
    .. warning::
        MajorShardingType is deprecating in the near feature.

    Values
    ----------
    SINGLE:
        Single process training.
    DP:
        Data parallel training.
    TP:
        Standard tensor parallel training.
    DPTP:
        Data and Standard tensor parallel training.
    """

    SINGLE = 0
    DP = 1
    TP = 2
    DPTP = 3


class ShardingType(Enum):
    """
    The sharding type to indicate sharding pattern.
    .. warning::
        ShardingType is deprecating in the near feature.

    Values
    ----------
    SINGLE:
        No sharding.
    DP:
        Sharding along data parallelism.
    TP_COL:
        Sharding along column-split tensor parallelism.
    TP_ROW:
        Sharding along row-split tensor parallelism.
    DP_TP_COL:
        Sharding along data and column-split tensor parallelism.
    DP_TP_ROW:
        Sharding along data and row-split tensor parallelism.
    """

    SINGLE = (MajorShardingType.SINGLE, "single")
    DP = (MajorShardingType.DP, "dp")
    TP_COL = (MajorShardingType.TP, "tp_col")
    TP_ROW = (MajorShardingType.TP, "tp_row")
    DP_TP_COL = (MajorShardingType.DPTP, "dp_tp_col")
    DP_TP_ROW = (MajorShardingType.DPTP, "dp_tp_row")
