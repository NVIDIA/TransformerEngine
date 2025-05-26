# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Sharding utilities for Transformer Engine in JAX.

This module provides utilities for managing tensor sharding in distributed training,
including support for various parallelism strategies like data parallelism (DP),
tensor parallelism (TP), pipeline parallelism (PP), and full-sharded data
parallelism (FSDP). It includes functions for sharding constraints, mesh management,
and collective operations.
"""
import os
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional
from jax.interpreters import pxla
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec

_PXLA_THREAD_RESOURCES = pxla.thread_resources

# Axis Names
BATCH_AXES = "nvte_batch"
SEQLEN_AXES = "nvte_seqlen"
SEQLEN_TP_AXES = "nvte_seqlen_tp"
SEQLEN_CP_AXES = "nvte_seqlen_cp"
HEAD_AXES = "nvte_head"
HIDDEN_AXES = "nvte_hidden"
HIDDEN_TP_AXES = "nvte_hidden_tp"
JOINED_AXES = "nvte_joined"
W_NO_SHARD_AXES = "nvte_w_no_shard"
W_FSDP_AXES = "nvte_w_fsdp"
W_TP_AXES = "nvte_w_tp"
W_JOINED_AXES = "nvte_w_joined"


def _get_mesh_info(resource: str, mesh: jax.sharding.Mesh):
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
        SEQLEN_CP_AXES: gsr.cp_resource,
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
    # mesh_axis_names = [rules[name] for name in logical_axis_names]
    mesh_axis_names = []
    for name in logical_axis_names:
        axis_name = rules[name] if name in rules else None
        mesh_axis_names.append(axis_name)
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


def with_sharding_constraint_by_logical_axes(
    x: jnp.array, logical_axis_names: Optional[tuple | list]
):
    """
    A wrapper function to jax.lax.with_sharding_constraint to accept logical axes.

    If logical_axis_names = None, this means no sharding constraint is applied.

    If logical_axis_names = (None, None, ...), this means a sharding constraint is applied and the tensor is replicated across all devices.

    Args:
        x: Input tensor to apply sharding constraint
        logical_axis_names: Logical axis names to apply sharding constraint
    Returns:
        Tensor with sharding constraint applied, or the original tensor if no logical axes are provided.

    """
    if not logical_axis_names:
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


def lax_paral_op(
    x: jnp.array, ops: Callable, mesh_resource: str, mesh: jax.sharding.Mesh, **kwargs
):
    """
    A wrapper function to invoke lax.p* operations, like psum.
    """
    if mesh_resource is not None:
        _, resource = _get_mesh_info(mesh_resource, mesh)
        return ops(x, resource, **kwargs)
    return x


def num_of_devices():
    """
    Get total number of detected devices
    """
    return len(jax.devices())


def get_mesh_axis_size(axis, mesh=None):
    """
    Get the axis size of the given mesh.
    If the mesh is None, it would be replaced
    by the global mesh.
    """
    if mesh is None:
        mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh

    if axis is None:
        return 1

    assert axis in mesh.shape, f"{axis} is not a axis of the given mesh {mesh.shape}"
    return mesh.shape[axis]


def get_mesh_axis_rank(axis: str, mesh=None):
    """
    Gets the local axis rank of the `axis` of the array.
    If the mesh is None the rank is 0.
    """
    if mesh is None:
        return 0
    _, axis_name = _get_mesh_info(axis, mesh)
    return jax.lax.axis_index(axis_name)


@dataclass
class MeshResource:
    """A data container for managing mesh resources in distributed training.

    This class defines the mapping between logical axes and physical mesh axes
    for different types of parallelism in distributed training.

    Attributes:
        dp_resource: Axis name for data parallelism (batch sharding), default is None
        tp_resource: Axis name for tensor parallelism (hidden dimension sharding), default is None
        fsdp_resource: Axis name for full-sharded data parallelism, default is None
        pp_resource: Axis name for pipeline parallelism (layer sharding), default is None
        cp_resource: Axis name for context parallelism (sequence sharding), default is None
    """

    dp_resource: str = None
    tp_resource: str = None
    fsdp_resource: str = None
    pp_resource: str = None
    cp_resource: str = None


_GLOBAL_MESH_RESOURCE = MeshResource()


@contextmanager
def global_shard_guard(resource: MeshResource):
    """Context manager for setting global sharding configuration.

    This context manager allows temporarily setting the global mesh resource
    configuration for sharding operations.

    Args:
        resource: MeshResource instance defining the sharding configuration
    """
    global _GLOBAL_MESH_RESOURCE
    old_resources = _GLOBAL_MESH_RESOURCE
    try:
        _GLOBAL_MESH_RESOURCE = resource
        yield
    finally:
        _GLOBAL_MESH_RESOURCE = old_resources


def global_mesh_resource() -> MeshResource:
    """Get the current global mesh resource configuration.

    Returns:
        The current MeshResource instance
    """
    return _GLOBAL_MESH_RESOURCE


def all_reduce_sum_along_dp_fsdp(x: jnp.array, mesh: jax.sharding.Mesh):
    """Perform all-reduce sum operation along data parallelism and FSDP axes.

    Args:
        x: Input tensor to reduce
        mesh: JAX mesh for distributed computation

    Returns:
        Reduced tensor
    """
    x = lax_paral_op(x, jax.lax.psum, global_mesh_resource().dp_resource, mesh)
    return lax_paral_op(x, jax.lax.psum, global_mesh_resource().fsdp_resource, mesh)


def all_reduce_max_along_all_axes_except_PP(x: jnp.array, mesh: jax.sharding.Mesh):
    """Perform all-reduce max operation along all axes except pipeline parallelism.

    Args:
        x: Input tensor to reduce
        mesh: JAX mesh for distributed computation

    Returns:
        Reduced tensor
    """
    all_axes = get_all_mesh_axes()
    for axis in all_axes:
        if axis != global_mesh_resource().pp_resource:
            x = lax_paral_op(x, jax.lax.pmax, axis, mesh)
    return x


# Deprecating Items ---------------------------------------------------------------
ShardingResource = MeshResource

global_shard_resource = global_mesh_resource


class MajorShardingType(Enum):
    """Enumeration of major sharding types for distributed training.

    This enum defines the basic sharding patterns available for distributed
    training. Note that this class is deprecated and will be removed in the future.

    Values:
        SINGLE: Single process training
        DP: Data parallel training
        TP: Standard tensor parallel training
        DPTP: Data and standard tensor parallel training
    """

    SINGLE = 0
    DP = 1
    TP = 2
    DPTP = 3


class ShardingType(Enum):
    """Enumeration of detailed sharding types for distributed training.

    This enum defines specific sharding patterns for distributed training,
    including combinations of data parallelism and different tensor parallelism
    strategies. Note that this class is deprecated and will be removed in the future.

    Values:
        SINGLE: No sharding
        DP: Sharding along data parallelism
        TP_COL: Sharding along column-split tensor parallelism
        TP_ROW: Sharding along row-split tensor parallelism
        DP_TP_COL: Sharding along data and column-split tensor parallelism
        DP_TP_ROW: Sharding along data and row-split tensor parallelism
    """

    SINGLE = (MajorShardingType.SINGLE, "single")
    DP = (MajorShardingType.DP, "dp")
    TP_COL = (MajorShardingType.TP, "tp_col")
    TP_ROW = (MajorShardingType.TP, "tp_row")
    DP_TP_COL = (MajorShardingType.DPTP, "dp_tp_col")
    DP_TP_ROW = (MajorShardingType.DPTP, "dp_tp_row")


def get_non_contracting_logical_axes(
    ndim, logical_axes: tuple[Optional[str]], contracting_dims
) -> tuple[Optional[str]]:
    """Get logical axes for non-contracting dimensions.

    Args:
        ndim: Number of dimensions in the tensor.
        logical_axes: Tuple of logical axes for each dimension.
        contracting_dims: Set of dimensions that are being contracted.

    Returns:
        Tuple of logical axes for non-contracting dimensions.
    """
    assert logical_axes is not None, "Logical axes must be a tuple and cannot be None."
    assert len(logical_axes) == ndim, "Logical axes must match the number of dimensions."

    non_contracting_dims = [i for i in range(ndim) if i not in contracting_dims]
    non_contracting_logical_axes = tuple(logical_axes[i] for i in non_contracting_dims)
    return non_contracting_logical_axes
