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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Optional
import warnings
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.interpreters import pxla
from jax.sharding import PartitionSpec, get_abstract_mesh
import numpy as np

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


def _validate_mesh_resource_configuration(mesh_resource):
    """Validate that the mesh resource configuration is consistent and conflict-free."""
    is_dp_enabled = (
        mesh_resource.dp_resource is not None and get_mesh_axis_size(mesh_resource.dp_resource) > 1
    )
    is_tp_enabled = (
        mesh_resource.tp_resource is not None and get_mesh_axis_size(mesh_resource.tp_resource) > 1
    )
    is_tpsp_enabled = (
        mesh_resource.tpsp_resource is not None
        and get_mesh_axis_size(mesh_resource.tpsp_resource) > 1
    )
    is_fsdp_enabled = (
        mesh_resource.fsdp_resource is not None
        and get_mesh_axis_size(mesh_resource.fsdp_resource) > 1
    )

    assert not (is_dp_enabled and is_fsdp_enabled), (
        "Data parallelism and full-sharded data parallelism cannot be enabled at the same time."
        f" Got dp_resource={mesh_resource.dp_resource} and"
        f" fsdp_resource={mesh_resource.fsdp_resource}"
    )
    assert not (is_tp_enabled and is_tpsp_enabled), (
        "Tensor parallelism and tensor sequence parallelism cannot be enabled at the same time."
        f" Got tp_resource={mesh_resource.tp_resource} and"
        f" tpsp_resource={mesh_resource.tpsp_resource}"
    )


def get_sharding_map_logic_axis_to_mesh_axis():
    """
    Generate a dict to map logical axes to mesh axes.
    """
    gsr = global_mesh_resource()

    is_tpsp_enabled = gsr.tpsp_resource is not None and get_mesh_axis_size(gsr.tpsp_resource) > 1
    is_fsdp_enabled = gsr.fsdp_resource is not None and get_mesh_axis_size(gsr.fsdp_resource) > 1

    te_logical_axis_to_mesh_axis = {
        BATCH_AXES: gsr.fsdp_resource if is_fsdp_enabled else gsr.dp_resource,
        SEQLEN_AXES: None,
        SEQLEN_TP_AXES: gsr.tpsp_resource,
        SEQLEN_CP_AXES: gsr.cp_resource,
        HEAD_AXES: gsr.tpsp_resource if is_tpsp_enabled else gsr.tp_resource,
        HIDDEN_AXES: None,
        HIDDEN_TP_AXES: gsr.tpsp_resource if is_tpsp_enabled else gsr.tp_resource,
        JOINED_AXES: None,
        W_NO_SHARD_AXES: None,
        W_FSDP_AXES: gsr.fsdp_resource,
        W_TP_AXES: gsr.tpsp_resource if is_tpsp_enabled else gsr.tp_resource,
        W_JOINED_AXES: None,
    }
    return te_logical_axis_to_mesh_axis


def _generate_pspec(logical_axis_names):
    """
    Convert TransformerEngine logical axes (e.g. BATCH_AXES) to a JAX PartitionSpec.
    Note, this method does not support Flax logical axes.

    Args:
        logical_axis_names: TransformerEngine logical axes to convert to a JAX PartitionSpec.
    Returns:
        A JAX PartitionSpec with the mesh axes corresponding to the given TransformerEngine logical axis names
    """
    rules = get_sharding_map_logic_axis_to_mesh_axis()

    mesh_axis_names = [rules.get(name) for name in logical_axis_names]
    pspec = jax.sharding.PartitionSpec(*mesh_axis_names)
    return pspec


def with_sharding_constraint(x: jnp.array, pspec: PartitionSpec):
    """
    A wrapper function to jax.lax.with_sharding_constraint
        1. Does nothing if mesh is empty.
        2. If all mesh axes are manual axes, replaces pspec with all Nones.
        3. Otherwise, strips only the manual axes.
    """
    if pspec is None:
        return x

    mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
    if mesh.empty:
        return x

    # We want to exclude the axes that already used by shard_map and shard_map
    # only sets those in the abstract_mesh, not the physical one
    manual_axis_names = get_abstract_mesh().manual_axes
    cleaned_axis_names = tuple(name if name not in manual_axis_names else None for name in pspec)

    cleaned_pspec = PartitionSpec(*cleaned_axis_names)
    return jax.lax.with_sharding_constraint(x, cleaned_pspec)


def with_sharding_constraint_by_logical_axes(
    x: jnp.array, logical_axis_names: Optional[tuple | list]
):
    """
    A wrapper function to flax.linen.with_logical_constraint.

    DEPRECATED USE CASE: If no Flax logical axis rules are available, this function falls back to jax.lax.with_sharding_constraint using a hardcoded logical axis rule table from TE rules, such as BATCH_AXES. This functionality will be removed in the future.

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

    try:
        # Check if Flax logical axis rules are available, if so use them
        import flax

        flax_rules = flax.linen.get_logical_axis_rules()
        if len(flax_rules) > 0:
            return flax.linen.with_logical_constraint(
                x, logical_axis_names, fallback=flax.linen.spmd.RulesFallback.AXIS_IS_UNSHARDED
            )
    except ImportError:
        pass

    warnings.warn(
        "TransformerEngine logical axes, such as BATCH_AXES, SEQLEN_AXES, etc. are deprecated and"
        " will be removed in a future version. Please use Flax logical axes with a"
        " flax.linen.logical_axis_rules context and optionally use"
        " transformer_engine.jax.flax.extend_logical_axis_rules to add BATCH_AXES, etc. to your"
        " rules.",
        DeprecationWarning,
    )

    # If no logical axis rules are available from Flax, fallback to TE's hardcoded logical axis rule table
    assert len(x.shape) == len(logical_axis_names)
    pspec = _generate_pspec(logical_axis_names)
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


def get_mesh_axis_rank_host(axis, mesh) -> int:
    """
    Same as get_mesh_axis_rank(), but return a host value instead of a
    traced device value.
    """
    if axis not in mesh.axis_names:
        raise ValueError(f"Axis {axis} not found in mesh axis names: {mesh.axis_names}")

    axis_index = mesh.axis_names.index(axis)

    # Convert mesh.devices (ndarray of Device objects) to flat list
    devices = mesh.devices
    local_device = jax.devices()[jax.process_index()]  # Pick one device on this host

    # Find index of local_device in mesh.devices
    coords = np.argwhere(devices == local_device)
    if coords.size == 0:
        raise ValueError(f"Local device {local_device} not found in mesh.devices.")
    coords = tuple(coords[0])  # Coordinates in the mesh array

    # Get the mesh rank along the specified axis
    rank = coords[axis_index]
    return int(rank)


@dataclass
class MeshResource:
    """A data container for managing mesh resources in distributed training.

    This class defines the mapping between logical axes and physical mesh axes
    for different types of parallelism in distributed training.

    Attributes:
        dp_resource: Axis name for data parallelism (batch sharding), default is None
        tp_resource: Axis name for tensor parallelism (hidden dimension sharding), default is None
        tpsp_resource: Axis name for tensor sequence parallelism (hidden and sequence sharding), default is None
        fsdp_resource: Axis name for full-sharded data parallelism, default is None
        pp_resource: Axis name for pipeline parallelism (layer sharding), default is None
        cp_resource: Axis name for context parallelism (sequence sharding), default is None
    """

    dp_resource: str = None
    tp_resource: str = None
    tpsp_resource: str = None
    fsdp_resource: str = None
    pp_resource: str = None
    cp_resource: str = None


_GLOBAL_MESH_RESOURCE = None


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
    assert _GLOBAL_MESH_RESOURCE is not None, (
        "Global mesh resource is not set. Please set the MeshResource via a global_shard_guard"
        " context. If you are not using multiple GPUs, you can use an empty MeshResource by"
        " wrapping your program in 'with global_shard_guard(MeshResource()):'"
    )
    _validate_mesh_resource_configuration(_GLOBAL_MESH_RESOURCE)
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


def tpsp_axis_size():
    """
    Get the size of the tensor parallelism axis.
    Return 1 if no TP axis is set.
    """
    return get_mesh_axis_size(global_mesh_resource().tpsp_resource)


def dp_or_fsdp_axis_size():
    """
    Get the size of the data parallelism or FSDP axis.
    Return 1 if no DP/FSDP axis is set.
    """
    dp_size = get_mesh_axis_size(global_mesh_resource().dp_resource)
    fsdp_size = get_mesh_axis_size(global_mesh_resource().fsdp_resource)
    return dp_size if dp_size > 1 else fsdp_size
