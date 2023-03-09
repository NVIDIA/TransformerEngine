# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Sharding Meta for xmap with CustomCall
"""

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Union, Tuple, Dict, Callable, Sequence
from jax.interpreters import pxla
import jax
import jax.numpy as jnp
from jax.experimental.maps import xmap

jax.config.update('experimental_xmap_spmd_lowering', True)
jax.config.update('experimental_xmap_spmd_lowering_manual', True)

_PXLA_THREAD_RESOURCES = pxla.thread_resources


def _get_mesh_info(resource: str):
    mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
    assert resource in mesh.axis_names, \
        f"{resource} is not in the axis_names of Mesh {mesh}."
    return mesh.shape[resource], resource


@dataclass
class ShardingResource:
    """
    A data container to indicate which axis in Mesh for data parallelism and
    which for tensor parallelism.

    Parameters
    ----------
    dp_resource : str, default = None
        The axis name in Mesh used to shard batches along.
        If it is None, then disabling data parallelism.
    tp_resource : str, default = None
        The axis name in Mesh used to split the hidden dimensions along.
        If it is None, then disabling tensor parallelism.
    """
    dp_resource: str = None
    tp_resource: str = None


_GLOBAL_SHARD_RESOURCE = ShardingResource()


@contextmanager
def global_shard_guard(resource: ShardingResource):
    """
    A context manager to switch the global ShardingResource
    """
    global _GLOBAL_SHARD_RESOURCE
    prev_gsr = _GLOBAL_SHARD_RESOURCE
    try:
        _GLOBAL_SHARD_RESOURCE = resource
        yield
    finally:
        _GLOBAL_SHARD_RESOURCE = prev_gsr


def global_shard_resource() -> ShardingResource:
    """
    A getter of  the global ShardingResource
    """
    return _GLOBAL_SHARD_RESOURCE


class MajorShardingType(Enum):
    """
    The major sharding type to indicate sharding pattern.
    `SINGLE` means single process training.
    `DP` means data parallel traiing.
    `TP` means tensor parallel traiing.
    `DPTP` means data and tensor parallel traiing.
    """
    SINGLE = 0
    DP = 1
    TP = 2
    DPTP = 3


class ShardingType(Enum):
    """
    The sharding type to indicate sharding pattern.
    `SINGLE` means no sharding.
    `DP` means sharding along data parallelism.
    `TP_COL` means sharding along column-split tensor parallelism.
    `TP_ROW` means sharding along row-split tensor parallelism.
    `DP_TP_COL` means sharding along data and column-split tensor parallelism.
    `DP_TP_ROW` means sharding along data and row-split tensor parallelism.
    """
    SINGLE = (MajorShardingType.SINGLE, "single")
    DP = (MajorShardingType.DP, "dp")
    TP_COL = (MajorShardingType.TP, "tp_col")
    TP_ROW = (MajorShardingType.TP, "tp_row")
    DP_TP_COL = (MajorShardingType.DPTP, "dp_tp_col")
    DP_TP_ROW = (MajorShardingType.DPTP, "dp_tp_row")


def infer_major_sharding_type() -> MajorShardingType:
    """
    Infer MajorShardingType from _GLOBAL_SHARD_RESOURCE
    """
    gsr = global_shard_resource()

    resources = [gsr.dp_resource, gsr.tp_resource]
    for idx, rs in enumerate(resources):
        try:
            size, _ = _get_mesh_info(rs)
            if size <= 1:
                resources[idx] = None
        except AssertionError as _:
            resources[idx] = None

    dp_resource = resources[0]
    tp_resource = resources[1]

    if dp_resource is not None and \
        tp_resource is not None :
        return MajorShardingType.DPTP

    if dp_resource is not None:
        return MajorShardingType.DP

    if tp_resource is not None:
        return MajorShardingType.TP

    return MajorShardingType.SINGLE


def infer_sharding_type(major_st: MajorShardingType = None) -> Tuple[ShardingType, ShardingType]:
    """
    Infer ShardingType via given MajorShardingType
    """
    if major_st is None:
        major_st = infer_major_sharding_type()

    if major_st is MajorShardingType.DP:
        return ShardingType.DP, ShardingType.DP
    if major_st is MajorShardingType.TP:
        return ShardingType.TP_COL, ShardingType.TP_ROW
    if major_st is MajorShardingType.DPTP:
        return ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW
    return ShardingType.SINGLE, ShardingType.SINGLE


def is_dp_enabled(mst: MajorShardingType) -> bool:
    """
    is_dp_enabled
    """
    return mst in (MajorShardingType.DP, MajorShardingType.DPTP)


def is_tp_enabled(mst: MajorShardingType) -> bool:
    """
    is_tp_enabled
    """
    return mst in (MajorShardingType.TP, MajorShardingType.DPTP)


def merge_axis_resources(ars: Tuple[Dict]) -> Dict:
    """
    merge_axis_resources
    """
    output = {}
    for ar in ars:
        for key in ar:
            if key not in output:
                output[key] = ar[key]
            else:
                assert output[key] == ar[key]
    return output


@dataclass
class ShardingMeta:
    """ShardingMeta"""
    in_axes: Union[Dict, Tuple[str, ...], Tuple[Union[Dict, Tuple], ...]]
    out_axes: Union[Dict, Tuple[str, ...], Tuple[Union[Dict, Tuple], ...]]
    axis_resources: Dict
    input_shapes: Tuple[Tuple[int, ...]]
    output_shapes: Tuple[Tuple[int, ...]]


class ShardingMetaGenerator:
    """
    ShardingMetaGenerator
    """

    def __init__(self):

        def get_single_sharding_meta(*argv, **kwargs) -> ShardingMeta:    # pylint: disable=unused-argument
            return None

        self.sharding_type_meta_map = {
            ShardingType.SINGLE: get_single_sharding_meta,
            ShardingType.DP: self.get_dp_sharding_meta,
            ShardingType.TP_COL: self.get_tp_col_sharding_meta,
            ShardingType.TP_ROW: self.get_tp_row_sharding_meta,
            ShardingType.DP_TP_COL: self.get_dp_tp_col_sharding_meta,
            ShardingType.DP_TP_ROW: self.get_dp_tp_row_sharding_meta
        }

    def get_sharding_meta(self, stype: ShardingType, *argv, **kwargs) -> ShardingMeta:
        """get_sharding_meta"""
        return self.sharding_type_meta_map[stype](*argv, **kwargs)

    def get_dp_sharding_meta(self, *argv, **kwargs) -> ShardingMeta:
        """get_dp_sharding_meta"""
        raise NotImplementedError

    def get_tp_col_sharding_meta(self, *argv, **kwargs) -> ShardingMeta:
        """get_tp_col_sharding_meta"""
        raise NotImplementedError

    def get_tp_row_sharding_meta(self, *argv, **kwargs) -> ShardingMeta:
        """get_tp_row_sharding_meta"""
        raise NotImplementedError

    def get_dp_tp_col_sharding_meta(self, *argv, **kwargs) -> ShardingMeta:
        """get_dp_tp_col_sharding_meta"""
        raise NotImplementedError

    def get_dp_tp_row_sharding_meta(self, *argv, **kwargs) -> ShardingMeta:
        """get_dp_tp_row_sharding_meta"""
        raise NotImplementedError


class FP8MetaShardingMetaGenerator(ShardingMetaGenerator):
    """
    FP8MetaShardingMetaGenerator
    """

    def get_dp_sharding_meta(self,
                             num_of_meta: int,
                             dp_axis_name: str = 'data',
                             tp_axis_name: str = 'model') -> ShardingMeta:
        return FP8MetaShardingMetaGenerator._generate_sharding_meta(MajorShardingType.DP,
                                                                    num_of_meta, dp_axis_name,
                                                                    tp_axis_name)

    def get_tp_col_sharding_meta(self,
                                 num_of_meta: int,
                                 dp_axis_name: str = 'data',
                                 tp_axis_name: str = 'model') -> ShardingMeta:
        return FP8MetaShardingMetaGenerator._generate_sharding_meta(MajorShardingType.TP,
                                                                    num_of_meta, dp_axis_name,
                                                                    tp_axis_name)

    def get_tp_row_sharding_meta(self,
                                 num_of_meta: int,
                                 dp_axis_name: str = 'data',
                                 tp_axis_name: str = 'model') -> ShardingMeta:
        return FP8MetaShardingMetaGenerator._generate_sharding_meta(MajorShardingType.TP,
                                                                    num_of_meta, dp_axis_name,
                                                                    tp_axis_name)

    def get_dp_tp_col_sharding_meta(self,
                                    num_of_meta: int,
                                    dp_axis_name: str = 'data',
                                    tp_axis_name: str = 'model') -> ShardingMeta:
        return FP8MetaShardingMetaGenerator._generate_sharding_meta(MajorShardingType.DPTP,
                                                                    num_of_meta, dp_axis_name,
                                                                    tp_axis_name)

    def get_dp_tp_row_sharding_meta(self,
                                    num_of_meta: int,
                                    dp_axis_name: str = 'data',
                                    tp_axis_name: str = 'model') -> ShardingMeta:
        return FP8MetaShardingMetaGenerator._generate_sharding_meta(MajorShardingType.DPTP,
                                                                    num_of_meta, dp_axis_name,
                                                                    tp_axis_name)

    @staticmethod
    def _stack_axes_meta(num_of_meta: int, mapping: Dict) -> Tuple:
        return tuple(mapping for _ in range(num_of_meta))

    @staticmethod
    def _generate_sharding_meta(type_: MajorShardingType,
                                num_of_meta: int,
                                dp_axis_name: str = 'data',
                                tp_axis_name: str = 'model') -> ShardingMeta:

        axis_resource = {}

        if is_dp_enabled(type_):
            axis_resource[dp_axis_name] = global_shard_resource().dp_resource

        if is_tp_enabled(type_):
            axis_resource[tp_axis_name] = global_shard_resource().tp_resource

        return ShardingMeta(FP8MetaShardingMetaGenerator._stack_axes_meta(num_of_meta, {}),
                            FP8MetaShardingMetaGenerator._stack_axes_meta(num_of_meta, {}),
                            axis_resource, (), ())


class DotShardingMetaGenerator(ShardingMetaGenerator):
    """
    DotShardingMetaGenerator
    """

    def get_dp_sharding_meta(
            self,
            a_shape: Tuple,
            b_shape: Tuple,
            batch_dim_of_a: int,
            model_dim_of_a: int,    # pylint: disable=unused-argument
            model_dim_of_b: int,    # pylint: disable=unused-argument
            contracting_dims: Tuple[Sequence[int], Sequence[int]],
            dp_axis_name: str = 'data',
            tp_axis_name: str = 'model'    # pylint: disable=unused-argument
    ) -> ShardingMeta:
        DotShardingMetaGenerator._is_supported(a_shape, b_shape, batch_dim_of_a, None,
                                               contracting_dims)

        out_shape = DotShardingMetaGenerator._infer_output_shape(a_shape, b_shape, contracting_dims)
        out_batch_dim = batch_dim_of_a

        dp_size, dp_mesh_axis = _get_mesh_info(global_shard_resource().dp_resource)
        assert a_shape[batch_dim_of_a] % dp_size == 0, \
            f"The dimension of batch in a_shape should be a multiple of data parallelism size," \
            f" but got {a_shape[batch_dim_of_a]=} and {dp_size=}."
        a_new_shape = (*a_shape[:batch_dim_of_a], dp_size, -1, *a_shape[batch_dim_of_a + 1:])
        return ShardingMeta(({
            batch_dim_of_a: dp_axis_name
        }, {}), ({
            out_batch_dim: dp_axis_name
        }), {dp_axis_name: dp_mesh_axis}, [a_new_shape, b_shape], [out_shape])

    def get_tp_col_sharding_meta(
            self,
            a_shape: Tuple,
            b_shape: Tuple,
            batch_dim_of_a: int,
            model_dim_of_a: int,    # pylint: disable=unused-argument
            model_dim_of_b: int,
            contracting_dims: Tuple[Sequence[int], Sequence[int]],
            dp_axis_name: str = 'data',    # pylint: disable=unused-argument
            tp_axis_name: str = 'model') -> ShardingMeta:
        DotShardingMetaGenerator._is_supported(a_shape, b_shape, batch_dim_of_a, None,
                                               contracting_dims)

        out_shape = DotShardingMetaGenerator._infer_output_shape(a_shape, b_shape, contracting_dims)

        out_model_idx = len(out_shape) - (len(b_shape) - model_dim_of_b)

        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)
        assert b_shape[model_dim_of_b] % tp_size == 0, \
            f"The dimension of model parallelism in b_shape should be a multiple of " \
            f"tensor parallelism size,but got {b_shape[model_dim_of_b]=} and {tp_size=}."
        b_new_shape = (*b_shape[:model_dim_of_b], tp_size, b_shape[model_dim_of_b] // tp_size,
                       *b_shape[model_dim_of_b + 1:])
        return ShardingMeta(({}, {
            model_dim_of_b: tp_axis_name
        }), ({
            out_model_idx: tp_axis_name
        }), {tp_axis_name: tp_mesh_axis}, [a_shape, b_new_shape], [out_shape])

    def get_tp_row_sharding_meta(
            self,
            a_shape: Tuple,
            b_shape: Tuple,
            batch_dim_of_a: int,
            model_dim_of_a: int,
            model_dim_of_b: int,
            contracting_dims: Tuple[Sequence[int], Sequence[int]],
            dp_axis_name: str = 'data',    # pylint: disable=unused-argument
            tp_axis_name: str = 'model') -> ShardingMeta:
        DotShardingMetaGenerator._is_supported(a_shape, b_shape, batch_dim_of_a, model_dim_of_a,
                                               contracting_dims)

        out_shape = DotShardingMetaGenerator._infer_output_shape(a_shape, b_shape, contracting_dims)

        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)
        assert a_shape[model_dim_of_a] % tp_size == 0, \
            f"The dimension of model parallelism in a_shape should be a multiple of " \
            f"tensor parallelism size,but got {a_shape[model_dim_of_a]=} and {tp_size=}."
        assert b_shape[model_dim_of_b] % tp_size == 0, \
            f"The dimension of model parallelism in b_shape should be a multiple of " \
            f"tensor parallelism size,but got {b_shape[model_dim_of_b]=} and {tp_size=}."
        a_new_shape = (*a_shape[:model_dim_of_a], tp_size, a_shape[model_dim_of_a] // tp_size,
                       *a_shape[model_dim_of_a + 1:])
        b_new_shape = (*b_shape[:model_dim_of_b], tp_size, b_shape[model_dim_of_b] // tp_size,
                       *b_shape[model_dim_of_b + 1:])
        return ShardingMeta(({
            model_dim_of_a: tp_axis_name
        }, {
            model_dim_of_b: tp_axis_name
        }), ({}), {tp_axis_name: tp_mesh_axis}, [a_new_shape, b_new_shape], [out_shape])

    def get_dp_tp_col_sharding_meta(
            self,
            a_shape: Tuple,
            b_shape: Tuple,
            batch_dim_of_a: int,
            model_dim_of_a: int,    # pylint: disable=unused-argument
            model_dim_of_b: int,
            contracting_dims: Tuple[Sequence[int], Sequence[int]],
            dp_axis_name: str = 'data',
            tp_axis_name: str = 'model') -> ShardingMeta:
        DotShardingMetaGenerator._is_supported(a_shape, b_shape, batch_dim_of_a, None,
                                               contracting_dims)

        out_shape = DotShardingMetaGenerator._infer_output_shape(a_shape, b_shape, contracting_dims)

        out_model_idx = len(out_shape) + 1 - (len(b_shape) - model_dim_of_b)

        dp_size, dp_mesh_axis = _get_mesh_info(global_shard_resource().dp_resource)
        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)
        assert a_shape[batch_dim_of_a] % dp_size == 0, \
            f"The dimension of batch in a_shape should be a multiple of data parallelism size," \
            f" but got {a_shape[batch_dim_of_a]=} and {dp_size=}."
        assert b_shape[model_dim_of_b] % tp_size == 0, \
            f"The dimension of model parallelism in b_shape should be a multiple of " \
            f"tensor parallelism size,but got {b_shape[model_dim_of_b]=} and {tp_size=}."
        a_new_shape = (*a_shape[:batch_dim_of_a], dp_size, a_shape[batch_dim_of_a] // dp_size,
                       *a_shape[batch_dim_of_a + 1:])
        b_new_shape = (*b_shape[:model_dim_of_b], tp_size, b_shape[model_dim_of_b] // tp_size,
                       *b_shape[model_dim_of_b + 1:])
        return ShardingMeta(({
            batch_dim_of_a: dp_axis_name
        }, {
            model_dim_of_b: tp_axis_name
        }), ({
            batch_dim_of_a: dp_axis_name,
            out_model_idx: tp_axis_name
        }), {
            dp_axis_name: dp_mesh_axis,
            tp_axis_name: tp_mesh_axis
        }, [a_new_shape, b_new_shape], [out_shape])

    def get_dp_tp_row_sharding_meta(self,
                                    a_shape: Tuple,
                                    b_shape: Tuple,
                                    batch_dim_of_a: int,
                                    model_dim_of_a: int,
                                    model_dim_of_b: int,
                                    contracting_dims: Tuple[Sequence[int], Sequence[int]],
                                    dp_axis_name: str = 'data',
                                    tp_axis_name: str = 'model') -> ShardingMeta:
        DotShardingMetaGenerator._is_supported(a_shape, b_shape, batch_dim_of_a, model_dim_of_a,
                                               contracting_dims)

        out_shape = DotShardingMetaGenerator._infer_output_shape(a_shape, b_shape, contracting_dims)

        dp_size, dp_mesh_axis = _get_mesh_info(global_shard_resource().dp_resource)
        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)
        assert a_shape[batch_dim_of_a] % dp_size == 0, \
            f"The dimension of batch in a_shape should be a multiple of data parallelism size," \
            f" but got {a_shape[batch_dim_of_a]=} and {dp_size=}."
        assert a_shape[model_dim_of_a] % tp_size == 0, \
            f"The dimension of model parallelism in a_shape should be a multiple of " \
            f"tensor parallelism size,but got {a_shape[model_dim_of_a]=} and {tp_size=}."
        assert b_shape[model_dim_of_b] % tp_size == 0, \
            f"The dimension of model parallelism in b_shape should be a multiple of " \
            f"tensor parallelism size,but {b_shape[model_dim_of_b]=} and {tp_size=}."
        a_new_shape = (*a_shape[:batch_dim_of_a], dp_size, a_shape[batch_dim_of_a] // dp_size,
                       *a_shape[batch_dim_of_a + 1:model_dim_of_a], tp_size,
                       a_shape[model_dim_of_a] // tp_size, *a_shape[model_dim_of_a + 1:])
        b_new_shape = (*b_shape[:model_dim_of_b], tp_size, b_shape[model_dim_of_b] // tp_size,
                       *b_shape[model_dim_of_b + 1:])
        return ShardingMeta(
            (
                {
                    batch_dim_of_a:
                        dp_axis_name,
        # "model_dim_of_a+1" is the index to tp_size in a_new_shape
                    model_dim_of_a + 1:
                        tp_axis_name
                },
                {
                    model_dim_of_b: tp_axis_name
                }),
            ({
                batch_dim_of_a: dp_axis_name
            }),
            {
                dp_axis_name: dp_mesh_axis,
                tp_axis_name: tp_mesh_axis
            },
            [a_new_shape, b_new_shape],
            [out_shape])

    @staticmethod
    def _is_supported(
        a_shape: Tuple,    # pylint: disable=unused-argument
        b_shape: Tuple,    # pylint: disable=unused-argument
        batch_dim_of_a: int,
        model_dim_of_a: int,
        contracting_dims: Tuple[Sequence[int], Sequence[int]],
    ):
        assert batch_dim_of_a not in contracting_dims[0], \
            "batch_dim_of_a should be one of contracting_dims[0]"
        assert batch_dim_of_a >= 0, \
            "Only support non-negative value of batch_dim_of_a."
        if model_dim_of_a is not None:
            assert model_dim_of_a >= 0, \
                "Only support non-negative value of model_dim_of_a"
            assert model_dim_of_a > batch_dim_of_a, \
                "Only support the case that model_dim_of_a > batch_dim_of_a."

    @staticmethod
    def _infer_output_shape(
        a_shape: Tuple,
        b_shape: Tuple,
        contracting_dims: Tuple[Sequence[int], Sequence[int]],
    ):
        lhs_contracting_dims, rhs_contracting_dims = contracting_dims
        return (*a_shape[:min(lhs_contracting_dims)], *b_shape[max(rhs_contracting_dims) + 1:])


class ElementwiseShardingMetaGenerator(ShardingMetaGenerator):
    """
    ElementwiseShardingMetaGenerator
    """

    def get_dp_sharding_meta(
            self,
            input_shape: Tuple,
            other_shape: Tuple,
            batch_dim: int,
            dp_axis_name: str = 'data',
            tp_axis_name: str = 'model'    # pylint: disable=unused-argument
    ) -> ShardingMeta:
        """get_dp_sharding_meta"""
        ElementwiseShardingMetaGenerator._is_supported(input_shape, other_shape, batch_dim)

        dp_size, dp_mesh_axis = _get_mesh_info(global_shard_resource().dp_resource)

        assert input_shape[batch_dim] % dp_size == 0, \
            f"The dimension of batch in input_shape should be a multiple of data parallelism " \
            f"size, but got {input_shape[batch_dim]=} and {dp_size=}."
        input_new_shape = (*input_shape[:batch_dim], dp_size, -1, *input_shape[batch_dim + 1:])
        in_axes = [{batch_dim: dp_axis_name}]
        input_new_shapes = [input_new_shape]
        if other_shape is not None:
            input_new_shapes.append(other_shape)
            in_axes.append({})

        return ShardingMeta(tuple(in_axes), ({
            batch_dim: dp_axis_name
        }), {dp_axis_name: dp_mesh_axis}, input_new_shapes, [input_shape])

    def get_tp_col_sharding_meta(
        self,
        input_shape: Tuple,
        other_shape: Tuple,
        batch_dim: int,    # pylint: disable=unused-argument
        dp_axis_name: str = 'data',    # pylint: disable=unused-argument
        tp_axis_name: str = 'model'    # pylint: disable=unused-argument
    ) -> ShardingMeta:
        """get_tp_col_sharding_meta"""
        ElementwiseShardingMetaGenerator._is_supported(input_shape, other_shape, 0)
        in_axes = [{}]
        input_new_shapes = [input_shape]
        if other_shape is not None:
            in_axes.append({})
            input_new_shapes.append(other_shape)

        return ShardingMeta(tuple(in_axes), ({}), {}, input_new_shapes, [input_shape])

    def get_tp_row_sharding_meta(
            self,
            input_shape: Tuple,
            other_shape: Tuple,
            batch_dim: int,    # pylint: disable=unused-argument
            dp_axis_name: str = 'data',    # pylint: disable=unused-argument
            tp_axis_name: str = 'model') -> ShardingMeta:
        """get_tp_row_sharding_meta"""
        ElementwiseShardingMetaGenerator._is_supported(input_shape, other_shape, 0)

        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)

        assert input_shape[-1] % tp_size == 0, \
            f"The last dimension in input_shape should be a multiple of tensor parallelism size," \
            f" but got {input_shape[-1]=} and {tp_size=}."
        input_new_shape = (*input_shape[:-1], tp_size, -1)

        in_axes = [{
        # "len(a_new_shape)-2" is the index to tp_size in a_new_shape
            len(input_new_shape) - 2:
                tp_axis_name
        }]
        input_new_shapes = [input_new_shape]

        if other_shape is not None:
            assert other_shape[0] % tp_size == 0, \
            f"The first dimension in other_shape should be a multiple of tensor parallelism size," \
            f" but got {other_shape[0]=} and {tp_size=}."
            other_new_shape = (tp_size, -1)
            in_axes.append({0: tp_axis_name})
            input_new_shapes.append(other_new_shape)

        return ShardingMeta(tuple(in_axes), ({
            len(input_new_shape) - 2: tp_axis_name
        }), {tp_axis_name: tp_mesh_axis}, input_new_shapes, [input_shape])

    def get_dp_tp_col_sharding_meta(self,
                                    input_shape: Tuple,
                                    other_shape: Tuple,
                                    batch_dim: int,
                                    dp_axis_name: str = 'data',
                                    tp_axis_name: str = 'model') -> ShardingMeta:
        """get_dp_tp_col_sharding_meta"""
        return self.get_dp_sharding_meta(input_shape, other_shape, batch_dim, dp_axis_name,
                                         tp_axis_name)

    def get_dp_tp_row_sharding_meta(self,
                                    input_shape: Tuple,
                                    other_shape: Tuple,
                                    batch_dim: int,
                                    dp_axis_name: str = 'data',
                                    tp_axis_name: str = 'model') -> ShardingMeta:
        """get_dp_tp_row_sharding_meta"""
        ElementwiseShardingMetaGenerator._is_supported(input_shape, other_shape, batch_dim)

        dp_size, dp_mesh_axis = _get_mesh_info(global_shard_resource().dp_resource)
        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)

        assert input_shape[batch_dim] % dp_size == 0, \
            f"The dimension of batch in input_shape should be a multiple of data parallelism" \
            f"size, but got {input_shape[batch_dim]=} and {dp_size=}."
        assert input_shape[-1] % tp_size == 0, \
            f"The last dimension in input_shape should be a multiple of tensor parallelism size," \
            f" but got {input_shape[-1]=} and {tp_size=}."
        input_new_shape = (*input_shape[:batch_dim], dp_size, -1, *input_shape[batch_dim + 1:-1],
                           tp_size, input_shape[-1] // tp_size)

        in_axes = [{
            batch_dim:
                dp_axis_name,
        # "len(a_new_shape)-2" is the index to tp_size in a_new_shape
            len(input_new_shape) - 2:
                tp_axis_name
        }]
        input_new_shapes = [input_new_shape]

        other_new_shape = other_shape
        if other_shape is not None:
            assert other_shape[0] % tp_size == 0, \
            f"The first dimension in other_shape should be a multiple of tensor parallelism size," \
            f" but got {other_shape[0]=} and {tp_size=}."
            other_new_shape = (tp_size, -1)
            in_axes.append({0: tp_axis_name})
            input_new_shapes.append(other_new_shape)

        return ShardingMeta(tuple(in_axes), ({
            batch_dim: dp_axis_name,
            len(input_new_shape) - 2: tp_axis_name
        }), {
            dp_axis_name: dp_mesh_axis,
            tp_axis_name: tp_mesh_axis
        }, input_new_shapes, [input_shape])

    @staticmethod
    def _is_supported(input_shape: Tuple, other_shape: Tuple, batch_dim: int):
        if other_shape is not None:
            assert len(other_shape) == 1, "Only support 1 dimension of other_shapes currently."
            assert input_shape[-1] == other_shape[0], \
                f"input_shape[-1] should equal to oshape[0], " \
                f"but got {input_shape[-1]} and {other_shape[0]}."

        assert batch_dim < len(input_shape)-1, \
            "batch_dim cannot be the latest dim"


class SoftmaxShardingMetaGenerator(ShardingMetaGenerator):
    """
    SoftmaxShardingMetaGenerator
    """

    def get_dp_sharding_meta(
            self,
            input_shape: Tuple,
            dp_dim: int = 0,
            tp_dim: int = 1,
            dp_axis_name: str = 'data',
            tp_axis_name: str = 'model'    # pylint: disable=unused-argument
    ) -> ShardingMeta:
        """get_dp_sharding_meta"""
        SoftmaxShardingMetaGenerator._is_supported(input_shape, dp_dim, tp_dim)

        dp_size, dp_mesh_axis = _get_mesh_info(global_shard_resource().dp_resource)

        assert input_shape[dp_dim] % dp_size == 0, \
            f"The dimension of batch in input_shape should be a multiple of data parallelism " \
            f"size, but got {input_shape[dp_dim]=} and {dp_size=}."
        input_new_shape = (*input_shape[:dp_dim], dp_size, -1, *input_shape[dp_dim + 1:])
        in_axes = [{dp_dim: dp_axis_name}]
        input_new_shapes = [input_new_shape]

        out_axes = in_axes[0]

        return ShardingMeta(tuple(in_axes), out_axes, {dp_axis_name: dp_mesh_axis},
                            input_new_shapes, [input_shape])

    def get_tp_col_sharding_meta(self,
                                 input_shape: Tuple,
                                 dp_dim: int = 0,
                                 tp_dim: int = 1,
                                 dp_axis_name: str = 'data',
                                 tp_axis_name: str = 'model') -> ShardingMeta:
        """get_tp_col_sharding_meta"""
        return SoftmaxShardingMetaGenerator._get_tp_sharding_meta(input_shape, dp_dim, tp_dim,
                                                                  dp_axis_name, tp_axis_name)

    def get_tp_row_sharding_meta(self,
                                 input_shape: Tuple,
                                 dp_dim: int = 0,
                                 tp_dim: int = 1,
                                 dp_axis_name: str = 'data',
                                 tp_axis_name: str = 'model') -> ShardingMeta:
        """get_tp_row_sharding_meta"""
        return SoftmaxShardingMetaGenerator._get_tp_sharding_meta(input_shape, dp_dim, tp_dim,
                                                                  dp_axis_name, tp_axis_name)

    def get_dp_tp_col_sharding_meta(self,
                                    input_shape: Tuple,
                                    dp_dim: int = 0,
                                    tp_dim: int = 1,
                                    dp_axis_name: str = 'data',
                                    tp_axis_name: str = 'model') -> ShardingMeta:
        """get_dp_tp_col_sharding_meta"""
        return SoftmaxShardingMetaGenerator._get_dptp_sharding_meta(input_shape, dp_dim, tp_dim,
                                                                    dp_axis_name, tp_axis_name)

    def get_dp_tp_row_sharding_meta(self,
                                    input_shape: Tuple,
                                    dp_dim: int = 0,
                                    tp_dim: int = 1,
                                    dp_axis_name: str = 'data',
                                    tp_axis_name: str = 'model') -> ShardingMeta:
        """get_dp_tp_row_sharding_meta"""
        return SoftmaxShardingMetaGenerator._get_dptp_sharding_meta(input_shape, dp_dim, tp_dim,
                                                                    dp_axis_name, tp_axis_name)

    @staticmethod
    def _is_supported(input_shape: Tuple, dp_dim: int, tp_dim: int):
        assert len(input_shape) == 4
        assert dp_dim == 0
        assert tp_dim == 1

    @staticmethod
    def _get_tp_sharding_meta(
        input_shape: Tuple,
        dp_dim: int = 0,
        tp_dim: int = 1,
        dp_axis_name: str = 'data',    # pylint: disable=unused-argument
        tp_axis_name: str = 'model'    # pylint: disable=unused-argument
    ) -> ShardingMeta:
        """get_tp_sharding_meta"""
        SoftmaxShardingMetaGenerator._is_supported(input_shape, dp_dim, tp_dim)

        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)

        assert input_shape[tp_dim] % tp_size == 0, \
            f"The dimension of tensor parallel in input_shape should be a multiple of data " \
            f"parallelism size, but got {input_shape[tp_dim]=} and {tp_size=}."
        input_new_shape = (*input_shape[:tp_dim], tp_size, -1, *input_shape[tp_dim + 1:])
        in_axes = [{tp_dim: tp_axis_name}]
        input_new_shapes = [input_new_shape]

        out_axes = in_axes[0]

        return ShardingMeta(tuple(in_axes), out_axes, {tp_axis_name: tp_mesh_axis},
                            input_new_shapes, [input_shape])

    @staticmethod
    def _get_dptp_sharding_meta(input_shape: Tuple,
                                dp_dim: int = 0,
                                tp_dim: int = 1,
                                dp_axis_name: str = 'data',
                                tp_axis_name: str = 'model') -> ShardingMeta:
        """get_dp_tp_sharding_meta"""
        SoftmaxShardingMetaGenerator._is_supported(input_shape, dp_dim, tp_dim)

        dp_size, dp_mesh_axis = _get_mesh_info(global_shard_resource().dp_resource)
        tp_size, tp_mesh_axis = _get_mesh_info(global_shard_resource().tp_resource)

        assert input_shape[dp_dim] % dp_size == 0, \
            f"The dimension of batch in input_shape should be a multiple of data parallelism " \
            f"size, but got {input_shape[dp_dim]=} and {dp_size=}."
        assert input_shape[tp_dim] % tp_size == 0, \
            f"The dimension of tensor parallel in input_shape should be a multiple of data " \
            f"parallelism size, but got {input_shape[tp_dim]=} and {tp_size=}."

        input_new_shape = (*input_shape[:dp_dim], dp_size, input_shape[dp_dim] // dp_size,
                           *input_shape[dp_dim + 1:tp_dim], tp_size, input_shape[tp_dim] // tp_size,
                           *input_shape[tp_dim + 1:])

        in_axes = [{dp_dim: dp_axis_name, tp_dim + 1: tp_axis_name}]
        input_new_shapes = [input_new_shape]

        out_axes = in_axes[0]

        return ShardingMeta(tuple(in_axes), out_axes, {
            dp_axis_name: dp_mesh_axis,
            tp_axis_name: tp_mesh_axis
        }, input_new_shapes, [input_shape])


def get_fp8_meta_sharding_meta(stype: ShardingType,
                               num_of_meta: int,
                               dp_axis_name: str = 'data',
                               tp_axis_name: str = 'model') -> ShardingMeta:
    """
    get_fp8_meta_sharding_meta
    """
    return FP8MetaShardingMetaGenerator().get_sharding_meta(stype, num_of_meta, dp_axis_name,
                                                            tp_axis_name)


def get_dot_sharding_meta(stype: ShardingType,
                          a_shape: Tuple,
                          b_shape: Tuple,
                          batch_dim_of_a: int,
                          model_dim_of_a: int,
                          model_dim_of_b: int,
                          contracting_dims: Tuple[Sequence[int], Sequence[int]] = ((-1,), (0,)),
                          dp_axis_name: str = 'data',
                          tp_axis_name: str = 'model') -> ShardingMeta:
    """
    get_dot_sharding_meta
    """
    if stype in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW):
        assert model_dim_of_b <= max(contracting_dims[1]), \
                f"The dimension of model parallelism in b_shape should be smaller than the max of" \
                f" contracting_dims[1], but got {model_dim_of_b=} and {contracting_dims[1]=}."
    if stype in (ShardingType.TP_COL, ShardingType.DP_TP_COL):
        assert model_dim_of_b > max(contracting_dims[1]), \
                f"The dimension of model parallelism in b_shape should be larger than the max of" \
                f" contracting_dims[1], but got {model_dim_of_b=} and {contracting_dims[1]=}."
    return DotShardingMetaGenerator().get_sharding_meta(stype, a_shape, b_shape, batch_dim_of_a,
                                                        model_dim_of_a, model_dim_of_b,
                                                        contracting_dims, dp_axis_name,
                                                        tp_axis_name)


def get_elementwise_sharding_meta(stype: ShardingType,
                                  input_shape: Tuple,
                                  other_shape: Tuple,
                                  batch_dim: int,
                                  dp_axis_name: str = 'data',
                                  tp_axis_name: str = 'model') -> ShardingMeta:
    """
    get_elementwise_sharding_meta
    """
    return ElementwiseShardingMetaGenerator().get_sharding_meta(stype, input_shape, other_shape,
                                                                batch_dim, dp_axis_name,
                                                                tp_axis_name)


def get_softmax_sharding_meta(stype: ShardingType,
                              input_shape: Tuple,
                              dp_dim: int = 0,
                              tp_dim: int = 1,
                              dp_axis_name: str = 'data',
                              tp_axis_name: str = 'model') -> ShardingMeta:
    """
    get_softmax_sharding_meta
    """
    return SoftmaxShardingMetaGenerator().get_sharding_meta(stype, input_shape, dp_dim, tp_dim,
                                                            dp_axis_name, tp_axis_name)


def xmap_runner(func: Callable, in_axes: Tuple[Dict, ...],
                out_axes: Union[Dict, Tuple[str, ...], Tuple[Union[Dict, Tuple], ...]],
                axis_resources: Dict, inputs: Tuple):
    """
    xmap_runner
    """
    assert isinstance(inputs, tuple)
    assert isinstance(in_axes, tuple)

    mesh = _PXLA_THREAD_RESOURCES.env.physical_mesh
    fake_in_axes = {}
    fake_axis_resource = {}

    # Fake related setup is a workaround to "NotImplementedError:
    # Collectives in manually partitioned computations are only supported
    # when all mesh axes are partitioned manually (no partial automatic
    # sharding). Make sure that you mention all mesh axes in axis_resources!"
    for i, mesh_axis_names in enumerate(mesh.axis_names):
        if mesh_axis_names not in axis_resources.values():
            fake_axis_name = f"{mesh_axis_names}_fake_{i}"
            fake_in_axes[i] = fake_axis_name
            fake_axis_resource[fake_axis_name] = mesh_axis_names

    fake_input = jnp.zeros(tuple(64 for _ in range(len(fake_in_axes) + 1)))

    xmapped = xmap(lambda func_input, _: func(*func_input),
                   in_axes=(in_axes, fake_in_axes),
                   out_axes=out_axes,
                   axis_resources={
                       **axis_resources,
                       **fake_axis_resource
                   })
    output = xmapped(inputs, fake_input)
    return output
