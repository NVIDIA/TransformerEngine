# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Helper module for fp8 meta management
"""
import os
from contextlib import contextmanager
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from transformer_engine_jax import DType
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.jax.sharding import global_shard_guard
from transformer_engine.jax.sharding import ShardingResource

Collection = Union[Dict, FrozenDict]


def _format2dtypes(format_: Format):
    if format_ == Format.E4M3:
        return DType.kFloat8E4M3, DType.kFloat8E4M3
    if format_ == Format.E5M2:
        return DType.kFloat8E5M2, DType.kFloat8E5M2
    if format_ == Format.HYBRID:
        return DType.kFloat8E4M3, DType.kFloat8E5M2
    return DType.kBFloat16, DType.kBFloat16


class FP8GemmPackage:
    """
    A container that contains all required data for
    FP8 GEMM
    """

    def __init__(
        self,
        num_of_gemm: int,
        inputs: jnp.ndarray,
        kernels: List[jnp.ndarray],
        fp8_max: jnp.ndarray,
        amax: jnp.ndarray,
        scale: jnp.ndarray,
        scale_inv: jnp.ndarray,
    ) -> None:
        self._num_of_gemm = num_of_gemm
        self._inputs = inputs

        assert len(kernels) == self._num_of_gemm
        self._kernels = kernels

        total_num_of_meta = self._num_of_gemm * FP8Helper.NUM_META_PER_GEMM
        assert fp8_max.shape[0] == total_num_of_meta
        self._fp8_max = fp8_max
        assert amax.shape[0] == total_num_of_meta
        self._amax = amax
        assert scale.shape[0] == total_num_of_meta
        self._scale = scale
        assert scale_inv.shape[0] == total_num_of_meta
        self._scale_inv = scale_inv

    @property
    def num_of_gemm(self) -> int:
        """
        num_of_gemm of this package
        """
        return self._num_of_gemm

    @property
    def inputs(self) -> jnp.ndarray:
        """
        inputs of this package
        """
        return self._inputs

    @property
    def kernels(self) -> List[jnp.ndarray]:
        """
        kernels of this package
        """
        return self._kernels

    @property
    def fp8_max(self) -> jnp.ndarray:
        """
        fp8_max of this package
        """
        return self._fp8_max

    @property
    def amax(self) -> jnp.ndarray:
        """
        amax of this package
        """
        return self._amax

    @property
    def scale(self) -> jnp.ndarray:
        """
        scale of this package
        """
        return self._scale

    @property
    def scale_inv(self) -> jnp.ndarray:
        """
        scale_inv of this package
        """
        return self._scale_inv


class AmaxComputeAlgo(Enum):
    """AmaxComputeAlgo."""
    MAX = "max"
    MOST_RECENT = "most_recent"


class FP8Helper:
    """
    FP8 helper to manage the FP8 meta
    """
    INITIALIZED = False
    MARGIN: float = 0.0
    FP8_FORMAT: Format = Format.HYBRID
    FWD_DTYPE: DType = DType.kFloat8E4M3
    BWD_DTYPE: DType = DType.kFloat8E5M2
    UPDATE_FP8META_INTERVAL: int = 1
    AMAX_HISTORY_LEN: int = 1
    AMAX_COMPUTE_ALGO: AmaxComputeAlgo = AmaxComputeAlgo.MOST_RECENT
    NUM_META_PER_GEMM: int = 3
    INPUT_META_IDX_PER_GEMM: int = 0
    KERNEL_META_IDX_PER_GEMM: int = 1
    GRAD_META_IDX_PER_GEMM: int = 2
    FP8_COLLECTION_NAME: str = "fp8_meta_collection"
    FP8_AMAX_NAME: str = "fp8_meta_amax"
    FP8_SCALE_NAME: str = "fp8_meta_scale"
    FP8_SCALE_INV_NAME: str = "fp8_meta_scale_inv"
    FP8_MAX_NAME: str = "fp8_max"
    FP8_2X_ACC_FPROP_ENV_VAR_NAME = "NVTE_JAX_FP8_2X_ACC_FPROP"
    FP8_2X_ACC_DGRAD_ENV_VAR_NAME = "NVTE_JAX_FP8_2X_ACC_DGRAD"
    FP8_2X_ACC_WGRAD_ENV_VAR_NAME = "NVTE_JAX_FP8_2X_ACC_WGRAD"
    FP8_2X_ACC_FPROP: bool = False
    FP8_2X_ACC_DGRAD: bool = True
    FP8_2X_ACC_WGRAD: bool = True

    @staticmethod
    def enable_fp8():
        """
        Indicate if fp8 training is enable or not.
        """
        return FP8Helper.INITIALIZED

    @staticmethod
    def initialize(margin: float = 0.0,
                   fp8_format: Format = Format.HYBRID,
                   update_fp8meta_interval: int = 1,
                   amax_history_len: int = 1,
                   amax_compute_algo: AmaxComputeAlgo = AmaxComputeAlgo.MOST_RECENT) -> None:
        """
        Initialize the FP8 meta
        """
        FP8Helper.INITIALIZED = True
        FP8Helper.MARGIN = margin
        FP8Helper.FP8_FORMAT = fp8_format
        FP8Helper.FWD_DTYPE, FP8Helper.BWD_DTYPE = \
            _format2dtypes(FP8Helper.FP8_FORMAT)
        FP8Helper.UPDATE_FP8META_INTERVAL = update_fp8meta_interval
        FP8Helper.AMAX_HISTORY_LEN = amax_history_len
        FP8Helper.AMAX_COMPUTE_ALGO = amax_compute_algo
        FP8Helper.FP8_2X_ACC_FPROP = False
        FP8Helper.FP8_2X_ACC_DGRAD = bool(
            int(os.environ.get(FP8Helper.FP8_2X_ACC_DGRAD_ENV_VAR_NAME, True)))
        FP8Helper.FP8_2X_ACC_WGRAD = bool(
            int(os.environ.get(FP8Helper.FP8_2X_ACC_WGRAD_ENV_VAR_NAME, True)))

    @staticmethod
    def finalize() -> None:
        """
        FP8 helper finalize
        """
        FP8Helper.INITIALIZED = False
        FP8Helper.MARGIN = 0.0
        FP8Helper.FP8_FORMAT = Format.HYBRID
        FP8Helper.FWD_DTYPE = DType.kFloat8E4M3
        FP8Helper.BWD_DTYPE = DType.kFloat8E5M2
        FP8Helper.UPDATE_FP8META_INTERVAL = 1
        FP8Helper.AMAX_HISTORY_LEN = 1

    @staticmethod
    def update_amax_history(amax_buffers: jnp.ndarray) -> jnp.ndarray:
        """
        Update the amax history
        """
        updated_amax_buffers = jnp.roll(amax_buffers, -1, 1)
        updated_amax_buffers.at[:, 0].set(0)
        return updated_amax_buffers

    @staticmethod
    def update_collections(new: Collection, original: Collection) -> Collection:
        """
        Update the collections
        """
        if not isinstance(original, FrozenDict):
            original = FrozenDict(original)
        for key in new:
            if key in original:
                original, _ = original.pop(key)
        return FrozenDict({**new, **original})

    @staticmethod
    def update_fp8_metas(state: Collection) -> Collection:
        """
        Update the FP8 metas
        """
        if FP8Helper.FP8_COLLECTION_NAME in state:
            if not isinstance(state, FrozenDict):
                state = FrozenDict(state)
            others, fp8_metas = state.pop(FP8Helper.FP8_COLLECTION_NAME)
            fp8_metas = FP8Helper._update_fp8_metas_impl(fp8_metas)
            return FrozenDict({**others, FP8Helper.FP8_COLLECTION_NAME: fp8_metas})
        return state

    @staticmethod
    def generate_fp8_max_array(num_of_meta):
        """
        Generate the FP8 max array
        """
        num_of_gemm = num_of_meta // FP8Helper.NUM_META_PER_GEMM
        fp8_max_fwd = FP8Helper.FP8_FORMAT.value.max_fwd
        fp8_max_bwd = FP8Helper.FP8_FORMAT.value.max_bwd
        fp8_max_per_gemm = []
        for i in range(FP8Helper.NUM_META_PER_GEMM):
            val = fp8_max_bwd if i == FP8Helper.GRAD_META_IDX_PER_GEMM \
                else fp8_max_fwd
            fp8_max_per_gemm.append([val])
        fp8_max_per_gemm = jnp.asarray(fp8_max_per_gemm, dtype=jnp.float32)
        return jnp.vstack([fp8_max_per_gemm] * num_of_gemm)

    @staticmethod
    def get_fp8_meta_indices(gemm_idx: int) -> Tuple[int]:
        """
        Obtain the index about FP8 metas by the given GEMM index.
        """
        input_idx = FP8Helper.NUM_META_PER_GEMM * gemm_idx + FP8Helper.INPUT_META_IDX_PER_GEMM
        kernel_idx = FP8Helper.NUM_META_PER_GEMM * gemm_idx + FP8Helper.KERNEL_META_IDX_PER_GEMM
        grad_idx = FP8Helper.NUM_META_PER_GEMM * gemm_idx + FP8Helper.GRAD_META_IDX_PER_GEMM
        return input_idx, kernel_idx, grad_idx

    @staticmethod
    @jax.jit
    def _update_fp8_metas_impl(fp8_metas: Collection) -> Collection:
        fp8_meta_arrays, treedef = jax.tree_util.tree_flatten(fp8_metas)
        num_of_meta_with_max = FP8Helper.NUM_META_PER_GEMM + 1
        num_of_gemm = len(fp8_meta_arrays) // num_of_meta_with_max
        for i in range(num_of_gemm):
            # flattern array is ordered in alphabetical order of collection names
            fp8_max_idx = i * num_of_meta_with_max
            fp8_amax_idx = fp8_max_idx + 1
            fp8_scale_idx = fp8_amax_idx + 1
            fp8_scale_inv_idx = fp8_scale_idx + 1

            fp8_max = fp8_meta_arrays[fp8_max_idx]
            if FP8Helper.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX:
                amax = jnp.max(fp8_meta_arrays[fp8_amax_idx], axis=1, keepdims=True)
            else:
                amax = fp8_meta_arrays[fp8_amax_idx][:, 0:1]
            scale = fp8_meta_arrays[fp8_scale_idx]

            exp = jnp.floor(jnp.log2(fp8_max / amax)) - FP8Helper.MARGIN
            sf = jnp.round(jnp.power(2, jnp.abs(exp)))
            sf = jnp.where(amax > 0.0, sf, scale)
            sf = jnp.where(jnp.isfinite(amax), sf, scale)
            scale = jnp.where(exp < 0, 1 / sf, sf)
            fp8_meta_arrays[fp8_scale_idx] = scale
            fp8_meta_arrays[fp8_scale_inv_idx] = 1 / scale

        return jax.tree_util.tree_unflatten(treedef, fp8_meta_arrays)


@contextmanager
def fp8_autocast(enabled: bool = False,
                 fp8_recipe: Optional[DelayedScaling] = None,
                 sharding_resource: Optional[ShardingResource] = None) -> None:
    r"""
    Context manager for FP8 usage.

    .. code-block:: python

        mesh_shape = (4, 2)
        dp_mesh_axis_name = 'data_parallel'
        tp_mesh_axis_name = 'tensor_parallel'
        devices = np.asarray(jax.devices()).reshape(*mesh_shape)

        with maps.Mesh(devices, (dp_mesh_axis_name, tp_mesh_axis_name)):
            sharding_resource=ShardingResource(dp_mesh_axis_name, tp_mesh_axis_name)

            with fp8_autocast(enabled=True, sharding_resource=sharding_resource):
                rules = extend_logical_axis_rules(tuple())
                transformer = TransformerLayer()

                with partitioning.axis_rules(rules):
                    pjit(transformer.init, ...)(...)

    .. note::
        We only support :attr:`margin`, :attr:`fp8_format`, :attr:`interval` and
        :attr:`amax_history_len` in recipe.DelayedScaling currently. Other parameters
        in recipe.DelayedScaling would be ignored, even is set.

    Parameters
    ----------
    enabled: bool, default = False
        whether or not to enable fp8
    fp8_recipe: recipe.DelayedScaling, default = None
        recipe used for FP8 training.
    sharding_resource: ShardingResource, defaule = None
        specify the mesh axes for data and tensor parallelism to shard along.
        If set to None, then ShardingResource() would be created.
    """
    if fp8_recipe is None:
        fp8_recipe = DelayedScaling()

    if sharding_resource is None:
        sharding_resource = ShardingResource()

    try:
        with global_shard_guard(sharding_resource):
            if enabled:
                amax_compute_algo = AmaxComputeAlgo.MOST_RECENT
                if fp8_recipe.amax_compute_algo == 'max':
                    amax_compute_algo = AmaxComputeAlgo.MAX

                FP8Helper.initialize(margin=fp8_recipe.margin,
                                     fp8_format=fp8_recipe.fp8_format,
                                     update_fp8meta_interval=fp8_recipe.interval,
                                     amax_history_len=fp8_recipe.amax_history_len,
                                     amax_compute_algo=amax_compute_algo)
            yield
    finally:
        FP8Helper.finalize()


# Function Wrappers
def update_collections(new: Collection, original: Collection) -> Collection:
    r"""
    A helper to update Flax's Collection. Collection is a union type of dict and
    Flax's FrozenDict.

    Collection = [dict, FrozenDict]

    Parameters
    ----------
    new: Collection
        A collection that includes new data.
    original: Collection
        The base collection.

    Returns
    -------
    outputs : Collection
        The updated collection.
    """
    return FP8Helper.update_collections(new, original)


def update_fp8_metas(state: Collection) -> Collection:
    r"""
    Calculate new fp8 scales and its inverse via the followed formula

    `exp` = floor(log2(`fp8_max` / `amax`)) - `margin`
    `sf` = round(power(2, abs(exp)))
    `sf` = `sf` if `amax` > 0.0, else original_scale
    `sf` = `sf` if isfinite(`amax`), else original_scale)
    `updated_scale` = `1/sf` if exp < 0, else `sf`
    `updated_scale_inv` = `1/updated_scale`

    Collection = [dict, FrozenDict]

    Parameters
    ----------
    state: Collection
        A collection that includes FP8 metas.

    Returns
    -------
    outputs : Collection
        The collection with updated FP8 metas.
    """
    return FP8Helper.update_fp8_metas(state)
