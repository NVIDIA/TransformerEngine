# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Helper module for fp8 meta management
"""
from contextlib import contextmanager
from enum import Enum
from typing import Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from transformer_engine_jax import DType
from transformer_engine_jax import get_cublasLt_version
from transformer_engine_jax import get_cuda_version, get_device_compute_capability
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.jax.sharding import global_shard_guard
from transformer_engine.jax.sharding import MeshResource

_is_fp8_available = None
_reason_for_no_fp8 = ""
Collection = Union[Dict, FrozenDict]


def _check_fp8_support(gpu_id) -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    gpu_arch = get_device_compute_capability(gpu_id)
    if gpu_arch >= 90:    # hopper and above
        return True, ""
    if gpu_arch < 89:    # pre-ada
        return False, "Device compute capability 8.9 or higher required for FP8 execution."
    if get_cublasLt_version() < 120103:
        return False, "CublasLt version 12.1.3.x or higher required for FP8 execution on Ada."
    if get_cuda_version() < 12010:
        return False, "Cuda version 12.1 or higher required for FP8 execution on Ada."
    return True, ""


def is_fp8_available(gpu_id=None) -> Tuple[bool, str]:
    """Return if fp8 support is available"""
    if gpu_id is not None:
        return _check_fp8_support(gpu_id)

    global _is_fp8_available, _reason_for_no_fp8
    if _is_fp8_available is None:
        _is_fp8_available = True
        # JAX doesn't provide the local GPU id.
        for local_gpu_id in range(len(jax.local_devices())):
            ret, msg = _check_fp8_support(local_gpu_id)
            if ret is False:
                _is_fp8_available = ret
                _reason_for_no_fp8 = msg
            break

    return _is_fp8_available, _reason_for_no_fp8


def _format2dtypes(format_: Format):
    if format_ == Format.E4M3:
        return jnp.float8_e4m3fn, jnp.float8_e4m3fn
    if format_ == Format.E5M2:
        return jnp.float8_e5m2, jnp.float8_e5m2
    if format_ == Format.HYBRID:
        return jnp.float8_e4m3fn, jnp.float8_e5m2
    return jnp.bfloat16, jnp.bfloat16


class FP8MetaPackage:
    """
    A container that contains all required meta data for FP8
    """

    def __init__(
        self,
        num_of_gemm: int,
        fp8_max: jnp.ndarray,
        amax: jnp.ndarray,
        scale: jnp.ndarray,
        scale_inv: jnp.ndarray,
    ) -> None:
        total_num_of_meta = num_of_gemm * FP8Helper.NUM_META_PER_GEMM
        self._num_of_gemm = num_of_gemm
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

    def get_package_by_gemm_idx(self, gemm_idx):
        """
        Get a sub package by gemm_idx
        """
        assert self.num_of_gemm > gemm_idx

        meta_start_idx = gemm_idx * FP8Helper.NUM_META_PER_GEMM
        meta_end_idx = (gemm_idx + 1) * FP8Helper.NUM_META_PER_GEMM
        return FP8MetaPackage(1, self.fp8_max[meta_start_idx:meta_end_idx],
                              self.amax[meta_start_idx:meta_end_idx],
                              self.scale[meta_start_idx:meta_end_idx],
                              self.scale_inv[meta_start_idx:meta_end_idx])


class AmaxComputeAlgo(Enum):
    """AmaxComputeAlgo."""
    MAX = "max"
    MOST_RECENT = "most_recent"


NVTE_FP8_COLLECTION_NAME = "fp8_meta_collection"


class FP8Helper:
    """
    FP8 helper to manage the FP8 meta
    """
    INITIALIZED = False
    MARGIN: float = 0.0
    FP8_FORMAT: Format = Format.HYBRID
    FWD_DTYPE: DType = _format2dtypes(Format.HYBRID)[0]
    BWD_DTYPE: DType = _format2dtypes(Format.HYBRID)[1]
    UPDATE_FP8META_INTERVAL: int = 1
    AMAX_HISTORY_LEN: int = 1024
    AMAX_COMPUTE_ALGO: AmaxComputeAlgo = AmaxComputeAlgo.MAX
    NUM_META_PER_GEMM: int = 3
    INPUT_META_IDX_PER_GEMM: int = 0
    KERNEL_META_IDX_PER_GEMM: int = 1
    GRAD_META_IDX_PER_GEMM: int = 2
    FP8_COLLECTION_NAME: str = NVTE_FP8_COLLECTION_NAME
    FP8_AMAX_NAME: str = "fp8_meta_amax"
    FP8_SCALE_NAME: str = "fp8_meta_scale"
    FP8_SCALE_INV_NAME: str = "fp8_meta_scale_inv"
    FP8_MAX_NAME: str = "fp8_max"
    FP8_2X_ACC_FPROP: bool = False
    FP8_2X_ACC_DGRAD: bool = True
    FP8_2X_ACC_WGRAD: bool = True

    @staticmethod
    def is_fp8_enabled():
        """
        Indicate if fp8 training is enable or not.
        """
        return FP8Helper.INITIALIZED

    @staticmethod
    def initialize(margin: float = 0.0,
                   fp8_format: Format = Format.HYBRID,
                   update_fp8meta_interval: int = 1,
                   amax_history_len: int = 1,
                   amax_compute_algo: AmaxComputeAlgo = AmaxComputeAlgo.MAX) -> None:
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
        FP8Helper.FP8_2X_ACC_DGRAD = True
        FP8Helper.FP8_2X_ACC_WGRAD = True

    @staticmethod
    def finalize() -> None:
        """
        FP8 helper finalize
        """
        FP8Helper.INITIALIZED = False
        FP8Helper.MARGIN = 0.0
        FP8Helper.FP8_FORMAT = Format.HYBRID
        FP8Helper.FWD_DTYPE, FP8Helper.BWD_DTYPE = \
            _format2dtypes(FP8Helper.FP8_FORMAT)
        FP8Helper.UPDATE_FP8META_INTERVAL = 1
        FP8Helper.AMAX_HISTORY_LEN = 1024
        FP8Helper.AMAX_COMPUTE_ALGO = AmaxComputeAlgo.MAX

    @staticmethod
    def update_collections(new: Collection, original: Collection) -> Collection:
        """
        Update the collections
        """
        assert isinstance(original, (dict, FrozenDict))
        assert isinstance(new, (dict, FrozenDict))
        frozen_original = FrozenDict(original) if not isinstance(original, FrozenDict) else original
        for key in new:
            if key in frozen_original:
                frozen_original, _ = frozen_original.pop(key)
        new_coll = FrozenDict({**new, **frozen_original})
        if not isinstance(original, FrozenDict):
            new_coll = new_coll.unfreeze()
        return new_coll

    @staticmethod
    def update_fp8_metas(state: Collection) -> Collection:
        """
        Update the FP8 metas
        """
        assert isinstance(state, (dict, FrozenDict))
        if FP8Helper.FP8_COLLECTION_NAME in state:
            frozen_state = FrozenDict(state) if not isinstance(state, FrozenDict) else state
            others, fp8_metas = frozen_state.pop(FP8Helper.FP8_COLLECTION_NAME)
            fp8_metas = FP8Helper._update_fp8_metas_impl(fp8_metas)
            new_state = FrozenDict({**others, FP8Helper.FP8_COLLECTION_NAME: fp8_metas})

            if not isinstance(state, FrozenDict):
                new_state = new_state.unfreeze()
            return new_state
        return state

    @staticmethod
    def generate_fp8_max_array(num_of_meta):
        """
        Generate the FP8 max array
        """
        num_of_gemm = num_of_meta // FP8Helper.NUM_META_PER_GEMM
        fp8_max_fwd = jnp.finfo(FP8Helper.FWD_DTYPE).max
        fp8_max_bwd = jnp.finfo(FP8Helper.BWD_DTYPE).max
        fp8_max_per_gemm = []
        for i in range(FP8Helper.NUM_META_PER_GEMM):
            val = fp8_max_bwd if i == FP8Helper.GRAD_META_IDX_PER_GEMM \
                else fp8_max_fwd
            fp8_max_per_gemm.append([val])
        fp8_max_per_gemm = jnp.asarray(fp8_max_per_gemm, dtype=jnp.float32)
        return jnp.vstack([fp8_max_per_gemm] * num_of_gemm)

    @staticmethod
    def get_fp8_meta_indices(gemm_idx: int) -> Tuple[int, int, int]:
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
                amax = jnp.max(fp8_meta_arrays[fp8_amax_idx], axis=-1, keepdims=True)
            else:
                amax = fp8_meta_arrays[fp8_amax_idx][..., 0:1]
            scale = fp8_meta_arrays[fp8_scale_idx]

            sf = (fp8_max / amax) / (2**FP8Helper.MARGIN)
            sf = jnp.where(amax > 0.0, sf, scale)
            sf = jnp.where(jnp.isfinite(amax), sf, scale)
            fp8_meta_arrays[fp8_scale_idx] = sf
            fp8_meta_arrays[fp8_scale_inv_idx] = 1 / sf

        return jax.tree_util.tree_unflatten(treedef, fp8_meta_arrays)

    @staticmethod
    def update_amax_history(amax: jnp.ndarray) -> jnp.ndarray:
        """
        Update the amax history
        """
        updated_amax = jnp.roll(amax, -1, -1)
        updated_amax = updated_amax.at[..., 0].set(0)
        return updated_amax

    @staticmethod
    @jax.jit
    def update_fp8_scale(fp8_max: jnp.ndarray, amax: jnp.ndarray,
                         scale: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate fp8 scale and scale_inv based on given amax.
        """
        if FP8Helper.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX:
            amax = jnp.max(amax, axis=-1, keepdims=True)
        else:
            amax = amax[..., 0:1]

        sf = (fp8_max / amax) / (2**FP8Helper.MARGIN)
        sf = jnp.where(amax > 0.0, sf, scale)
        sf = jnp.where(jnp.isfinite(amax), sf, scale)
        scale = sf
        scale_inv = 1 / sf

        return scale, scale_inv


@contextmanager
def fp8_autocast(enabled: bool = False,
                 fp8_recipe: Optional[DelayedScaling] = None,
                 mesh_resource: Optional[MeshResource] = None) -> None:
    r"""
    Context manager for FP8 usage.

    .. code-block:: python

        mesh_shape = (4, 2)
        dp_mesh_axis_name = 'data_parallel'
        tp_mesh_axis_name = 'tensor_parallel'
        devices = np.asarray(jax.devices()).reshape(*mesh_shape)

        with maps.Mesh(devices, (dp_mesh_axis_name, tp_mesh_axis_name)):
            mesh_resource=MeshResource(dp_mesh_axis_name, tp_mesh_axis_name)

            with fp8_autocast(enabled=True, mesh_resource=mesh_resource):
                rules = extend_logical_axis_rules(tuple())
                transformer = TransformerLayer()

                with partitioning.axis_rules(rules):
                    pjit(transformer.init, ...)(...)

    .. note::
        We only support :attr:`margin`, :attr:`fp8_format`,
        :attr:`interval`, :attr:`amax_history_len` and
        :attr:`amax_compute_algo`(with value 'max' and 'most_recent')
        in recipe.DelayedScaling currently. Other parameters in
        recipe.DelayedScaling will trigger an assertion.

    Parameters
    ----------
    enabled: bool, default = False
        Whether or not to enable fp8
    fp8_recipe: recipe.DelayedScaling, default = None
        Recipe used for FP8 training.
    mesh_resource: MeshResource, default = None
        Specify the mesh axes for data and tensor parallelism to shard along.
        If set to None, then no data or tensor parallelism will be used.

    """
    if fp8_recipe is None:
        fp8_recipe = DelayedScaling()

    assert fp8_recipe.amax_compute_algo in [
        "max", "most_recent"
    ], ("DelayedScaling amax_compute_algo only supports max and most_recent with TE/JAX.")
    assert fp8_recipe.scaling_factor_compute_algo is None, (
        "DelayedScaling scaling_factor_compute_algo isn't supported by TE/JAX.")
    assert fp8_recipe.override_linear_precision == (False, False, False), (
        "DelayedScaling override_linear_precision isn't supported by TE/JAX.")
    assert fp8_recipe.reduce_amax, ("DelayedScaling reduce_amax should be enabled for TE/JAX.")

    if mesh_resource is None:
        mesh_resource = MeshResource()

    try:
        with global_shard_guard(mesh_resource):
            if enabled:
                fp8_available, reason_for_no_fp8 = is_fp8_available()
                assert fp8_available, reason_for_no_fp8

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
def update_collections(new: Collection, original: Collection) -> FrozenDict:
    r"""
    A helper to update Flax's Collection.

    Collection = [dict, flax.core.frozen_dict.FrozenDict]

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

    .. code-block:: python

        sf = (fp8_max / amax) / (2 ^ margin)
        sf = sf if amax > 0.0, else original_scale
        updated_scale = sf if isfinite(amax), else original_scale)
        updated_scale_inv = 1/updated_scale

    Collection = [dict, flax.core.frozen_dict.FrozenDict]

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


def get_delayed_scaling():
    r"""
    Obtain an instance of  DelayedScaling which is set via fp8_autocast.

    .. note::
        We only store :attr:`margin`, :attr:`fp8_format`, :attr:`interval`,
        :attr:`amax_history_len` and :attr:`amax_compute_algo` via fp8_autocast.
        Other parameters in recipe.DelayedScaling would be returned as the default
        values.

    Returns
    -------
    delay_scaling : DelayedScaling
        an instance of  DelayedScaling which is set via fp8_autocast.
    """
    amax_compute_algo = "max" if FP8Helper.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX \
                        else "most_recent"
    return DelayedScaling(margin=int(FP8Helper.MARGIN),
                          interval=FP8Helper.UPDATE_FP8META_INTERVAL,
                          fp8_format=FP8Helper.FP8_FORMAT,
                          amax_history_len=FP8Helper.AMAX_HISTORY_LEN,
                          amax_compute_algo=amax_compute_algo)
