# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""
Helper module for fp4 meta management
"""
from contextlib import contextmanager
from enum import Enum
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
from flax.linen import fp4_ops

from transformer_engine.transformer_engine_jax import DType
from transformer_engine.transformer_engine_jax import get_cublasLt_version
from transformer_engine.transformer_engine_jax import (
    get_cuda_version,
    get_device_compute_capability,
)
from transformer_engine.common.recipe import DelayedScaling, Format
from transformer_engine.jax.sharding import global_shard_guard
from transformer_engine.jax.sharding import MeshResource

_is_fp4_available = None
_reason_for_no_fp4 = ""
Collection = Union[Dict, FrozenDict]

def _check_fp4_support(gpu_id) -> Tuple[bool, str]:
    """Return if fp4 support is available"""
    gpu_arch = get_device_compute_capability(gpu_id)
    if gpu_arch >= 100:  # hopper and above
        return True, ""
    if gpu_arch < 100:  # pre-ada
        return False, "Device compute capability 8.9 or lower not supported."
    return True, ""

def is_fp4_available(gpu_id=None) -> Tuple[bool, str]:
    """Return if fp4 support is available"""
    if gpu_id is not None:
        return _check_fp4_support(gpu_id)

    global _is_fp4_available, _reason_for_no_fp4
    if _is_fp4_available is None:
        _is_fp4_available = True
        # JAX doesn't provide the local GPU id.
        for local_gpu_id in range(len(jax.local_devices())):
            ret, msg = _check_fp4_support(local_gpu_id)
            if ret is False:
                _is_fp4_available = ret
                _reason_for_no_fp4 = msg
            break

    return _is_fp4_available, _reason_for_no_fp4

def _format2dtypes(format_: Format):
    if format_ == Format.E4M3:
        return jnp.float8_e4m3fn, jnp.float8_e4m3fn
    if format_ == Format.E5M2:
        return jnp.float8_e5m2, jnp.float8_e5m2
    if format_ == Format.HYBRID:
        return jnp.float8_e4m3fn, jnp.float8_e5m2
    return jnp.bfloat16, jnp.bfloat16


# fm32 is a custom dtype to specify the "add" rules as max operation.
# This is typically used in Pipeline Parallelism + "MiconBatching > 1",
# which is implemented via nn.scan. Without this custom dtype, nn.scan
# would sum gradients from all micro-batches, and this is not the expected
# behavior for fp4 meta. Instead, the summation of fp4 meta gradients should
# be "MAX".
FlaxFloatMeta32 = fp4_ops.fm32


class fp4MetaPackage:
    """
    A container that contains all required meta data for fp4
    """

    NUM_OF_META: int = 3
    INPUT_IDX: int = 0
    WEIGHT_IDX: int = 1
    GRAD_IDX: int = 2

    def __init__(
        self,
        input_amax: jnp.ndarray,
        input_scale: jnp.ndarray,
        weight_amax: jnp.ndarray,
        weight_scale: jnp.ndarray,
        grad_amax: jnp.ndarray,
        grad_scale: jnp.ndarray,
    ) -> None:

        self._amax_list = [None] * fp4MetaPackage.NUM_OF_META
        self._scale_list = [None] * fp4MetaPackage.NUM_OF_META

        self._amax_list[fp4MetaPackage.INPUT_IDX] = input_amax
        self._scale_list[fp4MetaPackage.INPUT_IDX] = input_scale
        self._amax_list[fp4MetaPackage.WEIGHT_IDX] = weight_amax
        self._scale_list[fp4MetaPackage.WEIGHT_IDX] = weight_scale
        self._amax_list[fp4MetaPackage.GRAD_IDX] = grad_amax
        self._scale_list[fp4MetaPackage.GRAD_IDX] = grad_scale

    @property
    def amax_list(self) -> List[jnp.ndarray]:
        """
        Get the amax list of this package.
        """
        return self._amax_list

    @property
    def scale_list(self) -> List[jnp.ndarray]:
        """
        Get the scale list of this package.
        """
        return self._scale_list

    @staticmethod
    def update_amax_list(amax_list: List[jnp.ndarray]) -> jnp.ndarray:
        """
        Update the amax history list
        """
        updated_amax_list = [fp4Helper.update_amax_history(amax) for amax in amax_list]
        return updated_amax_list

    @staticmethod
    def update_fp4_scale(
        amax_list: List[jnp.ndarray], scale_list: List[jnp.ndarray], fp4_dtype_list: List[DType]
    ) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
        """
        Get update scale and scale_inv list
        """
        update_scale_list = []
        update_scale_inv_list = []
        for amax, scale, fp4_dtype in zip(amax_list, scale_list, fp4_dtype_list):
            upadted_scale, updated_scale_inv = fp4Helper.update_fp4_scale(amax, scale, fp4_dtype)
            update_scale_list.append(upadted_scale)
            update_scale_inv_list.append(updated_scale_inv)
        return update_scale_list, update_scale_inv_list


class AmaxComputeAlgo(Enum):
    """AmaxComputeAlgo."""

    MAX = "max"
    MOST_RECENT = "most_recent"


NVTE_fp4_COLLECTION_NAME = "fp4_metas"


class fp4Helper:
    """
    fp4 helper to manage the fp4 meta
    """

    INITIALIZED = False
    MARGIN: float = 0.0
    fp4_FORMAT: Format = Format.HYBRID
    FWD_DTYPE: DType = _format2dtypes(Format.HYBRID)[0]
    BWD_DTYPE: DType = _format2dtypes(Format.HYBRID)[1]
    AMAX_HISTORY_LEN: int = 1024
    AMAX_COMPUTE_ALGO: AmaxComputeAlgo = AmaxComputeAlgo.MAX
    fp4_COLLECTION_NAME: str = NVTE_fp4_COLLECTION_NAME
    fp4_AMAX_NAME: str = "amax"
    fp4_SCALE_NAME: str = "scale"
    fp4_2X_ACC_FPROP: bool = False
    fp4_2X_ACC_DGRAD: bool = True
    fp4_2X_ACC_WGRAD: bool = True

    @staticmethod
    def is_fp4_enabled():
        """
        Indicate if fp4 training is enable or not.
        """
        return fp4Helper.INITIALIZED

    @staticmethod
    def initialize(
        margin: float = 0.0,
        fp4_format: Format = Format.HYBRID,
        amax_history_len: int = 1,
        amax_compute_algo: AmaxComputeAlgo = AmaxComputeAlgo.MAX,
    ) -> None:
        """
        Initialize the fp4 meta
        """
        fp4Helper.INITIALIZED = True
        fp4Helper.MARGIN = margin
        fp4Helper.fp4_FORMAT = fp4_format
        fp4Helper.FWD_DTYPE, fp4Helper.BWD_DTYPE = _format2dtypes(fp4Helper.fp4_FORMAT)
        fp4Helper.AMAX_HISTORY_LEN = amax_history_len
        fp4Helper.AMAX_COMPUTE_ALGO = amax_compute_algo
        fp4Helper.fp4_2X_ACC_FPROP = False
        fp4Helper.fp4_2X_ACC_DGRAD = True
        fp4Helper.fp4_2X_ACC_WGRAD = True

    @staticmethod
    def finalize() -> None:
        """
        fp4 helper finalize
        """
        fp4Helper.INITIALIZED = False
        fp4Helper.MARGIN = 0.0
        fp4Helper.fp4_FORMAT = Format.HYBRID
        fp4Helper.FWD_DTYPE, fp4Helper.BWD_DTYPE = _format2dtypes(fp4Helper.fp4_FORMAT)
        fp4Helper.AMAX_HISTORY_LEN = 1024
        fp4Helper.AMAX_COMPUTE_ALGO = AmaxComputeAlgo.MAX

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
    def generate_fp4_meta_dtype_converter_pair(*args):
        """
        Generate a pair of conversion fun in-between fm32 and fp32.
        """

        def identical_fun(*metas):
            return list(metas)

        def fm32_to_fp32_fun(*metas):
            for meta in metas:
                assert meta.dtype == FlaxFloatMeta32
            return [jax.lax.convert_element_type(meta, jnp.float32) for meta in metas]

        def fp32_to_fm32_fun(*metas):
            for meta in metas:
                assert meta.dtype == jnp.float32
            return [jax.lax.convert_element_type(meta, FlaxFloatMeta32) for meta in metas]

        # Make functions to be a vaild JAX type
        partial_identical_fun = jax.tree_util.Partial(identical_fun)
        partial_fm32_to_fp32_fun = jax.tree_util.Partial(fm32_to_fp32_fun)
        partial_fp32_to_fm32_fun = jax.tree_util.Partial(fp32_to_fm32_fun)

        if len(args) < 1:
            return partial_identical_fun, partial_identical_fun

        original_dtype = args[0].dtype
        for arg in args:
            assert arg.dtype == original_dtype

        if original_dtype == FlaxFloatMeta32:
            return partial_fm32_to_fp32_fun, partial_fp32_to_fm32_fun

        return partial_identical_fun, partial_identical_fun

    @staticmethod
    @jax.jit
    def update_amax_history(amax: jnp.ndarray) -> jnp.ndarray:
        """
        Update the amax history
        """
        updated_amax = jnp.roll(amax, -1, -1)
        updated_amax = updated_amax.at[0].set(0)
        return updated_amax

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def update_fp4_scale(amax: jnp.ndarray, scale: jnp.ndarray, fp4_dtype: DType) -> jnp.ndarray:
        """
        Calculate fp4 scale and scale_inv based on given amax.
        """
        fp4_max = jnp.astype(jnp.finfo(fp4_dtype).max, jnp.float32)

        if fp4Helper.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX:
            amax = jnp.max(amax, axis=-1, keepdims=True)
        else:
            amax = amax[0:1]

        sf = (fp4_max / amax) / (2**fp4Helper.MARGIN)
        sf = jnp.where(amax > 0.0, sf, scale)
        sf = jnp.where(jnp.isfinite(amax), sf, scale)
        scale = sf
        scale_inv = 1 / sf

        return scale, scale_inv


@contextmanager
def fp4_autocast(
    enabled: bool = False,
    fp4_recipe: Optional[DelayedScaling] = None,
    mesh_resource: Optional[MeshResource] = None,
) -> None:
    r"""
    Context manager for fp4 usage.

    .. code-block:: python

        mesh_shape = (4, 2)
        dp_mesh_axis_name = 'data_parallel'
        tp_mesh_axis_name = 'tensor_parallel'
        devices = np.asarray(jax.devices()).reshape(*mesh_shape)

        with maps.Mesh(devices, (dp_mesh_axis_name, tp_mesh_axis_name)):
            mesh_resource=MeshResource(dp_mesh_axis_name, tp_mesh_axis_name)

            with fp4_autocast(enabled=True, mesh_resource=mesh_resource):
                rules = extend_logical_axis_rules(tuple())
                transformer = TransformerLayer()

                with partitioning.axis_rules(rules):
                    pjit(transformer.init, ...)(...)

    .. note::
        We only support :attr:`margin`, :attr:`fp4_format`, :attr:`amax_history_len`,
        and :attr:`amax_compute_algo` (with value 'max' and 'most_recent') in
        recipe.DelayedScaling currently. Other parameters in recipe.DelayedScaling
        will trigger an assertion.

    Parameters
    ----------
    enabled: bool, default = False
        Whether or not to enable fp4
    fp4_recipe: recipe.DelayedScaling, default = None
        Recipe used for fp4 training.
    mesh_resource: MeshResource, default = None
        Specify the mesh axes for data and tensor parallelism to shard along.
        If set to None, then no data or tensor parallelism will be used.

    """
    if fp4_recipe is None:
        fp4_recipe = DelayedScaling()

    assert fp4_recipe.amax_compute_algo in [
        "max",
        "most_recent",
    ], "DelayedScaling amax_compute_algo only supports max and most_recent with TE/JAX."
    assert (
        fp4_recipe.scaling_factor_compute_algo is None
    ), "DelayedScaling scaling_factor_compute_algo isn't supported by TE/JAX."
    assert fp4_recipe.override_linear_precision == (
        False,
        False,
        False,
    ), "DelayedScaling override_linear_precision isn't supported by TE/JAX."
    assert fp4_recipe.reduce_amax, "DelayedScaling reduce_amax should be enabled for TE/JAX."

    if mesh_resource is None:
        mesh_resource = MeshResource()

    try:
        with global_shard_guard(mesh_resource):
            if enabled:
                fp4_available, reason_for_no_fp4 = is_fp4_available()
                assert fp4_available, reason_for_no_fp4

                amax_compute_algo = AmaxComputeAlgo.MOST_RECENT
                if fp4_recipe.amax_compute_algo == "max":
                    amax_compute_algo = AmaxComputeAlgo.MAX

                fp4Helper.initialize(
                    margin=fp4_recipe.margin,
                    fp4_format=fp4_recipe.fp4_format,
                    amax_history_len=fp4_recipe.amax_history_len,
                    amax_compute_algo=amax_compute_algo,
                )
            yield
    finally:
        fp4Helper.finalize()


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
    return fp4Helper.update_collections(new, original)


def get_delayed_scaling():
    r"""
    Obtain an instance of  DelayedScaling which is set via fp4_autocast.

    .. note::
        We only store :attr:`margin`, :attr:`fp4_format`, :attr:`amax_history_len`
        , and :attr:`amax_compute_algo` via fp4_autocast. Other parameters in
        recipe.DelayedScaling would be returned as the default values.

    Returns
    -------
    delay_scaling : DelayedScaling
        an instance of  DelayedScaling which is set via fp4_autocast.
    """
    amax_compute_algo = (
        "max" if fp4Helper.AMAX_COMPUTE_ALGO is AmaxComputeAlgo.MAX else "most_recent"
    )
    return DelayedScaling(
        margin=int(fp4Helper.MARGIN),
        fp4_format=fp4Helper.fp4_FORMAT,
        amax_history_len=fp4Helper.AMAX_HISTORY_LEN,
        amax_compute_algo=amax_compute_algo,
    )
