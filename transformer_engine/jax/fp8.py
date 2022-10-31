# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
Helper module for fp8 meta management
"""
from contextlib import contextmanager
from typing import Any, Optional
import flax
import jax
import jax.numpy as jnp
from transformer_engine_jax import DType
from transformer_engine.common.recipe import DelayedScaling, Format

Collection = Any


def _get_ctypes(format_: Format):
    if format_ == Format.E4M3:
        return DType.kFloat8E4M3, DType.kFloat8E4M3
    if format_ == Format.E5M2:
        return DType.kFloat8E5M2, DType.kFloat8E5M2
    if format_ == Format.HYBRID:
        return DType.kFloat8E4M3, DType.kFloat8E5M2
    return DType.kBFloat16, DType.kBFloat16


class FP8Helper():
    """
    FP8 helper to manage the FP8 meta
    """
    INITIALIZED = False
    MARGIN: float = 0.0
    FP8_FORMAT: Format = Format.HYBRID
    FWD_CTYPE: DType = DType.kFloat8E4M3
    BWD_CTYPE: DType = DType.kFloat8E5M2
    UPDATE_FP8META_INTERVAL: int = 1
    AMAX_HISTORY_SIZE: int = 1
    NUM_META_PER_GEMM: int = 3
    INPUT_META_IDX_PER_GEMM: int = 0
    KERNEL_META_IDX_PER_GEMM: int = 1
    GRAD_META_IDX_PER_GEMM: int = 2
    FP8_COLLECTION_NAME: str = "fp8_meta_collection"
    FP8_AMAX_NAME: str = "fp8_meta_amax"
    FP8_SCALE_NAME: str = "fp8_meta_scale"
    FP8_SCALE_INV_NAME: str = "fp8_meta_scale_inv"
    FP8_MAX_NAME: str = "fp8_max"

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
                   amax_history_size: int = 1) -> None:
        """
        Initialize the FP8 meta
        """
        FP8Helper.INITIALIZED = True
        FP8Helper.MARGIN = margin
        FP8Helper.FP8_FORMAT = fp8_format
        FP8Helper.FWD_CTYPE, FP8Helper.BWD_CTYPE = \
            _get_ctypes(FP8Helper.FP8_FORMAT)
        FP8Helper.UPDATE_FP8META_INTERVAL = update_fp8meta_interval
        FP8Helper.AMAX_HISTORY_SIZE = amax_history_size

    @staticmethod
    def finalize() -> None:
        """
        FP8 helper finalize
        """
        FP8Helper.INITIALIZED = False
        FP8Helper.MARGIN = 0.0
        FP8Helper.FP8_FORMAT = Format.HYBRID
        FP8Helper.FWD_CTYPE = DType.kFloat8E4M3
        FP8Helper.BWD_CTYPE = DType.kFloat8E5M2
        FP8Helper.UPDATE_FP8META_INTERVAL = 1
        FP8Helper.AMAX_HISTORY_SIZE = 1

    @staticmethod
    def update_fp8_metas(state: Collection) -> Collection:
        """
        Update the FP8 metas
        """
        if FP8Helper.FP8_COLLECTION_NAME in state:
            others, fp8_metas = state.pop(FP8Helper.FP8_COLLECTION_NAME)
            fp8_metas = FP8Helper._update_fp8_metas_impl(fp8_metas)
            return flax.core.frozen_dict.FrozenDict({
                **others, FP8Helper.FP8_COLLECTION_NAME:
                fp8_metas
            })
        return state

    @staticmethod
    @jax.jit
    def sync_fp8_metas(fp8_metas: Collection) -> None:
        """
        Sync the FP8 metas
        """
        raise NotImplementedError

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
    def update_collections(new: Collection, original: Collection) -> None:
        """
        Update the collections
        """
        for key in new:
            if key in original:
                original, _ = original.pop(key)
        return flax.core.frozen_dict.FrozenDict({**new, **original})

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

            exp = jnp.floor(
                jnp.log2(fp8_meta_arrays[fp8_max_idx] /
                         fp8_meta_arrays[fp8_amax_idx])) - FP8Helper.MARGIN
            sf = jnp.round(jnp.power(2, jnp.abs(exp)))
            sf = jnp.where(fp8_meta_arrays[fp8_amax_idx] > 0.0, sf,
                           fp8_meta_arrays[fp8_scale_idx])
            sf = jnp.where(jnp.isfinite(fp8_meta_arrays[fp8_amax_idx]), sf,
                           fp8_meta_arrays[fp8_scale_idx])
            fp8_meta_arrays[fp8_scale_idx] = jnp.where(exp < 0, 1 / sf, sf)
            fp8_meta_arrays[
                fp8_scale_inv_idx] = 1 / fp8_meta_arrays[fp8_scale_idx]

        return jax.tree_util.tree_unflatten(treedef, fp8_meta_arrays)


@contextmanager
def fp8_autocast(enabled: bool = False,
                 fp8_recipe: Optional[DelayedScaling] = None) -> None:
    """
    Context manager for FP8 usage.

    Parameters
    ----------
    enabled: bool, default = `False`
        whether or not to enable fp8
    fp8_recipe: recipe.DelayedScaling, default = `None`
        recipe used for FP8 training.
    """
    if fp8_recipe is None:
        fp8_recipe = DelayedScaling()

    assert fp8_recipe.amax_history_len == 1, \
        "It only support amax_history_len == 1 for now."

    try:
        if enabled:
            FP8Helper.initialize(margin=fp8_recipe.margin,
                                 fp8_format=fp8_recipe.fp8_format,
                                 update_fp8meta_interval=fp8_recipe.interval,
                                 amax_history_size=fp8_recipe.amax_history_len)
        yield
    finally:
        FP8Helper.finalize()
