# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""
DEPRECATED in favor of `transformer_engine.pytorch.quantization.py`.
"""

# pylint: disable=wrong-import-position,unused-import

import warnings

warnings.warn(
    "Using deprecated internal API from Transformer Engine. "
    "transformer_engine.pytorch.fp8 will be removed in a "
    "future release.",
    DeprecationWarning,
    stacklevel=2,
)


# There are some users indirectly importing these classes
# from fp8.py. This ensure backwards compatibility.
# https://github.com/Lightning-AI/lightning-thunder/pull/2635.
from transformer_engine.common.recipe import (
    Recipe,
    DelayedScaling,
    Format,
    MXFP8BlockScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    NVFP4BlockScaling,
    CustomRecipe,
)


# Importing each function instead of 'import *' allows us specify '__all__' in
# quantize.py and also makes any newer additions to quantize.py invisible via
# fp8.py so that we don't reinforce importing internal TE functions.
from .quantization import (
    check_fp8_support,
    check_mxfp8_support,
    check_nvfp4_support,
    check_fp8_block_scaling_support,
    check_recipe_support,
    get_default_fp8_recipe,
    get_fp8_torch_dtype,
    get_fp8_te_dtype,
    get_fp4_te_dtype,
    get_fp8_max,
    FP8GlobalStateManager,
    fp8_model_init,
    fp8_autocast,
    _update_amax_history,
    _default_get_amax_and_update_history,
    _default_sf_compute,
    _compute_amax_and_update_history,
    _compute_scaling_factor,
    _amax_and_scale_update,
    split_and_copy,
    RecipeState,
    DelayedScalingRecipeState,
    Float8CurrentScalingRecipeState,
    MXFP8BlockScalingRecipeState,
    Float8BlockScalingRecipeState,
    NVFP4BlockScalingRecipeState,
    CustomRecipeState,
)
