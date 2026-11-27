..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Common API
==========

FP8 GEMMs use accurate accumulation automatically, including split
accumulation on Hopper. There is no accumulation-precision setting to tune.
Blackwell and newer GPUs, FP8 block scaling, and MXFP8 do not require this
legacy option.

.. versionchanged:: 3.0

   Removed ``MMParams`` from ``transformer_engine.common.recipe`` and the
   ``fp8_gemm_fprop``, ``fp8_gemm_dgrad``, and ``fp8_gemm_wgrad`` recipe fields.
   Remove these options when constructing recipes. GEMM APIs no longer accept
   ``use_split_accumulator``; remove that argument from Python and C/C++ calls
   and rebuild native consumers. The corresponding matmul configuration
   attributes and C++ setters have also been removed.

   Custom recipe GEMMs no longer receive ``use_split_accumulator`` in
   ``transformer_engine.pytorch.custom_recipes.gemm.MMParams``. Custom
   implementations should use accurate accumulation, for example
   ``torch._scaled_mm(..., use_fast_accum=False)``.

   Checkpoints containing pickled legacy ``MMParams`` recipe objects require
   conversion before loading with TE 3.0.

   JAX always uses accurate FP8 accumulation, including its native fallback.
   ``NVTE_FP8_GEMM_HIGH_PRECISION_ACCUMULATION`` is no longer used.

.. autoapiclass:: transformer_engine.common.recipe.Format

.. autoapiclass:: transformer_engine.common.recipe.DelayedScaling(margin=0, fp8_format=Format.HYBRID, amax_history_len=1024, amax_compute_algo="max", scaling_factor_compute_algo=None)

.. autoapiclass:: transformer_engine.common.recipe.MXFP8BlockScaling(fp8_format=Format.E4M3)

.. autoapiclass:: transformer_engine.common.recipe.NVFP4BlockScaling(fp4_format=Format.E2M1)

.. autoapiclass:: transformer_engine.common.recipe.Float8CurrentScaling(fp8_format=Format.HYBRID)

.. autoapiclass:: transformer_engine.common.recipe.Float8BlockScaling(fp8_format=Format.E4M3)

.. autoapiclass:: transformer_engine.common.recipe.CustomRecipe(qfactory, fp8_dpa=False, fp8_mha=False)
