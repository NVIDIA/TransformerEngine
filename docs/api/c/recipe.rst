..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

recipe.h
========

Partial shard casting uses ``nvte_partial_cast`` for block FP8, MXFP8, and NVFP4.
Select the recipe through the output tensor's scaling mode and attach its data,
scales, and inverse scales with ``nvte_set_tensor_param_v2``. Output and scale
datatypes come from those tensor parameters. Matrix dimensions and the shard's
logical offset remain explicit because they cannot be inferred from a local shard.

The existing ``nvte_fp8_block_scaling_partial_cast``,
``nvte_mxfp8_scaling_partial_cast``, ``nvte_nvfp4_2d_partial_cast``, and
``nvte_nvfp4_2d_partial_cast_v2`` entry points retain their signatures and forward
to the combined API. Their adapters preserve the original scale conventions.
Legacy strided FP8 scales may require a temporary contiguous buffer; legacy
float32 NVFP4 tile scales require conversion to encoded, expanded inverse scales.
These adapters use stream-ordered temporary allocations because the old interfaces
do not accept caller-provided workspace. The combined API itself does not allocate
device memory.

New NVFP4 callers attach the encoded, expanded rowwise inverse scales already used
by GEMM, together with the scalar global encoding scale. MXFP8 columnwise casting
is selected by the presence of the output tensor's columnwise data field, regardless
of whether auxiliary columnwise scale metadata exists. When columnwise output is
present, its inverse scales are required. See the ``nvte_partial_cast`` reference
below for layouts and allocation requirements.
Partial amax computation APIs are unchanged.

.. doxygenfile:: recipe.h

