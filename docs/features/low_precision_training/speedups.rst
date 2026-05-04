..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

GEMM Speedups Across Precisions
=================================

Transformer Engine supports multiple low-precision formats for the linear-layer GEMMs
that dominate transformer training time: BF16, FP8 tensor-wise scaling
(CurrentScaling, DelayedScaling), FP8 Block Scaling, MXFP8, and NVFP4. Each step down
in precision can accelerate the 12 GEMMs per transformer layer (4 Fprop + 4 Dgrad +
4 Wgrad), but the actual speedup depends on your model's matrix dimensions.

A benchmark tool is provided at
`benchmarks/gemm/benchmark_gemm.py <https://github.com/NVIDIA/TransformerEngine/blob/main/benchmarks/gemm/benchmark_gemm.py>`__
to measure GEMM performance for your specific model config. See the
:doc:`full tutorial </examples/gemm_profiling/gemm_profiling>` for usage details.


Recommended Precision Recipes by Device
-----------------------------------------

.. note::

   **Both devices:** BF16, FP8 CurrentScaling (``Float8CurrentScaling``),
   FP8 DelayedScaling (``DelayedScaling``)

   **Hopper (SM90+):** Add FP8 Block Scaling (``Float8BlockScaling``)

   **Blackwell (SM100+):** Add MXFP8 (``MXFP8BlockScaling``), NVFP4 (``NVFP4BlockScaling``)

   FP8 Block Scaling on Blackwell is emulated via MXFP8 and supported only for
   backward compatibility. On Blackwell, prefer MXFP8 or NVFP4. FP8 Block Scaling
   targets Hopper, where it runs natively.


Speedup Is Shape-Dependent
----------------------------

**The speedup from lower-precision GEMMs depends directly on the matrix dimensions,
which are determined by your model config.** Larger matrices amortize the fixed
overhead of quantization (format conversion, block scaling, Hadamard transforms)
over more compute, so they benefit more from FP8 and FP4. Smaller matrices may
see zero benefit -- or even a slowdown -- because the quantization overhead costs
more than the GEMM kernel saves.

This means your model's architecture directly determines how much you stand to gain
from precision scaling:

- **Models with large hidden dimensions and intermediate sizes** produce large GEMMs
  where FP4/FP8 tensor cores have enough work to overcome overhead. These models see
  meaningful speedups.
- **Models with smaller dimensions** (or individual projections like attention output,
  which has K=N=hidden_size and no expansion) may see little to no benefit from lower
  precision, because the GEMM is too small for the faster kernel to outrun the
  quantization cost.
- **Batch size and sequence length also matter** -- they determine M, the token
  dimension. A larger M means more rows in each GEMM, which again helps amortize
  overhead.

This is why benchmarking with your actual config matters. The theoretical tensor core
speedup (e.g., 2x for FP4 vs FP8) is an upper bound that assumes the GEMM is large
enough to saturate the hardware. This also makes the tool useful for architecture
co-design: run candidate configs through the tool before committing to a training
run to see how each choice affects low-precision gains.


Example: 5B Model on B300 (Blackwell)
---------------------------------------

.. figure:: gemm_profiling/img/b300_model_config_speedup.png
   :align: center
   :width: 80%
   :alt: Autocast model config benchmark showing per-layer GEMM time breakdown across precisions.

   Autocast model config benchmark on NVIDIA B300 -- per-layer GEMM time breakdown by
   precision and operation (Fprop+Dgrad and Wgrad).

For a 5B-parameter model (hidden=4096, intermediate=16384, 24 layers), MXFP8 delivers
~1.42x and NVFP4 delivers ~1.98x over BF16 in autocast mode. FP8 DelayedScaling
reaches 1.64x, outperforming both FP8 CurrentScaling (1.39x) and MXFP8 on Blackwell.
In pre-quantized mode (raw kernel throughput), NVFP4 reaches 3.48x -- the gap is
quantization overhead.

Example: 5B Model on H200 (Hopper)
-------------------------------------

.. figure:: gemm_profiling/img/h200_model_config_speedup.png
   :align: center
   :width: 80%
   :alt: Autocast model config benchmark showing per-layer GEMM time breakdown across precisions on H200.

   Autocast model config benchmark on NVIDIA H200 NVL -- per-layer GEMM time breakdown by
   precision and operation (Fprop+Dgrad and Wgrad).

For the same 5B-parameter model on H200 (Hopper), the available precisions are BF16,
FP8 CurrentScaling, FP8 DelayedScaling, and FP8 Block Scaling. FP8 DelayedScaling
delivers ~1.69x over BF16, followed by FP8 CurrentScaling at ~1.58x and FP8 Block
Scaling at ~1.40x. FP8 Block Scaling runs natively on Hopper and is the only
block-scaled FP8 recipe available on this device. In pre-quantized mode (raw kernel
throughput), FP8 CurrentScaling reaches 1.90x over BF16.

See the :doc:`full tutorial </examples/gemm_profiling/gemm_profiling>` for detailed
analysis on both Blackwell and Hopper, including Fprop vs Dgrad comparisons, autocast
vs pre-quantized breakdowns, and manual shape mode for non-standard architectures.
