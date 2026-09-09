..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

FP8 Delayed Scaling
===================================

FP8 Delayed Scaling recipe estimates scaling factors from historical amax values rather than computing them
for each tensor. Compared to Current Scaling recipe, 
this reduces tensor reads per quantization from two to one, 
improving memory efficiency.

Both this and :doc:`FP8 Current Scaling <../fp8_current_scaling/fp8_current_scaling>` recipe use 
the same FP8 formats (E4M3/E5M2) with one FP32 scaling factor per tensor. 
Reading the FP8 Current Scaling documentation first is recommended.

Quantization with delayed scaling factors
-----------------------------------------

FP8 Current Scaling requires two tensor reads per quantization: one to compute amax, 
one to cast. FP8 Delayed Scaling eliminates the first read by predicting the scaling factor 
from historical amax values - hence *delayed* (using past values) versus *current* (using present values).

The quantization process works as follows:

1. **Compute scaling factor from history** (no tensor read needed):
   The scaling factor is derived from stored ``amax_history`` using the formula:
   
   ``scaling_factor = FP8_MAX / amax``
   
   where ``amax`` is computed from history using either ``max`` (maximum over window, default) or ``most_recent`` algorithm.

2. **Quantize the tensor** (one tensor read):
   Apply the scaling factor and cast to FP8. Values exceeding FP8 range are clipped.

3. **Update history**:
   Record the actual amax from this quantization for future iterations.

Each module maintains an ``amax_history`` tensor of configurable length (``amax_history_len``) 
for each quantized tensor.

.. raw:: html
   :file: img/scaling_comparison.svg

*Figure 1. Comparison of FP8 Current Scaling and FP8 Delayed Scaling quantization processes.*

Amax History Management
-----------------------

The ``amax_history`` buffer acts as a sliding window of recent amax values.
Position 0 serves as a staging area for the current amax, while positions 1 to N-1 
store the history from oldest to newest. Each quantization writes the observed amax 
to position 0, and after the pass completes, the history is rotated:

.. code-block:: text

   Before rotation: [amax_N, amax_1, amax_2, ..., amax_N-1]   (amax_N = current, amax_1 = oldest)
   After rotation:  [0,      amax_2, ..., amax_N-1, amax_N]   (amax_1 dropped, amax_N appended)

The scaling factor is computed **before** the rotation, so it uses all ``amax_history_len`` values.
Position 0 serves as a staging area — it is zeroed after the scale update, ready for the next iteration's amax.

The implementation differs between PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      Each module creates two ``amax_history`` tensors, initialized to zero:
      
      - Forward: shape ``(amax_history_len, num_gemms * 3)`` — three FP8 tensors per GEMM (input, weight, output)
      - Backward: shape ``(amax_history_len, num_gemms * 2)`` — two FP8 tensors per GEMM (grad_output, grad_input)
      
      When the autocast context exits, a single CUDA kernel processes all tensors at once — 
      performing amax reduction across GPUs and history rotation. This batched approach 
      minimizes kernel launch overhead compared to updating each tensor separately.

   .. tab:: JAX

      Each quantizer maintains its own ``amax_history`` with shape ``(amax_history_len,)``
      and updates independently.

Here's how to use FP8 Delayed Scaling in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. raw:: html

         <div class="code-block-header">
            Requires SM89 (Ada) or later
         </div>

      .. literalinclude:: pytorch_delayed_scaling_example.py
         :language: python
         :start-after: # START_DELAYED_SCALING_EXAMPLE
         :end-before: # END_DELAYED_SCALING_EXAMPLE

   .. tab:: JAX

      .. raw:: html

         <div class="code-block-header">
            Requires SM89 (Ada) or later
         </div>

      .. literalinclude:: jax_delayed_scaling_example.py
         :language: python
         :start-after: # START_DELAYED_SCALING_EXAMPLE
         :end-before: # END_DELAYED_SCALING_EXAMPLE


Distributed Training
--------------------

FP8 Delayed Scaling uses the same data formats as FP8 Current Scaling - quantized all-gather is supported.
However, amax reduction works slightly differently in different frameworks.

.. tabs::

   .. tab:: PyTorch

      Amax reduction is controlled by two parameters:
      
      - ``reduce_amax`` in recipe: enables/disables reduction (required for SP and CP)
      - ``amax_reduction_group`` in ``autocast``: specifies the process group for reduction
      
      We recommend reducing amax across all GPUs where the tensor is sharded, 
      including data parallel ranks.

      .. literalinclude:: pytorch_delayed_scaling_distributed_example.py
         :language: python
         :start-after: # START_AMAX_REDUCTION_EXAMPLE
         :end-before: # END_AMAX_REDUCTION_EXAMPLE

      In data parallel training, some modules may not execute on certain ranks 
      (e.g., MoE experts that receive no tokens). This is handled as follows:
      
      - **First iteration**: All modules must execute on all ranks to register 
        their ``amax_history`` tensors in the global buffer. Mismatched registration
        would cause the ``all_reduce`` to hang due to different tensor sizes across ranks.
      - **Subsequent iterations**: The ``autocast`` context must be entered and exited
        on all ranks (this triggers the collective reduction). Individual modules can be
        skipped - if no rank executes a module, its history is not rotated and scale 
        remains unchanged.


   .. tab:: JAX

      Amax reduction is always enabled and managed automatically.
      Reduction scope: all parallelism axes except pipeline parallelism (TP, SP, DP/FSDP).

      .. literalinclude:: jax_delayed_scaling_distributed_example.py
         :language: python
         :start-after: # START_AMAX_REDUCTION_EXAMPLE
         :end-before: # END_AMAX_REDUCTION_EXAMPLE

Supported devices
-----------------

Ada and later (SM 8.9+)

Quantizer
---------

.. tabs::

   .. tab:: PyTorch

      Delayed scaling uses
      :class:`~transformer_engine.pytorch.Float8Quantizer`. It does not
      compute the scaling factor from the current tensor: one-element
      ``scale`` and ``amax`` buffers are supplied at construction.
      Quantization applies the given scale and records the tensor's amax into
      the ``amax`` buffer.

      During training both buffers are views into the recipe state: ``scale``
      into its per-quantizer scale vector, ``amax`` into the current row of
      its ``(amax_history_len, num_quantizers)`` amax history. At the end of
      each step the recipe state computes a new scale from the history (its
      max or most recent entry, per ``amax_compute_algo``), rolls the history
      by one slot, and zeroes the current row — all in place, so the views
      held by the quantizer stay valid for the whole training run.

      .. code-block:: python

         import torch
         import transformer_engine.pytorch as te

         tensor = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)

         quantizer = te.Float8Quantizer(
             scale=torch.ones(1, device="cuda"),
             amax=torch.zeros(1, device="cuda"),
             fp8_dtype=te.DType.kFloat8E4M3,
         )

         qtensor = quantizer(tensor)
         roundtrip = qtensor.dequantize()

   .. tab:: JAX

      Delayed scaling uses ``DelayedScaleQuantizer``. The ``scale`` and the
      ``amax_history`` (1024 entries by default) are fields of the quantizer
      itself, carried through JAX transformations as its pytree state. Each
      ``quantize()`` call applies the current ``scale``, then updates the
      state: the tensor's amax is written into the history, a new scale is
      computed from the history (max or most-recent entry, per
      ``amax_compute_algo``), and the history is rolled by one slot.

      .. code-block:: python

         import jax.numpy as jnp
         from transformer_engine.jax.quantize import (
             QuantizerFactory, ScalingMode, QuantizeLayout,
         )

         x = jnp.ones((256, 256), dtype=jnp.bfloat16)

         quantizer = QuantizerFactory.create(
             scaling_mode=ScalingMode.DELAYED_TENSOR_SCALING,
             q_dtype=jnp.float8_e4m3fn,
             q_layout=QuantizeLayout.ROWWISE,
         )
         qtensor = quantizer.quantize(x)
         roundtrip = qtensor.dequantize()
