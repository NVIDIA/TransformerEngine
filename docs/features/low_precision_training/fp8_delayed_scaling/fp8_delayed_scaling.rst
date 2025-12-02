..
    Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

FP8 Delayed Scaling
===================================

FP8 Delayed Scaling estimates scaling factors from historical amax values rather than computing them
for each tensor. This reduces quantization from two tensor reads to one, improving memory bandwidth.

Both this recipe and :doc:`FP8 Current Scaling <../fp8_current_scaling/fp8_current_scaling>` use 
the same FP8 formats (E4M3/E5M2) with one float32 scaling factor per tensor. 
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
   
   where ``amax`` is computed from history using either ``max`` (default) or ``most_recent`` algorithm.

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

The effective history length is ``amax_history_len - 1`` since position 0 is reserved 
for the staging area.

The implementation differs between PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      Each module creates two ``amax_history`` tensors, initialized to zero:
      
      - Forward: shape ``(amax_history_len, num_gemms * 3)`` — three FP8 tensors per GEMM (input, weight, output)
      - Backward: shape ``(amax_history_len, num_gemms * 2)`` — two FP8 tensors per GEMM (grad_output, grad_input)
      
      During the first forward pass, modules register their ``amax_history`` tensors 
      to a **global buffer** associated with the autocast context. When the context exits,
      a single CUDA kernel processes all registered tensors at once - performing both 
      amax reduction across GPUs and history rotation.
      
      This batched approach (one kernel for all tensors instead of one kernel per tensor)
      minimizes kernel launch overhead.

   .. tab:: JAX

      Each quantizer maintains its own ``amax_history`` as a Flax variable with shape ``(amax_history_len,)``.
      There is no global buffer - each quantizer updates independently.
      
      The rotation is performed per-quantizer using ``jnp.roll``:
      
      .. code-block:: python
      
         updated_amax_history = jnp.roll(amax_history, -1, -1)
         amax_history = updated_amax_history.at[0].set(0.0)

Here's how to use FP8 Delayed Scaling in PyTorch and JAX:

.. tabs::

   .. tab:: PyTorch

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Requires SM89 (Ada) or later
         </div>

      .. literalinclude:: pytorch_delayed_scaling_example.py
         :language: python
         :start-after: # START_DELAYED_SCALING_EXAMPLE
         :end-before: # END_DELAYED_SCALING_EXAMPLE

   .. tab:: JAX

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Requires SM89 (Ada) or later
         </div>

      .. literalinclude:: jax_delayed_scaling_example.py
         :language: python
         :start-after: # START_DELAYED_SCALING_EXAMPLE
         :end-before: # END_DELAYED_SCALING_EXAMPLE


Distributed Training
--------------------

Since FP8 Delayed Scaling uses the same data formats as FP8 Current Scaling,
transpose gather is not supported. However, amax reduction works slightly differently in different frameworks.

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