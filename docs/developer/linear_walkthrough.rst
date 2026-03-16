..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _linear-walkthrough:

Linear Module: End-to-End Walkthrough
======================================

This page traces a complete forward and backward pass through ``te.Linear``, the most
fundamental Transformer Engine module. It connects concepts from across the codebase:
quantization, GEMM, distributed communication, and autograd.

The goal is not just to show *what* happens at each step, but *why* the code is structured
this way — what constraints drive the ordering, what trade-offs are being made, and how
low-precision quantization fundamentally changes the data flow compared to a standard
``torch.nn.Linear``.

.. figure:: ./img/linear_e2e_flow.svg
   :align: center
   :width: 95%

   End-to-end flow through Linear forward and backward with FP8 and tensor parallelism.

..
   Diagram description for ``linear_e2e_flow.svg``:
   Two horizontal swim lanes labeled "Forward" and "Backward".
   Forward lane (left to right):
     1. "Input (BF16)" →
     2. "All-gather (if column-parallel + SP)" →
     3. "input_quantizer(input)" producing "FP8 rowwise + columnwise" →
     4. "general_gemm(weight, input)" producing "Output (BF16)" →
     5. "Reduce-scatter (if row-parallel + SP) or All-reduce (if row-parallel)"
   Backward lane (right to left):
     1. "grad_output (BF16)" →
     2. "grad_output_quantizer(grad)" →
     3. "Dgrad GEMM: general_gemm(grad, weight)" producing "grad_input" →
     4. "Wgrad GEMM: general_gemm(input_columnwise, grad)" producing "grad_weight" →
     5. "All-reduce grad_weight (if column-parallel)"
   Dotted arrows from forward boxes 3→backward box 4 labeled "saved columnwise input"
   and forward weight→backward box 3 labeled "saved weight".

Why This Walkthrough Matters
-----------------------------

A standard ``torch.nn.Linear`` is straightforward: ``output = input @ weight.T + bias``.
TE's Linear does the same math, but introduces two major dimensions of complexity:

1. **Low-precision quantization** — Inputs and weights must be cast to a low-precision
   format (FP8, MXFP8, NVFP4) *before* GEMM, and the backward pass needs the data in a
   *different layout* (columnwise) than the forward pass (rowwise). This means the forward
   pass must proactively prepare data for backward.

2. **Tensor parallelism** — The weight matrix is sharded across GPUs, so collective
   communication (all-gather, reduce-scatter) must be interleaved with computation.
   The placement of communication relative to GEMM depends on whether this is a
   column-parallel or row-parallel linear.

These two concerns interact with each other at every step — for instance, quantizing
*before* an all-gather halves the communication volume. The walkthrough below makes these
interactions explicit.

The flow described here is **recipe-agnostic**: the same phases execute regardless of
whether the active recipe is MXFP8, current scaling, block scaling, or delayed scaling.
The quantizer abstraction (see :doc:`quantization/class_hierarchy`) hides recipe-specific
details — ``_Linear`` simply calls ``quantizer(tensor)`` and receives quantized data with
the appropriate scales, no matter how those scales were computed.

Setup
-----

.. code-block:: python

   import transformer_engine.pytorch as te
   from transformer_engine.common.recipe import MXFP8BlockScaling

   # Create a Linear module
   linear = te.Linear(4096, 16384, bias=True)

   # Enable FP8 with MXFP8 recipe
   recipe = MXFP8BlockScaling()
   with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
       output = linear(input)  # Triggers the flow below

Phase 1: Module Forward Entry
------------------------------

**File**: ``transformer_engine/pytorch/module/linear.py``, ``Linear.forward()`` (line ~1343)

**Why this phase exists**: Before any math can happen, TE needs to set up the quantization
infrastructure. Unlike BF16 where you just call GEMM directly, low-precision formats
require quantizer objects that know how to cast data and manage scales. This setup phase
bridges the gap between the user-facing recipe configuration and the quantizer objects
that actually perform casts.

1. ``prepare_forward()`` is called (inherited from ``TransformerEngineBaseModule``):

   - Checks if FP8 is enabled via ``FP8GlobalStateManager`` — a global singleton that
     tracks whether we're inside an ``fp8_autocast`` context. This global state exists
     because FP8 behavior must be coordinated across all TE modules in the model.
   - Creates or refreshes **quantizer** instances from the active recipe. Each recipe
     type produces a different quantizer class (``MXFP8Quantizer``,
     ``Float8CurrentScalingQuantizer``, ``Float8BlockQuantizer``, etc.).
   - Recipe-specific customization is applied — for example, current scaling configures
     ``amax_epsilon`` and ``power_2_scale`` on quantizers, while MXFP8 needs no such
     configuration.

2. Three quantizers are prepared, one for each tensor that will be cast to FP8:

   - ``input_quantizer`` — for the activation tensor (forward GEMM input)
   - ``weight_quantizer`` — for the weight parameter (forward GEMM input)
   - ``grad_output_quantizer`` — for the backward gradient (backward GEMM input)

   **Why three separate quantizers?** Each tensor has independent scaling state because
   their value distributions differ significantly — activations, weights, and gradients
   have different magnitudes and ranges. A single shared scale would waste dynamic range.

3. ``_Linear.apply()`` is called, entering the custom ``torch.autograd.Function``. This
   boundary separates the ``nn.Module`` API from the autograd machinery — everything
   beyond this point runs inside PyTorch's autograd graph and must carefully manage what
   is saved for backward.

Phase 2: _Linear.forward() — Input Quantization and Communication
-------------------------------------------------------------------

**File**: ``transformer_engine/pytorch/module/linear.py``, ``_Linear.forward()`` (line ~87)

**Why this is the most complex phase**: Input preparation must solve a chicken-and-egg
problem. The GEMM needs the *full* (ungathered) input, but communication is expensive.
With FP8, we also need both a rowwise *and* columnwise quantized version. The code must
decide: quantize before or after communication? The answer depends on the parallel mode.

**Column-parallel with sequence parallelism** (QKV projection, FC1):

Each GPU holds a shard of the input along the sequence dimension. The GEMM needs the
full sequence, so an all-gather is required. The key decision: **quantize locally first,
then all-gather the quantized data** — this communicates 1 byte per element instead of 2
(BF16), cutting communication volume in half.

.. code-block:: python

   # 1. Configure what quantization layouts we need.
   #    Rowwise: for the forward GEMM (this phase).
   #    Columnwise: for the backward wgrad GEMM (Phase 8, later).
   #    We quantize both now because the original BF16 input will be discarded.
   input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)

   # 2. Quantize the local shard — produces a QuantizedTensorStorage containing
   #    quantized rowwise data, columnwise data, and their scale inverses.
   inputmat = input_quantizer(inputmat)

   # 3. All-gather the quantized data across TP ranks.
   #    Each rank sends its local shard; all ranks receive the full tensor.
   inputmat_total, _ = gather_along_first_dim(inputmat, tp_group)

.. note::

   The columnwise data is NOT all-gathered along with the rowwise data in all cases.
   Some quantizer types (e.g. delayed scaling ``Float8Quantizer``) don't support
   columnwise all-gather, so columnwise is disabled before the gather and the input is
   re-quantized columnwise in the backward pass. This is a trade-off: less communication
   complexity at the cost of redundant quantization later.

**No parallelism** (standalone Linear):

No communication needed — the full input is already local. Quantization still produces
both layouts because the backward pass will need columnwise data regardless.

.. code-block:: python

   input_quantizer.set_usage(rowwise=True, columnwise=backward_needs_input)
   inputmat = input_quantizer(inputmat)
   inputmat_total = inputmat  # No gather needed

**What ``backward_needs_input`` means**: This flag is ``True`` when ``weight.requires_grad``
and gradients are enabled — i.e., during training. During inference or when the weight is
frozen, the backward wgrad GEMM won't run, so there's no point producing columnwise data.
This avoids a significant portion of the quantization cost during inference.

Phase 3: _Linear.forward() — Weight Quantization
--------------------------------------------------

**File**: ``transformer_engine/pytorch/module/linear.py``, ``_Linear.forward()`` (line ~250)

**Why weight handling differs from input**: Weights are persistent parameters that don't
change between microbatches during gradient accumulation. Inputs change every microbatch.
This asymmetry enables an important optimization: **quantized weight caching**.

.. code-block:: python

   # Configure weight quantizer — rowwise for forward GEMM,
   # columnwise for backward dgrad GEMM (where the weight is the "other" operand).
   weight_quantizer.set_usage(rowwise=True, columnwise=columnwise_usage)

   # Get quantized weight, potentially from cache.
   # On the first microbatch: quantize and cache.
   # On subsequent microbatches: return cached quantized weight.
   weightmat = module.get_weight_workspace(
       tensor=weight,
       quantizer=weight_quantizer,
       cache_name=(None if is_first_microbatch is None else "weight"),
       update_workspace=is_first_microbatch is None or is_first_microbatch,
   )

**Why cache quantized weights?** In gradient accumulation with *N* microbatches, the weight
doesn't change until the optimizer step. Without caching, we'd quantize the same weight
*N* times. With caching, we quantize once and reuse the quantized version for all *N*
forward passes. The ``is_first_microbatch`` parameter controls this: ``True`` triggers a
fresh quantization, ``False`` returns the cached version, ``None`` disables caching
entirely (quantize every time).

**Columnwise weight**: The weight also needs a columnwise layout for the backward dgrad
GEMM (Phase 7). The quantizer produces both during the same quantization call, just like
the input quantizer. The columnwise weight data is saved for backward.

Phase 4: _Linear.forward() — GEMM
-----------------------------------

**File**: ``transformer_engine/pytorch/cpp_extensions/gemm.py``

**Why this isn't just a matmul**: Low-precision GEMM requires additional metadata (scale
inverses) that standard matrix multiplication APIs don't handle. TE wraps cuBLASLt with
scale-aware logic and optional communication overlap.

.. code-block:: python

   # Forward GEMM: output = input @ weight^T
   # The result is produced in high precision (BF16/FP16) — cuBLASLt accumulates
   # in FP32 internally and casts the output down. This is why low-precision GEMM
   # doesn't lose as much accuracy as you might expect from 8-bit math.
   gemm_out, *_, reduce_scatter_out = general_gemm(
       weightmat,           # A operand (quantized + scale_inv)
       inputmat_total,      # B operand (quantized + scale_inv)
       out_dtype=activation_dtype,
       bias=bias,           # Fused bias add (avoids separate kernel launch)
       use_split_accumulator=use_split_accumulator,
       ub=ub_obj,           # Userbuffers overlap (optional)
       ub_type=ub_type,
   )

The call traverses four layers, matching the :doc:`architecture_overview`:

1. **Python** (``cpp_extensions/gemm.py``): Extracts raw data pointers and scale tensors
   from the ``QuantizedTensorStorage``. The C++ layer can't understand Python quantized
   tensor types, so this translation is mandatory.
2. **pybind11** (``csrc/extensions/gemm.cpp``): Constructs opaque ``NVTETensor`` handles —
   the C API's tensor abstraction. This is where TE crosses the Python/C++ boundary.
3. **C API** (``common/gemm/``): Selects the cuBLASLt algorithm and configures compute
   types based on the input precision and scaling mode.
4. **cuBLASLt**: Executes the actual matrix multiply on the GPU.

**``use_split_accumulator``**: When ``True``, cuBLASLt uses higher-precision intermediate
accumulators, trading some performance for numerical accuracy. This is controlled by the
recipe and defaults to ``False`` for the forward pass (where some accumulation error is
acceptable) and ``True`` for backward (where gradient accuracy matters more).

Phase 5: _Linear.forward() — Output Communication
----------------------------------------------------

**Why output communication depends on parallel mode**: Tensor parallelism shards the
weight matrix, and the parallel mode determines which dimension is sharded. This directly
determines what communication is needed to produce the correct output.

For **row-parallel** (output projection, FC2):

Each GPU computes a partial sum (because the input is split across the inner dimension).
These partial results must be summed across GPUs to produce the correct output.

.. code-block:: python

   if sequence_parallel:
       # Reduce-scatter: sum partial results AND scatter the output along the
       # sequence dimension, so each GPU only stores its local sequence shard.
       # This is more memory-efficient than all-reduce because the output is
       # smaller on each GPU.
       out, _ = reduce_scatter_along_first_dim(out, tp_group)
   elif tensor_parallel:
       # All-reduce: sum partial results and replicate the full output.
       # Used when sequence parallelism is disabled.
       out, _ = allreduce(out, tp_group)

For **column-parallel** (QKV, FC1): No communication needed. Each GPU holds a different
slice of the output features, and these slices are independent — they don't need to be
summed or gathered until a downstream operation requires the full output.

Phase 6: _Linear.forward() — Save for Backward
--------------------------------------------------

**Why this phase is critical for memory efficiency**: PyTorch autograd requires saving
tensors from the forward pass to compute gradients. For a standard ``nn.Linear``, this
means saving the full input and weight in BF16. With FP8, TE saves *quantized* tensors —
half the memory — but must carefully track which quantization layouts to keep.

.. code-block:: python

   # Discard rowwise data — it was only needed for the forward GEMM (Phase 4)
   # and is no longer useful. Keep only columnwise data for the wgrad GEMM.
   if backward_needs_input and own_quantized_input:
       inputmat.update_usage(rowwise_usage=False, columnwise_usage=True)

   ctx.save_for_backward(inputmat, weightmat, weight, bias)

**The key insight**: The forward GEMM consumes *rowwise* input, but the backward wgrad
GEMM needs *columnwise* input (see :doc:`quantization/rowwise_columnwise`). Since TE
quantized both layouts in Phase 2, it can now discard the rowwise data, keeping only
the columnwise data in the autograd cache. This means activation memory for FP8 is
roughly **half** of what it would be in BF16 — one byte per element (FP8 columnwise)
instead of two (BF16).

For column-parallel with sequence parallelism, only the *local shard* of the input is
saved, not the full all-gathered tensor. The all-gather will be repeated in the backward
pass. This trades communication for memory — a favorable trade-off in large model training
where activation memory is the binding constraint.

Phase 7: _Linear.backward() — Dgrad (Activation Gradient)
-----------------------------------------------------------

**File**: ``transformer_engine/pytorch/module/linear.py``, ``_Linear.backward()`` (line ~495)

**Why dgrad runs before wgrad**: The dgrad GEMM (``grad_input = grad_output @ weight``)
is on the critical path of backpropagation — downstream layers are waiting for
``grad_input`` to continue their own backward passes. The wgrad GEMM (``grad_weight``)
only updates this layer's parameters and doesn't block anything. Running dgrad first
minimizes the time other layers spend waiting.

.. code-block:: python

   # Quantize grad_output for the backward GEMMs.
   # Rowwise: needed for dgrad GEMM (this phase).
   # Columnwise: needed for wgrad GEMM (Phase 8).
   # Both are computed now because the BF16 grad_output will be discarded.
   grad_output_quantizer.set_usage(rowwise=True, columnwise=True)
   grad_output = grad_output_preprocess(ctx, grad_output, ...)

   # Dgrad GEMM: grad_input = grad_output @ weight
   # Uses rowwise grad_output and columnwise weight (saved from forward).
   dgrad, *_ = general_gemm(
       weight_fp8,       # Columnwise weight saved in Phase 6
       grad_output,      # Rowwise grad_output
       layout="NN",
       grad=True,        # Signals this is a backward GEMM (affects accumulator)
       out_dtype=activation_dtype,
   )

**Communication overlap opportunity**: For column-parallel, the dgrad output must be
reduce-scattered (or all-reduced) back to each GPU's local shard. This communication is
launched *asynchronously* — it runs on a separate CUDA stream while the wgrad GEMM
(Phase 8) executes on the compute stream. Similarly, if the wgrad needs a re-gathered
input (column-parallel + SP), that all-gather is launched asynchronously during dgrad
and synchronized before wgrad begins.

Phase 8: _Linear.backward() — Wgrad (Weight Gradient)
-------------------------------------------------------

**Why wgrad uses columnwise data**: The wgrad GEMM computes ``grad_weight = input^T @
grad_output``. The transpose of rowwise data is columnwise data. Rather than transposing
at this point (which would require a separate kernel launch and temporary memory), the
forward pass pre-computed the columnwise layout in Phase 2. This is the payoff for the
"dual layout" strategy.

.. code-block:: python

   # Synchronize any async communication from Phase 7
   if inputmat_total_work is not None:
       inputmat_total_work.wait()

   # Ensure columnwise data is available
   inputmat_total.update_usage(columnwise_usage=True)

   # Wgrad GEMM: grad_weight = input^T @ grad_output
   wgrad, *_ = general_gemm(
       inputmat_total,    # Columnwise input saved from forward
       grad_output,       # Columnwise grad_output from Phase 7
       out_dtype=activation_dtype,
       grad=True,
       ub=ub_obj_wgrad,
       ub_type=ub_type_wgrad,
   )

**``fuse_wgrad_accumulation``**: When enabled (common in Megatron-LM), the wgrad GEMM
accumulates directly into ``weight.main_grad`` instead of allocating a separate gradient
tensor. This avoids an extra memory allocation and addition kernel, which matters when
gradients are large (e.g., 4096 × 16384 = 67M parameters per Linear).

Phase 9: Post-Backward Cleanup
--------------------------------

After both GEMMs complete, the backward pass handles recipe-specific bookkeeping.

For most recipes (MXFP8, current scaling, block scaling), there is no post-backward state
to update — scales are computed inline during quantization and don't carry state across
iterations. This is one of the advantages of these recipes: they are stateless.

**Delayed scaling only**: The delayed scaling recipe is an exception. It maintains an
**amax history** — a rolling window of observed maximum values — that feeds forward into
the next iteration's scale computation. After the backward pass completes, the first FP8
module in the model triggers ``FP8GlobalStateManager.reduce_and_update_fp8_tensors()``,
which all-reduces amax values across TP ranks and updates the history buffer. This ensures
all ranks agree on scales. If you're not using delayed scaling, this code path is skipped.

Summary of Files Touched
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - File
     - Role in Linear
   * - ``pytorch/module/linear.py``
     - ``Linear.forward()``, ``_Linear`` autograd
   * - ``pytorch/module/base.py``
     - ``prepare_forward()``, quantizer setup
   * - ``pytorch/quantized_tensor.py``
     - ``Quantizer``, ``QuantizedTensorStorage`` base classes
   * - ``pytorch/tensor/mxfp8_tensor.py``
     - ``MXFP8Quantizer``, block-scaled FP8 cast
   * - ``pytorch/tensor/float8_tensor.py``
     - ``Float8Quantizer``, ``Float8CurrentScalingQuantizer``
   * - ``pytorch/cpp_extensions/gemm.py``
     - ``general_gemm()`` Python wrapper
   * - ``pytorch/csrc/extensions/gemm.cpp``
     - pybind11 GEMM binding
   * - ``common/gemm/cublaslt_gemm.cu``
     - cuBLASLt GEMM kernel dispatch
   * - ``pytorch/distributed.py``
     - ``allreduce()``, ``gather_along_first_dim()``, etc.
   * - ``pytorch/quantization.py``
     - ``FP8GlobalStateManager``, ``fp8_autocast``

See Also
--------

- :doc:`architecture_overview` — High-level system architecture
- :doc:`quantization/class_hierarchy` — Quantizer/Storage/Tensor design
- :doc:`quantization/rowwise_columnwise` — Why both layouts exist
- :doc:`pytorch_frontend/autograd_integration` — Autograd patterns
- :doc:`distributed/tensor_parallel` — TP communication in Linear
