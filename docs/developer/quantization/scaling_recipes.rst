..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _scaling-recipes:

Scaling Recipes
===============

Scaling recipes control *how* quantization parameters (scales, amax values) are computed
and updated across training iterations. They sit between the user-facing ``recipe`` API
and the per-tensor ``Quantizer`` instances.

Recipe Configuration
--------------------

Users configure quantization via recipe classes from ``transformer_engine.common.recipe``.
The recipe hierarchy:

- ``Recipe`` — abstract base class
- ``DelayedScaling`` — delayed tensor scaling with amax history
- ``Float8CurrentScaling`` — just-in-time per-tensor scaling
- ``MXFP8BlockScaling`` — MXFP8 with 32-element blocks
- ``Float8BlockScaling`` — configurable 1D/2D block scaling
- ``NVFP4BlockScaling`` — NVFP4 with 16-element blocks
- ``CustomRecipe`` — user-provided quantizer factory

Recipe classes specify:

- **FP8 format**: Which FP8 types to use for forward (typically E4M3) and backward
  (typically E5M2).
- **Amax history length**: How many past amax values to track (default: 1024).
- **Amax compute algorithm**: How to derive the scale from history (typically ``max``).
- **Scaling mode**: Delayed tensor, current tensor, MXFP8, block, or NVFP4.

.. code-block:: python

   import transformer_engine.pytorch as te
   from transformer_engine.common.recipe import DelayedScaling

   recipe = DelayedScaling(
       fp8_format=te.recipe.Format.HYBRID,  # E4M3 fwd, E5M2 bwd
       amax_history_len=1024,
       amax_compute_algo="max",
   )

   with te.autocast(enabled=True, recipe=recipe):
       output = model(input)

FP8GlobalStateManager
---------------------

The ``FP8GlobalStateManager`` (in ``transformer_engine/pytorch/quantization.py``) is a singleton
that coordinates FP8 state across all modules in a model. Key responsibilities:

- **Track global FP8 enabled/disabled state** for the current forward pass.
- **Distribute recipe configuration** to all ``TransformerEngineBaseModule`` instances.
- **Coordinate amax reduction** across distributed ranks (for delayed scaling, all ranks
  must agree on the global amax).

The manager is activated by the ``autocast`` context manager:

.. code-block:: python

   # Pseudocode for autocast
   @contextmanager
   def autocast(enabled, recipe):
       FP8GlobalStateManager.set_enabled(enabled)
       FP8GlobalStateManager.set_recipe(recipe)
       try:
           yield
       finally:
           FP8GlobalStateManager.reset()

Per-Module State
----------------

Each ``TransformerEngineBaseModule`` maintains its own quantization state:

- **Quantizer instances**: One per quantized tensor (input, weight, output gradient, weight
  gradient). Created based on the active recipe.
- **Amax history buffers**: Registered as module buffers for delayed scaling. Updated
  after each forward/backward pass.
- **Scale tensors**: Computed from amax history before each forward pass.

The module's ``init_fp8_metadata()`` method creates this state, and
``prepare_forward()`` / ``end_forward()`` manage the per-iteration lifecycle.

Amax Update Flow (Delayed Scaling)
-----------------------------------

For delayed tensor scaling, the amax update follows this sequence each iteration:

1. **Pre-forward**: Compute new scales from existing amax history.
2. **Forward**: GEMM kernels record the amax of their FP8 outputs into a buffer.
3. **Backward**: Same as forward for gradient GEMMs.
4. **Post-backward**: Roll the amax history window (shift old values, insert new amax).
5. **All-reduce**: If using distributed training, reduce amax across ranks.

For current-scaling and block-scaling modes, scales are computed just-in-time during
the cast kernel, so no amax history is maintained.

.. warning::

   For the amax all-reduce to work correctly, every rank must participate in the
   collective. This means each ``autocast`` region must contain at least one FP8
   module to trigger the reduction. The modules do not need to be the same across
   ranks — the reduction operates on amaxes from all layers simultaneously and updates
   only those that were touched (identified by checking against the initial zero value).

Recipe → Quantizer Mapping
--------------------------

The recipe's scaling mode determines which ``Quantizer`` subclass is instantiated:

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Recipe / Mode
     - Forward Quantizer
     - Backward Quantizer
   * - ``DelayedScaling``
     - ``Float8Quantizer``
     - ``Float8Quantizer``
   * - Current scaling
     - ``Float8CurrentScalingQuantizer``
     - ``Float8CurrentScalingQuantizer``
   * - MXFP8
     - ``MXFP8Quantizer``
     - ``MXFP8Quantizer``
   * - Block scaling
     - ``Float8BlockQuantizer``
     - ``Float8BlockQuantizer``
   * - NVFP4
     - ``NVFP4Quantizer``
     - ``NVFP4Quantizer``

See Also
--------

- :doc:`class_hierarchy` — The Quantizer/Storage/Tensor class design
- :doc:`/developer/pytorch_frontend/module_hierarchy` — How modules manage FP8 state
