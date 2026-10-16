..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Post-Training Quantization Calibration
======================================

TransformerEngine can collect and export quantization scaling metadata from
TransformerEngine modules for post-training quantization (PTQ), typically used
by inference engines to avoid recomputing global scales during inference.
Enable collection by passing a
:class:`~transformer_engine.pytorch.QuantizationCalibrationConfig` to
:class:`~transformer_engine.pytorch.autocast`.

.. code-block:: python

   import transformer_engine.pytorch as te
   from transformer_engine.common.recipe import Float8CurrentScaling

   calibration_config = te.QuantizationCalibrationConfig(
       # Calibration decay applied to past samples. Zero decay implies
       # taking only the most recent scaling factor, while unit decay
       # implies taking a maximum over the entire history of the model
       # and/or calibration dataset.
       transformer_engine_calibration_decay=0.0,
   )

   # Quantized forward while tracking FP8-CS weight and activation
   # `scale_inv` for export.
   with te.autocast(
       enabled=True,
       recipe=Float8CurrentScaling(),
       calibrating=True,
       calibration_config=calibration_config,
   ):
       model(inputs)

Calibration metadata is exposed on each TransformerEngine module as a non-persistent buffer
that should be manually reduced (across model or data parallelisms) and appended to a state
dictionary for checkpoint export. Resuming PTQ calibration from an checkpoint with exported
calibration buffers is not supported.

.. code-block:: python

   calibration = {}
   for module_fqn, module in model.named_modules():
       for buffer_name, value in module.named_buffers(recurse=False):
           if buffer_name.endswith("_te_ptq_calibrated"):
               key = f"{module_fqn}.{buffer_name}" if module_fqn else buffer_name
               calibration[key] = value.detach().cpu()

Buffer names follow this template:

.. code-block:: text

   {tensor_name}_{metadata_name}_{recipe_name}_te_ptq_calibrated

The fields identify:

* ``tensor_name``: the tensor's role in the module, such as ``input``, ``weight``,
  ``fc1_input``, or ``weight_gemm2``.
* ``metadata_name``: the collected quantity. FP8 current scaling stores ``scale_inv``;
  FP8 delayed scaling and NVFP4 store ``amax``.
* ``recipe_name``: the quantization recipe name standardized by ``Recipe.name()``.
  PTQ calibration metadata buffering currently supports ``fp8_current_scaling``, ``fp8_delayed_scaling``, and ``nvfp4``. Block- and row-wise recipes such as
  MXFP8 and row-wise NVFP4 are not supported and should be PTQ-recomputed.

For example, the NVFP4 weight metadata for local expert 2 is named:

.. code-block:: text

   weight_gemm2_amax_nvfp4_te_ptq_calibrated

GroupedLinear uses one ``input_gemmN`` and ``weight_gemmN`` entry per local expert/GEMM.
Distributed framework integrations are responsible for reducing or gathering local buffers
and mapping local expert indices to global expert indices before checkpoint export.

``transformer_engine_calibration_decay`` is a decay between 0 and 1 that is applied to
previously-buffered activation scaling factors before incorporating each new observation.
With zero decay, only the latest value is buffered, while higher decay can be utilized to
extract steady-state calibration values during training, where the model and activations
are dynamic. Weight calibration always buffers the current observed calibration value.

.. note::

   Calibration buffers are created and updated during a forward pass. If weight caching
   is utilized, then weight calibration is performed only during weight updates.
   Backward and activation recomputation passes do not update calibration metadata, so
   each observation is only updated (and decayed) once per forward step. Run a warmup
   forward pass before CUDA graph capture, then read the buffers only after graph
   replay has completed.
