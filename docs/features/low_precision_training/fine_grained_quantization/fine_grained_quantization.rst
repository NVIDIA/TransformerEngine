..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _fine-grained-quantization-recipes:

Fine-grained quantization recipes
=================================

Transformer Engine (TE) can select quantizers by module or operation type,
tensor role, module or operation instance name, and rowwise or columnwise
direction. This enables **per-GEMM granularity of the precision format and/or
quantization logic**. A
:class:`~transformer_engine.common.recipe.CustomRecipe` supplies a quantizer
factory to the standard :class:`~transformer_engine.pytorch.autocast` path. The
factory can compose TE-native quantizers with
:class:`~transformer_engine.pytorch.HybridQuantizer` and
:class:`~transformer_engine.pytorch.IdentityQuantizer`.

This guide covers PyTorch, TE-native quantizers, and static recipe construction.
It does not define a supported recipe or an expected accuracy/performance
ordering. Validate every configuration on the target model, hardware, and
distributed setup.

.. note::

   With TE-native low-precision quantizers on supported hardware and kernel
   paths, recipes use TE's native GPU quantization and low-precision GEMM
   implementations. No fake quantization or high-precision GEMM emulation is
   involved on these paths.

.. warning::

   Fine-grained recipes and their construction APIs are experimental. API,
   validation, and kernel coverage may change without notice. A configuration
   that can be expressed by the API is not necessarily executable for every
   module, GEMM shape, software version, or GPU. An executable configuration
   is not necessarily optimized or validated for accuracy and convergence on
   a particular workload.

Configuration readiness
-----------------------

Treat fine-grained construction as an experimental recipe exploration surface.
Keep these readiness levels distinct:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Level
     - Meaning
   * - Expressible
     - A factory can describe the role and direction assignment.
   * - Executable
     - The current module, GEMM backend, layout, software, and GPU accept it.
   * - Optimized
     - The selected path has an appropriate optimized kernel and integration.
   * - Workload-validated
     - Accuracy, convergence, throughput, and memory have been measured on the
       target workload.

Fine-grained assignments provide a way to explore an accuracy/performance
slider by varying precision and quantization logic per GEMM. The recipes that
can be realized efficiently are constrained by available kernels and
integrations. Accuracy and convergence experiments can run on functionally
executable, non-optimized paths before dedicated kernels are available.

Factory contract
----------------

Each TE module defines an ordered role list for the forward and backward
quantizer slots it needs. When module recipe state is initialized or rebuilt,
a ``CustomRecipe`` calls ``qfactory(role)`` once for every slot in that list.
It does not call the factory on every unchanged forward.

The role vocabulary includes:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Field
     - Examples
     - Meaning
   * - ``module_type``
     - ``"linear"``, ``"grouped_linear"``, ``"dpa"``
     - TE-defined module or operation type, populated by the TE module.
   * - ``tensor_type``
     - ``"input"``, ``"weight"``, ``"grad_output"``
     - TE-defined slot in that module's vocabulary, populated by the TE module.
   * - ``name``
     - ``"decoder.39.qkv"``, ``"decoder.39.fc2"``
     - Caller or framework-provided instance identity. Composite TE modules
       may append suffixes for nested operations.

``module_type`` and ``tensor_type`` are TE-defined selectors populated by the
module. The caller or framework supplies the root ``name``; composite TE
modules may extend it with suffixes such as ``.fc1``, ``.fc2``, and ``.proj``.

A robust factory follows four construction rules:

* Return a quantizer for every role. Return ``IdentityQuantizer`` for an
  intentional high-precision slot; do not return ``None``.
* Constructing a fresh quantizer for every call is recommended.
  ``HybridQuantizer`` owns and configures its rowwise and columnwise children.
* A module-level function is the most portable factory definition, especially
  when a launcher or checkpointing setup needs to import or pickle it.
* Treat role strings as selectors, not a fixed enumeration. Preserve a base
  factory fallback for roles the factory does not recognize.

For example, compose TE-native factories by keeping one named ``linear`` module
in high precision, using NVFP4 for every ``grouped_linear`` role, and retaining
MXFP8 as the global fallback:

.. code-block:: python

   from typing import Optional

   import transformer_engine.pytorch as te
   from transformer_engine.pytorch.custom_recipes.quantizer_factories import (
       mxfp8_factory,
       nvfp4_factory,
   )

   def my_factory(role: Optional[te.QuantizerRole]):
       if role is not None:
           if role.module_type == "linear" and role.name == "decoder.39.fc2":
               return te.IdentityQuantizer()
           if role.module_type == "grouped_linear":
               return nvfp4_factory(role)
       return mxfp8_factory(role)

The training framework or caller must pass semantic names to TE modules for
name-based selection, for example
``te.Linear(..., name="decoder.39.fc2")``.

Linear GEMM direction mapping
-----------------------------

``Linear`` and ``GroupedLinear`` training consume rowwise and columnwise
representations as follows:

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - GEMM
     - First operand
     - Second operand
   * - Forward (fprop)
     - ``weight.rowwise``
     - ``input.rowwise``
   * - Input gradient (dgrad)
     - ``weight.columnwise``
     - ``grad_output.rowwise``
   * - Weight gradient (wgrad)
     - ``input.columnwise``
     - ``grad_output.columnwise``

Therefore three per-GEMM formats, ``F`` for fprop, ``D`` for dgrad, and ``W``
for wgrad, map to tensor quantizers as:

.. code-block:: text

   input       = Hybrid(rowwise=F, columnwise=W)
   weight      = Hybrid(rowwise=F, columnwise=D)
   grad_output = Hybrid(rowwise=D, columnwise=W)

If two directions use the same quantizer configuration, a plain quantizer may
replace the corresponding hybrid. The two operands of each GEMM still need a
combination supported by that GEMM backend. TE may reject incompatible
quantizer pairs or unsupported layouts.

One factory may return both plain and hybrid quantizers (see the runnable
example below).

Combining rowwise and columnwise quantizers
-------------------------------------------

:class:`~transformer_engine.pytorch.HybridQuantizer` composes a rowwise and a
columnwise quantizer. Its output,
:class:`~transformer_engine.pytorch.HybridQuantizedTensor`, composes the
corresponding representations.

Choosing the columnwise source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``columnwise_source`` is a separate numerical recipe choice that controls the
source for the columnwise representation:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Columnwise source
   * - ``"original"``
     - The original high-precision tensor.
   * - ``"rowwise_dequantized"``
     - Dequantized rowwise representation.

For forward inputs and weights, ``"rowwise_dequantized"`` derives the backward
representation from the value consumed in the forward direction. This
can improve forward/backward numerical consistency and may affect convergence.
It does not recover information discarded by rowwise quantization.
``"original"`` instead derives both representations from the original tensor.
Choose the provenance as part of the numerical recipe.

Keeping directions in high precision
------------------------------------

:class:`~transformer_engine.pytorch.IdentityQuantizer` stores its input in the
held compute dtype, typically BF16, FP16, or FP32. It can keep a complete slot
in high precision or act as one child of a ``HybridQuantizer``:

.. code-block:: python

   return te.HybridQuantizer(
       rowwise_quantizer=mxfp8_factory(role),
       columnwise_quantizer=te.IdentityQuantizer(),
       columnwise_source="rowwise_dequantized",
   )

In this example, the rowwise direction uses MXFP8. The columnwise direction is
held in high precision, but its value is reconstructed from MXFP8. Use
``columnwise_source="original"`` when the high-precision direction should
retain the original input value instead.

Runnable example
----------------

The following synthetic example demonstrates base-factory composition, the
general three-format mapping, high-precision directions, and module/name
targeting. It uses only TE-native quantizers. MXFP8 and NVFP4 execution requires
supported hardware and software.

.. literalinclude:: pytorch_fine_grained_quantization_example.py
   :language: python
   :start-after: # START_FINE_GRAINED_QUANTIZATION_EXAMPLE
   :end-before: # END_FINE_GRAINED_QUANTIZATION_EXAMPLE

Run it from the repository root after installing TE:

.. code-block:: bash

   python docs/features/low_precision_training/fine_grained_quantization/pytorch_fine_grained_quantization_example.py

Recipe starting points
----------------------

The runnable example above is deliberately synthetic: it demonstrates the
expressiveness of the API, not a recommended training recipe. More realistic
starting points are available in
``transformer_engine/pytorch/custom_recipes/quantizer_factory_zoo.py``. Some
zoo factories encode externally described recipe structures or have specific
motivating evidence. They are still illustrative examples rather than
official, broadly validated defaults. Read each factory's rationale and
validate accuracy, convergence, and performance on the target workload.
Realizing the intended performance may require dedicated kernel enablement for
the selected operand formats, layouts, or module path; functional execution
does not imply that an optimized kernel path exists.

API reference
-------------

See the :doc:`PyTorch API <../../../api/pytorch>` for ``QuantizerRole``,
``HybridQuantizer``, ``IdentityQuantizer``, and their returned tensor types.
See the :doc:`Common API <../../../api/common>` for ``CustomRecipe``.
