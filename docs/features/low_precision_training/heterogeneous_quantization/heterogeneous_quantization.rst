..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _fine-grained-quantization-recipes:
.. _heterogeneous-quantization-recipes:

Heterogeneous quantization recipes
==================================

Transformer Engine (TE) supports heterogeneous quantization recipes that select
quantizers by module or operation type, tensor role, module or operation
instance name, and rowwise/columnwise direction or GEMM type. Heterogeneous recipes
provide role-aware mixed-precision and quantization configuration at module,
tensor-role, and GEMM-direction granularity. A
:class:`~transformer_engine.common.recipe.CustomRecipe` supplies a quantizer
factory to the standard :class:`~transformer_engine.pytorch.autocast` path. The
factory can compose TE-native quantizers with
:class:`~transformer_engine.pytorch.HybridQuantizer` and
:class:`~transformer_engine.pytorch.IdentityQuantizer`.

This guide covers PyTorch, TE-native quantizers, and static recipe construction.

Mixing formats at a glance
--------------------------

A single ``CustomRecipe`` can mix quantization formats and high precision
across modules and GEMM directions. The accompanying
:doc:`tutorial <../../../examples/heterogeneous_quantization/heterogeneous_quantization>`
makes the following assignments:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Module assignment
     - Fprop
     - Dgrad
     - Wgrad
   * - ``demo.fc1``
     - MXFP8
     - NVFP4
     - BF16
   * - ``demo.fc2``
     - BF16
     - BF16
     - BF16
   * - Other TE modules
     - MXFP8
     - MXFP8
     - MXFP8

Once the factory defines these assignments, the recipe uses the standard TE
autocast path:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         recipe = CustomRecipe(qfactory=quantizer_factory)

         with te.autocast(enabled=True, recipe=recipe):
             output = model(inputs)

The complete factory appears in the tutorial. The same machinery can also:

* assign formats by module or operation type;
* override a named module instance;
* choose fprop, dgrad, and wgrad formats independently; and
* keep selected slots or directions in high precision.

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

The training framework or caller must pass semantic names to TE modules for
name-based selection, for example
``te.Linear(..., name="decoder.39.fc2")``.

See the
:doc:`tutorial <../../../examples/heterogeneous_quantization/heterogeneous_quantization>`
for factory construction rules and complete examples.

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

.. raw:: html
   :file: img/heterogeneous_linear_mapping.svg

*Figure 1. Fine-grained tensor representations provide matching operand
formats for each Linear GEMM.*

If two directions use the same quantizer configuration, a plain quantizer may
replace the corresponding hybrid. The two operands of each GEMM still need a
combination supported by that GEMM backend. TE may reject incompatible
quantizer pairs or unsupported layouts.

One factory may return both plain and hybrid quantizers (see the
:doc:`tutorial <../../../examples/heterogeneous_quantization/heterogeneous_quantization>`).

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

.. raw:: html
   :file: img/hybrid_columnwise_source.svg

*Figure 2. The columnwise representation can be derived from the original
high-precision tensor or from the dequantized rowwise representation.*

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

.. tabs::

   .. tab:: PyTorch

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

Tutorial
--------

See :doc:`Building a heterogeneous quantization recipe
<../../../examples/heterogeneous_quantization/heterogeneous_quantization>` for
factory composition, a runnable example, recipe starting points, and workload
validation guidance.

Support status
--------------

.. note::

   With TE-native low-precision quantizers on supported hardware and kernel
   paths, recipes use TE's native GPU quantization and low-precision GEMM
   implementations. No fake quantization or high-precision GEMM emulation is
   involved on these paths.

.. warning::

   Fine-grained recipes and their construction APIs are experimental. API,
   validation, and kernel coverage may change without notice. This guide does
   not define a supported recipe or an expected accuracy/performance ordering.

API reference
-------------

See the :doc:`PyTorch API <../../../api/pytorch>` for ``QuantizerRole``,
``HybridQuantizer``, ``IdentityQuantizer``, and their returned tensor types.
See the :doc:`Common API <../../../api/common>` for ``CustomRecipe``.
