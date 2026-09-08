..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _fine-grained-quantization-recipes:
.. _heterogeneous-quantization-recipes:

Fine-grained quantization recipes
=================================

Standard TE recipes quantize the whole model the same way. That is often too
coarse: one sensitive layer may need BF16 while the rest runs in MXFP8, or a
gradient GEMM may tolerate a cheaper format than the forward pass. Fine-grained
recipes lift this restriction: you write a small factory function that picks a
quantizer for each slot TE asks about, and pass it via
:class:`~transformer_engine.common.recipe.CustomRecipe` to the usual
:class:`~transformer_engine.pytorch.autocast`.
"Fine-grained" refers to the granularity of that choice (per module, tensor
role, and GEMM direction), not to the block size of the scaling factors.

.. warning::

   Fine-grained recipes are currently available only in the PyTorch API of
   TE.

.. warning::

   Fine-grained recipes and their construction APIs are experimental: API,
   validation, and kernel coverage may change without notice. This guide does
   not define a supported recipe or an expected accuracy/performance ordering.


Example: mixing MXFP8, NVFP4, and BF16
--------------------------------------

The `runnable example <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fine_grained_quantization/pytorch_fine_grained_quantization_example.py>`__
makes the following assignments:

.. raw:: html
   :file: img/fine_grained_assignments.svg

*Figure 1. Precision assignments per module and GEMM used throughout this
guide.*

A minimal factory implementing these assignments, plugged into the standard
TE autocast path:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         import transformer_engine.pytorch as te
         from transformer_engine.common.recipe import CustomRecipe
         from transformer_engine.pytorch.custom_recipes.quantizer_factories import (
             mxfp8_factory,
             nvfp4_factory,
         )


         def quantizer_factory(role):
             if role is not None and role.name == "demo.fc1":
                 if role.tensor_type == "input":
                     # wgrad keeps the original BF16 input
                     return te.HybridQuantizer(
                         rowwise_quantizer=mxfp8_factory(role),
                         columnwise_quantizer=te.IdentityQuantizer(),
                         columnwise_source="original",
                     )
                 if role.tensor_type == "weight":
                     # fprop in MXFP8, dgrad in NVFP4
                     return te.HybridQuantizer(
                         rowwise_quantizer=mxfp8_factory(role),
                         columnwise_quantizer=nvfp4_factory(role),
                         columnwise_source="rowwise_dequantized",
                     )
                 if role.tensor_type == "grad_output":
                     # dgrad in NVFP4, wgrad keeps the original BF16 gradient
                     return te.HybridQuantizer(
                         rowwise_quantizer=nvfp4_factory(role),
                         columnwise_quantizer=te.IdentityQuantizer(),
                         columnwise_source="original",
                     )
             if role is not None and role.name == "demo.fc2":
                 return te.IdentityQuantizer()  # whole module stays in BF16
             return mxfp8_factory(role)  # every other TE module in MXFP8


         recipe = CustomRecipe(qfactory=quantizer_factory)

         with te.autocast(enabled=True, recipe=recipe):
             output = model(inputs)

The complete, runnable version is available
`on GitHub <https://github.com/NVIDIA/TransformerEngine/blob/main/docs/examples/fine_grained_quantization/pytorch_fine_grained_quantization_example.py>`__
(requires Blackwell or later); run it from the repository root after
installing TE:

.. code-block:: bash

   python docs/examples/fine_grained_quantization/pytorch_fine_grained_quantization_example.py

CustomRecipe and quantizer factory
----------------------------------

:class:`~transformer_engine.common.recipe.CustomRecipe` is used like any
other TE recipe (``DelayedScaling``, ``MXFP8BlockScaling``, ...), but carries
no quantization logic of its own: TE asks your ``qfactory`` for a quantizer
whenever a module needs one.

Each TE module defines an ordered role list for the forward and backward
quantizer slots it needs. When module recipe state is initialized or rebuilt,
a ``CustomRecipe`` calls ``qfactory(role)`` once for every slot in that list.
It does not call the factory on every unchanged forward.

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         # QuantizerRole describes the slot being configured (fields below):
         #
         #   @dataclasses.dataclass(frozen=True)
         #   class QuantizerRole:
         #       module_type: str = ""
         #       tensor_type: str = ""
         #       name: str = ""


         def quantizer_factory(role: Optional[te.QuantizerRole]):
             # construct a fresh quantizer on every call
             ...
             # Boundary slots may pass role=None or a role with empty fields, so
             # always end with a default that covers every remaining role.
             return mxfp8_factory(role)


         # The factory plugs into the standard TE autocast path:
         recipe = CustomRecipe(qfactory=quantizer_factory)

         with te.autocast(enabled=True, recipe=recipe):
             output = model(inputs)

      **Module type**

      The kind of TE module that owns the slot, filled in by TE itself:

      * ``"linear"`` — ``Linear``, ``LayerNormLinear``, ``fc1``/``fc2`` in
        ``LayerNormMLP``, ``qkv``/``proj`` in ``MultiheadAttention``;
      * ``"grouped_linear"`` — ``GroupedLinear``;
      * ``"dpa"`` — ``DotProductAttention``.

      **Tensor type**

      Which tensor of that module the quantizer will process, also filled in
      by TE. For ``"linear"`` and ``"grouped_linear"``:

      * ``"input"`` — the activation (fprop, wgrad);
      * ``"weight"`` — (fprop, dgrad);
      * ``"grad_output"`` — the incoming gradient (dgrad, wgrad).

      For ``"dpa"``:

      * ``"qkv"`` — the query/key/value tensor;
      * ``"s"`` — the softmax output;
      * ``"do"`` — the output gradient;
      * ``"dp"`` — the gradient of ``"s"``.

      **Name**

      The identity of one concrete module instance, supplied by the caller:
      ``te.Linear(..., name="decoder.39.fc2")``. Composite TE modules may
      append suffixes such as ``.fc1``, ``.fc2``, and ``.proj``.

      The role vocabulary is experimental and may grow between releases —
      one more reason to end the factory with a total default. Treat the role
      strings as selectors, not a fixed enumeration. Prefer a module-level
      function for the factory itself, so that launchers and checkpointing
      setups can import or pickle it.

      TE provides factories for its native quantizers in
      ``transformer_engine.pytorch.custom_recipes.quantizer_factories``
      (``mxfp8_factory``, ``nvfp4_factory``, ...). They can be used as
      defaults or to construct ``HybridQuantizer`` children. Additional
      specialized recipes are available in
      ``transformer_engine.pytorch.custom_recipes.quantizer_factory_zoo``.

      The factory is not limited to TE-native quantizers: it may return your
      own :class:`~transformer_engine.pytorch.Quantizer` subclass, and custom
      quantizers can also serve as ``HybridQuantizer`` children. The GEMMs
      still need to receive representations in formats they support.

HybridQuantizer
---------------

During training, each tensor of a ``Linear`` or ``GroupedLinear`` layer feeds
two different GEMMs: its rowwise representation feeds one, its columnwise
representation the other (the exact operand layout is described in the
:doc:`Introduction <../introduction/introduction>`). Since those two GEMMs may
want different formats, the tensor needs a quantizer per direction:
:class:`~transformer_engine.pytorch.HybridQuantizer` composes a rowwise and a
columnwise quantizer, and its output,
:class:`~transformer_engine.pytorch.HybridQuantizedTensor`, composes the
corresponding representations.

.. tabs::

   .. tab:: PyTorch

      The following is pseudocode illustrating the composition:

      .. code-block:: text

         quantizer = te.HybridQuantizer(
             rowwise_quantizer=MXFP8Quantizer(fp8_dtype=DType.kFloat8E4M3),
             columnwise_quantizer=NVFP4Quantizer(),
             columnwise_source="original",  # or "rowwise_dequantized"
         )

         # Quantization yields a HybridQuantizedTensor whose rowwise
         # representation is MXFP8 and columnwise representation is NVFP4;
         # each GEMM consumes the representation it needs.
         qtensor = quantizer(tensor)

.. raw:: html
   :file: img/hybrid_quantizer.svg

*Figure 2. HybridQuantizer composes a rowwise and a columnwise quantizer; each
representation of the result feeds a different GEMM.*

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

*Figure 3. The columnwise representation can be derived from the original
high-precision tensor or from the dequantized rowwise representation.*

For forward inputs and weights, ``"rowwise_dequantized"`` derives the backward
representation from the value consumed in the forward direction. This
can improve forward/backward numerical consistency and may affect convergence.
It does not recover information discarded by rowwise quantization.
``"original"`` instead derives both representations from the original tensor.
Choose the provenance as part of the numerical recipe.

IdentityQuantizer
-----------------

:class:`~transformer_engine.pytorch.IdentityQuantizer` stores its input in the
held compute dtype, typically BF16, FP16, or FP32. It can keep a complete slot
in high precision or act as one child of a ``HybridQuantizer``:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         # whole slot in high precision (e.g. a module kept in BF16)
         quantizer = te.IdentityQuantizer()

         # one direction in high precision, the other quantized
         quantizer = te.HybridQuantizer(
             rowwise_quantizer=mxfp8_factory(role),
             columnwise_quantizer=te.IdentityQuantizer(),
             columnwise_source="rowwise_dequantized",
         )

Note that in the second example the columnwise direction is high precision but
holds the value reconstructed from MXFP8, not the original input — see
`Choosing the columnwise source`_ above.

Example: one format per GEMM
----------------------------

A natural way to design a recipe is to pick one format for each GEMM. To
translate that into quantizers, look at what each GEMM consumes — both of its
operands must be in that GEMM's format:

* **fprop** consumes ``input.rowwise`` and ``weight.rowwise``;
* **dgrad** consumes ``grad_output.rowwise`` and ``weight.columnwise``;
* **wgrad** consumes ``input.columnwise`` and ``grad_output.columnwise``.

Reading the same table per tensor gives the ``HybridQuantizer`` for each role.
For the example assignments (fprop in MXFP8, dgrad in NVFP4, wgrad in BF16):

.. code-block:: text

   input       = HybridQuantizer(rowwise=MXFP8,  columnwise=BF16)    # fprop | wgrad
   weight      = HybridQuantizer(rowwise=MXFP8,  columnwise=NVFP4)   # fprop | dgrad
   grad_output = HybridQuantizer(rowwise=NVFP4,  columnwise=BF16)    # dgrad | wgrad

.. raw:: html
   :file: img/fine_grained_linear_mapping.svg

*Figure 4. Each GEMM consumes one representation of each of its two operand
tensors; giving both operands the same format sets that GEMM's precision.*

If two directions use the same quantizer configuration, a plain quantizer may
replace the corresponding hybrid; one factory may return both plain and hybrid
quantizers.
The two operands of each GEMM still need a combination supported by that GEMM
backend. TE may reject incompatible quantizer pairs or unsupported layouts.

.. note::

   On supported hardware these recipes run TE's regular quantized kernels: the
   tensors are quantized on the GPU and the GEMMs execute in the selected
   low-precision formats. TE does not fall back to fake quantization
   (quantize-dequantize followed by a high-precision GEMM).


Validating and optimizing a recipe
----------------------------------

The factory API can express more recipes than TE has kernels for, so any
assignment lands in one of three buckets:

* **Fast** — quantization hits TE's fused kernels and every GEMM runs a
  native low-precision implementation.
* **Correct but potentially unoptimized** — the recipe executes, but some
  selected paths may not have fused or optimized implementations in the
  current TE release. For example, ``HybridQuantizer`` may produce its rowwise
  and columnwise representations in separate kernel launches; future releases
  may fuse this work.
* **Rejected** — the two operands of some GEMM end up in a combination of
  formats or layouts that no GEMM backend supports, and TE raises an error.
  This can happen with plain and hybrid quantizers alike.

Before adopting a recipe for a real workload, check that:

* it executes at all on the target GPU, software version, and modules;
* it runs on optimized kernels rather than fallback paths;
* accuracy and convergence hold on the target model and distributed setup;
* throughput and memory actually improve on the target workload.

The unoptimized paths are still useful: accuracy and convergence experiments can run
on them before dedicated kernels exist, so the precision of each GEMM can be
treated as an accuracy/performance trade-off to explore.

API reference
-------------

See the :doc:`PyTorch API <../../../api/pytorch>` for ``QuantizerRole``,
``HybridQuantizer``, ``IdentityQuantizer``, and their returned tensor types.
See the :doc:`Common API <../../../api/common>` for ``CustomRecipe``.
