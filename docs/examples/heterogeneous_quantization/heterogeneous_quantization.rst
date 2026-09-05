..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _fine-grained-quantization-tutorial:
.. _heterogeneous-quantization-tutorial:

Building a heterogeneous quantization recipe
==============================================

This tutorial demonstrates factory composition, module and name targeting,
the general three-format GEMM mapping, and high-precision directions. See
:doc:`Heterogeneous quantization recipes
<../../features/low_precision_training/heterogeneous_quantization/heterogeneous_quantization>`
for the API concepts and direction mapping.

Constructing a factory
----------------------

A robust factory follows four construction rules:

* Return a quantizer for every role. Return ``IdentityQuantizer`` for an
  intentional high-precision slot; do not return ``None``.
* Constructing a fresh quantizer for every call is recommended.
  ``HybridQuantizer`` owns and configures its rowwise and columnwise children.
* A module-level function is the most portable factory definition, especially
  when a launcher or checkpointing setup needs to import or pickle it.
* Treat role strings as selectors, not a fixed enumeration. Preserve a base
  factory fallback for roles the factory does not recognize.

For example, compose a TE-native factory by keeping one named ``linear``
module in high precision, using NVFP4 for every ``grouped_linear`` role, and
retaining MXFP8 as the global fallback:

.. tabs::

   .. tab:: PyTorch

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

Runnable example
----------------

The following synthetic example demonstrates base-factory composition, the
general three-format mapping, high-precision directions, and module/name
targeting. It uses only TE-native quantizers. MXFP8 and NVFP4 execution requires
supported hardware and software.

.. tabs::

   .. tab:: PyTorch

      .. raw:: html

         <div style="background: #f0f4f8; border-left: 3px solid #5c7cfa; padding: 6px 12px; font-size: 13px; color: #495057; margin-bottom: 0; border-radius: 4px 4px 0 0;">
            Requires SM100 (Blackwell) or later
         </div>

      .. literalinclude:: pytorch_heterogeneous_quantization_example.py
         :language: python
         :start-after: # START_HETEROGENEOUS_QUANTIZATION_EXAMPLE
         :end-before: # END_HETEROGENEOUS_QUANTIZATION_EXAMPLE

Run it from the repository root after installing TE:

.. code-block:: bash

   python docs/examples/heterogeneous_quantization/pytorch_heterogeneous_quantization_example.py

Recipe starting points
----------------------

The runnable example above is deliberately synthetic: it demonstrates the
expressiveness of the API, not a recommended training recipe.

The TE-native base factories in
``transformer_engine/pytorch/custom_recipes/quantizer_factories.py`` construct
standard TE quantizers. They can be used directly as factory fallbacks or
composed as children of a ``HybridQuantizer``.

More specialized starting points are available in
``transformer_engine/pytorch/custom_recipes/quantizer_factory_zoo.py``. Some
zoo factories encode externally described recipe structures or have specific
motivating evidence. They remain illustrative examples rather than official,
broadly validated defaults; read each factory's rationale before adapting it.

Validating and optimizing a recipe
----------------------------------

A factory can describe assignments beyond current optimized kernel coverage.
Before adopting an assignment for a workload:

* Confirm that the module, GEMM layout and shape, software version, and GPU can
  execute it.
* Check whether the selected module path has an appropriate optimized kernel
  and integration.
* Validate accuracy and convergence on the target model and distributed setup.
* Benchmark throughput and memory on the target workload.

Fine-grained assignments provide a way to explore an accuracy/performance
slider by varying precision and quantization logic per GEMM. The recipes that
can be realized efficiently are constrained by available kernels and
integrations. Accuracy and convergence experiments can run on functionally
executable, non-optimized paths before dedicated kernels are available.
