..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Developer Guide
===============

This section documents the internal architecture, design decisions, and implementation
details of Transformer Engine. It is intended for contributors, maintainers, and AI
agents working on the codebase.

.. note::

   This guide describes the *internal* structure of Transformer Engine, not the
   user-facing API. For the public API, see :doc:`/api/pytorch` and :doc:`/api/jax`.

How to Use This Guide
---------------------

- **New contributors**: Start with :doc:`architecture_overview` for the big picture,
  then read :doc:`linear_walkthrough` for an end-to-end trace through a concrete PyTorch
  module. For building from source, see :doc:`/installation`. For running tests, see
  :doc:`testing`. For contributing guidelines, see ``CONTRIBUTING.rst`` at the repository
  root.
- **Working on new quantization recipe**: See :doc:`quantization/index` for the 
  Quantizer/Storage/Tensor design and :doc:`cpp_core/type_system` for the underlying C/C++ types.
- **Working on attention**: See :doc:`attention/index` for backend selection and kernel
  organization.
- **Working on distributed training**: See :doc:`distributed/index` for tensor/sequence parallelism
  and communication overlap.
- **Framework-specific work**: See :doc:`pytorch_frontend/index` or
  :doc:`jax_frontend/index`.

.. toctree::
   :caption: Developer Guide
   :maxdepth: 2

   architecture_overview
   linear_walkthrough
   cpp_core/index
   quantization/index
   pytorch_frontend/index
   jax_frontend/index
   attention/index
   distributed/index
   testing
