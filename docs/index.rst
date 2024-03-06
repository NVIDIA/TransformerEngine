..
    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Transformer Engine documentation
==============================================

.. ifconfig:: "dev" in release

   .. warning::
      You are currently viewing unstable developer preview of the documentation.
      To see the documentation for the latest stable release, refer to:

      * `Release Notes <https://docs.nvidia.com/deeplearning/transformer-engine/release-notes/index.html>`_
      * `Developer Guide <https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html>`_ (stable version of this page)

.. include:: ../README.rst
   :start-after: overview-begin-marker-do-not-remove
   :end-before: overview-end-marker-do-not-remove

.. toctree::
   :hidden:

   Home <self>

.. toctree::
   :hidden:
   :caption: Getting Started

   installation
   examples/quickstart.ipynb

.. toctree::
   :hidden:
   :caption: Python API documentation

   api/common
   api/framework

.. toctree::
   :hidden:
   :caption: Examples and Tutorials

   examples/fp8_primer.ipynb
   examples/advanced_optimizations.ipynb
   examples/te_llama/tutorial_accelerate_hf_llama_with_te.ipynb

.. toctree::
   :hidden:
   :caption: Advanced

   api/c/index
