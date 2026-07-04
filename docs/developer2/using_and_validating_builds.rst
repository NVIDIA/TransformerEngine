Using and Validating Development Builds
=======================================

Confirm the selected installation
---------------------------------

.. TODO: Show how to verify that Python imports the current checkout rather
   than a previously installed release, and how to locate the loaded common
   and framework extension libraries.

Run common functionality
------------------------

.. TODO: Define a minimal validation path for a common-only build and the
   public C API.

Run the PyTorch integration
---------------------------

.. TODO: Provide a minimal import and execution check that verifies both the
   Python package and compiled PyTorch binding.

Run the JAX integration
-----------------------

.. TODO: Provide a minimal import and execution check that verifies both the
   Python package and compiled JAX binding.

Choose validation after a change
--------------------------------

.. TODO: Map Python-only, binding, common C++, CUDA, packaging, and
   framework-specific changes to the smallest useful smoke checks before the
   full testing guidance is applied.

Use runnable examples
---------------------

.. TODO: Explain when the standalone programs under ``examples/`` are useful
   for validating a development build and how they differ from test and
   tutorial coverage.
