Unified Editable Build
======================

Purpose and scope
-----------------

.. TODO: Explain that the root source build compiles the common library and
   all selected framework extensions together for development.

Canonical command
-----------------

.. TODO: Document ``pip install -e . -v --no-build-isolation`` from the
   repository root, including its prerequisites and expected success criteria.

Select framework integrations
-----------------------------

.. TODO: Document framework auto-detection and explicit selection through
   ``NVTE_FRAMEWORK``, including common-only, PyTorch-only, JAX-only, and
   combined builds.

Build pipeline
--------------

.. TODO: Trace the root ``setup.py`` flow through the common CMake build,
   PyTorch ``CppExtension``, and JAX pybind11 extension. Identify the relevant
   functions and files without relying on line numbers.

Build outputs
-------------

.. TODO: Document the editable installation layout, common shared library,
   framework extension modules, CMake build tree, and framework-owned Python
   package files.

Incremental rebuilds
--------------------

.. TODO: Explain which edits require reinstalling or rebuilding, how
   ``NVTE_CMAKE_BUILD_DIR`` affects reuse, and how compiler caching and build
   parallelism fit into the development loop.

Clean rebuilds
--------------

.. TODO: Define a safe clean-build procedure and distinguish generated build
   artifacts from source files.
