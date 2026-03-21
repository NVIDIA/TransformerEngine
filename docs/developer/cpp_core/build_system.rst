..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _build-system:

Build System
============

Transformer Engine has two build paths: a monolithic developer build and a split
pip-wheel build for distribution.

Monolithic Developer Build
--------------------------

This is the build most developers use during day-to-day work. A single ``pip install``
invocation builds everything — the C++ core library and the framework-specific extensions
— into one package:

.. code-block:: bash

   pip install -e . -v --no-build-isolation

The build pipeline:

.. code-block:: text

   pip install -e .
         │
         ▼
   setup.py
     ├── Checks/fetches git submodules (CUTLASS, cuDNN-frontend)
     ├── Detects CUDA toolkit and GPU architectures
     ├── Invokes CMake to build libtransformer_engine.so (C++ core)
     └── Builds framework-specific extensions:
         ├── PyTorch: pybind11 extension via torch.utils.cpp_extension
         └── JAX:     XLA FFI extension via pybind11

The C++ core is built by CMake (``transformer_engine/common/CMakeLists.txt``) and
produces ``libtransformer_engine.so``. Framework-specific extensions are built by each
framework's own extension system (PyTorch uses ``torch.utils.cpp_extension.CppExtension``,
JAX uses pybind11 directly). Both link against the shared core library.

Pip Wheel Build
---------------

The distribution build (used for PyPI releases) creates multiple separate packages
rather than one monolithic install. This is controlled by the ``NVTE_RELEASE_BUILD``
environment variable. When set, ``setup.py`` produces a metapackage with no compiled
extensions — the core library and framework extensions are packaged and distributed
separately.

Key Environment Variables
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``NVTE_FRAMEWORK``
     - Select framework: ``pytorch``, ``jax``, ``all``, ``none``, or auto-detect (default)
   * - ``NVTE_CUDA_ARCHS``
     - Semicolon-separated GPU architectures (e.g., ``"80;90;100"``)
   * - ``MAX_JOBS``
     - Maximum parallel compilation jobs
   * - ``CUDA_PATH``
     - Custom CUDA toolkit path
   * - ``CUDNN_PATH``
     - Custom cuDNN path
   * - ``NVTE_RELEASE_BUILD``
     - Build as split pip-wheel metapackage (for distribution)
   * - ``NVTE_USE_CCACHE``
     - Set to ``1`` to enable ccache for faster incremental builds
   * - ``NVTE_CCACHE_BIN``
     - Path to ccache binary (default: ``ccache``). E.g. ``sccache`` for CI.
   * - ``NVTE_BUILD_DEBUG``
     - Set to ``1`` to build C++ extensions with debug symbols

NVTE_CUDA_ARCHS
^^^^^^^^^^^^^^^^

This variable controls which GPU architectures the CUDA kernels are compiled for. It
directly affects build time (more architectures = longer builds) and which GPUs the
resulting binary supports.

If not set, the build system auto-detects based on the installed CUDA toolkit version:

- CUDA 13.0+: ``"75;80;89;90;100;120"``
- CUDA 12.8–12.9: ``"70;80;89;90;100;120"``
- CUDA < 12.8: ``"70;80;89;90"``

For development, set this to only your target GPU to minimize build time:

.. code-block:: bash

   NVTE_CUDA_ARCHS="90" pip install -e . -v --no-build-isolation

Git Submodules
--------------

The build requires two submodules in ``3rdparty/``:

- **CUTLASS** (``3rdparty/cutlass``) — NVIDIA's CUDA Templates for Linear Algebra
  Subroutines. Used for custom GEMM kernels and attention kernels.
- **cuDNN Frontend** (``3rdparty/cudnn-frontend``) — C++ frontend for cuDNN graph API.
  Used for fused attention and normalization.

``setup.py`` automatically initializes these if they are missing, but manual setup is
sometimes needed:

.. code-block:: bash

   git submodule update --init --recursive

Developer Build Tips
--------------------

**Fast incremental rebuild** (C++ only, skip Python):

.. code-block:: bash

   cmake --build build/cmake --parallel 4

**Enable ccache** for fast rebuilds when iterating on C++ code:

.. code-block:: bash

   NVTE_USE_CCACHE=1 pip install -e . -v --no-build-isolation

   # Or with sccache (common in CI):
   NVTE_USE_CCACHE=1 NVTE_CCACHE_BIN=sccache pip install -e . -v --no-build-isolation

**Verbose CMake output** (for debugging build issues):

.. code-block:: bash

   cmake --build build/cmake --verbose --parallel 4

Common Build Failures
---------------------

Build failures are common for new developers. Here are the most frequent issues:

- ``fatal error: cudnn.h: No such file or directory`` → Set ``CUDNN_PATH`` to your
  cuDNN installation directory.
- ``nvcc fatal: Unsupported gpu architecture`` → Your ``NVTE_CUDA_ARCHS`` includes an
  architecture not supported by your CUDA toolkit version. Either upgrade CUDA or remove
  the unsupported architecture.
- Submodule errors (``CMake Error ... CUTLASS``) → Run
  ``git submodule update --init --recursive``.
- Out of memory during compilation → Reduce ``MAX_JOBS`` (try ``MAX_JOBS=1``). Each
  compilation unit for templated CUDA kernels can consume several GB of memory.
- ``ModuleNotFoundError: No module named 'torch'`` → You need ``--no-build-isolation``
  so that the build can find your installed PyTorch.
- Build succeeds but import fails with undefined symbols → This usually means the core
  library and the framework extension were built against different CUDA versions. Clean
  the build (``rm -rf build/``) and rebuild from scratch.
