..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _build-system:

Build System
============

Transformer Engine uses a hybrid build system: a Python ``setup.py`` orchestrates the
top-level install, while CMake handles the C++/CUDA compilation of the core library.

Build Pipeline
--------------

.. code-block:: text

   pip install -e . -v --no-build-isolation
         │
         ▼
   setup.py
     ├── Checks/fetches git submodules (CUTLASS, cuDNN-frontend)
     ├── Detects CUDA toolkit and GPU architectures
     ├── Invokes CMake to build transformer_engine/common/
     └── Builds framework-specific extensions:
         ├── PyTorch: pybind11 extensions via torch.utils.cpp_extension
         └── JAX:     XLA FFI extensions

Key Environment Variables
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``NVTE_FRAMEWORK``
     - Select framework: ``pytorch``, ``jax``, or auto-detect (default)
   * - ``NVTE_CUDA_ARCHS``
     - Semicolon-separated GPU architectures (e.g., ``"80;90;100"``)
   * - ``MAX_JOBS``
     - Maximum parallel compilation jobs
   * - ``CUDA_PATH``
     - Custom CUDA toolkit path
   * - ``CUDNN_PATH``
     - Custom cuDNN path
   * - ``NVTE_RELEASE_BUILD``
     - Enable release-mode optimizations

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

CMake Structure
---------------

The CMake build lives under ``transformer_engine/common/CMakeLists.txt`` and produces
``libtransformer_engine.so``. Key configuration:

- **Architecture flags**: ``NVTE_CUDA_ARCHS`` maps to CMake's
  ``CMAKE_CUDA_ARCHITECTURES``. Only specified architectures are compiled, which
  significantly affects build time.
- **Conditional compilation**: Some features are gated on CUDA version (e.g., FP4 support
  requires CUDA 12.8+).
- **Header paths**: cuDNN headers are found via ``CUDNN_PATH`` or system paths.

Developer Build Tips
--------------------

**Fast incremental rebuild** (C++ only, skip Python):

.. code-block:: bash

   cmake --build build/cmake --parallel 4

**Minimal architecture for fast iteration**:

.. code-block:: bash

   NVTE_CUDA_ARCHS="90" pip install -e . -v --no-build-isolation

**Verbose CMake output** (for debugging build issues):

.. code-block:: bash

   cmake --build build/cmake --verbose --parallel 4

**Common build issues:**

- ``fatal error: cudnn.h: No such file or directory`` → Set ``CUDNN_PATH``
- ``nvcc fatal: Unsupported gpu architecture`` → Check ``NVTE_CUDA_ARCHS`` matches
  your GPU
- Submodule errors → Run ``git submodule update --init --recursive``
- Out of memory during compilation → Reduce ``MAX_JOBS`` (try ``MAX_JOBS=1``)
