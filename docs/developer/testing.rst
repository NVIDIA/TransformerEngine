..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

.. _testing:

Testing Infrastructure
======================

This page describes how tests are organized, how to run them, and how to add new tests.

Test Hierarchy
--------------

Tests are organized into four levels by resource usage and execution frequency:

- **L0** — Fast, single-GPU unit tests. Run on every commit. All L0 tests must pass before
  submitting PRs.
- **L1** — Multi-GPU and integration tests. Run nightly. Includes distributed tests,
  Megatron-Core integration, ONNX export, and Thunder integration.
- **L2** — Extended tests. Run weekly. Broader coverage of edge cases and configurations.
- **L3** — Compatibility tests. Run per-release. Tests against specific external library
  versions (e.g., Flash Attention version compatibility).

Test scripts live in ``qa/L<level>_<name>/test.sh``. Each script sets up the environment
and invokes ``pytest`` or framework-specific test runners.

Test Locations
--------------

- ``tests/pytorch/`` — PyTorch unit tests (sanity, numerics, operators, etc.)
- ``tests/pytorch/distributed/`` — PyTorch distributed/multi-GPU tests
- ``tests/jax/`` — JAX unit tests
- ``tests/cpp/operator/`` — C++ CUDA kernel tests (cast, activation, transpose, softmax,
  normalization, GEMM, etc.)

Running Tests
-------------

**PyTorch — full L0 suite**:

.. code-block:: bash

   TE_PATH=/path/to/TransformerEngine bash qa/L0_pytorch_unittest/test.sh

**PyTorch — single test file**:

.. code-block:: bash

   python3 -m pytest tests/pytorch/test_sanity.py -v

**JAX — full L0 suite**:

.. code-block:: bash

   TE_PATH=/path/to/TransformerEngine bash qa/L0_jax_unittest/test.sh

**C++ kernel tests**:

C++ tests are built separately with CMake/Ninja and run via ``ctest``. These test CUDA
kernels directly (cast, activation, transpose, softmax, normalization, etc.) without
framework overhead.

.. code-block:: bash

   cd tests/cpp
   cmake -GNinja -Bbuild .
   cmake --build build
   ctest --test-dir build -j4

``LD_LIBRARY_PATH`` must include the path to the installed ``libtransformer_engine.so``.
The ``qa/L0_cppunittest/test.sh`` script handles this automatically by querying ``pip3
show transformer-engine``.

Key Environment Variables
-------------------------

Some tests require specific environment variables for reproducibility:

- ``PYTORCH_JIT=0`` and ``NVTE_TORCH_COMPILE=0`` — Disable JIT compilation and Torch
  Compile. Used for tests that need deterministic behavior, such as CUDA graph tests
  where capture correctness must be verified.
- ``TE_PATH`` — Root of the Transformer Engine source tree. Used by ``qa/`` test scripts
  to locate test files and configuration.

Adding a New Test
-----------------

1. Create the test file in the appropriate directory (``tests/pytorch/``, ``tests/jax/``,
   or ``tests/cpp/operator/``).
2. For Python tests, use ``pytest`` conventions (functions or classes prefixed with
   ``test_``).
3. For C++ tests, add a ``.cu`` file in ``tests/cpp/operator/`` and register it in the
   ``CMakeLists.txt`` in that directory.
4. Register the test in the corresponding ``qa/L0_*/test.sh`` script (or the appropriate
   level), setting any needed environment variables.
5. Verify the test passes locally before submitting.

Linting
-------

Pre-commit linting runs automatically on every commit via pre-commit.ci. You can also
run it locally:

.. code-block:: bash

   bash qa/format.sh

See ``CONTRIBUTING.rst`` at the repository root for full formatting and style guidelines.
