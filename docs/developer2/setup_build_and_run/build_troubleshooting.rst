Build and Setup Troubleshooting
===============================

Troubleshoot the first failing stage. A later import or runtime error is often
a consequence of an earlier environment or linking mismatch, so preserve the
original verbose output instead of rebuilding repeatedly with different
settings.

Collect build diagnostics
-------------------------

Reproduce the failure from the repository root and retain the complete log:

.. code-block:: bash

   pip install -e . -vvv --no-build-isolation 2>&1 | tee /tmp/te-build.log

Record the settings that affect native compilation:

.. code-block:: bash

   env | sort | grep -E '^(CUDA|CUDNN|NCCL|MPI|NVSHMEM|CUBLASMP|CUSOLVERMP|NVTE|MAX_JOBS|CC|CXX|LD_LIBRARY_PATH)='

Also capture the selected tools:

.. code-block:: bash

   python --version
   python -m pip --version
   nvcc --version
   c++ --version
   cmake --version
   ninja --version
   git submodule status --recursive

Record framework and backend information with the framework that failed:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         import torch

         print(torch.__version__)
         print(torch.version.cuda)
         print(torch._C._GLIBCXX_USE_CXX11_ABI)
         print(torch.cuda.is_available())

   .. tab:: JAX

      .. code-block:: python

         import jax

         print(jax.__version__)
         print(jax.devices())

For an import or runtime failure, also print the package and shared-library
paths from :doc:`using_and_validating_builds`. A useful report states the
container image or host OS, GPU model, driver, exact Git revision, framework
selection, CUDA architectures, and whether a new CMake build directory changes
the result.

Toolchain and configuration failures
------------------------------------

Use the last command printed before the error to identify the failing layer:

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Stage
     - Typical symptom
     - First checks
   * - Python build setup
     - Framework or pybind11 cannot be imported.
     - Confirm the intended Python and ``pip``, install the framework first,
       and retain ``--no-build-isolation``.
   * - CMake configure
     - CUDA compiler, headers, cuDNN, or another package is not found.
     - Check ``CUDA_HOME``, compiler paths, library prefixes, and enabled
       optional components.
   * - Common compilation
     - NVCC rejects an architecture or a header is missing.
     - Compare ``NVTE_CUDA_ARCHS`` with the selected NVCC and inspect the
       first compiler error.
   * - Framework binding compilation
     - XLA, PyTorch, CUDA, MPI, or NVSHMEM headers are missing.
     - Confirm the selected framework installation and that optional flags are
       consistent with the common build.
   * - Install or import
     - Shared library is absent or a symbol cannot be resolved.
     - Inspect loaded artifact paths and compare the build and runtime library
       environments.

If ``nvcc`` is missing or the wrong version is selected, set
``CUDA_HOME`` and ensure ``$CUDA_HOME/bin`` precedes other toolkits on
``PATH``. An ``unsupported gpu architecture`` error means at least one
``NVTE_CUDA_ARCHS`` target is not supported by that NVCC; remove the target
or select a newer compatible toolkit.

Missing optional-library headers should not be worked around by adding
unrelated include paths. Either provide the installation required by the
feature and its documented ``*_HOME`` variable, or disable the feature. For
JAX, ``build_tools/jax.py`` (``xla_path``) normally obtains XLA headers
from the installed JAX FFI package; ``XLA_HOME`` is the fallback for a
deliberate external XLA tree.

Submodule and source-dependency failures
----------------------------------------

Inspect the recorded state:

.. code-block:: bash

   git submodule status --recursive

A leading ``-`` indicates an uninitialized submodule. Initialize all recorded
revisions with:

.. code-block:: bash

   git submodule update --init --recursive

A leading ``+`` indicates that the submodule is checked out at a commit
other than the one recorded by the parent repository. If that was not
intentional, restore the recorded revision with the same update command. If it
was intentional development, use
``NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1`` and disclose the override in
test results.

Do not use the bypass merely to make a stale checkout compile. The build uses
submodule headers and sources directly, so an unintended revision can produce
compile errors or behavior that does not correspond to the parent commit.

Resource and incremental-build failures
---------------------------------------

CUDA template compilation may use substantial memory. If the system becomes
unresponsive, NVCC is killed, or the build fails with an out-of-memory error:

1. set ``MAX_JOBS=1``;
2. keep ``NVTE_BUILD_THREADS_PER_JOB=1``;
3. target only the needed GPU with ``NVTE_CUDA_ARCHS``;
4. retry from the existing build tree.

If the error appears only in an incremental build, point
``NVTE_CMAKE_BUILD_DIR`` at a new empty directory and rerun the same command.
Success with the same compiler, environment, and sources isolates the old
CMake state as the cause. Changing the cache directory or temporarily
disabling ``NVTE_USE_CCACHE`` can similarly isolate a bad compiler-cache
entry.

Do not begin troubleshooting by deleting the complete checkout or submodules.
Preserve the failing build log and user changes, and remove only identified
generated artifacts.

Import, linking, and ABI failures
---------------------------------

First confirm that ``transformer_engine.__file__`` points to the intended
checkout. An NGC container already contains a released Transformer Engine
installation; the editable package must take precedence during development.

``transformer_engine/common/__init__.py``
(``_get_shared_object_file``) raises an error when it finds either no
matching shared object or multiple candidates. Multiple candidates usually
mean that editable, source-install, and split-wheel artifacts have been mixed
in one environment. Use a clean environment or uninstall the unrelated
distribution rather than copying libraries until the import succeeds.

Undefined symbols and loader errors commonly indicate one of these mismatches:

* the framework was upgraded after its private TE binding was compiled;
* common and framework libraries came from different builds or versions;
* the build and runtime resolve different CUDA, cuDNN, NCCL, or C++ ABI
  libraries;
* a native optional component was enabled in only part of the build.

Rebuild the common library and selected binding together after a framework,
compiler, or CUDA change. For PyTorch, record
``torch._C._GLIBCXX_USE_CXX11_ABI``; the package builder normally mirrors
that value. For cuDNN loading failures in a virtual environment, ensure the
build and ``LD_LIBRARY_PATH`` resolve the same cuDNN installation rather
than mixing a system library with a Python-distributed library.

A JAX error reporting no registered CUDA implementation for a TE custom call
usually means the JAX extension was not built for the active environment.
Reinstall with ``NVTE_FRAMEWORK=jax`` and
``--no-build-isolation``, then confirm the loaded
``transformer_engine_jax`` path.

Distribution-build failures
---------------------------

Release artifacts add package metadata and compatibility dimensions to the
native build. Diagnose them in the container and wheelhouse produced by
``build_tools/wheel_utils/``:

* inspect ``wheelhouse/logs/metapackage.txt``, ``common.txt``,
  ``torch.txt``, or ``jax.txt`` for the first failure;
* verify that every artifact has the same Transformer Engine version;
* confirm the requested CUDA major and manylinux platform tag;
* distinguish framework source distributions from compiled wheels;
* install into a clean environment with the checkout absent from
  ``PYTHONPATH``;
* run ``python -m pip check`` before import and runtime tests.

The PyTorch package first looks for a cached wheel matching TE, CUDA major,
PyTorch, Python, platform, and C++ ABI. A missing cached wheel is not itself an
error: the builder falls back to source compilation. Set
``NVTE_PYTORCH_FORCE_BUILD=TRUE`` when the source-build path is the behavior
under test.

Use ``qa/L0_pytorch_wheel/test.sh`` and
``qa/L0_jax_wheel/test.sh`` as the executable references for package
assembly and clean import checks. See :doc:`split_distribution_builds` for
the artifact relationships.
