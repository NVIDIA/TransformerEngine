Build Configuration
===================

Configuration principles
------------------------

Transformer Engine's source build is configured primarily through environment
variables. Set them before invoking ``pip``; ``setup.py``, CMake, and the
framework extension builders read the environment while the build is being
configured.

Use three rules when changing build configuration:

* Set only the controls needed for the current change. The default development
  build should remain the baseline.
* Record non-default values in bug reports and test results. Architecture,
  optional-library, and numerical flags can materially change the compiled
  code.
* Reconfigure after changing a native option. If CMake reuse is questionable,
  select a new ``NVTE_CMAKE_BUILD_DIR`` rather than trusting cached state.

This page groups the important controls by purpose. ``docs/envvars.rst``
provides the broader environment-variable reference, while the implementation
in ``setup.py``, ``build_tools/``, and
``transformer_engine/common/CMakeLists.txt`` remains authoritative.

Framework selection
-------------------

``NVTE_FRAMEWORK`` is a comma-separated framework selection understood by
``build_tools/utils.py` (``get_frameworks``).

.. list-table::
   :header-rows: 1
   :widths: 28 26 46

   * - Value
     - Common library
     - Framework binding
   * - ``pytorch``
     - Built
     - PyTorch
   * - ``jax``
     - Built
     - JAX
   * - ``pytorch,jax`` or ``all``
     - Built once
     - PyTorch and JAX
   * - ``none``
     - Built
     - None
   * - Unset
     - Built
     - Every installed framework that can be imported

An explicit value is recommended for development and CI. Auto-detection is
sensitive to unrelated packages installed in the environment.

CUDA toolkit and target architectures
-------------------------------------

Toolkit discovery starts with ``CUDA_HOME``, then searches for ``nvcc`` on
``PATH``, and finally checks the conventional ``/usr/local/cuda``
location. The selected NVCC determines the CUDA toolkit version used to choose
default architecture targets.

.. list-table::
   :header-rows: 1
   :widths: 32 26 42

   * - Setting
     - Typical value
     - Effect
   * - ``CUDA_HOME``
     - ``/usr/local/cuda``
     - Selects the CUDA toolkit used for NVCC and headers when several
       toolkits are installed.
   * - ``CUDA_PATH``
     - Toolkit prefix
     - Additional CUDA location honored by CUDA-dependent components and
       library discovery.
   * - ``CUDNN_PATH`` or ``CUDNN_HOME``
     - cuDNN installation prefix
     - Identifies a nonstandard cuDNN installation. Runtime library paths must
       resolve the same installation used for the build.
   * - ``NVTE_CUDA_ARCHS``
     - ``"90"`` or ``"80;90"``
     - Selects compiled GPU architectures. More targets increase build time
       and binary size.
   * - ``NVTE_BUILD_USE_NVIDIA_WHEELS=1``
     - Disabled by default
     - Forces CUDA include discovery through installed NVIDIA Python packages
       instead of a toolkit include directory.
   * - ``NVTE_CMAKE_EXTRA_ARGS``
     - Space-separated CMake arguments
     - Appends advanced arguments to the common CMake configure command.

For local development, set ``NVTE_CUDA_ARCHS`` to the compute capabilities
that will actually execute the build. See
:ref:`developer-ngc-environment` for a discovery example. A target unsupported
by the selected NVCC fails during compilation; a target omitted from the list
is not expected to run on that GPU.

``NVTE_CMAKE_EXTRA_ARGS`` is an escape hatch rather than a stable public
interface. Prefer a named build option when one exists, and include the exact
value in any reproduction.

Build type, caching, and parallelism
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 22 46

   * - Setting
     - Default
     - Effect
   * - ``MAX_JOBS``
     - Maximum available
     - Standard cap on concurrent native compilation.
   * - ``NVTE_BUILD_MAX_JOBS``
     - Unset
     - TE-specific job cap. When set, the Python build helper gives it
       precedence over ``MAX_JOBS``.
   * - ``NVTE_BUILD_THREADS_PER_JOB``
     - ``1``
     - Threads used internally by each NVCC process.
   * - ``NVTE_BUILD_DEBUG=1``
     - ``0``
     - Selects a Debug common-library configuration and adds debug information
       and assertions to framework bindings.
   * - ``NVTE_CMAKE_BUILD_DIR``
     - ``build/cmake``
     - Chooses the common CMake state and object directory.
   * - ``NVTE_USE_CCACHE=1``
     - ``0``
     - Uses a compiler launcher for common C++ and CUDA compilation.
   * - ``NVTE_CCACHE_BIN``
     - ``ccache``
     - Selects the launcher executable; for example, an environment may use
       ``sccache``.

Parallelism and compiler threads multiply resource use. If compilation runs
out of memory, reduce ``MAX_JOBS`` first and leave
``NVTE_BUILD_THREADS_PER_JOB=1``. Caching affects build time but not the
resulting library; changing compilers, CUDA toolkits, or important flags should
use a separate cache namespace or a cleared cache.

Optional components
-------------------

These options change compiled capabilities and require additional libraries:

.. list-table::
   :header-rows: 1
   :widths: 34 20 46

   * - Setting
     - Default
     - Requirement and effect
   * - ``NVTE_WITH_NCCL_EP``
     - Enabled when applicable
     - Builds NCCL expert-parallel support for compute capability 9.0 or newer.
       ``NCCL_HOME`` can select the NCCL installation. Setting ``0``
       builds stubs and omits the NCCL EP implementation.
   * - ``NVTE_UB_WITH_MPI=1``
     - ``0``
     - Enables MPI bootstrap for userbuffers and requires ``MPI_HOME``.
   * - ``NVTE_ENABLE_NVSHMEM=1``
     - ``0``
     - Enables NVSHMEM support and requires ``NVSHMEM_HOME``.
   * - ``NVTE_WITH_CUBLASMP=1``
     - ``0``
     - Enables cuBLASMp paths and uses ``CUBLASMP_HOME`` or the matching
       installed NVIDIA package.
   * - ``NVTE_WITH_CUSOLVERMP=1``
     - ``0``
     - Enables distributed Newton-Schulz support and uses
       ``CUSOLVERMP_HOME``.

Enable the same capability in the common library and selected framework
binding by using one environment for the complete root build. Do not assemble
native outputs produced with conflicting feature flags.

Specialized semantic controls
-----------------------------

Some build options change numerical behavior rather than just build mechanics.
``NVTE_BUILD_ACTIVATION_WITH_FAST_MATH=1`` compiles selected activation
kernels with CUDA fast math, trading some numerical precision for performance.
``NVTE_BUILD_NUM_PHILOX_ROUNDS`` sets the number of Philox rounds compiled
into stochastic-rounding kernels, and must be a positive integer. Such
options should be changed only for targeted development or
experimentation. Record their value and rebuild every affected binary before
comparing numerical results.

Release-only controls
---------------------

The following controls belong to packaging and release workflows, not ordinary
editable builds:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Setting
     - Purpose
   * - ``NVTE_RELEASE_BUILD=1``
     - Switches the root build to the split-distribution layout and suppresses
       the local Git suffix in the package version.
   * - ``NVTE_BUILD_METAPACKAGE=1``
     - Builds the dependency-selecting ``transformer-engine`` metapackage
       without compiled extensions. It requires release-build mode.
   * - ``NVTE_NO_LOCAL_VERSION=1``
     - Suppresses the Git commit local-version suffix outside release mode.
   * - ``NVTE_PYTORCH_FORCE_BUILD=TRUE``
     - Prevents the PyTorch package builder from using a compatible cached
       release wheel and forces local compilation.
   * - ``NVTE_PYTORCH_FORCE_CXX11_ABI=TRUE``
     - Overrides the ABI reported by PyTorch for specialized packaging. An
       incorrect value produces an unloadable extension.

``NVTE_PROJECT_BUILDING`` is set internally by the setup scripts and should
not be set by contributors. The distribution workflow and its package-specific
controls are described in :doc:`split_distribution_builds`.
