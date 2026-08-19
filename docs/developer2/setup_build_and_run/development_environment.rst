Development Environment
=======================

The recommended way to prepare a Transformer Engine development environment
is to start from an NVIDIA GPU Cloud (NGC) framework container. The PyTorch
and JAX containers provide compatible framework, CUDA, cuDNN, compiler, and
Python environments, which avoids reproducing that compatibility setup on the
host.

.. _developer-ngc-environment:

Recommended: NGC container
--------------------------

Choose the container for the framework being modified:

.. list-table::
   :header-rows: 1
   :widths: 20 40 20

   * - Integration
     - Image
     - Build selection
   * - PyTorch
     - ``nvcr.io/nvidia/pytorch:<tag>``
     - ``NVTE_FRAMEWORK=pytorch``
   * - JAX
     - ``nvcr.io/nvidia/jax:<tag>``
     - ``NVTE_FRAMEWORK=jax``

For example, ``26.06-py3`` identifies the June 2026 container release. Treat
the tag as an example rather than a permanent recommendation. Select a
container compatible with the hardware and the version against which the
change will be tested. Current tags are listed in the `PyTorch NGC catalog
<https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`__ and the
`JAX NGC catalog
<https://catalog.ngc.nvidia.com/orgs/nvidia/containers/jax>`__.

Start the container
^^^^^^^^^^^^^^^^^^^

From the repository root on the host, initialize the Git submodules:

.. code-block:: bash

   git submodule update --init --recursive

Then start the container for the framework being developed:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: bash

         export TE_IMAGE=nvcr.io/nvidia/pytorch:26.06-py3

         docker run --gpus all --rm -it --ipc=host \
             --volume "$PWD:/workspace/TransformerEngine" \
             --workdir /workspace/TransformerEngine \
             "$TE_IMAGE"

   .. tab:: JAX

      .. code-block:: bash

         export TE_IMAGE=nvcr.io/nvidia/jax:26.06-py3

         docker run --gpus all --rm -it --ipc=host \
             --volume "$PWD:/workspace/TransformerEngine" \
             --workdir /workspace/TransformerEngine \
             "$TE_IMAGE"

These commands assume that the current directory is the Transformer Engine
checkout. Each provides all visible host GPUs, mounts the checkout at
``/workspace/TransformerEngine``, and makes that directory the container's
working directory.

``--ipc=host`` is not required for compilation, but it avoids the small
default shared-memory limit when the same container is subsequently used for
framework and distributed tests. Use a more restrictive GPU or IPC setup
when the local execution environment requires it.

Prepare the package environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

NGC containers set ``PIP_CONSTRAINT`` to keep Python packages compatible with
the versions shipped in the image. That constraint can prevent an editable
installation of a different Transformer Engine version. Unset it inside the
container before building the checkout:

.. code-block:: bash

   unset PIP_CONSTRAINT

Customize the development build
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default build targets several CUDA architectures and uses all available
parallel build jobs. Those defaults produce broadly usable binaries, but they
can make a local development build unnecessarily slow or exhaust the
container's memory. Configure both settings before invoking ``pip``.

Target GPU architectures
~~~~~~~~~~~~~~~~~~~~~~~~

``NVTE_CUDA_ARCHS`` is a semicolon-separated list of CUDA compute
capabilities. Each additional architecture causes another set of CUDA kernels
to be compiled, so a development build should normally include only the GPUs
on which it will run.

Use ``nvidia-smi`` to inspect the compute capabilities visible in the
container:

.. code-block:: bash

   nvidia-smi --query-gpu=compute_cap --format=csv,noheader

Remove the decimal point when setting the build target. For example, a reported
compute capability of ``9.0`` corresponds to:

.. code-block:: bash

   export NVTE_CUDA_ARCHS="90"

If the build must run on more than one GPU architecture, include each required
target, for example ``"80;90"``. Quote multi-architecture values so the
shell does not interpret the semicolon as a command separator. A binary built
for only one architecture is not intended to run on GPUs with a different
compute capability.

Limit parallel compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~

``MAX_JOBS`` limits the number of compiler processes that may run in
parallel. Native and CUDA compilation can consume substantial memory per job;
using every CPU core may cause memory pressure or an out-of-memory failure
instead of reducing the build time.

Choose a value appropriate for the CPU and memory available to the container.
For example:

.. code-block:: bash

   export MAX_JOBS=4

Reduce the value further if compilation exhausts memory; ``MAX_JOBS=1`` is
the lowest-resource fallback. A larger value may reduce wall-clock time on a
machine with enough memory, but does not change the resulting binaries.

Reuse compilation results
~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``build/cmake`` directory is inside the mounted checkout, so its
CMake state persists when the container exits. Enable compiler caching to
reuse unchanged C++ and CUDA compilation results during repeated builds:

.. code-block:: bash

   export NVTE_USE_CCACHE=1

The cache benefits repeated builds in the same container. To reuse it across
disposable container sessions, mount the ccache directory from the host. See
:doc:`build_configuration` for the cache location and build-directory
controls.

Build the selected integration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Set the framework explicitly and install the checkout in editable mode:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: bash

         export NVTE_FRAMEWORK=pytorch
         pip install -e . -v --no-build-isolation

   .. tab:: JAX

      .. code-block:: bash

         export NVTE_FRAMEWORK=jax
         pip install -e . -v --no-build-isolation

Explicit selection makes the result independent of any other framework that
happens to be installed in the image. The editable build compiles the common
library and the selected framework extension from the mounted checkout. See
:doc:`unified_editable_build` for the build stages, outputs, and incremental
rebuild behavior.

Confirm the environment
^^^^^^^^^^^^^^^^^^^^^^^

Before running a larger test, check that the selected framework sees a GPU and
that Transformer Engine resolves to the mounted checkout:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         import torch
         import transformer_engine
         import transformer_engine.pytorch

         print(f"PyTorch: {torch.__version__}")
         print(f"CUDA available: {torch.cuda.is_available()}")
         print(f"Transformer Engine: {transformer_engine.__file__}")

   .. tab:: JAX

      .. code-block:: python

         import jax
         import transformer_engine
         import transformer_engine.jax

         print(f"JAX: {jax.__version__}")
         print(f"JAX devices: {jax.devices()}")
         print(f"Transformer Engine: {transformer_engine.__file__}")

The reported Transformer Engine path should point into
``/workspace/TransformerEngine``. The PyTorch check should report CUDA as
available, and the JAX device list should contain at least one GPU. These
checks catch accidental use of the Transformer Engine version already shipped
in the container as well as failures to expose the host GPUs.

Existing compatible environment
--------------------------------

A container is not required if the host already provides a compatible
framework and native toolchain. This path is useful for persistent development
machines and CI images, but the contributor is responsible for keeping the
components compatible.

The authoritative supported-version information belongs in the `installation
guide <https://nvidia.github.io/TransformerEngine/installation.html>`__. In
addition to a supported GPU and driver, a source build needs:

* a supported Python environment;
* PyTorch, JAX, or both, installed with GPU support;
* a CUDA toolkit that includes NVCC and development headers;
* cuDNN and the other CUDA libraries required by the selected features;
* a C++17-capable host compiler;
* CMake, Ninja, Git, pybind11, setuptools, and wheel.

The framework must be installed before Transformer Engine. The root build
imports the selected framework while configuring its C++ extension, which is
why the installation uses ``--no-build-isolation``.

Verify the native tools from the same shell that will run ``pip``:

.. code-block:: bash

   python --version
   python -m pip --version
   nvcc --version
   c++ --version
   cmake --version
   ninja --version

Also confirm that pybind11 is visible to that Python interpreter:

.. code-block:: python

   import pybind11

   print(pybind11.__version__)
   print(pybind11.get_cmake_dir())

Finally, verify the selected framework and its GPU backend:

.. tabs::

   .. tab:: PyTorch

      .. code-block:: python

         import torch

         print(torch.__version__)
         print(torch.version.cuda)
         print(torch.cuda.is_available())

   .. tab:: JAX

      .. code-block:: python

         import jax

         print(jax.__version__)
         print(jax.devices())

If multiple CUDA installations are present, set ``CUDA_HOME`` to the toolkit
that should provide NVCC and headers. Library search paths must resolve a
compatible cuDNN, NCCL, and CUDA runtime at both build and execution time. See
:doc:`build_configuration` for discovery controls and
:doc:`build_troubleshooting` for mismatch symptoms.

Repository dependencies
-----------------------

Transformer Engine records its source dependencies as Git submodules.
Initialize the revisions recorded by the checkout before the first build:

.. code-block:: bash

   git submodule update --init --recursive

The root ``setup.py`` function ``git_check_submodules`` performs the same
initialization when Git and ``.gitmodules`` are available. It accepts an
uninitialized submodule or the commit recorded by the parent repository. A
submodule checked out at a different commit is treated as a configuration
error so that a stale dependency is not used silently.

``NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1`` bypasses both the revision
check and automatic update. Use it only when intentionally developing against
modified submodules, and record that fact when reporting test results or build
failures.

Optional native dependencies
----------------------------

Most contributors should begin with the default feature set. Enable an
optional component only when the change or test requires it because each
component adds discovery, compatibility, and linking requirements.

.. list-table::
   :header-rows: 1
   :widths: 24 30 46

   * - Component
     - Build selection
     - Additional requirement
   * - NCCL expert parallelism
     - ``NVTE_WITH_NCCL_EP``
     - Enabled by default for targets with compute capability 9.0 or newer.
       Requires NCCL headers and a runtime-compatible NCCL library, and builds
       the expert-parallel code from ``3rdparty/nccl``. Set the variable to
       ``0`` when this capability is not needed.
   * - MPI userbuffers bootstrap
     - ``NVTE_UB_WITH_MPI=1``
     - Requires ``MPI_HOME`` to identify the MPI installation prefix.
   * - NVSHMEM
     - ``NVTE_ENABLE_NVSHMEM=1``
     - Requires ``NVSHMEM_HOME``; the framework binding and common library
       must be built consistently.
   * - cuBLASMp
     - ``NVTE_WITH_CUBLASMP=1``
     - Uses ``CUBLASMP_HOME`` when set, otherwise the matching NVIDIA Python
       distribution must be installed.
   * - cuSolverMp
     - ``NVTE_WITH_CUSOLVERMP=1``
     - Uses ``CUSOLVERMP_HOME``, with ``/usr`` as the build default.

The exact option handling is defined by ``setup.py`` (``setup_common_extension``),
``build_tools/pytorch.py`` (``setup_pytorch_extension``),
``build_tools/jax.py`` (``setup_jax_extension``), and
``transformer_engine/common/CMakeLists.txt``. The configuration summary in
:doc:`build_configuration` explains which settings belong in normal
development and which are reserved for distribution builds.
