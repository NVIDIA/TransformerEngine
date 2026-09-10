Split Distribution Builds
=========================

Purpose and package boundaries
------------------------------

Transformer Engine's release installation is split so that the large common
library is selected by CUDA major version while each framework binding can be
built against the framework installed by the user. This is a packaging
workflow, not the recommended contributor build.

The distributions have distinct responsibilities:

.. list-table::
   :header-rows: 1
   :widths: 27 28 45

   * - Distribution
     - Artifact role
     - Relationship
   * - ``transformer-engine``
     - Dependency-selecting metapackage
     - Provides extras such as ``core``, ``core_cu12``,
       ``core_cu13``, ``pytorch``, and ``jax`` that select the
       concrete packages.
   * - ``transformer-engine-cu12`` or
       ``transformer-engine-cu13``
     - CUDA-major common package
     - Contains the Transformer Engine Python package and common shared
       library for one CUDA major.
   * - ``transformer-engine-torch``
     - PyTorch binding package
     - Depends on the exactly matching common package selected from the CUDA
       major reported by PyTorch.
   * - ``transformer-engine-jax``
     - JAX binding package
     - Depends on the exactly matching common package selected from the JAX
       CUDA backend.

At import time, ``transformer_engine/common/__init__.py``
(``sanity_checks_for_pypi_installation`` and
``load_framework_extension``) checks that the metapackage, common package,
and installed framework package have matching Transformer Engine versions.
The private bindings may vary with framework and ABI even though the public
Python package is shared.

Distribution entry points
-------------------------

Release packaging is coordinated by ``build_tools/wheel_utils/``:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Entry point
     - Responsibility
   * - ``Dockerfile.x86``
     - Creates a ``manylinux_2_28_x86_64`` environment with the requested
       CUDA toolkit, cuDNN, NCCL, and build tools.
   * - ``Dockerfile.aarch``
     - Creates the corresponding ``manylinux_2_28_aarch64`` environment.
   * - ``launch_x86.sh``
     - Builds and runs the x86 container for the CUDA and artifact
       combinations currently produced on x86.
   * - ``launch_aarch.sh``
     - Builds and runs the AArch64 container for the combinations currently
       produced on AArch64.
   * - ``build_wheels.sh``
     - Builds the requested metapackage, common wheel, and framework source
       distributions inside the manylinux container, and writes artifacts and
       logs to ``/wheelhouse``.

The launch scripts are the authoritative description of the current platform,
CUDA-major, and artifact matrix. Do not duplicate that matrix in prose; review
the build arguments passed by the relevant launcher when changing release
coverage.

.. warning::

   The launch scripts remove their existing named wheelhouse directories and
   perform clean container builds. Run them from an appropriate release
   checkout after preserving any output needed from an earlier run. They are
   substantially more expensive than the root editable build.

Common package
--------------

With ``NVTE_RELEASE_BUILD=1``, the root ``setup.py`` builds the common
CMake extension but omits the PyTorch and JAX extensions. The wheel script then:

1. builds a platform wheel from the root package;
2. unpacks the wheel;
3. changes the distribution name to
   ``transformer-engine-cu<CUDA_MAJOR>``;
4. updates the distribution metadata and wheel tag;
5. repacks it with a ``py3-none-<platform>`` tag.

The common native library does not use the Python C API, so the finished
artifact is independent of a particular Python minor version while remaining
platform-specific. The CUDA major is part of the distribution name because
the native dependencies are not interchangeable across CUDA majors.

The empty metapackage is built separately with both
``NVTE_RELEASE_BUILD=1`` and ``NVTE_BUILD_METAPACKAGE=1``. Its extras
select common and framework packages at the exact same TE version.

PyTorch package
---------------

``build_wheels.sh`` creates a source distribution from
``transformer_engine/pytorch/setup.py``. That source distribution contains
the binding sources, copied common headers, build helpers, and package
metadata, but not the common library itself.

When a wheel is requested from the PyTorch package, its
``CachedWheelsCommand`` first constructs a release URL from:

* Transformer Engine version;
* CUDA major reported by PyTorch;
* PyTorch version;
* Python ABI;
* operating-system platform;
* PyTorch's C++11 ABI setting.

If that exact cached wheel exists, the command uses it. Otherwise it compiles
``transformer_engine_torch`` from source against the installed PyTorch and
depends on the exactly matching CUDA-major common package.
``NVTE_PYTORCH_FORCE_BUILD=TRUE`` bypasses the cached-wheel attempt when
testing source compilation.

JAX package
-----------

``build_wheels.sh`` creates a source distribution from
``transformer_engine/jax/setup.py``. During installation, the setup script:

1. imports the installed JAX package and obtains its XLA FFI headers;
2. determines the CUDA major from the JAX GPU backend;
3. copies the common C and C++ headers required by the binding;
4. compiles ``transformer_engine_jax`` with pybind11;
5. depends on the exactly matching CUDA-major common package.

A GPU-enabled JAX environment is therefore required when building the JAX
extension package. Unlike the PyTorch setup path, the current JAX setup does
not implement a cached release-wheel lookup.

Platform and CUDA matrix
------------------------

The release matrix has three independent dimensions:

* manylinux platform: x86-64 or AArch64;
* CUDA major and the toolkit minor used to produce that artifact;
* artifact selection: metapackage, common package, PyTorch, and JAX.

The Dockerfiles define how one platform/toolkit environment is constructed.
The ``BUILD_METAPACKAGE``, ``BUILD_COMMON``, ``BUILD_PYTORCH``,
``BUILD_JAX``, and ``CUDA_MAJOR`` build arguments select its outputs.
The launch scripts provide the checked-in combinations. A matrix change should
update those scripts and their CI coverage together rather than only changing
this page.

Artifacts and logs
------------------

Each container writes products under ``/wheelhouse``. The launcher copies
that directory to a platform- and CUDA-specific directory on the host.

Expected contents include:

* a ``transformer_engine`` metapackage wheel when requested;
* a ``transformer_engine_cu<major>`` common wheel;
* a ``transformer_engine_torch`` source distribution when requested;
* a ``transformer_engine_jax`` source distribution when requested;
* ``logs/metapackage.txt``, ``common.txt``, ``torch.txt``, and
  ``jax.txt`` for the corresponding requested builds.

Artifact presence alone is not success. Inspect the log for the command that
created it, and confirm the filename, internal metadata, version, CUDA major,
Python tag, and platform tag agree.

Validate distribution artifacts
-------------------------------

Validate release products outside the source checkout so editable imports
cannot hide a packaging error:

1. create a clean environment with the intended framework and GPU backend;
2. install artifacts only from the candidate wheelhouse or source
   distributions;
3. run ``python -m pip check``;
4. inspect installed distribution names and versions;
5. import the common package and selected framework integration;
6. print the loaded common and framework shared-library paths;
7. execute a small forward and backward operation on a GPU;
8. repeat for every supported framework, CUDA major, Python, platform, and ABI
   combination represented by the artifact.

The executable repository references are
``qa/L0_pytorch_wheel/test.sh`` and
``qa/L0_jax_wheel/test.sh``. They assemble the common and metapackage
artifacts, build the framework package, install the candidates without
dependency substitution, and run
``tests/pytorch/test_sanity_import.py`` or
``tests/jax/test_sanity_import.py``. Release validation should add runtime
coverage beyond those import checks for the combinations being published.
