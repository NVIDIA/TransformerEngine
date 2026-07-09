Unified Editable Build
======================

Purpose and scope
-----------------

The root editable build is the normal development build. One invocation builds
the framework-independent common library, builds each selected framework
binding, and exposes the Python sources directly from the checkout. This keeps
changes to Python files immediately visible while native changes can be
recompiled in place.

This path is distinct from :doc:`split_distribution_builds`, which creates
separate release artifacts for the common, PyTorch, and JAX packages.

Canonical command
-----------------

Run the build from the repository root after preparing the environment and
selecting the desired framework:

.. code-block:: bash

   pip install -e . -v --no-build-isolation

The options are intentional:

* ``-e`` installs the Python package in editable mode, so imports resolve to
  the checkout.
* ``-v`` exposes the CMake and compiler commands needed to diagnose a
  failure.
* ``--no-build-isolation`` makes the build use the PyTorch or JAX
  installation already present in the development environment. An isolated
  build environment would not necessarily contain the selected framework or
  the matching CUDA packages.

A successful command must finish without a compiler or installer error. It is
not sufficient on its own: confirm the import path and execute a compiled
operation as described in :doc:`using_and_validating_builds`.

Select framework integrations
-----------------------------

Set ``NVTE_FRAMEWORK`` explicitly for reproducible development builds:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Value
     - Result
   * - ``pytorch``
     - Build the common library and PyTorch binding.
   * - ``jax``
     - Build the common library and JAX binding.
   * - ``pytorch,jax`` or ``all``
     - Build the common library and both bindings.
   * - ``none``
     - Build only the common library and framework-independent Python package.
   * - Unset
     - Import PyTorch and JAX if available and build every framework that is
       detected.

Auto-detection is convenient for an interactive experiment, but it can change
when packages are added to the environment. Contributor instructions, bug
reproductions, and CI jobs should normally set the value explicitly. The
selection logic is implemented by ``build_tools/utils.py``
(``get_frameworks``).

Build pipeline
--------------

The root ``setup.py`` coordinates these build stages:

1. ``git_check_submodules`` verifies and initializes the repository
   submodules unless the check was intentionally disabled.
2. ``get_frameworks`` resolves the explicit or detected framework list.
3. ``setup_common_extension`` creates a ``CMakeExtension`` for
   ``transformer_engine/common``. The build command in
   ``build_tools/build_ext.py`` configures, builds, and installs
   ``libtransformer_engine``.
4. For PyTorch, ``build_tools/pytorch.py``
   (``setup_pytorch_extension``) creates a
   ``torch.utils.cpp_extension.CppExtension`` named
   ``transformer_engine_torch``.
5. For JAX, ``build_tools/jax.py`` (``setup_jax_extension``) creates a
   pybind11 extension named ``transformer_engine_jax``.

The framework bindings contain C++ integration code but do not own CUDA
kernels. Kernel compilation remains in the common build, consistent with the
layering described in
:doc:`../project_and_architecture/layered_architecture`.

Build outputs
-------------

An editable installation combines files in the checkout with generated native
artifacts:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Output
     - Purpose
   * - Python source under ``transformer_engine/``
     - Imported directly from the checkout through editable-install metadata.
   * - ``build/cmake/``
     - Default CMake configure and object-file tree for the common library.
       ``NVTE_CMAKE_BUILD_DIR`` can place this tree elsewhere.
   * - ``libtransformer_engine`` shared library
     - Framework-independent compiled implementation.
   * - ``transformer_engine_torch`` shared module
     - Private PyTorch C++ binding, present only when PyTorch was selected.
   * - ``transformer_engine_jax`` shared module
     - Private JAX C++ binding, present only when JAX was selected.
   * - Editable distribution metadata
     - Allows ``pip`` and ``importlib.metadata`` to identify the source
       checkout as the installed ``transformer-engine`` distribution.

Shared-object placement varies between editable, regular source, and split
wheel installations. Runtime discovery is centralized in
``transformer_engine/common/__init__.py``
(``_get_shared_object_file``); callers should not hard-code a generated
filename or directory.

Incremental rebuilds
--------------------

Use the smallest rebuild that matches the changed files:

.. list-table::
   :header-rows: 1
   :widths: 31 34 35

   * - Change
     - Rebuild
     - Follow-up
   * - Python-only code
     - None for an existing editable install.
     - Restart the Python process if it already imported the changed module.
   * - Common C++ or CUDA
     - Rerun the editable-install command so CMake builds and installs the
       changed targets.
     - Run a common test and each affected framework test.
   * - PyTorch binding C++
     - Rerun with ``NVTE_FRAMEWORK=pytorch``.
     - Import PyTorch TE and run the affected operation.
   * - JAX binding C++
     - Rerun with ``NVTE_FRAMEWORK=jax``.
     - Import JAX TE and run the affected operation.
   * - Build scripts, compiler flags, framework selection, or optional feature
     - Rerun the complete editable build. Use a fresh build tree if the old
       configuration may be incompatible.
     - Recheck loaded library paths and enabled features.
   * - Git submodule revision or submodule sources
     - Update the submodule and rerun the native build.
     - Run tests covering the dependent component.

CMake reuses ``build/cmake`` by default. ``NVTE_USE_CCACHE=1`` adds
compiler-level reuse, while ``MAX_JOBS`` controls concurrent compilation.
See :doc:`build_configuration` for their precedence and scope.

Clean rebuilds
--------------

CMake and Ninja normally handle incremental changes. However, a clean rebuild is
needed when the generated build state refers to a toolchain or dependency that no
longer exists.

Typical symptoms include:

* Ninja reports that it cannot build a target because an old CUDA library,
  such as a particular ``libcublas`` version, is missing and there is no rule
  to produce it. This commonly happens after the CUDA installation is updated
  from version X to version Y while the old absolute library path remains in
  the build graph.
* CMake continues to use a compiler or CUDA toolkit path that no longer
  matches ``CUDA_HOME`` or ``PATH``.
* CMake reports that the configured generator differs from the requested
  generator.
* An optional library was upgraded, removed, or moved, but Ninja still refers
  to its previous headers or shared libraries.
* Architecture or native feature settings changed and the generated targets
  no longer agree with the current environment.

In the case of build in the default location:

.. code-block:: bash

   rm -rf build
   pip install -e . -v --no-build-isolation

If ``NVTE_CMAKE_BUILD_DIR`` points outside ``build/``, verify its value and
remove that directory instead. Removing the build directory does not clear the
compiler cache, so ccache can still reuse compatible compilation results.
