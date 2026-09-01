Setup, Build, and Run
=====================

This section takes a contributor from a source checkout to a development build
that can be imported, exercised, and rebuilt efficiently. The recommended path
uses a framework-specific NGC container; an existing compatible host
environment is also supported.

Start with these pages in order:

1. :doc:`development_environment` prepares PyTorch or JAX, the native
   toolchain, submodules, and optional dependencies.
2. :doc:`unified_editable_build` explains the root editable build and its
   generated artifacts.
3. :doc:`using_and_validating_builds` confirms that the checkout and compiled
   libraries are the ones actually executing.
4. :doc:`build_troubleshooting` organizes failures by build stage.

Transformer Engine has two build models:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Build model
     - Intended use
     - Entry point
   * - Unified editable build
     - Day-to-day implementation and testing
     - ``pip install -e . -v --no-build-isolation`` from the repository
       root
   * - Split distribution build
     - Release package construction and package-boundary validation
     - Containerized scripts under ``build_tools/wheel_utils/``

Use :doc:`build_configuration` when a change needs controls beyond the
recommended architecture, job-limit, and compiler-cache settings introduced
in the environment page. Use :doc:`split_distribution_builds` only for
packaging work or when validating the installed-package layout.

.. toctree::
   :maxdepth: 1

   development_environment
   unified_editable_build
   build_configuration
   using_and_validating_builds
   build_troubleshooting
   split_distribution_builds
