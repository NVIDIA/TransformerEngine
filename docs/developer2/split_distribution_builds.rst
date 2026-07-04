Split Distribution Builds
=========================

Purpose and package boundaries
------------------------------

.. TODO: Explain why release packaging separates the metapackage, CUDA-major
   common package, PyTorch package, and JAX package, and show their dependency
   relationships.

Distribution entry points
-------------------------

.. TODO: Document the responsibilities of ``build_tools/wheel_utils/``, the
   architecture-specific launch scripts, Dockerfiles, and
   ``build_wheels.sh``.

Common package
--------------

.. TODO: Outline how the common library is built, repackaged for a CUDA-major
   package name, and made independent of a particular Python minor version.

PyTorch package
---------------

.. TODO: Outline the PyTorch source-distribution and wheel flow, dependency on
   the matching common package, framework and CUDA compatibility tags, cached
   wheel lookup, and source-build fallback.

JAX package
-----------

.. TODO: Outline the JAX source-distribution and extension-build flow and its
   dependency on the matching common package.

Platform and CUDA matrix
------------------------

.. TODO: Document how x86-64 and AArch64 manylinux builds and CUDA-major
   variants are selected. Keep the current matrix in one authoritative place.

Artifacts and logs
------------------

.. TODO: Document wheelhouse contents, build logs, package naming, and how to
   distinguish wheels from source distributions and metapackages.

Validate distribution artifacts
-------------------------------

.. TODO: Define metadata, dependency, installation, import, ABI, and runtime
   checks for each distribution artifact.
