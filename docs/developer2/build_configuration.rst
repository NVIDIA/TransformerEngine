Build Configuration
===================

Configuration principles
------------------------

.. TODO: Explain which settings are intended for contributors, which are
   internal to release packaging, and which authoritative reference defines
   each environment variable.

Framework selection
-------------------

.. TODO: Cover framework auto-detection and explicit selection without
   duplicating the unified-build procedure.

CUDA toolkit and target architectures
-------------------------------------

.. TODO: Cover toolkit discovery, include paths, CUDA-major selection, and GPU
   architecture targets, including their effect on build time and portability.

Build type, caching, and parallelism
------------------------------------

.. TODO: Cover debug builds, CMake build directories, ccache or sccache, and
   controls for parallel compilation.

Optional components
-------------------

.. TODO: Group options for MPI, NVSHMEM, cuBLASMp, NCCL expert parallelism,
   fast-math kernels, and other optional native features by dependency and
   effect.

Release-only controls
---------------------

.. TODO: Identify release-build, metapackage, package-layout, and wheel-build
   settings that ordinary development builds should not set.
