Build and Setup Troubleshooting
===============================

Collect build diagnostics
-------------------------

.. TODO: Define the command output, environment details, tool versions, and
   generated logs needed to investigate a build failure.

Toolchain and configuration failures
------------------------------------

.. TODO: Organize failures involving CUDA discovery, unsupported target
   architectures, CMake, compilers, missing headers, and optional libraries.

Submodule and source-dependency failures
----------------------------------------

.. TODO: Cover uninitialized, stale, or locally modified submodules and the
   difference between correcting them and intentionally bypassing checks.

Resource and incremental-build failures
---------------------------------------

.. TODO: Cover compiler memory pressure, parallel-job limits, stale CMake
   state, compiler caches, and when a clean rebuild is appropriate.

Import, linking, and ABI failures
---------------------------------

.. TODO: Cover imports resolving to the wrong installation, missing shared
   libraries, mismatched CUDA or framework builds, undefined symbols, and
   conflicts between editable and distribution-package layouts.

Distribution-build failures
---------------------------

.. TODO: Cover container setup, platform tags, package metadata, wheel or
   source-distribution dependencies, cached wheel lookup, and wheelhouse logs.
