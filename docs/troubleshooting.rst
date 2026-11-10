..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Troubleshooting
---------------

Common issues and solutions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **ABI compatibility issues**

   * **Symptoms:** ``ImportError`` with undefined symbols when importing Transformer Engine.
   * **Solution:** Ensure PyTorch and Transformer Engine are built with the same C++ ABI setting.
     Rebuild PyTorch from source with a matching ABI if needed.
   * **Context:** This is particularly common with pip-installed PyTorch outside of containers.

2. **Missing headers or libraries**

   * **Symptoms:** CMake errors about missing headers such as ``cudnn.h``, ``cublas_v2.h``,
     or ``filesystem``.
   * **Solution:** Install the missing development packages or point to the correct locations:

     .. code-block:: bash

         export CUDA_PATH=/path/to/cuda
         export CUDNN_PATH=/path/to/cudnn

   * If CMake cannot find a C++ compiler, set the ``CXX`` environment variable.

3. **Build resource issues**

   * **Symptoms:** Compilation hangs, the system freezes, or the build runs out of memory.
   * **Solution:** Limit parallel builds:

     .. code-block:: bash

         MAX_JOBS=1 NVTE_BUILD_THREADS_PER_JOB=1 pip install ...

4. **Verbose build logging**

   Use verbose output to diagnose a build:

   .. code-block:: bash

       cd transformer_engine
       pip install -v -v -v --no-build-isolation .

UV and virtual environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Import error**

   Ensure the UV environment is active and install with
   ``uv pip install --no-build-isolation <package-or-source-directory>`` instead of installing
   into the system environment.

2. **cuDNN sublibrary loading failure**

   ``CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED`` can occur when Transformer Engine is built against
   the container's system cuDNN while packages inside the virtual environment install a different
   ``nvidia-cudnn-cu12`` or ``nvidia-cudnn-cu13`` version. When building from source, point the
   build and runtime to cuDNN in the virtual environment:

   .. code-block:: bash

       export CUDNN_PATH=$(pwd)/.venv/lib/python3.12/site-packages/nvidia/cudnn
       export CUDNN_HOME=$CUDNN_PATH
       export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$LD_LIBRARY_PATH

3. **Building wheels**

   Use ``uv build --wheel --no-build-isolation -v`` when building the wheel and
   ``uv pip install --no-build-isolation`` when installing it. Verbose output helps verify that
   the build is not pulling in a different PyTorch or JAX version from the active environment.

JAX-specific issues
^^^^^^^^^^^^^^^^^^^

**FFI registration error**

If you see ``No registered implementation for custom call to <some_te_ffi> for platform CUDA``,
ensure ``--no-build-isolation`` is used both when building and installing Transformer Engine.
