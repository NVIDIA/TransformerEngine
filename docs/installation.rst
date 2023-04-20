..
    Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Installation
============

Prerequisites
-------------
.. |driver link| replace:: NVIDIA Driver
.. _driver link: https://www.nvidia.com/drivers

1. Linux x86_64
2. `CUDA 11.8 <https://developer.nvidia.com/cuda-downloads>`__
3. |driver link|_ supporting CUDA 11.8 or later.
4. `cuDNN 8 <https://developer.nvidia.com/cudnn>`__ or later.
5. For FP8 fused attention, `CUDA 12.1 <https://developer.nvidia.com/cuda-downloads>`__ or later, |driver link|_ supporting CUDA 12.1 or later, and `cuDNN 8.9 <https://developer.nvidia.com/cudnn>`__ or later.


Transformer Engine in NGC Containers
------------------------------------

Transformer Engine library is preinstalled in the PyTorch container in versions 22.09 and later
on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_.


pip - from GitHub
-----------------------

Additional Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^

1. `CMake <https://cmake.org/>`__ version 3.18 or later: `pip install cmake`.
2. [For pyTorch support] `pyTorch <https://pytorch.org/>`__ with GPU support.
3. [For JAX support] `JAX <https://github.com/google/jax/>`__ with GPU support, version >= 0.4.7.
4. [For TensorFlow support] `TensorFlow <https://www.tensorflow.org/>`__ with GPU support.
5. `pybind11`: `pip install pybind11`.
6. [Optional] `Ninja <https://ninja-build.org/>`__: `pip install ninja`.

Installation (stable release)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to install the latest stable version of Transformer Engine:

.. code-block:: bash

  # Execute one of the following commands
  NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable    # Build TE for PyTorch only. The default.
  NVTE_FRAMEWORK=jax pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable        # Build TE for JAX only.
  NVTE_FRAMEWORK=tensorflow pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable # Build TE for TensorFlow only.
  NVTE_FRAMEWORK=all pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable        # Build TE for all supported frameworks.

Installation (development build)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   While the development build of Transformer Engine could contain new features not available in
   the official build yet, it is not supported and so its usage is not recommended for general
   use.

Execute the following command to install the latest development build of Transformer Engine:

.. code-block:: bash

  # Execute one of the following commands
  NVTE_FRAMEWORK=pytorch pip install git+https://github.com/NVIDIA/TransformerEngine.git@main    # Build TE for PyTorch only. The default.
  NVTE_FRAMEWORK=jax pip install git+https://github.com/NVIDIA/TransformerEngine.git@main        # Build TE for JAX only.
  NVTE_FRAMEWORK=tensorflow pip install git+https://github.com/NVIDIA/TransformerEngine.git@main # Build TE for TensorFlow only.
  NVTE_FRAMEWORK=all pip install git+https://github.com/NVIDIA/TransformerEngine.git@main        # Build TE for all supported frameworks.
