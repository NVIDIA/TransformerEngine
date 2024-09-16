..
    Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Installation
============

Prerequisites
-------------
.. |driver link| replace:: NVIDIA Driver
.. _driver link: https://www.nvidia.com/drivers

1. Linux x86_64
2. `CUDA 12.0 <https://developer.nvidia.com/cuda-downloads>`__
3. |driver link|_ supporting CUDA 12.0 or later.
4. `cuDNN 8.1 <https://developer.nvidia.com/cudnn>`__ or later.
5. For FP8/FP16/BF16 fused attention, `CUDA 12.1 <https://developer.nvidia.com/cuda-downloads>`__ or later, |driver link|_ supporting CUDA 12.1 or later, and `cuDNN 8.9.1 <https://developer.nvidia.com/cudnn>`__ or later.

If the CUDA Toolkit headers are not available at runtime in a standard
installation path, e.g. within `CUDA_HOME`, set
`NVTE_CUDA_INCLUDE_PATH` in the environment.

Transformer Engine in NGC Containers
------------------------------------

Transformer Engine library is preinstalled in the PyTorch container in versions 22.09 and later
on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_.


pip - from PyPI
-----------------------

Transformer Engine can be directly installed from `our PyPI <https://pypi.org/project/transformer-engine/>`_, e.g.

.. code-block:: bash

    pip install transformer_engine[pytorch]

To obtain the necessary Python bindings for Transformer Engine, the frameworks needed must be explicitly specified as extra dependencies in a comma-separated list (e.g. [jax,pytorch,paddle]). Transformer Engine ships wheels for the core library as well as the PaddlePaddle extensions. Source distributions are shipped for the JAX and PyTorch extensions.

pip - from GitHub
-----------------------

Additional Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^

1. [For PyTorch support] `PyTorch <https://pytorch.org/>`__ with GPU support.
2. [For JAX support] `JAX <https://github.com/google/jax/>`__ with GPU support, version >= 0.4.7.

Installation (stable release)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to install the latest stable version of Transformer Engine:

.. code-block:: bash

  pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable `NVTE_FRAMEWORK` to a comma-separated list (e.g. `NVTE_FRAMEWORK=jax,pytorch`).

Installation (development build)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   While the development build of Transformer Engine could contain new features not available in
   the official build yet, it is not supported and so its usage is not recommended for general
   use.

Execute the following command to install the latest development build of Transformer Engine:

.. code-block:: bash

  pip install git+https://github.com/NVIDIA/TransformerEngine.git@main

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable `NVTE_FRAMEWORK` to a comma-separated list (e.g. `NVTE_FRAMEWORK=jax,pytorch`). To only build the framework-agnostic C++ API, set `NVTE_FRAMEWORK=none`.

In order to install a specific PR, execute after changing NNN to the PR number:

.. code-block:: bash

  pip install git+https://github.com/NVIDIA/TransformerEngine.git@refs/pull/NNN/merge


Installation (from source)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following commands to install Transformer Engine from source:

.. code-block:: bash

  # Clone repository, checkout stable branch, clone submodules
  git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git

  cd TransformerEngine
  export NVTE_FRAMEWORK=pytorch   # Optionally set framework
  pip install .                   # Build and install

If the Git repository has already been cloned, make sure to also clone the submodules:

.. code-block:: bash

  git submodule update --init --recursive

Extra dependencies for testing can be installed by setting the "test" option:

.. code-block:: bash

  pip install .[test]

To build the C++ extensions with debug symbols, e.g. with the `-g` flag:

.. code-block:: bash

  pip install . --global-option=--debug
