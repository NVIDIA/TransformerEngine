..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Installation
============

Prerequisites
-------------
.. |driver link| replace:: NVIDIA Driver
.. _driver link: https://www.nvidia.com/drivers

1. Linux x86_64
2. `CUDA 12.1+ (12.8+ for Blackwell support) <https://developer.nvidia.com/cuda-downloads>`__
3. |driver link|_ supporting CUDA 12.1 or later.
4. `cuDNN 9.12 <https://developer.nvidia.com/cudnn>`__ or later.

If the CUDA Toolkit headers are not available at runtime in a standard
installation path, e.g. within `CUDA_HOME`, set
`NVTE_CUDA_INCLUDE_PATH` in the environment.

Transformer Engine in NGC Containers
------------------------------------

Transformer Engine library is preinstalled in the PyTorch container in versions 22.09 and later
on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_.


pip - from PyPI
---------------

Transformer Engine can be directly installed from `our PyPI <https://pypi.org/project/transformer-engine/>`_, e.g.

.. code-block:: bash

    pip3 install --no-build-isolation transformer_engine[pytorch]

To obtain the necessary Python bindings for Transformer Engine, the frameworks needed must be explicitly specified as extra dependencies in a comma-separated list (e.g. [jax,pytorch]). Transformer Engine ships wheels for the core library. Source distributions are shipped for the JAX and PyTorch extensions. To install the framework extensions with build isolation, see :ref:`build_isolation`.

The core package from Transformer Engine (without any framework extensions) can be installed via:

.. code-block:: bash

    pip3 install transformer_engine[core]

By default, this will install the core library compiled for CUDA 12. The cuda major version can be specified by modified the extra dependency to `core_cu12` or `core_cu13`.

pip - from GitHub
-----------------

Additional Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^

1. [For PyTorch support] `PyTorch <https://pytorch.org/>`__ with GPU support.
2. [For JAX support] `JAX <https://github.com/google/jax/>`__ with GPU support, version >= 0.4.7.

Installation (stable release)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to install the latest stable version of Transformer Engine:

.. code-block:: bash

  pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@stable

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable `NVTE_FRAMEWORK` to a comma-separated list (e.g. `NVTE_FRAMEWORK=jax,pytorch`).

Installation (development build)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   While the development build of Transformer Engine could contain new features not available in
   the official build yet, it is not supported and so its usage is not recommended for general
   use.

Execute the following command to install the latest development build of Transformer Engine:

.. code-block:: bash

  pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@main

This will automatically detect if any supported deep learning frameworks are installed and build Transformer Engine support for them. To explicitly specify frameworks, set the environment variable `NVTE_FRAMEWORK` to a comma-separated list (e.g. `NVTE_FRAMEWORK=jax,pytorch`). To only build the framework-agnostic C++ API, set `NVTE_FRAMEWORK=none`.

In order to install a specific PR, execute (after changing NNN to the PR number):

.. code-block:: bash

  pip3 install --no-build-isolation git+https://github.com/NVIDIA/TransformerEngine.git@refs/pull/NNN/merge


Installation (from source)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following commands to install Transformer Engine from source:

.. code-block:: bash

  # Clone repository, checkout stable branch, clone submodules
  git clone --branch stable --recursive https://github.com/NVIDIA/TransformerEngine.git

  cd TransformerEngine
  export NVTE_FRAMEWORK=pytorch         # Optionally set framework
  pip3 install --no-build-isolation .   # Build and install

If the Git repository has already been cloned, make sure to also clone the submodules:

.. code-block:: bash

  git submodule update --init --recursive

Extra dependencies for testing can be installed by setting the "test" option:

.. code-block:: bash

  pip3 install --no-build-isolation .[test]

To build the C++ extensions with debug symbols, e.g. with the `-g` flag:

.. code-block:: bash

  NVTE_BUILD_DEBUG=1 pip3 install --no-build-isolation .

.. _build_isolation:

pip - with build isolation
--------------------------

The commands above disable build isolation so that the framework extensions are compiled against the installed PyTorch or JAX. Transformer Engine can also be installed with build isolation, in which pip builds Transformer Engine in a separate environment and installs the build dependencies there. Since the framework extensions must be built against the same version of PyTorch or JAX that is used at runtime, pin the framework version in the build environment with a build constraints file. Build constraints require pip 25.3 or later.

Execute the following command to create a build constraints file that matches the installed framework:

.. code-block:: bash

  # PyTorch
  python3 -c "import torch; print(f'torch=={torch.__version__}')" > build-constraints.txt

  # JAX
  python3 -c "import jax, jaxlib; print(f'jax=={jax.__version__}'); print(f'jaxlib=={jaxlib.__version__}')" > build-constraints.txt

If the framework was installed from a separate package index, e.g. the PyTorch index for a specific CUDA version, pass the same `--index-url` or `--extra-index-url` options to pip so that the build environment can find the same version. Framework builds that are not published to a package index, such as the ones in the NGC containers, cannot be used with build isolation. Use `--no-build-isolation` for these.

Installation (from PyPI)
^^^^^^^^^^^^^^^^^^^^^^^^

The core library is shipped as wheels and does not need to be built. Execute the following command to install the PyTorch extension with build isolation:

.. code-block:: bash

  pip3 install --build-constraint build-constraints.txt transformer_engine[pytorch]

The JAX extension is installed in the same way:

.. code-block:: bash

  pip3 install --build-constraint build-constraints.txt transformer_engine[jax]

With build isolation, the JAX extension is built against JAX with CUDA 12 support and depends on the core library compiled for CUDA 12 by default. For JAX with CUDA 13 support, set the `cuda-version` config setting:

.. code-block:: bash

  pip3 install --build-constraint build-constraints.txt --config-settings cuda-version=13.0 transformer_engine[jax]

Installation (from GitHub or source)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With build isolation, frameworks installed in the current environment are not detected. To build Transformer Engine support for them, set the `framework` config setting to a comma-separated list (e.g. `framework=jax,pytorch`). If no frameworks are specified, only the framework-agnostic C++ API is built.

.. code-block:: bash

  cd TransformerEngine
  pip3 install --build-constraint build-constraints.txt --config-settings framework=pytorch .

The same options can be used to install from GitHub:

.. code-block:: bash

  pip3 install --build-constraint build-constraints.txt --config-settings framework=pytorch git+https://github.com/NVIDIA/TransformerEngine.git@stable

The CUDA compiler and libraries for the build are installed into the build environment, matching the version of the installed CUDA Toolkit. If no CUDA Toolkit is installed, specify the CUDA version with the `cuda-version` config setting (e.g. `cuda-version=13.0`). To build with an installed CUDA Toolkit instead, set `CUDA_HOME` to its location.

Optional dependencies, such as cuBLASMp and NVSHMEM, are enabled with config settings as well (e.g. `with-cublasmp=1`). In an isolated build, their Python packages are installed into the build environment. Each config setting has an equivalent environment variable, e.g. `NVTE_FRAMEWORK` for `framework` and `NVTE_CUDA_VERSION` for `cuda-version`. See :doc:`envvars` for the complete list.

Other installers support build constraints as well. For example, with `uv`:

.. code-block:: bash

  uv pip install --build-constraints build-constraints.txt --config-setting framework=pytorch .

.. include:: troubleshooting.rst
