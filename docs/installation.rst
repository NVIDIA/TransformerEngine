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


Transformer Engine in NGC Containers
------------------------------------

Transformer Engine library is preinstalled in the PyTorch container in versions 22.09 and later
on `NVIDIA GPU Cloud <https://ngc.nvidia.com>`_.


pip - from GitHub
-----------------------

Additional Prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^

1. `CMake <https://cmake.org/>`__ version 3.18 or later
2. `pyTorch <https://pytorch.org/>`__ with GPU support
3. `Ninja <https://ninja-build.org/>`__

Installation (stable release)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the following command to install the latest stable version of Transformer Engine:

.. code-block:: bash

   pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@stable

Installation (development build)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::

   While the development build of Transformer Engine could contain new features not available in
   the official build yet, it is not supported and so its usage is not recommended for general
   use.

Execute the following command to install the latest development build of Transformer Engine:

.. code-block:: bash

   pip install --upgrade git+https://github.com/NVIDIA/TransformerEngine.git@main

