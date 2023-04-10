..
    Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

Just-in-time kernel compilation
===============================

Transformer Engine uses `NVRTC <https://docs.nvidia.com/cuda/nvrtc>`__
to compile several GPU kernels at runtime. This enables performance
tuning for specific compute configurations, at the expense of adding
compilation overhead. This feature may be disabled by setting
`NVTE_DISABLE_NVRTC=0` in the environment, in which case Transformer
Engine will fall back to statically-compiled kernels without some
performance optimizations.

NVRTC requires access to the
`CUDA Toolkit headers <https://docs.nvidia.com/cuda/>`__
for compilation. The search path may be configured by setting
`NVTE_CUDA_INCLUDE_DIR` in the environment. If not set, Transformer
Engine will search in common installation paths, e.g. within
`CUDA_HOME`.
