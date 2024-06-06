# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Paddle-paddle related extensions."""
from pathlib import Path

import setuptools

from .utils import cuda_version


def setup_paddle_extension(
    csrc_source_files,
    csrc_header_files,
    common_header_files,
) -> setuptools.Extension:
    """Setup CUDA extension for Paddle support"""

    # Source files
    csrc_source_files = Path(csrc_source_files)
    sources = [
        csrc_source_files / "extensions.cu",
        csrc_source_files / "common.cpp",
        csrc_source_files / "custom_ops.cu",
    ]

    # Header files
    include_dirs = [
        common_header_files,
        common_header_files / "common",
        common_header_files / "common" / "include",
        csrc_header_files,
    ]

    # Compiler flags
    cxx_flags = ["-O3"]
    nvcc_flags = [
        "-O3",
        "-gencode",
        "arch=compute_70,code=sm_70",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ]

    # Version-dependent CUDA options
    try:
        version = cuda_version()
    except FileNotFoundError:
        print("Could not determine CUDA Toolkit version")
    else:
        if version >= (11, 2):
            nvcc_flags.extend(["--threads", "4"])
        if version >= (11, 0):
            nvcc_flags.extend(["-gencode", "arch=compute_80,code=sm_80"])
        if version >= (11, 8):
            nvcc_flags.extend(["-gencode", "arch=compute_90,code=sm_90"])

    # Construct Paddle CUDA extension
    sources = [str(path) for path in sources]
    include_dirs = [str(path) for path in include_dirs]
    from paddle.utils.cpp_extension import CUDAExtension

    ext = CUDAExtension(
        sources=sources,
        include_dirs=include_dirs,
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    )
    ext.name = "transformer_engine_paddle_pd_"
    return ext
