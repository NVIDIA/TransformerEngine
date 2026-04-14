import os
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from setuptools import setup

project_root = os.path.dirname(os.path.abspath(__file__))

air_topk_ext = CUDAExtension(
    name="air_topk_wrapper",
    sources=["kernels/air_topk_wrapper.cu"],
    include_dirs=[
        os.path.join(project_root, "external/simple_air_topk/include"),
    ],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math", "-arch=sm_100"],
    },
)

topk_per_row_ext = CUDAExtension(
    name="topk_per_row",
    sources=["kernels/topk_per_row.cu"],
    extra_compile_args={
        "cxx": ["-O3"],
        "nvcc": ["-O3", "--use_fast_math", "-arch=sm_100"],
    },
)

setup(
    name="topk-kernels",
    ext_modules=[air_topk_ext, topk_per_row_ext],
    cmdclass={"build_ext": BuildExtension},
)
