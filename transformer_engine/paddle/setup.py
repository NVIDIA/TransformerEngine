# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Installation script for TE paddle-paddle extensions."""

# pylint: disable=wrong-import-position,wrong-import-order

import sys
import os
import shutil
from pathlib import Path

import setuptools
from paddle.utils.cpp_extension import BuildExtension

try:
    import paddle  # pylint: disable=unused-import
except ImportError as e:
    raise RuntimeError("This package needs Paddle Paddle to build.") from e


current_file_path = Path(__file__).parent.resolve()
build_tools_dir = current_file_path.parent.parent / "build_tools"
if bool(int(os.getenv("NVTE_RELEASE_BUILD", "0"))) or os.path.isdir(build_tools_dir):
    shutil.copytree(build_tools_dir, current_file_path / "build_tools", dirs_exist_ok=True)


from build_tools.build_ext import get_build_ext  # pylint: disable=wrong-import-position
from build_tools.utils import (
    package_files,
    copy_common_headers,
)  # pylint: disable=wrong-import-position
from build_tools.te_version import te_version  # pylint: disable=wrong-import-position
from build_tools.paddle import setup_paddle_extension  # pylint: disable=wrong-import-position


CMakeBuildExtension = get_build_ext(BuildExtension)


if __name__ == "__main__":
    # Extensions
    common_headers_dir = "common_headers"
    copy_common_headers(current_file_path.parent, str(current_file_path / common_headers_dir))
    ext_modules = [
        setup_paddle_extension(
            "csrc", current_file_path / "csrc", current_file_path / common_headers_dir
        )
    ]

    # Configure package
    setuptools.setup(
        name="transformer_engine_paddle",
        version=te_version(),
        packages=["csrc", common_headers_dir, "build_tools"],
        description="Transformer acceleration library - Paddle Paddle Lib",
        ext_modules=ext_modules,
        cmdclass={"build_ext": CMakeBuildExtension},
        install_requires=["paddlepaddle-gpu"],
        tests_require=["numpy"],
        include_package_data=True,
        package_data={
            "csrc": package_files("csrc"),
            common_headers_dir: package_files(common_headers_dir),
            "build_tools": package_files("build_tools"),
        },
    )
    if any(x in sys.argv for x in (".", "sdist", "bdist_wheel")):
        shutil.rmtree(common_headers_dir)
