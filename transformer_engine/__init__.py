# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# pylint: disable=missing-module-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
"""Top level package"""
from . import common


def generate_error_msh(framework):
    return f"Skip importing transformer_engine.{framework}," \
           f" since that module is not found!" \
           f" Please ignore this if transformer_engine.{framework}" \
           f" is no needed, otherwise make sure transformer_engine is" \
           f" installed via \"FRAMEWORK={framework}\" or \"FRAMEWORK=all\""


try:
    from . import pytorch
except ImportError as e:
    print(generate_error_msh("pytorch"))

try:
    from . import jax
except ImportError as e:
    print(generate_error_msh("jax"))
