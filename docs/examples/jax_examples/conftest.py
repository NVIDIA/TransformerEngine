# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Pytest conftest for docs/examples/jax_examples.

Adds ``docs/examples/`` to ``sys.path`` so the example modules can do
``import quickstart_jax_utils`` regardless of the directory pytest was invoked
from.
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_ROOT = os.path.dirname(_HERE)
if _EXAMPLES_ROOT not in sys.path:
    sys.path.insert(0, _EXAMPLES_ROOT)
