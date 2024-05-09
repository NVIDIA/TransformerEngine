# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FW agnostic user-end APIs"""
import subprocess
import sys

def get_te_path():
    """Find Transformer Engine install path using pip"""

    command = [sys.executable, "-m", "pip", "show", "transformer_engine"]
    result = subprocess.run(command, capture_output=True, check=True, text=True)
    result = result.stdout.replace("\n", ":").split(":")
    return result[result.index("Location") + 1].strip()
