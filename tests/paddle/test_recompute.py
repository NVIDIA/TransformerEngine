# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Recompute"""

from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

from transformer_engine.paddle.fp8 import is_fp8_available

test_root = Path(__file__).resolve().parent
is_fp8_supported, reason = is_fp8_available()


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
@pytest.mark.parametrize("use_reentrant", [False, True])
def test_transformer_encoder_recompute(use_reentrant):
    """
    Test TransformerLayer encoder recompute
    """
    rtol = 1e-5
    atol = 1e-5

    def launch_subprocess_and_check_output(enable_recompute):
        """Launch training in subprocess and check output"""
        try:
            cmd = [
                "python",
                str(test_root / "recompute_tests" / "recompute_transformer_encoder.py"),
                str(int(enable_recompute)),
                str(int(use_reentrant)),
            ]
            result = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)

            print(result)

            loss_match = re.search(r"Loss:\s+(-?\d+\.\d+)", result)
            memory_match = re.search(r"Peak memory:\s+(\d+)", result)

            loss_value = float(loss_match.group(1))
            memory_value = int(memory_match.group(1))

            return loss_value, memory_value

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Subprocess failed with error: {e}") from e

    loss_recompute, peak_memory_recompute = launch_subprocess_and_check_output(True)
    loss_ref, peak_memory_ref = launch_subprocess_and_check_output(False)

    assert peak_memory_recompute < peak_memory_ref
    np.testing.assert_allclose(loss_recompute, loss_ref, rtol=rtol, atol=atol)
