# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""End-to-end numerical tests for contiguous GDN context parallelism."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import torch

from test_gdn_attention import _GDN_AVAILABLE
import transformer_engine


def test_gdn_with_cp():
    """Run genuine NCCL CP with all token and state gradients checked numerically."""
    available = torch.cuda.device_count()
    # Test only the largest supported CP size available, rather than all sizes, to reduce test time.
    size = next((size for size in (8, 4, 2) if available >= size), None)
    if size is None:
        pytest.skip(f"GDN CP test requires at least 2 GPUs, but found {available}")
    if not _GDN_AVAILABLE:
        if os.getenv("NVTE_GDN_CP_TEST_REQUIRED", "0") == "1":
            pytest.fail("Required GDN CP test needs the cuDNN frontend GDN runtime")
        pytest.skip("GDN CP test requires the cuDNN frontend GDN runtime")
    runner = Path(__file__).with_name("run_gdn_with_cp.py")
    env = os.environ.copy()
    # torchrun starts a script, so its sys.path[0] is the tests directory.
    # Preserve the TE package that pytest imported, including source worktrees.
    package_root = str(Path(transformer_engine.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join((package_root, env.get("PYTHONPATH", "")))
    with tempfile.TemporaryDirectory(prefix="te-gdn-cp-") as log_dir:
        log_path = Path(log_dir) / "worker.log"
        print(f"CP={size} progress log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    f"--nproc_per_node={size}",
                    str(runner),
                ],
                check=False,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1200,
                env=env,
            )
        output = log_path.read_text()
        assert result.returncode == 0, output
        print(output)
