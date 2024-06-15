# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine version string."""
import os
from pathlib import Path
import subprocess


def te_version() -> str:
    """Transformer Engine version string

    Includes Git commit as local version, unless suppressed with
    NVTE_NO_LOCAL_VERSION environment variable.

    """
    root_path = Path(__file__).resolve().parent
    with open(root_path / "VERSION.txt", "r") as f:
        version = f.readline().strip()
    if not int(os.getenv("NVTE_NO_LOCAL_VERSION", "0")) and not bool(
        int(os.getenv("NVTE_RELEASE_BUILD", "0"))
    ):
        try:
            output = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                cwd=root_path,
                check=True,
                universal_newlines=True,
            )
        except (subprocess.CalledProcessError, OSError):
            pass
        else:
            commit = output.stdout.strip()
            version += f"+{commit}"
    return version
