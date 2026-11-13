# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine version string."""
import os
from pathlib import Path
import subprocess

from packaging.version import InvalidVersion, Version


def te_raw_version() -> str:
    """Transformer Engine version string as written in VERSION.txt

    May be a SemVer string such as 1.2.3-rc.1. Use this for Git tags
    and use te_version() for anything consumed by Python packaging.

    """
    root_path = Path(__file__).resolve().parent
    with open(root_path / "VERSION.txt", "r") as f:
        return f.readline().strip()


def append_local_version(version: str, local: str) -> str:
    """Append a segment to the local version label, creating it if needed

    A PEP 440 version has at most one "+", so further segments are
    separated by ".".

    """
    separator = "." if "+" in version else "+"
    return f"{version}{separator}{local}"


def te_version() -> str:
    """Transformer Engine version string, normalized to PEP 440

    SemVer prerelease and build metadata are converted, e.g.
    1.2.3-rc.1 becomes 1.2.3rc1 and 4.5.6-dev.123+abc123 becomes
    4.5.6.dev123+abc123.

    Includes Git commit as local version, unless suppressed with
    NVTE_NO_LOCAL_VERSION environment variable.

    """
    raw_version = te_raw_version()
    try:
        parsed_version = Version(raw_version)
    except InvalidVersion as e:
        raise ValueError(
            f"Transformer Engine version {raw_version!r} has no PEP 440 equivalent. SemVer"
            " prerelease labels must be one of alpha, a, beta, b, c, rc, pre, preview, or dev,"
            " followed by an optional number, e.g. 1.2.3-rc.1."
        ) from e
    if "-" in raw_version.split("+")[0] and parsed_version.is_postrelease:
        # SemVer prereleases sort before the release, PEP 440 post-releases sort after it.
        raise ValueError(
            f"Transformer Engine version {raw_version!r} is a SemVer prerelease but is a"
            f" post-release ({parsed_version}) under PEP 440."
        )
    version = str(parsed_version)

    if not int(os.getenv("NVTE_NO_LOCAL_VERSION", "0")) and not bool(
        int(os.getenv("NVTE_RELEASE_BUILD", "0"))
    ):
        root_path = Path(__file__).resolve().parent
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
            version = str(Version(append_local_version(version, commit)))
    return version
