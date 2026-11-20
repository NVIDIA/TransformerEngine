# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Transformer Engine version string."""
import os
from pathlib import Path
import re
import subprocess

from packaging.version import Version

# The subset of SemVer whose PEP 440 equivalent has the same precedence. PEP 440 orders dev
# releases before alpha and beta releases, so only dev and rc prereleases are supported.
_SEMVER_PATTERN = re.compile(
    r"(?P<release>(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*))"
    r"(?:-(?P<label>dev|rc)\.(?P<number>0|[1-9][0-9]*))?"
    r"(?:\+(?P<build>[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*))?"
)


def te_raw_version() -> str:
    """Transformer Engine version string as written in VERSION.txt

    A SemVer string such as 1.2.3-rc.1. Use this for Git tags and use
    te_version() for anything consumed by Python packaging.

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

    VERSION.txt must contain MAJOR.MINOR.PATCH, optionally followed by
    a -dev.N or -rc.N prerelease and +BUILD metadata. For example,
    1.2.3-rc.1 becomes 1.2.3rc1 and 4.5.6-dev.123+abc123 becomes
    4.5.6.dev123+abc123.

    Includes Git commit as local version, unless suppressed with
    NVTE_NO_LOCAL_VERSION environment variable.

    """
    raw_version = te_raw_version()
    match = _SEMVER_PATTERN.fullmatch(raw_version)
    if match is None:
        raise ValueError(
            f"Transformer Engine version {raw_version!r} is not a supported SemVer version."
            " Expected MAJOR.MINOR.PATCH, optionally followed by -dev.N or -rc.N and +BUILD"
            " metadata, e.g. 1.2.3, 1.2.3-rc.1, or 1.2.3-dev.4+abc123."
        )
    version = match["release"]
    if match["label"] == "rc":
        version += f"rc{match['number']}"
    elif match["label"] == "dev":
        version += f".dev{match['number']}"
    if match["build"] is not None:
        version += f"+{match['build']}"
    version = str(Version(version))

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
