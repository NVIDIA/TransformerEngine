# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import subprocess
import sys
from pathlib import Path

import pytest
from packaging.utils import parse_wheel_filename
from packaging.version import Version

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from build_tools import te_version as te_version_module
from build_tools.te_version import append_local_version, te_version


@pytest.fixture
def raw_version(monkeypatch):
    """Set the contents of VERSION.txt."""

    def set_raw_version(raw):
        monkeypatch.setattr(te_version_module, "te_raw_version", lambda: raw)

    monkeypatch.setenv("NVTE_RELEASE_BUILD", "1")
    return set_raw_version


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("2.22.0", "2.22.0"),
        ("2.22.0.dev0", "2.22.0.dev0"),
        ("1.2.3-rc.1", "1.2.3rc1"),
        ("1.2.3-rc1", "1.2.3rc1"),
        ("1.2.3-alpha.2", "1.2.3a2"),
        ("1.2.3-beta", "1.2.3b0"),
        ("4.5.6-dev.123", "4.5.6.dev123"),
        ("4.5.6-dev.123+abc123", "4.5.6.dev123+abc123"),
    ],
)
def test_normalizes_semver(raw_version, raw, expected):
    raw_version(raw)
    assert te_version() == expected


@pytest.mark.parametrize(
    "raw,error",
    [
        ("", "has no PEP 440 equivalent"),
        ("abc", "has no PEP 440 equivalent"),
        ("1.2.3-nightly.5", "has no PEP 440 equivalent"),
        ("1.2.3-rc.1.2", "has no PEP 440 equivalent"),
        ("1.2.3-1", "is a post-release"),
        ("1.2.3-post.1", "is a post-release"),
        ("v1.2.3", "must not start with 'v'"),
        ("V1.2.3-rc.1", "must not start with 'v'"),
    ],
)
def test_rejects_invalid_versions(raw_version, raw, error):
    raw_version(raw)
    with pytest.raises(ValueError, match=error):
        te_version()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1.2.3-rc.1", "1.2.3rc1+abc1234"),
        ("4.5.6-dev.123+abc123", "4.5.6.dev123+abc123.abc1234"),
    ],
)
def test_appends_commit(raw_version, monkeypatch, raw, expected):
    raw_version(raw)
    monkeypatch.setenv("NVTE_RELEASE_BUILD", "0")
    monkeypatch.delenv("NVTE_NO_LOCAL_VERSION", raising=False)
    monkeypatch.setattr(
        te_version_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="abc1234\n"),
    )
    assert te_version() == expected


@pytest.mark.parametrize("raw", ["2.22.0", "1.2.3-rc.1", "4.5.6-dev.123+abc123"])
def test_framework_wheel_filename_is_valid(raw_version, raw):
    raw_version(raw)
    wheel_version = append_local_version(te_version(), "cu12torch2.9.1cxx11abiTRUE")
    name, version, _, _ = parse_wheel_filename(
        f"transformer_engine_torch-{wheel_version}-cp312-cp312-linux_x86_64.whl"
    )
    assert name == "transformer-engine-torch"
    assert version == Version(wheel_version)
    assert version.public == te_version().split("+")[0]
