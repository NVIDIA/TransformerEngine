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
        ("3.0.0", "3.0.0"),
        ("10.20.30", "10.20.30"),
        ("3.0.0-dev.0", "3.0.0.dev0"),
        ("1.2.3-alpha.1", "1.2.3a1"),
        ("1.2.3-beta.2", "1.2.3b2"),
        ("1.2.3-rc.1", "1.2.3rc1"),
        ("4.5.6-dev.123", "4.5.6.dev123"),
        ("4.5.6-dev.123+gabc1234", "4.5.6.dev123+gabc1234"),
        ("1.2.3+g0123456", "1.2.3+g0123456"),
        ("1.2.3+g" + "0" * 64, "1.2.3+g" + "0" * 64),
    ],
)
def test_normalizes_semver(raw_version, raw, expected):
    raw_version(raw)
    assert te_version() == expected
    assert str(Version(expected)) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "abc",
        "1.2",
        "1.2.3.4",
        "01.2.3",
        "v1.2.3",
        " 1.2.3",
        # PEP 440 rather than SemVer
        "3.0.0.dev0",
        "1.2.3rc1",
        "1.2.3.post1",
        # Unsupported prereleases
        "1.2.3-a.1",
        "1.2.3-alpha",
        "1.2.3-nightly.5",
        "1.2.3-1",
        "1.2.3-rc",
        "1.2.3-rc1",
        "1.2.3-RC.1",
        "1.2.3-rc.01",
        "1.2.3-rc.1.2",
        "1.2.3-rc.1.dev.2",
        # Invalid build metadata
        "1.2.3+",
        "1.2.3+g",
        "1.2.3+abc1234",
        "1.2.3+gabc123",
        "1.2.3+g" + "0" * 65,
        "1.2.3+gABC1234",
        "1.2.3+gabc123z",
        "1.2.3+gabc1234.abc1234",
        "1.2.3+build.5",
    ],
)
def test_rejects_unsupported_versions(raw_version, raw):
    raw_version(raw)
    with pytest.raises(ValueError, match="is not a supported SemVer version"):
        te_version()


def test_pep440_precedence(raw_version):
    """Package installers order prereleases by PEP 440, with dev releases first."""
    expected_order = [
        "3.0.0-dev.0",
        "3.0.0-dev.10",
        "3.0.0-alpha.1",
        "3.0.0-beta.1",
        "3.0.0-rc.1",
        "3.0.0-rc.2",
        "3.0.0",
        "3.0.1-dev.0",
        "3.1.0",
    ]
    pep440_versions = []
    for raw in expected_order:
        raw_version(raw)
        pep440_versions.append(Version(te_version()))
    assert pep440_versions == sorted(pep440_versions)
    assert len(set(pep440_versions)) == len(pep440_versions)


@pytest.mark.parametrize(
    "raw,commit,expected",
    [
        ("1.2.3-rc.1", "abc1234", "1.2.3rc1+gabc1234"),
        ("1.2.3", "0123456", "1.2.3+g0123456"),
        ("4.5.6-dev.123+gdef5678", "abc1234", "4.5.6.dev123+gdef5678.gabc1234"),
    ],
)
def test_appends_commit(raw_version, monkeypatch, raw, commit, expected):
    raw_version(raw)
    monkeypatch.setenv("NVTE_RELEASE_BUILD", "0")
    monkeypatch.delenv("NVTE_NO_LOCAL_VERSION", raising=False)
    monkeypatch.setattr(
        te_version_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=f"{commit}\n"),
    )
    assert te_version() == expected
    assert str(Version(expected)) == expected


@pytest.mark.parametrize("raw", ["3.0.0", "3.0.0-dev.0", "1.2.3-rc.1", "4.5.6-dev.123+gabc1234"])
def test_framework_wheel_filename_is_valid(raw_version, raw):
    raw_version(raw)
    wheel_version = append_local_version(te_version(), "cu12torch2.9.1cxx11abiTRUE")
    name, version, _, _ = parse_wheel_filename(
        f"transformer_engine_torch-{wheel_version}-cp312-cp312-linux_x86_64.whl"
    )
    assert name == "transformer-engine-torch"
    assert version == Version(wheel_version)
    assert version.public == te_version().split("+")[0]
