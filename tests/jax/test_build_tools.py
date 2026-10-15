# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Dependency metadata for builds that deserialize cuDNN frontend graphs."""

from email.parser import Parser
from io import StringIO

import pytest
from packaging.requirements import Requirement
from setuptools import Distribution

from build_tools import jax as jax_build


@pytest.mark.parametrize("frontend_version", ("1.29.0", "1.30.0"))
def test_jax_metadata_pins_build_frontend(monkeypatch, frontend_version):
    """Built metadata must reject runtime serializers other than the build version."""

    def distribution_version(package):
        assert package == "nvidia-cudnn-frontend"
        return frontend_version

    monkeypatch.setattr(jax_build, "distribution_version", distribution_version)
    dist = Distribution(
        {
            "name": "transformer_engine_jax",
            "version": "0.0.0",
            "install_requires": jax_build.install_requirements(),
        }
    )
    metadata = StringIO()
    dist.metadata.write_pkg_file(metadata)
    requirements = [
        Requirement(value)
        for value in Parser().parsestr(metadata.getvalue()).get_all("Requires-Dist")
    ]
    frontend = next(req for req in requirements if req.name == "nvidia-cudnn-frontend")
    assert frontend.specifier.contains(frontend_version)
    for other_version in ("1.28.0", "1.29.0", "1.29.1", "1.30.0", "1.30.1", "1.31.0"):
        assert frontend.specifier.contains(other_version) == (other_version == frontend_version)


def test_jax_build_requires_minimum_frontend(monkeypatch):
    """Pinning the installed version must preserve the minimum build requirement."""
    monkeypatch.setattr(jax_build, "distribution_version", lambda _package: "1.28.0")
    with pytest.raises(RuntimeError, match="nvidia-cudnn-frontend>=1.29.0"):
        jax_build.install_requirements()
