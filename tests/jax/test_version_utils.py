# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from types import SimpleNamespace

import pytest

from transformer_engine.jax import version_utils


@pytest.fixture(autouse=True)
def clear_collective_stream_caches():
    """Keep resolver tests independent from JAX imports in other tests."""
    version_utils.get_collective_stream_compute_on.cache_clear()
    version_utils.is_collective_stream_supported.cache_clear()
    yield
    version_utils.get_collective_stream_compute_on.cache_clear()
    version_utils.is_collective_stream_supported.cache_clear()


def test_collective_stream_compute_on_prefers_legacy_region_name(monkeypatch):
    """JAX 0.10.1 exposes the region transform as compute_on2."""

    def context_manager_compute_on():
        return None

    def region_compute_on():
        return None

    module = SimpleNamespace(
        compute_on=context_manager_compute_on,
        compute_on2=region_compute_on,
    )
    monkeypatch.setattr(version_utils, "jax_version_meet_requirement", lambda _: True)
    monkeypatch.setattr(version_utils, "import_module", lambda _: module)

    assert version_utils.get_collective_stream_compute_on() is region_compute_on
    assert version_utils.is_collective_stream_supported()


def test_collective_stream_compute_on_falls_back_to_current_name(monkeypatch):
    """Current JAX exposes only the renamed region transform as compute_on."""

    def region_compute_on():
        return None

    module = SimpleNamespace(compute_on=region_compute_on)
    monkeypatch.setattr(version_utils, "jax_version_meet_requirement", lambda _: True)
    monkeypatch.setattr(version_utils, "import_module", lambda _: module)

    assert version_utils.get_collective_stream_compute_on() is region_compute_on
    assert version_utils.is_collective_stream_supported()


def test_collective_stream_compute_on_requires_supported_jax(monkeypatch):
    monkeypatch.setattr(version_utils, "jax_version_meet_requirement", lambda _: False)
    monkeypatch.setattr(
        version_utils,
        "import_module",
        lambda _: pytest.fail("compute_on should not be imported for unsupported JAX"),
    )

    assert version_utils.get_collective_stream_compute_on() is None
    assert not version_utils.is_collective_stream_supported()


def test_collective_stream_compute_on_handles_missing_module(monkeypatch):
    def raise_import_error(_):
        raise ImportError

    monkeypatch.setattr(version_utils, "jax_version_meet_requirement", lambda _: True)
    monkeypatch.setattr(version_utils, "import_module", raise_import_error)

    assert version_utils.get_collective_stream_compute_on() is None
    assert not version_utils.is_collective_stream_supported()
