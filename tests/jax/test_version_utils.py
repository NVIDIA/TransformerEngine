# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from transformer_engine.jax import version_utils


_DEVICE_MEMORY_SPACE = object()


@contextmanager
def _context_manager_compute_on(compute_type):
    """Model JAX's old context-manager API."""
    del compute_type
    yield


def _region_compute_on(f=None, *, compute_type, out_memory_spaces, compiler_options=None):
    """Model JAX's region transform and validate TE's production call contract."""
    assert compute_type == "gpu_stream:collective"
    assert out_memory_spaces is _DEVICE_MEMORY_SPACE
    assert compiler_options is None

    def decorator(func):
        def wrapped():
            return func()

        return wrapped

    return decorator if f is None else decorator(f)


def _exercise_region_compute_on_contract(compute_on):
    wrapped = compute_on(
        compute_type="gpu_stream:collective",
        out_memory_spaces=_DEVICE_MEMORY_SPACE,
    )(lambda: 42)
    assert wrapped() == 42


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

    module = SimpleNamespace(
        compute_on=_context_manager_compute_on,
        compute_on2=_region_compute_on,
    )
    monkeypatch.setattr(version_utils, "jax_version_meet_requirement", lambda _: True)
    monkeypatch.setattr(version_utils, "import_module", lambda _: module)

    compute_on = version_utils.get_collective_stream_compute_on()
    assert compute_on is _region_compute_on
    _exercise_region_compute_on_contract(compute_on)
    assert version_utils.is_collective_stream_supported()


def test_collective_stream_compute_on_falls_back_to_current_name(monkeypatch):
    """Current JAX exposes only the renamed region transform as compute_on."""

    module = SimpleNamespace(compute_on=_region_compute_on)
    monkeypatch.setattr(version_utils, "jax_version_meet_requirement", lambda _: True)
    monkeypatch.setattr(version_utils, "import_module", lambda _: module)

    compute_on = version_utils.get_collective_stream_compute_on()
    assert compute_on is _region_compute_on
    _exercise_region_compute_on_contract(compute_on)
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
