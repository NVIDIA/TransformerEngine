# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""A broken FlashAttention install must not take down ``transformer_engine.pytorch``.

The distribution metadata can resolve to a supported version while the extension
itself fails to load (missing CUDA dependency, ABI mismatch). That used to
propagate out of ``backends.py`` and make the whole PyTorch interface
unimportable, even for code that never asks for FlashAttention.

Fault injection has to happen before ``transformer_engine`` is imported, so each
case runs in a fresh interpreter with a throwaway directory first on
``PYTHONPATH``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

PROBE = textwrap.dedent(
    """
    import sys

    import transformer_engine.pytorch  # noqa: F401  (the import under test)
    from transformer_engine.pytorch.attention.dot_product_attention import backends
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        FlashAttentionUtils,
    )

    distribution, expect_loaded = sys.argv[1], sys.argv[2] == "loaded"
    if distribution == "flash-attn":
        state = (FlashAttentionUtils.is_installed, backends.flash_attn_func is not None)
    elif distribution == "flash-attn-3":
        state = (FlashAttentionUtils.v3_is_installed, backends.flash_attn_func_v3 is not None)
    elif distribution == "flash-attn-4":
        state = (FlashAttentionUtils.v4_is_installed, backends.flash_attn_func_v4 is not None)
    else:
        raise ValueError(f"Unexpected FlashAttention distribution: {distribution}")
    assert state == (expect_loaded, expect_loaded), state
    print("PROBE_OK")
    """
)

# The FA4 entry points that backends.py binds; never called by the probe.
FAKE_FA4_INTERFACE = textwrap.dedent(
    """
    def _unused(*args, **kwargs):
        raise NotImplementedError

    flash_attn_func = flash_attn_varlen_func = _unused
    _validate_head_dims = _flash_attn_fwd = _flash_attn_bwd = _unused
    """
)


def _fake_dist_info(root: Path, name: str, version: str) -> None:
    """Distribution metadata that ``importlib.metadata.version`` resolves."""
    dist = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )
    (dist / "RECORD").write_text("", encoding="utf-8")


def _fake_module(root: Path, module: str, source: str) -> None:
    """A module, inside regular packages that shadow any installed ones."""
    *packages, name = module.split(".")
    path = root
    for package in packages:
        path = path / package
        path.mkdir(exist_ok=True)
        (path / "__init__.py").touch()
    (path / f"{name}.py").write_text(source, encoding="utf-8")


def _run_probe(root: Path, distribution: str, expect_loaded: bool) -> None:
    env = dict(os.environ)
    inherited = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(root) + (os.pathsep + inherited if inherited else "")
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, distribution, "loaded" if expect_loaded else "unavailable"],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "PROBE_OK" in proc.stdout


@pytest.mark.parametrize(
    "name,version,module",
    [
        ("flash-attn", "2.7.4.post1", "flash_attn_2_cuda"),
        ("flash-attn-3", "3.0.0b1", "flash_attn_interface"),
    ],
    ids=["fa2", "fa3"],
)
def test_broken_flash_attn_does_not_break_te_import(name, version, module):
    with tempfile.TemporaryDirectory(prefix="te-broken-fa-") as tmp:
        root = Path(tmp)
        _fake_dist_info(root, name, version)
        message = "libcudnn.so.9: cannot open shared object file"
        _fake_module(root, module, f"raise ImportError({message!r})\n")
        _run_probe(root, name, expect_loaded=False)


@pytest.mark.parametrize("fa4_version,expect_loaded", [("4.0.0b25", False), ("4.0.0b26", True)])
def test_fa4_minimum_version(fa4_version, expect_loaded):
    """An importable FA4 older than the supported minimum is reported unavailable."""
    with tempfile.TemporaryDirectory(prefix="te-fa4-version-") as tmp:
        root = Path(tmp)
        _fake_dist_info(root, "flash-attn-4", fa4_version)
        _fake_dist_info(root, "nvidia-cutlass-dsl", "4.8.0")
        _fake_module(root, "flash_attn.cute.interface", FAKE_FA4_INTERFACE)
        _run_probe(root, "flash-attn-4", expect_loaded)
