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
    import transformer_engine.pytorch  # noqa: F401  (the import under test)
    from transformer_engine.pytorch.attention.dot_product_attention import backends
    from transformer_engine.pytorch.attention.dot_product_attention.utils import (
        FlashAttentionUtils,
    )

    assert FlashAttentionUtils.is_installed is False
    assert backends.flash_attn_func is None
    if not FlashAttentionUtils.v3_is_installed:
        assert backends.flash_attn_func_v3 is None
    print("PROBE_OK")
    """
)


def _fake_broken_dist(root: Path, name: str, version: str, module: str, message: str) -> None:
    """A distribution whose metadata resolves but whose module raises ImportError."""
    dist = root / f"{name.replace('-', '_')}-{version}.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    (dist / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
    )
    (dist / "RECORD").write_text("", encoding="utf-8")
    path = root / (module.replace(".", "/") + ".py")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"raise ImportError({message!r})\n", encoding="utf-8")


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
        _fake_broken_dist(
            Path(tmp), name, version, module, "libcudnn.so.9: cannot open shared object file"
        )
        env = dict(os.environ)
        inherited = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = tmp + (os.pathsep + inherited if inherited else "")
        proc = subprocess.run(
            [sys.executable, "-c", PROBE], capture_output=True, text=True, env=env, timeout=900
        )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "PROBE_OK" in proc.stdout
