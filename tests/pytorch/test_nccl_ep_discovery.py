# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import transformer_engine


def _hide_packaged_library(monkeypatch):
    def _missing_library(_):
        raise FileNotFoundError

    monkeypatch.setattr(
        transformer_engine.common,
        "_get_shared_object_file",
        _missing_library,
    )


def test_nccl_ep_library_found_from_home(monkeypatch, tmp_path):
    home = tmp_path / "nccl_ep"
    library_dir = home / "lib"
    library_dir.mkdir(parents=True)
    (library_dir / "libnccl_ep.so.0.1").touch()

    monkeypatch.setenv("NCCL_EP_HOME", str(home))
    _hide_packaged_library(monkeypatch)
    monkeypatch.setattr(transformer_engine, "find_library", lambda _: None)

    assert transformer_engine._nccl_ep_library_installed()


def test_nccl_ep_library_found_in_package(monkeypatch, tmp_path):
    library = tmp_path / "libnccl_ep.so"
    library.touch()

    monkeypatch.delenv("NCCL_EP_HOME", raising=False)
    monkeypatch.setattr(
        transformer_engine.common,
        "_get_shared_object_file",
        lambda _: library,
    )
    monkeypatch.setattr(transformer_engine, "find_library", lambda _: None)

    assert transformer_engine._nccl_ep_library_installed()


def test_nccl_ep_library_found_by_dynamic_loader(monkeypatch):
    monkeypatch.delenv("NCCL_EP_HOME", raising=False)
    _hide_packaged_library(monkeypatch)
    monkeypatch.setattr(transformer_engine, "find_library", lambda _: "libnccl_ep.so.0")

    assert transformer_engine._nccl_ep_library_installed()


def test_nccl_ep_library_not_found(monkeypatch):
    monkeypatch.delenv("NCCL_EP_HOME", raising=False)
    _hide_packaged_library(monkeypatch)
    monkeypatch.setattr(transformer_engine, "find_library", lambda _: None)

    assert not transformer_engine._nccl_ep_library_installed()
