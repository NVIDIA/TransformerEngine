# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CPU-only checks for native CP initialization policy."""

import importlib.util
import os
from pathlib import Path
import sys
from unittest.mock import Mock

import pytest
import torch


@pytest.mark.parametrize("configured_value", [None, "512"])
def test_native_cp_preserves_nccl_environment(monkeypatch, configured_value):
    """Creating a transport must not limit GIN resources for other communicators."""
    gin_env_names = (
        "NCCL_GIN_NCONTEXTS",
        "NCCL_GIN_SIGNAL_POOL_SIZE",
        "NCCL_GIN_COUNTER_POOL_SIZE",
    )
    for name in gin_env_names:
        if configured_value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, configured_value)

    extension = Mock(cp_native_transport_create=Mock(return_value=(1, None)))
    monkeypatch.setitem(sys.modules, "transformer_engine_torch", extension)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0])
    path = (
        Path(__file__).resolve().parents[3]
        / "transformer_engine/pytorch/attention/native_cp_transport.py"
    )
    spec = importlib.util.spec_from_file_location("native_cp_transport_policy_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parent = Mock()
    parent._get_backend.return_value._comm_ptr.return_value = 123

    module.NativeCPTransport(parent, 256)

    extension.cp_native_transport_create.assert_called_once_with(123, 256)
    assert {name: os.environ.get(name) for name in gin_env_names} == {
        name: configured_value for name in gin_env_names
    }
