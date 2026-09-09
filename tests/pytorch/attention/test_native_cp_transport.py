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


@pytest.fixture
def native_module(monkeypatch):
    """Load the transport without importing GPU-only TE modules."""
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
    return module, extension


@pytest.mark.parametrize("configured_value", [None, "512"])
def test_native_cp_preserves_nccl_environment(monkeypatch, native_module, configured_value):
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
    module, extension = native_module
    parent = Mock()
    parent._get_backend.return_value._comm_ptr.return_value = 123

    module.NativeCPTransport(parent, 256)

    extension.cp_native_transport_create.assert_called_once_with(123, 256)
    assert {name: os.environ.get(name) for name in gin_env_names} == {
        name: configured_value for name in gin_env_names
    }


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool, torch.float32])
@pytest.mark.parametrize("peers", [(7, 11), (None, 11), (7, None), (None, None)])
def test_native_cp_halo_stages_noncontiguous_tensors(native_module, dtype, peers):
    """Halo staging reuses the arena and preserves missing-receive boundary fills."""
    module, extension = native_module
    transport = module.NativeCPTransport.__new__(module.NativeCPTransport)
    transport.handle = 123
    transport.arena = torch.empty(512, dtype=torch.uint8)
    transport._parent_rank = {7: 1, 11: 2}
    source = torch.arange(6).reshape(3, 2).to(dtype).t()
    output = torch.zeros(3, 2, dtype=dtype).t()
    expected = torch.ones_like(output)

    def exchange(handle, send, recv, send_peer, recv_peer, channel):
        assert handle == 123 and channel == 3
        assert send_peer == (-1 if peers[0] is None else 1)
        assert recv_peer == (-1 if peers[1] is None else 2)
        assert send.is_contiguous() and recv.is_contiguous()
        assert send.untyped_storage().data_ptr() == transport.arena.data_ptr()
        assert recv.untyped_storage().data_ptr() == transport.arena.data_ptr()
        if peers[0] is not None:
            torch.testing.assert_close(send, source, rtol=0, atol=0)
        if peers[1] is not None:
            recv.copy_(expected)
        return channel

    extension.cp_native_transport_send_recv.side_effect = exchange
    transport.exchange(source, peers[0], output, peers[1])
    torch.testing.assert_close(
        output, expected if peers[1] is not None else torch.zeros_like(output), rtol=0, atol=0
    )
    if peers == (None, None):
        extension.cp_native_transport_send_recv.assert_not_called()
        extension.cp_native_transport_wait.assert_not_called()
    else:
        extension.cp_native_transport_send_recv.assert_called_once()
        extension.cp_native_transport_wait.assert_called_once_with(123, 3)


def test_native_cp_halo_rejects_incompatible_buffers(native_module):
    """Bad halo metadata must fail before entering the communication kernel."""
    module, extension = native_module
    transport = module.NativeCPTransport.__new__(module.NativeCPTransport)
    transport.handle = 123
    transport.arena = torch.empty(512, dtype=torch.uint8)
    transport._parent_rank = {7: 1}
    with pytest.raises(ValueError, match="matching shapes and dtypes"):
        transport.exchange(torch.zeros(2), 7, torch.zeros(3), 7)
    with pytest.raises(ValueError, match="matching shapes and dtypes"):
        transport.exchange(torch.zeros(2), 7, torch.zeros(2, dtype=torch.int64), 7)
    with pytest.raises(RuntimeError, match="arena needs"):
        transport.exchange(torch.zeros(512), 7, torch.zeros(512), 7)
    transport.exchange(torch.empty(0), 7, torch.empty(0), 7)
    extension.cp_native_transport_send_recv.assert_not_called()
