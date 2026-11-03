# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CPU-only checks for native CP initialization, staging and ring collectives."""

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def native_module(monkeypatch):
    """Load the transport without importing GPU-only TE modules."""
    extension = Mock(
        cp_native_transport_create=Mock(return_value=(1, None)),
        cp_native_transport_get_unavailable_reason=Mock(return_value=""),
    )
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


@pytest.mark.parametrize("with_parent", [False, True])
def test_native_cp_capability_query_is_local(monkeypatch, native_module, with_parent):
    """Preflight must never initialize transports or launch communication."""
    module, extension = native_module
    parent = Mock()
    parent._get_backend.return_value._comm_ptr.return_value = 123
    barrier = Mock(side_effect=AssertionError("capability query must not synchronize"))
    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    monkeypatch.setattr(torch.cuda, "current_device", barrier)

    assert (
        module.get_native_cp_transport_unavailable_reason(parent if with_parent else None) is None
    )

    extension.cp_native_transport_get_unavailable_reason.assert_called_once_with(
        123 if with_parent else 0
    )
    extension.cp_native_transport_create.assert_not_called()
    extension.cp_native_transport_send_recv.assert_not_called()
    barrier.assert_not_called()
    assert not module._group_transports
    if not with_parent:
        parent._get_backend.assert_not_called()


@pytest.mark.parametrize(
    "missing_symbol, expected",
    [
        ("cp_native_transport_create", "not built with native CP transport"),
        ("cp_native_transport_get_unavailable_reason", "capability query is unavailable"),
    ],
)
def test_native_cp_capability_query_requires_build_symbols(native_module, missing_symbol, expected):
    module, extension = native_module
    delattr(extension, missing_symbol)
    parent = Mock()

    assert expected in module.get_native_cp_transport_unavailable_reason(parent)

    parent._get_backend.assert_not_called()


@pytest.mark.parametrize(
    "failure",
    [
        "NCCL Device API GIN requires matching compile/runtime versions",
        "The parent NCCL communicator does not support NCCL Device API",
        "The parent communicator spans multiple LSA domains but GIN is unavailable",
    ],
)
def test_native_cp_capability_query_preserves_reason(native_module, failure):
    module, extension = native_module
    extension.cp_native_transport_get_unavailable_reason.return_value = failure

    assert module.get_native_cp_transport_unavailable_reason() == failure

    extension.cp_native_transport_create.assert_not_called()


def test_native_cp_capability_query_requires_comm_ptr(native_module):
    module, extension = native_module
    parent = Mock()
    parent._get_backend.return_value = object()

    assert "does not expose _comm_ptr" in module.get_native_cp_transport_unavailable_reason(parent)

    extension.cp_native_transport_get_unavailable_reason.assert_not_called()


def test_native_cp_capability_query_rejects_uninitialized_parent(native_module):
    module, extension = native_module
    parent = Mock()
    parent._get_backend.return_value._comm_ptr.return_value = 0

    assert "not initialized" in module.get_native_cp_transport_unavailable_reason(parent)

    extension.cp_native_transport_get_unavailable_reason.assert_not_called()


@pytest.mark.parametrize("failure_site", ["backend", "comm_ptr", "properties"])
def test_native_cp_capability_query_reports_runtime_failure(native_module, failure_site):
    module, extension = native_module
    parent = Mock()
    parent._get_backend.return_value._comm_ptr.return_value = 123
    target = {
        "backend": parent._get_backend,
        "comm_ptr": parent._get_backend.return_value._comm_ptr,
        "properties": extension.cp_native_transport_get_unavailable_reason,
    }[failure_site]
    target.side_effect = RuntimeError("test query failure")

    assert (
        module.get_native_cp_transport_unavailable_reason(parent)
        == "Native CP capability query failed: test query failure"
    )

    extension.cp_native_transport_create.assert_not_called()


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


def _collective_transport(module, arena_bytes):
    transport = module.NativeCPTransport.__new__(module.NativeCPTransport)
    transport.handle = 123
    transport.arena = torch.empty(arena_bytes, dtype=torch.uint8)
    transport._parent_rank = {rank: index for index, rank in enumerate((11, 3, 17, 5, 13, 7, 2))}
    return transport


def _logical_group(size, rank):
    # The descriptor's order, not sorted global ranks, owns the rank-major shards.
    return SimpleNamespace(
        ranks=(11, 3, 17, 5, 13, 7, 2)[:size], size=lambda: size, rank=lambda: rank
    )


def _check_collective_ring(monkeypatch, transport, operation, size, rank, dtype, noncontiguous):
    """Simulate foreign shards/partial sums and validate every transmitted chunk."""
    group = _logical_group(size, rank)
    rows = 17 * size if operation == "reduce_scatter" else 17
    inputs = []
    for owner in range(size):
        values = torch.arange(rows * 9, dtype=dtype) + owner * 1000
        inputs.append(values.reshape(9, rows).t() if noncontiguous else values.reshape(rows, 9))
    source = inputs[rank]
    before = source.clone()
    assert source.is_contiguous() == (not noncontiguous)
    channel = {"all_gather": 0, "reduce_scatter": 1, "all_reduce": 2}[operation]
    capacity = (transport.payload_bytes // 512 * 256) // source.element_size()
    chunk_elements = source.numel() // size if operation == "reduce_scatter" else source.numel()
    rank_inputs = [
        value.reshape(size, -1) if operation == "reduce_scatter" else value.reshape(-1)
        for value in inputs
    ]
    works = []

    def partial(owners, shard, start, stop):
        return torch.stack([rank_inputs[owner][shard, start:stop] for owner in owners]).sum(0)

    def exchange(send, dst, recv, src, used_channel):
        call = len(works)
        chunk, phase = divmod(call, size - 1)
        step = phase + 1
        start, stop = chunk * capacity, min((chunk + 1) * capacity, chunk_elements)
        assert dst == group.ranks[(rank + 1) % size]
        assert src == group.ranks[(rank - 1) % size]
        assert used_channel == channel
        for staged in (send, recv):
            assert staged.shape == (stop - start,) and staged.dtype == dtype
            assert staged.is_contiguous()
            assert staged.untyped_storage().data_ptr() == transport.arena.data_ptr()
        assert send.data_ptr() != recv.data_ptr()
        if operation == "reduce_scatter":
            sent_owners = [(rank - offset) % size for offset in range(step)]
            recv_owners = [(rank - offset) % size for offset in range(1, step + 1)]
            expected_send = partial(sent_owners, (rank - step) % size, start, stop)
            incoming = partial(recv_owners, (rank - step - 1) % size, start, stop)
        else:
            expected_send = rank_inputs[(rank - step + 1) % size][start:stop]
            incoming = rank_inputs[(rank - step) % size][start:stop]
        torch.testing.assert_close(send, expected_send, rtol=0, atol=0)
        # Data becomes available only when the caller waits on this particular work.
        work = SimpleNamespace(wait=Mock(side_effect=lambda: recv.copy_(incoming)))
        works.append(work)
        return work

    exchange_mock = Mock(side_effect=exchange)
    monkeypatch.setattr(transport, "send_recv", exchange_mock)
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        Mock(side_effect=AssertionError("logical collectives must use descriptor.ranks")),
    )
    arena_pointer = transport.arena.data_ptr()
    result = getattr(transport, operation)(source, group)
    if operation == "all_gather":
        expected = torch.cat(inputs, dim=0)
    else:
        expected = torch.stack(inputs).sum(0)
        if operation == "reduce_scatter":
            expected = expected.chunk(size, dim=0)[rank]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert result.dtype == dtype
    assert result.untyped_storage().data_ptr() != arena_pointer
    assert transport.arena.data_ptr() == arena_pointer
    expected_calls = 0 if size == 1 else (chunk_elements + capacity - 1) // capacity * (size - 1)
    assert exchange_mock.call_count == expected_calls
    for work in works:
        work.wait.assert_called_once_with()
    if size > 1:
        assert chunk_elements > capacity  # Both fixed arenas must exercise multiple chunks.


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter", "all_reduce"])
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
@pytest.mark.parametrize("arena_bytes", [512, 1024])
@pytest.mark.parametrize("noncontiguous", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.int64])
def test_native_cp_collectives_follow_logical_ring(
    monkeypatch, native_module, operation, size, arena_bytes, noncontiguous, dtype
):
    """Every rank receives foreign data, including odd rings and non-natural rank order."""
    module, extension = native_module
    transport = _collective_transport(module, arena_bytes)
    for rank in range(size):
        _check_collective_ring(monkeypatch, transport, operation, size, rank, dtype, noncontiguous)
    extension.cp_native_transport_create.assert_not_called()


@pytest.mark.parametrize("arena_bytes", [512, 1024])
def test_native_cp_collectives_reuse_one_arena(monkeypatch, native_module, arena_bytes):
    module, extension = native_module
    transport = _collective_transport(module, arena_bytes)
    arena = transport.arena
    for size in (3, 5, 3):
        for operation in ("all_gather", "reduce_scatter", "all_reduce"):
            _check_collective_ring(monkeypatch, transport, operation, size, 1, torch.float32, True)
            assert transport.arena is arena
            assert transport.payload_bytes == arena_bytes
    extension.cp_native_transport_create.assert_not_called()


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter", "all_reduce"])
@pytest.mark.parametrize("size", [1, 2, 3, 5, 7])
def test_native_cp_collectives_accept_empty_tensors(monkeypatch, native_module, operation, size):
    module, _ = native_module
    transport = _collective_transport(module, 512)
    exchange = Mock(side_effect=AssertionError("empty tensors must not communicate"))
    monkeypatch.setattr(transport, "send_recv", exchange)
    source = torch.empty((0, 9))
    result = getattr(transport, operation)(source, _logical_group(size, 0))
    assert result.shape == source.shape and result.dtype == source.dtype
    exchange.assert_not_called()


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter", "all_reduce"])
@pytest.mark.parametrize("arena_bytes", [0, 256])
def test_native_cp_collectives_reject_small_arena(
    monkeypatch, native_module, operation, arena_bytes
):
    module, _ = native_module
    transport = _collective_transport(module, arena_bytes)
    exchange = Mock()
    monkeypatch.setattr(transport, "send_recv", exchange)
    source = torch.zeros((51 if operation == "reduce_scatter" else 17, 9))
    with pytest.raises(RuntimeError, match="arena needs at least 512 bytes"):
        getattr(transport, operation)(source, _logical_group(3, 0))
    exchange.assert_not_called()


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter", "all_reduce"])
def test_native_cp_collectives_reject_wrong_device(monkeypatch, native_module, operation):
    module, _ = native_module
    transport = _collective_transport(module, 512)
    exchange = Mock()
    monkeypatch.setattr(transport, "send_recv", exchange)
    source = torch.empty((51 if operation == "reduce_scatter" else 17, 9), device="meta")
    with pytest.raises(ValueError, match="must be on the arena device"):
        getattr(transport, operation)(source, _logical_group(3, 0))
    exchange.assert_not_called()


@pytest.mark.parametrize("operation", ["all_gather", "reduce_scatter"])
def test_native_cp_collectives_reject_scalar_shards(monkeypatch, native_module, operation):
    module, _ = native_module
    transport = _collective_transport(module, 512)
    exchange = Mock()
    monkeypatch.setattr(transport, "send_recv", exchange)
    with pytest.raises(ValueError, match="dimension|shards"):
        getattr(transport, operation)(torch.tensor(3.0), _logical_group(3, 0))
    exchange.assert_not_called()


@pytest.mark.parametrize("size", [2, 3, 5, 7])
def test_native_cp_reduce_scatter_rejects_unequal_shards(monkeypatch, native_module, size):
    module, _ = native_module
    transport = _collective_transport(module, 512)
    exchange = Mock()
    monkeypatch.setattr(transport, "send_recv", exchange)
    with pytest.raises(ValueError, match="equal first-dimension shards"):
        transport.reduce_scatter(torch.zeros((size + 1, 9)), _logical_group(size, 0))
    exchange.assert_not_called()


@pytest.mark.parametrize("size", [1, 3, 5])
def test_native_cp_all_reduce_accepts_scalar(monkeypatch, native_module, size):
    module, _ = native_module
    transport = _collective_transport(module, 512)
    source = torch.tensor(3.0)
    group = _logical_group(size, 0)
    calls = []

    def exchange(send, dst, recv, src, channel):
        step = len(calls) + 1
        assert dst == group.ranks[1] and src == group.ranks[-1] and channel == 2
        torch.testing.assert_close(send, torch.tensor([float((1 - step) % size + 3)]))
        work = SimpleNamespace(wait=Mock(side_effect=lambda: recv.fill_(float((-step) % size + 3))))
        calls.append(work)
        return work

    monkeypatch.setattr(transport, "send_recv", exchange)
    result = transport.all_reduce(source, group)
    torch.testing.assert_close(result, torch.tensor(float(sum(range(3, size + 3)))))
    torch.testing.assert_close(source, torch.tensor(3.0))
    assert result.shape == () and result.data_ptr() != source.data_ptr()
    assert len(calls) == size - 1
    for work in calls:
        work.wait.assert_called_once_with()
