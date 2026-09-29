# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Saved-tensor metadata restoration, including large grouped layers."""

import pytest
import torch

from transformer_engine.pytorch.quantized_tensor import (
    _SavedQuantizedTensor,
    prepare_for_saving,
    restore_from_saved,
    restore_from_func_ctx,
)


@pytest.mark.parametrize("sequence_type", [list, tuple])
@pytest.mark.parametrize("return_tail", [False, True])
@pytest.mark.parametrize("count", [0, 1, 1024])
def test_restore_tensor_identity(sequence_type, return_tail, count):
    value = torch.empty(2, dtype=torch.bfloat16)
    values = [None if i % 4 == 0 else value for i in range(count)]
    saved = sequence_type(values + [value])
    restored = restore_from_saved([None] * count, saved, return_tail)
    if return_tail:
        restored, tail = restored
        assert type(tail) is sequence_type
        assert len(tail) == 1 and tail[0] is value
        if not count:
            assert tail is saved
    assert len(restored) == count
    assert all(a is b for a, b in zip(restored, values))
    assert len(saved) == count + 1


def test_grouped_saved_tensor_roundtrip():
    captured = []
    weight = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
    empty = torch.empty(0)
    values = [None] * 256 + [weight] * 512 + [empty] * 256

    class Capture(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            saved, ctx.tensor_objects = prepare_for_saving(*values)
            ctx.save_for_backward(*saved)
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            captured.extend(restore_from_func_ctx(ctx))
            assert ctx.tensor_objects is None
            return grad

    x = torch.ones(1, requires_grad=True)
    Capture.apply(x).sum().backward()
    assert len(captured) == len(values)
    assert all(a is b for a, b in zip(captured, values))
    assert torch.equal(x.grad, torch.ones_like(x))


class _RestoredStorage:
    _INNER_TENSORS = (("data", "data"), ("scale", "scale"))

    def __init__(self, data=None, scale=None, label=None):
        self.data, self.scale, self.label = data, scale, label


@pytest.mark.parametrize("sequence_type", [list, tuple])
def test_restore_mixed_metadata_and_replacement_tail(sequence_type):
    ordinary, data, scale, legacy_value, replacement, remainder = [torch.empty(1) for _ in range(6)]
    flattened = _SavedQuantizedTensor(
        ("data", "scale"),
        {"cls": _RestoredStorage, "is_tensor": False, "nontensor_kwargs": {"label": "flat"}},
    )

    class LegacyStorage:
        def restore_from_saved(self, saved):
            assert type(saved) is sequence_type
            assert saved[0] is legacy_value
            return sequence_type([replacement, remainder])

    legacy = LegacyStorage()
    saved = sequence_type([ordinary, data, scale, None, legacy_value])
    result, tail = restore_from_saved([None, flattened, None, legacy, ordinary], saved, True)
    assert result[0] is ordinary
    assert result[1].data is data and result[1].scale is scale and result[1].label == "flat"
    assert result[2] is None and result[3] is legacy and result[4] is replacement
    assert type(tail) is sequence_type and len(tail) == 1 and tail[0] is remainder


def test_restore_empty_flattened_and_legacy_tail():
    value = torch.empty(1)
    flattened = _SavedQuantizedTensor(
        (), {"cls": _RestoredStorage, "is_tensor": False, "nontensor_kwargs": {}}
    )
    tail = (value,)

    class LegacyStorage:
        def restore_from_saved(self, saved):
            assert saved is tail
            return saved

    legacy = LegacyStorage()
    restored, remaining = restore_from_saved([flattened, legacy], tail, True)
    assert restored[0].data is None and restored[0].scale is None
    assert restored[1] is legacy and remaining is tail


@pytest.mark.parametrize("with_legacy", [False, True])
def test_restore_empty_flattened_copies_list_tail(with_legacy):
    value = torch.empty(1)
    saved = [value]
    flattened = _SavedQuantizedTensor(
        (), {"cls": _RestoredStorage, "is_tensor": False, "nontensor_kwargs": {}}
    )

    class MutatingStorage:
        def restore_from_saved(self, tail):
            assert tail is not saved
            assert tail.pop() is value
            return tail

    metadata = [flattened, MutatingStorage()] if with_legacy else [flattened]
    _, remaining = restore_from_saved(metadata, saved, True)
    assert saved == [value]
    assert remaining is not saved
    assert remaining == ([] if with_legacy else [value])


@pytest.mark.parametrize("empty_prefix", [False, True])
def test_restore_flattened_constructor_cannot_change_consumed_tail(empty_prefix):
    ordinary, data, after, replacement = [torch.empty(1) for _ in range(4)]
    saved = [data, after] if empty_prefix else [ordinary, data, after]

    class MutatingConstructor(_RestoredStorage):
        def __init__(self, data=None, scale=None, external=None):
            super().__init__(data=data, scale=scale)
            external[-1] = replacement

    prefix = (
        _SavedQuantizedTensor(
            (), {"cls": _RestoredStorage, "is_tensor": False, "nontensor_kwargs": {}}
        )
        if empty_prefix
        else None
    )
    flattened = _SavedQuantizedTensor(
        ("data",),
        {"cls": MutatingConstructor, "is_tensor": False, "nontensor_kwargs": {"external": saved}},
    )
    restored = restore_from_saved([prefix, flattened, None], saved)
    assert restored[1].data is data
    assert restored[-1] is after
    assert saved[-1] is replacement


def test_restore_short_saved_sequence():
    with pytest.raises(IndexError):
        restore_from_saved([None, None], [torch.empty(1)])


def test_restore_legacy_error():
    class BrokenStorage:
        def restore_from_saved(self, saved):
            assert saved == (None,)
            raise RuntimeError("restore failed")

    with pytest.raises(RuntimeError, match="restore failed"):
        restore_from_saved([None, BrokenStorage()], (None, None))


def test_restore_large_group_does_linear_sequence_work():
    copied = 0

    class CountedTuple(tuple):
        def __getitem__(self, key):
            nonlocal copied
            value = super().__getitem__(key)
            if isinstance(key, slice):
                copied += len(value)
                return CountedTuple(value)
            return value

    size = 1024
    restored = restore_from_saved([None] * size, CountedTuple([None] * size))
    assert restored == [None] * size
    assert copied <= size, "Restoring ordinary tensors must not repeatedly copy the remaining tail"
