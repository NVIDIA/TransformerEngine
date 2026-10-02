# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""CPU contracts for saved tensor reconstruction, including its linear copy bound."""

import pytest
import torch

from transformer_engine.pytorch.quantized_tensor import (
    _SavedQuantizedTensor,
    restore_from_func_ctx,
    restore_from_saved,
)


class CountedTuple(tuple):
    """Count copied references without a timing threshold."""

    copied = 0

    def __getitem__(self, index):
        value = super().__getitem__(index)
        if isinstance(index, slice):
            type(self).copied += len(value)
            return type(self)(value)
        return value


@pytest.mark.parametrize("count", [0, 1, 1024])
@pytest.mark.parametrize("container", [list, tuple, CountedTuple])
@pytest.mark.parametrize("return_tail", [False, True])
def test_restore_plain_identity_and_linear_copying(count, container, return_tail):
    value, tail = torch.empty(0), torch.empty(0)
    saved = container([value] * count + [tail])
    objects = [None if i % 2 else value for i in range(count)]
    CountedTuple.copied = 0
    result = restore_from_saved(objects, saved, return_saved_tensors=return_tail)
    if return_tail:
        result, remaining = result
        assert type(remaining) is container
        assert len(remaining) == 1 and remaining[0] is tail
        if count == 0:
            assert remaining is saved
    assert len(result) == count and all(item is value for item in result)
    assert len(saved) == count + 1 and saved[-1] is tail
    assert CountedTuple.copied <= len(saved)


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("compiled_count", [0, 2])
def test_restore_mixed_storage_and_context(container, compiled_count):
    class FlatStorage:
        _INNER_TENSORS = (("left", "left"), ("right", "right"))

        def __init__(self, left=None, right=None):
            self.left, self.right = left, right

    class EagerStorage:
        def restore_from_saved(self, saved):
            assert type(saved) is container
            self.values = saved[:2]
            return saved[2:]

    class Context:
        pass

    values = [torch.empty(0) for _ in range(7)]
    metadata = {"cls": FlatStorage, "is_tensor": False, "nontensor_kwargs": {}}
    flattened = _SavedQuantizedTensor(("left", "right")[:compiled_count], metadata)
    eager = EagerStorage()
    ctx = Context()
    ctx.tensor_objects = [None, flattened, eager, values[0], None]
    ctx.saved_tensors = container(values[: 1 + compiled_count + 2] + [values[5], None, values[6]])
    result, remaining = restore_from_func_ctx(ctx, return_saved_tensors=True)
    assert ctx.tensor_objects is None
    assert result[0] is values[0] and result[2] is eager
    assert result[1].left is (values[1] if compiled_count else None)
    assert result[1].right is (values[2] if compiled_count else None)
    assert eager.values[0] is values[1 + compiled_count]
    assert eager.values[1] is values[2 + compiled_count]
    assert result[3] is values[5] and result[4] is None
    assert type(remaining) is container and len(remaining) == 1
    assert remaining[0] is values[6]


def test_restore_missing_plain_tensor_still_fails():
    with pytest.raises(IndexError):
        restore_from_saved([None], [])


@pytest.mark.parametrize("container", [list, tuple])
def test_restore_consecutive_eager_storage_preserves_pass_through_identity(container):
    first_input = container([torch.empty(0), torch.empty(0)])
    second_input = container([torch.empty(0)])

    class FirstStorage:
        def restore_from_saved(self, saved):
            assert saved is first_input
            return second_input

    class SecondStorage:
        def restore_from_saved(self, saved):
            assert saved is second_input
            return saved

    first, second = FirstStorage(), SecondStorage()
    result, remaining = restore_from_saved([first, second], first_input, return_saved_tensors=True)
    assert result[0] is first and result[1] is second
    assert remaining is second_input


def test_restore_compiled_storage_linear_copying():
    class Storage:
        _INNER_TENSORS = (("data", "data"),)

        def __init__(self, data):
            self.data = data

    value = torch.empty(0)
    metadata = {"cls": Storage, "is_tensor": False, "nontensor_kwargs": {}}
    objects = [_SavedQuantizedTensor(("data",), metadata)] * 1024
    CountedTuple.copied = 0
    result = restore_from_saved(objects, CountedTuple([value] * 1024))
    assert len(result) == 1024 and all(item.data is value for item in result)
    assert CountedTuple.copied <= 1024


@pytest.mark.parametrize("with_eager", [False, True])
def test_restore_empty_compiled_storage_preserves_suffix_copy(with_eager):
    class EmptyStorage:
        _INNER_TENSORS = ()

    class MutatingEager:
        def restore_from_saved(self, saved):
            assert saved is not original
            saved.reverse()
            return saved

    original = [torch.empty(0), torch.empty(0)]
    metadata = {"cls": EmptyStorage, "is_tensor": False, "nontensor_kwargs": {}}
    objects = [_SavedQuantizedTensor((), metadata)]
    if with_eager:
        objects.append(MutatingEager())
    _, remaining = restore_from_saved(objects, original, return_saved_tensors=True)
    assert remaining is not original
    assert remaining[0] is original[1 if with_eager else 0]
    assert remaining[1] is original[0 if with_eager else 1]


def test_restore_autograd_with_saved_tensor_hooks():
    class Cube(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value, value.square())
            ctx.tensor_objects = [None, None]
            return value.pow(3)

        @staticmethod
        def backward(ctx, gradient):
            _, squared = restore_from_func_ctx(ctx)
            assert ctx.tensor_objects is None
            return gradient * 3 * squared

    value = torch.tensor([2.0, 3.0], requires_grad=True)
    with torch.autograd.graph.saved_tensors_hooks(
        lambda tensor: tensor.clone(), lambda tensor: tensor
    ):
        Cube.apply(value).sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([12.0, 27.0]))
