# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Host metadata lifetime and workspace ownership."""
import gc
import weakref

import pytest
import torch

from transformer_engine.pytorch.attention import packed_sequence as metadata


@pytest.mark.parametrize("offsets", [(0,), (0, 1), (0, 0, 257, 4097), (0, 16384, 65536)])
def test_registered_metadata_never_reads_tensor(offsets, monkeypatch):
    tensor = torch.tensor(offsets, dtype=torch.int32)
    monkeypatch.setattr(torch.Tensor, "tolist", lambda *_: pytest.fail("Unexpected readback"))
    metadata.register_cu_seqlens(tensor, offsets)
    assert metadata._get_cu_seqlens(tensor) == offsets


def test_mutation_and_clone_require_registration():
    tensor = torch.tensor([0, 3, 8], dtype=torch.int32)
    metadata.register_cu_seqlens(tensor, [0, 3, 8])
    assert metadata._get_cu_seqlens(tensor.clone()) is None
    alias = tensor.view_as(tensor)
    assert metadata._get_cu_seqlens(alias) == (0, 3, 8)
    alias[-1] = 9
    assert metadata._get_cu_seqlens(tensor) is None
    metadata.register_cu_seqlens(tensor, [0, 3, 9])
    assert metadata._get_cu_seqlens(tensor) == (0, 3, 9)


def test_registration_does_not_retain_tensor_or_mutable_host_list():
    tensor = torch.tensor([0, 3], dtype=torch.int32)
    key = (tensor.device, tensor.data_ptr())
    host = [0, 3]
    metadata.register_cu_seqlens(tensor, host)
    host[-1] = 7
    assert metadata._get_cu_seqlens(tensor) == (0, 3)
    ref = weakref.ref(tensor)
    del tensor
    gc.collect()
    assert ref() is None
    assert key not in metadata._PREFIXES


@pytest.mark.parametrize("offsets", [(1, 3), (0, 3, 2), (), (0, 2**31)])
def test_invalid_offsets(offsets):
    with pytest.raises(ValueError):
        metadata.register_cu_seqlens(torch.zeros(len(offsets), dtype=torch.int32), offsets)


def test_tensor_metadata_is_not_implicitly_copied_to_host():
    tensor = torch.tensor([0, 3], dtype=torch.int32)
    with pytest.raises(TypeError):
        metadata.register_cu_seqlens(tensor, tensor)


def test_shape_and_dtype_mismatch():
    with pytest.raises(ValueError):
        metadata.register_cu_seqlens(torch.tensor([0, 3]), [0, 3])
    with pytest.raises(ValueError):
        metadata.register_cu_seqlens(torch.zeros(3, dtype=torch.int32), [0, 3])


def test_inference_tensor_is_not_retained():
    with torch.inference_mode():
        tensor = torch.tensor([0, 3], dtype=torch.int32)
        metadata.register_cu_seqlens(tensor, [0, 3])
        assert metadata._get_cu_seqlens(tensor) is None


def test_workspace_nested_and_exception_cleanup():
    assert metadata._get_attention_backend_workspace() is None
    with pytest.raises(RuntimeError, match="training failed"):
        with metadata.attention_backend_workspace():
            owner = metadata._get_attention_backend_workspace()
            tensor = torch.empty(10)
            ref = weakref.ref(tensor)
            owner.buffers["test"] = tensor
            del tensor
            with metadata.attention_backend_workspace():
                assert metadata._get_attention_backend_workspace() is owner
            assert owner.active and ref() is not None
            raise RuntimeError("training failed")
    assert not owner.active and not owner.buffers and ref() is None
    assert metadata._get_attention_backend_workspace() is None
