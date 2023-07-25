from math import prod
from typing import Any
from .enums import DType
import transformer_engine_extensions as tex  # TODO: make this framework agnostic
from .generic_tensor import (
    GenericTensor,
    NativeTensor,
    FP8Tensor,
    TransformerEngineExtensionsFP8TensorMeta,
    TensorDescriptor,
)
from . import generic_tensor as f

AMAX_HISTORY_LEN = 1024
ALIGN_BYTES = 32


class TensorManager:
    tensor_descriptors = dict[str, TensorDescriptor]()
    meta_storage: TransformerEngineExtensionsFP8TensorMeta
    tensor_storage = dict[DType, NativeTensor]()
    tensor_indices = dict[str, int]()
    tensors = dict[str, NativeTensor]()
    allocated = False

    def register_tensor(self, name: str, desc: TensorDescriptor):
        if not self.allocated:
            raise RuntimeError("Storage already allocated")
        if name in self.tensor_descriptors:
            raise RuntimeError(f"Tensor {name} already registered")
        self.tensor_descriptors[name] = desc

    def allocate_storage(
        self,
    ):
        if self.allocated:
            raise RuntimeError("Storage already allocated")
        self.allocated = True

        self._allocate_fp8_meta()

        def align(x: int):
            return (x + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1)

        for dtype in [DType.FP8E4M3, DType.FP8E5M2, DType.FP16, DType.BF16, DType.FP32]:
            tensor_offsets = dict[str, int]()
            prev_offset = 0
            for name, desc in self.tensor_descriptors.items():
                if desc.dtype != dtype:
                    continue

                offset = align(prev_offset)
                tensor_offsets[name] = offset
                prev_offset = offset + prod(desc.shape)

            assert dtype not in self.tensor_storage

            self.tensor_storage[dtype] = f.empty((prev_offset,), dtype)

            for name, desc in self.tensor_descriptors.items():
                if desc.dtype != dtype:
                    continue

                offset = tensor_offsets[name]
                tensor = self.tensor_storage[dtype][
                    offset : offset + prod(desc.shape)
                ].view(desc.shape)
                assert tensor.is_contiguous()

                if desc.constructor is not None:
                    desc.constructor(desc.shape, desc.dtype, tensor)

                self.tensors[name] = tensor

    def retrieve_tensor(self, name: str) -> GenericTensor:
        if not self.allocated:
            raise RuntimeError("Storage not yet allocated")
        if name not in self.tensors:
            raise RuntimeError("This tensor wasn't registered")

        dtype = self.tensor_descriptors[name].dtype
        tensor = self.tensors[name]

        if dtype.is_fp8():
            meta = self.meta_storage
            index = self.tensor_indices[name]
            return FP8Tensor(dtype, tensor, meta, index)
        else:
            return tensor

    def _allocate_fp8_meta(self):
        self.meta_storage = self.make_tensor_meta()
        self.meta_storage.scale = f.empty((len(self.tensor_descriptors),), DType.FP32)
        self.meta_storage.scale_inv = f.empty(
            (len(self.tensor_descriptors),), DType.FP32
        )
        self.meta_storage.amax_history = f.empty(
            (
                AMAX_HISTORY_LEN,
                len(self.tensor_descriptors),
            ),
            DType.FP32,
        )
        f.ones(self.meta_storage.scale)
        f.ones(self.meta_storage.scale_inv)
        f.zeros(self.meta_storage.amax_history)
        self.tensor_indices = {
            name: i for i, name in enumerate(self.tensor_descriptors.keys())
        }

    def _sanity_check(self, *args: Any):
        if not self.allocated:
            raise RuntimeError("Storage not allocated")
        for arg in args:
            if not isinstance(arg, str):
                raise RuntimeError(f"Expected string, got {type(arg)}")
            if arg not in self.tensors:
                raise RuntimeError(f"Tensor {arg} not registered")

    def make_tensor_meta(self) -> TransformerEngineExtensionsFP8TensorMeta:
        return tex.FP8TensorMeta()  # type: ignore
