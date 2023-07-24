from abc import ABC, abstractmethod
from math import prod
import subprocess
from typing import Any, Generic
from attr import dataclass
from .enums import DType
from .framework_interface import FrameworkInterface, TensorType, ParamConstructor
from . import framework_interface as fi
import transformer_engine_extensions as tex  # TODO: make this framework agnostic


AMAX_HISTORY_LEN = 1024
ALIGN_BYTES = 32

@dataclass
class GenericTensor(Generic[TensorType]):
    _dtype: DType
    def dtype(self) -> DType:
        return self._dtype

@dataclass
class TransformerEngineExtensionsFP8TensorMeta(Generic[TensorType]):
    scale: TensorType
    scale_inv: TensorType
    amax_history: TensorType

@dataclass
class NativeTensor(GenericTensor[TensorType]):
    _tensor: TensorType

@dataclass
class FP8Tensor(GenericTensor[TensorType]):
    _tensor: TensorType
    _meta: TransformerEngineExtensionsFP8TensorMeta[TensorType]
    _index: int

@dataclass
class TensorDescriptor:
    shape: tuple[int, ...]
    dtype: DType
    init_method: ParamConstructor


def is_hopper():
    gpu_name = (
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        .decode("utf-8")
        .strip()
    )
    return "H100" in gpu_name


def cublas_workspace(fw: type[FrameworkInterface[TensorType]]) -> TensorType:
    global _cublas_workspace
    if "_cublas_workspace" not in globals():
        workspace_size = 33_554_432 if is_hopper() else 4_194_304
        _cublas_workspace: TensorType = fi.empty(
            fw, (workspace_size,), DType.FP8Any  # type: ignore
        )
    return _cublas_workspace


class TensorManagerBase(ABC, Generic[TensorType]):
    tensor_descriptors = dict[str, TensorDescriptor]()
    meta_storage: TransformerEngineExtensionsFP8TensorMeta[TensorType]
    tensor_storage = dict[DType, TensorType]()
    tensor_indices = dict[str, int]()
    tensors = dict[str, TensorType]()
    allocated = False

    def __init__(self, framework_interface: FrameworkInterface[TensorType]) -> None:
        self.framework_interface = framework_interface
        self.framework = type(framework_interface)

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

            self.tensor_storage[dtype] = self.framework.fi_empty((prev_offset,), dtype)

            for name, desc in self.tensor_descriptors.items():
                if desc.dtype != dtype:
                    continue

                offset = tensor_offsets[name]
                tensor = self.tensor_storage[dtype][
                    offset : offset + prod(desc.shape)
                ].view(desc.shape)
                assert tensor.is_contiguous()

                tensor = desc.init_method(self.framework, desc.shape, desc.dtype)

                self.tensors[name] = tensor

    def retrieve_tensor(self, name: str) -> GenericTensor[TensorType]:
        if not self.allocated:
            raise RuntimeError("Storage not yet allocated")
        if name not in tensors:
            raise RuntimeError("This tensor wasn't registered")

        dtype = self.tensor_descriptors[name].dtype
        tensor = self.tensors[name]

        if dtype.is_fp8():
            meta = self.meta_storage
            index = self.tensor_indices[name]
            return FP8Tensor[TensorType](dtype, tensor, meta, index)
        else:
            return NativeTensor[TensorType](dtype, tensor)

    def _allocate_fp8_meta(
        self,
    ):
        self.meta_storage = self.make_tensor_meta()
        self.meta_storage.scale = fi.ones(
            self.framework,
            (len(self.tensor_descriptors),),
            DType.FP32,
        )
        self.meta_storage.scale_inv = fi.ones(
            self.framework,
            (len(self.tensor_descriptors),),
            DType.FP32,
        )
        self.meta_storage.amax_history = fi.zeros(
            self.framework,
            (AMAX_HISTORY_LEN, len(self.tensor_descriptors)),
            DType.FP32,
        )
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

    def make_tensor_meta(self) -> TransformerEngineExtensionsFP8TensorMeta[TensorType]:
        return tex.FP8TensorMeta()  # type: ignore
