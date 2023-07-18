from enum import Enum
from math import prod
import subprocess
from typing import Any, Callable, Generic
from attr import dataclass
from .enums import DType
from .framework_interface import FrameworkInterface, TensorType, TensorTypeBase
from . import framework_interface as fi
import transformer_engine_extensions as tex  # TODO: make this framework agnostic


AMAX_HISTORY_LEN = 1024
ALIGN_BYTES = 32


@dataclass
class TransformerEngineExtensionsFP8TensorMeta(Generic[TensorType]):
    scale: TensorType
    scale_inv: TensorType
    amax_history: TensorType


@dataclass
class TensorDescriptor(Generic[TensorType]):
    shape: tuple[int, ...]
    dtype: DType
    init_method: Callable[
        [type[FrameworkInterface[TensorType]], tuple[int, ...], DType], TensorType
    ]


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


class TensorManager(Generic[TensorType]):
    tensor_descriptors = dict[str, TensorDescriptor[TensorType]]()
    meta_storage: TransformerEngineExtensionsFP8TensorMeta[TensorType]
    tensor_storage = dict[DType, TensorType]()
    tensor_indices = dict[str, int]()
    tensors = dict[str, TensorType]()
    allocated = False

    def __init__(self, framework_interface: FrameworkInterface[TensorType]) -> None:
        self.framework_interface = framework_interface
        self.framework = type(framework_interface)

    def register_tensor(self, name: str, desc: TensorDescriptor[TensorType]):
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

    def _is_fp8(self, tensor: str):
        return self.tensor_descriptors[tensor].dtype in [
            DType.FP8E4M3,
            DType.FP8E5M2,
        ]

    def te_dtype(self, tensor: str) -> object:
        assert self._is_fp8(tensor)
        if self.tensor_descriptors[tensor].dtype == DType.FP8E4M3:
            return tex.DType.FP8E4M3  # type: ignore
        else:
            return tex.DType.FP8E5M2  # type: ignore

    def make_tensor_meta(self) -> TransformerEngineExtensionsFP8TensorMeta[TensorType]:
        return tex.FP8TensorMeta()  # type: ignore

    def gemm(self, in1: str, in2: str, out: str):
        import torch
        from .pytorch_interface import PytorchInterface
        from transformer_engine.pytorch import cpp_extensions

        # TODO: make this framework agnostic

        assert issubclass(self.framework, PytorchInterface)

        self._sanity_check(in1, in2, out)

        out_torch_dtype = self.tensor_descriptors[out].dtype.value
        assert isinstance(out_torch_dtype, torch.dtype)

        if self._is_fp8(in1) and self._is_fp8(in2):
            in1_te_dtype = self.te_dtype(in1)
            in2_te_dtype = self.te_dtype(in2)
            out_te_dtype = self.te_dtype(out) if self._is_fp8(out) else None

            cpp_extensions.fp8_gemm(  # type: ignore
                self.tensors[in1],
                self.meta_storage.scale_inv,
                self.tensor_indices[in1],
                in1_te_dtype,
                self.tensors[in2],
                self.meta_storage.scale_inv,
                self.tensor_indices[in2],
                in2_te_dtype,
                out_torch_dtype,
                cublas_workspace(),
                out=self.tensors[out],
                fp8_meta_tensor=self.meta_storage if self._is_fp8(out) else None,
                out_index=self.tensor_indices[out] if self._is_fp8(out) else None,
                D_dtype=out_te_dtype,
            )

        elif self._is_fp8(in1) ^ self._is_fp8(in2):
            raise RuntimeError("Mixed precision `GEMM(FP8, not FP8)` not supported.")
        else:
            if self._is_fp8(out):
                raise RuntimeError(
                    "Mixed precision `GEMM(not FP8, not FP8) -> FP8` not supported."
                )

            cpp_extensions.gemm(  # type: ignore
                self.tensors[in1],
                self.tensors[in2],
                out_torch_dtype,
                cublas_workspace(),
                out=self.tensors[out],
            )
