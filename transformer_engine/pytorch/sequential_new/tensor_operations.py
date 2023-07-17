from enum import Enum
from math import prod
from typing import Any
from attr import dataclass
import torch
from transformer_engine.pytorch import cpp_extensions
import transformer_engine_extensions as tex
from .enums import DType


class InitMethod(Enum):
    ZEROS = (0,)
    ONES = (1,)
    RANDOM = 2


AMAX_HISTORY_LEN = 1024
ALIGN_BYTES = 32


@dataclass
class TransformerEngineExtensionsFP8TensorMeta:
    scale: torch.Tensor
    scale_inv: torch.Tensor
    amax_history: torch.Tensor


@dataclass
class TensorDescriptor:
    shape: tuple[int, ...]
    dtype: DType
    init_method: InitMethod


_cublas_workspace: torch.Tensor | None = None


def cublas_workspace():
    global _cublas_workspace
    if _cublas_workspace is None:
        is_hopper: bool = torch.cuda.get_device_properties(torch.cuda.current_device()).major >= 9  # type: ignore
        workspace_size = 33_554_432 if is_hopper else 4_194_304
        _cublas_workspace = torch.empty(
            workspace_size, dtype=torch.uint8, device="cuda"
        )
    return _cublas_workspace


class TensorManager:
    tensor_descriptors = dict[str, TensorDescriptor]()
    meta_storage: TransformerEngineExtensionsFP8TensorMeta
    tensor_storage = dict[DType, torch.Tensor]()
    tensor_indices = dict[str, int]()
    tensors = dict[str, torch.Tensor]()
    allocated = False

    def __init__(self) -> None:
        raise RuntimeError("This is a static class")

    @staticmethod
    def register_tensor(name: str, desc: TensorDescriptor):
        if not TensorManager.allocated:
            raise RuntimeError("Storage already allocated")
        if name in TensorManager.tensor_descriptors:
            raise RuntimeError(f"Tensor {name} already registered")
        TensorManager.tensor_descriptors[name] = desc

    @staticmethod
    def allocate_storage():
        if TensorManager.allocated:
            raise RuntimeError("Storage already allocated")
        TensorManager.allocated = True

        TensorManager._allocate_fp8_meta()

        def align(x: int):
            return (x + ALIGN_BYTES - 1) & ~(ALIGN_BYTES - 1)

        for dtype in [DType.FP8E4M3, DType.FP8E5M2, DType.FP16, DType.BF16, DType.FP32]:
            tensor_offsets = dict[str, int]()
            prev_offset = 0
            for name, desc in TensorManager.tensor_descriptors.items():
                if desc.dtype != dtype:
                    continue

                offset = align(prev_offset)
                tensor_offsets[name] = offset
                prev_offset = offset + prod(desc.shape)

            torch_dtype = dtype.value
            assert isinstance(torch_dtype, torch.dtype)
            assert dtype not in TensorManager.tensor_storage

            TensorManager.tensor_storage[dtype] = torch.empty(
                prev_offset, dtype=torch_dtype, device="cuda"
            )

            for name, desc in TensorManager.tensor_descriptors.items():
                if desc.dtype != dtype:
                    continue

                offset = tensor_offsets[name]
                tensor = TensorManager.tensor_storage[dtype][
                    offset : offset + prod(desc.shape)
                ].view(desc.shape)
                assert tensor.is_contiguous()

                if desc.init_method == InitMethod.ZEROS:
                    tensor.zero_()
                elif desc.init_method == InitMethod.ONES:
                    tensor.fill_(1)
                elif desc.init_method == InitMethod.RANDOM:
                    tensor.random_()

                TensorManager.tensors[name] = tensor

    @staticmethod
    def _allocate_fp8_meta():
        TensorManager.meta_storage = tex.FP8TensorMeta()  # type: ignore
        TensorManager.meta_storage.scale = torch.ones(
            len(TensorManager.tensor_descriptors),
            dtype=torch.float32,
            device="cuda",
        )
        TensorManager.meta_storage.scale_inv = torch.ones(
            len(TensorManager.tensor_descriptors),
            dtype=torch.float32,
            device="cuda",
        )
        TensorManager.meta_storage.amax_history = torch.zeros(
            AMAX_HISTORY_LEN,
            len(TensorManager.tensor_descriptors),
            dtype=torch.float32,
            device="cuda",
        )
        TensorManager.tensor_indices = {
            name: i for i, name in enumerate(TensorManager.tensor_descriptors.keys())
        }

    @staticmethod
    def _sanity_check(*args: Any):
        if not TensorManager.allocated:
            raise RuntimeError("Storage not allocated")
        for arg in args:
            if not isinstance(arg, str):
                raise RuntimeError(f"Expected string, got {type(arg)}")
            if arg not in TensorManager.tensors:
                raise RuntimeError(f"Tensor {arg} not registered")

    @staticmethod
    def _is_fp8(tensor: str):
        return TensorManager.tensor_descriptors[tensor].dtype in [
            DType.FP8E4M3,
            DType.FP8E5M2,
        ]

    @staticmethod
    def te_dtype(tensor: str) -> object:
        assert TensorManager._is_fp8(tensor)
        if TensorManager.tensor_descriptors[tensor].dtype == DType.FP8E4M3:
            return tex.DType.FP8E4M3  # type: ignore
        else:
            return tex.DType.FP8E5M2  # type: ignore

    @staticmethod
    def gemm(in1: str, in2: str, out: str):
        TensorManager._sanity_check(in1, in2, out)

        out_torch_dtype = TensorManager.tensor_descriptors[out].dtype.value
        assert isinstance(out_torch_dtype, torch.dtype)

        if TensorManager._is_fp8(in1) and TensorManager._is_fp8(in2):
            in1_te_dtype = TensorManager.te_dtype(in1)
            in2_te_dtype = TensorManager.te_dtype(in2)
            out_te_dtype = (
                TensorManager.te_dtype(out) if TensorManager._is_fp8(out) else None
            )

            cpp_extensions.fp8_gemm(  # type: ignore
                TensorManager.tensors[in1],
                TensorManager.meta_storage.scale_inv,
                TensorManager.tensor_indices[in1],
                in1_te_dtype,
                TensorManager.tensors[in2],
                TensorManager.meta_storage.scale_inv,
                TensorManager.tensor_indices[in2],
                in2_te_dtype,
                out_torch_dtype,
                cublas_workspace(),
                out=TensorManager.tensors[out],
                fp8_meta_tensor=TensorManager.meta_storage
                if TensorManager._is_fp8(out)
                else None,
                out_index=TensorManager.tensor_indices[out]
                if TensorManager._is_fp8(out)
                else None,
                D_dtype=out_te_dtype,
            )

        elif TensorManager._is_fp8(in1) ^ TensorManager._is_fp8(in2):
            raise RuntimeError("Mixed precision `GEMM(FP8, not FP8)` not supported.")
        else:
            if TensorManager._is_fp8(out):
                raise RuntimeError(
                    "Mixed precision `GEMM(not FP8, not FP8) -> FP8` not supported."
                )

            cpp_extensions.gemm(  # type: ignore
                TensorManager.tensors[in1],
                TensorManager.tensors[in2],
                out_torch_dtype,
                cublas_workspace(),
                out=TensorManager.tensors[out],
            )
