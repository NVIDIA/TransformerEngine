from ..common_back.generic_tensor import (
    TransformerEngineExtensionsFP8TensorMeta,
    FP8Tensor,
    NativeTensor,
)
from ..common_back.enums import DType
import torch
from dataclasses import dataclass
from ..multiple_dispatch import multiple_dispatch
from ... import cpp_extensions
import subprocess


@dataclass
class PytorchTransformerEngineExtensionsFP8TensorMeta(
    TransformerEngineExtensionsFP8TensorMeta
):
    scale: torch.Tensor
    scale_inv: torch.Tensor
    amax_history: torch.Tensor


class PytorchNativeTensor(NativeTensor):
    dtype: DType
    tensor: torch.Tensor


class PytorchFP8Tensor(FP8Tensor):
    tensor: torch.Tensor
    meta_ref: PytorchTransformerEngineExtensionsFP8TensorMeta
    index_in_meta: int


def is_hopper():
    gpu_name = (
        subprocess.check_output(
            "nvidia-smi --query-gpu=name --format=csv,noheader", shell=True
        )
        .decode("utf-8")
        .strip()
    )
    return "H100" in gpu_name


_cublas_workspace: torch.Tensor


def cublas_workspace():
    global _cublas_workspace
    if "_cublas_workspace" not in globals():
        workspace_size = 33_554_432 if is_hopper() else 4_194_304
        _cublas_workspace = torch.empty(workspace_size, dtype=torch.int8, device="cuda")
    return _cublas_workspace


@multiple_dispatch
def gemm(
    a: PytorchFP8Tensor,
    b: PytorchFP8Tensor,
    out: PytorchNativeTensor | PytorchFP8Tensor,
):
    cpp_extensions.fp8_gemm(  # type: ignore
        a.tensor,
        a.meta_ref.scale_inv,
        a.index_in_meta,
        a.dtype.tex_dtype(),
        b.tensor,
        b.meta_ref.scale_inv,
        b.index_in_meta,
        b.dtype.tex_dtype(),
        out.dtype.torch_dtype(),
        cublas_workspace(),
        out=out.tensor,
        fp8_meta_tensor=out.meta_ref if isinstance(out, PytorchFP8Tensor) else None,
        out_index=out.index_in_meta if isinstance(out, PytorchFP8Tensor) else None,
        D_dtype=out.dtype.tex_dtype(),
    )


@multiple_dispatch
def gemm(a: PytorchNativeTensor, b: PytorchNativeTensor, out: PytorchNativeTensor):
    cpp_extensions.gemm(  # type: ignore
        a.tensor,
        b.tensor,
        out.dtype.torch_dtype(),
        cublas_workspace(),
        out=out.tensor,
    )
