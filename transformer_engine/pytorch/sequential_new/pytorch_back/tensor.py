from ..common_back.generic_tensor import GenericTensor
import torch
from dataclasses import dataclass
from ..multiple_dispatch import multiple_dispatch
from ... import cpp_extensions
import subprocess


class TorchTensor(GenericTensor):
    pass


@dataclass
class TEFP8TorchTensorMetadata:
    scale: torch.Tensor
    scale_inv: torch.Tensor
    amax_history: torch.Tensor


class TEFP8TorchTensor(GenericTensor):
    tensor: torch.Tensor
    meta_ref: TEFP8TorchTensorMetadata
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
def gemm(a: TEFP8TorchTensor, b: TEFP8TorchTensor, out: GenericTensor):
    cpp_extensions.fp8_gemm(  # type: ignore
        a.tensor,
        a.meta_ref.scale_inv,
        a.index_in_meta,
        a.dtype.tex_dtype,
        b.tensor,
        b.meta_ref.scale_inv,
        b.index_in_meta,
        b.dtype.tex_dtype,
        out.dtype.torch_dtype(),
        cublas_workspace(),
        out=out.tensor,
        fp8_meta_tensor=out.meta_ref if isinstance(out, TEFP8TorchTensor) else None,
        out_index=out.index_in_meta if isinstance(out, TEFP8TorchTensor) else None,
        D_dtype=out.dtype.tex_dtype,
    )


@multiple_dispatch
def gemm(a: TorchTensor, b: TorchTensor, out: TorchTensor):
    cpp_extensions.gemm(  # type: ignore
        a.tensor,
        b.tensor,
        out.dtype.torch_dtype(),
        cublas_workspace(),
        out=out.tensor,
    )
