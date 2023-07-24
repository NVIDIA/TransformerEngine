from dataclasses import dataclass
from typing import Protocol
import torch
from .enums import DType

class GenericTensor(Protocol):
    def dtype() -> DType:
        raise NotImplementedError()

class TorchTensor(GenericTensor):
    _tensor: torch.Tensor

@dataclass
class TEFP8TorchTensorMetadata:
    scale: torch.Tensor
    scale_inv: torch.Tensor
    amax_history: torch.Tensor

class TEFP8TorchTensor(GenericTensor):
    _tensor: torch.Tensor
    _meta_ref: TEFP8TorchTensorMetadata
    _index_in_meta: int

# GEMM
def gemm(a: GenericTensor, b: GenericTensor, out: GenericTensor):
    if a.dtype().is_fp8() != b.dtype().is_fp8():
        raise NotImplementedError("Mixed precision GEMM(FP8, not FP8) is not supported")
    elif a.dtype().is_fp8(): # and b.dtype().is_fp8()
        assert isinstance(a, TEFP8TorchTensor)
        assert isinstance(b, TEFP8TorchTensor)
        _gemm_fp8(a, b, out)
    else: # neither is fp8
        if out.dtype().is_fp8():
            raise NotImplementedError("Mixed precision GEMM(not FP8, not FP8) -> FP8 is not supported")
        assert isinstance(a, TorchTensor)
        assert isinstance(b, TorchTensor)
        assert isinstance(out, TorchTensor)
        _gemm(a, b, out)

def _gemm_fp8(a: TEFP8TorchTensor, b: TEFP8TorchTensor, out: GenericTensor):
    cpp_extensions.fp8_gemm(  # type: ignore
        a._tensor,
        a._meta_ref.scale_inv,
        a._index_in_meta,
        a.dtype().tex_dtype(),
        b._tensor,
        b._meta_ref.scale_inv,
        b._index_in_meta,
        b.dtype().tex_dtype(),
        out.dtype().torch_dtype(),
        _cublas_workspace(),
        out=out._tensor,
        fp8_meta_tensor=out._meta_ref if isinstance(out, TEFP8TorchTensor) else None,
        out_index=out._index_in_meta if isinstance(out, TEFP8TorchTensor) else None,
        D_dtype=out.dtype().tex_dtype(),
    )

def _gemm(a: TorchTensor, b: TorchTensor, out: TorchTensor):
    cpp_extensions.gemm(  # type: ignore
        a._tensor,
        b._tensor,
        out.dtype().torch_dtype(),
        _cublas_workspace(),
        out=out._tensor,
    )
