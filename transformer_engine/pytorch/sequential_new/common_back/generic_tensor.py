from dataclasses import dataclass
import torch
from .enums import DType
from ... import cpp_extensions


class GenericTensor:
    tensor: torch.Tensor
    dtype: DType


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


# GEMM
def gemm(a: GenericTensor, b: GenericTensor, out: GenericTensor):
    if a.dtype.is_fp8() != b.dtype.is_fp8():
        raise NotImplementedError("Mixed precision GEMM(FP8, not FP8) is not supported")
    elif a.dtype.is_fp8():  # and b.dtype.is_fp8()
        assert isinstance(a, TEFP8TorchTensor)
        assert isinstance(b, TEFP8TorchTensor)
        _gemm_fp8(a, b, out)
    else:  # neither is fp8
        if out.dtype.is_fp8():
            raise NotImplementedError(
                "Mixed precision GEMM(not FP8, not FP8) -> FP8 is not supported"
            )
        assert isinstance(a, TorchTensor)
        assert isinstance(b, TorchTensor)
        assert isinstance(out, TorchTensor)
        _gemm(a, b, out)


def _gemm_fp8(a: TEFP8TorchTensor, b: TEFP8TorchTensor, out: GenericTensor):
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
        _cublas_workspace(),
        out=out.tensor,
        fp8_meta_tensor=out.meta_ref if isinstance(out, TEFP8TorchTensor) else None,
        out_index=out.index_in_meta if isinstance(out, TEFP8TorchTensor) else None,
        D_dtype=out.dtype.tex_dtype,
    )


def _gemm(a: TorchTensor, b: TorchTensor, out: TorchTensor):
    cpp_extensions.gemm(  # type: ignore
        a.tensor,
        b.tensor,
        out.dtype.torch_dtype(),
        _cublas_workspace(),
        out=out.tensor,
    )
