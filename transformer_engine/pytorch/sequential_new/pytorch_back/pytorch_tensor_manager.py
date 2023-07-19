import torch
from ..common_back.tensor_manager import TensorManagerBase, cublas_workspace
from .pytorch_interface import PytorchInterface
from ... import cpp_extensions


class PytorchTensorManager(TensorManagerBase[torch.Tensor]):
    def gemm(self, in1: str, in2: str, out: str):
        assert issubclass(self.framework, PytorchInterface)

        self._sanity_check(in1, in2, out)

        out_torch_dtype = self.tensor_descriptors[out].dtype.value
        assert isinstance(out_torch_dtype, torch.dtype)
        assert isinstance(self.tensors[in1], torch.Tensor)

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
                cublas_workspace(PytorchInterface),
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
                cublas_workspace(PytorchInterface),
                out=self.tensors[out],
            )
