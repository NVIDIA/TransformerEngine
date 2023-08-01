from enum import Enum
from typing import NewType
import torch
import transformer_engine_extensions as tex

TexDType = NewType("TexDType", int)


class DType(Enum):
    FP8E4M3 = "FP8E4M3"
    FP8E5M2 = "FP8E5M2"
    FP16 = "FP16"
    BF16 = "BF16"
    FP32 = "FP32"
    default = BF16

    @staticmethod
    def from_torch_dtype(torch_dtype: torch.dtype):
        if torch_dtype == torch.float16:
            return DType.FP16
        elif torch_dtype == torch.bfloat16:
            return DType.BF16
        elif torch_dtype == torch.float32:
            return DType.FP32
        else:
            raise ValueError(f"Unsupported torch dtype {torch_dtype}")

    def is_fp8(self):
        return "FP8" in self.name

    def tex_dtype(self) -> TexDType:
        name = self.name.replace("FP", "F").replace("F", "Float")
        return getattr(tex.DType, f"k{name}")  # type: ignore

    def torch_dtype(self) -> torch.dtype:
        if self.is_fp8():
            return torch.int8
        else:
            name = self.name.replace("FP", "F").replace("F", "Float")
            return getattr(torch, name.lower())  # type: ignore


class DTypeInfer:
    pass


class PType(Enum):
    NA = 0
    NCS = 1
    PA = 2
    NRS = 3
