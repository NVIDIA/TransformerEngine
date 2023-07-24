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

    infer = "infer"
    default = BF16

    def is_fp8(self):
        return "FP8" in self.name

    def tex_dtype(self) -> TexDType:
        if self is DType.infer:
            raise ValueError(
                "This type has not been inferred and doesn't correspond to any specific type"
            )

        name = self.name.replace("FP", "F").replace("F", "Float")
        return getattr(tex.DType, f"k{name}")  # type: ignore

    def torch_dtype(self) -> torch.dtype:
        if self is DType.infer:
            raise ValueError(
                "This type has not been inferred and doesn't correspond to any specific type"
            )

        if self.is_fp8():
            return torch.int8
        else:
            name = self.name.replace("FP", "F").replace("F", "Float")
            return getattr(torch, name.lower())  # type: ignore


class PType(Enum):
    NA = 0
    NCS = 1
    PA = 2
    NRS = 3
