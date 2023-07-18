from enum import Enum
import torch


class DType(Enum):
    FP8Any = "FP8Any"
    FP8E4M3 = "FP8E4M3"
    FP8E5M2 = "FP8E5M2"
    FP16 = "FP16"
    BF16 = "BF16"
    FP32 = "FP32"
    infer = "infer"
    default = BF16


class InitMethod(Enum):
    ZEROS = (0,)
    ONES = (1,)
    RANDOM = 2
