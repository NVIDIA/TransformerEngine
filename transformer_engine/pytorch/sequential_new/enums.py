from enum import Enum
import torch


class DType(Enum):
    FP8E4M3 = torch.int8
    FP8E5M2 = torch.int8
    FP16 = torch.float16
    BF16 = torch.bfloat16
    FP32 = torch.float32
    infer = "INFER"
    default = "DEFAULT"


class InitMethod(Enum):
    ZEROS = (0,)
    ONES = (1,)
    RANDOM = 2
