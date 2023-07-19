from abc import abstractmethod
from functools import reduce
import operator
from typing import Generic, NoReturn
from typing_extensions import TypeVarTuple, Unpack

from transformer_engine.pytorch.sequential_new.enums import DType
from .ops import (
    Bias,
    BiasGrad,
    Gelu,
    GeluGrad,
    Gemm,
    Grad,
    Op,
    ParamDescriptor,
    ResidualBegin,
    ResidualEnd,
    Dropout,
    DropoutGrad,
)


# Fused op base class
class FusedOp(Op):
    pass


# Auto fuser
Ops = TypeVarTuple("Ops", default=Unpack[tuple[Op]])


class AutoFuse(FusedOp, Generic[Unpack[Ops]]):
    pass


# Manual fusions
class GemmBias(FusedOp):
    pass


class GemmBiasGelu(FusedOp):
    pass


# Fuser
FusedOpTypes = TypeVarTuple("FusedOpTypes", default=Unpack[tuple[FusedOp]])


class Fuser(Generic[Unpack[FusedOpTypes]]):
    pass


TE_FUSER = Fuser[
    # GemmBias,
    # GemmBiasGelu,
    AutoFuse[GeluGrad, BiasGrad],
    AutoFuse[Bias, Dropout, ResidualEnd],
    AutoFuse[ResidualBegin, BiasGrad, DropoutGrad],
    AutoFuse[Bias, Dropout, ResidualEnd, ResidualBegin],
    AutoFuse[ResidualEnd, ResidualBegin, BiasGrad, DropoutGrad],
]()
