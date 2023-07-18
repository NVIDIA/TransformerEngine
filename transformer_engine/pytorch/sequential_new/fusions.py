from abc import abstractmethod
from functools import reduce
import operator
from typing import Generic, NoReturn
from typing_extensions import TypeVarTuple, Unpack

from transformer_engine.pytorch.sequential_new.enums import DType
from .ops import (
    Bias,
    BiasGrad,
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

Ops = TypeVarTuple("Ops", default=Unpack[tuple[Op]])
Grads = TypeVarTuple("Grads", default=Unpack[tuple[Grad]])


class FusedOp(Op, Generic[Unpack[Ops]]):
    def __init__(self, *ops: Unpack[Ops]):
        self.ops = ops
        super().__init__(
            "Fused " + " ".join(str(type(op)) for op in ops),
            self.ops[0].input_type,
            self.ops[-1].output_type,
        )

    def describe_params(self) -> dict[str, ParamDescriptor]:
        return reduce(operator.ior, [op.describe_params() for op in self.ops], {})


class FusedGrad(Grad, Generic[Unpack[Grads]]):
    def __init__(self, orig: Op, *ops: Unpack[Grads]):
        self.ops = ops
        super().__init__(orig)

    def io_types(self):
        return (self.ops[0].io_types()[0], self.ops[-1].io_types()[-1])


class PostBackwardFusedOp(FusedOp, Generic[Unpack[Ops]]):
    def bwd(self) -> NoReturn:
        raise RuntimeError("Backward generation should be done before this fusion")


class PreBackwardFusedOp(FusedOp, Generic[Unpack[Ops]]):
    def bwd(self):
        return type(self).grad_type()(op.bwd() for op in self.ops[::-1])

    @staticmethod
    @abstractmethod
    def grad_type() -> type[Grad]:
        ...


# Post-backward generation fusions
class GemmBias(PostBackwardFusedOp[Gemm, Bias]):
    pass


class GeluBiasGrad(PostBackwardFusedOp[GeluGrad, BiasGrad]):
    pass


# Pre-backward generation fusions
class BiasDropoutResidual(FusedOp[Bias, Dropout, ResidualEnd]):
    @staticmethod
    def grad_type():
        return ResidualDropoutBiasGrad


class ResidualDropoutBiasGrad(FusedGrad[ResidualBegin, BiasGrad, DropoutGrad]):
    pass


class BiasDropoutResiduals(FusedOp[Bias, Dropout, ResidualEnd, ResidualBegin]):
    @staticmethod
    def grad_type():
        return ResidualsDropoutBiasGrad


class ResidualsDropoutBiasGrad(
    FusedGrad[ResidualEnd, ResidualBegin, BiasGrad, DropoutGrad]
):
    pass
