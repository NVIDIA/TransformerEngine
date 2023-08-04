from __future__ import annotations
from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Generator, final

from transformer_engine.new.common.generic_tensor import GenericTensor
from .enums import DType, PType, DTypeInfer
from .generic_tensor import (
    GenericTensor,
    TensorDescriptor,
    ParamInitializer,
)
from . import generic_tensor as f

ExecutionFlow = list["Op"]
Parallelism = tuple[PType, PType]
Context = dict[str, GenericTensor]
ParamGrads = dict[str, GenericTensor]


# Op Protocol
class Unset:
    pass


class Op(ABC):
    """
    An Op represents a transformation applied to the main data tensor.
    It is akin to an nn.Module + autograd.Function pair in PyTorch.

    Inside ComputePipeline, the following methods are called in the order
    in which they are declared below.
    """

    name: str
    original_source: object
    __world_size: int | Unset
    __input_type: DType | DTypeInfer
    __output_type: DType | DTypeInfer
    __parallellism: Parallelism | Unset

    @property
    def world_size(self):
        assert not isinstance(self.__world_size, Unset)
        return self.__world_size

    @property
    def raw_input_type(self):
        return self.__input_type

    @raw_input_type.setter
    def raw_input_type(self, value: DType | DTypeInfer):
        self.__input_type = value

    @property
    def raw_output_type(self):
        return self.__output_type

    @raw_output_type.setter
    def raw_output_type(self, value: DType | DTypeInfer):
        self.__output_type = value

    @property
    def input_type(self):
        assert isinstance(self.__input_type, DType)
        return self.__input_type

    @property
    def output_type(self):
        assert isinstance(self.__output_type, DType)
        return self.__output_type

    @property
    def parallellism(self):
        assert not isinstance(self.__parallellism, Unset)
        return self.__parallellism

    def __init__(
        self,
        relative_name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
    ):
        self.name = relative_name
        self.__input_type = input_type
        self.__output_type = output_type
        self.__world_size = Unset()
        self.__parallellism = Unset()

    def set_parent_name(self, parent_module_name: str):
        self.name = parent_module_name + "." + self.name
        return self

    def set_world_size(self, world_size: int):
        self.__world_size = world_size
        return self

    def set_types_inferred(
        self, inferred_input_type: DType, inferred_output_type: DType
    ):
        self.__input_type = inferred_input_type
        self.__output_type = inferred_output_type
        return self

    @abstractmethod
    def describe_parallellism(self) -> list[ExecutionFlow]:
        raise NotImplementedError()

    def set_parallelism(self, chosen_parallelism: Parallelism):
        self.__parallellism = chosen_parallelism
        self._pre_describe_tensors()
        return self

    @abstractmethod
    def _pre_describe_tensors(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def describe_tensors(self) -> dict[str, TensorDescriptor]:
        raise NotImplementedError()

    def set_tensors_allocated(self, **tensors: GenericTensor):
        for name, tensor in tensors.items():
            setattr(self, name, tensor)

    @abstractmethod
    def forward(
        self, x: GenericTensor, **tensors: GenericTensor
    ) -> tuple[GenericTensor, Context]:
        raise NotImplementedError()

    @abstractmethod
    def backward(
        self, grad: GenericTensor, **context: GenericTensor
    ) -> tuple[GenericTensor, ParamGrads]:
        raise NotImplementedError()

    @abstractmethod
    def inference_optimized(
        self, x: GenericTensor, **tensors: GenericTensor
    ) -> GenericTensor:
        raise NotImplementedError()


# Base classes
class NoTensorOp(Op):
    """
    Base class for ops that do not have any trainable parameters.
    """

    def describe_tensors(self) -> dict[str, TensorDescriptor]:
        return {}


# Parallelism base classes
class ParallelismClass:
    NORMAL = (PType.NA, PType.NA)
    ROWP = (PType.NRS, PType.NRS)
    COLP = (PType.NCS, PType.NCS)
    CGEMM = (PType.NA, PType.NCS)
    RGEMM = (PType.NCS, PType.PA)
    S = (PType.NA, PType.NRS)
    RS = (PType.PA, PType.NRS)
    AG = (PType.NRS, PType.NA)
    AR = (PType.PA, PType.NA)


def __single_parallel(parallelism: Parallelism, op: Op):
    op = deepcopy(op)
    op.set_parallelism(parallelism)
    return [op]


_normal = partial(__single_parallel, ParallelismClass.NORMAL)
_row_parallel = partial(__single_parallel, ParallelismClass.ROWP)
_column_parallel = partial(__single_parallel, ParallelismClass.COLP)


class PointwiseOp(Op):
    def describe_parallellism(self):
        return [_normal(self), _row_parallel(self), _column_parallel(self)]


class RowwiseOp(Op):
    def describe_parallellism(self):
        return [_normal(self), _row_parallel(self)]


class ColumnwiseOp(Op):
    def describe_parallellism(self):
        return [_normal(self), _column_parallel(self)]


class EnvObliviousOp(Op):
    def _pre_describe_tensors(self) -> None:
        return


class NonParallelOp(EnvObliviousOp):
    def describe_parallellism(self):
        return [_normal(self)]


# Identity
@final
class Identity(
    PointwiseOp,
    NoTensorOp,
    EnvObliviousOp,
):
    def forward(self, x: GenericTensor, **_):
        return x, Context()

    def backward(self, grad: GenericTensor, **_):
        return grad, ParamGrads()

    def inference_optimized(self, x: GenericTensor, **_):
        return x


# Transpose
@final
class Transpose(NonParallelOp, NoTensorOp):
    def forward(self, x: GenericTensor, **_):
        return f.transpose(x), Context()

    def backward(self, grad: GenericTensor, **_):
        return f.transpose(grad), ParamGrads()

    def inference_optimized(self, x: GenericTensor, **_):
        return f.transpose(x)


# Normalization
@final
class LayerNorm(RowwiseOp, EnvObliviousOp):
    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        features: int,
        eps: float,
        zero_centered_gamma: bool,
    ):
        super().__init__(name, input_type, output_type)
        self.param_type = param_type
        self.features = features
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma

    def describe_tensors(
        self,
    ) -> dict[str, TensorDescriptor]:
        return {
            "weight": TensorDescriptor(
                (self.features,),
                f.zeros if self.zero_centered_gamma else f.ones,
                self.param_type,
                True,
            ),
            "bias": TensorDescriptor((self.features,), f.zeros, self.param_type, True),
        }

    def forward(
        self, x: GenericTensor, *, weight: GenericTensor, bias: GenericTensor, **_
    ):
        act, mu, rsigma = f.layer_norm(
            x,
            weight,
            bias,
            self.eps,
            self.zero_centered_gamma,
        )
        return act, {"x": x, "weight": weight, "mu": mu, "rsigma": rsigma}

    def backward(
        self,
        grad: GenericTensor,
        *,
        x: GenericTensor,
        weight: GenericTensor,
        mu: GenericTensor,
        rsigma: GenericTensor,
        **_,
    ):
        dgrad, wgrad, bgrad = f.dlayer_norm(
            grad,
            x,
            weight,
            self.zero_centered_gamma,
            mu,
            rsigma,
        )
        return dgrad, {"weight": wgrad, "bias": bgrad}

    def inference_optimized(
        self, x: GenericTensor, *, weight: GenericTensor, bias: GenericTensor, **_
    ):
        return f.layer_norm_inf(
            x,
            weight,
            bias,
            self.eps,
            self.zero_centered_gamma,
        )


# Linear
_rgemm = partial(__single_parallel, ParallelismClass.RGEMM)
_cgemm = partial(__single_parallel, ParallelismClass.CGEMM)


def _rgemm_rs(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    op.set_parallelism(ParallelismClass.RGEMM)
    rs = (
        ReduceScatter("post-rs", op.output_type, op.output_type)
        .set_parent_name(op.name)
        .set_world_size(op.world_size)
        .set_types_inferred(op.output_type, op.output_type)
        .set_parallelism(ParallelismClass.RS)
    )
    return _rgemm(op) + [rs]


def _ag_cgemm(op: Gemm) -> list[Op]:
    op = deepcopy(op)
    ag = (
        AllGather("pre-ag", op.input_type, op.input_type)
        .set_parent_name(op.name)
        .set_world_size(op.world_size)
        .set_types_inferred(op.output_type, op.output_type)
        .set_parallelism(ParallelismClass.AG)
    )
    op.set_parallelism(ParallelismClass.CGEMM)
    return [ag] + _cgemm(op)


@final
class Gemm(Op):
    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        in_features: int,
        out_features: int,
        init_method: ParamInitializer,
    ):
        super().__init__(name, input_type, output_type)
        self.in_features = in_features
        self.out_features = out_features
        self.param_type = param_type
        self.init_method = init_method

    def _pre_describe_tensors(self):
        assert self.parallellism in [
            ParallelismClass.NORMAL,
            ParallelismClass.RGEMM,
            ParallelismClass.CGEMM,
        ]

        workers = (
            self.world_size
            if self.parallellism in [ParallelismClass.RGEMM, ParallelismClass.CGEMM]
            else 1
        )
        if self.parallellism == ParallelismClass.RGEMM:
            if self.in_features % workers != 0:
                raise ValueError(
                    "Number of input features must be divisible by the distributed group size"
                )
            self.in_features //= workers
        elif self.parallellism == ParallelismClass.CGEMM:
            if self.out_features % workers != 0:
                raise ValueError(
                    "Number of output features must be divisible by the distributed group size"
                )
            self.out_features //= workers

    def describe_parallellism(self):
        return [
            _normal(self),
            _rgemm(self),
            _cgemm(self),
            _rgemm_rs(self),
            _ag_cgemm(self),
        ]

    def describe_tensors(self) -> dict[str, TensorDescriptor]:
        return {
            "weight": TensorDescriptor(
                (self.in_features, self.out_features),
                self.init_method,
                self.param_type,
                True,
            ),
        }

    def forward(
        self, x: GenericTensor, *, weight: GenericTensor, **_
    ) -> tuple[GenericTensor, Context]:
        if self.input_type.is_fp8() and not self.param_type.is_fp8():
            weight_fp8, weight_t = f.cast_transpose_fp8(weight)

        return f.gemm(x, weight), {"x": x, "weight": weight}

    def backward(
        self,
        grad: GenericTensor,
        *,
        x: GenericTensor,
        weight: GenericTensor,
        **_,
    ) -> tuple[GenericTensor, ParamGrads]:
        dgrad = f.gemm(grad, f.transpose(weight))
        wgrad = f.gemm(f.transpose(x), grad)
        return dgrad, {"weight": wgrad}

    def inference_optimized(self, x: GenericTensor, *, weight: GenericTensor, **_):
        return f.gemm(x, weight)


@final
class Bias(PointwiseOp):
    bias: GenericTensor
    bias_grad: GenericTensor

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        param_type: DType,
        features: int,
        init_method: ParamInitializer,
    ):
        super().__init__(name, input_type, output_type)
        self.param_dtype = param_type
        self.features = features
        self.init_method = init_method

    def _pre_describe_tensors(self):
        assert self.parallellism in [
            ParallelismClass.NORMAL,
            ParallelismClass.ROWP,
            ParallelismClass.COLP,
        ]
        workers = self.world_size if self.parallellism == ParallelismClass.COLP else 1
        if self.features % workers != 0:
            raise ValueError(
                "Number of features must be divisible by the distributed group size"
            )
        self.features //= workers

    def describe_tensors(self) -> dict[str, TensorDescriptor]:
        return {
            "bias": TensorDescriptor(
                (self.features,), self.init_method, self.param_dtype, True
            ),
        }

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.add(x, self.bias)
        self.bias_grad = grad
        yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        return f.add(x, self.bias)


# Attention
HEAD_PARALLEL_SCATTERING = (PType.NA, PType.NCS)
HEAD_PARALLEL_GATHERING = (PType.NCS, PType.NA)


@final
class DotProductAttention(NoTensorOp):  # TODO
    features_per_head: int

    def __init__(
        self,
        name: str,
        input_type: DType | DTypeInfer,
        output_type: DType | DTypeInfer,
        features_per_head: int,
    ):
        super().__init__(
            name,
            input_type,
            output_type,
        )
        self.features_per_head = features_per_head

    def describe_parallellism(self):
        return [
            _normal(self),
            _column_parallel(self),
            HEAD_PARALLEL_SCATTERING,
            HEAD_PARALLEL_GATHERING,
        ]


# Residual
@final
class ResidualBegin(PointwiseOp, NoTensorOp, EnvObliviousOp):
    end: ResidualEnd | None = None
    fwd_residue: GenericTensor

    def __init__(self, name: str, end: ResidualEnd | None = None):
        super().__init__(name, DType.default, DType.default)
        self.end = end

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        assert self.end is not None
        self.fwd_residue = x
        grad = yield x
        self.end.bwd_residue = f.add(self.end.bwd_residue, grad)
        yield self.end.bwd_residue

    def inference(self, x: GenericTensor) -> GenericTensor:
        self.fwd_residue = x
        return x


@final
class ResidualEnd(PointwiseOp, NoTensorOp, EnvObliviousOp):
    begin: ResidualBegin
    bwd_residue: GenericTensor

    def __init__(self, name: str, begin: ResidualBegin):
        super().__init__(name, DType.default, DType.default)
        self.begin = begin

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        self.begin.fwd_residue = f.add(self.begin.fwd_residue, x)
        grad = yield self.begin.fwd_residue
        self.bwd_residue = grad
        yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        self.begin.fwd_residue = f.add(self.begin.fwd_residue, x)
        return self.begin.fwd_residue


# Dropout
@final
class Dropout(
    PointwiseOp,
    NoTensorOp,
    EnvObliviousOp,
):
    p: float

    def __init__(self, name: str, p: float):
        super().__init__(name, DTypeInfer(), DTypeInfer())
        self.p = p

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.dropout(x, self.p)
        yield f.dropout(grad, self.p)

    def inference(self, x: GenericTensor) -> GenericTensor:
        return f.dropout(x, self.p)


# Activation
@final
class Gelu(PointwiseOp, NoTensorOp, EnvObliviousOp):
    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.gelu(x)
        yield f.dgelu(grad, x)

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        return f.gelu(x)


@final
class Relu(PointwiseOp, NoTensorOp, EnvObliviousOp):
    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.relu(x)
        yield f.drelu(grad, x)

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        return f.relu(x)


# Cast
@final
class Cast(PointwiseOp, NoTensorOp, EnvObliviousOp):
    act: GenericTensor

    def training(
        self,
        x: GenericTensor,
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.cast(x, self.output_type)
        yield f.cast(grad, self.input_type)

    def inference(
        self,
        x: GenericTensor,
    ) -> GenericTensor:
        return f.cast(x, self.output_type)


# Communication

_scatter = partial(__single_parallel, ParallelismClass.S)
_reduce_scatter = partial(__single_parallel, ParallelismClass.RS)
_all_gather = partial(__single_parallel, ParallelismClass.AG)
_all_reduce = partial(__single_parallel, ParallelismClass.AR)


@final
class Scatter(NoTensorOp, EnvObliviousOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_scatter(self)]

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.scatter(x)
        yield f.gather(grad)

    def inference(self, x: GenericTensor) -> GenericTensor:
        return f.scatter(x)


@final
class ReduceScatter(NoTensorOp, EnvObliviousOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_reduce_scatter(self)]

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.reduce_scatter(x)
        yield f.all_gather(grad)

    def inference(self, x: GenericTensor) -> GenericTensor:
        return f.reduce_scatter(x)


@final
class AllGather(NoTensorOp, EnvObliviousOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_all_gather(self)]

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.all_gather(x)
        yield f.reduce_scatter(grad)

    def inference(self, x: GenericTensor) -> GenericTensor:
        return f.all_gather(x)


@final
class AllReduce(NoTensorOp, EnvObliviousOp):
    act: GenericTensor

    def describe_parallellism(self) -> list[ExecutionFlow]:
        return [_all_reduce(self)]

    def training(
        self, x: GenericTensor
    ) -> Generator[GenericTensor, GenericTensor, None]:
        grad = yield f.all_reduce(x)
        yield grad

    def inference(self, x: GenericTensor) -> GenericTensor:
        return f.all_reduce(x)
