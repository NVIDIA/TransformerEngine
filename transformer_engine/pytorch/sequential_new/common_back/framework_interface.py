from __future__ import annotations
from typing import Protocol, Sequence, TypeVar, overload
from .enums import DType


class TensorTypeBase(Protocol):
    @overload
    def view(self, size: Sequence[int], /) -> TensorTypeBase:
        ...

    @overload
    def view(self, *size: int) -> TensorTypeBase:
        ...

    def view(self, *size: int | Sequence[int]) -> TensorTypeBase:
        raise NotImplementedError()

    def __getitem__(self, indices: int | slice | tuple[int | slice]) -> TensorTypeBase:
        raise NotImplementedError()

    def is_contiguous(self) -> bool:
        raise NotImplementedError()


TensorType = TypeVar(
    "TensorType",
    bound=TensorTypeBase,
)


class FrameworkInterface(Protocol[TensorType]):
    Tensor: type[TensorType]

    @staticmethod
    def fi_empty(shape: tuple[int, ...], dtype: DType) -> "Tensor":
        raise NotImplementedError()

    @staticmethod
    def fi_zeros(
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_ones(
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_normal(
        mean: float,
        std: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_uniform(
        min: float,
        max: float,
        shape: tuple[int, ...] | None,
        dtype: DType | None,
        out: "Tensor" | None,
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_relu(x: "Tensor", out: "DType | Tensor") -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_drelu(grad: "Tensor", x: "Tensor", out: "DType | Tensor") -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_gelu(x: "Tensor", out: "DType | Tensor") -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_dgelu(grad: "Tensor", x: "Tensor", out: "DType | Tensor") -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_swiglu(x: "Tensor", out: "DType | Tensor") -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_dswiglu(
        grad: "Tensor", x: "Tensor", out: "DType | Tensor"
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_geglu(x: "Tensor", out: "DType | Tensor") -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_dgeglu(
        grad: "Tensor", x: "Tensor", out: "DType | Tensor"
    ) -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_reglu(x: "Tensor", out: "DType | Tensor") -> "Tensor" | None:
        raise NotImplementedError()

    @staticmethod
    def fi_dreglu(
        grad: "Tensor", x: "Tensor", out: "DType | Tensor"
    ) -> "Tensor" | None:
        raise NotImplementedError()

    def fi_register_buffer(self, name: str, tensor: "Tensor") -> None:
        raise NotImplementedError()


def empty(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
):
    return fi.fi_empty(shape, dtype)


@overload
def zeros(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def zeros(
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def zeros(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    /,
    *,
    out: TensorType | None = None,
):
    return fi.fi_zeros(shape, dtype, out)


@overload
def ones(
    fi: type[FrameworkInterface[TensorType]], shape: tuple[int, ...], dtype: DType, /
) -> TensorType:
    ...


@overload
def ones(
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def ones(
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    out: TensorType | None = None,
):
    return fi.fi_ones(shape, dtype, out)


@overload
def normal(
    mean: float,
    std: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def normal(
    mean: float,
    std: float,
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def normal(
    mean: float,
    std: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    out: TensorType | None = None,
):
    return fi.fi_normal(mean, std, shape, dtype, out)


@overload
def uniform(
    min: float,
    max: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...],
    dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def uniform(
    min: float,
    max: float,
    fi: type[FrameworkInterface[TensorType]],
    /,
    *,
    out: TensorType,
) -> None:
    ...


def uniform(
    min: float,
    max: float,
    fi: type[FrameworkInterface[TensorType]],
    shape: tuple[int, ...] | None = None,
    dtype: DType | None = None,
    out: TensorType | None = None,
):
    return fi.fi_uniform(min, max, shape, dtype, out)


@overload
def relu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out_dtype: DType, /
) -> TensorType:
    ...


@overload
def relu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out: TensorType, /
) -> None:
    ...


def relu(
    fi: type[FrameworkInterface[TensorType]],
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_relu(x, out)


@overload
def drelu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out_dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def drelu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: TensorType,
    /,
) -> None:
    ...


def drelu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_drelu(grad, x, out)


@overload
def gelu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out_dtype: DType, /
) -> TensorType:
    ...


@overload
def gelu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out: TensorType, /
) -> None:
    ...


def gelu(
    fi: type[FrameworkInterface[TensorType]],
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_gelu(x, out)


@overload
def dgelu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out_dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def dgelu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: TensorType,
    /,
) -> None:
    ...


def dgelu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_dgelu(grad, x, out)


@overload
def reglu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out_dtype: DType, /
) -> TensorType:
    ...


@overload
def reglu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out: TensorType, /
) -> None:
    ...


def reglu(
    fi: type[FrameworkInterface[TensorType]],
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_reglu(x, out)


@overload
def dreglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out_dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def dreglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: TensorType,
    /,
) -> None:
    ...


def dreglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_dreglu(grad, x, out)


@overload
def geglu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out_dtype: DType, /
) -> TensorType:
    ...


@overload
def geglu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out: TensorType, /
) -> None:
    ...


def geglu(
    fi: type[FrameworkInterface[TensorType]],
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_geglu(x, out)


@overload
def dgeglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out_dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def dgeglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: TensorType,
    /,
) -> None:
    ...


def dgeglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_dgeglu(grad, x, out)


@overload
def swiglu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out_dtype: DType, /
) -> TensorType:
    ...


@overload
def swiglu(
    fi: type[FrameworkInterface[TensorType]], x: TensorType, out: TensorType, /
) -> None:
    ...


def swiglu(
    fi: type[FrameworkInterface[TensorType]],
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_swiglu(x, out)


@overload
def dswiglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out_dtype: DType,
    /,
) -> TensorType:
    ...


@overload
def dswiglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: TensorType,
    /,
) -> None:
    ...


def dswiglu(
    fi: type[FrameworkInterface[TensorType]],
    grad: TensorType,
    x: TensorType,
    out: DType | TensorType,
):
    return fi.fi_dswiglu(grad, x, out)


class ParamConstructor(Protocol):
    @staticmethod
    def __call__(
        fi: type[FrameworkInterface[TensorType]],
        shape: tuple[int, ...],
        dtype: DType,
        /,
    ) -> TensorType:
        raise NotImplementedError()


class Activation(Protocol):
    @staticmethod
    @overload
    def __call__(
        fi: type[FrameworkInterface[TensorType]],
        x: TensorType,
        out_dtype: DType,
        /,
    ) -> TensorType:
        ...

    @staticmethod
    @overload
    def __call__(
        fi: type[FrameworkInterface[TensorType]],
        x: TensorType,
        out: TensorType,
        /,
    ) -> None:
        ...


class Gradient(Protocol):
    @staticmethod
    @overload
    def __call__(
        fi: type[FrameworkInterface[TensorType]],
        grad: TensorType,
        x: TensorType,
        out_dtype: DType,
        /,
    ) -> TensorType:
        ...

    @staticmethod
    @overload
    def __call__(
        fi: type[FrameworkInterface[TensorType]],
        grad: TensorType,
        x: TensorType,
        out: TensorType,
        /,
    ) -> None:
        ...
