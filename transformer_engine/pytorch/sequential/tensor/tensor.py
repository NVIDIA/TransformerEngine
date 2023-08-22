import torch
from ..cpp_extensions import Tensor as TensorBase, DType
from .printing import tensor_repr


class Tensor(TensorBase):
    _cached_dtype: DType
    _cached_shape: tuple[int, ...]

    def __init__(
        self,
        dtype: DType,
        data: torch.Tensor,
        amax: torch.Tensor,
        scale: torch.Tensor,
        scale_inv: torch.Tensor,
    ):
        self._cached_dtype = dtype
        self._cached_shape = data.shape
        super().__init__(dtype, data, amax, scale, scale_inv)

    @property
    def dtype(self):  # type: ignore[incompatible-override]
        return self._cached_dtype

    @property
    def shape(self):  # type: ignore[incompatible-override]
        return self._cached_shape

    def query_shape_and_dtype_(self):
        self._cached_dtype = super().dtype
        self._cached_shape = tuple(super().shape)
        return self

    def __repr__(self):
        return tensor_repr(self)
