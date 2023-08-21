import torch
from ._nvte import Tensor as TensorBase, DType
from .dtype import dtype_name


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
        self.query_shape_and_dtype_()
        data_repr = repr(self.data)
        data_repr = data_repr[::-1][data_repr[::-1].find("]") :][::-1]
        data_repr = "T" + data_repr[1:]
        return f"""\
{data_repr},
       dtype = {dtype_name(self.dtype)},\
amax = {self.amax[0].item() if self.amax.numel() else None},\
scale = {self.scale.item() if self.scale.numel() else None},\
scale_inv = {self.scale_inv.item() if self.scale_inv.numel() else None}\
)"""
