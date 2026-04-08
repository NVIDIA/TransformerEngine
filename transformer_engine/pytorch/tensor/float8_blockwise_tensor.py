# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data quantized with NxN tiles"""
from __future__ import annotations
from collections.abc import Iterable
import math
import warnings
from typing import Any, Optional, Tuple, Union

import torch
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType
from transformer_engine.common.recipe import Float8BlockScaling, Recipe
from .storage.float8_blockwise_tensor_storage import Float8BlockwiseQTensorStorage
from ..quantized_tensor import QuantizedTensor, Quantizer
from ._quantization_helpers import _IdentityFunc
from ..utils import devices_match, round_up_to_nearest_multiple

aten = torch.ops.aten


class Float8BlockQuantizer(Quantizer):
    """Builder class for tensors quantized with current scaling using
    NxN quantization tilings to choose scale.

    This class is typically used to convert a high-precision tensor
    (e.g. in FP32 or BF16) into a quantized tensor (e.g. in FP8).

    """

    dtype: TE_DType
    block_len: int
    amax_epsilon: float
    force_pow_2_scales: bool
    block_scaling_dim: int

    def __init__(
        self,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool,
        columnwise: bool,
        amax_epsilon: float = 0.0,
        force_pow_2_scales: bool = True,
        block_scaling_dim: int = 2,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype
        self.block_len = 128
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon
        self.block_scaling_dim = block_scaling_dim

    def copy(self) -> Float8BlockQuantizer:
        """Create shallow copy"""

        quantizer = Float8BlockQuantizer(
            fp8_dtype=self.dtype,
            rowwise=self.rowwise_usage,
            columnwise=self.columnwise_usage,
            block_scaling_dim=self.block_scaling_dim,
            amax_epsilon=self.amax_epsilon,
            force_pow_2_scales=self.force_pow_2_scales,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm

        return quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Update the quantized tensor with data from the source tensor.

        This method quantizes the input tensor and stores the result in the destination tensor.

        Parameters
        ----------
        src : torch.Tensor
            Source tensor containing the data to be quantized
        dst : QuantizedTensor
            Destination tensor where the quantized data will be stored
        noop_flag : Optional[torch.Tensor]
            Optional flag tensor indicating whether to skip the quantization operation

        Returns
        -------
        QuantizedTensor
            The destination tensor containing the quantized data

        Raises
        ------
        AssertionError
            If the destination tensor is not a Float8BlockwiseQTensor
        """
        assert isinstance(
            dst, Float8BlockwiseQTensor
        ), f"Cannot store quantized blockwise tensor in {type(dst)} type."
        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        dst._fp8_dtype = self.dtype
        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        return tex.quantize(tensor, self)

    def get_scale_shape(self, shape: Iterable[int], columnwise: bool) -> Tuple[int, int]:
        """Scaling tensor shape.

        This method determines the shape of the scaling tensor based
        on the quantizer configuration. The scales are padded to
        multiples of 4 for compatibility with GEMM.

        Parameters
        ----------
        shape : Iterable[int]
            Logical tensor shape.
        columnwise : bool
            Whether the data is scaled column-wise (True) or row-wise (False).

        Returns
        -------
        Tuple[int, int]
            Scaling tensor shape.

        """

        # Flatten tensor to 2D
        dim0 = math.prod(shape[:-1])
        dim1 = shape[-1] if shape else 1

        # Check block dims
        if self.block_scaling_dim not in (1, 2):
            raise RuntimeError(
                "Only 1D or 2D blocks are supported, "
                f"but got block_scaling_dim={self.block_scaling_dim}"
            )

        # 128x128 block scaling
        if self.block_scaling_dim == 2:
            scale_dim0 = (dim0 + self.block_len - 1) // self.block_len
            scale_dim1 = (dim1 + self.block_len - 1) // self.block_len
            if columnwise:
                return (scale_dim1, round_up_to_nearest_multiple(scale_dim0, 4))
            return (scale_dim0, round_up_to_nearest_multiple(scale_dim1, 4))

        # 1x128 block scaling
        if columnwise:
            return (
                (dim0 + self.block_len - 1) // self.block_len,
                round_up_to_nearest_multiple(dim1, 4),
            )
        return (
            (dim1 + self.block_len - 1) // self.block_len,
            round_up_to_nearest_multiple(dim0, 4),
        )

    def get_columnwise_shape(self, shape: Iterable[int]) -> Tuple[int, ...]:
        """Column-wise data shape

        GEMMs expect that the column-wise data is transposed relative
        to the logical tensor shape.

        Parameters
        ----------
        shape : Iterable[int]
            Logical tensor shape.

        Returns
        -------
        Tuple[int, ...]
            Column-wise data shape.
        """
        colwise_shape = []
        if shape:
            colwise_shape.append(shape[-1])
        colwise_shape.extend(shape[:-1])
        return tuple(colwise_shape)

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        shape = inp.size()
        if len(shape) < 2:
            return False
        if shape[-1] % self.block_len != 0:
            return False
        if math.prod(shape[:-1]) % self.block_len != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
        pin_memory: bool = False,
    ) -> Float8BlockwiseQTensor:
        """Construct quantized tensor with uninitialized data"""

        tensor_kwargs = {
            "device": torch.device("cuda") if device is None else device,
            "pin_memory": pin_memory,
        }

        # Allocate buffers for row-scaled data
        rowwise_data = None
        rowwise_scale_inv = None
        if self.rowwise_usage:
            rowwise_data = torch.empty(shape, dtype=torch.uint8, **tensor_kwargs)
            rowwise_scale_inv = torch.empty(
                self.get_scale_shape(shape, columnwise=False),
                dtype=torch.float32,
                **tensor_kwargs,
            )

        # Allocate buffers for column-scaled data
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty(
                self.get_columnwise_shape(shape),
                dtype=torch.uint8,
                **tensor_kwargs,
            )
            columnwise_scale_inv = torch.empty(
                self.get_scale_shape(shape, columnwise=True),
                dtype=torch.float32,
                **tensor_kwargs,
            )

        # Construct FP8 tensor
        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=dtype,
            fp8_dtype=self.dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            quantizer=self,
            is_2D_scaled=self.block_scaling_dim == 2,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # NOTE: This interface is specific to requirements like delayed scaling
        # where state from an estimator influences distribution parameters.
        pass

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return Float8BlockScaling


class Float8BlockwiseQTensor(Float8BlockwiseQTensorStorage, QuantizedTensor):
    """Tensor class with FP8 data quantized via NxN blocks or 1xN blocks.

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    rowwise_data : torch.Tensor
          FP8 data in a uint8 tensor matching shape of dequantized tensor.
    rowwise_scale_inv : torch.Tensor
          FP32 dequantization scales in GEMM format for dequantizing rowwise_data.
    columnwise_data : Optional[torch.Tensor]
          FP8 data in a uint8 tensor matching shape of dequantized tensor transpose.
    columnwise_scale_inv : Optional[torch.Tensor]
          FP32 dequantization scales in GEMM format for dequantizing columnwise_data.

    fp8_dtype : transformer_engine_torch.DType, default = kFloat8E4M3
               FP8 format.
    quantizer : Quantizer - the Float8BlockQuantizer that quantized this tensor and
               holds configuration about quantization and dequantization modes.
    """

    # NOTE: We reorder the *args so that we can instantiate a Float8BlockwiseQTensorStorage with positional args,
    # which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: TE_DType,
        quantizer: Quantizer,
        is_2D_scaled: bool,
        **kwargs,
    ):
        instance = super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            fp8_dtype,
            quantizer,
            is_2D_scaled,
            *args,
            **kwargs,
        )

        return instance

    def __repr__(self, *, tensor_contents=None):
        return (
            f"Float8BlockwiseQTensor(fp8_dtype={self._fp8_dtype},"
            f" is_2D_scaled={self._is_2D_scaled},"
            f" data={self.dequantize()})"
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8BlockwiseQTensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize())
        return super().quantize_(tensor, noop_flag=noop_flag)

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8BlockwiseQTensor

        By default the resulting tensor's dtype is the
        Float8BlockwiseQTensor's pre-quantized dtype.
        """
        if dtype is not None:
            dequant_dtype = dtype
        else:
            dequant_dtype = self.dtype
        return super().dequantize(dtype=dequant_dtype)

    def detach(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return Float8BlockwiseQTensor.make_like(self)

    def clone(self) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        rowwise_data = None
        if self._rowwise_data is not None:
            rowwise_data = self._rowwise_data.detach().clone()
        columnwise_data = None
        if self._columnwise_data is not None:
            columnwise_data = self._columnwise_data.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "rowwise_data": rowwise_data,
                "columnwise_data": columnwise_data,
            },
        )

    def view(self, *shape: Tuple[int]) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def untyped_storage(self) -> torch.UntypedStorage:
        """Return the underlying UntypedStorage of the FP8 data.

        Note that FP8 block-scaled tensor may involve multiple
        buffers: row-wise FP8 data, row-wise scales, column-wise FP8
        data, column-wise scales. The UntypedStorage of the row-wise
        FP8 data is returned if it exists, and otherwise the
        UntypedStorage of the column-wise FP8 data.

        """
        data = self._rowwise_data if self._rowwise_data is not None else self._columnwise_data
        if data is not None:
            return data.untyped_storage()
        return torch.UntypedStorage(0, device=self.device)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):

        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            if data is None:
                # Columnwise data only.
                super().__torch_dispatch__(func, types, args, kwargs)
            orig_size = data.size()
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            if orig_size != out_data.size():
                raise NotImplementedError(
                    "Changing shape with view not implemented "
                    " (scales and columnwise data untouched)."
                )
            return Float8BlockwiseQTensor.make_like(tensor)

        # as_strided op — applied by FSDP2 on the unsharded param.
        # When shape and strides match (no-op), return self to preserve the quantized type.
        # If shape differs (e.g. padding needed), fall through to dequantize.
        if func == aten.as_strided.default:
            tensor = args[0]
            shape = args[1]
            strides = args[2]
            if (
                len(shape) == len(strides) == 2
                and tuple(strides) == (shape[-1], 1)
                and tuple(shape) == tuple(tensor.size())
            ):
                return Float8BlockwiseQTensor.make_like(tensor)

        # slice op — applied by FSDP2 when shards need unpadding.
        # When the slice is a no-op (covers entire dimension), return self.
        if func == aten.slice.Tensor:
            tensor = args[0]
            dim = args[1]
            start = args[2]
            length = args[3]
            if start == 0 and length == tensor.size(dim):
                return Float8BlockwiseQTensor.make_like(tensor)

        # record stream op
        if func == torch.ops.aten.record_stream.default:
            qt, stream = args
            for t in (
                qt._rowwise_data,
                qt._columnwise_data,
                qt._rowwise_scale_inv,
                qt._columnwise_scale_inv,
            ):
                if t is not None and t.is_cuda:
                    t.record_stream(stream)
            return None

        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8BlockwiseQTensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if (
            self._rowwise_data is not None
            and self._rowwise_data.is_contiguous(memory_format=memory_format)
            and (
                (self._columnwise_data is None)
                or (self._columnwise_data.is_contiguous(memory_format=memory_format))
            )
        ):
            return self
        raise ValueError("Float8BlockwiseQTensor does not support different memory formats!")

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        dtype: torch.dtype,
        quantizer: Quantizer,
        is_2D_scaled: bool,
        data_format: Any = None,  # pylint: disable=unused-argument
    ) -> Float8BlockwiseQTensor:
        """Build Float8BlockwiseQTensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return Float8BlockwiseQTensor(
            shape=shape,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            fp8_dtype=fp8_dtype,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            dtype=dtype,
            quantizer=quantizer,
            is_2D_scaled=is_2D_scaled,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8BlockwiseQTensor._make_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
                self._quantizer,
                self._is_2D_scaled,
                None,  # data_format
            ),
        )

    def _get_data(self) -> Float8BlockwiseQTensor:
        """Get tensor data property"""
        return self

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a Float8BlockwiseQTensor. Otherwise
        casts to FP8.

        """
        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device

        def _set_from_tensor(dst: Float8BlockwiseQTensor, src: Float8BlockwiseQTensor):
            dst._rowwise_data = src._rowwise_data
            dst._columnwise_data = src._columnwise_data
            dst._quantizer = src._quantizer.copy()
            dst._fp8_dtype = src._fp8_dtype
            dst._rowwise_scale_inv = src._rowwise_scale_inv
            dst._columnwise_scale_inv = src._columnwise_scale_inv

        # Check that tensor dimensions match
        if (
            self.size() != tensor.size()
            or self.stride() != tensor.stride()
            or self.layout != tensor.layout
        ):
            raise ValueError("Invalid tensor for updating Float8BlockwiseQTensor data")

        # Just copy FP8 data if other tensor is Float8BlockwiseQTensor
        if (
            isinstance(tensor, Float8BlockwiseQTensor)
            and self.storage_offset() == tensor.storage_offset()
            and devices_match(self.device, new_device)
        ):
            _set_from_tensor(self, tensor)
            return

        if isinstance(tensor, Float8BlockwiseQTensor):
            assert tensor._quantizer is not None, "Can't quantize without a quantizer"
            quantizer = tensor._quantizer
        else:
            assert self._quantizer is not None, "Can't quantize without a quantizer"
            quantizer = self._quantizer

        # Quantize to FP8
        quantizer.update_quantized(tensor, self)

    # Cast to FP8 when setting Float8BlockwiseQTensor.data
    data = property(_get_data, _set_data)

    @property
    def shape(self):
        """Return the shape of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.shape
        if self._columnwise_data is not None:
            return self._columnwise_data.shape
        return torch.Tensor.size(self)

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA device."""
        if self._rowwise_data is not None:
            return self._rowwise_data.is_cuda
        if self._columnwise_data is not None:
            return self._columnwise_data.is_cuda
        raise RuntimeError("Float8BlockwiseQTensor has no data!")

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Called by FSDP2 before all-gather of weights for forward and backward passes.

        Args:
            mesh: DeviceMesh used by FSDP2 to shard the weights.
            orig_size: Original size of the weight tensor.
            contiguous_orig_stride: Original stride of the weight tensor.
            module: FSDP-wrapped module containing this tensor.
            mp_policy: Mixed precision policy used by FSDP2.

        Returns:
            sharded_tensors: Tuple of tensors to be all-gathered.
            metadata: Metadata needed for reconstructing the tensor after all-gather.
        """
        # pylint: disable=unused-argument
        # PyTorch FSDP2 private API – tested with PyTorch 2.5+;
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from transformer_engine.pytorch.distributed import _get_module_fsdp_state

        if not self._is_2D_scaled:
            raise NotImplementedError(
                "FSDP2 is only supported for Float8BlockwiseQTensors with 2D block scaling "
                "(block_scaling_dim=2). 1D block scaling is not supported because the scale "
                "layout has M in dim1, which is incompatible with FSDP2 dim0 all-gather."
            )

        if self._rowwise_data is None or self._rowwise_scale_inv is None:
            raise RuntimeError(
                "Rowwise data must be available for FSDP2 all-gather with 2D block scaling."
            )

        fsdp_state = _get_module_fsdp_state(module)
        param_group = fsdp_state._fsdp_param_group
        if param_group is None:
            raise RuntimeError(
                "FSDP state for this module has no parameter group; "
                "cannot determine reshard_after_forward."
            )
        reshard_after_forward = param_group._reshard_after_forward

        # If weights are resharded after forward pass, only the relevant usage
        # is needed based on whether it's a forward or backward pass.
        # If not resharded, the same all-gathered weights are reused in backward,
        # so both usages may be needed.
        if reshard_after_forward:
            training_state = param_group._training_state
            is_backward_pass = training_state == TrainingState.PRE_BACKWARD
            rowwise_usage = not is_backward_pass
            columnwise_usage = is_backward_pass
        else:
            rowwise_usage = True
            columnwise_usage = self._quantizer.columnwise_usage

        # For 2D block scaling (128x128 blocks), columnwise data and scales are
        # the transpose of rowwise data and scales. Only all-gather the rowwise
        # tensors; columnwise will be derived locally via _create_columnwise()
        # in post_all_gather, halving all-gather communication volume.
        sharded_tensors = (self._rowwise_data, self._rowwise_scale_inv)
        metadata = (self._fp8_dtype, self._is_2D_scaled, rowwise_usage, columnwise_usage)
        return sharded_tensors, metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Float8BlockwiseQTensor] = None,
    ):
        """Called by FSDP2 after all-gather of weights for forward and backward passes.

        Args:
            all_gather_outputs: All-gathered tensors from fsdp_pre_all_gather.
            metadata: Metadata from fsdp_pre_all_gather.
            param_dtype: High-precision dtype of the tensor.
            out: Existing tensor to update in-place (None on first iteration).

        Returns:
            Tuple of (Float8BlockwiseQTensor, all_gather_outputs).
        """
        fp8_dtype, is_2D_scaled, rowwise_usage, columnwise_usage = metadata

        # Only rowwise data+scales were all-gathered (columnwise is derived locally).
        rowwise_data, rowwise_scale_inv = all_gather_outputs[:2]
        data_shape = rowwise_data.shape

        if out is not None:
            out._rowwise_data = rowwise_data
            out._rowwise_scale_inv = rowwise_scale_inv
        else:
            out = Float8BlockwiseQTensor(
                shape=data_shape,
                dtype=param_dtype,
                fp8_dtype=fp8_dtype,
                rowwise_data=rowwise_data,
                rowwise_scale_inv=rowwise_scale_inv,
                columnwise_data=None,
                columnwise_scale_inv=None,
                quantizer=self._quantizer,
                is_2D_scaled=is_2D_scaled,
            )

        # For 2D block scaling, derive columnwise data and scales from rowwise
        # via local fp8 transpose.
        if columnwise_usage:
            out._create_columnwise()
        # remove usages if not needed.
        out.update_usage(
            rowwise_usage=rowwise_usage,
            columnwise_usage=columnwise_usage,
        )
        out._quantizer.set_usage(rowwise=rowwise_usage, columnwise=columnwise_usage)
        return out, all_gather_outputs


class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8BlockwiseQTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[list[int]] = None,
    ) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(ctx.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        if tensor._is_2D_scaled:
            # For the case of 2D scaled tensor, the last 2 dimensions should not change
            if shape[-1] != ctx.shape[-1] or shape[-2] != ctx.shape[-2]:
                warnings.warn(
                    "2D scaled Float8BlockwiseQTensor does not support view "
                    "the last 2 dimensions "
                    f"(attempted to view dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().view(*shape)
        else:
            # For the case of 1D scaled tensor, the last dimension should not change
            if shape[-1] != ctx.shape[-1]:
                warnings.warn(
                    "1D scaled Float8BlockwiseQTensor does not support view "
                    "the last dimension "
                    f"(attempted to view dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().view(*shape)

        if list(shape) == list(tensor.shape):
            return tensor

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=tensor.dtype,
            fp8_dtype=tensor._fp8_dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            quantizer=tensor._quantizer,
            is_2D_scaled=tensor._is_2D_scaled,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            new_data = (
                grad._rowwise_data.view(*ctx.shape) if grad._rowwise_data is not None else None
            )
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            else:
                new_columnwise_data = None
            dgrad = Float8BlockwiseQTensor(
                shape=ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
                is_2D_scaled=grad._is_2D_scaled,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8BlockwiseQTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8BlockwiseQTensor,
        shape: Optional[list[int]] = None,
    ) -> Float8BlockwiseQTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        ctx.shape = tensor.shape
        if shape is None:
            return tensor

        # Canonicalize shape
        if not isinstance(shape, Iterable):
            shape = [shape]
        elif len(shape) == 1 and isinstance(shape[0], Iterable):
            shape = shape[0]
        if -1 in shape:
            shape = list(shape)
            d_inferred = -math.prod(tensor.shape) // math.prod(shape)
            for i, d in enumerate(shape):
                if d == -1:
                    shape[i] = d_inferred
                    break

        if tensor._is_2D_scaled:
            # For the case of 2D scaled tensor, the last 2 dimensions should not change
            if shape[-1] != ctx.shape[-1] or shape[-2] != ctx.shape[-2]:
                warnings.warn(
                    "2D scaled Float8BlockwiseQTensor does not support reshaping "
                    "the last 2 dimensions "
                    f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().reshape(*shape)
        else:
            # For the case of 1D scaled tensor, the last dimension should not change
            if shape[-1] != ctx.shape[-1]:
                warnings.warn(
                    "1D scaled Float8BlockwiseQTensor does not support reshaping "
                    "the last dimension "
                    f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)}). "
                    "If you are using this for FSDP2 without compiled_autograd_enabled, "
                    "then ignore this warning since this view is not going to be used anywhere.",
                    stacklevel=2,
                )
                return tensor.dequantize().reshape(*shape)
        if list(shape) == list(tensor.shape):
            return tensor

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)
        if tensor._columnwise_data is not None:
            columnwise_shape = [shape[-1]] + list(shape[:-1])
            new_columnwise_data = tensor._columnwise_data.view(columnwise_shape)

        return Float8BlockwiseQTensor(
            shape=shape,
            dtype=tensor.dtype,
            fp8_dtype=tensor._fp8_dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            quantizer=tensor._quantizer,
            is_2D_scaled=tensor._is_2D_scaled,
            requires_grad=tensor.requires_grad,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, Float8BlockwiseQTensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*ctx.shape)
            if grad._columnwise_data is not None:
                columnwise_shape = [ctx.shape[-1]] + list(ctx.shape[:-1])
                new_columnwise_data = grad._columnwise_data.view(columnwise_shape)
            dgrad = Float8BlockwiseQTensor(
                shape=ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
                is_2D_scaled=grad._is_2D_scaled,
                requires_grad=grad.requires_grad,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
