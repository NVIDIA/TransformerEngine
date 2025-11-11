# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with MXFP8 data"""
from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional, Tuple, Union, Any
import warnings

import torch
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import MXFP8BlockScaling, Recipe
from ..constants import MXFP8_BLOCK_SCALING_SIZE
from ..utils import devices_match, round_up_to_nearest_multiple
from .storage.mxfp8_tensor_storage import MXFP8TensorStorage, _FromMXFP8Func
from ..quantized_tensor import QuantizedTensor, Quantizer
from ._quantization_helpers import _IdentityFunc

aten = torch.ops.aten


class MXFP8Quantizer(Quantizer):
    """Builder class for FP8 tensors with MX block scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized by
    dividing them into groups of 32 elements, each scaled and cast
    separately using current data.

    """

    dtype: TE_DType

    def __init__(
        self,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = fp8_dtype

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:

        assert isinstance(dst, MXFP8Tensor), f"Cannot store quantized MXFP8 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        # Update FP8 dtype
        dst._fp8_dtype = self.dtype

        return dst

    def quantize_impl(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Quantize tensor implementation"""
        return tex.quantize(tensor, self)

    def is_quantizable(self, inp: torch.Tensor) -> bool:
        """Returns whether or not given inp can be quantized"""
        if inp.ndim < 2:
            return False
        if inp.shape[-1] % MXFP8_BLOCK_SCALING_SIZE != 0:
            return False
        if math.prod(inp.shape[:-1]) % MXFP8_BLOCK_SCALING_SIZE != 0:
            return False
        return True

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> MXFP8Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        assert (
            shape[-1] % MXFP8_BLOCK_SCALING_SIZE == 0
            and math.prod(shape[:-1]) % MXFP8_BLOCK_SCALING_SIZE == 0
        ), (
            f"Incorrect shape {shape} for MXFP8. Tensor dims must be divisible by"
            f" {MXFP8_BLOCK_SCALING_SIZE}"
        )

        # Allocate FP8 data
        data = torch.empty(shape, dtype=torch.uint8, device=device)
        scale_inv = torch.empty(
            round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),
            round_up_to_nearest_multiple(shape[-1] // MXFP8_BLOCK_SCALING_SIZE, 4),
            dtype=torch.uint8,
            device=device,
        )

        # Allocate FP8 data transpose if needed
        columnwise_data = None
        columnwise_scale_inv = None
        if self.columnwise_usage:
            columnwise_data = torch.empty_like(data)
            columnwise_scale_inv = torch.empty(
                round_up_to_nearest_multiple(math.prod(shape[:-1]) // MXFP8_BLOCK_SCALING_SIZE, 4),
                round_up_to_nearest_multiple(shape[-1], 128),
                dtype=torch.uint8,
                device=device,
            )

        # Construct FP8 tensor
        return MXFP8Tensor(
            shape=shape,
            dtype=dtype,
            fp8_dtype=self.dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            quantizer=self,
            requires_grad=requires_grad,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # TODO(ksivamani): No calibration needed for mxfp8?
        pass

    def create_tensor_from_data(
        self,
        data: torch.Tensor,
        scale_inv: torch.Tensor,
        fake_dtype: torch.dtype,
        fp8_dtype: TE_DType = tex.DType.kFloat8E4M3,
    ) -> MXFP8Tensor:
        """Create a new MXFP8Tensor from data and scale_inv."""
        return MXFP8Tensor(
            shape=data.shape,
            dtype=fake_dtype,
            rowwise_data=data,
            rowwise_scale_inv=scale_inv,
            columnwise_data=None,
            columnwise_scale_inv=None,
            fp8_dtype=fp8_dtype,
            quantizer=self,
        )

    def onnx_quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        if tensor.dtype != torch.float32:
            tensor = tensor.to(dtype=torch.float32)
        data, scale_inv = torch.ops.tex.mxfp8_quantize(tensor)
        return self.create_tensor_from_data(data, scale_inv, fake_dtype=torch.float32)

    def onnx_dequantize(self, tensor: Union[MXFP8TensorStorage, MXFP8Tensor]) -> torch.Tensor:
        return torch.ops.tex.mxfp8_dequantize(tensor._rowwise_data, tensor._rowwise_scale_inv)

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return MXFP8BlockScaling


class MXFP8Tensor(MXFP8TensorStorage, QuantizedTensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    data: torch.Tensor
          Raw FP8 data in a uint8 tensor
    fp8_dtype: transformer_engine_torch.DType, default = kFloat8E4M3
               FP8 format.
    fp8_scale_inv: torch.Tensor
                   Reciprocal of the scaling factor applied when
                   casting to FP8, i.e. the scaling factor that must
                   be applied when casting from FP8 to higher
                   precision.
    dtype: torch.dtype, default = torch.float32
           Nominal tensor datatype.

    """

    # NOTE: We reorder the *args so that we can instantiate a MXFP8TensorStorage with positional args,
    # which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: TE_DType,
        quantizer: Optional[Quantizer],
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
            *args,
            **kwargs,
        )
        return instance

    def __repr__(self, *, tensor_contents=None):
        return f"MXFP8Tensor(fp8_dtype={self._fp8_dtype}, data={self.dequantize(dtype=self.dtype)})"

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from MXFP8Tensor

        By default the resulting tensor's dtype is the
        MXFP8Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromMXFP8Func.apply(self, dtype)
        return _FromMXFP8Func.forward(None, self, dtype)

    def _build_default_quantizer(self) -> Optional[Quantizer]:
        """Build default quantizer for the tensor"""
        return MXFP8Quantizer(fp8_dtype=self._fp8_dtype)

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> MXFP8Tensor:
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

    def detach(self) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        # TODO(ksivamani): Fix the detach bug
        return MXFP8Tensor.make_like(self)

    def clone(self) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        assert self._rowwise_data is not None
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

    def view(self, *shape: Tuple[int]) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> MXFP8Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> MXFP8Tensor:
        """Returns tensor with data in provided memory format
        Returns `self` if data is already in correct memory format.

        """
        if self._rowwise_data is not None and self._rowwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        if self._columnwise_data is not None and self._columnwise_data.is_contiguous(
            memory_format=memory_format
        ):
            return self
        raise ValueError("MXFP8Tensor does not support different memory formats!")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # View op
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._rowwise_data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            return MXFP8Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                rowwise_data=out_data,
                rowwise_scale_inv=tensor._rowwise_scale_inv,
                columnwise_data=tensor._columnwise_data,
                columnwise_scale_inv=tensor._columnwise_scale_inv,
                quantizer=tensor._quantizer,
                requires_grad=False,
                fp8_dtype=tensor._fp8_dtype,
            )

        if func == torch.ops.aten.copy_.default:
            dst, src = args[0], args[1]
            if isinstance(src, MXFP8Tensor) and isinstance(dst, MXFP8Tensor):
                # Booleans to check if src has all the usages that dst needs to respect dst quantizer usages.
                # If not, default to base class behavior.
                rowwise_matches = src._rowwise_data is not None or dst._rowwise_data is None
                columnwise_matches = (
                    src._columnwise_data is not None or dst._columnwise_data is None
                )
                if rowwise_matches and columnwise_matches:
                    if dst._rowwise_data is not None:
                        dst._rowwise_data.copy_(src._rowwise_data.detach())
                        dst._rowwise_scale_inv.copy_(src._rowwise_scale_inv.detach())
                    if dst._columnwise_data is not None:
                        dst._columnwise_data.copy_(src._columnwise_data.detach())
                        dst._columnwise_scale_inv.copy_(src._columnwise_scale_inv.detach())
                    return dst

        # FSDP2 related functions.
        if func == aten.split.Tensor:
            # This is called if entire model is initialized on CUDA device and
            # then splitted. Finally the shard needed by the process is used
            # and other splitted shards are discarded.
            if "dim" in kwargs:
                dim_to_split = kwargs["dim"]
            else:
                dim_to_split = args[2] if len(args) > 2 else 0
            tensor = args[0]
            split_size = args[1]
            dim0_size = tensor.size(0)
            dimlast_size = math.prod(tensor.shape[1:])
            if (
                dim0_size % split_size != 0
                or dim_to_split != 0
                or split_size % MXFP8_BLOCK_SCALING_SIZE != 0
                or dimlast_size % MXFP8_BLOCK_SCALING_SIZE != 0
            ):
                # Handle splitting by dequantizing and splitting the hp tensor
                return super().__torch_dispatch__(func, types, args, kwargs)

            out_data = []
            for data in [tensor._rowwise_data, tensor._columnwise_data]:
                func_out = (
                    data.__torch_dispatch__(
                        func,
                        types,
                        [data] + list(args[1:]),
                        kwargs,
                    )
                    if data is not None
                    else None
                )
                out_data.append(func_out)

            scale_invs = [tensor._rowwise_scale_inv, tensor._columnwise_scale_inv]
            split_sizes_for_scale = [split_size, split_size // MXFP8_BLOCK_SCALING_SIZE]
            # Padding requirements: rowwise dim0 should be divisble by 128, columnwise dim0 should be divisble by 4
            padding_multiples = [128, 4]
            for scale_inv, scale_split_size, pad_multiple in zip(
                scale_invs, split_sizes_for_scale, padding_multiples
            ):
                scale_inv_out = (
                    scale_inv.__torch_dispatch__(
                        func,
                        types,
                        [scale_inv, scale_split_size] + list(args[2:]),
                        kwargs,
                    )
                    if scale_inv is not None
                    else None
                )
                # Pad scale_inv_out to be a multiple of pad_multiple
                if scale_inv_out is not None:
                    current_shape = scale_inv_out.shape
                    pad_dim0 = (pad_multiple - current_shape[0] % pad_multiple) % pad_multiple
                    if pad_dim0 > 0:
                        scale_inv_out = torch.nn.functional.pad(scale_inv_out, (0, 0, 0, pad_dim0))

                out_data.append(scale_inv_out)
            return [
                MXFP8Tensor(
                    shape=(
                        splitted_tensor_data[0].size()
                        if splitted_tensor_data[0] is not None
                        else splitted_tensor_data[1].size()
                    ),
                    dtype=tensor.dtype,
                    rowwise_data=splitted_tensor_data[0],
                    rowwise_scale_inv=splitted_tensor_data[2],
                    columnwise_data=splitted_tensor_data[1],
                    columnwise_scale_inv=splitted_tensor_data[3],
                    quantizer=tensor._quantizer,
                    requires_grad=False,
                    fp8_dtype=tensor._fp8_dtype,
                )
                for splitted_tensor_data in zip(*out_data)
            ]
        if func == torch.ops.aten.as_strided.default:
            # Applied on unsharded param in FSDP2. In our case, this should be a no-op
            # This is needed for the case where some MXFP8 shards need padding i.e dimension 0
            # of the unsharded param is not a multiple of the world size. If that is the case,
            # we down the dequantization route and weights are allgathered in high precision.
            # If weight doesnt need padding, this is just a no-op.
            shape = args[1]
            strides = args[2]
            tensor = args[0]
            if (
                len(shape) != 2
                or len(strides) != 2
                or strides[1] != 1
                or shape[0] != tensor.shape[0]
                or shape[1] != tensor.shape[1]
            ):
                return super().__torch_dispatch__(func, types, args, kwargs)

            return MXFP8Tensor.make_like(tensor)

        if func == aten.slice.Tensor:
            # FSDP2 needed function.
            # We need slicing for the case where some MXFP8 weight shards need padding i.e dimension 0
            # of the unsharded param is not a multiple of the world size. If that is the case,
            # we down the dequantization route and weights are allgathered in high precision instead.
            # If sharded weight doesnt have padding, this is just a no-op.
            dim = args[1]
            start = args[2]
            length = args[3]
            tensor = args[0]
            if (
                dim != 0
                or length != tensor.shape[0]
                or start != 0
                or length % MXFP8_BLOCK_SCALING_SIZE != 0
                or start % MXFP8_BLOCK_SCALING_SIZE != 0
            ):
                return super().__torch_dispatch__(func, types, args, kwargs)
            return MXFP8Tensor.make_like(tensor)

        if func == aten.new_zeros.default:
            rowwise_data = None
            columnwise_data = None
            rowwise_scale_inv = None
            columnwise_scale_inv = None
            tensor = args[0]
            shape = args[1]
            first_dim = math.prod(shape[:-1])
            last_dim = shape[-1]
            if (
                first_dim % MXFP8_BLOCK_SCALING_SIZE != 0
                or last_dim % MXFP8_BLOCK_SCALING_SIZE != 0
            ):
                return super().__torch_dispatch__(func, types, args, kwargs)
            rowwise_scale_inv_shape = [first_dim, last_dim // MXFP8_BLOCK_SCALING_SIZE]
            columnwise_scale_inv_shape = [
                first_dim // MXFP8_BLOCK_SCALING_SIZE,
                last_dim,
            ]
            if tensor._rowwise_data is not None:
                rowwise_data = tensor._rowwise_data.__torch_dispatch__(
                    func,
                    types,
                    [tensor._rowwise_data] + list(args[1:]),
                    kwargs,
                )
                rowwise_scale_inv = tensor._rowwise_scale_inv.__torch_dispatch__(
                    func,
                    types,
                    [tensor._rowwise_scale_inv, rowwise_scale_inv_shape] + list(args[2:]),
                    kwargs,
                )
            if tensor._columnwise_data is not None:
                columnwise_data = tensor._columnwise_data.__torch_dispatch__(
                    func,
                    types,
                    [tensor._columnwise_data] + list(args[1:]),
                    kwargs,
                )
                columnwise_scale_inv = tensor._columnwise_scale_inv.__torch_dispatch__(
                    func,
                    types,
                    [tensor._columnwise_scale_inv, columnwise_scale_inv_shape] + list(args[2:]),
                    kwargs,
                )
            return MXFP8Tensor(
                shape=args[1],
                dtype=tensor.dtype,
                rowwise_data=rowwise_data,
                rowwise_scale_inv=rowwise_scale_inv,
                columnwise_data=columnwise_data,
                columnwise_scale_inv=columnwise_scale_inv,
                quantizer=tensor._quantizer.copy(),
                requires_grad=False,
                fp8_dtype=tensor._fp8_dtype,
            )
        # Default case
        return super().__torch_dispatch__(func, types, args, kwargs)

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Functions FSDP2 calls before all-gather of the
        weights for both forward and backward passes.
        Args:
            mesh (torch.distributed.DeviceMesh): DeviceMesh used by FSDP2
            to shard the weights.
            orig_size (torch.Size): Original size of the weight tensor.(For us same as self.shape)
            contiguous_orig_stride (Tuple[int]): Original stride of the weight tensor
            (For us same as self.stride()).
            module (FSDPModule): FSDP module. FSDP wrapped module wrapped using fully_shard
            that contains this MXFP8 tensor.
            mp_policy (MixedPrecisionPolicy): Mixed precision policy used by FSDP2.

        Returns:
            sharded_tensors: Tuple[torch.Tensor, ...]: Tuple of tensors
            that need to be all-gathered.
            metadata: Tuple[Any]: Metadata needed for reconstructing the
            MXFP8Tensor after all-gather.
        """
        # pylint: disable=unused-argument
        from transformer_engine.pytorch.distributed import _get_module_fsdp_state

        fsdp_state = _get_module_fsdp_state(module)
        reshard_after_forward = fsdp_state._fsdp_param_group._reshard_after_forward
        quantizer = self._quantizer.copy()
        # Remove padding from scale inverses before allgather
        # Rowwise scale_inv should be divisible by [128,4], columnwise by [4, 128]
        rowwise_scale_inv = self._rowwise_scale_inv
        columnwise_scale_inv = self._columnwise_scale_inv
        shape = self.shape
        if rowwise_scale_inv is not None:
            # Remove padding from rowwise scale_inv
            flattened_in_shape0 = math.prod(shape[:-1])
            if rowwise_scale_inv.size(0) != flattened_in_shape0:
                rowwise_scale_inv = rowwise_scale_inv[:flattened_in_shape0]

        if columnwise_scale_inv is not None:
            # Remove padding from columnwise scale_inv
            flattened_in_shape0 = math.prod(shape[:-1]) // MXFP8_BLOCK_SCALING_SIZE
            if columnwise_scale_inv.size(0) != flattened_in_shape0:
                columnwise_scale_inv = columnwise_scale_inv[:flattened_in_shape0]

        sharded_tensors = (self._rowwise_data, rowwise_scale_inv)
        # If weights are resharded after forward pass, then its enough to set the quantizer usages
        # based on whether its forward or backward pass for the allgathered weights.
        # If not resharded after forward pass, the same weights allgathered in forward
        # are used again in backward. And hence if we need the columnwise data/scale_inv,
        # we need to send them as well for allgather in forward pass itself.
        if reshard_after_forward:
            training_state = fsdp_state._fsdp_param_group._training_state
            is_backward_pass = training_state == TrainingState.PRE_BACKWARD
            # Allgather only the necessary tensors based on forward/backward pass
            quantizer.set_usage(rowwise=not is_backward_pass, columnwise=is_backward_pass)
            sharded_tensors = (
                (self._columnwise_data, columnwise_scale_inv)
                if is_backward_pass
                else sharded_tensors
            )
        else:
            if quantizer.columnwise_usage:
                # If weights are not resharded after forward, then both
                # rowwise and columnwise data/scale_inv need to be allgathered.
                sharded_tensors += (self._columnwise_data, columnwise_scale_inv)
        metadata = (self._fp8_dtype, quantizer)
        return sharded_tensors, metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[MXFP8Tensor] = None,
    ):
        """Functions FSDP2 calls after all-gather of the
        weights for both forward and backward passes.
        Args:
            all_gather_outputs (Tuple[torch.Tensor, ...]): sharded_tensors sent out in fsdp_pre_all_gather from each rank
            are all-gathered and received here as a tuple.
            metadata (Any): metadata sent out in fsdp_pre_all_gather used for reconstructing the MXFP8Tensor.
            param_dtype (torch.dtype): high precision dtype of the MXFP8Tensor.
            out (Optional[torch.Tensor], optional): _description_. Defaults to None.
        Returns:
            Tuple[MXFP8Tensor, Tuple[torch.Tensor, ...]]: Allgathered MXFP8Tensor and tuple of internal tensors
            used by the MXFP8Tensor that was being computed after allgather.
        """
        fp8_dtype, quantizer = metadata
        rowwise_data, rowwise_scale_inv = (
            all_gather_outputs[:2] if quantizer.rowwise_usage else (None, None)
        )
        columnwise_data, columnwise_scale_inv = (
            all_gather_outputs[-2:] if quantizer.columnwise_usage else (None, None)
        )

        # Add padding to scale_inv tensors to be multiples of [128, 4]for rowwise and [4, 128] for columnwise
        if rowwise_scale_inv is not None:
            # Pad rowwise_scale_inv to be a multiple of [128, 4]
            current_shape = rowwise_scale_inv.shape
            pad_dim0 = (128 - current_shape[0] % 128) % 128
            if pad_dim0 > 0:
                rowwise_scale_inv = torch.nn.functional.pad(rowwise_scale_inv, (0, 0, 0, pad_dim0))

        if columnwise_scale_inv is not None:
            # Pad columnwise_scale_inv to be a multiple of [4, 128]
            current_shape = columnwise_scale_inv.shape
            pad_dim0 = (4 - current_shape[0] % 4) % 4
            if pad_dim0 > 0:
                columnwise_scale_inv = torch.nn.functional.pad(
                    columnwise_scale_inv, (0, 0, 0, pad_dim0)
                )

        if out is not None:
            out._rowwise_data = rowwise_data
            out._rowwise_scale_inv = rowwise_scale_inv
            out._columnwise_data = columnwise_data
            out._columnwise_scale_inv = columnwise_scale_inv
            out._quantizer = quantizer
        else:
            out = MXFP8Tensor(
                rowwise_data=rowwise_data,
                rowwise_scale_inv=rowwise_scale_inv,
                columnwise_data=columnwise_data,
                columnwise_scale_inv=columnwise_scale_inv,
                fp8_dtype=fp8_dtype,
                dtype=param_dtype,
                shape=rowwise_data.shape if rowwise_data is not None else columnwise_data.shape,
                quantizer=quantizer,
            )

        return out, all_gather_outputs

    @classmethod
    def _make_in_reduce_ex(
        cls,
        rowwise_data: torch.Tensor,
        rowwise_scale_inv: torch.Tensor,
        columnwise_data: torch.Tensor,
        columnwise_scale_inv: torch.Tensor,
        fp8_dtype: TE_DType,
        dtype: torch.dtype,
        shape: torch.shape,
        quantizer: Optional[Quantizer] = None,
    ) -> MXFP8Tensor:
        """Build MXFP8Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return MXFP8Tensor(
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            fp8_dtype=fp8_dtype,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            dtype=dtype,
            shape=shape,
            quantizer=quantizer,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            MXFP8Tensor._make_in_reduce_ex,
            (
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._fp8_dtype,
                self.dtype,
                self.shape,
                self._quantizer,
            ),
        )

    def _get_data(self) -> MXFP8Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a MXFP8Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP8 data if other tensor is MXFP8Tensor
        if isinstance(tensor, MXFP8Tensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    MXFP8Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(MXFP8Tensor, type(self)).data.__set__(self, dummy_tensor)
            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer.copy()
            self._fp8_dtype = tensor._fp8_dtype
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.internal = False
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting MXFP8Tensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the MXFP8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP8Tensor,
        shape: Optional[list[int]] = None,
    ) -> MXFP8Tensor:
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
        if shape[-1] != ctx.shape[-1]:
            warnings.warn(
                "MXFP8Tensor does not support reshaping inner dimension. "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
                "If you are using this for FSDP2 without compiled_autograd_enabled,"
                "then ignore this warning. Since this view is not going to be used anywhere. ",
                stacklevel=2,
            )
            return tensor.dequantize().view(*shape)

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.view(*shape)
        if tensor._columnwise_data is not None:
            new_columnwise_data = tensor._columnwise_data.view(*shape)
        return MXFP8Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP8Tensor):
            new_data = (
                grad._rowwise_data.view(*ctx.shape) if grad._rowwise_data is not None else None
            )
            if grad._columnwise_data is not None:
                new_columnwise_data = grad._columnwise_data.view(*ctx.shape)
            else:
                new_columnwise_data = None
            dgrad = MXFP8Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the MXFP8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: MXFP8Tensor,
        shape: Optional[list[int]] = None,
    ) -> MXFP8Tensor:
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
        if shape[-1] != ctx.shape[-1]:
            raise RuntimeError(
                "MXFP8Tensor does not support reshaping inner dimension "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
            )

        # Construct new tensor if shape is provided
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            new_rowwise_data = tensor._rowwise_data.reshape(*shape)
        if tensor._columnwise_data is not None:
            new_columnwise_data = tensor._columnwise_data.view(*shape)

        return MXFP8Tensor(
            shape,
            tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, MXFP8Tensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                new_rowwise_data = grad._rowwise_data.view(*ctx.shape)
            if grad._columnwise_data is not None:
                new_columnwise_data = grad._columnwise_data.view(*ctx.shape)
            dgrad = MXFP8Tensor(
                ctx.shape,
                grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                fp8_dtype=grad._fp8_dtype,
                quantizer=grad._quantizer,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
