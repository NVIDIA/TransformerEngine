# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with FP8 data"""
from __future__ import annotations
from typing import Any, Optional, Tuple, Iterable, Union
import warnings
import torch
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import DelayedScaling, Float8CurrentScaling, Recipe
from ..utils import canonicalize_process_group, devices_match
from .storage.float8_tensor_storage import Float8TensorStorage, _FromFloat8Func
from ..quantized_tensor import QuantizedTensor, Quantizer
from ._quantization_helpers import _IdentityFunc
from ..constants import dist_group_type

aten = torch.ops.aten

_ops_to_preserve_subclass_in_fsdp2 = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.copy_.default,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


class Float8Quantizer(Quantizer):
    """Builder class for FP8 tensors with per-tensor delayed scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized by
    multiplying with a scaling factor and casting to FP8. The max-abs
    value ("amax") in the tensor is also computed, which can be used
    for updating the scaling factor (handled externally by
    DelayedScalingRecipeState and FP8GlobalStateManager).

    """

    """Scaling factor to multiply when quantizing to FP8"""
    scale: torch.Tensor
    """Max-abs value from last FP8 cast"""
    amax: torch.Tensor
    """FP8 datatype"""
    dtype: TE_DType

    def __init__(
        self,
        scale: torch.Tensor,
        amax: torch.Tensor,
        fp8_dtype: TE_DType,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.scale = scale
        self.amax = amax
        self.dtype = fp8_dtype

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        if not isinstance(dst, Float8Tensor):
            raise ValueError("Float8Quantizer can only update Float8Tensor")

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

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Float8Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Allocate FP8 data
        data = torch.empty(shape, dtype=torch.uint8, device=device)

        # Allocate FP8 data transpose if needed
        data_transpose = None
        if self.columnwise_usage:
            transpose_shape = [data.size(-1)] + list(data.shape[:-1])
            data_transpose = torch.empty(
                transpose_shape,
                dtype=torch.uint8,
                device=device,
            )

        # Construct FP8 tensor
        return Float8Tensor(
            shape=shape,
            dtype=dtype,
            data=data,
            fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=device),
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=data_transpose,
            quantizer=self,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        amin, amax = tensor.aminmax()
        self.amax.copy_(torch.max(-amin, amax))

    def create_tensor_from_data(
        self,
        data: torch.Tensor,
        fake_dtype=torch.float32,
        requires_grad: bool = False,
        internal: bool = False,
    ):
        """Create Float8Tensor from raw uint8 data"""
        assert data.dtype in [
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ]
        if internal:
            return Float8TensorStorage(
                data=data,
                fp8_scale_inv=1 / self.scale,
                fp8_dtype=self.dtype,
                requires_grad=requires_grad,
                data_transpose=None,
                quantizer=self,
            )
        return Float8Tensor(
            shape=data.shape,
            dtype=fake_dtype,
            data=data,
            fp8_scale_inv=1 / self.scale,
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=None,
            quantizer=self,
        )

    def onnx_quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Function using primitives with ONNX defined translations."""
        # Q inputs are currently constrained to FP32 due to a similar limitation in ORT
        # custom ops, so cast the input if needed.
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        data = torch.ops.tex.fp8_quantize(tensor, self.scale.item())
        return self.create_tensor_from_data(data, fake_dtype=torch.float32)

    def onnx_dequantize(self, tensor: QuantizedTensor) -> torch.Tensor:
        """Function using primitives with ONNX defined translations."""
        out = torch.ops.tex.fp8_dequantize(tensor._data, tensor._scale_inv)
        out = out.to(tensor.dtype)
        return out

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return DelayedScaling

    def supports_only_rowwise_all_gather(self) -> bool:
        """
        Float8Quantizer supports only rowwise all-gather
        """
        return True


class Float8CurrentScalingQuantizer(Quantizer):
    """Builder class for FP8 tensors with per-tensor current scaling

    High-precision tensors (e.g. in FP32 or BF16) are quantized by
    multiplying with a scaling factor and casting to FP8. The max-abs
    value ("amax") in the tensor is computed directly by scanning the input
    high-precision tensor, without the need of any history window.

    Unlike delayed scaling, scale and amax tensors are not needed to initialize the
    quantizer, becuse they are simply GPU buffers that will be filled by current
    scaling quantization kernels, instead of using values taken from delayed scaling
    history window. Therefore, device parameter is needed for tensor allocation.

    Both Float8CurrentScalingQuantizer and Float8Quantizer produces Float8Tensor,
    because they are both per-tensor scaling, ie. one scaling factor per tensor.

    """

    """Workspace buffer for FP8 scaling factor"""
    scale: torch.Tensor
    """Workspace buffer for max-abs value"""
    amax: torch.Tensor
    """FP8 datatype"""
    dtype: TE_DType
    """amax update options"""
    use_existing_amax: bool
    """amax reduction options"""
    with_amax_reduction: bool
    amax_reduction_group: Optional[dist_group_type]
    """Options about how to quantize the tensor"""
    force_pow_2_scales: bool
    amax_epsilon: float

    def __init__(
        self,
        fp8_dtype: TE_DType,
        device: torch.device,
        *,
        rowwise: bool = True,
        columnwise: bool = True,
        use_existing_amax: bool = False,
        with_amax_reduction: bool = False,
        amax_reduction_group: Optional[dist_group_type] = None,
        force_pow_2_scales: bool = False,
        amax_epsilon: float = 0.0,
    ) -> None:
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.scale = torch.empty(1, dtype=torch.float32, device=device)
        self.amax = torch.empty(1, dtype=torch.float32, device=device)
        self.dtype = fp8_dtype
        self.use_existing_amax = use_existing_amax
        self.with_amax_reduction = with_amax_reduction
        self.amax_reduction_group = amax_reduction_group
        self.force_pow_2_scales = force_pow_2_scales
        self.amax_epsilon = amax_epsilon

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        if not isinstance(dst, Float8Tensor):
            raise ValueError("Float8CurrentScalingQuantizer can only update Float8Tensor")

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

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Float8Tensor:

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Allocate FP8 data
        data = torch.empty(shape, dtype=torch.uint8, device=device)

        # Allocate FP8 data transpose if needed
        data_transpose = None
        if self.columnwise_usage:
            transpose_shape = [data.size(-1)] + list(data.shape[:-1])
            data_transpose = torch.empty(
                transpose_shape,
                dtype=torch.uint8,
                device=device,
            )
        # Construct FP8 tensor
        return Float8Tensor(
            shape=shape,
            dtype=dtype,
            data=data,
            fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=device),
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=data_transpose,
            quantizer=self,
        )

    def calibrate(self, tensor: torch.Tensor) -> None:
        # current scaling don't need to calibrate
        return

    def create_tensor_from_data(
        self,
        data: torch.Tensor,
        fake_dtype=torch.float32,
        requires_grad: bool = False,
        internal: bool = False,
    ):
        """
        Create Float8Tensor from raw uint8 data, unlike delayed scaling,
        self.scale doesn't mean anything, so we are simply creating empty scale_inv
        """
        assert data.dtype in [
            torch.uint8,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
        ]
        if internal:
            return Float8TensorStorage(
                data=data,
                fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=data.device),
                fp8_dtype=self.dtype,
                requires_grad=requires_grad,
                data_transpose=None,
                quantizer=self,
            )
        return Float8Tensor(
            shape=data.shape,
            dtype=fake_dtype,
            data=data,
            fp8_scale_inv=torch.empty(1, dtype=torch.float32, device=data.device),
            fp8_dtype=self.dtype,
            requires_grad=requires_grad,
            data_transpose=None,
            quantizer=self,
        )

    def onnx_quantize(self, tensor: torch.Tensor) -> QuantizedTensor:
        """Function using primitives with ONNX defined translations."""
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        data, scale_inv = torch.ops.tex.fp8_cs_quantize(tensor)
        return Float8Tensor(
            shape=data.shape,
            dtype=torch.float32,
            data=data,
            fp8_scale_inv=scale_inv,
            fp8_dtype=self.dtype,
            requires_grad=False,
            data_transpose=None,
            quantizer=self,
        )

    def onnx_dequantize(self, tensor: QuantizedTensor) -> torch.Tensor:
        """Function using primitives with ONNX defined translations."""
        out = torch.ops.tex.fp8_dequantize(tensor._data, tensor._scale_inv)
        out = out.to(tensor.dtype)
        return out

    def _canonicalized_amax_reduction_group(self) -> dist_group_type:
        """Get process group for amax reduction"""
        return canonicalize_process_group(self.amax_reduction_group)

    def _get_compatible_recipe(self) -> Union[type[Recipe], None]:
        return Float8CurrentScaling

    def supports_only_rowwise_all_gather(self) -> bool:
        """
        Float8CurrentScalingQuantizer supports only rowwise all-gather
        """
        return True


class Float8Tensor(Float8TensorStorage, QuantizedTensor):
    """Experimental tensor class with FP8 data

    The tensor presents as having a standard, higher-precision dtype,
    but the data itself is (scaled) FP8. For most tensor operations,
    the data will be cast to the nominal dtype before performing the
    operation.

    Parameters
    ----------
    shape: int or iterable of int
        Tensor dimensions.
    dtype: torch.dtype
        Nominal tensor datatype.
    requires_grad: bool, optional = False
        Whether to compute gradients for this tensor.
    data: torch.Tensor
        Raw FP8 data in a uint8 tensor
    fp8_scale_inv: torch.Tensor
        Reciprocal of the scaling factor applied when casting to FP8,
        i.e. the scaling factor that must be applied when casting from
        FP8 to higher precision.
    fp8_dtype: transformer_engine_torch.DType
        FP8 format.
    data_transpose: torch.Tensor, optional
        FP8 transpose data in a uint8 tensor
    quantizer: Float8Quantizer, Float8CurrentScalingQuantizer, optional
        Builder class for FP8 tensors

    """

    def __repr__(self, *, tensor_contents=None):
        return (
            "Float8Tensor("
            f"fp8_dtype={self._fp8_dtype}, "
            f"scale_inv={self._scale_inv.item()}, "
            f"data={self.dequantize(dtype=self.dtype)}"
            ")"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8Tensor

        By default the resulting tensor's dtype is the
        Float8Tensor's nominal dtype.
        """
        # Convert PyTorch dtype to TE dtype
        if dtype is None:
            dtype = self.dtype

        if torch.is_grad_enabled():
            return _FromFloat8Func.apply(self, dtype)
        return _FromFloat8Func.forward(None, self, dtype)

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> Float8Tensor:
        """Update FP8 data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, QuantizedTensor):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        return super().quantize_(tensor, noop_flag=noop_flag)

    def detach(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return Float8Tensor.make_like(self)

    def clone(self) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        assert self._data is not None
        data = self._data.detach().clone()
        data_transpose = None
        if self._transpose is not None:
            data_transpose = self._transpose.detach().clone()
        return _IdentityFunc.apply(
            self,
            {
                "data": data,
                "data_transpose": data_transpose,
            },
        )

    def view(self, *shape: Tuple[int]) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> Float8Tensor:
        """Returns tensor with data in provided memory format

        Returns `self` if data is already in correct memory format.

        """
        if self._data is not None and self._data.is_contiguous(memory_format=memory_format):
            return self
        if self._transpose is not None and self._transpose.is_contiguous(
            memory_format=memory_format
        ):
            return self
        return Float8Tensor.make_like(tensor=self, data=self._data.contiguous())

        # raise ValueError("Float8Tensor does not support different memory formats!")

    def _reset_caches(self) -> None:
        """
        Set transpose cache as invalid.
        Should be called after any in-place operation.
        """
        self._transpose_invalid = True

    def remove_caches(self) -> None:
        """
        Remove transpose cache and mark it as invalid.
        """
        self._transpose_invalid = True
        del self._transpose  # explicitly deletes the data for safety
        self._transpose = None

    @classmethod
    def make_like(
        cls,
        tensor: QuantizedTensor,
        *,
        shape: Optional[Iterable[int]] = None,
        dtype: Optional[torch.dtype] = None,
        requires_grad: bool = False,
        data: Optional[torch.Tensor] = None,
        data_transpose: Optional[torch.Tensor] = None,
    ) -> QuantizedTensor:
        """Create new quantized tensor

        By default, new tensor has the same attributes and underlying
        data.

        """
        if shape is None and data is not None:
            shape = data.shape
        new_tensor = super().make_like(
            tensor, shape=shape, dtype=dtype, requires_grad=requires_grad
        )
        if data is not None:
            new_tensor._data = data
        if data_transpose is not None:
            new_tensor._transpose = data_transpose
            new_tensor._transpose_invalid = False
        return new_tensor

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func == aten.view.default:
            tensor = args[0]
            data = tensor._data
            out_data = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            out_shape = out_data.size()
            out_transpose = None if tensor._transpose_invalid else tensor._transpose
            if out_transpose is not None:
                out_transpose_shape = out_transpose.size()
                if (
                    out_transpose_shape[0] != out_shape[-1]
                    or out_transpose_shape[1:] != out_shape[:-1]
                ):
                    out_transpose = None
                else:
                    view_shape_for_transpose = [out_shape[-1]] + list(out_shape[:-1])
                    out_transpose = out_transpose.view(*view_shape_for_transpose)
            return Float8Tensor(
                shape=out_shape,
                dtype=tensor.dtype,
                requires_grad=False,
                data=out_data,
                fp8_scale_inv=tensor._scale_inv,
                fp8_dtype=tensor._fp8_dtype,
                data_transpose=out_transpose,
                quantizer=tensor._quantizer,
            )

        if func in [aten.slice.Tensor, aten.select.int]:
            tensor = args[0]
            data = tensor._data
            data_slice = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            return Float8Tensor.make_like(tensor, data=data_slice, shape=data_slice.shape)

        # Related to FSDP2
        if func == aten.split.Tensor:
            tensor = args[0]
            data = tensor._data
            func_out = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            t_func_out = [None] * len(func_out)
            # Compute corresponding split of the transpose cache if available
            if tensor._transpose is not None and not tensor._transpose_invalid:
                transpose = tensor._transpose
                ndim = data.dim()
                # Figure out the original split dim
                if "dim" in kwargs:
                    dim_to_split = kwargs["dim"]
                else:
                    dim_to_split = args[2] if len(args) > 2 else 0
                # Dimension along which transpose needs to be split
                t_dim = 0 if dim_to_split == ndim - 1 else dim_to_split + 1
                t_func_out = transpose.__torch_dispatch__(
                    func,
                    types,
                    [transpose, args[1], t_dim],
                    kwargs,
                )
            outs = [
                Float8Tensor.make_like(
                    tensor,
                    data=split_tensor,
                    data_transpose=split_transpose_tensor,
                    shape=split_tensor.shape,
                )
                for split_tensor, split_transpose_tensor in zip(func_out, t_func_out)
            ]
            return outs

        if func == aten.new_zeros.default:
            # create fresh new tensor with zeros.
            tensor = args[0]
            data = tensor._data
            func_out = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            func_transposed_out = None
            if tensor._transpose is not None and not tensor._transpose_invalid:
                transpose = tensor._transpose
                size = args[1]
                t_shape = [size[-1]] + list(size[:-1])
                func_transposed_out = transpose.__torch_dispatch__(
                    func,
                    types,
                    [transpose, t_shape] + list(args[2:]),
                    kwargs,
                )
            # deep copy the scale inverse tensor and quantizer as well.
            scale_inv = tensor._scale_inv.detach().clone()
            quantizer = tensor._quantizer.copy()
            out_tensor = Float8Tensor(
                data=func_out,
                shape=func_out.shape,
                dtype=tensor.dtype,
                fp8_dtype=tensor._fp8_dtype,
                fp8_scale_inv=scale_inv,
                data_transpose=func_transposed_out,
                quantizer=quantizer,
            )
            return out_tensor

        if func == torch.ops.aten.as_strided.default:
            tensor = args[0]
            data = tensor._data
            # Apply as_strided to the primary uint8 data
            func_out = data.__torch_dispatch__(
                func,
                types,
                [data] + list(args[1:]),
                kwargs,
            )
            func_transposed_out = None
            if tensor._transpose is not None and not tensor._transpose_invalid:
                transpose = tensor._transpose
                size = args[1]
                stride = args[2]
                if "storage_offset" in kwargs:
                    storage_offset = kwargs["storage_offset"]
                else:
                    storage_offset = args[3] if len(args) > 3 else 0
                # Shape and strided needed for transpose matrix
                t_size = [size[-1]] + list(size[:-1])
                t_stride = [stride[-1]] + list(stride[:-1])
                func_transposed_out = transpose.__torch_dispatch__(
                    func,
                    types,
                    [transpose, t_size, t_stride, storage_offset] + list(args[4:]),
                    kwargs,
                )
            return Float8Tensor.make_like(
                tensor, data=func_out, data_transpose=func_transposed_out, shape=func_out.shape
            )

        if func == torch.ops.aten.detach.default:
            return cls.detach(args[0])
        if func == torch.ops.aten.clone.default:
            return cls.clone(args[0])
        if func == torch.ops.aten.copy_.default:
            dst, src = args[0], args[1]
            # Just copy FP8 attrs if copying between Float8Tensors
            if isinstance(src, Float8Tensor) and isinstance(dst, Float8Tensor):
                dst._data.copy_(src._data.detach())
                dst._scale_inv.copy_(src._scale_inv.view(dst._scale_inv.size()))
                if src._transpose is not None or dst._transpose is not None:
                    dst._create_transpose()
                return dst
        elif func in _ops_to_preserve_subclass_in_fsdp2:
            # Ops in the _ops_to_preserve_subclass_in_fsdp2 are recommened to return the same class instance to work fine with the torch fsdp2
            warnings.warn(
                f"A function call({func}) in {cls} may not return {cls} tensor as an output. It"
                " might cause an error in torch FSDP2!"
            )
        else:
            pass
        return super().__torch_dispatch__(func, types, args, kwargs)

    def fsdp_pre_all_gather(self, mesh, orig_size, contiguous_orig_stride, module, mp_policy):
        """Functions FSDP2 calls before all-gather of the
        weights for both forward and backward passes.
        Args:
            mesh (torch.distributed.DeviceMesh): DeviceMesh used by FSDP2
            to shard the weights.
            orig_size (torch.Size): Original size of the weight tensor.(For us same as self.shape)
            contiguous_orig_stride (Tuple[int]): Original stride of the weight tensor
            (For us same as self.stride())
            module (FSDPModule): FSDP module. FSDP wrapped module wrapped using fully_shard
            that contains this FP8 tensor.
            mp_policy (MixedPrecisionPolicy): Mixed precision policy used by FSDP2.

        Returns:
            shareded_tensors: Tuple[torch.Tensor, ...]: Tuple of tensors
            that need to be all-gathered.(In this case uint8 data tensor)
            metadata: Tuple[Any]: Metadata needed for reconstructing the
            Float8Tensor after all-gather.
        """
        # pylint: disable=unused-argument
        # Importing here to avoid circular imports
        from transformer_engine.pytorch.distributed import _get_module_fsdp_state

        if isinstance(self._quantizer, Float8CurrentScalingQuantizer) and mesh is not None:
            # When sharded weight is updated after reduce scattering the gradients in FSDP2,
            # we need to do amax reduction across the mesh to make sure all weight shards are
            # updated with same scale inverse. Setting the state below in the quantizer will make
            # sure that updated Quantized weight tensor have same scale inverse across all shards.
            self._quantizer.amax_reduction_group = mesh.get_group()
            self._quantizer.with_amax_reduction = True
        quantizer = self._quantizer.copy()  # quantizer to be used for allgathered weights
        fsdp_state = _get_module_fsdp_state(module)
        reshard_after_forward = fsdp_state._fsdp_param_group._reshard_after_forward
        # If weights are resharded after forward pass, then its enough to set the quantizer usages
        # based on whether its forward or backward pass for the allgathered weights.
        # If not resharded after forward pass, the same weights allgathered in forward
        # are used again in backward and so we dont change the quantizer usages which might need
        # both rowwise and columnwise usages.
        if reshard_after_forward:
            training_state = fsdp_state._fsdp_param_group._training_state
            is_backward_pass = training_state == TrainingState.PRE_BACKWARD
            # In case of hopper/L40, only one of data/transpose is needed
            # based on forward or backward pass. So setting the quantizer usages appropriately.
            quantizer.set_usage(rowwise=not is_backward_pass, columnwise=is_backward_pass)
        sharded_tensors = (self._data,)
        metadata = (self._scale_inv, self._fp8_dtype, quantizer)
        return sharded_tensors, metadata

    def fsdp_post_all_gather(
        self,
        all_gather_outputs: Tuple[torch.Tensor, ...],
        metadata: Any,
        param_dtype: torch.dtype,
        *,
        out: Optional[Float8Tensor] = None,
    ):
        """Functions FSDP2 calls after all-gather of the
        weights for both forward and backward passes.
        Args:
            all_gather_outputs (Tuple[torch.Tensor, ...]): sharded_tensors sent out in fsdp_pre_all_gather from each rank
            are all-gathered and received here as a tuple.
            metadata (Any): metadata sent out in fsdp_pre_all_gather used for reconstructing the Float8Tensor.
            param_dtype (torch.dtype): high precision dtype of the Float8Tensor.
            out (Optional[torch.Tensor], optional): _description_. Defaults to None.

        Returns:
            Tuple[Float8Tensor, Tuple[torch.Tensor, ...]]: Allgathered Float8Tensor and tuple of internal tensors
            used by the Float8Tensor that was being computed after allgather.
        """

        (data,) = all_gather_outputs
        (fp8_scale_inv, fp8_dtype, quantizer) = metadata
        orig_shape = data.size()
        # Quantizer has only columnwise usage set for backward pass
        # In Blackwell+ architectures, transpose is not needed at all,
        # even if columnwise usage is set. and is going to be handled
        # internally in the update_usage method.
        if out is not None:
            out._data = data
        else:
            fp8_args = {
                "shape": orig_shape,
                "dtype": param_dtype,
                "fp8_scale_inv": fp8_scale_inv,
                "fp8_dtype": fp8_dtype,
                "quantizer": quantizer,
                "requires_grad": False,
                "data": data,
            }
            out = Float8Tensor(**fp8_args)

        out.update_usage(
            rowwise_usage=quantizer.rowwise_usage,
            columnwise_usage=quantizer.columnwise_usage,
        )
        return out, all_gather_outputs

    @classmethod
    def _make_in_reduce_ex(
        cls,
        data: torch.Tensor,
        fp8_dtype: TE_DType,
        fp8_scale_inv: torch.Tensor,
        dtype: torch.dtype,
        shape: torch.shape,
    ) -> Float8Tensor:
        """Build Float8Tensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional
        arguments.

        """
        return Float8Tensor(
            data=data,
            fp8_dtype=fp8_dtype,
            fp8_scale_inv=fp8_scale_inv,
            dtype=dtype,
            shape=shape,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling to remove references to FP8 metadata objects"""
        return (
            Float8Tensor._make_in_reduce_ex,
            (self._data, self._fp8_dtype, self._scale_inv, self.dtype, self.shape),
        )

    def _get_data(self) -> Float8Tensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes FP8 data if setting from a Float8Tensor. Otherwise
        casts to FP8.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy FP8 data if other tensor is Float8Tensor
        if isinstance(tensor, Float8Tensor):

            # PyTorch tensor attributes
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    Float8Tensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(Float8Tensor, type(self)).data.__set__(self, dummy_tensor)

            # Float8Tensor attributes
            self._data = tensor._data
            self._quantizer = tensor._quantizer.copy()
            self._fp8_dtype = tensor._fp8_dtype
            self._scale_inv = tensor._scale_inv
            self._transpose = tensor._transpose
            self._transpose_invalid = tensor._transpose_invalid
            return

        # Quantize to FP8
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.internal = False
        self.data = self._quantizer.quantize(tensor)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to FP8 when setting Float8Tensor.data
    data = property(_get_data, _set_data)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        shape: Optional[list[int]] = None,
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        ctx.shape = tensor.shape
        if shape is None:
            return tensor.detach()
        out_data = tensor._data.view(*shape)
        out_shape = out_data.size()
        out_transpose = None if tensor._transpose_invalid else tensor._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if out_transpose_shape[0] != out_shape[-1] or out_transpose_shape[1:] != out_shape[:-1]:
                out_transpose = None
            else:
                view_shape_for_transpose = [shape[-1]] + list(shape[:-1])
                out_transpose = out_transpose.view(*view_shape_for_transpose)
        return Float8Tensor(
            shape=out_shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            data=out_data,
            fp8_scale_inv=tensor._scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            data_transpose=out_transpose,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad.reshape(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the Float8Tensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: Float8Tensor,
        shape: Tuple[int],
    ) -> Float8Tensor:
        # pylint: disable=missing-function-docstring
        ctx.shape = tensor.shape
        if shape is None:
            return tensor.detach()
        out_data = tensor._data.reshape(*shape)
        out_shape = out_data.size()
        out_transpose = None if tensor._transpose_invalid else tensor._transpose
        if out_transpose is not None:
            out_transpose_shape = out_transpose.size()
            if out_transpose_shape[0] != out_shape[-1] or out_transpose_shape[1:] != out_shape[:-1]:
                out_transpose = None
            else:
                reshape_shape_for_transpose = [shape[-1]] + list(shape[:-1])
                out_transpose = out_transpose.reshape(*reshape_shape_for_transpose)
        return Float8Tensor(
            shape=out_shape,
            dtype=tensor.dtype,
            requires_grad=tensor.requires_grad,
            data=out_data,
            fp8_scale_inv=tensor._scale_inv,
            fp8_dtype=tensor._fp8_dtype,
            data_transpose=out_transpose,
            quantizer=tensor._quantizer,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring
        return grad.reshape(ctx.shape), None
