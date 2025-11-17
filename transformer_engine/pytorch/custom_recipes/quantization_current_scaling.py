# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Current scaling recipe reference implementation."""

import dataclasses
import math
from typing import Optional, Tuple, Iterable

import torch

from transformer_engine.pytorch.custom_recipes import quantization
from transformer_engine.pytorch.custom_recipes import utils
from transformer_engine.pytorch.quantized_tensor import QuantizedTensorStorage, Quantizer


def current_scaling_ref_quantizer_factory(role):
    """Factory function for current scaling reference quantizer.

    Usage with CustomRecipe and autocast:
        custom_recipe = recipe.CustomRecipe(qfactory=current_scaling_ref_quantizer_factory)
        with autocast(recipe=custom_recipe):
            output = model(input)
    """
    if role in ("linear_input", "linear_weight"):
        dtype = torch.float8_e4m3fn
    elif role in ("linear_output", "linear_grad_output"):
        dtype = torch.float8_e5m2
    else:
        return None
    return CurrentScalingQuantizerRef(
        dtype=dtype,
        rowwise=True,
        columnwise=True,
        pow_2_scales=False,
        eps=0.0,
    )


@dataclasses.dataclass
class CurrentScalingTensorRef(QuantizedTensorStorage):
    """Reference implementation of current scaling quantized tensor"""

    data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    data_t: Optional[torch.Tensor] = None
    scale_t: Optional[torch.Tensor] = None

    dtype: Optional[torch.dtype] = None
    device: Optional[torch.device] = None
    quant_dtype: Optional[torch.dtype] = None
    original_shape: Optional[Tuple[int, ...]] = None
    _quantizer: Optional[Quantizer] = None

    @property
    def custom(self) -> bool:
        """Flag to indicate this quantized tensor is custom."""
        return True

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], QuantizedTensorStorage]:
        """Prepare the quantization result for saving for backward"""
        tensors = [self.data, self.data_t, self.scale, self.scale_t]
        self.data = None
        self.data_t = None
        self.scale = None
        self.scale_t = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the quantization result from the saved tensors"""
        self.data = tensors[0]
        self.data_t = tensors[1]
        self.scale = tensors[2]
        self.scale_t = tensors[3]
        return tensors[4:]

    # Compatibility
    @property
    def _data(self):
        return self.data

    @_data.setter
    def _data(self, value):
        self.data = value

    @property
    def _scale_inv(self):
        return self.scale

    @_scale_inv.setter
    def _scale_inv(self, value):
        self.scale = value

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"quant_dtype={self.quant_dtype}, "
            f"original_shape={self.original_shape}"
            ")"
        )

    def update_usage(
        self,
        rowwise_usage: Optional[bool] = None,
        columnwise_usage: Optional[bool] = None,
    ):
        """Generate or remove quantized data based on provided usage."""
        has_data = self.data is not None
        has_data_transpose = self.data_t is not None
        needs_data = has_data
        needs_data_transpose = has_data_transpose

        if rowwise_usage is not None:
            needs_data = rowwise_usage
        if columnwise_usage is not None:
            needs_data_transpose = columnwise_usage

        # Generate data that is required
        if needs_data and not has_data:
            raise RuntimeError("Cannot generate FP8 data, even from FP8 data transpose")
        if needs_data_transpose and not has_data_transpose:
            if not has_data:
                raise RuntimeError("FP8 data is required to generate FP8 data transpose")
            self._create_transpose()

        # Delete data that is not required
        if not needs_data:
            self.data = None
        if not needs_data_transpose:
            self.data_t = None

    def _create_transpose(self):
        """Create transposed quantized tensor"""
        if not self.data.is_contiguous():
            self.data = self.data.contiguous()
        self.data_t = self.data.t().contiguous()
        self.scale_t = self.scale

    def size(self, *args, **kwargs):
        """Get the size of the quantized tensor"""
        if self.data is not None:
            return self.data.size(*args, **kwargs)
        size = self.data_t.size(*args, **kwargs)
        return torch.Size([size[-1], math.prod(size[:-1])])


def _scale_from_amax_tensor(
    x_dtype: torch.dtype,
    amax: torch.Tensor,
    quant_dtype: torch.dtype,
    *,
    eps: float,
    pow_2_scales: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Derives quantization and dequantization from amax and options.

    Reference implementation for scale calculation.

    Returns:
    - scale: quantization scales
    - scale_inv: dequantization scales
    - amax: Amax tensor with updates made for extrema values.
    """
    assert amax.dtype == torch.float, "amax must be a float tensor."
    fp8_max = torch.finfo(quant_dtype).max

    # Clamping amax to avoid division by small numbers
    amax = torch.max(amax, torch.tensor(eps))

    # Compute scale factor
    scale = torch.div(fp8_max, amax)

    # Take care of inf before pow_2_scales
    scale = torch.where(scale == torch.inf, torch.finfo(x_dtype).max, scale)

    if pow_2_scales:
        _, exp = torch.frexp(scale)
        exp = exp - 1
        assert (exp > -127).all()
        unity = torch.tensor([1.0], device=exp.device)
        torch.ldexp(unity, exp, out=scale)
        scale = torch.where(amax == float("inf"), 0.0, scale)

    # Handle overflow cases for amax zero causing NaN
    scale = torch.where(amax == 0, 1.0, scale)

    # Compute scale_inv
    scale_inv = torch.reciprocal(scale)

    return scale, scale_inv, amax


class CurrentScalingQuantizerRef(Quantizer):
    """Reference implementation of current scaling quantizer"""

    def __init__(
        self,
        dtype: torch.dtype,
        rowwise: bool = True,
        columnwise: bool = True,
        pow_2_scales: bool = False,
        eps: float = 0.0,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.internal = True

        self.dtype = dtype
        self.pow_2_scales = pow_2_scales
        self.eps = eps

        self.with_amax_reduction = False
        self.amax_reduction_group = None

    @property
    def custom(self) -> bool:
        """Flag to indicate this quantizer is custom."""
        return True

    @property
    def supports_allgather_fp8(self) -> bool:
        """Flag to indicate this quantizer supports allgather fp8"""
        return True

    @classmethod
    def compute_scale(
        cls,
        x: torch.Tensor,
        quant_dtype: torch.dtype,
        eps=0.0,
        pow_2_scales: bool = False,
    ):
        """Compute the scale from the amax tensor"""
        # Use float32 for computation
        x_fp32 = x.to(torch.float32)

        if x_fp32.numel() == 0:
            amax = torch.empty(1, dtype=torch.float32, device=x.device)
        else:
            amax = torch.amax(torch.abs(x_fp32)).view(1)

        return _scale_from_amax_tensor(
            x.dtype,
            amax=amax,
            quant_dtype=quant_dtype,
            eps=eps,
            pow_2_scales=pow_2_scales,
        )

    def _quantize(self, tensor: torch.Tensor) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Python implementation of quantization (c++ kernel can be used as an option instead).

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to quantize (should be 2D)

        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
            (qx, sx, qx_t, sx_t) where:
            - qx: quantized data in row-major order (if rowwise_usage), None otherwise
            - sx: empty scale tensor for qx (if rowwise_usage), None otherwise
            - qx_t: quantized data in column-major order (if columnwise_usage), None otherwise
            - sx_t: empty scale tensor for qx_t (if columnwise_usage), None otherwise
        """
        # Handle amax reduction if enabled
        if self.with_amax_reduction:
            assert (
                self.amax_reduction_group is not None
            ), "amax_reduction_group must be set when with_amax_reduction is True"

            # Compute local amax
            if tensor.numel() == 0:
                amax = torch.empty(1, dtype=torch.float32, device=tensor.device)
            else:
                amax = torch.amax(torch.abs(tensor)).view(1).to(torch.float32)

            # Reduce amax across all ranks
            torch.distributed.all_reduce(
                amax, group=self.amax_reduction_group, op=torch.distributed.ReduceOp.MAX
            )

            # Compute scale using the global amax
            scale, scale_inv, _ = _scale_from_amax_tensor(
                tensor.dtype,
                amax=amax,
                quant_dtype=self.dtype,
                eps=self.eps,
                pow_2_scales=self.pow_2_scales,
            )
        else:
            # compute scale factor using local amax
            scale, scale_inv, _ = self.compute_scale(
                tensor,
                self.dtype,
                eps=self.eps,
                pow_2_scales=self.pow_2_scales,
            )

        qx: Optional[torch.Tensor] = (tensor.float() * scale).to(self.dtype)
        sx: Optional[torch.Tensor] = scale_inv

        # transpose if needed
        if self.columnwise_usage:
            assert qx is not None
            qx_t = qx.t().contiguous()
            sx_t = sx
        else:
            qx_t, sx_t = None, None

        if not self.rowwise_usage:
            qx = None
            sx = None

        return qx, sx, qx_t, sx_t

    def quantize(
        self,
        tensor: torch.Tensor,
        **kwargs,  # pylint: disable=unused-argument
    ) -> CurrentScalingTensorRef:
        # sanity checks
        assert tensor.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        # Make it work with 3D tensors
        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.shape[-1])

        qx, sx, qx_t, sx_t = self._quantize(tensor)

        return CurrentScalingTensorRef(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=tensor.dtype,
            device=tensor.device,
            quant_dtype=self.dtype,
            _quantizer=self,
            original_shape=original_shape,
        )

    def dequantize(
        self, tensor: torch.Tensor, scale: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        tensor = tensor.to(torch.float32) * scale
        if dtype is None:
            return tensor
        return tensor.to(dtype)

    def qgemm(
        self,
        qx: torch.Tensor,
        qw: torch.Tensor,
        m_params: quantization.MMParams,
        out_dtype: torch.dtype,
        sx: torch.Tensor,
        sw: torch.Tensor,
        bias: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
        accumulate: bool = False,
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,  # pylint: disable=unused-argument
        qresult_x: QuantizedTensorStorage | None = None,  # pylint: disable=unused-argument
        qresult_w: QuantizedTensorStorage | None = None,  # pylint: disable=unused-argument
    ) -> torch.Tensor:
        """Python implementation of quantized gemm."""
        M, K = qx.shape
        N, _ = qw.shape

        if M == 0 or K == 0 or N == 0:
            if accumulate:
                assert out is not None
                y = out
            else:
                y = torch.zeros((M, N), dtype=out_dtype, device=qx.device)
            if bias is not None:
                y += bias
            return y

        # cublas fp8 gemm does not support fp32 bias
        use_bias_in_gemm = (
            bias is not None and out_dtype != torch.float32 and bias.dtype != torch.float32
        )

        # Run quantized gemm: y = qw * qx
        scaled_mm_res = torch._scaled_mm(
            qx,
            qw.transpose(-1, -2),
            scale_a=sx,
            scale_b=sw,
            out_dtype=out_dtype,
            use_fast_accum=not m_params.use_split_accumulator,
            bias=bias if use_bias_in_gemm else None,
        )
        y = scaled_mm_res[0] if isinstance(scaled_mm_res, tuple) else scaled_mm_res

        if bias is not None and not use_bias_in_gemm:
            # Check number of elements in bias tensor because it can be an empty tensor
            if bias.numel():
                y += bias

        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            out.add_(y)
            y = out
        else:
            assert out is None, "Output tensor should be None when accumulate is False."

        return y

    def transpose_qresult(self, qresult: CurrentScalingTensorRef) -> CurrentScalingTensorRef:
        """Python implementation of transpose qresult."""
        qx = qresult.data
        scale = qresult.scale
        assert qresult.data_t is None
        assert qresult.scale_t is None
        assert qx is not None
        qx_t = qx.transpose(-2, -1).contiguous()
        scale_t = scale
        qresult.data_t = qx_t
        qresult.scale_t = scale_t
        return qresult

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensorStorage,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> QuantizedTensorStorage:
        """Update the quantized tensor with the given tensor in-place

        Parameters
        ----------
        src: torch.Tensor
            Source tensor to copy from
        dst: ExperimentalQuantizedTensor
            Destination ExperimentalQuantizedTensor to update
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update
        """
        # Handle noop flag
        if noop_flag is not None and noop_flag.item() != 0:
            return dst

        # Make sure input is in expected format
        if not src.is_contiguous():
            src = src.contiguous()

        # Store the original shape and reshape for processing
        original_shape = src.shape
        if src.ndim > 2:
            src = src.view(-1, src.shape[-1])

        qx, sx, qx_t, sx_t = self._quantize(src)

        # Update the destination with new data
        dst.data = qx
        dst.scale = sx
        dst.data_t = qx_t
        dst.scale_t = sx_t
        dst.dtype = src.dtype
        dst.quant_dtype = self.dtype
        dst.original_shape = original_shape

        return dst

    def make_empty(
        self,
        shape: Iterable[int],
        *,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,  # pylint: disable=unused-argument
    ) -> CurrentScalingTensorRef:
        assert len(shape) == 2, "shape is not 2d"

        # Canonicalize tensor attributes
        if device is None:
            device = torch.device("cuda")

        # Allocate quantized data
        qx = torch.empty(shape, dtype=self.dtype, device=device)
        sx = torch.empty(1, dtype=torch.float32, device=device)

        # Allocate quantized data transpose if needed
        qx_t = None
        sx_t = None
        if self.columnwise_usage:
            inner_dim = qx.size(-1)
            qx_t = torch.empty(
                inner_dim,
                qx.numel() // inner_dim,
                dtype=self.dtype,
                device=device,
            )
            sx_t = torch.empty(1, dtype=torch.float32, device=device)

        # Construct quantized tensor
        return CurrentScalingTensorRef(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            dtype=dtype,
            device=device,
            quant_dtype=self.dtype,
            _quantizer=self,
            original_shape=shape,
        )
