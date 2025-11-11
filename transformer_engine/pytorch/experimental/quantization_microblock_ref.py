# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFP4 implementations for experimental middleware between Transformer Engine and Kitchen."""

from typing import Optional, Tuple

import torch

from transformer_engine.pytorch.experimental import quantization
from transformer_engine.pytorch.experimental import utils
from transformer_engine.pytorch.experimental.quantization import (
    ExperimentalQuantizedTensor,
    ExperimentalQuantizer,
)


def cast_to_fp4x2(x):
    """Quantize a tensor to FP4 E2M1 and store in a byte tensor"""

    result = torch.zeros_like(x, dtype=torch.uint8)
    result[(x >= 0.0) & (x <= 0.25)] = 0
    result[(x > 0.25) & (x < 0.75)] = 1
    result[(x >= 0.75) & (x <= 1.25)] = 2
    result[(x > 1.25) & (x < 1.75)] = 3
    result[(x >= 1.75) & (x <= 2.5)] = 4
    result[(x > 2.5) & (x < 3.5)] = 5
    result[(x >= 3.5) & (x <= 5.0)] = 6
    result[x > 5.0] = 7

    result[(x >= -0.25) & (x < -0.0)] = 8
    result[(x < -0.25) & (x > -0.75)] = 9
    result[(x <= -0.75) & (x >= -1.25)] = 10
    result[(x < -1.25) & (x > -1.75)] = 11
    result[(x <= -1.75) & (x >= -2.5)] = 12
    result[(x < -2.5) & (x > -3.5)] = 13
    result[(x <= -3.5) & (x >= -5.0)] = 14
    result[x < -5.0] = 15

    return result[:, ::2] + result[:, 1::2] * 16


def cast_from_fp4x2(x, dq_dtype):
    """Dequantize FP4 E2M1 tensor that has been represented in a byte tensor"""
    fp4_values = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=x.device,
        dtype=dq_dtype,
    )

    # Convert to long integers for indexing
    second_bit = torch.div(x, 16, rounding_mode="floor").to(torch.long)
    first_bit = (x - second_bit * 16).to(torch.long)

    # Use the long integers to index fp4_values
    first_bit_values = fp4_values[first_bit]
    second_bit_values = fp4_values[second_bit]

    result = torch.zeros(
        (first_bit_values.shape[0], first_bit_values.shape[1] * 2),
        device=x.device,
        dtype=dq_dtype,
    )
    result[:, ::2] = first_bit_values
    result[:, 1::2] = second_bit_values

    return result


def cast_to_e8(decode_scale):
    """Cast to a value that is representable in FP8 E8M0.

    The result is in FP32, not FP8 E8M0.
    """
    max_exponent = torch.tensor(127, device=decode_scale.device, dtype=torch.float32)
    exponent = torch.ceil(torch.log2(decode_scale))
    exponent = torch.clamp(exponent, min=-max_exponent, max=max_exponent)

    return torch.tensor(2.0, device=decode_scale.device, dtype=torch.float32) ** exponent


def cast_to_e4m3(decode_scale, global_amax):
    """Scale and cast to FP8 E4M3.

    decode_scale is actually the encoding scaling factor. global_amax
    can be any data tensor and not just the amax.

    TODO(etsykunov): Make less unintuitive.
    """
    decode_scale = decode_scale * global_amax
    FLOAT8_E4M3_MAX = torch.tensor(448.0, device=decode_scale.device, dtype=torch.float32)
    decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
    return decode_scale.to(torch.float8_e4m3fn)


def high_precision_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    out_dtype: torch.dtype,
    accumulate: bool = False,
    is_a_transposed: bool = False,
    is_b_transposed: bool = False,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    scale_alpha: float = 1.0,
) -> torch.Tensor:
    """GEMM implementation with unquantized data"""
    # Handle transpositions
    mat1, mat2 = a, b
    if is_a_transposed:
        mat1 = a.T
    if is_b_transposed:
        mat2 = b.T

    # Ensure dtype compatibility for torch.addmm
    mat1 = mat1.to(out_dtype)
    mat2 = mat2.to(out_dtype)

    # Determine output shape
    y_shape = (mat1.size(0), mat2.size(1))

    if bias is not None:
        assert not accumulate, "Bias is not supported with accumulation"
        bias = bias.to(out_dtype)
        # With bias case
        if out_dtype == torch.float32:
            y_ref = torch.addmm(bias.repeat(mat1.size(0), 1), mat1, mat2, beta=1, alpha=1)
        else:
            y_ref = torch.addmm(bias, mat1, mat2, beta=1, alpha=scale_alpha)
    else:
        # Without bias case
        if accumulate and out is not None:
            y_ref = out.clone().to(out_dtype)
        else:
            y_ref = torch.zeros(y_shape, dtype=out_dtype, device=a.device)
        torch.addmm(y_ref, mat1, mat2, beta=1, alpha=scale_alpha, out=y_ref)

    return y_ref


class NVFP4TensorRef(ExperimentalQuantizedTensor):
    """NVFP4 tensor for middleware between Transformer Engine and Kitchen"""

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"dtype={self.dtype}, "
            f"device={self.device}, "
            f"quant_dtype={self.quant_dtype}, "
            f"data={self.dequantize(dtype=self.dtype)}, "
            f"original_shape={self.original_shape}"
            ")"
        )

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> ExperimentalQuantizedTensor:
        """In-place update of quantized data

        Parameters
        ----------
        tensor: torch.Tensor
            Tensor to copy from
        noop_flag: torch.Tensor, optional
            float32 flag indicating whether to avoid performing update

        """
        if isinstance(tensor, ExperimentalQuantizedTensor):
            return self.quantize_(tensor.dequantize(), noop_flag=noop_flag)
        self.get_quantizer().update_quantized(tensor, self, noop_flag=noop_flag)
        return self

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from quantized tensor
        """
        if dtype is None:
            dtype = self.dtype

        # Ignore data_t for now
        assert self.data is not None, "QuantizedTensor has no valid tensor data"
        assert self.scale is not None, "QuantizedTensor has no valid scale"
        tensor_data = self.data
        tensor_scale = self.scale
        # Dispatch to the quantizer
        return self.get_quantizer().dequantize(tensor_data, tensor_scale, dtype=dtype)

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

    def size(self, *args, **kwargs):  # pylint: disable=unused-argument
        """Return the original tensor shape, not the internal packed data shape.

        FP4 quantization packs two 4-bit values into each 8-bit value, which reduces
        the second dimension by half. This method returns the logical shape that
        users expect, not the internal packed storage shape.
        """
        assert self.original_shape is not None
        return torch.Size(self.original_shape)


def get_wgrad_sign_vector() -> torch.Tensor:
    """Hard-coded signs for Hadamard transform"""
    return torch.tensor(
        [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0],
        dtype=torch.float32,
    )


class NVFP4QuantizerRef(ExperimentalQuantizer):
    """NVFP4 quantizer for middleware between Transformer Engine and Kitchen"""

    def __init__(
        self,
        dtype: utils.Fp4Formats,
        rowwise: bool = True,
        columnwise: bool = True,
        pow_2_scales: bool = False,
        eps: float = 0.0,
        quant_tile_shape: Tuple[int, int] = (1, 16),
        with_rht: bool = False,
        with_random_sign_mask: bool = True,
    ):
        super().__init__(rowwise=rowwise, columnwise=columnwise)
        self.dtype = dtype
        self.pow_2_scales = pow_2_scales
        self.eps = eps
        self.quant_tile_shape = quant_tile_shape
        self.with_rht = with_rht
        self.with_random_sign_mask = with_random_sign_mask

    @staticmethod
    def _build_hadamard_matrix(
        size: int, device: torch.device, dtype: torch.dtype, with_random_sign_mask: bool = True
    ) -> torch.Tensor:
        """Construct a Hadamard matrix of given power-of-two size with entries +-1.

        Uses Sylvester construction to avoid SciPy dependency.
        """
        assert (size & (size - 1)) == 0, "Hadamard size must be a power of two"
        h = torch.ones((1, 1), device=device, dtype=torch.float32)
        while h.shape[0] < size:
            h = torch.cat(
                [
                    torch.cat([h, h], dim=1),
                    torch.cat([h, -h], dim=1),
                ],
                dim=0,
            )
        if with_random_sign_mask:
            sign_mat = get_wgrad_sign_vector().to(device) * torch.eye(
                size, device=device, dtype=torch.float32
            )
            h = sign_mat @ h
        return h.to(dtype)

    def _apply_rht(self, x: torch.Tensor) -> torch.Tensor:
        """Apply randomized Hadamard transform without random signs (reference path).

        This matches the reference used in tests: x_reshaped @ (H * (1/sqrt(g))).
        """
        # Only apply when enabled
        if not self.with_rht:
            return x

        # RHT dimension equals the quantization tile length (NVFP4 uses 16)
        rht_dim = self.quant_tile_shape[1]
        assert (
            x.shape[-1] % rht_dim == 0
        ), f"Inner dimension {x.shape[-1]} must be divisible by hadamard dimension {rht_dim}"

        # Build H and scale
        H = self._build_hadamard_matrix(rht_dim, x.device, x.dtype, self.with_random_sign_mask)
        scale = 1.0 / float(rht_dim) ** 0.5

        # Perform blockwise transform along the last dimension
        original_shape = x.shape
        x_mat = x.contiguous().view(-1, rht_dim)
        # Random sign matrix is identity in this reference (no sign flipping)
        transform = H * scale
        out = x_mat @ transform
        return out.view(original_shape)

    @staticmethod
    def _recover_swizzled_scales(
        swizzled_scale: bool, scale: torch.Tensor, m: int, n: int, block_length: int
    ) -> torch.Tensor:
        if not swizzled_scale:
            return scale
        rounded_m = utils.roundup_div(m, 128) * 128
        scale_n = utils.roundup_div(n, block_length)
        rounded_n = utils.roundup_div(scale_n, 4) * 4
        # Recover swizzled scaling factor layout -> linear layout
        tmp = torch.reshape(scale, (rounded_m // 128, rounded_n // 4, 32, 4, 4))
        # after permutation, the layout is [rounded_m // 128, 4, 32, rounded_n // 4, 4]
        tmp = torch.permute(tmp, (0, 3, 2, 1, 4))
        result = torch.reshape(tmp, (rounded_m, rounded_n))
        return result[:m, :scale_n]

    @classmethod
    def _quantize_blockwise_reference(
        cls,
        x: torch.Tensor,
        global_amax: torch.Tensor,
        tile_len_x: int,
        tile_len_y: int,
        *,
        pow_2_scales: bool,
        eps: float,  # pylint: disable=unused-argument
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        assert x.ndim == 2
        using_2d_quantization = tile_len_x == 16 and tile_len_y == 16
        m, n = x.shape
        # Compute vec_max based on the original x (before reshape)
        # For 1D quantization: amax over each row chunk of 16
        # For 2D quantization: amax over each 16x16 block, but output shape is still (128, 8, 1), filled with block amax
        if using_2d_quantization:
            # x shape: (128, 128)
            x_blocks = (
                x.unfold(0, tile_len_y, tile_len_y)
                .unfold(1, tile_len_x, tile_len_x)
                .to(torch.float32)
            )  # (8, 8, 16, 16)
            block_amax = torch.amax(torch.abs(x_blocks), dim=(-1, -2))  # (8, 8)
            # Now, expand to (128, 8, 1) by repeating each block_amax for 16 rows
            vec_max = block_amax.repeat_interleave(tile_len_y, dim=0).unsqueeze(-1)  # (128, 8, 1)
        else:
            # x shape: (128, 128)
            x_reshaped = x.view(m, n // tile_len_x, tile_len_x)  # (128, 8, 16)
            vec_max = torch.amax(torch.abs(x_reshaped), dim=-1, keepdim=True).to(
                torch.float32
            )  # (128, 8, 1)
        x = x.view(m, n // tile_len_x, tile_len_x)
        FLOAT4_E2M1_MAX = torch.tensor(6.0, device=x.device, dtype=torch.float32)
        FLOAT8_E4M3_MAX = torch.tensor(448.0, device=x.device, dtype=torch.float32)
        decode_scale = torch.div(vec_max, FLOAT4_E2M1_MAX)

        if pow_2_scales:
            decode_scale = cast_to_e8(decode_scale)
            encode_scale = torch.div(
                torch.tensor(1.0, device=x.device, dtype=torch.float32),
                decode_scale.to(torch.float32),
            )
        else:
            global_encode_scale = torch.div(FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX, global_amax)
            global_encode_scale = torch.min(
                global_encode_scale,
                torch.tensor(
                    torch.finfo(torch.float32).max,
                    device=global_encode_scale.device,
                    dtype=torch.float32,
                ),
            )
            if global_encode_scale == torch.tensor(0.0, device=x.device, dtype=torch.float32):
                global_encode_scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
            global_decode_scale = torch.div(1.0, global_encode_scale)

            decode_scale = decode_scale * global_encode_scale
            decode_scale = torch.min(
                decode_scale,
                torch.tensor(
                    torch.finfo(torch.float32).max,
                    device=decode_scale.device,
                    dtype=torch.float32,
                ),
            )
            decode_scale = torch.clamp(decode_scale, min=-FLOAT8_E4M3_MAX, max=FLOAT8_E4M3_MAX)
            decode_scale = decode_scale.to(torch.float8_e4m3fn)

            encode_scale = torch.min(
                torch.div(1.0, decode_scale.to(torch.float32) * global_decode_scale),
                torch.tensor(
                    torch.finfo(torch.float32).max,
                    device=decode_scale.device,
                    dtype=torch.float32,
                ),
            )

        scaled_x = x.to(torch.float32) * encode_scale

        clipped_x = torch.clamp(scaled_x, -FLOAT4_E2M1_MAX, FLOAT4_E2M1_MAX).reshape(m, n)

        return cast_to_fp4x2(clipped_x), decode_scale.squeeze(-1)

    @staticmethod
    def _pad_tensor(
        tensor: torch.Tensor, row_divisor: Optional[int], col_divisor: Optional[int]
    ) -> torch.Tensor:

        assert tensor.dim() == 2, "only supports 2D tensors"
        M, N = tensor.shape
        padding_needed_rows = 0
        padding_needed_cols = 0

        if row_divisor is not None and M % row_divisor != 0:
            padding_needed_rows = row_divisor - (M % row_divisor)
        # Check and calculate column padding if col_divisor is provided
        if col_divisor is not None and N % col_divisor != 0:
            padding_needed_cols = col_divisor - (N % col_divisor)

        # Return original tensor if no padding is needed
        if padding_needed_rows == 0 and padding_needed_cols == 0:
            return tensor

        # pad the tensor
        out = torch.nn.functional.pad(
            tensor,
            (0, padding_needed_cols, 0, padding_needed_rows),
            mode="constant",
            value=0.0,
        ).contiguous()

        return out

    @staticmethod
    def _rm_pad_tensor(tensor: torch.Tensor, original_size: tuple[int, ...]) -> torch.Tensor:

        assert tensor.dim() == 2, "only supports 2D tensors"
        M, N = original_size
        out = tensor[:M, :N].contiguous()
        return out

    def _quantize(self, tensor: torch.Tensor) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Python implementation of microblock FP4 quantization.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor to quantize (should be 2D)

        Returns
        -------
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], torch.Tensor]
            (qx, sx, qx_t, sx_t, global_amax) where:
            - qx: quantized data in row-major order (if rowwise_usage), None otherwise
            - sx: scale tensor for qx (if rowwise_usage), None otherwise
            - qx_t: quantized data in column-major order (if columnwise_usage), None otherwise
            - sx_t: scale tensor for qx_t (if columnwise_usage), None otherwise
            - global_amax: global amax tensor
        """
        if self.pow_2_scales:
            assert self.quant_tile_shape == (
                1,
                32,
            ), "MXFP4 only supports 1x32 tile shape."
            # TODO(etsykunov): Fix bug where global_amax_row and
            # global_amax_col are not defined
            # global_amax = torch.empty(0, device=tensor.device, dtype=torch.float32)
        else:
            assert self.quant_tile_shape in (
                (1, 16),
                (16, 16),
            ), "NVFP4 only supports 1x16 or 16x16 tile shape."
            # Prepare inputs once so we can reuse for both amax and quantization
            # Row-input will always be the original input.
            row_input = tensor
            col_input = (
                self._apply_rht(tensor.t().contiguous())
                if self.with_rht
                else tensor.t().contiguous()
            )
            # Compute amax for rowwise and columnwise paths separately
            global_amax_row = torch.max(torch.abs(row_input)).to(torch.float32).view(1)
            global_amax_col = (
                torch.max(torch.abs(col_input)).to(torch.float32).view(1)
                if self.columnwise_usage
                else global_amax_row
            )

        transpose_scales = False

        M, N = tensor.shape
        if self.rowwise_usage:
            x_input = row_input
            x_padded = self._pad_tensor(
                x_input, row_divisor=self.quant_tile_shape[0], col_divisor=self.quant_tile_shape[1]
            )

            qx, sx = self._quantize_blockwise_reference(
                x_padded,
                global_amax_row,
                self.quant_tile_shape[1],
                self.quant_tile_shape[0],
                pow_2_scales=self.pow_2_scales,
                eps=self.eps,
            )
            if transpose_scales:
                sx = sx.T

            qx = self._rm_pad_tensor(qx, (M, N // 2))

        else:
            qx = None
            sx = None

        if self.columnwise_usage:
            x_t = col_input
            x_t_padded = self._pad_tensor(
                x_t, row_divisor=self.quant_tile_shape[0], col_divisor=self.quant_tile_shape[1]
            )

            qx_t, sx_t = self._quantize_blockwise_reference(
                x_t_padded,
                global_amax_col,
                self.quant_tile_shape[1],
                self.quant_tile_shape[0],
                pow_2_scales=self.pow_2_scales,
                eps=self.eps,
            )

            qx_t = self._rm_pad_tensor(qx_t, (N, M // 2))

            if transpose_scales:
                sx_t = sx_t.T
        else:
            qx_t = None
            sx_t = None

        return qx, sx, qx_t, sx_t, global_amax_row, global_amax_col

    def quantize(
        self,
        tensor: torch.Tensor,
        **kwargs,  # pylint: disable=unused-argument
    ) -> NVFP4TensorRef:
        # sanity checks
        assert tensor.dtype in utils.HIGH_PRECISION_FLOAT_DTYPES, "Unsupported input dtype."

        # Make it work with 3D tensors
        original_shape = tensor.shape
        if tensor.ndim > 2:
            tensor = tensor.view(-1, tensor.shape[-1])

        qx, sx, qx_t, sx_t, global_amax_row, global_amax_col = self._quantize(tensor)

        return NVFP4TensorRef(
            data=qx,
            scale=sx,
            data_t=qx_t,
            scale_t=sx_t,
            global_amax_row=global_amax_row,
            global_amax_col=global_amax_col,
            dtype=tensor.dtype,
            device=tensor.device,
            quant_dtype=self.dtype,
            quantizer=self,
            original_shape=original_shape,
        )

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: ExperimentalQuantizedTensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> ExperimentalQuantizedTensor:
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

        qx, sx, qx_t, sx_t, global_amax = self._quantize(src)

        # Update the destination with new data
        dst.data = qx
        dst.scale = sx
        dst.data_t = qx_t
        dst.scale_t = sx_t
        dst.global_amax = global_amax
        dst.dtype = src.dtype
        dst.quant_dtype = self.dtype
        dst.original_shape = original_shape

        return dst

    @property
    def supports_allgather_fp8(self) -> bool:
        """Whether the tensor data can be all-gathered with an FP8 all-gather.

        TODO(etsykunov): Confirm docstring is correct. Also, this API
        seems too FP8-specific and should be reconsidered.
        """
        return False

    def transpose_qresult(
        self, qresult: quantization.ExperimentalQuantizedTensor
    ) -> quantization.ExperimentalQuantizedTensor:
        """Convert row-wise data to column-wise data (?)

        TODO(etsykunov): Confirm docstring is correct.
        """
        raise NotImplementedError("Transpose qresult is not implemented for FP4.")

    @property
    def supports_dequantize(self) -> bool:
        """Whether quantized tensor can converted to high-precision tensor"""
        return False

    @property
    def is_data_t_transposed_in_memory(self) -> bool:
        """Whether column-wise data is stored in transposed layout.

        TODO(etsykunov): Confirm docstring is correct.
        """
        raise NotImplementedError("Not implemented yet")

    def dequantize(
        self, tensor: torch.Tensor, scale: torch.Tensor, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        """Dequantize the quantized tensor"""
        raise NotImplementedError("Not implemented yet")

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
        gemm_type: quantization.GEMMType = quantization.GEMMType.FPROP,
        qresult_x: quantization.ExperimentalQuantizedTensor | None = None,
        qresult_w: quantization.ExperimentalQuantizedTensor | None = None,
    ) -> torch.Tensor:
        assert bias is None, "Bias is implemented for FP4 GEMM."

        high_precision_x = cast_from_fp4x2(qx, out_dtype)
        high_precision_w = cast_from_fp4x2(qw, out_dtype)

        if self.pow_2_scales:

            if sx.dtype == torch.uint8:
                # if scaling factor is stored in uint8 container
                sx = torch.tensor(2.0, device=sx.device, dtype=torch.float32) ** (
                    (
                        sx.to(torch.float32)
                        - torch.tensor(127, device=sx.device, dtype=torch.float32)
                    )
                )
                sw = torch.tensor(2.0, device=sw.device, dtype=torch.float32) ** (
                    (
                        sw.to(torch.float32)
                        - torch.tensor(127, device=sw.device, dtype=torch.float32)
                    )
                )
            else:
                # if scaling factor is torch.float8_e8m0fnu
                sx = sx.to(torch.float32)
                sw = sw.to(torch.float32)

            alpha = torch.tensor(1.0, device=high_precision_x.device, dtype=torch.float32)

        else:

            assert qresult_x is not None
            assert qresult_w is not None

            assert qresult_x.global_amax_row is not None
            assert qresult_w.global_amax_col is not None

            sx = sx.to(torch.float32)
            sw = sw.to(torch.float32)

            factor = 6.0 * 6.0 * 448.0 * 448.0

            if gemm_type == quantization.GEMMType.WGRAD:
                partial_alpha = qresult_x.global_amax_col * qresult_w.global_amax_col
            else:
                partial_alpha = qresult_x.global_amax_row * qresult_w.global_amax_row
            alpha = torch.div(partial_alpha, factor).squeeze(-1)

        M, K = high_precision_x.shape
        N, K_w = high_precision_w.shape
        assert K == K_w, "K dimension mismatch between qx and qw"

        assert K % 32 == 0, "K dimension must be divisible by 32"
        assert N % 8 == 0, "N dimension must be divisible by 8"

        block_length = 32 if self.pow_2_scales else 16

        grid_k = K // block_length

        assert sx.shape == (
            M,
            K // block_length,
        ), f"sx shape mismatch: expected ({M}, {K//block_length}), got {sx.shape}"
        assert sw.shape == (
            N,
            K // block_length,
        ), f"sw shape mismatch: expected ({N}, {K//block_length}), got {sw.shape}"

        y = torch.zeros(M, N, dtype=torch.float32, device=qx.device)

        # below implementation is to match the FP4 tensor core implementation
        # Each output element (i, j) is fp32 accumulation of (K // block_length) inner products
        # Each inner product is sx * sw * (1, block_length) x (block_length, 1) with precision in fp32
        # Then batch the computation in M, N dimension
        for k in range(grid_k):
            k_start = k * block_length
            k_end = k_start + block_length

            qx_block = high_precision_x[:, k_start:k_end].clone().contiguous()
            qw_block = high_precision_w[:, k_start:k_end].clone().contiguous()

            # Extract scaling factors for the current blocks
            sx_block = sx[:, k]
            sw_block = sw[:, k]

            y += torch.outer(sx_block, sw_block) * high_precision_gemm_ref(
                qx_block, qw_block, torch.float32, is_b_transposed=True
            )

        if not self.pow_2_scales and K > 0:
            # only apply global scale for NVFP4 and non-empty cases
            y = alpha * y

        # accumulation happens at epilogue in float32
        if accumulate:
            assert out is not None, "Output tensor must be provided for accumulation."
            y += out.to(torch.float32)
        else:
            assert out is None, "Output tensor should be None when accumulate is False."

        y = y.to(out_dtype)
        return y
