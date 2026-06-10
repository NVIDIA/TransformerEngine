# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Mixin class holding data specific for Float8BlockwiseQTensor"""

from __future__ import annotations
import math
from typing import Optional, Dict, Any, Tuple, Union
import torch

import transformer_engine_torch as tex

from ...quantized_tensor import QuantizedTensorStorage, Quantizer

from ...constants import TE_DType_To_Torch, DType

from ...utils import _empty_tensor, round_up_to_nearest_multiple


class Float8BlockwiseQTensorStorage(QuantizedTensorStorage):
    """Mixin class that holds data attributes of Float8BlockwiseQTensor.

    Float8BlockwiseQTensor inherits from the PyTorch tensor class and this
    mixin class. If this class is instantiated directly, it has the same
    data, lower CPU overhead, and less functionality. It should only
    be instantiated directly for performance-critical internal usage.
    """

    _rowwise_data: Optional[torch.Tensor]
    _columnwise_data: Optional[torch.Tensor]
    _quantizer: Quantizer
    _fp8_dtype: DType
    _rowwise_scale_inv: Optional[torch.Tensor]
    _columnwise_scale_inv: Optional[torch.Tensor]
    _is_2D_scaled: bool

    def __new__(
        cls,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        fp8_dtype: Union[DType, tex.DType],
        quantizer: Quantizer,
        is_2D_scaled: bool,
        *args,
        fake_dtype: Optional[torch.dtype] = None,
        **kwargs,
    ):
        if cls is Float8BlockwiseQTensorStorage:
            instance = object.__new__(cls)
            instance._dtype = fake_dtype if fake_dtype is not None else torch.float32
        else:
            instance = super().__new__(cls, *args, fake_dtype=fake_dtype, **kwargs)
        instance._rowwise_data = rowwise_data
        instance._columnwise_data = columnwise_data
        instance._quantizer = quantizer.copy() if quantizer is not None else None
        instance._fp8_dtype = DType.cast(fp8_dtype)
        instance._rowwise_scale_inv = rowwise_scale_inv
        instance._columnwise_scale_inv = columnwise_scale_inv
        instance._is_2D_scaled = is_2D_scaled

        return instance

    def clear(self):
        """Deallocate this tensor's memory. Typically not needed and must be used carefully."""
        for t in (
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ):
            if t is not None:
                t.data = _empty_tensor()

    def copy_from_storage(self, src: QuantizedTensorStorage) -> None:
        """Copy data buffers from another Float8BlockwiseQTensorStorage."""
        if not isinstance(src, Float8BlockwiseQTensorStorage):
            raise TypeError("copy_from_storage expects Float8BlockwiseQTensorStorage")
        if self._fp8_dtype != src._fp8_dtype:
            raise RuntimeError("FP8 dtype mismatch in copy_from_storage")
        if self._is_2D_scaled != src._is_2D_scaled:
            raise RuntimeError("Scale layout mismatch in copy_from_storage")

        def _copy_optional(dst: Optional[torch.Tensor], src_tensor: Optional[torch.Tensor]):
            if dst is not None and src_tensor is not None:
                dst.copy_(src_tensor)

        _copy_optional(self._rowwise_data, src._rowwise_data)
        _copy_optional(self._columnwise_data, src._columnwise_data)
        _copy_optional(self._rowwise_scale_inv, src._rowwise_scale_inv)
        _copy_optional(self._columnwise_scale_inv, src._columnwise_scale_inv)

    def get_metadata(self) -> Dict[str, Any]:
        """Get this tensor's metadata."""
        return {
            "rowwise_data": self._rowwise_data,
            "rowwise_scale_inv": self._rowwise_scale_inv,
            "columnwise_data": self._columnwise_data,
            "columnwise_scale_inv": self._columnwise_scale_inv,
            "fp8_dtype": self._fp8_dtype,
            "quantizer": self._quantizer,
            "is_2D_scaled": self._is_2D_scaled,
            "fake_dtype": self._dtype,
        }

    def prepare_for_saving(
        self,
    ) -> Tuple[list[Optional[torch.Tensor]], Float8BlockwiseQTensorStorage]:
        """
        Prepare the tensor base for saving for backward
        """
        tensors = [
            self._rowwise_data,
            self._columnwise_data,
            self._rowwise_scale_inv,
            self._columnwise_scale_inv,
        ]
        self._rowwise_data = None
        self._columnwise_data = None
        self._rowwise_scale_inv = None
        self._columnwise_scale_inv = None
        return tensors, self

    def restore_from_saved(
        self, tensors: list[Optional[torch.Tensor]]
    ) -> list[Optional[torch.Tensor]]:
        """Restore the tensor base data from the saved tensors list."""
        self._rowwise_data = tensors[0]
        self._columnwise_data = tensors[1]
        self._rowwise_scale_inv = tensors[2]
        self._columnwise_scale_inv = tensors[3]
        return tensors[4:]

    def get_data_tensors(self, rowwise_data: bool = True, columnwise_data: bool = True):
        """Get this Tensor's data."""
        if rowwise_data and columnwise_data:
            return self._rowwise_data, self._columnwise_data
        if rowwise_data:
            return self._rowwise_data
        if columnwise_data:
            return self._columnwise_data
        raise ValueError("No data to get, both rowwise_data and columnwise_data are False")

    def _transpose_dq_columnwise_output(self, columnwise_dq: torch.Tensor) -> torch.Tensor:
        """Takes dequantized columnwise data and permutes to a rowwise shape"""
        if columnwise_dq.dim() < 2:
            return columnwise_dq
        permute_dims = list(range(1, columnwise_dq.dim()))
        permute_dims.append(0)
        return torch.permute(columnwise_dq, tuple(permute_dims)).contiguous()

    def _dequantize_vectorwise(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if dtype is None:
            dtype = self._dtype
        block_len = 128

        q_M, q_K = 1, 1
        if self._rowwise_data is not None:
            q = self._rowwise_data
            scale_inv = self._rowwise_scale_inv
            transpose_output = False
            if len(q.shape) >= 1:
                q_K = q.shape[-1]
            for i in range(len(q.shape) - 1):
                q_M *= q.shape[i]
            inner_q_dimension_tiled = True
            scales_tiled_dim, scales_untiled_dim = scale_inv.shape
        else:
            assert self._columnwise_data is not None, "No data to dequantize"
            q = self._columnwise_data
            scale_inv = self._columnwise_scale_inv
            scales_tiled_dim, scales_untiled_dim = scale_inv.shape
            inner_q_dimension_tiled = True
            transpose_output = True
            if len(q.shape) >= 1:
                q_M = q.shape[0]
            for i in range(1, len(q.shape)):
                q_K *= q.shape[i]

        orig_shape = q.shape
        q = q.reshape(q_M, q_K)
        if inner_q_dimension_tiled:
            if q_K % block_len != 0:
                k_pad_amount = (block_len - (q_K % block_len)) % block_len
                q = torch.nn.functional.pad(
                    q, (0, k_pad_amount, 0, 0), mode="constant", value=0
                ).contiguous()
            padded_M, padded_K = q.shape
            q_tiled = q.reshape(q_M, scales_tiled_dim, block_len)
        else:
            if q_M % block_len != 0:
                m_pad_amount = (block_len - (q_M % block_len)) % block_len
                q = torch.nn.functional.pad(
                    q, (0, 0, 0, m_pad_amount), mode="constant", value=0
                ).contiguous()
            padded_M, padded_K = q.shape
            q_tiled = q.reshape(scales_tiled_dim, block_len, q_K)
        if scales_untiled_dim > q_M:
            # untiled scale dimension is 4 element aligned.
            scale_inv = scale_inv[:, :q_M].contiguous()
        dq_scale = scale_inv.transpose(-2, -1).contiguous().reshape(q_M, scales_tiled_dim, 1)
        torch_q_dtype = TE_DType_To_Torch[self._fp8_dtype]
        result = q_tiled.view(torch_q_dtype).to(torch.float32) * dq_scale
        if padded_M != q_M or padded_K != q_K:
            result = result.reshape(padded_M, padded_K)[:q_M, :q_K]
        result = result.to(dtype)
        if len(orig_shape) == 0:
            result = result.reshape([])
        else:
            result = result.reshape(*orig_shape).contiguous()

        if transpose_output:
            return self._transpose_dq_columnwise_output(result)
        return result

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """
        Construct plain PyTorch tensor from Float8BlockwiseQTensor
        """
        if dtype is None:
            dtype = self._dtype

        if self._rowwise_data is not None and self._rowwise_data.numel() == 0:
            return torch.empty(self.size(), dtype=dtype, device=self.device)

        block_len = 128
        if not self._is_2D_scaled:
            return self._dequantize_vectorwise(dtype=dtype)

        def format_scale_as_logical_shape(q_K, scales, block_len):
            # The GEMM for 2D blocks required padding in the scales.
            derived_scale_k_shape = math.ceil(q_K / block_len)
            _, scale_K = scales.shape
            if derived_scale_k_shape == scale_K:
                return scales
            return scales[:, :derived_scale_k_shape].contiguous()

        q_M, q_K = 1, 1
        if self._rowwise_data is not None:
            q = self._rowwise_data
            scale_inv = self._rowwise_scale_inv
            transpose_output = False
            if len(q.shape) >= 1:
                q_K = q.shape[-1]
            for i in range(len(q.shape) - 1):
                q_M *= q.shape[i]
        else:
            assert self._columnwise_data is not None, "No data to dequantize"
            q = self._columnwise_data
            scale_inv = self._columnwise_scale_inv
            transpose_output = True
            if len(q.shape) >= 1:
                q_M = q.shape[0]
            for i in range(1, len(q.shape)):
                q_K *= q.shape[i]

        orig_shape = q.shape
        q = q.reshape(q_M, q_K)
        formatted_scales = format_scale_as_logical_shape(q_K, scale_inv, block_len)
        assert len(formatted_scales.shape) == 2
        m_tiles, k_tiles = formatted_scales.shape
        unpadded_m, unpadded_k = q_M, q_K
        m_block_len = block_len
        k_block_len = block_len
        if q_M % m_block_len != 0 or q_K % k_block_len != 0:
            m_pad_amount = (m_block_len - (q_M % m_block_len)) % m_block_len
            k_pad_amount = (k_block_len - (q_K % k_block_len)) % k_block_len
            q = torch.nn.functional.pad(
                q, (0, k_pad_amount, 0, m_pad_amount), mode="constant", value=0
            ).contiguous()
        padded_M, padded_K = q.shape
        q_tiled = q.reshape(m_tiles, m_block_len, k_tiles, k_block_len)

        torch_q_dtype = TE_DType_To_Torch[self._fp8_dtype]

        result = q_tiled.view(torch_q_dtype).to(torch.float32) * formatted_scales.view(
            m_tiles, 1, k_tiles, 1
        )
        result = result.view(padded_M, padded_K).to(dtype)
        if padded_M != unpadded_m or padded_K != unpadded_k:
            result = result[:unpadded_m, :unpadded_k]
        if len(orig_shape) == 0:
            result = result.reshape([])
        else:
            result = result.reshape(*orig_shape).contiguous()
        if transpose_output:
            return self._transpose_dq_columnwise_output(result)
        return result

    def size(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        if self._rowwise_data is not None:
            return self._rowwise_data.size(*args, **kwargs)
        dims = list(self._columnwise_data.size(*args, **kwargs))
        reordered = []
        for i in range(1, len(dims)):
            reordered.append(dims[i])
        reordered.append(dims[0])
        return torch.Size(reordered)

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("Float8BlockwiseQTensorStorage has no data!")

    def _create_columnwise(self):
        """
        Update columnwise data and columnwise scale inv. Can only be used when using 2D scaling.
        """
        assert self._is_2D_scaled, "Cannot create columnwise data when not using 2D scaling."

        rowwise_data = self._rowwise_data
        if not rowwise_data.is_contiguous():
            rowwise_data = rowwise_data.contiguous()
        self._columnwise_data = tex.fp8_transpose(
            rowwise_data, self._fp8_dtype, out=self._columnwise_data
        )

        if self._columnwise_scale_inv is None:
            assert self._quantizer is not None, (
                "._quantizer of Float8BlockwiseQTensor cannot be None because all the blockwise "
                "quantized tensors are supposed to be generated from the quantizer."
            )
            columnwise_scale_inv_shape = self._quantizer.get_scale_shape(rowwise_data.shape, True)
            self._columnwise_scale_inv = torch.empty(
                columnwise_scale_inv_shape,
                dtype=self._rowwise_scale_inv.dtype,
                device=self._rowwise_scale_inv.device,
            )
        assert len(self._rowwise_scale_inv.shape) == 2
        assert len(self._columnwise_scale_inv.shape) == 2
        rowwise_scale_inv = self._rowwise_scale_inv
        columnwise_scale_inv = rowwise_scale_inv.transpose(-2, -1)
        h = min(self._columnwise_scale_inv.shape[0], columnwise_scale_inv.shape[0])
        w = min(self._columnwise_scale_inv.shape[1], columnwise_scale_inv.shape[1])
        self._columnwise_scale_inv[0:h, 0:w].copy_(columnwise_scale_inv[0:h, 0:w])

    def _transpose_columnwise_data(self):
        """Plainly transpose the columnwise data and scale inv."""
        if self._columnwise_data is not None:
            # TODO(yuzhongw, tmoon): Figure out why _old_data is not automatically
            # deallocated by GC. Manually deallocating is a temporary hack.
            _old_data = self._columnwise_data
            self._columnwise_data = tex.fp8_transpose(
                self._columnwise_data, self._fp8_dtype, out=None
            )
            _old_data.data = _empty_tensor()
            del _old_data

    def __repr__(self):
        if self._rowwise_data is not None:
            data = self.dequantize()
            descriptor = "rowwise"
        else:
            data = self.dequantize()
            descriptor = "columnwise"
        return (
            "Float8BlockwiseQTensorStorage("
            f"fp8_dtype={self._fp8_dtype}, "
            f"{descriptor}_scaled_data={data})"
        )

    def update_usage(
        self, rowwise_usage: Optional[bool] = None, columnwise_usage: Optional[bool] = None
    ):
        """
        update_usage can be used to clear out one of two possible copies of the data.
        """

        if rowwise_usage is None:
            rowwise_usage = self._rowwise_data is not None
        if columnwise_usage is None:
            columnwise_usage = self._columnwise_data is not None
        assert (
            columnwise_usage or rowwise_usage
        ), "Must retain some data either columnwise or rowwise"

        if columnwise_usage and rowwise_usage:
            if not self._is_2D_scaled:
                # For 1D scaling, we cannot create columnwise data/scale_inv from rowwise
                # data/scale_inv because their scale values are different.
                assert (
                    self._rowwise_data is not None
                    and self._rowwise_scale_inv is not None
                    and self._columnwise_data is not None
                    and self._columnwise_scale_inv is not None
                ), "Cannot update to rowwise and columnwise usage."
            else:
                # For 2D scaling, if columnwise data/scale_inv is None, we can create them from
                # rowwise data/scale_inv.
                assert (
                    self._rowwise_data is not None and self._rowwise_scale_inv is not None
                ), "Cannot update to rowwise and columnwise usage because rowwise data is None."
                if self._columnwise_data is None or self._columnwise_scale_inv is None:
                    self._create_columnwise()
            return

        if rowwise_usage:
            assert (
                self._rowwise_data is not None and self._rowwise_scale_inv is not None
            ), "Cannot update to rowwise usage."
            self._columnwise_data = None
            self._columnwise_scale_inv = None
            return
        if columnwise_usage:
            assert (
                self._columnwise_data is not None and self._columnwise_scale_inv is not None
            ), "Cannot update to columnwise usage."
            self._rowwise_data = None
            self._rowwise_scale_inv = None
            return

        return

    def get_usages(self) -> Dict[str, bool]:
        """Get the usage of the tensor"""
        return {
            "rowwise": self._rowwise_data is not None,
            "columnwise": self._columnwise_data is not None,
        }

    # ── FSDP2 sub-storage buffer protocol ────────────────────────────
    #
    # Float8Block stores columnwise data N-major (transposed) for the GEMM, so
    # it cannot be dim-0 all-gathered directly. Each direction is made
    # self-contained: the columnwise direction fp8-transposes its own data to
    # M-major for the gather and back on assign, using only its own buffers (no
    # dependency on a rowwise sibling, which in a hybrid tensor may be a
    # different format). Block-scale GEMM alignment padding (round-up-to-4) is
    # stripped before the gather and re-applied after. Only 2D block scaling is
    # supported -- the 1D scale layout has M in dim1, incompatible with FSDP2's
    # dim-0 all-gather.

    _FSDP_BLOCK_LEN = 128

    def _fsdp_logical_mn(self) -> Tuple[int, int]:
        """Flattened ``(M, N)`` of this sub-storage's logical shape."""
        shape = self.size()
        last_dim = shape[-1] if len(shape) > 0 else 1
        leading = 1
        for dim in shape[:-1]:
            leading *= dim
        return leading, last_dim

    def fsdp_buffer_fields(self) -> Tuple[str, ...]:
        """Fields gathered by FSDP2 for Float8 block scaling (2D scaling only)."""
        if not self._is_2D_scaled:
            raise NotImplementedError(
                "FSDP2 for Float8BlockwiseQTensor requires 2D block scaling "
                "(block_scaling_dim=2). 1D block scaling is not supported because "
                "its scale layout has M in dim1, which is incompatible with FSDP2 "
                "dim-0 all-gather."
            )
        fields = []
        if self._rowwise_data is not None:
            fields.extend(("_rowwise_data", "_rowwise_scale_inv"))
        if self._columnwise_data is not None:
            fields.extend(("_columnwise_data", "_columnwise_scale_inv"))
        return tuple(fields)

    def fsdp_extract_buffers(
        self,
    ) -> Tuple[Tuple[Optional[torch.Tensor], ...], Dict[str, Any]]:
        """Extract M-major, alignment-stripped buffers for dim-0 all-gather.

        Rowwise data is already M-major; columnwise data is N-major and is
        fp8-transposed to M-major here (and transposed back in
        :meth:`fsdp_assign_gathered`). The block-scale round-up-to-4 alignment
        padding is stripped so dim-0 concatenation across shards is well-defined.
        """
        names = self.fsdp_buffer_fields()
        block_len = self._FSDP_BLOCK_LEN
        m, n = self._fsdp_logical_mn()
        m_tiles = (m + block_len - 1) // block_len
        last_tiles = (n + block_len - 1) // block_len

        if self._rowwise_data is not None:
            # Rowwise scale is (m_tiles, round_up(last_tiles, 4)); m_tiles sits in
            # dim-0 (sharded/gathered) unpadded, the round-up padding is on dim-1
            # (not sharded). Strip dim-1 to the compact tile count.
            scale = self._rowwise_scale_inv
            if scale is not None and scale.size(1) > last_tiles:
                scale = scale[:, :last_tiles].contiguous()
            buffers = (self._rowwise_data, scale)
            direction = "rowwise"
        else:
            # Columnwise data is N-major (N, M); transpose to M-major (M, N).
            col_data = self._columnwise_data
            if not col_data.is_contiguous():
                col_data = col_data.contiguous()
            data_m = tex.fp8_transpose(col_data, self._fp8_dtype, out=None)
            # Columnwise scale is (last_tiles, round_up(m_tiles, 4)); transpose to
            # (round_up(m_tiles, 4), last_tiles) and strip dim-0 to m_tiles so the
            # gathered (dim-0) axis is the M-tiles, matching the rowwise layout.
            scale = self._columnwise_scale_inv.transpose(0, 1).contiguous()
            if scale.size(0) > m_tiles:
                scale = scale[:m_tiles].contiguous()
            buffers = (data_m, scale)
            direction = "columnwise"

        return buffers, {"direction": direction, "field_names": names}

    def fsdp_assign_gathered(
        self,
        gathered: Tuple[Optional[torch.Tensor], ...],
        meta: Dict[str, Any],
    ) -> None:
        """Write gathered buffers back, re-applying transpose + scale padding.

        Inverse of :meth:`fsdp_extract_buffers`: rowwise re-pads the scale's
        last-dim alignment; columnwise transposes the M-major gathered data back
        to N-major and re-pads/transposes the scale to the GEMM scale layout
        produced by ``get_scale_shape(..., columnwise=True)``.
        """
        block_len = self._FSDP_BLOCK_LEN
        direction = meta["direction"]
        data, scale = gathered

        if direction == "rowwise":
            last_dim = data.size(-1)
            last_tiles = (last_dim + block_len - 1) // block_len
            if scale is not None:
                pad = round_up_to_nearest_multiple(last_tiles, 4) - last_tiles
                if pad > 0:
                    scale = torch.nn.functional.pad(scale, (0, pad))
            self._rowwise_data = data
            self._rowwise_scale_inv = scale
            return

        # Columnwise: gathered data is M-major (M_full, N); transpose to N-major.
        data_m = data if data.is_contiguous() else data.contiguous()
        self._columnwise_data = tex.fp8_transpose(data_m, self._fp8_dtype, out=None)
        m_full = 1
        for dim in data.shape[:-1]:
            m_full *= dim
        m_tiles_full = (m_full + block_len - 1) // block_len
        # Gathered scale is compact (m_tiles_full, last_tiles); transpose to
        # (last_tiles, m_tiles_full) and re-pad the M-tile dim to multiple of 4.
        scale_t = scale.transpose(0, 1).contiguous()
        pad = round_up_to_nearest_multiple(m_tiles_full, 4) - m_tiles_full
        if pad > 0:
            scale_t = torch.nn.functional.pad(scale_t, (0, pad))
        self._columnwise_scale_inv = scale_t.contiguous()
