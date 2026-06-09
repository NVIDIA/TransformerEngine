# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tensor class with hybrid (mixed-format) quantized data"""
from __future__ import annotations
from collections.abc import Iterable
import math
from typing import Optional, Tuple, Union, Any
import warnings

import torch
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from transformer_engine.pytorch.ops.basic.quantize import Quantize
import transformer_engine_torch as tex
from transformer_engine_torch import DType as TE_DType

from transformer_engine.common.recipe import MXFP8BlockScaling, Recipe
from ..constants import MXFP8_BLOCK_SCALING_SIZE, NVFP4_BLOCK_SCALING_SIZE
from ..utils import canonicalize_shape, devices_match, round_up_to_nearest_multiple
from .storage.flex_tensor_storage import FlexTensorStorage, _FromFlexFunc
from ..quantized_tensor import QuantizedTensor, Quantizer
from ._quantization_helpers import _IdentityFunc

aten = torch.ops.aten

class FlexQuantizer(Quantizer):
    """Builder class for Flex tensors that are quantized in potentially both directions

    High-precision tensors (e.g. in FP32 or BF16) are quantized in row-wise and column-wise
    directions with potentially different quantization formats. For example, it can be
    quantized in MXFP8 in one direction and NVFP4 in another, or simply only quantized in only one.

    The quantization & dequantization logic is implemented in
    the new experimental CuTeDSL kernel instead of the CUDA C++ kernel.
    """

    dtype_row: TE_DType
    dtype_column: TE_DType

    def __init__(
        self,
        *,
        dtype_row: Optional[TE_DType],
        dtype_column: Optional[TE_DType],
        quantize_func: str,
        dequantize_func: str,
        stochastic_rounding: bool = False,
    ):
        """
        Parameters
        ----------
        dtype_row, dtype_column : Optional[TE_DType]
            Per-direction quantized formats. ``None`` means that direction is
            not produced (the corresponding data/scale/amax slots are passed as
            None to the kernel).
        quantize_func, dequantize_func : str
            Names of the tvm-ffi global functions (registered CuTeDSL kernels)
            that C++ ``FlexQuantizer::quantize`` / ``::dequantize`` resolve by
            name (``Function::GetGlobal``) and call.

            These names double as the kernel **cache key**, and that key MUST
            encode *exactly* everything the kernel was specialized for at
            compile time — every constexpr the ``@cute.jit`` baked in: the
            high-precision input dtype, the per-direction quantized formats,
            the swizzle flag, and any baked tensor extents (e.g. a literal N).
            Two requirements follow:

            * If a constexpr changes, the name MUST change too (recompile +
              register under the new name). The name is the contract: a given
              name *guarantees* a kernel specialized for that exact signature.
            * Conversely, anything NOT baked (e.g. a sym-int dim) must NOT
              appear in the name, or you fragment the cache and recompile
              needlessly.

            Because the name is the only thing C++ carries to the call site,
            there is no deeper signature check: if the constexpr signature and
            the name ever disagree, the lookup simply misses. C++ then raises
            (see ``call_tvm_ffi``): a miss means no kernel was registered for
            this exact signature — the constexpr guarantee is broken — and that
            is surfaced as an error rather than silently mis-dispatching.
        stochastic_rounding : bool
            If True, quantize() mints a fresh RNG state (seed/offset) from the
            torch CUDA generator each call and passes it in the ``rng_state``
            slot below. The kernel must be a variant compiled with SR enabled.
            (Not implemented yet — C++ throws when this is set.)

        tvm-ffi calling protocol (POSITIONAL — there is no runtime signature
        check beyond what the compiled kernel itself enforces; both functions
        MUST accept arguments in exactly this order):

            idx  quantize_func           dequantize_func         dtype
            ---  ----------------------  ----------------------  -----------------
             0   mX   (high-prec INPUT)  mO   (high-prec OUTPUT)  fp32/bf16/fp16
             1   mO_row    data  OUTPUT  mX_row    data  INPUT    uint8 (fp8/fp4)
             2   mS_row    scale OUTPUT  mS_row    scale INPUT    uint8 (e8m0/e4m3)
             3   mA_row    amax  OUTPUT  mA_row    amax  INPUT    fp32 (None: MXFP8)
             4   mO_col    data  OUTPUT  mX_col    data  INPUT    uint8 (fp8/fp4)
             5   mS_col    scale OUTPUT  mS_col    scale INPUT    uint8 (e8m0/e4m3)
             6   mA_col    amax  OUTPUT  mA_col    amax  INPUT    fp32 (None: MXFP8)
             7   rng_state (SR seed)     stream                  int64 / handle
             8   stream                  --                      int64 handle

        Both functions share the SAME order for slots 0..6 — the only
        difference is direction: for quantize, slot 0 is the high-precision
        INPUT and slots 1..6 are the quantized OUTPUTS; for dequantize it is
        reversed (slot 0 is the high-precision OUTPUT, slots 1..6 the quantized
        INPUTS). Per-direction slots are grouped {data, scale_inv, amax}; a
        disabled direction (dtype is None) or a format without amax (e.g. MXFP8)
        passes None for that slot. quantize_func additionally takes ``rng_state``
        (slot 7, None unless stochastic_rounding) before the trailing CUDA
        ``stream`` (passed as an int64 handle); dequantize_func omits rng_state,
        so its ``stream`` is slot 7.

        What each slot is for (so a kernel can take only what it needs and the
        rest are passed as None):

        - mX / mO (slot 0): the high-precision tensor. ALWAYS present — the
          value being quantized (quantize: read) or reconstructed (dequantize:
          write).

        - Row-wise group (slots 1..3) and column-wise group (slots 4..6): the
          tensor quantized along two orientations. Block scaling is applied
          along the last (contiguous) axis for the row-wise output and along
          the first axis for the column-wise output. Training needs BOTH
          because the forward and backward GEMMs consume the operand in
          opposite layouts (e.g. ``x`` row-wise for fprop, column-wise for the
          wgrad/dgrad GEMM). Inference or a one-sided use needs only ONE — set
          the unused direction's dtype to None and its three slots arrive as
          None.

            * data  (slots 1 / 4): the packed FP8/FP4 bytes for that direction.
              The actual quantized payload; produce it for any direction you
              want to keep.
            * scale_inv (slots 2 / 5): the per-block scale factors for that
              direction (E8M0 for MXFP8, E4M3 for NVFP4). Always paired with
              ``data`` — a block-scaled format is meaningless without its
              scales, so if you produce ``data`` you must produce ``scale_inv``.
            * amax (slots 3 / 6): the per-direction amax (max |x|). Only formats
              whose scale derives from a tensor/global amax need this (e.g.
              NVFP4's FP32 global scale, current-scaling recipes). MXFP8 picks a
              power-of-two scale per 32-block directly from the block, so it
              needs NO amax → pass None.

        - rng_state (quantize slot 7): seed/offset for the stochastic-rounding
          RNG. Only needed if the kernel does stochastic rounding; otherwise
          None (and leave ``stochastic_rounding=False``). Absent from
          dequantize_func entirely.

        - stream (trailing): the CUDA stream to launch on, passed by C++ as an
          int64 handle (decoded as an opaque pointer). Always present.
        """
        if dtype_row is None and dtype_column is None:
            raise ValueError(
                "FlexQuantizer requires at least one direction to be quantized, "
                "but both dtype_row and dtype_column are None."
            )
        super().__init__(rowwise=dtype_row is not None, columnwise=dtype_column is not None)
        self.dtype_row = dtype_row
        self.dtype_column = dtype_column
        self.quantize_func = quantize_func
        self.dequantize_func = dequantize_func
        # FIXME(flex): kernels are bound to one (cfg, shape). Take an is_valid(tensor)
        # and a recompile(tensor) -> new func name per direction; in quantize_impl/
        # dequantize, recompile when invalid instead of erroring on shape change.
        # When True, quantize() mints a fresh RNG state (seed/offset) from the
        # torch CUDA generator each call and passes it to the kernel as the
        # trailing arg, for stochastic rounding. The kernel must be a variant
        # compiled with stochastic rounding enabled.
        self.stochastic_rounding = stochastic_rounding

    def copy(self) -> FlexQuantizer:
        """Create shallow copy"""

        quantizer = FlexQuantizer(
            dtype_row=self.dtype_row,
            dtype_column=self.dtype_column,
            quantize_func=self.quantize_func,
            dequantize_func=self.dequantize_func,
            stochastic_rounding=self.stochastic_rounding,
        )
        quantizer.internal = self.internal
        quantizer.optimize_for_gemm = self.optimize_for_gemm
        return quantizer

    def update_quantized(
        self,
        src: torch.Tensor,
        dst: QuantizedTensor,
        *,
        noop_flag = None
    ) -> QuantizedTensor:
        assert isinstance(dst, FlexTensor), f"Cannot store quantized MXFP8 in {type(dst)} type."

        # Make sure input is in expected format
        if not devices_match(src.device, dst.device):
            src = src.to(device=dst.device)
        if not src.is_contiguous():
            src = src.contiguous()

        # Launch cast kernel
        tex.quantize(src, self, dst, noop_flag)

        # Update quantized tensor metadata
        dst._dtype_row = self.dtype_row
        dst._dtype_column = self.dtype_column

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

    def calibrate(self, tensor: torch.Tensor) -> None:
        pass  # Calibration is no-op since this supports only blockwise quantization, which doesn't require calibration.

    def get_scale_shape(
        self,
        shape: Iterable[int],
        columnwise: bool,
        dtype: TE_DType
    ) -> Tuple[int, int]:
        """Calculate the shape of the scaling tensor for Flex quantization.

        Parameters
        ----------
        shape : Iterable[int]
            Shape of the input tensor to be quantized
        columnwise : bool
            Whether to use columnwise scaling (True) or rowwise scaling (False)

        Returns
        -------
        Tuple[int, int]
            Shape of the scaling tensor as (outer_dim, inner_dim)
            For MXFP8 1D blockwise quantization, blocksize is 32
            For NXFP4 1D blockwise quantization, blocksize is 16
                - If columnwise: (round_to_multiple(K, 128), round_to_multiple(roundup(M / 16), 4))
                - If rowwise: (round_to_multiple(M, 128), round_to_multiple(roundup(K / 16), 4))
        Swizzle kernel will be performed before GEMM to suit the need of CuBLAS.
        CuBLAS doc: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
        """
        if FlexTensor.is_nvfp4_dtype(dtype):
            M, K = 1, 1
            M = math.prod(shape[:-1])
            K = shape[-1]
            if columnwise:
                outer = round_up_to_nearest_multiple(K, 128)
                inner = round_up_to_nearest_multiple(math.ceil(M / NVFP4_BLOCK_SCALING_SIZE), 4)
                return (outer, inner)
            outer = round_up_to_nearest_multiple(M, 128)
            inner = round_up_to_nearest_multiple(math.ceil(K / NVFP4_BLOCK_SCALING_SIZE), 4)
            return (outer, inner)

        elif FlexTensor.is_mxfp8_dtype(dtype):
            if columnwise:
                return (
                    round_up_to_nearest_multiple(math.prod(shape[:-1]) // MXFP8_BLOCK_SCALING_SIZE, 4),
                    round_up_to_nearest_multiple(shape[-1], 128),
                )
            return (
                round_up_to_nearest_multiple(math.prod(shape[:-1]), 128),
                round_up_to_nearest_multiple(shape[-1] // MXFP8_BLOCK_SCALING_SIZE, 4),
            )
        else:
            raise ValueError(f"Unsupported quantization dtype {dtype} for Flex quantizer!")

    def get_data_shape(
        self,
        shape: Iterable[int],
        columnwise: bool,
        dtype: TE_DType,
    ) -> Tuple[int, ...]:
        """Physical (byte) shape of the quantized DATA buffer for one direction.

        Mirrors ``FlexQuantizer::create_tensor`` (C++): FP8 stores the logical
        shape in both directions; NVFP4 packs two values per byte (and the
        column-wise buffer is transposed), so the packed dim is halved.

        Parameters
        ----------
        shape : Iterable[int]
            Logical shape of the tensor being quantized.
        columnwise : bool
            Column-wise data buffer (True) or row-wise (False).
        dtype : TE_DType
            Per-direction quantized format.

        Returns
        -------
        Tuple[int, ...]
            Shape of the uint8 data buffer for that direction.
        """
        if columnwise:
            return _flex_columnwise_byte_shape(dtype, tuple(shape))
        return _flex_rowwise_byte_shape(dtype, tuple(shape))

    @staticmethod
    def get_columnwise_shape(shape: Iterable[int]) -> Tuple[int, ...]:
        # TODO: probably need to fix this for dist
        raise NotImplementedError("Not implemented yet.")

    def _get_compatible_recipe(self) -> Union[Recipe, None]:
        """Get a compatible recipe for this quantizer, if any."""
        # TODO: really?
        return None  # Flex quantizer does not have a specific compatible recipe since it's orthogonal to the choice of recipe.

class FlexTensor(FlexTensorStorage, QuantizedTensor):
    """Tensor class for flex tensors with quantization in both directions.

    The tensor presents as having a standard, higher-precision dtype, but its
    data is quantized -- potentially with a different format per direction. For
    example, it can be quantized row-wise in NVFP4 and column-wise in MXFP8, or
    use the same format in both directions. The per-direction format is fixed by
    the quantizer that created the tensor (``_dtype_row`` / ``_dtype_column``).

    This is the autograd-visible PyTorch tensor subclass; the data buffers and
    the operations on them live in the ``FlexTensorStorage`` mixin.
    """

    # NOTE: We reorder the *args so that we can instantiate a FlexTensorStorage with positional
    # args, which significantly reduces the Pybind11 overhead when calling the constructor from C++.
    def __new__(
        cls,
        *args,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        amax_rowwise: Optional[torch.Tensor],
        amax_columnwise: Optional[torch.Tensor],
        dtype_row: Optional[TE_DType],
        dtype_column: Optional[TE_DType],
        quantizer: Optional[Quantizer],
        with_gemm_swizzled_scales: bool,
        **kwargs,
    ):
        return super().__new__(
            cls,
            rowwise_data,
            rowwise_scale_inv,
            columnwise_data,
            columnwise_scale_inv,
            amax_rowwise,
            amax_columnwise,
            dtype_row,
            dtype_column,
            quantizer,
            with_gemm_swizzled_scales,
            *args,
            **kwargs,
        )

    def __repr__(self, *, tensor_contents=None):
        return (
            f"FlexTensor(dtype_row={self._dtype_row}, dtype_column={self._dtype_column}, "
            f"data={self.dequantize()})"
        )

    def dequantize(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        """Construct a plain PyTorch tensor from this FlexTensor.

        By default the resulting tensor's dtype is the FlexTensor's nominal
        (high-precision) dtype.
        """
        if dtype is None:
            dtype = self.dtype
        if torch.is_grad_enabled():
            return _FromFlexFunc.apply(self, dtype)
        return _FromFlexFunc.forward(None, self, dtype)

    def _build_default_quantizer(self) -> Optional[Quantizer]:
        """Build a default quantizer matching this tensor's per-direction formats."""
        return FlexQuantizer(dtype_row=self._dtype_row, dtype_column=self._dtype_column)

    def quantize_(
        self,
        tensor: torch.Tensor,
        *,
        noop_flag: Optional[torch.Tensor] = None,
    ) -> FlexTensor:
        """Update Flex tensor data

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

    def detach(self) -> FlexTensor:
        # pylint: disable=missing-function-docstring
        return FlexTensor.make_like(self)

    def clone(self) -> FlexTensor:
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
            }
        )

    def view(self, *shape: Tuple[int]) -> FlexTensor:
        # pylint: disable=missing-function-docstring
        return _ViewFunc.apply(self, shape)

    def reshape(self, *shape: Tuple[int]) -> FlexTensor:
        # pylint: disable=missing-function-docstring
        return _ReshapeFunc.apply(self, shape)

    def contiguous(
        self,
        memory_format: torch.memory_format = torch.contiguous_format,
    ) -> FlexTensor:
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
        raise ValueError("FlexTensor does not support different memory formats!")

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        # pylint: disable=missing-function-docstring
        # TODO: handle aten ops (view/copy_/split/...) per direction,
        # following MXFP8Tensor.__torch_dispatch__.
        raise NotImplementedError(
            f"FlexTensor.__torch_dispatch__ does not support {func} yet"
        )

    # ------------------------------------------------------------------
    # FSDP2 is not supported yet. Define the hooks so any accidental
    # `fully_shard` use fails loudly here instead of cryptically deep in
    # FSDP2's DTensor machinery. Single-GPU (non-FSDP2) use does not call these.
    # ------------------------------------------------------------------
    def fsdp_pre_all_gather(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        # TODO: probably need to fix this for dist
        raise NotImplementedError(
            "FlexTensor does not support FSDP2 yet. "
            "Use a single-GPU (non-fully_shard) setup for now."
        )

    def fsdp_post_all_gather(self, *args, **kwargs):
        # pylint: disable=missing-function-docstring
        # TODO: probably need to fix this for dist
        raise NotImplementedError(
            "FlexTensor does not support FSDP2 yet. "
            "Use a single-GPU (non-fully_shard) setup for now."
        )

    @classmethod
    def _make_in_reduce_ex(
        cls,
        shape: torch.Size,
        rowwise_data: Optional[torch.Tensor],
        rowwise_scale_inv: Optional[torch.Tensor],
        columnwise_data: Optional[torch.Tensor],
        columnwise_scale_inv: Optional[torch.Tensor],
        amax_rowwise: Optional[torch.Tensor],
        amax_columnwise: Optional[torch.Tensor],
        dtype_row: Optional[TE_DType],
        dtype_column: Optional[TE_DType],
        dtype: torch.dtype,
        quantizer: Optional[Quantizer],
        with_gemm_swizzled_scales: bool = False,
    ) -> FlexTensor:
        """Build FlexTensor, for use in __reduce__

        __reduce_ex__ assumes object constructor has positional arguments.

        """
        return FlexTensor(
            shape=shape,
            dtype=dtype,
            rowwise_data=rowwise_data,
            rowwise_scale_inv=rowwise_scale_inv,
            columnwise_data=columnwise_data,
            columnwise_scale_inv=columnwise_scale_inv,
            amax_rowwise=amax_rowwise,
            amax_columnwise=amax_columnwise,
            dtype_row=dtype_row,
            dtype_column=dtype_column,
            quantizer=quantizer,
            with_gemm_swizzled_scales=with_gemm_swizzled_scales,
        )

    def __reduce_ex__(self, protocol: int) -> tuple:
        """Custom pickling"""
        return (
            FlexTensor._make_in_reduce_ex,
            (
                self.shape,
                self._rowwise_data,
                self._rowwise_scale_inv,
                self._columnwise_data,
                self._columnwise_scale_inv,
                self._amax_rowwise,
                self._amax_columnwise,
                self._dtype_row,
                self._dtype_column,
                self.dtype,
                self._quantizer,
                self._with_gemm_swizzled_scales,
            ),
        )

    @property
    def device(self):
        """Return the device of the tensor. Define this to avoid expensive PyObject lookups."""
        if self._rowwise_data is not None:
            return self._rowwise_data.device
        if self._columnwise_data is not None:
            return self._columnwise_data.device
        raise RuntimeError("FlexTensor has no data!")

    @property
    def shape(self):
        """Return the logical shape of the tensor.

        Each direction's buffer is laid out per its own format: MXFP8 stores the
        logical shape directly, while NVFP4 is packed (and column-wise NVFP4 is
        also transposed), so the byte shape is unpacked back to the logical shape.
        """
        if self._rowwise_data is not None:
            byte_shape = self._rowwise_data.shape
            if self.is_nvfp4_dtype(self._dtype_row):
                return torch.Size(byte_shape[:-1] + (byte_shape[-1] * 2,))
            return byte_shape
        if self._columnwise_data is not None:
            byte_shape = self._columnwise_data.shape
            if self.is_nvfp4_dtype(self._dtype_column):
                return torch.Size(byte_shape[1:-1] + (byte_shape[-1] * 2, byte_shape[0]))
            return byte_shape
        return torch.Tensor.size(self)

    @property
    def is_cuda(self):
        """Return whether the tensor is on a CUDA device."""
        if self._rowwise_data is not None:
            return self._rowwise_data.is_cuda
        if self._columnwise_data is not None:
            return self._columnwise_data.is_cuda
        raise RuntimeError("FlexTensor has no data!")

    def _get_data(self) -> FlexTensor:
        """Get tensor data property"""
        return super().data

    @torch.no_grad()
    def _set_data(self, tensor: torch.Tensor) -> None:
        """Set tensor data property

        Just takes the quantized data if setting from a FlexTensor. Otherwise
        casts to the flex format.

        """

        # Tensor device
        new_device = tensor.device if tensor.is_cuda else self.device
        if not devices_match(new_device, tensor.device):
            tensor = tensor.to(device=new_device)

        # Just copy quantized data if other tensor is a FlexTensor
        if isinstance(tensor, FlexTensor):
            if (  # pylint: disable=too-many-boolean-expressions
                self.size() != tensor.size()
                or self.stride() != tensor.stride()
                or self.storage_offset() != tensor.storage_offset()
                or self.dtype != tensor.dtype
                or self.layout != tensor.layout
                or not devices_match(self.device, new_device)
            ):
                dummy_tensor = torch.Tensor._make_wrapper_subclass(
                    FlexTensor,
                    tensor.size(),
                    strides=tensor.stride(),
                    storage_offset=tensor.storage_offset(),
                    dtype=tensor.dtype,
                    layout=tensor.layout,
                    requires_grad=tensor.requires_grad,
                    device=new_device,
                )
                # pylint: disable=unnecessary-dunder-call
                super(FlexTensor, type(self)).data.__set__(self, dummy_tensor)

            self._rowwise_data = tensor._rowwise_data
            self._columnwise_data = tensor._columnwise_data
            self._quantizer = tensor._quantizer
            self._rowwise_scale_inv = tensor._rowwise_scale_inv
            self._columnwise_scale_inv = tensor._columnwise_scale_inv
            self._amax_rowwise = tensor._amax_rowwise
            self._amax_columnwise = tensor._amax_columnwise
            self._dtype_row = tensor._dtype_row
            self._dtype_column = tensor._dtype_column
            self._with_gemm_swizzled_scales = tensor._with_gemm_swizzled_scales
            return

        # Quantize to the flex format
        assert self._quantizer is not None, "Can't quantize without a quantizer"
        self._quantizer.update_quantized(tensor, self)
        if self.requires_grad != tensor.requires_grad:
            self.requires_grad_(requires_grad=tensor.requires_grad)

    # Cast to the flex format when setting FlexTensor.data
    data = property(_get_data, _set_data)


def _flex_rowwise_byte_shape(dtype: Optional[TE_DType], shape) -> tuple:
    """Physical row-wise buffer shape for a logical ``shape``.

    MXFP8 stores the logical shape; NVFP4 packs 2 elements per byte, so the last
    dim is halved.
    """
    if FlexTensor.is_nvfp4_dtype(dtype):
        if shape[-1] % 2 != 0:
            raise ValueError(
                "Cannot represent row-wise NVFP4 quantized data for Flex tensor "
                f"with shape={tuple(shape)} as byte array."
            )
        return shape[:-1] + (shape[-1] // 2,)
    return tuple(shape)

def _flex_columnwise_byte_shape(dtype: Optional[TE_DType], shape) -> tuple:
    """Physical column-wise buffer shape for a logical ``shape``.

    MXFP8 stores the logical shape (untransposed); NVFP4 is transposed and packed
    -- (K, prod(leading_dims) // 2).
    """
    if FlexTensor.is_nvfp4_dtype(dtype):
        columnwise_shape = (shape[-1], math.prod(shape[:-1]))
        if columnwise_shape[-1] % 2 != 0:
            raise ValueError(
                "Cannot represent column-wise NVFP4 quantized data for Flex tensor "
                f"with shape={tuple(shape)} as byte array."
            )
        return (columnwise_shape[0], columnwise_shape[1] // 2)
    return tuple(shape)


class _ViewFunc(torch.autograd.Function):
    """View function

    View the FlexTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: FlexTensor,
        shape: Optional[list[int]] = None,
    ) -> FlexTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        cur_shape = tensor.shape
        if ctx is not None:
            ctx.shape = cur_shape
        if shape is None:
            return tensor

        shape = canonicalize_shape(shape, cur_shape)
        if shape[-1] != cur_shape[-1]:
            warnings.warn(
                "FlexTensor does not support reshaping inner dimension. "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
                "If you are using this for FSDP2 without compiled_autograd_enabled,"
                "then ignore this warning. Since this view is not going to be used anywhere. ",
                stacklevel=2,
            )
            return tensor.dequantize().view(*shape)

        # Construct new tensor if shape is provided. Each direction is viewed
        # according to its own format; a None dtype means that direction is not
        # quantized (its data buffer is None, so it is skipped).
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            byte_shape = _flex_rowwise_byte_shape(tensor._dtype_row, shape)
            new_rowwise_data = tensor._rowwise_data.view(byte_shape)
        if tensor._columnwise_data is not None:
            byte_shape = _flex_columnwise_byte_shape(tensor._dtype_column, shape)
            new_columnwise_data = tensor._columnwise_data.view(byte_shape)

        # Construct tensor
        return FlexTensor(
            shape,
            dtype=tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            dtype_row=tensor._dtype_row,
            dtype_column=tensor._dtype_column,
            requires_grad=tensor.requires_grad,
            with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, FlexTensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                byte_shape = _flex_rowwise_byte_shape(grad._dtype_row, ctx.shape)
                new_rowwise_data = grad._rowwise_data.view(byte_shape)
            if grad._columnwise_data is not None:
                byte_shape = _flex_columnwise_byte_shape(grad._dtype_column, ctx.shape)
                new_columnwise_data = grad._columnwise_data.view(byte_shape)
            dgrad = FlexTensor(
                ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                dtype_row=grad._dtype_row,
                dtype_column=grad._dtype_column,
                requires_grad=grad.requires_grad,
                with_gemm_swizzled_scales=grad._with_gemm_swizzled_scales,
            )
            return dgrad, None
        return grad.view(ctx.shape), None


class _ReshapeFunc(torch.autograd.Function):
    """Reshape function

    Reshape the FlexTensor using the provided shape.

    """

    @staticmethod
    def forward(
        ctx,
        tensor: FlexTensor,
        shape: Optional[list[int]] = None,
    ) -> FlexTensor:
        # pylint: disable=missing-function-docstring

        # Return input tensor if shape is not provided
        cur_shape = tensor.shape
        if ctx is not None:
            ctx.shape = cur_shape
        if shape is None:
            return tensor

        shape = canonicalize_shape(shape, cur_shape)
        if shape[-1] != cur_shape[-1]:
            warnings.warn(
                "FlexTensor does not support reshaping inner dimension. "
                f"(attempted to reshape dims={tuple(tensor.shape)} to {tuple(shape)})"
                "If you are using this for FSDP2 without compiled_autograd_enabled,"
                "then ignore this warning. Since this view is not going to be used anywhere. ",
                stacklevel=2,
            )
            return tensor.dequantize().reshape(*shape)

        # Construct new tensor if shape is provided. Each direction is reshaped
        # according to its own format; a None dtype means that direction is not
        # quantized (its data buffer is None, so it is skipped).
        new_rowwise_data = None
        new_columnwise_data = None
        if tensor._rowwise_data is not None:
            byte_shape = _flex_rowwise_byte_shape(tensor._dtype_row, shape)
            new_rowwise_data = tensor._rowwise_data.reshape(byte_shape)
        if tensor._columnwise_data is not None:
            byte_shape = _flex_columnwise_byte_shape(tensor._dtype_column, shape)
            new_columnwise_data = tensor._columnwise_data.reshape(byte_shape)

        # Construct tensor
        return FlexTensor(
            shape,
            dtype=tensor.dtype,
            rowwise_data=new_rowwise_data,
            rowwise_scale_inv=tensor._rowwise_scale_inv,
            columnwise_data=new_columnwise_data,
            columnwise_scale_inv=tensor._columnwise_scale_inv,
            amax_rowwise=tensor._amax_rowwise,
            amax_columnwise=tensor._amax_columnwise,
            quantizer=tensor._quantizer,
            dtype_row=tensor._dtype_row,
            dtype_column=tensor._dtype_column,
            requires_grad=tensor.requires_grad,
            with_gemm_swizzled_scales=tensor._with_gemm_swizzled_scales,
        )

    @staticmethod
    def backward(
        ctx,
        grad: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        # pylint: disable=missing-function-docstring

        if isinstance(grad, FlexTensor):
            new_rowwise_data = None
            new_columnwise_data = None
            if grad._rowwise_data is not None:
                byte_shape = _flex_rowwise_byte_shape(grad._dtype_row, ctx.shape)
                new_rowwise_data = grad._rowwise_data.reshape(byte_shape)
            if grad._columnwise_data is not None:
                byte_shape = _flex_columnwise_byte_shape(grad._dtype_column, ctx.shape)
                new_columnwise_data = grad._columnwise_data.reshape(byte_shape)
            dgrad = FlexTensor(
                ctx.shape,
                dtype=grad.dtype,
                rowwise_data=new_rowwise_data,
                rowwise_scale_inv=grad._rowwise_scale_inv,
                columnwise_data=new_columnwise_data,
                columnwise_scale_inv=grad._columnwise_scale_inv,
                amax_rowwise=grad._amax_rowwise,
                amax_columnwise=grad._amax_columnwise,
                quantizer=grad._quantizer,
                dtype_row=grad._dtype_row,
                dtype_column=grad._dtype_column,
                requires_grad=grad.requires_grad,
                with_gemm_swizzled_scales=grad._with_gemm_swizzled_scales,
            )
            return dgrad, None
        return grad.view(ctx.shape), None
