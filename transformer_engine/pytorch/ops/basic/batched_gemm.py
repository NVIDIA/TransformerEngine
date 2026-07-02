# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for strided batched linear transformations."""

from __future__ import annotations

from collections.abc import Iterable
import math
from typing import Optional

import torch

from transformer_engine.common.recipe import Recipe

from ...cpp_extensions import strided_batched_gemm
from ...module.base import _2X_ACC_DGRAD, _2X_ACC_FPROP, _2X_ACC_WGRAD
from ...quantization import FP8GlobalStateManager, QuantizerRole
from ...tensor import MXFP8Quantizer, Quantizer
from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ...utils import canonicalize_device, canonicalize_dtype, devices_match
from .._common import is_quantized_tensor, maybe_autocast_dtype, maybe_dequantize
from ..op import BasicOperation, OperationContext


class BatchedGEMM(BasicOperation):
    """Apply a batch of linear transformations with one weight per batch.

    The weight has shape ``[G, R, D]``. Two contiguous input layouts are
    supported:

    * ``[..., G, D]`` with ``batch_dim=-2``
    * ``[G, ..., D]`` with ``batch_dim=0``

    The corresponding output replaces the final ``D`` dimension with ``R``.
    High-precision and MXFP8 compute are supported in both forward and backward.
    MXFP8 quantizes each full input buffer without changing its data layout.
    Training forward caches row-wise and column-wise compact operands for reuse
    by backward. The GEMM extension packs the required scaling direction for
    each cuBLASLt call. Weights created inside
    :func:`transformer_engine.pytorch.quantized_model_init` remain in compact
    MXFP8 format.

    Parameters
    ----------
    num_gemms : int
        Number of independent GEMMs (``G``).
    in_features : int
        Input feature dimension (``D``).
    out_features : int
        Output feature dimension (``R``).
    batch_dim : {0, -2}, default = -2
        Position of the GEMM batch dimension in the input.
    device : torch.device, default = default CUDA device
        Weight device.
    dtype : torch.dtype, default = default dtype
        Weight datatype.
    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        batch_dim: int = -2,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()

        for name, value in (
            ("num_gemms", num_gemms),
            ("in_features", in_features),
            ("out_features", out_features),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer (got {value!r})")
        if batch_dim not in (0, -2):
            raise ValueError(
                f"BatchedGEMM supports only batch_dim=0 or batch_dim=-2 (got {batch_dim})"
            )

        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.batch_dim = batch_dim

        device = canonicalize_device(device)
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

        self._with_quantized_weight: bool = FP8GlobalStateManager.with_fp8_parameters()
        if self._with_quantized_weight:
            self.reset_recipe_state(recipe=FP8GlobalStateManager.get_fp8_recipe())

        weight = torch.empty(
            num_gemms,
            out_features,
            in_features,
            device=device,
            dtype=dtype,
        )
        self.weight: torch.nn.Parameter
        self.register_parameter("weight", torch.nn.Parameter(weight))
        if weight.device.type != "meta":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the weight."""
        weight = self.weight
        device = weight.device
        if device.type == "meta":
            device = canonicalize_device(None)
        if is_quantized_tensor(weight):
            weight = torch.empty(weight.size(), dtype=weight.dtype, device=device)
        elif not devices_match(weight.device, device):
            weight = torch.empty_like(weight, device=device)
        torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

        if self._with_quantized_weight:
            quantizer = self.get_quantizer("forward", 1)
            if quantizer is None:
                raise RuntimeError(
                    "Tried to quantize BatchedGEMM weight after deferred initialization, "
                    "but no quantizer was available"
                )
            self._configure_quantizer(quantizer, internal=False)
            quantizer.set_usage(rowwise=True, columnwise=torch.is_grad_enabled())
            with torch.no_grad():
                weight = quantizer(weight)

        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight)
        self.weight = weight

    def pre_first_fuser_forward(self) -> None:
        super().pre_first_fuser_forward()
        if self.weight.device.type == "meta":
            self.reset_parameters()

    def num_quantizers(self, mode: str) -> int:
        if mode == "forward":
            return 2
        if mode == "backward":
            return 1
        return 0

    def get_quantizer_roles(self, mode: str) -> Optional[list[QuantizerRole]]:
        name = getattr(self, "name", "") or ""
        if mode == "forward":
            return [
                QuantizerRole(module_type="batched_gemm", tensor_type="input", name=name),
                QuantizerRole(module_type="batched_gemm", tensor_type="weight", name=name),
            ]
        if mode == "backward":
            return [QuantizerRole(module_type="batched_gemm", tensor_type="grad_output", name=name)]
        return None

    def get_input_quantizer(self) -> None:
        # Input scales must be packed with knowledge of this op's batch dimension.
        return None

    def get_grad_output_quantizer(self) -> None:
        # Grad-output scales also need this op's batch-aware packing.
        return None

    @staticmethod
    def _configure_quantizer(quantizer: Quantizer, *, internal: bool = True) -> None:
        if not isinstance(quantizer, MXFP8Quantizer):
            raise RuntimeError("BatchedGEMM expected an MXFP8 quantizer")
        quantizer.set_usage(rowwise=True, columnwise=False)
        quantizer.internal = internal
        quantizer.optimize_for_gemm = False

    @staticmethod
    def _validate_recipe(recipe: Recipe) -> None:
        if not recipe.mxfp8():
            raise ValueError(
                "BatchedGEMM supports only high-precision compute or the MXFP8 recipe "
                f"(got {recipe.__class__.__name__})"
            )
        if recipe.backward_override is not None:
            raise ValueError(
                "BatchedGEMM does not support MXFP8 backward_override "
                f"(got {recipe.backward_override!r})"
            )

    def reset_recipe_state(self, *, recipe: Optional[Recipe]) -> None:
        if recipe is not None:
            self._validate_recipe(recipe)
        super().reset_recipe_state(recipe=recipe)
        if recipe is None:
            return

        self._configure_quantizer(self.get_quantizer("forward", 0))
        weight = getattr(self, "weight", None)
        weight_is_quantized = is_quantized_tensor(weight)
        weight_quantizer = self.get_quantizer("forward", 1)
        self._configure_quantizer(
            weight_quantizer,
            internal=not (
                FP8GlobalStateManager.with_fp8_parameters()
                or getattr(self, "_with_quantized_weight", False)
                or weight_is_quantized
            ),
        )
        self._configure_quantizer(self.get_quantizer("backward", 0))

        if isinstance(weight, MXFP8TensorStorage):
            if weight._quantizer is not None:
                weight_quantizer.set_usage(
                    rowwise=weight._quantizer.rowwise_usage,
                    columnwise=weight._quantizer.columnwise_usage,
                )
            weight.update_quantizer(weight_quantizer.copy())

    def pre_fuser_forward(self, *, requires_grad: bool) -> None:
        super().pre_fuser_forward(requires_grad=requires_grad)
        if not FP8GlobalStateManager.is_fp8_enabled():
            return
        recipe = FP8GlobalStateManager.get_fp8_recipe()
        self._validate_recipe(recipe)
        self._configure_quantizer(self.get_quantizer("forward", 0))
        self._configure_quantizer(
            self.get_quantizer("forward", 1),
            internal=not (
                getattr(self, "_with_quantized_weight", False)
                or is_quantized_tensor(getattr(self, "weight", None))
            ),
        )
        self._configure_quantizer(self.get_quantizer("backward", 0))

    def _validate_input(self, input_: torch.Tensor) -> int:
        if is_quantized_tensor(input_):
            raise ValueError("BatchedGEMM expects a high-precision input tensor")
        if not isinstance(input_, torch.Tensor):
            raise TypeError(
                f"BatchedGEMM expects a torch.Tensor input (got {type(input_).__name__})"
            )
        if input_.device.type != "cuda":
            raise ValueError(f"BatchedGEMM requires a CUDA input tensor (got {input_.device})")
        if not devices_match(input_.device, self.weight.device):
            raise ValueError(
                "BatchedGEMM input and weight must be on the same device "
                f"(got {input_.device} and {self.weight.device})"
            )
        if input_.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"BatchedGEMM input has unsupported dtype {input_.dtype}")
        if not input_.is_contiguous():
            raise ValueError("BatchedGEMM requires a contiguous input tensor")
        if input_.ndim < 2:
            raise ValueError(
                f"BatchedGEMM input must have at least two dimensions (got {input_.ndim})"
            )

        batch_axis = 0 if self.batch_dim == 0 else input_.ndim - 2
        if input_.size(batch_axis) != self.num_gemms:
            raise ValueError(
                "BatchedGEMM input batch dimension has invalid size "
                f"(expected {self.num_gemms}, got {input_.size(batch_axis)})"
            )
        if input_.size(-1) != self.in_features:
            raise ValueError(
                "BatchedGEMM input feature dimension has invalid size "
                f"(expected {self.in_features}, got {input_.size(-1)})"
            )
        rows = input_.numel() // (self.num_gemms * self.in_features)
        if rows <= 0:
            raise ValueError("BatchedGEMM does not support empty input matrices")
        return rows

    def _validate_weight(self) -> None:
        expected_shape = (self.num_gemms, self.out_features, self.in_features)
        if tuple(self.weight.shape) != expected_shape:
            raise ValueError(
                f"BatchedGEMM weight has invalid shape (expected {expected_shape}, "
                f"got {tuple(self.weight.shape)})"
            )
        if self.weight.device.type != "cuda":
            raise ValueError(f"BatchedGEMM requires a CUDA weight (got {self.weight.device})")
        if not self.weight.is_contiguous():
            raise ValueError("BatchedGEMM requires a contiguous weight")
        if is_quantized_tensor(self.weight) and not isinstance(self.weight, MXFP8TensorStorage):
            raise ValueError("BatchedGEMM supports only MXFP8 quantized weights")
        if isinstance(self.weight, MXFP8TensorStorage) and self.weight._with_gemm_swizzled_scales:
            raise ValueError("BatchedGEMM quantized weights must use compact MXFP8 scales")

    def _matrix_strides(self, rows: int, features: int) -> tuple[int, int]:
        if self.batch_dim == 0:
            return features, rows * features
        return self.num_gemms * features, features

    def _validate_mxfp8_dimensions(self, rows: int) -> None:
        for name, size in (
            ("rows per GEMM", rows),
            ("in_features", self.in_features),
            ("out_features", self.out_features),
        ):
            if size % 32 != 0:
                raise ValueError(f"MXFP8 BatchedGEMM requires {name} divisible by 32 (got {size})")

    @staticmethod
    def _quantize_for_batched_gemm(
        tensor: torch.Tensor,
        quantizer: Quantizer,
        batch_dim: int,
        num_gemms: int,
        *,
        rowwise: bool,
        columnwise: bool,
    ) -> torch.Tensor:
        if not isinstance(quantizer, MXFP8Quantizer):
            raise RuntimeError("BatchedGEMM expected an MXFP8 quantizer")
        if not rowwise and not columnwise:
            raise RuntimeError("BatchedGEMM quantization requires at least one scaling direction")
        features = tensor.size(-1)
        rows = tensor.numel() // (num_gemms * features)
        quantizer_input = tensor
        if columnwise and batch_dim == -2:
            quantizer_input = tensor.view(rows, num_gemms * features)
        quantizer.set_usage(rowwise=rowwise, columnwise=columnwise)
        quantizer.internal = True
        quantizer.optimize_for_gemm = False
        return quantizer(quantizer_input)

    @staticmethod
    def _validate_quantized_weight_usage(
        weight: MXFP8TensorStorage,
        *,
        columnwise: bool,
    ) -> None:
        if weight._rowwise_data is None or weight._rowwise_scale_inv is None:
            raise RuntimeError("BatchedGEMM MXFP8 weight is missing row-wise data")
        if columnwise and (weight._columnwise_data is None or weight._columnwise_scale_inv is None):
            raise RuntimeError(
                "BatchedGEMM MXFP8 weight is missing column-wise data required for backward"
            )

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        del prev_op_grad_output_quantizer, next_op_input_quantizer

        rows = self._validate_input(input_)
        self._validate_weight()
        dtype = maybe_autocast_dtype(default_dtype=self.weight.dtype)
        x = maybe_dequantize(input_, dtype)
        input_requires_grad = ctx.requires_grad
        weight_requires_grad = ctx.requires_grad and self.weight.requires_grad
        with_mxfp8_compute = FP8GlobalStateManager.is_fp8_enabled()
        if with_mxfp8_compute and isinstance(self.weight, MXFP8TensorStorage):
            w = self.weight
        else:
            w = maybe_dequantize(self.weight, dtype)

        gemm_x: torch.Tensor = x
        gemm_w: torch.Tensor = w
        if with_mxfp8_compute:
            self._validate_mxfp8_dimensions(rows)
            gemm_x = self._quantize_for_batched_gemm(
                x,
                self.get_quantizer("forward", 0),
                self.batch_dim,
                self.num_gemms,
                rowwise=True,
                columnwise=weight_requires_grad,
            )
            if isinstance(w, MXFP8TensorStorage):
                self._validate_quantized_weight_usage(w, columnwise=input_requires_grad)
                gemm_w = w
            else:
                gemm_w = self._quantize_for_batched_gemm(
                    w,
                    self.get_quantizer("forward", 1),
                    0,
                    self.num_gemms,
                    rowwise=True,
                    columnwise=input_requires_grad,
                )

        output_shape = list(input_.shape)
        output_shape[-1] = self.out_features
        output = torch.empty(output_shape, dtype=dtype, device=input_.device)
        ldb, strideb = self._matrix_strides(rows, self.in_features)
        ldd, strided = self._matrix_strides(rows, self.out_features)
        strided_batched_gemm(
            gemm_w,
            gemm_x,
            output,
            m=self.out_features,
            n=rows,
            k=self.in_features,
            batch_count=self.num_gemms,
            lda=self.in_features,
            stridea=self.out_features * self.in_features,
            ldb=ldb,
            strideb=strideb,
            ldd=ldd,
            strided=strided,
            layout="TN",
            use_split_accumulator=_2X_ACC_FPROP,
        )

        if ctx.requires_grad:
            saved_input = gemm_x if with_mxfp8_compute else x
            saved_weight = gemm_w if with_mxfp8_compute else w
            ctx.save_for_backward(
                saved_input if weight_requires_grad else None,
                saved_weight if input_requires_grad else None,
            )
            ctx.input_requires_grad = input_requires_grad
            ctx.weight_requires_grad = weight_requires_grad
            ctx.input_shape = tuple(input_.shape)
            ctx.output_shape = tuple(output_shape)
            ctx.rows = rows
            ctx.dtype = dtype
            ctx.with_mxfp8_compute = with_mxfp8_compute
            ctx.grad_output_quantizer = self.get_quantizer("backward", 0)

        return output

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        x, w = ctx.saved_tensors
        dy = maybe_dequantize(grad_output, ctx.dtype)
        if tuple(dy.shape) != ctx.output_shape:
            raise ValueError(
                "BatchedGEMM grad output has invalid shape "
                f"(expected {ctx.output_shape}, got {tuple(dy.shape)})"
            )

        gemm_dy = dy
        if ctx.with_mxfp8_compute:
            gemm_dy = self._quantize_for_batched_gemm(
                dy,
                ctx.grad_output_quantizer,
                self.batch_dim,
                self.num_gemms,
                rowwise=ctx.input_requires_grad,
                columnwise=ctx.weight_requires_grad,
            )

        grad_input = None
        if ctx.input_requires_grad:
            if w is None:
                raise RuntimeError("BatchedGEMM weight was not saved for input gradient")
            grad_input = torch.empty(ctx.input_shape, dtype=ctx.dtype, device=dy.device)
            ldb, strideb = self._matrix_strides(ctx.rows, self.out_features)
            ldd, strided = self._matrix_strides(ctx.rows, self.in_features)
            strided_batched_gemm(
                w,
                gemm_dy,
                grad_input,
                m=self.in_features,
                n=ctx.rows,
                k=self.out_features,
                batch_count=self.num_gemms,
                lda=self.in_features,
                stridea=self.out_features * self.in_features,
                ldb=ldb,
                strideb=strideb,
                ldd=ldd,
                strided=strided,
                layout="NN",
                use_split_accumulator=_2X_ACC_DGRAD,
            )

        grad_weight = None
        if ctx.weight_requires_grad:
            if x is None:
                raise RuntimeError("BatchedGEMM input was not saved for weight gradient")
            grad_weight = torch.empty(
                self.num_gemms,
                self.out_features,
                self.in_features,
                dtype=ctx.dtype,
                device=dy.device,
            )
            lda, stridea = self._matrix_strides(ctx.rows, self.in_features)
            ldb, strideb = self._matrix_strides(ctx.rows, self.out_features)
            strided_batched_gemm(
                x,
                gemm_dy,
                grad_weight,
                m=self.in_features,
                n=self.out_features,
                k=ctx.rows,
                batch_count=self.num_gemms,
                lda=lda,
                stridea=stridea,
                ldb=ldb,
                strideb=strideb,
                ldd=self.in_features,
                strided=self.out_features * self.in_features,
                layout="NT",
                use_split_accumulator=_2X_ACC_WGRAD,
            )

        return grad_input, [grad_weight]
