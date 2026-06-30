# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for strided batched linear transformations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import contextlib
from typing import Any, Optional
import warnings

import torch

from transformer_engine.common.recipe import Recipe

from ...cpp_extensions import strided_batched_gemm
from ...distributed import CudaRNGStatesTracker
from ...module.base import _2X_ACC_DGRAD, _2X_ACC_FPROP, _2X_ACC_WGRAD
from ...quantization import FP8GlobalStateManager, QuantizerRole
from ...tensor import MXFP8Quantizer, Quantizer
from ...tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ...utils import (
    canonicalize_device,
    canonicalize_dtype,
    devices_match,
    get_default_init_method,
)
from .._common import (
    get_accumulate_flag_in_param,
    get_dummy_wgrads_for_params,
    get_main_grad_from_param,
    is_quantized_tensor,
    maybe_autocast_dtype,
    maybe_dequantize,
    view_main_grad_as_grouped_buffer,
)
from ..op import BasicOperation, OperationContext


_HIGH_PRECISION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


class BatchedLinear(BasicOperation):
    """Apply one linear transformation per batch entry.

    The weight has shape ``[G, R, D]``. Inputs may use either contiguous
    ``[..., G, D]`` storage with ``batch_dim=-2`` or contiguous
    ``[G, ..., D]`` storage with ``batch_dim=0``. The output replaces ``D``
    with ``R``. High-precision and MXFP8 forward and backward computation are
    supported. Tensor parallelism is not supported.

    Parameters
    ----------
    num_gemms : int
        Number of independent linear transformations (``G``).
    in_features : int
        Input feature dimension (``D``).
    out_features : int
        Output feature dimension (``R``).
    batch_dim : {0, -2}, default = -2
        Position of the GEMM batch dimension in the input.
    bias : bool, default = True
        Add one learned bias of shape ``[G, R]``.
    return_bias : bool, default = False
        Return the bias separately instead of applying it.
    device : torch.device, default = default CUDA device
        Parameter device.
    dtype : torch.dtype, default = default dtype
        Parameter datatype.
    rng_state_tracker_function : callable, optional
        Function returning a ``CudaRNGStatesTracker`` used during parameter
        initialization.
    accumulate_into_main_grad : bool, default = False
        Write weight gradients directly into the externally allocated
        ``weight.main_grad`` buffer. Setting ``weight.overwrite_main_grad`` to
        ``True`` overwrites that buffer instead of accumulating into it.
    init_method : callable, optional
        Weight initialization method. The default is TE's normal initializer.
    name : str, optional
        Name used by quantizer-role dispatch and debugging.

    Notes
    -----
    Constructing this operation under ``quantized_model_init`` supports only
    an MXFP8 recipe. Meta-device parameter materialization relies on the
    operation fuser's deferred-initialization support.
    """

    num_extra_outputs: int = 0

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        batch_dim: int = -2,
        bias: bool = True,
        return_bias: bool = False,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
        accumulate_into_main_grad: bool = False,
        init_method: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        name: Optional[str] = None,
    ) -> None:
        # The fuser allocates extra-output routing state in BasicOperation.__init__.
        self.num_extra_outputs = int(bias and return_bias)
        super().__init__()

        for arg_name, value in (
            ("num_gemms", num_gemms),
            ("in_features", in_features),
            ("out_features", out_features),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{arg_name} must be a positive integer (got {value!r})")
        if batch_dim not in (0, -2):
            raise ValueError(
                f"BatchedLinear supports only batch_dim=0 or batch_dim=-2 (got {batch_dim})"
            )

        device = canonicalize_device(device)
        dtype = canonicalize_dtype(dtype)
        if dtype not in _HIGH_PRECISION_DTYPES:
            raise ValueError(
                f"BatchedLinear parameters must use float32, float16, or bfloat16 (got {dtype})"
            )

        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.batch_dim = batch_dim
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.name = name
        self._rng_state_tracker_function = rng_state_tracker_function
        self._accumulate_into_main_grad = accumulate_into_main_grad
        self._init_method = get_default_init_method() if init_method is None else init_method

        # Initialize recipe state if the weight itself is stored in MXFP8.
        self._with_quantized_weight = FP8GlobalStateManager.with_fp8_parameters()
        if self._with_quantized_weight:
            self.reset_recipe_state(recipe=FP8GlobalStateManager.get_fp8_recipe())

        weight = torch.empty(
            num_gemms,
            out_features,
            in_features,
            dtype=dtype,
            device=device,
        )
        self.weight: torch.nn.Parameter
        self.register_parameter("weight", torch.nn.Parameter(weight))

        bias_tensor = None
        if bias:
            bias_tensor = torch.empty(
                num_gemms,
                out_features,
                dtype=dtype,
                device=device,
            )
            bias_tensor = torch.nn.Parameter(bias_tensor)
        self.bias: Optional[torch.nn.Parameter]
        self.register_parameter("bias", bias_tensor)

        if device.type != "meta":
            self.reset_parameters()

    def reset_parameters(self) -> None:
        """Allocate and initialize parameter values."""
        old_weight = self.weight
        device = old_weight.device
        if device.type == "meta":
            device = canonicalize_device(None)

        if is_quantized_tensor(old_weight):
            weight = torch.empty(old_weight.size(), dtype=old_weight.dtype, device=device)
        elif not devices_match(old_weight.device, device):
            weight = torch.empty_like(old_weight, device=device)
        else:
            weight = old_weight

        init_context = contextlib.nullcontext()
        if self._rng_state_tracker_function is not None:
            init_context = self._rng_state_tracker_function().fork()
        with torch.no_grad(), init_context:
            self._init_method(weight)

        if self._with_quantized_weight:
            quantizer = self.get_quantizer("forward", 1)
            if quantizer is None:
                raise RuntimeError(
                    "Tried to quantize BatchedLinear weight after deferred initialization, "
                    "but no quantizer was available. The forward pass must run under "
                    "MXFP8 autocast."
                )
            self._configure_quantizer(quantizer, internal=False)
            quantizer.set_usage(rowwise=True, columnwise=torch.is_grad_enabled())
            with torch.no_grad():
                weight = quantizer(weight)

        if not isinstance(weight, torch.nn.Parameter):
            weight = torch.nn.Parameter(weight, requires_grad=old_weight.requires_grad)
        self.weight = weight

        if self.use_bias:
            old_bias = self.bias
            if old_bias is None:
                raise RuntimeError("BatchedLinear bias parameter is missing")
            if devices_match(old_bias.device, device):
                bias = old_bias
            else:
                bias = torch.empty_like(old_bias, device=device)
            with torch.no_grad():
                bias.zero_()
            if not isinstance(bias, torch.nn.Parameter):
                bias = torch.nn.Parameter(bias, requires_grad=old_bias.requires_grad)
            self.bias = bias

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
        name = self.name or ""
        if mode == "forward":
            return [
                QuantizerRole(module_type="batched_linear", tensor_type="input", name=name),
                QuantizerRole(module_type="batched_linear", tensor_type="weight", name=name),
            ]
        if mode == "backward":
            return [
                QuantizerRole(
                    module_type="batched_linear",
                    tensor_type="grad_output",
                    name=name,
                )
            ]
        return None

    def get_input_quantizer(self) -> None:
        # Input scales must be packed with knowledge of this op's batch dimension.
        return None

    def get_grad_output_quantizer(self) -> None:
        # Grad-output scales also require batch-aware packing.
        return None

    @staticmethod
    def _configure_quantizer(quantizer: Quantizer, *, internal: bool = True) -> None:
        if not isinstance(quantizer, MXFP8Quantizer):
            raise RuntimeError("BatchedLinear expected an MXFP8 quantizer")
        quantizer.set_usage(rowwise=True, columnwise=False)
        quantizer.internal = internal
        quantizer.optimize_for_gemm = False

    @staticmethod
    def _validate_recipe(recipe: Recipe) -> None:
        if not recipe.mxfp8():
            raise ValueError(
                "BatchedLinear supports only high-precision compute or the MXFP8 recipe "
                f"(got {recipe.__class__.__name__})"
            )
        if recipe.backward_override is not None:
            raise ValueError(
                "BatchedLinear does not support MXFP8 backward_override "
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
        self._validate_recipe(FP8GlobalStateManager.get_fp8_recipe())
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
        if not isinstance(input_, torch.Tensor):
            raise TypeError(
                f"BatchedLinear expects a torch.Tensor input (got {type(input_).__name__})"
            )
        if is_quantized_tensor(input_):
            raise ValueError("BatchedLinear expects a high-precision input tensor")
        if input_.device.type != "cuda":
            raise ValueError(f"BatchedLinear requires a CUDA input tensor (got {input_.device})")
        if not devices_match(input_.device, self.weight.device):
            raise ValueError(
                "BatchedLinear input and weight must be on the same device "
                f"(got {input_.device} and {self.weight.device})"
            )
        if input_.dtype not in _HIGH_PRECISION_DTYPES:
            raise ValueError(f"BatchedLinear input has unsupported dtype {input_.dtype}")
        if not input_.is_contiguous():
            raise ValueError("BatchedLinear requires a contiguous input tensor")
        if input_.ndim < 2:
            raise ValueError(
                f"BatchedLinear input must have at least two dimensions (got {input_.ndim})"
            )

        batch_axis = 0 if self.batch_dim == 0 else input_.ndim - 2
        if input_.size(batch_axis) != self.num_gemms:
            raise ValueError(
                "BatchedLinear input batch dimension has invalid size "
                f"(expected {self.num_gemms}, got {input_.size(batch_axis)})"
            )
        if input_.size(-1) != self.in_features:
            raise ValueError(
                "BatchedLinear input feature dimension has invalid size "
                f"(expected {self.in_features}, got {input_.size(-1)})"
            )
        rows = input_.numel() // (self.num_gemms * self.in_features)
        if rows <= 0:
            raise ValueError("BatchedLinear does not support empty input matrices")
        return rows

    def _validate_parameters(self) -> None:
        expected_weight_shape = (self.num_gemms, self.out_features, self.in_features)
        if tuple(self.weight.shape) != expected_weight_shape:
            raise ValueError(
                "BatchedLinear weight has invalid shape "
                f"(expected {expected_weight_shape}, got {tuple(self.weight.shape)})"
            )
        if self.weight.device.type != "cuda":
            raise ValueError(f"BatchedLinear requires a CUDA weight (got {self.weight.device})")
        if not self.weight.is_contiguous():
            raise ValueError("BatchedLinear requires a contiguous weight")
        if is_quantized_tensor(self.weight) and not isinstance(self.weight, MXFP8TensorStorage):
            raise ValueError("BatchedLinear supports only MXFP8 quantized weights")
        if isinstance(self.weight, MXFP8TensorStorage) and self.weight._with_gemm_swizzled_scales:
            raise ValueError("BatchedLinear quantized weights must use compact MXFP8 scales")

        if self.use_bias:
            if self.bias is None:
                raise ValueError("BatchedLinear bias parameter is missing")
            expected_bias_shape = (self.num_gemms, self.out_features)
            if tuple(self.bias.shape) != expected_bias_shape:
                raise ValueError(
                    "BatchedLinear bias has invalid shape "
                    f"(expected {expected_bias_shape}, got {tuple(self.bias.shape)})"
                )
            if not devices_match(self.bias.device, self.weight.device):
                raise ValueError(
                    "BatchedLinear bias and weight must be on the same device "
                    f"(got {self.bias.device} and {self.weight.device})"
                )
            if self.bias.dtype not in _HIGH_PRECISION_DTYPES:
                raise ValueError(f"BatchedLinear bias has unsupported dtype {self.bias.dtype}")
            if not self.bias.is_contiguous():
                raise ValueError("BatchedLinear requires a contiguous bias")

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
                raise ValueError(
                    f"MXFP8 BatchedLinear requires {name} divisible by 32 (got {size})"
                )

    @staticmethod
    def _quantize_for_batched_gemm(
        tensor: torch.Tensor,
        quantizer: Quantizer,
        batch_dim: int,
        num_gemms: int,
        *,
        rowwise: bool,
        columnwise: bool,
    ) -> MXFP8TensorStorage:
        if not isinstance(quantizer, MXFP8Quantizer):
            raise RuntimeError("BatchedLinear expected an MXFP8 quantizer")
        if not rowwise and not columnwise:
            raise RuntimeError("BatchedLinear quantization requires at least one scaling direction")
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
            raise RuntimeError("BatchedLinear MXFP8 weight is missing row-wise data")
        if columnwise and (weight._columnwise_data is None or weight._columnwise_scale_inv is None):
            raise RuntimeError(
                "BatchedLinear MXFP8 weight is missing column-wise data required for backward"
            )

    def _apply_bias(self, output: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if not self.apply_bias:
            return output
        if self.bias is None:
            raise RuntimeError("BatchedLinear bias parameter is missing")
        bias = maybe_dequantize(self.bias, dtype)
        if self.batch_dim == 0:
            bias_shape = [self.num_gemms] + [1] * (output.ndim - 2) + [self.out_features]
            bias = bias.view(bias_shape)
        return output + bias

    def _reduce_bias_gradient(self, grad_output: torch.Tensor) -> torch.Tensor:
        if self.batch_dim == 0:
            reduce_dims = tuple(range(1, grad_output.ndim - 1))
        else:
            reduce_dims = tuple(range(grad_output.ndim - 2))
        if reduce_dims:
            grad_output = grad_output.sum(dim=reduce_dims)
        if self.bias is not None and grad_output.dtype != self.bias.dtype:
            grad_output = grad_output.to(dtype=self.bias.dtype)
        return grad_output

    def op_forward(
        self,
        ctx: OperationContext,
        input_: torch.Tensor,
        *,
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
    ) -> torch.Tensor:
        del prev_op_grad_output_quantizer, next_op_input_quantizer

        rows = self._validate_input(input_)
        self._validate_parameters()
        dtype = maybe_autocast_dtype(default_dtype=self.weight.dtype)
        x = maybe_dequantize(input_, dtype)
        input_requires_grad = ctx.requires_grad
        weight_requires_grad = ctx.requires_grad and self.weight.requires_grad
        bias_requires_grad = ctx.requires_grad and self.bias is not None and self.bias.requires_grad
        with_mxfp8_compute = FP8GlobalStateManager.is_fp8_enabled()

        if not with_mxfp8_compute and is_quantized_tensor(self.weight):
            warnings.warn(
                "BatchedLinear is using an MXFP8 weight without MXFP8 compute. "
                "The weight will be dequantized.",
                stacklevel=2,
            )
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
        output = self._apply_bias(output, dtype)

        if ctx.requires_grad:
            ctx.save_for_backward(
                gemm_x if weight_requires_grad else None,
                gemm_w if input_requires_grad else None,
            )
            ctx.input_requires_grad = input_requires_grad
            ctx.weight_requires_grad = weight_requires_grad
            ctx.bias_requires_grad = bias_requires_grad
            ctx.input_shape = tuple(input_.shape)
            ctx.output_shape = tuple(output_shape)
            ctx.rows = rows
            ctx.dtype = dtype
            ctx.with_mxfp8_compute = with_mxfp8_compute
            ctx.grad_output_quantizer = self.get_quantizer("backward", 0)
            ctx.apply_bias = self.apply_bias

        return output

    def _op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
        grad_returned_bias: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        x, w = ctx.saved_tensors
        dy = maybe_dequantize(grad_output, ctx.dtype).contiguous()
        if tuple(dy.shape) != ctx.output_shape:
            raise ValueError(
                "BatchedLinear grad output has invalid shape "
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
                raise RuntimeError("BatchedLinear weight was not saved for input gradient")
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
                raise RuntimeError("BatchedLinear input was not saved for weight gradient")
            accumulate_wgrad = False
            if self._accumulate_into_main_grad:
                main_grad = get_main_grad_from_param(
                    self.weight,
                    op_label="BatchedLinear",
                ).detach()
                grad_weight = view_main_grad_as_grouped_buffer(
                    main_grad,
                    self.num_gemms,
                    (self.out_features, self.in_features),
                    label="BatchedLinear weight",
                )
                if not grad_weight.is_contiguous():
                    raise RuntimeError("BatchedLinear weight main_grad must be contiguous")
                if not devices_match(grad_weight.device, dy.device):
                    raise RuntimeError(
                        "BatchedLinear weight main_grad must be on the grad output device "
                        f"(got {grad_weight.device} and {dy.device})"
                    )
                if grad_weight.dtype not in _HIGH_PRECISION_DTYPES:
                    raise RuntimeError(
                        "BatchedLinear weight main_grad must have a high-precision dtype "
                        f"(got {grad_weight.dtype})"
                    )
                accumulate_wgrad = get_accumulate_flag_in_param(self.weight)
            else:
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
                accumulate=accumulate_wgrad,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

            if self._accumulate_into_main_grad:
                grad_weight = get_dummy_wgrads_for_params([self.weight])[0]

        grad_bias = None
        if ctx.bias_requires_grad:
            if ctx.apply_bias:
                grad_bias = self._reduce_bias_gradient(dy)
            elif grad_returned_bias is not None:
                grad_bias = maybe_dequantize(grad_returned_bias, self.bias.dtype)

        grad_params = [grad_weight]
        if self.use_bias:
            grad_params.append(grad_bias)
        return grad_input, grad_params

    def op_backward(
        self,
        ctx: OperationContext,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, Iterable[Optional[torch.Tensor]]]:
        return self._op_backward(ctx, grad_output, None)

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: Sequence[Sequence[Optional[torch.Tensor]]],
        prev_op_grad_output_quantizer: Optional[Quantizer],
        next_op_input_quantizer: Optional[Quantizer],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Sequence[Sequence[Optional[torch.Tensor]]]]:
        del basic_op_extra_inputs
        output = self.op_forward(
            basic_op_ctxs[0],
            input_,
            prev_op_grad_output_quantizer=prev_op_grad_output_quantizer,
            next_op_input_quantizer=next_op_input_quantizer,
            **basic_op_kwargs[0],
        )
        if self.num_extra_outputs == 0:
            return output, [()]
        if self.bias is None:
            raise RuntimeError("BatchedLinear bias parameter is missing")
        return output, [(maybe_dequantize(self.bias, output.dtype),)]

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: Sequence[Sequence[Optional[torch.Tensor]]],
    ) -> tuple[
        torch.Tensor,
        Sequence[Sequence[Optional[torch.Tensor]]],
        Sequence[Sequence[Optional[torch.Tensor]]],
    ]:
        grad_returned_bias = None
        if self.num_extra_outputs > 0:
            grad_returned_bias = basic_op_grad_extra_outputs[0][0]
        grad_input, grad_params = self._op_backward(
            basic_op_ctxs[0],
            grad_output,
            grad_returned_bias,
        )
        return grad_input, [grad_params], [()]

    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        *extra_inputs: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
        output = super().forward(input, *extra_inputs, **kwargs)
        if self.return_bias and not self.use_bias:
            return output, None
        return output
