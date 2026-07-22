# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""BatchedLinear API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import warnings
import weakref

import torch

from transformer_engine.common.recipe import Recipe

from .base import (
    TransformerEngineBaseModule,
    _2X_ACC_DGRAD,
    _2X_ACC_FPROP,
    _2X_ACC_WGRAD,
    get_dummy_wgrad,
)
from ..constants import FP8BwdTensorIdx, FP8FwdTensorIdx
from ..cpp_extensions import strided_batched_gemm
from ..distributed import CudaRNGStatesTracker
from ..jit import no_torch_dynamo
from ..quantization import FP8GlobalStateManager, QuantizerRole
from ..quantized_tensor import (
    QuantizedTensorStorage,
    Quantizer,
    prepare_for_saving,
    restore_from_func_ctx,
)
from ..tensor import MXFP8Quantizer
from ..tensor.storage.mxfp8_tensor_storage import MXFP8TensorStorage
from ..utils import cast_if_needed, devices_match, init_method_constant

__all__ = ["BatchedLinear"]


_HIGH_PRECISION_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _maybe_dequantize(
    tensor: Union[torch.Tensor, QuantizedTensorStorage],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a high-precision tensor with the requested dtype."""
    if isinstance(tensor, QuantizedTensorStorage):
        return tensor.dequantize(dtype=dtype)
    return cast_if_needed(tensor, dtype)


def _matrix_strides(
    batch_dim: int,
    num_gemms: int,
    rows: int,
    features: int,
) -> Tuple[int, int]:
    """Return matrix leading dimension and batch stride for a supported layout."""
    if batch_dim == 0:
        return features, rows * features
    return num_gemms * features, features


def _configure_mxfp8_quantizer(
    quantizer: Quantizer,
    *,
    internal: bool,
    rowwise: bool,
    columnwise: bool,
) -> None:
    """Configure an MXFP8 quantizer for compact strided-batched operands."""
    if not isinstance(quantizer, MXFP8Quantizer):
        raise RuntimeError("BatchedLinear expected an MXFP8 quantizer")
    quantizer.set_usage(rowwise=rowwise, columnwise=columnwise)
    quantizer.internal = internal
    quantizer.optimize_for_gemm = False


def _quantize_for_batched_gemm(
    tensor: torch.Tensor,
    quantizer: Quantizer,
    batch_dim: int,
    num_gemms: int,
    *,
    rowwise: bool,
    columnwise: bool,
) -> MXFP8TensorStorage:
    """Quantize while preserving batch-major or interleaved matrix storage."""
    if not rowwise and not columnwise:
        raise RuntimeError("BatchedLinear quantization requires at least one scaling direction")
    features = tensor.size(-1)
    rows = tensor.numel() // (num_gemms * features)
    quantizer_input = tensor
    if columnwise and batch_dim == -2:
        quantizer_input = tensor.view(rows, num_gemms * features)
    _configure_mxfp8_quantizer(
        quantizer,
        internal=True,
        rowwise=rowwise,
        columnwise=columnwise,
    )
    return quantizer(quantizer_input)


def _validate_quantized_weight_usage(
    weight: MXFP8TensorStorage,
    *,
    columnwise: bool,
) -> None:
    """Check that a primary MXFP8 weight has the data needed by this pass."""
    if weight._rowwise_data is None or weight._rowwise_scale_inv is None:
        raise RuntimeError("BatchedLinear MXFP8 weight is missing row-wise data")
    if columnwise and (weight._columnwise_data is None or weight._columnwise_scale_inv is None):
        raise RuntimeError(
            "BatchedLinear MXFP8 weight is missing column-wise data required for backward"
        )


def _get_main_grad(weight: torch.nn.Parameter) -> torch.Tensor:
    """Return the externally managed main-grad buffer for a weight."""
    if hasattr(weight, "__fsdp_param__"):
        weight.main_grad = weight.get_main_grad()
    if not hasattr(weight, "main_grad") or weight.main_grad is None:
        raise RuntimeError(
            "BatchedLinear is configured with accumulate_into_main_grad=True, "
            "but weight does not have a valid main_grad attribute"
        )
    return weight.main_grad


def _view_main_grad(
    main_grad: torch.Tensor,
    shape: Tuple[int, int, int],
) -> torch.Tensor:
    """View a main-grad allocation as the batched weight without copying."""
    if tuple(main_grad.shape) == shape:
        return main_grad
    if main_grad.numel() != torch.Size(shape).numel():
        raise RuntimeError(
            f"BatchedLinear weight main_grad expected shape {shape} or matching numel, "
            f"but got {tuple(main_grad.shape)}"
        )
    try:
        return main_grad.view(shape)
    except RuntimeError as exc:
        raise RuntimeError(
            f"BatchedLinear weight main_grad must be viewable as {shape} without copy, "
            f"but got shape {tuple(main_grad.shape)} and stride {tuple(main_grad.stride())}"
        ) from exc


@dataclass
class _BatchedLinearArgs:
    """Non-tensor arguments shared by forward and backward."""

    num_gemms: int
    in_features: int
    out_features: int
    batch_dim: int
    rows: int
    activation_dtype: torch.dtype
    with_mxfp8_compute: bool
    input_quantizer: Optional[Quantizer]
    weight_quantizer: Optional[Quantizer]
    grad_output_quantizer: Optional[Quantizer]
    accumulate_into_main_grad: bool


class _BatchedLinear(torch.autograd.Function):
    """Autograd implementation backed by strided batched GEMMs."""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        weight: torch.Tensor,
        args: _BatchedLinearArgs,
    ) -> torch.Tensor:
        """Run the batched linear forward pass and save tensors needed by backward."""
        input_requires_grad, weight_requires_grad = ctx.needs_input_grad[:2]
        x = _maybe_dequantize(inp, args.activation_dtype)
        if args.with_mxfp8_compute and isinstance(weight, MXFP8TensorStorage):
            w = weight
        else:
            w = _maybe_dequantize(weight, args.activation_dtype)

        gemm_x = x
        gemm_w = w
        if args.with_mxfp8_compute:
            gemm_x = _quantize_for_batched_gemm(
                x,
                args.input_quantizer,
                args.batch_dim,
                args.num_gemms,
                rowwise=True,
                columnwise=weight_requires_grad,
            )
            if isinstance(w, MXFP8TensorStorage):
                _validate_quantized_weight_usage(w, columnwise=input_requires_grad)
                gemm_w = w
            else:
                gemm_w = _quantize_for_batched_gemm(
                    w,
                    args.weight_quantizer,
                    0,
                    args.num_gemms,
                    rowwise=True,
                    columnwise=input_requires_grad,
                )

        output_shape = list(inp.shape)
        output_shape[-1] = args.out_features
        output = torch.empty(output_shape, dtype=args.activation_dtype, device=inp.device)
        ldb, strideb = _matrix_strides(
            args.batch_dim,
            args.num_gemms,
            args.rows,
            args.in_features,
        )
        ldd, strided = _matrix_strides(
            args.batch_dim,
            args.num_gemms,
            args.rows,
            args.out_features,
        )
        strided_batched_gemm(
            gemm_w,
            gemm_x,
            output,
            m=args.out_features,
            n=args.rows,
            k=args.in_features,
            batch_count=args.num_gemms,
            lda=args.in_features,
            stridea=args.out_features * args.in_features,
            ldb=ldb,
            strideb=strideb,
            ldd=ldd,
            strided=strided,
            layout="TN",
            use_split_accumulator=_2X_ACC_FPROP,
        )

        if input_requires_grad or weight_requires_grad:
            tensors_to_save, tensor_objects = prepare_for_saving(
                gemm_x if weight_requires_grad else None,
                gemm_w if input_requires_grad else None,
            )
            ctx.save_for_backward(*tensors_to_save)
            ctx.tensor_objects = tensor_objects
            ctx.args = args
            ctx.input_requires_grad = input_requires_grad
            ctx.weight_requires_grad = weight_requires_grad
            ctx.input_shape = tuple(inp.shape)
            ctx.output_shape = tuple(output_shape)
            if args.accumulate_into_main_grad and weight_requires_grad:
                ctx.origin_weight_ref = weakref.ref(weight)

        return output

    @staticmethod
    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], None]:
        """Compute input and weight gradients with strided batched GEMMs."""
        args: _BatchedLinearArgs = ctx.args
        x, w = restore_from_func_ctx(ctx)
        dy = _maybe_dequantize(grad_output, args.activation_dtype).contiguous()
        if tuple(dy.shape) != ctx.output_shape:
            raise ValueError(
                "BatchedLinear grad output has invalid shape "
                f"(expected {ctx.output_shape}, got {tuple(dy.shape)})"
            )

        gemm_dy = dy
        if args.with_mxfp8_compute:
            gemm_dy = _quantize_for_batched_gemm(
                dy,
                args.grad_output_quantizer,
                args.batch_dim,
                args.num_gemms,
                rowwise=ctx.input_requires_grad,
                columnwise=ctx.weight_requires_grad,
            )

        grad_input = None
        if ctx.input_requires_grad:
            if w is None:
                raise RuntimeError("BatchedLinear weight was not saved for input gradient")
            grad_input = torch.empty(
                ctx.input_shape,
                dtype=args.activation_dtype,
                device=dy.device,
            )
            ldb, strideb = _matrix_strides(
                args.batch_dim,
                args.num_gemms,
                args.rows,
                args.out_features,
            )
            ldd, strided = _matrix_strides(
                args.batch_dim,
                args.num_gemms,
                args.rows,
                args.in_features,
            )
            strided_batched_gemm(
                w,
                gemm_dy,
                grad_input,
                m=args.in_features,
                n=args.rows,
                k=args.out_features,
                batch_count=args.num_gemms,
                lda=args.in_features,
                stridea=args.out_features * args.in_features,
                ldb=ldb,
                strideb=strideb,
                ldd=ldd,
                strided=strided,
                layout="NN",
                use_split_accumulator=_2X_ACC_DGRAD,
            )

        grad_weight = None
        origin_weight = None
        if ctx.weight_requires_grad:
            if x is None:
                raise RuntimeError("BatchedLinear input was not saved for weight gradient")
            accumulate = False
            if args.accumulate_into_main_grad:
                origin_weight_ref = ctx.origin_weight_ref
                ctx.origin_weight_ref = None
                origin_weight = origin_weight_ref()
                if origin_weight is None:
                    raise RuntimeError(
                        "BatchedLinear weight was removed while accumulate_into_main_grad=True"
                    )
                main_grad = _get_main_grad(origin_weight)
                origin_weight.main_grad = main_grad
                grad_weight = _view_main_grad(
                    main_grad.detach(),
                    (args.num_gemms, args.out_features, args.in_features),
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
                accumulate = not getattr(origin_weight, "overwrite_main_grad", False)
            else:
                grad_weight = torch.empty(
                    args.num_gemms,
                    args.out_features,
                    args.in_features,
                    dtype=args.activation_dtype,
                    device=dy.device,
                )

            lda, stridea = _matrix_strides(
                args.batch_dim,
                args.num_gemms,
                args.rows,
                args.in_features,
            )
            ldb, strideb = _matrix_strides(
                args.batch_dim,
                args.num_gemms,
                args.rows,
                args.out_features,
            )
            strided_batched_gemm(
                x,
                gemm_dy,
                grad_weight,
                m=args.in_features,
                n=args.out_features,
                k=args.rows,
                batch_count=args.num_gemms,
                lda=lda,
                stridea=stridea,
                ldb=ldb,
                strideb=strideb,
                ldd=args.in_features,
                strided=args.out_features * args.in_features,
                layout="NT",
                accumulate=accumulate,
                use_split_accumulator=_2X_ACC_WGRAD,
            )

            if args.accumulate_into_main_grad:
                grad_weight = None
                if hasattr(origin_weight, "grad_added_to_main_grad"):
                    origin_weight.grad_added_to_main_grad = True
                    grad_weight = get_dummy_wgrad(
                        list(origin_weight.shape),
                        origin_weight.dtype,
                        zero=getattr(origin_weight, "zero_out_wgrad", False),
                    )

        return grad_input, grad_weight, None


class BatchedLinear(TransformerEngineBaseModule):
    """Apply one linear transformation per batch entry.

    The weight has shape ``[G, R, D]``. Inputs may use either contiguous
    ``[..., G, D]`` storage with ``batch_dim=-2`` or contiguous
    ``[G, ..., D]`` storage with ``batch_dim=0``. The output replaces ``D``
    with ``R``. High-precision and MXFP8 forward and backward computation are
    supported.

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
    rng_state_tracker_function : callable, optional
        Function returning a ``CudaRNGStatesTracker`` used during parameter
        initialization.
    accumulate_into_main_grad : bool, default = False
        Write weight gradients directly into the externally allocated
        ``weight.main_grad`` buffer.
    init_method : callable, optional
        Weight initialization method. The default is TE's normal initializer.
    bias : bool, default = True
        Add one learned bias of shape ``[G, R]``.
    return_bias : bool, default = False
        Return the bias separately instead of applying it.
    params_dtype : torch.dtype, optional
        Parameter dtype. Defaults to ``torch.get_default_dtype()``.
    device : torch.device or str, default = "cuda"
        Parameter device.
    name : str, optional
        Module name used by quantizer-role dispatch and debugging.

    Notes
    -----
    Constructing this module under ``quantized_model_init`` supports only an
    MXFP8 recipe. ``preserve_high_precision_init_val=True`` is handled by the
    common Transformer Engine module parameter initialization path.
    """

    def __init__(
        self,
        num_gemms: int,
        in_features: int,
        out_features: int,
        *,
        batch_dim: int = -2,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
        accumulate_into_main_grad: bool = False,
        init_method: Optional[Callable] = None,
        bias: bool = True,
        return_bias: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        device: Union[torch.device, str] = "cuda",
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name)

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

        params_dtype = torch.get_default_dtype() if params_dtype is None else params_dtype
        if params_dtype not in _HIGH_PRECISION_DTYPES:
            raise ValueError(
                "BatchedLinear parameters must use float32, float16, or bfloat16 "
                f"(got {params_dtype})"
            )

        self.num_gemms = num_gemms
        self.in_features = in_features
        self.out_features = out_features
        self.batch_dim = batch_dim
        self.accumulate_into_main_grad = accumulate_into_main_grad
        self.rng_state_tracker_function = rng_state_tracker_function
        self.use_bias = bias
        self.return_bias = return_bias
        self.apply_bias = bias and not return_bias
        self.weight_names = ["weight"]

        if self.primary_weights_in_fp8:
            self._validate_recipe(FP8GlobalStateManager.get_fp8_recipe())

        weight = torch.empty(
            num_gemms,
            out_features,
            in_features,
            dtype=params_dtype,
            device=device,
        )
        self.register_parameter(
            "weight",
            torch.nn.Parameter(weight),
            init_fn=init_method,
            get_rng_state_tracker=rng_state_tracker_function,
            fp8_meta_index=FP8FwdTensorIdx.GEMM1_WEIGHT,
        )

        if bias:
            bias_tensor = torch.empty(
                num_gemms,
                out_features,
                dtype=params_dtype,
                device=device,
            )
            self.register_parameter(
                "bias",
                torch.nn.Parameter(bias_tensor),
                init_fn=init_method_constant(0.0),
            )
        else:
            self.bias = torch.empty(0, dtype=params_dtype, device=device)

        if self.primary_weights_in_fp8:
            self.init_fp8_metadata()
            _configure_mxfp8_quantizer(
                self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT],
                internal=False,
                rowwise=True,
                columnwise=torch.is_grad_enabled(),
            )

        self.reset_parameters(defer_init=torch.device(device).type == "meta")

    @staticmethod
    def _validate_recipe(recipe: Recipe) -> None:
        """Reject recipes that the strided-batched backend cannot execute."""
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

    def get_quantizer_roles(
        self,
        *,
        fwd: bool,
        num_quantizers: int,
    ) -> Optional[List[QuantizerRole]]:
        """Return quantizer roles for input, weight, and grad output."""
        name = self.name or ""
        if fwd:
            base = [
                QuantizerRole(module_type="batched_linear", tensor_type="input", name=name),
                QuantizerRole(module_type="batched_linear", tensor_type="weight", name=name),
                self._output_quantizer_role,
            ]
        else:
            base = [
                QuantizerRole(
                    module_type="batched_linear",
                    tensor_type="grad_output",
                    name=name,
                ),
                self._grad_input_quantizer_role,
            ]
        return [base[i % len(base)] for i in range(num_quantizers)]

    def _get_weight_tensors(self) -> List[Union[torch.Tensor, QuantizedTensorStorage]]:
        """Return the weight in the representation required by current compute."""
        weight = self.weight
        if not self.fp8 and isinstance(weight, QuantizedTensorStorage):
            warnings.warn(
                "You are using a quantized BatchedLinear weight without quantized compute. "
                "Please make sure this is intentional."
            )
            weight = weight.dequantize()
        return [weight]

    def _get_weight_quantizers(self) -> List[Optional[Quantizer]]:
        """Return the MXFP8 weight quantizer used by this module."""
        if not self.fp8 and not self.fp8_calibration and not self.primary_weights_in_fp8:
            return [None]
        quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
        _configure_mxfp8_quantizer(
            quantizer,
            internal=not self.primary_weights_in_fp8,
            rowwise=True,
            columnwise=torch.is_grad_enabled(),
        )
        return [quantizer]

    def _validate_input_and_weight(self, inp: torch.Tensor) -> int:
        """Validate the supported layouts and return rows per GEMM."""
        if isinstance(inp, QuantizedTensorStorage):
            raise ValueError("BatchedLinear expects a high-precision input tensor")
        if inp.device.type != "cuda":
            raise ValueError(f"BatchedLinear requires a CUDA input tensor (got {inp.device})")
        if not devices_match(inp.device, self.weight.device):
            raise ValueError(
                "BatchedLinear input and weight must be on the same device "
                f"(got {inp.device} and {self.weight.device})"
            )
        if inp.dtype not in _HIGH_PRECISION_DTYPES:
            raise ValueError(f"BatchedLinear input has unsupported dtype {inp.dtype}")
        if not inp.is_contiguous():
            raise ValueError("BatchedLinear requires a contiguous input tensor")
        if inp.ndim < 2:
            raise ValueError(
                f"BatchedLinear input must have at least two dimensions (got {inp.ndim})"
            )

        batch_axis = 0 if self.batch_dim == 0 else inp.ndim - 2
        if inp.size(batch_axis) != self.num_gemms:
            raise ValueError(
                "BatchedLinear input batch dimension has invalid size "
                f"(expected {self.num_gemms}, got {inp.size(batch_axis)})"
            )
        if inp.size(-1) != self.in_features:
            raise ValueError(
                "BatchedLinear input feature dimension has invalid size "
                f"(expected {self.in_features}, got {inp.size(-1)})"
            )

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
        if isinstance(self.weight, QuantizedTensorStorage) and not isinstance(
            self.weight, MXFP8TensorStorage
        ):
            raise ValueError("BatchedLinear supports only MXFP8 quantized weights")
        if isinstance(self.weight, MXFP8TensorStorage) and self.weight._with_gemm_swizzled_scales:
            raise ValueError("BatchedLinear quantized weights must use compact MXFP8 scales")

        rows = inp.numel() // (self.num_gemms * self.in_features)
        if rows <= 0:
            raise ValueError("BatchedLinear does not support empty input matrices")
        return rows

    def _validate_mxfp8_dimensions(self, rows: int) -> None:
        """Check MXFP8 block-alignment requirements."""
        for name, size in (
            ("rows per GEMM", rows),
            ("in_features", self.in_features),
            ("out_features", self.out_features),
        ):
            if size % 32 != 0:
                raise ValueError(
                    f"MXFP8 BatchedLinear requires {name} divisible by 32 (got {size})"
                )

    @no_torch_dynamo()
    def forward(
        self,
        inp: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Apply the batched linear transformation."""
        if not isinstance(inp, torch.Tensor):
            raise TypeError(
                f"BatchedLinear expects a torch.Tensor input (got {type(inp).__name__})"
            )
        if FP8GlobalStateManager.is_fp8_enabled():
            self._validate_recipe(FP8GlobalStateManager.get_fp8_recipe())

        rows = self._validate_input_and_weight(inp)
        inp = self.prepare_forward(inp, allow_non_contiguous=True)
        try:
            input_quantizer = None
            weight_quantizer = None
            grad_output_quantizer = None
            if self.fp8:
                self._validate_mxfp8_dimensions(rows)
                input_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_INPUT]
                weight_quantizer = self.quantizers["scaling_fwd"][FP8FwdTensorIdx.GEMM1_WEIGHT]
                grad_output_quantizer = self.quantizers["scaling_bwd"][FP8BwdTensorIdx.GRAD_OUTPUT1]
                for quantizer, internal in (
                    (input_quantizer, True),
                    (weight_quantizer, not self.primary_weights_in_fp8),
                    (grad_output_quantizer, True),
                ):
                    _configure_mxfp8_quantizer(
                        quantizer,
                        internal=internal,
                        rowwise=True,
                        columnwise=False,
                    )

            args = _BatchedLinearArgs(
                num_gemms=self.num_gemms,
                in_features=self.in_features,
                out_features=self.out_features,
                batch_dim=self.batch_dim,
                rows=rows,
                activation_dtype=self.activation_dtype,
                with_mxfp8_compute=self.fp8,
                input_quantizer=input_quantizer,
                weight_quantizer=weight_quantizer,
                grad_output_quantizer=grad_output_quantizer,
                accumulate_into_main_grad=self.accumulate_into_main_grad,
            )
            out = _BatchedLinear.apply(inp, self.weight, args)
        finally:
            self.end_forward()

        bias = cast_if_needed(self.bias, self.activation_dtype) if self.use_bias else None
        if self.apply_bias:
            if self.batch_dim == 0:
                bias_shape = [self.num_gemms] + [1] * (out.ndim - 2) + [self.out_features]
                out = out + bias.view(bias_shape)
            else:
                out = out + bias
        if self.return_bias:
            return out, bias
        return out
