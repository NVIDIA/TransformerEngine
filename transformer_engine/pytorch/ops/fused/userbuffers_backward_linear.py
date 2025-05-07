# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear layer backward with Userbuffers communication."""

from __future__ import annotations
from typing import Optional
import warnings

import torch

from transformer_engine_torch import CommOverlapType
from ...cpp_extensions import general_gemm
from ...distributed import gather_along_first_dim, get_distributed_world_size
from ...module.base import (
    fill_userbuffers_buffer_for_all_gather,
    get_ub,
    get_workspace,
    _2X_ACC_DGRAD,
    _2X_ACC_WGRAD,
)
from ...tensor.quantized_tensor import QuantizedTensorBase, Quantizer
from ...tensor.mxfp8_tensor import MXFP8Quantizer
from ...utils import canonicalize_device, canonicalize_dtype, clear_tensor_data
from ..basic import BasicLinear, Bias, ReduceScatter
from ..op import FusedOperation, FusibleOperation, OperationContext


class UserbuffersBackwardLinear(FusedOperation):
    """Linear backward implementation using Userbuffers

    This operation is equivalent to a linear operation's backward
    pass, but it uses Userbuffers to overlap tensor-parallel
    communication with compute.

    """

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
        reduce_scatter: Optional[ReduceScatter],
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = {"linear": None, "bias": None, "reduce_scatter": None}
        ops = []
        if reduce_scatter is not None:
            op_idxs["reduce_scatter"] = len(ops)
            ops.append(reduce_scatter)
        if bias is not None:
            op_idxs["bias"] = len(ops)
            ops.append(bias)
        op_idxs["linear"] = len(ops)
        ops.append(linear)

        # Initialize base class
        super().__init__(ops)

        # Index of each basic operations
        self._op_idxs: dict[str, Optional[int]] = op_idxs

        # Tensor parallelism configuration
        self.tensor_parallel_mode: Optional[str]
        self.tensor_parallel_group: Optional[torch.distributed.ProcessGroup]
        self.tensor_parallel_size: int
        self.sequence_parallel: bool
        if reduce_scatter is None:
            self.tensor_parallel_mode = linear.tensor_parallel_mode
            self.tensor_parallel_group = linear.tensor_parallel_group
            self.tensor_parallel_size = linear.tensor_parallel_size
            self.sequence_parallel = linear.sequence_parallel
        else:
            self.tensor_parallel_mode = "row"
            self.tensor_parallel_group = reduce_scatter.process_group
            self.tensor_parallel_size = reduce_scatter.process_group_size
            self.sequence_parallel = True

    @staticmethod
    def _functional_backward(
        grad_output: torch.Tensor,
        input: Optional[torch.Tensor],  # pylint: disable=redefined-builtin
        weight: Optional[torch.Tensor],
        *,
        input_requires_grad: bool = True,
        weight_requires_grad: bool = True,
        bias_requires_grad: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        grad_weight: Optional[torch.Tensor] = None,
        accumulate_into_grad_weight: bool = False,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        with_quantized_compute: bool = False,
        input_quantizer: Optional[Quantizer] = None,
        weight_quantizer: Optional[Quantizer] = None,
        grad_output_quantizer: Optional[Quantizer] = None,
        grad_input_quantizer: Optional[Quantizer] = None,
        ub_comm_name: str,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """Functional API for backward pass

        Parameters
        ----------
        grad_output: torch.Tensor
            Loss gradient w.r.t. output tensor
        input: torch.Tensor, optional
            Input tensor. Required to compute loss gradient w.r.t.
            weight.
        weight: torch.Tensor, optional
            Weight tensor. Required to compute loss gradient w.r.t.
            input.
        weight_requires_grad: bool
            Whether to compute loss gradient w.r.t. weight tensor
        bias_requires_grad: bool
            Whether to compute loss gradient w.r.t. bias tensor
        device: torch.device, default = default CUDA device
            Tensor device
        dtype: torch.dtype, default = default dtype
            Tensor datatype
        grad_weight: torch.Tensor, optional
            Loss gradient w.r.t. weight tensor
        accumulate_into_grad_weight: bool, default = `False`
            Add result to weight grad instead of overwriting
        tensor_parallel_mode: {`None`, "column", "row"}, default = `None`
            Mode for tensor parallelism
        tensor_parallel_group: torch.distributed.ProcessGroup, default = world group
            Process group for tensor parallelism
        sequence_parallel: bool, default = `False`
            Whether to apply sequence parallelism together with tensor
            parallelism, i.e. distributing input or output tensors
            along outer dimension (sequence or batch dim) when not
            distributing along inner dimension (embedding dim)
        with_quantized_compute: bool, default = `False`
            Whether to perform compute with quantized data.
        input_quantizer: Quantizer, optional
            Builder class for quantized input tensor.
        weight_quantizer: Quantizer, optional
            Builder class for quantized weight tensor.
        grad_output_quantizer: Quantizer, optional
            Builder class for quantized loss gradient w.r.t. output
            tensor.
        grad_input_quantizer: Quantizer, optional
            Builder class for quantized loss gradient w.r.t. input
            tensor.
        ub_comm_name: str
            Layer type (e.g. "qkv", "proj", "fc1", "fc2"). This is
            used to access the corresponding Userbuffers communicators
            (e.g. "qkv_dgrad", "qkv_wgrad").

        Returns
        -------
        torch.Tensor
            Loss gradient w.r.t. input tensor
        torch.Tensor
            Loss gradient w.r.t. weight tensor
        dict
            Extra output tensors. "grad_bias" is loss gradient w.r.t.
            the bias tensor.

        """

        # Configuration-specific outputs
        extra_outputs = {}

        # Check device
        if device is None:
            if weight is not None:
                device = weight.device
            else:
                device = grad_output.device
        device = canonicalize_device(device)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")

        # Check datatype
        if dtype is None:
            if weight is not None:
                dtype = weight.dtype
            else:
                dtype = grad_output.dtype
        dtype = canonicalize_dtype(dtype)
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError(f"Supported dtypes are float32, float16, bfloat16 (got {dtype})")

        # Check tensor parallel group
        if tensor_parallel_size is None:
            tensor_parallel_size = get_distributed_world_size(tensor_parallel_group)
        if tensor_parallel_size == 1:
            tensor_parallel_mode = None
        if tensor_parallel_mode not in ("column", "row"):
            raise RuntimeError(
                "Invalid configuration for Userbuffers "
                f"({tensor_parallel_size=}, {tensor_parallel_mode=})"
            )
        if not sequence_parallel:
            raise RuntimeError(f"Invalid configuration for Userbuffers ({sequence_parallel=})")

        # dgrad GEMM is required
        if not input_requires_grad:
            warnings.warn(
                "Linear input doesn't require gradient, "
                "but Userbuffers implementation requires dgrad GEMM."
            )
            input_requires_grad = True

        # Check quantizers
        if with_quantized_compute:
            if weight_requires_grad and input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if input_requires_grad and weight_quantizer is None:
                raise ValueError("Missing quantizer for weight tensor")
            if grad_output_quantizer is None:
                raise ValueError("Missing quantizer for grad output tensor")
            if grad_input_quantizer is not None:
                raise ValueError("Quantized grad input is not supported")
        else:
            input_quantizer = None
            weight_quantizer = None
            grad_output_quantizer = None
            grad_input_quantizer = None

        # Get Userbuffers communicators
        # Note: Communication patterns are (1) overlap dy all-gather
        # with dgrad GEMM, (2) overlap x all-gather with dgrad GEMM
        # and dx reduce-scatter with wgrad GEMM, (3) overlap dx
        # reduce-scatter with dgrad GEMM
        ub_comm_dgrad = None
        ub_comm_wgrad = None
        ub_type_dgrad = None
        ub_type_wgrad = None
        with_bulk_overlap = False
        with_dgrad_all_gather_dy = False
        with_dgrad_reduce_scatter_dx = False
        with_dgrad_all_gather_x = False
        with_wgrad_reduce_scatter_dx = False
        if tensor_parallel_mode == "row":
            ub_comm_dgrad = get_ub(ub_comm_name + "_dgrad")
            ub_type_dgrad = CommOverlapType.AG
            with_dgrad_all_gather_dy = True
        elif tensor_parallel_mode == "column":
            if input_requires_grad and weight_requires_grad:
                with_bulk_overlap = True
                ub_comm_dgrad = get_ub(ub_comm_name + "_dgrad")
                ub_type_dgrad = CommOverlapType.AG
                with_dgrad_all_gather_x = True
                ub_comm_wgrad = get_ub(ub_comm_name + "_wgrad")
                ub_type_wgrad = CommOverlapType.RS
                with_wgrad_reduce_scatter_dx = True
                if ub_comm_wgrad.is_fp8_ubuf():
                    raise RuntimeError(
                        "Userbuffers reduce-scatter is not supported with FP8 buffers"
                    )
            else:
                ub_comm_dgrad = get_ub(ub_comm_name + "_dgrad")
                ub_type_dgrad = CommOverlapType.RS
                with_dgrad_reduce_scatter_dx = True
                if ub_comm_dgrad.is_fp8_ubuf():
                    raise RuntimeError(
                        "Userbuffers reduce-scatter is not supported with FP8 buffers"
                    )

        # Compute grad bias if needed
        db = None
        db_async = None
        if bias_requires_grad:
            db = grad_output.sum(tuple(range(grad_output.dim() - 1)))
            if tensor_parallel_mode == "row":
                db_async = torch.distributed.all_reduce(
                    db,
                    group=tensor_parallel_group,
                    async_op=True,
                )

        # Cast grad output tensor dtype if needed
        dy_local = grad_output
        if with_quantized_compute:
            if not isinstance(dy_local, QuantizedTensorBase):
                with_columnwise = weight_requires_grad
                if (
                    with_columnwise
                    and with_dgrad_all_gather_dy
                    and not isinstance(grad_output_quantizer, MXFP8Quantizer)
                ):
                    with_columnwise = False
                grad_output_quantizer.set_usage(
                    rowwise=True,
                    columnwise=with_columnwise,
                )
                dy_local = grad_output_quantizer(dy_local)
        else:
            if isinstance(dy_local, QuantizedTensorBase):
                dy_local = dy_local.dequantize(dtype=dtype)
            elif dy_local.dtype != dtype:
                dy_local = dy_local.to(dtype=dtype)

        # Cast weight tensor dtype if needed
        if weight is None:
            raise ValueError("Weight tensor is required to compute input grad")
        w = weight
        if with_quantized_compute:
            if not isinstance(w, QuantizedTensorBase):
                weight_quantizer.set_usage(columnwise=True)
                w = weight_quantizer(w)
        else:
            if isinstance(w, QuantizedTensorBase):
                w = w.dequantize(dtype=dtype)
            elif w.dtype != dtype:
                w = w.to(dtype=dtype)

        # Cast input tensor dtype if needed
        x_local = None
        if weight_requires_grad:
            if input is None:
                raise ValueError("Input tensor is required to compute weight grad")
            x_local = input
            if with_quantized_compute:
                if not isinstance(x_local, QuantizedTensorBase):
                    input_quantizer.set_usage(columnwise=True)
                    x_local = input_quantizer(x_local)
            else:
                if isinstance(x_local, QuantizedTensorBase):
                    x_local = x_local.dequantize(dtype=dtype)
                elif x_local.dtype != dtype:
                    x_local = x_local.to(dtype=dtype)

        # dgrad GEMM
        dx_local = None
        dx = None
        dy = None
        x = None
        if input_requires_grad:

            # Initialize grad output
            if with_dgrad_all_gather_dy:
                if grad_output_quantizer is not None:
                    grad_output_quantizer.set_usage(rowwise=True, columnwise=False)
                dy, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_comm_dgrad,
                    dy_local,
                    grad_output_quantizer,
                    tensor_parallel_group,
                )
            else:
                dy = dy_local

            # Construct grad input tensor if needed
            if with_dgrad_reduce_scatter_dx or with_wgrad_reduce_scatter_dx:
                dx_size = list(dy.size())
                dx_size[-1] = w.size(-1)
                dx_local_size = list(dx_size)
                dx_local_size[0] //= tensor_parallel_size
                if with_dgrad_reduce_scatter_dx:
                    dx_local = torch.empty(
                        dx_local_size,
                        dtype=dtype,
                        device=device,
                    )
                elif with_wgrad_reduce_scatter_dx:
                    dx_local = ub_comm_wgrad.get_buffer(
                        local_chunk=True,
                        shape=dx_local_size,
                    )
                    dx = ub_comm_wgrad.get_buffer(
                        local_chunk=False,
                        shape=dx_size,
                    )

            # Initialize input tensor if needed
            if with_dgrad_all_gather_x:
                if input_quantizer is not None:
                    input_quantizer.set_usage(rowwise=False, columnwise=True)
                x, _ = fill_userbuffers_buffer_for_all_gather(
                    ub_comm_dgrad,
                    x_local,
                    input_quantizer,
                    tensor_parallel_group,
                )

            # Perform dgrad GEMM
            dx, *_ = general_gemm(
                w,
                dy,
                get_workspace(),
                out_dtype=dtype,
                quantization_params=grad_input_quantizer,
                layout="NN",
                out=dx,
                use_split_accumulator=_2X_ACC_DGRAD,
                grad=True,
                ub=ub_comm_dgrad,
                ub_type=ub_type_dgrad,
                extra_output=dx_local if with_dgrad_reduce_scatter_dx else None,
                bulk_overlap=with_bulk_overlap,
            )
            if not (with_dgrad_reduce_scatter_dx or with_wgrad_reduce_scatter_dx):
                dx_local = dx

        # wgrad GEMM
        dw = None
        if weight_requires_grad:

            # Initialize grad output
            if tensor_parallel_mode == "row" and isinstance(grad_output_quantizer, MXFP8Quantizer):
                # UB does not support overlapping grad output
                # all-gather with wgrad GEMM. Also, MXFP8 does not
                # allow reusing the grad output that was gathered for
                # the dgrad GEMM. We work around with blocking
                # all-gather for column-scaled MXFP8 data.
                grad_output_quantizer.set_usage(rowwise=False, columnwise=True)
                dy, _ = gather_along_first_dim(
                    grad_output,
                    tensor_parallel_group,
                    quantizer=grad_output_quantizer,
                )
            if tensor_parallel_mode == "column":
                dy = dy_local
            if dy is None:
                raise RuntimeError(
                    "wgrad GEMM requires grad output tensor, which has not been initialized"
                )
            if isinstance(dy, QuantizedTensorBase):
                dy.update_usage(rowwise_usage=False, columnwise_usage=True)

            # Initialize input tensor
            if tensor_parallel_mode == "row":
                x = x_local
            if x is None:
                raise RuntimeError(
                    "wgrad GEMM requires input tensor, which has not been initialized"
                )
            if isinstance(x, QuantizedTensorBase):
                x.update_usage(rowwise_usage=False, columnwise_usage=True)

            # Check grad weight tensor
            dw = grad_weight
            dw_dtype = dtype
            if dw is None:
                if accumulate_into_grad_weight:
                    raise ValueError(
                        "Attempted to accumulate into grad weight tensor "
                        "without providing grad weight tensor"
                    )
            else:
                dw_dtype = dw.dtype

            # Perform wgrad GEMM
            dw, *_ = general_gemm(
                x,
                dy,
                get_workspace(),
                out_dtype=dw_dtype,
                accumulate=accumulate_into_grad_weight,
                layout="NT",
                out=dw,
                use_split_accumulator=_2X_ACC_WGRAD,
                grad=True,
                ub=ub_comm_wgrad,
                ub_type=ub_type_wgrad,
                bulk_overlap=with_bulk_overlap,
            )

            # Bulk overlap reduce-scatter with non-FP8 buffer is
            # in-place. Need to copy grad input tensor to avoid data
            # corruption in Userbuffers buffer.
            if with_wgrad_reduce_scatter_dx:
                dx_local = dx_local.clone()

        # Compute grad bias if needed
        if db_async is not None:
            db_async.wait()
        if bias_requires_grad:
            extra_outputs["grad_bias"] = db

        return dx_local, dw, extra_outputs

    def fuser_backward(
        self,
        basic_op_ctxs: list[OperationContext],
        grad_output: torch.Tensor,
        *,
        basic_op_grad_extra_outputs: list[tuple[torch.Tensor, ...]],
    ) -> tuple[
        torch.Tensor,
        list[tuple[Optional[torch.Tensor], ...]],
        list[tuple[()]],
    ]:

        # Get basic operations
        idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[idx]
        linear_op_ctx = basic_op_ctxs[idx]
        bias_op = None
        if self._op_idxs["bias"] is not None:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]

        # Saved tensors from forward pass
        (x_local,) = linear_op_ctx.saved_tensors

        # wgrad fusion
        accumulate_into_main_grad = linear_op._accumulate_into_main_grad
        grad_weight = None
        if linear_op_ctx.weight_requires_grad and accumulate_into_main_grad:
            if not hasattr(linear_op.weight, "main_grad"):
                raise RuntimeError(
                    "BasicLinear op is configured with "
                    "accumulate_into_main_grad=True, "
                    "but weight parameter does not have main_grad attribute"
                )
            grad_weight = linear_op.weight.main_grad.detach()
        else:
            accumulate_into_main_grad = False

        # Linear backward pass
        retval = UserbuffersBackwardLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=linear_op.weight,
            weight_requires_grad=linear_op_ctx.weight_requires_grad,
            bias_requires_grad=(bias_op is not None),
            dtype=linear_op_ctx.dtype,
            grad_weight=grad_weight,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_quantized_compute=linear_op_ctx.with_quantized_compute,
            input_quantizer=linear_op_ctx.input_quantizer,
            weight_quantizer=linear_op_ctx.weight_quantizer,
            grad_output_quantizer=linear_op_ctx.grad_output_quantizer,
            grad_input_quantizer=None,  # Not supported
            ub_comm_name=linear_op._userbuffers_options["comm_name"],
        )
        grad_input, grad_weight, extra_outputs = retval
        grad_bias = None
        if bias_op is not None:
            grad_bias = extra_outputs["grad_bias"]

        # Clear input tensor if possible
        if linear_op_ctx.has_prev_op:
            clear_tensor_data(x_local)

        # Return gradients
        grad_params = [() for _ in range(len(self.basic_ops))]
        if accumulate_into_main_grad:
            grad_weight = None
        grad_params[self._op_idxs["linear"]] = (grad_weight,)
        if bias_op is not None:
            grad_params[self._op_idxs["bias"]] = (grad_bias,)
        grad_extra_inputs = [() for _ in range(len(self.basic_ops))]
        return grad_input, grad_params, grad_extra_inputs


def fuse_userbuffers_backward_linear(
    ops: list[tuple[FusibleOperation, list[int]]],
) -> list[tuple[FusibleOperation, list[int]]]:
    """Substitute linear operations with Userbuffers implementation

    Parameters
    ----------
    ops: list of tuples
        Forward pass operations and the indices of the corresponding
        basic operations.

    Returns
    -------
    ops: list of tuples
        Updated forward pass operations

    """

    # Return immediately if environment is not distributed
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return ops

    # Sliding window in list of ops
    window = []

    def peek_next_op() -> Optional[FusibleOperation]:
        """Get next op in list of ops"""
        nonlocal ops
        if not ops:
            return None
        return ops[-1][0]

    def pop_next_op() -> FusibleOperation:
        """Remove next op from list of ops and add to sliding window"""
        nonlocal ops, window
        window.insert(0, ops[-1])
        ops = ops[:-1]
        return window[0][0]

    # Scan through ops in reverse order, fusing if possible
    out_reversed = []
    while ops:
        out_reversed.extend(reversed(window))
        window.clear()

        # Check if next op is linear
        next_op = pop_next_op()
        if not isinstance(next_op, BasicLinear):
            continue
        linear = next_op
        if linear._userbuffers_options is None:
            continue

        # Check if next op is bias
        bias = None
        if linear.tensor_parallel_mode != "row" and isinstance(peek_next_op(), Bias):
            bias = pop_next_op()

        # Check if next op is reduce-scatter
        reduce_scatter = None
        if linear.tensor_parallel_mode is None and isinstance(peek_next_op(), ReduceScatter):
            reduce_scatter = pop_next_op()

        # Check for invalid combinations
        if reduce_scatter is None:
            if linear.tensor_parallel_mode is None:
                continue
            if linear.tensor_parallel_size == 1:
                continue
            if linear.tensor_parallel_mode == "row" and bias is not None:
                continue
        else:
            if linear.tensor_parallel_mode is not None:
                continue
            if reduce_scatter.process_group_size == 1:
                continue

        # Replace window with fused op
        op = UserbuffersBackwardLinear(
            linear=linear,
            bias=bias,
            reduce_scatter=reduce_scatter,
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out_reversed.extend(reversed(window))
    out = out_reversed
    out.reverse()
    return out
