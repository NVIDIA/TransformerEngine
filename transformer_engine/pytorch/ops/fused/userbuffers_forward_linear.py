# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear layer forward with Userbuffers communication."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformer_engine_torch import CommOverlapType
from ...cpp_extensions import general_gemm
from ...distributed import get_distributed_world_size
from ...fp8 import FP8GlobalStateManager
from ...module.base import (
    fill_userbuffers_buffer_for_all_gather,
    get_ub,
    get_workspace,
    _2X_ACC_FPROP,
)
from ...tensor.quantized_tensor import QuantizedTensorBase, Quantizer
from ...tensor.float8_tensor import Float8Quantizer
from ...tensor._internal.float8_tensor_base import Float8TensorBase
from ...utils import canonicalize_device, canonicalize_dtype
from ..basic import BasicLinear, Bias, ReduceScatter
from ..op import (
    BasicOperation,
    FusedOperation,
    FusibleOperation,
    OperationContext,
)


class UserbuffersForwardLinear(FusedOperation):
    """Linear forward implementation using Userbuffers

    This operation is equivalent to a linear operation's forward pass,
    but it uses Userbuffers to overlap tensor-parallel communication
    with compute.

    """

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
        reduce_scatter: Optional[ReduceScatter],
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = {"linear": 0, "bias": None, "reduce_scatter": None}
        ops = [linear]
        if bias is not None:
            op_idxs["bias"] = len(ops)
            ops.append(bias)
        if reduce_scatter is not None:
            op_idxs["reduce_scatter"] = len(ops)
            ops.append(reduce_scatter)

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
    def _functional_forward(
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        weight: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        with_quantized_compute: bool = False,
        input_quantizer: Optional[Quantizer] = None,
        weight_quantizer: Optional[Quantizer] = None,
        output_quantizer: Optional[Quantizer] = None,
        ub_comm_name: str,
    ) -> tuple[torch.Tensor, dict]:
        """Functional API for forward pass

        Parameters
        ----------
        input: torch.Tensor
            Input tensor
        weight: torch.Tensor
            Weight tensor
        bias: torch.Tensor, optional
            Bias tensor
        device: torch.device, default = default CUDA device
            Tensor device
        dtype: torch.dtype, default = default dtype
            Tensor datatype
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
        output_quantizer: Quantizer, optional
            Builder class for quantized output tensor.
        ub_comm_name: str
            Layer type (e.g. "qkv", "proj", "fc1", "fc2"). This is
            used to access the corresponding Userbuffers communicators
            (e.g. "qkv_fprop").

        Returns
        -------
        torch.Tensor
            Output tensor
        dict
            Extra output tensors. "input" is the input tensor,
            possibly cast and reshaped from the provided input tensor.

        """

        # Check device
        if device is None:
            device = weight.device
        device = canonicalize_device(device)
        if device.type != "cuda":
            raise ValueError(f"Only CUDA devices are supported (got {device})")

        # Check datatype
        if dtype is None:
            dtype = weight.dtype
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

        # Check quantizers
        if with_quantized_compute:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if weight_quantizer is None:
                raise ValueError("Missing quantizer for weight tensor")
            if output_quantizer is not None:
                raise ValueError("FP8 output is not supported")
        else:
            input_quantizer = None
            weight_quantizer = None
            output_quantizer = None

        # Get Userbuffers communicator
        ub_comm = get_ub(ub_comm_name + "_fprop")
        with_ub_all_gather = tensor_parallel_mode == "column"
        with_ub_reduce_scatter = tensor_parallel_mode == "row"
        ub_type = CommOverlapType.AG if with_ub_all_gather else CommOverlapType.RS

        # Initialize input tensor
        x_local = input
        x = None
        if with_ub_all_gather:
            if input_quantizer is not None:
                if not isinstance(x_local, QuantizedTensorBase):
                    input_quantizer.set_usage(rowwise=True, columnwise=True)
                    if isinstance(input_quantizer, Float8Quantizer):
                        input_quantizer.set_usage(columnwise=False)
                    x_local = input_quantizer(x_local)
                input_quantizer.set_usage(rowwise=True, columnwise=False)
            x, x_local = fill_userbuffers_buffer_for_all_gather(
                ub_comm,
                x_local,
                input_quantizer,
                tensor_parallel_group,
            )
        else:
            if with_quantized_compute:
                if not isinstance(x_local, QuantizedTensorBase):
                    input_quantizer.set_usage(rowwise=True, columnwise=True)
                    x_local = input_quantizer(x_local)
            else:
                if isinstance(x_local, QuantizedTensorBase):
                    x_local = x_local.dequantize(dtype=dtype)
                if x_local.dtype != dtype:
                    x_local = x_local.to(dtype=dtype)
            x = x_local

        # Initialize weight tensor
        w = weight
        w_is_quantized = isinstance(w, QuantizedTensorBase)
        if with_quantized_compute and not w_is_quantized:
            weight_quantizer.set_usage(rowwise=True)
            w = weight_quantizer(w)
        elif not with_quantized_compute and w_is_quantized:
            w = w.dequantize()
        if not with_quantized_compute and w.dtype != dtype:
            w = w.to(dtype=dtype)

        # Construct output tensor if needed
        reduce_scatter_output = None
        if with_ub_reduce_scatter:
            y_local_size = list(x.size())
            y_local_size[0] //= tensor_parallel_size
            y_local_size[-1] = w.size(0)
            reduce_scatter_output = torch.empty(y_local_size, dtype=dtype, device=device)

        # Perform GEMM
        gemm_output, *_, reduce_scatter_output = general_gemm(
            w,
            x,
            get_workspace(),
            out_dtype=dtype,
            quantization_params=output_quantizer,
            bias=bias,
            use_split_accumulator=_2X_ACC_FPROP,
            ub=ub_comm,
            ub_type=ub_type,
            extra_output=reduce_scatter_output,
        )
        if with_ub_reduce_scatter:
            y_local = reduce_scatter_output
        else:
            y_local = gemm_output

        # Detach input tensor if needed
        # Note: PyTorch autograd produces esoteric errors if we save
        # input tensor as context for backward pass.
        if x_local is input:
            x_local = x_local.detach()

        # Configure input tensor for backward pass
        if with_quantized_compute and isinstance(x_local, QuantizedTensorBase):
            if not (isinstance(x_local, Float8TensorBase) and with_ub_all_gather):
                # FP8 does not support all-gather of transpose data
                x_local.update_usage(rowwise_usage=False, columnwise_usage=True)

        # Return cast tensors
        extra_outputs = {"input": x_local, "weight": w}
        return y_local, extra_outputs

    def fuser_forward(
        self,
        basic_op_ctxs: list[OperationContext],
        input_: torch.Tensor,
        *,
        basic_op_extra_inputs: list[tuple[torch.Tensor, ...]],
        basic_op_prev_ops: list[Optional[BasicOperation]],
        basic_op_next_ops: list[Optional[BasicOperation]],
        basic_op_kwargs: list[dict[str, Any]],
    ) -> tuple[torch.Tensor, Iterable[Iterable[torch.Tensor]]]:

        # Get basic operations
        idx = self._op_idxs["linear"]
        linear_op = self.basic_ops[idx]
        linear_op_ctx = basic_op_ctxs[idx]
        bias_op = None
        bias = None
        if self._op_idxs["bias"] is not None:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]
            bias = bias_op.bias
            if basic_op_kwargs[idx]:
                raise ValueError("Bias operation forward does not expect keyword arguments")

        # Quantization metadata
        with_quantized_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_quantizer = None
        weight_quantizer = None
        grad_output_quantizer = None
        grad_input_quantizer = None
        if with_quantized_compute:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if not recipe.delayed() and not recipe.mxfp8():
                raise RuntimeError("Userbuffers is only supported with FP8 delayed scaling recipe")
            input_quantizer = linear_op.get_quantizer("forward", 0)
            weight_quantizer = linear_op.get_quantizer("forward", 1)
            grad_output_quantizer = linear_op.get_quantizer("backward", 0)
            prev_op = basic_op_prev_ops[0]
            if prev_op is not None and prev_op.num_quantizers("backward") > 0 and recipe.delayed():
                grad_input_quantizer = prev_op.get_quantizer("backward", 0)

        # Get autocast dtype if needed
        dtype = None
        if torch.is_autocast_enabled():
            dtype = torch.get_autocast_dtype("cuda")

        # Userbuffers options
        if linear_op._userbuffers_options is None:
            raise RuntimeError("Linear op is missing dict for Userbuffers options")

        # Linear forward
        output, extra_outputs = UserbuffersForwardLinear._functional_forward(
            input=input_,
            weight=linear_op.weight,
            bias=bias,
            dtype=dtype,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            tensor_parallel_size=self.tensor_parallel_size,
            sequence_parallel=self.sequence_parallel,
            with_quantized_compute=with_quantized_compute,
            input_quantizer=input_quantizer,
            weight_quantizer=weight_quantizer,
            output_quantizer=None,  # Not supported
            ub_comm_name=linear_op._userbuffers_options["comm_name"],
        )
        x_local = extra_outputs["input"]

        # Save state for backward pass
        linear_op_ctx.save_for_backward(x_local)
        linear_op_ctx.with_quantized_compute = with_quantized_compute
        linear_op_ctx.input_quantizer = input_quantizer
        linear_op_ctx.weight_quantizer = weight_quantizer
        linear_op_ctx.grad_output_quantizer = grad_output_quantizer
        linear_op_ctx.grad_input_quantizer = grad_input_quantizer
        linear_op_ctx.dtype = dtype
        linear_op_ctx.input_dims = input_.size()
        linear_op_ctx.input_requires_grad = input_.requires_grad
        linear_op_ctx.weight_requires_grad = linear_op.weight.requires_grad
        linear_op_ctx.has_prev_op = basic_op_prev_ops[0] is not None

        return output, [() for _ in range(len(self.basic_ops))]


def fuse_userbuffers_forward_linear(
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
        return ops[0][0]

    def pop_next_op() -> FusibleOperation:
        """Remove next op from list of ops and add to sliding window"""
        nonlocal ops, window
        window.append(ops[0])
        ops = ops[1:]
        return window[-1][0]

    # Scan through ops, fusing if possible
    out = []
    while ops:
        out.extend(window)
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
        op = UserbuffersForwardLinear(
            linear=linear,
            bias=bias,
            reduce_scatter=reduce_scatter,
        )
        basic_op_idxs = [basic_op_idxs[0] for _, basic_op_idxs in window]
        window = [(op, basic_op_idxs)]

    # Return list of ops
    out.extend(window)
    return out
