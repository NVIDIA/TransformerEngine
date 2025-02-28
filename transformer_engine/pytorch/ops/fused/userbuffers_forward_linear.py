# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear layer forward with Userbuffers communication."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformer_engine_torch import CommOverlapAlgo
from ...cpp_extensions import general_gemm
from ...distributed import get_distributed_world_size
from ...float8_tensor import Float8Tensor
from ...fp8 import FP8GlobalStateManager, get_fp8_te_dtype
from ...module.base import get_ub, get_workspace
from ...utils import canonicalize_device, canonicalize_dtype
from ..basic import BasicLinear, Bias, ReduceScatter
from ..op import (
    BasicOperation,
    FusedOperation,
    FusibleOperation,
    OperationContext,
)
from .._common import (
    convert_tensor,
    get_fp8_meta_from_fp8_tensor,
    is_float8_tensor,
    reshape,
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

        # Check tensor dims
        w_size = weight.size()
        if len(w_size) != 2:
            raise ValueError(f"Expected 2D weight tensor (got shape={tuple(w_size)})")
        x_local_size = input.size()
        if len(x_local_size) < 2:
            raise ValueError(
                f"Expected input tensor with at least 2 dims (got shape={tuple(x_size)})"
            )
        if x_local_size[-1] != w_size[1]:
            raise ValueError(
                f"Input tensor (shape={tuple(x_size)}) is not compatible with "
                f"weight tensor (shape={tuple(w_size)})"
            )
        if tensor_parallel_mode == "row" and x_local_size[0] % tensor_parallel_size != 0:
            raise ValueError(
                f"Input tensor (shape={tuple(x_size)}) is not compatible with "
                f"row tensor parallelism (size={tensor_parallel_size})"
            )
        y_local_size = list(x_local_size)
        if tensor_parallel_mode == "row":
            y_local_size[0] //= tensor_parallel_size
        else:
            y_local_size[0] *= tensor_parallel_size
        y_local_size[-1] = w_size[0]

        # Check quantizers
        if with_quantized_compute:
            if input_quantizer is None:
                raise ValueError("Missing quantizer for input tensor")
            if not isinstance(input_quantizer, Float8Quantizer):
                raise ValueError(
                    "Invalid quantizer for input tensor (Userbuffers only supports FP8)"
                )
            if weight_quantizer is None:
                raise ValueError("Missing quantizer for weight tensor")
            if not isinstance(weight_quantizer, Float8Quantizer):
                raise ValueError(
                    "Invalid quantizer for weight tensor (Userbuffers only supports FP8)"
                )
            if output_quantizer is not None and not isinstance(output_quantizer, Float8Quantizer):
                raise ValueError(
                    "Invalid quantizer for output tensor (Userbuffers only supports FP8)"
                )
        else:
            input_quantizer = None
            weight_quantizer = None
            output_quantizer = None
        with_quantized_output = (
            with_quantized_compute
            and tensor_parallel_mode != "row"
            and output_quantizer is not None
        )

        # Get Userbuffers communicator
        ub_comm = get_ub(ub_comm_name + "_fprop")
        ub_local_buffer = ub_comm.get_ubuf_output(0)
        ub_global_buffer = ub_comm.get_ubuf_output(1)
        with_ub_all_gather = tensor_parallel_mode == "column"
        with_ub_reduce_scatter = tensor_parallel_mode == "row"
        ub_type = (
            tex.CommOverlapType.AG if with_ub_all_gather else tex.CommOverlapType.RS
        )

        # Cast input tensor to correct dtype
        x_local = input
        own_quantized_x_local = False
        if with_quantized_compute:
            if not isinstance(x_local, Float8Tensor):
                if with_ub_all_gather:
                    input_quantizer.set_usage(rowwise=True, columnwise=False)
                    x_local_fp8 = input_quantizer.create_tensor_from_data(
                        ub_local_buffer,
                        fake_dtype=dtype,
                    )
                    x_local_fp8.copy_(x_local)
                    x_local = x_local_fp8
                else:
                    input_quantizer.set_usage(rowwise=True)
                    x_local = input_quantizer(x_local)
                own_quantized_x_local = True
        else:
            if isinstance(x_local, QuantizedTensor):
                x_local = x_local.dequantize(dtype=dtype)
            elif x_local.dtype != dtype:
                x_local = x_local.to(dtype=dtype)

        # Initialize buffers for UB all-gather if needed
        x = x_local
        if with_ub_all_gather:
            if with_quantized_compute:
                x = input_quantizer.create_tensor_from_data(
                    ub_global_buffer,
                    fake_dtype=dtype,
                )
                if x_local._data.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(x_local._data)
                else:
                    x_local._data = torch.empty_like(x_local._data)
            else:
                x = ub_global_buffer
                if x_local.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(x_local)
                else:
                    x_local = torch.empty_like(x_local)

        # Check weight tensor
        w = weight
        w_is_quantized = isinstance(w, QuantizedTensor)
        if with_quantized_compute and not w_is_quantized:
            weight_quantizer.set_usage(rowwise=True)
            w = weight_quantizer(w)
        elif not with_quantized_compute and w_is_quantized:
            w = w.dequantize()
        if not with_quantized_compute and w.dtype != dtype:
            w = w.to(dtype=dtype)

        # Construct output tensor
        y = None
        y_local = None
        if with_ub_reduce_scatter:
            y_local = torch.empty(y_local_size, dtype=dtype, device=device)
            if with_quantized_output:
                output_quantizer.set_usage(rowwise=True, columnwise=False)
                y = output_quantizer.create_tensor_from_data(
                    ub_global_buffer,
                    fake_dtype=dtype,
                )
                ub_comm.set_ubuf_scale_inv(y._scale_inv)
            else:
                y = ub_global_buffer

        # Perform GEMM
        y, *_, rs_output = general_gemm(
            w,
            x,
            get_workspace(),
            out_dtype=dtype,
            quantization_params=output_quantizer,
            out=y,
            bias=bias,
            use_split_accumulator=_2X_ACC_FPROP,
            ub=ub,
            ub_type=ub_type,
            extra_output=y_local,
        )
        if with_ub_reduce_scatter:
            y_local = rs_output
        else:
            y_local = y

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
        output_quantizer = None
        grad_output_quantizer = None
        grad_input_quantizer = None
        if with_quantized_compute:
            recipe = FP8GlobalStateManager.get_fp8_recipe()
            if not recipe.delayed():
                raise RuntimeError(
                    "Userbuffers is only supported with FP8 delayed scaling recipe"
                )
            input_quantizer = linear_op.get_quantizer("forward", 0)
            weight_quantizer = linear_op.get_quantizer("forward", 1)
            next_op = basic_op_next_ops[-1]
            if (
                next_op is not None
                and next_op.num_quantizers("forward") > 0
                and recipe.delayed()
            ):
                output_quantizer = next_op.get_quantizer("forward", 0)
            grad_output_quantizer = linear_op.get_quantizer("backward", 0)
            prev_op = basic_op_prev_ops[0]
            if (
                prev_op is not None
                and prev_op.num_quantizers("backward") > 0
                and recipe.delayed()
            ):
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
            output_quantizer=output_quantizer,
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
