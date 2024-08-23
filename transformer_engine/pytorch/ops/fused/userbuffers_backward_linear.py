# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear layer backward with Userbuffers communication."""

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional

import torch

from ...cpp_extensions import FP8TensorMeta, UbufOverlapAlgo, fp8_gemm, gemm
from ...distributed import get_distributed_world_size
from ...float8_tensor import Float8Tensor
from ...fp8 import FP8GlobalStateManager, get_fp8_te_dtype
from ...module.base import get_ub, get_workspace
from ..basic import BasicLinear, Bias, ReduceScatter
from ..op import (
    BasicOperation,
    FusedOperation,
    FusibleOperation,
    OperationContext,
)
from .._common import (
    canonicalize_device,
    canonicalize_dtype,
    convert_tensor,
    is_float8_tensor,
    reshape,
)

class UserbuffersBackwardLinear(FusedOperation):

    def __init__(
        self,
        *,
        linear: BasicLinear,
        bias: Optional[Bias],
        reduce_scatter: Optional[ReduceScatter],
    ) -> None:

        # Basic operations that comprise this fused operation
        op_idxs = dict(
            linear=None,
            bias=None,
            reduce_scatter=None,
        )
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
    def _functional_backward_dgrad(
        grad_output_local: torch.Tensor,
        weight: Optional[torch.Tensor],
        *,
        weight_requires_grad: bool,
        device: torch.device,
        dtype: torch.dtype,
        tensor_parallel_mode: str,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup],
        tensor_parallel_size: int,
        with_fp8_compute: bool,
        with_fp8_grad_input: bool,
        weight_fp8_meta: Optional[dict[str, Any]],
        grad_output_fp8_meta: Optional[dict[str, Any]],
        grad_input_fp8_meta: Optional[dict[str, Any]],
        ub_comm_name: str,
    ):

        # Get Userbuffers communicator
        ub_comm = get_ub(ub_comm_name + "_dgrad")
        ub_local_buffer = ub_comm.get_ubuf_output(0)
        ub_global_buffer = ub_comm.get_ubuf_output(1)
        with_ub_reduce_scatter = tensor_parallel_mode == "column"
        with_ub_all_gather = tensor_parallel_mode == "row"

        # Choose Userbuffers communication algorithm
        ub_algo = None
        if with_ub_all_gather:
            if with_fp8_compute and ub_comm.is_atomic_gemm():
                ub_algo = UbufOverlapAlgo.ATOMIC_GEMM_AG_P2P
            else:
                ub_algo = UbufOverlapAlgo.SPLIT_PIPELINED_AG_P2P
        elif with_ub_reduce_scatter:
            ub_algo = UbufOverlapAlgo.BULK_OVERLAP_AG  ### TODO Is this right?
        else:
            raise RuntimeError("Could not choose Userbuffers communication algorithm")

        # Cast grad output tensor to correct dtype
        dy_local = grad_output_local
        if with_fp8_compute and not is_float8_tensor(dy_local):
            fp8_dtype = get_fp8_te_dtype(
                grad_output_fp8_meta["recipe"],
                fprop_tensor=False,
            )
            if with_ub_all_gather:
                data = ub_local_buffer
            else:
                data = torch.empty_like(dy_local, dtype=torch.uint8)
            dy_fp8 = Float8Tensor(
                data=data,
                fp8_meta=grad_output_fp8_meta,
                fp8_meta_forward=False,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=torch.empty([1], dtype=torch.float32, device=device),
                dtype=dtype,
            )
            with_cast_transpose = weight_requires_grad and not with_ub_all_gather
            if with_cast_transpose:
                dy_fp8.cast_transpose_(dy_local)
            else:
                dy_fp8.copy_(dy_local)
            dy_local = dy_fp8
        elif not with_fp8_compute and is_float8_tensor(dy_local):
            if with_ub_all_gather:
                dy_local = ub_local_buffer.copy_(dy_local)
            else:
                dy_local = dy_local.from_float8()

        # Initialize buffers for UB all-gather if needed
        dy = dy_local
        if with_ub_all_gather:
            if with_fp8_compute:
                dy = Float8Tensor.make_like(dy_local, data=ub_global_buffer)
                if dy_local._data.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(dy_local._data)
            else:
                dy = ub_global_buffer
                if dy_local.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(dy_local)

        # Check weight tensor
        w = convert_tensor(
            weight,
            device=device,
            dtype=dtype,
            memory_format=torch.contiguous_format,
        )
        if with_fp8_compute and not is_float8_tensor(w):
            fp8_dtype = get_fp8_te_dtype(
                weight_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            w_fp8 = Float8Tensor(
                data=torch.empty_like(w, dtype=torch.uint8),
                fp8_meta=weight_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                fp8_scale_inv=torch.empty([1], dtype=torch.float32, device=device),
                dtype=dtype,
            )
            w_fp8.cast_transpose_(w)
            w = w_fp8
        elif not with_fp8_compute and is_float8_tensor(w):
            w = w.from_float8()

        # Construct grad input tensor
        dx = None
        dx_local = None
        if with_ub_reduce_scatter:
            # Initialize buffers for UB reduce-scatter
            dx = ub_global_buffer
            dx_local = torch.empty(
                (dy.size(0) // tensor_parallel_size, w.size(-1)),
                dtype=dtype,
                device=device,
            )
        else:
            # Allocate grad input tensor
            if with_fp8_grad_input:
                fp8_dtype = get_fp8_te_dtype(
                    grad_input_fp8_meta["recipe"],
                    fprop_tensor=False,
                )
                data = torch.empty(
                    (dy.size(0), w.size(-1)),
                    dtype=torch.uint8,
                    device=device,
                )
                dx = Float8Tensor(
                    data=data,
                    fp8_meta=grad_input_fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    dtype=dtype,
                )
            else:
                dx = torch.empty(
                    (dy.size(0), w.size(-1)),
                    dtype=dtype,
                    device=device,
                )
            dx_local = dx

        # Perform dgrad GEMM
        if with_fp8_compute:
            kwargs = dict(
                out=dx,
                use_split_accumulator=False, ### TODO ?
                ub_algo=ub_algo,
                ub=ub_comm,
            )
            if with_ub_reduce_scatter:
                kwargs["extra_output_tensor"] = dx_local
            if with_fp8_grad_input:
                if dx._fp8_meta is None:
                    # Hackily create FP8TensorMeta if needed
                    fp8_meta = FP8TensorMeta()
                    fp8_meta.scale = dx._scale_inv.reciprocal()
                    fp8_meta.amax_history = torch.empty(1, 1, dtype=torch.float32, device=device)
                    fp8_meta.scale_inv = dx._scale_inv
                    fp8_meta_index = 0
                else:
                    # Get FP8TensorMeta from Float8Tensor
                    fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(
                        forward=dx._fp8_meta_forward,
                    )
                    fp8_meta = dx._fp8_meta[fp8_meta_key]
                    fp8_meta_index = dx._fp8_meta_index
                kwargs.update(
                    dict(
                        out=dx._data,
                        out_index=fp8_meta_index,
                        fp8_meta_tensor=fp8_meta,
                        D_dtype=dx._fp8_dtype,
                    )
                )
            fp8_gemm(
                w.transpose_2d(),
                w._scale_inv,
                0,
                w._fp8_dtype,
                dy._data,
                dy._scale_inv,
                0,
                dy._fp8_dtype,
                dy.dtype,
                get_workspace(),
                **kwargs,
            )
        else:
            kwargs = dict(
                layout="NN",
                out=dx,
                ub_algo=ub_algo,
                ub=ub_comm,
            )
            if with_ub_reduce_scatter:
                kwargs["extra_output_tensor"] = dx_local
            gemm(w, dy, dx.dtype, get_workspace(), **kwargs)

        return dx_local, dy

    @staticmethod
    def _functional_backward(
        grad_output: torch.Tensor,
        input: Optional[torch.Tensor],  # pylint: disable=redefined-builtin
        weight: Optional[torch.Tensor],
        input_dims: Iterable[int],
        weight_dims: Iterable[int],
        *,
        weight_requires_grad: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        grad_weight: Optional[torch.Tensor] = None,
        accumulate_into_grad_weight: bool = False,
        grad_bias: Optional[torch.Tensor] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        tensor_parallel_size: Optional[int] = None,
        sequence_parallel: bool = False,
        with_fp8_compute: bool = False,
        weight_fp8_meta: Optional[dict[str, Any]] = None,
        grad_output_fp8_meta: Optional[dict[str, Any]] = None,
        grad_input_fp8_meta: Optional[dict[str, Any]] = None,
        ub_comm_name: str,
    ):

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

        # Check input tensor
        output_dims = tuple(grad_output.size())
        input_dims = tuple(input_dims)
        weight_dims = tuple(weight_dims)
        if len(weight_dims) != 2:
            raise ValueError(f"Weight tensor is not 2D (shape={weight_dims})")
        if len(input_dims) == 0 or weight_dims[1] != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) "
                "are not compatible"
            )
        if weight_dims[0] != output_dims[-1]:
            raise ValueError(
                f"Grad output tensor (shape={output_dims}) "
                f"and weight tensor (shape={weight_dims}) "
                "are not compatible"
            )

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
            raise RuntimeError(
                f"Invalid configuration for Userbuffers ({sequence_parallel=})"
            )

        # Check if FP8 is enabled
        if with_fp8_compute:
            if grad_output_fp8_meta is None and not is_float8_tensor(grad_output):
                raise ValueError("No FP8 metadata was provided for casting output gradient to FP8")
        else:
            weight_fp8_meta = None
            grad_output_fp8_meta = None
            grad_input_fp8_meta = None
        with_fp8_grad_input = (
            with_fp8_compute
            and tensor_parallel_mode != "column"
            and grad_input_fp8_meta is not None
        )

        # Perform dgrad GEMM
        dy_local = reshape(
            grad_output,
            (-1, output_dims[-1]),
            device=device,
            dtype=dtype,
        )
        dx_local, dy = UserbuffersBackwardLinear._functional_backward_dgrad(
            grad_output_local=dy_local,
            weight=weight,
            weight_requires_grad=weight_requires_grad,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=tensor_parallel_group,
            tensor_parallel_size=tensor_parallel_size,
            with_fp8_compute=with_fp8_compute,
            with_fp8_grad_input=with_fp8_grad_input,
            weight_fp8_meta=weight_fp8_meta,
            grad_output_fp8_meta=grad_output_fp8_meta,
            grad_input_fp8_meta=grad_output_fp8_meta,
            ub_comm_name=ub_comm_name,
        )
        grad_input = reshape(dx_local, input_dims)

        # Perform wgrad GEMM
        if not weight_requires_grad:
            grad_weight = None
        else:
            raise NotImplementedError()  ### TODO Implement

        return grad_input, grad_weight

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
        bias = None
        if self._op_idxs["bias"] is not None:
            idx = self._op_idxs["bias"]
            bias_op = self.basic_ops[idx]
            bias = bias_op.bias

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
        grad_bias = None  ### TODO Implement
        grad_input, grad_weight = UserbuffersBackwardLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=linear_op.weight,
            input_dims=linear_op_ctx.input_dims,
            weight_dims=linear_op.weight.size(),
            weight_requires_grad=linear_op_ctx.weight_requires_grad,
            device=linear_op.device,
            dtype=linear_op.dtype,
            grad_weight=grad_weight,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            grad_bias=grad_bias,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_fp8_compute=linear_op_ctx.with_fp8_compute,
            weight_fp8_meta=linear_op_ctx.weight_fp8_meta,
            grad_output_fp8_meta=linear_op_ctx.grad_output_fp8_meta,
            grad_input_fp8_meta=linear_op_ctx.grad_input_fp8_meta,
            ub_comm_name=linear_op._userbuffers_options["comm_name"],
        )
        if accumulate_into_main_grad:
            grad_weight = None

        # Clear input tensor if possible
        if linear_op_ctx.has_prev_op:
            clear_tensor_data(x_local)

        # Return gradients
        grad_params = [() for _ in range(len(self.basic_ops))]
        grad_params[self._op_idxs["linear"]] = (grad_weight,)
        if bias_op is not None:
            grad_params[self._op_idxs["bias"]] = (grad_bias,)
        grad_extra_inputs = [() for _ in range(len(self.basic_ops))]
        return grad_input, grad_params, grad_extra_inputs


def fuse_userbuffers_backward_linear(
    ops: list[tuple[FusibleOperation, list[int]]],
) -> list[tuple[FusibleOperation, list[int]]]:

    # Sliding window in list of ops
    window = []

    def peek_next_op() -> Optional[FusibleOperation]:
        nonlocal ops
        if not ops:
            return None
        return ops[-1][0]

    def pop_next_op() -> FusibleOperation:
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
            if linear.tensor_parallel_mode == "row" and bias is not None:
                continue
        else:
            if linear.tensor_parallel_mode is not None:
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
