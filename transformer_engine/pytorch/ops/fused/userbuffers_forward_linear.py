# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear layer forward with Userbuffers communication."""

# pylint: skip-file  ### TODO Debug Userbuffers support

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

        ### TODO Debug Userbuffers support
        raise NotImplementedError("Userbuffers support has been broken by recent refactors")

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
        with_fp8_compute: bool = False,
        input_fp8_meta: Optional[dict[str, Any]] = None,
        weight_fp8_meta: Optional[dict[str, Any]] = None,
        output_fp8_meta: Optional[dict[str, Any]] = None,
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
        with_fp8_compute: bool, default = `False`
            Whether to perform compute in FP8
        input_fp8_meta: dict, optional
            FP8 metadata for casting input tensor to FP8. Required for
            FP8 compute if input is not already in FP8.
        weight_fp8_meta: dict, optional
            FP8 metadata for casting weight tensor to FP8. Required for
            FP8 compute if weight is not already in FP8.
        output_fp8_meta: dict, optional
            FP8 metadata for casting output tensor to FP8
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

        # Input tensor dims
        input_dims = tuple(input.size())
        weight_dims = tuple(weight.size())
        if len(weight_dims) != 2:
            raise ValueError(f"Weight tensor is not 2D (shape={weight_dims})")
        if len(input_dims) == 0 or weight_dims[1] != input_dims[-1]:
            raise ValueError(
                f"Input tensor (shape={input_dims}) "
                f"and weight tensor (shape={weight_dims}) "
                "are not compatible"
            )

        # Output tensor dims
        output_dims = list(input_dims)
        output_dims[0] = -1
        output_dims[-1] = weight_dims[0]

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

        # Check if FP8 is enabled
        if with_fp8_compute:
            if input_fp8_meta is None and not is_float8_tensor(input):
                raise ValueError("No FP8 metadata was provided for casting input to FP8")
            if weight_fp8_meta is None and not is_float8_tensor(weight):
                raise ValueError("No FP8 metadata was provided for casting weight to FP8")
        else:
            input_fp8_meta = None
            weight_fp8_meta = None
            output_fp8_meta = None
        with_fp8_output = (
            with_fp8_compute and tensor_parallel_mode != "row" and output_fp8_meta is not None
        )

        # Get Userbuffers communicator
        ub_comm = get_ub(ub_comm_name + "_fprop")
        ub_local_buffer = ub_comm.get_ubuf_output(0)
        ub_global_buffer = ub_comm.get_ubuf_output(1)
        with_ub_all_gather = tensor_parallel_mode == "column"
        with_ub_reduce_scatter = tensor_parallel_mode == "row"

        # Choose Userbuffers communication algorithm
        ub_algo = None
        if with_ub_all_gather:
            if with_fp8_compute and ub_comm.is_atomic_gemm():
                ub_algo = CommOverlapAlgo.ATOMIC_GEMM_AG_P2P
            else:
                ub_algo = CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P
        elif with_ub_reduce_scatter:
            is_atomic_gemm = with_fp8_compute and ub_comm.is_atomic_gemm()
            ub_algo = {
                (True, True): CommOverlapAlgo.ATOMIC_GEMM_RS_P2P,
                (True, False): CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P,
                (False, True): CommOverlapAlgo.ATOMIC_GEMM_RS,
                (False, False): CommOverlapAlgo.SPLIT_PIPELINED_RS,
            }[(ub_comm.is_p2p_overlap(), is_atomic_gemm)]
        else:
            raise RuntimeError("Could not choose Userbuffers communication algorithm")

        # Cast input tensor to correct dtype
        x_local = reshape(
            input,
            (-1, input_dims[-1]),
            device=device,
            dtype=dtype,
        )
        if with_fp8_compute and not is_float8_tensor(x_local):
            fp8_dtype = get_fp8_te_dtype(
                input_fp8_meta["recipe"],
                fprop_tensor=True,
            )
            with_transpose_cache = weight.requires_grad
            if tensor_parallel_mode == "column" and sequence_parallel:
                with_transpose_cache = False
            x_local = Float8Tensor.to_float8(
                x_local,
                fp8_meta=input_fp8_meta,
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                data=(ub_local_buffer if with_ub_all_gather else None),
                with_transpose_cache=with_transpose_cache,
            )
        elif not with_fp8_compute and is_float8_tensor(x_local):
            if with_ub_all_gather:
                x_local = ub_local_buffer.copy_(x_local)
            else:
                x_local = x_local.dequantize()

        # Initialize buffers for UB all-gather if needed
        x = x_local
        if with_ub_all_gather:
            if with_fp8_compute:
                x = Float8Tensor.make_like(x_local, data=ub_global_buffer)
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
            w = Float8Tensor.to_float8(
                w,
                fp8_meta=weight_fp8_meta,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
            )
        elif not with_fp8_compute and is_float8_tensor(w):
            w = w.dequantize()

        # Check bias tensor
        b = None
        if bias is not None:
            b = convert_tensor(
                bias,
                device=device,
                dtype=dtype,
                memory_format=torch.contiguous_format,
            )

        # Construct output tensor
        y = None
        y_local = None
        if with_ub_reduce_scatter:
            # Initialize buffers for UB reduce-scatter
            if with_fp8_output:
                fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=True)
                fp8_dtype = get_fp8_te_dtype(
                    output_fp8_meta["recipe"],
                    fprop_tensor=True,
                )
                y = Float8Tensor(
                    data=ub_global_buffer,
                    fp8_meta=output_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=output_fp8_meta[fp8_meta_key].scale_inv[0],
                    dtype=dtype,
                )
                ub_comm.set_ubuf_scale_inv(y._scale_inv)
            else:
                y = ub_global_buffer
            y_local = torch.empty(
                (x.size(0) // tensor_parallel_size, weight_dims[0]),
                dtype=dtype,
                device=device,
            )
        else:
            # Allocate output tensor
            if with_fp8_output:
                fp8_dtype = get_fp8_te_dtype(
                    output_fp8_meta["recipe"],
                    fprop_tensor=True,
                )
                data = torch.empty(
                    (x.size(0), weight_dims[0]),
                    dtype=torch.uint8,
                    device=device,
                )
                y = Float8Tensor(
                    data=data,
                    fp8_meta=output_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    dtype=dtype,
                )
            else:
                y = torch.empty(
                    (x.size(0), weight_dims[0]),
                    dtype=dtype,
                    device=device,
                )
            y_local = y

        # Perform GEMM
        if with_fp8_compute:
            kwargs = {
                "out": y,
                "bias": b,
                "use_bias": (b is not None),
                "use_split_accumulator": False,
                "ub_algo": ub_algo,
                "ub": ub_comm,
            }
            if with_ub_all_gather:
                kwargs["extra_output_tensor"] = x_local._data
            if with_ub_reduce_scatter:
                kwargs["extra_output_tensor"] = y_local
            if with_fp8_output:
                fp8_meta, fp8_meta_index = get_fp8_meta_from_fp8_tensor(y)
                kwargs.update(
                    {
                        "out": y._data,
                        "out_index": fp8_meta_index,
                        "fp8_meta_tensor": fp8_meta,
                        "D_dtype": y._fp8_dtype,
                    }
                )
            fp8_gemm(
                w._data,
                w._scale_inv,
                0,
                w._fp8_dtype,
                x._data,
                x._scale_inv,
                0,
                x._fp8_dtype,
                y.dtype,
                get_workspace(),
                **kwargs,
            )
        else:
            kwargs = {
                "out": y,
                "bias": b,
                "use_bias": (b is not None),
                "ub_algo": ub_algo,
                "ub": ub_comm,
            }
            if with_ub_all_gather:
                kwargs["extra_output_tensor"] = x_local
            if with_ub_reduce_scatter:
                kwargs["extra_output_tensor"] = y_local
            gemm(w, x, y.dtype, get_workspace(), **kwargs)

        # Reshape output tensor
        out = reshape(y_local, output_dims)

        # Return cast tensors
        extra_outputs = {"input": x_local, "weight": w}
        return out, extra_outputs

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

        # FP8 metadata
        with_fp8_compute = FP8GlobalStateManager.is_fp8_enabled()
        input_fp8_meta = None
        weight_fp8_meta = None
        output_fp8_meta = None
        grad_output_fp8_meta = None
        grad_input_fp8_meta = None
        if with_fp8_compute:
            input_fp8_meta = linear_op.get_fp8_meta("input")
            weight_fp8_meta = linear_op.get_fp8_meta("param")
            next_op = basic_op_next_ops[-1]
            if next_op is not None and next_op.num_fp8_scales("input") > 0:
                output_fp8_meta = next_op.get_fp8_meta("input")
            grad_output_fp8_meta = linear_op.get_fp8_meta("grad_output")
            prev_op = basic_op_prev_ops[0]
            if prev_op is not None and prev_op.num_fp8_scales("grad_output") > 0:
                grad_input_fp8_meta = prev_op.get_fp8_meta("grad_output")

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
            device=linear_op.device,
            dtype=dtype,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            tensor_parallel_size=self.tensor_parallel_size,
            sequence_parallel=self.sequence_parallel,
            with_fp8_compute=with_fp8_compute,
            input_fp8_meta=input_fp8_meta,
            weight_fp8_meta=weight_fp8_meta,
            output_fp8_meta=output_fp8_meta,
            ub_comm_name=linear_op._userbuffers_options["comm_name"],
        )
        x_local = extra_outputs["input"]

        # Save state for backward pass
        linear_op_ctx.save_for_backward(x_local)
        linear_op_ctx.with_fp8_compute = with_fp8_compute
        linear_op_ctx.weight_fp8_meta = weight_fp8_meta
        linear_op_ctx.grad_output_fp8_meta = grad_output_fp8_meta
        linear_op_ctx.grad_input_fp8_meta = grad_input_fp8_meta
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

    return ops  ### TODO Debug Userbuffers support

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
