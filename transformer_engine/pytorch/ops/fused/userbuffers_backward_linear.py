# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Linear layer backward with Userbuffers communication."""

# pylint: skip-file  ### TODO Debug Userbuffers support

from __future__ import annotations
from collections.abc import Iterable
from typing import Any, Optional
import warnings

import torch

from transformer_engine_torch import CommOverlapAlgo
from ...cpp_extensions import general_gemm
from ...distributed import get_distributed_world_size
from ...float8_tensor import Float8Tensor
from ...fp8 import FP8GlobalStateManager, get_fp8_te_dtype
from ...module.base import get_ub, get_workspace
from ...utils import canonicalize_device, canonicalize_dtype, clear_tensor_data
from ..basic import BasicLinear, Bias, ReduceScatter
from ..op import FusedOperation, FusibleOperation, OperationContext
from .._common import (
    convert_tensor,
    get_fp8_meta_from_fp8_tensor,
    is_float8_tensor,
    reshape,
)


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

        ### TODO Debug Userbuffers support
        raise NotImplementedError("Userbuffers support has been broken by recent refactors")

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
        input_dims: Iterable[int],
        weight_dims: Iterable[int],
        *,
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
        with_fp8_compute: bool = False,
        input_fp8_meta: Optional[dict[str, Any]] = None,
        weight_fp8_meta: Optional[dict[str, Any]] = None,
        grad_output_fp8_meta: Optional[dict[str, Any]] = None,
        grad_input_fp8_meta: Optional[dict[str, Any]] = None,
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
        input_dims: iterable of int
            Input tensor dimensions
        weight_dims: iterable of int
            Weight tensor dimensions
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
        with_fp8_compute: bool, default = `False`
            Whether to perform compute in FP8
        input_fp8_meta: dict, optional
            FP8 metadata for casting input tensor to FP8. Required for
            FP8 compute if input is not already in FP8.
        weight_fp8_meta: dict, optional
            FP8 metadata for casting weight tensor to FP8. Required for
            FP8 compute if weight is not already in FP8.
        grad_output_fp8_meta: dict, optional
            FP8 metadata for casting loss gradient w.r.t. output
            tensor to FP8. Required if output grad is not already in
            FP8.
        grad_input_fp8_meta: dict, optional
            FP8 metadata for casting loss gradient w.r.t. input
            tensor to FP8
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
            raise RuntimeError(f"Invalid configuration for Userbuffers ({sequence_parallel=})")

        # Check if FP8 is enabled
        if with_fp8_compute:
            if grad_output_fp8_meta is None and not is_float8_tensor(grad_output):
                raise ValueError("No FP8 metadata was provided for casting output gradient to FP8")
        else:
            input_fp8_meta = None
            weight_fp8_meta = None
            grad_output_fp8_meta = None
            grad_input_fp8_meta = None
        with_fp8_grad_input = (
            with_fp8_compute
            and tensor_parallel_mode != "column"
            and grad_input_fp8_meta is not None
        )

        # Get Userbuffers communicators and algorithms
        # Note: communication patterns are (1) overlap dy all-gather
        # with dgrad GEMM, (2) overlap x all-gather with dgrad GEMM
        # and dx reduce-scatter with wgrad GEMM, (3) overlap dx
        # reduce-scatter with dgrad GEMM.
        with_ub_all_gather_dy = False
        with_ub_reduce_scatter_dx = False
        with_ub_all_gather_x = False
        ub_comm_dy = None
        ub_comm_dx = None
        ub_comm_x = None
        ub_algo_dy = None
        ub_algo_dx = None
        ub_algo_x = None
        if tensor_parallel_mode == "row":
            with_ub_all_gather_dy = True
            ub_comm_dy = get_ub(ub_comm_name + "_dgrad")
            if with_fp8_compute and ub_comm_dy.is_atomic_gemm():
                ub_algo_dy = CommOverlapAlgo.ATOMIC_GEMM_AG_P2P
            else:
                ub_algo_dy = CommOverlapAlgo.SPLIT_PIPELINED_AG_P2P
        elif tensor_parallel_mode == "column":
            with_ub_reduce_scatter_dx = True
            if weight_requires_grad:
                with_ub_all_gather_x = True
                ub_comm_dx = get_ub(ub_comm_name + "_wgrad")
                ub_comm_x = get_ub(ub_comm_name + "_dgrad")
                ub_algo_dx = CommOverlapAlgo.BULK_OVERLAP_RS
                ub_algo_x = CommOverlapAlgo.BULK_OVERLAP_AG
            else:
                with_ub_all_gather_x = False
                ub_comm_dx = get_ub(ub_comm_name + "_dgrad")
                is_atomic_gemm = with_fp8_compute and ub_comm_dx.is_atomic_gemm()
                ub_algo_dx = {
                    (True, True): CommOverlapAlgo.ATOMIC_GEMM_RS_P2P,
                    (True, False): CommOverlapAlgo.SPLIT_PIPELINED_RS_P2P,
                    (False, True): CommOverlapAlgo.ATOMIC_GEMM_RS,
                    (False, False): CommOverlapAlgo.SPLIT_PIPELINED_RS,
                }[(ub_comm_dx.is_p2p_overlap(), is_atomic_gemm)]

        # Check grad output tensor
        # Note: Possibly fuse cast with computing grad bias
        dy_local = reshape(
            grad_output,
            (-1, output_dims[-1]),
            device=device,
            dtype=dtype,
        )
        db = None
        db_async = None
        if bias_requires_grad and with_fp8_compute and with_ub_all_gather_dy:
            # We don't have a grad bias impl that takes FP8 input. For
            # cases where we cast to FP8 and all-gather, it's better
            # to compute the grad bias on ungathered, non-FP8 values.
            db = dy_local.sum(dim=0)
            db_async = torch.distributed.all_reduce(
                db,
                group=tensor_parallel_group,
                async_op=True,
            )
        if with_fp8_compute and not is_float8_tensor(dy_local):
            fp8_dtype = get_fp8_te_dtype(
                grad_output_fp8_meta["recipe"],
                fprop_tensor=False,
            )
            if bias_requires_grad and db is None:
                # Fused cast-transpose-bgrad
                fp8_meta_key = FP8GlobalStateManager.get_meta_tensor_key(forward=False)
                fp8_scale_inv = torch.empty([1], dtype=torch.float32, device=device)
                db, data, data_transpose = fp8_cast_transpose_bgrad_fused(
                    dy_local,
                    grad_output_fp8_meta[fp8_meta_key],
                    0,
                    fp8_dtype,
                    scale_inv=fp8_scale_inv,
                )
                if with_ub_all_gather_dy:
                    data = ub_comm_dy.get_ubuf_output(0).copy_(data)
                dy_local = Float8Tensor(
                    data=data,
                    fp8_meta=grad_output_fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    fp8_scale_inv=fp8_scale_inv,
                    dtype=dtype,
                    data_transpose=data_transpose,
                )
            else:
                dy_local = Float8Tensor.to_float8(
                    dy_local,
                    fp8_meta=grad_output_fp8_meta,
                    fp8_meta_forward=False,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    data=(ub_comm_dy.get_ubuf_output(0) if with_ub_all_gather_dy else None),
                    with_transpose_cache=(not with_ub_all_gather_dy),
                )
        elif not with_fp8_compute and is_float8_tensor(dy_local):
            if with_ub_all_gather_dy:
                ub_local_buffer = ub_comm_dy.get_ubuf_output(0)
                dy_local = ub_local_buffer.copy_(dy_local)
            else:
                dy_local = dy_local.dequantize()

        if bias_requires_grad and db is None and with_fp8_compute and with_ub_all_gather_dy:
            # We don't have a fused grad bias impl that takes FP8
            # input. For cases where we cast to FP8 and all-gather,
            # it's better to compute the grad bias on ungathered,
            # non-FP8 values.
            db = dy_local.sum(dim=0)
            db_async = torch.distributed.all_reduce(
                db,
                group=tensor_parallel_group,
                async_op=True,
            )

        # Check input tensor
        x_local = None
        if weight_requires_grad:
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
                x_local = Float8Tensor.to_float8(
                    x_local,
                    fp8_meta=input_fp8_meta,
                    fp8_meta_forward=True,
                    fp8_meta_index=0,
                    fp8_dtype=fp8_dtype,
                    data=(ub_comm_x.get_ubuf_output(0) if with_ub_all_gather_x else None),
                    with_transpose_cache=(not with_ub_all_gather_x),
                )
            elif not with_fp8_compute and is_float8_tensor(x_local):
                if with_ub_all_gather_x:
                    ub_local_buffer = ub_comm_x.get_ubuf_output(0)
                    x_local = ub_local_buffer.copy_(x_local)
                else:
                    x_local = x_local.dequantize()

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
                fp8_meta_forward=True,
                fp8_meta_index=0,
                fp8_dtype=fp8_dtype,
                with_transpose_cache=True,
            )
        elif not with_fp8_compute and is_float8_tensor(w):
            w = w.dequantize()

        # Initialize buffers for UB all-gather if needed
        dy = dy_local
        x = x_local
        if with_ub_all_gather_dy:
            ub_local_buffer = ub_comm_dy.get_ubuf_output(0)
            ub_global_buffer = ub_comm_dy.get_ubuf_output(1)
            if with_fp8_compute:
                dy = Float8Tensor.make_like(dy_local, data=ub_global_buffer)
                if dy_local._data.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(dy_local._data)
            else:
                dy = ub_global_buffer
                if dy_local.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(dy_local)
        if with_ub_all_gather_x:
            ub_local_buffer = ub_comm_x.get_ubuf_output(0)
            ub_global_buffer = ub_comm_x.get_ubuf_output(1)
            if with_fp8_compute:
                x = Float8Tensor.make_like(x_local, data=ub_global_buffer)
                if x_local._data.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(x_local._data)
            else:
                x = ub_global_buffer
                if x_local.data_ptr() != ub_local_buffer.data_ptr():
                    ub_local_buffer.copy_(x_local)

        # Construct grad input tensor
        dx = None
        dx_local = None
        if with_ub_reduce_scatter_dx:
            # Initialize buffers for UB reduce-scatter
            dx = ub_comm_dx.get_ubuf_output(1)
            ub_local_buffer = ub_comm_dx.get_ubuf_output(0)
            if with_ub_all_gather_x:
                dx_local = ub_local_buffer
            else:
                dx_local = torch.empty_like(ub_local_buffer)
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

        # Allocate grad input tensor
        if grad_weight is None:
            if accumulate_into_grad_weight:
                raise ValueError(
                    "Attempted to accumulate into grad weight bufferwithout providing grad weight"
                )
            grad_weight = torch.empty(
                weight_dims,
                dtype=dtype,
                device=device,
                memory_format=torch.contiguous_format,
            )

        # Perform dgrad GEMM
        if with_fp8_compute:
            kwargs = {"out": dx, "use_split_accumulator": True}
            if with_ub_all_gather_dy:
                kwargs["ub_algo"] = ub_algo_dy
                kwargs["ub"] = ub_comm_dy
            elif with_ub_all_gather_x:
                kwargs["ub_algo"] = ub_algo_x
                kwargs["ub"] = ub_comm_x
            elif with_ub_reduce_scatter_dx:
                kwargs["ub_algo"] = ub_algo_dx
                kwargs["ub"] = ub_comm_dx
                kwargs["extra_output_tensor"] = dx_local
            if with_fp8_grad_input:
                fp8_meta, fp8_meta_index = get_fp8_meta_from_fp8_tensor(dx)
                kwargs.update(
                    {
                        "out": dx._data,
                        "out_index": fp8_meta_index,
                        "fp8_meta_tensor": fp8_meta,
                        "D_dtype": dx._fp8_dtype,
                    }
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
            kwargs = {"grad": True, "layout": "NN", "out": dx}
            if with_ub_all_gather_dy:
                kwargs["ub_algo"] = ub_algo_dy
                kwargs["ub"] = ub_comm_dy
            elif with_ub_all_gather_x:
                kwargs["ub_algo"] = ub_algo_x
                kwargs["ub"] = ub_comm_x
            elif with_ub_reduce_scatter_dx:
                kwargs["ub_algo"] = ub_algo_dx
                kwargs["ub"] = ub_comm_dx
                kwargs["extra_output_tensor"] = dx_local
            gemm(w, dy, dx.dtype, get_workspace(), **kwargs)
        grad_input = reshape(dx_local, input_dims)

        # Perform wgrad GEMM
        if not weight_requires_grad:
            pass
        elif with_fp8_compute:
            kwargs = {
                "accumulate": accumulate_into_grad_weight,
                "out": grad_weight,
                "use_split_accumulator": True,
            }
            if with_ub_reduce_scatter_dx:
                kwargs["ub_algo"] = ub_algo_dx
                kwargs["ub"] = ub_comm_dx
            fp8_gemm(
                x.transpose_2d(),
                x._scale_inv,
                0,
                x._fp8_dtype,
                dy.transpose_2d(),
                dy._scale_inv,
                0,
                dy._fp8_dtype,
                grad_weight.dtype,
                get_workspace(),
                **kwargs,
            )
        else:
            kwargs = {
                "accumulate": accumulate_into_grad_weight,
                "layout": "NT",
                "grad": True,
                "use_bias": bias_requires_grad,
                "out": grad_weight,
            }
            if with_ub_reduce_scatter_dx:
                kwargs["ub_algo"] = ub_algo_dx
                kwargs["ub"] = ub_comm_dx
            grad_weight, db, _ = gemm(
                x,
                dy,
                grad_weight.dtype,
                get_workspace(),
                **kwargs,
            )

        # Compute grad bias if needed
        if db_async is not None:
            db_async.wait()
        if bias_requires_grad:
            if db is None:
                db = dy.sum(dim=0)
            extra_outputs["grad_bias"] = db

        return grad_input, grad_weight, extra_outputs

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

        # Hackily workaround Userbuffers bug with non-FP8 dgrad
        # reduce-scatter overlap
        weight_requires_grad = linear_op_ctx.weight_requires_grad
        if not linear_op_ctx.with_fp8_compute and not weight_requires_grad:
            warnings.warn(
                "There is a correctness bug when using Userbuffers "
                "to overlap a dgrad reduce-scatter with a non-FP8 dgrad GEMM. "
                "Hackily working around by overlapping dgrad reduce-scatter "
                "with wgrad GEMM, even though wgrad isn't needed. "
                "Please contact Transformer Engine team "
                "if you encounter this use-case."
            )
            weight_requires_grad = True

        # Linear backward pass
        retval = UserbuffersBackwardLinear._functional_backward(
            grad_output=grad_output,
            input=x_local,
            weight=linear_op.weight,
            input_dims=linear_op_ctx.input_dims,
            weight_dims=linear_op.weight.size(),
            weight_requires_grad=weight_requires_grad,
            bias_requires_grad=(bias_op is not None),
            device=linear_op.device,
            dtype=linear_op_ctx.dtype,
            grad_weight=grad_weight,
            accumulate_into_grad_weight=accumulate_into_main_grad,
            tensor_parallel_mode=self.tensor_parallel_mode,
            tensor_parallel_group=self.tensor_parallel_group,
            sequence_parallel=self.sequence_parallel,
            with_fp8_compute=linear_op_ctx.with_fp8_compute,
            weight_fp8_meta=linear_op_ctx.weight_fp8_meta,
            grad_output_fp8_meta=linear_op_ctx.grad_output_fp8_meta,
            grad_input_fp8_meta=linear_op_ctx.grad_input_fp8_meta,
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
