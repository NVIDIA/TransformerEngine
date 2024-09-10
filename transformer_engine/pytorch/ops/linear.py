# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Fusible operation for linear layer."""

from __future__ import annotations
from collections.abc import Callable
from typing import Any, Optional

import torch

from transformer_engine.pytorch.ops.basic import (
    AllReduce,
    BasicLinear,
    Bias,
    ReduceScatter,
)
from transformer_engine.pytorch.distributed import CudaRNGStatesTracker
from transformer_engine.pytorch.ops.op import FusedOperation


class Linear(FusedOperation):
    """Apply linear transformation: :math:`y = x A^T + b`

    This is a drop-in replacement for `torch.nn.Linear`.

    Parameters
    ----------
    in_features: int
        Inner dimension of input tensor
    out_features: int
        Inner dimension of output tensor
    bias: bool, default = `True`
        Apply additive bias
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
        parallelism, i.e. distributing input or output tensors along
        outer dimension (sequence or batch dim) when not distributing
        along inner dimension (embedding dim)
    rng_state_tracker_function: callable, optional
        Function that returns CudaRNGStatesTracker, which is used for
        model-parallel weight initialization
    rng_state_tracker_name: str, optional
        Key passed to `CudaRNGStatesTracker` to get a specific RNG
        tracker
    accumulate_into_main_grad: bool, default = `False`
        Whether to directly accumulate weight gradients into the
        weight's `main_grad` attribute instead of relying on PyTorch
        autograd. The weight's `main_grad` must be set externally and
        there is no guarantee that `grad` will be set or be
        meaningful.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
        tensor_parallel_mode: Optional[str] = None,
        tensor_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        sequence_parallel: bool = False,
        rng_state_tracker_function: Optional[Callable[[], CudaRNGStatesTracker]] = None,
        rng_state_tracker_name: Optional[str] = None,
        accumulate_into_main_grad: bool = False,
    ) -> None:

        # Tensor parallel configuration
        (
            tensor_parallel_mode,
            tensor_parallel_group,
            tensor_parallel_size,
            sequence_parallel,
            local_in_features,
            local_out_features,
        ) = BasicLinear._canonicalize_tensor_parallelism(
            mode=tensor_parallel_mode,
            process_group=tensor_parallel_group,
            sequence_parallel=sequence_parallel,
            in_features=in_features,
            out_features=out_features,
        )

        # List of operations that comprise this fused operation
        ops = []
        op_idxs = dict(
            linear=None,
            bias=None,
            all_gather=None,
            reduce_scatter=None,
            all_reduce=None,
        )

        def add_op(op: BasicOperation, name: str) -> None:
            """Add basic op to this fused operation"""
            op_idxs[name] = len(ops)
            ops.append(op)

        # Construct basic ops
        linear_kwargs = dict(
            in_features=in_features,
            out_features=out_features,
            device=device,
            dtype=dtype,
            tensor_parallel_mode=tensor_parallel_mode,
            tensor_parallel_group=tensor_parallel_group,
            sequence_parallel=sequence_parallel,
            rng_state_tracker_function=rng_state_tracker_function,
            rng_state_tracker_name=rng_state_tracker_name,
            accumulate_into_main_grad=accumulate_into_main_grad,
        )
        bias_kwargs = dict(
            size=out_features,
            device=device,
            dtype=dtype,
            tensor_parallel=(tensor_parallel_mode is not None),
            tensor_parallel_group=tensor_parallel_group,
        )
        if tensor_parallel_mode == "row":
            # Row TP: GEMM + bias + reduction
            linear_kwargs["in_features"] = local_in_features
            linear_kwargs["out_features"] = local_out_features
            linear_kwargs["tensor_parallel_mode"] = None
            linear_kwargs["tensor_parallel_group"] = None
            linear_kwargs["sequence_parallel"] = False
            bias_kwargs["size"] *= tensor_parallel_size
            add_op(BasicLinear(**linear_kwargs), "linear")
            if bias:
                add_op(Bias(**bias_kwargs), "bias")
            if sequence_parallel:
                add_op(ReduceScatter(tensor_parallel_group), "reduce_scatter")
            else:
                add_op(AllReduce(tensor_parallel_group), "all_reduce")
        else:
            # Column TP or no TP: (gather + GEMM) + bias
            add_op(BasicLinear(**linear_kwargs), "linear")
            if bias:
                add_op(Bias(**bias_kwargs), "bias")

        # Initialize base class
        super().__init__(ops)

        # Index of each basic operation
        self._op_idxs = op_idxs

        # Register parameters
        # Note: Work around checks in module base class by manually
        # filling self._parameters
        self._parameters["weight"] = None
        self._parameters["bias"] = None
        self.register_parameter("weight", self.weight)
        self.register_parameter("bias", self.bias if bias else None)

    @property
    def weight(self) -> torch.nn.Parameter:
        linear_op_idx = self._op_idxs["linear"]
        return self.basic_ops[linear_op_idx].weight

    def _set_weight(self, value: torch.nn.Parameter) -> None:
        linear_op_idx = self._op_idxs["linear"]
        if not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(weight)
        self.basic_ops[linear_op_idx].weight = value

    @property
    def bias(self) -> torch.nn.Parameter:
        bias_op_idx = self._op_idxs["bias"]
        if bias_op_idx is None:
            return None
        return self.basic_ops[bias_op_idx].bias

    def _set_bias(self, value: torch.nn.Parameter) -> None:

        # Handle edge cases
        bias_op_idx = self._op_idxs["bias"]
        if bias_op_idx is None:
            if value is not None:
                raise ValueError(
                    "Attempted to set bias tensor in linear operation, but bias is disabled"
                )
            return

        # Set bias tensor
        if value is not None and not isinstance(value, torch.nn.Parameter):
            value = torch.nn.Parameter(value)
        self.basic_ops[bias_op_idx].bias = value

    def __setattr__(self, name: str, value: Any) -> None:

        # Manually set weight and bias tensor
        # Note: It would be nicer to implement "weight" and "bias" as
        # class properties, but the base module class implementation
        # of __setattr__ takes precedence over property setters.
        if name == "weight":
            self._set_weight(value)
        elif name == "bias":
            self._set_bias(value)

        # Base class implementation
        super().__setattr__(name, value)
