# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    rng_state_tracker_function: callable
        Function that returns CudaRNGStatesTracker, which is used for
        model-parallel weight initialization
    accumulate_into_main_grad: bool, default = `False`
        Whether to directly accumulate weight gradients into the
        weight's `main_grad` attribute instead of relying on PyTorch
        autograd. The weight's `main_grad` must be set externally and
        there is no guarantee that `grad` will be set or be
        meaningful. This is primarily intented to integrate with
        Megatron-LM.

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

        # Construct basic ops
        ops = []
        linear_idx = None
        bias_idx = None
        linear_kwargs = {
            "in_features": in_features,
            "out_features": out_features,
            "device": device,
            "dtype": dtype,
            "tensor_parallel_mode": tensor_parallel_mode,
            "tensor_parallel_group": tensor_parallel_group,
            "sequence_parallel": sequence_parallel,
            "rng_state_tracker_function": rng_state_tracker_function,
            "accumulate_into_main_grad": accumulate_into_main_grad,
        }
        bias_kwargs = {
            "size": out_features,
            "device": device,
            "dtype": dtype,
            "tensor_parallel": (tensor_parallel_mode is not None),
            "tensor_parallel_group": tensor_parallel_group,
        }
        if tensor_parallel_mode == "row":
            # Row TP: GEMM + bias + reduction
            linear_idx = len(ops)
            linear_kwargs["in_features"] = local_in_features
            linear_kwargs["out_features"] = local_out_features
            linear_kwargs["tensor_parallel_mode"] = None
            linear_kwargs["tensor_parallel_group"] = None
            linear_kwargs["sequence_parallel"] = False
            ops.append(BasicLinear(**linear_kwargs))
            if bias:
                bias_idx = len(ops)
                bias_kwargs["size"] *= tensor_parallel_size
                ops.append(Bias(**bias_kwargs))
            if sequence_parallel:
                ops.append(ReduceScatter(tensor_parallel_group))
            else:
                ops.append(AllReduce(tensor_parallel_group))
        else:
            # Column TP or no TP: (gather + GEMM) + bias
            linear_idx = len(ops)
            ops.append(BasicLinear(**linear_kwargs))
            if bias:
                bias_idx = len(ops)
                ops.append(Bias(**bias_kwargs))

        # Initialize base class
        super().__init__(ops)

        # Register parameters
        self._linear_idx: Optional[int] = linear_idx
        self._bias_idx: Optional[int] = bias_idx
        self.register_parameter("weight", self.basic_ops[self._linear_idx].weight)
        bias = None
        if self._bias_idx is not None:
            bias = self.basic_ops[self._bias_idx].bias
        self.register_parameter("bias", bias)

    def register_parameter(self, name: str, param: Optional[torch.nn.Parameter]) -> None:
        """Add a parameter to the module

        Also updates the basic operation that owns the parameter.

        """
        if name == "bias" and self._bias_idx is None and param is not None:
            raise ValueError(
                "Attempted to set bias parameter in Linear operation "
                "that does not have bias enabled"
            )
        super().register_parameter(name, param)
        if name == "weight":
            self.basic_ops[self._linear_idx].weight = param
        elif name == "bias" and self._bias_idx is not None:
            self.basic_ops[self._bias_idx].bias = param

    def state_dict(self, *, prefix: str = "", **kwargs) -> dict[str, Any]:
        """Save state"""
        state_dict = super().state_dict(prefix=prefix, **kwargs)

        # Remove basic op params from state dict
        # Note: Logically, basic ops own params and fused ops are
        # considered as stateless. However, we register weight and
        # bias params in the linear op for convenience. We remove
        # these redudant params from the checkpoint for backward
        # compatibility.
        if f"{prefix}weight" in state_dict:
            del state_dict[f"{prefix}weight"]
        if f"{prefix}bias" in state_dict:
            del state_dict[f"{prefix}bias"]

        return state_dict

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        *args,
        **kwargs,
    ) -> None:

        # Add basic op params to state dict
        # Note: Logically, basic ops own params and fused ops are
        # considered as stateless. However, we register weight and
        # bias params in the linear op for convenience. We remove
        # these redudant params from the checkpoint for backward
        # compatibility.
        if f"{prefix}weight" not in state_dict:
            state_dict[f"{prefix}weight"] = state_dict[
                f"{prefix}basic_ops.{self._linear_idx}.weight"
            ]
        if f"{prefix}bias" not in state_dict:
            if self._bias_idx is None:
                state_dict[f"{prefix}bias"] = None
            else:
                state_dict[f"{prefix}bias"] = state_dict[f"{prefix}basic_ops.{self._bias_idx}.bias"]

        # Load state dict
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
