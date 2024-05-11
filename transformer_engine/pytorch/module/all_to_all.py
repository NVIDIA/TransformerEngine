# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""FP8 AlltoAll API"""
from typing import Union, Optional, Tuple, List, Dict, Any

import torch

from .. import cpp_extensions as tex

from .base import TransformerEngineBaseModule

from ..cpp_extensions import cast_to_fp8, cast_from_fp8
from ..constants import dist_group_type, TE_DType
from ..fp8 import get_fp8_te_dtype, FP8GlobalStateManager
from ..graph import is_graph_capturing
from ..jit import no_torch_dynamo
from ..utils import requires_grad

__all__ = ["FP8AllToAll"]


class _FP8AllToAll(torch.autograd.Function):
    """Functional FP8AllToAll
    Calls custom cuda extensions.
    """

    @staticmethod
    def forward(
        ctx,
        group: dist_group_type,
        inp: torch.Tensor,
        output_split_sizes: Union[List, None],
        input_split_sizes: Union[List, None],
        is_grad_enabled: bool,
        activation_dtype: torch.dtype,
        fp8_meta: Dict[str, Any],
    ) -> torch.Tensor:

        world_size = torch.distributed.get_world_size(group=group)
        # Bypass the function if we are using only 1 GPU.
        if world_size == 1:
            return inp

        inp = inp.contiguous()
        fp8_dtype_forward = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        inp_fp8 = cast_to_fp8(
            inp,
            fp8_meta["scaling_fwd"],
            tex.FP8FwdTensors.GEMM1_INPUT,
            fp8_dtype_forward,
        )
  
        if output_split_sizes is None:
            # Equal split (all2all)
            output_fp8 = torch.empty_like(inp_fp8)
        else:
            # Unequal split (all2all-v)
            output_fp8 = torch.empty(
                size=[sum(output_split_sizes)] + list(inp_fp8.size()[1:]),
                dtype=inp_fp8.dtype,
                device=inp_fp8.device,
            )
        torch.distributed.all_to_all_single(
            output_fp8,
            inp_fp8,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        if output_fp8.nelement() == 0:
            output = torch.empty_like(output_fp8, dtype=activation_dtype)
        else:
            output = cast_from_fp8(
                output_fp8,
                fp8_meta["scaling_fwd"],
                tex.FP8FwdTensors.GEMM1_INPUT,
                fp8_dtype_forward,
                TE_DType[activation_dtype],
            )

        if is_grad_enabled:
            ctx.group = group
            ctx.output_split_sizes = output_split_sizes
            ctx.input_split_sizes = input_split_sizes
            ctx.activation_dtype = activation_dtype
            ctx.fp8_meta = fp8_meta
            ctx.reduce_and_update_bwd_fp8_tensors = False
            if requires_grad(inp,):
                ctx.reduce_and_update_bwd_fp8_tensors = (
                    ctx.reduce_and_update_bwd_fp8_tensors
                    or FP8GlobalStateManager.is_first_fp8_module()
                )

        return output

    @staticmethod
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Union[torch.Tensor, None], ...]:

        with torch.cuda.nvtx.range("_FP8AllToAll"):
            world_size = torch.distributed.get_world_size(group=ctx.group)
            # Bypass the function if we are using only 1 GPU.
            if world_size == 1:
                return grad_output

            grad_output = grad_output.contiguous()
            fp8_dtype_backward = get_fp8_te_dtype(
                ctx.fp8_meta["recipe"], fprop_tensor=False
            )
            grad_output_fp8 = cast_to_fp8(
                grad_output,
                ctx.fp8_meta["scaling_bwd"],
                tex.FP8BwdTensors.GRAD_OUTPUT1,
                fp8_dtype_backward,
            )

            if ctx.input_split_sizes is None:
                # Equal split (all2all)
                dgrad_fp8 = torch.empty_like(grad_output_fp8)
            else:
                # Unequal split (all2all-v)
                dgrad_fp8 = torch.empty(
                    size=[sum(ctx.input_split_sizes)] + list(grad_output_fp8.size()[1:]),
                    dtype=grad_output_fp8.dtype,
                    device=grad_output_fp8.device,
                )
            torch.distributed.all_to_all_single(
                dgrad_fp8,
                grad_output_fp8,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )

            if dgrad_fp8.nelement() == 0:
                dgrad = torch.empty_like(dgrad_fp8, dtype=ctx.activation_dtype)
            else:
                dgrad = cast_from_fp8(
                    dgrad_fp8,
                    ctx.fp8_meta["scaling_bwd"],
                    tex.FP8BwdTensors.GRAD_OUTPUT1,
                    fp8_dtype_backward,
                    TE_DType[ctx.activation_dtype],
                )

        if ctx.reduce_and_update_bwd_fp8_tensors and not is_graph_capturing():
            FP8GlobalStateManager.reduce_and_update_fp8_tensors(forward=False)

        return (
            None,
            dgrad,
            None,
            None,
            None,
            None,
            None,
        )


class FP8AllToAll(TransformerEngineBaseModule):
    r"""
    Cast_to_fp8 -> FP8 AllToAll -> Cast_from_fp8.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_fp8_weights_scratchpad(
        self,
        is_first_microbatch: Union[bool, None],
    ) -> List[torch.Tensor]:
        return [None, None]

    @no_torch_dynamo()
    def forward(
        self,
        group: dist_group_type,
        inp: torch.Tensor,
        output_split_sizes: Union[List, None],
        input_split_sizes: Union[List, None],
    ) -> torch.Tensor:
        """
        FP8AllToAll FWD.

        Parameters
        ----------
        inp : torch.Tensor
             Input tensor.
        """
        with self.prepare_forward(inp, None) as inp:
            assert self.fp8, "Need to run inside fp8_autocast region."
            self.set_activation_dtype(inp)

            if torch.is_grad_enabled():
                fwd_fn = _FP8AllToAll.apply
                args = []
            else:
                fwd_fn = _FP8AllToAll.forward
                args = [None]

            args += (
                group,
                inp,
                output_split_sizes,
                input_split_sizes,
                torch.is_grad_enabled(),
                self.activation_dtype,
                self.fp8_meta,
            )

            return fwd_fn(*args)
