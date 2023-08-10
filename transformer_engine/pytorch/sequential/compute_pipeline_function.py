import torch
from torch import autograd
from torch.autograd.function import FunctionCtx
from torch import nn
import transformer_engine_cuda as nvte

from .ops import Context, Op

from .nvte_utils import is_fp8, make_nvte_tensor

from .compute_pipeline import ComputePipeline


class ComputePipelineFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        exposed_x: torch.Tensor,
        *exposed_tensors: torch.Tensor,
        op: Op,
        nvte_x: nvte.Tensor
    ):
        """
        exposed_x is used only to let autograd construct the computation graph
        real input and output is nvte_x
        exposed_tensors are exposed for the optimizer to later apply gradients
        """
        del exposed_tensors

        y, to_save = op.forward(nvte_x)

        # Expose backward context for tracing
        bwd_ctx = list[torch.Tensor]()
        for _, tensor in to_save.items():
            bwd_ctx.append(tensor.data)
            if tensor.amax.numel():
                bwd_ctx.append(tensor.amax)
            if tensor.scale.numel():
                bwd_ctx.append(tensor.scale)
            if tensor.scale_inv.numel():
                bwd_ctx.append(tensor.scale_inv)
        ctx.save_for_backward(*bwd_ctx)

        # Save real context
        setattr(ctx, "nvte_ctx", to_save)
        setattr(ctx, "nvte_op", op)

        # Actually store the result
        nvte_x.data, nvte_x.amax, nvte_x.scale, nvte_x.scale_inv = (
            y.data,
            y.amax,
            y.scale,
            y.scale_inv,
        )

        # Preserve computation graph
        exposed_x.data = y.data

        return exposed_x

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):
        # The context needs to think that the tensors were read
        _ = ctx.saved_tensors()  # type: ignore

        # Get real context
        saved: Context = getattr(ctx, "nvte_ctx")
        op: Op = getattr(ctx, "nvte_op")

        data_grad, param_grads = op.backward(saved, make_nvte_tensor(grad_output))

        # Check that gradients are not fp8 and can be processed by the optimizer
        # TODO: change this when fp8 optimizer comes along
        assert not is_fp8(data_grad)
        assert all(g is None or not is_fp8(g) for g in param_grads)

        torch_grads = [data_grad.data] + [
            g.data if g is not None else None for g in param_grads
        ]

        return (*torch_grads, None, None)


def apply(x: torch.Tensor, pipeline: ComputePipeline, training: bool) -> torch.Tensor:
    nvte_x = make_nvte_tensor(x)
    if not training:
        y = pipeline.run_inference(nvte_x)
        assert not is_fp8(y)
        return y.data
    else:
        for contained_op in pipeline.functions:
            nvte_tensors = contained_op.args()
            exposed_tensors = list[torch.Tensor]()
            for nvte_tensor in nvte_tensors:
                assert not is_fp8(
                    nvte_tensor
                )  # TODO: change when fp8 optimizer comes along
                exposed_tensors.append(nvte_tensor.data)
            x = ComputePipelineFunction.apply(  # type: ignore
                x, *exposed_tensors, op=contained_op, nvte_x=nvte_x
            )
        return x
