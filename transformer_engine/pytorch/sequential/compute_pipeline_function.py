import torch
from torch import autograd
from torch.autograd.function import FunctionCtx
from . import nvte
from .ops import Context, Op
from .compute_pipeline import ComputePipeline


class ComputePipelineFunction(autograd.Function):
    @staticmethod
    def forward(  # type: ignore[arg-type]
        ctx: FunctionCtx,
        exposed_x: torch.Tensor,
        *args: torch.Tensor | Op | list[nvte.Tensor]
    ):
        """
        exposed_x is used only to let autograd construct the computation graph
        real input and output is in list, as nvte.Tensor is immutable
        exposed_tensors are exposed for the optimizer to later apply gradients
        """
        exposed_tensors, op, nvte_x_container = args[:-2], args[-2], args[-1]
        del exposed_tensors

        assert isinstance(op, Op)
        assert isinstance(nvte_x_container, list)
        assert len(nvte_x_container) == 1
        nvte_x = nvte_x_container[0]
        assert isinstance(nvte_x, nvte.Tensor)

        nvte.set_current_pass("forward")
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
        nvte_x_container[0] = y

        # Expose result for Pytorch
        exposed_y = torch.Tensor()
        exposed_y.shape = torch.Size(y.shape)  # needed for autograd to not complain
        exposed_y.grad_fn = exposed_x.grad_fn  # needed to preserve computation graph

        return exposed_y

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):  # type: ignore[arg-type]
        # The context needs to think that the tensors were read
        _ = ctx.saved_tensors  # type: ignore

        # Get real context
        saved: Context = getattr(ctx, "nvte_ctx")
        op: Op = getattr(ctx, "nvte_op")

        nvte.set_current_pass("backward")
        data_grad, param_grads = op.backward(saved, nvte.make_nvte_tensor(grad_output))

        # Check that gradients are not fp8 and can be processed by the optimizer
        # TODO: change this when fp8 optimizer comes along
        assert not nvte.is_fp8(data_grad)
        assert all(not nvte.is_fp8(g) for g in param_grads)

        torch_grads = [data_grad.data] + [g.data for g in param_grads]

        return (*torch_grads, None, None)


def apply(x: torch.Tensor, pipeline: ComputePipeline, training: bool) -> torch.Tensor:
    nvte_x = nvte.make_nvte_tensor(x)
    if not training:
        nvte.set_current_pass("inference")
        y = pipeline.run_inference(nvte_x)
        assert not nvte.is_fp8(y)
        return y.data
    else:
        for contained_op in pipeline.functions:
            nvte_tensors = contained_op.args()
            exposed_tensors = list[torch.Tensor]()
            for nvte_tensor in nvte_tensors:
                assert not nvte.is_fp8(
                    nvte_tensor
                )  # TODO: change when fp8 optimizer comes along
                exposed_tensors.append(nvte_tensor.data)
            nvte_x_container = [nvte_x]
            x = ComputePipelineFunction.apply(  # type: ignore
                x, *exposed_tensors, contained_op, nvte_x_container
            )
            nvte_x = nvte_x_container[0]
        return x
