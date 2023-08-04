import torch.autograd

from ...common.enums import DType
from ...pytorch_back.tensor import PytorchNativeTensor, PytorchTensor
from ...common import ComputePipeline, GenericTensor, Op


class ComputePipelineFunction(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore
        ctx: torch.autograd.function.FunctionCtx,
        x: torch.Tensor,
        *exposed_tensors: torch.Tensor,
        op: Op,
        forward_args: dict[str, GenericTensor],
        **_
    ):
        raw_activation, raw_ctx = op.forward(
            PytorchNativeTensor(DType.from_torch_dtype(x.dtype), x), **forward_args
        )
        assert isinstance(raw_activation, PytorchTensor)
        activation = raw_activation.tensor
        exposed_ctx = (
            tensor.tensor
            for tensor in raw_ctx.values()
            if isinstance(tensor, PytorchTensor)
        )
        ctx.save_for_backward(activation, *exposed_ctx)
        setattr(ctx, "op", op)
        setattr(ctx, "raw_ctx", raw_ctx)
        setattr(ctx, "forward_args", forward_args)
        return activation

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, *grads: torch.Tensor):
        assert len(grads) == 1
        grad = grads[0]
        _ = ctx.saved_tensors  # type: ignore
        op: Op = getattr(ctx, "op")  # type: ignore
        raw_ctx: dict[str, GenericTensor] = getattr(ctx, "raw_ctx")  # type: ignore
        forward_args: dict[str, GenericTensor] = getattr(ctx, "forward_args")  # type: ignore
        raw_grad, raw_param_grads = op.backward(
            PytorchNativeTensor(DType.from_torch_dtype(grad.dtype), grad), **raw_ctx
        )
        assert isinstance(raw_grad, PytorchTensor)
        grad = raw_grad.tensor
        ordering = {name: i for i, name in enumerate(forward_args.keys())}
        ordered_raw_param_grads = {
            ordering[name]: grad for name, grad in raw_param_grads.items()
        }
        raw_all_grads = [
            ordered_raw_param_grads[i] if i in ordered_raw_param_grads else None
            for i in range(len(forward_args))
        ]
        all_grads = [
            grad.tensor if isinstance(grad, PytorchTensor) else None
            for grad in raw_all_grads
        ]
        return (grad, *all_grads, None, None)


def apply(x: torch.Tensor, pipeline: ComputePipeline, training: bool) -> torch.Tensor:
    if not training:
        result = pipeline.inference_optimized(
            PytorchNativeTensor(DType.from_torch_dtype(x.dtype), x)
        )
        assert isinstance(result, PytorchNativeTensor)
        return result.tensor
    else:
        assert pipeline._compiled is not None
        for op in pipeline._compiled:
            forward_args = pipeline._parameters[op] | pipeline._buffers[op]
            exposed_tensors = tuple(forward_args.values())
            x = ComputePipelineFunction.apply(  # type: ignore
                x, *exposed_tensors, op=op, forward_args=forward_args
            )
        return x
