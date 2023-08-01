from typing import Generator
import torch
from torch.autograd.function import FunctionCtx

from ..common_back.generic_tensor import GenericTensor

from .tensor import PytorchNativeTensor
from ..common_back.enums import DType
from ..common_back.compute_pipeline import ComputePipeline


class PipelineFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: FunctionCtx | None, input: torch.Tensor, pipeline: ComputePipeline):  # type: ignore
        tensor = PytorchNativeTensor(DType.from_torch_dtype(input.dtype), input)
        if pipeline.environment.training:
            assert ctx is not None
            gen = pipeline.training(tensor)
            output = next(gen)
            ctx.gen = gen  # type: ignore
            assert isinstance(output, PytorchNativeTensor)
            return output.tensor
        else:
            output = pipeline.inference(tensor)
            assert isinstance(output, PytorchNativeTensor)
            return output.tensor

    @staticmethod
    def backward(ctx: FunctionCtx, *grad_outputs: torch.Tensor):
        assert len(grad_outputs) == 1
        grad_output = grad_outputs[0]
        tensor = PytorchNativeTensor(
            DType.from_torch_dtype(grad_output.dtype), grad_output
        )
        gen: Generator[GenericTensor, GenericTensor, None] = ctx.gen  # type: ignore
        assert isinstance(gen, Generator)
        grad = gen.send(tensor)
        assert isinstance(grad, PytorchNativeTensor)
        return grad.tensor, None
