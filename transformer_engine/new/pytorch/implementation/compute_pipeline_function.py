import torch.autograd

from ...common_back import ComputePipeline


def apply(x: torch.Tensor, pipeline: ComputePipeline) -> torch.Tensor:
    ...  # TODO
