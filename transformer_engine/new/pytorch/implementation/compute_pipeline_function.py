import torch.autograd

from ...common import ComputePipeline


def apply(x: torch.Tensor, pipeline: ComputePipeline, training: bool) -> torch.Tensor:
    ...  # TODO
