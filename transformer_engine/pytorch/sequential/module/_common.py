from typing import Callable
import torch

ParameterInitMethod = Callable[[torch.Tensor], torch.Tensor]
