from __future__ import annotations
from typing import Callable
import torch

ParameterInitMethod = Callable[[torch.Tensor], torch.Tensor]
