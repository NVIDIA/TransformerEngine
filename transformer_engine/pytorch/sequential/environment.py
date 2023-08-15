import torch
from dataclasses import dataclass
from contextlib import contextmanager

_lowp: torch.dtype = torch.float32
_world_size: int = 1


@dataclass
class Environment:
    lowp: torch.dtype
    world_size: int

    @staticmethod
    def current():
        return Environment(_lowp, _world_size)


@contextmanager
def environment(lowp: torch.dtype = torch.float32, world_size: int = 1):
    global _lowp, _world_size

    prev_lowp = _lowp
    prev_world_size = _world_size

    _lowp = lowp
    _world_size = world_size

    try:
        yield
    finally:
        _lowp = prev_lowp
        _world_size = prev_world_size
