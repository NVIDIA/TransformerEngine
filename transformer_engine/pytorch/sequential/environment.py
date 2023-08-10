from dataclasses import dataclass


@dataclass
class Environment:
    fp8_enabled: bool
    world_size: int
