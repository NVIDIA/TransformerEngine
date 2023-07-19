from __future__ import annotations
from attr import dataclass
from ...fp8 import is_fp8_enabled


@dataclass
class CompileEnv:
    fp8: bool

    @staticmethod
    def current() -> CompileEnv:
        return CompileEnv(is_fp8_enabled())


__all__ = ["CompileEnv"]
