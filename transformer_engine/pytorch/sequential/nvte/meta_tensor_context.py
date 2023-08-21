from __future__ import annotations
from typing import Literal
import torch
from ._pass import pass_


class MetaTensorContext:
    last_iter_fwd: int | None
    last_iter_bwd: int | None
    current_pass: Literal["forward", "backward"]
    current_iter: int
    is_first_iter: bool
    prev: MetaTensorContext | None
    metatensors: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    metatensors_fwd: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None
    metatensors_bwd: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None

    def __init__(self):
        self.last_iter_fwd = None
        self.last_iter_bwd = None
        self.metatensors_fwd = None
        self.metatensors_bwd = None

    def __call__(self, current_pass: Literal["forward", "backward"], current_iter: int):
        last_iter = (
            self.last_iter_fwd if current_pass == "forward" else self.last_iter_bwd
        )
        if last_iter is not None and current_iter != last_iter + 1:
            raise ValueError(
                "Detected skipped iteration. This would most likely invalidate the current metatensors. Recreate the context instead."
            )

        self.current_pass = current_pass
        self.current_iter = current_iter
        return self

    def __enter__(self):
        global _current
        self.prev = _current
        if self.current_pass == "forward":
            self.metatensors = self.metatensors_fwd or {}
            self.is_first_iter = self.last_iter_fwd is None
        else:
            self.metatensors = self.metatensors_bwd or {}
            self.is_first_iter = self.last_iter_bwd is None
        self.current_tensor = 0
        global pass_
        pass_ = self.current_pass
        _current = self

    def __exit__(self, exc_type: type, exc_value: object, exc_tb: object):
        global _current
        _current = self.prev
        if self.current_pass == "forward":
            self.last_iter_fwd = self.current_iter
            self.metatensors_fwd = self.metatensors
        else:
            self.last_iter_bwd = self.current_iter
            self.metatensors_bwd = self.metatensors
        del self.current_pass
        del self.current_iter
        del self.is_first_iter
        del self.prev
        del self.current_tensor
        del self.metatensors

    def next_tensor(self):
        self.current_tensor += 1

    def has_metatensors(self):
        assert self.current_pass is not None
        if self.is_first_iter:
            return False
        assert self.current_tensor in self.metatensors
        return True

    def set_metatensors(self, mts: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        assert self.is_first_iter
        assert self.current_tensor not in self.metatensors
        self.metatensors[self.current_tensor] = mts

    def get_metatensors(self):
        assert not self.is_first_iter
        assert self.current_tensor in self.metatensors
        amax_history, scale, scale_inv = self.metatensors[self.current_tensor]
        amax = amax_history[self.current_iter % amax_history.shape[0]]
        # TODO compute scale and scale_inv
        return amax, scale, scale_inv


_current: MetaTensorContext | None = None


def current():
    assert _current is not None
    return _current
