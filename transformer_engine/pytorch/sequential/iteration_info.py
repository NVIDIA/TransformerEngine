from __future__ import annotations
from abc import ABC, abstractmethod


class IterationAware:
    __iter_info: IterationInfoProvider
    __cur_iter: int | None = None
    __index: int = 0
    __max_index: int = 0

    def __init__(self, iter_info: IterationInfoProvider):
        self.__iter_info = iter_info

    def iteration(self):
        return self.__iter_info.iteration()

    def is_new_iteration(self):
        return self.__iter_info.is_new_iteration(self)

    def index_within_iteration(self):
        return self.__iter_info.index_within_iteration(self)

    def max_index(self):
        assert self.iteration() != 1
        return self.__max_index


class IterationInfoProvider(ABC):
    @abstractmethod
    def iteration(self) -> int:
        ...

    def __is_new_iteration(self, asker: IterationAware, __update: bool):
        if asker.__cur_iter is None or asker.__cur_iter == self.iteration() - 1:
            if __update:
                asker.__cur_iter = self.iteration()
            return True
        elif asker.__cur_iter == self.iteration():
            return False
        else:
            raise AssertionError()

    def is_new_iteration(self, asker: IterationAware):
        return self.__is_new_iteration(asker, True)

    def index_within_iteration(self, asker: IterationAware):
        if self.__is_new_iteration(asker, False):
            asker.__index = 1
        else:
            asker.__index += 1
            if self.iteration() == 1:
                asker.__max_index = asker.__index
            else:
                assert asker.__index <= asker.__max_index
        return asker.__index - 1
