from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, TypeVar


T = TypeVar("T")


class Persistent(Generic[T], ABC):
    """
    Storage for data that is to be persisted between iterations.
    Examples include fp8 metatensors (during training)
    and KV cache (during inference).
    """

    # abstract
    @abstractmethod
    def _generate(self) -> T:
        ...

    # public
    def __call__(self):
        result = self._generate()
        if __debug__:
            if self._iteration() == 1:
                self.__values.append(result)
            else:
                assert self.__values[self.__index_within_iteration(False)] is result
        return result

    def next_iteration(self):
        self.__user_set_iteration += 1

    # protected
    def _iteration(self):
        assert self.__user_set_iteration > 0
        return self.__user_set_iteration

    def _is_new_iteration(self):
        return self.__is_new_iteration(True)

    def _index_within_iteration(self):
        return self.__index_within_iteration(True)

    def _max_index(self):
        assert self._iteration() != 1
        return self.__max_index

    # private
    __index: int = 0
    __max_index: int = 0
    __user_set_iteration: int = 0
    __derived_seen_iteration: int = 0
    if __debug__:
        __values = list[T]()

    def __is_new_iteration(self, update: bool):
        if self.__derived_seen_iteration == self._iteration() - 1:
            if update:
                self.__derived_seen_iteration = self._iteration()
            return True
        elif self.__derived_seen_iteration == self._iteration():
            return False
        elif self.__derived_seen_iteration > self._iteration():
            raise AssertionError("Iteration cannot decrease.")
        else:  # self.__cur_iter == self._iteration() - k, k > 1
            raise AssertionError("Cannot skip iterations.")

    def __index_within_iteration(self, update: bool):
        if update:
            if self.__is_new_iteration(False):
                self.__index = 1
            else:
                self.__index += 1
                if self._iteration() == 1:
                    self.__max_index = self.__index

        assert self.__index > 0
        assert self.__index <= self.__max_index

        return self.__index - 1
