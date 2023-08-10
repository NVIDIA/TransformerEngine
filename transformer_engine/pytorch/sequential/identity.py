from typing import Generic, TypeVar
import inspect


def identity():
    return hash(tuple((info.filename, info.positions) for info in inspect.stack()))


T = TypeVar("T")


class Persistent(Generic[T]):
    identity: int
    value: T

    def __init__(self, value: T):
        self.identity = identity()
        self.value = value


for i in range(10):
    if i % 2 == 0:
        print(Persistent[int](i).identity)
    else:
        print(Persistent[int](i).identity)
