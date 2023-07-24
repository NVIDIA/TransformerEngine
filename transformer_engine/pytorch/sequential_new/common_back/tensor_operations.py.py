from typing import Any, NewType, Protocol

TensorHandle = NewType("TensorHandle", int)


class OpManListener(Protocol):
    def callback(self, func_name: str, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError()


class OpMan:
    listener: OpManListener

    def __init__(self, listener: OpManListener):
        self.listener = listener

    def __getattribute__(self, __name: str):
        def func(*args: Any, **kwargs: Any):
            self.listener.callback(__name, *args, **kwargs)

        return func
