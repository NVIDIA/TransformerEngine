from .base import Op


class OpAdd(Op):
    def __init__(self, a: Op, b: Op):
        super().__init__()
        self.a = a
        self.b = b

    def backward_a(self, graph: "OpGraph", grad: Op):
        return grad

    def backward_b(self, graph: "OpGraph", grad: Op):
        return grad
