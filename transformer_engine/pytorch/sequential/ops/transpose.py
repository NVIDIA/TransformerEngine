from .base import Op


class OpTranspose(Op):
    def __init__(self, a: Op):
        super().__init__()
        self.a = a

    def backward_a(self, graph: "OpGraph", grad: Op):
        return graph.t_(grad)
