from .base import Op


class OpScale(Op):
    def __init__(self, a: Op, b: Op):
        super().__init__()
        self.a = a
        self.b = b

    def backward_a(self, graph: "OpGraph", grad: Op):
        return graph.scale_(self.b, grad)

    def backward_b(self, graph: "OpGraph", grad: Op):
        return graph.scale_(self.a, grad)
