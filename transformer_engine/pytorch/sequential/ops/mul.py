from .base import Op


class OpMul(Op):
    def __init__(self, a: Op, b: Op):
        super().__init__()
        self.a = a
        self.b = b

    def backward_a(self, graph: "OpGraph", grad: Op):
        bT = graph.t_(self.b)
        return graph.mul_(bT, grad)

    def backward_b(self, graph: "OpGraph", grad: Op):
        aT = graph.t_(self.a)
        return graph.mul_(aT, grad)
