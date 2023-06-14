from .base import Op


class OpFGelu(Op):
    def __init__(self, a: Op):
        super().__init__()
        self.a = a

    def backward_a(self, graph: "OpGraph", grad: Op):
        df = graph.df_gelu_(self.a)
        return graph.scale_(df, grad)


class OpDFGelu(Op):
    def __init__(self, a: Op):
        super().__init__()
        self.a = a
