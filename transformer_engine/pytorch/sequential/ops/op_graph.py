from enum import Enum


class ParamDescriptor:
    features: int
    name: str

    def __init__(self, features: int, name: str):
        self.features = features
        self.name = name


def _gen_node_id() -> int:
    if "id" not in _gen_node_id.__dict__:
        _gen_node_id.id = 0
    _gen_node_id.id += 1
    return _gen_node_id.id


class Op:
    grad: "Op | None" = None

    def __init__(self):
        self.id = _gen_node_id()


class OpInputPlaceholder(Op):
    def replace(self, op: Op):
        self.__class__ = op.__class__
        self.__dict__ = op.__dict__


class OpParam(Op):
    def __init__(self, param: ParamDescriptor):
        super().__init__()
        self.param = param


class OpAdd(Op):
    def __init__(self, a: Op, b: Op):
        super().__init__()
        self.a = a
        self.b = b

    def backward_a(self, graph: "OpGraph", grad: Op):
        return grad

    def backward_b(self, graph: "OpGraph", grad: Op):
        return grad


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
        return graph.mul_(grad, aT)


class OpTranspose(Op):
    def __init__(self, a: Op):
        super().__init__()
        self.a = a

    def backward_a(self, graph: "OpGraph", grad: Op):
        return graph.t_(grad)


class OpFLayerNorm(Op):
    def __init__(
        self, a: Op, gamma: Op, beta: Op, eps: float, zero_centered_gamma: bool
    ):
        super().__init__()
        self.a = a
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma


class OpFGelu(Op):
    def __init__(self, a: Op):
        super().__init__()
        self.a = a


class OpGraph:
    trainable_parameters: list[ParamDescriptor]
    nodes: list[Op]
    in_nodes: list[Op]
    out_nodes: list[Op]

    def __init__(self):
        self.trainable_parameters = []
        self.nodes = []
        self.in_nodes = []
        self.out_nodes = []

    def in_(self) -> Op:
        """Adds an input node with the given number of features to the graph. Returns the node, so it can be used in further computations."""
        self.in_nodes.append(OpInputPlaceholder())
        return self.in_nodes[-1]

    def param_(self, features: int, name: str) -> Op:
        """Adds a trainable parameter with the given number of features to the graph. Returns the parameter, so it can be used in further computations."""
        self.trainable_parameters.append(ParamDescriptor(features, name))
        self.nodes.append(OpParam(self.trainable_parameters[-1]))
        return self.nodes[-1]

    def out_(self, op: Op) -> Op:
        """Marks the given op as an output node of the graph. The op must be a node of the graph. Returns the op back."""
        assert op in self.nodes
        self.out_nodes.append(op)
        return op

    def mul_(self, a: Op, b: Op) -> Op:
        """Adds a multiplication node to the graph. Returns the node with the result."""
        self.nodes.append(OpMul(a, b))
        return self.nodes[-1]

    def add_(self, a: Op, b: Op) -> Op:
        """Adds an addition node to the graph. Returns the node with the result."""
        self.nodes.append(OpAdd(a, b))
        return self.nodes[-1]

    def f_layernorm_(
        self, a: Op, gamma: Op, beta: Op, eps: float, zero_centered_gamma: bool
    ) -> Op:
        """Adds a layernorm node to the graph. Returns the node with the result."""
        self.nodes.append(OpFLayerNorm(a, gamma, beta, eps, zero_centered_gamma))
        return self.nodes[-1]

    def f_gelu_(self, a: Op) -> Op:
        """Adds a gelu node to the graph. Returns the node with the result."""
        self.nodes.append(OpFGelu(a))
        return self.nodes[-1]

    def t_(self, a: Op) -> Op:
        """Adds a transpose node to the graph. Returns the node with the result."""
        self.nodes.append(OpTranspose(a))
        return self.nodes[-1]

    def create_backward_graph_(self, node: Op, grad: Op):
        """Creates the graph of the backward pass, assuming that grad is the gradient of the loss with respect to the node."""
        assert node in self.out_nodes
        assert grad in self.in_nodes

        node.grad = grad
        bfs = [node]
        while len(bfs) > 0:
            cur = bfs.pop()
            params = cur.__dict__.items()
            for name, param in params:
                if isinstance(param, Op) and name != "grad":
                    param_grad_func = getattr(cur, f"backward_{name}")
                    param.grad = param_grad_func(self, cur.grad)
                    bfs.append(param)

    @staticmethod
    def combine_graphs(a: "OpGraph", b: "OpGraph") -> "OpGraph":
        assert len(a.out_nodes) == len(b.in_nodes)
        graph = OpGraph()
        graph.trainable_parameters = a.trainable_parameters + b.trainable_parameters
        graph.nodes = a.nodes + b.nodes
        graph.in_nodes = a.in_nodes
        graph.out_nodes = b.out_nodes
        for out_node, in_node in zip(a.out_nodes, b.in_nodes):
            in_node.replace(out_node)
        return graph
