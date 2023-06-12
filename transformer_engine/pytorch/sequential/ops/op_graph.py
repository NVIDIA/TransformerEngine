from enum import Enum


class ParamDescriptor:
    features: int

    def __init__(self, features: int):
        self.features = features


class Op:
    ...


class OpInputPlaceholder(Op):
    def __init__(self):
        pass

    def replace(self, op: Op):
        self.__class__ = op.__class__
        self.__dict__ = op.__dict__


class OpParam(Op):
    def __init__(self, param: ParamDescriptor):
        self.param = param


class OpAdd(Op):
    def __init__(self, a: Op, b: Op):
        self.a = a
        self.b = b


class OpMul(Op):
    def __init__(self, a: Op, b: Op):
        self.a = a
        self.b = b


class OpFLayerNorm(Op):
    def __init__(
        self, a: Op, gamma: Op, beta: Op, eps: float, zero_centered_gamma: bool
    ):
        self.a = a
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma


class OpFGelu(Op):
    def __init__(self, a: Op):
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

    def param_(self, features: int) -> Op:
        """Adds a trainable parameter with the given number of features to the graph. Returns the parameter, so it can be used in further computations."""
        self.trainable_parameters.append(ParamDescriptor(features))
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
