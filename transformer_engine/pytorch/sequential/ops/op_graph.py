from .base import Op, OpInputPlaceholder
from .add import OpAdd
from .mul import OpMul
from .transpose import OpTranspose
from .layernorm import OpFLayerNormCore, OpDFLayerNormCore
from .gelu import OpFGelu, OpDFGelu
from .scale import OpScale


class ParamDescriptor:
    features: int
    name: str

    def __init__(self, features: int, name: str):
        self.features = features
        self.name = name


class OpParam(Op):
    def __init__(self, param: ParamDescriptor):
        super().__init__()
        self.param = param


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

    def scale_(self, a: Op, b: Op) -> Op:
        """Adds a scale (element-wise multiply) node to the graph. Returns the node with the result."""
        self.nodes.append(OpScale(a, b))
        return self.nodes[-1]

    def f_layernorm_(
        self, a: Op, gamma: Op, beta: Op, eps: float, zero_centered_gamma: bool
    ) -> Op:
        """Adds a layernorm node to the graph. Returns the node with the result."""
        self.nodes.append(OpFLayerNormCore(a, eps))
        core = self.nodes[-1]
        node = self.scale_(core, gamma)
        if zero_centered_gamma:
            node = self.add_(node, core)
        node = self.add_(node, beta)
        return node

    def df_layernorm_core_(self, a: Op, eps: float) -> Op:
        """Adds a layernorm' (derivative) node to the graph. Returns the node with the result."""
        self.nodes.append(OpDFLayerNormCore(a, eps))
        return self.nodes[-1]

    def f_gelu_(self, a: Op) -> Op:
        """Adds a gelu node to the graph. Returns the node with the result."""
        self.nodes.append(OpFGelu(a))
        return self.nodes[-1]

    def df_gelu_(self, a: Op) -> Op:
        """Adds a gelu' (derivative) node to the graph. Returns the node with the result."""
        self.nodes.append(OpDFGelu(a))
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
