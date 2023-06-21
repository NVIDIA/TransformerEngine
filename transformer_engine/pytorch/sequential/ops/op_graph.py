from types import EllipsisType
from typing import TypeVar, overload, ContextManager
from .base import Op, OpInputPlaceholder
from .add import OpAdd
from .bmm import OpBMM
from .scale import OpScale
from .view import OpView
from .layernorm import OpFLayerNormCore, OpDFLayerNormCore
from .dropout import OpFDropout
from .softmax import OpFSoftmax, OpDFSoftmax
from .gelu import OpFGelu, OpDFGelu
from .mul import OpMul

T = TypeVar("T", bound=Op)


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
    in_nodes: list[OpInputPlaceholder]
    out_nodes: list[Op]

    def __init__(self):
        self.trainable_parameters = []
        self.nodes = []
        self.in_nodes = []
        self.out_nodes = []

    def in_(self):
        """Adds an input node with the given number of features to the graph. Returns the node, so it can be used in further computations."""
        node = OpInputPlaceholder()
        self.in_nodes.append(node)
        return node

    def param_(self, features: int, name: str):
        """Adds a trainable parameter with the given number of features to the graph. Returns the parameter, so it can be used in further computations."""
        self.trainable_parameters.append(ParamDescriptor(features, name))
        node = OpParam(self.trainable_parameters[-1])
        self.nodes.append(node)
        return node

    def out_(self, op: T) -> T:
        """Marks the given op as an output node of the graph. The op must be a node of the graph. Returns the op back."""
        assert op in self.nodes
        self.out_nodes.append(op)
        return op

    def bmm_(self, a: Op, b: Op):
        """Adds a batched matrix multiplication node to the graph. Returns the node with the result."""
        node = OpBMM(a, b)
        self.nodes.append(node)
        return node

    def add_(self, a: Op, b: Op):
        """Adds an addition node to the graph. Returns the node with the result."""
        node = OpAdd(a, b)
        self.nodes.append(node)
        return node

    def mul_(self, a: Op, b: Op):
        """Adds an element-wise multiplication node to the graph. Returns the node with the result."""
        node = OpMul(a, b)
        self.nodes.append(node)
        return node

    def f_layernorm_(
        self, a: Op, gamma: Op, beta: Op, eps: float, zero_centered_gamma: bool
    ) -> Op:
        """Adds a layernorm node to the graph. Returns the node with the result."""
        self.nodes.append(OpFLayerNormCore(a, eps))
        core = self.nodes[-1]
        node = self.mul_(core, gamma)
        if zero_centered_gamma:
            node = self.add_(node, core)
        node = self.add_(node, beta)
        return node

    def df_layernorm_core_(self, a: Op, eps: float):
        """Adds a layernorm' (derivative) node to the graph. Returns the node with the result."""
        node = OpDFLayerNormCore(a, eps)
        self.nodes.append(node)
        return node

    def f_dropout_(self, a: Op, p: float, rng_ctx: ContextManager[None]):
        """Adds a dropout node to the graph. Returns the node with the result"""
        node = OpFDropout(a, p, rng_ctx)
        self.nodes.append(node)
        return node

    def f_softmax_(self, a: Op):
        """Adds a softmax node to the graph. Returns the node with the result"""
        node = OpFSoftmax(a)
        self.nodes.append(node)
        return node

    def df_softmax_(self, a: Op):
        """Adds a softmax' (derivative) node to the graph. Returns the node with the result"""
        node = OpDFSoftmax(a)
        self.nodes.append(node)
        return node

    def f_gelu_(self, a: Op):
        """Adds a gelu node to the graph. Returns the node with the result."""
        node = OpFGelu(a)
        self.nodes.append(node)
        return node

    def df_gelu_(self, a: Op):
        """Adds a gelu' (derivative) node to the graph. Returns the node with the result."""
        node = OpDFGelu(a)
        self.nodes.append(node)
        return node

    def t_(self, a: Op):
        """Adds a transpose node to the graph. Returns the node with the result."""
        return self.view_(a, [..., -1, -2])

    @overload
    def scale_(self, a: float, b: Op) -> OpScale:
        ...

    @overload
    def scale_(self, a: Op, b: float) -> OpScale:
        ...

    def scale_(self, a: Op | float, b: Op | float):
        """Adds a scaling node to the graph. Returns the node with the result."""
        if isinstance(a, float):
            assert isinstance(b, Op)
            node = OpScale(b, a)
        else:
            assert isinstance(b, float)
            node = OpScale(a, b)
        self.nodes.append(node)
        return node

    def view_(self, a: Op, shape: list[int | EllipsisType]):
        """Adds a view node that permutes the order of dimensions of the Op. Returns the node with the result."""
        node = OpView(a, shape)
        self.nodes.append(node)
        return node

    def create_backward_graph_(self, node: Op, grad: Op):
        """Creates the graph of the backward pass, assuming that grad is the gradient of the loss with respect to the node."""
        assert node in self.out_nodes
        assert grad in self.in_nodes

        def set_grad(node: Op, grad: Op):
            if node.grad is None:
                node.grad = grad
            else:
                node.grad = self.add_(node.grad, grad)

        set_grad(node, grad)
        bfs = [node]
        while len(bfs) > 0:
            cur = bfs.pop()
            params = cur.__dict__.items()
            for name, param in params:
                if isinstance(param, Op) and name != "grad":
                    param_grad_func = getattr(cur, f"backward_{name}")
                    set_grad(param, param_grad_func(self, cur.grad))
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
