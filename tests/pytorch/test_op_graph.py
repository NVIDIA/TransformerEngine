import transformer_engine.pytorch as te
from transformer_engine.pytorch.sequential import ComputePipeline
from transformer_engine.pytorch.sequential.ops.op_graph import (
    Op,
    OpInputPlaceholder,
    OpParam,
)

seq = te.Sequential(
    te.LayerNormLinear(1, 1),
    te.LayerNormMLP(1, 1),
    te.Sequential(te.LayerNormLinear(1, 1), te.LayerNormMLP(1, 1)),
)
c = ComputePipeline(*seq._modules.values())


def to_graph(node, top=True):
    if top:
        print('digraph G {\nrankdir="LR"')
    own_name = f"{node.__class__.__name__} at {hex(id(node))}"
    if isinstance(node, OpInputPlaceholder):
        print(f'"{own_name}" [group="input"]')
    elif isinstance(node, OpParam):
        print(f'"{own_name}" [group="param", label="{node.param.name}"]')
    else:
        print(
            f'"{own_name}" [group="op", label="{node.__class__.__name__}", shape="box"]'
        )
    for name, value in node.__dict__.items():
        if isinstance(value, Op):
            print(
                f'"{value.__class__.__name__} at {hex(id(value))}" -> "{own_name}" [label="{name}"]'
            )
            to_graph(value, False)
    if top:
        print("}")


to_graph(c._graph.out_nodes[0])
