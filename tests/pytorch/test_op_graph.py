import transformer_engine.pytorch as te
from transformer_engine.pytorch.sequential import ComputePipeline
from transformer_engine.pytorch.sequential.ops.op_graph import (
    Op,
    OpInputPlaceholder,
    OpParam,
    OpGraph,
)

seq = te.Sequential(te.Linear(1, 1), te.Linear(1, 1), te.Linear(1, 1), te.Linear(1, 1))
c = ComputePipeline(*seq._modules.values())


def render(graph: OpGraph):
    s = ""
    for node in graph.nodes:
        own_name = f"{node.__class__.__name__} at {hex(node.id)}"
        if isinstance(node, OpInputPlaceholder):
            s += (f'"{own_name}"') + "\n"
        if isinstance(node, OpParam):
            s += (f'"{own_name}" [group=param, label="{node.param.name}"]') + "\n"
        else:
            s += (
                f'"{own_name}" [group=op, label="{node.__class__.__name__}", shape="box"]'
            ) + "\n"
        for name, value in node.__dict__.items():
            if isinstance(value, Op) and name != "grad":
                s += (
                    f'"{value.__class__.__name__} at {hex(value.id)}" -> "{own_name}" [label="{name}"]'
                    + "\n"
                )
    return s


fwd = render(c._graph).splitlines()
c.backward()
print()
combined = render(c._graph).splitlines()

bwd = []
for line in combined:
    if line not in fwd:
        try:
            a, b = line.split(" -> ")
            b, c = b.split(" [", 1)
            c = " [" + c
            line = f"{b} -> {a}{c}[dir=back]"
        except:
            pass
        bwd.append(line)

s = ""
s += ("digraph G {") + "\n"
s += ("rankdir=LR") + "\n"
s += ("subgraph cluster_0 {") + "\n"
s += ("\n".join(fwd)) + "\n"
s += ("} subgraph cluster_1 {") + "\n"
s += ("\n".join(bwd).replace("op", "opbwd").replace("param", "parambwd")) + "\n"
s += ("}") + "\n"
s += ("}") + "\n"
print(s)
