from __future__ import annotations
from dataclasses import dataclass
from typing import final

from transformer_engine.pytorch.sequential_new.common_back.generic_tensor import (
    GenericTensor,
)
from .enums import DTypeInfer, PType
from .ops import (
    NoSupplementaryTensorsOp,
    Op,
    ParameterFreeOp,
    ShapePreserveOp,
    Identity,
    Transpose,
    Scatter,
    AllGather,
    AllReduce,
    _normal,
)


@dataclass
class Connection:
    src: PType
    dst: PType
    ops: list[Op]

    def cost(self):
        weight: int
        if (self.src, self.dst) == (PType.NA, PType.NA):
            weight = 1
        elif (self.src, self.dst) == (PType.PA, PType.PA):
            weight = 1
        else:
            weight = 0
        return weight * len(self.ops)


@dataclass
class Node:
    connections: list[Connection]


def model_parallel_transform(ops: list[Op]) -> list[Op]:
    ops = [Input("input")] + ops + [Output("output")]
    graph: list[Node] = []
    for op in ops:
        possible_chains: list[list[Op]] = op.describe_parallellism()
        connections = list[Connection]()
        for subops in possible_chains:
            for i, subop in enumerate(subops[:-1]):
                after = subops[i + 1]
                if subop.parallellism[1] != after.parallellism[0]:
                    raise ValueError("Parallelism type mismatch in chain")
            connections.append(
                Connection(
                    subops[0].parallellism[0],
                    subops[-1].parallellism[1],
                    subops,
                )
            )
        graph.append(Node(connections))
    return _bfs01(graph)


def _bfs01(graph: list[Node]) -> list[Op]:
    vertices = (len(graph) + 1) * len(PType)
    START = 0
    FINISH = vertices - (len(PType))
    UNREACHABLE = 1000000000

    dist = [UNREACHABLE] * vertices
    prev: list[Connection | None] = [None] * vertices
    q0: list[tuple[int, int, Connection | None]] = [(START, 0, None)]
    q1: list[tuple[int, int, Connection | None]] = []

    def reachable(v: int):
        return dist[v] <= len(graph)

    while q0 or q1:
        for q in [q0, q1]:
            while q:
                v, d, p = q.pop()
                if d < dist[v]:
                    dist[v], prev[v] = d, p

                node = v // len(PType)
                endpoint = PType(v % len(PType))

                if node < len(graph):
                    for conn in graph[node].connections:
                        if conn.src == endpoint:
                            u = (node + 1) * len(PType) + conn.dst.value
                            if not reachable(u):
                                cost = conn.cost()
                                if cost == 0:
                                    q0.append((u, dist[v], conn))
                                else:
                                    q1.append((u, dist[v] + cost, conn))
    assert dist[FINISH] <= len(graph) - 2  # at least as good as no parallelism

    path: list[Connection] = []
    v = FINISH
    while v != START:
        conn = prev[v]
        assert conn is not None
        path.append(conn)
        node = v // len(PType)
        v = (node - 1) * len(PType) + conn.src.value
    path.reverse()
    ops = [op for conn in path for op in conn.ops]
    return ops


@final
class Input(ParameterFreeOp, ShapePreserveOp, NoSupplementaryTensorsOp):
    def __init__(self, name: str):
        super().__init__(name, DTypeInfer(), DTypeInfer())

    def training(self, x: GenericTensor):
        raise RuntimeError("Input op should not be called directly")

    def inference(self, x: GenericTensor):
        raise RuntimeError("Input op should not be called directly")

    def describe_parallellism(self) -> list[list[Op]]:
        i = Identity("identity", self.input_type, self.output_type).named(self.name)
        s = Scatter("scatter").named(self.name)
        t = Transpose("transpose").named(self.name)
        return [
            _normal(i),
            [s, t],
            [s],
        ]


@final
class Output(ParameterFreeOp, ShapePreserveOp, NoSupplementaryTensorsOp):
    def __init__(self, name: str):
        super().__init__(name, DTypeInfer(), DTypeInfer())

    def training(self, x: GenericTensor):
        raise RuntimeError("Output op should not be called directly")

    def inference(self, x: GenericTensor):
        raise RuntimeError("Output op should not be called directly")

    def describe_parallellism(self) -> list[list[Op]]:
        i = Identity("identity", self.input_type, self.output_type).named(self.name)
        ar = AllReduce("allreduce").named(self.name)
        ag = AllGather("allgather").named(self.name)
        t = Transpose("transpose", self.input_type, self.output_type).named(self.name)
        return [
            _normal(i),
            [t, ag],
            [ar],
            [ag],
        ]
