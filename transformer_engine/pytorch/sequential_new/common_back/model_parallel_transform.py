from __future__ import annotations
from dataclasses import dataclass
from .enums import PType
from .ops import OpBase


@dataclass
class Connection:
    src: PType
    dst: PType
    op: OpBase

    def cost(self):
        weight: int
        if (self.src, self.dst) == (PType.NA, PType.NA):
            weight = 1
        elif (self.src, self.dst) == (PType.PA, PType.PA):
            weight = 1
        else:
            weight = 0
        return weight


@dataclass
class Node:
    connections: list[Connection]


def model_parallel_transform(ops: list[OpBase]):
    graph: list[Node] = []
    for op in ops:
        graph.append(
            Node([Connection(src, dst, op) for src, dst in op.describe_parallellism()])
        )
    _bfs01(graph)


def _bfs01(graph: list[Node]):
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
    for conn in path:
        conn.op.parallellism = (conn.src, conn.dst)
    path.reverse()
