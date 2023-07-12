from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from math import inf
from .ops import (
    Op,
    Gemm,
    Add,
    Gelu,
    PassthroughOp,
    Relu,
    ResidualBegin,
    ResidualEnd,
    LayerNorm,
    Transpose,
    DotProductAttention,
)


def model_parallel_transform(ops: list[Op]) -> list[Op]:
    graph: list[Node] = []
    for op in ops:
        if isinstance(op, Gemm):
            graph.append(_ugemm(op))
        elif isinstance(op, DotProductAttention):
            graph.append(_dot_product_attention(op))
        elif type(op) in POINTWISE_OPS:
            graph.append(_pointwise(op))
        elif type(op) in ROWWISE_OPS:
            graph.append(_rowwise(op))
        else:
            graph.append(_unknown(op))
    graph = [_pre(ops[0])] + graph + [_post(ops[-1])]

    return _bfs01(graph)


POINTWISE_OPS = [Add, Gelu, Relu, ResidualBegin, ResidualEnd]
ROWWISE_OPS = [LayerNorm]


def _bfs01(graph: list[Node]) -> list[Op]:
    vertices = (len(graph) + 1) * len(EndPoint)
    START = 0
    FINISH = vertices - (len(EndPoint))
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

                node = v // len(EndPoint)
                endpoint = EndPoint(v % len(EndPoint))

                if node < len(graph):
                    for conn in graph[node].connections:
                        if conn.src == endpoint:
                            u = (node + 1) * len(EndPoint) + conn.dst.value
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
        node = v // len(EndPoint)
        v = (node - 1) * len(EndPoint) + conn.src.value
    path.reverse()
    ops = [op for conn in path for op in conn.ops]
    return ops


class EndPoint(Enum):
    NA = 0
    "normal all"
    NCS = 1
    "normal column-split"
    PA = 2
    "partial all"
    NRS = 3
    "normal row-split"


@dataclass
class Connection:
    src: EndPoint
    dst: EndPoint
    ops: list[Op]

    def cost(self):
        weight: int
        if (self.src, self.dst) == (EndPoint.NA, EndPoint.NA):
            weight = 1
        elif (self.src, self.dst) == (EndPoint.PA, EndPoint.PA):
            weight = 1
        else:
            weight = 0
        return weight * len(self.ops)


@dataclass
class Node:
    connections: list[Connection]


def _ugemm(gemm: Gemm) -> Node:
    rgemm = GemmRowParallel(
        "rgemm", gemm.input_type, gemm.output_type, gemm.in_features, gemm.out_features
    ).named(gemm.name)
    cgemm = GemmColParallel(
        "cgemm", gemm.input_type, gemm.output_type, gemm.in_features, gemm.out_features
    ).named(gemm.name)
    rs = ReduceScatter("rs", gemm.output_type, gemm.output_type).named(gemm.name)
    ag = AllGather("ag", gemm.input_type, gemm.input_type).named(gemm.name)

    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, [gemm]),
            Connection(EndPoint.NA, EndPoint.NCS, [cgemm]),
            Connection(EndPoint.NCS, EndPoint.PA, [rgemm]),
            Connection(EndPoint.NCS, EndPoint.NRS, [rgemm, rs]),
            Connection(EndPoint.NRS, EndPoint.NCS, [ag, cgemm]),
        ]
    )


def _dot_product_attention(dpa: DotProductAttention) -> Node:
    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, [dpa]),
            Connection(EndPoint.NA, EndPoint.NCS, [dpa]),
            Connection(EndPoint.NCS, EndPoint.NCS, [dpa]),
            Connection(EndPoint.NCS, EndPoint.NA, [dpa]),
        ]
    )


def _pre(firstOp: Op) -> Node:
    s = Scatter("pre-scatter").named(firstOp.name)
    t = Transpose("pre-transpose").named(firstOp.name)

    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, []),
            Connection(EndPoint.NA, EndPoint.NCS, [s, t]),
            Connection(EndPoint.NA, EndPoint.NRS, [s]),
        ]
    )


def _post(lastOp: Op) -> Node:
    ar = AllReduce("post-allreduce").named(lastOp.name)
    ag = AllGather("post-allgather").named(lastOp.name)
    t = Transpose("post-transpose").named(lastOp.name)

    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, []),
            Connection(EndPoint.NCS, EndPoint.NA, [t, ag]),
            Connection(EndPoint.PA, EndPoint.NA, [ar]),
            Connection(EndPoint.NRS, EndPoint.NA, [ag]),
        ]
    )


def _pointwise(op: Op) -> Node:
    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, [op]),
            Connection(EndPoint.NCS, EndPoint.NCS, [op]),
            # TODO: for ex. Add could work with PA if it was divided
            Connection(EndPoint.NRS, EndPoint.NRS, [op]),
        ]
    )


def _rowwise(op: Op) -> Node:
    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, [op]),
            Connection(EndPoint.NRS, EndPoint.NRS, [op]),
        ]
    )


def _unknown(op: Op) -> Node:
    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, [op]),
        ]
    )


class ReduceScatter(PassthroughOp):
    pass


class Scatter(PassthroughOp):
    pass


class AllGather(PassthroughOp):
    pass


class AllReduce(PassthroughOp):
    pass


class GemmRowParallel(Gemm):
    pass


class GemmColParallel(Gemm):
    pass
