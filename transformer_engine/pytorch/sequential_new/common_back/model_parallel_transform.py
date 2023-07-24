from __future__ import annotations
from dataclasses import dataclass
from .enums import PType
from .ops import (
    OpBase,
    Gemm,
    Bias,
    Gelu,
    PassthroughOp,
    Relu,
    ResidualBegin,
    ResidualEnd,
    LayerNorm,
    Transpose,
    DotProductAttention,
    Dropout,
)


@dataclass
class Connection:
    src: PType
    dst: PType
    ops: list[OpBase]

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


def model_parallel_transform(ops: list[OpBase]) -> list[OpBase]:
    graph: list[Node] = []
    for op in ops:
        graph.append(op.describe_parallellism())
    graph = [_pre(ops[0])] + graph + [_post(ops[-1])]

    return _bfs01(graph)


def _bfs01(graph: list[Node]) -> list[OpBase]:
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


def _ugemm(gemm: Gemm) -> Node:
    rgemm = GemmRowParallel(
        "rgemm",
        gemm.input_type,
        gemm.output_type,
        gemm.in_features,
        gemm.out_features,
        gemm.init_method,
    ).named(gemm.name)
    cgemm = GemmColParallel(
        "cgemm",
        gemm.input_type,
        gemm.output_type,
        gemm.in_features,
        gemm.out_features,
        gemm.init_method,
    ).named(gemm.name)
    rs = ReduceScatter("rs", gemm.output_type, gemm.output_type).named(gemm.name)
    ag = AllGather("ag", gemm.input_type, gemm.input_type).named(gemm.name)

    return Node(
        [
            Connection(PType.NA, PType.NA, [gemm]),
            Connection(PType.NA, PType.NCS, [cgemm]),
            Connection(PType.NA, PType.PA, [rgemm]),
            Connection(PType.NCS, PType.PA, [rgemm]),
            Connection(PType.NCS, PType.NRS, [rgemm, rs]),
            Connection(PType.NRS, PType.NCS, [ag, cgemm]),
        ]
    )


def _dot_product_attention(dpa: DotProductAttention) -> Node:
    return Node(
        [
            Connection(PType.NA, PType.NA, [dpa]),
            Connection(PType.NA, PType.NCS, [dpa]),
            Connection(PType.NCS, PType.NCS, [dpa]),
            Connection(PType.NCS, PType.NA, [dpa]),
        ]
    )


def _pre(firstOp: OpBase) -> Node:
    s = Scatter("pre-scatter").named(firstOp.name)
    t = Transpose("pre-transpose").named(firstOp.name)

    return Node(
        [
            Connection(PType.NA, PType.NA, []),
            Connection(PType.NA, PType.NCS, [s, t]),
            Connection(PType.NA, PType.NRS, [s]),
        ]
    )


def _post(lastOp: OpBase) -> Node:
    ar = AllReduce("post-allreduce").named(lastOp.name)
    ag = AllGather("post-allgather").named(lastOp.name)
    t = Transpose("post-transpose").named(lastOp.name)

    return Node(
        [
            Connection(PType.NA, PType.NA, []),
            Connection(PType.NCS, PType.NA, [t, ag]),
            Connection(PType.PA, PType.NA, [ar]),
            Connection(PType.NRS, PType.NA, [ag]),
        ]
    )


def _pointwise(op: OpBase) -> Node:
    return Node(
        [
            Connection(PType.NA, PType.NA, [op]),
            Connection(PType.NCS, PType.NCS, [op]),
            # TODO: for ex. Bias could work with PA if it was divided
            Connection(PType.NRS, PType.NRS, [op]),
        ]
    )


def _rowwise(op: OpBase) -> Node:
    return Node(
        [
            Connection(PType.NA, PType.NA, [op]),
            Connection(PType.NRS, PType.NRS, [op]),
        ]
    )


def _unknown(op: OpBase) -> Node:
    return Node(
        [
            Connection(PType.NA, PType.NA, [op]),
        ]
    )


class ReduceScatter(PassthroughOp):
    def bwd(self):
        raise ValueError("This operation is meant for internal use only")


class Scatter(PassthroughOp):
    def bwd(self):
        raise ValueError("This operation is meant for internal use only")


class AllGather(PassthroughOp):
    def bwd(self):
        raise ValueError("This operation is meant for internal use only")


class AllReduce(PassthroughOp):
    def bwd(self):
        raise ValueError("This operation is meant for internal use only")


class GemmRowParallel(Gemm):
    def bwd(self):
        raise ValueError("This operation is meant for internal use only")


class GemmColParallel(Gemm):
    def bwd(self):
        raise ValueError("This operation is meant for internal use only")
