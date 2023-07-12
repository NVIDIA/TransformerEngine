from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
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
)


def model_parallel_transform(ops: list[Op]) -> list[Op]:
    graph: list[Node] = []
    for op in ops:
        if isinstance(op, Gemm):
            graph.append(_ugemm(op))
        elif type(op) in POINTWISE_OPS:
            graph.append(_pointwise(op))
        elif type(op) in ROWWISE_OPS:
            graph.append(_rowwise(op))
        else:
            graph.append(_unknown(op))
    best_path = _bfs01(graph)
    return [op for conn in best_path for op in conn.ops]


POINTWISE_OPS = [Add, Gelu, Relu, ResidualBegin, ResidualEnd]
ROWWISE_OPS = [LayerNorm]


def _bfs01(graph: list[Node]) -> list[Connection]:
    ...


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
        if (self.src, self.dst) == (EndPoint.NA, EndPoint.NA):
            return 1
        elif (self.src, self.dst) == (EndPoint.PA, EndPoint.PA):
            return 1
        else:
            return 0


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
    rs = ReduceScatter("rs").named(gemm.name)
    ag = AllGather("ag").named(gemm.name)

    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, [gemm]),
            Connection(EndPoint.NA, EndPoint.NCS, [cgemm]),
            Connection(EndPoint.NCS, EndPoint.PA, [rgemm]),
            Connection(EndPoint.NCS, EndPoint.NRS, [rgemm, rs]),
            Connection(EndPoint.NRS, EndPoint.NCS, [ag, cgemm]),
        ]
    )


def _pre(firstOp: Op) -> Node:
    s = Scatter("pre-scatter").named(firstOp.name)
    t = Transpose("pre-transpose").named(firstOp.name)

    return Node(
        [
            Connection(EndPoint.NA, EndPoint.NA, []),
            Connection(EndPoint.NA, EndPoint.NCS, [s, t]),
            Connection(EndPoint.NCS, EndPoint.NRS, [s]),
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
