from __future__ import annotations

from functools import partial
from ..ops import Op
from typing import Literal
from ... import nvte
from ..ops_types import (
    BackwardFused,
    ForwardFused,
    Grads,
    Context,
    Inference,
)
from ._storage import FUSIONS_FWD, FUSIONS_BWD, FUSIONS_INF


class FusedOp(Op):
    def __init__(
        self,
        ops: list[Op],
        forward: ForwardFused | None = None,
        backward: BackwardFused | None = None,
        inference: Inference | None = None,
    ):
        self.forward_ = forward
        self.backward_ = backward
        self.inference_ = inference
        self.ops = ops

    def inference(self, x: nvte.Tensor) -> nvte.Tensor:
        assert self.inference_ is not None
        return self.inference_(x)

    def forward(self, x: nvte.Tensor):
        assert self.forward_ is not None
        y, ctxs = self.forward_(x)
        full_ctx: Context = {}
        for op, ctx in zip(self.ops, ctxs):
            op_name = getattr(op, "name")
            ctx: Context = {op_name + name: tensor for name, tensor in ctx.items()}
            full_ctx.update(ctx)
        return y, full_ctx

    def backward(self, ctx: Context, dy: nvte.Tensor):
        assert self.backward_ is not None
        ctxs: list[Context] = []
        for op in self.ops:
            op_name = getattr(op, "name")
            ctxs.append(
                {
                    name[len(op_name) :]: tensor
                    for name, tensor in ctx.items()
                    if name.startswith(op_name)
                }
            )

        dx, grads = self.backward_(*ctxs, dy)
        grads_total: Grads = [grad for op_grads in grads for grad in op_grads]
        return dx, grads_total

    def require_grad(self):
        list_: list[nvte.Tensor] = []
        for op in self.ops:
            list_.extend(op.require_grad())
        return list_

    def __repr__(self):
        return f"""FusedOp{self.ops}"""


def get_fused_op_list(
    ops: list[Op], fuse_by: Literal["forward", "backward", "inference"]
):
    ops = ops.copy()
    if fuse_by == "forward":
        fusion_dict = FUSIONS_FWD
    elif fuse_by == "backward":
        fusion_dict = FUSIONS_BWD
    else:  # pass_ == "inference":
        fusion_dict = FUSIONS_INF
    fusions = [(len(arg_types), arg_types, f) for arg_types, f in fusion_dict.items()]
    fusions.sort(key=lambda x: x[0], reverse=True)  # largest first
    for cnt, arg_types, f in fusions:
        startPos = 0
        while startPos < len(ops) - cnt + 1:
            if all(isinstance(ops[startPos + i], arg_types[i]) for i in range(cnt)):
                fused_ops = ops[startPos : startPos + cnt]
                func = partial(f, *fused_ops)
                fused_op = FusedOp(fused_ops, **{fuse_by: func})
                ops[startPos : startPos + cnt] = [fused_op]
            startPos += 1
    return ops


__all__ = ["FusedOp", "get_fused_op_list"]
