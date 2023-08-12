import copy
from copy import deepcopy
from functools import partial, reduce
import operator
from typing import Callable, Literal
from typing_extensions import Unpack
from .utils import set_attribute
import transformer_engine_cuda as nvte
from .nvte_utils import is_fp8
from .ops import (
    BackwardFused,
    ForwardFused,
    Grads,
    Op,
    FUSIONS_INF,
    FUSIONS_FWD,
    FUSIONS_BWD,
    Context,
    Inference,
)
from .environment import Environment


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
        full_ctx = Context()
        for op, ctx in zip(self.ops, ctxs):
            op_name = getattr(op, "name")
            ctx: Context = {op_name + name: tensor for name, tensor in ctx.items()}
            full_ctx |= ctx
        return y, full_ctx

    def backward(self, ctx: Context, dy: nvte.Tensor):
        assert self.backward_ is not None
        ctxs = list[Context]()
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

    def args(self):
        return list(sum((op.args() for op in self.ops), list[nvte.Tensor]()))

    def __repr__(self):
        return f"""FusedOp{self.ops}"""


class SelfContainedOp(Op):
    def __init__(self, fwds: list[Op], bwds: list[Op]) -> None:
        self.fwds = fwds
        self.bwds = bwds

    def inference(self, x: nvte.Tensor) -> nvte.Tensor:
        raise AssertionError("Not used for inference")

    def forward(self, x: nvte.Tensor):
        full_ctx = Context()
        for op in self.fwds:
            x, ctx = op.forward(x)
            if not isinstance(op, FusedOp):
                op_name = getattr(op, "name")
                ctx = {op_name + name: tensor for name, tensor in ctx.items()}
            full_ctx |= ctx
        return x, full_ctx

    def backward(self, ctx: Context, dy: nvte.Tensor):
        ctxs = list[Context]()
        for op in self.bwds:
            if isinstance(op, FusedOp):
                ctxs.append(ctx)
            else:
                op_name = getattr(op, "name")
                ctxs.append(
                    {
                        name[len(op_name) :]: tensor
                        for name, tensor in ctx.items()
                        if name.startswith(op_name)
                    }
                )

        full_grads = Grads()
        for op, ctx in list(zip(self.bwds, ctxs))[::-1]:
            dy, grads = op.backward(ctx, dy)
            full_grads += grads
        return dy, full_grads

    def args(self):
        return list(sum((op.args() for op in self.fwds), list[nvte.Tensor]()))


def force_use_bf16(ops: list[Op]):
    for op in ops:
        attributes = dir(op)
        dtype_attributes = [attr for attr in attributes if attr.endswith("_dtype")]
        for dtype_attribute in dtype_attributes:
            attr_val = getattr(op, dtype_attribute)
            if isinstance(attr_val, nvte.DType) and is_fp8(attr_val):
                setattr(op, dtype_attribute, nvte.DType.BFloat16)


def model_parallel_transform(ops: list[Op]):
    raise NotImplementedError()


def get_list(ops: list[Op], fuse_by: Literal["forward", "backward", "inference"]):
    ops = ops.copy()
    if fuse_by == "forward":
        fusion_dict = FUSIONS_FWD
    elif fuse_by == "backward":
        fusion_dict = FUSIONS_BWD
    else:  # pass_ == "inference":
        fusion_dict = FUSIONS_INF
    fusions = [(len(arg_types), arg_types, f) for arg_types, f in fusion_dict.items()]
    fusions.sort(key=lambda x: x[0], reverse=True)  # largest first
    for _, arg_types, f in fusions:
        for startPos in range(len(ops) - len(arg_types) + 1):
            if all(
                isinstance(ops[i], arg_types[i - startPos])
                for i in range(len(arg_types))
            ):
                fused_ops = ops[startPos : startPos + len(arg_types)]
                func = partial(f, *fused_ops)
                fused_op = FusedOp(fused_ops, **{fuse_by: func})
                ops[startPos : startPos + len(arg_types)] = [fused_op]
    return ops


def name_ops(ops: list[Op]):
    for i, op in enumerate(ops):
        setattr(op, "name", f"{i}({op.__class__.__name__})")


def split_into_self_contained(fwds: list[Op], bwds: list[Op]):
    functions = list[SelfContainedOp]()
    while fwds or bwds:
        fwd = fwds.pop(0)
        unmatched_fwd_ops: set[Op] = {
            *reduce(operator.iadd, [fwd.ops if isinstance(fwd, FusedOp) else [fwd]], [])
        }
        used_forwards = [fwd]
        used_backwards = list[Op]()
        unmatched_bwd_ops: set[Op] = set()
        while unmatched_fwd_ops or unmatched_bwd_ops:
            while unmatched_fwd_ops:
                bwd = bwds.pop(0)
                used_backwards.append(bwd)
                ops = bwd.ops if isinstance(bwd, FusedOp) else [bwd]
                for op in ops:
                    if op in unmatched_fwd_ops:
                        unmatched_fwd_ops.remove(op)
                    else:
                        unmatched_bwd_ops.add(op)
            while unmatched_bwd_ops:
                fwd = fwds.pop(0)
                used_forwards.append(fwd)
                ops = fwd.ops if isinstance(fwd, FusedOp) else [fwd]
                for op in ops:
                    if op in unmatched_bwd_ops:
                        unmatched_bwd_ops.remove(op)
                    else:
                        unmatched_fwd_ops.add(op)
        functions.append(SelfContainedOp(used_forwards, used_backwards))
    return functions


def copy_op_list(ops: list[Op]):
    "Deep copy ops, except for tensors"
    with set_attribute(nvte.Tensor, "__deepcopy__", lambda self, memo: self):
        return copy.deepcopy(ops)


class ComputePipeline:
    def __init__(self, ops: list[Op], env: Environment):
        ops = copy_op_list(ops)

        name_ops(ops)
        if not env.fp8_enabled:
            force_use_bf16(ops)
        if env.world_size > 1:
            model_parallel_transform(ops)

        self._inf = get_list(ops, "inference")

        self.functions = split_into_self_contained(
            get_list(ops, "forward"), get_list(ops, "backward")
        )

    def run_inference(self, x: nvte.Tensor) -> nvte.Tensor:
        for op in self._inf:
            x = op.inference(x)
        return x

    def __repr__(self):
        return f"""ComputePipeline(
    forward: {tuple(
        op
        for f in self.functions
        for op in f.fwds
    )},
    backward: {tuple(
        op
        for f in self.functions
        for op in f.bwds
    )},
)"""
