import copy
from functools import reduce
import operator
from . import nvte
from .ops import Op, Grads, Context
from .fusions import FusedOp, get_fused_op_list
from .utils import set_attribute
from .recipe import Recipe
from .meta import PersistentFP8Meta


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

    def require_grad(self):
        return list(sum((op.require_grad() for op in self.fwds), list[nvte.Tensor]()))


def force_use_precision(ops: list[Op], allowed: nvte.DType):
    PRECISION = {
        nvte.DType.Float8E4M3: 0,
        nvte.DType.Float8E5M2: 0,
        nvte.DType.BFloat16: 1,
        nvte.DType.Float16: 2,
        nvte.DType.Float32: 3,
        nvte.DType.Int64: 4,
    }

    for op in ops:
        attributes = dir(op)
        dtype_attributes = [attr for attr in attributes if attr.endswith("_dtype")]
        for dtype_attribute in dtype_attributes:
            attr_val = getattr(op, dtype_attribute)
            if (
                isinstance(attr_val, nvte.DType)
                and PRECISION[attr_val] < PRECISION[allowed]
            ):
                setattr(op, dtype_attribute, allowed)


def model_parallel_transform(ops: list[Op]):
    raise NotImplementedError()  # TODO


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


# Needed for copy_op_list
# Shouldn't cause any issues
setattr(nvte.Tensor, "__deepcopy__", lambda self, memo: self) # type: ignore

def copy_op_list(ops: list[Op]):
    "Deep copy ops, except for tensors"
    return copy.deepcopy(ops)


class ComputePipeline:
    def __init__(self, ops: list[Op], env: Recipe):
        ops = copy_op_list(ops)

        name_ops(ops)
        force_use_precision(ops, env.lowp)
        if env.world_size > 1:
            model_parallel_transform(ops)

        self._inf = get_fused_op_list(ops, "inference")

        self.functions = split_into_self_contained(
            get_fused_op_list(ops, "forward"), get_fused_op_list(ops, "backward")
        )
        self.forward = tuple(op for f in self.functions for op in f.fwds)
        self.backward = tuple(op for f in self.functions for op in f.bwds)
        self.meta_fwd = PersistentFP8Meta()
        self.meta_bwd = PersistentFP8Meta()

    def run_inference(self, x: nvte.Tensor):
        for op in self._inf:
            x = op.inference(x)
        return x

    def next_iteration(self):
        self.meta_fwd.next_iteration()
        self.meta_bwd.next_iteration()

    def __repr__(self):
        return f"""ComputePipeline(
    forward: {self.forward},
    backward: {self.backward},
)"""
