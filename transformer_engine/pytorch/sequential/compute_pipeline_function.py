from dataclasses import dataclass
import torch
from torch import autograd
from torch.autograd.function import FunctionCtx
from . import nvte
from .ops import Context, Op
from .compute_pipeline import ComputePipeline


@dataclass
class CommWithLoop:
    nvte_x: nvte.Tensor
    is_exposed_x_squished_now: bool


class ComputePipelineFunction(autograd.Function):
    @staticmethod
    def forward(  # type: ignore[arg-type]
        ctx: FunctionCtx,
        exposed_x: torch.Tensor,
        *args: torch.Tensor | Op | CommWithLoop,
    ):
        """
        exposed_x is used only to let autograd construct the computation graph
        real input and output is in list, as nvte.Tensor is immutable
        exposed_tensors are exposed for the optimizer to later apply gradients
        """
        exposed_tensors, op, comm = args[:-2], args[-2], args[-1]
        del exposed_tensors

        assert isinstance(op, Op)
        assert isinstance(comm, CommWithLoop)
        nvte_x = comm.nvte_x
        assert isinstance(nvte_x, nvte.Tensor)

        nvte.set_current_pass("forward")
        y, to_save = op.forward(nvte_x)

        # Expose backward context for tracing
        bwd_ctx = list[torch.Tensor]()
        for _, tensor in to_save.items():
            bwd_ctx.append(tensor.data)
            if tensor.amax.numel():
                bwd_ctx.append(tensor.amax)
            if tensor.scale.numel():
                bwd_ctx.append(tensor.scale)
            if tensor.scale_inv.numel():
                bwd_ctx.append(tensor.scale_inv)
        ctx.save_for_backward(*bwd_ctx)

        # Save real context
        setattr(ctx, "nvte_ctx", to_save)
        setattr(ctx, "nvte_op", op)

        # Actually store the result
        comm.nvte_x = y

        # Unsquish x if needed:
        if comm.is_exposed_x_squished_now:
            # Intentionally commented out - _unsquish(exposed_x)
            # We don't need to perform the unsquish itself, as this
            # data will not be read anyway.
            # Actually, we cannot do that, as x,
            # cannot be modified in place.
            # It is only really neccesarry to notify
            # the backward.
            comm.is_exposed_x_squished_now = False
            # If the input to the forward was squished,
            # Pytorch will expect its gradient to be squished
            # as well. The backward of this forward will be
            # responsible for producing the gradient of
            # this squished input, so it is responsible for
            # squishing it.
            setattr(ctx, "nvte_squish_outgoing_dgrad", True)
        else:
            setattr(ctx, "nvte_squish_outgoing_dgrad", False)

        # Expose result for Pytorch
        x_data = exposed_x.data
        exposed_x.data = torch.Tensor()  # avoid copy
        exposed_y = exposed_x.clone()  # copy history
        exposed_x.data = x_data
        exposed_y.data = y.data

        # Squish y if fp8:
        if exposed_y.data.dtype == torch.int8:
            # Pytorch will break the computation graph
            # if it will see an output tensor of an integer type.
            # As fp8 tensors internally have dtype int8,
            # we need to pretend that this type is actually different.
            # This doesn't work in TorchScript, but this code
            # won't run at inference anyway.
            _squish(exposed_y)
            # Because the output is squished, the gradient also needs to be.
            # The backward of this forward recieves the gradient of the
            # output as its input. So, the backward before it needs
            # to squish it, while the backward coresponding to this
            # forward needs to unsquish it.
            setattr(ctx, "nvte_unsquish_incoming_dgrad", True)
            comm.is_exposed_x_squished_now = True
        else:
            setattr(ctx, "nvte_unsquish_incoming_dgrad", False)
            comm.is_exposed_x_squished_now = False

        return exposed_y

    @staticmethod
    def backward(ctx: FunctionCtx, grad_output: torch.Tensor):  # type: ignore[arg-type]
        # The context needs to think that the tensors were read
        _ = ctx.saved_tensors  # type: ignore

        # Get real context
        saved: Context = getattr(ctx, "nvte_ctx")
        op: Op = getattr(ctx, "nvte_op")

        # Make sure that the gradient is contiguous
        grad_output = grad_output.contiguous()  # TODO: use transposed gradient instead

        # Check if incoming gradient needs to be unsquished
        unsquish_incoming_dgrad: bool = getattr(ctx, "nvte_unsquish_incoming_dgrad")
        if unsquish_incoming_dgrad:
            _unsquish(grad_output)

        nvte.set_current_pass("backward")
        data_grad, param_grads = op.backward(saved, nvte.make_nvte_tensor(grad_output))

        # Check that gradients are not fp8 and can be processed by the optimizer
        # TODO: change this when fp8 optimizer comes along
        assert all(not nvte.is_fp8(g) for g in param_grads)

        # Check if outgoing gradient needs to be squished
        exposed_dgrad = data_grad.data
        squish_outgoing_dgrad: bool = getattr(ctx, "nvte_squish_outgoing_dgrad")
        if squish_outgoing_dgrad:
            _squish(exposed_dgrad)

        torch_grads = [exposed_dgrad] + [g.data for g in param_grads]

        return (*torch_grads, None, None)


def apply(x: torch.Tensor, pipeline: ComputePipeline, training: bool) -> torch.Tensor:
    if not training:
        nvte.set_current_pass("inference")
        y = pipeline.run_inference(nvte.make_nvte_tensor(x))
        assert not nvte.is_fp8(y)
        return y.data
    else:
        comm = CommWithLoop(nvte.make_nvte_tensor(x), False)
        for contained_op in pipeline.functions:
            nvte_tensors = contained_op.require_grad()
            exposed_tensors = list[torch.Tensor]()
            for nvte_tensor in nvte_tensors:
                assert not nvte.is_fp8(
                    nvte_tensor
                )  # TODO: change when fp8 optimizer comes along
                exposed_tensors.append(nvte_tensor.data)
            x = ComputePipelineFunction.apply(  # type: ignore
                x, *exposed_tensors, contained_op, comm
            )
        return x


# The squish needs to be invertible and
# always reduce the size of the tensor by the same
# amount.
#
# If a tensor is to be squished, it must have been
#   1. an fp8 result from forward
#   2. an outgoing gradient
#
# The outgoing gradient could have any type,
# but it is reasonable to assume that if someone is
# using fp8, they are also probably using bfloat16
# rather than float16.
#
# And they probably won't be using float64.
SQUISH_TABLE = {
    torch.int8: torch.float16,
    torch.bfloat16: torch.float32,
    torch.float32: torch.float64,
}
UNSQUISH_TABLE = {v: k for k, v in SQUISH_TABLE.items()}


def _unsquish(t: torch.Tensor):
    assert t.data.dtype in UNSQUISH_TABLE
    t.data = t.data.view(UNSQUISH_TABLE[t.data.dtype])


def _squish(t: torch.Tensor):
    if t.data.dtype in SQUISH_TABLE:
        t.data = t.data.view(SQUISH_TABLE[t.data.dtype])
    else:
        raise RuntimeError("Invalid dtype of gradient for FP8 tensor.")
