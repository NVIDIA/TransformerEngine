# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""LayerNormLinear DDP placeholders must match the original weight shard.

A single-GPU DistributedWeight stub supplies a full weight and GEMM gradient
buffer while retaining only a shard as the leaf parameter. This exercises TE's
autograd contract without depending on Megatron or distributed communication.
"""

import pytest
import torch

import transformer_engine.pytorch as te


class _ShardedWeight(torch.nn.Parameter):
    """Stand in for one shard owner, with local gradient finalization."""

    is_distributed_weight = True

    def __new__(cls, full_weight, shards):
        shard = full_weight[: full_weight.shape[0] // shards].clone()
        return super().__new__(cls, shard, requires_grad=True)

    def __init__(self, full_weight, shards):
        self.full_weight = full_weight.detach().clone()
        self.main_grad = torch.zeros(self.shape, dtype=torch.float32, device=self.device)
        self.scratch = torch.empty_like(self.full_weight, dtype=torch.float32)
        self.grad_added_to_main_grad = False
        self.calls = {"forward": 0, "backward": 0, "finalize": 0}

    def materialize_group_for_forward(self):
        self.calls["forward"] += 1
        return self.full_weight.detach().requires_grad_(True)

    def materialize_group_for_backward(self):
        self.calls["backward"] += 1
        return self.full_weight.detach().requires_grad_(True)

    def grad_buffer(self):
        return self.scratch

    def finalize_group_grads(self, gradient):
        self.calls["finalize"] += 1
        assert gradient.shape == self.full_weight.shape
        self.main_grad.add_(gradient[: self.shape[0]])
        # The real gradient lives in main_grad. TE may replace this placeholder
        # for DDP, but that replacement must still have the leaf parameter shape.
        return torch.zeros(self.shape, dtype=self.dtype, device=self.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("shards", [1, 2])
@pytest.mark.parametrize("zero_dummy", [False, True])
@pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
def test_fused_ddp_dummy_gradient_shape(shards, zero_dummy, normalization):
    """Check both DDP branches and real gradient accumulation over two microbatches."""
    torch.manual_seed(271828)
    module, reference = (
        te.LayerNormLinear(
            128,
            256,
            normalization=normalization,
            bias=False,
            fuse_wgrad_accumulation=True,
            params_dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(2)
    )
    reference.load_state_dict(module.state_dict())
    module.weight = _ShardedWeight(module.weight.detach(), shards)
    reference.weight.main_grad = torch.zeros_like(reference.weight, dtype=torch.float32)
    reference.weight.grad_added_to_main_grad = False
    for weight in (module.weight, reference.weight):
        weight.zero_out_wgrad = zero_dummy

    placeholder_shapes = []

    def observe_placeholder(gradient):
        placeholder_shapes.append(tuple(gradient.shape))
        if zero_dummy:
            assert torch.count_nonzero(gradient) == 0
        return gradient

    module.weight.register_hook(observe_placeholder)
    for microbatch in range(2):
        inputs = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
        actual_input = inputs.clone().requires_grad_(True)
        reference_input = inputs.clone().requires_grad_(True)
        module.weight.grad_added_to_main_grad = False
        actual_output = module(actual_input, is_first_microbatch=microbatch == 0)
        reference_output = reference(reference_input, is_first_microbatch=microbatch == 0)
        actual_output.float().square().mean().backward()
        reference_output.float().square().mean().backward()

        for actual, expected in (
            (actual_output, reference_output),
            (actual_input.grad, reference_input.grad),
            (module.weight.main_grad, reference.weight.main_grad[: module.weight.shape[0]]),
            (module.layer_norm_weight.grad, reference.layer_norm_weight.grad),
        ):
            torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-5, atol=1e-5)
        if normalization == "LayerNorm":
            torch.testing.assert_close(
                module.layer_norm_bias.grad.float(),
                reference.layer_norm_bias.grad.float(),
                rtol=1e-5,
                atol=1e-5,
            )
        assert module.weight.grad_added_to_main_grad
        assert module.weight.main_grad.shape == module.weight.shape
        assert module.weight.scratch.shape == reference.weight.shape
        assert placeholder_shapes[-1] == tuple(module.weight.shape)
        # DDP discards the placeholder after its hook; do not accumulate its
        # unspecified contents in .grad across microbatches.
        module.weight.grad = None
        reference.weight.grad = None

    assert module.weight.calls == {"forward": 2, "backward": 2, "finalize": 2}
    assert len(placeholder_shapes) == 2
