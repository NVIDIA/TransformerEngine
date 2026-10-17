# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DistributedWeight dispatch in the fused grouped MLP (``ops.fused.GroupedMLP_CuTeGEMM*``).

``test_ops_grouped_linear_distributed_weight.py`` covers the unfused ``ops.GroupedLinear`` path.
The fused op has its own weight plumbing, so it is covered here with the same kind of in-repo fake:

  * A DistributedWeight whose shards are plain BF16 (e.g. a Megatron GTP expert under a per-module
    MXFP8 recipe, which creates no native-quantized params) gathers as BF16. The fused op must
    gather *before* quantizing in forward, and in backward must quantize the re-gathered BF16
    weight before the dgrad GEMMs read quantized storage.
  * A DistributedWeight whose shards are native-quantized gathers as quantized tensors, which
    must pass through unchanged (no re-quantization).

The per-module recipe is MXFP8 entered inside a global NVFP4 autocast, mirroring how Megatron
applies a ``--te-precision-config-file`` override around a fused expert MLP.

As in the unfused test, the fakes apply distinct power-of-two scales in the materialize hooks so
the gathered weights are observably the ones the GEMMs consumed. A power-of-two scale commutes
exactly through MXFP8 quantization (E8M0 block scales shift, E4M3 data is unchanged) and through
BF16 rounding, so the scale relations below hold bit-exactly. FC1 gets no forward scale because the
activation between the two GEMMs is nonlinear.
"""

from contextlib import nullcontext

import pytest
import torch

import transformer_engine.pytorch as te
from transformer_engine.pytorch import MXFP8Quantizer, MXFP8Tensor
from transformer_engine.pytorch.ops.fused.grouped_mlp import GroupedMLP_CuTeGEMMGLU
from transformer_engine.pytorch.ops._common import is_quantized_tensor

from utils import dtype_tols, make_recipe

mxfp8_available, reason_for_no_mxfp8 = te.is_mxfp8_available(return_reason=True)
nvfp4_available, reason_for_no_nvfp4 = te.is_nvfp4_available(return_reason=True)

# Distinct powers of two: each commutes exactly through quantization and BF16 rounding.
FC1_BWD_SCALE = 2.0
FC2_FWD_SCALE = 4.0
FC2_BWD_SCALE = 8.0


def _mxfp8_quantize(tensor: torch.Tensor) -> MXFP8Tensor:
    """Native-quantized stand-in: both usages, like a shard the module's own quantizer created."""
    quantizer = MXFP8Quantizer(fp8_dtype=te.DType.kFloat8E4M3, rowwise=True, columnwise=True)
    return quantizer(tensor)


class _FakeDistWeight(torch.nn.Parameter):
    """Single-process fake DistributedWeight shard of a routed-expert group.

    Every shard of the group is one of these (as with a real implementer such as GTP); only the
    leader (``weight0``) carries ``weight_list``/``_group`` and is called back by the op.
    ``materialize_group_for_forward`` / ``materialize_group_for_backward`` return the whole
    group's "gathered" weights scaled by ``fwd_scale`` / ``bwd_scale``: plain BF16 tensors (a
    BF16 shard all-gather) or MXFP8 tensors when ``gather_quantized`` (a native-quantized shard
    all-gather). ``finalize_group_grads`` models the main-grad contract: accumulate the wgrads
    into each shard's ``main_grad`` in-place and return dummies that the op discards.
    ``grad_buffer`` hands the wgrad GEMM a fresh full-shape buffer in the compute dtype.
    """

    is_distributed_weight = True
    # The fused op only takes the distributed path for a routed-expert group leader.
    is_routed_expert = True

    def _gathered(self, scale: float):
        gathered = [w.detach() * scale for w in self._group]
        if self.gather_quantized:
            gathered = [_mxfp8_quantize(w) for w in gathered]
        return gathered

    def materialize_group_for_forward(self):
        self.calls["fwd"] += 1
        return self._gathered(self.fwd_scale)

    def materialize_group_for_backward(self, **kwargs):
        self.calls["bwd"] += 1
        return self._gathered(self.bwd_scale)

    def finalize_group_grads(self, wgrads, **kwargs):
        self.calls["finalize"] += 1
        wl = list(wgrads) if isinstance(wgrads, (list, tuple)) else [wgrads]
        for w, g in zip(self._group, wl):
            w.main_grad.add_(g.to(w.main_grad.dtype))
            w.grad_added_to_main_grad = True
        return [torch.zeros_like(g) for g in wl]

    def grad_buffer(self):
        return torch.empty_like(self.data)


def _make_fake_dist_leader(op, num_gemms, *, fwd_scale, bwd_scale, gather_quantized):
    """Turn ``op.weight{i}`` into fake distributed shards; return the leader (``weight0``)."""
    shards = []
    for i in range(num_gemms):
        shard = _FakeDistWeight(getattr(op, f"weight{i}").data)
        shard.main_grad = torch.zeros_like(shard, dtype=torch.float32)
        shard.grad_added_to_main_grad = False  # DDP initializes this on every param
        setattr(op, f"weight{i}", shard)
        shards.append(shard)
    leader = shards[0]
    leader.calls = {"fwd": 0, "bwd": 0, "finalize": 0}
    leader.fwd_scale = fwd_scale
    leader.bwd_scale = bwd_scale
    leader.gather_quantized = gather_quantized
    leader._group = shards
    leader.weight_list = shards  # routed-expert group membership, asserted by the op
    return leader


def _skip_unless_fused_mxfp8_supported(device: str, dtype: torch.dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if not mxfp8_available:
        pytest.skip(reason_for_no_mxfp8)
    if not GroupedMLP_CuTeGEMMGLU.is_supported():
        pytest.skip("Fused grouped MLP (CuTeDSL) is not supported on this system")
    if dtype != torch.bfloat16:
        pytest.skip("Fused grouped MLP requires BF16 data")


def _make_grouped_mlp(group_size, hidden_size, split_alignment, glu_interleave_size, dtype, device):
    fc1 = te.ops.GroupedLinear(
        group_size,
        hidden_size,
        2 * hidden_size,
        bias=False,
        device=device,
        dtype=dtype,
        split_alignment=split_alignment,
    )
    fc2 = te.ops.GroupedLinear(
        group_size,
        hidden_size,
        hidden_size,
        bias=False,
        device=device,
        dtype=dtype,
        split_alignment=split_alignment,
    )
    module = te.ops.Sequential(
        fc1, te.ops.ScaledSwiGLU(glu_interleave_size=glu_interleave_size), fc2
    )
    return module, fc1, fc2


def _assert_fused(module) -> None:
    forward_ops = module._module_groups[0]._forward_ops
    assert len(forward_ops) == 1
    assert isinstance(forward_ops[0][0], GroupedMLP_CuTeGEMMGLU)


@pytest.mark.parametrize("outer_recipe", (None, "nvfp4"))
@pytest.mark.parametrize("gather_quantized", (False, True))
def test_fused_grouped_mlp_distributed_weight(
    outer_recipe,
    gather_quantized,
    *,
    group_size: int = 4,
    hidden_size: int = 256,
    split_alignment: int = 256,
    glu_interleave_size: int = 32,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Fused grouped MLP with distributed weights: gather-then-quantize in fwd and bwd.

    ``gather_quantized=False`` is the BF16-shard case (per-module MXFP8 without native-quantized
    params); ``gather_quantized=True`` is the native-quantized-shard case, whose gathered tensors
    must pass through. ``outer_recipe="nvfp4"`` wraps the per-module MXFP8 autocast in a global
    NVFP4 one, as Megatron does for an overridden expert MLP.
    """
    _skip_unless_fused_mxfp8_supported(device, dtype)
    if outer_recipe == "nvfp4" and not nvfp4_available:
        pytest.skip(reason_for_no_nvfp4)
    torch.manual_seed(0)

    split_sizes = torch.tensor(
        [split_alignment, 2 * split_alignment, split_alignment, 2 * split_alignment],
        dtype=torch.int,
        device=device,
    )
    assert split_sizes.numel() == group_size
    total_tokens = int(split_sizes.sum().item())

    module, fc1, fc2 = _make_grouped_mlp(
        group_size, hidden_size, split_alignment, glu_interleave_size, dtype, device
    )
    reference, ref_fc1, ref_fc2 = _make_grouped_mlp(
        group_size, hidden_size, split_alignment, glu_interleave_size, dtype, device
    )
    reference.load_state_dict(module.state_dict())

    fc1_leader = _make_fake_dist_leader(
        fc1, group_size, fwd_scale=1.0, bwd_scale=FC1_BWD_SCALE, gather_quantized=gather_quantized
    )
    fc2_leader = _make_fake_dist_leader(
        fc2,
        group_size,
        fwd_scale=FC2_FWD_SCALE,
        bwd_scale=FC2_BWD_SCALE,
        gather_quantized=gather_quantized,
    )

    x = torch.rand((total_tokens, hidden_size), dtype=dtype, device=device) - 0.5
    x.requires_grad_(True)
    ref_x = x.detach().clone().requires_grad_(True)
    probs = torch.rand((total_tokens,), dtype=dtype, device=device)
    dy = torch.rand((total_tokens, hidden_size), dtype=dtype, device=device) - 0.5

    mxfp8_recipe = make_recipe("mxfp8")
    outer = make_recipe(outer_recipe) if outer_recipe is not None else None

    def _run(mod, inp):
        outer_ctx = te.autocast(enabled=True, recipe=outer) if outer is not None else nullcontext()
        # Per-module MXFP8 override entered inside the global autocast (Megatron's pattern).
        with outer_ctx, te.autocast(enabled=True, recipe=mxfp8_recipe):
            y = mod(inp, split_sizes, probs, split_sizes)
        y.backward(dy)
        return y.detach()

    out = _run(module, x)
    ref_out = _run(reference, ref_x)

    # The CuTeDSL fused op was formed on both (not the unfused fallback).
    _assert_fused(module)
    _assert_fused(reference)

    # Every dispatch hook fired on both groups.
    for leader in (fc1_leader, fc2_leader):
        assert leader.calls["fwd"] > 0 and leader.calls["bwd"] > 0 and leader.calls["finalize"] > 0

    # fwd used the gathered FC2 weights (scale commutes exactly through quantization).
    torch.testing.assert_close(out, FC2_FWD_SCALE * ref_out, rtol=0, atol=0)
    # dgrad used the re-gathered FC2 and FC1 weights.
    torch.testing.assert_close(x.grad, FC1_BWD_SCALE * FC2_BWD_SCALE * ref_x.grad, rtol=0, atol=0)
    # wgrads were reduce-scattered into main_grad: FC2's does not depend on the weight scales,
    # FC1's sees the FC2-backward scale through the activation gradient.
    tols = dtype_tols(dtype)
    for i in range(group_size):
        fc2_w, ref_fc2_w = getattr(fc2, f"weight{i}"), getattr(ref_fc2, f"weight{i}")
        fc1_w, ref_fc1_w = getattr(fc1, f"weight{i}"), getattr(ref_fc1, f"weight{i}")
        torch.testing.assert_close(fc2_w.main_grad.to(dtype), ref_fc2_w.grad, **tols)
        torch.testing.assert_close(
            fc1_w.main_grad.to(dtype), FC2_BWD_SCALE * ref_fc1_w.grad, **tols
        )
        for w in (fc1_w, fc2_w):
            assert w.grad_added_to_main_grad is True
            assert w.grad is not None, "distributed weights receive a dummy .grad"


def test_quantize_unquantized_weights_passes_through_quantized(
    *,
    group_size: int = 2,
    hidden_size: int = 128,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """Already-quantized gathered weights are returned as-is; plain ones are quantized."""
    # Imported here so the end-to-end test above still collects on a tree without the helper.
    from transformer_engine.pytorch.ops.fused.grouped_mlp import _quantize_unquantized_weights

    _skip_unless_fused_mxfp8_supported(device, dtype)
    torch.manual_seed(0)
    module, fc1, fc2 = _make_grouped_mlp(group_size, hidden_size, 256, 32, dtype, device)
    split_sizes = torch.tensor([256] * group_size, dtype=torch.int, device=device)
    total_tokens = int(split_sizes.sum().item())
    x = torch.rand((total_tokens, hidden_size), dtype=dtype, device=device) - 0.5
    probs = torch.rand((total_tokens,), dtype=dtype, device=device)

    # A forward initializes the op's per-expert weight quantizers for the recipe.
    with te.autocast(enabled=True, recipe=make_recipe("mxfp8")):
        module(x, split_sizes, probs, split_sizes)
        pre_quantized = _mxfp8_quantize(fc2.weight0.detach())
        plain = fc2.weight1.detach()
        out = _quantize_unquantized_weights(
            [pre_quantized, plain], fc2, rowwise=True, columnwise=True
        )

    assert out[0] is pre_quantized, "quantized gathered weight must pass through unchanged"
    assert is_quantized_tensor(out[1]), "BF16 gathered weight must be quantized"
    torch.testing.assert_close(
        out[1]._rowwise_data, _mxfp8_quantize(plain)._rowwise_data, rtol=0, atol=0
    )
