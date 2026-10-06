# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Isolate MoE weight layout errors using real packing and FSDP collectives.

The fallback tests replace FP8 quantization and grouped GEMM with lossless
wrappers and JAX matmul to isolate layout/axis handling from kernel accuracy.
Known failures are strict xfails; remove the marks when fixing the bugs.
"""

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P

moe = importlib.import_module("transformer_engine.jax.moe")


def fsdp_mesh():
    if len(jax.devices()) < 2:
        pytest.skip("Requires two JAX devices for a real FSDP all-gather")
    return Mesh(np.asarray(jax.devices()[:2]), ("fsdp",))


@pytest.mark.xfail(
    strict=True,
    reason="Packing contiguous gated-dimension shards changes global gate/up pairing",
    raises=AssertionError,
)
def test_standard_gated_shards_match_unsharded_swiglu():
    mesh = fsdp_mesh()
    # Each rank receives 64 contiguous columns: rank 0 gets only gates (1),
    # rank 1 only ups (3). Each local half still meets pack's 32-column rule.
    gate = jnp.ones((1, 32, 64), jnp.float32)
    up = 3 * gate
    wi = jnp.concatenate((gate, up), axis=-1)
    x = jnp.ones((1, 32), jnp.float32) / 32
    wo = jnp.ones((64, 1), jnp.float32) / 64

    def output(packed):
        global_gate, global_up = moe.tex.unpack_swiglu_pair(packed)
        return (jax.nn.silu(x @ global_gate[0]) * (x @ global_up[0])) @ wo

    expected = output(moe.tex.pack_swiglu_pair(gate, up))

    def local_forward(local_wi):
        local_gate, local_up = jnp.split(local_wi, 2, axis=-1)
        packed = moe.tex.pack_swiglu_pair(local_gate, local_up)
        gathered = jax.lax.all_gather(packed, "fsdp", axis=2, tiled=True)
        return output(gathered)

    actual = shard_map(
        local_forward,
        mesh=mesh,
        in_specs=P(None, None, "fsdp"),
        out_specs=P(),
        check_rep=False,
    )(wi)
    # expected=3*silu(1)=2.193176; actual=(silu(1)+3*silu(3))/2=4.652113.
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize("native_layout", [False, True])
def test_unfused_k_sharded_weight_matches_unsharded(monkeypatch, native_layout):
    mesh = fsdp_mesh()

    class LosslessTensor:
        def __init__(self, data):
            self.data = data

        def get_tensor(self, **kwargs):
            return self

        def checkpoint(self, quantizer):
            return self

    def quantize(data, *args, **kwargs):
        return LosslessTensor(data)

    def gather(tensor, fsdp_axis, fsdp_size, sharded_axis):
        assert fsdp_size == 2
        return LosslessTensor(
            jax.lax.all_gather(tensor.data, fsdp_axis, axis=sharded_axis, tiled=True)
        )

    def gemm(lhs, rhs, *, contracting_dims, bias):
        assert contracting_dims == ((1,), (1,))
        # One expert is sufficient to reproduce the matrix-axis error.
        return lhs.data @ rhs.data[0]

    monkeypatch.setattr(moe.tex, "grouped_quantize", quantize)
    monkeypatch.setattr(moe, "_gather_quantized_weight", gather)
    monkeypatch.setattr(moe.tex, "grouped_gemm", gemm)
    quantizers = SimpleNamespace(x=None, kernel=None)
    gate = jnp.ones((1, 64, 64), jnp.float32) / 64
    up = 2 * gate
    standard_wi = jnp.concatenate((gate, up), axis=-1)
    wi = moe.tex.pack_swiglu_pair(gate, up).transpose(0, 2, 1) if native_layout else standard_wi
    x = jnp.ones((1, 1, 64), jnp.float32)
    wo = jnp.ones((1, 64, 64), jnp.float32) / 64

    def forward(weight, quant_before_fsdp_ag):
        result, _ = moe._ffn_fwd_per_shard(
            x,
            jnp.ones((1, 1)),
            jnp.asarray([1]),
            weight,
            wo,
            None,
            None,
            None,
            (quantizers, quantizers),
            num_local_experts=1,
            activation_type="silu",
            apply_topk_weights_early=False,
            use_cudnn_jax_fusion=False,
            wi_0_checkpoint_name=None,
            wi_1_checkpoint_name=None,
            wo_checkpoint_name=None,
            cudnn_native_weight_layout=native_layout,
            quant_before_fsdp_ag=quant_before_fsdp_ag,
            fsdp_axis="fsdp",
            fsdp_size=2,
            wi_fsdp_axis=(2 if native_layout else 1),
            wo_fsdp_axis=None,
        )
        return result

    expected = forward(wi, False)
    independent_reference = (jax.nn.silu(x @ gate[0]) * (x @ up[0])) @ wo[0]
    np.testing.assert_allclose(expected, independent_reference, rtol=1e-6)
    actual = shard_map(
        lambda local_wi: forward(local_wi, True),
        mesh=mesh,
        in_specs=P(None, None, "fsdp") if native_layout else P(None, "fsdp", None),
        out_specs=P(),
        check_rep=False,
    )(wi)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize(
    "shape,flag,expected",
    [
        ((2, 128, 256), None, False),
        ((2, 256, 128), None, True),
        ((2, 128, 128), False, False),
        ((2, 128, 128), True, True),
        ((2, 128, 128), None, ValueError),
        ((2, 128, 256), True, ValueError),
        ((2, 256, 128), False, ValueError),
        ((2, 128, 256), "false", TypeError),
    ],
)
def test_layout_resolution(shape, flag, expected):
    wi = jnp.zeros(shape)
    if isinstance(expected, type):
        with pytest.raises(expected):
            moe._resolve_cudnn_native_weight_layout(wi, 128, flag)
    else:
        assert moe._resolve_cudnn_native_weight_layout(wi, 128, flag) is expected


def test_layout_argument_is_optional_and_static():
    import inspect

    assert inspect.signature(moe.moe).parameters["cudnn_native_weight_layout"].default is None
    assert inspect.signature(moe._moe).parameters["cudnn_native_weight_layout"].default is None
    assert 33 in moe._moe.nondiff_argnums
