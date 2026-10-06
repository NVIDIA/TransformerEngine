# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Isolate MoE weight layout errors using real packing and FSDP collectives.

The fallback tests replace FP8 quantization and grouped GEMM with lossless
wrappers and JAX matmul to isolate layout/axis handling from kernel accuracy.
Weight preparation tests use real MXFP8 quantization, scales, and all-gathers.
"""

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from transformer_engine.jax.quantize import QuantizerFactory, QuantizeLayout, ScalingMode
from transformer_engine.jax.quantize.dequantizer import Dequantizer

moe = importlib.import_module("transformer_engine.jax.moe")


def fsdp_mesh():
    if len(jax.devices()) < 2:
        pytest.skip("Requires two JAX devices for a real FSDP all-gather")
    return Mesh(np.asarray(jax.devices()[:2]), ("fsdp",))


def test_standard_gated_shards_match_unsharded_swiglu():
    mesh = fsdp_mesh()
    # Each rank receives 128 contiguous columns: rank 0 gets only gates (1),
    # rank 1 only ups (3). Each local half still meets pack's 32-column rule.
    gate = jnp.ones((1, 128, 128), jnp.bfloat16)
    up = 3 * gate
    wi = jnp.concatenate((gate, up), axis=-1)
    x = jnp.ones((1, 128), jnp.float32) / 128
    wo = jnp.ones((128, 1), jnp.float32) / 128
    quantizer = weight_quantizer(1)

    def output(tensor):
        packed = jnp.concatenate(tensor.rowwise_tensor.dequantize())
        global_gate, global_up = moe.tex.unpack_swiglu_pair(packed)
        return (jax.nn.silu(x @ global_gate[0]) * (x @ global_up[0])) @ wo

    expected = output(prepare_weight(wi, quantizer, fused=True, native=False))

    def local_forward(local_wi):
        return prepare_weight(local_wi, quantizer, fused=True, native=False, axis=2)

    gathered = shard_map(
        local_forward,
        mesh=mesh,
        in_specs=P(None, None, "fsdp"),
        out_specs=P(),
        check_rep=False,
    )(wi)
    actual = output(gathered)
    # Both paths now yield 3*silu(1)=2.193176, rather than FSDP yielding 4.652113.
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def weight_quantizer(experts):
    return QuantizerFactory.create(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        q_dtype=jnp.float8_e4m3fn,
        q_layout=QuantizeLayout.ROWWISE_COLWISE,
        n_groups=experts,
    )


def prepare_weight(wi, quantizer, *, fused, native, axis=None):
    return moe._prepare_fc1_weight(
        wi,
        quantizer,
        fused=fused,
        native_layout=native,
        quant_before_fsdp_ag=axis is not None,
        fsdp_axis="fsdp",
        fsdp_size=2,
        sharded_axis=axis,
    )


def plain_weight_scales(tensor):
    """Ignore unused allocation padding while checking every active scale byte."""
    kwargs = dict(
        data_layout=tensor.data_layout,
        is_colwise=tensor.is_colwise,
        flatten_axis=tensor.flatten_axis - 1,
    )
    padded = tensor.scaling_mode.get_scale_shape(
        tensor.original_shape[1:], is_padded=True, **kwargs
    )
    unpadded = tensor.scaling_mode.get_scale_shape(
        tensor.original_shape[1:], is_padded=False, **kwargs
    )
    scale_size = int(np.prod(padded))
    return jnp.stack(
        [
            moe._unswizzle_mxfp8_grouped_scale(
                tensor.scale_inv[e * scale_size : (e + 1) * scale_size], padded, tensor.is_colwise
            )[: unpadded[0], : unpadded[1]]
            for e in range(tensor.original_shape[0])
        ]
    ).view(jnp.uint8)


def reference_quantize_weight(data, *args, **kwargs):
    """Pure-JAX MXFP8 encoding for positive powers of two, including small shards.

    This avoids the legacy grouped quantizer's host-copy path for dimensions
    below 128. The production gather, FP8 permutation, and swizzle still run.
    """
    copies = []
    mode = ScalingMode.MXFP8_1D_SCALING
    experts, rows, columns = data.shape
    for colwise in (False, True):
        blocks = (
            data.reshape(experts, rows // 32, 32, columns)
            if colwise
            else data.reshape(experts, rows, columns // 32, 32)
        )
        maxima = jnp.max(blocks.astype(jnp.float32), axis=2 if colwise else 3)
        exponent = jnp.log2(maxima).astype(jnp.int32) - 8
        inverse_scale = jnp.exp2(exponent.astype(jnp.float32))
        broadcast_scale = jnp.repeat(inverse_scale, 32, axis=1 if colwise else 2)
        quantized = (data / broadcast_scale).astype(jnp.float8_e4m3fn)
        scales = (exponent + 127).astype(jnp.uint8)
        padded = mode.get_scale_shape((rows, columns), is_colwise=colwise, is_padded=True)
        swizzled = [
            moe.swizzled_scale(
                jnp.pad(scale, ((0, padded[0] - scale.shape[0]), (0, padded[1] - scale.shape[1]))),
                1,
                colwise,
            ).reshape(-1)
            for scale in scales
        ]
        scale_inv = jnp.concatenate(swizzled)
        scale_size = mode.get_grouped_scale_shape(data.shape, experts, colwise, flatten_axis=2)[0]
        scale_inv = jnp.pad(scale_inv, (0, scale_size - scale_inv.size))
        copies.append(
            moe.GroupedScaledTensor1x(
                data=quantized.reshape(-1),
                scale_inv=scale_inv,
                amax=jnp.empty((0,), jnp.float32),
                first_dims=None,
                last_dims=None,
                scaling_mode=mode,
                dq_dtype=data.dtype,
                _dq_func=Dequantizer.grouped_dequantize,
                is_colwise=colwise,
                data_layout="N",
                flatten_axis=2,
                original_shape=data.shape,
                pre_swizzled=True,
            )
        )
    return moe.ScaledTensor2x(*copies)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("width", [64, 192])
def test_small_gated_shards_preserve_quantized_pairs(monkeypatch, native, width):
    mesh = fsdp_mesh()
    source = jnp.exp2(
        (jnp.arange(64) // 32)[None, :, None]
        + (jnp.arange(width) // 32)[None, None, :]
        + jnp.arange(2)[:, None, None]
    ).astype(jnp.bfloat16)
    gate, up = jnp.split(source, 2, axis=-1)
    wi = moe.tex.pack_swiglu_pair(gate, up).transpose(0, 2, 1) if native else source
    monkeypatch.setattr(moe.tex, "grouped_quantize", reference_quantize_weight)
    expected = prepare_weight(wi, None, fused=not native, native=native)
    actual = shard_map(
        lambda shard: prepare_weight(
            shard, None, fused=not native, native=native, axis=1 if native else 2
        ),
        mesh=mesh,
        in_specs=P(None, "fsdp", None) if native else P(None, None, "fsdp"),
        out_specs=P(),
        check_rep=False,
    )(wi)
    for actual_copy, expected_copy in zip(
        (actual.rowwise_tensor, actual.colwise_tensor),
        (expected.rowwise_tensor, expected.colwise_tensor),
    ):
        np.testing.assert_array_equal(
            actual_copy.data.view(jnp.uint8), expected_copy.data.view(jnp.uint8)
        )
        np.testing.assert_array_equal(
            plain_weight_scales(actual_copy), plain_weight_scales(expected_copy)
        )
        decoded = jnp.concatenate(actual_copy.dequantize())
        execution_weight = moe.tex.pack_swiglu_pair(gate, up) if not native else source
        np.testing.assert_array_equal(decoded, execution_weight)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("shard_dimension", ["expert", "k", "gated"])
@pytest.mark.parametrize("width", [256, 768])
def test_quantized_fc1_gather_matches_unsharded(monkeypatch, native, fused, shard_dimension, width):
    mesh = fsdp_mesh()
    # Distinct column-block magnitudes, row magnitudes, and expert magnitudes
    # expose scale permutations that equal-valued/constant-scale inputs hide.
    source = jax.random.normal(jax.random.PRNGKey(9), (2, 256, width))
    column_magnitude = jnp.exp2(jnp.arange(width) // 32 % 7 - 3)
    row_magnitude = jnp.exp2(jnp.arange(256) // 32 - 3)
    source = (
        source
        * column_magnitude
        * row_magnitude[None, :, None]
        * jnp.asarray([1.0, 4.0])[:, None, None]
    ).astype(jnp.bfloat16)
    gate, up = jnp.split(source, 2, axis=-1)
    wi = moe.tex.pack_swiglu_pair(gate, up).transpose(0, 2, 1) if native else source
    axis = {"expert": 0, "k": 2 if native else 1, "gated": 1 if native else 2}[shard_dimension]
    quantizer = weight_quantizer(2)
    expected = prepare_weight(wi, quantizer, fused=fused, native=native)
    spec = [None, None, None]
    spec[axis] = "fsdp"
    quantized_shapes = []
    original_quantize = moe.tex.grouped_quantize

    def record_quantize(data, *args, **kwargs):
        quantized_shapes.append(data.shape)
        return original_quantize(data, *args, **kwargs)

    monkeypatch.setattr(moe.tex, "grouped_quantize", record_quantize)
    actual = shard_map(
        lambda shard: prepare_weight(shard, quantizer, fused=fused, native=native, axis=axis),
        mesh=mesh,
        in_specs=P(*spec),
        out_specs=P(),
        check_rep=False,
    )(wi)
    # Only a local shard is quantized, once; no global BF16 requantization.
    assert quantized_shapes
    assert all(int(np.prod(shape)) == wi.size // 2 for shape in quantized_shapes)
    for actual_copy, expected_copy in zip(
        (actual.rowwise_tensor, actual.colwise_tensor),
        (expected.rowwise_tensor, expected.colwise_tensor),
    ):
        np.testing.assert_array_equal(
            actual_copy.data.view(jnp.uint8), expected_copy.data.view(jnp.uint8)
        )
        np.testing.assert_array_equal(
            plain_weight_scales(actual_copy), plain_weight_scales(expected_copy)
        )
        np.testing.assert_array_equal(
            jnp.concatenate(actual_copy.dequantize()), jnp.concatenate(expected_copy.dequantize())
        )


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
