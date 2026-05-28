# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Partitioning tests for grouped quantize and grouped GEMM."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from transformer_engine.jax.cpp_extensions.gemm import GroupedGemmPrimitive
from transformer_engine.jax.cpp_extensions.quantization import GroupedQuantizePrimitive
from transformer_engine.jax.dense import grouped_dense
from transformer_engine.jax.quantize import QuantizeLayout, QuantizerFactory, ScalingMode
from transformer_engine.jax.sharding import MeshResource, global_shard_guard


def _mesh():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("Grouped GEMM partitioning tests require at least 4 visible GPUs.")
    return Mesh(np.asarray(devices[:4]).reshape(2, 2), ("expert", "fsdp"))


def _arg_info(mesh, shape, spec):
    return SimpleNamespace(
        shape=shape,
        ndim=len(shape),
        size=int(np.prod(shape)),
        sharding=NamedSharding(mesh, PartitionSpec(*spec)),
    )


def _normalize_spec(spec):
    if isinstance(spec, PartitionSpec):
        return tuple(spec)
    return spec


def _mxfp8_grouped_quantizer_set(n_groups):
    return QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e4m3fn,
        is_2x2x=True,
        n_groups=n_groups,
    )


def test_grouped_quantize_specs_preserve_ep_and_fsdp_for_block_scales():
    mesh = _mesh()
    with global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        _, _, out_shardings, _ = GroupedQuantizePrimitive.partition(
            jnp.float8_e4m3fn,
            ScalingMode.MXFP8_1D_SCALING.value,
            QuantizeLayout.ROWWISE,
            -1,
            jnp.float8_e8m0fnu,
            mesh,
            (
                _arg_info(mesh, (8, 128, 64), ("expert", None, "fsdp")),
                _arg_info(mesh, (8,), ("expert",)),
                _arg_info(mesh, (8,), ("expert",)),
            ),
            (),
        )

    specs = tuple(tuple(sharding.spec) for sharding in out_shardings)
    assert _normalize_spec(specs[0]) == (("expert", "fsdp"),)
    assert _normalize_spec(specs[2]) == (("expert", "fsdp"),)
    assert _normalize_spec(specs[4]) == ("expert",)


def test_grouped_quantize_mxfp8_colwise_specs_preserve_ep_and_fsdp():
    mesh = _mesh()
    with global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        _, _, out_shardings, _ = GroupedQuantizePrimitive.partition(
            jnp.float8_e4m3fn,
            ScalingMode.MXFP8_1D_SCALING.value,
            QuantizeLayout.ROWWISE_COLWISE,
            -1,
            jnp.float8_e8m0fnu,
            mesh,
            (
                _arg_info(mesh, (8, 128, 128), ("expert", None, "fsdp")),
                _arg_info(mesh, (8,), ("expert",)),
                _arg_info(mesh, (8,), ("expert",)),
            ),
            (),
        )

    specs = tuple(tuple(sharding.spec) for sharding in out_shardings)
    assert _normalize_spec(specs[0]) == (("expert", "fsdp"),)
    assert _normalize_spec(specs[1]) == (("expert", "fsdp"),)
    assert _normalize_spec(specs[2]) == (("expert", "fsdp"),)
    assert _normalize_spec(specs[3]) == (("expert", "fsdp"),)
    assert _normalize_spec(specs[4]) == ("expert",)


def test_grouped_gemm_rhs_weight_specs_gather_fsdp_but_preserve_ep():
    mesh = _mesh()
    arg_infos = (
        _arg_info(mesh, (8192,), (None,)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (65536,), (("expert", "fsdp"),)),
        _arg_info(mesh, (2048,), (("expert", "fsdp"),)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (8,), ("expert",)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (8,), ("expert",)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (1,), (None,)),
        _arg_info(mesh, (0,), (None,)),
    )
    with global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        _, _, out_sharding, arg_shardings = GroupedGemmPrimitive.partition(
            False,
            False,
            ScalingMode.NO_SCALING.value,
            jnp.bfloat16,
            False,
            False,
            False,
            1,
            1,
            (1, 128, 64),
            128,
            64,
            128,
            64,
            mesh,
            arg_infos,
            (),
        )

    assert tuple(arg_shardings[2].spec) == ("expert",)
    assert tuple(arg_shardings[3].spec) == ("expert",)
    assert tuple(out_sharding[0].spec) == (None, None, None)


def test_grouped_partitioning_shardy_rules_smoke():
    mesh = _mesh()
    quantize_rule = GroupedQuantizePrimitive.shardy_sharding_rule(
        jnp.float8_e4m3fn,
        ScalingMode.MXFP8_1D_SCALING.value,
        QuantizeLayout.ROWWISE,
        -1,
        jnp.float8_e8m0fnu,
        mesh,
        (
            SimpleNamespace(shape=(8, 128, 64)),
            SimpleNamespace(shape=(8,)),
            SimpleNamespace(shape=(8,)),
        ),
        (
            SimpleNamespace(shape=(8 * 128 * 64,)),
            SimpleNamespace(shape=(1,)),
            SimpleNamespace(shape=(8 * 128 * 64,)),
            SimpleNamespace(shape=(1,)),
            SimpleNamespace(shape=(8,)),
        ),
    )
    gemm_rule = GroupedGemmPrimitive.shardy_sharding_rule(
        False,
        False,
        ScalingMode.NO_SCALING.value,
        jnp.bfloat16,
        False,
        False,
        False,
        1,
        2,
        (128, 64),
        128,
        64,
        128,
        64,
        mesh,
        tuple(SimpleNamespace(shape=(1,)) for _ in range(13)),
        (SimpleNamespace(shape=(128, 64)),),
    )

    assert quantize_rule is not None
    assert gemm_rule is not None


def test_grouped_dense_mxfp8_ep_fsdp_outside_shard_map_single_process():
    mesh = _mesh()
    n_groups = 4
    group_tokens = 128
    hidden = 256
    out_hidden = 128
    x_shape = (n_groups * group_tokens, hidden)
    w_shape = (n_groups, hidden, out_hidden)

    x_sharding = NamedSharding(mesh, PartitionSpec("expert", None))
    w_sharding = NamedSharding(mesh, PartitionSpec("expert", "fsdp", None))
    group_sharding = NamedSharding(mesh, PartitionSpec("expert"))
    out_sharding = NamedSharding(mesh, PartitionSpec("expert", None))

    quantizer_set = _mxfp8_grouped_quantizer_set(n_groups)

    with mesh, global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        x = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(20), x_shape, dtype=jnp.bfloat16)
            * jnp.asarray(0.01, dtype=jnp.bfloat16),
            x_sharding,
        )
        w = jax.device_put(
            jax.random.normal(jax.random.PRNGKey(21), w_shape, dtype=jnp.bfloat16)
            * jnp.asarray(0.01, dtype=jnp.bfloat16),
            w_sharding,
        )
        group_sizes = jax.device_put(
            jnp.full((n_groups,), group_tokens, dtype=jnp.int32),
            group_sharding,
        )

        def apply_with_vjp(x, w, group_sizes):
            def apply(x, w):
                return grouped_dense(
                    x,
                    w,
                    group_sizes,
                    contracting_dims=((1,), (1,)),
                    quantizer_set=quantizer_set,
                    kernel_fsdp_info=("fsdp", 1),
                )

            out, vjp_fn = jax.vjp(apply, x, w)
            dx, dw = vjp_fn(out)
            return out, dx, dw

        out, dx, dw = jax.jit(
            apply_with_vjp,
            in_shardings=(x_sharding, w_sharding, group_sharding),
            out_shardings=(out_sharding, x_sharding, w_sharding),
        )(x, w, group_sizes)
        out, dx, dw = jax.block_until_ready((out, dx, dw))

    assert tuple(out.sharding.spec) == ("expert", None)
    assert tuple(dx.sharding.spec) == ("expert", None)
    assert tuple(dw.sharding.spec) == ("expert", "fsdp", None)
    for value in (out, dx, dw):
        local_value = np.asarray(jax.device_get(value.addressable_data(0)))
        assert np.all(np.isfinite(local_value))
        assert np.any(local_value != 0.0)
