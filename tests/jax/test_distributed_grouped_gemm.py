# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Partitioning tests for grouped quantize and grouped GEMM."""

import math
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


def _mesh_with_dp_tp():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("Grouped GEMM partitioning tests require at least 4 visible GPUs.")
    return Mesh(np.asarray(devices[:4]).reshape(2, 1, 2, 1), ("expert", "dp", "fsdp", "tp"))


def _mesh_with_arbitrary_axis():
    devices = jax.devices()
    if len(devices) < 4:
        pytest.skip("Grouped GEMM partitioning tests require at least 4 visible GPUs.")
    return Mesh(
        np.asarray(devices[:4]).reshape(2, 1, 2, 1),
        ("expert", "dp", "fsdp", "myaxis123"),
    )


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


def _spec_contains_axis(spec, axis):
    for axis_spec in spec:
        axis_tuple = axis_spec if isinstance(axis_spec, tuple) else (axis_spec,)
        if axis in axis_tuple:
            return True
    return False


def _mxfp8_grouped_quantizer_set(n_groups):
    return QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e4m3fn,
        is_2x2x=True,
        n_groups=n_groups,
    )


def test_grouped_quantize_large_abstract_shape_preserves_inferred_hidden_dim():
    input_shape = (786432, 4096)
    group_count = 64
    outputs = GroupedQuantizePrimitive.abstract(
        jax.core.ShapedArray(input_shape, jnp.bfloat16),
        jax.core.ShapedArray((group_count,), jnp.float32),
        jax.core.ShapedArray((group_count,), jnp.int32),
        out_dtype=jnp.float8_e4m3fn,
        scaling_mode=ScalingMode.MXFP8_1D_SCALING.value,
        q_layout=QuantizeLayout.ROWWISE,
        flatten_axis=-1,
        scale_dtype=jnp.float8_e8m0fnu,
        uniform_groups=False,
    )

    assert outputs[0].shape == input_shape
    assert max(outputs[0].shape) < 2**31
    assert math.prod(outputs[0].shape) == 3221225472


def test_grouped_quantize_preserves_output_side_fsdp_for_uniform_kernel():
    mesh = _mesh()
    with global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        _, _, out_shardings, arg_shardings = GroupedQuantizePrimitive.partition(
            jnp.float8_e4m3fn,
            ScalingMode.MXFP8_1D_SCALING.value,
            QuantizeLayout.ROWWISE,
            -1,
            jnp.float8_e8m0fnu,
            True,
            mesh,
            (
                _arg_info(mesh, (8, 128, 256), ("expert", None, "fsdp")),
                _arg_info(mesh, (8,), ("expert",)),
                _arg_info(mesh, (8,), ("expert",)),
            ),
            (),
        )

    assert tuple(arg_shardings[0].spec) == ("expert", None, "fsdp")
    specs = tuple(tuple(sharding.spec) for sharding in out_shardings)
    assert _normalize_spec(specs[0]) == ("expert", None, "fsdp")
    assert _normalize_spec(specs[2]) == ("expert", None, "fsdp")
    assert _normalize_spec(specs[4]) == ("expert",)


def test_grouped_quantize_mxfp8_colwise_scale_tracks_output_side_fsdp():
    mesh = _mesh()
    with global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        _, _, out_shardings, arg_shardings = GroupedQuantizePrimitive.partition(
            jnp.float8_e4m3fn,
            ScalingMode.MXFP8_1D_SCALING.value,
            QuantizeLayout.ROWWISE_COLWISE,
            -1,
            jnp.float8_e8m0fnu,
            True,
            mesh,
            (
                _arg_info(mesh, (8, 128, 256), ("expert", None, "fsdp")),
                _arg_info(mesh, (8,), ("expert",)),
                _arg_info(mesh, (8,), ("expert",)),
            ),
            (),
        )

    assert tuple(arg_shardings[0].spec) == ("expert", None, "fsdp")
    specs = tuple(tuple(sharding.spec) for sharding in out_shardings)
    assert _normalize_spec(specs[0]) == ("expert", None, "fsdp")
    assert _normalize_spec(specs[1]) == ("expert", None, "fsdp")
    assert _normalize_spec(specs[2]) == ("expert", None, "fsdp")
    assert _normalize_spec(specs[3]) == ("expert", "fsdp", None)
    assert _normalize_spec(specs[4]) == ("expert",)


def test_grouped_quantize_preserves_row_side_fsdp_for_kernel():
    mesh = _mesh()
    with global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        _, _, out_shardings, arg_shardings = GroupedQuantizePrimitive.partition(
            jnp.float8_e4m3fn,
            ScalingMode.MXFP8_1D_SCALING.value,
            QuantizeLayout.ROWWISE,
            -1,
            jnp.float8_e8m0fnu,
            True,
            mesh,
            (
                _arg_info(mesh, (8, 256, 128), ("expert", "fsdp", None)),
                _arg_info(mesh, (8,), ("expert",)),
                _arg_info(mesh, (8,), ("expert",)),
            ),
            (),
        )

    assert tuple(arg_shardings[0].spec) == ("expert", "fsdp", None)
    specs = tuple(tuple(sharding.spec) for sharding in out_shardings)
    assert _normalize_spec(specs[0]) == ("expert", "fsdp", None)
    assert _normalize_spec(specs[2]) == ("expert", "fsdp", None)


def test_grouped_quantize_strips_unsupported_axes_and_preserves_supported_axes():
    mesh = _mesh_with_dp_tp()
    with jax.set_mesh(mesh), global_shard_guard(
        MeshResource(dp_resource="dp", tp_resource="tp", fsdp_resource="fsdp", ep_resource="expert")
    ):
        with pytest.warns(RuntimeWarning, match="Grouped quantize.*tp"):
            _, _, out_shardings, arg_shardings = GroupedQuantizePrimitive.partition(
                jnp.float8_e4m3fn,
                ScalingMode.MXFP8_1D_SCALING.value,
                QuantizeLayout.ROWWISE,
                -1,
                jnp.float8_e8m0fnu,
                True,
                mesh,
                (
                    _arg_info(mesh, (8, 128, 256), ("expert", "dp", ("fsdp", "tp"))),
                    _arg_info(mesh, (8,), (("expert", "tp"),)),
                    _arg_info(mesh, (8,), (("expert", "tp"),)),
                ),
                (),
            )

    assert tuple(arg_shardings[0].spec) == ("expert", "dp", "fsdp")
    assert tuple(arg_shardings[1].spec) == ("expert",)
    assert tuple(arg_shardings[2].spec) == ("expert",)

    out_specs = tuple(tuple(sharding.spec) for sharding in out_shardings)
    assert _normalize_spec(out_specs[0]) == ("expert", "dp", "fsdp")
    assert _normalize_spec(out_specs[2]) == ("expert", "dp", "fsdp")
    assert _normalize_spec(out_specs[4]) == ("expert",)
    for spec in (*out_specs, *(tuple(sharding.spec) for sharding in arg_shardings)):
        assert not _spec_contains_axis(spec, "tp")


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


def test_grouped_gemm_gathers_smaller_moe_rhs_across_fsdp_group_axis():
    """A global MoE RHS has E groups while token counts have dp * E groups."""
    mesh = _mesh()
    arg_infos = (
        _arg_info(mesh, (8192,), (None,)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (32, 128, 64), (("fsdp", "expert"), None, None)),
        _arg_info(mesh, (2048,), (("fsdp", "expert"),)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (64,), (("fsdp", "expert"),)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (64,), (("fsdp", "expert"),)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (1,), (None,)),
        _arg_info(mesh, (0,), (None,)),
    )
    with global_shard_guard(MeshResource(fsdp_resource="fsdp", ep_resource="expert")):
        _, _, _, arg_shardings = GroupedGemmPrimitive.partition(
            False,
            False,
            ScalingMode.NO_SCALING.value,
            jnp.bfloat16,
            False,
            False,
            False,
            1,
            1,
            (64, 128, 64),
            128,
            64,
            128,
            64,
            mesh,
            arg_infos,
            (),
        )

    assert tuple(arg_shardings[2].spec) == ("expert", None, None)
    assert tuple(arg_shardings[3].spec) == ("expert",)


def test_grouped_gemm_strips_unsupported_axes_preserves_dp_and_gathers_rhs_fsdp():
    mesh = _mesh_with_dp_tp()
    arg_infos = (
        _arg_info(mesh, (8192,), (("dp", "tp"),)),
        _arg_info(mesh, (0,), (("tp",),)),
        _arg_info(mesh, (65536,), (("expert", "fsdp", "tp"),)),
        _arg_info(mesh, (2048,), (("expert", "fsdp", "tp"),)),
        _arg_info(mesh, (0,), (("fsdp", "tp"),)),
        _arg_info(mesh, (8,), (("expert", "tp"),)),
        _arg_info(mesh, (0,), (("tp",),)),
        _arg_info(mesh, (0,), (("tp",),)),
        _arg_info(mesh, (0,), (("tp",),)),
        _arg_info(mesh, (8,), (("expert", "tp"),)),
        _arg_info(mesh, (0,), (("tp",),)),
        _arg_info(mesh, (1,), (("tp",),)),
        _arg_info(mesh, (0,), (("tp",),)),
    )
    result_infos = (_arg_info(mesh, (1, 128, 64), ("expert", "tp", None)),)
    with jax.set_mesh(mesh), global_shard_guard(
        MeshResource(dp_resource="dp", tp_resource="tp", fsdp_resource="fsdp", ep_resource="expert")
    ):
        with pytest.warns(RuntimeWarning, match="Grouped GEMM.*tp"):
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
                result_infos,
            )

    assert tuple(arg_shardings[0].spec) == ("dp",)
    assert tuple(arg_shardings[2].spec) == ("expert",)
    assert tuple(arg_shardings[3].spec) == ("expert",)
    assert tuple(arg_shardings[5].spec) == ("expert",)
    assert tuple(out_sharding[0].spec) == ("expert", None, None)
    for spec in (
        *(tuple(sharding.spec) for sharding in arg_shardings),
        tuple(out_sharding[0].spec),
    ):
        assert not _spec_contains_axis(spec, "tp")


def test_grouped_gemm_reduce_axis_skips_ep_and_uses_dp():
    mesh = _mesh_with_dp_tp()
    arg_infos = (
        _arg_info(mesh, (8192,), (("expert", "dp"),)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (8192,), (("expert", "dp"),)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (8,), ("expert",)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (8,), ("expert",)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (8,), ("expert",)),
        _arg_info(mesh, (0,), (None,)),
        _arg_info(mesh, (1,), (None,)),
        _arg_info(mesh, (0,), (None,)),
    )

    with jax.set_mesh(mesh), global_shard_guard(
        MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="expert")
    ):
        _, _, reduce_axis = GroupedGemmPrimitive._parse_partition_specs(
            mesh,
            arg_infos,
            (),
            out_shape=(1, 128, 64),
            lhs_is_trans=False,
            lhs_axis_boundary=1,
        )

    assert reduce_axis == "dp"


def test_grouped_partitioning_strips_arbitrary_unsupported_axis():
    mesh = _mesh_with_arbitrary_axis()
    mesh_resource = MeshResource(dp_resource="dp", fsdp_resource="fsdp", ep_resource="expert")

    with jax.set_mesh(mesh), global_shard_guard(mesh_resource):
        with pytest.warns(RuntimeWarning, match="Grouped quantize.*myaxis123"):
            _, _, quantize_out_shardings, quantize_arg_shardings = (
                GroupedQuantizePrimitive.partition(
                    jnp.float8_e4m3fn,
                    ScalingMode.MXFP8_1D_SCALING.value,
                    QuantizeLayout.ROWWISE,
                    -1,
                    jnp.float8_e8m0fnu,
                    True,
                    mesh,
                    (
                        _arg_info(mesh, (8, 128, 256), ("expert", "myaxis123", ("dp", "fsdp"))),
                        _arg_info(mesh, (8,), (("expert", "myaxis123"),)),
                        _arg_info(mesh, (8,), (("expert", "myaxis123"),)),
                    ),
                    (),
                )
            )

        gemm_arg_infos = (
            _arg_info(mesh, (8192,), (("dp", "myaxis123"),)),
            _arg_info(mesh, (0,), (("myaxis123",),)),
            _arg_info(mesh, (65536,), (("expert", "fsdp", "myaxis123"),)),
            _arg_info(mesh, (2048,), (("expert", "fsdp", "myaxis123"),)),
            _arg_info(mesh, (0,), (("fsdp", "myaxis123"),)),
            _arg_info(mesh, (8,), (("expert", "myaxis123"),)),
            _arg_info(mesh, (0,), (("myaxis123",),)),
            _arg_info(mesh, (0,), (("myaxis123",),)),
            _arg_info(mesh, (0,), (("myaxis123",),)),
            _arg_info(mesh, (8,), (("expert", "myaxis123"),)),
            _arg_info(mesh, (0,), (("myaxis123",),)),
            _arg_info(mesh, (1,), (("myaxis123",),)),
            _arg_info(mesh, (0,), (("myaxis123",),)),
        )
        gemm_result_infos = (_arg_info(mesh, (1, 128, 64), ("expert", "myaxis123", None)),)
        with pytest.warns(RuntimeWarning, match="Grouped GEMM.*myaxis123"):
            _, _, gemm_out_sharding, gemm_arg_shardings = GroupedGemmPrimitive.partition(
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
                gemm_arg_infos,
                gemm_result_infos,
            )

    assert tuple(quantize_arg_shardings[0].spec) == ("expert", None, ("dp", "fsdp"))
    assert tuple(quantize_arg_shardings[1].spec) == ("expert",)
    quantize_out_specs = tuple(tuple(sharding.spec) for sharding in quantize_out_shardings)
    assert _normalize_spec(quantize_out_specs[0]) == ("expert", None, ("dp", "fsdp"))
    assert _normalize_spec(quantize_out_specs[2]) == ("expert", None, ("dp", "fsdp"))

    assert tuple(gemm_arg_shardings[0].spec) == ("dp",)
    assert tuple(gemm_arg_shardings[2].spec) == ("expert",)
    assert tuple(gemm_arg_shardings[3].spec) == ("expert",)
    assert tuple(gemm_out_sharding[0].spec) == ("expert", None, None)

    all_specs = (
        *quantize_out_specs,
        *(tuple(sharding.spec) for sharding in quantize_arg_shardings),
        *(tuple(sharding.spec) for sharding in gemm_arg_shardings),
        tuple(gemm_out_sharding[0].spec),
    )
    for spec in all_specs:
        assert not _spec_contains_axis(spec, "myaxis123")


def test_grouped_partitioning_shardy_rules_smoke():
    mesh = _mesh()
    quantize_rule = GroupedQuantizePrimitive.shardy_sharding_rule(
        jnp.float8_e4m3fn,
        ScalingMode.MXFP8_1D_SCALING.value,
        QuantizeLayout.ROWWISE,
        -1,
        jnp.float8_e8m0fnu,
        True,
        mesh,
        (
            SimpleNamespace(shape=(8, 128, 128)),
            SimpleNamespace(shape=(8,)),
            SimpleNamespace(shape=(8,)),
        ),
        (
            SimpleNamespace(shape=(8, 128, 128)),
            SimpleNamespace(shape=(1,)),
            SimpleNamespace(shape=(8, 1, 512)),
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


@pytest.mark.parametrize(
    "weight_spec",
    [
        ("expert", "fsdp", None),
        ("expert", None, "fsdp"),
    ],
    ids=("contracting-fsdp", "output-fsdp"),
)
def test_grouped_dense_mxfp8_ep_fsdp_outside_shard_map_single_process(weight_spec):
    mesh = _mesh()
    n_groups = 4
    group_tokens = 128
    hidden = 256
    out_hidden = 256
    x_shape = (n_groups * group_tokens, hidden)
    w_shape = (n_groups, hidden, out_hidden)

    x_sharding = NamedSharding(mesh, PartitionSpec("expert", None))
    w_sharding = NamedSharding(mesh, PartitionSpec(*weight_spec))
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
    assert tuple(dw.sharding.spec) == weight_spec
    for value in (out, dx, dw):
        local_value = np.asarray(jax.device_get(value.addressable_data(0)))
        assert np.all(np.isfinite(local_value))
        assert np.any(local_value != 0.0)

    x_global = np.asarray(jax.device_get(x)).reshape(n_groups, group_tokens, hidden)
    w_global = np.asarray(jax.device_get(w))
    reference = np.einsum(
        "gth,gho->gto", x_global.astype(np.float32), w_global.astype(np.float32)
    ).reshape(x_shape[0], out_hidden)
    np.testing.assert_allclose(
        np.asarray(jax.device_get(out)).astype(np.float32),
        reference.astype(np.float32),
        atol=5e-3,
        rtol=5e-2,
    )
