# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as jem
import numpy as np
from jax.experimental import shard_map
from jax.sharding import NamedSharding, PartitionSpec

from transformer_engine.jax.dense import grouped_dense as te_grouped_dense
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    ScalingMode,
)
from transformer_engine.jax.sharding import MeshResource, global_shard_guard

from utils import assert_allclose, dtype_tols


N_GROUP = 8
EP_AXIS_NAME = "ep"
FSDP_AXIS_NAME = "fsdp"
MESH_AXIS_NAME = FSDP_AXIS_NAME


def _mxfp8_grouped_quantizer_set(n_groups):
    return QuantizerFactory.create_set(
        scaling_mode=ScalingMode.MXFP8_1D_SCALING,
        fwd_dtype=jnp.float8_e4m3fn,
        bwd_dtype=jnp.float8_e4m3fn,
        is_2x2x=True,
        n_groups=n_groups,
    )


def test_grouped_gemm_fp8_allgather(data_shapes, kernel_fsdp_axis):
    assert kernel_fsdp_axis in [1, 2]
    x_shape, w_shape = data_shapes

    x_sharding = NamedSharding(mesh, PartitionSpec(None, MESH_AXIS_NAME, None, None, None))
    w_sharding = (
        NamedSharding(mesh, PartitionSpec(None, None, MESH_AXIS_NAME))
        if kernel_fsdp_axis == 2
        else NamedSharding(mesh, PartitionSpec(None, MESH_AXIS_NAME, None))
    )
    b_sharding = (
        NamedSharding(mesh, PartitionSpec(None, MESH_AXIS_NAME))
        if kernel_fsdp_axis == 2
        else NamedSharding(mesh, PartitionSpec(None, None))
    )
    w_no_sharding = NamedSharding(mesh, PartitionSpec(None, None, None))
    b_no_sharding = NamedSharding(mesh, PartitionSpec(None, None))

    def init_data():
        x_key = jax.random.PRNGKey(0)
        w_key = jax.random.PRNGKey(1)
        b_key = jax.random.PRNGKey(2)
        x = (
            jax.random.normal(x_key, shape=(N_GROUP, *x_shape), dtype=jnp.bfloat16)
            * jnp.asarray(0.01, dtype=jnp.bfloat16)
        )
        w = (
            jax.random.normal(w_key, shape=(N_GROUP, *w_shape), dtype=jnp.bfloat16)
            * jnp.asarray(0.01, dtype=jnp.bfloat16)
        )
        b = (
            jax.random.normal(b_key, shape=(N_GROUP, w_shape[-1]), dtype=jnp.bfloat16)
            * jnp.asarray(0.01, dtype=jnp.bfloat16)
        )
        return x, w, w, b, b

    def test_func(outter_x, outter_w, outter_b):
        in_specs = (x_sharding.spec, w_sharding.spec, b_sharding.spec)
        out_specs = x_sharding.spec

        @partial(
            shard_map.shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
        def sharded_group_gemm(x, w, b):
            group_size = x.shape[0]
            x_reshaped = x.reshape(-1, x.shape[-1])
            n_groups = jnp.full(group_size, x_reshaped.shape[0] // group_size)

            quantizer_set = _mxfp8_grouped_quantizer_set(group_size)

            output = te_grouped_dense(
                x_reshaped,
                w,
                n_groups,
                bias=b,
                quantizer_set=quantizer_set,
            )
            output = output.reshape(*x.shape[:-1], -1)
            return output

        def run(x, w, b):
            output = sharded_group_gemm(x, w, b)
            return output

        output, vjp_fn = jax.vjp(run, outter_x, outter_w, outter_b)
        dx, dw, db = vjp_fn(output)
        return output, dx, dw, db

    def ref_func(outter_x, outter_w, outter_b):

        in_specs = (x_sharding.spec, w_no_sharding.spec, b_no_sharding.spec)
        out_specs = x_sharding.spec

        @partial(
            shard_map.shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
        def sharded_group_gemm(x, w, b):
            group_size = x.shape[0]
            x_reshaped = x.reshape(-1, x.shape[-1])
            n_groups = jnp.full(group_size, x_reshaped.shape[0] // group_size)

            quantizer_set = _mxfp8_grouped_quantizer_set(group_size)
            output = te_grouped_dense(
                x_reshaped,
                w,
                n_groups,
                bias=b,
                quantizer_set=quantizer_set,
            )
            output = output.reshape(*x.shape[:-1], -1)
            return output

        def run(x, w, b):
            output = sharded_group_gemm(x, w, b)
            return output

        output, vjp_fn = jax.vjp(run, outter_x, outter_w, outter_b)
        dx, dw, db = vjp_fn(output)
        return output, dx, dw, db

    init_func = jax.jit(
        init_data,
        out_shardings=(x_sharding, w_sharding, w_no_sharding, b_sharding, b_no_sharding),
    )
    x, w, w_global, b, b_global = init_func()

    o_sharding = x_sharding
    test_func_jitted = jax.jit(
        test_func,
        in_shardings=(x_sharding, w_sharding, b_sharding),
        out_shardings=(o_sharding, x_sharding, w_sharding, b_sharding),
    )
    ref_func_jitted = jax.jit(
        ref_func,
        in_shardings=(x_sharding, w_no_sharding, b_no_sharding),
        out_shardings=(o_sharding, x_sharding, w_no_sharding, b_no_sharding),
    )

    out, dx, dw, db = test_func_jitted(x, w, b)
    ref_out, ref_dx, ref_dw, ref_db = ref_func_jitted(x, w_global, b_global)

    # Avoid creating a host scalar JAX array under the multi-process mesh in dtype_tols.
    e4m3_tols = dtype_tols(jnp.float8_e4m3fn, rtol=0.25, atol=0.25)

    out, ref_out = jem.process_allgather((out, ref_out), tiled=True)
    dx, ref_dx = jem.process_allgather((dx, ref_dx), tiled=True)
    dw, ref_dw = jem.process_allgather((dw, ref_dw), tiled=True)
    db, ref_db = jem.process_allgather((db, ref_db), tiled=True)

    assert_allclose(out, ref_out, **e4m3_tols)
    assert_allclose(dx, ref_dx, **e4m3_tols)
    assert_allclose(dw, ref_dw, **e4m3_tols)
    assert_allclose(db, ref_db, **e4m3_tols)


def run_grouped_dense_mxfp8_ep_fsdp_outside_shard_map():
    n_groups = 4
    group_tokens = 128
    hidden = 256
    out_hidden = 128
    x_shape = (n_groups * group_tokens, hidden)
    w_shape = (n_groups, hidden, out_hidden)
    quantizer_set = _mxfp8_grouped_quantizer_set(n_groups)

    x_sharding = NamedSharding(mesh, PartitionSpec(EP_AXIS_NAME, None))
    w_sharding = NamedSharding(mesh, PartitionSpec(EP_AXIS_NAME, FSDP_AXIS_NAME, None))
    group_sharding = NamedSharding(mesh, PartitionSpec(EP_AXIS_NAME))
    out_sharding = NamedSharding(mesh, PartitionSpec(EP_AXIS_NAME, None))

    with mesh, global_shard_guard(
        MeshResource(ep_resource=EP_AXIS_NAME, fsdp_resource=FSDP_AXIS_NAME)
    ):
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
                return te_grouped_dense(
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

    assert tuple(out.sharding.spec) == (EP_AXIS_NAME, None)
    assert tuple(dx.sharding.spec) == (EP_AXIS_NAME, None)
    assert tuple(dw.sharding.spec) == (EP_AXIS_NAME, FSDP_AXIS_NAME, None)
    for value in (out, dx, dw):
        local_value = np.asarray(jax.device_get(value.addressable_data(0)))
        assert np.all(np.isfinite(local_value))
        assert np.any(local_value != 0.0)


if __name__ == "__main__":
    import sys

    coord_addr = sys.argv[1]
    proc_id = int(sys.argv[2])
    num_procs = int(sys.argv[3])

    jax.distributed.initialize(
        coordinator_address=coord_addr, num_processes=num_procs, process_id=proc_id
    )

    mesh = jax.make_mesh((num_procs,), (FSDP_AXIS_NAME,))

    with mesh:
        data_shapes = [((4, 16, 128, 7168), (7168, 2048))]
        for data_shape in data_shapes:
            for kernel_fsdp_axis in [1, 2]:
                test_grouped_gemm_fp8_allgather(data_shape, kernel_fsdp_axis)

    if num_procs == 4:
        mesh = jax.make_mesh((2, 2), (EP_AXIS_NAME, FSDP_AXIS_NAME))
        run_grouped_dense_mxfp8_ep_fsdp_outside_shard_map()
