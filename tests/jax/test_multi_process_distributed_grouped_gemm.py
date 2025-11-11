# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from functools import partial

import jax
import jax.numpy as jnp
import jax.experimental.multihost_utils as jem

from transformer_engine.jax.dense import grouped_dense as te_grouped_dense
from transformer_engine.jax.quantize import (
    QuantizerFactory,
    ScalingMode,
)

from utils import assert_allclose, dtype_tols


N_GROUP = 8
MESH_AXIS_NAME = "fsdp"


def test_grouped_gemm_fp8_allgather(data_shapes, kernel_fsdp_axis):
    assert kernel_fsdp_axis in [1, 2]
    x_shape, w_shape = data_shapes

    x_sharding = NamedSharding(mesh, PartitionSpec(None, MESH_AXIS_NAME, None, None, None))
    w_sharding = (
        NamedSharding(mesh, PartitionSpec(None, None, MESH_AXIS_NAME))
        if kernel_fsdp_axis == 2
        else NamedSharding(mesh, PartitionSpec(None, MESH_AXIS_NAME, None))
    )
    w_no_sharding = NamedSharding(mesh, PartitionSpec(None, None, None))

    def init_data():
        x_key = jax.random.PRNGKey(0)
        w_key = jax.random.PRNGKey(1)
        x = jax.random.normal(x_key, shape=(N_GROUP, *x_shape), dtype=jnp.bfloat16)
        w = jax.random.normal(w_key, shape=(N_GROUP, *w_shape), dtype=jnp.bfloat16)
        w_amax = jnp.max(jnp.abs(w), axis=range(1, w.ndim))
        return x, w, w, w_amax

    def test_func(outter_x, outter_w, outter_w_amax):
        in_specs = (x_sharding.spec, w_sharding.spec, None)
        out_specs = x_sharding.spec

        @partial(
            shard_map.shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
        def sharded_group_gemm(x, w, w_amax):
            group_size = x.shape[0]
            x_reshaped = x.reshape(-1, x.shape[-1])
            n_groups = jnp.full(group_size, x_reshaped.shape[0] // group_size)

            quantizer_set = QuantizerFactory.create_set(
                scaling_mode=ScalingMode.CURRENT_TENSOR_SCALING,
                fwd_dtype=jnp.float8_e4m3fn,
                bwd_dtype=jnp.float8_e5m2,
                is_2x2x=True,
                n_groups=group_size,
            )

            output = te_grouped_dense(
                x_reshaped,
                w,
                n_groups,
                kernel_amax=w_amax,
                quantizer_set=quantizer_set,
                kernel_fsdp_info=(MESH_AXIS_NAME, kernel_fsdp_axis),
            )
            output = output.reshape(*x.shape[:-1], -1)
            return output

        def run(x, w, w_amax):
            output = sharded_group_gemm(x, w, w_amax)
            return output

        output, vjp_fn = jax.vjp(run, outter_x, outter_w, outter_w_amax)
        dx, dw, _ = vjp_fn(output)
        return output, dx, dw

    def ref_func(outter_x, outter_w):

        in_specs = (x_sharding.spec, w_no_sharding.spec)
        out_specs = x_sharding.spec

        @partial(
            shard_map.shard_map,
            mesh=mesh,
            in_specs=in_specs,
            out_specs=out_specs,
            check_rep=False,
        )
        def sharded_group_gemm(x, w):
            group_size = x.shape[0]
            x_reshaped = x.reshape(-1, x.shape[-1])
            n_groups = jnp.full(group_size, x_reshaped.shape[0] // group_size)

            quantizer_set = QuantizerFactory.create_set(
                scaling_mode=ScalingMode.CURRENT_TENSOR_SCALING,
                fwd_dtype=jnp.float8_e4m3fn,
                bwd_dtype=jnp.float8_e5m2,
                is_2x2x=True,
                n_groups=group_size,
            )
            output = te_grouped_dense(x_reshaped, w, n_groups, quantizer_set=quantizer_set)
            output = output.reshape(*x.shape[:-1], -1)
            return output

        def run(x, w):
            output = sharded_group_gemm(x, w)
            return output

        output, vjp_fn = jax.vjp(run, outter_x, outter_w)
        dx, dw = vjp_fn(output)
        return output, dx, dw

    init_func = jax.jit(init_data, out_shardings=(x_sharding, w_sharding, w_no_sharding, None))
    x, w, w_global, w_amax = init_func()

    o_sharding = x_sharding
    test_func_jitted = jax.jit(
        test_func,
        in_shardings=(x_sharding, w_sharding, None),
        out_shardings=(o_sharding, x_sharding, w_sharding),
    )
    ref_func_jitted = jax.jit(
        ref_func,
        in_shardings=(x_sharding, w_no_sharding),
        out_shardings=(o_sharding, x_sharding, w_no_sharding),
    )

    out, dx, dw = test_func_jitted(x, w, w_amax)
    ref_out, ref_dx, ref_dw = ref_func_jitted(x, w_global)

    e4m3_tols = dtype_tols(jnp.float8_e4m3fn)
    e5m2_tols = dtype_tols(jnp.float8_e5m2)

    out, ref_out = jem.process_allgather((out, ref_out))
    dx, ref_dx = jem.process_allgather((dx, ref_dx))
    dw, ref_dw = jem.process_allgather((dw, ref_dw))

    jnp.allclose(out, ref_out, **e4m3_tols)
    jnp.allclose(dx, ref_dx, **e5m2_tols)
    jnp.allclose(dw, ref_dw, **e5m2_tols)


if __name__ == "__main__":
    from jax.sharding import NamedSharding, PartitionSpec
    from jax.experimental import shard_map
    import sys

    coord_addr = sys.argv[1]
    proc_id = int(sys.argv[2])
    num_procs = int(sys.argv[3])

    jax.distributed.initialize(
        coordinator_address=coord_addr, num_processes=num_procs, process_id=proc_id
    )

    mesh = jax.make_mesh((num_procs,), (MESH_AXIS_NAME,))

    with mesh:
        data_shapes = [((4, 16, 128, 7168), (7168, 2048))]
        for data_shape in data_shapes:
            for kernel_fsdp_axis in [1, 2]:
                test_grouped_gemm_fp8_allgather(data_shape, kernel_fsdp_axis)
