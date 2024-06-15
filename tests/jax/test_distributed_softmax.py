# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import warnings
import pytest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from distributed_test_base import generate_configs, generate_collectives_count
from distributed_test_base import compare_ops
from utils import make_causal_mask, make_self_mask
from transformer_engine.jax import fp8_autocast
from transformer_engine.jax.softmax import SoftmaxType, softmax

DTYPES = [jnp.float16, jnp.bfloat16]


class TestDistributedSoftmax:

    def generate_collectives_count_ref(self):
        # for loss
        all_reduce_loss_bytes = 4  # 1 * FP32
        return generate_collectives_count(allreduce=all_reduce_loss_bytes, allgather=0, other=0)

    def generate_inputs(self, shape, mesh_resource, softmax_type, dtype, bad_sharding):
        batch, _, sqelen, _ = shape

        x = random.normal(random.PRNGKey(1124), shape, dtype=dtype)
        if softmax_type == SoftmaxType.SCALED_UPPER_TRIANG_MASKED:
            mask = make_causal_mask(batch, sqelen)
        else:
            mask = make_self_mask(batch, sqelen)

        if not bad_sharding:
            x_pspec = PartitionSpec(
                mesh_resource.dp_resource, mesh_resource.tp_resource, None, None
            )
        else:
            x_pspec = PartitionSpec(
                mesh_resource.dp_resource, None, None, mesh_resource.tp_resource
            )
        mask_pspec = PartitionSpec(mesh_resource.dp_resource, None, None, None)

        return (x, mask), (x_pspec, mask_pspec)

    @staticmethod
    def target_func(x, mask, scale_factor=1.0, softmax_type=SoftmaxType.SCALED):
        return jnp.mean(softmax(x, mask, scale_factor=scale_factor, softmax_type=softmax_type))

    @staticmethod
    def ref_func(x, mask, scale_factor=1.0, dtype=jnp.float16):
        bias = None
        if mask is not None:
            bias = jax.lax.select(
                mask > 0,
                jnp.full(mask.shape, -1e10).astype(dtype),
                jnp.full(mask.shape, 0.0).astype(dtype),
            )
        if bias is not None:
            x = x + bias.astype(dtype)
        output = jax.nn.softmax(x * scale_factor)
        return jnp.mean(output)

    @pytest.mark.parametrize("device_count,mesh_shape,mesh_axes,mesh_resource", generate_configs())
    @pytest.mark.parametrize("data_shape", [[32, 12, 128, 128], [64, 16, 1024, 1024]])
    @pytest.mark.parametrize(
        "softmax_type",
        [SoftmaxType.SCALED, SoftmaxType.SCALED_MASKED, SoftmaxType.SCALED_UPPER_TRIANG_MASKED],
    )
    @pytest.mark.parametrize("scale_factor", [1.0, 3.0])
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("bad_sharding", [False, True])
    def test_softmax(
        self,
        device_count,
        mesh_shape,
        mesh_axes,
        mesh_resource,
        data_shape,
        softmax_type,
        scale_factor,
        dtype,
        bad_sharding,
    ):

        target_func = partial(
            self.target_func, scale_factor=scale_factor, softmax_type=softmax_type
        )
        ref_func = partial(self.ref_func, scale_factor=scale_factor, dtype=dtype)

        (x, mask), (x_pspec, mask_pspec) = self.generate_inputs(
            data_shape, mesh_resource, softmax_type, dtype, bad_sharding
        )
        collective_count_ref = self.generate_collectives_count_ref()
        devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
        mesh = Mesh(devices, mesh_axes)
        with mesh, fp8_autocast(mesh_resource=mesh_resource):
            x_ = jax.device_put(x, NamedSharding(mesh, x_pspec))
            mask_ = jax.device_put(mask, NamedSharding(mesh, mask_pspec))

            with warnings.catch_warnings(record=True) as warns:
                try:
                    compare_ops(
                        target_func,
                        ref_func,
                        [x_, mask_],
                        collective_count_ref,
                        grad_args=(0,),
                        metric_fwd_dtype=dtype,
                        metric_bwd_dtype=dtype,
                        in_shardings=(x_pspec, mask_pspec),
                        out_shardings=(None, (x_pspec,)),
                    )
                except AssertionError as err:
                    # Softmax should still produce the correct numerical result with
                    # bad sharding. However, the collective count may not be the same
                    # when XLA is forced to unshard the hidden dimension. We can catch
                    # and ignore that specific error here.
                    if not bad_sharding or "Expected collective count" not in str(err):
                        raise err
                finally:
                    for w in warns:
                        assert "Sharding the hidden dimension is not supported" in str(w), (
                            "Softmax primitive did not raise the correct warning for "
                            "unsupported sharding in the hidden dimension."
                        )
