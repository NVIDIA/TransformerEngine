# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest

import jax
import jax.numpy as jnp

from utils import assert_allclose
from transformer_engine.jax.flax.module import _apply_low_rank_adaptation
from transformer_engine.jax.flax.module import _normalize_axes


class TestLoRA:

    def reference(x, la, lb, pattern):
        out = jnp.einsum(pattern, x, la, lb)
        return out

    @pytest.mark.parametrize('shape', [(32, 1024), (32, 128, 1024)])
    @pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize('axis_features_pattern', [((-1,), (1024,), '...h,hr,rk->...k'),
                                                       ((-1,), (3, 1024), '...h,hkr,krz->...kz')])
    @pytest.mark.parametrize('rank', [32, 16])
    def test_lora(self, shape, dtype, axis_features_pattern, rank):
        axis, features, pattern = axis_features_pattern
        axis = _normalize_axes(axis, len(shape))
        shape_in_axis = tuple(shape[ax] for ax in axis)

        key = jax.random.key(1124)
        key, x_key = jax.random.split(key)
        x = jax.random.normal(x_key, shape, dtype)

        key, la_key = jax.random.split(key)
        la_shape = (*shape_in_axis, *features[:-1], rank)
        la = jax.random.normal(la_key, la_shape, dtype)

        key, lb_key = jax.random.split(key)
        lb_shape = (*features[:-1], rank, features[-1])
        lb = jax.random.normal(lb_key, lb_shape, dtype)

        out_target = _apply_low_rank_adaptation(x, axis, features, la, lb)
        out_ref = TestLoRA.reference(x, la, lb, pattern)

        assert_allclose(out_target, out_ref, dtype=dtype)
