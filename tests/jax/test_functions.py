# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest

import jax
import jax.numpy as jnp

from utils import assert_allclose
from transformer_engine.jax.flax.module import _apply_low_rank_adaptation
from transformer_engine.jax.flax.module import _normalize_axes
from transformer_engine.jax.flax.transformer import LoRAScope
from transformer_engine.jax.flax.transformer import _canonicalize_lora_scope


class TestLoRA:

    def reference(x, la, lb, pattern, scale):
        out = jnp.einsum(pattern, x, la, lb)
        return out * scale

    @pytest.mark.parametrize("shape", [(32, 1024), (32, 128, 1024)])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
    @pytest.mark.parametrize(
        "axis_features_pattern",
        [((-1,), (1024,), "...h,hr,rk->...k"), ((-1,), (3, 1024), "...h,hkr,krz->...kz")],
    )
    @pytest.mark.parametrize("rank", [32, 16])
    @pytest.mark.parametrize("alpha", [None, 4, 8])
    def test_lora(self, shape, dtype, axis_features_pattern, rank, alpha):
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

        out_target = _apply_low_rank_adaptation(x, axis, features, la, lb, alpha)
        scale_ref = alpha / rank if alpha is not None else 1.0
        out_ref = TestLoRA.reference(x, la, lb, pattern, scale_ref)

        assert_allclose(out_target, out_ref, dtype=dtype)

    @pytest.mark.parametrize(
        "scope_ref_assert",
        [
            ("none", LoRAScope(False, False, False), False),
            ("all", LoRAScope(True, True, True), False),
            ("qkv_proj", LoRAScope(True, False, False), False),
            ("output_proj", LoRAScope(False, True, False), False),
            ("mlp", LoRAScope(False, False, True), False),
            ("exclude_qkv_proj", LoRAScope(False, True, True), False),
            ("exclude_output_proj", LoRAScope(True, False, True), False),
            ("exclude_mlp", LoRAScope(True, True, False), False),
            ("messing_up", LoRAScope(), True),
        ],
    )
    def test_lora_scope_generator(self, scope_ref_assert):
        scope, reference, need_assert = scope_ref_assert
        try:
            lora_scope = _canonicalize_lora_scope(scope)
            assert lora_scope == reference
        except AssertionError as ae:
            assert need_assert, f"{ae.args}"
