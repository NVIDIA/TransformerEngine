# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from utils import assert_allclose
from transformer_engine.common.recipe import (
    DelayedScaling,
    MXFP8BlockScaling,
    Float8CurrentScaling,
    NVFP4BlockScaling,
)
from transformer_engine.common.recipe import Format as FP8Format
from transformer_engine.jax import fp8_autocast
from transformer_engine.jax.quantize import (
    get_quantize_config,
    is_scaling_mode_supported,
    ScalingMode,
    update_collections,
    TensorSource,
)
from transformer_engine.jax.quantize.helper import _format2dtypes
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource

is_fp8_supported, reason = is_scaling_mode_supported(ScalingMode.DELAYED_TENSOR_SCALING)
is_mxfp8_supported, mxfp8_reason = is_scaling_mode_supported(ScalingMode.MXFP8_1D_SCALING)
is_nvfp4_supported, nvfp4_reason = is_scaling_mode_supported(ScalingMode.NVFP4_1D_SCALING)


class TestHelper(unittest.TestCase):

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_update_collections(self):
        original_val = 0.0
        updated_val = 10.0

        original_state = {
            "test1": original_val,
            "test2": original_val,
        }
        updated_state = update_collections({"test1": updated_val}, original_state)
        self.assertEqual(updated_state["test1"], updated_val)
        self.assertEqual(updated_state["test2"], original_val)

        original_state = flax.core.frozen_dict.FrozenDict(original_state)
        updated_state = update_collections({"test1": updated_val}, original_state)
        self.assertEqual(updated_state["test1"], updated_val)
        self.assertEqual(updated_state["test2"], original_val)


class TestFP8Functions(unittest.TestCase):

    def _check_default_state(self):
        self.assertFalse(get_quantize_config().is_fp8_enabled())

    def _compare_delay_scaling(self, test):
        self.assertEqual(get_quantize_config().MARGIN, test.margin)
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp8_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp8_format)[1])
        self.assertEqual(get_quantize_config().AMAX_HISTORY_LEN, test.amax_history_len)
        self.assertEqual(get_quantize_config().AMAX_COMPUTE_ALGO.value, test.amax_compute_algo)

    def _compare_current_scaling(self, test):
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp8_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp8_format)[1])
        for tensor_source in TensorSource:
            self.assertEqual(
                get_quantize_config().get_scaling_mode(tensor_source),
                ScalingMode.CURRENT_TENSOR_SCALING,
            )

    def _compare_mxfp8_scaling(self, test):
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp8_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp8_format)[1])
        for tensor_source in TensorSource:
            self.assertEqual(
                get_quantize_config().get_scaling_mode(tensor_source), ScalingMode.MXFP8_1D_SCALING
            )

    def _compare_nvfp4_scaling(self, test):
        self.assertEqual(get_quantize_config().FWD_DTYPE, _format2dtypes(test.fp4_format)[0])
        self.assertEqual(get_quantize_config().BWD_DTYPE, _format2dtypes(test.fp4_format)[1])
        for tensor_source in TensorSource:
            target_scaling_mode = (
                ScalingMode.NVFP4_2D_SCALING
                if tensor_source == TensorSource.KERNEL
                else ScalingMode.NVFP4_1D_SCALING
            )
            self.assertEqual(
                get_quantize_config().get_scaling_mode(tensor_source), target_scaling_mode
            )

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast_delayed_scaling(self):
        self._check_default_state()

        with fp8_autocast(enabled=False, fp8_recipe=DelayedScaling(), mesh_resource=MeshResource()):
            self._check_default_state()

        self._check_default_state()

        ds = DelayedScaling(margin=5.0, fp8_format=FP8Format.E4M3, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_delay_scaling(ds)

        self._check_default_state()

        ds = DelayedScaling(margin=3.0, fp8_format=FP8Format.HYBRID, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_delay_scaling(ds)

        self._check_default_state()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast_current_scaling(self):
        self._check_default_state()

        with fp8_autocast(
            enabled=False, fp8_recipe=Float8CurrentScaling(), mesh_resource=MeshResource()
        ):
            self._check_default_state()

        self._check_default_state()

        cs = Float8CurrentScaling(fp8_format=FP8Format.E4M3)
        with fp8_autocast(enabled=True, fp8_recipe=cs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_current_scaling(cs)

        self._check_default_state()

        cs = Float8CurrentScaling(fp8_format=FP8Format.HYBRID)
        with fp8_autocast(enabled=True, fp8_recipe=cs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_current_scaling(cs)

        self._check_default_state()

    @unittest.skipIf(not is_mxfp8_supported, reason=mxfp8_reason)
    def test_fp8_autocast_mxfp8_block_scaling(self):
        self._check_default_state()

        with fp8_autocast(
            enabled=False, fp8_recipe=MXFP8BlockScaling(), mesh_resource=MeshResource()
        ):
            self._check_default_state()

        self._check_default_state()

        bs = MXFP8BlockScaling()
        with fp8_autocast(enabled=True, fp8_recipe=bs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_mxfp8_scaling(bs)

        self._check_default_state()

    @unittest.skipIf(not is_nvfp4_supported, reason=nvfp4_reason)
    def test_fp8_autocast_nvfp4_block_scaling(self):
        self._check_default_state()

        with fp8_autocast(
            enabled=False, fp8_recipe=NVFP4BlockScaling(), mesh_resource=MeshResource()
        ):
            self._check_default_state()

        self._check_default_state()

        bs = NVFP4BlockScaling()
        with fp8_autocast(enabled=True, fp8_recipe=bs, mesh_resource=MeshResource()):
            self.assertTrue(get_quantize_config().is_fp8_enabled())
            self._compare_nvfp4_scaling(bs)

        self._check_default_state()
