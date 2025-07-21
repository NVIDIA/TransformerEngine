# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from utils import assert_allclose
from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling, Float8CurrentScaling
from transformer_engine.common.recipe import Format as FP8Format
from transformer_engine.jax import fp8_autocast, get_delayed_scaling
from transformer_engine.jax.quantize import (
    QuantizeConfig,
    is_fp8_available,
    ScalingMode,
    update_collections,
)
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource

is_fp8_supported, reason = is_fp8_available()
is_mxfp8_supported, mxfp8_reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)


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
        self.assertFalse(QuantizeConfig.is_fp8_enabled())

    def _compare_delay_scaling(self, ref, test):
        self.assertTrue(ref.margin == test.margin)
        self.assertTrue(ref.fp8_format == test.fp8_format)
        self.assertTrue(ref.amax_history_len == test.amax_history_len)
        self.assertTrue(ref.amax_compute_algo == test.amax_compute_algo)

    def _compare_current_scaling(self, test):
        self.assertEqual(QuantizeConfig.MARGIN, test.margin)
        self.assertEqual(QuantizeConfig.FP8_FORMAT, test.fp8_format)
        self.assertEqual(QuantizeConfig.SCALING_MODE, ScalingMode.CURRENT_TENSOR_SCALING)

    def _compare_mxfp8_scaling(self, test):
        self.assertEqual(QuantizeConfig.MARGIN, test.margin)
        self.assertEqual(QuantizeConfig.FP8_FORMAT, test.fp8_format)
        self.assertEqual(QuantizeConfig.SCALING_MODE, ScalingMode.MXFP8_1D_SCALING)

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast_delayed_scaling(self):
        QuantizeConfig.finalize()  # Ensure the testing not affect by previous tests.
        self._check_default_state()

        with fp8_autocast(enabled=False, fp8_recipe=DelayedScaling()):
            self._check_default_state()

        self._check_default_state()

        ds = DelayedScaling(margin=5.0, fp8_format=FP8Format.E4M3, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            self.assertTrue(QuantizeConfig.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), ds)

        self._check_default_state()

        ds = DelayedScaling(margin=3.0, fp8_format=FP8Format.HYBRID, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            self.assertTrue(QuantizeConfig.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), ds)

        self._check_default_state()

    @unittest.skipIf(not is_mxfp8_supported, reason=mxfp8_reason)
    def test_fp8_autocast_mxfp8_scaling(self):
        QuantizeConfig.finalize()  # Ensure the testing not affect by previous tests.
        self._check_default_state()

        with fp8_autocast(enabled=False, fp8_recipe=Float8CurrentScaling()):
            self._check_default_state()

        self._check_default_state()

        cs = Float8CurrentScaling(margin=5.0, fp8_format=FP8Format.E4M3)
        with fp8_autocast(enabled=True, fp8_recipe=cs):
            self.assertTrue(QuantizeConfig.is_fp8_enabled())
            self._compare_current_scaling(cs)

        self._check_default_state()

        cs = Float8CurrentScaling(margin=3.0, fp8_format=FP8Format.HYBRID)
        with fp8_autocast(enabled=True, fp8_recipe=cs):
            self.assertTrue(QuantizeConfig.is_fp8_enabled())
            self._compare_current_scaling(cs)

        self._check_default_state()

    @unittest.skipIf(not is_mxfp8_supported, reason=mxfp8_reason)
    def test_fp8_autocast_mxfp8_scaling(self):
        QuantizeConfig.finalize()  # Ensure the testing not affect by previous tests.
        self._check_default_state()

        with fp8_autocast(enabled=False, fp8_recipe=MXFP8BlockScaling()):
            self._check_default_state()

        self._check_default_state()

        bs = MXFP8BlockScaling(margin=5.0, fp8_format=FP8Format.E4M3)
        with fp8_autocast(enabled=True, fp8_recipe=bs):
            self.assertTrue(QuantizeConfig.is_fp8_enabled())
            self._compare_mxfp8_scaling(bs)

        self._check_default_state()

        bs = MXFP8BlockScaling(margin=3.0, fp8_format=FP8Format.HYBRID)
        with fp8_autocast(enabled=True, fp8_recipe=bs):
            self.assertTrue(QuantizeConfig.is_fp8_enabled())
            self._compare_mxfp8_scaling(bs)

        self._check_default_state()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast_with_sharding_resource(self):
        QuantizeConfig.finalize()  # Ensure the testing not affect by previous tests.
        self._check_default_state()

        ds = DelayedScaling(margin=5.0, fp8_format=FP8Format.E4M3, amax_history_len=1)

        mesh_s = (
            (MeshResource(None, None)),
            (MeshResource("dp", None)),
            (MeshResource(None, "tp")),
            (MeshResource("dp", "tp")),
        )
        # TODO (Ming Huang): Support multi-GPUs testing. # pylint: disable=fixme
        mesh_shape = (1, 1)
        devices = np.asarray(jax.devices()[:1]).reshape(*mesh_shape)
        with jax.sharding.Mesh(devices, ("dp", "tp")):
            for sr in mesh_s:
                with fp8_autocast(enabled=True, fp8_recipe=ds, mesh_resource=sr):
                    self.assertTrue(QuantizeConfig.is_fp8_enabled())
                    self._compare_delay_scaling(get_delayed_scaling(), ds)
                    self.assertEqual(sr, global_mesh_resource())

                self._check_default_state()
