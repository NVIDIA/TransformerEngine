# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest

import flax
import jax
import jax.numpy as jnp
import numpy as np

from utils import assert_allclose
from transformer_engine.common.recipe import DelayedScaling
from transformer_engine.common.recipe import Format as FP8Format
from transformer_engine.jax import fp8_autocast, get_delayed_scaling
from transformer_engine.jax.fp8 import FP8Helper, is_fp8_available, AmaxComputeAlgo
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource

is_fp8_supported, reason = is_fp8_available()


class TestFP8Helper(unittest.TestCase):

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_initialize(self):
        margin = 5.0
        fp8_format = FP8Format.E4M3
        amax_history_len = 10

        FP8Helper.initialize(
            margin=margin, fp8_format=fp8_format, amax_history_len=amax_history_len
        )

        self.assertEqual(
            FP8Helper.MARGIN,
            margin,
            f"FP8Helper.MARGIN initialization failed, should be {margin}"
            f" but got {FP8Helper.MARGIN}.",
        )
        self.assertEqual(
            FP8Helper.FP8_FORMAT,
            fp8_format,
            f"FP8Helper.FP8_FORMAT initialization failed, should be {fp8_format}"
            f" but got {FP8Helper.FP8_FORMAT}.",
        )
        self.assertEqual(
            FP8Helper.AMAX_HISTORY_LEN,
            amax_history_len,
            f"FP8Helper.AMAX_HISTORY_LEN initialization failed, should be {amax_history_len}"
            f" but got {FP8Helper.AMAX_HISTORY_LEN}.",
        )

        FP8Helper.finalize()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_update_collections(self):
        original_val = 0.0
        updated_val = 10.0

        original_state = {
            "test1": original_val,
            "test2": original_val,
        }
        updated_state = FP8Helper.update_collections({"test1": updated_val}, original_state)
        self.assertEqual(updated_state["test1"], updated_val)
        self.assertEqual(updated_state["test2"], original_val)

        original_state = flax.core.frozen_dict.FrozenDict(original_state)
        updated_state = FP8Helper.update_collections({"test1": updated_val}, original_state)
        self.assertEqual(updated_state["test1"], updated_val)
        self.assertEqual(updated_state["test2"], original_val)


class TestFP8Functions(unittest.TestCase):

    def _check_defult_state(self):
        self.assertFalse(FP8Helper.is_fp8_enabled())

    def _compare_delay_scaling(self, ref, test):
        self.assertTrue(ref.margin == test.margin)
        self.assertTrue(ref.fp8_format == test.fp8_format)
        self.assertTrue(ref.amax_history_len == test.amax_history_len)
        self.assertTrue(ref.amax_compute_algo == test.amax_compute_algo)

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast(self):
        FP8Helper.finalize()  # Ensure the testing not affect by previous tests.
        self._check_defult_state()

        with fp8_autocast(enabled=False, fp8_recipe=DelayedScaling()):
            self.assertFalse(FP8Helper.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), DelayedScaling())

        self._check_defult_state()

        ds = DelayedScaling(margin=5.0, fp8_format=FP8Format.E4M3, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            self.assertTrue(FP8Helper.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), ds)

        self._check_defult_state()

        ds = DelayedScaling(margin=3.0, fp8_format=FP8Format.HYBRID, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            self.assertTrue(FP8Helper.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), ds)

        self._check_defult_state()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast_with_sharding_resource(self):
        FP8Helper.finalize()  # Ensure the testing not affect by previous tests.
        self._check_defult_state()

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
                    self.assertTrue(FP8Helper.is_fp8_enabled())
                    self._compare_delay_scaling(get_delayed_scaling(), ds)
                    self.assertEqual(sr, global_mesh_resource())

                self._check_defult_state()
