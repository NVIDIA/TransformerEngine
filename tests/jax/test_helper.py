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
        update_fp8meta_interval = 10
        amax_history_len = 10

        FP8Helper.initialize(margin=margin,
                             fp8_format=fp8_format,
                             update_fp8meta_interval=update_fp8meta_interval,
                             amax_history_len=amax_history_len)

        self.assertEqual(
            FP8Helper.MARGIN, margin, f"FP8Helper.MARGIN initialization failed, should be {margin}"
            f" but got {FP8Helper.MARGIN}.")
        self.assertEqual(
            FP8Helper.FP8_FORMAT, fp8_format,
            f"FP8Helper.FP8_FORMAT initialization failed, should be {fp8_format}"
            f" but got {FP8Helper.FP8_FORMAT}.")
        self.assertEqual(
            FP8Helper.UPDATE_FP8META_INTERVAL, update_fp8meta_interval,
            "FP8Helper.UPDATE_FP8META_INTERVAL initialization failed, should be"
            f"{update_fp8meta_interval} but got {FP8Helper.UPDATE_FP8META_INTERVAL}.")
        self.assertEqual(
            FP8Helper.AMAX_HISTORY_LEN, amax_history_len,
            f"FP8Helper.AMAX_HISTORY_LEN initialization failed, should be {amax_history_len}"
            f" but got {FP8Helper.AMAX_HISTORY_LEN}.")

        FP8Helper.finalize()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_update_fp8_metas(self):
        FP8Helper.initialize(margin=3.0, amax_history_len=3)

        seed = 0
        key1, key2 = jax.random.split(jax.random.PRNGKey(seed))
        num_of_gemm = 10
        num_of_meta = FP8Helper.NUM_META_PER_GEMM * num_of_gemm

        def select_amax(amaxes):
            if FP8Helper.AMAX_COMPUTE_ALGO == AmaxComputeAlgo.MAX:
                return jnp.max(amaxes, axis=-1, keepdims=True)
            return amaxes[:, 0:1]

        def get_fp8_scale(fp8_max, amax, scale):
            fp8_max = np.array(fp8_max)
            amax = np.array(amax)
            scale = np.array(scale)

            sf = (fp8_max / amax) / (2**FP8Helper.MARGIN)
            sf = jnp.where(amax > 0.0, sf, scale)
            sf = jnp.where(jnp.isfinite(amax), sf, scale)
            return sf

        amax_meta_shape = (num_of_meta, FP8Helper.AMAX_HISTORY_LEN)
        scale_meta_shape = (num_of_meta, 1)
        fp8_max_array = FP8Helper.generate_fp8_max_array(num_of_meta)
        fp8_amax_array1 = jax.random.uniform(key1, shape=amax_meta_shape)
        fp8_scale_array1 = get_fp8_scale(fp8_max_array, select_amax(fp8_amax_array1),
                                         jnp.ones(scale_meta_shape))
        fp8_scale_inv_array1 = 1 / fp8_scale_array1
        fp8_amax_array2 = jax.random.uniform(key2, shape=amax_meta_shape)
        fp8_scale_array2 = get_fp8_scale(fp8_max_array, select_amax(fp8_amax_array2),
                                         jnp.ones(scale_meta_shape))
        fp8_scale_inv_array2 = 1 / fp8_scale_array2

        state = flax.core.frozen_dict.FrozenDict({
            FP8Helper.FP8_COLLECTION_NAME: {
                "test_update_fp8_metas1": {
                    FP8Helper.FP8_MAX_NAME: fp8_max_array,
                    FP8Helper.FP8_AMAX_NAME: fp8_amax_array1,
                    FP8Helper.FP8_SCALE_NAME: jnp.ones(scale_meta_shape),
                    FP8Helper.FP8_SCALE_INV_NAME: jnp.ones(scale_meta_shape)
                },
                "test_update_fp8_metas2": {
                    FP8Helper.FP8_MAX_NAME: fp8_max_array,
                    FP8Helper.FP8_AMAX_NAME: fp8_amax_array2,
                    FP8Helper.FP8_SCALE_NAME: jnp.ones(scale_meta_shape),
                    FP8Helper.FP8_SCALE_INV_NAME: jnp.ones(scale_meta_shape)
                }
            }
        })

        updated_state = FP8Helper.update_fp8_metas(state)

        state_array, _ = jax.tree_util.tree_flatten(updated_state)
        meta_per_gemm = FP8Helper.NUM_META_PER_GEMM + 1
        scale_shift = 2
        scale_inv_shift = 3
        assert_allclose(state_array[0 * meta_per_gemm + scale_shift], fp8_scale_array1)
        assert_allclose(state_array[0 * meta_per_gemm + scale_inv_shift], fp8_scale_inv_array1)
        assert_allclose(state_array[1 * meta_per_gemm + scale_shift], fp8_scale_array2)
        assert_allclose(state_array[1 * meta_per_gemm + scale_inv_shift], fp8_scale_inv_array2)

        FP8Helper.finalize()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_generate_fp8_max_array(self):
        num_of_meta = FP8Helper.NUM_META_PER_GEMM * 2

        def get_ref(format_for_test):
            refer_list = []
            for i in range(num_of_meta):
                val = format_for_test.value.max_bwd \
                    if i % FP8Helper.NUM_META_PER_GEMM == FP8Helper.GRAD_META_IDX_PER_GEMM \
                    else format_for_test.value.max_fwd
                refer_list.append([val])
            return jnp.asarray(refer_list)

        for fp8_format in FP8Format:
            FP8Helper.initialize(fp8_format=fp8_format)
            assert_allclose(get_ref(fp8_format), FP8Helper.generate_fp8_max_array(num_of_meta))
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
        self.assertTrue(ref.interval == test.interval)
        self.assertTrue(ref.fp8_format == test.fp8_format)
        self.assertTrue(ref.amax_history_len == test.amax_history_len)
        self.assertTrue(ref.amax_compute_algo == test.amax_compute_algo)

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast(self):
        FP8Helper.finalize()    # Ensure the testing not affect by previous tests.
        self._check_defult_state()

        with fp8_autocast(enabled=False, fp8_recipe=DelayedScaling()):
            self.assertFalse(FP8Helper.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), DelayedScaling())

        self._check_defult_state()

        ds = DelayedScaling(margin=5.0, interval=3, fp8_format=FP8Format.E4M3, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            self.assertTrue(FP8Helper.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), ds)

        self._check_defult_state()

        ds = DelayedScaling(margin=3.0, interval=1, fp8_format=FP8Format.HYBRID, amax_history_len=1)
        with fp8_autocast(enabled=True, fp8_recipe=ds):
            self.assertTrue(FP8Helper.is_fp8_enabled())
            self._compare_delay_scaling(get_delayed_scaling(), ds)

        self._check_defult_state()

    @unittest.skipIf(not is_fp8_supported, reason=reason)
    def test_fp8_autocast_with_sharding_resource(self):
        FP8Helper.finalize()    # Ensure the testing not affect by previous tests.
        self._check_defult_state()

        ds = DelayedScaling(margin=5.0, interval=3, fp8_format=FP8Format.E4M3, amax_history_len=1)

        mesh_s = (
            (MeshResource(None, None)),
            (MeshResource('dp', None)),
            (MeshResource(None, 'tp')),
            (MeshResource('dp', 'tp')),
        )
        # TODO (Ming Huang): Support multi-GPUs testing. # pylint: disable=fixme
        mesh_shape = (1, 1)
        devices = np.asarray(jax.devices()[:1]).reshape(*mesh_shape)
        with jax.sharding.Mesh(devices, ('dp', 'tp')):
            for sr in mesh_s:
                with fp8_autocast(enabled=True, fp8_recipe=ds, mesh_resource=sr):
                    self.assertTrue(FP8Helper.is_fp8_enabled())
                    self._compare_delay_scaling(get_delayed_scaling(), ds)
                    self.assertEqual(sr, global_mesh_resource())

                self._check_defult_state()
