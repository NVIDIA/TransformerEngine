# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
import numpy as np
import pytest

from utils import is_devices_enough
from transformer_engine.jax.flax import extend_logical_axis_rules
from transformer_engine.jax.sharding import get_dot_sharding_meta
from transformer_engine.jax.sharding import get_elementwise_sharding_meta
from transformer_engine.jax.sharding import get_fp8_meta_sharding_meta
from transformer_engine.jax.sharding import global_shard_guard
from transformer_engine.jax.sharding import infer_major_sharding_type
from transformer_engine.jax.sharding import is_dp_enabled, is_tp_enabled
from transformer_engine.jax.sharding import ShardingMeta, ShardingResource, ShardingType


def _get_sharding_resource(mesh_names, sharding_type):
    dp_r = None
    tp_r = None

    if sharding_type in (ShardingType.DP, ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
        dp_r = mesh_names[0]

    if sharding_type in (ShardingType.TP_COL, ShardingType.TP_ROW):
        tp_r = mesh_names[0]

    if sharding_type in (ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
        tp_r = mesh_names[1]
    return ShardingResource(dp_r, tp_r)


DEVICE_COUNT = 4
MESH_CONFIG = [((4,), ("dp",), ShardingType.DP), ((4,), ("tp",), ShardingType.TP_COL),
               ((4,), ("tp",), ShardingType.TP_ROW), ((2, 2), ("dp", "tp"), ShardingType.DP_TP_COL),
               ((2, 2), ("dp", "tp"), ShardingType.DP_TP_ROW)]

LOGICAL_RULES = [
    [(('a1', None), ('a2', 'ma2')), False],
    [(('a1', None), ('a2', 'ma2'), ('a3', ('ma31', 'ma32'))), True],
    [(('a1', None), ('a2', 'ma2'), ('a3', 'ma31'), ('a3', 'ma32')), False],
    [(('a1', None), ('a2', 'ma2'), ('batch', 'batch_1200234')), True],
    [(('a1', None), ('a2', 'ma2'), ('a2', 'ma1'), ('batch', 'model'), ('batch', 'data')), True],
]
SRS = [
    ShardingResource(),
    ShardingResource('data', None),
    ShardingResource(None, 'model'),
    ShardingResource('data', 'model')
]


class TestShardingSideAPI:

    @pytest.mark.parametrize('base_rules,need_assert', LOGICAL_RULES)
    @pytest.mark.parametrize('sr', SRS)
    def test_extend_logical_axis_rules(self, base_rules, need_assert, sr):
        with global_shard_guard(sr):
            try:
                target_te_rules = extend_logical_axis_rules(tuple())
                extended_rules = extend_logical_axis_rules(base_rules)
                assert extended_rules == (*base_rules, *target_te_rules)
                assert not need_assert
            except AssertionError as ae:
                assert need_assert, f"{ae.args}"


class TestGeneralFunc:

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type', MESH_CONFIG)
    @pytest.mark.skipif(not is_devices_enough(DEVICE_COUNT), reason='Num of GPU is not enough')
    def test_infer_major_sharding_type(
            self,
            mesh_shape,    # pylint: disable=unused-argument
            mesh_names,
            sharding_type):
        devices = np.asarray(jax.devices()[:DEVICE_COUNT]).reshape(*mesh_shape)
        with global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            with jax.sharding.Mesh(devices, mesh_names):
                assert infer_major_sharding_type() is sharding_type.value[0]

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type', MESH_CONFIG)
    def test_is_dp_enabled(
            self,
            mesh_shape,    # pylint: disable=unused-argument
            mesh_names,    # pylint: disable=unused-argument
            sharding_type):
        if sharding_type in (ShardingType.DP, ShardingType.DP_TP_COL, ShardingType.DP_TP_ROW):
            assert is_dp_enabled(sharding_type.value[0])
        else:
            assert not is_dp_enabled(sharding_type.value[0])

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type', MESH_CONFIG)
    def test_is_tp_enabled(
            self,
            mesh_shape,    # pylint: disable=unused-argument
            mesh_names,    # pylint: disable=unused-argument
            sharding_type):
        if sharding_type is ShardingType.DP:
            assert not is_tp_enabled(sharding_type.value[0])
        else:
            assert is_tp_enabled(sharding_type.value[0])


class TestShardingMetaGenerator:

    BATCH_AXIS_NAME = 'batch'
    MODEL_AXIS_NAME = 'model'

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type', MESH_CONFIG)
    @pytest.mark.skipif(not is_devices_enough(DEVICE_COUNT), reason='Num of GPU is not enough')
    def test_fp8_meta(self, mesh_shape, mesh_names, sharding_type, num_of_fp8_meta=4):

        def stack_axes_meta(mapping):
            return tuple(mapping for _ in range(num_of_fp8_meta))

        def get_ref_sm():
            if sharding_type == ShardingType.DP:
                return ShardingMeta(stack_axes_meta({}), stack_axes_meta({}),
                                    {TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0]}, (),
                                    ())

            if sharding_type == ShardingType.TP_COL:
                return ShardingMeta(stack_axes_meta({}), stack_axes_meta({}),
                                    {TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[0]}, (),
                                    ())

            if sharding_type == ShardingType.TP_ROW:
                return ShardingMeta(stack_axes_meta({}), stack_axes_meta({}),
                                    {TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[0]}, (),
                                    ())

            if sharding_type == ShardingType.DP_TP_COL:
                return ShardingMeta(
                    stack_axes_meta({}), stack_axes_meta({}), {
                        TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0],
                        TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[1]
                    }, (), ())

            if sharding_type == ShardingType.DP_TP_ROW:
                return ShardingMeta(
                    stack_axes_meta({}), stack_axes_meta({}), {
                        TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0],
                        TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[1]
                    }, (), ())
            return None

        devices = np.asarray(jax.devices()[:DEVICE_COUNT]).reshape(*mesh_shape)
        with global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            with jax.sharding.Mesh(devices, mesh_names):
                test_sm = get_fp8_meta_sharding_meta(
                    sharding_type,
                    num_of_fp8_meta,
                    dp_axis_name=TestShardingMetaGenerator.BATCH_AXIS_NAME,
                    tp_axis_name=TestShardingMetaGenerator.MODEL_AXIS_NAME)
                assert test_sm == get_ref_sm()

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type', MESH_CONFIG)
    @pytest.mark.parametrize('a_shape, b_shape', [((64, 128, 256), (256, 512)),
                                                  ((128, 64, 512), (512, 256))])
    @pytest.mark.parametrize('batch_dim_of_a', [0, 1])
    @pytest.mark.skipif(not is_devices_enough(DEVICE_COUNT), reason='Num of GPU is not enough')
    def test_dot(self, mesh_shape, mesh_names, sharding_type, a_shape, b_shape, batch_dim_of_a):
        model_dim_of_a = len(a_shape) - 1
        model_dim_of_b = 0 if sharding_type in (ShardingType.TP_ROW, ShardingType.DP_TP_ROW) else 1
        contracting_dims = ((-1,), (0,))

        def get_ref_sm():
            out_shape = (*a_shape[:min(contracting_dims[0])],
                         *b_shape[max(contracting_dims[1]) + 1:])
            if sharding_type == ShardingType.DP:
                a_new_shape = (*a_shape[:batch_dim_of_a], mesh_shape[0], -1,
                               *a_shape[batch_dim_of_a + 1:])
                return ShardingMeta(({
                    batch_dim_of_a: TestShardingMetaGenerator.BATCH_AXIS_NAME
                }, {}), ({
                    batch_dim_of_a: TestShardingMetaGenerator.BATCH_AXIS_NAME
                }), {TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0]},
                                    [a_new_shape, b_shape], [out_shape])

            if sharding_type == ShardingType.TP_COL:
                b_new_shape = (b_shape[0], mesh_shape[0], b_shape[1] // mesh_shape[0])
                return ShardingMeta(({}, {
                    1: TestShardingMetaGenerator.MODEL_AXIS_NAME
                }), ({
                    len(out_shape) - 1: TestShardingMetaGenerator.MODEL_AXIS_NAME
                }), {TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[0]},
                                    [a_shape, b_new_shape], [out_shape])

            if sharding_type == ShardingType.TP_ROW:
                a_new_shape = (*a_shape[:-1], mesh_shape[0], a_shape[-1] // mesh_shape[0])
                b_new_shape = (mesh_shape[0], b_shape[0] // mesh_shape[0], b_shape[1])
                return ShardingMeta(({
                    len(a_new_shape) - 2: TestShardingMetaGenerator.MODEL_AXIS_NAME
                }, {
                    0: TestShardingMetaGenerator.MODEL_AXIS_NAME
                }), ({}), {TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[0]},
                                    [a_new_shape, b_new_shape], [out_shape])

            if sharding_type == ShardingType.DP_TP_COL:
                a_new_shape = (*a_shape[:batch_dim_of_a], mesh_shape[0],
                               a_shape[batch_dim_of_a] // mesh_shape[0],
                               *a_shape[batch_dim_of_a + 1:])
                b_new_shape = (b_shape[0], mesh_shape[1], b_shape[1] // mesh_shape[1])
                return ShardingMeta(
                    ({
                        batch_dim_of_a: TestShardingMetaGenerator.BATCH_AXIS_NAME
                    }, {
                        1: TestShardingMetaGenerator.MODEL_AXIS_NAME
                    }), ({
                        batch_dim_of_a: TestShardingMetaGenerator.BATCH_AXIS_NAME,
                        len(out_shape): TestShardingMetaGenerator.MODEL_AXIS_NAME
                    }), {
                        TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0],
                        TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[1]
                    }, [a_new_shape, b_new_shape], [out_shape])

            if sharding_type == ShardingType.DP_TP_ROW:
                a_new_shape = (*a_shape[:batch_dim_of_a], mesh_shape[0],
                               a_shape[batch_dim_of_a] // mesh_shape[0],
                               *a_shape[batch_dim_of_a + 1:-1], mesh_shape[1],
                               a_shape[-1] // mesh_shape[1])
                b_new_shape = (mesh_shape[1], b_shape[0] // mesh_shape[1], b_shape[1])
                return ShardingMeta(
                    ({
                        batch_dim_of_a: TestShardingMetaGenerator.BATCH_AXIS_NAME,
                        len(a_new_shape) - 2: TestShardingMetaGenerator.MODEL_AXIS_NAME
                    }, {
                        0: TestShardingMetaGenerator.MODEL_AXIS_NAME
                    }), ({
                        batch_dim_of_a: TestShardingMetaGenerator.BATCH_AXIS_NAME
                    }), {
                        TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0],
                        TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[1]
                    }, [a_new_shape, b_new_shape], [out_shape])
            return None

        devices = np.asarray(jax.devices()[:DEVICE_COUNT]).reshape(*mesh_shape)
        with global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            with jax.sharding.Mesh(devices, mesh_names):
                test_sm = get_dot_sharding_meta(
                    sharding_type,
                    a_shape,
                    b_shape,
                    batch_dim_of_a,
                    model_dim_of_a,
                    model_dim_of_b,
                    contracting_dims,
                    dp_axis_name=TestShardingMetaGenerator.BATCH_AXIS_NAME,
                    tp_axis_name=TestShardingMetaGenerator.MODEL_AXIS_NAME)
                assert test_sm == get_ref_sm()

    @pytest.mark.parametrize('mesh_shape,mesh_names,sharding_type', MESH_CONFIG)
    @pytest.mark.parametrize('input_shape', [(64, 128, 256), (128, 64, 512)])
    @pytest.mark.parametrize('other_shape', [(256,), (512,)])
    @pytest.mark.parametrize('batch_dim', [0, 1])
    @pytest.mark.skipif(not is_devices_enough(DEVICE_COUNT), reason='Num of GPU is not enough')
    def test_elementwise(self, mesh_shape, mesh_names, sharding_type, input_shape, other_shape,
                         batch_dim):

        def get_ref_sm():
            need_assert = True
            ref_sharding_meta = None
            if input_shape[-1] != other_shape[0]:
                need_assert = True
                ref_sharding_meta = None
            elif sharding_type is (ShardingType.DP_TP_COL, ShardingType.DP):
                need_assert = False
                input_new_shape = (*input_shape[:batch_dim], mesh_shape[0], -1,
                                   *input_shape[batch_dim + 1:])
                ref_sharding_meta = ShardingMeta(({
                    batch_dim: TestShardingMetaGenerator.BATCH_AXIS_NAME
                }, {}), ({
                    batch_dim: TestShardingMetaGenerator.BATCH_AXIS_NAME
                }), {TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0]},
                                                 [input_new_shape, other_shape], [input_shape])
            elif sharding_type is ShardingType.TP_COL:
                need_assert = False
                ref_sharding_meta = ShardingMeta(({}, {}), ({}), {}, [input_shape, other_shape],
                                                 [input_shape])
            elif sharding_type is ShardingType.TP_ROW:
                need_assert = False
                input_new_shape = (*input_shape[:-1], mesh_shape[0], -1)
                other_new_shape = (mesh_shape[0], -1)

                ref_sharding_meta = ShardingMeta(({
                    len(input_new_shape) - 2: TestShardingMetaGenerator.MODEL_AXIS_NAME
                }, {
                    0: TestShardingMetaGenerator.MODEL_AXIS_NAME
                }), ({
                    len(input_new_shape) - 2: TestShardingMetaGenerator.MODEL_AXIS_NAME
                }), {TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[0]},
                                                 [input_new_shape, other_new_shape], [input_shape])
            elif sharding_type is ShardingType.DP_TP_ROW:
                need_assert = False
                input_new_shape = (*input_shape[:batch_dim], mesh_shape[0], -1,
                                   *input_shape[batch_dim + 1:-1], mesh_shape[1],
                                   input_shape[-1] // mesh_shape[1])
                other_new_shape = (mesh_shape[0], -1)

                ref_sharding_meta = ShardingMeta(
                    ({
                        batch_dim: TestShardingMetaGenerator.BATCH_AXIS_NAME,
                        len(input_new_shape) - 2: TestShardingMetaGenerator.MODEL_AXIS_NAME
                    }, {
                        0: TestShardingMetaGenerator.MODEL_AXIS_NAME
                    }), ({
                        batch_dim: TestShardingMetaGenerator.BATCH_AXIS_NAME,
                        len(input_new_shape) - 2: TestShardingMetaGenerator.MODEL_AXIS_NAME
                    }), {
                        TestShardingMetaGenerator.BATCH_AXIS_NAME: mesh_names[0],
                        TestShardingMetaGenerator.MODEL_AXIS_NAME: mesh_names[1]
                    }, [input_new_shape, other_new_shape], [input_shape])

            return ref_sharding_meta, need_assert

        devices = np.asarray(jax.devices()[:DEVICE_COUNT]).reshape(*mesh_shape)
        with global_shard_guard(_get_sharding_resource(mesh_names, sharding_type)):
            with jax.sharding.Mesh(devices, mesh_names):
                ref_sm, need_assert = get_ref_sm()
                try:
                    test_sm = get_elementwise_sharding_meta(
                        sharding_type,
                        input_shape,
                        other_shape,
                        batch_dim,
                        dp_axis_name=TestShardingMetaGenerator.BATCH_AXIS_NAME,
                        tp_axis_name=TestShardingMetaGenerator.MODEL_AXIS_NAME)
                    assert not need_assert
                    assert test_sm == ref_sm
                except (NotImplementedError, AssertionError) as e:
                    assert need_assert, f"{e.args}"
