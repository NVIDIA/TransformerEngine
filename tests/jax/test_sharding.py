# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
import pytest
import numpy as np

from transformer_engine.jax.flax import extend_logical_axis_rules
from transformer_engine.jax.sharding import get_group_of_mesh_axis, get_rank_of_mesh_axis
from transformer_engine.jax.sharding import global_shard_guard, num_of_devices
from transformer_engine.jax.sharding import MeshResource

LOGICAL_RULES = [
    [(("a1", None), ("a2", "ma2")), False],
    [(("a1", None), ("a2", "ma2"), ("a3", ("ma31", "ma32"))), True],
    [(("a1", None), ("a2", "ma2"), ("a3", "ma31"), ("a3", "ma32")), False],
    [(("a1", None), ("a2", "ma2"), ("batch", "batch_1200234")), True],
    [(("a1", None), ("a2", "ma2"), ("a2", "ma1"), ("batch", "model"), ("batch", "data")), True],
]

MeshS = [
    MeshResource(),
    MeshResource("data", None),
    MeshResource(None, "model"),
    MeshResource("data", "model"),
]

MESH_INFO = [
    (
        4,
        (2, 2),
        ("a1", "a2"),
        {"a1": [0, 0, 1, 1], "a2": [0, 1, 0, 1]},
        {"a1": [0, 1, 0, 1], "a2": [0, 0, 1, 1]},
    ),
    (
        4,
        (4, 1),
        ("a1", "a2"),
        {"a1": [0, 1, 2, 3], "a2": [0, 0, 0, 0]},
        {"a1": [0, 0, 0, 0], "a2": [0, 1, 2, 3]},
    ),
    (
        4,
        (1, 4),
        ("a1", "a2"),
        {"a1": [0, 0, 0, 0], "a2": [0, 1, 2, 3]},
        {"a1": [0, 1, 2, 3], "a2": [0, 0, 0, 0]},
    ),
    (
        8,
        (2, 2, 2),
        ("a1", "a2", "a3"),
        {
            "a1": [0, 0, 0, 0, 1, 1, 1, 1],
            "a2": [0, 0, 1, 1, 0, 0, 1, 1],
            "a3": [0, 1, 0, 1, 0, 1, 0, 1],
        },
        {
            "a1": [0, 1, 2, 3, 0, 1, 2, 3],
            "a2": [0, 1, 0, 1, 2, 3, 2, 3],
            "a3": [0, 0, 1, 1, 2, 2, 3, 3],
        },
    ),
]


class TestShardingSideAPI:

    @pytest.mark.parametrize("base_rules,need_assert", LOGICAL_RULES)
    @pytest.mark.parametrize("sr", MeshS)
    def test_extend_logical_axis_rules(self, base_rules, need_assert, sr):
        with global_shard_guard(sr):
            try:
                target_te_rules = extend_logical_axis_rules(tuple())
                extended_rules = extend_logical_axis_rules(base_rules)
                assert extended_rules == (*base_rules, *target_te_rules)
                assert not need_assert
            except AssertionError as ae:
                assert need_assert, f"{ae.args}"

    @pytest.mark.parametrize("mesh_info", MESH_INFO)
    def test_get_rank_and_group_of_mesh_axis(self, mesh_info):
        num_device, mesh_shape, mesh_axes, rank_ref, group_ref = mesh_info

        if num_of_devices() < num_device:
            pytest.skip("Not enough devices for this test.")

        devices = np.asarray(jax.devices()[:num_device]).reshape(*mesh_shape)
        mesh = jax.sharding.Mesh(devices, mesh_axes)

        for d_id in range(num_device):
            for axis in mesh_axes:
                rank = get_rank_of_mesh_axis(d_id, axis, mesh)
                assert rank == rank_ref[axis][d_id]
                group = get_group_of_mesh_axis(d_id, axis, mesh)
                assert group == group_ref[axis][d_id]
