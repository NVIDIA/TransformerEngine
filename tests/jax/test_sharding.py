# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest

from transformer_engine.jax.flax import extend_logical_axis_rules
from transformer_engine.jax.sharding import global_shard_guard, MeshResource

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
