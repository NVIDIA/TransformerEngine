# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for group sharding"""

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer import (
    DygraphShardingOptimizer,
)

from utils import assert_allclose, set_random_seed
import transformer_engine.paddle as te


class TestGroupSharding(unittest.TestCase):
    """Tests group sharding"""

    def setUp(self):
        self.set_attr()
        self.init_dist_env()
        paddle.set_default_dtype(self.global_dtype)

    def set_attr(self):
        """Set test configs"""
        self.sharding_degree = 2
        self.global_dtype = "float32"
        self.rtol = 1e-5
        self.atol = 1e-5
        self.batch_size = 16
        self.in_channels = 16
        self.out_channels = 32
        self.fp8 = False

    def init_dist_env(self):
        """Init Paddle Fleet environment"""
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": self.sharding_degree,
        }
        self.strategy = strategy
        fleet.init(is_collective=True, strategy=strategy)

    def _get_model_and_optimizer(self, model, stage):
        if stage == 1:
            optimizer = DygraphShardingOptimizer(
                paddle.optimizer.AdamW(learning_rate=0.01, parameters=model.parameters()),
                fleet.get_hybrid_communicate_group(),
            )
            model = fleet.distributed_model(model)
            optimizer = fleet.distributed_optimizer(optimizer)
        elif stage in [2, 3]:
            optimizer = paddle.optimizer.AdamW(learning_rate=0.01, parameters=model.parameters())
            group = fleet.get_hybrid_communicate_group().get_sharding_parallel_group()

            class ShardingLevel:  # pylint: disable=too-few-public-methods,
                """Paddle sharding options"""

                kStage1 = "os"
                kStage2 = "os_g"
                kStage3 = "p_g_os"

            level = ShardingLevel.kStage3 if stage == 3 else ShardingLevel.kStage2
            model, optimizer, _ = paddle.distributed.sharding.group_sharded_parallel(
                model=model,
                optimizer=optimizer,
                level=level,
                group=group,
                segment_size=256,
            )
        else:
            raise ValueError(f"Stage {stage} not supported")
        return model, optimizer

    def test_group_sharding_stage1(self):
        """Tests group sharding training"""
        set_random_seed(1024)
        model_te = te.Linear(self.in_channels, self.out_channels)
        model_pd = paddle.nn.Linear(self.in_channels, self.out_channels)
        model_pd.weight.copy_(model_te.weight.T, True)
        model_pd.bias.copy_(model_te.bias, True)

        model_te, optimizer_te = self._get_model_and_optimizer(model_te, stage=1)
        model_pd, optimizer_pd = self._get_model_and_optimizer(model_pd, stage=1)

        rank_id = paddle.distributed.get_rank()
        paddle.seed(rank_id)

        def train_one_step(model, inp, optimizer):
            out = model(inp)
            loss = out.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            return loss

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.in_channels], self.global_dtype)
            with te.fp8_autocast(enabled=False):
                loss_te = train_one_step(model_te, inp, optimizer_te)
            loss_pd = train_one_step(model_pd, inp, optimizer_pd)
            assert_allclose(loss_te, loss_pd, rtol=self.rtol, atol=self.atol)

        assert (
            len(optimizer_te.state_dict()) == 4
        ), "Expect each rank to hold 4 optimizer state entries."

    def test_group_sharding_stage2(self):
        """Tests group sharding training"""
        set_random_seed(1024)
        model_te = te.Linear(self.in_channels, self.out_channels)
        model_pd = paddle.nn.Linear(self.in_channels, self.out_channels)
        model_pd.weight.copy_(model_te.weight.T, True)
        model_pd.bias.copy_(model_te.bias, True)

        model_te, optimizer_te = self._get_model_and_optimizer(model_te, stage=2)
        model_pd, optimizer_pd = self._get_model_and_optimizer(model_pd, stage=2)

        rank_id = paddle.distributed.get_rank()
        paddle.seed(rank_id)

        def train_one_step(model, inp, optimizer):
            out = model(inp)
            loss = out.mean()
            loss.backward()
            # Check gradients are split to different trainers
            if rank_id == 0:
                assert model.bias.grad is None and model.weight.grad is not None
            elif rank_id == 1:
                assert model.weight.grad is None and model.bias.grad is not None
            optimizer.step()
            optimizer.clear_grad()
            return loss

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.in_channels], self.global_dtype)
            with te.fp8_autocast(enabled=False):
                loss_te = train_one_step(model_te, inp, optimizer_te)
            loss_pd = train_one_step(model_pd, inp, optimizer_pd)
            assert_allclose(loss_te, loss_pd, rtol=self.rtol, atol=self.atol)

        assert (
            len(optimizer_te.state_dict()) == 4
        ), "Expect each rank to hold 4 optimizer state entries."

    def test_group_sharding_stage3(self):
        """Tests group sharding training"""
        set_random_seed(1024)
        model_te = te.Linear(self.in_channels, self.out_channels)
        model_pd = paddle.nn.Linear(self.in_channels, self.out_channels)
        model_pd.weight.copy_(model_te.weight.T, True)
        model_pd.bias.copy_(model_te.bias, True)

        model_te, optimizer_te = self._get_model_and_optimizer(model_te, stage=3)
        model_pd, optimizer_pd = self._get_model_and_optimizer(model_pd, stage=3)

        rank_id = paddle.distributed.get_rank()
        paddle.seed(rank_id)

        def train_one_step(model, inp, optimizer):
            out = model(inp)
            loss = out.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            return loss

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.in_channels], self.global_dtype)
            with te.fp8_autocast(enabled=False):
                loss_te = train_one_step(model_te, inp, optimizer_te)
            loss_pd = train_one_step(model_pd, inp, optimizer_pd)
            assert_allclose(loss_te, loss_pd, rtol=self.rtol, atol=self.atol)

        for name, value in optimizer_te.state_dict().items():
            if name.endswith("w_0_moment1_0"):
                assert (
                    value.numel() == self.in_channels * self.out_channels // self.sharding_degree
                ), "Expect optimizer state to be sharded across trainers."


if __name__ == "__main__":
    unittest.main()
