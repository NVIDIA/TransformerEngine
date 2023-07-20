# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for group sharding"""

import unittest

import paddle
from paddle.distributed import fleet

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
        self.global_dtype = 'float32'
        self.rtol = 1e-5
        self.atol = 1e-5
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
        fleet.init(is_collective=True, strategy=strategy)

    def test_group_sharding(self):
        """Tests group sharding training"""
        set_random_seed(1024)
        model_te = te.Linear(16, 16)
        model_pd = paddle.nn.Linear(16, 16)
        model_pd.weight.copy_(model_te.weight.T, True)
        model_pd.bias.copy_(model_te.bias, True)

        optimizer_te = paddle.optimizer.AdamW(learning_rate=0.01, parameters=model_te.parameters())
        optimizer_pd = paddle.optimizer.AdamW(learning_rate=0.01, parameters=model_pd.parameters())

        model_te = fleet.distributed_model(model_te)
        model_pd = fleet.distributed_model(model_pd)
        optimizer_te = fleet.distributed_optimizer(optimizer_te)
        optimizer_pd = fleet.distributed_optimizer(optimizer_pd)

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
            inp = paddle.uniform([16, 16], self.global_dtype)
            with te.fp8_autocast(enabled=False):
                loss_te = train_one_step(model_te, inp, optimizer_te)
            loss_pd = train_one_step(model_pd, inp, optimizer_pd)
            assert_allclose(loss_te, loss_pd, rtol=self.rtol, atol=self.atol)


if __name__ == '__main__':
    unittest.main()
