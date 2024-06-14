# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Linear layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, set_random_seed
import transformer_engine.paddle as te


def assert_allclose_across_ranks(tensor, group=None):
    """Assert tensor is identical in all ranks"""
    gathered_list = []
    paddle.distributed.all_gather(gathered_list, tensor, group=group)
    assert len(gathered_list) > 1
    for gathered_tensor in gathered_list:
        assert_allclose(tensor, gathered_tensor)


class TestAmaxReduction(unittest.TestCase):
    """Tests Amax reduction"""

    def setUp(self):
        self.data_parallel_size = 2
        self.init_dist_env()
        self.global_dtype = "bfloat16"
        paddle.set_default_dtype(self.global_dtype)

    def init_dist_env(self):
        """Init Paddle Fleet environment"""
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": self.data_parallel_size,
            "mp_degree": 1,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

    def test_amax_reduction(self):
        """Tests column parallel linear"""
        set_random_seed(1024)
        layer1 = te.Linear(16, 16)
        layer2 = te.Linear(16, 16)
        model = paddle.nn.Sequential(layer1, layer2)
        model = fleet.distributed_model(model)

        rank_id = paddle.distributed.get_rank()
        set_random_seed(rank_id)

        optimizer = paddle.optimizer.SGD(learning_rate=10.0, parameters=model.parameters())
        optimizer = fleet.distributed_optimizer(optimizer)

        def train_one_step(layer, inp, optimizer):
            inp = paddle.to_tensor(inp)
            inp.stop_gradient = False
            out = layer(inp)
            loss = out.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            return loss

        for _ in range(5):
            inp = paddle.uniform([16, 16], self.global_dtype)
            with te.fp8_autocast(enabled=True):
                train_one_step(model, inp, optimizer)

            assert_allclose_across_ranks(layer1.fp8_meta["scaling_fwd"].amax_history[-1])
            assert_allclose_across_ranks(layer1.fp8_meta["scaling_fwd"].scale)
            assert_allclose_across_ranks(layer1.fp8_meta["scaling_fwd"].scale_inv)
            assert_allclose_across_ranks(layer2.fp8_meta["scaling_fwd"].amax_history[-1])
            assert_allclose_across_ranks(layer2.fp8_meta["scaling_fwd"].scale)
            assert_allclose_across_ranks(layer2.fp8_meta["scaling_fwd"].scale_inv)
            assert_allclose_across_ranks(layer1.fp8_meta["scaling_bwd"].amax_history[-1])
            assert_allclose_across_ranks(layer1.fp8_meta["scaling_bwd"].scale)
            assert_allclose_across_ranks(layer1.fp8_meta["scaling_bwd"].scale_inv)
            assert_allclose_across_ranks(layer2.fp8_meta["scaling_bwd"].amax_history[-1])
            assert_allclose_across_ranks(layer2.fp8_meta["scaling_bwd"].scale)
            assert_allclose_across_ranks(layer2.fp8_meta["scaling_bwd"].scale_inv)


if __name__ == "__main__":
    unittest.main()
