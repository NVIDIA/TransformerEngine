# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Linear layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops

from utils import assert_allclose, set_random_seed
import transformer_engine.paddle as te


class TestLinearTp(unittest.TestCase):
    """Tests Linear layer with column/row parallelism in BF16"""

    def setUp(self):
        self.set_attr()
        self.init_dist_env()
        paddle.set_default_dtype(self.global_dtype)

    def init_dist_env(self):
        """Init Paddle Fleet environment"""
        strategy = fleet.DistributedStrategy()
        self.model_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": self.model_parallel_size,
            "pp_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()
        self.hcg = fleet.get_hybrid_communicate_group()
        self.tp_group = self.hcg.get_model_parallel_group()
        self.world_size = self.hcg.get_model_parallel_world_size()

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.in_features = 32
        self.out_features = 64
        self.global_dtype = 'bfloat16'
        self.rtol = 1e-3
        self.atol = 1e-3
        self.fp8 = False

    def test_column_parallel_layer(self):
        """Tests column parallel linear"""
        set_random_seed(1024)
        layer_te = te.Linear(
            self.in_features,
            self.out_features,
            parallel_mode='column',
        )
        layer_pd = te.Linear(
            self.in_features,
            self.out_features,
            backend='paddle',
        )
        # Get total weight
        total_weight = []
        partial_weight = layer_te.weight.clone().detach()
        paddle.distributed.all_gather(total_weight, partial_weight, group=self.tp_group)
        total_weight = paddle.concat(total_weight, axis=0)
        assert total_weight.T.shape == layer_pd.weight.shape, \
                f"Shapes of src:{total_weight.T.shape} and " \
                f"dst:{layer_pd.weight.shape} do not match."
        layer_pd.weight.copy_(total_weight.T, True)

        optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
        optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

        layer_te = fleet.distributed_model(layer_te)
        optimizer_te = fleet.distributed_optimizer(optimizer_te)

        def train_one_step(layer, inp, optimizer, gather=False):
            inp = paddle.to_tensor(inp)
            inp.stop_gradient = False
            out = layer(inp)
            if gather:
                total_out = mp_ops._c_concat(out, group=self.tp_group)
            else:
                total_out = out
            loss = total_out.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            return loss, inp.grad

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.in_features], self.global_dtype)
            with te.fp8_autocast(enabled=self.fp8):
                loss_tp, grad_input = train_one_step(layer_te, inp, optimizer_te, gather=True)
            loss_ref, grad_input_ref = train_one_step(layer_pd, inp, optimizer_pd)
            assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
            assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)

    def test_row_parallel_layer(self):
        """Tests row parallel linear"""
        set_random_seed(1024)
        layer_te = te.Linear(
            self.in_features,
            self.out_features,
            parallel_mode='row',
        )
        layer_pd = te.Linear(
            self.in_features,
            self.out_features,
            backend='paddle',
        )
        # Get total weight
        total_weight = []
        partial_weight = layer_te.weight.clone().detach()
        paddle.distributed.all_gather(total_weight, partial_weight, group=self.tp_group)
        total_weight = paddle.concat(total_weight, axis=1)
        assert total_weight.T.shape == layer_pd.weight.shape, \
                f"Shapes of src:{total_weight.shape} and dst:{layer_pd.weight.shape} do not match."
        layer_pd.weight.copy_(total_weight.T, True)

        optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
        optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

        # Note(tizheng): For this test, we cannot wrap model with fleet.distributed_model,
        # because it will broadcast inputs across mp group. However, RPL expects splitted
        # inputs, which is different on each rank.

        def train_one_step(layer, inp, optimizer, split=False):
            inp = paddle.to_tensor(inp, stop_gradient=True)
            if split:
                # TODO(tizheng): Why not working?
                # issue: https://github.com/PaddlePaddle/Paddle/issues/55565
                # input_parallel = mp_ops._c_split(inp, group=layer.tp_group)
                split_size = inp.shape[1] // self.world_size
                input_parallel = inp[:, split_size * self.rank:split_size * (self.rank + 1)]
            else:
                input_parallel = inp
            input_parallel.stop_gradient = False
            out = layer(input_parallel)
            loss = out.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            if split:
                grad_input = []
                paddle.distributed.all_gather(grad_input, input_parallel.grad, group=self.tp_group)
                grad_input = paddle.concat(grad_input, axis=1)
            else:
                grad_input = input_parallel.grad
            return loss, grad_input

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.in_features], self.global_dtype)
            with te.fp8_autocast(enabled=self.fp8):
                loss_tp, grad_input = train_one_step(layer_te, inp, optimizer_te, split=True)
            loss_ref, grad_input_ref = train_one_step(layer_pd, inp, optimizer_pd)
            assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
            assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)


class TestLinearTpFP8(TestLinearTp):
    """Tests Linear layer with column/row parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.in_features = 32
        self.out_features = 64
        self.global_dtype = 'bfloat16'
        self.rtol = 1e-2
        self.atol = 1e-2
        self.fp8 = True


if __name__ == '__main__':
    unittest.main()
