# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for LayerNormLinear layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops

from utils import assert_allclose, assert_shape, set_random_seed
import transformer_engine.paddle as te


class TestLayerNormLinearTp(unittest.TestCase):
    """Tests LayerNormLinear layer with column/row parallelism in BF16"""

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
        self.hcg = fleet.get_hybrid_communicate_group()
        self.tp_group = self.hcg.get_model_parallel_group()

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.in_features = 32
        self.out_features = 64
        self.global_dtype = 'bfloat16'
        self.rtol = 1e-3
        self.atol = 1e-3
        self.eps = 1e-3
        self.fp8 = False

    def test_column_parallel_layer(self):
        """Tests column parallel LayerNormLinear"""
        set_random_seed(1024)
        layer_te = te.LayerNormLinear(
            self.in_features,
            self.out_features,
            eps=self.eps,
            parallel_mode='column',
        )
        layer_pd = te.LayerNormLinear(
            self.in_features,
            self.out_features,
            eps=self.eps,
            backend='paddle',
        )
        # Get total weight
        total_weight = []
        partial_weight = layer_te.weight.clone().detach()
        paddle.distributed.all_gather(total_weight, partial_weight, group=self.tp_group)
        total_weight = paddle.concat(total_weight, axis=0)
        layer_pd.weight.copy_(total_weight.T, True)

        assert_shape(layer_te.weight,
                     [self.out_features // self.model_parallel_size, self.in_features])
        assert_shape(layer_te.bias, [self.out_features // self.model_parallel_size])

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


class TestLayerNormLinearTpFp8(TestLayerNormLinearTp):
    """Tests LayernormLinear layer with column/row parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.in_features = 32
        self.out_features = 64
        self.global_dtype = 'bfloat16'
        self.rtol = 1e-2
        self.atol = 1e-2
        self.eps = 1e-3
        self.fp8 = True


if __name__ == '__main__':
    unittest.main()
