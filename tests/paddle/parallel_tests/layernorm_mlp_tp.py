# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for LayerNormMLP layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, set_random_seed
import transformer_engine.paddle as te


class TestLayerNormMLPTp(unittest.TestCase):
    """Tests LayerNormMLP layer with model parallel in BF16"""

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
        self.hidden_size = 32
        self.ffn_hidden_size = 64
        self.global_dtype = 'bfloat16'
        self.rtol = 1e-3
        self.atol = 1e-3
        self.eps = 1e-3
        self.fp8 = False

    def test_parallel_layer(self):
        """Tests parallel LayerNormMLP"""
        set_random_seed(1024)
        layer_te = te.LayerNormMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            eps=self.eps,
            set_parallel_mode=True,
        )
        layer_pd = te.LayerNormMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            eps=self.eps,
            set_parallel_mode=False,
            backend='paddle',
        )

        def _get_total_weight(local_weight, tp_group, axis):
            total_weight = []
            partial_weight = local_weight.clone().detach()
            paddle.distributed.all_gather(total_weight, partial_weight, group=tp_group)
            total_weight = paddle.concat(total_weight, axis=axis)
            return total_weight

        # Get total weight
        total_fc1_weight = _get_total_weight(layer_te.fc1_weight, tp_group=self.tp_group, axis=0)
        total_fc2_weight = _get_total_weight(layer_te.fc2_weight, tp_group=self.tp_group, axis=1)
        layer_pd.fc1_weight.copy_(total_fc1_weight.T, True)
        layer_pd.fc2_weight.copy_(total_fc2_weight.T, True)

        optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
        optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

        layer_te = fleet.distributed_model(layer_te)
        optimizer_te = fleet.distributed_optimizer(optimizer_te)

        def train_one_step(layer, inp, optimizer):
            inp = paddle.to_tensor(inp)
            inp.stop_gradient = False
            out = layer(inp)
            loss = out.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            return loss, inp.grad

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.hidden_size], self.global_dtype)
            with te.fp8_autocast(enabled=self.fp8):
                loss_tp, grad_input = train_one_step(layer_te, inp, optimizer_te)
            loss_ref, grad_input_ref = train_one_step(layer_pd, inp, optimizer_pd)
            assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
            assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)


class TestLayerNormMLPTpFp8(TestLayerNormMLPTp):
    """Tests LayerNormMLP layer with tensor parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 32
        self.ffn_hidden_size = 64
        self.global_dtype = 'bfloat16'
        self.rtol = 1e-2
        self.atol = 1e-2
        self.eps = 1e-3
        self.fp8 = True


if __name__ == '__main__':
    unittest.main()
