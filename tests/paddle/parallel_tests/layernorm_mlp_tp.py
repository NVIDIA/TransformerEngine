# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for LayerNormMLP layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops

from utils import assert_allclose, assert_shape, set_random_seed
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
        strategy.hybrid_configs["mp_configs"].need_broadcast_data = False
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()
        self.hcg = fleet.get_hybrid_communicate_group()
        self.tp_group = self.hcg.get_model_parallel_group()
        self.world_size = self.hcg.get_model_parallel_world_size()

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 32
        self.ffn_hidden_size = 64
        self.global_dtype = "bfloat16"
        self.rtol = 1e-3
        self.atol = 1e-3
        self.eps = 1e-3
        self.fp8 = False
        self.sequence_parallel = False

    def _train_one_step(self, layer, inp, optimizer, split_input="none", gather_output=False):
        inp = paddle.to_tensor(inp, stop_gradient=True)
        assert split_input in ["none", "column", "row"]
        if split_input == "column":
            split_size = inp.shape[1] // self.world_size
            input_parallel = inp[:, split_size * self.rank : split_size * (self.rank + 1)]
        elif split_input == "row":
            split_size = inp.shape[0] // self.world_size
            input_parallel = inp[split_size * self.rank : split_size * (self.rank + 1), :]
        else:
            input_parallel = inp
        input_parallel.stop_gradient = False
        out = layer(input_parallel)
        if gather_output:
            # Need to concat on the first dim, while _c_concat concats on the last dim
            total_out = mp_ops._c_concat(out.T, group=self.tp_group).T
        else:
            total_out = out
        loss = total_out.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if split_input != "none":
            grad_input = []
            paddle.distributed.all_gather(grad_input, input_parallel.grad, group=self.tp_group)
            if split_input == "column":
                grad_input = paddle.concat(grad_input, axis=1)
            elif split_input == "row":
                grad_input = paddle.concat(grad_input, axis=0)
        else:
            grad_input = input_parallel.grad
        return loss, grad_input

    def test_parallel_layer(self):
        """Tests parallel LayerNormMLP"""
        set_random_seed(1024)
        layer_te = te.LayerNormMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            eps=self.eps,
            set_parallel_mode=True,
            sequence_parallel=self.sequence_parallel,
        )
        layer_pd = te.LayerNormMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            eps=self.eps,
            set_parallel_mode=False,
            backend="paddle",
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

        assert_shape(
            layer_te.fc1_weight,
            [self.ffn_hidden_size // self.model_parallel_size, self.hidden_size],
        )
        assert_shape(layer_te.fc1_bias, [self.ffn_hidden_size // self.model_parallel_size])
        assert_shape(
            layer_te.fc2_weight,
            [self.hidden_size, self.ffn_hidden_size // self.model_parallel_size],
        )
        assert_shape(layer_te.fc2_bias, [self.hidden_size])

        optimizer_te = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_te.parameters())
        optimizer_pd = paddle.optimizer.SGD(learning_rate=0.001, parameters=layer_pd.parameters())

        layer_te = fleet.distributed_model(layer_te)
        optimizer_te = fleet.distributed_optimizer(optimizer_te)

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.hidden_size], self.global_dtype)
            with te.fp8_autocast(enabled=self.fp8):
                loss_tp, grad_input = self._train_one_step(
                    layer_te,
                    inp,
                    optimizer_te,
                    split_input="row" if self.sequence_parallel else "none",
                    gather_output=self.sequence_parallel,
                )
            loss_ref, grad_input_ref = self._train_one_step(layer_pd, inp, optimizer_pd)
            assert_allclose(loss_tp, loss_ref, rtol=self.rtol, atol=self.atol)
            assert_allclose(grad_input, grad_input_ref, rtol=self.rtol, atol=self.atol)


class TestLayerNormMLPTpFp8(TestLayerNormMLPTp):
    """Tests LayerNormMLP layer with tensor parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 32
        self.ffn_hidden_size = 64
        self.global_dtype = "bfloat16"
        self.rtol = 1e-2
        self.atol = 1e-2
        self.eps = 1e-3
        self.fp8 = True
        self.sequence_parallel = False


class TestLayerNormMLPSp(TestLayerNormMLPTp):
    """Tests LayerNormMLP layer with sequence parallel in BF16"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 32
        self.ffn_hidden_size = 64
        self.global_dtype = "bfloat16"
        self.rtol = 1e-3
        self.atol = 1e-3
        self.eps = 1e-3
        self.fp8 = False
        self.sequence_parallel = True


class TestLayerNormMLPSpFp8(TestLayerNormMLPTp):
    """Tests LayerNormMLP layer with sequence parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 32
        self.ffn_hidden_size = 64
        self.global_dtype = "bfloat16"
        self.rtol = 1e-2
        self.atol = 1e-2
        self.eps = 1e-3
        self.fp8 = True
        self.sequence_parallel = True


if __name__ == "__main__":
    unittest.main()
