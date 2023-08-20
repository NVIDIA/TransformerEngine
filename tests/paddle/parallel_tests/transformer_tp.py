# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Transformer layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet

from utils import assert_allclose, set_random_seed
import transformer_engine.paddle as te


class TestTransformerTp(unittest.TestCase):
    """Tests Transformer layer with model parallel in BF16"""

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
        self.hidden_size = 1024
        self.num_heads = 16
        self.ffn_hidden_size = 4096
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.mask_type = 'padding'
        self.layer_type = 'encoder'
        self.global_dtype = 'bfloat16'
        self.rtol = 5e-2
        self.atol = 5e-2
        self.eps = 1e-3
        self.fp8 = False

    def test_parallel_layer(self):
        """Tests parallel Transformer"""
        set_random_seed(1024)
        common_args = [
            self.hidden_size,
            self.ffn_hidden_size,
            self.num_heads,
        ]
        common_kwargs = {
            'layernorm_epsilon': self.eps,
            'hidden_dropout': 0.0,
            'attention_dropout': 0.0,
            'self_attn_mask_type': self.mask_type,
            'layer_type': self.layer_type,
        }
        layer_tp = te.TransformerLayer(*common_args, **common_kwargs, set_parallel_mode=True)
        layer_single = te.TransformerLayer(*common_args, **common_kwargs, set_parallel_mode=False)

        def _get_total_weight(local_weight, tp_group, axis):
            total_weight = []
            partial_weight = local_weight.clone().detach()
            paddle.distributed.all_gather(total_weight, partial_weight, group=tp_group)
            total_weight = paddle.concat(total_weight, axis=axis)
            return total_weight

        def _get_weight(obj, weight_names):
            for name in weight_names:
                obj = getattr(obj, name)
            return obj

        def copy_weight(layer_src, layer_dst, partition_mode, weight_names):
            weight_src = _get_weight(layer_src, weight_names)
            weight_dst = _get_weight(layer_dst, weight_names)
            if partition_mode is None:
                total_weight = weight_src
            elif partition_mode == 'column':
                total_weight = _get_total_weight(weight_src, tp_group=self.tp_group, axis=0)
            elif partition_mode == 'row':
                total_weight = _get_total_weight(weight_src, tp_group=self.tp_group, axis=1)
            else:
                raise ValueError(f"Partition Mode {partition_mode} is not supported.")
            assert weight_dst.shape == total_weight.shape, \
                    f"Shapes of src:{total_weight.shape} and dst:{weight_dst.shape} do not match."
            weight_dst.copy_(total_weight, True)

        copy_weight(layer_tp, layer_single, None, ['self_attention', 'layernorm_qkv', 'ln_weight'])
        copy_weight(layer_tp, layer_single, 'column', ['self_attention', 'layernorm_qkv', 'weight'])
        copy_weight(layer_tp, layer_single, 'row', ['self_attention', 'proj', 'weight'])
        copy_weight(layer_tp, layer_single, None, ['layernorm_mlp', 'ln_weight'])
        copy_weight(layer_tp, layer_single, 'column', ['layernorm_mlp', 'fc1_weight'])
        copy_weight(layer_tp, layer_single, 'row', ['layernorm_mlp', 'fc2_weight'])

        optimizer_tp = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer_tp.parameters())
        optimizer_single = paddle.optimizer.SGD(learning_rate=0.1,
                                                parameters=layer_single.parameters())

        layer_tp = fleet.distributed_model(layer_tp)
        optimizer_tp = fleet.distributed_optimizer(optimizer_tp)

        def train_one_step(layer, inp_list, optimizer, fp8_enabled):
            with te.fp8_autocast(enabled=fp8_enabled):
                out = layer(*inp_list)
            loss = out.mean()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            return loss

        for _ in range(5):
            inp = paddle.uniform([self.batch_size, self.q_seqlen, self.hidden_size],
                                 self.global_dtype)
            mask = paddle.ones(shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen),
                               dtype='bool')
            loss_tp = train_one_step(layer_tp, [inp, mask], optimizer_tp, self.fp8)
            loss_single = train_one_step(layer_single, [inp, mask], optimizer_single, self.fp8)
            assert_allclose(loss_tp, loss_single, rtol=self.rtol, atol=self.atol)


class TestTransformerTpFp8(TestTransformerTp):
    """Tests Transformer layer with tensor parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 1024
        self.num_heads = 16
        self.ffn_hidden_size = 4096
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.mask_type = 'padding'
        self.layer_type = 'encoder'
        self.global_dtype = 'bfloat16'
        self.rtol = 5e-2
        self.atol = 5e-2
        self.eps = 1e-3
        self.fp8 = True


if __name__ == '__main__':
    unittest.main()
