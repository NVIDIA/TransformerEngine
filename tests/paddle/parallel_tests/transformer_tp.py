# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Transformer layer in tensor parallel"""

import unittest

import paddle
from paddle.distributed import fleet
from paddle.distributed.fleet.layers.mpu import mp_ops

from utils import assert_allclose, set_random_seed, register_sequence_parallel_allreduce_hooks
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
        strategy.hybrid_configs["mp_configs"].need_broadcast_data = False
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()
        self.hcg = fleet.get_hybrid_communicate_group()
        self.tp_group = self.hcg.get_model_parallel_group()
        self.world_size = self.hcg.get_model_parallel_world_size()

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 1024
        self.num_heads = 16
        self.ffn_hidden_size = 4096
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.mask_type = "padding"
        self.layer_type = "encoder"
        self.global_dtype = "bfloat16"
        self.rtol = 5e-2
        self.atol = 5e-2
        self.eps = 1e-3
        self.fp8 = False
        self.sequence_parallel = False

    def _train_one_step(self, layer, inp_list, optimizer, fp8_enabled, sequence_parallel=False):
        inp, mask = inp_list
        if sequence_parallel:
            split_size = inp.shape[0] // self.world_size
            input_parallel = inp[split_size * self.rank : split_size * (self.rank + 1), :]
        else:
            input_parallel = inp
        with te.fp8_autocast(enabled=fp8_enabled):
            out = layer(input_parallel, mask)
        if sequence_parallel:
            total_out = mp_ops._c_concat(out, group=self.tp_group)
            total_out = paddle.concat(paddle.split(total_out, self.world_size, axis=-1), axis=0)
        else:
            total_out = out
        loss = total_out.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        return loss, total_out

    def test_parallel_layer(self):
        """Tests parallel Transformer"""
        set_random_seed(1024)
        common_args = [
            self.hidden_size,
            self.ffn_hidden_size,
            self.num_heads,
        ]
        common_kwargs = {
            "layernorm_epsilon": self.eps,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "self_attn_mask_type": self.mask_type,
            "layer_type": self.layer_type,
        }
        layer_tp = te.TransformerLayer(
            *common_args,
            **common_kwargs,
            set_parallel_mode=True,
            sequence_parallel=self.sequence_parallel,
        )
        layer_single = te.TransformerLayer(*common_args, **common_kwargs, set_parallel_mode=False)

        def _get_total_weight(local_weight, tp_group, axis, interleave=False):
            total_weight = []
            partial_weight = local_weight.clone().detach()
            paddle.distributed.all_gather(total_weight, partial_weight, group=tp_group)
            if interleave:
                # Due to the interleaved qkv layout, need to concat on num_head
                # dimension for column parallel linear in MultiHeadAttention layer
                assert axis == 0
                assert [
                    3 * self.hidden_size // self.world_size,
                    self.hidden_size,
                ] == partial_weight.shape
                local_num_head = self.num_heads // self.world_size
                for idx, _ in enumerate(total_weight):
                    total_weight[idx] = total_weight[idx].reshape(
                        [3, local_num_head, -1, self.hidden_size]
                    )
                total_weight = paddle.concat(total_weight, axis=1).reshape([-1, self.hidden_size])
            else:
                total_weight = paddle.concat(total_weight, axis=axis)
            return total_weight

        def _get_weight(obj, weight_names):
            for name in weight_names:
                obj = getattr(obj, name)
            return obj

        def copy_weight(layer_src, layer_dst, partition_mode, weight_names, interleave=False):
            weight_src = _get_weight(layer_src, weight_names)
            weight_dst = _get_weight(layer_dst, weight_names)
            if partition_mode is None:
                total_weight = weight_src
            elif partition_mode == "column":
                total_weight = _get_total_weight(
                    weight_src, tp_group=self.tp_group, axis=0, interleave=interleave
                )
            elif partition_mode == "row":
                total_weight = _get_total_weight(weight_src, tp_group=self.tp_group, axis=1)
            else:
                raise ValueError(f"Partition Mode {partition_mode} is not supported.")
            assert (
                weight_dst.shape == total_weight.shape
            ), f"Shapes of src:{total_weight.shape} and dst:{weight_dst.shape} do not match."
            weight_dst.copy_(total_weight, True)

        copy_weight(layer_tp, layer_single, None, ["self_attention", "layernorm_qkv", "ln_weight"])
        copy_weight(
            layer_tp,
            layer_single,
            "column",
            ["self_attention", "layernorm_qkv", "weight"],
            interleave=True,
        )
        copy_weight(layer_tp, layer_single, "row", ["self_attention", "proj", "weight"])
        copy_weight(layer_tp, layer_single, None, ["layernorm_mlp", "ln_weight"])
        copy_weight(layer_tp, layer_single, "column", ["layernorm_mlp", "fc1_weight"])
        copy_weight(layer_tp, layer_single, "row", ["layernorm_mlp", "fc2_weight"])

        if self.sequence_parallel:
            register_sequence_parallel_allreduce_hooks(layer_tp, accumulation_steps=1)

        optimizer_tp = paddle.optimizer.SGD(learning_rate=0.01, parameters=layer_tp.parameters())
        optimizer_single = paddle.optimizer.SGD(
            learning_rate=0.01, parameters=layer_single.parameters()
        )

        layer_tp = fleet.distributed_model(layer_tp)
        optimizer_tp = fleet.distributed_optimizer(optimizer_tp)

        for _ in range(5):
            inp = paddle.uniform(
                [self.batch_size, self.q_seqlen, self.hidden_size], self.global_dtype
            )
            mask = paddle.zeros(
                shape=(self.batch_size, 1, self.q_seqlen, self.kv_seqlen), dtype="bool"
            )
            loss_tp, out_tp = self._train_one_step(
                layer_tp, [inp, mask], optimizer_tp, self.fp8, self.sequence_parallel
            )
            loss_single, out_single = self._train_one_step(
                layer_single, [inp, mask], optimizer_single, self.fp8
            )
            assert_allclose(out_tp, out_single, rtol=self.rtol, atol=self.atol)
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
        self.mask_type = "padding"
        self.layer_type = "encoder"
        self.global_dtype = "bfloat16"
        self.rtol = 5e-2
        self.atol = 0.5
        self.eps = 1e-3
        self.fp8 = True
        self.sequence_parallel = False


class TestTransformerSp(TestTransformerTp):
    """Tests Transformer layer with sequence parallel in BF16"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 1024
        self.num_heads = 16
        self.ffn_hidden_size = 4096
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.mask_type = "padding"
        self.layer_type = "encoder"
        self.global_dtype = "bfloat16"
        self.rtol = 5e-2
        self.atol = 5e-2
        self.eps = 1e-3
        self.fp8 = False
        self.sequence_parallel = True


class TestTransformerSpFp8(TestTransformerSp):
    """Tests Transformer layer with sequence parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 16
        self.hidden_size = 1024
        self.num_heads = 16
        self.ffn_hidden_size = 4096
        self.q_seqlen = 128
        self.kv_seqlen = 128
        self.mask_type = "padding"
        self.layer_type = "encoder"
        self.global_dtype = "bfloat16"
        self.rtol = 5e-2
        self.atol = 0.5
        self.eps = 1e-3
        self.fp8 = True
        self.sequence_parallel = True


if __name__ == "__main__":
    unittest.main()
