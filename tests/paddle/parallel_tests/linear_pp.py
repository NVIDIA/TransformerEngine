# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Unittest for Linear layer in pipeline parallel"""

import unittest

import numpy as np

import paddle
from paddle.distributed import fleet

from paddle.distributed.fleet.meta_parallel import (
    LayerDesc,
    PipelineLayer,
)

from utils import assert_allclose, set_random_seed
import transformer_engine.paddle as te


class TELinear(te.Linear):
    """To pass is_first_microbatch"""

    def __init__(self, *args, **kwargs):
        assert "accumulate_steps" in kwargs
        self.accumulate_steps = kwargs["accumulate_steps"]
        del kwargs["accumulate_steps"]
        self._micro_batch_id = 0
        super().__init__(*args, **kwargs)

    def forward(self, *args, **kwargs):
        kwargs["is_first_microbatch"] = (self._micro_batch_id % self.accumulate_steps) == 0
        if paddle.is_grad_enabled() and self.training:
            self._micro_batch_id += 1
        return super().forward(*args, **kwargs)


class TEPipelineModel(PipelineLayer):
    """Model for pipeline parallel test"""

    def __init__(
        self,
        in_features,
        hidden_features,
        weight_attrs,
        use_te=True,
        use_fp8=False,
        accumulate_steps=1,
        **kwargs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.fp8 = use_fp8
        hcg = fleet.get_hybrid_communicate_group()
        self.dp_group = hcg.get_data_parallel_group()

        Linear = TELinear if use_te else paddle.nn.Linear
        extra_kwargs = {}
        if use_te:
            extra_kwargs["accumulate_steps"] = accumulate_steps

        model_desc = [
            LayerDesc(
                Linear,
                self.in_features,
                self.hidden_features,
                weight_attr=weight_attrs[0],
                **extra_kwargs,
            ),
            LayerDesc(
                Linear,
                self.hidden_features,
                self.in_features,
                weight_attr=weight_attrs[1],
                **extra_kwargs,
            ),
        ]
        super().__init__(layers=model_desc, loss_fn=paddle.nn.CrossEntropyLoss(), **kwargs)

    def forward(self, *args, **kwargs):
        with te.fp8_autocast(enabled=self.fp8, fp8_group=self.dp_group):
            return super().forward(*args, **kwargs)


class StandaloneModel(paddle.nn.Layer):
    """Model for pipeline parallel test"""

    def __init__(self, in_features, hidden_features, weight_attrs):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        Linear = paddle.nn.Linear
        self.layer = paddle.nn.Sequential(
            Linear(self.in_features, self.hidden_features, weight_attr=weight_attrs[0]),
            Linear(self.hidden_features, self.in_features, weight_attr=weight_attrs[1]),
        )
        self.loss = paddle.nn.CrossEntropyLoss()

    def forward(self, inp):
        out = self.layer(inp[0])
        loss = self.loss(out, inp[1])
        return loss


class TestLinearPipelineParallel(unittest.TestCase):
    """Tests Linear layer with pipeline parallel"""

    def setUp(self):
        self.set_attr()
        self.init_dist_env()
        paddle.set_default_dtype(self.global_dtype)

    def init_dist_env(self):
        """Init Paddle Fleet environment"""
        strategy = fleet.DistributedStrategy()
        self.pipeline_parallel_size = 2
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": self.pipeline_parallel_size,
        }
        self.accumulate_steps = self.batch_size // self.micro_batch_size
        strategy.pipeline_configs = {
            "accumulate_steps": self.accumulate_steps,
            "micro_batch_size": self.micro_batch_size,
        }
        fleet.init(is_collective=True, strategy=strategy)
        self.rank = fleet.worker_index()
        self.hcg = fleet.get_hybrid_communicate_group()

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 32
        self.micro_batch_size = 16
        self.in_features = 32
        self.hidden_features = 64
        self.global_dtype = "float32"
        self.rtol = 1e-5
        self.atol = 1e-5
        self.iter = 10
        self.fp8 = False

    def test_pipeline_train(self):
        """Test pipeline parallel training"""
        set_random_seed(1024)
        np.random.seed(1024)

        weight1_np = np.random.normal(size=[self.in_features, self.hidden_features])
        weight2_np = np.random.normal(size=[self.hidden_features, self.in_features])
        weight_attrs = [
            paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(weight1_np)),
            paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(weight2_np)),
        ]
        weight_attrs_transposed = [
            paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(weight1_np.T)),
            paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(weight2_np.T)),
        ]

        pipe_model = TEPipelineModel(
            self.in_features,
            self.hidden_features,
            weight_attrs_transposed,
            use_te=True,
            use_fp8=self.fp8,
            seg_method="layer:Linear",
            num_stages=self.pipeline_parallel_size,
            accumulate_steps=self.accumulate_steps,
        )

        # Check if model is split across ranks as expected
        for name, sublayer in pipe_model.named_sublayers():
            if name in ("_loss_fn", "shared_layers"):
                continue
            if self.rank == 0:
                assert tuple(sublayer.weight.shape) == weight1_np.T.shape, (
                    f"Shape does not match, expect: {weight1_np.T.shape}, "
                    f"actual: {tuple(sublayer.weight.shape)}"
                )
            elif self.rank == 1:
                assert tuple(sublayer.weight.shape) == weight2_np.T.shape, (
                    f"Shape does not match, expect: {weight2_np.T.shape}, "
                    f"actual: {tuple(sublayer.weight.shape)}"
                )

        standalone_model = StandaloneModel(
            self.in_features,
            self.hidden_features,
            weight_attrs,
        )

        optimizer_te = paddle.optimizer.SGD(learning_rate=0.1, parameters=pipe_model.parameters())
        optimizer_pd = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=standalone_model.parameters()
        )

        pipe_model = fleet.distributed_model(pipe_model)
        optimizer_te = fleet.distributed_optimizer(optimizer_te)

        def train_one_step(layer, inp, optimizer):
            loss = layer(inp)
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            return loss

        for i in range(self.iter):
            inp = paddle.to_tensor(
                np.random.normal(size=[self.batch_size, self.in_features]), dtype=self.global_dtype
            )
            label = paddle.to_tensor(np.random.randint(self.in_features, size=[self.batch_size, 1]))
            loss_te = pipe_model.train_batch([inp, label], optimizer_te)
            loss_pd = train_one_step(standalone_model, [inp, label], optimizer_pd)
            print(f"Iter: {i}, loss_te: {loss_te.item()}, loss_pd: {loss_pd.item()}")
            assert_allclose(loss_te, loss_pd, rtol=self.rtol, atol=self.atol)


class TestLinearPipelineParallelFP8(TestLinearPipelineParallel):
    """Tests Linear layer with column/row parallelism in FP8"""

    def set_attr(self):
        """Set test configs"""
        self.batch_size = 32
        self.micro_batch_size = 16
        self.in_features = 32
        self.hidden_features = 64
        self.global_dtype = "float32"
        self.rtol = 5e-2
        self.atol = 5e-2
        self.iter = 10
        self.fp8 = True


if __name__ == "__main__":
    unittest.main()
