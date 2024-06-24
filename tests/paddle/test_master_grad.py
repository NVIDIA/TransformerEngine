# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TransformerLayer encoder main_grad"""

import numpy as np
import pytest

import paddle
from paddle.distributed.fleet.utils import mix_precision_utils

import transformer_engine.paddle as te
from transformer_engine.paddle.fp8 import is_fp8_available

is_fp8_supported, reason = is_fp8_available()


def create_optimizer(model, use_pure_bf16, use_main_grad):
    """Create optimizer"""
    if use_main_grad:
        assert use_pure_bf16
        model = mix_precision_utils.MixPrecisionLayer(model, dtype="bfloat16")
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        learning_rate=0.0001,
        multi_precision=use_pure_bf16,
    )
    if use_main_grad:
        optimizer = mix_precision_utils.MixPrecisionOptimizer(optimizer)

    return optimizer


class Net(paddle.nn.Layer):
    """Network use for main_grad testing"""

    def __init__(self, fuse_wgrad_accumulation):
        super().__init__()
        self.layer = te.TransformerLayer(
            4096,
            16384,
            32,
            layer_type="encoder",
            fuse_wgrad_accumulation=fuse_wgrad_accumulation,
        )

    def forward(self, inp):
        out = self.layer(inp)
        return out


def train(enable_master_grad, fuse_wgrad_accumulation=False):
    """Train function"""
    paddle.seed(10)

    accumulate_steps = 4

    if fuse_wgrad_accumulation:
        assert enable_master_grad, "fuse_wgrad_accumulation requires enable_master_grad"

    model = Net(fuse_wgrad_accumulation)

    optimizer = create_optimizer(model, use_pure_bf16=True, use_main_grad=enable_master_grad)

    loss_list = []
    for step_id in range(16):
        inp = paddle.uniform([2, 1024, 4096], dtype="float32")
        inp.stop_gradient = False
        with te.fp8_autocast(enabled=True):
            out = model(inp)
        loss = out.mean()
        loss_list.append(loss)
        loss.backward()

        # gradient accumulation
        if (step_id + 1) % accumulate_steps == 0:
            optimizer.step()
            optimizer.clear_grad()

    return loss_list


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
def test_master_grad():
    """Test main_grad"""
    paddle.set_default_dtype("float32")
    loss1 = train(enable_master_grad=False)
    loss2 = train(enable_master_grad=True)
    loss3 = train(enable_master_grad=True, fuse_wgrad_accumulation=True)

    np.testing.assert_allclose(loss1, loss2, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(loss1, loss3, rtol=1e-5, atol=1e-5)
