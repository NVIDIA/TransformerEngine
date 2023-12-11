# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import transformer_engine.jax as te
try:
    from te.examples.mnist.single_gpu import mnist_parser, train_and_evaluate
except ModuleNotFoundError as e:
    example_path = te.examples.mnist.__file__.strip('/__init__.py')
    err_msg = f'{str(e)}. Please install TransformerEngine with `pip install .[test]` ' + \
        'or run `pip install -r {example_path}/requirements.txt`.'
    raise ModuleNotFoundError(err_msg)

def verify(actual):
        """Check If loss and accuracy match target"""
        desired_traing_loss = 0.055
        desired_traing_accuracy = 0.98
        desired_test_loss = 0.04
        desired_test_accuracy = 0.098
        assert actual[0] < desired_traing_loss
        assert actual[1] > desired_traing_accuracy
        assert actual[2] < desired_test_loss
        assert actual[3] > desired_test_accuracy

@pytest.fixture()
def config():
    yield mnist_parser(["--epochs", "5"])


class TestExampleMNIST:

    gpu_has_fp8, reason = te.fp8.is_fp8_available()

    def test_te_bf16(self, config):
        """Test Transformer Engine with BF16"""
        config.use_te = True
        config.use_fp8 = False
        actual = train_and_evaluate(config)
        self.verify(actual)

    @pytest.mark.skipif(not gpu_has_fp8, reason=reason)
    def test_te_fp8(self, config):
        """Test Transformer Engine with FP8"""
        config.use_fp8 = True
        actual = train_and_evaluate(config)
        self.verify(actual)
