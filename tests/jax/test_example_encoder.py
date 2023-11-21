# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import transformer_engine.jax as te
try:
    from te.examples.encoder.single_gpu import encoder_parser, train_and_evaluate
except ModuleNotFoundError as e:
    example_path = te.examples.encoder.__file__.strip('/__init__.py')
    err_msg = f'{str(e)}. Please install TransformerEngine with `pip install .[test]` ' + \
        'or run `pip install -r {example_path}/requirements.txt`.'
    raise ModuleNotFoundError(err_msg)

@pytest.fixture()
def config():
    yield encoder_parser(["--epochs", "3"])


class TestExampleEncoder:

    gpu_has_fp8, reason = te.fp8.is_fp8_available()

    def test_te_bf16(self, config):
        """Test Transformer Engine with BF16"""
        actual = train_and_evaluate(config)
        assert actual[0] < 0.45 and actual[1] > 0.79

    @pytest.mark.skipif(not gpu_has_fp8, reason=reason)
    def test_te_fp8(self, config):
        """Test Transformer Engine with FP8"""
        config.use_fp8 = True
        actual = train_and_evaluate(config)
        assert actual[0] < 0.45 and actual[1] > 0.79
