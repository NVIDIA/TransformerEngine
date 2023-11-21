# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import multiprocessing as mp
import transformer_engine.jax as te
import transformer_engine.jax.examples.encoder.multi_gpu as multi_gpu_encoder
import transformer_engine.jax.examples.encoder.model_parallel as model_parallel_encoder
import transformer_engine.jax.examples.encoder.multiprocessing as multiprocessing_encoder
from utils import is_devices_enough


@pytest.mark.skipif(not is_devices_enough(2), reason='Need at least 2 GPUs for distributed tests.')
class TestDistributedEncoder:

    gpu_has_fp8, reason = te.fp8.is_fp8_available()
    parsers  = [ multi_gpu_encoder.encoder_parser,     model_parallel_encoder.encoder_parser ]
    trainers = [ multi_gpu_encoder.train_and_evaluate, model_parallel_encoder.train_and_evaluate ]
    args = ["--epochs", "3"]

    @pytest.mark.parametrize('encoder_parser,train_and_evaluate', zip(parsers, trainers))
    def test_te_bf16(self, encoder_parser, train_and_evaluate):
        """Test Transformer Engine with BF16"""
        actual = train_and_evaluate(encoder_parser(self.args))
        assert actual[0] < 0.45 and actual[1] > 0.79

    @pytest.mark.skipif(not gpu_has_fp8, reason=reason)
    @pytest.mark.parametrize('encoder_parser,train_and_evaluate', zip(parsers, trainers))
    def test_te_fp8(self, encoder_parser, train_and_evaluate):
        """Test Transformer Engine with FP8"""
        actual = train_and_evaluate(encoder_parser(self.args))
        assert actual[0] < 0.45 and actual[1] > 0.79


@pytest.mark.skipif(not is_devices_enough(2), reason='Need at least 2 GPUs for distributed tests.')
class TestMultiprocessingEncoder:

    num_gpu, gpu_has_fp8, reason = multiprocessing_encoder.unittest_query_gpu()

    def exec(self, use_fp8):
        """Run 3 epochs for testing"""
        num_gpu = self.num_gpu
        tp_size = 2 if num_gpu > 1 and num_gpu % 2 == 0 else 1
        dp_size = num_gpu // tp_size
        batch_size = 64 // dp_size

        arg_list = []
        for i in range(num_gpu):
            args = multiprocessing_encoder.encoder_parser([])
            args.num_process = num_gpu
            args.use_fp8 = use_fp8
            args.batch_size = batch_size
            args.test_batch_size = batch_size
            args.process_id = i
            arg_list.append(args)

        with mp.Pool(self.num_gpu) as p:
            results = p.map(multiprocessing_encoder.train_and_evaluate, arg_list)

        return results

    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        results = self.exec(False)
        actual = results[0]
        assert actual[0] < 0.45 and actual[1] > 0.79

    @pytest.mark.skipif(not gpu_has_fp8, reason=reason)
    def test_te_fp8(self):
        """Test Transformer Engine with FP8"""
        results = self.exec(True)
        actual = results[0]
        assert actual[0] < 0.45 and actual[1] > 0.79