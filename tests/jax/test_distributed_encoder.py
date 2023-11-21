# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import multiprocessing as mp
import transformer_engine.jax as te
import transformer_engine.jax.examples.encoder.data_parallel as data_parallel
import transformer_engine.jax.examples.encoder.model_parallel as model_parallel
import transformer_engine.jax.examples.encoder.data_model_parallel as data_model_parallel


class TestMultiGPUEncoder:

    gpu_has_fp8, reason = te.fp8.is_fp8_available()
    parsers  = [ data_parallel.encoder_parser,     model_parallel.encoder_parser ]
    trainers = [ data_parallel.train_and_evaluate, model_parallel.train_and_evaluate ]
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


class TestMultiGPUEncoderDPTP:

    num_gpu, gpu_has_fp8, reason = data_model_parallel.unittest_query_gpu()

    def exec(self, use_fp8):
        """Run 3 epochs for testing"""
        num_gpu = self.num_gpu
        tp_size = 2 if num_gpu > 1 and num_gpu % 2 == 0 else 1
        dp_size = num_gpu // tp_size
        batch_size = 64 // dp_size

        arg_list = []
        for i in range(num_gpu):
            args = data_model_parallel.encoder_parser([])
            args.num_process = num_gpu
            args.use_fp8 = use_fp8
            args.batch_size = batch_size
            args.test_batch_size = batch_size
            args.process_id = i
            arg_list.append(args)

        with mp.Pool(self.num_gpu) as p:
            results = p.map(data_model_parallel.train_and_evaluate, arg_list)

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