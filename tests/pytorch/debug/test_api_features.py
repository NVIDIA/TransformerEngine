# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor, Float8Quantizer

import nvdlfw_inspect.api as debug_api

try:
    import transformer_engine
    import transformer_engine_torch as tex
except (ImportError, ModuleNotFoundError):
    print("Could not find TransformerEngine package.")
    exit(1)


def test_transformer_engine_no_config(feature_dirs):
    debug_api.initialize("", feature_dirs=feature_dirs)
    try:

        tensor = torch.rand(24, 2046).cuda()

        # FP8 enabled - true by the default
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="fprop", iteration=0
        )

        # modify_tensor_enabled - False by default
        assert not debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.attn.qkv", gemm="fprop", tensor_name="activation", iteration=0
        )

        # inspect_tensor_enabled - False by default
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.1.attn.qkv", tensor_name="activation", iteration=0
        )

        # inspect_tensor_postquantize - False by default
        assert not debug_api.transformer_engine.inspect_tensor_postquantize_enabled(
            "decoder.1.attn.qkv", gemm="fprop", tensor_name="activation", iteration=0
        )

    finally:
        debug_api.end_debug()


def test_disable_fp8_gemm(configs_dir, feature_dirs):
    try:
        debug_api.initialize(configs_dir + "disable_fp8_gemms.yaml", feature_dirs=feature_dirs)

        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="fprop", iteration=0
        )
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="dgrad", iteration=0
        )
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="wgrad", iteration=0
        )

        # caching
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="fprop", iteration=0
        )
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="dgrad", iteration=0
        )
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="wgrad", iteration=0
        )

    finally:
        debug_api.end_debug()


def test_disable_fp8_layer(configs_dir, feature_dirs):
    try:
        debug_api.initialize(configs_dir + "disable_fp8_layer.yaml", feature_dirs=feature_dirs)

        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.mlp.fc1", gemm="fprop", iteration=0
        )
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.mlp.fc1", gemm="wgrad", iteration=0
        )
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.mlp.fc1", gemm="dgrad", iteration=0
        )
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="fprop", iteration=0
        )
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="wgrad", iteration=0
        )
        assert not debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.attn.qkv", gemm="dgrad", iteration=0
        )

    finally:
        debug_api.end_debug()


def test_per_tensor_scaling(configs_dir, feature_dirs):
    try:

        debug_api.initialize(configs_dir + "per_tensor_scaling.yaml", feature_dirs=feature_dirs)

        tensor = torch.rand(24, 2046).cuda()

        # check modify_tensor_enabled
        assert debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="activation", iteration=0
        )
        assert debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="weight", iteration=0
        )
        assert debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="gradient", iteration=0
        )
        assert not debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="weight", iteration=0
        )
        assert not debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor_name="gradient", iteration=0
        )
        assert not debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="wgrad", tensor_name="activation", iteration=0
        )

        # check modify_tensor

        default_quantizer1 = Float8Quantizer(
            scale=torch.tensor([1]).cuda(),
            amax=torch.tensor([0]).cuda(),
            fp8_dtype=tex.DType.kFloat8E4M3,
        )
        default_quantizer2 = Float8Quantizer(
            scale=torch.tensor([1]).cuda(),
            amax=torch.tensor([0]).cuda(),
            fp8_dtype=tex.DType.kFloat8E5M2,
        )

        output1 = debug_api.transformer_engine.modify_tensor(
            layer_name="decoder.1.mlp.fc1",
            gemm="fprop",
            tensor_name="activation",
            default_quantizer=default_quantizer1,
            iteration=0,
            tensor=tensor,
        )
        assert type(output1) == Float8Tensor
        assert output1._fp8_dtype == tex.DType.kFloat8E4M3

        output2 = debug_api.transformer_engine.modify_tensor(
            "decoder.1.mlp.fc1",
            gemm="dgrad",
            tensor=tensor,
            tensor_name="gradient",
            default_quantizer=default_quantizer2,
            iteration=0,
        )
        assert type(output2) == Float8Tensor
        assert output2._fp8_dtype == tex.DType.kFloat8E5M2

        assert not debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1",
            gemm="wgrad",
            tensor_name="gradient",
            iteration=0,
        )

        assert not debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc4",
            gemm="fprop",
            tensor_name="activation",
            iteration=0,
        )
    finally:
        debug_api.end_debug()


def test_fake_quant(configs_dir, feature_dirs):
    try:
        debug_api.initialize(
            configs_dir + "fake_quantization_config.yaml", feature_dirs=feature_dirs
        )

        tensor = torch.rand(24, 2046).cuda()

        # modify_tensor_enabled
        assert debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="fprop", tensor_name="activation", iteration=0
        )

        assert debug_api.transformer_engine.modify_tensor_enabled(
            "decoder.1.mlp.fc1", gemm="dgrad", tensor_name="gradient", iteration=0
        )

        # modify_tensor
        debug_api.transformer_engine.modify_tensor(
            "decoder.1.mlp.fc1",
            gemm="fprop",
            tensor=tensor,
            tensor_name="activation",
            iteration=0,
            default_quantizer=None,
        )

        debug_api.transformer_engine.modify_tensor(
            "decoder.1.mlp.fc1",
            gemm="dgrad",
            tensor=tensor,
            tensor_name="gradient",
            iteration=0,
            default_quantizer=None,
        )

        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.fc2", gemm="wgrad", iteration=0
        )
        # caching
        assert debug_api.transformer_engine.fp8_gemm_enabled(
            "decoder.1.fc2", gemm="wgrad", iteration=0
        )
    finally:
        debug_api.end_debug()


def test_statistics_collection(configs_dir, feature_dirs):
    try:
        debug_api.initialize(
            config_file=configs_dir + "stats_collection_test_config.yaml",
            feature_dirs=feature_dirs,
            default_logging_enabled=False,
        )

        tensor = torch.randn((100, 100, 5)).cuda()
        tensor_fp8 = Float8Tensor(
            data=tensor.to(torch.uint8).cuda(),
            fp8_scale_inv=torch.full([1], 1.0).cuda(),
            fp8_dtype=tex.DType.kFloat8E4M3,
            shape=tensor.shape,
            dtype=torch.float32,
        )

        def log():
            from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS

            return STATS_BUFFERS.log_stats()

        def assert_empty():
            stats = log()
            assert len(stats) == 0

        # TE tensor stats --
        debug_api.transformer_engine.inspect_tensor(
            "decoder.1.mlp.fc1",
            tensor=tensor,
            tensor_name="activation",
            iteration=200,
            tp_group=None,
        )
        stats = log()
        assert stats[("decoder.1.mlp.fc1", "activation", "cur_amax", 200)] == tensor.abs().max()
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.1.mlp.fc1", tensor_name="activation", iteration=201
        )
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.2.mlp.fc1", tensor_name="activation", iteration=200
        )
        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.1.mlp.fc1", tensor_name="gradient", iteration=200
        )

        expected_underflows = (tensor_fp8._data == 0).sum() * 100 / (100 * 100 * 5)
        expected_overflows = (tensor_fp8._data == 126).sum() * 100 / (100 * 100 * 5)

        # TE FP8 tensor stats --
        assert debug_api.transformer_engine.inspect_tensor_postquantize_enabled(
            "decoder.1.mlp.fc1", tensor_name="gradient", gemm="wgrad", iteration=200
        )
        debug_api.transformer_engine.inspect_tensor_postquantize(
            "decoder.1.mlp.fc1",
            tensor=tensor_fp8,
            tensor_name="gradient",
            iteration=200,
            rowwise=True,
            tp_group=None,
        )
        stats = log()
        torch.testing.assert_close(
            stats[("decoder.1.mlp.fc1", "gradient", "underflows%", 200)], expected_underflows
        )

        assert not debug_api.transformer_engine.inspect_tensor_postquantize_enabled(
            "decoder.1.mlp.fc1", tensor_name="activation", gemm="fprop", iteration=201
        )
        assert not debug_api.transformer_engine.inspect_tensor_postquantize_enabled(
            "decoder.2.mlp.fc1", tensor_name="gradient", gemm="wgrad", iteration=200
        )

        # Second config in same yaml
        tensor = torch.rand((100, 100, 5))
        debug_api.transformer_engine.inspect_tensor(
            "decoder.6.mlp.fc1",
            tensor=tensor,
            tensor_name="activation",
            iteration=200,
            tp_group=None,
        )
        stats = log()
        stats_names = [x[3] for x in stats.keys()]
        all(s in stats_names for s in ["cur_amax", "dynamic_range", "mean", "std", "l1_norm"])
        assert stats[("decoder.6.mlp.fc1", "activation", "mean", 200)] == tensor.mean()

        debug_api.transformer_engine.inspect_tensor(
            "decoder.7.mlp.fc1",
            tensor=tensor,
            tensor_name="weight",
            iteration=200,
            tp_group=None,
        )
        stats = log()
        stats_names = [x[3] for x in stats.keys()]
        all(s in stats_names for s in ["mean", "std", "l1_norm", "min", "max"])
        assert stats[("decoder.7.mlp.fc1", "weight", "max", 200)] == tensor.max()

        assert not debug_api.transformer_engine.inspect_tensor_enabled(
            "decoder.7.mlp.fc1", tensor_name="weight", iteration=201
        )
        assert_empty()

    finally:
        debug_api.end_debug()


def test_statistics_multi_run(configs_dir, feature_dirs):
    try:
        debug_api.initialize(
            config_file=configs_dir + "stats_collection_test_config.yaml",
            feature_dirs=feature_dirs,
            default_logging_enabled=False,
        )

        def feed(tensor, tensor_fp8):
            debug_api.transformer_engine.inspect_tensor(
                "decoder.5.mlp.fc1",
                tensor=tensor,
                tensor_name="activation",
                iteration=1,
                tp_group=None,
            )
            debug_api.transformer_engine.inspect_tensor_postquantize(
                "decoder.5.mlp.fc1",
                tensor=tensor_fp8,
                tensor_name="activation",
                iteration=1,
                rowwise=True,
                tp_group=None,
            )

        def log_stats():
            from transformer_engine.debug.features.utils.stats_buffer import STATS_BUFFERS

            return STATS_BUFFERS.log_stats()

        def fp8_tensor(t):
            return Float8Tensor(
                data=t.to(torch.uint8).cuda(),
                fp8_scale_inv=torch.ones([1]).cuda(),
                fp8_dtype=tex.DType.kFloat8E4M3,
                shape=t.shape,
                dtype=torch.float32,
            )

        shape = [1024, 1024]
        tensors = [torch.randn(shape) for _ in range(2)]
        tensors_fp8 = [fp8_tensor(tensors[i]) for i in range(2)]

        feed(tensors[0], tensors_fp8[0])
        feed(tensors[1], tensors_fp8[1])
        stats1 = log_stats()

        tensor2 = torch.cat((tensors[0], tensors[1])).cuda()
        fp8tensor2 = fp8_tensor(tensor2)
        feed(tensor2, fp8tensor2)
        stats2 = log_stats()

        assert len(stats1.keys()) > 0
        for k in stats1.keys():
            torch.testing.assert_close(stats1[k], stats2[k])
    finally:
        debug_api.end_debug()


if __name__ == "__main__":
    pass
