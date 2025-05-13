# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import pathlib, os

from nvdlfw_inspect.config_manager import ConfigManager

import nvdlfw_inspect.api as debug_api

try:
    import transformer_engine
    from transformer_engine.debug.features.api import TEConfigAPIMapper
except (ImportError, ModuleNotFoundError):
    print("Could not find TransformerEngine debug module.")
    exit(1)


def test_transformer_engine_config_parsing(feature_dirs):
    debug_api.initialize(
        config_file=pathlib.Path(__file__).resolve().parent
        / "test_configs/tensor_manipulation_transformer_engine.yaml",
        feature_dirs=feature_dirs,
        log_dir="./log",
    )

    cfg_fc1 = ConfigManager.get_config_for_layer("decoder.1.mlp.fc1")["transformer_engine"]
    cfg_fc2 = ConfigManager.get_config_for_layer("decoder.1.mlp.fc2")["transformer_engine"]
    assert cfg_fc1 and cfg_fc2

    gemm_parsing = True
    tensor_parsing = True

    # Per tensor scaling set for dgrad, filter based on gemm
    ret, _ = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc1["PerTensorScaling"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="wgrad",
        tensor_name="activation",
    )
    assert not ret

    # per tensor scaling set for gradient, filter based on tensor name
    ret, _ = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc1["PerTensorScaling"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="dgrad",
        tensor_name="activation",
    )
    assert not ret

    ret, parsed_cfg_fc1 = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc1["PerTensorScaling"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="dgrad",
        tensor_name="gradient",
    )
    assert ret
    assert parsed_cfg_fc1 == {"gemm": "dgrad", "tensor": "gradient"}

    # Test tensor struct
    ret, parsed_cfg_fc1_act = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc1["FakeQuant"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="fprop",
        tensor_name="activation",
    )
    ret, parsed_cfg_fc1_wei = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc1["FakeQuant"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="fprop",
        tensor_name="weight",
    )
    assert ret
    assert parsed_cfg_fc1_act == {
        "gemm": "fprop",
        "tensor": "activation",
        "quant_format": "FP8E4M3",
    }
    assert parsed_cfg_fc1_wei == {
        "gemm": "fprop",
        "tensor": "weight",
        "quant_format": "FP8E4M3",
    }

    # Test gemms struct
    ret, parsed_cfg_fc2_grad = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc2["FakeQuant"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="dgrad",
        tensor_name="gradient",
    )
    assert ret
    assert parsed_cfg_fc2_grad == {"gemm": "dgrad", "tensor": "gradient", "quant_format": "FP8E5M2"}
    ret, parsed_cfg_fc2_wei = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc2["FakeQuant"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="dgrad",
        tensor_name="weight",
    )
    assert ret
    assert parsed_cfg_fc2_wei == {"gemm": "dgrad", "tensor": "weight", "quant_format": "FP8E5M2"}

    # Test gemm + tensor struct
    ret, parsed_cfg_fc2_fprop_act = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc2["PerTensorScaling"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="fprop",
        tensor_name="activation",
    )
    assert ret
    assert parsed_cfg_fc2_fprop_act == {"gemm": "fprop", "tensor": "activation"}

    ret, parsed_cfg_fc2_fprop_wei = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc2["PerTensorScaling"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="fprop",
        tensor_name="weight",
    )
    assert ret
    assert parsed_cfg_fc2_fprop_wei == {"gemm": "fprop", "tensor": "weight"}

    ret, parsed_cfg_fc2_wgrad_act = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc2["PerTensorScaling"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="wgrad",
        tensor_name="activation",
    )
    assert ret
    assert parsed_cfg_fc2_wgrad_act == {"gemm": "wgrad", "tensor": "activation"}

    ret, parsed_cfg_fc2_wgrad_grad = TEConfigAPIMapper().parse_config_and_api(
        cfg_fc2["PerTensorScaling"],
        gemm_parsing=gemm_parsing,
        tensor_parsing=tensor_parsing,
        gemm="wgrad",
        tensor_name="gradient",
    )
    assert ret
    assert parsed_cfg_fc2_wgrad_grad == {"gemm": "wgrad", "tensor": "gradient"}

    ConfigManager.reset()
