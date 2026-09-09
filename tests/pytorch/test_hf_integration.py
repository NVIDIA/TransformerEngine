# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from unittest import mock
import pytest
from transformer_engine.pytorch import TransformerLayer, quantized_model_init, QuantizedTensor

# I think this API is changing quite fast, looks like ToT torchao and transformers are incompatible.
# (pstjohn, 10/16/2025)
with mock.patch("transformers.utils.is_torchao_available", return_value=False):
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel


class SimpleTEModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.my_layer = TransformerLayer(
            hidden_size=320,
            num_attention_heads=16,
            ffn_hidden_size=1024,
            layer_number=None,
            fuse_qkv_params=True,
            qkv_weight_interleaved=True,
        )

    def forward(self, hidden_states, attention_mask):
        return self.my_layer(hidden_states, attention_mask)


class SimpleTEModelNoQKVFusion(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.my_layer = TransformerLayer(
            hidden_size=320,
            num_attention_heads=16,
            ffn_hidden_size=1024,
            layer_number=None,
            fuse_qkv_params=False,
        )

    def forward(self, hidden_states, attention_mask):
        return self.my_layer(hidden_states, attention_mask)


@pytest.fixture
def checkpoint_path(tmp_path):
    config = PretrainedConfig()
    model = SimpleTEModel(config)
    model.save_pretrained(tmp_path / "simple_te_model")
    return tmp_path / "simple_te_model"


def test_save_hf_model(tmp_path):
    config = PretrainedConfig()
    model = SimpleTEModel(config)
    model.save_pretrained(tmp_path / "simple_te_model")


def test_save_and_load_hf_model(checkpoint_path):
    model = SimpleTEModel.from_pretrained(checkpoint_path)
    assert model is not None


def test_quantized_model_init_context_manager(checkpoint_path):
    config = PretrainedConfig()
    with quantized_model_init(enabled=True):
        model = SimpleTEModel(config)
    assert isinstance(model.my_layer.layernorm_mlp.fc1_weight, QuantizedTensor)


def test_quantized_model_init_context_manager_no_qkv_fusion(checkpoint_path):
    # RuntimeError: Splitting QuantizedTensor into multiple params is not supported
    config = PretrainedConfig()
    with quantized_model_init(enabled=True):
        model = SimpleTEModelNoQKVFusion(config)
    assert isinstance(model.my_layer.layernorm_mlp.fc1_weight, QuantizedTensor)


def test_from_pretrained_with_quantized_model_init(checkpoint_path):
    # TypeError: Float8TensorStorage.__new__() missing 3 required keyword-only arguments: 'data',
    # 'fp8_scale_inv', and 'fp8_dtype'
    with quantized_model_init(enabled=True):
        model = SimpleTEModel.from_pretrained(checkpoint_path)
    assert isinstance(model.my_layer.layernorm_mlp.fc1_weight, QuantizedTensor)


def test_save_pretrained_with_quantized_model_init(tmp_path):
    # RuntimeError: Attempted to access the data pointer on an invalid python storage.
    config = PretrainedConfig()
    with quantized_model_init(enabled=True):
        model = SimpleTEModel(config)

    model.save_pretrained(tmp_path / "simple_te_model")
