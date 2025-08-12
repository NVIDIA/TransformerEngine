# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from transformer_engine.pytorch.transformer import TransformerLayer


class SimpleTEModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.my_layer = TransformerLayer(
            hidden_size=320,
            num_attention_heads=16,
            ffn_hidden_size=1024,
            layer_number=None,
        )

    def forward(self, hidden_states, attention_mask):
        return self.my_layer(hidden_states, attention_mask)


def test_save_hf_model(tmp_path):
    model = SimpleTEModel(PretrainedConfig())
    model.save_pretrained(tmp_path / "simple_te_model")


@pytest.mark.xfail(reason="This test is failing until huggingface/transformers#38155 is merged.")
def test_save_and_load_hf_model(tmp_path):
    model = SimpleTEModel(PretrainedConfig())
    model.save_pretrained(tmp_path / "simple_te_model")
    del model
    model = SimpleTEModel.from_pretrained(tmp_path / "simple_te_model")
    assert model is not None
