import pytest
import torch
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from transformer_engine.pytorch.transformer import TransformerLayer
from transformer_engine.pytorch.utils import is_bf16_compatible

param_types = [torch.float32, torch.float16]
if is_bf16_compatible():  # bf16 requires sm_80 or higher
    param_types.append(torch.bfloat16)


all_activations = ["gelu", "relu"]
all_normalizations = ["LayerNorm", "RMSNorm"]


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("activation", all_activations)
@pytest.mark.parametrize("normalization", all_normalizations)
def test_save_and_load_hf_model(tmp_path, dtype, activation, normalization):
    class SimpleTEModel(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config: PretrainedConfig):
            super().__init__(config)
            self.my_layer = TransformerLayer(
                hidden_size=320,
                num_attention_heads=16,
                ffn_hidden_size=1024,
                layer_number=None,
                params_dtype=dtype,
                activation=activation,
                normalization=normalization,
            )

        def forward(self, hidden_states, attention_mask):
            return self.my_layer(hidden_states, attention_mask)

    model = SimpleTEModel(PretrainedConfig())

    model.save_pretrained(tmp_path / "simple_te_model")
    del model
    SimpleTEModel.from_pretrained(tmp_path / "simple_te_model")
