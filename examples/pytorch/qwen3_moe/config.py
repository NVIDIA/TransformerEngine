"""Configuration for Qwen3 MoE model.

Default values match the HuggingFace Transformers Qwen3MoeConfig.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen3MoeConfig:
    """Configuration class for Qwen3 MoE model.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimensionality of the hidden representations.
        moe_intermediate_size: Dimensionality of each MoE expert's intermediate layer.
        num_hidden_layers: Number of decoder layers.
        num_attention_heads: Number of query attention heads.
        num_key_value_heads: Number of key/value attention heads (for GQA).
        max_position_embeddings: Maximum sequence length supported by RoPE.
        initializer_range: Standard deviation for weight initialization.
        rms_norm_eps: Epsilon for RMSNorm layers.
        rope_theta: Base frequency for RoPE.
        attention_bias: Whether to use bias in attention projections.
        attention_dropout: Dropout rate for attention weights.
        num_experts: Total number of MoE experts.
        top_k: Number of experts selected per token (top-k).
        norm_topk_prob: Whether to renormalize top-k routing probabilities.
    """

    vocab_size: int = 151936
    hidden_size: int = 2048
    moe_intermediate_size: int = 768
    num_hidden_layers: int = 24
    num_attention_heads: int = 32
    num_key_value_heads: int = 4
    max_position_embeddings: int = 32768
    initializer_range: float = 0.02
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    attention_bias: bool = False
    attention_dropout: float = 0.0
    num_experts: int = 128
    top_k: int = 8
    norm_topk_prob: bool = False

    @property
    def head_dim(self) -> int:
        """Dimensionality of each attention head."""
        return self.hidden_size // self.num_attention_heads
