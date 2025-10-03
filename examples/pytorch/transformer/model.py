from dataclasses import dataclass
import torch
import transformer_engine.pytorch
from transformer_engine.pytorch.attention.rope import RotaryPositionEmbedding
from typing import Dict
from transformer_engine.pytorch.module.layernorm import LayerNorm


@dataclass
class SimpleConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    layer_norm_eps: float
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    num_hidden_layers: int
    vocab_size: int
    micro_batch_size: int
    max_seq_length: int


# THD Model
class SimpleThDModel(torch.nn.Module):
    def __init__(self, config: SimpleConfig):
        super(SimpleThDModel, self).__init__()
        self.config = config
        print("Config: ", config)
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.transformer_layers = torch.nn.ModuleList(
            [
                transformer_engine.pytorch.TransformerLayer(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    layernorm_epsilon=config.layer_norm_eps,
                    hidden_dropout=0.0,
                    attention_dropout=0.1,
                    qkv_weight_interleaved=True,
                    layer_number=i + 1,
                    layer_type="encoder",
                    self_attn_mask_type="padding",
                    activation="gelu",
                    attn_input_format="thd",
                    seq_length=config.max_seq_length,  # Note: Do we need this anymore?
                    micro_batch_size=config.micro_batch_size,
                    num_gqa_groups=config.num_attention_heads,
                    fuse_qkv_params=True,
                    params_dtype=torch.bfloat16,
                    window_size=(-1, -1),
                    sequence_parallel=False,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.emb_layer_norm_after = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_embeddings = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ):
        """Forward pass of the SimpleModel.

        Args:
            batch (Dict[str, torch.Tensor]): The batch. which has the following keys
                - input_ids: The input ids.
                - labels: The labels.
                - position_ids: The position ids.
                - cu_seqlens_q: The cu_seqlens_q.
                - cu_seqlens_kv: The cu_seqlens_kv.
                - cu_seqlens_q_padded: The cu_seqlens_q_padded.
                - cu_seqlens_kv_padded: The cu_seqlens_kv_padded.
                - pad_between_seqs: The pad_between_seqs.
                - max_seqlen_q: The max_seqlen_q.
                - max_seqlen_kv: The max_seqlen_kv.
        """
        # The Embedding part - handle both 1D and 2D input
        input_ids = batch["input_ids"]
        if input_ids.dim() == 2:
            # If 2D (batch_size, seq_len), flatten to 1D for THD format
            input_ids = input_ids.view(-1)

        hidden_states = self.embedding(input_ids)

        # The Encoder part.
        te_rope_emb = self.rotary_embeddings(max_seq_len=batch["cu_seqlens_q_padded"][-1]).cuda()
        for layer_module in self.transformer_layers:
            hidden_states = layer_module(
                hidden_states=hidden_states,  # Remove squeeze(0) since we already flattened
                rotary_pos_emb=te_rope_emb,
                cu_seqlens_q=batch["cu_seqlens_q"],
                cu_seqlens_kv=batch["cu_seqlens_kv"],
                cu_seqlens_q_padded=batch["cu_seqlens_q_padded"],
                cu_seqlens_kv_padded=batch["cu_seqlens_kv_padded"],
                pad_between_seqs=batch["pad_between_seqs"],
                max_seqlen_q=batch["max_seqlen_q"],
                max_seqlen_kv=batch["max_seqlen_kv"],
            )
        hidden_states = self.emb_layer_norm_after(hidden_states)
        logits = self.linear(hidden_states)

        return logits


# BSHD Model.
class SimpleBSHDModel(torch.nn.Module):
    def __init__(self, config: SimpleConfig):
        super(SimpleBSHDModel, self).__init__()
        self.config = config
        print("Config: ", config)
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.transformer_layers = torch.nn.ModuleList(
            [
                transformer_engine.pytorch.TransformerLayer(
                    hidden_size=config.hidden_size,
                    ffn_hidden_size=config.intermediate_size,
                    num_attention_heads=config.num_attention_heads,
                    layernorm_epsilon=config.layer_norm_eps,
                    hidden_dropout=0.0,
                    attention_dropout=0.1,
                    qkv_weight_interleaved=True,
                    layer_number=i + 1,
                    layer_type="encoder",
                    self_attn_mask_type="no_mask",
                    activation="gelu",
                    attn_input_format="bshd",
                    seq_length=config.max_seq_length,
                    micro_batch_size=config.micro_batch_size,
                    num_gqa_groups=config.num_attention_heads,
                    fuse_qkv_params=True,
                    params_dtype=torch.bfloat16,
                    window_size=(-1, -1),
                    sequence_parallel=False,
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.emb_layer_norm_after = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_embeddings = RotaryPositionEmbedding(
            config.hidden_size // config.num_attention_heads
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        """Forward pass of the SimpleBSHDModel.

        Here we process the input in BSHD format.
        Args:
            batch (Dict[str, torch.Tensor]): The batch. which has the following keys
                - input_ids: The input ids. of shape [batch_size, seq_length]
                - labels: The labels. of shape [batch_size, seq_length]
                - position_ids: The position ids. of shape [batch_size, seq_length]
                - cu_seqlens_q: The cu_seqlens_q. of shape [batch_size + 1]
                - cu_seqlens_kv: The cu_seqlens_kv. of shape [batch_size + 1]
                - pad_between_seqs: boolean indicating if there is padding between sequences
                - max_seqlen_q: The max_seqlen_q.
                - max_seqlen_kv: The max_seqlen_kv.
        Returns:
            logits: The logits. of shape [batch_size, seq_length, vocab_size]
        """
        input_ids = batch["input_ids"]
        # input_ids of shape [batch_size, seq_length]

        hidden_states = self.embedding(input_ids)
        # hidden_states of shape [batch_size, seq_length, hidden_size]

        # The Encoder part.
        te_rope_emb = self.rotary_embeddings(max_seq_len=batch["cu_seqlens_q"][-1]).cuda()
        for layer_module in self.transformer_layers:
            hidden_states = layer_module(
                hidden_states=hidden_states,  # Remove squeeze(0) since we already flattened
                rotary_pos_emb=te_rope_emb,
                cu_seqlens_q=batch["cu_seqlens_q"],
                cu_seqlens_kv=batch["cu_seqlens_kv"],
                pad_between_seqs=batch["pad_between_seqs"],
                max_seqlen_q=batch["max_seqlen_q"],
                max_seqlen_kv=batch["max_seqlen_kv"],
            )
        hidden_states = self.emb_layer_norm_after(hidden_states)
        logits = self.linear(hidden_states)
        # Logits of shape [batch_size, seq_length, vocab_size]
        return logits
