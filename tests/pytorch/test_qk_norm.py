# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from transformer_engine.pytorch import MultiheadAttention

import pytest
import torch


@pytest.mark.parametrize("use_qk_norm", [False, True])
@pytest.mark.parametrize("attention_type", ["self", "cross"])
@pytest.mark.parametrize("qk_norm_eps", [1e-6, 1e-5])
def test_qk_norm_functionality(use_qk_norm, attention_type, qk_norm_eps) -> None:
    """Test QK normalization functionality, module structure, and numerical behavior."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 128

    # Create MultiheadAttention module
    mha = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_type=attention_type,
        use_qk_norm=use_qk_norm,
        qk_norm_eps=qk_norm_eps,
        bias=False,
        device="cuda",
    ).cuda()

    # Check module structure based on use_qk_norm parameter
    if use_qk_norm:
        assert hasattr(mha, "qk_norm"), "Should have qk_norm module when use_qk_norm=True"
        assert not hasattr(mha, "q_l2norm"), "Should not have separate q_l2norm module"
        assert not hasattr(mha, "k_l2norm"), "Should not have separate k_l2norm module"
        # Check that the module is L2Norm type
        from transformer_engine.pytorch.ops.basic.l2norm import L2Norm

        assert isinstance(mha.qk_norm, L2Norm), "qk_norm should be an L2Norm module"
    else:
        assert not hasattr(mha, "qk_norm"), "Should not have qk_norm module when use_qk_norm=False"

    # Create input tensors
    batch_size = 2  # Use a fixed batch size for testing
    hidden_states = torch.randn(
        seq_len, batch_size, hidden_size, device="cuda", dtype=torch.float32
    )

    if attention_type == "cross":
        encoder_output = torch.randn(
            seq_len, batch_size, hidden_size, device="cuda", dtype=torch.float32
        )
    else:
        encoder_output = None

    # Test forward pass
    with torch.no_grad():
        if attention_type == "cross":
            output = mha(hidden_states, encoder_output=encoder_output)
        else:
            output = mha(hidden_states)

    # Check output shape and numerical properties
    assert output.shape == (
        seq_len,
        batch_size,
        hidden_size,
    ), f"Output shape mismatch: {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaN"
    assert not torch.isinf(output).any(), "Output contains Inf"

    # Test with RoPE (if self-attention)
    if attention_type == "self":
        head_dim = hidden_size // num_attention_heads
        rotary_dim = head_dim // 2
        rotary_pos_emb = torch.randn(seq_len, 1, 1, rotary_dim, device="cuda", dtype=torch.float32)

        with torch.no_grad():
            output_with_rope = mha(hidden_states, rotary_pos_emb=rotary_pos_emb)

        assert output_with_rope.shape == (
            seq_len,
            batch_size,
            hidden_size,
        ), "Output shape with RoPE mismatch"
        assert not torch.isnan(output_with_rope).any(), "RoPE output contains NaN"
        assert not torch.isinf(output_with_rope).any(), "RoPE output contains Inf"


def test_qk_norm_output_difference() -> None:
    """Test that QK normalization actually changes the output compared to no normalization."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 128
    batch_size = 2

    # Use same random seed to ensure identical weight initialization
    current_rng_state = torch.get_rng_state()
    current_cuda_rng_state = torch.cuda.get_rng_state()

    # Reset to a known seed for reproducible initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create model with QK normalization
    mha_with_norm = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_qk_norm=True,
        bias=False,
        device="cuda",
    ).cuda()

    # Reset to same seed for identical initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create identical model without QK normalization
    mha_no_norm = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        use_qk_norm=False,
        bias=False,
        device="cuda",
    ).cuda()

    # Create input tensors
    hidden_states = torch.randn(
        seq_len, batch_size, hidden_size, device="cuda", dtype=torch.float32
    )

    # Compare outputs with identical weights but different QK norm settings
    with torch.no_grad():
        output_with_norm = mha_with_norm(hidden_states)
        output_no_norm = mha_no_norm(hidden_states)

    # Outputs should be different when QK normalization is enabled
    assert not torch.allclose(
        output_with_norm, output_no_norm, atol=1e-6
    ), "QK normalization should change the output, but outputs are identical"


def test_qk_norm_with_fused_qkv() -> None:
    """Test QK normalization works with fused QKV parameters."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 64

    mha = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        fuse_qkv_params=True,
        use_qk_norm=True,
        bias=False,
        device="cuda",
    ).cuda()

    # Create input and test forward pass
    batch_size = 2  # Use a fixed batch size for testing
    hidden_states = torch.randn(
        seq_len, batch_size, hidden_size, device="cuda", dtype=torch.float32
    )

    with torch.no_grad():
        output = mha(hidden_states)

    assert output.shape == (
        seq_len,
        batch_size,
        hidden_size,
    ), f"Output shape mismatch: {output.shape}"


def test_qk_norm_transformer_layer_output_difference() -> None:
    """Test that QK normalization actually changes TransformerLayer output compared to no normalization."""
    from transformer_engine.pytorch import TransformerLayer

    hidden_size = 256
    ffn_hidden_size = 1024
    num_attention_heads = 8
    seq_len = 128
    batch_size = 2

    # Use same random seed to ensure identical weight initialization
    current_rng_state = torch.get_rng_state()
    current_cuda_rng_state = torch.cuda.get_rng_state()

    # Reset to a known seed for reproducible initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create TransformerLayer with QK normalization
    transformer_with_norm = TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        use_qk_norm=True,
        bias=False,
        device="cuda",
    ).cuda()

    # Reset to same seed for identical initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create identical TransformerLayer without QK normalization
    transformer_no_norm = TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        use_qk_norm=False,
        bias=False,
        device="cuda",
    ).cuda()

    # Create input tensors
    hidden_states = torch.randn(
        seq_len, batch_size, hidden_size, device="cuda", dtype=torch.float32
    )

    # Compare outputs with identical weights but different QK norm settings
    with torch.no_grad():
        output_with_norm = transformer_with_norm(hidden_states)
        output_no_norm = transformer_no_norm(hidden_states)

    # Outputs should be different when QK normalization is enabled
    assert not torch.allclose(
        output_with_norm, output_no_norm, atol=1e-6
    ), "QK normalization should change the TransformerLayer output, but outputs are identical"

    # Check that outputs have expected shapes and properties
    assert output_with_norm.shape == (
        seq_len,
        batch_size,
        hidden_size,
    ), f"Output shape mismatch: {output_with_norm.shape}"
    assert not torch.isnan(output_with_norm).any(), "Output with QK norm contains NaN"
    assert not torch.isinf(output_with_norm).any(), "Output with QK norm contains Inf"
    assert not torch.isnan(output_no_norm).any(), "Output without QK norm contains NaN"
    assert not torch.isinf(output_no_norm).any(), "Output without QK norm contains Inf"
