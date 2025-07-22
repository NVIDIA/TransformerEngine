# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from transformer_engine.pytorch import MultiheadAttention

import pytest
import torch


@pytest.mark.parametrize("qk_norm_type", [None, "L2Normalization", "RMSNorm", "LayerNorm"])
@pytest.mark.parametrize("attention_type", ["self", "cross"])
@pytest.mark.parametrize("qk_norm_eps", [1e-6, 1e-5])
def test_qk_norm_functionality(qk_norm_type, attention_type, qk_norm_eps) -> None:
    """Test QK normalization functionality, module structure, and numerical behavior."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 128

    # Create MultiheadAttention module
    mha = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_type=attention_type,
        qk_norm_type=qk_norm_type,
        qk_norm_eps=qk_norm_eps,
        bias=False,
        device="cuda",
    ).cuda()

    # Check module structure based on qk_norm_type parameter
    if qk_norm_type is not None:
        assert mha.q_norm is not None, "Should have q_norm module when qk_norm_type is not None"
        assert mha.k_norm is not None, "Should have k_norm module when qk_norm_type is not None"

        # Check that the modules are of the correct type
        if qk_norm_type == "L2Normalization":
            from transformer_engine.pytorch.ops.basic.l2normalization import L2Normalization

            assert isinstance(
                mha.q_norm, L2Normalization
            ), "q_norm should be an L2Normalization module"
            assert isinstance(
                mha.k_norm, L2Normalization
            ), "k_norm should be an L2Normalization module"
            # For L2 normalization, q_norm and k_norm should be the same instance (parameter-free)
            assert (
                mha.q_norm is mha.k_norm
            ), "q_norm and k_norm should be the same instance for L2 normalization"

        elif qk_norm_type == "RMSNorm":
            from transformer_engine.pytorch.module.rmsnorm import RMSNorm

            assert isinstance(mha.q_norm, RMSNorm), "q_norm should be an RMSNorm module"
            assert isinstance(mha.k_norm, RMSNorm), "k_norm should be an RMSNorm module"
            # For RMS normalization, q_norm and k_norm should be separate instances
            assert (
                mha.q_norm is not mha.k_norm
            ), "q_norm and k_norm should be separate instances for RMS normalization"

        elif qk_norm_type == "LayerNorm":
            from transformer_engine.pytorch.module.layernorm import LayerNorm

            assert isinstance(mha.q_norm, LayerNorm), "q_norm should be a LayerNorm module"
            assert isinstance(mha.k_norm, LayerNorm), "k_norm should be a LayerNorm module"
            # For LayerNorm, q_norm and k_norm should be separate instances
            assert (
                mha.q_norm is not mha.k_norm
            ), "q_norm and k_norm should be separate instances for LayerNorm"

        else:
            # For extensibility - just ensure they exist
            assert mha.q_norm is not None, f"q_norm should exist for qk_norm_type={qk_norm_type}"
            assert mha.k_norm is not None, f"k_norm should exist for qk_norm_type={qk_norm_type}"
    else:
        assert mha.q_norm is None, "Should not have q_norm module when qk_norm_type is None"
        assert mha.k_norm is None, "Should not have k_norm module when qk_norm_type is None"

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


@pytest.mark.parametrize("qk_norm_type", ["L2Normalization", "RMSNorm", "LayerNorm"])
def test_qk_norm_output_difference(qk_norm_type) -> None:
    """Test that QK normalization actually changes the output compared to no normalization."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 128
    batch_size = 2

    # Reset to a known seed for reproducible initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create model with QK normalization
    mha_with_norm = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        qk_norm_type=qk_norm_type,
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
        qk_norm_type=None,
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
    ), f"QK normalization ({qk_norm_type}) should change the output, but outputs are identical"


@pytest.mark.parametrize("qk_norm_type", ["L2Normalization", "RMSNorm", "LayerNorm"])
def test_qk_norm_with_fused_qkv(qk_norm_type) -> None:
    """Test QK normalization works with fused QKV parameters."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 64

    mha = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        fuse_qkv_params=True,
        qk_norm_type=qk_norm_type,
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


@pytest.mark.parametrize("qk_norm_type", ["L2Normalization", "RMSNorm", "LayerNorm"])
def test_qk_norm_transformer_layer_output_difference(qk_norm_type) -> None:
    """Test that QK normalization actually changes TransformerLayer output compared to no normalization."""
    from transformer_engine.pytorch import TransformerLayer

    hidden_size = 256
    ffn_hidden_size = 1024
    num_attention_heads = 8
    seq_len = 128
    batch_size = 2

    # Reset to a known seed for reproducible initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create TransformerLayer with QK normalization
    transformer_with_norm = TransformerLayer(
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_attention_heads,
        qk_norm_type=qk_norm_type,
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
        qk_norm_type=None,
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
    assert not torch.allclose(output_with_norm, output_no_norm, atol=1e-6), (
        f"QK normalization ({qk_norm_type}) should change the TransformerLayer output, but outputs"
        " are identical"
    )

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


@pytest.mark.parametrize("qk_norm_type", ["L2Normalization", "RMSNorm", "LayerNorm"])
def test_qk_norm_before_after_rope(qk_norm_type) -> None:
    """Test that QK normalization before and after RoPE works without errors."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 64
    batch_size = 2

    # Create model with QK norm after RoPE (default)
    mha_after = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        qk_norm_type=qk_norm_type,
        qk_norm_before_rope=False,
        bias=False,
        device="cuda",
    ).cuda()

    # Create model with QK norm before RoPE
    mha_before = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        qk_norm_type=qk_norm_type,
        qk_norm_before_rope=True,
        bias=False,
        device="cuda",
    ).cuda()

    hidden_states = torch.randn(
        seq_len, batch_size, hidden_size, device="cuda", dtype=torch.float32
    )

    # Create RoPE embeddings
    head_dim = hidden_size // num_attention_heads
    rotary_dim = head_dim // 2
    rotary_pos_emb = torch.randn(seq_len, 1, 1, rotary_dim, device="cuda", dtype=torch.float32)

    with torch.no_grad():
        output_after_rope = mha_after(hidden_states, rotary_pos_emb=rotary_pos_emb)
        output_before_rope = mha_before(hidden_states, rotary_pos_emb=rotary_pos_emb)

        output_after_no_rope = mha_after(hidden_states)
        output_before_no_rope = mha_before(hidden_states)

    # Check output shapes and properties
    expected_shape = (seq_len, batch_size, hidden_size)
    for output in [
        output_after_rope,
        output_before_rope,
        output_after_no_rope,
        output_before_no_rope,
    ]:
        assert output.shape == expected_shape, f"Output shape mismatch: {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN"
        assert not torch.isinf(output).any(), "Output contains Inf"

    assert output_after_rope.shape == output_before_rope.shape, "Outputs should have same shape"
    assert mha_after.qk_norm_before_rope == False, "mha_after should have qk_norm_before_rope=False"
    assert mha_before.qk_norm_before_rope == True, "mha_before should have qk_norm_before_rope=True"


def test_different_qk_norm_types_produce_different_outputs() -> None:
    """Test that different QK normalization types produce different outputs."""
    hidden_size = 256
    num_attention_heads = 8
    seq_len = 128
    batch_size = 2

    # Use same random seed to ensure identical weight initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create model with L2 normalization
    mha_l2 = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        qk_norm_type="L2Normalization",
        bias=False,
        device="cuda",
    ).cuda()

    # Reset to same seed for identical initialization
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Create model with RMS normalization
    mha_rms = MultiheadAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        qk_norm_type="RMSNorm",
        bias=False,
        device="cuda",
    ).cuda()

    # Create input tensors
    hidden_states = torch.randn(
        seq_len, batch_size, hidden_size, device="cuda", dtype=torch.float32
    )

    # Compare outputs with identical weights but different QK norm types
    with torch.no_grad():
        output_l2 = mha_l2(hidden_states)
        output_rms = mha_rms(hidden_states)

    # Outputs should be different when using different normalization types
    assert not torch.allclose(
        output_l2, output_rms, atol=1e-6
    ), "L2 and RMS normalization should produce different outputs, but outputs are identical"

    # Check that outputs have expected shapes and properties
    assert output_l2.shape == output_rms.shape, "L2 and RMS outputs should have same shape"
    assert not torch.isnan(output_l2).any(), "L2 output contains NaN"
    assert not torch.isinf(output_l2).any(), "L2 output contains Inf"
    assert not torch.isnan(output_rms).any(), "RMS output contains NaN"
    assert not torch.isinf(output_rms).any(), "RMS output contains Inf"
