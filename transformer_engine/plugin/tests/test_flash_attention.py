# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import math
import torch
import torch.nn.functional as F

from transformer_engine.plugin.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
)


class FlashAttentionTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Flash Attention",
            "Test correctness of Flash Attention implementation across backends"
        )
        self.backends = get_available_backends()
        self.device = device

    def _reference_attention(
        self,
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
    ):
        """Reference implementation of scaled dot-product attention
        Input format: sbhd [seq, batch, heads, dim]
        """
        # Convert sbhd to bhsd for computation
        q = query.permute(1, 2, 0, 3)  # [batch, heads, seq, dim]
        k = key.permute(1, 2, 0, 3)
        v = value.permute(1, 2, 0, 3)

        L, S = q.size(-2), k.size(-2)
        if scale is None:
            scale_factor = 1 / math.sqrt(q.size(-1))
        else:
            scale_factor = scale

        attn_weight = q @ k.transpose(-2, -1) * scale_factor

        if is_causal:
            causal_mask = torch.triu(
                torch.full((L, S), float('-inf'), dtype=q.dtype, device=q.device),
                diagonal=1
            )
            attn_weight = attn_weight + causal_mask

        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask

        attn_weight = F.softmax(attn_weight, dim=-1)

        if dropout_p > 0.0:
            attn_weight = F.dropout(attn_weight, p=dropout_p, training=True)

        out = attn_weight @ v
        # Convert bhsd back to sbhd
        return out.permute(2, 0, 1, 3)  # [seq, batch, heads, dim]

    def test_flash_attention_forward_basic(self, seq_len=16, batch_size=2, num_heads=4, head_dim=32):
        """Test basic flash attention forward pass with sbhd layout and bf16"""
        print(f"\n  Testing Flash Attention forward sbhd bf16 (seq={seq_len}, batch={batch_size}, heads={num_heads}, dim={head_dim})")

        # Shape: (seq_len, batch, num_heads, head_dim) - sbhd layout
        query = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device
        )
        key = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device
        )
        value = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device
        )

        scale = 1.0 / math.sqrt(head_dim)

        # Reference attention (compute in float32 for accuracy)
        reference = self._reference_attention(
            query.float(), key.float(), value.float(),
            scale=scale, is_causal=False
        ).to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                FlashAttentionClass = backend.get_flash_attention_class()
                flash_attn = FlashAttentionClass(
                    softmax_scale=scale,
                    attention_dropout=0.0,
                    attention_type="self",
                    deterministic=True,
                )

                # Run forward pass with sbhd layout
                output = flash_attn(
                    query_layer=query,
                    key_layer=key,
                    value_layer=value,
                    attention_mask=None,
                    qkv_layout="sb3hd",
                    attn_mask_type="no_mask",
                    window_size=(-1, -1),  # Required by flash_attn 2.7+
                )

                # Output shape: sbhd -> view to sb(h*d)
                expected_shape = (seq_len, batch_size, num_heads * head_dim)
                if output.shape != expected_shape:
                    # Try to reshape reference for comparison
                    reference_flat = reference.contiguous().reshape(seq_len, batch_size, -1)
                    self.assert_close(
                        output.float(), reference_flat.float(), rtol=1e-2, atol=1e-2,
                        msg=f"Flash Attention forward mismatch for {backend_name}"
                    )
                else:
                    reference_flat = reference.contiguous().reshape(seq_len, batch_size, -1)
                    self.assert_close(
                        output.float(), reference_flat.float(), rtol=1e-2, atol=1e-2,
                        msg=f"Flash Attention forward mismatch for {backend_name}"
                    )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")
                import traceback
                traceback.print_exc()

    def test_flash_attention_forward_causal(self, seq_len=16, batch_size=2, num_heads=4, head_dim=32):
        """Test flash attention forward pass with causal mask"""
        print(f"\n  Testing Flash Attention forward causal sbhd bf16 (seq={seq_len}, batch={batch_size}, heads={num_heads}, dim={head_dim})")

        query = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device
        )
        key = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device
        )
        value = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device
        )

        scale = 1.0 / math.sqrt(head_dim)

        # Reference attention with causal mask
        reference = self._reference_attention(
            query.float(), key.float(), value.float(),
            scale=scale, is_causal=True
        ).to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                FlashAttentionClass = backend.get_flash_attention_class()
                flash_attn = FlashAttentionClass(
                    softmax_scale=scale,
                    attention_dropout=0.0,
                    attention_type="self",
                    deterministic=True,
                )

                output = flash_attn(
                    query_layer=query,
                    key_layer=key,
                    value_layer=value,
                    attention_mask=None,
                    qkv_layout="sb3hd",
                    attn_mask_type="causal",
                    window_size=(-1, -1),  # Required by flash_attn 2.7+
                )

                reference_flat = reference.contiguous().reshape(seq_len, batch_size, -1)
                self.assert_close(
                    output.float(), reference_flat.float(), rtol=1e-2, atol=1e-2,
                    msg=f"Flash Attention forward causal mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")
                import traceback
                traceback.print_exc()

    def test_flash_attention_backward(self, seq_len=16, batch_size=2, num_heads=4, head_dim=32):
        """Test flash attention backward pass with sbhd layout, bf16, and causal mask.

        Note: FlagGems backward currently only supports causal attention.
        """
        print(f"\n  Testing Flash Attention backward causal sbhd bf16 (seq={seq_len}, batch={batch_size}, heads={num_heads}, dim={head_dim})")

        query = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device, requires_grad=True
        )
        key = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device, requires_grad=True
        )
        value = generate_random_tensor(
            (seq_len, batch_size, num_heads, head_dim),
            dtype=torch.bfloat16, device=self.device, requires_grad=True
        )
        # grad_output shape matches output: sb(h*d)
        grad_output = generate_random_tensor(
            (seq_len, batch_size, num_heads * head_dim),
            dtype=torch.bfloat16, device=self.device
        )

        scale = 1.0 / math.sqrt(head_dim)

        # Reference backward (compute in float32 for accuracy)
        # Note: FlagGems backward only supports causal attention
        query_f32 = query.float().detach().requires_grad_(True)
        key_f32 = key.float().detach().requires_grad_(True)
        value_f32 = value.float().detach().requires_grad_(True)

        ref_output = self._reference_attention(query_f32, key_f32, value_f32, scale=scale, is_causal=True)
        ref_output_flat = ref_output.contiguous().reshape(seq_len, batch_size, -1)
        ref_output_flat.backward(grad_output.float())
        ref_grad_q = query_f32.grad.clone().to(torch.bfloat16)
        ref_grad_k = key_f32.grad.clone().to(torch.bfloat16)
        ref_grad_v = value_f32.grad.clone().to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                FlashAttentionClass = backend.get_flash_attention_class()
                flash_attn = FlashAttentionClass(
                    softmax_scale=scale,
                    attention_dropout=0.0,
                    attention_type="self",
                    deterministic=True,
                )

                # Forward pass
                q_copy = query.detach().requires_grad_(True)
                k_copy = key.detach().requires_grad_(True)
                v_copy = value.detach().requires_grad_(True)

                output = flash_attn(
                    query_layer=q_copy,
                    key_layer=k_copy,
                    value_layer=v_copy,
                    attention_mask=None,
                    qkv_layout="sb3hd",
                    attn_mask_type="causal",
                    window_size=(-1, -1),  # Required by flash_attn 2.7+
                )

                # Backward pass
                output.backward(grad_output)

                # bf16 backward has higher numerical error due to accumulated precision loss
                self.assert_close(
                    q_copy.grad.float(), ref_grad_q.float(), rtol=2e-2, atol=2e-2,
                    msg=f"Flash Attention backward grad_q mismatch for {backend_name}"
                )
                self.assert_close(
                    k_copy.grad.float(), ref_grad_k.float(), rtol=2e-2, atol=2e-2,
                    msg=f"Flash Attention backward grad_k mismatch for {backend_name}"
                )
                self.assert_close(
                    v_copy.grad.float(), ref_grad_v.float(), rtol=2e-2, atol=2e-2,
                    msg=f"Flash Attention backward grad_v mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")
                import traceback
                traceback.print_exc()

    def run_all_tests(self):
        print("\n" + "="*60)
        print("Testing Flash Attention")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        # Basic forward tests with sbhd layout and bf16
        self.test_flash_attention_forward_basic(seq_len=16, batch_size=2, num_heads=4, head_dim=32)
        self.test_flash_attention_forward_basic(seq_len=32, batch_size=4, num_heads=8, head_dim=64)

        # Causal mask tests
        self.test_flash_attention_forward_causal(seq_len=16, batch_size=2, num_heads=4, head_dim=32)

        # Backward tests
        self.test_flash_attention_backward(seq_len=16, batch_size=2, num_heads=4, head_dim=32)

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device != "cuda":
        print("Warning: Flash Attention tests require CUDA. Skipping.")
        return 0
    test_suite = FlashAttentionTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
