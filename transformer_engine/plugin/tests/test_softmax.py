# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
import torch.nn.functional as F

from transformer_engine.plugin.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
)


class SoftmaxTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Softmax Operations",
            "Test correctness of all softmax operations across backends"
        )
        self.backends = get_available_backends()
        self.device = device

    def test_scaled_softmax_forward(self, shape=(2, 4, 8, 16)):
        print(f"\n  Testing scaled softmax forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)
        scale = 0.125
        reference = F.softmax(x.float() * scale, dim=-1).to(x.dtype)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.scaled_softmax_forward(x, scale)
                self.assert_close(
                    output, reference, rtol=1e-2, atol=1e-3,
                    msg=f"Scaled softmax forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_softmax_backward(self, shape=(2, 4, 8, 16)):
        print(f"\n  Testing scaled softmax backward with shape {shape}")

        # Use bf16 for all computation to match backend precision
        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        scale = 0.125
        grad_output = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)

        # Compute reference gradient using autograd (in float32 for precision, then convert)
        x_f32 = x.float().detach().requires_grad_(True)
        softmax_output_f32 = F.softmax(x_f32 * scale, dim=-1)
        loss = (softmax_output_f32 * grad_output.float()).sum()
        loss.backward()
        reference_grad = x_f32.grad.clone()

        # Get softmax output in bf16 for backend
        softmax_out_test = softmax_output_f32.detach().to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                # Clone inputs as some backends may modify them in-place
                grad_input = backend.scaled_softmax_backward(
                    grad_output.clone(), softmax_out_test.clone(), scale
                )
                self.assert_close(
                    grad_input.float(), reference_grad, rtol=1e-2, atol=1e-2,
                    msg=f"Scaled softmax backward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_masked_softmax_forward(self, shape=(2, 4, 8, 16)):
        print(f"\n  Testing scaled masked softmax forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        scale = 0.125

        # Create boolean mask and corresponding masks
        batch = shape[0]
        seq_q, seq_k = shape[-2], shape[-1]
        bool_mask = torch.rand((batch, 1, seq_q, seq_k), device=self.device) > 0.5

        # CUDA uses uint8 mask (1=masked, 0=unmasked)
        uint8_mask = bool_mask.to(torch.uint8)

        # Additive mask for reference computation
        additive_mask = torch.zeros((batch, 1, seq_q, seq_k), dtype=x.dtype, device=self.device)
        additive_mask = additive_mask.masked_fill(bool_mask, float('-inf'))
        additive_mask_expanded = additive_mask.expand(shape)

        # Reference: F.softmax(x * scale + additive_mask, dim=-1)
        reference = F.softmax(x * scale + additive_mask_expanded, dim=-1)

        # Use bf16 for all backends
        x_test = x.to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.scaled_masked_softmax_forward(x_test, uint8_mask, scale)
                self.assert_close(
                    output.float(), reference.float(), rtol=1e-2, atol=1e-3,
                    msg=f"Scaled masked softmax forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_masked_softmax_backward(self, shape=(2, 4, 8, 16)):
        print(f"\n  Testing scaled masked softmax backward with shape {shape}")

        # Use bf16 for all computation
        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        scale = 0.125
        grad_output = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)

        # Compute reference gradient using autograd (in float32 for precision)
        x_f32 = x.float().detach().requires_grad_(True)
        softmax_output_f32 = F.softmax(x_f32 * scale, dim=-1)
        loss = (softmax_output_f32 * grad_output.float()).sum()
        loss.backward()
        reference_grad = x_f32.grad.clone()

        # Get softmax output in bf16 for backend
        softmax_out_test = softmax_output_f32.detach().to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                # Clone inputs as some backends may modify them in-place
                grad_input = backend.scaled_masked_softmax_backward(
                    grad_output.clone(), softmax_out_test.clone(), scale
                )
                self.assert_close(
                    grad_input.float(), reference_grad, rtol=1e-2, atol=1e-2,
                    msg=f"Scaled masked softmax backward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_upper_triang_masked_softmax_forward(self, shape=(8, 16, 16)):
        print(f"\n  Testing scaled upper triang masked softmax forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)
        scale = 0.125
        seq_len = shape[-1]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=x.dtype, device=self.device),
            diagonal=1
        )
        reference = F.softmax(x.float() * scale + causal_mask.float(), dim=-1).to(x.dtype)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.scaled_upper_triang_masked_softmax_forward(x, scale)
                self.assert_close(
                    output, reference, rtol=1e-2, atol=1e-3,
                    msg=f"Scaled upper triang masked softmax forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_upper_triang_masked_softmax_backward(self, shape=(8, 16, 16)):
        print(f"\n  Testing scaled upper triang masked softmax backward with shape {shape}")

        # Use bf16 for all computation
        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        scale = 0.125
        seq_len = shape[-1]
        grad_output = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1
        )

        # Compute reference gradient using autograd (in float32 for precision)
        x_f32 = x.float().detach().requires_grad_(True)
        softmax_output_f32 = F.softmax(x_f32 * scale + causal_mask, dim=-1)
        loss = (softmax_output_f32 * grad_output.float()).sum()
        loss.backward()
        reference_grad = x_f32.grad.clone()

        # Get softmax output in bf16 for backend
        softmax_out_test = softmax_output_f32.detach().to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                # Clone inputs as some backends may modify them in-place
                grad_input = backend.scaled_upper_triang_masked_softmax_backward(
                    grad_output.clone(), softmax_out_test.clone(), scale
                )
                self.assert_close(
                    grad_input.float(), reference_grad, rtol=1e-2, atol=1e-2,
                    msg=f"Scaled upper triang masked softmax backward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_aligned_causal_masked_softmax_forward(self, shape=(2, 4, 16, 16)):
        """Test scaled aligned causal masked softmax forward.

        Note: CUDA backend requires 4D tensor (batch, heads, seq, seq).
        """
        print(f"\n  Testing scaled aligned causal masked softmax forward with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)
        scale = 0.125
        seq_len = shape[-1]

        # Aligned causal mask (lower triangular)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=x.dtype, device=self.device),
            diagonal=1
        )
        reference = F.softmax(x.float() * scale + causal_mask.float(), dim=-1).to(x.dtype)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.scaled_aligned_causal_masked_softmax_forward(x, scale)
                self.assert_close(
                    output, reference, rtol=1e-2, atol=1e-3,
                    msg=f"Scaled aligned causal masked softmax forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_aligned_causal_masked_softmax_backward(self, shape=(2, 4, 16, 16)):
        """Test scaled aligned causal masked softmax backward.

        Note: All backends use bf16 for consistency.
        """
        print(f"\n  Testing scaled aligned causal masked softmax backward with shape {shape}")

        # Use bf16 for all computation
        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device, requires_grad=True)
        scale = 0.125
        seq_len = shape[-1]
        grad_output = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), dtype=torch.float32, device=self.device),
            diagonal=1
        )

        # Compute reference gradient using autograd (in float32 for precision)
        x_f32 = x.float().detach().requires_grad_(True)
        softmax_output_f32 = F.softmax(x_f32 * scale + causal_mask, dim=-1)
        loss = (softmax_output_f32 * grad_output.float()).sum()
        loss.backward()
        reference_grad = x_f32.grad.clone()

        # Get softmax output in bf16 for backend
        softmax_out_test = softmax_output_f32.detach().to(torch.bfloat16)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                # Clone inputs as some backends may modify them in-place
                grad_input = backend.scaled_aligned_causal_masked_softmax_backward(
                    grad_output.clone(), softmax_out_test.clone(), scale
                )
                self.assert_close(
                    grad_input.float(), reference_grad, rtol=1e-2, atol=1e-2,
                    msg=f"Scaled aligned causal masked softmax backward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def run_all_tests(self):
        print("\n" + "="*60)
        print("Testing Softmax Operations")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        # Scaled softmax tests
        self.test_scaled_softmax_forward((4, 8, 16, 16))
        self.test_scaled_softmax_forward((2, 4, 32, 32))
        self.test_scaled_softmax_backward((4, 8, 16, 16))
        self.test_scaled_softmax_backward((2, 4, 32, 32))

        # Masked softmax tests
        self.test_scaled_masked_softmax_forward((4, 8, 16, 16))
        self.test_scaled_masked_softmax_backward((4, 8, 16, 16))

        # Upper triangular (causal) masked softmax tests
        self.test_scaled_upper_triang_masked_softmax_forward((16, 32, 32))
        self.test_scaled_upper_triang_masked_softmax_forward((8, 64, 64))
        self.test_scaled_upper_triang_masked_softmax_backward((16, 32, 32))
        self.test_scaled_upper_triang_masked_softmax_backward((8, 64, 64))

        # Aligned causal masked softmax tests (4D tensor required by CUDA)
        self.test_scaled_aligned_causal_masked_softmax_forward((2, 4, 32, 32))
        self.test_scaled_aligned_causal_masked_softmax_backward((2, 4, 32, 32))

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = SoftmaxTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
