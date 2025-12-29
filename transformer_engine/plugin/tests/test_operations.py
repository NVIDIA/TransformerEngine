# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
import torch.nn.functional as F
import sys

from transformer_engine.plugin.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
)


class OperationsTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Operations (GEMM, Softmax, Dropout)",
            "Test correctness of GEMM, Softmax, and Dropout operations"
        )
        self.backends = get_available_backends()
        self.device = device

    def test_gemm_basic(self, M=32, N=64, K=48):
        print(f"\n  Testing GEMM ({M}x{K}) @ ({K}x{N})")

        A = generate_random_tensor((K, N), dtype=torch.float32, device=self.device)
        B = generate_random_tensor((M, K), dtype=torch.float32, device=self.device)
        reference = B @ A

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                D = torch.empty((M, N), dtype=torch.float32, device=self.device)
                workspace = torch.empty(1024, dtype=torch.uint8, device=self.device)

                output, _, _, _ = backend.generic_gemm(
                    A, False, B, False, D,
                    None, torch.float32, None, None,
                    False, None, False,
                    workspace, 1024, False, False
                )

                self.assert_close(
                    output, reference, rtol=5e-2, atol=1e-2,
                    msg=f"GEMM output mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_gemm_transpose_a(self, M=32, N=64, K=48):
        print(f"\n  Testing GEMM transpose A ({N}x{K}).T @ ({M}x{K})")

        A = generate_random_tensor((N, K), dtype=torch.float32, device=self.device)
        B = generate_random_tensor((M, K), dtype=torch.float32, device=self.device)
        reference = B @ A.T

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                D = torch.empty((M, N), dtype=torch.float32, device=self.device)
                workspace = torch.empty(1024, dtype=torch.uint8, device=self.device)

                output, _, _, _ = backend.generic_gemm(
                    A, True, B, False, D,
                    None, torch.float32, None, None,
                    False, None, False,
                    workspace, 1024, False, False
                )

                self.assert_close(
                    output, reference, rtol=5e-2, atol=1e-2,
                    msg=f"GEMM transpose A mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_gemm_3d(self, B=2, M=16, N=32, K=24):
        print(f"\n  Testing 3D GEMM ({B}x{M}x{K}) @ ({K}x{N})")

        A = generate_random_tensor((B, M, K), dtype=torch.float32, device=self.device)
        B_mat = generate_random_tensor((K, N), dtype=torch.float32, device=self.device)
        reference = torch.matmul(A, B_mat)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                D = torch.empty((B, M, N), dtype=torch.float32, device=self.device)
                workspace = torch.empty(1024, dtype=torch.uint8, device=self.device)

                output, _, _, _ = backend.generic_gemm(
                    B_mat, False, A, False, D,
                    None, torch.float32, None, None,
                    False, None, False,
                    workspace, 1024, False, False
                )

                self.assert_close(
                    output, reference, rtol=5e-2, atol=1e-2,
                    msg=f"3D GEMM mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_scaled_softmax(self, shape=(2, 4, 8, 16)):
        print(f"\n  Testing scaled softmax with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)
        scale = 0.125
        reference = F.softmax(x.float() * scale, dim=-1).to(x.dtype)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.scaled_softmax_forward(x, scale)
                self.assert_close(
                    output, reference, rtol=1e-2, atol=1e-3,
                    msg=f"Scaled softmax mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_causal_masked_softmax(self, shape=(8, 16, 16)):
        print(f"\n  Testing causal masked softmax with shape {shape}")

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
                    msg=f"Causal masked softmax mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_dropout(self, shape=(4, 8, 16)):
        print(f"\n  Testing dropout with shape {shape}")

        x = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)
        dropout_prob = 0.1

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output, mask = backend.dropout_fwd(x, dropout_prob)

                num_nonzero = (output != 0).sum().item()
                total_elements = output.numel()
                nonzero_ratio = num_nonzero / total_elements
                expected_ratio = 1.0 - dropout_prob

                assert abs(nonzero_ratio - expected_ratio) < 0.2, \
                    f"Dropout ratio mismatch for {backend_name}: {nonzero_ratio:.3f} vs {expected_ratio:.3f}"

                assert torch.all(output[output == 0] == 0), \
                    f"Dropped elements should be zero for {backend_name}"

                expected_scale = 1.0 / (1.0 - dropout_prob)
                non_zero_output = output[output != 0]
                non_zero_input = x[output != 0]

                if len(non_zero_output) > 0:
                    self.assert_close(
                        non_zero_output, non_zero_input * expected_scale,
                        rtol=1e-2, atol=1e-3,
                        msg=f"Dropout scaling mismatch for {backend_name}"
                    )

                grad_output = generate_random_tensor(shape, dtype=torch.bfloat16, device=self.device)
                grad_input = backend.dropout_bwd(grad_output, mask, dropout_prob)

                grad_nonzero_mask = (grad_input != 0)
                output_nonzero_mask = (output != 0)
                assert torch.all(grad_nonzero_mask == output_nonzero_mask), \
                    f"Dropout backward sparsity mismatch for {backend_name}"

                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def run_all_tests(self):
        print("\n" + "="*60)
        print("Testing Operations (GEMM, Softmax, Dropout)")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        self.test_gemm_basic(M=32, N=64, K=48)
        self.test_gemm_basic(M=64, N=128, K=96)
        self.test_gemm_transpose_a(M=32, N=64, K=48)
        self.test_gemm_3d(B=2, M=16, N=32, K=24)

        self.test_scaled_softmax((4, 8, 16, 16))
        self.test_scaled_softmax((2, 4, 32, 32))
        self.test_causal_masked_softmax((16, 32, 32))
        self.test_causal_masked_softmax((8, 64, 64))

        self.test_dropout((4, 8, 16))
        self.test_dropout((8, 16, 32))

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = OperationsTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
