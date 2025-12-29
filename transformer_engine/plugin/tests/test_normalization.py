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


class NormalizationTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Normalization Functions",
            "Test correctness of LayerNorm and RMSNorm across backends"
        )
        self.backends = get_available_backends()
        self.eps = 1e-5
        self.device = device

    def _reference_layernorm_forward(self, x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        rsigma = torch.rsqrt(var + eps)
        normalized = (x - mean) * rsigma
        output = normalized * weight + bias
        return output, mean.squeeze(-1), rsigma.squeeze(-1)

    def _reference_rmsnorm_forward(self, x, weight, eps):
        var = (x ** 2).mean(dim=-1, keepdim=True)
        rsigma = torch.rsqrt(var + eps)
        normalized = x * rsigma
        output = normalized * weight
        return output, None, rsigma.squeeze(-1)

    def test_layernorm_forward(self, shape=(2, 4, 8)):
        print(f"\n  Testing LayerNorm forward with shape {shape}")

        hidden_size = shape[-1]
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        weight = torch.ones(hidden_size, dtype=torch.float32, device=self.device)
        bias = torch.zeros(hidden_size, dtype=torch.float32, device=self.device)

        ref_output, ref_mean, ref_rsigma = self._reference_layernorm_forward(
            x, weight, bias, self.eps
        )

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output, mean, rsigma = backend.layernorm_fwd(
                    x, weight, bias, self.eps,
                    None, None, torch.float32, 0, False
                )
                self.assert_close(
                    output, ref_output, rtol=1e-5, atol=1e-7,
                    msg=f"LayerNorm forward output mismatch for {backend_name}"
                )
                self.assert_close(
                    mean, ref_mean, rtol=1e-5, atol=1e-7,
                    msg=f"LayerNorm forward mean mismatch for {backend_name}"
                )
                self.assert_close(
                    rsigma, ref_rsigma, rtol=1e-4, atol=1e-6,
                    msg=f"LayerNorm forward rsigma mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_layernorm_backward(self, shape=(2, 4, 8)):
        print(f"\n  Testing LayerNorm backward with shape {shape}")

        hidden_size = shape[-1]
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        weight = torch.ones(hidden_size, dtype=torch.float32, device=self.device, requires_grad=True)
        bias = torch.zeros(hidden_size, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        output, mean, rsigma = self._reference_layernorm_forward(x, weight, bias, self.eps)
        output.backward(grad_output)
        ref_grad_x = x.grad.clone()
        ref_grad_weight = weight.grad.clone()
        ref_grad_bias = bias.grad.clone()

        x.grad = None
        weight.grad = None
        bias.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                x_copy = x.detach()
                weight_copy = weight.detach()

                grad_x, grad_weight, grad_bias = backend.layernorm_bwd(
                    grad_output, x_copy, mean.detach(), rsigma.detach(),
                    weight_copy, 0, False
                )

                self.assert_close(
                    grad_x, ref_grad_x, rtol=1e-4, atol=1e-6,
                    msg=f"LayerNorm backward grad_x mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_weight, ref_grad_weight, rtol=1e-4, atol=1e-6,
                    msg=f"LayerNorm backward grad_weight mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-5,
                    msg=f"LayerNorm backward grad_bias mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_rmsnorm_forward(self, shape=(2, 4, 8)):
        print(f"\n  Testing RMSNorm forward with shape {shape}")

        hidden_size = shape[-1]
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        weight = torch.ones(hidden_size, dtype=torch.float32, device=self.device)

        ref_output, _, ref_rsigma = self._reference_rmsnorm_forward(x, weight, self.eps)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output, _, rsigma = backend.rmsnorm_fwd(
                    x, weight, self.eps,
                    None, None, torch.float32, 0, False
                )
                self.assert_close(
                    output, ref_output, rtol=1e-5, atol=1e-7,
                    msg=f"RMSNorm forward output mismatch for {backend_name}"
                )
                self.assert_close(
                    rsigma, ref_rsigma, rtol=1e-4, atol=1e-6,
                    msg=f"RMSNorm forward rsigma mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_rmsnorm_backward(self, shape=(2, 4, 8)):
        print(f"\n  Testing RMSNorm backward with shape {shape}")

        hidden_size = shape[-1]
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        weight = torch.ones(hidden_size, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        output, _, rsigma = self._reference_rmsnorm_forward(x, weight, self.eps)
        output.backward(grad_output)
        ref_grad_x = x.grad.clone()
        ref_grad_weight = weight.grad.clone()

        x.grad = None
        weight.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                x_copy = x.detach()
                weight_copy = weight.detach()

                grad_x, grad_weight = backend.rmsnorm_bwd(
                    grad_output, x_copy, rsigma.detach(),
                    weight_copy, 0, False, self.eps
                )

                self.assert_close(
                    grad_x, ref_grad_x, rtol=1e-4, atol=1e-6,
                    msg=f"RMSNorm backward grad_x mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_weight, ref_grad_weight, rtol=1e-4, atol=1e-6,
                    msg=f"RMSNorm backward grad_weight mismatch for {backend_name}"
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
        print("Testing Normalization Functions")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        shapes = [
            (8, 16),
            (32, 64),
            (64, 128),
            (16, 256),
        ]

        for shape in shapes:
            self.test_layernorm_forward(shape)
            self.test_layernorm_backward(shape)
            self.test_rmsnorm_forward(shape)
            self.test_rmsnorm_backward(shape)

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = NormalizationTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
