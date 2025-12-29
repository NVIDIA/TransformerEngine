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
    generate_test_shapes,
)


class ActivationTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Activation Functions",
            "Test correctness of all activation functions across backends"
        )
        self.backends = get_available_backends()
        self.reference_backend = "reference"
        self.device = device

    # ==================== Reference implementations ====================
    def _get_reference_gelu(self, x):
        return F.gelu(x, approximate='tanh')

    def _get_reference_geglu(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.gelu(a, approximate='tanh') * b

    def _get_reference_qgelu(self, x):
        return x * torch.sigmoid(1.702 * x)

    def _get_reference_qgeglu(self, x):
        a, b = x.chunk(2, dim=-1)
        return a * torch.sigmoid(1.702 * a) * b

    def _get_reference_relu(self, x):
        return F.relu(x)

    def _get_reference_reglu(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.relu(a) * b

    def _get_reference_srelu(self, x):
        return torch.square(F.relu(x))

    def _get_reference_sreglu(self, x):
        a, b = x.chunk(2, dim=-1)
        return torch.square(F.relu(a)) * b

    def _get_reference_silu(self, x):
        return F.silu(x)

    def _get_reference_swiglu(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

    def _get_reference_clamped_swiglu(self, x, limit=7.0, alpha=1.702):
        """Reference implementation matching CUDA clamped_swiglu.

        CUDA implementation:
        - a (activation): clamp to upper bound only: min(a, limit)
        - b (gate): clamp to [-limit, limit], then add 1
        - output = (a_clamped * sigmoid(alpha * a_clamped)) * b_clamped
        """
        a, b = x.chunk(2, dim=-1)
        # CUDA only clamps a to upper bound
        a_clamped = torch.clamp(a, max=limit)
        # CUDA clamps b to [-limit, limit] and adds 1
        b_clamped = torch.clamp(b, -limit, limit) + 1
        return a_clamped * torch.sigmoid(alpha * a_clamped) * b_clamped

    # ==================== Forward tests ====================
    def test_gelu_forward(self, shape=(4, 8)):
        print(f"\n  Testing GELU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_gelu(x)
        self._test_activation_forward("gelu", x, reference)

    def test_geglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing GEGLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_geglu(x)
        self._test_activation_forward("geglu", x, reference)

    def test_qgelu_forward(self, shape=(4, 8)):
        print(f"\n  Testing QGELU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_qgelu(x)
        self._test_activation_forward("qgelu", x, reference)

    def test_qgeglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing QGEGLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_qgeglu(x)
        self._test_activation_forward("qgeglu", x, reference)

    def test_relu_forward(self, shape=(4, 8)):
        print(f"\n  Testing ReLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_relu(x)
        self._test_activation_forward("relu", x, reference, rtol=1e-6, atol=1e-8)

    def test_reglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing ReGLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_reglu(x)
        self._test_activation_forward("reglu", x, reference, rtol=1e-6, atol=1e-8)

    def test_srelu_forward(self, shape=(4, 8)):
        print(f"\n  Testing SReLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_srelu(x)
        self._test_activation_forward("srelu", x, reference)

    def test_sreglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing SReGLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_sreglu(x)
        self._test_activation_forward("sreglu", x, reference)

    def test_silu_forward(self, shape=(4, 8)):
        print(f"\n  Testing SiLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_silu(x)
        self._test_activation_forward("silu", x, reference)

    def test_swiglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing SwiGLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_swiglu(x)
        self._test_activation_forward("swiglu", x, reference)

    def test_clamped_swiglu_forward(self, shape=(4, 16)):
        print(f"\n  Testing Clamped SwiGLU forward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        reference = self._get_reference_clamped_swiglu(x)
        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output = backend.clamped_swiglu(x, None, 7.0, 1.702)
                self.assert_close(
                    output, reference, rtol=1e-4, atol=1e-6,
                    msg=f"clamped_swiglu forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def _test_activation_forward(self, op_name, x, reference, rtol=1e-4, atol=1e-6):
        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                op_fn = getattr(backend, op_name)
                output = op_fn(x, None)
                self.assert_close(
                    output, reference, rtol=rtol, atol=atol,
                    msg=f"{op_name} forward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    # ==================== Backward tests ====================
    def test_gelu_backward(self, shape=(4, 8)):
        print(f"\n  Testing GELU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        y = self._get_reference_gelu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dgelu", x, grad_output, reference_grad)

    def test_geglu_backward(self, shape=(4, 16)):
        print(f"\n  Testing GEGLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor((shape[0], shape[1] // 2) if len(shape) == 2 else (*shape[:-1], shape[-1] // 2),
                                              dtype=torch.float32, device=self.device)
        y = self._get_reference_geglu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dgeglu", x, grad_output, reference_grad)

    def test_qgelu_backward(self, shape=(4, 8)):
        print(f"\n  Testing QGELU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        y = self._get_reference_qgelu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dqgelu", x, grad_output, reference_grad)

    def test_qgeglu_backward(self, shape=(4, 16)):
        print(f"\n  Testing QGEGLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor((shape[0], shape[1] // 2) if len(shape) == 2 else (*shape[:-1], shape[-1] // 2),
                                              dtype=torch.float32, device=self.device)
        y = self._get_reference_qgeglu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dqgeglu", x, grad_output, reference_grad)

    def test_relu_backward(self, shape=(4, 8)):
        print(f"\n  Testing ReLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        y = self._get_reference_relu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("drelu", x, grad_output, reference_grad)

    def test_reglu_backward(self, shape=(4, 16)):
        print(f"\n  Testing ReGLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor((shape[0], shape[1] // 2) if len(shape) == 2 else (*shape[:-1], shape[-1] // 2),
                                              dtype=torch.float32, device=self.device)
        y = self._get_reference_reglu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dreglu", x, grad_output, reference_grad)

    def test_srelu_backward(self, shape=(4, 8)):
        print(f"\n  Testing SReLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        y = self._get_reference_srelu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dsrelu", x, grad_output, reference_grad)

    def test_sreglu_backward(self, shape=(4, 16)):
        print(f"\n  Testing SReGLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor((shape[0], shape[1] // 2) if len(shape) == 2 else (*shape[:-1], shape[-1] // 2),
                                              dtype=torch.float32, device=self.device)
        y = self._get_reference_sreglu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dsreglu", x, grad_output, reference_grad)

    def test_silu_backward(self, shape=(4, 8)):
        print(f"\n  Testing SiLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)
        y = self._get_reference_silu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dsilu", x, grad_output, reference_grad)

    def test_swiglu_backward(self, shape=(4, 16)):
        print(f"\n  Testing SwiGLU backward with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor((shape[0], shape[1] // 2) if len(shape) == 2 else (*shape[:-1], shape[-1] // 2),
                                              dtype=torch.float32, device=self.device)
        y = self._get_reference_swiglu(x)
        y.backward(grad_output)
        reference_grad = x.grad.clone()
        x.grad = None
        self._test_activation_backward("dswiglu", x, grad_output, reference_grad)

    def _test_activation_backward(self, op_name, x, grad_output, reference_grad, rtol=1e-4, atol=1e-6):
        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                op_fn = getattr(backend, op_name)
                grad_input = op_fn(grad_output, x.detach(), None)
                self.assert_close(
                    grad_input, reference_grad, rtol=rtol, atol=atol,
                    msg=f"{op_name} backward mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    # ==================== Bias + backward tests ====================
    def test_dbias_dgelu(self, shape=(4, 8)):
        print(f"\n  Testing dbias_dgelu with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        # Reference: compute dgelu and sum for bias grad
        y = self._get_reference_gelu(x)
        y.backward(grad_output)
        ref_grad_input = x.grad.clone()
        ref_grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        x.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                grad_input, grad_bias = backend.dbias_dgelu(grad_output, x.detach(), None)
                self.assert_close(
                    grad_input, ref_grad_input, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dgelu grad_input mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dgelu grad_bias mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except RuntimeError as e:
                # CUDA requires a valid quantizer for dbias_d* fused ops
                if "NoneQuantizer does not support" in str(e):
                    self.skipped += 1
                    print(f"    ⊘ {backend_name} (requires FP8 quantizer for fused op)")
                else:
                    self.failed += 1
                    print(f"    ✗ {backend_name}: {e}")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_dbias_dsilu(self, shape=(4, 8)):
        print(f"\n  Testing dbias_dsilu with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        y = self._get_reference_silu(x)
        y.backward(grad_output)
        ref_grad_input = x.grad.clone()
        ref_grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        x.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                grad_input, grad_bias = backend.dbias_dsilu(grad_output, x.detach(), None)
                self.assert_close(
                    grad_input, ref_grad_input, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dsilu grad_input mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dsilu grad_bias mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except RuntimeError as e:
                # CUDA requires a valid quantizer for dbias_d* fused ops
                if "NoneQuantizer does not support" in str(e):
                    self.skipped += 1
                    print(f"    ⊘ {backend_name} (requires FP8 quantizer for fused op)")
                else:
                    self.failed += 1
                    print(f"    ✗ {backend_name}: {e}")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_dbias_drelu(self, shape=(4, 8)):
        print(f"\n  Testing dbias_drelu with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        y = self._get_reference_relu(x)
        y.backward(grad_output)
        ref_grad_input = x.grad.clone()
        ref_grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        x.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                grad_input, grad_bias = backend.dbias_drelu(grad_output, x.detach(), None)
                self.assert_close(
                    grad_input, ref_grad_input, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_drelu grad_input mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_drelu grad_bias mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except RuntimeError as e:
                # CUDA requires a valid quantizer for dbias_d* fused ops
                if "NoneQuantizer does not support" in str(e):
                    self.skipped += 1
                    print(f"    ⊘ {backend_name} (requires FP8 quantizer for fused op)")
                else:
                    self.failed += 1
                    print(f"    ✗ {backend_name}: {e}")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_dbias_dqgelu(self, shape=(4, 8)):
        print(f"\n  Testing dbias_dqgelu with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        y = self._get_reference_qgelu(x)
        y.backward(grad_output)
        ref_grad_input = x.grad.clone()
        ref_grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        x.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                grad_input, grad_bias = backend.dbias_dqgelu(grad_output, x.detach(), None)
                self.assert_close(
                    grad_input, ref_grad_input, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dqgelu grad_input mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dqgelu grad_bias mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except RuntimeError as e:
                # CUDA requires a valid quantizer for dbias_d* fused ops
                if "NoneQuantizer does not support" in str(e):
                    self.skipped += 1
                    print(f"    ⊘ {backend_name} (requires FP8 quantizer for fused op)")
                else:
                    self.failed += 1
                    print(f"    ✗ {backend_name}: {e}")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_dbias_dsrelu(self, shape=(4, 8)):
        print(f"\n  Testing dbias_dsrelu with shape {shape}")
        x = generate_random_tensor(shape, dtype=torch.float32, device=self.device, requires_grad=True)
        grad_output = generate_random_tensor(shape, dtype=torch.float32, device=self.device)

        y = self._get_reference_srelu(x)
        y.backward(grad_output)
        ref_grad_input = x.grad.clone()
        ref_grad_bias = grad_output.sum(dim=tuple(range(grad_output.ndim - 1)))
        x.grad = None

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                grad_input, grad_bias = backend.dbias_dsrelu(grad_output, x.detach(), None)
                self.assert_close(
                    grad_input, ref_grad_input, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dsrelu grad_input mismatch for {backend_name}"
                )
                self.assert_close(
                    grad_bias, ref_grad_bias, rtol=1e-4, atol=1e-6,
                    msg=f"dbias_dsrelu grad_bias mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except RuntimeError as e:
                # CUDA requires a valid quantizer for dbias_d* fused ops
                if "NoneQuantizer does not support" in str(e):
                    self.skipped += 1
                    print(f"    ⊘ {backend_name} (requires FP8 quantizer for fused op)")
                else:
                    self.failed += 1
                    print(f"    ✗ {backend_name}: {e}")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def run_all_tests(self):
        print("\n" + "="*60)
        print("Testing Activation Functions")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        shapes = [(4, 8), (8, 16), (2, 4, 8)]
        glu_shapes = [(4, 16), (8, 32), (2, 4, 16)]

        # Forward tests - non-gated activations
        for shape in shapes:
            self.test_gelu_forward(shape)
            self.test_qgelu_forward(shape)
            self.test_relu_forward(shape)
            self.test_srelu_forward(shape)
            self.test_silu_forward(shape)

        # Forward tests - gated activations
        for shape in glu_shapes:
            self.test_geglu_forward(shape)
            self.test_qgeglu_forward(shape)
            self.test_reglu_forward(shape)
            self.test_sreglu_forward(shape)
            self.test_swiglu_forward(shape)
            self.test_clamped_swiglu_forward(shape)

        # Backward tests - non-gated activations
        for shape in shapes:
            self.test_gelu_backward(shape)
            self.test_qgelu_backward(shape)
            self.test_relu_backward(shape)
            self.test_srelu_backward(shape)
            self.test_silu_backward(shape)

        # Backward tests - gated activations
        for shape in glu_shapes:
            self.test_geglu_backward(shape)
            self.test_qgeglu_backward(shape)
            self.test_reglu_backward(shape)
            self.test_sreglu_backward(shape)
            self.test_swiglu_backward(shape)

        # Note: dbias_d* tests are skipped because CUDA requires FP8 quantizer
        # for these fused ops. These will be tested separately with FP8 quantizer.

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = ActivationTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
