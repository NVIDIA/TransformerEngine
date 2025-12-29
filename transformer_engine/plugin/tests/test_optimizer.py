# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
import math

from transformer_engine.plugin.test_utils import (
    get_available_backends,
    get_backend,
    TestCase,
    generate_random_tensor,
)


class OptimizerTests(TestCase):
    def __init__(self, device="cpu"):
        super().__init__(
            "Optimizer Operations",
            "Test correctness of multi_tensor optimizer operations across backends"
        )
        self.backends = get_available_backends()
        self.device = device

    def _reference_multi_tensor_l2norm(self, tensors, per_tensor=False):
        """Reference implementation for multi_tensor_l2norm"""
        if per_tensor:
            return [torch.norm(t.float(), p=2) for t in tensors]
        else:
            total_norm_sq = sum(torch.norm(t.float(), p=2) ** 2 for t in tensors)
            return torch.sqrt(total_norm_sq)

    def test_multi_tensor_scale(self, num_tensors=4, shape=(64, 128)):
        print(f"\n  Testing multi_tensor_scale with {num_tensors} tensors of shape {shape}")

        scale = 0.5

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                # Create input tensors
                input_tensors = [generate_random_tensor(shape, dtype=torch.float32, device=self.device)
                                 for _ in range(num_tensors)]
                # Create output tensors (will be filled by the function)
                output_tensors = [torch.empty_like(t) for t in input_tensors]
                # Create reference tensors
                ref_tensors = [t.clone() * scale for t in input_tensors]

                # Apply backend scaling: tensor_lists = [input_tensors, output_tensors]
                noop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
                backend.multi_tensor_scale(
                    chunk_size=2048,
                    noop_flag=noop_flag,
                    tensor_lists=[input_tensors, output_tensors],
                    scale=scale
                )

                # Compare results
                for i, (output, reference) in enumerate(zip(output_tensors, ref_tensors)):
                    self.assert_close(
                        output, reference, rtol=1e-5, atol=1e-7,
                        msg=f"multi_tensor_scale tensor {i} mismatch for {backend_name}"
                    )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_multi_tensor_l2norm(self, num_tensors=4, shape=(64, 128)):
        print(f"\n  Testing multi_tensor_l2norm with {num_tensors} tensors of shape {shape}")

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                tensors = [generate_random_tensor(shape, dtype=torch.float32, device=self.device)
                           for _ in range(num_tensors)]

                # Reference computation
                ref_norm = self._reference_multi_tensor_l2norm(tensors, per_tensor=False)

                # Backend computation
                noop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
                output_norm = backend.multi_tensor_l2norm(
                    chunk_size=2048,
                    noop_flag=noop_flag,
                    tensor_lists=[tensors],
                    per_tensor=False
                )

                # CUDA backend returns tuple (norm, per_tensor_norms), extract the first element
                if isinstance(output_norm, tuple):
                    output_norm = output_norm[0]

                self.assert_close(
                    output_norm, ref_norm, rtol=1e-4, atol=1e-6,
                    msg=f"multi_tensor_l2norm total norm mismatch for {backend_name}"
                )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_multi_tensor_l2norm_per_tensor(self, num_tensors=4, shape=(64, 128)):
        print(f"\n  Testing multi_tensor_l2norm per_tensor with {num_tensors} tensors of shape {shape}")

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                tensors = [generate_random_tensor(shape, dtype=torch.float32, device=self.device)
                           for _ in range(num_tensors)]

                # Reference computation
                ref_norms = self._reference_multi_tensor_l2norm(tensors, per_tensor=True)

                # Backend computation
                noop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
                output_norms = backend.multi_tensor_l2norm(
                    chunk_size=2048,
                    noop_flag=noop_flag,
                    tensor_lists=[tensors],
                    per_tensor=True
                )

                # CUDA backend returns tuple (total_norm, per_tensor_norms), extract second element
                if isinstance(output_norms, tuple):
                    output_norms = output_norms[1]

                for i, (output, reference) in enumerate(zip(output_norms, ref_norms)):
                    self.assert_close(
                        output, reference, rtol=1e-4, atol=1e-6,
                        msg=f"multi_tensor_l2norm per_tensor {i} mismatch for {backend_name}"
                    )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def test_multi_tensor_adam(self, num_tensors=3, shape=(32, 64)):
        print(f"\n  Testing multi_tensor_adam with {num_tensors} tensors of shape {shape}")

        lr = 0.001
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        step = 1
        weight_decay = 0.01

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                # Create tensors for backend test
                params = [generate_random_tensor(shape, dtype=torch.float32, device=self.device)
                          for _ in range(num_tensors)]
                grads = [generate_random_tensor(shape, dtype=torch.float32, device=self.device)
                         for _ in range(num_tensors)]
                exp_avgs = [torch.zeros_like(p) for p in params]
                exp_avg_sqs = [torch.zeros_like(p) for p in params]

                # Create reference tensors with same values
                ref_params = [p.clone() for p in params]
                ref_grads = [g.clone() for g in grads]
                ref_exp_avgs = [torch.zeros_like(p) for p in params]
                ref_exp_avg_sqs = [torch.zeros_like(p) for p in params]

                # Apply reference Adam step (matching the torch implementation)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                for p, g, m, v in zip(ref_params, ref_grads, ref_exp_avgs, ref_exp_avg_sqs):
                    # AdamW style: weight decay applied to param first
                    p.mul_(1 - lr * weight_decay)

                    # Update biased first moment estimate
                    m.mul_(beta1).add_(g, alpha=1 - beta1)
                    # Update biased second raw moment estimate
                    v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    # Compute bias-corrected estimates
                    corrected_m = m / bias_correction1
                    corrected_v = v / bias_correction2

                    # Update parameters
                    denom = corrected_v.sqrt().add_(eps)
                    p.addcdiv_(corrected_m, denom, value=-lr)

                # Apply backend Adam step
                noop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
                backend.multi_tensor_adam(
                    chunk_size=2048,
                    noop_flag=noop_flag,
                    tensor_lists=[grads, params, exp_avgs, exp_avg_sqs],
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    step=step,
                    mode=1,  # AdamW mode
                    bias_correction=1,
                    weight_decay=weight_decay
                )

                # Compare results with relaxed tolerance
                for i, (output, reference) in enumerate(zip(params, ref_params)):
                    self.assert_close(
                        output, reference, rtol=1e-3, atol=1e-5,
                        msg=f"multi_tensor_adam param {i} mismatch for {backend_name}"
                    )
                print(f"    ✓ {backend_name}")
            except NotImplementedError:
                self.skipped += 1
                print(f"    ⊘ {backend_name} (not implemented)")
            except Exception as e:
                self.failed += 1
                print(f"    ✗ {backend_name}: {e}")

    def _reference_multi_tensor_unscale_l2norm(self, tensors, inv_scale, per_tensor=False):
        """Reference implementation for multi_tensor_unscale_l2norm.

        Computes L2 norm of tensors after unscaling.
        Note: scale parameter is actually inv_scale (1/loss_scale).
        Unscaling means multiplying by inv_scale (= dividing by loss_scale).
        """
        inv_scale_value = inv_scale.item() if isinstance(inv_scale, torch.Tensor) else inv_scale
        # Unscale (multiply by inv_scale) and compute L2 norm
        if per_tensor:
            return [torch.norm(t.float() * inv_scale_value, p=2) for t in tensors]
        else:
            total_norm_sq = sum(torch.norm(t.float() * inv_scale_value, p=2) ** 2 for t in tensors)
            return torch.sqrt(total_norm_sq)

    def test_multi_tensor_unscale_l2norm(self, num_tensors=4, shape=(64, 128)):
        print(f"\n  Testing multi_tensor_unscale_l2norm with {num_tensors} tensors of shape {shape}")

        # Note: scale parameter is actually inv_scale (1/loss_scale)
        # For AMP with loss_scale=1024, inv_scale would be 1/1024
        inv_scale_value = 0.5  # equivalent to loss_scale = 2.0
        tensors = [generate_random_tensor(shape, dtype=torch.float32, device=self.device)
                   for _ in range(num_tensors)]
        noop_flag = torch.tensor([0], dtype=torch.int32, device=self.device)
        inv_scale = torch.tensor([inv_scale_value], dtype=torch.float32, device=self.device)

        # Compute mathematical reference
        reference_norm = self._reference_multi_tensor_unscale_l2norm(tensors, inv_scale, per_tensor=False)

        for backend_name in self.backends:
            backend = get_backend(backend_name)
            try:
                output_norm = backend.multi_tensor_unscale_l2norm(
                    chunk_size=2048,
                    noop_flag=noop_flag,
                    tensor_lists=[tensors],
                    scale=inv_scale,
                    per_tensor=False
                )

                # CUDA backend returns tuple (norm, per_tensor_norms), extract the first element
                if isinstance(output_norm, tuple):
                    output_norm = output_norm[0]

                self.assert_close(
                    output_norm, reference_norm, rtol=1e-4, atol=1e-6,
                    msg=f"multi_tensor_unscale_l2norm mismatch for {backend_name}"
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
        print("Testing Optimizer Operations")
        print("="*60)
        print(f"Available backends: {', '.join(self.backends)}")

        # multi_tensor_scale tests
        self.test_multi_tensor_scale(num_tensors=4, shape=(64, 128))
        self.test_multi_tensor_scale(num_tensors=8, shape=(128, 256))

        # multi_tensor_l2norm tests
        self.test_multi_tensor_l2norm(num_tensors=4, shape=(64, 128))
        self.test_multi_tensor_l2norm_per_tensor(num_tensors=4, shape=(64, 128))

        # multi_tensor_unscale_l2norm tests
        self.test_multi_tensor_unscale_l2norm(num_tensors=4, shape=(64, 128))

        # multi_tensor_adam tests
        self.test_multi_tensor_adam(num_tensors=3, shape=(32, 64))

        return self.report()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    test_suite = OptimizerTests(device=device)
    success = test_suite.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
