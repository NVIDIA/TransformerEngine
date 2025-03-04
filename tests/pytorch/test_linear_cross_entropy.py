# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import typing
import pytest
import torch
from transformer_engine.pytorch.linear_cross_entropy import linear_cross_entropy

def run_torch_entropy(hidden: torch.Tensor,
                    weight: torch.Tensor,
                    labels: torch.Tensor) -> typing.List[torch.Tensor]:
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32)) # [num_tokens, vocab_size]
    pd = torch.nn.functional.softmax(logits, dim=-1) # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1) # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1) # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels) # [1]
    return logprobs, entropy

def vanilla_test():
    num_tokens = 80
    hidden_size = 4096
    vocab_size = 152064

    dtype = torch.bfloat16

    enabled_fileds = {
        "forward": {"Torch": True, "Kernel": True},
        "backward": {"Torch": True, "Kernel": True}
    }

    # set_backward_method(BackwardEnum._Total_Separate)

    iterations = 5
    for i in range(iterations):
        print(f"[INFO]: ---------- Iteration {i} starts. ----------")
        with torch.cuda.nvtx.range(f"iteration_{i}"):
            hidden = (torch.empty((num_tokens, hidden_size), dtype=dtype, device="cuda")
                    .uniform_(-0.5, 0.5)
                    .requires_grad_())
            weight = (torch.empty((hidden_size, vocab_size), dtype=dtype, device="cuda")
                    .uniform_(-0.5, 0.5)
                    .requires_grad_())
            labels = torch.randint(0, vocab_size, (num_tokens,), device="cuda")

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            if enabled_fileds["forward"]["Torch"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
                end_event.record()
                torch.cuda.synchronize()
                torch_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Forward pass: Torch implementation peak memory: {torch_max_memory:.2f} MB")
                print(f"[INFO]: Forward pass: Torch implementation time: {start_event.elapsed_time(end_event):.2f} ms")

            if enabled_fileds["forward"]["Kernel"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels)
                end_event.record()
                torch.cuda.synchronize()
                kernel_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Forward pass: Kernel implementation peak memory: {kernel_max_memory:.2f} MB")
                print(f"[INFO]: Forward pass: Kernel implementation time: {start_event.elapsed_time(end_event):.2f} ms")

                if enabled_fileds["forward"]["Torch"]:
                    torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-3, rtol=1e-3)
                    torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-3, rtol=1e-3)
                    print(f"[INFO]: Forward pass: Kernel implementation passed.")

            if enabled_fileds["backward"]["Torch"] or enabled_fileds["backward"]["Kernel"]:
                g_entropy = (torch.empty((num_tokens,), dtype=dtype, device="cuda")
                            .uniform_(-0.5, 0.5))
                g_logprobs = (torch.empty((), dtype=dtype, device="cuda")
                            .uniform_(-1, 1))

            if enabled_fileds["backward"]["Torch"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
                end_event.record()
                torch.cuda.synchronize()
                torch_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Backward pass: torch implementation peak memory: {torch_backward_max_memory:.2f} MB")
                print(f"[INFO]: Backward pass: torch gradient time: {start_event.elapsed_time(end_event):.2f} ms")

            if enabled_fileds["backward"]["Kernel"]:
                torch.cuda.reset_peak_memory_stats()
                start_event.record()
                (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
                end_event.record()
                torch.cuda.synchronize()
                kernel_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
                print(f"[INFO]: Backward pass: kernel implementation peak memory: {kernel_backward_max_memory:.2f} MB")
                print(f"[INFO]: Backward pass: kernel gradient time: {start_event.elapsed_time(end_event):.2f} ms")

                if enabled_fileds["backward"]["Torch"]:
                    torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=1e-2, rtol=1e-3)
                    torch.testing.assert_close(d_torch_weight, d_kernel_weight, atol=1e-2, rtol=1e-3)
                    print(f"[INFO]: Backward pass: kernel implementation passed.")

class TestLinearCrossEntropy:
    def clearnup(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import gc
        gc.collect()
        torch.cuda.synchronize()
    

    def generate_hyper(self):
        self.num_tokens = 80
        self.hidden_size = 4096
        self.vocab_size = 152064
        self.dtype = torch.bfloat16

    def generate_forward_input(self):
        hidden = (torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        weight = (torch.empty((self.hidden_size, self.vocab_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        labels = torch.randint(0, self.vocab_size, (self.num_tokens,), device="cuda")
        return hidden, weight, labels
    
    def generate_backward_input(self):
        g_entropy = (torch.empty((self.num_tokens,), dtype=self.dtype, device="cuda")
                            .uniform_(-0.5, 0.5))
        g_logprobs = (torch.empty((), dtype=self.dtype, device="cuda")
                            .uniform_(-1, 1))
        return g_entropy, g_logprobs

    def test_correctness(self):
        self.clearnup()
        self.generate_hyper()

        iterations = 5
        torch_forward_latency = list()
        torch_backward_latency = list()
        kernel_forward_latency = list()
        kernel_backward_latency = list()
        for i in range(iterations):
            hidden, weight, labels = self.generate_forward_input()
            g_entropy, g_logprobs = self.generate_backward_input()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-3, rtol=1e-3)

            start_event.record()
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            kernel_backward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=1e-2, rtol=1e-3)
            torch.testing.assert_close(d_torch_weight, d_kernel_weight, atol=1e-2, rtol=1e-3)
        
        # remove first latency
        torch_forward_latency = torch_forward_latency[1:]
        torch_backward_latency = torch_backward_latency[1:]
        kernel_forward_latency = kernel_forward_latency[1:]
        kernel_backward_latency = kernel_backward_latency[1:]   

        print()
        print(f"[INFO]: Forward pass: Torch implementation average time: {sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms")
        print(f"[INFO]: Backward pass: torch implementation average time: {sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms")
        print(f"[INFO]: Forward pass: Kernel implementation average time: {sum(kernel_forward_latency) / len(kernel_forward_latency):.2f} ms")
        print(f"[INFO]: Backward pass: kernel implementation average time: {sum(kernel_backward_latency) / len(kernel_backward_latency):.2f} ms")

    def test_torch_storage(self):
        self.clearnup()
        self.generate_hyper()
        hidden, weight, labels = self.generate_forward_input()
        g_entropy, g_logprobs = self.generate_backward_input()

        print()
        torch.cuda.reset_peak_memory_stats()
        (torch_logprobs, torch_entropy) = run_torch_entropy(hidden, weight, labels)
        torch.cuda.synchronize()
        torch_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Torch Forward pass peak memory: {torch_max_memory:.2f} MB")

        torch.cuda.reset_peak_memory_stats()
        (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_entropy, torch_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
        torch.cuda.synchronize()
        torch_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Torch Backward pass peak memory: {torch_backward_max_memory:.2f} MB")
        
    def test_kernel_storage(self):
        self.clearnup()
        self.generate_hyper()
        hidden, weight, labels = self.generate_forward_input()
        g_entropy, g_logprobs = self.generate_backward_input()

        print(end="\n")
        torch.cuda.reset_peak_memory_stats()
        (kernel_logprobs, kernel_entropy) = linear_cross_entropy(hidden, weight, labels)
        torch.cuda.synchronize()
        kernel_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Kernel Forward pass peak memory: {kernel_max_memory:.2f} MB")

        torch.cuda.reset_peak_memory_stats()
        (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_entropy, kernel_logprobs),
                                                                            (hidden, weight),
                                                                            (g_entropy, g_logprobs),
                                                                            retain_graph=False)
        torch.cuda.synchronize()
        kernel_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Kernel Backward pass peak memory: {kernel_backward_max_memory:.2f} MB")
        
        
        
        
        
        
        

if __name__ == "__main__":
    vanilla_test()