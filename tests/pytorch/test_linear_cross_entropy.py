# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import typing
import pytest
import torch
from transformer_engine.pytorch.linear_cross_entropy import linear_cross_entropy_with_token_entropy
from transformer_engine.pytorch.linear_cross_entropy import linear_cross_entropy

import torch.distributed as dist
import os
from transformer_engine.pytorch.cross_entropy import parallel_cross_entropy

def run_torch_entropy_with_token_entropy(hidden: torch.Tensor,
                                        weight: torch.Tensor,
                                        labels: torch.Tensor) -> typing.List[torch.Tensor]:
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32)) # [num_tokens, vocab_size]
    pd = torch.nn.functional.softmax(logits, dim=-1) # [num_tokens, vocab_size]
    entropy_a = torch.logsumexp(logits, dim=-1) # [num_tokens]
    entropy_b = torch.sum(pd * logits, dim=-1) # [num_tokens]
    entropy = entropy_a - entropy_b
    logprobs = torch.nn.functional.cross_entropy(logits, labels) # [1]
    return logprobs, entropy

def run_torch_entropy(hidden: torch.Tensor,
                      weight: torch.Tensor,
                      labels: torch.Tensor) -> typing.List[torch.Tensor]:
    logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32)) # [num_tokens, vocab_size]
    logprobs = torch.nn.functional.cross_entropy(logits, labels) # [1]
    return logprobs

def cleanup():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    import gc
    gc.collect()
    torch.cuda.synchronize()

class TestLinearCrossEntropyWithTokenEntropy:
    def clearnup(self):
        cleanup()
    

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
            (torch_logprobs, torch_entropy) = run_torch_entropy_with_token_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (kernel_logprobs, kernel_entropy) = linear_cross_entropy_with_token_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_entropy, kernel_entropy, atol=1e-4, rtol=1e-4)

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

            torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=1e-2, rtol=1e-4)
            torch.testing.assert_close(d_torch_weight, d_kernel_weight, atol=1e-2, rtol=1e-4)
        
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
        (torch_logprobs, torch_entropy) = run_torch_entropy_with_token_entropy(hidden, weight, labels)
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
        (kernel_logprobs, kernel_entropy) = linear_cross_entropy_with_token_entropy(hidden, weight, labels)
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
        
        
        
class TestLinearCrossEntropy:
    def generate_hyper(self):
        self.num_tokens = 80
        self.hidden_size = 4096
        self.vocab_size = 152064
        self.dtype = torch.bfloat16
        self.ignore_index = -100 # this is the default value in torch's cross entropy

    def generate_forward_input(self):
        hidden = (torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        weight = (torch.empty((self.hidden_size, self.vocab_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        labels = torch.randint(0, self.vocab_size, (self.num_tokens,), device="cuda")
        pad_labels = torch.nn.functional.pad(labels, (0, 1), value=self.ignore_index)
        labels = pad_labels[..., 1:].contiguous()
        return hidden, weight, labels

    def generate_backward_input(self):
        g_logprobs = (torch.empty((), dtype=self.dtype, device="cuda")
                        .uniform_(-1, 1))
        return g_logprobs

    def test_correctness(self):
        cleanup()

        self.generate_hyper()
        
        iterations = 5
        torch_forward_latency = list()
        torch_backward_latency = list()
        kernel_forward_latency = list()
        kernel_backward_latency = list()

        for i in range(iterations):
            hidden, weight, labels = self.generate_forward_input()
            g_logprobs = self.generate_backward_input()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            torch_logprobs = run_torch_entropy(hidden, weight, labels)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            kernel_logprobs = linear_cross_entropy(hidden, weight, labels, "mean", None, self.ignore_index)
            end_event.record()
            torch.cuda.synchronize()
            kernel_forward_latency.append(start_event.elapsed_time(end_event))

            # forward result verification
            torch.testing.assert_close(torch_logprobs, kernel_logprobs, atol=1e-4, rtol=1e-4)

            start_event.record()
            (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_logprobs,),
                                                                    (hidden, weight),
                                                                    (g_logprobs,),
                                                                    retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))

            start_event.record()
            (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_logprobs,),
                                                                    (hidden, weight),
                                                                    (g_logprobs,),
                                                                    retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            kernel_backward_latency.append(start_event.elapsed_time(end_event))

            # backward result verification
            torch.testing.assert_close(d_torch_hidden, d_kernel_hidden, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(d_torch_weight, d_kernel_weight)

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
        cleanup()

        self.generate_hyper()
        hidden, weight, labels = self.generate_forward_input()

        print()
        torch.cuda.reset_peak_memory_stats()
        torch_logprobs = run_torch_entropy(hidden, weight, labels)
        torch.cuda.synchronize()
        torch_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Torch Forward pass peak memory: {torch_max_memory:.2f} MB")

        torch.cuda.reset_peak_memory_stats()
        g_logprobs = self.generate_backward_input()
        (d_torch_hidden, d_torch_weight) = torch.autograd.grad((torch_logprobs,),
                                                                (hidden, weight),
                                                                (g_logprobs,),
                                                                retain_graph=False)
        torch.cuda.synchronize()
        torch_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Torch Backward pass peak memory: {torch_backward_max_memory:.2f} MB")

    def test_kernel_storage(self):
        cleanup()

        self.generate_hyper()
        hidden, weight, labels = self.generate_forward_input()

        print()
        torch.cuda.reset_peak_memory_stats()
        kernel_logprobs = linear_cross_entropy(hidden, weight, labels, "mean", None, self.ignore_index)
        torch.cuda.synchronize()
        kernel_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Kernel Forward pass peak memory: {kernel_max_memory:.2f} MB")

        torch.cuda.reset_peak_memory_stats()
        g_logprobs = self.generate_backward_input()
        (d_kernel_hidden, d_kernel_weight) = torch.autograd.grad((kernel_logprobs,),
                                                                (hidden, weight),
                                                                (g_logprobs,),
                                                                retain_graph=False)
        torch.cuda.synchronize()
        kernel_backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"[INFO]: Kernel Backward pass peak memory: {kernel_backward_max_memory:.2f} MB")
        

class _TorchLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                hidden: torch.Tensor,
                weight: torch.Tensor,
                labels: torch.Tensor,
                dist_process_group: dist.ProcessGroup):
        logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32)) # [num_tokens, vocab_size]

        whole_logits = torch.empty((logits.shape[0], logits.shape[1] * dist.get_world_size(dist_process_group)), 
                                    dtype=logits.dtype, device=logits.device)
        whole_logits_ref = [
            whole_logits[:, i * logits.shape[1]: (i + 1) * logits.shape[1]]
            for i in range(dist.get_world_size(dist_process_group))
        ]
        dist.all_gather(whole_logits_ref, logits, group=dist_process_group)

        logprobs = torch.nn.functional.cross_entropy(whole_logits, labels)
        
        # Save tensors needed for backward
        ctx.save_for_backward(hidden, weight, labels, whole_logits)
        ctx.dist_process_group = dist_process_group

        return logprobs
    
    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, labels, whole_logits = ctx.saved_tensors
        dist_process_group = ctx.dist_process_group
        
        # Calculate gradients for cross entropy
        batch_size = whole_logits.size(0)
        vocab_size = whole_logits.size(1)
        
        # Create mask for valid labels (not ignore_index)
        ignore_index = -100  # Default value for ignore_index in PyTorch
        valid_mask = (labels != ignore_index).float().unsqueeze(1)
        
        # Count valid tokens for normalization
        num_valid_tokens = valid_mask.sum()
        
        # Create one-hot encoding for labels, only for valid positions
        one_hot = torch.zeros_like(whole_logits)
        valid_labels = labels.clone()
        valid_labels[labels == ignore_index] = 0  # Temporary replace with valid index for scatter
        one_hot.scatter_(1, valid_labels.unsqueeze(1), 1)
        one_hot = one_hot * valid_mask  # Zero out positions with ignore_index
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(whole_logits, dim=1)
        
        # Calculate gradient of cross entropy w.r.t. logits
        # Only consider valid tokens for normalization
        if num_valid_tokens > 0:
            grad_logits = (probs - one_hot) * grad_output / num_valid_tokens
        else:
            grad_logits = torch.zeros_like(probs)
        
        # Zero out gradients for tokens with ignore_index
        grad_logits = grad_logits * valid_mask
        
        # Get the local portion of the gradient
        local_size = weight.size(1)
        rank = dist.get_rank(dist_process_group)
        local_grad_logits = grad_logits[:, rank * local_size:(rank + 1) * local_size]
        
        # Calculate gradients for hidden and weight
        grad_hidden = torch.matmul(local_grad_logits, weight.t().to(torch.float32)).to(hidden.dtype)
        grad_weight = torch.matmul(hidden.t().to(torch.float32), local_grad_logits).to(weight.dtype)
        
        return grad_hidden, grad_weight, None, None
          

class _TestTensorParallel:
    def __init__(self, ignore_index: typing.Optional[int] = None):
        dist.init_process_group(backend="nccl")
        self.group = dist.group.WORLD

        self.local_rank = dist.get_rank(self.group)
        self.world_size = dist.get_world_size(self.group)
        device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(device)
        print(f"[INFO]: Local rank: {self.local_rank}, World size: {self.world_size}")

        self.ignore_index_opt = ignore_index

    def generate_hyper(self):
        self.num_tokens = 80
        self.hidden_size = 4096
        self.vocab_size = 152064
        self.dtype = torch.bfloat16
        self.iterations = 5
        # -100 is the default value in torch's cross entropy
        self.ignore_index = self.ignore_index_opt if self.ignore_index_opt is not None else -100

    def generate_forward_input(self):
        hidden = (torch.empty((self.num_tokens, self.hidden_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        weight = (torch.empty((self.hidden_size, self.vocab_size), dtype=self.dtype, device="cuda")
                .uniform_(-0.5, 0.5)
                .requires_grad_())
        labels = torch.randint(0, self.vocab_size * self.world_size, (self.num_tokens,), device="cuda")
        if self.ignore_index_opt is not None:
            pad_labels = torch.nn.functional.pad(labels, (0, 1), value=self.ignore_index)
            labels = pad_labels[..., 1:].contiguous()
        return hidden, weight, labels

    def generate_backward_input(self):
        g_logprobs = (torch.empty((), dtype=self.dtype, device="cuda")
                        .uniform_(-1, 1))
        return g_logprobs

    def verify_torch_correctness_with_single_gpu(self):
        cleanup()
        self.generate_hyper()

        for i in range(self.iterations):
            hidden, weight, labels = self.generate_forward_input()
            
            # synchronize hidden and labels among Process Group
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            logprobs = _TorchLinearCrossEntropy.apply(hidden, weight, labels, self.group)
            
            # single GPU verification
            whole_weight = torch.empty((weight.shape[0], weight.shape[1] * self.world_size),
                                        dtype=weight.dtype, device=weight.device)
            whole_weight_ref = [
                whole_weight[:, i * weight.shape[1]: (i + 1) * weight.shape[1]]
                for i in range(self.world_size)
            ]
            dist.all_gather(whole_weight_ref, weight, group=self.group)

            whole_weight_ = whole_weight.clone().requires_grad_()

            single_logprobs = run_torch_entropy(hidden, whole_weight_, labels)
            torch.testing.assert_close(logprobs, single_logprobs)
            
            # backward pass
            g_logprobs = self.generate_backward_input()
            dist.broadcast(g_logprobs, src=0, group=self.group)

            (d_hidden, d_weight) = torch.autograd.grad((logprobs,),
                                                        (hidden, weight),
                                                        (g_logprobs,),
                                                        retain_graph=False)
            dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=self.group)
            
            (d_whole_hidden, d_whole_weight) = torch.autograd.grad((single_logprobs,),
                                                                    (hidden, whole_weight_),
                                                                    (g_logprobs,),
                                                                    retain_graph=False)
            
            torch.testing.assert_close(d_hidden, d_whole_hidden, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(d_weight, 
                d_whole_weight[:, self.local_rank * d_weight.shape[1] : (self.local_rank + 1) * d_weight.shape[1]])

        if self.local_rank == 0:
            print(f"[INFO] rank {self.local_rank} torch correctness with single GPU verified")


    def verify_linear_combined_parallel_cross_entropy(self):
        cleanup()
        self.generate_hyper()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch_forward_latency = list()
        custom_forward_latency = list()
        torch_backward_latency = list()
        custom_backward_latency = list()

        for i in range(self.iterations):
            hidden, weight, labels = self.generate_forward_input()

            # synchronize hidden and labels among Process Group
            # this step is not included in the latency measurement
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            start_event.record()
            torch_logprobs = _TorchLinearCrossEntropy.apply(hidden, weight, labels, self.group)
            end_event.record()
            torch.cuda.synchronize()
            torch_forward_latency.append(start_event.elapsed_time(end_event))

            # combine linear with parallel cross entropy
            start_event.record()
            logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32))
            logits_view = logits.view(-1, logits.shape[0], logits.shape[1])
            parallel_logprobs = parallel_cross_entropy(logits_view, labels, 0.0, True, self.group)
            end_event.record()
            torch.cuda.synchronize()
            custom_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(torch_logprobs, parallel_logprobs)
            
            # torch backward pass
            g_logprobs = self.generate_backward_input()
            dist.broadcast(g_logprobs, src=0, group=self.group)

            start_event.record()
            (torch_d_hidden, torch_d_weight) = torch.autograd.grad((torch_logprobs,),
                                                                    (hidden, weight),
                                                                    (g_logprobs,),
                                                                    retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            torch_backward_latency.append(start_event.elapsed_time(end_event))
            # as forward didn't include broadcast, we need to exclude it in the latency measurement as well
            dist.all_reduce(torch_d_hidden, op=dist.ReduceOp.SUM, group=self.group)

            # linear combined parallel cross entropy
            start_event.record()
            d_logits = torch.autograd.grad((parallel_logprobs,),
                                            (logits_view,),
                                            (g_logprobs,),
                                            retain_graph=False)
            d_logits = d_logits[0]
            
            d_hidden = torch.matmul(d_logits, weight.T.to(torch.float32))
            d_weight = torch.matmul(hidden.T.to(torch.float32), d_logits)
            end_event.record()
            torch.cuda.synchronize()
            custom_backward_latency.append(start_event.elapsed_time(end_event))

            dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=self.group)
            d_hidden = d_hidden.view(hidden.shape)
            d_weight = d_weight.view(weight.shape)
            
            # parallel_cross_entropy always return FP32 for gradients
            # so we need to convert torch's gradient to FP32
            torch.testing.assert_close(torch_d_hidden.to(torch.float32), d_hidden, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_d_weight.to(torch.float32), d_weight, atol=1e-4, rtol=1e-4)

        if self.local_rank == 0:
            print(f"[INFO] rank {self.local_rank} linear combined parallel cross entropy correctness verified")

            # remove first latency
            torch_forward_latency = torch_forward_latency[1:]
            torch_backward_latency = torch_backward_latency[1:]
            custom_forward_latency = custom_forward_latency[1:]
            custom_backward_latency = custom_backward_latency[1:]
            
            print()
            print(f"[INFO] rank {self.local_rank}, torch forward latency "
                  f"{sum(torch_forward_latency) / len(torch_forward_latency):.2f} ms")
            print(f"[INFO] rank {self.local_rank}, linear combined parallel cross entropy forward latency "
                  f"{sum(custom_forward_latency) / len(custom_forward_latency):.2f} ms")
            print(f"[INFO] rank {self.local_rank}, torch backward latency "
                  f"{sum(torch_backward_latency) / len(torch_backward_latency):.2f} ms")
            print(f"[INFO] rank {self.local_rank}, linear combined parallel cross entropy backward latency "
                  f"{sum(custom_backward_latency) / len(custom_backward_latency):.2f} ms")

    def check_linear_combined_parallel_cross_entropy_storage(self):
        cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_input()
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        logits = torch.matmul(hidden.to(torch.float32), weight.to(torch.float32))
        logits_view = logits.view(-1, logits.shape[0], logits.shape[1])
        parallel_logprobs = parallel_cross_entropy(logits_view, labels, 0.0, True, self.group)
        torch.cuda.synchronize()
        forward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024

        g_logprobs = self.generate_backward_input()
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        d_logits = torch.autograd.grad((parallel_logprobs,),
                                        (logits_view,),
                                        (g_logprobs,),
                                        retain_graph=False)
        d_logits = d_logits[0]  
        d_hidden = torch.matmul(d_logits, weight.T.to(torch.float32))
        d_weight = torch.matmul(hidden.T.to(torch.float32), d_logits)
        torch.cuda.synchronize()
        backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print()
            print(f"[INFO] rank {self.local_rank}, linear combined parallel cross entropy "
                  f"forward max memory: {forward_max_memory:.2f} MB")
            print(f"[INFO] rank {self.local_rank}, linear combined parallel cross entropy "
                  f"backward max memory: {backward_max_memory:.2f} MB")
        
    def verify_linear_cross_entropy(self):
        cleanup()
        self.generate_hyper()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        custom_forward_latency = list()
        custom_backward_latency = list()

        for i in range(self.iterations):
            hidden, weight, labels = self.generate_forward_input()
            dist.broadcast(hidden, src=0, group=self.group)
            dist.broadcast(labels, src=0, group=self.group)

            torch_logprobs = _TorchLinearCrossEntropy.apply(hidden, weight, labels, self.group)

            start_event.record()
            custom_logprobs = linear_cross_entropy(hidden, weight, labels, "mean", self.group, self.ignore_index)
            end_event.record()
            torch.cuda.synchronize()
            custom_forward_latency.append(start_event.elapsed_time(end_event))

            torch.testing.assert_close(custom_logprobs, torch_logprobs, atol=1e-4, rtol=1e-4)

            # backward pass
            g_logprobs = self.generate_backward_input()
            dist.broadcast(g_logprobs, src=0, group=self.group)

            torch_d_hidden, torch_d_weight = torch.autograd.grad((torch_logprobs,),
                                                                  (hidden, weight),
                                                                  (g_logprobs,),
                                                                  retain_graph=False)
            dist.all_reduce(torch_d_hidden, op=dist.ReduceOp.SUM, group=self.group)
            torch.cuda.synchronize()

            start_event.record()
            custom_d_hidden, custom_d_weight = torch.autograd.grad((custom_logprobs,),
                                                                     (hidden, weight),
                                                                     (g_logprobs,),
                                                                     retain_graph=False)
            end_event.record()
            torch.cuda.synchronize()
            custom_backward_latency.append(start_event.elapsed_time(end_event))
            dist.all_reduce(custom_d_hidden, op=dist.ReduceOp.SUM, group=self.group)
            
            torch.testing.assert_close(torch_d_hidden, custom_d_hidden, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(torch_d_weight, custom_d_weight, atol=1e-4, rtol=1e-4)

        # remove first latency
        custom_forward_latency = custom_forward_latency[1:]
        custom_backward_latency = custom_backward_latency[1:]
        if self.local_rank == 0:
            print()
            print(f"[INFO] rank {self.local_rank}, linear cross entropy correctness verified")
            print()
            print(f"[INFO] rank {self.local_rank}, linear cross entropy forward latency "
                  f"{sum(custom_forward_latency) / len(custom_forward_latency):.2f} ms")
            print(f"[INFO] rank {self.local_rank}, linear cross entropy backward latency "
                  f"{sum(custom_backward_latency) / len(custom_backward_latency):.2f} ms")

    def check_linear_cross_entropy_storage(self):
        cleanup()
        self.generate_hyper()

        hidden, weight, labels = self.generate_forward_input()
        dist.broadcast(hidden, src=0, group=self.group)
        dist.broadcast(labels, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        custom_logprobs = linear_cross_entropy(hidden, weight, labels, "mean", self.group, self.ignore_index)
        torch.cuda.synchronize()
        forward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024

        g_logprobs = self.generate_backward_input()
        dist.broadcast(g_logprobs, src=0, group=self.group)

        torch.cuda.reset_peak_memory_stats()
        (d_hidden, d_weight) = torch.autograd.grad((custom_logprobs,),
                                                    (hidden, weight),
                                                    (g_logprobs,),
                                                    retain_graph=False)
        torch.cuda.synchronize()
        backward_max_memory = torch.cuda.max_memory_reserved() / 1024 / 1024
        dist.all_reduce(d_hidden, op=dist.ReduceOp.SUM, group=self.group)

        if self.local_rank == 0:
            print()
            print(f"[INFO] rank {self.local_rank}, linear cross entropy "
                  f"forward max memory: {forward_max_memory:.2f} MB")
            print(f"[INFO] rank {self.local_rank}, linear cross entropy "
                  f"backward max memory: {backward_max_memory:.2f} MB")
        


    def shutdown(self):
        dist.destroy_process_group()

if __name__ == "__main__":
    # torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/pytorch/test_linear_cross_entropy.py
    torch.manual_seed(233376)

    ignore_index = -100 # this is the default value in torch's cross entropy
    # ignore_index = None # comment this line if you want to test with ignore_index
    tp_test = _TestTensorParallel(ignore_index)

    tp_test.verify_torch_correctness_with_single_gpu()
    if ignore_index is None:
        tp_test.verify_linear_combined_parallel_cross_entropy()
        tp_test.check_linear_combined_parallel_cross_entropy_storage()
    tp_test.verify_linear_cross_entropy()
    tp_test.check_linear_cross_entropy_storage()

    tp_test.shutdown()