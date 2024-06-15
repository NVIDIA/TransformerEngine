# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""NVFuser functions and JIT utilities"""
import os
from typing import Callable, Optional, Tuple

import torch

jit_fuser = torch.jit.script
if torch.__version__ >= "2" and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
    jit_fuser = torch.compile

# See: https://github.com/NVIDIA/TransformerEngine/issues/597
dropout_fuser = torch.jit.script
if torch.__version__ >= "2.2" and bool(int(os.getenv("NVTE_TORCH_COMPILE", "1"))):
    dropout_fuser = torch.compile

# Decorator to disable Torch Dynamo
# See: https://github.com/NVIDIA/TransformerEngine/issues/308
no_torch_dynamo = lambda recursive=True: lambda func: func
if torch.__version__ >= "2":
    import torch._dynamo

    if torch.__version__ >= "2.1":
        no_torch_dynamo = lambda recursive=True: lambda f: torch._dynamo.disable(
            f, recursive=recursive
        )
    else:
        # no "recursive" option in pyTorch 2.0 - it acts as if recursive was True
        no_torch_dynamo = lambda recursive=True: torch._dynamo.disable


def set_jit_fusion_options() -> None:
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    TORCH_MAJOR = int(torch.__version__.split(".")[0])
    TORCH_MINOR = int(torch.__version__.split(".")[1])
    if TORCH_MAJOR == 2 and TORCH_MINOR >= 2:
        pass
    elif (TORCH_MAJOR == 2) or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10):
        # nvfuser
        torch._C._jit_set_profiling_executor(True)
        torch._C._jit_set_profiling_mode(True)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._debug_set_autodiff_subgraph_inlining(False)
    else:
        # legacy pytorch fuser
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)


@jit_fuser
def bias_gelu_fused_(inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Bias-GeLU fused"""
    x = inp + bias
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@jit_fuser
def gelu_fused_(inp: torch.Tensor) -> torch.Tensor:
    """
    GeLU fused, this is copy of bias_gelu_fused cause jit fusion doesn't allow conditioning.
    """
    x = inp
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@jit_fuser
def bgrad_dgelu_fused_(
    grad_output: torch.Tensor, inp: torch.Tensor, bias: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bgrad-Dgelu fused"""
    x = inp + bias
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    dgelu = ff * grad_output
    bgrad = dgelu.sum(dim=0)
    return bgrad, dgelu


@jit_fuser
def dgelu_fused_(grad_output: torch.Tensor, inp: torch.Tensor) -> torch.Tensor:
    """
    Dgelu fused, this is copy of bgrad_dgelu_fused_ cause jit fusion doesn't allow conditioning.
    """
    x = inp
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    dgelu = ff * grad_output
    return dgelu


def bias_gelu_fused(inp: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """Disable native AMP for bias_gelu_fused_"""
    with torch.cuda.amp.autocast(enabled=False):
        if bias.numel() != 0:
            return bias_gelu_fused_(inp, bias)
        return gelu_fused_(inp)


def bgrad_dgelu_fused(
    grad_output: torch.Tensor, inp: torch.Tensor, bias: torch.Tensor
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Disable native AMP for `bgrad_dgelu_fused_`"""
    with torch.cuda.amp.autocast(enabled=False):
        if bias.numel() != 0:
            return bgrad_dgelu_fused_(grad_output, inp, bias)
        return None, dgelu_fused_(grad_output, inp)


def bias_dropout_add(
    x: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    prob: float,
    training: bool,
) -> torch.Tensor:
    """dropout(inp + bias) + residual"""
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training: bool) -> Callable:
    """bias_dropout_add based on training or not"""

    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)

    return _bias_dropout_add


@dropout_fuser
def bias_dropout_add_fused_train_(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    """Jit fused bias_dropout_add for training"""
    return bias_dropout_add(x, bias, residual, prob, True)


def bias_dropout_add_fused_train(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    """Disable native AMP and enable grad for BDA"""
    with torch.enable_grad():
        with torch.cuda.amp.autocast(enabled=False):
            return bias_dropout_add_fused_train_(x, bias, residual, prob)


@dropout_fuser
def bias_dropout_add_fused_inference_(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    """Jit fused bias_dropout_add for inference"""
    return bias_dropout_add(x, bias, residual, prob, False)


def bias_dropout_add_fused_inference(
    x: torch.Tensor, bias: torch.Tensor, residual: torch.Tensor, prob: float
) -> torch.Tensor:
    """Disable native AMP for BDA"""
    with torch.cuda.amp.autocast(enabled=False):
        return bias_dropout_add_fused_inference_(x, bias, residual, prob)


def warmup_jit_bias_dropout_add(
    hidden_size: int, dtype: torch.dtype, seq_length: int, micro_batch_size: int
) -> None:
    """Compile BDA JIT function before the main training steps"""

    # Save cuda RNG state to ensure warmup does not affect reproducibility.
    rng_state = torch.cuda.get_rng_state()

    inp = torch.rand((seq_length, micro_batch_size, hidden_size), dtype=dtype, device="cuda")
    residual = torch.rand((seq_length, micro_batch_size, hidden_size), dtype=dtype, device="cuda")
    bias = torch.rand((hidden_size), dtype=dtype, device="cuda")
    dropout_rate = 0.1
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for input_grad, bias_grad, residual_grad in zip([False, True], [True, True], [True, True]):
        inp.requires_grad = input_grad
        bias.requires_grad = bias_grad
        residual.requires_grad = residual_grad
        for _ in range(5):
            output = bias_dropout_add_fused_train(inp, bias, residual, dropout_rate)
    del bias, inp, residual, output

    torch.cuda.empty_cache()
    torch.cuda.set_rng_state(rng_state)


def warmup_jit_bias_dropout_add_all_dtypes(
    hidden_size: int, seq_length: int, micro_batch_size: int
) -> None:
    """Call `warmup_jit_bias_dropout_add` for all training dtypes"""
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        warmup_jit_bias_dropout_add(hidden_size, dtype, seq_length, micro_batch_size)


def warmup_jit_bias_gelu(
    ffn_hidden_size_per_partition: int,
    dtype: torch.dtype,
    seq_length: int,
    micro_batch_size: int,
) -> None:
    """Compile bias-gelu JIT function before the main training steps"""

    # Save cuda RNG state to ensure warmup does not affect reproducibility.
    rng_state = torch.cuda.get_rng_state()

    bias = torch.rand(ffn_hidden_size_per_partition, dtype=dtype, device="cuda")
    inp = torch.rand(
        (seq_length * micro_batch_size, ffn_hidden_size_per_partition),
        dtype=dtype,
        device="cuda",
    )
    # Warmup JIT fusions with the input grad_enable state of both forward
    # prop and recomputation
    for bias_grad, input_grad in zip([True, True], [False, True]):
        bias.requires_grad, inp.requires_grad = bias_grad, input_grad
        for _ in range(5):
            _ = bias_gelu_fused_(inp, bias)
            _ = gelu_fused_(inp)
    del bias, inp

    torch.cuda.empty_cache()
    torch.cuda.set_rng_state(rng_state)


def warmup_jit_bias_gelu_all_dtypes(
    ffn_hidden_size: int, seq_length: int, micro_batch_size: int
) -> None:
    """Call `warmup_jit_bias_gelu` for all training dtypes"""
    for dtype in [torch.float32, torch.bfloat16, torch.float16]:
        warmup_jit_bias_gelu(ffn_hidden_size, dtype, seq_length, micro_batch_size)
