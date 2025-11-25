# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import torch
from transformer_engine.pytorch import LayerNormMLP
import pytest

torch.manual_seed(1234)
device = torch.device("cuda")


class _Sequential(torch.nn.Sequential):
    """Sequential model that forwards keyword arguments to modules"""

    def forward(self, input_: torch.Tensor, **kwargs) -> torch.Tensor:
        x = input_
        for module in self:
            x = module(x, **kwargs)
        return x


class ModelConfig:
    def __init__(
        self,
        hidden_size: int = 128,
        ffn_hidden_size: int = 512,
        layers: int = 1,
    ):
        self._hidden_size = hidden_size
        self._ffn_hidden_size = ffn_hidden_size
        self._layers = layers

    def build(self):

        ln_list, sln_list = [], []
        for _ in range(self._layers):
            ln = LayerNormMLP(self._hidden_size, self._ffn_hidden_size, checkpoint=False).to(device)
            sln = LayerNormMLP(self._hidden_size, self._ffn_hidden_size, checkpoint=True).to(device)
            with torch.no_grad():
                sln.layer_norm_weight = torch.nn.Parameter(ln.layer_norm_weight.clone())
                sln.layer_norm_bias = torch.nn.Parameter(ln.layer_norm_bias.clone())
                sln.fc1_weight = torch.nn.Parameter(ln.fc1_weight.clone())
                sln.fc2_weight = torch.nn.Parameter(ln.fc2_weight.clone())
                sln.fc1_bias = torch.nn.Parameter(ln.fc1_bias.clone())
                sln.fc2_bias = torch.nn.Parameter(ln.fc2_bias.clone())
            ln_list.append(ln)
            sln_list.append(sln)

        ln_model = _Sequential(*ln_list)
        sln_model = _Sequential(*sln_list)

        return ln_model, sln_model


config = {
    "small": ModelConfig(128, 512, 12),
    "medium": ModelConfig(512, 2048, 12),
    "large": ModelConfig(1024, 4096, 12),
    "huge": ModelConfig(2048, 8192, 12),
}

seq_sizes = [2**7, 2**10, 2**14, 2**16]


def _warmup(model, tensor):
    for _ in range(3):
        model(tensor).sum().backward()


def _run_fwd(model, tensor):

    torch.cuda.reset_peak_memory_stats(device)
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated(device)
    start_time.record()
    out = model(tensor)
    end_time.record()
    end_time.synchronize()
    elapsed = start_time.elapsed_time(end_time)
    peak_mem = torch.cuda.max_memory_allocated(device)
    mem = float(peak_mem - start_mem)

    return out, elapsed, mem


def _run_bwd(model, out):

    model.zero_grad(set_to_none=False)
    loss = out.sum()

    torch.cuda.reset_peak_memory_stats(device)
    start_time, end_time = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated(device)
    start_time.record()
    loss.backward()
    end_time.record()
    end_time.synchronize()
    elapsed = start_time.elapsed_time(end_time)
    peak_mem = torch.cuda.max_memory_allocated(device)
    mem = float(peak_mem - start_mem)

    param_grads = _collect_param_grads(model)
    return param_grads, elapsed, mem


def _max_diff(ref, other):
    """Return max absolute difference between two tensors or collections."""
    if ref is None or other is None:
        return 0.0
    if isinstance(ref, (list, tuple)):
        diffs = [_max_diff(r, o) for r, o in zip(ref, other)]
        return max(diffs) if diffs else 0.0
    return torch.max(torch.abs(ref.detach() - other.detach())).item()


def _collect_param_grads(model):
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        key = _param_key(name)
        if key is not None:
            grads[key] = param.grad.detach().clone()
    return grads


def _param_key(name):
    return name.split(".")[-1]


@pytest.mark.parametrize("size", config.keys())
@pytest.mark.parametrize("seq_size", seq_sizes)
def test_selective_activation_checkpoint(size, seq_size):

    ln_model, sln_model = config[size].build()
    data = torch.randn((seq_size, config[size]._hidden_size), device=device)

    _warmup(ln_model, data)
    ln_fwd_out, ln_fwd_time, ln_fwd_mem = _run_fwd(ln_model, data)
    ln_grads, ln_bwd_time, ln_bwd_mem = _run_bwd(ln_model, ln_fwd_out)

    _warmup(sln_model, data)
    sln_fwd_out, sln_fwd_time, sln_fwd_mem = _run_fwd(sln_model, data)
    sln_grads, sln_bwd_time, sln_bwd_mem = _run_bwd(sln_model, sln_fwd_out)

    assert ln_fwd_mem > 6 * sln_fwd_mem, (
        "selective activation checkpointing does not reduce forward memory by 6X, only by"
        f" {ln_fwd_mem/sln_fwd_mem}!"
    )
    assert ln_bwd_time < sln_bwd_time, (
        "selective activation activation checkpointing backward pass is NOT slower than native!"
        f" got Native LayerNormMLP Backward Time: {ln_bwd_time} ms and Selective Activation"
        f" Checkpointed LayerNormMLP Backward Time: {sln_bwd_time} ms"
    )
    diff = _max_diff(ln_fwd_out, sln_fwd_out)
    assert diff == 0.0, f"outputs are not equal! maximum difference {diff}"
    for key in [
        "layer_norm_weight",
        "layer_norm_bias",
        "fc1_weight",
        "fc1_bias",
        "fc2_weight",
        "fc2_bias",
    ]:
        diff = _max_diff(ln_grads[key], sln_grads[key])
        assert diff == 0.0, f"gradients for {key} are not equal! maximum difference: {diff}"
