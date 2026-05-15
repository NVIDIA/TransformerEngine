# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Forward + backward parity check for te_mixtral (BF16 and MXFP8) vs HF.

Compares logits, loss, and weight gradients (``model.embed_tokens.weight``
and ``lm_head.weight``) between the HuggingFace reference and the
TransformerEngine port in both BF16 and MXFP8 modes.
"""

import torch
from transformers import MixtralConfig, MixtralForCausalLM

import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe

from te_mixtral import NVMixtralForCausalLM, replace_params as replace_params_bf16
from te_mixtral_mxfp8 import (
    NVMixtralMXFP8ForCausalLM,
    replace_params as replace_params_mxfp8,
)


# BF16 should match HF very closely.
BF16_TOL = 0.01

# MXFP8 quantizes activations to FP8 with per-tile bf16 scales, so we expect
# larger logit / gradient drift than the BF16 case.
MXFP8_LOGITS_ATOL = 1.5
MXFP8_LOGITS_RTOL = 0.05
MXFP8_LOSS_ATOL = 0.05
MXFP8_LOSS_RTOL = 0.05
MXFP8_GRAD_ATOL = 1.0
MXFP8_GRAD_RTOL = 0.1


def _build_config():
    return MixtralConfig(
        hidden_size=256,
        intermediate_size=512,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=1024,
        max_position_embeddings=128,
        router_jitter_noise=0.0,
        rms_norm_eps=1e-5,
    )


def _load_te_weights(model_te, model_hf, replace_params_fn):
    te_state_dict = model_te.state_dict()
    replace_params_fn(model_hf.state_dict(), te_state_dict, model_te.config)
    missing, unexpected = model_te.load_state_dict(te_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected TE keys during load: {unexpected}")
    allowed_missing = [k for k in missing if k.endswith("_extra_state")]
    if len(allowed_missing) != len(missing):
        raise RuntimeError(f"Unexpected missing TE keys during load: {missing}")


def _zero_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


def _forward_backward(model, input_ids, attention_mask, labels, *, fp8_recipe=None):
    """Return (logits, loss, embed_grad, lm_head_grad), all detached as float32."""
    _zero_grads(model)
    if fp8_recipe is not None:
        with te.autocast(enabled=True, recipe=fp8_recipe):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
    else:
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
    return (
        out.logits.detach().float(),
        out.loss.detach().float(),
        model.model.embed_tokens.weight.grad.detach().float().clone(),
        model.lm_head.weight.grad.detach().float().clone(),
    )


def _compare(label, hf, te_, *, atol, rtol):
    diff = (hf - te_).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"  {label:<14s} max={max_diff:.6f}  mean={mean_diff:.6f}")
    torch.testing.assert_close(te_, hf, atol=atol, rtol=rtol)


def _make_inputs(cfg, device):
    torch.manual_seed(1)
    # seq divisible by 32 so the MXFP8 path is happy.
    input_ids = torch.randint(0, cfg.vocab_size, (2, 64), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    return input_ids, attention_mask, labels


def _build_hf(cfg, device, dtype):
    torch.manual_seed(0)
    model = MixtralForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model


def _run_bf16(cfg, model_hf, inputs, device, dtype):
    print("=" * 64)
    print("BF16 parity check (forward + backward)")
    print("=" * 64)

    te_cfg = NVMixtralForCausalLM.config_class(**cfg.to_dict())
    model_te = NVMixtralForCausalLM(te_cfg).to(device=device, dtype=dtype)
    _load_te_weights(model_te, model_hf, replace_params_bf16)
    model_te.eval()

    input_ids, attention_mask, labels = inputs
    hf_logits, hf_loss, hf_embed_g, hf_lm_g = _forward_backward(
        model_hf, input_ids, attention_mask, labels
    )
    te_logits, te_loss, te_embed_g, te_lm_g = _forward_backward(
        model_te, input_ids, attention_mask, labels
    )

    print(f"  logits shape   {tuple(hf_logits.shape)}")
    print(f"  HF loss        {hf_loss.item():.6f}")
    print(f"  TE loss        {te_loss.item():.6f}")
    _compare("logits", hf_logits, te_logits, atol=BF16_TOL, rtol=0.0)
    _compare("loss", hf_loss, te_loss, atol=BF16_TOL, rtol=0.0)
    _compare("embed.grad", hf_embed_g, te_embed_g, atol=BF16_TOL, rtol=0.0)
    _compare("lm_head.grad", hf_lm_g, te_lm_g, atol=BF16_TOL, rtol=0.0)
    print("BF16 parity OK.\n")


def _run_mxfp8(cfg, model_hf, inputs, device, dtype):
    print("=" * 64)
    print("MXFP8 parity check (forward + backward)")
    print("=" * 64)

    te_cfg = NVMixtralMXFP8ForCausalLM.config_class(**cfg.to_dict())
    te_cfg.attn_input_format = "bshd"
    te_cfg.self_attn_mask_type = "causal"
    te_cfg.expert_parallel_size = 1
    te_cfg.dtype = dtype
    recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
    model_te = NVMixtralMXFP8ForCausalLM(te_cfg, fp8_recipe=recipe).to(device=device, dtype=dtype)
    _load_te_weights(model_te, model_hf, replace_params_mxfp8)
    model_te.eval()

    input_ids, attention_mask, labels = inputs
    hf_logits, hf_loss, hf_embed_g, hf_lm_g = _forward_backward(
        model_hf, input_ids, attention_mask, labels
    )
    te_logits, te_loss, te_embed_g, te_lm_g = _forward_backward(
        model_te, input_ids, attention_mask, labels, fp8_recipe=recipe
    )

    print(f"  logits shape   {tuple(hf_logits.shape)}")
    print(f"  HF loss        {hf_loss.item():.6f}")
    print(f"  TE loss        {te_loss.item():.6f}")
    _compare("logits", hf_logits, te_logits, atol=MXFP8_LOGITS_ATOL, rtol=MXFP8_LOGITS_RTOL)
    _compare("loss", hf_loss, te_loss, atol=MXFP8_LOSS_ATOL, rtol=MXFP8_LOSS_RTOL)
    _compare("embed.grad", hf_embed_g, te_embed_g, atol=MXFP8_GRAD_ATOL, rtol=MXFP8_GRAD_RTOL)
    _compare("lm_head.grad", hf_lm_g, te_lm_g, atol=MXFP8_GRAD_ATOL, rtol=MXFP8_GRAD_RTOL)
    print("MXFP8 parity OK.\n")


def main() -> None:
    assert torch.cuda.is_available(), "CUDA required."

    cfg = _build_config()
    device = "cuda"
    dtype = torch.bfloat16

    model_hf = _build_hf(cfg, device, dtype)
    inputs = _make_inputs(cfg, device)

    _run_bf16(cfg, model_hf, inputs, device, dtype)
    _run_mxfp8(cfg, model_hf, inputs, device, dtype)


if __name__ == "__main__":
    main()
