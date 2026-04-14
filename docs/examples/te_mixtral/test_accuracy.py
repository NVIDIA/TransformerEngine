import torch
from transformers import MixtralConfig, MixtralForCausalLM

from te_mixtral import NVMixtralForCausalLM, replace_params


def main():
    assert torch.cuda.is_available(), "A CUDA-capable GPU is required for this example."

    # Small config so the parity check runs quickly.
    cfg = MixtralConfig(
        hidden_size=256,
        intermediate_size=512,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        vocab_size=1024,
    )

    device = "cuda"
    dtype = torch.bfloat16

    # Reference Hugging Face model with random weights.
    model_hf = MixtralForCausalLM(cfg).to(device=device, dtype=dtype)
    model_hf.eval()

    # TE model with the same architecture, populated via replace_params().
    te_config = NVMixtralForCausalLM.config_class(**cfg.to_dict())
    model_te = NVMixtralForCausalLM(te_config).to(device=device, dtype=dtype)
    te_state_dict = model_te.state_dict()
    replace_params(model_hf.state_dict(), te_state_dict, model_te.config)
    missing, unexpected = model_te.load_state_dict(te_state_dict, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected TE keys during load: {unexpected}")
    allowed_missing = [key for key in missing if key.endswith("_extra_state")]
    if len(allowed_missing) != len(missing):
        raise RuntimeError(f"Unexpected missing TE keys during load: {missing}")
    model_te.eval()

    # Compare outputs on the same random input.
    input_ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        hf_logits = model_hf(input_ids=input_ids, attention_mask=attention_mask).logits
        te_logits = model_te(input_ids=input_ids, attention_mask=attention_mask).logits

    max_diff = (hf_logits - te_logits).abs().max().item()
    mean_diff = (hf_logits - te_logits).abs().mean().item()

    print(f"HF logits shape : {tuple(hf_logits.shape)}")
    print(f"TE logits shape : {tuple(te_logits.shape)}")
    print(f"Max abs diff    : {max_diff:.6f}")
    print(f"Mean abs diff   : {mean_diff:.6f}")

    assert max_diff < 0.05, f"Outputs diverged: {max_diff}"
    print("Weight mapping verified -- HF and TE models produce equivalent outputs.")


if __name__ == "__main__":
    main()
