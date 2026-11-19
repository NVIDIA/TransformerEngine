# Checking a recipe

`te` is `transformer_engine.pytorch` throughout this skill.

Three levels, cheapest first. Run every level the environment allows and report the level you
reached. A recipe that passed only the static check is "written, not executed".

## 1. Static check (always)

```bash
python scripts/check_recipe.py te_<family>.py [other_recipe_files.py ...]
```

Pass every file that makes up the recipe; the mapping may live in a second file. The script
parses the files without importing them, so it needs neither TE nor a GPU. It reports:

- **fail** when the files do not import TE, contain no fused TE block, have no
  weight-mapping function, or have a mapping that never writes a fused parameter
  (`layernorm_qkv`, `layernorm_mlp`, expert weights);
- **warn** for things that are often mistakes: no RoPE base passed, `strict=False` with no
  coverage check, an MoE recipe whose router is not excluded from quantization, deprecated
  `fp8_autocast` / `fp8_model_init`;
- **info** on whether a `te.autocast` precision hook is present, which a fine-tuning recipe
  needs and a plain conversion does not;
- the TE constructor keyword arguments it found, to compare with `inspect_hf_config.py`.

Exit code 0 means no failures. Warnings do not change the exit code; read them.

## 2. Structural check (needs TE and a GPU, no checkpoint)

Build both models from a small random config of the same family and compare structure:

```python
config = AutoConfig.from_pretrained(model_dir)        # or build the config by hand
config.num_hidden_layers = 2                          # keep it small
torch.set_default_dtype(torch.bfloat16)
hf_model = AutoModelForCausalLM.from_config(config).cuda()
te_model = TELlamaForCausalLM(config).cuda()          # the recipe's model class

replace_params(hf_model.state_dict(), te_model.state_dict(), config)
te_model.load_state_dict(hf_model.state_dict(), strict=False)
```

Then check that the TE model has the same number of parameter *elements* as the HF model, and
that no TE layer parameter was left unwritten (the NaN sentinel in
`references/dense-decoder.md`). A mismatch in element count means a bias, a norm or an
expert was created that the HF model does not have, or the other way round.

## 3. Numerical parity (needs TE and a GPU)

With the two models from level 2 holding the same weights:

```python
torch.manual_seed(0)
input_ids = torch.randint(0, config.vocab_size, (2, 64), device="cuda")
hf_model.eval(); te_model.eval()
with torch.no_grad(), te.autocast(enabled=False):
    ref = hf_model(input_ids).logits.float()
    out = te_model(input_ids).logits.float()

rel_err = (out - ref).norm() / ref.norm()
top1 = (out.argmax(-1) == ref.argmax(-1)).float().mean()
print(f"relative error {rel_err:.3e}, top-1 agreement {top1:.4f}")
```

How to read the result:

- BF16 arithmetic is not bit-reproducible across two implementations: fused modules round at
  different points. Expect a small relative error and near-total top-1 agreement, not zero
  error. Running both models in FP32 separates rounding from real mismatches; a real mismatch
  stays large in FP32.
- An error near 1.0 or agreement near chance means a wrong mapping or wrong constructor
  argument. The usual causes, in order: QKV layout, gate/up order in `fc1`, RoPE base, a missing
  `zero_centered_gamma`, `head_dim` not passed as `kv_channels`, mask type.
- Compare one layer at a time (feed the same hidden states to layer 0 of each model) to find
  which block differs.

Use unpadded, equal-length inputs first. Then repeat with a right-padded batch and compare
only the real token positions, because padding is where HF and TE mask conventions differ.

With the real checkpoint, also compare the loss on a few real batches and generate a short
continuation from each model.

### Extra checks for MoE recipes

- **Routing parity.** The selected expert indices must match the HF model exactly for the same
  input; routing weights must match to rounding error.
- **Permutation round trip.** `moe_unpermute(moe_permute(x))` with unit routing weights and one
  expert per token returns `x`.
- **Per-expert outputs.** Run one expert's tokens through the HF expert and through the grouped
  path and compare.
- **Fused path fired.** For the MXFP8 fused grouped MLP, confirm that the fused op is what
  runs, not the three unfused ops. The operation fuser falls back silently.
- **Expert parallelism.** Compare `ep_size=1` and `ep_size>1` outputs on the same input.

### After turning a recipe on

Parity against BF16 is no longer the test under FP8 or MXFP8; quantization changes values by
design. Compare training loss curves over a few hundred steps against the BF16 run, and measure
step time so the speedup is a number instead of an expectation.

## What to report

State, without softening:

- which levels ran and their results (the relative error and agreement numbers, not "passed");
- which levels could not run and why ("no GPU in this environment", "TE not installed",
  "checkpoint is gated");
- which facts were assumed from the family notes instead of read from the installed source;
- the TE and `transformers` versions, when they could be read;
- any block left native, with the reason.
