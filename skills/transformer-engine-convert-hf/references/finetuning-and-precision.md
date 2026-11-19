# Fine-tuning and precision

Read this when the user asks for a fine-tuning recipe or for FP8, MXFP8 or NVFP4. A plain
conversion request stops at a BF16 model that matches the HF model.

`te` is `transformer_engine.pytorch` throughout this skill.

## Order of operations

1. Build the TE model and load the HF checkpoint (the recipe's loader), then move it to the GPU.
2. Verify it against the HF model in BF16 with quantization off (`references/validation.md`).
3. Create the optimizer and any training wrappers (Accelerate, DDP, PEFT, `torch.compile`).
4. Turn on a quantization recipe and train.

Step 2 before step 4 matters: a low-precision run that diverges tells you nothing about whether
the weight mapping or the quantization is at fault.

## What to put in the recipe file

For a fine-tuning request, add two things to the recipe module:

1. The `select_recipe` helper below, as executable code.
2. A usage example in the module docstring that loads the model and runs one training step
   under `te.autocast`.

A complete training loop (data loading, scheduler, logging) belongs in the user's own script.
Write one only when they ask, and keep it in a separate file from the model recipe.

```python
import transformer_engine.pytorch as te
from transformer_engine.common import recipe as te_recipe


def select_recipe(precision="auto"):
    """Return the TE quantization recipe for `precision`, or None for BF16.

    "auto" picks the lowest precision this GPU supports. Asking for a precision the GPU
    does not support raises with TE's own explanation instead of silently falling back.
    """
    candidates = {
        "mxfp8": (te.is_mxfp8_available, te_recipe.MXFP8BlockScaling),  # Blackwell
        "fp8": (te.is_fp8_available, te_recipe.DelayedScaling),  # Hopper and newer
    }
    if precision == "bf16":
        print("TE precision: BF16 (no quantization)")
        return None
    if precision == "auto":
        for name, (is_available, recipe_cls) in candidates.items():
            if is_available():
                print(f"TE precision: {name}")
                return recipe_cls()
        print("TE precision: BF16 (no quantized recipe is available on this GPU)")
        return None
    if precision not in candidates:
        raise ValueError(f"Unknown precision {precision!r}; use auto, bf16, fp8 or mxfp8")
    is_available, recipe_cls = candidates[precision]
    available, reason = is_available(return_reason=True)
    if not available:
        raise RuntimeError(f"{precision} is not available: {reason}")
    print(f"TE precision: {precision}")
    return recipe_cls()
```

The docstring example, with the names of the Llama recipe:

```python
config = AutoConfig.from_pretrained(model_dir)
config.use_cache = False
model = TELlamaForCausalLM.from_pretrained_local(model_dir, config=config).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)  # after the conversion
fp8_recipe = select_recipe("auto")

with te.autocast(enabled=fp8_recipe is not None, recipe=fp8_recipe):
    loss = model(input_ids=input_ids, labels=labels).loss
loss.backward()  # outside the autocast block
optimizer.step()
optimizer.zero_grad()
```

## Recipes

Recipes live in `transformer_engine.common.recipe`. Ask TE what the GPU supports, as
`select_recipe` does, instead of inferring it from the GPU name.

| Recipe | Scaling | Typical hardware |
|---|---|---|
| `DelayedScaling` | Per-tensor scale from a history of maxima | Hopper and newer |
| `Float8CurrentScaling` | Per-tensor scale from the current tensor | Hopper and newer |
| `MXFP8BlockScaling` | One power-of-two scale per 32 values | Blackwell |
| `NVFP4BlockScaling` | 4-bit values with block scales | Blackwell |

Master weights stay in BF16; TE quantizes weights and activations around each GEMM.

`fp8_autocast` and `fp8_model_init` are deprecated names for `autocast` and
`quantized_model_init`; use the new names unless the installed TE predates them.

With Hugging Face Accelerate, `Accelerator(mixed_precision="fp8",
kwargs_handlers=[FP8RecipeKwargs(backend="te")])` applies TE's autocast for you, which is what
the published Llama tutorial does. Use one mechanism or the other, not both: wrapping a model
that already manages its own `autocast` in Accelerate's FP8 mode applies the recipe twice.

## What stays in high precision

| Component | Why | How |
|---|---|---|
| MoE router gate | Its output selects experts; noise changes the routing, not just the values | Build under `te.quantized_model_init(enabled=False)`; call under `te.autocast(enabled=False)` |
| LM head | Feeds the loss directly | Keep it an HF `nn.Linear`, or a `te.Linear` called under `te.autocast(enabled=False)` |
| Router softmax and routing weights | Small tensors where rounding is visible | Compute in `torch.float32` |
| Embeddings, RoPE frequencies | Not GEMMs | Leave as they are |

Individual layers can also be kept in BF16 (for example the first and last) by entering
`te.autocast(enabled=False)` around them; the published Mixtral tutorial does this with a
per-layer precision list.

## Memory: quantized parameters

`te.quantized_model_init(enabled=True)` around model construction stores weights in the
quantized format only, which saves memory for inference and for frozen layers. Do not use it for
ordinary fine-tuning, where the optimizer needs the high-precision master weights.

## Packed batches

Padding tokens cost attention and MLP time. With `attn_input_format="thd"` (or
`qkv_format="thd"` on `MultiheadAttention`) the batch is flattened to the real tokens only, and
the per-sequence boundaries are passed as `cu_seqlens_q` / `cu_seqlens_kv` with
`max_seqlen_q` / `max_seqlen_kv`. For MoE models this also removes padding tokens from the
expert GEMMs. It requires owning the model-level forward (see "Hosting the layer" in
`references/moe-decoder.md`), so add it on request.
