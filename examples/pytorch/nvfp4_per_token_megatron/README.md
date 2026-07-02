# NVFP4 per-token training with Megatron-Core

This example shows how to train a small Mixture-of-Experts (MoE) model with
the **NVFP4 per-token** quantization recipe on a single GPU using
[Megatron-Core](https://github.com/NVIDIA/Megatron-LM), and how to compare it
against the per-tensor NVFP4 recipe and an unquantized BF16 baseline.

The same model / data / seed are used across all modes; only the GEMM precision
changes, so the runs are directly comparable.

## How per-token interacts with Megatron-Core

Megatron-Core builds a plain `transformer_engine.common.recipe.NVFP4BlockScaling`
for `--fp4-format e2m1` and has **no CLI flag for per-token**. Per-token is
selected entirely through Transformer Engine environment variables, read when the
recipe is constructed:

| Variable | Effect |
| --- | --- |
| `NVTE_NVFP4_PER_TOKEN=1` | **Required**: Flip the recipe into per-token mode (per-row/per-col outer amax + fused CUTLASS GEMM) |
| `NVTE_NVFP4_PER_TOKEN_RHT=1` | Opt into the random Hadamard transform (off by default) |
| `NVTE_NVFP4_PER_TOKEN_SR=1` | Opt into stochastic rounding (off by default) |
| `NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1` | Use the transposition-invariant 2D weight cast in per-token layout |

For the per-tensor recipe, the analogous knobs are
`NVTE_NVFP4_DISABLE_RHT`, `NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING`, and
`NVTE_NVFP4_DISABLE_2D_QUANTIZATION`.

See the
[NVFP4 documentation](../../../docs/features/low_precision_training/nvfp4/nvfp4.rst)
("Per-token NVFP4") and `docs/envvars.rst` for full details. Equivalently, code
that constructs its own recipe can use the public
`transformer_engine.common.recipe.NVFP4PerTokenBlockScaling` class instead of the
env var.

Keeping the first/last transformer layers in BF16 is a Megatron-Core CLI feature
(`--first-last-layers-bf16 --num-layers-at-start-in-bf16 N
--num-layers-at-end-in-bf16 M`); those layers simply skip the FP4 autocast. This
is also supported with the per-token recipe.

## Prerequisites

- A Blackwell GPU (SM100+) — NVFP4 training requires it.
- Transformer Engine built from this repository **with per-token support**
  (`NVTE_CUDA_ARCHS=100a NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch pip install -e . --no-build-isolation`).
- A [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) checkout (provides
  `pretrain_gpt.py` and the `megatron` package).
- A tokenized dataset and tokenizer (the scripts default to an OLMo-1124 corpus
  with the Moonlight-16B-A3B tokenizer).
- For Weights & Biases logging: authenticate with `wandb login` or export
  `WANDB_API_KEY` in your environment.

## Files

| File | Purpose |
| --- | --- |
| `run_moe_nvfp4_singlegpu.sh` | Core launcher. Run **inside** the container from a shell that can see `pretrain_gpt.py`. Takes one mode: `bf16`, `prod` (== `pertensor`), or `pertoken`. |
| `sbatch_moe_nvfp4_singlegpu.sh` | Slurm wrapper: starts the container, (re)installs the editable TE build, then runs one or more variants (so far one GPU each). |
| `submit_chain.sh` | Submit a chain of dependent Slurm jobs that auto-resume from the stable checkpoint dir. |

## Quick start (standalone, inside a container)

```bash
# Point at your Megatron-LM checkout, data, and tokenizer.
export MLM_DIR=/path/to/Megatron-LM
export DATA_PATH=/path/to/datasets/your_data
export TOKENIZER_MODEL=/path/to/tokenizers/Moonlight-16B-A3B
export TRAIN_ITERS=2000

bash run_moe_nvfp4_singlegpu.sh pertoken

# Compare against per-tensor NVFP4 and BF16:
bash run_moe_nvfp4_singlegpu.sh pertensor
bash run_moe_nvfp4_singlegpu.sh bf16
```

To enable per-token RHT / SR / 2D-weight, export the knobs before launching:

```bash
export NVTE_NVFP4_PER_TOKEN_RHT=1
export NVTE_NVFP4_PER_TOKEN_SR=1
export NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=1
bash run_moe_nvfp4_singlegpu.sh pertoken
```

## Slurm

Edit the **host-side config block** at the top of
`sbatch_moe_nvfp4_singlegpu.sh` (Slurm account, container `IMAGE`, `HOST_MOUNT`,
`TE_DIR`, and `HOST_LOG_DIR` / the `#SBATCH --output/--error` paths) for your
cluster, then:

```bash
# One mode:
sbatch sbatch_moe_nvfp4_singlegpu.sh pertoken

# Up to 4 variants concurrently (one GPU each):
sbatch sbatch_moe_nvfp4_singlegpu.sh "bf16,pertensor+rht+sr,pertoken"

# Override knobs via --export:
sbatch --export=ALL,TRAIN_ITERS=2000,SEED=42 sbatch_moe_nvfp4_singlegpu.sh pertoken
```

Spec syntax: `<mode>[+rht][+sr][+1d][+2d][+fb]` where `mode` is
`bf16 | prod (== pertensor) | pertoken`. `+rht`/`+sr` turn those features on,
`+1d` forces 1D weights (per-tensor only), `+2d` enables the per-token 2D-weight
route, and `+fb` keeps the first/last layers in BF16.

For runs that exceed one Slurm wall-clock window, chain dependent jobs that
resume from the stable per-variant checkpoint dir:

```bash
CHAIN=3 bash submit_chain.sh \
    --export=ALL,IMAGE=/path/to/te_pertoken.sqsh,SKIP_BUILD=1,TRAIN_ITERS=60000 \
    sbatch_moe_nvfp4_singlegpu.sh pertoken
```

## Notes and current limitations

The per-token recipe is currently intended for **accuracy evaluation and
comparison** (per-token vs per-tensor vs BF16), **not** for optimized production
deployment. Concretely:

- **Not tested with CUDA graphs.** The per-token path has not been validated under
  Megatron's CUDA graph capture; leave CUDA graphs disabled for now.
- **Kernels are not yet performance-optimal.** Several per-token cast / GEMM
  kernels are functional but not tuned, so wall-clock throughput is not
  representative of the recipe's eventual performance. Use this example for
  numerical/accuracy comparison, not perf benchmarking.
- `--no-gradient-accumulation-fusion` is required: the per-token kernel does not
  yet support fused wgrad accumulation. The scripts set it for every mode so
  only the GEMM precision differs.
- The example reduces the MoE expert count to 64 so all experts stay local at
  EP=1 on a single GPU (TE's grouped-NVFP4 kernels cap at 64 tensors per launch).
  Real training shards experts via EP>1.
