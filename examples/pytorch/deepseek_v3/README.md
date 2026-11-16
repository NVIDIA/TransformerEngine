# DeepSeekV3Layer with expert parallelism

A forward + backward benchmark of one `DeepSeekV3Layer`: RMSNorm, multi-latent attention,
and MoE with a shared expert. Routed experts are sharded across GPUs using NCCL EP.

## Quick start

Requires SM90+ GPUs with NVLink and an NCCL EP-enabled TE build;
see [requirements](#requirements). From this directory:

```bash
bash run_deepseek_v3_layer_ep.sh --dsv3
```

For MXFP8 with the fused grouped MLP (SM100-class GPUs):

```bash
NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 bash run_deepseek_v3_layer_ep.sh --dsv3 --recipe mxfp8
```

## Requirements

The following NCCL EP requirements apply to `--impl te`. The naive variants use
ordinary NCCL collectives and can also run on older GPUs without NVLink.

- SM90 or newer GPUs connected with NVLink.
- NCCL >= 2.30.4, PyTorch >= 2.11 (symmetric memory), Transformer Engine built with NCCL EP
  enabled (`NVTE_WITH_NCCL_EP`, on by default when an SM90+ architecture is targeted;
  `NVTE_WITH_NCCL_EP=0` disables it).
- `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` additionally needs SM100-class GPUs and the CuTe DSL
  (`nvidia-cutlass-dsl`) for the fused MXFP8 grouped MLP.

## Running

Run these commands from this directory. Single node, all local GPUs:

```bash
bash run_deepseek_v3_layer_ep.sh                         # small dims, bf16
bash run_deepseek_v3_layer_ep.sh --dsv3                  # DeepSeek-V3 layer dims, bf16
bash run_deepseek_v3_layer_ep.sh --dsv3 --recipe mxfp8   # MXFP8 experts (unfused grouped GEMM)
NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 bash run_deepseek_v3_layer_ep.sh --dsv3 --recipe mxfp8
bash run_deepseek_v3_layer_ep.sh --dsv3 --impl naive          # all_to_all + loop over experts
bash run_deepseek_v3_layer_ep.sh --dsv3 --impl naive_grouped  # all_to_all + TE grouped GEMM
```

`--impl` selects the MoE block inside the same layer (attention and norms are identical):

- `te` (default): `DeepSeekV3MoE`, NCCL EP dispatch/combine, experts as one grouped GEMM.
- `naive`: MoE written with plain PyTorch, no TE MoE code: sigmoid top-k router with expert
  bias, `all_to_all_single` dispatch and combine (two host syncs per layer for the split sizes),
  a Python loop over the local experts with dense `F.linear` SwiGLU MLPs, `index_copy` /
  `index_add` to place results, and a shared expert. This is what an EP MoE looks like before
  any fused kernels.
- `naive_grouped`: the same all_to_all dispatch and combine, but the received rows are sorted by
  local expert and run through one `te.ops.GroupedLinear` / `ScaledSwiGLU` / `GroupedLinear`
  stack. Isolates the cost of the Python loop from the cost of the communication path.

Multi-node: launch `torchrun` yourself, EP spans every rank:

```bash
torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=<head>:29500 \
    deepseek_v3_layer_ep.py --dsv3 --recipe mxfp8
```

Every rank owns `--num-local-experts` experts (default 8), so the expert count is
`8 * world_size`. Other knobs: `--tokens-per-rank`, `--topk`, `--hidden`, `--num-heads`,
`--moe-ffn`, the MLA dims (`--q-lora-rank`, `--kv-lora-rank`, `--qk-nope-head-dim`,
`--qk-rope-head-dim`, `--v-head-dim`), `--warmup`, `--iters`.
`--warmup 0` is supported; `--iters` must be positive and `--tokens-per-rank` must be a
positive multiple of four. MXFP8 requires `--impl te`.

## Results

GB200 and GB300 GPUs, 4096 tokens per rank, top-k 8, 8 local experts per GPU.
Times cover one layer's forward + backward; throughput is global, in millions of tokens/s.
Every number is the median of three independent runs (spread within 0.3 ms).
See [benchmark setup](#benchmark-setup) below for the full configuration.

### TE vs. plain PyTorch (`--dsv3`)

Same layer and dimensions, BF16 unless noted.

All variants run the router backward and clear parameter gradients before each step.

| MoE implementation | 4 GB200 (32 experts) | 4 GB300 (32 experts) | 8 GB300 (64 experts) |
|---|---:|---:|---:|
| `naive`: all_to_all + loop over experts | 28.61 ms | 27.21 ms | 27.70 ms |
| `naive_grouped`: all_to_all + TE grouped GEMM | 17.79 ms | 16.72 ms | 17.13 ms |
| `te`: NCCL EP + grouped GEMM | 13.68 ms | 12.37 ms | 14.09 ms |
| `te`, mxfp8 (unfused grouped GEMM) | 10.99 ms | 10.32 ms | 12.11 ms |
| `te`, mxfp8 fused | 10.11 ms | 9.49 ms | 10.45 ms |

### TE throughput (`--dsv3`)

| Precision | 4 GB200 · Mtok/s | 4 GB300 · Mtok/s | 8 GB300 · Mtok/s |
|---|---:|---:|---:|
| BF16 | 1.20 | 1.33 | 2.32 |
| MXFP8 | 1.49 | 1.59 | 2.71 |
| MXFP8 fused | 1.62 | 1.73 | 3.14 |

4 GPUs = 1 node / 32 experts; 8 GPUs = 2 nodes / 64 experts.
Both nodes share one NVLink domain (MNNVL).

“Fused” enables `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`.

### Benchmark setup

| Setting | Value |
|---|---|
| hardware | GB200 (SM100), one node; GB300 (SM103), one or two nodes; 4 GPUs per node, both GB300 nodes in one NVLink domain (MNNVL) |
| software | CUDA 13.3, NCCL 2.30.7, PyTorch 2.13.0a0+9186a08b2c.nv26.07, cuDNN 9.26, cuDNN Frontend 1.29.0, CUTLASS DSL 4.6.2 |
| NVIDIA driver | GB200: 580.173.02; GB300: 580.173.10 |
| `--dsv3` dims | hidden 7168, 128 heads, MLA q_lora 1536 / kv_lora 512 / nope 128 / rope 64 / v 128, expert ffn 2048, shared expert ffn 2048 |
| MoE | 8 local experts per rank (32 on 4 GPUs, 64 on 8), top-k 8, 4096 tokens per rank |
| precision | bf16 params and activations; `--recipe mxfp8` = MXFP8 block scaling for the expert GEMMs and dense projections |
| timing | fwd + bwd, 5 warmup, 10 timed iterations per run, median of 3 runs; wall-clock timing with CUDA synchronization, no CUDA graphs or profiler |
| environment | `OMP_NUM_THREADS=8`, `NVTE_GROUPED_LINEAR_SINGLE_PARAM=0`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=1`, `NVTE_FLASH_ATTN_V2=1`, `NVTE_FLASH_ATTN_V3=0`, `NVTE_FLASH_ATTN_V4=0`; EP buffer reused |

Each run reuses a normally distributed random input and backpropagates an all-ones
output gradient. Parameter and input gradients are cleared before each iteration.

`ep_bootstrap` is given the recv capacity returned by
`DeepSeekV3MoE.ep_recv_capacity(ep_size, tokens_per_rank, topk, num_local_experts)`, with
per-expert zones aligned to 256 rows.
