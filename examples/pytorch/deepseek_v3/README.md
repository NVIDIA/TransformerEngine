# DeepSeekV3Layer with expert parallelism

A forward + backward benchmark of one `DeepSeekV3Layer`: RMSNorm, multi-latent attention,
and MoE with a shared expert. Routed experts are sharded across GPUs using NCCL EP.

## Results

GB300 GPUs, 4096 tokens per rank, top-k 8, 8 local experts per GPU.
Times cover one layer's forward + backward; throughput is global, in millions of tokens/s.
See [benchmark configuration](#c-benchmark-configuration) for the full setup.

### TE vs. plain PyTorch (`--dsv3`)

Same layer and dimensions, BF16 unless noted.

The naive measurements predate the router-backward fix and omit probability-gradient
communication; updated timings are pending. They do not yet establish training speedups.

| MoE implementation | 4 GPUs (32 experts) | 8 GPUs (64 experts) |
|---|---:|---:|
| `naive`: all_to_all + loop over experts | 26.83 ms | 27.34 ms |
| `naive_grouped`: all_to_all + TE grouped GEMM | 16.84 ms | 17.20 ms |
| `te`: NCCL EP + grouped GEMM | 13.09 ms | 15.06 ms |
| `te`, mxfp8 (unfused grouped GEMM) | 11.02 ms | 12.88 ms |
| `te`, mxfp8 fused | 10.19 ms | 11.32 ms |

At the small default dims (4 GPUs): `naive` 10.08 ms, `naive_grouped` 6.80 ms, `te` 6.17 ms.

### TE precision and throughput (`--dsv3`)

| Precision | 4 GPUs · ms/iter | 4 GPUs · Mtok/s | 8 GPUs · ms/iter | 8 GPUs · Mtok/s |
|---|---:|---:|---:|---:|
| BF16 | 13.09 | 1.25 | 15.06 | 2.18 |
| MXFP8 | 11.02 | 1.49 | 12.88 | 2.54 |
| MXFP8 fused | 10.19 | 1.61 | 11.32 | 2.89 |

4 GPUs = 1 node / 32 experts; 8 GPUs = 2 nodes / 64 experts.
Both nodes share one NVLink domain (MNNVL).

### TE at small default dimensions

1 node, 4 GPUs, hidden 2048, 16 heads, expert FFN 1024, 32 experts.

| Precision | ms/iter | Mtok/s |
|---|---:|---:|
| BF16 | 6.17 | 2.65 |
| MXFP8 fused | 8.4 | 1.95 |

“Fused” enables `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`.
At the small dimensions, MXFP8 fused is slower than BF16.

## Quick start

Requires SM90+ GPUs with NVLink and an NCCL EP-enabled TE build;
see [full requirements](#a-requirements). From this directory:

```bash
bash run_deepseek_v3_layer_ep.sh --dsv3
```

For MXFP8 with the fused grouped MLP (SM100-class GPUs):

```bash
NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 bash run_deepseek_v3_layer_ep.sh --dsv3 --recipe mxfp8
```

## Appendix

[Requirements](#a-requirements) · [Running](#b-running) ·
[Configuration](#c-benchmark-configuration) · [Profiling](#d-profiling-with-nsys) ·
[TE kernels](#e-te-kernel-breakdown) ·
[Naive kernel profiles](#f-naive-kernel-profiles) ·
[EP internals](#g-ep-implementation-notes)

### A. Requirements

The following NCCL EP requirements apply to `--impl te`. The naive variants use
ordinary NCCL collectives and can also run on older GPUs without NVLink.

- SM90 or newer GPUs connected with NVLink (NCCL EP falls back to the network transport and
  deadlocks on PCIe-only nodes).
- NCCL >= 2.30.4, PyTorch >= 2.11 (symmetric memory), Transformer Engine built with the
  `3rdparty/nccl-extensions` submodule.
- `NVTE_CUTEDSL_FUSED_GROUPED_MLP=1` additionally needs SM100-class GPUs and the CuTe DSL
  (`nvidia-cutlass-dsl`) for the fused MXFP8 grouped MLP.

### B. Running

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

### C. Benchmark configuration

| Setting | Value |
|---|---|
| hardware | GB300 (SM103), 4 GPUs per node, both nodes in one NVLink domain (MNNVL) |
| software | CUDA 13.3, NCCL 2.30.7, PyTorch 2.13.0a0+9186a08b2c (NGC 26.07 build), cuDNN 9.24, Transformer Engine from this PR |
| `--dsv3` dims | hidden 7168, 128 heads, MLA q_lora 1536 / kv_lora 512 / nope 128 / rope 64 / v 128, expert ffn 2048, shared expert ffn 2048 |
| MoE | 8 local experts per rank (32 on 4 GPUs, 64 on 8), top-k 8, 4096 tokens per rank |
| precision | bf16 params and activations; `--recipe mxfp8` = MXFP8 block scaling for the expert GEMMs and dense projections |
| timing | fwd + bwd, 5 warmup, 10 timed iterations, no CUDA graphs |

Each iteration uses random data. After timing, the script checks the final output and
gradients on every rank, including the presence of input and router gradients.

### D. Profiling with nsys

The timed iterations run inside a `torch.cuda.profiler.start()` / `stop()` window, so
`-c cudaProfilerApi` records only them, one NVTX range per iteration:

```bash
NSYS=1 NVTE_CUTEDSL_FUSED_GROUPED_MLP=1 bash run_deepseek_v3_layer_ep.sh --dsv3 --recipe mxfp8
# -> results/deepseek_v3_layer_ep_<hostname>.nsys-rep
nsys stats --report cuda_gpu_kern_sum results/deepseek_v3_layer_ep_<hostname>.nsys-rep
```

The launcher wraps `torchrun`, so all local ranks land in one report. For multi-node runs put
the same `nsys profile ... -o <path>_%q{SLURM_NODEID}` in front of `torchrun` on each node.

At the small default dims MXFP8 is slower than bf16: the fused path launches many small
quantization kernels and the layer becomes CPU-launch-bound. Running under `nsys` adds
about 1.5 ms per iteration to these numbers.

### E. TE kernel breakdown

8 GPUs, `--dsv3`, MXFP8 fused. Per GPU and iteration, from `nsys stats --report cuda_gpu_kern_sum` on one node
(kernel time 11.7 ms of an 11.3 ms iteration: the GPU is busy back to back):

| Group | ms | Kernels |
|---|---:|---|
| NCCL EP all-to-all | 2.3 | `nccl_ep_jit_ht_dispatch_kernel` (0.99), `nccl_ep_jit_ht_combine_kernel` (1.31), each twice per iteration (fwd + bwd) |
| NCCL EP local permute + routing all-gather | 1.3 | `local_permute_dup/reduce` (0.62), `ncclDevKernel_AllGather_RING_LL` (0.65, includes waiting for slower ranks) |
| fused grouped MLP (cuDNN, MXFP8) | 2.7 | fc1+SwiGLU fwd (0.58), fc2 fwd (0.9), dGLU bwd (0.35), wgrad (0.85) |
| MXFP8 quantization | 1.1 | `group_quantize_mxfp8` on the recv buffer (0.4), `quantize_mxfp8_kernel_cast_only` for dense GEMM inputs (0.7) |
| dense MXFP8 GEMMs (MLA projections, shared expert) | 1.3 | `nvjet_sm103_qqtst_*` |
| attention (cuDNN SDPA) | 0.7 | flash fprop (0.19) + bprop (0.51) |
| elementwise | 1.1 | residual/shared-expert adds (0.73), RMSNorm fwd+bwd (0.39) |

Both nodes sit in one NVLink domain, so dispatch and combine move roughly 470 MB per GPU per
call over NVLink at close to link bandwidth; on an InfiniBand-connected pair of nodes the
all-to-all share would be much larger.

### F. Naive kernel profiles

The naive timings and profiles below predate the routing-probability autograd fix:
they omit router backward and probability-gradient communication. They are historical
measurements and must be rerun before drawing training-speedup conclusions. The current
benchmark also clears parameter gradients before each step.

Per GPU and iteration, the `naive` MoE spends (8 GPUs, kernel time 25.9 ms of a 28.5 ms
iteration):

| Group | ms | Details |
|---|---:|---|
| `ncclDevKernel_SendRecv` | 5.2 | 7 all_to_all launches per iteration (tokens fwd/bwd, results fwd/bwd, counts, indices, probs) |
| expert and dense GEMMs (`nvjet_*`) | 6.4 | 8 separate GEMM pairs per rank instead of one grouped GEMM, plus the MLA projections |
| elementwise adds | 4.1 | `index_add` and its backward, residuals |
| `FillFunctor` (zeros) | 2.5 | `zeros_like` for the per-expert output buffer and `index_add` targets |
| device-to-device copies | 2.3 | `index_copy` and gathers materialising per-expert slices |
| indexing kernels | 3.0 | `x[tok]`, `nonzero` masks, `index_copy`, `indexing_backward` |
| attention, norms | 1.1 | same as in the TE variant |

`naive_grouped` (8 GPUs, kernel time 17.1 ms of a 17.2 ms iteration):

| Group | ms | Details |
|---|---:|---|
| `ncclDevKernel_SendRecv` | 3.8 | the same 7 all_to_all launches, less time because the GPU is no longer stalled between them |
| grouped GEMMs (`nvjet_*_ptrGroup_*`) | 4.1 | fc1 / fc2 forward, dgrad, wgrad as grouped GEMMs, same as in `te` |
| sorting rows by expert and back | 2.6 | `argsort`, gathers (`x[tok]`, `x_recv[by_expert]`), `index_copy`, `indexing_backward` |
| dense GEMMs (MLA projections, shared expert) | 2.4 | same as in `te` |
| elementwise adds | 0.7 | `index_add`, residuals |
| attention, norms | 1.1 | same as in `te` |

Reading the three rows of the table together: the Python loop over experts costs about
10 ms per iteration (`naive` -> `naive_grouped`, 8 separate GEMM pairs, `nonzero` masks,
zero-filled buffers, copies); replacing torch `all_to_all` plus the surrounding sort / gather /
scatter with NCCL EP dispatch and combine, which write straight into the expert-major layout
and zero-fill the padding, saves another 2 ms (`naive_grouped` -> `te`). Both `naive`
variants also synchronise with the host twice per layer to learn the all_to_all split sizes;
`te` does not. `--recipe mxfp8` is only supported by `te`: the naive variants would need the
per-expert row counts padded to the MXFP8 block size.

### G. EP implementation notes

- `ep_bootstrap` must be given the same recv capacity the layer uses:
  `DeepSeekV3MoE.ep_recv_capacity(ep_size, tokens_per_rank, topk, num_local_experts)`.
- Per-expert zones in the recv buffer are aligned to 256 rows; the fused grouped MLP requires
  that alignment and reads a little past the last expert, which is why the layer keeps a
  zeroed margin after the received tokens.
- The recv and grad buffers are allocated uninitialized: NCCL EP zero-fills the alignment
  padding between experts itself.
