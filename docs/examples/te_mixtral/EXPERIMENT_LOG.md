# TE Mixtral — Experimental Log

The project scope is to create an implementation of Transformer Engine for a large language model. mixtral - How to wrap MoE layers with TE modules. All the files we modified is the docs/exa,ples/te_mixtral.

Investigating why TE `GroupedLinear` underperforms a naive Python loop with
`F.linear` on Mixtral-8x7B. 
All experiments on **8x B300, NGC pytorch-25.12-py3, 5-10 warmup + 30 timed. 
steps** unless otherwise noted. Reported metric is the **median** step time. "last" is the final step time as a sanity check on
steady state.

### Commands used

Wall-clock sweep (no profiler, produces `*.log`):

```bash
torchrun --standalone --nproc_per_node=8 run_finetune_ep.py \
    --improvement <TIER> --ep-size 2 --batch-size <B> --max-seq-length 8192 \
    --warmup-steps 10 --train-steps 200 \
    2>&1 | tee logs/sweep_seq8k_ep2_8gpus/seq8k_batch${B}_ep2_tier${TIER}.log
```

Nsys capture with NVTX markers (Exp 3, produces `*.nsys-rep`). NVTX
ranges are pushed around `dataloader / forward / backward /
optimizer_step` in `utils.py:finetune_model` and around the expert FFN
body (`expert_ffn_loop` / `expert_ffn_grouped`) in
`te_mixtral.py:_expert_ffn`:

```bash
nsys profile --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
    --output=logs/sweep_seq8k_ep2_8gpus/seq8k_batch8_ep2_tier${TIER}_nvtx \
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py \
        --improvement ${TIER} --ep-size 2 --batch-size 8 \
        --max-seq-length 8192 --warmup-steps 5 --train-steps 30
```

Post-process traces into CSV summaries:

```bash
# per-NVTX-range totals (used for the per-phase tables in Exp 3)
nsys stats --report nvtx_sum         --format csv --output . <trace>.nsys-rep
# per-CUDA-kernel totals (used for the GEMM / FillFunctor / splitKreduce counts)
nsys stats --report cuda_gpu_kern_sum --format csv --output . <trace>.nsys-rep
```

### Log folder layout

```
docs/examples/te_mixtral/logs/
├── sweep_seq8k_ep2_5tiers/                              # Exp 1 — 5-tier sweep at batch=2
└── sweep_seq8k_ep2_8gpus/                               # Exp 2/3 — batch sweep + nsys
    ├── seq8k_batch{1,2,4,8,16}_ep2_tier{2,3}.log        #   wall-clock sweep
    ├── seq8k_batch1_ep2_tier{1..5}.log                  #   batch=1 5-tier
    ├── seq8k_batch8_ep2_tier{2,3}.nsys-rep              #   Exp 3 kernel trace
    ├── seq8k_batch8_ep2_tier{2,3}_nvtx.nsys-rep         #   Exp 3 NVTX trace
    ├── seq8k_batch8_ep2_tier{2,3}_nvtx_nvtx_sum.csv     #   nvtx_sum CSV
    └── routing_{hf,te-loop,te-grouped}*.log             #   router-balance probe
```

Shared copies of the Exp 3 NVTX artifacts live at
`/lustre/share/coreai_prod_infbench/exp3_nvtx_per_phase/`.

---

All experiments run on a **SLURM cluster**. The login node has no CUDA, so any
experiment must execute inside an allocation on a compute node with 8 GPUs.

### Reserving a fresh node (8x GPU, 4h, exclusive)

```bash
srun -A coreai_prod_infbench \
     -p batch \
     -N 1 \
     -J myjob \
     -t 04:00:00 \
     --exclusive \
     --mpi=pmix \
     --container-image=/lustre/fsw/coreai_prod_infbench/faradawny/docker/pytorch-25.12-py3.sqsh \
     --container-save=/lustre/fsw/coreai_prod_infbench/faradawny/docker/pytorch-25.12-py3.sqsh \
     --container-name=mycontainer \
     --container-mounts=/lustre:/lustre \
     --pty bash
```

This drops you into an interactive shell inside the NGC `pytorch-25.12-py3`
container with `/lustre` mounted and the container saved as `mycontainer`.

### Reusing an existing reservation

Before requesting a new allocation, check for ongoing jobs and attach to one
instead — the reservation queue can be long, and an existing allocation
already has the container warmed up:

```bash
# List your running jobs
squeue -u $USER

# Attach to an existing job's allocation (overlap = share the node)
srun --jobid=<JOBID> --overlap --container-name=mycontainer --pty bash

# Or run a one-shot command inside the existing allocation
srun --jobid=<JOBID> --overlap --container-name=mycontainer \
     python3 docs/examples/te_mixtral/inspect_moe_block.py
```

The `--container-name=mycontainer` flag reuses the same container that the
original `srun` created, so installs and Python state persist across attaches.

---

## exp 0 - EP8 - false data (didn't reach steady state - too few training steps)
2026-04-17 Solve the Perf Bottleneck on Why TE is worse than HF
Slack: https://nvidia.slack.com/archives/C03V462SAMS/p1775171427302349	
Why a bigger sequence length seems to reduce the performance (of TE vs HF)? 
Benchmark Results (8x B300, 10 timed steps)

Tier                  Description               seq=256 (b=4)   seq=2048 (b=16)
--------------------  ------------------------  --------------  ---------------
1 - HF baseline BF16  device_map="auto"         582 ms          1030 ms
2 - Naive EP BF16     AllToAll (NCCL)            357 ms          2085 ms
3 - Fused EP BF16     DeepEP                     355 ms          2279 ms
4 - Fused EP FP8      DeepEP + DelayedScaling    404 ms          2056 ms

Peter St John
IMO we should figure out why we’re doing worse than HF on longer sequence lengths… the typical minimum context length for mixtral training would be 8192. have codex iterate with nsys and figure out why you’re slower
Sd

Using nsys-skill We found that the bottom neck is NCCL. Which is taking ninety two percent of the time. 


Apr 21, 2026 Meeting with Peter St. John and Timur to investigate pref bottleneck

seq=8192 (8 GPUs, EP=8)

┌──────┬──────────────────────┬─────────┬─────────┬─────────┬──────────┬──────────┬──────────┬──────────┐
│ Tier │ Configuration        │ Batch=2 │ Batch=4 │ Batch=8 │ Batch=16 │ Batch=32 │ Batch=64 │ Batch=128│
├──────┼──────────────────────┼─────────┼─────────┼─────────┼──────────┼──────────┼──────────┼──────────┤
│  1   │ HF BF16              │  597 ms │  721 ms │  782 ms │  1020 ms │  1890 ms │  4403 ms │ 10917 ms │
│  2   │ TE NCCL EP BF16      │ 2262 ms │ 2203 ms │ 2864 ms │  1690 ms │    OOM   │    OOM   │    OOM   │
│  3   │ TE Fused EP BF16     │ 2224 ms │ 1857 ms │ 3035 ms │  1762 ms │    OOM   │    OOM   │    OOM   │
│  4   │ TE Fused EP MXFP8    │ 1879 ms │ 2048 ms │ 3080 ms │  1650 ms │  2185 ms │    OOM   │    OOM   │
└──────┴──────────────────────┴─────────┴─────────┴─────────┴──────────┴──────────┴─

This is false data because we haven't reached a steady state. This is only using three training steps. There are a lot of variations. 


Because I only trained for ten steps, they suggested to run twenty training steps to reach a stable state. We shouldn't take the average of all the training steps Because the later training steps tend to be faster since the initialization of building the cache would take a lot of time. 

Steady state statistics 

Results — Median step time (seq=8192, 10 warmup + 200 timed steps, 8× B300)                                                                                     
                                                   
  ┌──────┬──────────────────────┬─────────┬─────────┬──────────┐                                                                                                  
  │ Tier │    Configuration     │ Batch=2 │ Batch=4 │ Batch=8  │                                                                                                  
  ├──────┼──────────────────────┼─────────┼─────────┼──────────┤                                                                                                  
  │  1   │ HF BF16 (device_map) │ 537 ms  │ 571 ms  │  694 ms  │                                                                                                  
  ├──────┼──────────────────────┼─────────┼─────────┼──────────┤                                                                                                  
  │  2   │ TE NCCL EP=8 BF16    │ 287 ms  │ 307 ms  │  crash¹  │                                                                                                  
  ├──────┼──────────────────────┼─────────┼─────────┼──────────┤                                                                                                  
  │  3   │ TE Fused DeepEP BF16 │ 275 ms  │ 308 ms  │ missing² │                                                                                                  
  ├──────┼──────────────────────┼─────────┼─────────┼──────────┤                                                                                                  
  │  4   │ TE Fused EP MXFP8    │ 315 ms  │ 318 ms  │ missing² │  
  └──────┴──────────────────────┴─────────┴─────────┴──────────┘                                                                                                  
                                                         



## Exp 1 - EP8 

 ┌───────────────────────────────────┬─────────┬─────────┬───────────┐                                                                                                                                                                                                                             
  │               Tier                │ batch=2 │ batch=4 │  batch=8  │
  ├───────────────────────────────────┼─────────┼─────────┼───────────┤                                                                                                                                                                                                                             
  │ 1 (HF)                            │  537 ms │  571 ms │    694 ms │                                                                                                                                                                                                                           
  ├───────────────────────────────────┼─────────┼─────────┼───────────┤
  │ 2 (NCCL alltoall + GroupedLinear) │  287 ms │  308 ms │     crash │
  ├───────────────────────────────────┼─────────┼─────────┼───────────┤                                                                                                                                                                                                                             
  │ 3 (Fused DeepEP + GroupedLinear)  │  275 ms │  308 ms │ (not run) │
  ├───────────────────────────────────┼─────────┼─────────┼───────────┤                                                                                                                                                                                                                             
  │ 4 (MXFP8 + Fused DeepEP)          │  315 ms │  318 ms │ (not run) │                                                                                                                                                                                                                           
  └───────────────────────────────────┴─────────┴─────────┴───────────┘   

all the experimental number below should be increased by one. 

## Experiment 1 —  more fine-grained tiers (5 tiers) to showcase groupedlinear (5 tiers) with EP2


**Setup:** 8× B300, EP=2 (DP=4), Mixtral-8x7B, batch=2, seq=8192,
5 warmup + 50 timed steps. Logs: `logs/sweep_seq8k_ep2_5tiers/`.

| Tier | Config | Median (ms) |
|:---:|---|---:|
| 1 | HF baseline BF16 (`device_map="auto"`) | 539 |
| 2 | TE EP=2 BF16 — Python loop, F.linear per expert | **259** |
| 3 | TE EP=2 BF16 — GroupedLinear | 479 |
| 4 | TE EP=2 BF16 — GroupedLinear + Fused DeepEP | 468 |
| 5 | TE EP=2 FP8 (Float8CurrentScaling) + GroupedLinear + DeepEP | 511 |

**Hypothesis FALSIFIED.** Tier 2 (loop) is **1.85× faster** than Tier 3 (GroupedLinear).

---

## Experiment 2 — See if batch size can improve GroupedLinear performance

**Setup:** 8× B300, EP=2 (DP=4), 4 experts/rank, BF16, seq=8192, Mixtral-8x7B
10 warmup + 200 timed steps. Logs: `logs/sweep_seq8k_ep2_8gpus/`.

| Config | batch=1 | batch=2 | batch=4 | batch=8 | batch=16 |
|---|---:|---:|---:|---:|---:|
| Tier 2 (loop)        |   240 ms |   242 ms |   242 ms |   306 ms¹ |  506 ms |
| Tier 3 (grouped)     |   449 ms |   474 ms |   511 ms |   617 ms¹ |   OOM²  |

Using agent skill to do nsys profiling. Only on batch size of 2.

**Setup:** 8× B300, EP=2, batch=2, seq=8192, 5 warmup + 30 timed steps.
nsys traces: `logs/sweep_seq8k_ep2_8gpus/nsys_tier{2,3}_batch2.nsys-rep`.

| Kernel category | Tier 2 (loop) | Tier 3 (grouped) | Note |
|---|---:|---:|---|
| `ncclDevKernel_SendRecv` | 46.2 s (66.3%) | 37.1 s (26.8%) | comm: T3 ~80% of T2 |
| BF16 GEMM (`nvjet_sm103_tst_*`) total launches | 10,693 | **66,083** | T3 fires **6.2× more GEMM kernels** |
| `cublasLt::splitKreduce_kernel` | absent | 48,823 (1.2%) | T3 uses **split-K** GEMM |
| `FillFunctor<bf16>` (zero-init) | absent from top 15 | 138,296 (9.6%) | T3 zeros many intermediate buffers |
| `CUDAFunctor_add` (elementwise) | absent from top 15 | 98,560 (12.1%) | T3 has lots of intermediate adds |
| AdamW kernels (per-kind) | ~45,640 | ~63,560 | T3 has 40% more |


Then, chatted with Pawel engineer. He says For tier 2, from batch size of 2 to batch of 4, the loop's implementation stays the same. It is CPU Bound. Thus, use batch size 8 to do nsys profile.

## Experiment 3 — nsys profile of Tier 2 vs Tier 3 (batch=8)

**Setup:** 8× B300, EP=2, batch=8, seq=8192, 5 warmup + 30 timed steps.
nsys traces: `logs/sweep_seq8k_ep2_8gpus/seq8k_batch8_ep2_tier{2,3}.nsys-rep`.

| Kernel | Tier 2 (loop) | Tier 3 (grouped) | T3 / T2 |
|---|---:|---:|---:|
| `ncclDevKernel_SendRecv` | 50.6 s (46.6%) | 51.6 s (25.8%) | 1.02× |
| `nvjet_sm103_tst_*` BF16 GEMM (sum of all variants) | **224,840** | **296,520** | **1.32×** |
| `cublasLt::splitKreduce_kernel` | 0 | **11,377** (0.2%) | T3-only |
| `FillFunctor<bf16>` (zero-init) | (not in top 15) | **138,240** (6.8%) | T3-only |
| `CUDAFunctor_add` (intermediate adds) | (not in top 15) | **98,560** (8.5%) | T3-only |
| **Total estimated GPU kernel time** | ~108 s | **~200 s** | **1.84×** |

**Wall-time at batch=8:** T2 = 306 ms, T3 = 617 ms → **2.02×** (matches GPU-time ratio within noise).

### batch=2 → batch=8 — what changed in the cuBLAS path

| Metric | batch=2 | batch=8 | Δ |
|---|---:|---:|---:|
| T3 GEMM launches | 66,083 | 296,520 | +349% |
| T2 GEMM launches | 10,693 | 224,840 | +2003% (cuBLAS picks more kernel variants at larger M) |
| **T3 / T2 GEMM ratio** | **6.18×** | **1.32×** | **dropped 5×** |
| `cublasLt::splitKreduce_kernel` (T3) | 48,823 | **11,377** | -77% |
| `FillFunctor<bf16>` (T3) | 138,296 | 138,240 | unchanged |
| `CUDAFunctor_add` (T3) | 98,560 | 98,560 | unchanged |
| **GPU kernel time ratio T3/T2** | 1.99× | 1.84× | partially closed |

### Two findings
1. **cuBLAS partially recovers at larger M.** The grouped-matmul heuristic
   reduces split-K usage from 48,823 reductions (batch=2) to 11,377
   (batch=8) — 4× drop. The 6.18× GEMM-count multiplier shrinks to 1.32×.
   The heuristic *is* making better choices at larger per-expert M.

2. **The constant per-step overhead persists.** `FillFunctor<bf16>` (~138k)
   and `CUDAFunctor_add` (~99k) are identical at batch=2 and batch=8 —
   they're tied to the *number of GroupedLinear invocations* per training
   step (~32 layers × N sub-steps), not per-call workload. This fixed
   ~180k-extra-kernels-per-step tax is what keeps the wall-time gap at 2×
   even when cuBLAS picks better GEMMs.

**Implication:** even if cuBLAS's split-K selection is fixed for this shape
class, the buffer-management scaffolding (zero-fill + intermediate adds)
inside the grouped path is a second source of overhead. Worth investigating
whether those buffers can be pooled / reused across calls instead of
zero-filled each invocation.

### Per-phase NVTX breakdown (batch=8)

Re-ran with `torch.cuda.nvtx.range_push/pop` markers around `dataloader`,
`forward`, `backward`, `optimizer_step`, and around the expert FFN body
(`expert_ffn_loop` / `expert_ffn_grouped`). Numbers are **median per step
per rank** from `nsys stats --report nvtx_sum` over 30 timed steps × 8
ranks. Traces:
`logs/sweep_seq8k_ep2_8gpus/seq8k_batch8_ep2_tier{2,3}_nvtx.nsys-rep`.

| Phase               | Tier 2 (loop) | Tier 3 (grouped) |   Δ      |
|---------------------|--------------:|-----------------:|---------:|
| `:dataloader`       |        7 ms   |          7 ms    |     0    |
| `:forward`          |      193 ms   |        189 ms    |   −4 ms  |
| `:backward`         |      137 ms   |        235 ms    | **+98 ms**  |
| `:optimizer_step`   |       19 ms   |         87 ms    | **+68 ms**  |
| **Sum (≈ step)**    |  **357 ms**   |     **518 ms**   | **+161 ms** |

**Where is the FFN time inside `:forward`?**

| Range                       | Tier 2 (loop) | Tier 3 (grouped) |
|-----------------------------|--------------:|-----------------:|
| `:expert_ffn_*` median/call |       1.27 ms |          1.31 ms |
| × 32 layers (per fwd)       |      40.6 ms  |         41.9 ms  |
| as % of `:forward`          |        21 %   |           22 %   |

→ **The forward FFN is the same speed in both modes.** GroupedLinear
forward is *not* the regression.

**Where the regression actually lives — answers to (a)/(b)/(c):**

- **(a) Forward FFN share, loop vs grouped.** ~21 % vs ~22 % of `:forward`.
  Essentially identical. The forward GroupedLinear kernel itself is fine
  at 4 experts/rank.
- **(b) FFN forward+backward split.** Forward is matched (≈ 41 ms in both
  modes). The 161 ms regression is concentrated in `:backward`
  (+98 ms, ~60 % of the gap) and `:optimizer_step` (+68 ms, ~42 %).
  `:nvte_flash_attn_bwd` and `:nvte_rmsnorm_bwd` medians are unchanged
  between tiers, so the extra backward time is the GroupedLinear
  dgrad/wgrad path — see GEMM counts below.
- **(c) Optimizer step share.** Tier 2: 19 ms ≈ **5.4 %** of step.
  Tier 3: 87 ms ≈ **16.8 %** of step → **4.5× wall time, 3.1× share**.
  Tier 3 holds AdamW state for one big stacked weight per layer; the
  optimizer is no longer cheap.

**GEMM dispatch volume per step per rank** (CPU-side range count and time
inside `:forward + :backward`):

| Range                        | Tier 2 (loop) | Tier 3 (grouped) |
|------------------------------|--------------:|-----------------:|
| `:nvte_cublas_gemm_v2` calls |       339     |      **1,235**   |
| `:nvte_multi_tensor_gemm` calls |    0       |      **224**     |
| Combined GEMM-launch CPU time |     ~33 ms   |     **~121 ms**  |

Tier 3 dispatches **3.6× more** plain cuBLAS GEMMs *and* adds an entirely
separate `nvte_multi_tensor_gemm` path (absent in Tier 2). The extra
calls land mostly in backward — matching the +98 ms backward Δ above.

**Takeaway.** The earlier "GroupedLinear is slow at small M" narrative was
incomplete. At batch=8 / 4-experts-per-rank the grouped *forward* matches
the loop. The 2× wall-clock gap is two separate issues: (1) GroupedLinear
**backward** spawns a high-volume grouped dgrad/wgrad through
`multi_tensor_gemm` plus 3.6× more `cublas_gemm_v2` calls, and (2) the
**optimizer** step on the stacked expert weights is ~4.5× slower than
AdamW on the per-expert tensors used by the loop.
