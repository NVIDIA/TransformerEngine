# H20 tensor-parallel reduction validation

The option selects the communication dtype and casts the sum back to the original
dtype. These results measure correctness and cost; they do not establish an output
accuracy improvement for two BF16 addends or a general inference speedup.

## Source and environment

- Reviewed baseline: `bfc06495d0ec8d734f23700afd7ca7de7e103359`.
- Tested implementation: `c0fda11ad0ab0216f3d7780c4a31624bad328141`. Every formal worker imported its
  selected clean source tree. Both use the same native extension built from the
  reviewed head's unchanged C++/CUDA sources, pinned CUTLASS and SM90 target.
- 2 × NVIDIA H20, driver 580.105.08, Torch 2.7.0+cu128, CUDA/NVCC 12.8,
  cuDNN 9.7.1.26, NCCL 2.26.2. Task-local dependency environment reuses root Torch.
  `CUDA_DEVICE_MAX_CONNECTIONS=1`, `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, OMP=8.
- [Raw samples, PIDs, commands and checkpoint manifest](results/h20_reduction_measurements.json).

## Correctness

| Check | Result |
|---|---|
| Two-rank NCCL matrix | 256 passed per rank; no skips or uncaught warnings |
| Reviewed-head negative control | 3 new feature cases fail as expected |
| Original affected sanity matrix | 7,668 passed, 8,964 skipped, 4,403 deselected |
| Full checkpoint inference | All 16 formal worker pairs finish with exit 0; prefill logits and 32 greedy tokens exactly match the native-dtype TE reference |
| L0 license | Pass |
| pre-commit on all changed files | Pass |
| L0 Python lint | Same pre-existing E1135 on `pools` in reviewed and updated source; no additional diagnostics |

The matrix covers FP16/BF16/FP32 inputs with native/FP16/BF16/FP32/FP64 reductions,
all-reduce/reduce-scatter, contiguous/strided inputs, sync/async cast-back, explicit
output identity, singleton validation, compile eager fallback and Userbuffers
fusion refusal. Native-dtype all-reduce retains NCCL's contiguous-input requirement.
Each communication sum is checked against independently gathered FP64 addends with
exact comparison. LayerNormLinear normalization gradients also use a separate FP64
PyTorch reference with the existing dtype tolerances from `tests/pytorch/utils.py`.
Other backward gradients and same-dtype/default outputs match their controls exactly.
The original sanity matrix skips unsupported recipe/model/override combinations.

LayerNormLinear publicly rejects row parallelism in this version. Its dtype option
therefore applies to the column-parallel input-gradient all-reduce or reduce-scatter;
its forward output is unchanged. Linear/LayerNormMLP/ops.Linear select forward
reduction dtype and leave backward unchanged.

## Measurement method

Each comparison uses one fresh APPA and one fresh PAAP quartet, with two independently
launched ranks per arm. No ranks or timing samples are reused across arms. Each
number is the geometric mean of four process-pair medians per role. For each timing
round, the slower rank supplies the critical-path sample. Speedup is A/P; below 1
means P is slower. Two independent quartets support descriptive ratios, not a
statistical significance or equivalence claim.

Graph cases execute complete forward **and backward**, including GEMMs and real NCCL
communication, with BF16 input/weights/output, hidden/output width 1024 and MLP width
4096. After side-stream warm-up, each graph contains three passes; 15 rounds each
replay 20 times (900 measured passes). Graph objects are reset before communicator
shutdown. The default comparison pins source; the cost comparison pins the updated
source and changes only BF16 versus FP32 communication.

### Default source control: forward/backward

| Path | SP | Global rows | Reviewed head (µs) | Updated default (µs) | Speedup A/P |
|---|---:|---:|---:|---:|---:|
| Linear | 0 | 2 | 20.466 | 20.544 | 0.9962× |
| Linear | 0 | 512 | 38.620 | 38.787 | 0.9957× |
| Linear | 1 | 2 | 27.892 | 27.903 | 0.9996× |
| Linear | 1 | 512 | 49.043 | 48.143 | 1.0187× |
| LayerNormLinear | 0 | 2 | 27.938 | 27.972 | 0.9988× |
| LayerNormLinear | 0 | 512 | 44.184 | 44.303 | 0.9973× |
| LayerNormLinear | 1 | 2 | 45.219 | 45.445 | 0.9950× |
| LayerNormLinear | 1 | 512 | 69.498 | 68.695 | 1.0117× |
| LayerNormMLP | 0 | 2 | 61.670 | 62.011 | 0.9945× |
| LayerNormMLP | 0 | 512 | 177.743 | 179.156 | 0.9921× |
| LayerNormMLP | 1 | 2 | 94.359 | 97.092 | 0.9719× |
| LayerNormMLP | 1 | 512 | 231.238 | 235.058 | 0.9837× |
| ops.Linear | 0 | 2 | 20.738 | 20.846 | 0.9948× |
| ops.Linear | 0 | 512 | 38.442 | 38.449 | 0.9998× |
| ops.Linear | 1 | 2 | 27.332 | 27.089 | 1.0090× |
| ops.Linear | 1 | 512 | 48.834 | 47.770 | 1.0223× |

### FP32 communication cost: forward/backward

| Path | SP | Global rows | BF16 communication (µs) | FP32 communication (µs) | Speedup A/P |
|---|---:|---:|---:|---:|---:|
| Linear | 0 | 2 | 20.654 | 26.143 | 0.7900× |
| Linear | 0 | 512 | 39.244 | 59.512 | 0.6594× |
| Linear | 1 | 2 | 28.573 | 34.998 | 0.8164× |
| Linear | 1 | 512 | 49.398 | 59.801 | 0.8260× |
| LayerNormLinear | 0 | 2 | 28.165 | 33.669 | 0.8365× |
| LayerNormLinear | 0 | 512 | 44.343 | 68.968 | 0.6429× |
| LayerNormLinear | 1 | 2 | 47.340 | 46.968 | 1.0079× |
| LayerNormLinear | 1 | 512 | 68.605 | 74.109 | 0.9257× |
| LayerNormMLP | 0 | 2 | 61.906 | 67.337 | 0.9193× |
| LayerNormMLP | 0 | 512 | 179.142 | 206.638 | 0.8669× |
| LayerNormMLP | 1 | 2 | 96.344 | 100.372 | 0.9599× |
| LayerNormMLP | 1 | 512 | 235.300 | 252.474 | 0.9320× |
| ops.Linear | 0 | 2 | 20.810 | 26.242 | 0.7930× |
| ops.Linear | 0 | 512 | 38.624 | 59.389 | 0.6504× |
| ops.Linear | 1 | 2 | 27.037 | 34.448 | 0.7849× |
| ops.Linear | 1 | 512 | 47.803 | 60.005 | 0.7967× |

## Full pretrained model

Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, BF16, batch 1,
128/512 input tokens, exactly 32 greedy output tokens, fresh KV cache per request.
All 28 attention output projections use row-parallel module Linear or ops.Linear,
with weights split along the input dimension; all 28 post-attention RMSNorm/SwiGLU
MLPs use LayerNormMLP.
HF SDPA attention stays replicated. Sequence-parallel outputs are gathered for the
HF caller; odd decoder rows are zero-padded to two and discarded after gathering.
This validates the complete checkpoint in this explicit TE integration.

Every arm has two warm-up generations and five timed complete generations per case.
Real collective calls are counted by dtype: 8,960 measured reductions per rank per
case. No requested FP32 path silently communicates in BF16. Weight addresses remain
stable within a case, all workers use the same checkpoint and inputs, and outputs
match the independently started native-dtype TE reference exactly. Checkpoint files
were removed after all model workers exited, freeing 1,519,183,121 bytes.

### Default source control: complete generation

| Path | SP | Input tokens | Reviewed head (ms) | Updated default (ms) | Speedup A/P |
|---|---:|---:|---:|---:|---:|
| Linear | 0 | 128 | 836.397 | 853.224 | 0.9803× |
| Linear | 0 | 512 | 842.047 | 871.269 | 0.9665× |
| Linear | 1 | 128 | 1205.393 | 1196.570 | 1.0074× |
| Linear | 1 | 512 | 1171.371 | 1176.668 | 0.9955× |
| ops.Linear | 0 | 128 | 932.642 | 961.323 | 0.9702× |
| ops.Linear | 0 | 512 | 951.119 | 960.137 | 0.9906× |
| ops.Linear | 1 | 128 | 1246.733 | 1251.004 | 0.9966× |
| ops.Linear | 1 | 512 | 1245.460 | 1247.512 | 0.9984× |

### FP32 communication cost: complete generation

| Path | SP | Input tokens | BF16 communication (ms) | FP32 communication (ms) | Speedup A/P |
|---|---:|---:|---:|---:|---:|
| Linear | 0 | 128 | 870.969 | 873.775 | 0.9968× |
| Linear | 0 | 512 | 867.581 | 883.284 | 0.9822× |
| Linear | 1 | 128 | 1165.837 | 1217.726 | 0.9574× |
| Linear | 1 | 512 | 1184.607 | 1287.183 | 0.9203× |
| ops.Linear | 0 | 128 | 979.327 | 991.415 | 0.9878× |
| ops.Linear | 0 | 512 | 929.249 | 989.944 | 0.9387× |
| ops.Linear | 1 | 128 | 1248.165 | 1296.724 | 0.9626× |
| ops.Linear | 1 | 512 | 1233.319 | 1292.667 | 0.9541× |

## Reproduce

Build the reviewed head for SM90 with the pinned source dependencies and retain that
native extension for both Python trees. Launch each arm with the same private runtime:

```bash
PYTHONPATH=/path/to/source CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_FRAMEWORK=pytorch NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 OMP_NUM_THREADS=8 \
python launch.py --label graph-A --output-dir /path/to/results -- \
  python benchmark.py --reduction default --output /path/to/results/graph-A.json

PYTHONPATH=/path/to/source CUDA_DEVICE_MAX_CONNECTIONS=1 \
NVTE_FRAMEWORK=pytorch NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 OMP_NUM_THREADS=8 \
python launch.py --label model-A --output-dir /path/to/results -- \
  python model_e2e.py --model /path/to/pinned-checkpoint \
  --reduction default --output /path/to/results/model-A.json
```

The first model arm creates the diagnostic reference; subsequent arms pass
`--reference /path/to/results/model-A-reference`. Repeat in APPA and PAAP orders.
For the cost comparison, both sources are the updated tree and reductions are
`bf16` for A and `fp32` for P. The launcher waits for both natural exits.
Earlier reference-layout and graph-stream diagnostics remain in the retained
evidence and are excluded from formal timing. Two earlier diagnostic ranks blocked
at communicator shutdown and later self-aborted through NCCL's watchdog (exit -6).
The launcher sent no signals and waited for their exits. All formal ranks exited 0.
