#!/usr/bin/env bash
# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
#
# 4-rank launcher for ep_bench.py.
# Examples:
#   bash run_ep_bench.sh                       # plain run, stdout only
#   bash run_ep_bench.sh --cuda-graph          # enable XLA command-buffer (cudaGraph), min_size=1
#   bash run_ep_bench.sh --nsys                # nsys on rank 0 -> results/jax_nsys.nsys-rep
#   bash run_ep_bench.sh --xplane              # jax.profiler on rank 0 -> results/xplane/
#
# Notes:
#   * nsys + xplane cannot be combined (both attach CUPTI -> MULTIPLE_SUBSCRIBERS).
#   * nsys + --cuda-graph is rejected: cudaGraph fires kernels via cuGraphLaunch
#     and detaches the host NVTX context, breaking per-stage attribution.
#   * stdout per rank lands in results/stdout_<tag>_rank_<i>.txt.

set -uo pipefail

NSYS=0; XPLANE=0; CGRAPH=0; SECOND_STEP=0
for a in "$@"; do
  case "$a" in
    --nsys)        NSYS=1 ;;
    --xplane)      XPLANE=1 ;;
    --cuda-graph)  CGRAPH=1 ;;
    --second-step) SECOND_STEP=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done
if [ "${NSYS}" -eq 1 ] && [ "${XPLANE}" -eq 1 ]; then
  echo "--nsys and --xplane both attach CUPTI; pick one." >&2; exit 2
fi
if [ "${NSYS}" -eq 1 ] && [ "${CGRAPH}" -eq 1 ]; then
  echo "--nsys and --cuda-graph cannot be combined: cudaGraph launches detach the" \
       "host NVTX context, so nvtx_kern_sum cannot attribute kernels to our ranges." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TE_REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
RESULTS="${SCRIPT_DIR}/results"
mkdir -p "${RESULTS}"
export PYTHONPATH="${TE_REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -lt 4 ]; then
  echo "EP bench requires >=4 GPUs (found ${NUM_GPUS}); SKIPPING."; exit 0
fi

# NCCL EP requires active NVLink P2P among ranks on the node.
if ! nvidia-smi nvlink --status 2>/dev/null | grep -qE 'Link [0-9]+:.*GB/s'; then
  echo "NVLink not detected on this platform — EP bench requires NVLink; SKIPPING."
  exit 0
fi

NUM=4
COORD="${COORD:-127.0.0.1:23457}"
TIMEOUT_S="${TIMEOUT_S:-1800}"

XLA_BASE="${XLA_BASE:---xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_graph_min_graph_size=1}"

if [ "${CGRAPH}" -eq 1 ]; then
  TAG="cudagraph"
  export XLA_FLAGS="${XLA_BASE} --xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL --xla_gpu_graph_min_graph_size=1"
else
  TAG="vanilla"
  export XLA_FLAGS="${XLA_BASE} --xla_gpu_enable_command_buffer="
fi
[ "${SECOND_STEP}" -eq 1 ] && TAG="${TAG}_step2"

: "${NCCL_EP_JIT_CACHE_DIR:=${TMPDIR:-/tmp}/nccl_ep_jit_cache_$(id -u)}"
export NCCL_EP_JIT_CACHE_DIR
mkdir -p "${NCCL_EP_JIT_CACHE_DIR}"

# JAX/XLA persistent compilation cache: first run pays full compile cost
# (cudaGraph capture + EP custom_calls is minutes); subsequent runs reuse it.
: "${JAX_COMPILATION_CACHE_DIR:=${TMPDIR:-/tmp}/jax_cache_$(id -u)}"
export JAX_COMPILATION_CACHE_DIR
mkdir -p "${JAX_COMPILATION_CACHE_DIR}"

export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.2}"
export NVTE_EP_SILENCE_NONSYMM_WARN="${NVTE_EP_SILENCE_NONSYMM_WARN:-1}"

ALL_RANKS_ARGS=()
R0_ONLY_ARGS=()
NSYS_PREFIX=()
SUFFIX=""
if [ "${SECOND_STEP}" -eq 1 ]; then
  ALL_RANKS_ARGS+=(--second-step)
fi
if [ "${XPLANE}" -eq 1 ]; then
  R0_ONLY_ARGS+=(--xplane "${RESULTS}/xplane_${TAG}")
  SUFFIX="_xplane"
fi
if [ "${NSYS}" -eq 1 ]; then
  SUFFIX="_nsys"
  export EP_BENCH_FLUSH_CUPTI=1
  NSYS_PREFIX=(nsys profile
               --output "${RESULTS}/jax_${TAG}_nsys"
               --force-overwrite=true
               --trace=cuda,nvtx
               --gpu-metrics-devices=none
               --cuda-um-cpu-page-faults=false
               --cuda-um-gpu-page-faults=false)
fi

OUT_PREFIX="stdout_${TAG}${SUFFIX}_rank"

for f in "${RESULTS}/${OUT_PREFIX}_"*.txt \
         "${RESULTS}/jax_${TAG}_nsys.nsys-rep" \
         "${RESULTS}/jax_${TAG}_nsys.sqlite" \
         "${RESULTS}/jax_${TAG}_nsys_nvtx_kern_sum.csv" \
         "${RESULTS}/jax_${TAG}_nsys_kern_sum.csv" \
         "${RESULTS}/summary_${TAG}${SUFFIX}.md"; do
  [ -f "$f" ] && mv -f "$f" "$f.prev"
done

PIDS=()
cleanup() { for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

for ((i=1; i<NUM; i++)); do
  timeout --foreground --signal=KILL "${TIMEOUT_S}" \
    python -u "${SCRIPT_DIR}/ep_bench.py" \
      --coordinator-address "${COORD}" --process-id "$i" --num-processes "${NUM}" \
      "${ALL_RANKS_ARGS[@]}" \
      > "${RESULTS}/${OUT_PREFIX}_${i}.txt" 2>&1 &
  PIDS+=($!)
done

R0_CMD=(python -u "${SCRIPT_DIR}/ep_bench.py"
        --coordinator-address "${COORD}" --process-id 0 --num-processes "${NUM}"
        "${ALL_RANKS_ARGS[@]}" "${R0_ONLY_ARGS[@]}")
if [ "${NSYS}" -eq 1 ]; then
  R0_CMD=("${NSYS_PREFIX[@]}" "${R0_CMD[@]}")
fi

WATCHDOG_PID=""
if [ "${NSYS}" -eq 1 ]; then
  ( while ! grep -q "kernel breakout" "${RESULTS}/${OUT_PREFIX}_0.txt" 2>/dev/null; do
      sleep 2
    done
    sleep 20
    pkill -INT -f "nsys profile --output ${RESULTS}/jax_${TAG}_nsys" 2>/dev/null || true
  ) &
  WATCHDOG_PID=$!
fi

timeout --foreground --signal=KILL "${TIMEOUT_S}" "${R0_CMD[@]}" 2>&1 | tee "${RESULTS}/${OUT_PREFIX}_0.txt"
if [ -n "${WATCHDOG_PID}" ]; then
  kill "${WATCHDOG_PID}" 2>/dev/null || true
fi
wait

SUMMARY="${RESULTS}/summary_${TAG}${SUFFIX}.md"
RANK0_LOG="${RESULTS}/${OUT_PREFIX}_0.txt"

{
  echo "# JAX EP bench summary — tag=${TAG}${SUFFIX}"
  echo ""
  echo "Generated: $(date -Iseconds)"
  echo "Rank-0 log: \`${RANK0_LOG}\`"
  echo ""
  echo "## Per-stage runtime (rank 0)"
  echo ""
  echo '```'
  awk '/^\| stage / {flag=1} flag {print; if (/combine[ ]+vjp-fwd/) {flag=0}}' "${RANK0_LOG}" || true
  echo '```'
} > "${SUMMARY}"

if [ "${NSYS}" -eq 1 ]; then
  NSYS_REP="${RESULTS}/jax_${TAG}_nsys.nsys-rep"
  NVTX_CSV="${RESULTS}/jax_${TAG}_nsys_nvtx_kern_sum.csv"
  KERN_CSV="${RESULTS}/jax_${TAG}_nsys_kern_sum.csv"
  if [ -f "${NSYS_REP}" ] && command -v nsys >/dev/null 2>&1; then
    PROJ_CSV="${RESULTS}/jax_${TAG}_nsys_nvtx_gpu_proj_sum.csv"
    echo "Extracting NVTX-range + kernel summaries from ${NSYS_REP} ..."
    nsys stats --report nvtx_kern_sum --format csv \
      --output - "${NSYS_REP}" > "${NVTX_CSV}" 2>&1 || true
    nsys stats --report cuda_gpu_kern_sum --format csv \
      --output - "${NSYS_REP}" > "${KERN_CSV}" 2>&1 || true
    nsys stats --report nvtx_gpu_proj_sum --format csv \
      --output - "${NSYS_REP}" > "${PROJ_CSV}" 2>&1 || true

    BREAKOUT=$(python3 - "${NVTX_CSV}" "${PROJ_CSV}" <<'PYEOF'
import csv, sys, collections, re
path = sys.argv[1]

STAGE_PATTERNS = {
    "dispatch_fwd":    re.compile(r"(^|:)dispatch_fwd(\[[^\]]*\])?$"),
    "ep_dispatch_vjp": re.compile(r"(^|:)ep_dispatch_vjp(\[[^\]]*\])?$"),
    "combine_fwd":     re.compile(r"(^|:)combine_fwd(\[[^\]]*\])?$"),
    "ep_combine_vjp":  re.compile(r"(^|:)ep_combine_vjp(\[[^\]]*\])?$"),
}
STAGE_ORDER = ("dispatch_fwd", "ep_dispatch_vjp", "combine_fwd", "ep_combine_vjp")

stages = collections.defaultdict(list)
try:
    with open(path) as f:
        lines = [ln for ln in f]
        header_idx = next((i for i, ln in enumerate(lines)
                           if ln.lstrip().startswith("NVTX Range,")), -1)
        if header_idx < 0:
            print("(NVTX header not found)"); sys.exit(0)
        reader = csv.reader(lines[header_idx:])
        header = next(reader, None)
        def col(name):
            for i, h in enumerate(header):
                if h.strip().lower() == name.lower():
                    return i
            return -1
        i_range = col("NVTX Range")
        i_total = col("Total Time (ns)")
        i_inst  = col("Kern Inst")
        i_name  = col("Kernel Name")
        if min(i_range, i_total, i_inst, i_name) < 0:
            print(f"(missing expected columns; got {header})"); sys.exit(0)
        for row in reader:
            if len(row) <= i_name: continue
            rname = row[i_range].strip()
            try:
                total_ns = int(row[i_total].replace(',', ''))
                inst = int(row[i_inst].replace(',', ''))
            except ValueError:
                continue
            kname = row[i_name].strip()
            for stage, pat in STAGE_PATTERNS.items():
                if pat.search(rname):
                    stages[stage].append((total_ns, inst, kname))
                    break
except FileNotFoundError:
    print("(nvtx_kern_sum CSV not found)"); sys.exit(0)

if not stages:
    print("(no kernels matched expected NVTX ranges)")
    sys.exit(0)

proj_csv = sys.argv[2] if len(sys.argv) > 2 else None
proj = {}
if proj_csv:
    try:
        with open(proj_csv) as f:
            plines = list(f)
        hidx = next((i for i, ln in enumerate(plines)
                     if ln.lstrip().startswith("Range,")), -1)
        if hidx >= 0:
            pr = csv.reader(plines[hidx:])
            ph = next(pr, None)
            def pcol(n):
                for i, h in enumerate(ph):
                    if h.strip().lower() == n.lower(): return i
                return -1
            pi_range = pcol("Range")
            pi_total = pcol("Total Proj Time (ns)")
            pi_inst  = pcol("Range Instances")
            pi_gpuops = pcol("Total GPU Ops")
            for row in pr:
                if len(row) <= max(pi_range, pi_total, pi_inst): continue
                rname = row[pi_range].strip()
                for stage, pat in STAGE_PATTERNS.items():
                    if pat.search(rname):
                        try:
                            t = int(row[pi_total].replace(',', ''))
                            n = int(row[pi_inst].replace(',', ''))
                            ops = int(row[pi_gpuops].replace(',', '')) if pi_gpuops >= 0 else 0
                        except ValueError:
                            continue
                        proj[stage] = (t / 1e3, n)
                        break
    except FileNotFoundError:
        pass

print("### Per-stage GPU activity (kernels + memops, from nvtx_gpu_proj_sum)")
print()
print("| stage | iters | GPU activity total (us) | per-iter (us) | kernel sum (us) | per-iter (us) | gap = memops+idle (us) |")
print("|------|-----:|----------------------:|------------:|--------------:|------------:|---------------------:|")
for stage in STAGE_ORDER:
    rows = stages.get(stage, [])
    kern_total_us = sum(r[0] for r in rows) / 1e3
    iters = max(rows, key=lambda r: r[0])[1] if rows else 0
    gpu_total_us, _ = proj.get(stage, (0.0, 0))
    per_iter_gpu = gpu_total_us / iters if iters else 0
    per_iter_kern = kern_total_us / iters if iters else 0
    gap = per_iter_gpu - per_iter_kern
    print(f"| `{stage}` | {iters} | {gpu_total_us:18.1f} | {per_iter_gpu:11.1f} | {kern_total_us:13.1f} | {per_iter_kern:11.1f} | {gap:20.1f} |")
print()

def _kern_per_iter(rows, needle):
    tot_ns = 0; inst = 0
    for tns, n, kname in rows:
        if needle in kname:
            tot_ns += tns; inst += n
    return (tot_ns / inst / 1e3) if inst else None

KEY_KERNELS = {
    "dispatch_fwd":    [("dispatch",      "nccl_ep_jit_ht_dispatch_kernel"),
                        ("permute",       "nccl_ep_jit_ht_permute_kernel")],
    "ep_dispatch_vjp": [("dispatch",      "nccl_ep_jit_ht_dispatch_kernel"),
                        ("permute",       "nccl_ep_jit_ht_permute_kernel")],
    "combine_fwd":     [("combine",       "nccl_ep_jit_ht_combine_kernel"),
                        ("local_reduce",  "nccl_ep_jit_ht_local_reduce_kernel")],
    "ep_combine_vjp":  [("combine",       "nccl_ep_jit_ht_combine_kernel"),
                        ("local_reduce",  "nccl_ep_jit_ht_local_reduce_kernel")],
}

print("### Key NCCL EP kernel time per iter (us)")
print()
print("| stage | primary kernel (us/iter) | secondary kernel (us/iter) | kernel sum/iter (us) |")
print("|------|--------------------:|-----------------------:|------------------:|")
for stage in STAGE_ORDER:
    rows = stages.get(stage, [])
    iters = max(rows, key=lambda r: r[0])[1] if rows else 0
    per_iter_kern = (sum(r[0] for r in rows) / 1e3 / iters) if iters else 0.0
    keys = KEY_KERNELS.get(stage, [])
    cells = []
    for label, needle in keys:
        v = _kern_per_iter(rows, needle)
        cells.append(f"{label}: {v:.1f}" if v is not None else f"{label}: -")
    while len(cells) < 2:
        cells.append("-")
    print(f"| `{stage}` | {cells[0]:>20} | {cells[1]:>22} | {per_iter_kern:17.1f} |")
print()

for stage in STAGE_ORDER:
    rows = stages.get(stage, [])
    if not rows:
        print(f"### Stage `{stage}` top kernels — none"); print(); continue
    agg = collections.defaultdict(lambda: [0, 0])
    for tns, inst, kname in rows:
        agg[kname][0] += tns
        agg[kname][1] += inst
    items = sorted(([k, v[0], v[1]] for k, v in agg.items()), key=lambda x: -x[1])
    total_us = sum(v[1] for v in items) / 1e3
    print(f"### Stage `{stage}` — top 20 kernels ({len(items)} distinct; kernel-sum {total_us:.1f} us)")
    print()
    print("| # | total (us) | inst | avg (us) | kernel |")
    print("|--:|-----------:|-----:|---------:|--------|")
    for i, (kname, tns, inst) in enumerate(items[:20], 1):
        avg_us = (tns / inst) / 1e3 if inst else 0
        short = kname if len(kname) <= 80 else kname[:77] + "..."
        print(f"| {i} | {tns/1e3:10.1f} | {inst:4d} | {avg_us:8.2f} | `{short}` |")
    print()
PYEOF
)
    {
      echo ""
      echo "## Kernel breakout per NVTX range (rank 0)"
      echo ""
      echo "${BREAKOUT}"
      echo "Full CSVs:"
      echo "- per-range: \`${NVTX_CSV}\`"
      echo "- overall:   \`${KERN_CSV}\`"
    } | tee -a "${RANK0_LOG}" >> "${SUMMARY}"
  fi
fi

echo "Done. Logs in ${RESULTS}/${OUT_PREFIX}_*.txt"
echo "Summary: ${SUMMARY}"
