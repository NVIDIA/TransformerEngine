#!/bin/bash

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# ============================================================================
# sbatch wrapper for the single-GPU MoE NVFP4 smoke run.
# Submits a batch job that: launches the container, applies per-container
# hygiene (flash-attn / huggingface-hub), (re)registers the editable TE build
# from the workspace, then runs run_moe_nvfp4_singlegpu.sh -- so you never have
# to srun --pty into the node by hand.
#
# Usage:
#     sbatch sbatch_moe_nvfp4_singlegpu.sh            # runs ALL 3 modes sequentially (1 GPU)
#     sbatch sbatch_moe_nvfp4_singlegpu.sh bf16       # just bf16
#     sbatch sbatch_moe_nvfp4_singlegpu.sh prod
#     sbatch sbatch_moe_nvfp4_singlegpu.sh pertoken
# Pick EXACTLY which variants run concurrently (one GPU each) with a comma list.
# QOS forces a full 4-GPU node per job, so packing up to 4 variants wastes nothing:
#     sbatch sbatch_moe_nvfp4_singlegpu.sh "bf16,pertensor+rht+sr,pertoken"
#     sbatch sbatch_moe_nvfp4_singlegpu.sh "pertensor,pertensor+rht,pertensor+sr,pertensor+rht+sr"
#     sbatch sbatch_moe_nvfp4_singlegpu.sh "pertoken,pertoken+sr,pertoken+rht,pertoken+rht+sr"
#   spec syntax: <mode>[+rht][+sr][+1d][+2d][+fb]  where mode = bf16 | prod(==pertensor) | pertoken
#     +fb -> keep first/last layers in bf16 (Megatron --first-last-layers-bf16;
#            defaults: 0 at start, 3 at end; override via NUM_LAYERS_START_BF16 /
#            NUM_LAYERS_END_BF16). e.g. "pertoken+2d+fb", "pertensor+rht+sr+fb".
#   ADDITIVE RHT/SR: +rht/+sr turn the feature ON; bare = OFF for BOTH paths.
#     -> bare `pertensor`/`prod` = block-scaling only (NO RHT/SR);
#     -> production-default per-tensor = `pertensor+rht+sr`;
#     -> per-token default (no RHT/SR) = `pertoken`.
#   each spec gets GPU i and port BASE_PORT+i; <=4 specs (one per GPU).
# Convenience aliases:
#     parallel  == "bf16,pertensor+rht+sr,pertoken"
#     parallel4 == "bf16,pertensor+rht+sr,pertoken,pertoken+sr"
#
# Pin a single mode to a specific GPU with GPU_ID:
#     sbatch --export=ALL,GPU_ID=2 sbatch_moe_nvfp4_singlegpu.sh pertoken
#
# Override knobs via --export, e.g.:
#     sbatch --export=ALL,TRAIN_ITERS=2000 sbatch_moe_nvfp4_singlegpu.sh pertoken
#     sbatch --export=ALL,SKIP_BUILD=1     sbatch_moe_nvfp4_singlegpu.sh prod
#     sbatch --export=ALL,SEED=42          sbatch_moe_nvfp4_singlegpu.sh prod
#     sbatch --export=ALL,PT_SR=1,PT_RHT=1 sbatch_moe_nvfp4_singlegpu.sh pertoken
# ============================================================================
#SBATCH -N 1
#SBATCH -p batch
#SBATCH -q short
#SBATCH -A your_slurm_account          # EDIT: your Slurm account
#SBATCH --gres=gpu:4
#SBATCH --time=2:00:00
#SBATCH -J nvfp4-moe-singlegpu
# NOTE: #SBATCH --output/--error are resolved by Slurm on the HOST filesystem
# (the batch body runs on the host, outside the container), so these MUST be the
# real host path, NOT the in-container mount target. EDIT to a writable host dir.
#SBATCH --output=/path/to/TransformerEngine/examples/pytorch/nvfp4_per_token_megatron/slurm_logs/nvfp4-moe-%j.out
#SBATCH --error=/path/to/TransformerEngine/examples/pytorch/nvfp4_per_token_megatron/slurm_logs/nvfp4-moe-%j.err

set -euo pipefail

IMAGE="${IMAGE:-/path/to/te_container_image.sqsh}"
HOST_MOUNT="${HOST_MOUNT:-/path/to/host/workspace:/workspace}"   # host:container
TE_DIR="${TE_DIR:-/workspace/TransformerEngine}"                 # in-container path
EXAMPLE_DIR="${EXAMPLE_DIR:-${TE_DIR}/examples/pytorch/nvfp4_per_token_megatron}"
RUN_SCRIPT="${EXAMPLE_DIR}/run_moe_nvfp4_singlegpu.sh"
# HOST_LOG_DIR is the HOST path matching the #SBATCH --output/--error dir above.
HOST_LOG_DIR="${HOST_LOG_DIR:-/path/to/TransformerEngine/examples/pytorch/nvfp4_per_token_megatron/slurm_logs}"

# slurm_logs must exist on the HOST before the job writes there.
mkdir -p "$HOST_LOG_DIR"

# --- Knobs (overridable via sbatch --export) -------------------------------
export RUN_MODE="${1:-all}"            # single mode | all | "a,b,c" list | parallel | parallel4
export SKIP_BUILD="${SKIP_BUILD:-0}"   # 1 = skip the pip install -e . step
export TRAIN_ITERS="${TRAIN_ITERS:-20000}"
export SEED="${SEED:-1234}"            # same seed across modes for a fair compare
export GPU_ID="${GPU_ID:-}"            # pin a single-mode run to this GPU (e.g. 2)
export BASE_PORT="${BASE_PORT:-29502}" # torchrun rendezvous base port (parallel uses +0/+1/+2)
export RUN_TAG="${RUN_TAG:-}"          # optional suffix on the run name (e.g. an experiment label)
export SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"  # checkpoint every N iters (stable per-variant dir)
export SAVE_RETAIN_INTERVAL="${SAVE_RETAIN_INTERVAL:-10000}"
export EXIT_DURATION_MIN="${EXIT_DURATION_MIN:-110}"
export PT_SR="${PT_SR:-0}"             # 1 = NVTE_NVFP4_PER_TOKEN_SR=1 (pertoken only)
export PT_RHT="${PT_RHT:-0}"           # 1 = NVTE_NVFP4_PER_TOKEN_RHT=1 (pertoken only)
# Carry the in-container paths into the container env (--export=ALL below).
export TE_DIR EXAMPLE_DIR RUN_SCRIPT

echo "[sbatch] job=$SLURM_JOB_ID node=$(hostname) mode=$RUN_MODE iters=$TRAIN_ITERS seed=$SEED save_interval=$SAVE_INTERVAL retain_interval=$SAVE_RETAIN_INTERVAL exit_min=$EXIT_DURATION_MIN gpu_id=${GPU_ID:-<auto>} skip_build=$SKIP_BUILD"

# --- Run everything inside the container -----------------------------------
srun --container-image="$IMAGE" \
     --container-writable \
     --container-mounts="$HOST_MOUNT" \
     --container-remap-root \
     --container-workdir="/workspace" \
     --export=ALL \
     bash <<'EOF'
set -euo pipefail

# TE_DIR / EXAMPLE_DIR / RUN_SCRIPT are inherited from the host via --export=ALL.
: "${TE_DIR:?TE_DIR not set -- edit the host-side config block}"
EXAMPLE_DIR="${EXAMPLE_DIR:-${TE_DIR}/examples/pytorch/nvfp4_per_token_megatron}"
RUN_SCRIPT="${RUN_SCRIPT:-${EXAMPLE_DIR}/run_moe_nvfp4_singlegpu.sh}"

# 1. Per-container hygiene -----------------------------------------------------
#    (a) flash-attn ABI mismatch -> remove it (TE skips the FA backend cleanly)
pip uninstall -y flash-attn flash_attn flash_attn_2_cuda >/dev/null 2>&1 || true
#    (b) transformers needs huggingface-hub <1.0 (image often ships 1.x)
python - <<'PY' 2>/dev/null || pip install -q "huggingface_hub>=0.34,<1.0"
import sys
from importlib.metadata import version
from packaging.version import parse
sys.exit(0 if parse(version("huggingface_hub")) < parse("1.0") else 1)
PY

# 2. (Re)register the editable TE build from the workspace --------------------
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo "[job] (re)installing editable TE from $TE_DIR ..."
    cd "$TE_DIR"
    NVTE_CUDA_ARCHS=100a NVTE_BUILD_THREADS_PER_JOB=8 NVTE_FRAMEWORK=pytorch \
        pip install -e . --no-build-isolation 2>&1 | tee "build_sbatch_${SLURM_JOB_ID}.log"
else
    echo "[job] SKIP_BUILD=1 -> assuming TE already registered"
fi

# Sanity: TE must import AND expose a per-token symbol 
python - <<'PY'
import transformer_engine            # libtransformer_engine.so first
import transformer_engine_torch as tex
assert hasattr(tex, "nvfp4_per_token_quantize") or hasattr(tex, "nvfp4_cutlass_per_token_gemm"), \
    "per-token TE symbols missing -> wrong/old TE is active"
print("[job] TE per-token symbols present: OK")
PY

# 3. Run the experiment(s) ---------------------------------------------------

WORK_ROOT="${WORK_ROOT:-${EXAMPLE_DIR}/work}"

# Map (mode, sr, rht, oned) -> run name (mirrors run_moe_nvfp4_singlegpu.sh exp
# names) so the wrapper's console log lands in the SAME per-run dir as wandb/tb.
#   exp_name <mode> [sr] [rht] [oned]
# oned=1 (prod/pertensor only) forces 1D weight quant (disables the default 2D)
# for the per-tensor-1d-vs-2d weight-quant ablation; named with a -1d suffix.
exp_name () {
    local mode="$1" sr="${2:-${PT_SR:-0}}" rht="${3:-${PT_RHT:-0}}" oned="${4:-0}" twod="${5:-0}" fb="${6:-0}" base
    case "$mode" in
        pertoken)       base="nvfp4-pertoken" ;;
        prod|pertensor) base="nvfp4-pertensor" ;;
        *)              echo "bf16"; return ;;
    esac
    [[ "$rht" == "1" ]] && base+="-rht"
    [[ "$sr"  == "1" ]] && base+="-sr"
    # -1d only meaningful for per-tensor (per-token weights are already 1D).
    [[ "$oned" == "1" && "$mode" != "pertoken" ]] && base+="-1d"
    # -2d only meaningful for per-token (Route A: 2D weight quant via
    # NVTE_NVFP4_PER_TOKEN_WEIGHT_2D). No-op on per-tensor (already 2D).
    [[ "$twod" == "1" && "$mode" == "pertoken" ]] && base+="-2d"
    # -fb: keep first/last layers in bf16 (Megatron --first-last-layers-bf16).
    [[ "$fb" == "1" ]] && base+="-fb"
    echo "$base"
}

# Emit the env-var assignments that realize (sr,rht,oned) for a given mode.
#   mode_env <mode> <sr> <rht> [oned]
mode_env () {
    local mode="$1" sr="$2" rht="$3" oned="${4:-0}" twod="${5:-0}"
    case "$mode" in
        pertoken)
            # twod (Route A) opts the forward WEIGHT into prod 2D block scaling
            # (16x16 inner + scalar outer) while activations/gradients stay
            # per-token 1D. Safe to combine with rht/sr: those only touch
            # act/grad, the fwd weight is always no-rht/no-sr.
            echo "NVTE_NVFP4_PER_TOKEN_RHT=$rht NVTE_NVFP4_PER_TOKEN_SR=$sr NVTE_NVFP4_PER_TOKEN_WEIGHT_2D=$twod" ;;
        prod|pertensor)
            # disable=1 when the feature is OFF (additive: feature ON iff flag==1).
            # 2D weight quant is ON by default for prod; +1d disables it (1D weights).
            echo "NVTE_NVFP4_DISABLE_RHT=$(( rht == 1 ? 0 : 1 )) NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=$(( sr == 1 ? 0 : 1 )) NVTE_NVFP4_DISABLE_2D_QUANTIZATION=$(( oned == 1 ? 1 : 0 ))" ;;
        *) echo "" ;;
    esac
}
#   run_dir_for <mode> [sr] [rht] [oned] [twod] [fb]  ->  work/<run name>-seed<N>
#   (stable, no timestamp; seed in the dir name so different seeds never collide)
run_dir_for () { echo "${WORK_ROOT}/$(exp_name "$1" "${2:-}" "${3:-}" "${4:-}" "${5:-}" "${6:-}")${RUN_TAG:+-${RUN_TAG}}-seed${SEED}"; }

# Foreground single run (optionally pinned to GPU_ID). Console log -> work dir.
run_one () {
    local spec="$1" mode sr rht oned twod fb
    # Parse +rht/+sr/+1d/+2d/+fb suffixes so a single spec works the same as in
    # launch_list (e.g. "pertoken+rht", "pertensor+1d", "pertoken+2d+fb"). Fall
    # back to PT_SR/PT_RHT when no suffix is given.
    mode="${spec%%+*}"
    sr="${PT_SR:-0}"; rht="${PT_RHT:-0}"; oned=0; twod=0; fb=0
    [[ "$spec" == *"+sr"*  ]] && sr=1
    [[ "$spec" == *"+rht"* ]] && rht=1
    [[ "$spec" == *"+1d"*  ]] && oned=1
    [[ "$spec" == *"+2d"*  ]] && twod=1
    [[ "$spec" == *"+fb"*  ]] && fb=1
    [[ "$mode" == "pertensor" ]] && mode="prod"
    case "$mode" in
        bf16|prod|pertoken) ;;
        *) echo "[job] ERROR: bad mode '$mode' in spec '$spec' (want bf16|prod|pertoken[+rht][+sr][+1d][+2d][+fb])" >&2; exit 1 ;;
    esac
    local run_name; run_name="$(exp_name "$mode" "$sr" "$rht" "$oned" "$twod" "$fb")${RUN_TAG:+-${RUN_TAG}}-seed${SEED}"
    local run_dir="${WORK_ROOT}/${run_name}"
    mkdir -p "$run_dir"
    echo "============================================================"
    echo "[job] === running mode=$mode sr=$sr rht=$rht oned=$oned twod=$twod fb=$fb (iters=$TRAIN_ITERS, GPU=${GPU_ID:-<auto>}) -> $run_dir ==="
    echo "============================================================"
    local gpu_env=()
    [[ -n "${GPU_ID:-}" ]] && gpu_env=(CUDA_VISIBLE_DEVICES="$GPU_ID")
    env "${gpu_env[@]}" TRAIN_ITERS="$TRAIN_ITERS" SEED="$SEED" MASTER_PORT="$BASE_PORT" \
        SAVE_INTERVAL="$SAVE_INTERVAL" SAVE_RETAIN_INTERVAL="$SAVE_RETAIN_INTERVAL" \
        EXIT_DURATION_MIN="$EXIT_DURATION_MIN" \
        $(mode_env "$mode" "$sr" "$rht" "$oned" "$twod") \
        FIRST_LAST_BF16="$fb" \
        WORK_ROOT="$WORK_ROOT" RUN_NAME="$run_name" \
        bash "$RUN_SCRIPT" "$mode" 2>&1 | tee "${run_dir}/console.log"
}

# Background run pinned to a specific GPU + port; everything in its own work dir.
# Per-run SR/RHT (args 4/5) override the global PT_SR/PT_RHT so one job can run
# several pertoken variants side by side.
#   run_bg <mode> <gpu> <port> [sr] [rht] [oned] [twod] [fb]
run_bg () {
    local mode="$1" gpu="$2" port="$3" sr="${4:-${PT_SR:-0}}" rht="${5:-${PT_RHT:-0}}" oned="${6:-0}" twod="${7:-0}" fb="${8:-0}"
    local run_name; run_name="$(exp_name "$mode" "$sr" "$rht" "$oned" "$twod" "$fb")${RUN_TAG:+-${RUN_TAG}}-seed${SEED}"
    local run_dir="${WORK_ROOT}/${run_name}"
    mkdir -p "$run_dir"
    echo "[job] launch mode=$mode sr=$sr rht=$rht oned=$oned twod=$twod fb=$fb on GPU $gpu (port $port) -> $run_dir/console.log"
    env CUDA_VISIBLE_DEVICES="$gpu" TRAIN_ITERS="$TRAIN_ITERS" SEED="$SEED" MASTER_PORT="$port" \
        SAVE_INTERVAL="$SAVE_INTERVAL" SAVE_RETAIN_INTERVAL="$SAVE_RETAIN_INTERVAL" \
        EXIT_DURATION_MIN="$EXIT_DURATION_MIN" \
        $(mode_env "$mode" "$sr" "$rht" "$oned" "$twod") \
        FIRST_LAST_BF16="$fb" \
        WORK_ROOT="$WORK_ROOT" RUN_NAME="$run_name" \
        bash "$RUN_SCRIPT" "$mode" >"${run_dir}/console.log" 2>&1 &
}

N_GPUS="${N_GPUS:-4}"   # GPUs available on the node (QOS gives a full 4-GPU node)

# Launch a list of run specs concurrently, one GPU each.
# Each spec: <mode>[+rht][+sr]  e.g. bf16, pertensor+rht+sr, pertoken, pertoken+sr
launch_list () {
    local specs=("$@")
    local n=${#specs[@]}
    if (( n > N_GPUS )); then
        echo "[job] ERROR: requested $n runs but only $N_GPUS GPUs available" >&2
        echo "[job]        specs: ${specs[*]}" >&2
        exit 1
    fi
    local i=0 spec mode sr rht oned twod fb
    for spec in "${specs[@]}"; do
        mode="${spec%%+*}"; sr=0; rht=0; oned=0; twod=0; fb=0
        [[ "$spec" == *"+sr"*  ]] && sr=1
        [[ "$spec" == *"+rht"* ]] && rht=1
        [[ "$spec" == *"+1d"*  ]] && oned=1
        [[ "$spec" == *"+2d"*  ]] && twod=1
        [[ "$spec" == *"+fb"*  ]] && fb=1
        [[ "$mode" == "pertensor" ]] && mode="prod"   # alias
        case "$mode" in
            bf16|prod|pertoken) ;;
            *) echo "[job] ERROR: bad mode '$mode' in spec '$spec' (want bf16|prod|pertoken[+rht][+sr][+1d][+2d][+fb])" >&2; exit 1 ;;
        esac
        run_bg "$mode" "$i" "$((BASE_PORT + i))" "$sr" "$rht" "$oned" "$twod" "$fb"
        i=$((i + 1))
    done
    local rc=0 pid
    for pid in $(jobs -p); do wait "$pid" || rc=1; done
    echo "[job] runs finished (rc=$rc). Per-run dirs (console.log + wandb/ + tb/):"
    for spec in "${specs[@]}"; do
        mode="${spec%%+*}"; sr=0; rht=0; oned=0; twod=0; fb=0
        [[ "$spec" == *"+sr"*  ]] && sr=1
        [[ "$spec" == *"+rht"* ]] && rht=1
        [[ "$spec" == *"+1d"*  ]] && oned=1
        [[ "$spec" == *"+2d"*  ]] && twod=1
        [[ "$spec" == *"+fb"*  ]] && fb=1
        echo "         $(run_dir_for "$mode" "$sr" "$rht" "$oned" "$twod" "$fb")"
    done
    exit "$rc"
}

# Aliases expand to a spec list; a comma-separated RUN_MODE is taken verbatim.
case "$RUN_MODE" in
    parallel)  RUN_MODE="bf16,pertensor+rht+sr,pertoken" ;;
    parallel4) RUN_MODE="bf16,pertensor+rht+sr,pertoken,pertoken+sr" ;;
esac

if [[ "$RUN_MODE" == *","* ]]; then
    IFS=',' read -ra specs <<< "$RUN_MODE"
    echo "[job] concurrent runs (GPU 0..$((${#specs[@]} - 1))): ${specs[*]}"
    launch_list "${specs[@]}"
elif [[ "$RUN_MODE" == "all" ]]; then
    for m in bf16 prod pertoken; do run_one "$m"; done
else
    run_one "$RUN_MODE"
fi

echo "[job] done."
EOF

echo "[sbatch] job $SLURM_JOB_ID finished."
