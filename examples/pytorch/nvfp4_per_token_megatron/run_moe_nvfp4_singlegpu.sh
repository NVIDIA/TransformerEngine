#!/bin/bash

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# ============================================================================
# Single-GPU Megatron-Core *MoE* NVFP4 smoke run (bf16 vs prod vs per-token).

# Usage (run INSIDE the compute-node container, from the Megatron-LM root):
#     bash run_moe_nvfp4_singlegpu.sh bf16        # unquantized bf16 baseline
#     bash run_moe_nvfp4_singlegpu.sh prod        # production NVFP4 (per-tensor block-scaling)
#     bash run_moe_nvfp4_singlegpu.sh pertoken    # NVFP4 per-token recipe
#
# Identical model / data / seed across modes; only the GEMM precision changes due to different quant recipe.
# ============================================================================
set -euo pipefail

MODE="${1:-prod}"
[[ "$MODE" == "pertensor" ]] && MODE="prod"   # alias: pertensor == prod (NVFP4 per-tensor)

# ---------------------------------------------------------------------------
# 0. Per-container hygiene: drop image-baked flash-attn (TE c10 ABI mismatch).
# ---------------------------------------------------------------------------
pip uninstall -y flash-attn flash_attn flash_attn_2_cuda >/dev/null 2>&1 || true

# ---------------------------------------------------------------------------
# 1. Recipe selection.
# ---------------------------------------------------------------------------
if [[ "$MODE" == "pertoken" ]]; then
    export NVTE_NVFP4_PER_TOKEN=1
    # NVTE_NORM_FWD_USE_CUDNN=1 is no longer required: the norm forward now
    # auto-selects the unfused path for per-token NVFP4 (its per-row/per-col
    # outer amax is computed by the per-token cast's K1 amax kernel).
    echo "[run] MODE=pertoken  -> NVTE_NVFP4_PER_TOKEN=1"
elif [[ "$MODE" == "prod" ]]; then
    unset NVTE_NVFP4_PER_TOKEN || true
    export NVTE_NVFP4_DISABLE_RHT="${NVTE_NVFP4_DISABLE_RHT:-1}"
    export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING="${NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING:-1}"
    # 2D weight quant is ON by default for prod; +1d sets DISABLE_2D=1 (1D weights)
    # for the per-tensor 1D-vs-2D weight-quant ablation.
    export NVTE_NVFP4_DISABLE_2D_QUANTIZATION="${NVTE_NVFP4_DISABLE_2D_QUANTIZATION:-0}"
    _rht=$([[ "$NVTE_NVFP4_DISABLE_RHT" == "0" ]] && echo on || echo off)
    _sr=$([[ "$NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING" == "0" ]] && echo on || echo off)
    _2d=$([[ "$NVTE_NVFP4_DISABLE_2D_QUANTIZATION" == "0" ]] && echo 2D || echo 1D)
    echo "[run] MODE=prod      -> NVFP4 per-tensor block-scaling (RHT=$_rht SR=$_sr, weight=$_2d)"
elif [[ "$MODE" == "bf16" ]]; then
    unset NVTE_NVFP4_PER_TOKEN || true
    echo "[run] MODE=bf16      -> unquantized bf16 baseline"
else
    echo "[run] ERROR: unknown MODE '$MODE' (expected 'bf16', 'prod' or 'pertoken')" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Locate the Megatron-LM root (where pretrain_gpt.py + the megatron package live)
# so this script works whether it sits inside Megatron-LM/ or one level up next
# to a Megatron-LM/ checkout (override with MLM_DIR=/path/to/Megatron-LM).
if [[ -n "${MLM_DIR:-}" ]]; then
    :
elif [[ -f "${SCRIPT_DIR}/pretrain_gpt.py" ]]; then
    MLM_DIR="$SCRIPT_DIR"
elif [[ -f "${SCRIPT_DIR}/Megatron-LM/pretrain_gpt.py" ]]; then
    MLM_DIR="${SCRIPT_DIR}/Megatron-LM"
else
    echo "[run] ERROR: cannot find pretrain_gpt.py under '$SCRIPT_DIR' or '$SCRIPT_DIR/Megatron-LM'." >&2
    echo "[run]        Set MLM_DIR=/path/to/Megatron-LM and re-run." >&2
    exit 1
fi
echo "[run] Megatron-LM root: $MLM_DIR"
cd "$MLM_DIR"
export PYTHONPATH="${MLM_DIR}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export NCCL_NVLS_ENABLE=0

WANDB_PROJECT="${WANDB_PROJECT:-NVFP4 per-token recipe MoE}"
if [[ "$MODE" == "pertoken" ]]; then
    WANDB_EXP_NAME="nvfp4-pertoken"
    # Append the active per-token recipe knobs so each variant gets its OWN
    # work dir + wandb exp name (e.g. nvfp4-pertoken-rht-sr). Keep rht before sr.
    [[ "${NVTE_NVFP4_PER_TOKEN_RHT:-0}" == "1" ]] && WANDB_EXP_NAME+="-rht"
    [[ "${NVTE_NVFP4_PER_TOKEN_SR:-0}"  == "1" ]] && WANDB_EXP_NAME+="-sr"
    [[ "${NVTE_NVFP4_PER_TOKEN_WEIGHT_2D:-0}" == "1" ]] && WANDB_EXP_NAME+="-2d"
elif [[ "$MODE" == "prod" ]]; then
    WANDB_EXP_NAME="nvfp4-pertensor"
    # Additive suffixes: feature ON iff its DISABLE env is 0 (mirrors wrapper).
    [[ "${NVTE_NVFP4_DISABLE_RHT:-1}" == "0" ]] && WANDB_EXP_NAME+="-rht"
    [[ "${NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING:-1}" == "0" ]] && WANDB_EXP_NAME+="-sr"
    [[ "${NVTE_NVFP4_DISABLE_2D_QUANTIZATION:-0}" == "1" ]] && WANDB_EXP_NAME+="-1d"
else
    WANDB_EXP_NAME="bf16"
fi
# +fb: keep the first/last few transformer layers in bf16 (only meaningful for
# the quantized modes; no-op suffix for bf16).
[[ "${FIRST_LAST_BF16:-0}" == "1" && "$MODE" != "bf16" ]] && WANDB_EXP_NAME+="-fb"

WORK_ROOT="${WORK_ROOT:-${SCRIPT_DIR}/work}"

RUN_NAME="${RUN_NAME:-${WANDB_EXP_NAME}${RUN_TAG:+-${RUN_TAG}}-seed${SEED:-1234}}"

WANDB_EXP_NAME="${WANDB_EXP_PREFIX-}${RUN_NAME}"
RUN_DIR="${WORK_ROOT}/${RUN_NAME}"
WANDB_SAVE_DIR="${WANDB_DIR:-${RUN_DIR}/wandb}"
TB_DIR="${TB_DIR:-${RUN_DIR}/tb}"
CKPT_DIR="${CKPT_DIR:-${RUN_DIR}/checkpoints}"
mkdir -p "$WANDB_SAVE_DIR" "$TB_DIR" "$CKPT_DIR"
echo "[run] work dir: $RUN_DIR  (ckpt=$CKPT_DIR)"
echo "[run] wandb: project='$WANDB_PROJECT' exp=$WANDB_EXP_NAME (WANDB_MODE=${WANDB_MODE:-online})"

LOG_ARGS=(
    --wandb-project "$WANDB_PROJECT"
    --wandb-exp-name "$WANDB_EXP_NAME"
    --wandb-save-dir "$WANDB_SAVE_DIR"
    # Disable OneLogger: with enable_one_logger=True it spins up a SECOND wandb
    # run (project=one_logger_project="megatron-lm", auto name like "woven-leaf"),
    # which becomes the active wandb.run and steals wandb.log() metrics from our
    # named run -> loss never shows in our named run. Off => single wandb run.
    --no-one-logger
    --tensorboard-dir "$TB_DIR"
    --tensorboard-log-interval 10
    --log-interval 10
    --log-num-zeros-in-grad
    --eval-iters 0
    --eval-interval 100000
    # Checkpointing: stable per-variant dir, same path for save+load so each
    # resubmission auto-resumes this variant.
    #   SAVE_INTERVAL        how often to write a ckpt (def 2000).
    #   SAVE_RETAIN_INTERVAL rolling cleanup: mcore keeps the LATEST ckpt always
    --save "$CKPT_DIR"
    --load "$CKPT_DIR"
    --save-interval "${SAVE_INTERVAL:-2000}"
    --save-retain-interval "${SAVE_RETAIN_INTERVAL:-10000}"
)

# ---------------------------------------------------------------------------
# Model: Ling-mini-v2 MoE block. All GEMM dims %128 (per-token alignment):
#    hidden=2048, ffn=5120, moe-ffn=512, shared-expert=512, seq=4096.
#    Routing: 64 experts / 8 groups (8 each), group_topk 4, topk 8
#    (8 <= group_topk*experts_per_group = 4*8 = 32). Reduced from Ling-mini-v2's
#    256 experts because EP=1 on one GPU exceeds TE's 64-tensor grouped cap.
#    Layer 0 dense, layers 1-19 MoE (moe-layer-freq); num-layers 20 = full
#    Ling-mini-v2 depth (fits one GB200 now that experts are down to 64).
# ---------------------------------------------------------------------------
MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl transformer_engine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --swiglu
    --position-embedding-type rope
    --no-rope-fusion
    --rotary-base 10000
    --rotary-percent 0.5
    --rotary-scaling-factor 40
    --normalization RMSNorm
    --norm-epsilon 1e-6
    --group-query-attention
    --num-attention-heads 16
    --num-query-groups 4
    --qk-layernorm
    --hidden-dropout 0
    --attention-dropout 0
    --num-layers 20
    --hidden-size 2048
    --ffn-hidden-size 5120
    --seq-length 4096
    --max-position-embeddings 4096
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
)

# MoE block based on Ling-mini-v2 (topk8 / 8 groups / group_topk4 / moe-ffn 512 /
# shared 512). num-layers 20 = full Ling-mini-v2 depth (fits one GB200 now that
# experts are down to 64). num-experts (64 vs 256) reduced because EP=1 on one
# GPU puts all experts local and TE's grouped-NVFP4 kernels cap at 64
# tensors/launch (real training shards via EP>1 so local experts <=64).
# 8 groups * 8 experts/group = 64.
MOE_ARGS=(
    --num-experts 64
    --moe-layer-freq "[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]"
    --moe-ffn-hidden-size 512
    --moe-shared-expert-intermediate-size 512
    --moe-grouped-gemm
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.001
    --moe-z-loss-coeff 0.0000035
    --moe-router-topk 8
    --moe-router-topk-scaling-factor 2.5
    --moe-router-num-groups 8
    --moe-router-group-topk 4
    --moe-router-dtype fp32
    --moe-router-score-function sigmoid
    --moe-router-enable-expert-bias
    --moe-router-bias-update-rate 1e-3
    --moe-token-dispatcher-type alltoall
    --moe-router-fusion
    --moe-permute-fusion
)

# ---------------------------------------------------------------------------
# Precision recipe (the part under test).
#    --no-gradient-accumulation-fusion MANDATORY for per-token (accumulate=True
#    unsupported); kept in all modes so only GEMM precision differs.
# ---------------------------------------------------------------------------
QUANT_ARGS=(
    --bf16
    --no-gradient-accumulation-fusion
)
if [[ "$MODE" != "bf16" ]]; then
    QUANT_ARGS+=(--fp4-format e2m1)
fi
# +fb variant: keep the first/last few transformer layers in bf16 (skip NVFP4)
# via Megatron's --first-last-layers-bf16. Gated by FIRST_LAST_BF16=1 (set by the
# sbatch wrapper for a +fb spec). Layer counts overridable; defaults 0 start / 3 end.
if [[ "${FIRST_LAST_BF16:-0}" == "1" && "$MODE" != "bf16" ]]; then
    QUANT_ARGS+=(
        --first-last-layers-bf16
        --num-layers-at-start-in-bf16 "${NUM_LAYERS_START_BF16:-0}"
        --num-layers-at-end-in-bf16 "${NUM_LAYERS_END_BF16:-3}"
    )
    echo "[run] FIRST_LAST_BF16=1 -> first ${NUM_LAYERS_START_BF16:-0} / last ${NUM_LAYERS_END_BF16:-3} layers in bf16"
fi

# ---------------------------------------------------------------------------
# Parallelism: single GPU. EP/TP/PP/CP=1. No sequence-parallel (needs TP>1),
#    no expert/comm overlap.
# ---------------------------------------------------------------------------
PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-distributed-optimizer
)

# ---------------------------------------------------------------------------
# Data / tokenizer.  DATA_MODE=real (default) uses the downloaded OLMo-1124
#    algebraic_stack corpus; DATA_MODE=mock falls back to synthetic data (no
#    files, loss won't converge -- only for a pure "does it run" smoke test).
#    Override DATA_PATH / TOKENIZER_MODEL via env if you use a different corpus.
# ---------------------------------------------------------------------------
DATA_MODE="${DATA_MODE:-real}"
DATA_PATH="${DATA_PATH:-/path/to/datasets/olmo-1124/algebraic_stack_text_document}"
# Tokenizer that ORIGINALLY preprocessed this corpus = Moonlight-16B-A3B (custom
# tiktoken tokenizer -> needs --trust-remote-code). Download tokenizer-only via:
#   hf download moonshotai/Moonlight-16B-A3B --include "tiktoken.model" \
#       "tokenization_moonshot.py" "tokenizer_config.json" --local-dir <TOKENIZER_MODEL>
TOKENIZER_MODEL="${TOKENIZER_MODEL:-/path/to/tokenizers/Moonlight-16B-A3B}"
if [[ "$DATA_MODE" == "mock" ]]; then
    echo "[run] DATA_MODE=mock  -> synthetic data (NullTokenizer, no convergence)"
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 32000
        --make-vocab-size-divisible-by 128
    )
else
    echo "[run] DATA_MODE=real  -> $DATA_PATH (tokenizer=$TOKENIZER_MODEL)"
    DATA_ARGS=(
        --data-path "$DATA_PATH"
        --split 99,1,0
        --tokenizer-type HuggingFaceTokenizer
        --tokenizer-model "$TOKENIZER_MODEL"
        --trust-remote-code
        --make-vocab-size-divisible-by 128
    )
fi

# ---------------------------------------------------------------------------
# Schedule: gbs/mbs = 8 microbatches/step. Steps via TRAIN_ITERS (def 20000).
# ---------------------------------------------------------------------------
TRAIN_ITERS="${TRAIN_ITERS:-20000}"
LR_WARMUP_ITERS=$(( TRAIN_ITERS * 3 / 100 ))
SEED="${SEED:-1234}"
export WANDB_TAGS="seed=${SEED}${WANDB_TAGS:+,${WANDB_TAGS}}"
echo "[run] wandb tags: $WANDB_TAGS"
# torchrun rendezvous port: must be UNIQUE per concurrent run on the same node
# (running bf16/prod/pertoken in parallel on different GPUs needs different
# ports, else they collide on the default 29502). Overridable via MASTER_PORT.
MASTER_PORT="${MASTER_PORT:-29502}"
TRAIN_ARGS=(
    --seed "$SEED"
    --micro-batch-size 1
    --global-batch-size 8
    --train-iters "$TRAIN_ITERS"
    --lr 1e-4
    --min-lr 1e-5
    --lr-decay-style cosine
    --lr-warmup-iters "$LR_WARMUP_ITERS"
    --lr-decay-iters "$TRAIN_ITERS"
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --init-method-std 0.02
    --attention-backend auto
)

# Wall-clock guard for SLURM job chaining: when set, mcore saves a checkpoint
# and exits cleanly once training has run EXIT_DURATION_MIN minutes (counted from
# train start, NOT job start -> leave headroom for container startup + final
# save under the #SBATCH --time wall). The next (dependent) job resumes from the
# ckpt. Empty/0 -> disabled (run until --train-iters). See submit_chain.sh.
EXIT_ARGS=()
if [[ -n "${EXIT_DURATION_MIN:-}" && "${EXIT_DURATION_MIN}" != "0" ]]; then
    EXIT_ARGS+=(--exit-duration-in-mins "$EXIT_DURATION_MIN")
    echo "[run] exit-duration-in-mins=$EXIT_DURATION_MIN (will save+exit, resume via --load on next job)"
fi

echo "[run] launching pretrain_gpt.py MoE ($MODE) on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>} port=$MASTER_PORT ..."
set -x
torchrun --nproc_per_node=1 --nnodes=1 \
    --master_addr=127.0.0.1 --master_port="$MASTER_PORT" \
    pretrain_gpt.py \
    "${MODEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${QUANT_ARGS[@]}" \
    "${PARALLEL_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${EXIT_ARGS[@]}" \
    "${LOG_ARGS[@]}"
