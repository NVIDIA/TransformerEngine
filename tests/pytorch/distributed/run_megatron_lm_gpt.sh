#!/bin/bash

# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# This script allows flexibly running various sizes of
# GPT3 models with named hyperparameters.

# Trick to get kwargs.
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done

# Set defaults for all arguments.
: ${DP_SIZE:="1"}
: ${TP_SIZE:="1"}
: ${PP_SIZE:="1"}
: ${NUM_LAYERS:="12"}
: ${HIDDEN_SIZE:="768"}
: ${NHEADS:="12"}
: ${SEQLEN:="2048"}
: ${MAX_POSITION_EMBEDDINGS:="2048"}
: ${MBS:="8"}
: ${GBS:="32"}
: ${STEPS:="400"}
: ${LR:="6.0e-4"}
: ${MIN_LR:="6.0e-5"}
: ${SAVE_INTERVAL:="1000"}
: ${SPLIT:="98,2,0"}
: ${CLIP_GRAD:="1.0"}
: ${WEIGHT_DECAY:="0.1"}
: ${ADAM_BETA1:="0.9"}
: ${ADAM_BETA2:="0.95"}
: ${INIT_METHOD_STD:="0.023"}
: ${SP:="False"}
: ${DTYPE:="bf16"}
: ${WGRAD_FUSION:="True"}
: ${FP8:="False"}
: ${FP8_AMAX_HISTORY_LEN:="32"}
: ${TRANSFORMER_IMPL:="transformer_engine"}
: ${FILENAME:="log.txt"}

# Logging.
DIR=`pwd`
TENSORBOARD_DIR="${DIR}/tensorboard"
CHECKPOINT_DIR="${DIR}/checkpoints"
mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${CHECKPOINT_DIR}

# Dataset.
. /data/gpt3/pile-cc1-cc2-shuf/gpt3_blend.sh

# Set GP3 options.
options=" \
    --exit-duration-in-mins 230 \
    --tensor-model-parallel-size ${TP_SIZE} \
    --pipeline-model-parallel-size ${PP_SIZE} \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --num-attention-heads ${NHEADS} \
    --seq-length ${SEQLEN} \
    --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters ${STEPS} \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --log-interval 1 \
    --eval-iters 50 \
    --eval-interval 2000 \
    --data-path ${DATA_BLEND} \
    --vocab-file /data/gpt3/pile-cc1-cc2-shuf/bpe/gpt2-vocab.json \
    --merge-file /data/gpt3/pile-cc1-cc2-shuf/bpe/gpt2-merges.txt \
    --save-interval ${SAVE_INTERVAL} \
    --save ${CHECKPOINT_DIR} \
    --split ${SPLIT} \
    --clip-grad ${CLIP_GRAD} \
    --weight-decay ${WEIGHT_DECAY} \
    --adam-beta1 ${ADAM_BETA1} \
    --adam-beta2 ${ADAM_BETA2} \
    --init-method-std ${INIT_METHOD_STD} \
    --log-params-norm \
    --log-num-zeros-in-grad \
    --transformer-impl ${TRANSFORMER_IMPL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --fp8-margin 0 \
    --fp8-interval 1 \
    --fp8-amax-history-len ${FP8_AMAX_HISTORY_LEN} \
    --fp8-amax-compute-algo max"

if [[ "$SP" == "True" ]]; then
        options+=" --sequence-parallel"
fi

if [[ "$WGRAD_FUSION" == "False" ]]; then
        options+=" --no-gradient-accumulation-fusion"
fi

if [[ "$FP8" != "False" ]]; then
        options+=" --fp8-format ${FP8}"
fi

if [[ "$DTYPE" != "fp32" ]]; then
        options+=" --${DTYPE}"
fi

# Run GPT3.
NUM_GPUS=$((${DP_SIZE}*${TP_SIZE}*${PP_SIZE}))
NVTE_TORCH_COMPILE=0 NVTE_ALLOW_NONDETERMINISTIC_ALGO=0 NVTE_FLASH_ATTN=1 NVTE_FWD_LAYERNORM_SM_MARGIN=0 NVTE_BWD_LAYERNORM_SM_MARGIN=0 CUDA_DEVICE_MAX_CONNECTIONS=1 NVTE_BIAS_GELU_NVFUSION=0 NVTE_BIAS_DROPOUT_FUSION=0 python -m torch.distributed.launch --use_env --nnodes=1 --nproc_per_node=${NUM_GPUS} ${DIR}/pretrain_gpt.py ${options} 2>&1 | tee $FILENAME

# Remove checkpoints.
rm -rf ${CHECKPOINT_DIR}/*
