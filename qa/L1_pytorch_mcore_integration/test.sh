# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Paths
: ${TE_PATH:=/opt/transformerengine}
: ${MCORE_PATH:=${TE_PATH}/qa/L1_pytorch_mcore_integration/Megatron-LM}

# Check whether FP8 is supported
DEVICE_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | sed 's/[^0-9]//g')
if [[ ${DEVICE_ARCH} -ge 89 ]]; then
    WITH_FP8=1
fi

# Download Megatron-LM if needed
if [ ! -d "${MCORE_PATH}" ]; then
    pushd $(dirname ${MCORE_PATH})
    git clone -b core_r0.12.0 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
    popd
fi

# Create mock vocab
VOCAB_FILE=${TE_PATH}/qa/L1_pytorch_mcore_integration/vocab.json
printf "" > ${VOCAB_FILE}
printf "{" >> ${VOCAB_FILE}
printf "\"<|endoftext|>\": 0" >> ${VOCAB_FILE}
seq 1 4095 | awk '{ printf(", \"%d\": %d", $1, $1) }' >> ${VOCAB_FILE}
printf "}" >> ${VOCAB_FILE}

# Megatron-LM invocation
COMMAND="
NVTE_TORCH_COMPILE=0
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
NVTE_FLASH_ATTN=1
NVTE_FWD_LAYERNORM_SM_MARGIN=0
NVTE_BWD_LAYERNORM_SM_MARGIN=0
CUDA_DEVICE_MAX_CONNECTIONS=1
NVTE_BIAS_GELU_NVFUSION=0
NVTE_BIAS_DROPOUT_FUSION=0

python3
-m torch.distributed.launch
--use_env
--nnodes=1
--nproc_per_node=1

${MCORE_PATH}/pretrain_gpt.py
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--use-cpu-initialization
--num-layers 2
--hidden-size 128
--num-attention-heads 8
--seq-length 128
--max-position-embeddings 128
--micro-batch-size 1
--global-batch-size 8
--train-iters 10
--eval-iters 10
--lr 1e-4
--mock-data
--vocab-file ${VOCAB_FILE}
--merge-file ${TE_PATH}/qa/L1_pytorch_mcore_integration/merges.txt
--transformer-impl transformer_engine
${WITH_FP8:+--fp8-format hybrid}
"
COMMAND=$(echo "${COMMAND}" | tr '\n' ' ')

# Launch Megatron-LM
bash -c "${COMMAND}"
