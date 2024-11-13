# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Paths
: ${TE_PATH:=/opt/transformerengine}
: ${MCORE_PATH:=${TE_PATH}/qa/L1_pytorch_mcore_integrationtest/Megatron-LM}

# Download Megatron-LM if needed
if [ ! -d "${MCORE_PATH}" ]; then
    pushd $(dirname ${MCORE_PATH})
    git clone -b core_r0.9.0 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
    popd
fi

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

python
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
--max-position-embeddings 2048
--micro-batch-size 1
--global-batch-size 8
--train-iters 10
--eval-iters 10
--lr 1e-4
--mock-data
--vocab-file /data/gpt3/pile-cc1-cc2-shuf/bpe/gpt2-vocab.json
--merge-file /data/gpt3/pile-cc1-cc2-shuf/bpe/gpt2-merges.txt
--transformer-impl transformer_engine
--fp8-format hybrid
"
COMMAND=$(echo "${COMMAND}" | tr '\n' ' ')

# Launch Megatron-LM
bash -c "${COMMAND}"
