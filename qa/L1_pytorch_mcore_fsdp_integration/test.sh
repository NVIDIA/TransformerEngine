# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Megatron-LM / Megatron-FSDP commit for main branch on Apr. 10, 2026.
# Necessary to support wgrad accumulate fusion and Megatron-FSDP NCCL UBR,
# and fixes decoupled_grad <> DistOpt usage in Megatron-LM.
MCORE_REF=${1:-ab43d43f0bc04f4656d4af15afb6e7e4c9ad71c8}

# Paths
: ${TE_PATH:=/opt/transformerengine}
: ${MCORE_PATH:=${TE_PATH}/qa/L1_pytorch_mcore_fsdp_integration/Megatron-LM}

# Download Megatron-LM if needed
if [ ! -d "${MCORE_PATH}" ]; then
    pushd $(dirname ${MCORE_PATH})
    git clone https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
    pushd Megatron-LM && git checkout "${MCORE_REF}" && popd
    popd
fi

# Create mock vocab
VOCAB_FILE=${TE_PATH}/qa/L1_pytorch_mcore_fsdp_integration/vocab.json
printf "" > ${VOCAB_FILE}
printf "{" >> ${VOCAB_FILE}
printf "\"<|endoftext|>\": 0" >> ${VOCAB_FILE}
seq 1 4095 | awk '{ printf(", \"%d\": %d", $1, $1) }' >> ${VOCAB_FILE}
printf "}" >> ${VOCAB_FILE}

# Setting CUDA_DEVICE_MAX_CONNECTIONS limits
# Megatron-FSDP stream parallelism.
unset CUDA_DEVICE_MAX_CONNECTIONS
export NVTE_TORCH_COMPILE=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NVTE_FLASH_ATTN=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_BIAS_GELU_NVFUSION=0
export NVTE_BIAS_DROPOUT_FUSION=0

# V1 offloading has bugs that are exposed by Megatron-FSDP.
# This test will focus on validating the new offloading code.
# Un-set the Megatron-LM default of V1.
export NVTE_CPU_OFFLOAD_V1=0

# Megatron-LM command to run Megatron-FSDP.
python3 \
-m torch.distributed.launch \
--use_env \
--nnodes=1 \
--nproc_per_node=$(nvidia-smi -L | wc -l) \
${MCORE_PATH}/pretrain_gpt.py \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--num-layers 2 \
--hidden-size 128 \
--num-attention-heads 8 \
--swiglu \
--seq-length 128 \
--max-position-embeddings 128 \
--micro-batch-size 1 \
--global-batch-size 8 \
--train-iters 10 \
--eval-iters 10 \
--eval-interval 100 \
--lr 1e-4 \
--mock-data \
--vocab-file ${VOCAB_FILE} \
--merge-file ${TE_PATH}/qa/L1_pytorch_mcore_fsdp_integration/merges.txt \
--transformer-impl transformer_engine \
--use-megatron-fsdp \
--data-parallel-sharding-strategy optim_grads_params \
--use-distributed-optimizer \
--use-precision-aware-optimizer \
--num-distributed-optimizer-instances 2 \
--outer-dp-sharding-strategy optim \
--use-nccl-ub \
--fsdp-double-buffer \
--fsdp-manual-registration \
--fp8-format hybrid \
--fp8-param-gather \
--fp8-recipe mxfp8 \
--cpu-offloading-num-layers 1 \
--overlap-grad-reduce \
--overlap-param-gather \
--ckpt-format fsdp_dtensor \
--init-model-with-meta-device \
--bf16 \
--grad-reduce-in-bf16
