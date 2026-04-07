# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

set -e

# Paths
: ${TE_PATH:=/opt/transformerengine}
: ${MCORE_PATH:=${TE_PATH}/qa/L1_pytorch_mcore_fsdp_integration/Megatron-LM}

# Download Megatron-LM if needed
if [ ! -d "${MCORE_PATH}" ]; then
    pushd $(dirname ${MCORE_PATH})
    git clone -b core_v0.16.1 https://github.com/NVIDIA/Megatron-LM.git Megatron-LM
    popd
fi

# Create mock vocab
VOCAB_FILE=${TE_PATH}/qa/L1_pytorch_mcore_fsdp_integration/vocab.json
printf "" > ${VOCAB_FILE}
printf "{" >> ${VOCAB_FILE}
printf "\"<|endoftext|>\": 0" >> ${VOCAB_FILE}
seq 1 4095 | awk '{ printf(", \"%d\": %d", $1, $1) }' >> ${VOCAB_FILE}
printf "}" >> ${VOCAB_FILE}

# Megatron-LM command to run Megatron-FSDP.
# TODO(@cspades): Megatron-Core 0.16.1 doesn't have the NCCL UBR / double-buffer
# fix for wgrad accumulate fusion yet. Next version bump of Megatron-Core, add:
# --use-nccl-ub
# --fsdp-double-buffer
# --fsdp-manual-registration
COMMAND="
NVTE_TORCH_COMPILE=0
NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
NVTE_FLASH_ATTN=1
NVTE_FWD_LAYERNORM_SM_MARGIN=0
NVTE_BWD_LAYERNORM_SM_MARGIN=0
NVTE_BIAS_GELU_NVFUSION=0
NVTE_BIAS_DROPOUT_FUSION=0
unset CUDA_DEVICE_MAX_CONNECTIONS

python3
-m torch.distributed.launch
--use_env
--nnodes=1
--nproc_per_node=4

${MCORE_PATH}/pretrain_gpt.py
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 1
--use-cpu-initialization
--num-layers 2
--hidden-size 128
--num-attention-heads 8
--swiglu
--seq-length 128
--max-position-embeddings 128
--micro-batch-size 1
--global-batch-size 8
--train-iters 10
--eval-iters 10
--lr 1e-4
--mock-data
--vocab-file ${VOCAB_FILE}
--merge-file ${TE_PATH}/qa/L1_pytorch_mcore_fsdp_integration/merges.txt
--transformer-impl transformer_engine
--use-megatron-fsdp
--data-parallel-sharding-strategy optim_grads_params
--use-distributed-optimizer
--use-precision-aware-optimizer
--num-distributed-optimizer-instances 2
--outer-dp-sharding-strategy optim
--fp8-format hybrid
--fp8-param-gather
--fp8-recipe tensorwise
--cpu-offloading-num-layers 1
--overlap-grad-reduce
--overlap-param-gather
--ckpt-format fsdp_dtensor
--init-model-with-meta-device
--bf16
--grad-reduce-in-bf16
"
COMMAND=$(echo "${COMMAND}" | tr '\n' ' ')

# Launch Megatron-LM
bash -c "${COMMAND}"
