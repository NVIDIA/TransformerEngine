/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_EP_NCCL_EP_PROVIDER_H_
#define TRANSFORMER_ENGINE_COMMON_EP_NCCL_EP_PROVIDER_H_

#include <nccl_ep.h>

namespace transformer_engine {
namespace ep {

namespace nccl_ep {

/*! \brief Load and validate libnccl_ep.so. Idempotent and thread-safe. */
void initialize();

/*! \brief Get a function pointer from the runtime NCCL EP library. */
void* get_symbol(const char* symbol);

ncclResult_t create_group(ncclEpGroup_t* ep_group, ncclComm_t comm,
                          const ncclEpGroupConfig_t* config);
ncclResult_t group_destroy(ncclEpGroup_t ep_group);
ncclResult_t handle_destroy(ncclEpHandle_t handle);
ncclResult_t init_handle(ncclEpHandle_t* handle, ncclEpGroup_t ep_group, ncclEpLayout_t layout,
                         const ncclEpHandleConfig_t* config, int num_topk,
                         const ncclEpTensor_t* handle_mem);
ncclResult_t handle_mem_size(ncclEpGroup_t ep_group, ncclEpLayout_t layout,
                             const ncclEpHandleConfig_t* config, size_t* size_out, int num_topk);
ncclResult_t update_handle(ncclEpHandle_t handle, const ncclEpTensor_t* topk_idx,
                           const ncclEpLayoutInfo_t* layout_info, cudaStream_t stream);
ncclResult_t dispatch(ncclEpHandle_t handle, const ncclEpDispatchInputs_t* inputs,
                      const ncclEpDispatchOutputs_t* outputs, const ncclEpLayoutInfo_t* layout_info,
                      const ncclEpDispatchConfig_t* config, cudaStream_t stream);
ncclResult_t combine(ncclEpHandle_t handle, const ncclEpCombineInputs_t* inputs,
                     const ncclEpCombineOutputs_t* outputs, const ncclEpCombineConfig_t* config,
                     cudaStream_t stream);

}  // namespace nccl_ep

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_NCCL_EP_PROVIDER_H_
