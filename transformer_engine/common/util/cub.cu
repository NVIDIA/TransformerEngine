/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/cub.h>
#include "../common.h"
#include <cuda/std/execution>
#include <cub/device/device_topk.cuh>

void nvte_cub_topk(cudaStream_t stream, const NVTETensor keys_in, const NVTETensor values_in,
                   NVTETensor keys_out, NVTETensor values_out, NVTETensor workspace,
                   int num_items, int k, size_t workspace_bytes) {
  NVTE_API_CALL(nvte_cub_topk);
  using namespace transformer_engine;

  const Tensor *keys_in_tensor = convertNVTETensorCheck(keys_in);
  const Tensor *values_in_tensor = convertNVTETensorCheck(values_in);
  Tensor *keys_out_tensor = convertNVTETensor(keys_out);
  Tensor *values_out_tensor = convertNVTETensor(values_out);
  Tensor *workspace_tensor = convertNVTETensor(workspace);
  auto keys_in_dtype = keys_in_tensor->data.dtype;
  auto values_in_dtype = values_in_tensor->data.dtype;

  auto requirements = cuda::execution::require(
    cuda::execution::determinism::not_guaranteed,
    cuda::execution::output_ordering::unsorted
  );
  cuda::stream_ref stream_ref{stream};
  auto env = cuda::std::execution::env{stream_ref, requirements};

  #define DISPATCH_CUB_TOPK(KeyT, ValueT)  \
  do {  \
    KeyT *d_keys_in = reinterpret_cast<KeyT *>(keys_in_tensor->data.dptr);  \
    KeyT *d_keys_out = reinterpret_cast<KeyT *>(keys_out_tensor->data.dptr);  \
    ValueT *d_values_in = reinterpret_cast<ValueT *>(values_in_tensor->data.dptr);  \
    ValueT *d_values_out = reinterpret_cast<ValueT *>(values_out_tensor->data.dptr);  \
    void *d_workspace = reinterpret_cast<void *>(workspace_tensor->data.dptr);  \
    cub::DeviceTopK::MaxPairs(  \
      d_workspace, workspace_bytes,  \
      d_keys_in, d_keys_out,  \
      d_values_in, d_values_out,  \
      num_items, k, env  \
    );  \
  } while (0);

  if (keys_in_dtype == DType::kFloat32 && values_in_dtype == DType::kInt32) {
    DISPATCH_CUB_TOPK(float, int);
  } else if (keys_in_dtype == DType::kFloat16 && values_in_dtype == DType::kInt32) {
    DISPATCH_CUB_TOPK(__half, int);
  } else if (keys_in_dtype == DType::kBFloat16 && values_in_dtype == DType::kInt32) {
    DISPATCH_CUB_TOPK(__nv_bfloat16, int);
  } else {
    NVTE_ERROR("Unsupported input key and value data types");
  }
  #undef DISPATCH_CUB_TOPK
}
