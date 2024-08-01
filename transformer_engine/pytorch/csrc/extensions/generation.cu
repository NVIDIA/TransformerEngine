/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "extensions.h"

// Kernel used to update KV chache when attention layout is "thd".
template <typename scalar_t>
__global__ void attention_copy_kernel(scalar_t* cache_tensor, int* seq_len, int* incoming_seq_len,
                                      scalar_t* hidden_tensor, int max_incoming_seq_len,
                                      int max_seq_len, int b, int s) {
  for (int batch_idx = blockIdx.x; batch_idx < b; batch_idx += gridDim.x) {
    int to_copy = s * incoming_seq_len[batch_idx];
    int offset = seq_len[batch_idx];

    scalar_t* begin_cache_copy = cache_tensor + max_seq_len * s * batch_idx + s * offset;
    scalar_t* begin_hidden_copy = hidden_tensor + s * batch_idx * max_incoming_seq_len;

    for (int i = threadIdx.x; i < to_copy; i += blockDim.x) {
      *(begin_cache_copy + i) = *(begin_hidden_copy + i);
    }
  }
}

template <typename scalar_t>
void attention_copy_launcher(torch::Tensor A, torch::Tensor seq_len, torch::Tensor incoming_seq_len,
                             torch::Tensor B, int max_incoming_seq_len, int max_seq_len, int b,
                             int s) {
  attention_copy_kernel<<<16, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<scalar_t*>(A.data_ptr<scalar_t>()), seq_len.data_ptr<int>(),
      incoming_seq_len.data_ptr<int>(), reinterpret_cast<scalar_t*>(B.data_ptr<scalar_t>()),
      max_incoming_seq_len, max_seq_len, b, s);
}

void attention_copy(torch::Tensor A, torch::Tensor seq_len, torch::Tensor incoming_seq_len,
                    torch::Tensor B, int max_incoming_seq_len, int max_seq_len, int b, int s) {
  if (A.scalar_type() == at::ScalarType::Half) {
    using dtype = at::Half;
    attention_copy_launcher<dtype>(A, seq_len, incoming_seq_len, B, max_incoming_seq_len,
                                   max_seq_len, b, s);

  } else if (A.scalar_type() == at::ScalarType::BFloat16) {
    using dtype = at::BFloat16;
    attention_copy_launcher<dtype>(A, seq_len, incoming_seq_len, B, max_incoming_seq_len,
                                   max_seq_len, b, s);
  } else if (A.scalar_type() == at::ScalarType::Float) {
    using dtype = float;
    attention_copy_launcher<dtype>(A, seq_len, incoming_seq_len, B, max_incoming_seq_len,
                                   max_seq_len, b, s);
  } else {
    NVTE_ERROR("Unsupported dtype of out\n");
  }
}
