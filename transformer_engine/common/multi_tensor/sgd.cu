/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <assert.h>
#include <cuda_fp8.h>
#include <transformer_engine/multi_tensor.h>
#include <transformer_engine/transformer_engine.h>

#include "../utils.cuh"
#include "multi_tensor_apply.cuh"

namespace transformer_engine {
namespace multi_tensor_sgd {

#define BLOCK_SIZE 512
#define ILP 4

/**
 * Perform fused SGD on multiple buffers
 * N: number of tensors
 * tl[0] : gradients
 * tl[1] : weights
 * tl[2] : momentum buffers
 * tl[3] : fp16 weights (if appropriate)
 * wd : weight_decay (scalar)
 * momentum : momentum (scalar)
 * dampening : momentum dampening (scalar)
 * lr : learning rate (scalar)
 * nesterov : enable nesterov (bool)
 * first run : necessary for proper momentum handling & init
 * wd_after_momentum : apply weight decay _after_ momentum instead of before
 **/
template <int N, typename T_grad, typename T_weight>
struct SGDFunctor {
  __device__ __forceinline__ void operator()(int chunk_size, volatile int* noop_gmem,
                                             TensorListMetadata<N>& tl,  // NOLINT(*)
                                             float wd, float momentum, float dampening, float lr,
                                             bool nesterov, bool first_run, bool wd_after_momentum,
                                             float scale) {
    // Early exit if we don't need to do anything
    if (*noop_gmem) return;

    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.sizes[tensor_loc];

    T_grad* grad_in = reinterpret_cast<T_grad*>(tl.addresses[0][tensor_loc]);
    grad_in += chunk_idx * chunk_size;

    T_weight* weight_in = reinterpret_cast<T_weight*>(tl.addresses[1][tensor_loc]);
    weight_in += chunk_idx * chunk_size;

    T_weight* mom_in = reinterpret_cast<T_weight*>(tl.addresses[2][tensor_loc]);
    mom_in += chunk_idx * chunk_size;

    fp16* model_weights_out = nullptr;
    if (N == 4) {
      model_weights_out = reinterpret_cast<fp16*>(tl.addresses[3][tensor_loc]);
      model_weights_out += chunk_idx * chunk_size;
    }

    n -= chunk_idx * chunk_size;

    // Non-divergent exit condition for the __syncthreads
    float incoming_grads[ILP];
    float incoming_weights[ILP];
    float incoming_moms[ILP];
    for (int i_start = 0; i_start < n && i_start < chunk_size; i_start += blockDim.x * ILP) {
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        incoming_grads[ii] = 0;
        incoming_weights[ii] = 0;
        incoming_moms[ii] = 0;
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          incoming_grads[ii] = static_cast<float>(grad_in[i]) * scale;
          incoming_weights[ii] = static_cast<float>(weight_in[i]);
          incoming_moms[ii] = static_cast<float>(mom_in[i]);
        }
      }

// note for clarification to future michael:
// From a pure memory dependency perspective, there's likely no point unrolling
// the write loop, since writes just fire off once their LDGs arrive.
// Put another way, the STGs are dependent on the LDGs, but not on each other.
// There is still compute ILP benefit from unrolling the loop though.
#pragma unroll
      for (int ii = 0; ii < ILP; ii++) {
        int i = i_start + threadIdx.x + ii * blockDim.x;
        if (i < n && i < chunk_size) {
          // apply weight decay before momentum if necessary
          if (wd != 0.f && !wd_after_momentum) incoming_grads[ii] += wd * incoming_weights[ii];

          if (momentum != 0.f) {
            if (!first_run)
              incoming_moms[ii] =
                  incoming_moms[ii] * momentum + (1.f - dampening) * incoming_grads[ii];
            else  // initialize momentums to current incoming grads
              incoming_moms[ii] = incoming_grads[ii];

            if (nesterov)
              incoming_grads[ii] += momentum * incoming_moms[ii];
            else
              incoming_grads[ii] = incoming_moms[ii];
          }

          // Apply WD after momentum if desired
          if (wd != 0.f && wd_after_momentum) incoming_grads[ii] += wd * incoming_weights[ii];

          // adjust the weight and write out
          weight_in[i] += (-lr * incoming_grads[ii]);

          // if necessary, write out an fp16 copy of the weights
          if (N == 4) model_weights_out[i] = static_cast<fp16>(weight_in[i]);

          // also write out the new momentum
          if (momentum != 0.f) mom_in[i] = incoming_moms[ii];
        }
      }
    }
  }
};

void multi_tensor_sgd_cuda(int chunk_size, Tensor noop_flag,
                           std::vector<std::vector<Tensor*>> tensor_lists, float wd, float momentum,
                           float dampening, float lr, bool nesterov, bool first_run,
                           bool wd_after_momentum, float scale, const int device_id,
                           cudaStream_t stream) {
  const size_t num_tensor_lists = tensor_lists.size();
  const size_t num_tensors_per_list = tensor_lists[0].size();

  auto grad_type = tensor_lists[0][0]->dtype();
  auto weight_type = tensor_lists[1][0]->dtype();

  if (num_tensor_lists == 4) {
    for (int i = 0; i < num_tensors_per_list; i++)
      NVTE_CHECK(tensor_lists[3][i]->dtype() == DType::kFloat16,
                 "Additional output tensors should always be fp16.");
  }

  // We have 3 possibilities to handle here, in terms of
  // grad_type, param_type, momentum_type, requires_fp16_copy
  // 1. fp16, fp16, fp16, No
  // 2. fp32, fp32, fp32, No
  // 3. fp16, fp32, fp32, Yes
  // 4. fp32, fp32, fp32, Yes // this is the materialize_master_grads=True case
  // It's easier to hardcode these possibilities than to use
  // switches etc. to handle the cross-product of cases where
  // we don't want the majority of them.

  // Case 1. fp16, fp16, fp16, No
  if (grad_type == DType::kFloat16 && weight_type == DType::kFloat16 && num_tensor_lists == 3) {
    multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                          SGDFunctor<3, fp16, fp16>(), device_id, stream, wd, momentum, dampening,
                          lr, nesterov, first_run, wd_after_momentum, scale);
  }
  // Case 2. fp32, fp32, fp32, No
  else if (grad_type == DType::kFloat32 &&  // NOLINT(*)
           weight_type == DType::kFloat32 && num_tensor_lists == 3) {
    multi_tensor_apply<3>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                          SGDFunctor<3, float, float>(), device_id, stream, wd, momentum, dampening,
                          lr, nesterov, first_run, wd_after_momentum, scale);
  }
  // Case 3. fp16, fp32, fp32, Yes
  else if (grad_type == DType::kFloat16 &&  // NOLINT(*)
           weight_type == DType::kFloat32 && num_tensor_lists == 4) {
    multi_tensor_apply<4>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                          SGDFunctor<4, fp16, float>(), device_id, stream, wd, momentum, dampening,
                          lr, nesterov, first_run, wd_after_momentum, scale);
  }
  // Case 4. fp32, fp32, fp32, Yes
  else if (grad_type == DType::kFloat32 &&  // NOLINT(*)
           weight_type == DType::kFloat32 && num_tensor_lists == 4) {
    multi_tensor_apply<4>(BLOCK_SIZE, chunk_size, noop_flag, tensor_lists,
                          SGDFunctor<4, float, float>(), device_id, stream, wd, momentum, dampening,
                          lr, nesterov, first_run, wd_after_momentum, scale);
  } else {
    NVTE_ERROR("Unsupported combination of weight and gradient types.");
  }

  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace multi_tensor_sgd
}  // namespace transformer_engine

void nvte_multi_tensor_sgd_cuda(int chunk_size, NVTETensor noop_flag, NVTETensor** tensor_lists,
                                const size_t num_tensor_lists, const size_t num_tensors_per_list,
                                float wd, float momentum, float dampening, float lr, int nesterov,
                                int first_run, int wd_after_momentum, float scale,
                                const int device_id, cudaStream_t stream) {
  NVTE_API_CALL(nvte_multi_tensor_sgd_cuda);
  using namespace transformer_engine;

  multi_tensor_sgd::multi_tensor_sgd_cuda(
      chunk_size, *reinterpret_cast<Tensor*>(noop_flag),
      convert_tensor_array(tensor_lists, num_tensor_lists, num_tensors_per_list), wd, momentum,
      dampening, lr, nesterov, first_run, wd_after_momentum, scale, device_id, stream);
}
