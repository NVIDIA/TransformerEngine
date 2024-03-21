/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/recipe.h>

#include <cmath>
#include <string>

#include "../common.h"
#include "../util/logging.h"
#include "../util/cuda_runtime.h"

namespace transformer_engine {
namespace delayed_scaling_recipe {

namespace {

// amax value to use for updating scaling factor
enum class AmaxComputeAlgo { INVALID, MOST_RECENT, MAX };

const char* dtype_name(DType dtype) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, Type,
    return TypeInfo<Type>::name;
  );  // NOLINT(*)
  return "";
}

// Maximum representable value of an FP8 dtype
inline float fp8_dtype_max(DType dtype) {
  switch (dtype) {
  case DType::kFloat8E4M3: return 448;
  case DType::kFloat8E5M2: return 57344;
  default:
    NVTE_ERROR("Expected FP8 dtype, but got ", dtype_name(dtype));
  }
  return 0;
}

// struct for amax parameters
struct AmaxParam {
  int num_scale = 0;
  float* amax_history = nullptr;
  float* scale = nullptr;
  float* scale_inv = nullptr;
};

// dummy struct for kernel_bulk's other params
struct OtherParams {
  float* a;
  size_t b;
  AmaxComputeAlgo c;
  float d;
};

#if CUDART_VERSION >= 12010
constexpr size_t max_constant_memory_per_kernel = 32768;
constexpr size_t AMAX_PARAMS_LIMIT = (
  max_constant_memory_per_kernel - sizeof(OtherParams)) / sizeof(AmaxParam);
#else
constexpr size_t max_constant_memory_per_kernel = 4096;
constexpr size_t AMAX_PARAMS_LIMIT = (
  max_constant_memory_per_kernel - sizeof(OtherParams)) / sizeof(AmaxParam);
#endif

struct AmaxParams {
  AmaxParam param[AMAX_PARAMS_LIMIT];
};

namespace amax_and_scale_update_impl {

// CUDA block size
constexpr size_t bsize = 256;

/* CUDA kernel to bulk-update amax history and FP8 scaling factors
 *
 * Block dims: bsize x 1 x 1
 *
 * Grid dims: num_tensors x 1 x 1
 */
__global__ void __launch_bounds__(bsize)
kernel_bulk(
       float* amax_reduction_buffer,
       AmaxParams p,
       size_t amax_history_length,
       AmaxComputeAlgo amax_compute_algo,
       float scaled_max) {
  const size_t bid = blockIdx.x;
  const size_t tid = threadIdx.x;
  const int num_scale = p.param[bid].num_scale;

  int offset_in_buffer = 0;
  for (int j = 0; j < bid; j++) {
    offset_in_buffer += p.param[j].num_scale;
  }

  for (int count = 0; count < num_scale; count++) {
    // Update amax
    float amax = 0;
    {
      // Roll amax history
      const auto& length = amax_history_length;
      const auto& stride = p.param[bid].num_scale;
      auto* amax_history = p.param[bid].amax_history+count;
      const auto last_amax = ((amax_reduction_buffer != nullptr)
            && (amax_reduction_buffer[offset_in_buffer+count] != 0.0f)) ?
            amax_reduction_buffer[offset_in_buffer+count] : amax_history[0];
      for (size_t off = 0; off < length; off += bsize) {
        const size_t i = off + tid;
        float a = 0;
        if (i < length) {
          a = (i < length - 1) ? amax_history[(i+1)*stride] : last_amax;
          amax = fmaxf(amax, a);
        }
        __syncthreads();  // Inplace roll
        if (i < length) {
          amax_history[i*stride] = (i > 0) ? a : 0;
        }
      }

      // Compute amax to use for scaling factor
      switch (amax_compute_algo) {
      case AmaxComputeAlgo::MOST_RECENT:
        amax = last_amax;
        break;
      case AmaxComputeAlgo::MAX:
        {
          __shared__ float shared_amax[bsize];
          shared_amax[tid] = amax;
          __syncthreads();
#pragma unroll
          for (size_t off = bsize / 2; off > 0; off /= 2) {
            if (tid < off) {
              shared_amax[tid] = fmaxf(shared_amax[tid], shared_amax[tid + off]);
            }
            __syncthreads();
          }
          amax = shared_amax[tid];
        }
        break;
      default:
        amax = 0;
      }
    }

    // Update scale and scale inverse
    if (tid == 0) {
      float scale;
      if (isfinite(amax) && amax > 0) {
        scale = scaled_max / amax;
      } else {
        scale = p.param[bid].scale[count];
      }
      p.param[bid].scale[count] = scale;
      p.param[bid].scale_inv[count] = 1 / scale;
    }
  }
}

}  // namespace amax_and_scale_update_impl

}  // namespace


void amax_and_scale_update_after_reduction(const Tensor &amax_reduction_buffer,
                                           std::vector<Tensor*> amax_histories,
                                           std::vector<Tensor*> scales,
                                           std::vector<Tensor*> scale_invs,
                                           const std::string &amax_compute_algo,
                                           DType fp8_dtype,
                                           float margin,
                                           cudaStream_t stream) {
  using namespace transformer_engine;

  // amax value to use for updating scaling factor
  AmaxComputeAlgo amax_compute_algo_ = AmaxComputeAlgo::INVALID;
  if (amax_compute_algo == "max") {
    amax_compute_algo_ = AmaxComputeAlgo::MAX;
  } else if (amax_compute_algo == "most_recent") {
    amax_compute_algo_ = AmaxComputeAlgo::MOST_RECENT;
  } else {
    NVTE_ERROR("Unsupported amax compute algorithm (", amax_compute_algo, ")");
  }

  // Expected maximum value after scale is applied
  const float scaled_max = fp8_dtype_max(fp8_dtype) * std::pow(2.f, -margin);

  // Number of elements in tensor
  auto numel = [] (const Tensor *tensor) -> size_t {
    size_t acc = 1;
    for (const auto& dim : tensor->data.shape) {
      acc *= dim;
    }
    return acc;
  };

  // Number of tensors in the bulk
  const size_t num_tensors = amax_histories.size();
  const int num_kernels = (num_tensors+AMAX_PARAMS_LIMIT-1)/AMAX_PARAMS_LIMIT;
  size_t amax_history_length = 0;
  if (num_tensors > 0) {
    amax_history_length = amax_histories[0]->data.shape[0];
  }

  // amax parameters
  float* amax_buffer = static_cast<float*>(amax_reduction_buffer.data.dptr);
  AmaxParams p;
  for (int iter = 0; iter < num_kernels; iter++) {
    size_t kernel_num_scales = 0;
    size_t kernel_num_tensors = (iter == (num_kernels -1))
          ? num_tensors % AMAX_PARAMS_LIMIT: AMAX_PARAMS_LIMIT;
    for (size_t pi = 0; pi < kernel_num_tensors; pi++) {
      size_t i = iter * AMAX_PARAMS_LIMIT + pi;

      // Check tensors
      int num_scale = amax_histories[i]->data.shape[1];
      NVTE_CHECK(amax_histories[i]->data.dtype == DType::kFloat32,
                 "Found ", dtype_name(amax_histories[i]->data.dtype), ".");
      NVTE_CHECK(amax_histories[i]->data.shape.size() == 2,
                 "Found ", amax_histories[i]->data.shape.size(), " dims");
      NVTE_CHECK(numel(amax_histories[i]) == amax_history_length * num_scale,
                 "Expected ", amax_history_length * num_scale, " elements, ",
                 "but found ", numel(amax_histories[i]), ".");
      NVTE_CHECK(scales[i]->data.dtype == DType::kFloat32,
                 "Found ", dtype_name(scales[i]->data.dtype), ".");
      NVTE_CHECK(scales[i]->data.shape.size() == 1,
                 "Found ", scales[i]->data.shape.size(), " dims");
      NVTE_CHECK(numel(scales[i]) == num_scale,
                 "Expected ", num_scale, " elements, ",
                 "Found ", numel(scales[i]), ".");

      // amax parameters
      kernel_num_scales += num_scale;
      p.param[pi].num_scale = num_scale;
      p.param[pi].amax_history = static_cast<float*>(amax_histories[i]->data.dptr);
      p.param[pi].scale = static_cast<float*>(scales[i]->data.dptr);
      p.param[pi].scale_inv = static_cast<float*>(scale_invs[i]->data.dptr);
    }

    // Launch CUDA kernel
    size_t grid_size = kernel_num_tensors;
    const size_t block_size = amax_and_scale_update_impl::bsize;
    amax_and_scale_update_impl::kernel_bulk
      <<<grid_size, block_size, 0, stream>>>(
        amax_buffer,
        p,
        amax_history_length,
        amax_compute_algo_,
        scaled_max);
    NVTE_CHECK_CUDA(cudaGetLastError());

    // shift amax buffer pointer
    if (amax_buffer != nullptr) {
      amax_buffer += kernel_num_scales;
    }
  }
}


}  // namespace delayed_scaling_recipe
}  // namespace transformer_engine


void nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
                           const NVTETensor amax_reduction_buffer,
                           std::vector<NVTETensor> amax_histories,
                           std::vector<NVTETensor> scales,
                           std::vector<NVTETensor> scale_invs,
                           const char *amax_compute_algo,
                           NVTEDType fp8_dtype,
                           float margin,
                           cudaStream_t stream) {
  NVTE_API_CALL(nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction);
  using namespace transformer_engine;
  size_t num_tensors = amax_histories.size();
  std::vector<Tensor*> t_amax_histories, t_scales, t_scale_invs;
  for (size_t i = 0; i < num_tensors; i++) {
    t_amax_histories.push_back(reinterpret_cast<Tensor*>(amax_histories[i]));
    t_scales.push_back(reinterpret_cast<Tensor*>(scales[i]));
    t_scale_invs.push_back(reinterpret_cast<Tensor*>(scale_invs[i]));
  }
  delayed_scaling_recipe::amax_and_scale_update_after_reduction(
    *reinterpret_cast<const Tensor*>(amax_reduction_buffer),
    t_amax_histories,
    t_scales,
    t_scale_invs,
    amax_compute_algo,
    static_cast<DType>(fp8_dtype),
    margin,
    stream);
}
