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

namespace amax_and_scale_update_impl {

// CUDA block size
constexpr size_t bsize = 256;

/* CUDA kernel to update amax history and FP8 scaling factors
 *
 * Block dims: bsize x 1 x 1
 *
 * Grid dims: num_scales x 1 x 1
 */
__global__ void __launch_bounds__(bsize)
kernel(const float* amax_history_ptr,
       const float* scale_ptr,
       const float* scale_inv_ptr,
       const unsigned char* scale_inv_mask_ptr,
       float* updated_amax_history_ptr,
       float* updated_scale_ptr,
       float* updated_scale_inv_ptr,
       size_t amax_history_length,
       size_t amax_history_stride,
       AmaxComputeAlgo amax_compute_algo,
       float scaled_max) {
  const size_t tid = threadIdx.x;
  const size_t bid = blockIdx.x;

  // Update amax
  float amax = 0;
  {
    // Roll amax history
    const auto* amax_history = amax_history_ptr + bid;
    auto* updated_amax_history = updated_amax_history_ptr + bid;
    const auto last_amax = amax_history[0];
    const auto& length = amax_history_length;
    const auto& stride = amax_history_stride;
    for (size_t off = 0; off < length; off += bsize) {
      const size_t i = off + tid;
      float a = 0;
      if (i < length) {
        a = (i < length - 1) ? amax_history[(i+1)*stride] : last_amax;
        amax = fmaxf(amax, a);
      }
      __syncthreads();  // In case roll is in-place
      if (i < length) {
        updated_amax_history[i*stride] = (i > 0) ? a : 0;
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
    // Update scale
    float scale;
    if (isfinite(amax) && amax > 0) {
      scale = scaled_max / amax;
    } else {
      scale = scale_ptr[bid];
    }
    updated_scale_ptr[bid] = scale;

    // Update scale inverse
    float scale_inv;
    if (scale_inv_mask_ptr == nullptr || scale_inv_mask_ptr[bid]) {
      scale_inv = 1 / scale;
    } else {
      scale_inv = scale_inv_ptr[bid];
    }
    updated_scale_inv_ptr[bid] = scale_inv;
  }
}

}  // namespace amax_and_scale_update_impl


}  // namespace

void amax_and_scale_update(const Tensor &amax_history,
                           const Tensor &scale,
                           const Tensor &scale_inv,
                           const Tensor &scale_inv_mask,
                           Tensor *updated_amax_history_,
                           Tensor *updated_scale_,
                           Tensor *updated_scale_inv_,
                           const std::string &amax_compute_algo,
                           DType fp8_dtype,
                           float margin,
                           cudaStream_t stream) {
  auto& updated_amax_history = *updated_amax_history_;
  auto& updated_scale = *updated_scale_;
  auto& updated_scale_inv = *updated_scale_inv_;

  // Number of elements in tensor
  auto numel = [] (const Tensor &tensor) -> size_t {
    size_t acc = 1;
    for (const auto& dim : tensor.data.shape) {
      acc *= dim;
    }
    return acc;
  };

  // Check tensors
  NVTE_CHECK(amax_history.data.shape.size() == 2,
             "Found ", amax_history.data.shape.size(), " dims");
  const size_t amax_history_length = amax_history.data.shape[0];
  const size_t num_scales = amax_history.data.shape[1];
  NVTE_CHECK(amax_history.data.dtype == DType::kFloat32,
             "Found ", dtype_name(amax_history.data.dtype), ".");
  NVTE_CHECK(numel(scale) == num_scales,
             "Expected ", num_scales, " elements, ",
             "but found ", numel(scale), ".");
  NVTE_CHECK(scale.data.dtype == DType::kFloat32,
             "Found ", dtype_name(scale.data.dtype), ".");
  if (scale_inv_mask.data.dptr != nullptr) {
    NVTE_CHECK(numel(scale_inv) == num_scales,
               "Expected ", num_scales, " elements, ",
               "but found ", numel(scale_inv), ".");
    NVTE_CHECK(scale_inv.data.dtype == DType::kFloat32);
    NVTE_CHECK(numel(scale_inv_mask) == num_scales,
               "Expected ", num_scales, " elements, ",
               "but found ", numel(scale_inv_mask), ".");
    NVTE_CHECK(scale_inv_mask.data.dtype == DType::kByte,
               "Found ", dtype_name(scale_inv_mask.data.dtype), ".");
  }
  NVTE_CHECK(updated_amax_history.data.shape.size() == 2,
             "Found ", updated_amax_history.data.shape.size(), " dims.");
  NVTE_CHECK(updated_amax_history.data.shape[0] == amax_history_length,
             "Expected ", amax_history_length, ", ",
             "but found ", updated_amax_history.data.shape[0]);
  NVTE_CHECK(updated_amax_history.data.shape[1] == num_scales,
             "Expected ", num_scales, ", ",
             "but found ", updated_amax_history.data.shape[1]);
  NVTE_CHECK(updated_amax_history.data.dtype == DType::kFloat32,
             "Got ", dtype_name(updated_amax_history.data.dtype), ".");
  NVTE_CHECK(numel(updated_scale) == num_scales,
             "Expected ", num_scales, " elements, ",
             "but found ", numel(updated_scale), ".");
  NVTE_CHECK(updated_scale.data.dtype == DType::kFloat32,
             "Got ", dtype_name(updated_scale.data.dtype), ".");
  NVTE_CHECK(numel(updated_scale_inv) == num_scales,
             "Expected ", num_scales, " elements, ",
             "but found ", numel(updated_scale_inv), ".");
  NVTE_CHECK(updated_scale_inv.data.dtype == DType::kFloat32,
             "Got ", dtype_name(updated_scale_inv.data.dtype), ".");

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

  // Launch CUDA kernel
  constexpr size_t block_size = amax_and_scale_update_impl::bsize;
  const size_t grid_size = num_scales;
  amax_and_scale_update_impl::kernel
    <<<grid_size, block_size, 0, stream>>>(
      static_cast<const float*>(amax_history.data.dptr),
      static_cast<const float*>(scale.data.dptr),
      static_cast<const float*>(scale_inv.data.dptr),
      static_cast<const unsigned char*>(scale_inv_mask.data.dptr),
      static_cast<float*>(updated_amax_history.data.dptr),
      static_cast<float*>(updated_scale.data.dptr),
      static_cast<float*>(updated_scale_inv.data.dptr),
      amax_history_length,
      num_scales,
      amax_compute_algo_,
      scaled_max);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace delayed_scaling_recipe
}  // namespace transformer_engine

void nvte_delayed_scaling_recipe_amax_and_scale_update(const NVTETensor amax_history,
                                                       const NVTETensor scale,
                                                       const NVTETensor scale_inv,
                                                       const NVTETensor scale_inv_mask,
                                                       NVTETensor updated_amax_history,
                                                       NVTETensor updated_scale,
                                                       NVTETensor updated_scale_inv,
                                                       const char *amax_compute_algo,
                                                       NVTEDType fp8_dtype,
                                                       float margin,
                                                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_delayed_scaling_recipe_amax_and_scale_update);
  using namespace transformer_engine;
  delayed_scaling_recipe::amax_and_scale_update(
    *reinterpret_cast<const Tensor*>(amax_history),
    *reinterpret_cast<const Tensor*>(scale),
    *reinterpret_cast<const Tensor*>(scale_inv),
    *reinterpret_cast<const Tensor*>(scale_inv_mask),
    reinterpret_cast<Tensor*>(updated_amax_history),
    reinterpret_cast<Tensor*>(updated_scale),
    reinterpret_cast<Tensor*>(updated_scale_inv),
    amax_compute_algo,
    static_cast<DType>(fp8_dtype),
    margin,
    stream);
}
