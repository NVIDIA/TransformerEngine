/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <bit>

#include "./common.h"
#include "./utils.cuh"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"

namespace transformer_engine {

namespace {

__global__ void __launch_bounds__(1)
    update_tensor_scale_inv_kernel(const float *__restrict__ scale_ptr,
                                   float *__restrict__ scale_inv_ptr) {
  const float scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  reciprocal<float>(scale_inv_ptr, scale);
}

}  // namespace

void update_tensor_scale_inv(Tensor *t, cudaStream_t stream) {
  if (is_fp8_dtype(t->data.dtype) && is_tensor_scaling(t->scaling_mode)) {
    NVTE_CHECK(t->scale_inv.dptr != nullptr, "Tensor should have allocated scale_inv.");
    update_tensor_scale_inv_kernel<<<1, 1, 0, stream>>>(
        reinterpret_cast<const float *>(t->scale.dptr),
        reinterpret_cast<float *>(t->scale_inv.dptr));
  }
}

namespace {

constexpr size_t kThreadsPerBlock = 256;
template <typename TVectorized>
__global__ void __launch_bounds__(kThreadsPerBlock)
    memset_kernel(void *__restrict__ ptr, int value, size_t size_in_bytes) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx * sizeof(TVectorized) >= size_in_bytes) {
    return;  // Out of bounds
  }

  if ((idx + 1) * sizeof(TVectorized) > size_in_bytes) {
    // If the buffer size is not an even multiple of the vectorization, manually set the remaining bytes unvectorized.
    size_t remaining_bytes = size_in_bytes - idx * sizeof(TVectorized);
    memset(reinterpret_cast<uint8_t *>(ptr) + idx * sizeof(TVectorized), value, remaining_bytes);
    return;
  }

  union {
    TVectorized value;
    uint8_t data[sizeof(TVectorized)];
  } data;
  for (size_t i = 0; i < sizeof(TVectorized); ++i) {
    data.data[i] = static_cast<uint8_t>(value);
  }
  reinterpret_cast<TVectorized *>(ptr)[idx] = data.value;
}

}  // namespace

#define MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, vectorizedType, stream) \
  if (size_in_bytes >= sizeof(vectorizedType) &&                                             \
      reinterpret_cast<size_t>(ptr) % sizeof(vectorizedType) == 0) {                         \
    size_t numBlocks = DIVUP(size_in_bytes, kThreadsPerBlock * sizeof(vectorizedType));      \
    dim3 grid(numBlocks, 1, 1);                                                              \
    memset_kernel<vectorizedType>                                                            \
        <<<grid, kThreadsPerBlock, 0, stream>>>(ptr, value, size_in_bytes);                  \
    return;                                                                                  \
  }

extern "C" {
void nvte_memset(void *ptr, int value, size_t size_in_bytes, cudaStream_t stream) {
  NVTE_API_CALL(nvte_memset);
  NVTE_CHECK(ptr != nullptr, "Pointer for memset must be allocated.");

  if (size_in_bytes > 4096) {
    // Use cudaMemsetAsync for larger sizes.
    cudaMemsetAsync(ptr, value, size_in_bytes, stream);
    return;
  }

  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, float4, stream);
  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, float2, stream);
  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, float, stream);
  MEMSET_VECTORIZED_KERNEL_DISPATCH(ptr, size_in_bytes, value, uint8_t, stream);
}
}  // extern "C"

void checkCuDriverContext(CUstream stream) {
  CUcontext ctx;
  const CUresult driver_status = cuda_driver::call("cuStreamGetCtx", stream, &ctx);
  switch (driver_status) {
    case CUDA_SUCCESS:
      break;

    case CUDA_ERROR_INVALID_CONTEXT:
      int current_device;
      NVTE_CHECK_CUDA(cudaGetDevice(&current_device));
      NVTE_CALL_CHECK_CUDA_DRIVER(cuDevicePrimaryCtxRetain, &ctx, current_device);
      NVTE_CALL_CHECK_CUDA_DRIVER(cuCtxSetCurrent, ctx);
      break;

    default:
      const char *desc_NVTE_CHECK_CUDA_DRIVER;
      cuda_driver::call("cuGetErrorString", driver_status, &desc_NVTE_CHECK_CUDA_DRIVER);
      NVTE_ERROR("CUDA Error: ", desc_NVTE_CHECK_CUDA_DRIVER);
  }
}

CUtensorMapDataType get_CUtensorMapDataType(DType dtype) {
  static const std::unordered_map<DType, CUtensorMapDataType> dtypeMapping = {
      {DType::kByte, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat32, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32},
      {DType::kFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16},
      {DType::kBFloat16, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16},
      {DType::kFloat8E4M3, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8},
      {DType::kFloat8E5M2, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT8}};
  return dtypeMapping.at(dtype);
}

// Set up parameters to create TMA descriptor.
void create_2D_tensor_map(CUtensorMap &tensorMap, const SimpleTensor &tensor,
                          const uint64_t globalY, const uint64_t globalX, const uint32_t shmemY,
                          const uint32_t shmemX, const uint32_t stride_elems,
                          const uint32_t offset_elems, const size_t type_size) {
  // Get a function pointer to the cuTensorMapEncodeTiled driver API
  static PFN_cuTensorMapEncodeTiled cuDriverTensorMapEncodeTiled = []() {
    void *driver_ptr = cuda_driver::get_symbol("cuTensorMapEncodeTiled");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
  }();
  // rank is the number of dimensions of the array
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {globalX, globalY};

  // The stride is the number of bytes to traverse from the first element of one row to the next
  uint64_t stride[rank - 1] = {stride_elems * type_size};

  // The boxSize is the size of the shared memory buffer that is used as the
  // source/destination of a TMA transfer
  uint32_t boxSize[rank] = {shmemX, shmemY};

  // The distance between elements in units of sizeof(element)
  uint32_t elemStride[rank] = {1, 1};

  const CUtensorMapDataType tensorDataType = get_CUtensorMapDataType(tensor.dtype);
  void *dataPtr =
      reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(tensor.dptr) + offset_elems * type_size);

  NVTE_CHECK(is_aligned_ptr(dataPtr, TMA_gmem_alignment),
             "Tensor data pointer must be 16B aligned");

  const int TMA_needed_size = TMA_gmem_alignment / type_size;
  NVTE_CHECK(globalX % TMA_needed_size == 0, "Shape not supported. For ", type_size,
             "-byte data type, expected multiple of ", TMA_needed_size, ", got ", globalX);

  // Create the tensor descriptor.
  NVTE_CHECK_CUDA_DRIVER(cuDriverTensorMapEncodeTiled(
      &tensorMap,  // CUtensorMap *tensorMap,
      tensorDataType,
      rank,        // cuuint32_t tensorRank,
      dataPtr,     // void *globalAddress,
      size,        // const cuuint64_t *globalDim,
      stride,      // const cuuint64_t *globalStrides,
      boxSize,     // const cuuint32_t *boxDim,
      elemStride,  // const cuuint32_t *elementStrides,
      // Interleave patterns can be used to accelerate loading of values that
      // are less than 4 bytes long.
      CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

      // Swizzling can be used to avoid shared memory bank conflicts.
      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,

      // L2 Promotion can be used to widen the effect of a cache-policy to a wider
      // set of L2 cache lines.
      CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
      // CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_256B,

      // Any element that is outside of bounds will be set to zero by the TMA transfer.
      CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

bool is_supported_by_CC_100() {
  int deviceComputeCapability = cuda::sm_arch(cuda::current_device());

  return deviceComputeCapability >= 100;
}

std::vector<std::vector<Tensor *>> convert_tensor_array(NVTETensor **nvte_tensors,
                                                        size_t outer_size, size_t inner_size) {
  std::vector<std::vector<Tensor *>> ret;
  for (size_t i = 0; i < outer_size; ++i) {
    ret.emplace_back();
    for (size_t j = 0; j < inner_size; ++j) {
      ret.back().push_back(reinterpret_cast<Tensor *>(nvte_tensors[i][j]));
    }
  }
  return ret;
}

}  // namespace transformer_engine
