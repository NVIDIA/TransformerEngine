/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMMON_H_
#define TRANSFORMER_ENGINE_COMMON_COMMON_H_

#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <transformer_engine/transformer_engine.h>

#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "./nvtx.h"
#include "./util/cuda_driver.h"
#include "./util/logging.h"

namespace transformer_engine {

struct SimpleTensor {
  void *dptr;
  std::vector<size_t> shape;
  DType dtype;

  SimpleTensor(void *dptr, const std::vector<size_t> &shape, DType dtype)
      : dptr(dptr), shape(shape), dtype(dtype) {}
  SimpleTensor() : SimpleTensor(nullptr, {}, DType::kFloat32) {}

  int numel() const {
    size_t acc = 1;
    for (const auto &dim : shape) {
      acc *= dim;
    }
    return acc;
  }
};

class ScalingMode : public NVTEScalingMode {
 public:
  ScalingMode() {
    x = -1;
    y = -1;
    delayed_scaling = true;
  }

  ScalingMode(const NVTEScalingMode &other) {  // NOLINT(runtime/explicit)
    x = other.x;
    y = other.y;
    delayed_scaling = other.delayed_scaling;
  }
};

struct Tensor {
  SimpleTensor data;
  SimpleTensor columnwise_data;
  SimpleTensor amax;
  SimpleTensor scale;
  SimpleTensor scale_inv;
  SimpleTensor columnwise_scale_inv;

  ScalingMode scaling_mode;

  Tensor()
      : data(),
        columnwise_data(),
        amax(nullptr, {1}, DType::kFloat32),
        scale(nullptr, {1}, DType::kFloat32),
        scale_inv(nullptr, {1}, DType::kFloat32),
        columnwise_scale_inv(nullptr, {1}, DType::kFloat32),
        scaling_mode() {}

  int numel() const {
    size_t acc = 1;
    for (const auto &dim : data.shape) {
      acc *= dim;
    }
    return acc;
  }
};

template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
  return (((x) + ((y)-1)) / (y));
}

using byte = uint8_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;
#if CUDA_VERSION >= 12080
    using fp8e8m0 = __nv_fp8_e8m0;
#endif
using e8m0_t = uint8_t;

namespace detail {

template <typename T>
constexpr inline const char *type_name() noexcept;
#define TRANSFORMER_ENGINE_TYPE_NAME(T)                  \
  template <>                                            \
  inline constexpr const char *type_name<T>() noexcept { \
    return #T;                                           \
  }
TRANSFORMER_ENGINE_TYPE_NAME(uint8_t)
TRANSFORMER_ENGINE_TYPE_NAME(int32_t)
TRANSFORMER_ENGINE_TYPE_NAME(int64_t)
TRANSFORMER_ENGINE_TYPE_NAME(float)
TRANSFORMER_ENGINE_TYPE_NAME(half)
TRANSFORMER_ENGINE_TYPE_NAME(nv_bfloat16)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e4m3)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e5m2)
#undef TRANSFORMER_ENGINE_TYPE_NAME

}  // namespace detail

template <typename T>
struct TypeInfo {
  using types = std::tuple<byte, int32, int64, fp32, fp16, bf16, fp8e4m3, fp8e5m2>;

  template <typename U, DType current>
  struct Helper {
    constexpr static DType getType() {
      constexpr int i = static_cast<int>(current);
      if (std::is_same<U, typename std::tuple_element<i, types>::type>::value) {
        return current;
      } else {
        return Helper<U, static_cast<DType>(i + 1)>::getType();
      }
    }
  };

  template <typename U>
  struct Helper<U, DType::kNumTypes> {
    constexpr static DType getType() { return DType::kNumTypes; }
  };

  template <typename U>
  constexpr static DType getType() {
    return Helper<U, DType::kByte>::getType();
  }

  constexpr static DType dtype = getType<T>();
  constexpr static size_t size = sizeof(T);
  constexpr static const char *name = detail::type_name<T>();
};

#define TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                           \
    using namespace transformer_engine;                      \
    case DType::kByte: {                                     \
      using type = unsigned char;                            \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kInt32: {                                    \
      using type = int32_t;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kInt64: {                                    \
      using type = int64_t;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat32: {                                  \
      using type = float;                                    \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat16: {                                  \
      using type = fp16;                                     \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kBFloat16: {                                 \
      using type = bf16;                                     \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat8E4M3: {                               \
      using type = fp8e4m3;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kFloat8E5M2: {                               \
      using type = fp8e5m2;                                  \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    default:                                                 \
      NVTE_ERROR("Invalid type.");                           \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(dtype, type, ...) \
  switch (dtype) {                                              \
    using namespace transformer_engine;                         \
    case DType::kFloat32: {                                     \
      using type = float;                                       \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kFloat16: {                                     \
      using type = fp16;                                        \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kBFloat16: {                                    \
      using type = bf16;                                        \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kFloat8E5M2: {                                  \
      using type = fp8e5m2;                                     \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    case DType::kFloat8E4M3: {                                  \
      using type = fp8e4m3;                                     \
      { __VA_ARGS__ }                                           \
    } break;                                                    \
    default:                                                    \
      NVTE_ERROR("Invalid type.");                              \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(dtype, type, ...) \
  switch (dtype) {                                                   \
    using namespace transformer_engine;                              \
    case DType::kFloat32: {                                          \
      using type = float;                                            \
      { __VA_ARGS__ }                                                \
    } break;                                                         \
    case DType::kFloat16: {                                          \
      using type = fp16;                                             \
      { __VA_ARGS__ }                                                \
    } break;                                                         \
    case DType::kBFloat16: {                                         \
      using type = bf16;                                             \
      { __VA_ARGS__ }                                                \
    } break;                                                         \
    default:                                                         \
      NVTE_ERROR("Invalid type.");                                   \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(dtype, type, ...) \
  switch (dtype) {                                               \
    using namespace transformer_engine;                          \
    case DType::kFloat8E5M2: {                                   \
      using type = fp8e5m2;                                      \
      { __VA_ARGS__ }                                            \
    } break;                                                     \
    case DType::kFloat8E4M3: {                                   \
      using type = fp8e4m3;                                      \
      { __VA_ARGS__ }                                            \
    } break;                                                     \
    default:                                                     \
      NVTE_ERROR("Invalid type.");                               \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, type, ...) \
  switch (dtype) {                                             \
    using namespace transformer_engine;                        \
    case DType::kFloat32: {                                    \
      using type = float;                                      \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case DType::kFloat16: {                                    \
      using type = fp16;                                       \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case DType::kBFloat16: {                                   \
      using type = bf16;                                       \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case DType::kFloat8E5M2:                                   \
    case DType::kFloat8E4M3: {                                 \
      NVTE_ERROR("FP8 type not instantiated for input.");      \
    } break;                                                   \
    default:                                                   \
      NVTE_ERROR("Invalid type.");                             \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(dtype, type, ...) \
  switch (dtype) {                                             \
    using namespace transformer_engine;                        \
    case DType::kFloat16: {                                    \
      using type = fp16;                                       \
      __VA_ARGS__;                                             \
      break;                                                   \
    }                                                          \
    case DType::kBFloat16: {                                   \
      using type = bf16;                                       \
      __VA_ARGS__;                                             \
      break;                                                   \
    }                                                          \
    default:                                                   \
      NVTE_ERROR("Invalid type for 16 bit.");                  \
  }

#define TRANSFORMER_ENGINE_MX_SCALE_DIM_SWITCH(SCALE_DIM, DIM, ...) \
  switch (SCALE_DIM) {                                              \
    case 1: {                                                       \
      constexpr size_t DIM = 1;                                     \
      { __VA_ARGS__ }                                               \
    } break;                                                        \
    case 32: {                                                      \
      constexpr size_t DIM = 32;                                    \
      { __VA_ARGS__ }                                               \
    } break;                                                        \
    default: {                                                      \
      NVTE_ERROR("Invalid size of the MX scaling factor.");         \
    }                                                               \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

inline size_t product(const std::vector<size_t> &shape) {
  size_t ret = 1;
  for (const auto &elem : shape) {
    ret *= elem;
  }
  return ret;
}

inline int log2_ceil(int value) {
  int log2_value = 0;
  while ((1 << log2_value) < value) ++log2_value;
  return log2_value;
}

template <size_t B>
inline size_t alignTo(size_t x) {
  size_t r = x % B;
  if (r == 0) return x;

  return x + B - r;
}

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

size_t typeToSize(const DType type);

void CheckInputTensor(const Tensor &t, const std::string &name);
void CheckOutputTensor(const Tensor &t, const std::string &name, bool allow_empty = false);

bool is_fp8_dtype(const DType t);

template <typename T>
std::string to_string(const std::vector<T> &v) {
  std::string s = "(";
  for (size_t i = 0; i < v.size(); ++i) {
    s += std::to_string(v[i]);
    if (i < v.size() - 1) {
      s += ", ";
    }
  }
  s += ")";
  return s;
}

std::string to_string(const DType type);
std::string to_string(const ScalingMode &type);

inline bool is_tensor_scaling(const ScalingMode &mode) { return (mode.x == -1) && (mode.y == -1); }

inline bool is_block_scaling(const ScalingMode &mode) {
  return (mode.x > 0) && (mode.y > 0) && (mode.delayed_scaling == 0);
}

inline bool is_rowwise_block_scaling(const ScalingMode &mode) {
  return (mode.x < mode.y) && (mode.delayed_scaling == 0);
}

inline bool is_delayed_tensor_scaling(const ScalingMode &mode) {
  return is_tensor_scaling(mode) && mode.delayed_scaling;
}

bool is_columnwise_block_scaling(const Tensor *t);

/*! \brief Update a tensor's FP8 scale-inverse
 *
 * The FP8 scale-inverse (dequantization scaling factor) is updated
 * with the reciprocal of the FP8 scale (quantization scaling factor).
 */
void update_tensor_scale_inv(Tensor *t, cudaStream_t stream);

#define NVTE_API_CALL(api_name) \
  transformer_engine::nvtx::NVTXWrapper _##api_name##_nvtx_wrapper(#api_name);

void checkCuDriverContext(CUstream stream);

CUtensorMapDataType get_CUtensorMapDataType(DType dtype);

inline bool isPointerAligned(const void *const ptr, const int alignment);

// Set up parameters to create TMA descriptor.
template <typename T>
void create_2D_tensor_map(CUtensorMap &tensorMap, const Tensor *tensor_ptr, const uint64_t globalY,
                          const uint64_t globalX, const uint32_t shmemY, const uint32_t shmemX);

bool is_supported_by_CC_100();
bool is_mxfp8_cast_supported_shape(const Tensor *output);
bool is_fp8_cast_supported_shape(const Tensor *output);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_COMMON_H_
