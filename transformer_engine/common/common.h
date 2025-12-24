/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMMON_H_
#define TRANSFORMER_ENGINE_COMMON_COMMON_H_

#include <cudaTypedefs.h>
#define FP4_TYPE_SUPPORTED (CUDA_VERSION >= 12080)

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
#endif

#include <cuda_runtime_api.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
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

std::string to_string(const DType type);
std::string to_string(const NVTEScalingMode &mode);

inline bool is_tensor_scaling(const NVTEScalingMode &mode) {
  return mode == NVTE_DELAYED_TENSOR_SCALING;
}

inline bool is_block_scaling(const NVTEScalingMode &mode) { return !is_tensor_scaling(mode); }

inline bool is_delayed_tensor_scaling(const NVTEScalingMode &mode) {
  return mode == NVTE_DELAYED_TENSOR_SCALING;
}

inline bool is_nvfp4_scaling(const NVTEScalingMode &mode) { return mode == NVTE_NVFP4_1D_SCALING; }

inline bool is_mxfp8_scaling(const NVTEScalingMode &mode) { return mode == NVTE_MXFP8_1D_SCALING; }

inline bool is_mxfp_scaling(const NVTEScalingMode &mode) { return mode == NVTE_MXFP8_1D_SCALING; }

inline bool is_nvfp_scaling(const NVTEScalingMode &mode) { return mode == NVTE_NVFP4_1D_SCALING; }

inline size_t product(const std::vector<size_t> &shape, const size_t begin, const size_t end) {
  NVTE_CHECK(begin <= end && end <= shape.size(), "Attempted to access entries ", begin, " to ",
             end, " in a vector with ", shape.size(), " entries");
  size_t ret = 1;
  for (size_t i = begin; i < end; ++i) {
    ret *= shape[i];
  }
  return ret;
}

inline size_t product(const std::vector<size_t> &shape) {
  size_t ret = 1;
  for (const auto &elem : shape) {
    ret *= elem;
  }
  return ret;
}

size_t get_buffer_size_bytes(const size_t N, const DType buffer_dtype);
size_t get_buffer_size_bytes(const size_t dim_first, const size_t dim_last,
                             const DType buffer_dtype);

struct SimpleTensor {
  void *dptr;
  std::vector<size_t> shape;
  DType dtype;

  SimpleTensor(void *dptr, std::vector<size_t> shape, DType dtype)
      : dptr{dptr}, shape{std::move(shape)}, dtype{dtype} {}

  SimpleTensor(const NVTEBasicTensor &tensor)  // NOLINT
      : dptr(tensor.data_ptr),
        shape(tensor.shape.data, tensor.shape.data + tensor.shape.ndim),
        dtype(static_cast<DType>(tensor.dtype)) {}

  SimpleTensor() : SimpleTensor(nullptr, std::vector<size_t>{0}, DType::kFloat32) {}

  operator NVTEBasicTensor() const {
    return {dptr, static_cast<NVTEDType>(dtype),
            nvte_make_shape(this->shape.data(), this->shape.size())};
  }

  /*! Number of tensor elements. */
  size_t numel() const { return product(shape); }

  /*! Whether the tensor is initialized.
   *
   *  Tensors with non-trivial shapes are considered initialized. This
   *  means that there is no guarantee that the data pointer can be
   *  safely accessed.
   */
  bool has_data() const { return !(dptr == nullptr && shape.size() == 1 && shape[0] == 0); }

  /*! Buffer size in bytes. */
  size_t buffer_size_bytes() const { return get_buffer_size_bytes(numel(), dtype); }

  /*! Reset to uninitialized tensor. */
  void clear() {
    dptr = nullptr;
    shape.resize(1);
    shape[0] = 0;
    dtype = DType::kFloat32;
  }
};

struct Tensor {
 public:
  SimpleTensor data;
  SimpleTensor columnwise_data;
  SimpleTensor amax;
  SimpleTensor columnwise_amax;
  SimpleTensor scale;
  SimpleTensor scale_inv;
  SimpleTensor columnwise_scale_inv;

  NVTEScalingMode scaling_mode;
  NVTETensor nvte_tensor;

  Tensor() : scaling_mode{NVTE_DELAYED_TENSOR_SCALING}, nvte_tensor{0} {}

  /*! Reset tensor data. */
  void clear() {
    data.clear();
    columnwise_data.clear();
    amax.clear();
    columnwise_amax.clear();
    scale.clear();
    scale_inv.clear();
    columnwise_scale_inv.clear();
    scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
  }

  explicit operator NVTETensor() const noexcept { return nvte_tensor; }

  /*! Number of tensor elements. */
  size_t numel() const {
    if (!has_data() && has_columnwise_data()) {
      return product(columnwise_data.shape);
    }
    return product(data.shape);
  }

  /*! Whether the tensor data buffer is not uninitialized.
   *
   *  Buffers with non-trivial shapes are considered initialized. This
   *  means that there is no guarantee that the data pointer can be
   *  safely accessed.
   */
  bool has_data() const { return data.has_data(); }

  /*! Whether the tensor column-wise data buffer is not uninitialized.
   *
   *  Buffers with non-trivial shapes are considered initialized. This
   *  means that there is no guarantee that the data pointer can be
   *  safely accessed.
   */
  bool has_columnwise_data() const { return columnwise_data.has_data(); }

  /*! Datatype of tensor elements. */
  DType dtype() const {
    if (!has_data() && has_columnwise_data()) {
      return columnwise_data.dtype;
    }
    return data.dtype;
  }

  /*! Number of tensor dimensions. */
  size_t dim() const {
    if (!has_data() && has_columnwise_data()) {
      return columnwise_data.shape.size();
    }
    return data.shape.size();
  }

  /*! Tensor dimensions.
   *
   *  This is the logical tensor shape. The underlying data may have a
   *  different shape, e.g. the column-wise data for some tensor
   *  formats are transposed.
   */
  std::vector<size_t> shape() const {
    // Each tensor format interprets its data differently
    switch (scaling_mode) {
      case NVTE_DELAYED_TENSOR_SCALING:
      case NVTE_BLOCK_SCALING_1D:
      case NVTE_BLOCK_SCALING_2D:
      case NVTE_NVFP4_1D_SCALING: {
        // Row-wise data shape matches tensor logical shape,
        // column-wise data shape is transpose of logical shape
        if (!has_data() && has_columnwise_data()) {
          std::vector<size_t> ret;
          if (!columnwise_data.shape.empty()) {
            ret.reserve(columnwise_data.shape.size());
            for (size_t i = 1; i < columnwise_data.shape.size(); i++) {
              ret.push_back(columnwise_data.shape[i]);
            }
            ret.push_back(columnwise_data.shape.front());
          }
          return ret;
        }
        return data.shape;
      }
      case NVTE_MXFP8_1D_SCALING: {
        // Row-wise and column-wise data shapes both match tensor
        // logical shape
        if (!has_data() && has_columnwise_data()) {
          return columnwise_data.shape;
        }
        return data.shape;
      }
      default:
        NVTE_ERROR("Cannot parse tensor shape with scaling mode \"", to_string(scaling_mode), "\"");
    }
  }

  /*! Matrix height after tensor is flattened to 2D
   *
   * If a tensor has dimensions (D1, D2, ..., Dn), it is reinterpreted
   * as a (D1*D2*...*D(n-1), Dn) matrix.
   */
  size_t flat_first_dim() const {
    const auto &full_shape = shape();
    size_t ret = 1;
    if (!full_shape.empty()) {
      for (size_t i = 0; i < full_shape.size() - 1; i++) {
        ret *= full_shape[i];
      }
    }
    return ret;
  }

  /*! Matrix width after tensor is flattened to 2D
   *
   * If a tensor has dimensions (D1, D2, ..., Dn), it is reinterpreted
   * as a (D1*D2*...*D(n-1), Dn) matrix.
   */
  size_t flat_last_dim() const {
    const auto &full_shape = shape();
    if (full_shape.empty()) {
      return 1;
    } else {
      return full_shape.back();
    }
  }
};

struct GroupedTensor {
 public:
  /* EXPERIMENTAL FEATURE AND SUBJECT TO CHANGE. */
  /*
  Grouped tensor is a collection of tensors with different shapes but the same dtype and scaling mode

  Shape Representation:
  - logical_shape: 2D shape representing the conceptual layouy, i.e. the shape when member tensors are flattened to 2D and stacked together (REQUIRED)
    + When all_same_shape(): [num_tensors * M, N] where each tensor is (M, N)
    + When varying_first_dim(): [~sum_of_first_dims, N] where N is common
    + When varying_last_dim(): [M, ~sum_of_last_dims] where M is common
    + When varying_both_dims(): [1, total_elements] (fully flattened)

  - first_dims and last_dims are OPTIONAL (empty if dimension is uniform)
    + Empty first_dims: all tensors have the same first dimension
    + Empty last_dims: all tensors have the same last dimension
    + Both empty: all tensors have identical shapes
    + Both set: each tensor has unique shape (first_dims[i], last_dims[i])

  Data Layout:
  - ALL data fields are stored as 1D flattened arrays (data, columnwise_data, scale_inv, etc.)
  - logical_shape provides the conceptual 2D interpretation
  - All data is stored on device in contiguous layout
  */

  SimpleTensor data;
  SimpleTensor columnwise_data;
  SimpleTensor scale_inv;
  SimpleTensor columnwise_scale_inv;
  SimpleTensor amax;
  SimpleTensor columnwise_amax;
  SimpleTensor scale;  // for FP8-DS only

  // Shape information (OPTIONAL - empty if dimension is uniform across all tensors)
  // first_dims[i] = first dimension of tensor i (empty if all tensors have same first dim)
  // last_dims[i] = last dimension of tensor i (empty if all tensors have same last dim)
  SimpleTensor first_dims;  // Device pointer to int64_t array of length num_tensors (or empty)
  SimpleTensor last_dims;   // Device pointer to int64_t array of length num_tensors (or empty)

  // Offsets for indexing into contiguous 1D layout (OPTIONAL - not needed if all_same_shape())
  // tensor_offsets[i] = element offset to start of tensor i (cumulative sum of numel for tensors 0..i-1)
  // Usage: tensor_i_ptr = (char*)data.dptr + tensor_offsets[i] * element_size
  // If empty and all_same_shape(): offset[i] = i * M * N (where M, N are common dimensions)
  SimpleTensor tensor_offsets;  // Device pointer to int64_t array of length num_tensors (or empty)

  // Logical shape: conceptual 2D shape of the grouped data (REQUIRED)
  // Represents how the 1D flattened data should be interpreted as 2D
  // Always 2D with positive dimensions
  NVTEShape logical_shape;

  NVTEScalingMode scaling_mode;
  size_t num_tensors;
  NVTEGroupedTensor nvte_tensor;

  GroupedTensor(NVTEScalingMode scaling_mode, size_t num_tensors)
      : data(),
        columnwise_data(),
        scale_inv(),
        columnwise_scale_inv(),
        amax(),
        columnwise_amax(),
        scale(),
        num_tensors(num_tensors),
        first_dims(nullptr, std::vector<size_t>{0}, DType::kInt64),
        last_dims(nullptr, std::vector<size_t>{0}, DType::kInt64),
        tensor_offsets(nullptr, std::vector<size_t>{0}, DType::kInt64),
        logical_shape(nvte_make_shape(nullptr, 1)),
        scaling_mode(scaling_mode),
        nvte_tensor(0) {}

  explicit operator NVTEGroupedTensor() const noexcept { return nvte_tensor; }

  bool has_data() const noexcept { return data.has_data(); }
  bool has_columnwise_data() const noexcept { return columnwise_data.has_data(); }

  bool all_same_first_dim() const noexcept { return !first_dims.has_data(); }
  bool all_same_last_dim() const noexcept { return !last_dims.has_data(); }
  bool all_same_shape() const noexcept { return !first_dims.has_data() && !last_dims.has_data(); }
  bool varying_both_dims() const noexcept { return first_dims.has_data() && last_dims.has_data(); }

  size_t get_common_first_dim() const {
    NVTE_CHECK(all_same_first_dim(), "First dim varies across tensors");
    NVTE_CHECK(logical_shape.ndim == 2, "Logical shape must be 2D");
    if (all_same_shape()) {
      // When both dims are uniform: logical_shape = [num_tensors * M, N]
      return logical_shape.data[0] / num_tensors;
    } else {
      // When varying last dims but not first dim: logical_shape = [M, sum_of_last_dims]
      return logical_shape.data[0];
    }
  }
  size_t get_common_last_dim() const {
    NVTE_CHECK(all_same_last_dim(), "Last dim varies across tensors");
    NVTE_CHECK(logical_shape.ndim == 2, "Logical shape must be 2D");
    // For both uniform and varying first dim cases: logical_shape[1] is the common last dim
    return logical_shape.data[1];
  }

  DType dtype() const {
    if (!has_data() && has_columnwise_data()) {
      return columnwise_data.dtype;
    }
    return data.dtype;
  }

  void clear() {
    data.clear();
    columnwise_data.clear();
    scale_inv.clear();
    columnwise_scale_inv.clear();
    amax.clear();
    columnwise_amax.clear();
    scale.clear();
    first_dims.clear();
    last_dims.clear();
    tensor_offsets.clear();
    logical_shape = nvte_make_shape(nullptr, 1);
    num_tensors = 0;
    scaling_mode = NVTE_DELAYED_TENSOR_SCALING;
    nvte_tensor = 0;
  }
};

struct QuantizationConfig {
  bool force_pow_2_scales = false;
  float amax_epsilon = 0.0f;
  NVTETensor noop_tensor = nullptr;
  Float8BlockScaleTensorFormat float8_block_scale_tensor_format =
      Float8BlockScaleTensorFormat::GEMM_READY;
  NVTETensor rng_state = nullptr;
  bool nvfp4_2d_quantization = false;
  bool stochastic_rounding = false;
  bool use_fast_math = false;

  static constexpr size_t attr_sizes[] = {
      sizeof(bool),                          // force_pow_2_scales
      sizeof(float),                         // amax_epsilon
      sizeof(NVTETensor),                    // noop_tensor
      sizeof(Float8BlockScaleTensorFormat),  // float8_block_scale_tensor_format
      sizeof(NVTETensor),                    // rng_seed and offset
      sizeof(bool),                          // nvfp4_2d_quantization
      sizeof(bool),                          // stochastic_rounding
      sizeof(bool)                           // use_fast_math
  };
};

cudaDataType_t get_cuda_dtype(const transformer_engine::DType t);

template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
  return (((x) + ((y)-1)) / (y));
}

template <typename T1, typename T2>
constexpr __device__ __host__ __forceinline__ uint64_t DIVUP_TO_MULTIPLE(const T1 &N, const T2 &M) {
  static_assert(std::is_integral<T1>::value && std::is_integral<T2>::value,
                "Integral type required.");
  return DIVUP(static_cast<uint64_t>(N), static_cast<uint64_t>(M)) * M;
}

using byte = uint8_t;
using int16 = int16_t;
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
#if FP4_TYPE_SUPPORTED
using fp4e2m1 = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
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
TRANSFORMER_ENGINE_TYPE_NAME(int16_t)
TRANSFORMER_ENGINE_TYPE_NAME(int32_t)
TRANSFORMER_ENGINE_TYPE_NAME(int64_t)
TRANSFORMER_ENGINE_TYPE_NAME(float)
TRANSFORMER_ENGINE_TYPE_NAME(half)
TRANSFORMER_ENGINE_TYPE_NAME(nv_bfloat16)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e4m3)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e5m2)
#if CUDA_VERSION >= 12080
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e8m0)
#endif
#if FP4_TYPE_SUPPORTED
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp4_e2m1)
#endif
#undef TRANSFORMER_ENGINE_TYPE_NAME

template <typename T>
struct TypeExtrema;

#if FP4_TYPE_SUPPORTED
template <>
struct TypeExtrema<fp4e2m1> {
  static constexpr float max = 6.0f;
  static constexpr float max_inverse = 1.0 / max;
};
#endif

template <>
struct TypeExtrema<fp8e4m3> {
  static constexpr float max = 448.0f;
  static constexpr float max_inverse = 1.0 / max;
};

template <>
struct TypeExtrema<fp8e5m2> {
  static constexpr float max = 57344.0f;
  static constexpr float max_inverse = 1.0 / max;
};

template <>
struct TypeExtrema<bf16> {
  // Hex float format of 1.(7 bits of 1) * 2 ^ 127
  static constexpr float max = 0x1.FEp127;
};

template <>
struct TypeExtrema<fp16> {
  // Hex float format of 1.(10 bits of 1) * 2 ^ 15
  static constexpr float max = 0x1.FFCp15;
};

template <typename T>
struct TypeExtrema {
  static constexpr float max = std::numeric_limits<T>::max();
};

}  // namespace detail

template <typename T>
struct BitsNumber;

#if FP4_TYPE_SUPPORTED
template <>
struct BitsNumber<fp4e2m1> {
  static constexpr size_t num_bits = 4;
};
#endif

template <typename T>
struct BitsNumber {
  static constexpr size_t num_bits = 8 * sizeof(T);
};

template <typename T>
struct TypeInfo {
#if FP4_TYPE_SUPPORTED
  using types = std::tuple<byte, int16, int32, int64, fp32, fp16, bf16, fp8e4m3, fp8e5m2, fp4e2m1
#if CUDA_VERSION >= 12080
                           ,
                           fp8e8m0
#endif
                           >;
#else
  using types = std::tuple<byte, int16, int32, int64, fp32, fp16, bf16, fp8e4m3, fp8e5m2
#if CUDA_VERSION >= 12080
                           ,
                           fp8e8m0
#endif
                           >;
#endif

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
  constexpr static size_t size = BitsNumber<T>::num_bits;
  constexpr static float max_finite_value = detail::TypeExtrema<T>::max;
  constexpr static const char *name = detail::type_name<T>();
};

#if FP4_TYPE_SUPPORTED
#define SWITCH_FP4_TYPE_HANDLE(type, ...) \
  case DType::kFloat4E2M1: {              \
    using type = fp4e2m1;                 \
    { __VA_ARGS__ }                       \
  } break;
#else
#define SWITCH_FP4_TYPE_HANDLE(type, ...)  // do nothing
#endif

#define TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, type, ...) \
  switch (dtype) {                                           \
    using namespace transformer_engine;                      \
    case DType::kByte: {                                     \
      using type = unsigned char;                            \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
    case DType::kInt16: {                                    \
      using type = int16_t;                                  \
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
    case DType::kFloat8E8M0: {                               \
      using type = byte;                                     \
      { __VA_ARGS__ }                                        \
    } break;                                                 \
      SWITCH_FP4_TYPE_HANDLE(type, __VA_ARGS__)              \
    default:                                                 \
      NVTE_ERROR("Invalid type.");                           \
  }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FLOAT(dtype, type, ...) \
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
    case DType::kFloat8E4M3: {                                 \
      using type = fp8e4m3;                                    \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    case DType::kFloat8E5M2: {                                 \
      using type = fp8e5m2;                                    \
      { __VA_ARGS__ }                                          \
    } break;                                                   \
    default:                                                   \
      NVTE_ERROR("Invalid type.");                             \
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

// Add a pack_size argument to select the packed type for FP4
#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP4x2_ONLY(dtype, pack_size, type, ...) \
  switch (dtype) {                                                             \
    using namespace transformer_engine;                                        \
    case DType::kFloat4E2M1: {                                                 \
      using type = __nv_fp4x2_storage_t;                                       \
      { __VA_ARGS__ }                                                          \
    } break;                                                                   \
    default:                                                                   \
      NVTE_ERROR("Invalid type.");                                             \
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
    case DType::kFloat4E2M1: {                                 \
      NVTE_ERROR("FP4 type not instantiated for input.");      \
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

#define TRANSFORMER_ENGINE_SWITCH_CONDITION(CONDITION, FLAG, ...) \
  if (CONDITION) {                                                \
    constexpr bool FLAG = true;                                   \
    { __VA_ARGS__ }                                               \
  } else {                                                        \
    constexpr bool FLAG = false;                                  \
    { __VA_ARGS__ }                                               \
  }

////////////////////////////////////////////////////////////////////////////////////////////////////

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

template <typename T>
struct is_fp4 : std::false_type {};

#if FP4_TYPE_SUPPORTED
template <>
struct is_fp4<fp4e2m1> : std::true_type {};
#endif

// [128,4] rowwise and [4,128] colwise alignment requirements for the tensor with scaling factors
constexpr size_t scale_tensor_alignment_X_rowwise = 4;
constexpr size_t scale_tensor_alignment_Y_rowwise = 128;
constexpr size_t scale_tensor_alignment_X_colwise = 128;
constexpr size_t scale_tensor_alignment_Y_colwise = 4;

// Alignment requirements for the Tensor Memory Accelerator (TMA)
constexpr size_t TMA_GMEM_ALIGNMENT = 16;    // global memory address alignment
constexpr size_t TMA_SHMEM_ALIGNMENT = 128;  // shared memory address alignment

inline bool is_aligned_ptr(const void *ptr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

inline bool is_aligned_tensor_data(const Tensor &t, size_t alignment) {
  return is_aligned_ptr(static_cast<const void *>(t.data.dptr), alignment);
}

size_t typeToSize(const DType type);
size_t typeToNumBits(const DType type);

void CheckNoopTensor(const Tensor &t, const std::string &name);
void CheckInputTensor(const Tensor &t, const std::string &name);
void CheckOutputTensor(const Tensor &t, const std::string &name, bool allow_empty = false);

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

// Set up parameters to create TMA descriptor.
void create_2D_tensor_map(
    CUtensorMap &tensorMap, const SimpleTensor &tensor, const uint64_t globalY,
    const uint64_t globalX, const uint32_t shmemY, const uint32_t shmemX,
    const uint32_t stride_elems, const uint32_t offset_elems, const size_t type_num_bits,
    const CUtensorMapSwizzle swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE);

bool is_supported_by_CC_100();

std::vector<std::vector<Tensor *>> convert_tensor_array(NVTETensor **nvte_tensors,
                                                        size_t outer_size, size_t inner_size);

Tensor *convertNVTETensor(const NVTETensor tensor);
Tensor *convertNVTETensorCheck(const NVTETensor tensor);

GroupedTensor *convertNVTEGroupedTensor(const NVTEGroupedTensor tensor);
GroupedTensor *convertNVTEGroupedTensorCheck(const NVTEGroupedTensor tensor);

// Helper functions for GroupedTensor validation
void CheckGroupedTensorShapeArrays(const GroupedTensor &t, const std::string &name);
void CheckInputGroupedTensor(const GroupedTensor &t, const std::string &name);
void CheckOutputGroupedTensor(const GroupedTensor &t, const std::string &name,
                              bool allow_empty = false);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_COMMON_H_
