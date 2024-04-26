/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMMON_H_
#define TRANSFORMER_ENGINE_COMMON_COMMON_H_

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <transformer_engine/transformer_engine.h>
#include "./nvtx.h"
#include "./util/logging.h"

namespace transformer_engine {

struct SimpleTensor {
  void *dptr;
  std::vector<size_t> shape;
  DType dtype;

  SimpleTensor(void *dptr, const std::vector<size_t> &shape, DType dtype) :
    dptr(dptr), shape(shape), dtype(dtype) {}
  SimpleTensor() : SimpleTensor(nullptr, {}, DType::kFloat32) {}
};

struct Tensor {
  SimpleTensor data;
  SimpleTensor amax;
  SimpleTensor scale;
  SimpleTensor scale_inv;

  Tensor() : data(),
             amax(nullptr, {1}, DType::kFloat32),
             scale(nullptr, {1}, DType::kFloat32),
             scale_inv(nullptr, {1}, DType::kFloat32) {}
};

template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
    return (((x) + ((y)-1)) / (y));
}

using byte = uint8_t;
using int32 = int32_t;
using fp32 = float;
using fp16 = sycl::half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

namespace detail {

template <typename T>
constexpr inline const char *type_name() noexcept;
#define TRANSFORMER_ENGINE_TYPE_NAME(T) \
  template <> inline constexpr const char *type_name<T>() noexcept { return #T; }
TRANSFORMER_ENGINE_TYPE_NAME(uint8_t)
TRANSFORMER_ENGINE_TYPE_NAME(int32_t)
TRANSFORMER_ENGINE_TYPE_NAME(float)
TRANSFORMER_ENGINE_TYPE_NAME(sycl::half)
TRANSFORMER_ENGINE_TYPE_NAME(nv_bfloat16)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e4m3)
TRANSFORMER_ENGINE_TYPE_NAME(__nv_fp8_e5m2)
#undef TRANSFORMER_ENGINE_TYPE_NAME

}  // namespace detail

template <typename T>
struct TypeInfo{
    using types = std::tuple<byte,
                             int32,
                             fp32,
                             fp16,
                             bf16,
                             fp8e4m3,
                             fp8e5m2>;

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
        constexpr static DType getType() {
            return DType::kNumTypes;
        }
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
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kByte: \
            { \
                using type = unsigned char; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kInt32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat16: \
            { \
                using type = fp16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kBFloat16: \
            { \
                using type = bf16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E4M3: \
            { \
                using type = fp8e4m3; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E5M2: \
            { \
                using type = fp8e5m2; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kFloat32: \
            { \
                using type = float; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat16: \
            { \
                using type = fp16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kBFloat16: \
            { \
                using type = bf16; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E5M2: \
            { \
                using type = fp8e5m2; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E4M3: \
            { \
                using type = fp8e4m3; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kFloat8E5M2: \
            { \
                using type = fp8e5m2; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kFloat8E4M3: \
            { \
                using type = fp8e4m3; \
                {__VA_ARGS__} \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, type, ...)                 \
    switch (dtype) {                                                           \
        using namespace transformer_engine;                                    \
    case DType::kFloat32: {                                                    \
        using type = float;                                                    \
            { {                                                                \
                dpct::has_capability_or_fail(__VA_ARGS__->get_device(),        \
                                             {sycl::aspect::fp16});            \
                                                                               \
                __VA_ARGS__->submit([&](sycl::handler &cgh) {                  \
                    /*                                                         \
                    DPCT1083:552: The size of local memory in the migrated     \
                    code may be different from the original code. Check that   \
                    the allocated memory size in the migrated code is correct. \
                    */                                                         \
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(       \
                        sycl::range<1>(__VA_ARGS__), cgh);                     \
                    sycl::local_accessor<float, 1> staging_acc_ct1(            \
                        sycl::range<1>(cast_transpose_num_threads /            \
                                       THREADS_PER_WARP),                      \
                        cgh);                                                  \
                                                                               \
                    const InputType *__VA_ARGS___ct0 = __VA_ARGS__;            \
                    OutputType *__VA_ARGS___ct1 = __VA_ARGS__;                 \
                    OutputType *__VA_ARGS___ct2 = __VA_ARGS__;                 \
                    const fp32 *__VA_ARGS___ct3 = __VA_ARGS__;                 \
                    fp32 *__VA_ARGS___ct4 = __VA_ARGS__;                       \
                    const size_t __VA_ARGS___ct5 = __VA_ARGS__;                \
                    const size_t __VA_ARGS___ct6 = __VA_ARGS__;                \
                    const size_t __VA_ARGS___ct7 = __VA_ARGS__;                \
                                                                               \
                    cgh.parallel_for(                                          \
                        sycl::nd_range<3>(                                     \
                            sycl::range<3>(1, 1, __VA_ARGS__) *                \
                                sycl::range<3>(1, 1, __VA_ARGS__),             \
                            sycl::range<3>(1, 1, __VA_ARGS__)),                \
                        [=](sycl::nd_item<3> item_ct1)                         \
                            [[intel::reqd_sub_group_size(32)]] {               \
                                cast_transpose_kernel<, , , , >(               \
                                    __VA_ARGS___ct0, __VA_ARGS___ct1,          \
                                    __VA_ARGS___ct2, __VA_ARGS___ct3,          \
                                    __VA_ARGS___ct4, __VA_ARGS___ct5,          \
                                    __VA_ARGS___ct6, __VA_ARGS___ct7,          \
                                    item_ct1,                                  \
                                    dpct_local_acc_ct1                         \
                                        .get_multi_ptr<                        \
                                            sycl::access::decorated::no>()     \
                                        .get(),                                \
                                    staging_acc_ct1                            \
                                        .get_multi_ptr<                        \
                                            sycl::access::decorated::no>()     \
                                        .get());                               \
                            });                                                \
                });                                                            \
}}                                                                             \
    } break;                                                                   \
    case DType::kFloat16: {                                                    \
        using type = fp16;                                                     \
            { {                                                                \
                dpct::has_capability_or_fail(__VA_ARGS__->get_device(),        \
                                             {sycl::aspect::fp16});            \
                                                                               \
                __VA_ARGS__->submit([&](sycl::handler &cgh) {                  \
                    /*                                                         \
                    DPCT1083:553: The size of local memory in the migrated     \
                    code may be different from the original code. Check that   \
                    the allocated memory size in the migrated code is correct. \
                    */                                                         \
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(       \
                        sycl::range<1>(__VA_ARGS__), cgh);                     \
                    sycl::local_accessor<float, 1> staging_acc_ct1(            \
                        sycl::range<1>(cast_transpose_num_threads /            \
                                       THREADS_PER_WARP),                      \
                        cgh);                                                  \
                                                                               \
                    const InputType *__VA_ARGS___ct0 = __VA_ARGS__;            \
                    OutputType *__VA_ARGS___ct1 = __VA_ARGS__;                 \
                    OutputType *__VA_ARGS___ct2 = __VA_ARGS__;                 \
                    const fp32 *__VA_ARGS___ct3 = __VA_ARGS__;                 \
                    fp32 *__VA_ARGS___ct4 = __VA_ARGS__;                       \
                    const size_t __VA_ARGS___ct5 = __VA_ARGS__;                \
                    const size_t __VA_ARGS___ct6 = __VA_ARGS__;                \
                    const size_t __VA_ARGS___ct7 = __VA_ARGS__;                \
                                                                               \
                    cgh.parallel_for(                                          \
                        sycl::nd_range<3>(                                     \
                            sycl::range<3>(1, 1, __VA_ARGS__) *                \
                                sycl::range<3>(1, 1, __VA_ARGS__),             \
                            sycl::range<3>(1, 1, __VA_ARGS__)),                \
                        [=](sycl::nd_item<3> item_ct1)                         \
                            [[intel::reqd_sub_group_size(32)]] {               \
                                cast_transpose_kernel<, , , , >(               \
                                    __VA_ARGS___ct0, __VA_ARGS___ct1,          \
                                    __VA_ARGS___ct2, __VA_ARGS___ct3,          \
                                    __VA_ARGS___ct4, __VA_ARGS___ct5,          \
                                    __VA_ARGS___ct6, __VA_ARGS___ct7,          \
                                    item_ct1,                                  \
                                    dpct_local_acc_ct1                         \
                                        .get_multi_ptr<                        \
                                            sycl::access::decorated::no>()     \
                                        .get(),                                \
                                    staging_acc_ct1                            \
                                        .get_multi_ptr<                        \
                                            sycl::access::decorated::no>()     \
                                        .get());                               \
                            });                                                \
                });                                                            \
}}                                                                             \
    } break;                                                                   \
    case DType::kBFloat16: {                                                   \
        using type = bf16;                                                     \
            { {                                                                \
                dpct::has_capability_or_fail(__VA_ARGS__->get_device(),        \
                                             {sycl::aspect::fp16});            \
                                                                               \
                __VA_ARGS__->submit([&](sycl::handler &cgh) {                  \
                    /*                                                         \
                    DPCT1083:554: The size of local memory in the migrated     \
                    code may be different from the original code. Check that   \
                    the allocated memory size in the migrated code is correct. \
                    */                                                         \
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(       \
                        sycl::range<1>(__VA_ARGS__), cgh);                     \
                    sycl::local_accessor<float, 1> staging_acc_ct1(            \
                        sycl::range<1>(cast_transpose_num_threads /            \
                                       THREADS_PER_WARP),                      \
                        cgh);                                                  \
                                                                               \
                    const InputType *__VA_ARGS___ct0 = __VA_ARGS__;            \
                    OutputType *__VA_ARGS___ct1 = __VA_ARGS__;                 \
                    OutputType *__VA_ARGS___ct2 = __VA_ARGS__;                 \
                    const fp32 *__VA_ARGS___ct3 = __VA_ARGS__;                 \
                    fp32 *__VA_ARGS___ct4 = __VA_ARGS__;                       \
                    const size_t __VA_ARGS___ct5 = __VA_ARGS__;                \
                    const size_t __VA_ARGS___ct6 = __VA_ARGS__;                \
                    const size_t __VA_ARGS___ct7 = __VA_ARGS__;                \
                                                                               \
                    cgh.parallel_for(                                          \
                        sycl::nd_range<3>(                                     \
                            sycl::range<3>(1, 1, __VA_ARGS__) *                \
                                sycl::range<3>(1, 1, __VA_ARGS__),             \
                            sycl::range<3>(1, 1, __VA_ARGS__)),                \
                        [=](sycl::nd_item<3> item_ct1)                         \
                            [[intel::reqd_sub_group_size(32)]] {               \
                                cast_transpose_kernel<, , , , >(               \
                                    __VA_ARGS___ct0, __VA_ARGS___ct1,          \
                                    __VA_ARGS___ct2, __VA_ARGS___ct3,          \
                                    __VA_ARGS___ct4, __VA_ARGS___ct5,          \
                                    __VA_ARGS___ct6, __VA_ARGS___ct7,          \
                                    item_ct1,                                  \
                                    dpct_local_acc_ct1                         \
                                        .get_multi_ptr<                        \
                                            sycl::access::decorated::no>()     \
                                        .get(),                                \
                                    staging_acc_ct1                            \
                                        .get_multi_ptr<                        \
                                            sycl::access::decorated::no>()     \
                                        .get());                               \
                            });                                                \
                });                                                            \
}}                                                                             \
    } break;                                                                   \
    case DType::kFloat8E5M2:                                                   \
    case DType::kFloat8E4M3: {                                                 \
        NVTE_ERROR("FP8 type not instantiated for input.");                    \
    } break;                                                                   \
    default:                                                                   \
        NVTE_ERROR("Invalid type.");                                           \
    }

#define TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(dtype, type, ...)                 \
  switch (dtype)                                                               \
    {                                                                          \
    using namespace transformer_engine;                                        \
    case DType::kFloat16:                                                      \
      {                                                                        \
          using type = fp16;                                                   \
          __VA_ARGS__;                                                         \
          break;                                                               \
      }                                                                        \
    case DType::kBFloat16:                                                     \
      {                                                                        \
          using type = bf16;                                                   \
          __VA_ARGS__;                                                         \
          break;                                                               \
      }                                                                        \
    default:                                                                   \
          NVTE_ERROR("Invalid type for 16 bit.");                              \
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

#define NVTE_API_CALL(api_name) \
  transformer_engine::nvtx::NVTXWrapper _ ## api_name ## _nvtx_wrapper(#api_name);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_COMMON_H_
