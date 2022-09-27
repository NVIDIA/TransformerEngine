/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_COMMON_H_
#define TRANSFORMER_ENGINE_COMMON_COMMON_H_

#include <transformer_engine/transformer_engine.h>
#include <transformer_engine/logging.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>
#include <type_traits>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace transformer_engine {

struct Tensor {
    void* dptr;
    std::vector<size_t> shape;
    DType dtype;

    Tensor() : dptr(nullptr), shape(), dtype(DType::kFloat32) {}
};

template <typename T>
constexpr T DIVUP(const T &x, const T &y) {
    return (((x) + ((y)-1)) / (y));
}

using byte = uint8_t;
using int32 = int32_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;


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
};

#define TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kByte: \
            { \
                using type = float; \
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

#define TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, type, ...) \
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
        case DType::kFloat8E4M3: \
            { \
                NVTE_ERROR("FP8 type not instantiated for input."); \
            } \
        break; \
        default: \
            NVTE_ERROR("Invalid type."); \
    }

template<typename T>
struct TypeId{};

template<>
struct TypeId<fp16>{
    constexpr static uint32_t Value = 0;
};

template<>
struct TypeId<bf16>{
    constexpr static uint32_t Value = 1;
};

template<>
struct TypeId<fp32>{
    constexpr static uint32_t Value = 2;
};

template<>
struct TypeId<fp8e4m3>{
    constexpr static uint32_t Value = 3;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T, int S>
struct Type2Key{
    constexpr static uint32_t Value = TypeId<T>::Value << S;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct WeightType2Key : public Type2Key<T, 0>{};

template<typename T>
struct InputType2Key : public Type2Key<T, 2>{};

template<typename T>
struct OutputType2Key : public Type2Key<T, 4>{};

template<typename T>
struct ComputeType2Key : public Type2Key<T, 6>{};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C>
struct Types2Key{
    constexpr static uint32_t Value = WeightType2Key<W>::Value | InputType2Key<I>::Value |
                                      OutputType2Key<O>::Value | ComputeType2Key<C>::Value;
    constexpr static inline uint64_t get(const uint64_t hidden_size){
        constexpr uint64_t type_key = Value;
        return (type_key << 32) | hidden_size;
    }
};

inline size_t product(const std::vector<size_t> &shape) {
    size_t ret = 1;
    for (const auto &elem : shape) {
        ret *= elem;
    }
    return ret;
}

template <typename T>
struct is_fp8 : std::false_type {};

template <>
struct is_fp8<fp8e4m3> : std::true_type {};

template <>
struct is_fp8<fp8e5m2> : std::true_type {};

size_t typeToSize(const DType type);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_COMMON_H_
