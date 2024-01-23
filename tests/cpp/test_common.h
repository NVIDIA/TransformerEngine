/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once

#include <iostream>
#include <memory>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime_api.h>

#include <transformer_engine/transformer_engine.h>
#include "util/logging.h"

namespace test {
using namespace transformer_engine;

template <size_t i>
struct BytesToType {};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
};

using byte = uint8_t;
using int32 = int32_t;
using int64 = int64_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

template <typename T>
struct TypeInfo{
    using types = std::tuple<byte,
                             int32,
                             int64,
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

class Tensor {
 public:
  Tensor(const NVTEShape &shape, const DType type);

  Tensor(const std::vector<size_t> &shape, const DType type) :
    Tensor(NVTEShape{shape.data(), shape.size()}, type) {}

  Tensor() {}

  Tensor& operator=(const Tensor &other) = delete;
  Tensor(const Tensor &other) = delete;

  Tensor(Tensor &&other) = default;
  Tensor& operator=(Tensor &&other) = default;

  ~Tensor() {
    if (tensor_.dptr() != nullptr) {
      cudaFree(tensor_.dptr());
    }
  }
  NVTETensor data() const noexcept {
    return tensor_.data();
  }

  const NVTEShape shape() const noexcept {
    return tensor_.shape();
  }

  DType dtype() const noexcept {
    return tensor_.dtype();
  }

  void *dptr() const noexcept {
    return tensor_.dptr();
  }

  template <typename T>
  T *cpu_dptr() const {
    NVTE_CHECK(TypeInfo<T>::dtype == tensor_.dtype(), "Invalid type!");
    return reinterpret_cast<T *>(cpu_data_.get());
  }

  float amax() const {
    if(amax_cpu_data_) {
      to_cpu();
      return *amax_cpu_data_;
    } else {
      return 0;
    }
  }

  float scale() const {
    if(scale_cpu_data_) {
      to_cpu();
      return *scale_cpu_data_;
    } else {
      return 1;
    }
  }

  float scale_inv() const {
    if(scale_inv_cpu_data_) {
      to_cpu();
      return *scale_inv_cpu_data_;
    } else {
      return 1;
    }
  }

  void to_cpu() const;
  void from_cpu() const;
  void set_scale(float scale);
  void set_scale_inv(float scale_inv);
  void shareFP8Meta(const Tensor &other);

 private:
  TensorWrapper tensor_;
  std::unique_ptr<unsigned char[]> cpu_data_;
  std::shared_ptr<float> amax_cpu_data_;
  std::shared_ptr<float> scale_cpu_data_;
  std::shared_ptr<float> scale_inv_cpu_data_;
};

size_t typeToSize(DType type);
size_t product(const NVTEShape &shape);

bool areShapesEqual(const NVTEShape &s1, const NVTEShape &s2);

void compareResults(const std::string &name, const Tensor &test, const void *ref,
                    double atol = 1e-5, double rtol = 1e-8);
void compareResults(const std::string &name, const float test, const float ref,
                    double atol = 1e-5, double rtol = 1e-8);

std::pair<double, double> getTolerances(const DType type);

void fillUniform(Tensor *t);
void setRandomScale(Tensor *t);

constexpr int THREADS_PER_WARP = 32;

const std::string &typeName(DType type);

extern std::vector<DType> all_fp_types;

bool isFp8Type(DType type);

}  // namespace test

#define TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(dtype, type, ...) \
    switch (dtype) { \
        using namespace transformer_engine; \
        case DType::kByte: \
            { \
                using type = byte; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kInt32: \
            { \
                using type = int32; \
                {__VA_ARGS__} \
            } \
        break; \
        case DType::kInt64: \
            { \
                using type = int64; \
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
