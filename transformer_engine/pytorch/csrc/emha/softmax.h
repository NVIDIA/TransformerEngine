#pragma once
/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

#include <ATen/cuda/CUDAGraphsUtils.cuh>
#include <unordered_map>

namespace softmax {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum MaskMode { SELF = 0, CAUSAL = 1 };

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Params>
struct LaunchParams {
  cudaDeviceProp *props;

  cudaStream_t stream;

  Params params;

  MaskMode mask_mode;

  int elts_per_thread;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ParamsBase {
  ParamsBase() : b(0), h(0), sq(0), sk(0), p_keep(0.f) {}

  // Problem dimensions.
  int b;
  int h;
  int sq;
  int sk;

  // Scale to multiply the input before softmax.
  float scale_pre_softmax;

  // 1 - p_dropout
  float p_keep;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FwdParams : public ParamsBase {
  FwdParams() : ParamsBase(), z(nullptr), cu_seqlens(nullptr), x(nullptr) {}

  // Output of FWD.
  void *z;

  // Common data pointers.
  void *x;

  // Sequence offsets for masked self attention.
  int *cu_seqlens;

  // PRNG State.
  at::PhiloxCudaState philox_args;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct BwdParams : public ParamsBase {
  BwdParams() : ParamsBase(), dz(nullptr), smat_dmask(nullptr), dx(nullptr) {}

  // Input: gradient wrt. FWD output.
  void *dz;

  // Input: S and dmask
  void *smat_dmask;

  // Output: Dgrad.
  void *dx;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using FwdFunction = std::function<void(LaunchParams<FwdParams> &, const bool)>;
using BwdFunction = std::function<void(LaunchParams<BwdParams> &, const bool)>;
using FunctionKey = uint64_t;
using FwdRegistry = std::unordered_map<FunctionKey, FwdFunction>;
using BwdRegistry = std::unordered_map<FunctionKey, BwdFunction>;

extern FwdRegistry FWD_FUNCS;
extern BwdRegistry BWD_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct TypeId {};

template <>
struct TypeId<fp16> {
  constexpr static uint32_t Value = 0;
};

template <>
struct TypeId<bf16> {
  constexpr static uint32_t Value = 1;
};

template <>
struct TypeId<fp32> {
  constexpr static uint32_t Value = 2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int S>
struct Type2Key {
  constexpr static uint32_t Value = TypeId<T>::Value << S;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct InputType2Key : public Type2Key<T, 0> {};

template <typename T>
struct OutputType2Key : public Type2Key<T, 2> {};

template <typename T>
struct ComputeType2Key : public Type2Key<T, 4> {};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename I, typename O, typename C>
struct Types2Key {
  constexpr static uint32_t Value = InputType2Key<I>::Value |
                                    OutputType2Key<O>::Value |
                                    ComputeType2Key<C>::Value;
  constexpr static inline uint64_t get(const uint64_t hidden_size,
                                       const MaskMode mask_mode) {
    constexpr uint64_t type_key = Value;
    const uint64_t mask_mode_(mask_mode);
    return (((type_key << 32) | hidden_size) << 2) | mask_mode_;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename I, typename O, typename C, uint64_t HIDDEN_SIZE,
          MaskMode MASK_MODE>
struct FwdRegistrar {
  explicit FwdRegistrar(FwdFunction f) {
    uint64_t key = Types2Key<I, O, C>::get(HIDDEN_SIZE, MASK_MODE);
    FWD_FUNCS.insert({key, f});
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct BwdRegistrar {
  explicit BwdRegistrar(BwdFunction f) {
    uint64_t key = Types2Key<I, O, C>::get(HIDDEN_SIZE, SELF);
    BWD_FUNCS.insert({key, f});
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace softmax
