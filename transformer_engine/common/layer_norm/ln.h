/*************************************************************************
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_H_
#define TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_H_

#include <transformer_engine/transformer_engine.h>
#include <functional>
#include <map>
#include <stdexcept>
#include <vector>
#include <unordered_map>

#include "../common.h"

namespace transformer_engine {
namespace layer_norm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Params>
struct LaunchParams{
    size_t workspace_bytes;
    size_t barrier_size;

    int multiprocessorCount;
    cudaStream_t stream;

    Params params;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ParamsBase {
    ParamsBase()
        : ctas_per_col(0)
        , rows(0)
        , cols(0)
        , x(nullptr)
        , mu(nullptr)
        , rs(nullptr)
        , gamma(nullptr)
        , workspace(nullptr)
        , barrier(nullptr)
        , zero_centered_gamma(false) {}


    // For Multi-CTA, number of different CTA groups. Otherwise same as gridDim.x.
    int ctas_per_col;
    // Size of CTA group.
    int ctas_per_row;

    // Input is interpreted as matrix. We normalize across columns.
    int rows;
    int cols;

    // Common data pointers.
    void *x;
    void *mu;
    void *rs;
    void *gamma;

    // Multi-CTA workspace in gmem.
    void *workspace;

    // Multi-CTA sync barriers in gmem.
    int *barrier;

    // Whether gamma is centered around 0
    bool zero_centered_gamma;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct FwdParams : public ParamsBase {
    FwdParams()
        : ParamsBase()
        , z(nullptr)
        , beta(nullptr)
        , epsilon(0.f)
        , fp8_out(false) {}

    // Output of LN FWD.
    void *z;
    void *beta;
    float epsilon;

    // Scaling factor
    void *scale;

    // AMax output
    void *amax;

    // Whether to compute scale and amax
    bool fp8_out;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct BwdParams : public ParamsBase {
    BwdParams()
        : ParamsBase()
        , dz(nullptr)
        , dbeta_part(nullptr)
        , dgamma_part(nullptr)
        , dx(nullptr)
        , dbeta(nullptr)
        , dgamma(nullptr) {}

    // Input: gradient wrt. LN FWD output.
    void *dz;

    // Workspace for Wgrad pre-reduction.
    void *dbeta_part;
    void *dgamma_part;

    // Output: Dgrad.
    void *dx;
    // Output: Wgrad.
    void *dbeta;
    void *dgamma;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

using FwdFunction = std::function<void(LaunchParams<FwdParams>&, const bool)>;
using BwdFunction = std::function<void(LaunchParams<BwdParams>&, const bool)>;
using FunctionKey = uint64_t;
using FwdTunedRegistry = std::unordered_map<FunctionKey, FwdFunction>;
using BwdTunedRegistry = std::unordered_map<FunctionKey, BwdFunction>;
using FwdGeneralRegistry = std::unordered_map<FunctionKey, std::map<uint64_t, FwdFunction>>;
using BwdGeneralRegistry = std::unordered_map<FunctionKey, std::map<uint64_t, BwdFunction>>;

extern FwdTunedRegistry FWD_TUNED_FUNCS;
extern BwdTunedRegistry BWD_TUNED_FUNCS;
extern FwdGeneralRegistry FWD_GENERAL_FUNCS;
extern BwdGeneralRegistry BWD_GENERAL_FUNCS;

////////////////////////////////////////////////////////////////////////////////////////////////////

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

template<typename T, int S>
struct Type2Key{
    constexpr static uint32_t Value = TypeId<T>::Value << S;
};

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

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct FwdTunedRegistrar{
    explicit FwdTunedRegistrar(FwdFunction f){
        uint64_t key = Types2Key<W, I, O, C>::get(HIDDEN_SIZE);
        FWD_TUNED_FUNCS.insert({ key, f });
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct FwdGeneralRegistrar{
    explicit FwdGeneralRegistrar(FwdFunction f){
        uint64_t key = Types2Key<W, I, O, C>::get(0);
        FWD_GENERAL_FUNCS[key].insert({ HIDDEN_SIZE, f });
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct BwdTunedRegistrar{
    explicit BwdTunedRegistrar(BwdFunction f){
        uint64_t key = Types2Key<W, I, O, C>::get(HIDDEN_SIZE);
        BWD_TUNED_FUNCS.insert({ key, f });
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct BwdGeneralRegistrar{
    explicit BwdGeneralRegistrar(BwdFunction f){
        uint64_t key = Types2Key<W, I, O, C>::get(0);
        BWD_GENERAL_FUNCS[key].insert({ HIDDEN_SIZE, f });
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

layer_norm::BwdFunction & get_bwd_launcher(DType wtype,
                                           DType itype,
                                           DType otype,
                                           DType ctype,
                                           uint32_t hidden_size);

//////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace layer_norm
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_H_
