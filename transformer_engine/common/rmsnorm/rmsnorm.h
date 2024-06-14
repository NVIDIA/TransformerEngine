/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_H_
#define TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_H_

#include <transformer_engine/transformer_engine.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../common.h"
#include "../layer_norm/ln.h"

namespace transformer_engine {
namespace rmsnorm {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Params>
struct LaunchParams : public transformer_engine::layer_norm::LaunchParams<Params> {};
struct FwdParams : public transformer_engine::layer_norm::FwdParams {};
struct BwdParams : public transformer_engine::layer_norm::BwdParams {};

////////////////////////////////////////////////////////////////////////////////////////////////////

using FwdFunction = std::function<void(LaunchParams<FwdParams> &, const bool)>;
using BwdFunction = std::function<void(LaunchParams<BwdParams> &, const bool)>;
using FunctionKey = uint64_t;
using FwdTunedRegistry = std::unordered_map<FunctionKey, FwdFunction>;
using BwdTunedRegistry = std::unordered_map<FunctionKey, BwdFunction>;
using FwdGeneralRegistry = std::unordered_map<FunctionKey, std::map<uint64_t, FwdFunction>>;
using BwdGeneralRegistry = std::unordered_map<FunctionKey, std::map<uint64_t, BwdFunction>>;

extern FwdTunedRegistry FWD_TUNED_FUNCS;
extern BwdTunedRegistry BWD_TUNED_FUNCS;
extern FwdGeneralRegistry FWD_GENERAL_FUNCS;
extern BwdGeneralRegistry BWD_GENERAL_FUNCS;

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct FwdTunedRegistrar {
  explicit FwdTunedRegistrar(FwdFunction f) {
    uint64_t key = layer_norm::Types2Key<W, I, O, C>::get(HIDDEN_SIZE);
    FWD_TUNED_FUNCS.insert({key, f});
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct FwdGeneralRegistrar {
  explicit FwdGeneralRegistrar(FwdFunction f) {
    uint64_t key = layer_norm::Types2Key<W, I, O, C>::get(0);
    FWD_GENERAL_FUNCS[key].insert({HIDDEN_SIZE, f});
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct BwdTunedRegistrar {
  explicit BwdTunedRegistrar(BwdFunction f) {
    uint64_t key = layer_norm::Types2Key<W, I, O, C>::get(HIDDEN_SIZE);
    BWD_TUNED_FUNCS.insert({key, f});
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////

template <typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE>
struct BwdGeneralRegistrar {
  explicit BwdGeneralRegistrar(BwdFunction f) {
    uint64_t key = layer_norm::Types2Key<W, I, O, C>::get(0);
    BWD_GENERAL_FUNCS[key].insert({HIDDEN_SIZE, f});
  }
};

}  // namespace rmsnorm
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_RMSNORM_RMSNORM_H_
