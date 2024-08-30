/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_H_
#define TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_H_

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>
#include <transformer_engine/transformer_engine.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../common.h"

namespace transformer_engine {

template <typename Params>
struct LaunchParams {
  size_t workspace_bytes;
  size_t barrier_size;

  int multiprocessorCount;
  cudaStream_t stream;

  Params params;
};

struct ParamsBase {
  ParamsBase()
      : ctas_per_col(0),
        rows(0),
        cols(0),
        x(nullptr),
        mu(nullptr),
        rs(nullptr),
        gamma(nullptr),
        workspace(nullptr),
        barrier(nullptr),
        zero_centered_gamma(false) {}

  // For Multi-CTA, number of different CTA groups. Otherwise same as gridDim.x.
  int ctas_per_col;
  // Size of CTA group.
  int ctas_per_row;

  // Input is interpreted as matrix. We normalize across columns.
  int rows;
  int cols;

  // Common data pointers.
  void* x;
  void* mu;
  void* rs;
  void* gamma;

  // Multi-CTA workspace in gmem.
  void* workspace;

  // Multi-CTA sync barriers in gmem.
  int* barrier;

  // Whether gamma is centered around 0
  bool zero_centered_gamma;
};

struct FwdParams : public ParamsBase {
  FwdParams() : ParamsBase(), z(nullptr), beta(nullptr), epsilon(0.f), fp8_out(false) {}

  // Output of LN FWD.
  void* z;
  void* beta;
  float epsilon;

  // Scaling factor
  void* scale;
  int scale_byte_size;

  // Inverse of scaling factor
  void* scale_inv;

  // AMax output
  void* amax;
  int amax_byte_size;

  // Whether to compute scale and amax
  bool fp8_out;
};

struct BwdParams : public ParamsBase {
  BwdParams()
      : ParamsBase(),
        dz(nullptr),
        dbeta_part(nullptr),
        dgamma_part(nullptr),
        dx(nullptr),
        dbeta(nullptr),
        dgamma(nullptr) {}

  // Input: gradient wrt. LN FWD output.
  void* dz;

  // Workspace for Wgrad pre-reduction.
  void* dbeta_part;
  void* dgamma_part;

  // Output: Dgrad.
  void* dx;
  // Output: Wgrad.
  void* dbeta;
  void* dgamma;
};

enum NVTE_NORM_TYPE {
  LN_FWD_TE,
  LN_BWD_TE,
  LN_FWD_CUDNN,
  LN_BWD_CUDNN,
  RMS_FWD_TE,
  RMS_BWD_TE,
  RMS_FWD_CUDNN,
  RMS_BWD_CUDNN,
};

template <NVTE_NORM_TYPE NormEnum>
constexpr bool IF_TE_NORMS() {
  return (NormEnum == NVTE_NORM_TYPE::LN_FWD_TE || NormEnum == NVTE_NORM_TYPE::LN_BWD_TE ||
          NormEnum == NVTE_NORM_TYPE::RMS_FWD_TE || NormEnum == NVTE_NORM_TYPE::RMS_BWD_TE);
};

template <NVTE_NORM_TYPE NormEnum>
constexpr bool IF_TE_FWD_NORMS() {
  return (NormEnum == NVTE_NORM_TYPE::LN_FWD_TE || NormEnum == NVTE_NORM_TYPE::RMS_FWD_TE);
};

template <NVTE_NORM_TYPE NormEnum>
constexpr bool IF_TE_BWD_NORMS() {
  return (NormEnum == NVTE_NORM_TYPE::LN_BWD_TE || NormEnum == NVTE_NORM_TYPE::RMS_BWD_TE);
};

template <NVTE_NORM_TYPE NormEnum>
constexpr bool IF_CUDNN_NORMS() {
  return (NormEnum == NVTE_NORM_TYPE::LN_FWD_CUDNN || NormEnum == NVTE_NORM_TYPE::LN_BWD_CUDNN ||
          NormEnum == NVTE_NORM_TYPE::RMS_FWD_CUDNN || NormEnum == NVTE_NORM_TYPE::RMS_BWD_CUDNN);
};

template <NVTE_NORM_TYPE NormEnum>
constexpr bool IF_CUDNN_FWD_NORMS() {
  return (NormEnum == NVTE_NORM_TYPE::LN_FWD_CUDNN || NormEnum == NVTE_NORM_TYPE::RMS_FWD_CUDNN);
};

template <NVTE_NORM_TYPE NormEnum>
constexpr bool IF_CUDNN_BWD_NORMS() {
  return (NormEnum == NVTE_NORM_TYPE::LN_BWD_CUDNN || NormEnum == NVTE_NORM_TYPE::RMS_BWD_CUDNN);
};

using FwdFunction = std::function<void(LaunchParams<FwdParams>&, const bool)>;
using BwdFunction = std::function<void(LaunchParams<BwdParams>&, const bool)>;
using FunctionKey = uint64_t;
using FwdTunedRegistry = std::unordered_map<FunctionKey, FwdFunction>;
using BwdTunedRegistry = std::unordered_map<FunctionKey, BwdFunction>;
using FwdGeneralRegistry = std::unordered_map<FunctionKey, std::map<uint64_t, FwdFunction>>;
using BwdGeneralRegistry = std::unordered_map<FunctionKey, std::map<uint64_t, BwdFunction>>;

template <NVTE_NORM_TYPE NormEnum, typename Enable = void>
struct LauncherType;

template <NVTE_NORM_TYPE NormEnum>
struct LauncherType<NormEnum, typename std::enable_if<IF_TE_NORMS<NormEnum>()>::type> {
  using ParamsType = std::conditional_t<IF_TE_FWD_NORMS<NormEnum>(), LaunchParams<FwdParams>,
                                        LaunchParams<BwdParams>>;
  using FunctionType = std::conditional_t<IF_TE_FWD_NORMS<NormEnum>(), FwdFunction, BwdFunction>;
};

extern FwdTunedRegistry LN_FWD_TUNED_FUNCS;
extern BwdTunedRegistry LN_BWD_TUNED_FUNCS;
extern FwdGeneralRegistry LN_FWD_GENERAL_FUNCS;
extern BwdGeneralRegistry LN_BWD_GENERAL_FUNCS;

extern FwdTunedRegistry RMS_FWD_TUNED_FUNCS;
extern BwdTunedRegistry RMS_BWD_TUNED_FUNCS;
extern FwdGeneralRegistry RMS_FWD_GENERAL_FUNCS;
extern BwdGeneralRegistry RMS_BWD_GENERAL_FUNCS;

template <NVTE_NORM_TYPE NormEnum, bool IF_TUNED, typename Enable = void>
struct RegistryType {};

template <NVTE_NORM_TYPE NormEnum, bool IF_TUNED>
struct RegistryType<NormEnum, IF_TUNED, typename std::enable_if<IF_TE_NORMS<NormEnum>()>::type> {
  using type = std::conditional_t<
      IF_TUNED, std::conditional_t<IF_TE_FWD_NORMS<NormEnum>(), FwdTunedRegistry, BwdTunedRegistry>,
      std::conditional_t<IF_TE_FWD_NORMS<NormEnum>(), FwdGeneralRegistry, BwdGeneralRegistry>>;
};

template <NVTE_NORM_TYPE NormEnum, bool IF_TUNED>
constexpr typename RegistryType<NormEnum, IF_TUNED>::type& GET_REGISTRY() {
  if constexpr (!IF_TE_NORMS<NormEnum>()) NVTE_ERROR("Unexpected NVTE_NORM_TYPE!");
  if constexpr (IF_TUNED) {
    if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_TE)
      return LN_FWD_TUNED_FUNCS;
    else if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_TE)
      return LN_BWD_TUNED_FUNCS;
    else if constexpr (NormEnum == NVTE_NORM_TYPE::RMS_FWD_TE)
      return RMS_FWD_TUNED_FUNCS;
    else if constexpr (NormEnum == NVTE_NORM_TYPE::RMS_BWD_TE)
      return RMS_BWD_TUNED_FUNCS;
  } else {
    if constexpr (NormEnum == NVTE_NORM_TYPE::LN_FWD_TE)
      return LN_FWD_GENERAL_FUNCS;
    else if constexpr (NormEnum == NVTE_NORM_TYPE::LN_BWD_TE)
      return LN_BWD_GENERAL_FUNCS;
    else if constexpr (NormEnum == NVTE_NORM_TYPE::RMS_FWD_TE)
      return RMS_FWD_GENERAL_FUNCS;
    else if constexpr (NormEnum == NVTE_NORM_TYPE::RMS_BWD_TE)
      return RMS_BWD_GENERAL_FUNCS;
  }
};

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

template <>
struct TypeId<fp8e4m3> {
  constexpr static uint32_t Value = 3;
};

template <typename T, int S>
struct Type2Key {
  constexpr static uint32_t Value = TypeId<T>::Value << S;
};

template <typename T>
struct WeightType2Key : public Type2Key<T, 0> {};

template <typename T>
struct InputType2Key : public Type2Key<T, 2> {};

template <typename T>
struct OutputType2Key : public Type2Key<T, 4> {};

template <typename T>
struct ComputeType2Key : public Type2Key<T, 6> {};

template <typename W, typename I, typename O, typename C>
struct Types2Key {
  constexpr static uint32_t Value = WeightType2Key<W>::Value | InputType2Key<I>::Value |
                                    OutputType2Key<O>::Value | ComputeType2Key<C>::Value;
  constexpr static inline uint64_t get(const uint64_t hidden_size) {
    constexpr uint64_t type_key = Value;
    return (type_key << 32) | hidden_size;
  }
};

template <typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE,
          NVTE_NORM_TYPE NormEnum, bool IF_TUNED, typename Enable = void>
struct NormRegistrar {};

template <typename W, typename I, typename O, typename C, uint64_t HIDDEN_SIZE,
          NVTE_NORM_TYPE NormEnum, bool IF_TUNED>
struct NormRegistrar<W, I, O, C, HIDDEN_SIZE, NormEnum, IF_TUNED,
                     typename std::enable_if<IF_TE_NORMS<NormEnum>()>::type> {
  explicit NormRegistrar(typename LauncherType<NormEnum>::FunctionType f) {
    auto& registry = GET_REGISTRY<NormEnum, IF_TUNED>();
    if constexpr (IF_TUNED) {
      uint64_t key = Types2Key<W, I, O, C>::get(HIDDEN_SIZE);
      registry.insert({key, f});
    } else {
      uint64_t key = Types2Key<W, I, O, C>::get(0);
      registry[key].insert({HIDDEN_SIZE, f});
    }
  }
};

class NormBase {
 public:
  virtual void initialize() = 0;

  virtual void execute() = 0;

  virtual void set_workspace_and_barrier(void* workspace_ptr, void* barrier_ptr) {};

  virtual std::vector<size_t> get_workspace_shape() { return {0}; };

  virtual std::vector<size_t> get_barrier_shape() { return {0}; };
};

template <NVTE_NORM_TYPE NormEnum>
class NormFwdTe : public NormBase {
 public:
  NormFwdTe();

  NormFwdTe(const Tensor& x, const Tensor& gamma, const Tensor& beta, const float epsilon,
            Tensor* z, Tensor* mu, Tensor* rsigma, cudaStream_t stream,
            const int multiprocessorCount, Tensor* workspace, Tensor* barrier,
            const bool zero_centered_gamma);

  void initialize() override;

  void set_workspace_and_barrier(void* workspace_ptr, void* barrier_ptr) override;

  void set_amax();

  void execute() override;

  std::vector<size_t> get_workspace_shape() override;

  std::vector<size_t> get_barrier_shape() override;

  typename LauncherType<NormEnum>::ParamsType _launch_params;

  typename LauncherType<NormEnum>::FunctionType _launcher;
};

template <NVTE_NORM_TYPE NormEnum>
class NormBwdTe : public NormFwdTe<NormEnum> {
 public:
  NormBwdTe(const Tensor& dz, const Tensor& x, const Tensor& mu, const Tensor& rsigma,
            const Tensor& gamma, Tensor* dx, Tensor* dgamma, Tensor* dbeta, Tensor* dgamma_part,
            Tensor* dbeta_part, cudaStream_t stream, const int multiprocessorCount,
            Tensor* workspace, Tensor* barrier, const bool zero_centered_gamma);

  std::vector<size_t> get_dgamma_shape();
};

template <NVTE_NORM_TYPE NormEnum>
class NormFwdCudnn : public NormBase {
 public:
  NormFwdCudnn();

  NormFwdCudnn(const Tensor& x, const Tensor& gamma, const Tensor& beta, const float epsilon,
               Tensor* z, Tensor* mu, Tensor* rsigma, cudaStream_t stream,
               const int multiprocessorCount, Tensor* workspace, const bool zero_centered_gamma);

  void initialize() override;

  void set_workspace_and_barrier(void* workspace_ptr, void* barrier_ptr) override;

  // Launch the kernel.
  void execute() override;

  std::vector<size_t> get_workspace_shape() override;

  ~NormFwdCudnn() { cudnnDestroy(_handle); }

  cudnn_frontend::graph::Graph _graph;
  std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*>
      _variant_pack;
  cudnnHandle_t _handle;
  void* _workspace;

  float _epsilon;
  std::unique_ptr<char[]> _scalar_offset;
};

template <NVTE_NORM_TYPE NormEnum>
class NormBwdCudnn : public NormFwdCudnn<NormEnum> {
 public:
  NormBwdCudnn(const Tensor& dz, const Tensor& x, const Tensor& mu, const Tensor& rsigma,
               const Tensor& gamma, Tensor* dx, Tensor* dgamma, Tensor* dbeta, cudaStream_t stream,
               const int multiprocessorCount, Tensor* workspace, const bool zero_centered_gamma);
};

template <NVTE_NORM_TYPE NormEnum, typename NormType>
void norms_launcher(NormType& Norm, Tensor* workspace, Tensor* barrier = nullptr,
                    Tensor* dgamma_part = nullptr, Tensor* dbeta_part = nullptr);

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_LAYER_NORM_LN_H_
