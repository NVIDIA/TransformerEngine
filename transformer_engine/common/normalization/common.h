/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_NORM_COMMON_H_
#define TRANSFORMER_ENGINE_COMMON_NORM_COMMON_H_

#include <cudnn.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>
#include <transformer_engine/transformer_engine.h>

#include <functional>
#include <map>
#include <stdexcept>
#include <tuple>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/system.h"

namespace transformer_engine {

namespace normalization {

namespace fe = cudnn_frontend;

template <typename KernelParamsType>
struct LaunchParams {
  size_t workspace_bytes = 0;
  size_t barrier_bytes = 0;
  size_t dgamma_part_bytes = 0;
  int multiprocessorCount;
  cudaStream_t stream;

  KernelParamsType params;

  size_t getTotalWorkspaceBytes(const bool _is_layernorm = true) const {
    return (workspace_bytes + barrier_bytes + size_t(_is_layernorm + 1) * dgamma_part_bytes);
  }
  void alignWorkspace(size_t alignment = 16) {
    workspace_bytes = DIVUP(workspace_bytes, alignment) * alignment;
    barrier_bytes = DIVUP(barrier_bytes, alignment) * alignment;
    dgamma_part_bytes = DIVUP(dgamma_part_bytes, alignment) * alignment;
  }
};

struct KernelParamsBase {
  KernelParamsBase()
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

struct ForwardKernelParams : public KernelParamsBase {
  ForwardKernelParams()
      : KernelParamsBase(), z(nullptr), beta(nullptr), epsilon(0.f), fp8_out(false) {}

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

struct BackwardKernelParams : public KernelParamsBase {
  BackwardKernelParams()
      : KernelParamsBase(),
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

enum class NVTE_Norm_Backend { Te, Cudnn };
enum class NVTE_Norm_Type { LayerNorm, RMSNorm };
enum class NVTE_Norm_Stage { Forward, Backward };

using TupleKeyType = std::tuple<uint64_t, uint64_t, uint64_t, bool>;
struct TupleHash {
  size_t operator()(const TupleKeyType& t) const {
    // Generate a hash for a tuple by combining the hashes of its entries
    // See: https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
    size_t seed = 0;
    std::hash<uint64_t> hasher;
    seed ^= hasher(std::get<0>(t)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hasher(std::get<1>(t)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    seed ^= hasher(std::get<2>(t)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    return seed;
  }
};

// Note: the default mode here should match with the default mode with QTensor
TupleKeyType get_key(NVTE_Norm_Backend NormBackend, NVTE_Norm_Type NormType,
                     NVTE_Norm_Stage NormStage, DType wtype, DType itype, DType otype, DType ctype,
                     uint64_t batch_size, uint64_t hidden_size, bool zero_centered_gamma,
                     bool is_tuned, NVTEScalingMode mode = NVTE_DELAYED_TENSOR_SCALING,
                     bool training = true);

template <typename KernelParamsType>
class TeNormalizationRegistry {
 private:
  using Function = std::function<void(LaunchParams<KernelParamsType>&, const bool)>;
  std::unordered_map<TupleKeyType, Function, TupleHash> tuned_function_map;
  std::unordered_map<uint64_t, std::map<uint64_t, Function>> general_function_map;

  TeNormalizationRegistry() = default;

  static TeNormalizationRegistry& getInstance() {
    static TeNormalizationRegistry registry;
    return registry;
  }

 public:
  static int registerFunction(TupleKeyType key,
                              void (*func)(LaunchParams<KernelParamsType>&, const bool)) {
    auto [general_key, batch_size, hidden_size, is_tuned] = key;
    if (is_tuned)
      getInstance().tuned_function_map.emplace(key, Function(func));
    else
      getInstance().general_function_map[general_key].emplace(hidden_size, Function(func));
    return 0;
  }

  static Function getKernel(TupleKeyType key) {
    auto& instance = getInstance();
    auto [general_key, batch_size, hidden_size, is_tuned] = key;
    if (is_tuned) {
      auto it = instance.tuned_function_map.find(key);
      if (it != instance.tuned_function_map.end()) return it->second;
    }
    if (instance.general_function_map.count(general_key) == 0) {
      NVTE_ERROR("Unavailable kernel for this normalization config.");
    }
    auto& general_func_map = instance.general_function_map.at(general_key);
    auto func_iter = general_func_map.lower_bound(hidden_size);
    if (func_iter == general_func_map.end()) {
      return general_func_map.rbegin()->second;  // Hidden size is too big, need to use multi-CTA
    } else {
      return func_iter->second;
    }
  }

  TeNormalizationRegistry(const TeNormalizationRegistry&) = delete;
  TeNormalizationRegistry& operator=(const TeNormalizationRegistry&) = delete;
  TeNormalizationRegistry(TeNormalizationRegistry&&) = delete;
  TeNormalizationRegistry& operator=(TeNormalizationRegistry&&) = delete;
};

class NormalizationPlanBase {
 public:
  virtual ~NormalizationPlanBase() = default;
  virtual std::vector<size_t> getWorkspaceShape() const = 0;

  virtual void execute(Tensor* z, void* x_dptr, void* gamma_dptr, void* beta_dptr, void* mean_dptr,
                       void* eps_dptr, void* rsigma_dptr, void* workspace_dptr,
                       cudaStream_t stream) = 0;

  virtual void execute(void* x_dptr, void* gamma_dptr, void* mean_dptr, void* rsigma_dptr,
                       void* dx_dptr, void* dz_dptr, void* dbeta_dptr, void* dgamma_dptr,
                       void* workspace_dptr, cudaStream_t stream) = 0;

 private:
  virtual void _build() = 0;
};

template <typename KernelParamsType>
class TeNormalizationPlan : public NormalizationPlanBase {
 public:
  TeNormalizationPlan(NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage, DType wtype, DType itype,
                      DType otype, DType ctype, const size_t batch_size, const size_t hidden_size,
                      const size_t sm_count, const bool zero_centered_gamma, const bool is_tuned);
  std::vector<size_t> getWorkspaceShape() const override;

  void execute(Tensor* z, void* x_dptr, void* gamma_dptr, void* beta_dptr, void* mean_dptr,
               void* eps_dptr, void* rsigma_dptr, void* workspace_dptr,
               cudaStream_t stream) override;

  void execute(void* x_dptr, void* gamma_dptr, void* mean_dptr, void* rsigma_dptr, void* dx_dptr,
               void* dz_dptr, void* dbeta_dptr, void* dgamma_dptr, void* workspace_dptr,
               cudaStream_t stream) override;

 private:
  void _set_workspace();
  void _build();

  using KernelRegistry = TeNormalizationRegistry<KernelParamsType>;
  LaunchParams<KernelParamsType> _launch_params;
  std::function<void(LaunchParams<KernelParamsType>&, const bool)> _kernel;

  const bool _is_layernorm;
};

class CudnnNormalizationPlan : public NormalizationPlanBase {
 public:
  CudnnNormalizationPlan(NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage, DType wtype,
                         DType itype, DType otype, DType ctype, const size_t batch_size,
                         const size_t hidden_size, const size_t sm_count,
                         const bool zero_centered_gamma, const NVTEScalingMode mode,
                         const bool training);

  std::vector<size_t> getWorkspaceShape() const override;

  void execute(Tensor* z, void* x_dptr, void* gamma_dptr, void* beta_dptr, void* mean_dptr,
               void* eps_dptr, void* rsigma_dptr, void* workspace_dptr,
               cudaStream_t stream) override;

  void execute(void* x_dptr, void* gamma_dptr, void* mean_dptr, void* rsigma_dptr, void* dx_dptr,
               void* dz_dptr, void* dbeta_dptr, void* dgamma_dptr, void* workspace_dptr,
               cudaStream_t stream) override;

 private:
  void _build() override;

  const bool _zero_centered, _fp8_out;
  int _ndim_scale_block;
  const NVTE_Norm_Stage _norm_stage;
  const NVTE_Norm_Type _norm_type;
  std::unique_ptr<char[]> _scalar_dptr;
  std::unique_ptr<float> _one_dptr = std::make_unique<float>(1.0f);
  // FWD
  std::shared_ptr<fe::graph::Tensor_attributes> _x, _gamma_zero, _scalar_offset, _gamma, _beta,
      _eps, _mean, _rsigma, _z, _z_scale, _one_for_div, _z_scale_inv, _amax, _z_fp8;
  // MX FWD
  std::shared_ptr<fe::graph::Tensor_attributes> _z_mx_row, _z_mx_col, _sf_row, _sf_col;
  const bool _training;
  // BWD
  std::shared_ptr<fe::graph::Tensor_attributes> _dz, _dx, _dgamma, _dbeta;

  fe::graph::Graph _graph;
  std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> _variant_pack;
  cudnnHandle_t _handle;
};

class NormalizationPlanRegistry {
 public:
  static NormalizationPlanRegistry& getInstance() {
    static thread_local NormalizationPlanRegistry instance;
    return instance;
  }

  NormalizationPlanBase* getNormalizationPlan(
      NVTE_Norm_Backend NormBackend, NVTE_Norm_Type NormType, NVTE_Norm_Stage NormStage,
      DType wtype, DType itype, DType otype, const size_t batch_size, const size_t hidden_size,
      const size_t sm_count, const bool zero_centered_gamma, const bool is_aligned,
      const NVTEScalingMode mode = NVTE_DELAYED_TENSOR_SCALING, const bool training = true);

 private:
  NormalizationPlanRegistry() {}
  NormalizationPlanRegistry(const NormalizationPlanRegistry&) = delete;
  NormalizationPlanRegistry& operator=(const NormalizationPlanRegistry&) = delete;

  std::unordered_map<TupleKeyType, std::unique_ptr<NormalizationPlanBase>, TupleHash>
      normalizationPlanMap;
};

using byte = uint8_t;
using int32 = int32_t;
using fp32 = float;
using fp16 = half;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

template <typename T>
struct TypeToDType;

template <>
struct TypeToDType<fp32> {
  static constexpr DType value = DType::kFloat32;
};
template <>
struct TypeToDType<fp16> {
  static constexpr DType value = DType::kFloat16;
};
template <>
struct TypeToDType<bf16> {
  static constexpr DType value = DType::kBFloat16;
};
template <>
struct TypeToDType<fp8e4m3> {
  static constexpr DType value = DType::kFloat8E4M3;
};
template <>
struct TypeToDType<fp8e5m2> {
  static constexpr DType value = DType::kFloat8E5M2;
};
template <>
struct TypeToDType<int32> {
  static constexpr DType value = DType::kInt32;
};
template <>
struct TypeToDType<byte> {
  static constexpr DType value = DType::kByte;
};

#define IS_TUNED(x) (strcmp(#x, "tuned") == 0 ? 1 : 0)

// TE kernels have no template for batch_size and zero_centered_gamma, thus zero out those
#define REGISTER_NORM_BASE(NORM_TYPE, NORM_STAGE, LAUNCH_TYPE, HIDDEN_SIZE, WTYPE, ITYPE, OTYPE,                    \
                           CTYPE, FUNC_NAME)                                                                        \
  static int                                                                                                        \
      register_##NORM_TYPE##_##NORM_STAGE##_##LAUNCH_TYPE##_##HIDDEN_SIZE##_##WTYPE##_##ITYPE##_##OTYPE##_##CTYPE = \
          TeNormalizationRegistry<NORM_STAGE##KernelParams>::registerFunction(                                      \
              (get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::NORM_TYPE,                                            \
                       NVTE_Norm_Stage::NORM_STAGE, (TypeToDType<WTYPE>::value),                                    \
                       (TypeToDType<ITYPE>::value), (TypeToDType<OTYPE>::value),                                    \
                       (TypeToDType<CTYPE>::value), 0, HIDDEN_SIZE, 0, IS_TUNED(LAUNCH_TYPE))),                     \
              FUNC_NAME)

// Alignment check
template <size_t Alignment = 16, typename... Args>
bool is_ptr_aligned(const Args*... ptrs) {
  return ((reinterpret_cast<uintptr_t>(ptrs) % Alignment == 0) && ...);
}

bool use_cudnn_norm_fwd();
bool use_cudnn_norm_bwd();

}  // namespace normalization
}  // namespace transformer_engine

#endif
