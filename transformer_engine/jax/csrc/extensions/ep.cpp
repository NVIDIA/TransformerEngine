/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef NVTE_WITH_NCCL_EP

#include "transformer_engine/ep.h"

#include <nccl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>

#include "../extensions.h"
#include "common.h"
#include "transformer_engine/gemm.h"

namespace transformer_engine {
namespace jax {

// NCCL comm + EPBackend lifetime tracks live JAX executables via XLA stateful FFI.

struct EpBootstrapParams {
  std::array<uint8_t, 128> uid_bytes{};
  int ep_size = 0;
  int rank_within_group = 0;
  int num_experts = 0;
  int max_tokens_per_rank = 0;
  int max_recv_tokens_per_rank = 0;
  int hidden_dim = 0;
  int max_num_sms = 0;
  NVTEDType max_token_dtype = kNVTEBFloat16;
};

class EpResources {
 public:
  explicit EpResources(const EpBootstrapParams& p) {
    ncclUniqueId uid;
    std::memcpy(&uid, p.uid_bytes.data(), sizeof(uid));
    NVTE_CHECK_NCCL(ncclCommInitRank(&comm_, p.ep_size, uid, p.rank_within_group));
    // zero_copy=0: JAX EP path always stages payloads; the zero-copy fast path
    // requires NVTECommWindow-backed tensors, which JAX bindings don't expose.
    NVTEEpGroupConfig cfg{.struct_size = sizeof(NVTEEpGroupConfig),
                          .ep_size = p.ep_size,
                          .num_experts = p.num_experts,
                          .max_tokens_per_rank = p.max_tokens_per_rank,
                          .max_recv_tokens_per_rank = p.max_recv_tokens_per_rank,
                          .hidden_dim = p.hidden_dim,
                          .num_comm_sms = p.max_num_sms,
                          .max_token_dtype = p.max_token_dtype,
                          .zero_copy = 0};
    try {
      nvte_ep_initialize(static_cast<void*>(comm_), &cfg);
    } catch (...) {
      ncclCommDestroy(comm_);
      comm_ = nullptr;
      throw;
    }
  }

  ~EpResources() {
    if (comm_ == nullptr) return;
    nvte_ep_shutdown();
    ncclCommDestroy(comm_);
  }

  EpResources(const EpResources&) = delete;
  EpResources& operator=(const EpResources&) = delete;

  ncclComm_t comm() const { return comm_; }

 private:
  ncclComm_t comm_{nullptr};
};

struct EpInstanceState {
  static ::xla::ffi::TypeId id;
  static ::xla::ffi::TypeInfo info;
  std::shared_ptr<EpResources> resources;
};

::xla::ffi::TypeId EpInstanceState::id = {};
::xla::ffi::TypeInfo EpInstanceState::info = ::xla::ffi::MakeTypeInfo<EpInstanceState>();

namespace {

std::mutex g_ep_mu;
EpBootstrapParams g_ep_params;
bool g_ep_params_set = false;
std::weak_ptr<EpResources> g_ep_resources_weak;
// Python-held anchor so trace-time handle_mem allocs find EPBackend ready.
std::shared_ptr<EpResources> g_ep_resources_anchor;

std::shared_ptr<EpResources> AcquireEpResources() {
  std::lock_guard<std::mutex> lock(g_ep_mu);
  NVTE_CHECK(g_ep_params_set,
             "EP bootstrap params not set; call transformer_engine_jax."
             "set_ep_bootstrap_params() (typically via ep_bootstrap) first.");
  auto sp = g_ep_resources_weak.lock();
  if (sp) return sp;
  sp = std::make_shared<EpResources>(g_ep_params);
  g_ep_resources_weak = sp;
  return sp;
}

}  // namespace

// top_k and dispatch_output_per_expert_alignment are baked as static FFI
// attributes; prepare passes them to the C API as NVTEEpLayerConfig, and the
// per-step ops carry top_k only to validate the topk_idx last dim.

struct EpConfig {
  int64_t top_k;
  int64_t dispatch_output_per_expert_alignment;
};

// ── Bootstrap helpers ─────────────────────────────────────────────────────────

// Caches uid + group config and eagerly creates the NCCL comm (ranks
// synchronize via the UID broadcast).
void SetEpBootstrapParams(pybind11::bytes unique_id_bytes_obj, int ep_size, int rank_within_group,
                          int num_experts, int max_tokens_per_rank, int max_recv_tokens_per_rank,
                          int hidden_dim, int max_num_sms, int max_token_dtype) {
  std::string uid_str = unique_id_bytes_obj;
  NVTE_CHECK(static_cast<int>(uid_str.size()) >= 128,
             "unique_id_bytes must be at least 128 bytes (ncclUniqueId size).");
  std::shared_ptr<EpResources> anchor;
  {
    std::lock_guard<std::mutex> lock(g_ep_mu);
    NVTE_CHECK(!g_ep_resources_anchor,
               "EP bootstrap already initialized; call release_ep_resources() before re-init.");
    std::memcpy(g_ep_params.uid_bytes.data(), uid_str.data(), 128);
    g_ep_params.ep_size = ep_size;
    g_ep_params.rank_within_group = rank_within_group;
    g_ep_params.num_experts = num_experts;
    g_ep_params.max_tokens_per_rank = max_tokens_per_rank;
    g_ep_params.max_recv_tokens_per_rank = max_recv_tokens_per_rank;
    g_ep_params.hidden_dim = hidden_dim;
    g_ep_params.max_num_sms = max_num_sms;
    g_ep_params.max_token_dtype = static_cast<NVTEDType>(max_token_dtype);
    g_ep_params_set = true;
  }
  // Acquire outside the lock: EpResources ctor runs ncclCommInitRank which is
  // a collective and may block on peer ranks.
  anchor = AcquireEpResources();
  std::lock_guard<std::mutex> lock(g_ep_mu);
  g_ep_resources_anchor = std::move(anchor);
}

// Drops the anchor; comm tears down once the last executable also releases.
void ReleaseEpResources() {
  std::shared_ptr<EpResources> to_drop;
  {
    std::lock_guard<std::mutex> lock(g_ep_mu);
    to_drop = std::move(g_ep_resources_anchor);
  }
  // to_drop dtor runs outside the lock.
}

size_t EpHandleMemSize(int top_k, size_t dispatch_output_per_expert_alignment) {
  NVTEEpLayerConfig layer_cfg{
      .struct_size = sizeof(NVTEEpLayerConfig),
      .top_k = top_k,
      .dispatch_output_per_expert_alignment = dispatch_output_per_expert_alignment};
  return nvte_ep_handle_mem_size(&layer_cfg);
}

pybind11::capsule GetEpInstanceStateTypeIdCapsule() {
  return pybind11::capsule(static_cast<void*>(&EpInstanceState::id), "xla.ffi.type_id");
}

pybind11::capsule GetEpInstanceStateTypeInfoCapsule() {
  return pybind11::capsule(static_cast<void*>(&EpInstanceState::info), "xla.ffi.type_info");
}

// ── Instantiate handler ─────────────────────────────────────────────────────

static ::xla::ffi::ErrorOr<std::unique_ptr<EpInstanceState>> EpInstantiateImpl() {
  auto state = std::make_unique<EpInstanceState>();
  try {
    state->resources = AcquireEpResources();
  } catch (const std::exception& e) {
    return ::xla::ffi::Unexpected(
        ::xla::ffi::Error::Internal(std::string("EP instantiate failed: ") + e.what()));
  }
  return state;
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpInstantiateHandler, EpInstantiateImpl, FFI::BindInstantiate());

// ── ep_prepare ────────────────────────────────────────────────────────────────

Error_Type EpPrepareFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type topk_idx,
                        Result_Type recv_tokens_per_expert, Result_Type handle_mem,
                        Result_Type workspace, EpConfig config) {
  (void)ep_state;  // lifetime only.
  auto topk_dims = topk_idx.dimensions();
  NVTE_CHECK(topk_dims.size() >= 2,
             "topk_idx must be at least 2D [..., top_k], got ndim=", topk_dims.size());
  auto idx_etype = topk_idx.element_type();
  NVTE_CHECK(idx_etype == ::xla::ffi::DataType::S64 || idx_etype == ::xla::ffi::DataType::S32,
             "topk_idx must be int32 or int64; got element_type=", static_cast<int>(idx_etype));

  std::vector<size_t> topk_shape = {product(topk_dims, 0, topk_dims.size() - 1),
                                    static_cast<size_t>(topk_dims.back())};
  // NCCL EP currently requires int64 topk_idx; upcast int32 on-stream.
  // TODO(phuong): drop once NCCL EP accepts int32.
  void* topk_idx_data = topk_idx.untyped_data();
  if (idx_etype == ::xla::ffi::DataType::S32) {
    const size_t n = topk_shape[0] * topk_shape[1];
    NVTE_CHECK(static_cast<size_t>(workspace->element_count()) >= n,
               "workspace too small for int32 → int64 upcast: element_count=",
               workspace->element_count(), " < required ", n);
    int64_t* ws = reinterpret_cast<int64_t*>(workspace->untyped_data());
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t*>(topk_idx_data), ws, n, stream);
    topk_idx_data = ws;
  }
  auto topk_idx_ = TensorWrapper(topk_idx_data, topk_shape, DType::kInt64);

  std::vector<size_t> tc_shape = {static_cast<size_t>(recv_tokens_per_expert->element_count())};
  auto recv_tokens_per_expert_ =
      TensorWrapper(recv_tokens_per_expert->untyped_data(), tc_shape, DType::kInt32);

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem->element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem->untyped_data(), hm_shape, DType::kByte);

  NVTEEpLayerConfig layer_cfg{.struct_size = sizeof(NVTEEpLayerConfig),
                              .top_k = static_cast<int>(config.top_k),
                              .dispatch_output_per_expert_alignment =
                                  static_cast<size_t>(config.dispatch_output_per_expert_alignment)};
  nvte_ep_prepare(handle_mem_.data(), topk_idx_.data(), recv_tokens_per_expert_.data(),
                  /*total_recv_tokens_per_rank=*/nullptr, &layer_cfg, stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpPrepareHandler, EpPrepareFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // topk_idx
                                  .Ret<Buffer_Type>()  // recv_tokens_per_expert
                                  .Ret<Buffer_Type>()  // handle_mem
                                  .Ret<Buffer_Type>()  // workspace (FFI scratch)
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch ───────────────────────────────────────────────────────────────

Error_Type EpDispatchFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                         Buffer_Type topk_idx, Buffer_Type tokens, Buffer_Type topk_weights,
                         Result_Type recv_tokens, Result_Type recv_topk_weights,
                         Result_Type workspace, EpConfig config) {
  (void)ep_state;
  auto token_dims = tokens.dimensions();
  NVTE_CHECK(token_dims.size() >= 2,
             "tokens must be at least 2D [..., H], got ndim=", token_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  auto idx_dims = topk_idx.dimensions();
  NVTE_CHECK(idx_dims.size() >= 2,
             "topk_idx must be at least 2D [..., top_k], got ndim=", idx_dims.size());
  auto idx_etype = topk_idx.element_type();
  NVTE_CHECK(idx_etype == ::xla::ffi::DataType::S64 || idx_etype == ::xla::ffi::DataType::S32,
             "topk_idx must be int32 or int64; got element_type=", static_cast<int>(idx_etype));
  NVTE_CHECK(static_cast<int64_t>(idx_dims.back()) == config.top_k, "top_k attr (", config.top_k,
             ") must match topk_idx last dim (", idx_dims.back(), ")");
  std::vector<size_t> idx_shape = {product(idx_dims, 0, idx_dims.size() - 1),
                                   static_cast<size_t>(idx_dims.back())};
  // NCCL EP currently requires int64 topk_idx; upcast int32 on-stream.
  // TODO(phuong): drop once NCCL EP accepts int32.
  void* topk_idx_data = topk_idx.untyped_data();
  if (idx_etype == ::xla::ffi::DataType::S32) {
    const size_t n = idx_shape[0] * idx_shape[1];
    NVTE_CHECK(static_cast<size_t>(workspace->element_count()) >= n,
               "workspace too small for int32 → int64 upcast: element_count=",
               workspace->element_count(), " < required ", n);
    int64_t* ws = reinterpret_cast<int64_t*>(workspace->untyped_data());
    nvte_convert_int32_to_int64(reinterpret_cast<const int32_t*>(topk_idx_data), ws, n, stream);
    topk_idx_data = ws;
  }
  auto topk_idx_ = TensorWrapper(topk_idx_data, idx_shape, DType::kInt64);

  const size_t T_flat = product(token_dims, 0, token_dims.size() - 1);
  const size_t H = static_cast<size_t>(token_dims.back());
  std::vector<size_t> tok_shape = {T_flat, H};
  auto token_dtype = convert_ffi_datatype_to_te_dtype(tokens.element_type());
  auto tokens_ = TensorWrapper(tokens.untyped_data(), tok_shape, token_dtype);

  auto tw_dims = topk_weights.dimensions();
  NVTE_CHECK(tw_dims.size() >= 2,
             "topk_weights must be at least 2D [..., top_k], got ndim=", tw_dims.size());
  std::vector<size_t> tw_shape = {product(tw_dims, 0, tw_dims.size() - 1),
                                  static_cast<size_t>(tw_dims.back())};
  auto topk_weights_ = TensorWrapper(topk_weights.untyped_data(), tw_shape, DType::kFloat32);

  // recv_tokens: flatten any leading dims into recv_capacity_per_rank.
  auto recv_dims = recv_tokens->dimensions();
  NVTE_CHECK(recv_dims.size() >= 2,
             "recv_tokens must be at least 2D [..., recv_pr, H]; got ndim=", recv_dims.size());
  const size_t recv_capacity_per_rank = product(recv_dims, 0, recv_dims.size() - 1);
  std::vector<size_t> recv_shape = {recv_capacity_per_rank, H};
  auto recv_tokens_ = TensorWrapper(recv_tokens->untyped_data(), recv_shape, token_dtype);

  auto recv_w_dims = recv_topk_weights->dimensions();
  NVTE_CHECK(recv_w_dims.size() >= 1,
             "recv_topk_weights must be at least 1D; got ndim=", recv_w_dims.size());
  const size_t recv_w_total = product(recv_w_dims, 0, recv_w_dims.size());
  NVTE_CHECK(recv_w_total == recv_capacity_per_rank, "recv_topk_weights total (", recv_w_total,
             ") must match recv_tokens recv_pr (", recv_capacity_per_rank, ")");
  std::vector<size_t> recv_w_shape = {recv_capacity_per_rank};
  auto recv_topk_weights_ =
      TensorWrapper(recv_topk_weights->untyped_data(), recv_w_shape, DType::kFloat32);

  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_dispatch(handle_mem_.data(), topk_idx_.data(), tokens_.data(), no_win,
                   topk_weights_.data(), no_win, recv_tokens_.data(), no_win,
                   recv_topk_weights_.data(), no_win, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchHandler, EpDispatchFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()                         // topk_idx
                                  .Arg<Buffer_Type>()                         // tokens
                                  .Arg<Buffer_Type>()                         // topk_weights
                                  .Ret<Buffer_Type>()                         // recv_tokens
                                  .Ret<Buffer_Type>()                         // recv_topk_weights
                                  .Ret<Buffer_Type>()  // workspace (FFI scratch)
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine ────────────────────────────────────────────────────────────────

Error_Type EpCombineFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                        Buffer_Type expert_out, Result_Type result, EpConfig config) {
  (void)ep_state;
  auto eo_dims = expert_out.dimensions();
  NVTE_CHECK(eo_dims.size() >= 2,
             "expert_out must be at least 2D [..., recv_pr, H]; got ndim=", eo_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t recv_capacity_per_rank = product(eo_dims, 0, eo_dims.size() - 1);
  const size_t H = static_cast<size_t>(eo_dims.back());
  std::vector<size_t> eo_shape = {recv_capacity_per_rank, H};
  auto eo_dtype = convert_ffi_datatype_to_te_dtype(expert_out.element_type());
  auto expert_out_ = TensorWrapper(expert_out.untyped_data(), eo_shape, eo_dtype);

  auto res_dims = result->dimensions();
  NVTE_CHECK(res_dims.size() >= 2,
             "result must be at least 2D [..., H]; got ndim=", res_dims.size());
  const size_t res_T_flat = product(res_dims, 0, res_dims.size() - 1);
  std::vector<size_t> res_shape = {res_T_flat, H};
  auto result_ = TensorWrapper(result->untyped_data(), res_shape, eo_dtype);

  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_combine(handle_mem_.data(), expert_out_.data(), no_win, result_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineHandler, EpCombineFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()                         // expert_out
                                  .Ret<Buffer_Type>()                         // result
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch_bwd ───────────────────────────────────────────────────────────

Error_Type EpDispatchBwdFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                            Buffer_Type grad, Buffer_Type g_recv_topk_weights,
                            Result_Type grad_tokens, Result_Type grad_topk_weights,
                            EpConfig config) {
  (void)ep_state;
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() >= 2,
             "grad must be at least 2D [..., recv_pr, H]; got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t recv_capacity_per_rank = product(grad_dims, 0, grad_dims.size() - 1);
  const size_t H = static_cast<size_t>(grad_dims.back());
  std::vector<size_t> g_shape = {recv_capacity_per_rank, H};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  auto gw_dims = g_recv_topk_weights.dimensions();
  NVTE_CHECK(
      gw_dims.size() >= 1,
      "g_recv_topk_weights rank must flatten to recv_capacity_per_rank; got ndim=", gw_dims.size());
  const size_t gw_total = product(gw_dims, 0, gw_dims.size());
  NVTE_CHECK(gw_total == recv_capacity_per_rank, "g_recv_topk_weights total (", gw_total,
             ") must match grad recv_pr (", recv_capacity_per_rank, ")");
  std::vector<size_t> gw_shape = {recv_capacity_per_rank};
  auto g_recv_topk_weights_ =
      TensorWrapper(g_recv_topk_weights.untyped_data(), gw_shape, DType::kFloat32);

  auto out_dims = grad_tokens->dimensions();
  NVTE_CHECK(out_dims.size() >= 2,
             "grad_tokens must be at least 2D [..., H], got ndim=", out_dims.size());
  const size_t T_flat = product(out_dims, 0, out_dims.size() - 1);
  std::vector<size_t> out_shape = {T_flat, H};
  auto grad_tokens_ = TensorWrapper(grad_tokens->untyped_data(), out_shape, g_dtype);

  auto gtw_dims = grad_topk_weights->dimensions();
  NVTE_CHECK(gtw_dims.size() >= 2,
             "grad_topk_weights must be at least 2D [..., top_k]; got ndim=", gtw_dims.size());
  const size_t gtw_T_flat = product(gtw_dims, 0, gtw_dims.size() - 1);
  NVTE_CHECK(gtw_T_flat == T_flat, "grad_topk_weights leading-dim product (", gtw_T_flat,
             ") must equal grad_tokens leading-dim product (", T_flat, ")");
  const size_t top_k = static_cast<size_t>(gtw_dims.back());
  NVTE_CHECK(static_cast<int64_t>(top_k) == config.top_k, "top_k attr (", config.top_k,
             ") must match grad_topk_weights last dim (", top_k, ")");
  std::vector<size_t> gtw_shape = {T_flat, top_k};
  auto grad_topk_weights_ =
      TensorWrapper(grad_topk_weights->untyped_data(), gtw_shape, DType::kFloat32);

  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_dispatch_bwd(handle_mem_.data(), grad_.data(), no_win, g_recv_topk_weights_.data(),
                       no_win, grad_tokens_.data(), grad_topk_weights_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchBwdHandler, EpDispatchBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()  // grad (w.r.t. recv_tokens)
                                  .Arg<Buffer_Type>()  // g_recv_topk_weights
                                  .Ret<Buffer_Type>()  // grad_tokens
                                  .Ret<Buffer_Type>()  // grad_topk_weights
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine_bwd ────────────────────────────────────────────────────────────

Error_Type EpCombineBwdFFI(cudaStream_t stream, EpInstanceState* ep_state, Buffer_Type handle_mem,
                           Buffer_Type grad, Result_Type grad_expert_out, EpConfig config) {
  (void)ep_state;
  auto grad_dims = grad.dimensions();
  NVTE_CHECK(grad_dims.size() >= 2,
             "grad must be at least 2D [..., H], got ndim=", grad_dims.size());

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem.element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem.untyped_data(), hm_shape, DType::kByte);

  const size_t T_flat = product(grad_dims, 0, grad_dims.size() - 1);
  const size_t H = static_cast<size_t>(grad_dims.back());
  std::vector<size_t> g_shape = {T_flat, H};
  auto g_dtype = convert_ffi_datatype_to_te_dtype(grad.element_type());
  auto grad_ = TensorWrapper(grad.untyped_data(), g_shape, g_dtype);

  auto out_dims = grad_expert_out->dimensions();
  NVTE_CHECK(out_dims.size() >= 2,
             "grad_expert_out must be at least 2D [..., recv_pr, H]; got ndim=", out_dims.size());
  const size_t recv_capacity_per_rank = product(out_dims, 0, out_dims.size() - 1);
  const size_t out_H = static_cast<size_t>(out_dims.back());
  NVTE_CHECK(out_H == H, "grad_expert_out hidden dim (", out_H, ") must match grad H (", H, ")");
  std::vector<size_t> out_shape = {recv_capacity_per_rank, H};
  auto grad_expert_out_ = TensorWrapper(grad_expert_out->untyped_data(), out_shape, g_dtype);

  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_combine_bwd(handle_mem_.data(), grad_.data(), no_win, grad_expert_out_.data(), no_win,
                      stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineBwdHandler, EpCombineBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()                     // stream
                                  .Ctx<::xla::ffi::State<EpInstanceState>>()  // EP state
                                  .Arg<Buffer_Type>()                         // handle_mem
                                  .Arg<Buffer_Type>()  // grad (w.r.t. result)
                                  .Ret<Buffer_Type>()  // grad_expert_out
                                  .Attrs<EpConfig>(),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    transformer_engine::jax::EpConfig, ::xla::ffi::StructMember<int64_t>("top_k"),
    ::xla::ffi::StructMember<int64_t>("dispatch_output_per_expert_alignment"));

#endif  // NVTE_WITH_NCCL_EP
