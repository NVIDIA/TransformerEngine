/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef NVTE_WITH_NCCL_EP

#include "transformer_engine/ep.h"

#include <nccl.h>

#include <cstdint>
#include <cstring>
#include <mutex>

#include "../extensions.h"
#include "common.h"
#include "transformer_engine/gemm.h"

namespace transformer_engine {
namespace jax {

namespace {

// Process-lifetime owner of the EP ncclComm_t. Created from a broadcast
// ncclUniqueId during EpInitialize; destroyed by EpShutdown (registered as a
// Python atexit hook from ep.py so it runs before C++ static destructors).
class EpCommManager {
 public:
  static EpCommManager& get() {
    static EpCommManager inst;
    return inst;
  }

  void init_from_uid(const uint8_t* uid_bytes, int ep_size, int rank_within_group) {
    std::lock_guard<std::mutex> lock(mutex_);
    NVTE_CHECK(comm_ == nullptr, "EP comm already initialized for this process");
    ncclUniqueId uid;
    std::memcpy(&uid, uid_bytes, sizeof(uid));
    NVTE_CHECK_NCCL(ncclCommInitRank(&comm_, ep_size, uid, rank_within_group));
  }

  ncclComm_t comm() const { return comm_; }

  void shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (comm_ == nullptr) return;
    ncclCommDestroy(comm_);
    comm_ = nullptr;
  }

 private:
  EpCommManager() = default;
  // Intentionally no NCCL teardown in the destructor: this runs at static-dtor
  // time, after Python has finalized and possibly after the CUDA driver
  // detaches the context. Calling ncclCommDestroy there has been observed to
  // hang or report cudartUnloading. Normal teardown goes through the Python
  // atexit hook (shutdown_ep_communicator) registered from ep.py; any path
  // that skips that (os._exit, fatal signal) leaks the comm, which the OS
  // reaps on process exit.
  ~EpCommManager() = default;
  EpCommManager(const EpCommManager&) = delete;
  EpCommManager& operator=(const EpCommManager&) = delete;

  std::mutex mutex_;
  ncclComm_t comm_{nullptr};
};

}  // namespace

// handle_id is baked at jit trace time and carried as a static FFI attribute.

struct EpPrepareConfig {
  int64_t handle_id;
  int64_t dispatch_output_per_expert_alignment;
};

struct EpDispatchConfig {
  int64_t handle_id;
  int64_t top_k;
};

struct EpCombineConfig {
  int64_t handle_id;
  int64_t num_local_tokens;
};

struct EpDispatchBwdConfig {
  int64_t handle_id;
  int64_t num_local_tokens;
  int64_t top_k;
};

struct EpCombineBwdConfig {
  int64_t handle_id;
};

// ── Bootstrap helpers ─────────────────────────────────────────────────────────

void EpInitialize(pybind11::bytes unique_id_bytes_obj, int ep_size, int rank_within_group,
                  int num_experts, int max_tokens_per_rank, int max_recv_tokens_per_rank,
                  int hidden_dim, int max_num_sms) {
  std::string uid_str = unique_id_bytes_obj;
  NVTE_CHECK(static_cast<int>(uid_str.size()) >= 128,
             "unique_id_bytes must be at least 128 bytes (ncclUniqueId size).");
  EpCommManager::get().init_from_uid(reinterpret_cast<const uint8_t*>(uid_str.data()), ep_size,
                                     rank_within_group);
  NVTEEpGroupConfig cfg{.ep_size = ep_size,
                        .num_experts = num_experts,
                        .max_tokens_per_rank = max_tokens_per_rank,
                        .max_recv_tokens_per_rank = max_recv_tokens_per_rank,
                        .hidden_dim = hidden_dim,
                        .max_num_sms = max_num_sms};
  // If common rejects the config (validate_config / ncclEpCreateGroup), roll
  // the comm back so the two singletons don't end up in inconsistent states
  // and the comm doesn't strand until process exit.
  try {
    nvte_ep_initialize(static_cast<void*>(EpCommManager::get().comm()), cfg);
  } catch (...) {
    EpCommManager::get().shutdown();
    throw;
  }
}

void EpShutdown() {
  // Order matters: ep_group_ in common reads from the comm, so tear it down
  // first, then destroy the comm.
  nvte_ep_shutdown();
  EpCommManager::get().shutdown();
}

pybind11::tuple EpRegisterLayer(int top_k, size_t dispatch_output_per_expert_alignment) {
  NVTEEpLayerConfig layer_cfg{0, top_k, dispatch_output_per_expert_alignment};
  size_t handle_mem_size = 0;
  uint64_t handle_id = nvte_ep_register_layer(layer_cfg, &handle_mem_size);
  return pybind11::make_tuple(handle_id, handle_mem_size);
}

// ── ep_prepare ────────────────────────────────────────────────────────────────

Error_Type EpPrepareFFI(cudaStream_t stream, Buffer_Type topk_idx, Result_Type token_counts,
                        Result_Type handle_mem, Result_Type workspace, EpPrepareConfig config) {
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

  std::vector<size_t> tc_shape = {static_cast<size_t>(token_counts->element_count())};
  auto token_counts_ = TensorWrapper(token_counts->untyped_data(), tc_shape, DType::kInt32);

  std::vector<size_t> hm_shape = {static_cast<size_t>(handle_mem->element_count())};
  auto handle_mem_ = TensorWrapper(handle_mem->untyped_data(), hm_shape, DType::kByte);

  NVTEEpHandle handle{static_cast<uint64_t>(config.handle_id), handle_mem_.data()};
  nvte_ep_prepare(handle, topk_idx_.data(), token_counts_.data(),
                  static_cast<size_t>(config.dispatch_output_per_expert_alignment), stream);
  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpPrepareHandler, EpPrepareFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // topk_idx
                                  .Ret<Buffer_Type>()      // token_counts
                                  .Ret<Buffer_Type>()      // handle_mem
                                  .Ret<Buffer_Type>()      // workspace (FFI scratch)
                                  .Attrs<EpPrepareConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch ───────────────────────────────────────────────────────────────

Error_Type EpDispatchFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type topk_idx,
                         Buffer_Type tokens, Buffer_Type topk_weights, Result_Type recv_tokens,
                         Result_Type recv_topk_weights, Result_Type workspace,
                         EpDispatchConfig config) {
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

  NVTEEpHandle handle{static_cast<uint64_t>(config.handle_id), handle_mem_.data()};
  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_dispatch(handle, topk_idx_.data(), tokens_.data(), no_win, topk_weights_.data(), no_win,
                   recv_tokens_.data(), no_win, recv_topk_weights_.data(), no_win, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchHandler, EpDispatchFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // topk_idx
                                  .Arg<Buffer_Type>()      // tokens
                                  .Arg<Buffer_Type>()      // topk_weights
                                  .Ret<Buffer_Type>()      // recv_tokens
                                  .Ret<Buffer_Type>()      // recv_topk_weights
                                  .Ret<Buffer_Type>()      // workspace (FFI scratch)
                                  .Attrs<EpDispatchConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine ────────────────────────────────────────────────────────────────

Error_Type EpCombineFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type expert_out,
                        Result_Type result, EpCombineConfig config) {
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
  NVTE_CHECK(static_cast<int64_t>(res_T_flat) == config.num_local_tokens,
             "result leading-dim product (", res_T_flat, ") must equal num_local_tokens (",
             config.num_local_tokens, ")");
  std::vector<size_t> res_shape = {res_T_flat, H};
  auto result_ = TensorWrapper(result->untyped_data(), res_shape, eo_dtype);

  NVTEEpHandle handle{static_cast<uint64_t>(config.handle_id), handle_mem_.data()};
  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_combine(handle, expert_out_.data(), no_win, result_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineHandler, EpCombineFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // expert_out
                                  .Ret<Buffer_Type>()      // result
                                  .Attrs<EpCombineConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_dispatch_bwd ───────────────────────────────────────────────────────────

Error_Type EpDispatchBwdFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type grad,
                            Buffer_Type g_recv_topk_weights, Result_Type grad_tokens,
                            Result_Type grad_topk_weights, EpDispatchBwdConfig config) {
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
  NVTE_CHECK(static_cast<int64_t>(T_flat) == config.num_local_tokens,
             "grad_tokens leading-dim product (", T_flat, ") must equal num_local_tokens (",
             config.num_local_tokens, ")");
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

  NVTEEpHandle handle{static_cast<uint64_t>(config.handle_id), handle_mem_.data()};
  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_dispatch_bwd(handle, grad_.data(), no_win, g_recv_topk_weights_.data(), no_win,
                       grad_tokens_.data(), grad_topk_weights_.data(), stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpDispatchBwdHandler, EpDispatchBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // grad (w.r.t. recv_tokens)
                                  .Arg<Buffer_Type>()      // g_recv_topk_weights
                                  .Ret<Buffer_Type>()      // grad_tokens
                                  .Ret<Buffer_Type>()      // grad_topk_weights
                                  .Attrs<EpDispatchBwdConfig>(),
                              FFI_CudaGraph_Traits);

// ── ep_combine_bwd ────────────────────────────────────────────────────────────

Error_Type EpCombineBwdFFI(cudaStream_t stream, Buffer_Type handle_mem, Buffer_Type grad,
                           Result_Type grad_expert_out, EpCombineBwdConfig config) {
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

  NVTEEpHandle handle{static_cast<uint64_t>(config.handle_id), handle_mem_.data()};
  NVTECommWindow no_win{nullptr, 0};
  nvte_ep_combine_bwd(handle, grad_.data(), no_win, grad_expert_out_.data(), no_win, stream);

  return ffi_with_cuda_error_check();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(EpCombineBwdHandler, EpCombineBwdFFI,
                              FFI::Bind()
                                  .Ctx<FFI_Stream_Type>()  // stream
                                  .Arg<Buffer_Type>()      // handle_mem
                                  .Arg<Buffer_Type>()      // grad (w.r.t. result)
                                  .Ret<Buffer_Type>()      // grad_expert_out
                                  .Attrs<EpCombineBwdConfig>(),
                              FFI_CudaGraph_Traits);

}  // namespace jax
}  // namespace transformer_engine

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(
    transformer_engine::jax::EpPrepareConfig, ::xla::ffi::StructMember<int64_t>("handle_id"),
    ::xla::ffi::StructMember<int64_t>("dispatch_output_per_expert_alignment"));

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpDispatchConfig,
                                      ::xla::ffi::StructMember<int64_t>("handle_id"),
                                      ::xla::ffi::StructMember<int64_t>("top_k"));

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpCombineConfig,
                                      ::xla::ffi::StructMember<int64_t>("handle_id"),
                                      ::xla::ffi::StructMember<int64_t>("num_local_tokens"));

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpDispatchBwdConfig,
                                      ::xla::ffi::StructMember<int64_t>("handle_id"),
                                      ::xla::ffi::StructMember<int64_t>("num_local_tokens"),
                                      ::xla::ffi::StructMember<int64_t>("top_k"));

XLA_FFI_REGISTER_STRUCT_ATTR_DECODING(transformer_engine::jax::EpCombineBwdConfig,
                                      ::xla::ffi::StructMember<int64_t>("handle_id"));

#endif  // NVTE_WITH_NCCL_EP
