/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.cpp
 *  \brief EPBackend implementation. See ep_backend.h for the op flow.
 */

#include "ep_backend.h"

#include <nccl_ep.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "../util/logging.h"

namespace transformer_engine {
namespace ep {

namespace {

ncclDataType_t te_dtype_to_nccl_dtype(NVTEDType dtype) {
  switch (dtype) {
    case kNVTEFloat32:
      return ncclFloat32;
    case kNVTEFloat16:
      return ncclFloat16;
    case kNVTEBFloat16:
      return ncclBfloat16;
    case kNVTEInt32:
      return ncclInt32;
    case kNVTEInt64:
      return ncclInt64;
    case kNVTEByte:
      return ncclUint8;
    case kNVTEFloat8E4M3:
      return ncclFloat8e4m3;
    case kNVTEFloat8E5M2:
      return ncclFloat8e5m2;
    default:
      NVTE_ERROR("Unsupported NVTEDType for NCCL dtype conversion: ", static_cast<int>(dtype));
  }
  return ncclFloat32;  // unreachable
}

// shape_out is caller-owned; desc.sizes aliases shape_out.data and must
// outlive the NCCL EP call.
inline ncclEpTensor_t make_nccl_ep_tensor(const NVTETensor t, NVTEShape& shape_out,
                                          const NVTECommWindow& win = {}) {
  shape_out = nvte_tensor_shape(t);
  ncclEpTensor_t desc = NCCL_EP_TENSOR_INIT;
  desc.ndim = shape_out.ndim;
  desc.sizes = shape_out.data;
  desc.datatype = te_dtype_to_nccl_dtype(nvte_tensor_type(t));
  if (win.window != nullptr) {
    desc.win_hdl = win.window;
    desc.win_offset = win.offset;
  } else {
    desc.data = nvte_tensor_data(t);
    NVTE_CHECK(desc.data != nullptr, "tensor data must not be null");
  }
  return desc;
}

}  // namespace

// ---------------------------------------------------------------------------
// Singleton + bootstrap
// ---------------------------------------------------------------------------

EPBackend& EPBackend::instance() {
  static EPBackend inst;
  return inst;
}

EPBackend& EPBackend::get() {
  EPBackend& inst = instance();
  NVTE_CHECK(inst.initialized_, "EPBackend not initialized. Call nvte_ep_initialize() first.");
  return inst;
}

void EPBackend::validate_config(const NVTEEpGroupConfig& config) {
  NVTE_CHECK(config.ep_size > 0, "ep_size must be positive, got ", config.ep_size);
  NVTE_CHECK(config.num_experts > 0, "num_experts must be positive, got ", config.num_experts);
  NVTE_CHECK(config.max_tokens_per_rank > 0, "max_tokens_per_rank must be positive, got ",
             config.max_tokens_per_rank);
  NVTE_CHECK(config.max_recv_tokens_per_rank > 0, "max_recv_tokens_per_rank must be positive, got ",
             config.max_recv_tokens_per_rank);
  NVTE_CHECK(config.hidden_dim > 0, "hidden_dim must be positive, got ", config.hidden_dim);
  NVTE_CHECK(config.max_token_dtype >= 0 && config.max_token_dtype < kNVTENumTypes,
             "max_token_dtype out of range, got ", static_cast<int>(config.max_token_dtype));
  const size_t elem_bytes = typeToSize(static_cast<DType>(config.max_token_dtype));
  const size_t row_bytes = static_cast<size_t>(config.hidden_dim) * elem_bytes;
  NVTE_CHECK(row_bytes >= 16,
             "hidden_dim * sizeof(max_token_dtype) must be >= 16 (NCCL EP 16B row alignment); "
             "got hidden_dim=",
             config.hidden_dim, ", element_bytes=", elem_bytes);
  // NCCL EP packs row size into ncclEpGroupConfig::max_token_bytes (unsigned int).
  NVTE_CHECK(row_bytes <= static_cast<size_t>(UINT_MAX),
             "hidden_dim * sizeof(max_token_dtype) exceeds 4 GiB; got ", row_bytes, " bytes");
  NVTE_CHECK(config.num_experts % config.ep_size == 0, "num_experts (", config.num_experts,
             ") must be divisible by ep_size (", config.ep_size, ")");
  NVTE_CHECK(config.num_comm_sms >= 0, "num_comm_sms must be >= 0 (0 = auto), got ",
             config.num_comm_sms);

  const int sm = cuda::sm_arch();
  NVTE_CHECK(sm >= 90, "NCCL EP requires SM_90+ (Hopper or later), but current device is SM_", sm);
}

void EPBackend::initialize(ncclComm_t ep_comm, NVTEEpGroupConfig config) {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  NVTE_CHECK(!inst.initialized_, "EP already initialized. Call initialize only once per process.");
  NVTE_CHECK(ep_comm != nullptr, "ep_comm must not be null");

  // Runtime gate: NCCL >= 2.30.4 (matches the submodule pin).
  constexpr int kMinNcclVersion = 23004;
  int nccl_version = 0;
  NVTE_CHECK_NCCL(ncclGetVersion(&nccl_version));
  NVTE_CHECK(nccl_version >= kMinNcclVersion, "NCCL EP requires NCCL >= 2.30.4, found ",
             nccl_version / 10000, ".", (nccl_version / 100) % 100, ".", nccl_version % 100,
             " at runtime.");

  validate_config(config);

  int comm_size = 0;
  NVTE_CHECK_NCCL(ncclCommCount(ep_comm, &comm_size));
  NVTE_CHECK(comm_size == config.ep_size, "ep_comm size (", comm_size, ") must equal ep_size (",
             config.ep_size, "). Pass the EP sub-communicator, not the world comm.");

  inst.init(ep_comm, config);
}

void EPBackend::shutdown() {
  EPBackend& inst = instance();
  std::lock_guard<std::mutex> lock(inst.mutex_);
  if (!inst.initialized_) return;
  for (auto& e : inst.lru_) {
    if (e.handle != nullptr) ncclEpHandleDestroy(e.handle);
  }
  inst.lru_.clear();
  inst.index_.clear();
  inst.fallback_layer_cfg_.reset();
  // ncclEpGroupDestroy reads from ep_comm_; destroy group while comm is still alive.
  if (inst.ep_group_ != nullptr) {
    ncclEpGroupDestroy(inst.ep_group_);
    inst.ep_group_ = nullptr;
  }
  inst.ep_comm_ = nullptr;  // borrowed; caller destroys
  inst.initialized_ = false;
}

ncclEpHandle_t EPBackend::open_handle(void* handle_mem, size_t handle_mem_size, int num_topk,
                                      size_t dispatch_output_per_expert_alignment) {
  size_t hm_sizes[1] = {handle_mem_size};
  ncclEpTensor_t routing_desc = NCCL_EP_TENSOR_INIT;
  routing_desc.ndim = 1;
  routing_desc.datatype = ncclUint8;
  routing_desc.data = handle_mem;
  routing_desc.sizes = hm_sizes;
  ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
  hcfg.dispatch_output_per_expert_alignment = dispatch_output_per_expert_alignment;
  ncclEpHandle_t handle;
  NVTE_CHECK_NCCL(ncclEpInitHandle(&handle, ep_group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, num_topk,
                                   &routing_desc));
  return handle;
}

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

// Static-dtor teardown: skip NCCL calls (CUDA context / borrowed ep_comm_ may
// already be gone) and release in-memory state only.
EPBackend::~EPBackend() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!initialized_) return;
  lru_.clear();
  index_.clear();
  fallback_layer_cfg_.reset();
  ep_group_ = nullptr;
  ep_comm_ = nullptr;
  initialized_ = false;
}

void EPBackend::init(ncclComm_t ep_comm, NVTEEpGroupConfig group_config) {
  NVTE_CHECK(!initialized_, "EPBackend already initialized");

  group_config_ = group_config;

  ncclEpGroupConfig_t cfg = NCCL_EP_GROUP_CONFIG_INIT;
  cfg.algorithm = NCCL_EP_ALGO_HIGH_THROUGHPUT;
  cfg.num_experts = static_cast<unsigned int>(group_config.num_experts);
  cfg.max_dispatch_tokens_per_rank = static_cast<unsigned int>(group_config.max_tokens_per_rank);
  const size_t elem_bytes = typeToSize(static_cast<DType>(group_config.max_token_dtype));
  cfg.max_token_bytes = static_cast<unsigned int>(group_config.hidden_dim * elem_bytes);
  cfg.rdma_buffer_size = NCCL_EP_AUTO;
  cfg.num_qp_per_rank = NCCL_EP_AUTO;
  cfg.num_channels = NCCL_EP_AUTO;
  cfg.max_num_sms = group_config.num_comm_sms > 0
                        ? static_cast<unsigned int>(group_config.num_comm_sms)
                        : NCCL_EP_AUTO;
  // Must be > 0; NCCL EP errors out on 0.
  cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(group_config.max_recv_tokens_per_rank);
  cfg.zero_copy = group_config.zero_copy ? NCCL_EP_ZERO_COPY_ON : NCCL_EP_ZERO_COPY_OFF;

  NVTE_CHECK_NCCL(ncclEpCreateGroup(&ep_group_, ep_comm, &cfg));

  ep_comm_ = ep_comm;

  initialized_ = true;
}

// ---------------------------------------------------------------------------
// Pointer-keyed LRU cache
// ---------------------------------------------------------------------------

size_t EPBackend::cache_cap_locked() {
  if (handle_cache_cap_ == 0) {
    const char* cap_env = std::getenv("NVTE_EP_HANDLE_CACHE_SIZE");
    if (cap_env != nullptr) {
      const int64_t v = static_cast<int64_t>(std::atol(cap_env));
      if (v < 0) {
        // Unlimited cache. WAR for JAX until XLA fixes handle_mem
        // reloc between runs.
        handle_cache_cap_ = SIZE_MAX;
      } else {
        NVTE_CHECK(v > 0,
                   "NVTE_EP_HANDLE_CACHE_SIZE=0 is invalid; use -1 for unlimited or a positive "
                   "cap.");
        handle_cache_cap_ = static_cast<size_t>(v);
      }
    } else {
      handle_cache_cap_ = 4096;
    }
  }
  return handle_cache_cap_;
}

ncclEpHandle_t EPBackend::prepare_handle_locked(void* handle_mem, NVTEEpLayerConfig layer_cfg) {
  // Update the program-wide fallback cfg so dispatch/combine/_bwd can
  // reconstruct the handle on a pointer-cache miss (WAR for XLA buffer reloc
  // between runs; one cfg per process). Remove this once XLA preserves the
  // handle_mem device pointer across runs.
  if (fallback_layer_cfg_.has_value()) {
    NVTE_CHECK(fallback_layer_cfg_->top_k == layer_cfg.top_k, "EP prepare top_k=", layer_cfg.top_k,
               " disagrees with process-wide cached top_k=", fallback_layer_cfg_->top_k);
    NVTE_CHECK(fallback_layer_cfg_->dispatch_output_per_expert_alignment ==
                   layer_cfg.dispatch_output_per_expert_alignment,
               "EP prepare alignment=", layer_cfg.dispatch_output_per_expert_alignment,
               " disagrees with process-wide cached alignment=",
               fallback_layer_cfg_->dispatch_output_per_expert_alignment);
  } else {
    fallback_layer_cfg_ = layer_cfg;
  }

  auto it = index_.find(handle_mem);
  if (it != index_.end()) {
    lru_.splice(lru_.begin(), lru_, it->second);
    return it->second->handle;
  }
  ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
  hcfg.dispatch_output_per_expert_alignment = layer_cfg.dispatch_output_per_expert_alignment;
  size_t hm_size = 0;
  NVTE_CHECK_NCCL(ncclEpHandleMemSize(ep_group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, &hm_size,
                                      layer_cfg.top_k));
  ncclEpHandle_t h = open_handle(handle_mem, hm_size, layer_cfg.top_k,
                                 layer_cfg.dispatch_output_per_expert_alignment);
  lru_.push_front(HandleEntry{handle_mem, h, layer_cfg, hm_size});
  index_.emplace(handle_mem, lru_.begin());
  while (lru_.size() > cache_cap_locked()) {
    HandleEntry& victim = lru_.back();
    if (victim.handle != nullptr) ncclEpHandleDestroy(victim.handle);
    index_.erase(victim.handle_mem);
    lru_.pop_back();
  }
  return h;
}

ncclEpHandle_t EPBackend::lookup_handle_locked(void* handle_mem) {
  auto it = index_.find(handle_mem);
  if (it != index_.end()) {
    lru_.splice(lru_.begin(), lru_, it->second);
    return it->second->handle;
  }
  // Miss: reconstruct from the process-wide cached cfg. XLA may relocate
  // handle_mem between runs, breaking the pointer key; the fallback cfg lets
  // us open a fresh handle on the new buffer. Drop this branch once XLA
  // preserves buffer pointers.
  const uintptr_t hm_addr = reinterpret_cast<uintptr_t>(handle_mem);
  NVTE_CHECK(fallback_layer_cfg_.has_value(), "ep op on handle_mem=0x", hm_addr,
             " with no cached entry and no prior nvte_ep_prepare; call prepare first.");
  return prepare_handle_locked(handle_mem, *fallback_layer_cfg_);
}

// ---------------------------------------------------------------------------
// Per-step operations
// ---------------------------------------------------------------------------

size_t EPBackend::handle_mem_size(NVTEEpLayerConfig layer_cfg) {
  NVTE_CHECK(layer_cfg.top_k > 0, "top_k must be > 0, got ", layer_cfg.top_k);
  std::lock_guard<std::mutex> lock(mutex_);
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
  hcfg.dispatch_output_per_expert_alignment = layer_cfg.dispatch_output_per_expert_alignment;
  size_t hm_size = 0;
  NVTE_CHECK_NCCL(ncclEpHandleMemSize(ep_group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, &hm_size,
                                      layer_cfg.top_k));
  return hm_size;
}

void EPBackend::prepare(void* handle_mem, const NVTETensor topk_idx,
                        NVTETensor recv_tokens_per_expert,
                        NVTETensor /*total_recv_tokens_per_rank*/, NVTEEpLayerConfig layer_cfg,
                        cudaStream_t stream) {
  // total_recv_tokens_per_rank is a reserved placeholder; not yet populated.
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");
  NVTE_CHECK(layer_cfg.top_k > 0, "top_k must be > 0, got ", layer_cfg.top_k);
  NVTE_CHECK(nvte_tensor_shape(topk_idx).ndim == 2, "topk_idx must be 2D [T, top_k]");

  NVTEShape topk_idx_shape;
  ncclEpTensor_t nccl_topk_idx = make_nccl_ep_tensor(topk_idx, topk_idx_shape);

  // ncclEpUpdateHandle writes per-expert counts via expert_counters.
  NVTEShape recv_tokens_per_expert_shape;
  ncclEpTensor_t recv_tokens_per_expert_desc;
  if (recv_tokens_per_expert != nullptr) {
    recv_tokens_per_expert_desc =
        make_nccl_ep_tensor(recv_tokens_per_expert, recv_tokens_per_expert_shape);
  }
  ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
  layout_info.expert_counters =
      (recv_tokens_per_expert != nullptr) ? &recv_tokens_per_expert_desc : nullptr;

  std::lock_guard<std::mutex> lock(mutex_);
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  ncclEpHandle_t h = prepare_handle_locked(handle_mem, layer_cfg);
  NVTE_CHECK_NCCL(ncclEpUpdateHandle(h, &nccl_topk_idx, &layout_info, stream));
}

void EPBackend::dispatch(void* handle_mem, const NVTETensor topk_idx, const NVTETensor tokens,
                         const NVTECommWindow& tokens_win, const NVTETensor topk_weights,
                         const NVTECommWindow& topk_weights_win, NVTETensor recv_tokens,
                         const NVTECommWindow& recv_tokens_win, NVTETensor recv_topk_weights,
                         const NVTECommWindow& recv_topk_weights_win, cudaStream_t stream) {
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");
  NVTE_CHECK(nvte_tensor_shape(tokens).ndim == 2, "tokens must be 2D [T, hidden_dim]");
  NVTE_CHECK(nvte_tensor_shape(recv_tokens).ndim == 2,
             "recv_tokens must be 2D [recv_T, hidden_dim]");

  NVTEDType tok_dtype = nvte_tensor_type(tokens);
  NVTE_CHECK(typeToSize(static_cast<DType>(tok_dtype)) <=
                 typeToSize(static_cast<DType>(group_config_.max_token_dtype)),
             "tokens dtype (", static_cast<int>(tok_dtype), ") wider than group max_token_dtype (",
             static_cast<int>(group_config_.max_token_dtype), ")");
  NVTEDType recv_dtype = nvte_tensor_type(recv_tokens);
  NVTE_CHECK(typeToSize(static_cast<DType>(recv_dtype)) <=
                 typeToSize(static_cast<DType>(group_config_.max_token_dtype)),
             "recv_tokens dtype (", static_cast<int>(recv_dtype),
             ") wider than group max_token_dtype (",
             static_cast<int>(group_config_.max_token_dtype), ")");

  NVTEShape tokens_shape, recv_tokens_shape;
  ncclEpTensor_t nccl_tokens_in = make_nccl_ep_tensor(tokens, tokens_shape, tokens_win);
  ncclEpTensor_t nccl_tokens_out =
      make_nccl_ep_tensor(recv_tokens, recv_tokens_shape, recv_tokens_win);

  // Routing is cached in handle_mem by ep_prepare; dispatch only needs
  // topk_weights to reconstruct the sparse-to-dense prob map.
  const bool is_forward = (topk_weights != nullptr);
  NVTEShape topk_weights_shape, recv_topk_weights_shape;
  ncclEpTensor_t nccl_topk_weights_in;
  ncclEpTensor_t nccl_topk_weights_out;
  if (is_forward) {
    NVTE_CHECK(topk_idx != nullptr, "topk_idx required in forward dispatch");
    NVTE_CHECK(nvte_tensor_shape(topk_idx).ndim == 2, "topk_idx must be 2D [T, top_k]");
    NVTE_CHECK(nvte_tensor_shape(topk_weights).ndim == 2, "topk_weights must be 2D [T, top_k]");
    NVTE_CHECK(recv_topk_weights != nullptr,
               "recv_topk_weights must not be null in forward dispatch");
    NVTE_CHECK(nvte_tensor_shape(recv_topk_weights).ndim == 1,
               "recv_topk_weights must be 1D [recv_capacity]");
    nccl_topk_weights_in = make_nccl_ep_tensor(topk_weights, topk_weights_shape, topk_weights_win);
    nccl_topk_weights_out =
        make_nccl_ep_tensor(recv_topk_weights, recv_topk_weights_shape, recv_topk_weights_win);
  }

  ncclEpDispatchInputs_t in_struct = NCCL_EP_DISPATCH_INPUTS_INIT;
  in_struct.tokens = &nccl_tokens_in;
  in_struct.topk_weights = is_forward ? &nccl_topk_weights_in : nullptr;

  ncclEpDispatchOutputs_t out_struct = NCCL_EP_DISPATCH_OUTPUTS_INIT;
  out_struct.tokens = &nccl_tokens_out;
  out_struct.topk_weights = is_forward ? &nccl_topk_weights_out : nullptr;

  ncclEpDispatchConfig_t dispatch_cfg = NCCL_EP_DISPATCH_CONFIG_INIT;
  dispatch_cfg.pass_direction = is_forward ? NCCL_EP_FWD_PASS : NCCL_EP_BWD_PASS;

  std::lock_guard<std::mutex> lock(mutex_);
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  ncclEpHandle_t h = lookup_handle_locked(handle_mem);
  NVTE_CHECK_NCCL(ncclEpDispatch(h, &in_struct, &out_struct,
                                 /*layout_info=*/nullptr, &dispatch_cfg, stream));
}

void EPBackend::combine(void* handle_mem, const NVTETensor expert_out,
                        const NVTECommWindow& expert_out_win, NVTETensor result,
                        cudaStream_t stream) {
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");
  NVTE_CHECK(nvte_tensor_shape(expert_out).ndim == 2, "expert_out must be 2D [recv_T, hidden_dim]");
  NVTE_CHECK(nvte_tensor_shape(result).ndim == 2, "result must be 2D [T, hidden_dim]");

  NVTEShape expert_out_shape, result_shape;
  ncclEpTensor_t nccl_expert_in = make_nccl_ep_tensor(expert_out, expert_out_shape, expert_out_win);
  ncclEpTensor_t nccl_result_out = make_nccl_ep_tensor(result, result_shape);

  ncclEpCombineInputs_t in_struct = NCCL_EP_COMBINE_INPUTS_INIT;
  in_struct.tokens = &nccl_expert_in;

  ncclEpCombineOutputs_t out_struct = NCCL_EP_COMBINE_OUTPUTS_INIT;
  out_struct.tokens = &nccl_result_out;

  std::lock_guard<std::mutex> lock(mutex_);
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  ncclEpHandle_t h = lookup_handle_locked(handle_mem);
  NVTE_CHECK_NCCL(ncclEpCombine(h, &in_struct, &out_struct, /*config=*/nullptr, stream));
}

void EPBackend::dispatch_bwd(void* handle_mem, const NVTETensor grad,
                             const NVTECommWindow& grad_win, const NVTETensor g_recv_topk_weights,
                             const NVTECommWindow& g_recv_topk_weights_win, NVTETensor grad_tokens,
                             NVTETensor grad_topk_weights, cudaStream_t stream) {
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");
  NVTE_CHECK(nvte_tensor_shape(grad).ndim == 2, "grad must be 2D [recv_capacity, hidden_dim]");
  NVTE_CHECK(nvte_tensor_shape(grad_tokens).ndim == 2, "grad_tokens must be 2D [T, hidden_dim]");

  // g_recv_topk_weights must be 1D [recv_capacity]; caller flattens.
  NVTE_CHECK(nvte_tensor_shape(g_recv_topk_weights).ndim == 1,
             "g_recv_topk_weights must be 1D [recv_capacity]; caller must flatten leading dims");
  NVTE_CHECK(nvte_tensor_shape(grad_topk_weights).ndim == 2,
             "grad_topk_weights must be 2D [T, top_k]");

  NVTEShape grad_shape, g_recv_w_shape, grad_tokens_shape, grad_w_shape;
  ncclEpTensor_t nccl_tok_in = make_nccl_ep_tensor(grad, grad_shape, grad_win);
  ncclEpTensor_t nccl_w_in =
      make_nccl_ep_tensor(g_recv_topk_weights, g_recv_w_shape, g_recv_topk_weights_win);
  ncclEpTensor_t nccl_tok_out = make_nccl_ep_tensor(grad_tokens, grad_tokens_shape);
  ncclEpTensor_t nccl_w_out = make_nccl_ep_tensor(grad_topk_weights, grad_w_shape);

  ncclEpCombineInputs_t in_struct = NCCL_EP_COMBINE_INPUTS_INIT;
  in_struct.tokens = &nccl_tok_in;
  in_struct.topk_weights = &nccl_w_in;

  ncclEpCombineOutputs_t out_struct = NCCL_EP_COMBINE_OUTPUTS_INIT;
  out_struct.tokens = &nccl_tok_out;
  out_struct.topk_weights = &nccl_w_out;

  ncclEpCombineConfig_t cfg = NCCL_EP_COMBINE_CONFIG_INIT;
  cfg.pass_direction = NCCL_EP_BWD_PASS;

  std::lock_guard<std::mutex> lock(mutex_);
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  ncclEpHandle_t h = lookup_handle_locked(handle_mem);
  NVTE_CHECK_NCCL(ncclEpCombine(h, &in_struct, &out_struct, &cfg, stream));
}

void EPBackend::combine_bwd(void* handle_mem, const NVTETensor grad, const NVTECommWindow& grad_win,
                            NVTETensor grad_expert_out, const NVTECommWindow& grad_expert_out_win,
                            cudaStream_t stream) {
  // Backward of combine = reverse-direction dispatch.
  dispatch(handle_mem, /*topk_idx=*/nullptr, grad, grad_win,
           /*topk_weights=*/nullptr, /*topk_weights_win=*/NVTECommWindow{}, grad_expert_out,
           grad_expert_out_win,
           /*recv_topk_weights=*/nullptr, /*recv_topk_weights_win=*/NVTECommWindow{}, stream);
}

}  // namespace ep
}  // namespace transformer_engine
