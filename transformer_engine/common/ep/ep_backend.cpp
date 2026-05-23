/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.cpp
 *  \brief EPBackend implementation. See ep_backend.h for the op flow.
 */

#include "ep_backend.h"

#include <algorithm>
#include <atomic>
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

// Build a by-value ncclEpTensor_t descriptor. `sizes` is caller-owned and must
// outlive any NCCL EP call that consumes the descriptor.
inline ncclEpTensor_t make_tensor(void* data, unsigned int ndim, ncclDataType_t datatype,
                                  size_t* sizes) {
  ncclEpTensor_t t = NCCL_EP_TENSOR_INIT;
  t.ndim = ndim;
  t.datatype = datatype;
  t.data = data;
  t.sizes = sizes;
  return t;
}

// Payload descriptor: prefer the symmem window when set, else fall back to the
// NVTETensor's raw device pointer.
inline ncclEpTensor_t make_payload_tensor(const NVTETensor t, const NVTECommWindow& win,
                                          unsigned int ndim, ncclDataType_t datatype,
                                          size_t* sizes) {
  ncclEpTensor_t desc = NCCL_EP_TENSOR_INIT;
  desc.ndim = ndim;
  desc.datatype = datatype;
  desc.sizes = sizes;
  if (win.window != nullptr) {
    desc.win_hdl = win.window;
    desc.win_offset = win.offset;
  } else {
    desc.data = nvte_tensor_data(t);
    NVTE_CHECK(desc.data != nullptr, "payload tensor data must not be null");
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
  NVTE_CHECK(config.hidden_dim * sizeof(nv_bfloat16) >= 16,
             "hidden_dim * 2 must be >= 16 (NCCL EP 16B row alignment); got hidden_dim=",
             config.hidden_dim);
  NVTE_CHECK(config.num_experts % config.ep_size == 0, "num_experts (", config.num_experts,
             ") must be divisible by ep_size (", config.ep_size, ")");
  NVTE_CHECK(config.max_num_sms >= 0, "max_num_sms must be >= 0 (0 = auto), got ",
             config.max_num_sms);

  int device, major;
  NVTE_CHECK_CUDA(cudaGetDevice(&device));
  NVTE_CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  NVTE_CHECK(major >= 9,
             "NCCL EP requires SM_90+ (Hopper or later), "
             "but current device has compute capability ",
             major, ".x");

  // NCCL EP needs CUDA multicast (NVLS); init hangs without it.
  NVTE_CHECK(cuda::supports_multicast(device),
             "NCCL EP requires CUDA multicast (NVLS) support on device ", device,
             " but CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED reports 0.");
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
  for (auto& kv : inst.handles_) {
    if (kv.second.cached_handle != nullptr) {
      ncclEpHandleDestroy(kv.second.cached_handle);
      kv.second.cached_handle = nullptr;
      kv.second.cached_handle_mem = nullptr;
    }
  }
  inst.handles_.clear();
  // ncclEpGroupDestroy reads from ep_comm_; destroy group while comm is still alive.
  if (inst.ep_group_ != nullptr) {
    ncclEpGroupDestroy(inst.ep_group_);
    inst.ep_group_ = nullptr;
  }
  inst.ep_comm_ = nullptr;  // borrowed — caller destroys
  inst.initialized_ = false;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

ncclDataType_t EPBackend::nvte_dtype_to_nccl(NVTEDType dtype) {
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
      NVTE_ERROR("Unsupported NVTEDType for NCCL EP conversion: ", static_cast<int>(dtype));
  }
  return ncclFloat32;  // unreachable
}

// Open a fresh ncclEpHandle over handle_mem. Caller (or cache) owns the result.
ncclEpHandle_t EPBackend::open_handle(void* handle_mem, size_t handle_mem_size, int num_topk,
                                      size_t dispatch_output_per_expert_alignment) {
  size_t hm_sizes[1] = {handle_mem_size};
  ncclEpTensor_t routing_desc = make_tensor(handle_mem, 1, ncclUint8, hm_sizes);
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
  handles_.clear();
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
  cfg.max_token_bytes = static_cast<unsigned int>(group_config.hidden_dim * sizeof(nv_bfloat16));
  cfg.rdma_buffer_size = NCCL_EP_AUTO;
  cfg.num_qp_per_rank = NCCL_EP_AUTO;
  cfg.num_channels = NCCL_EP_AUTO;
  cfg.max_num_sms = group_config.max_num_sms > 0
                        ? static_cast<unsigned int>(group_config.max_num_sms)
                        : NCCL_EP_AUTO;
  // Must be > 0; NCCL EP errors out on 0.
  cfg.max_recv_tokens_per_rank = static_cast<unsigned int>(group_config.max_recv_tokens_per_rank);

  NVTE_CHECK_NCCL(ncclEpCreateGroup(&ep_group_, ep_comm, &cfg));

  ep_comm_ = ep_comm;

  initialized_ = true;
}

// ---------------------------------------------------------------------------
// Per-handle_id config cache
// ---------------------------------------------------------------------------

uint64_t EPBackend::insert_new_entry(size_t handle_mem_size, int top_k, size_t alignment) {
  if (handle_cache_cap_ == 0) {
    const char* cap_env = std::getenv("NVTE_EP_HANDLE_CACHE_SIZE");
    handle_cache_cap_ = (cap_env != nullptr) ? std::max<size_t>(1, std::atoi(cap_env)) : 8192;
  }
  NVTE_CHECK(handles_.size() < handle_cache_cap_, "EP handle cache full (", handle_cache_cap_,
             " entries). Raise via NVTE_EP_HANDLE_CACHE_SIZE.");
  uint64_t id = next_handle_id_.fetch_add(1, std::memory_order_relaxed);
  handles_.emplace(id, HandleEntry{handle_mem_size, alignment, top_k});
  return id;
}

EPBackend::HandleEntry& EPBackend::lookup_config(uint64_t handle_id) {
  auto it = handles_.find(handle_id);
  NVTE_CHECK(it != handles_.end(), "ep op on handle_id=", handle_id,
             " with no cached config — call ep_prepare first.");
  return it->second;
}

ncclEpHandle_t EPBackend::get_or_open_handle(HandleEntry& cfg, void* handle_mem) {
  if (cfg.cached_handle != nullptr && cfg.cached_handle_mem == handle_mem) {
    return cfg.cached_handle;
  }
  if (cfg.cached_handle != nullptr) {
    NVTE_CHECK(group_config_.allow_handle_mem_reloc != 0,
               "EP handle_mem relocated for cached handle (old=",
               reinterpret_cast<uintptr_t>(cfg.cached_handle_mem),
               ", new=", reinterpret_cast<uintptr_t>(handle_mem),
               "). Set NVTEEpGroupConfig.allow_handle_mem_reloc=1 to allow rebuild.");
    ncclEpHandleDestroy(cfg.cached_handle);
    cfg.cached_handle = nullptr;
    cfg.cached_handle_mem = nullptr;
  }
  ncclEpHandle_t h = open_handle(handle_mem, cfg.handle_mem_size, cfg.top_k, cfg.alignment);
  cfg.cached_handle = h;
  cfg.cached_handle_mem = handle_mem;
  return h;
}

// ---------------------------------------------------------------------------
// Per-step operations
// ---------------------------------------------------------------------------

uint64_t EPBackend::register_layer(NVTEEpLayerConfig layer_config, size_t* handle_mem_size) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(layer_config.top_k > 0, "NVTEEpLayerConfig.top_k must be > 0");
  NVTE_CHECK(handle_mem_size != nullptr, "handle_mem_size must not be null");
  ncclEpHandleConfig_t hcfg = NCCL_EP_HANDLE_CONFIG_INIT;
  hcfg.dispatch_output_per_expert_alignment = layer_config.dispatch_output_per_expert_alignment;
  size_t hm_size = 0;
  NVTE_CHECK_NCCL(ncclEpHandleMemSize(ep_group_, NCCL_EP_LAYOUT_EXPERT_MAJOR, &hcfg, &hm_size,
                                      layer_config.top_k));
  *handle_mem_size = hm_size;
  std::lock_guard<std::mutex> lock(mutex_);
  return insert_new_entry(hm_size, layer_config.top_k,
                          layer_config.dispatch_output_per_expert_alignment);
}

void EPBackend::prepare(uint64_t handle_id, const NVTETensor topk_idx, NVTETensor token_counts,
                        void* handle_mem, size_t dispatch_output_per_expert_alignment,
                        cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
  void* idx_data = nvte_tensor_data(topk_idx);
  NVTE_CHECK(idx_data != nullptr, "topk_idx data must not be null");

  const size_t num_tokens = idx_shape.data[0];
  const size_t top_k = idx_shape.ndim > 1 ? idx_shape.data[1] : 1;
  const size_t num_local_experts =
      static_cast<size_t>(group_config_.num_experts / group_config_.ep_size);

  size_t idx_sizes[2] = {num_tokens, top_k};
  ncclEpTensor_t nccl_topk_idx = make_tensor(idx_data, 2, ncclInt64, idx_sizes);

  // ncclEpUpdateHandle writes per-expert counts via expert_counters.
  size_t cnt_sizes[1] = {num_local_experts};
  ncclEpTensor_t token_counts_desc;
  void* token_counts_data = (token_counts != nullptr) ? nvte_tensor_data(token_counts) : nullptr;
  if (token_counts_data != nullptr) {
    token_counts_desc = make_tensor(token_counts_data, 1, ncclInt32, cnt_sizes);
  }
  ncclEpLayoutInfo_t layout_info = NCCL_EP_LAYOUT_INFO_INIT;
  layout_info.expert_counters = (token_counts_data != nullptr) ? &token_counts_desc : nullptr;

  std::lock_guard<std::mutex> lock(mutex_);
  HandleEntry& cfg = lookup_config(handle_id);
  NVTE_CHECK(cfg.alignment == dispatch_output_per_expert_alignment,
             "ep_prepare: alignment mismatch for handle_id=", handle_id, " (cached=", cfg.alignment,
             ", got=", dispatch_output_per_expert_alignment, ")");
  ncclEpHandle_t h = get_or_open_handle(cfg, handle_mem);
  NVTE_CHECK_NCCL(ncclEpUpdateHandle(h, &nccl_topk_idx, &layout_info, stream));
}

void EPBackend::dispatch(uint64_t handle_id, void* handle_mem, const NVTETensor topk_idx,
                         const NVTETensor tokens, const NVTECommWindow& tokens_win,
                         const NVTETensor topk_weights, const NVTECommWindow& topk_weights_win,
                         NVTETensor recv_tokens, const NVTECommWindow& recv_tokens_win,
                         NVTETensor recv_topk_weights, const NVTECommWindow& recv_topk_weights_win,
                         cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape tok_shape = nvte_tensor_shape(tokens);
  NVTEDType tok_dtype = nvte_tensor_type(tokens);

  const size_t num_tokens = tok_shape.data[0];
  const size_t hidden_dim = tok_shape.data[1];

  size_t tok_sizes[2] = {num_tokens, hidden_dim};
  ncclEpTensor_t nccl_tokens_in =
      make_payload_tensor(tokens, tokens_win, 2, nvte_dtype_to_nccl(tok_dtype), tok_sizes);

  const bool is_forward = (topk_weights != nullptr);

  // Routing is cached in handle_mem by ep_prepare; dispatch only needs
  // topk_weights to reconstruct the sparse-to-dense prob map.
  size_t weights_in_sizes[2] = {0, 0};
  ncclEpTensor_t nccl_topk_weights_in;
  if (is_forward) {
    NVTE_CHECK(topk_idx != nullptr, "topk_idx required in forward dispatch");
    NVTEShape idx_shape = nvte_tensor_shape(topk_idx);
    const size_t top_k = idx_shape.ndim > 1 ? idx_shape.data[1] : 1;
    weights_in_sizes[0] = num_tokens;
    weights_in_sizes[1] = top_k;
    nccl_topk_weights_in =
        make_payload_tensor(topk_weights, topk_weights_win, 2, ncclFloat32, weights_in_sizes);
  }

  NVTEShape recv_shape = nvte_tensor_shape(recv_tokens);
  NVTEDType recv_dtype = nvte_tensor_type(recv_tokens);

  size_t recv_sizes[2] = {recv_shape.data[0], recv_shape.data[1]};
  ncclEpTensor_t nccl_tokens_out = make_payload_tensor(recv_tokens, recv_tokens_win, 2,
                                                       nvte_dtype_to_nccl(recv_dtype), recv_sizes);

  size_t weights_out_sizes[1] = {recv_shape.data[0]};
  ncclEpTensor_t nccl_topk_weights_out;
  if (is_forward) {
    NVTE_CHECK(recv_topk_weights != nullptr,
               "recv_topk_weights must not be null in forward dispatch");
    NVTEShape recv_w_shape = nvte_tensor_shape(recv_topk_weights);
    NVTE_CHECK(recv_w_shape.ndim == 1, "recv_topk_weights must be 1D [recv_capacity]");
    nccl_topk_weights_out = make_payload_tensor(recv_topk_weights, recv_topk_weights_win, 1,
                                                ncclFloat32, weights_out_sizes);
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
  HandleEntry& cfg = lookup_config(handle_id);
  ncclEpHandle_t h = get_or_open_handle(cfg, handle_mem);
  NVTE_CHECK_NCCL(ncclEpDispatch(h, &in_struct, &out_struct,
                                 /*layout_info=*/nullptr, &dispatch_cfg, stream));
}

void EPBackend::combine(uint64_t handle_id, void* handle_mem, const NVTETensor expert_out,
                        const NVTECommWindow& expert_out_win, NVTETensor result,
                        cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape exp_shape = nvte_tensor_shape(expert_out);
  NVTEDType exp_dtype = nvte_tensor_type(expert_out);

  size_t exp_sizes[2] = {exp_shape.data[0], exp_shape.data[1]};
  ncclEpTensor_t nccl_expert_in =
      make_payload_tensor(expert_out, expert_out_win, 2, nvte_dtype_to_nccl(exp_dtype), exp_sizes);

  NVTEShape res_shape = nvte_tensor_shape(result);
  void* res_data = nvte_tensor_data(result);
  NVTEDType res_dtype = nvte_tensor_type(result);
  NVTE_CHECK(res_data != nullptr, "result data must not be null");

  size_t res_sizes[2] = {res_shape.data[0], res_shape.data[1]};
  ncclEpTensor_t nccl_result_out =
      make_tensor(res_data, 2, nvte_dtype_to_nccl(res_dtype), res_sizes);

  ncclEpCombineInputs_t in_struct = NCCL_EP_COMBINE_INPUTS_INIT;
  in_struct.tokens = &nccl_expert_in;

  ncclEpCombineOutputs_t out_struct = NCCL_EP_COMBINE_OUTPUTS_INIT;
  out_struct.tokens = &nccl_result_out;

  std::lock_guard<std::mutex> lock(mutex_);
  HandleEntry& cfg = lookup_config(handle_id);
  ncclEpHandle_t h = get_or_open_handle(cfg, handle_mem);
  NVTE_CHECK_NCCL(ncclEpCombine(h, &in_struct, &out_struct, /*config=*/nullptr, stream));
}

void EPBackend::dispatch_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                             const NVTECommWindow& grad_win, const NVTETensor g_recv_topk_weights,
                             const NVTECommWindow& g_recv_topk_weights_win, NVTETensor grad_tokens,
                             NVTETensor grad_topk_weights, cudaStream_t stream) {
  NVTE_CHECK(initialized_, "EPBackend not initialized");
  NVTE_CHECK(handle_mem != nullptr, "handle_mem must not be null");

  NVTEShape g_shape = nvte_tensor_shape(grad);
  NVTEDType g_dtype = nvte_tensor_type(grad);
  size_t g_sizes[2] = {g_shape.data[0], g_shape.data[1]};
  ncclEpTensor_t nccl_tok_in =
      make_payload_tensor(grad, grad_win, 2, nvte_dtype_to_nccl(g_dtype), g_sizes);

  // g_recv_topk_weights must be 1D [recv_capacity] — caller flattens.
  NVTEShape gw_shape = nvte_tensor_shape(g_recv_topk_weights);
  NVTE_CHECK(gw_shape.ndim == 1,
             "g_recv_topk_weights must be 1D [recv_capacity]; caller must flatten leading dims");
  size_t gw_sizes[1] = {gw_shape.data[0]};
  ncclEpTensor_t nccl_w_in =
      make_payload_tensor(g_recv_topk_weights, g_recv_topk_weights_win, 1, ncclFloat32, gw_sizes);

  NVTEShape gt_shape = nvte_tensor_shape(grad_tokens);
  void* gt_data = nvte_tensor_data(grad_tokens);
  NVTE_CHECK(gt_data != nullptr, "grad_tokens data must not be null");
  size_t gt_sizes[2] = {gt_shape.data[0], gt_shape.data[1]};
  ncclEpTensor_t nccl_tok_out = make_tensor(gt_data, 2, nvte_dtype_to_nccl(g_dtype), gt_sizes);

  NVTEShape gtw_shape = nvte_tensor_shape(grad_topk_weights);
  void* gtw_data = nvte_tensor_data(grad_topk_weights);
  NVTE_CHECK(gtw_data != nullptr, "grad_topk_weights data must not be null");
  NVTE_CHECK(gtw_shape.ndim == 2, "grad_topk_weights must be 2D [T, top_k]");
  size_t gtw_sizes[2] = {gtw_shape.data[0], gtw_shape.data[1]};
  ncclEpTensor_t nccl_w_out = make_tensor(gtw_data, 2, ncclFloat32, gtw_sizes);

  ncclEpCombineInputs_t in_struct = NCCL_EP_COMBINE_INPUTS_INIT;
  in_struct.tokens = &nccl_tok_in;
  in_struct.topk_weights = &nccl_w_in;

  ncclEpCombineOutputs_t out_struct = NCCL_EP_COMBINE_OUTPUTS_INIT;
  out_struct.tokens = &nccl_tok_out;
  out_struct.topk_weights = &nccl_w_out;

  ncclEpCombineConfig_t cfg = NCCL_EP_COMBINE_CONFIG_INIT;
  cfg.pass_direction = NCCL_EP_BWD_PASS;

  std::lock_guard<std::mutex> lock(mutex_);
  HandleEntry& entry = lookup_config(handle_id);
  ncclEpHandle_t h = get_or_open_handle(entry, handle_mem);
  NVTE_CHECK_NCCL(ncclEpCombine(h, &in_struct, &out_struct, &cfg, stream));
}

void EPBackend::combine_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                            const NVTECommWindow& grad_win, NVTETensor grad_expert_out,
                            const NVTECommWindow& grad_expert_out_win, cudaStream_t stream) {
  // Backward of combine = reverse-direction dispatch.
  dispatch(handle_id, handle_mem, /*topk_idx=*/nullptr, grad, grad_win, /*topk_weights=*/nullptr,
           /*topk_weights_win=*/NVTECommWindow{}, grad_expert_out, grad_expert_out_win,
           /*recv_topk_weights=*/nullptr, /*recv_topk_weights_win=*/NVTECommWindow{}, stream);
}

}  // namespace ep
}  // namespace transformer_engine
