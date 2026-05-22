/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.h
 *  \brief Internal NCCL EP singleton; not part of the public API.
 *
 *  Per handle_id the cache stores config only (no device pointers), so
 *  handle_mem may be relocated between ops. Cap: NVTE_EP_HANDLE_CACHE_SIZE
 *  (default 8192); overflow throws.
 */

#ifndef TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
#define TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_

#include <cuda_runtime_api.h>
#include <nccl.h>
#include <nccl_ep.h>
#include <transformer_engine/ep.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace transformer_engine {
namespace ep {

/*! \brief EP backend singleton — owns the NCCL EP group; borrows the comm. */
class EPBackend {
 public:
  /*! \brief Access the singleton. Aborts if not initialized. */
  static EPBackend& get();

  /*! \brief Bootstrap from an existing EP sub-communicator.
   *  ep_comm is borrowed; the caller keeps it alive until shutdown() returns
   *  and must span exactly config.ep_size ranks.
   */
  static void initialize(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  /*! \brief Tear down the backend. Idempotent. Does not destroy ep_comm_. */
  static void shutdown();

  // Host-only: reserve a fresh handle_id, cache the layer config, and report
  // the handle_mem buffer size the caller must allocate.
  uint64_t register_layer(NVTEEpLayerConfig layer_config, size_t* handle_mem_size);

  void prepare(uint64_t handle_id, const NVTETensor topk_idx, NVTETensor token_counts,
               void* handle_mem, size_t dispatch_output_per_expert_alignment, cudaStream_t stream);

  void dispatch(uint64_t handle_id, void* handle_mem, const NVTETensor topk_idx,
                const NVTETensor tokens, const NVTECommWindow& tokens_win,
                const NVTETensor topk_weights, const NVTECommWindow& topk_weights_win,
                NVTETensor recv_tokens, const NVTECommWindow& recv_tokens_win,
                NVTETensor recv_topk_weights, const NVTECommWindow& recv_topk_weights_win,
                cudaStream_t stream);

  void combine(uint64_t handle_id, void* handle_mem, const NVTETensor expert_out,
               const NVTECommWindow& expert_out_win, NVTETensor result, cudaStream_t stream);

  // g_recv_topk_weights: 1D [recv_capacity] f32; grad_topk_weights: 2D [T, top_k] f32.
  void dispatch_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                    const NVTECommWindow& grad_win, const NVTETensor g_recv_topk_weights,
                    const NVTECommWindow& g_recv_topk_weights_win, NVTETensor grad_tokens,
                    NVTETensor grad_topk_weights, cudaStream_t stream);

  void combine_bwd(uint64_t handle_id, void* handle_mem, const NVTETensor grad,
                   const NVTECommWindow& grad_win, NVTETensor grad_expert_out,
                   const NVTECommWindow& grad_expert_out_win, cudaStream_t stream);

 private:
  EPBackend() = default;
  ~EPBackend();
  EPBackend(const EPBackend&) = delete;
  EPBackend& operator=(const EPBackend&) = delete;

  // ep_comm is borrowed — caller retains ownership across the backend lifetime.
  void init(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  static EPBackend& instance();  // Meyers singleton accessor
  static void validate_config(const NVTEEpGroupConfig& config);

  static ncclDataType_t nvte_dtype_to_nccl(NVTEDType dtype);
  // Open a transient ncclEpHandle over handle_mem. num_topk=-1 for paths
  // that don't carry per-token weights.
  ncclEpHandle_t open_handle(void* handle_mem, size_t handle_mem_size, int num_topk,
                             size_t dispatch_output_per_expert_alignment);

  ncclEpGroup_t ep_group_{nullptr};
  ncclComm_t ep_comm_{nullptr};
  NVTEEpGroupConfig group_config_{};
  bool initialized_{false};
  std::mutex mutex_;
  struct HandleEntry {
    size_t handle_mem_size;
    size_t alignment;
    int top_k;
  };
  std::unordered_map<uint64_t, HandleEntry> handles_;
  std::atomic<uint64_t> next_handle_id_{1};  // 0 reserved as "no id"
  size_t handle_cache_cap_{0};               // set lazily from NVTE_EP_HANDLE_CACHE_SIZE

  // Caller must hold mutex_. Throws on cap overflow.
  uint64_t insert_new_entry(size_t handle_mem_size, int top_k, size_t alignment);
  HandleEntry& lookup_config(uint64_t handle_id);
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
