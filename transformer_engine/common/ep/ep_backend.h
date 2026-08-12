/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.h
 *  \brief Internal NCCL EP singleton; not part of the public API. See ep.h.
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
#include <list>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace transformer_engine {
namespace ep {

/*! \brief EP backend singleton; owns the NCCL EP group, borrows the comm. */
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

  // Host-only: report handle_mem byte size for layer_cfg.
  size_t handle_mem_size(NVTEEpLayerConfig layer_cfg);

  // Seeds the cache for handle_mem with layer_cfg and runs the routing AllGather.
  void prepare(void* handle_mem, const NVTETensor topk_idx, NVTETensor recv_tokens_per_expert,
               NVTETensor total_recv_tokens_per_rank, NVTEEpLayerConfig layer_cfg,
               cudaStream_t stream);

  // Per-step ops below require a prior prepare().
  void dispatch(void* handle_mem, const NVTETensor topk_idx, const NVTETensor tokens,
                const NVTECommWindow& tokens_win, const NVTETensor topk_weights,
                const NVTECommWindow& topk_weights_win, NVTETensor recv_tokens,
                const NVTECommWindow& recv_tokens_win, NVTETensor recv_topk_weights,
                const NVTECommWindow& recv_topk_weights_win, cudaStream_t stream);

  void combine(void* handle_mem, const NVTETensor expert_out, const NVTECommWindow& expert_out_win,
               NVTETensor result, cudaStream_t stream);

  // g_recv_topk_weights: 1D [recv_capacity] f32; grad_topk_weights: 2D [T, top_k] f32.
  void dispatch_bwd(void* handle_mem, const NVTETensor grad, const NVTECommWindow& grad_win,
                    const NVTETensor g_recv_topk_weights,
                    const NVTECommWindow& g_recv_topk_weights_win, NVTETensor grad_tokens,
                    NVTETensor grad_topk_weights, cudaStream_t stream);

  void combine_bwd(void* handle_mem, const NVTETensor grad, const NVTECommWindow& grad_win,
                   NVTETensor grad_expert_out, const NVTECommWindow& grad_expert_out_win,
                   cudaStream_t stream);

 private:
  EPBackend() = default;
  ~EPBackend();
  EPBackend(const EPBackend&) = delete;
  EPBackend& operator=(const EPBackend&) = delete;

  // ep_comm is borrowed; caller retains ownership across the backend lifetime.
  void init(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  static EPBackend& instance();  // Meyers singleton accessor
  static void validate_config(const NVTEEpGroupConfig& config);

  // Open a fresh ncclEpHandle over handle_mem. num_topk=-1 for paths
  // that don't carry per-token weights.
  ncclEpHandle_t open_handle(void* handle_mem, size_t handle_mem_size, int num_topk,
                             size_t dispatch_output_per_expert_alignment);

  // LRU cache: most-recently-used at the front of lru_, evict from the back.
  struct HandleEntry {
    void* handle_mem;
    ncclEpHandle_t handle;
    NVTEEpLayerConfig layer_cfg;
    size_t handle_mem_size;
  };

  ncclEpGroup_t ep_group_{nullptr};
  ncclComm_t ep_comm_{nullptr};
  NVTEEpGroupConfig group_config_{};
  std::atomic<bool> initialized_{false};
  std::mutex mutex_;
  std::list<HandleEntry> lru_;
  std::unordered_map<void*, std::list<HandleEntry>::iterator> index_;
  size_t handle_cache_cap_{0};  // set lazily from NVTE_EP_HANDLE_CACHE_SIZE
  std::optional<NVTEEpLayerConfig> fallback_layer_cfg_;

  // Caller must hold mutex_.
  ncclEpHandle_t prepare_handle_locked(void* handle_mem, NVTEEpLayerConfig layer_cfg);
  ncclEpHandle_t lookup_handle_locked(void* handle_mem);
  size_t cache_cap_locked();
};

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
