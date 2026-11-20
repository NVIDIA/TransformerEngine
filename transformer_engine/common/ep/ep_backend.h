/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file ep_backend.h
 *  \brief Internal NCCL EP backend; not part of the public API. See ep.h.
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
#include <memory>
#include <mutex>
#include <unordered_map>

namespace transformer_engine {
namespace ep {

/*! \brief Readiness flags of an overlapped op: 1D int32, one flag per `granularity` units of the
 *         signaled tensor (rows of recv_tokens for a dispatch, hidden columns for a combine). */
struct EpSignal {
  // int32 readiness signal, one entry per granularity units; reset to -1, set to the slot index i
  // when that unit is ready (consumer waits for signal[s] == s).
  NVTETensor ready;
  size_t granularity;
};

/*! \brief EP backend per CUDA device; owns the NCCL EP group, borrows the comm. */
class EPBackend {
 public:
  /*! \brief Access the current device's backend. Aborts if not initialized. */
  static EPBackend& get();

  /*! \brief Bootstrap from an existing EP sub-communicator.
   *  ep_comm is borrowed; the caller keeps it alive until shutdown() returns
   *  and must span exactly config.ep_size ranks.
   *  Initializes the backend on the current CUDA device.
   */
  static void initialize(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  /*! \brief Tear down all local backends. Idempotent. Does not destroy ep_comm_. */
  static void shutdown();

  // Host-only: report handle_mem byte size for layer_cfg.
  // Returns the maximum across initialized local devices.
  static size_t handle_mem_size(NVTEEpLayerConfig layer_cfg);

  // Runs the routing AllGather into handle_mem; seeds the handle cache with layer_cfg when enabled.
  void prepare(NVTETensor handle_mem, const NVTETensor topk_idx, NVTETensor recv_tokens_per_expert,
               NVTETensor total_recv_tokens_per_rank, NVTEEpLayerConfig layer_cfg,
               cudaStream_t stream);

  // Per-step ops below require a prior prepare(). A non-null layer_cfg binds the handle from
  // handle_mem; null uses the handle cache (NVTE_EP_HANDLE_CACHE_SIZE != 0).
  // A non-null dispatch_signal publishes one flag per signal granularity rows of recv_tokens.
  void dispatch(NVTETensor handle_mem, const NVTETensor topk_idx, const NVTETensor tokens,
                const NVTECommWindow& tokens_win, const NVTETensor topk_weights,
                const NVTECommWindow& topk_weights_win, NVTETensor recv_tokens,
                const NVTECommWindow& recv_tokens_win, NVTETensor recv_topk_weights,
                const NVTECommWindow& recv_topk_weights_win, const NVTEEpLayerConfig* layer_cfg,
                cudaStream_t stream, const EpSignal* dispatch_signal = nullptr);

  // Fused prepare + dispatch: seeds routing then dispatches in one call. Routing writes the
  // per-expert recv counts to recv_tokens_per_expert and the scalar pre-drop per-rank recv total
  // to total_recv_tokens_per_rank (nullable); the dispatch then reads the counts from the handle.
  // expert_token_offsets (nullable): the padded per-expert token offsets; NCCL EP requires them,
  // the counts and the total, when given, to share one integer type.
  // dispatch_signal: as for dispatch().
  void prepare_and_dispatch(NVTETensor handle_mem, const NVTETensor topk_idx,
                            const NVTETensor tokens, const NVTECommWindow& tokens_win,
                            const NVTETensor topk_weights, const NVTECommWindow& topk_weights_win,
                            NVTETensor recv_tokens, const NVTECommWindow& recv_tokens_win,
                            NVTETensor recv_topk_weights,
                            const NVTECommWindow& recv_topk_weights_win,
                            NVTETensor recv_tokens_per_expert,
                            NVTETensor total_recv_tokens_per_rank, NVTEEpLayerConfig layer_cfg,
                            cudaStream_t stream, NVTETensor expert_token_offsets = nullptr,
                            const EpSignal* dispatch_signal = nullptr);

  // A non-null combine_signal makes the combine start on column chunks as the producer of
  // expert_out publishes them.
  void combine(NVTETensor handle_mem, const NVTETensor expert_out,
               const NVTECommWindow& expert_out_win, NVTETensor result,
               const NVTEEpLayerConfig* layer_cfg, cudaStream_t stream,
               const EpSignal* combine_signal = nullptr);

  // g_recv_topk_weights: 1D [recv_capacity] f32; grad_topk_weights: 2D [T, top_k] f32.
  // combine_signal: as for combine().
  void dispatch_bwd(NVTETensor handle_mem, const NVTETensor grad, const NVTECommWindow& grad_win,
                    const NVTETensor g_recv_topk_weights,
                    const NVTECommWindow& g_recv_topk_weights_win, NVTETensor grad_tokens,
                    NVTETensor grad_topk_weights, const NVTEEpLayerConfig* layer_cfg,
                    cudaStream_t stream, const EpSignal* combine_signal = nullptr);

  // dispatch_signal: as for dispatch(), over the rows of grad_expert_out.
  void combine_bwd(NVTETensor handle_mem, const NVTETensor grad, const NVTECommWindow& grad_win,
                   NVTETensor grad_expert_out, const NVTECommWindow& grad_expert_out_win,
                   const NVTEEpLayerConfig* layer_cfg, cudaStream_t stream,
                   const EpSignal* dispatch_signal = nullptr);

  const NVTEEpGroupConfig& group_config() const { return group_config_; }

  ~EPBackend();

 private:
  EPBackend() = default;
  EPBackend(const EPBackend&) = delete;
  EPBackend& operator=(const EPBackend&) = delete;

  // ep_comm is borrowed; caller retains ownership across the backend lifetime.
  void init(ncclComm_t ep_comm, NVTEEpGroupConfig config);

  // Caller must hold mutex_.
  void shutdown_locked();

  static EPBackend& instance(int device);
  static void validate_config(const NVTEEpGroupConfig& config);

  // Open a fresh ncclEpHandle over handle_mem. num_topk=-1 for paths
  // that don't carry per-token weights.
  ncclEpHandle_t open_handle(void* handle_mem, size_t handle_mem_size, int num_topk,
                             size_t dispatch_output_per_expert_alignment);

  // Pointer-keyed handle cache, capped at NVTE_EP_HANDLE_CACHE_SIZE live handles.
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
  std::list<HandleEntry> handles_;
  std::unordered_map<void*, std::list<HandleEntry>::iterator> index_;
  size_t handle_cache_cap_{
      0};  // read lazily from NVTE_EP_HANDLE_CACHE_SIZE; 0 (default) = cache disabled
  bool handle_cache_cap_set_{false};

  // Caller must hold mutex_.
  ncclEpHandle_t prepare_handle_locked(void* handle_mem, size_t handle_mem_size,
                                       NVTEEpLayerConfig layer_cfg);
  ncclEpHandle_t lookup_handle_locked(void* handle_mem, size_t handle_mem_size);
  size_t cache_cap_locked();

  // Bind a fresh handle to handle_mem's already-prepared contents via ncclEpImportHandle.
  // The caller owns the returned handle. Caller must hold mutex_.
  ncclEpHandle_t import_handle_locked(void* handle_mem, size_t handle_mem_size, int num_tokens,
                                      NVTEEpLayerConfig layer_cfg);

  // Per-step handle: imported when layer_cfg is set (*owned = true), else cached (*owned =
  // false). Caller must hold mutex_.
  ncclEpHandle_t acquire_step_handle_locked(void* handle_mem, size_t handle_mem_size,
                                            int num_tokens, const NVTEEpLayerConfig* layer_cfg,
                                            bool* owned);

  // Build the dispatch in/out structs and issue ncclEpDispatch on the resolved
  // handle. When recv_tokens_per_expert != nullptr (count mode), it is wired to
  // layout_info.expert_counters so the dispatch writes per-expert recv counts; likewise
  // expert_token_offsets to layout_info.expert_offsets.
  // Caller must hold mutex_.
  void issue_dispatch_locked(ncclEpHandle_t handle, const NVTETensor topk_idx,
                             const NVTETensor tokens, const NVTECommWindow& tokens_win,
                             const NVTETensor topk_weights, const NVTECommWindow& topk_weights_win,
                             NVTETensor recv_tokens, const NVTECommWindow& recv_tokens_win,
                             NVTETensor recv_topk_weights,
                             const NVTECommWindow& recv_topk_weights_win,
                             NVTETensor recv_tokens_per_expert,
                             NVTETensor total_recv_tokens_per_rank, NVTETensor expert_token_offsets,
                             const EpSignal* dispatch_signal, cudaStream_t stream);

  // devices_mutex_ protects the map; each backend's mutex_ protects its state.
  static std::mutex devices_mutex_;
  static std::unordered_map<int, std::unique_ptr<EPBackend>> devices_;
};

// A caller's versioned layer config copied into the current layout.
NVTEEpLayerConfig normalize_layer_config(const NVTEEpLayerConfig* cfg);

}  // namespace ep
}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_EP_EP_BACKEND_H_
