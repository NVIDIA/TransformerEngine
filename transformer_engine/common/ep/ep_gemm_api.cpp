/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/ep_gemm.h>

#include "../util/logging.h"

#if defined(NVTE_WITH_NCCL_EP) && defined(NVTE_WITH_CUTEDSL)

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../common.h"
#include "../util/cuda_runtime.h"
#include "ep_backend.h"
#include "ep_gemm_config.h"
#include "ep_overlap_workspace.h"

namespace ep = transformer_engine::ep;

namespace {

constexpr size_t kWorkspaceAlign = 128;
// Rows per GEMM M tile (cluster tile M). Smaller expert alignments end an expert in a tail tile.
constexpr size_t kRowsPerTile = 256;
// GEMM CTA tile N; combine readiness flags cover a multiple of it.
constexpr size_t kTileN = 256;
constexpr size_t kClusterSize = 2;

size_t align_up(size_t value) {
  return (value + kWorkspaceAlign - 1) / kWorkspaceAlign * kWorkspaceAlign;
}

// Per-device scratch for the readiness flags and GEMM state, grown geometrically on demand. An
// outgrown block stays allocated until EP shutdown: a captured CUDA graph may still use it.
struct ScratchBlock {
  void* ptr = nullptr;
  size_t bytes = 0;
  std::vector<void*> retired;

  void release() {
    retired.push_back(ptr);
    for (void* p : retired) {
      if (p != nullptr) {
        cudaFree(p);
      }
    }
    ptr = nullptr;
    bytes = 0;
    retired.clear();
  }
};

std::mutex scratch_mutex;
std::map<int, ScratchBlock>& scratch_blocks() {
  // Leaked: no CUDA call at static teardown.
  static auto* blocks = new std::map<int, ScratchBlock>();
  return *blocks;
}

uint8_t* get_scratch(size_t bytes) {
  std::lock_guard<std::mutex> lock(scratch_mutex);
  ScratchBlock& block = scratch_blocks()[transformer_engine::cuda::current_device()];
  if (block.bytes < bytes) {
    void* ptr = nullptr;
    const size_t grown = std::max(bytes, 2 * block.bytes);
    NVTE_CHECK_CUDA(cudaMalloc(&ptr, grown));
    block.retired.push_back(block.ptr);
    block.ptr = ptr;
    block.bytes = grown;
  }
  return static_cast<uint8_t*>(block.ptr);
}

using transformer_engine::tvm_ffi_bridge::DLTensorWrapper;

// Operands live on the current device.
DLTensorWrapper wrap_data(const NVTEBasicTensor& data, bool flatten_2d) {
  return DLTensorWrapper(data, flatten_2d, transformer_engine::cuda::current_device());
}

DLTensorWrapper wrap_rowwise(NVTETensor t, bool flatten_2d) {
  return wrap_data(NVTEBasicTensor{nvte_tensor_data(t), nvte_tensor_type(t), nvte_tensor_shape(t)},
                   flatten_2d);
}

DLTensorWrapper wrap_raw(void* ptr, NVTEDType dtype, size_t count) {
  NVTEShape shape{};
  shape.ndim = 1;
  shape.data[0] = count;
  return wrap_data(NVTEBasicTensor{ptr, dtype, shape}, false);
}

bool is_mxfp8(NVTETensor t) { return nvte_tensor_scaling_mode(t) == NVTE_MXFP8_1D_SCALING; }

// An MXFP8 operand: E4M3 data with compact or GEMM-swizzled scales.
void check_mxfp8(NVTETensor t, bool swizzled, const char* name) {
  NVTE_CHECK(t != nullptr && is_mxfp8(t) && nvte_tensor_type(t) == kNVTEFloat8E4M3, name,
             " must be MXFP8 E4M3");
  uint8_t with_swizzled = 0;
  nvte_get_tensor_param_v2(t, kNVTEWithGEMMSwizzledScales, &with_swizzled, sizeof(with_swizzled),
                           nullptr);
  NVTE_CHECK((with_swizzled != 0) == swizzled, name,
             swizzled ? " needs GEMM-swizzled scales" : " needs compact scales");
}

// Shapes the MXFP8 GEMMs take: compact scale rows of whole 16-byte TMA boxes, SwiGLU tiles and
// expert slabs of whole 128-row scale tiles.
void check_mxfp8_dims(size_t hidden, size_t ffn, size_t alignment) {
  NVTE_CHECK(hidden % 512 == 0 && ffn % 256 == 0, "MXFP8 needs hidden_dim (", hidden,
             ") a multiple of 512 and ffn_dim (", ffn, ") a multiple of 256");
  NVTE_CHECK(alignment >= 128, "MXFP8 needs dispatch_output_per_expert_alignment 128 or ",
             kRowsPerTile);
}

// Columnwise data of an MXFP8 operand.
DLTensorWrapper wrap_columnwise(NVTETensor t, bool flatten_2d) {
  const NVTEBasicTensor data = nvte_get_tensor_param(t, kNVTEColumnwiseData);
  NVTE_CHECK(data.data_ptr != nullptr, "MXFP8 operand without columnwise data");
  return wrap_data(data, flatten_2d);
}

// E8M0 scales of a [rows, cols] MXFP8 operand, rowwise or columnwise.
DLTensorWrapper wrap_scales(NVTETensor t, bool columnwise, size_t rows, size_t cols) {
  const NVTEBasicTensor scales =
      nvte_get_tensor_param(t, columnwise ? kNVTEColumnwiseScaleInv : kNVTERowwiseScaleInv);
  size_t count = 1;
  for (size_t i = 0; i < scales.shape.ndim; ++i) {
    count *= scales.shape.data[i];
  }
  NVTE_CHECK(scales.data_ptr != nullptr && count == rows * cols / 32, "expected ", rows * cols / 32,
             columnwise ? " columnwise" : " rowwise", " MXFP8 scales, got ", count);
  return wrap_raw(scales.data_ptr, kNVTEFloat8E8M0, count);
}

// Value of a positive integer environment variable, 0 when unset.
size_t env_size(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return 0;
  }
  char* end = nullptr;
  const int64_t parsed = std::strtoll(value, &end, 10);
  NVTE_CHECK(*end == '\0' && parsed > 0, name, " must be a positive integer, got '", value, "'");
  return static_cast<size_t>(parsed);
}

// Columns per combine flag: the whole row or a multiple of 1024 dividing it, the chunks NCCL EP
// accepts for bfloat16 rows. NVTE_EP_OVERLAP_COMBINE_CHUNK_COLS unset picks 1024 when it divides n, else
// the whole row.
size_t combine_chunk(size_t n) {
  NVTE_CHECK(n % kTileN == 0, "output columns (", n, ") must be a multiple of ", kTileN);
  const size_t requested = env_size("NVTE_EP_OVERLAP_COMBINE_CHUNK_COLS");
  if (requested == 0) {
    return n % 1024 == 0 ? 1024 : n;
  }
  NVTE_CHECK(requested == n || (requested % 1024 == 0 && n % requested == 0),
             "NVTE_EP_OVERLAP_COMBINE_CHUNK_COLS (", requested, ") must be the hidden size (", n,
             ") or a multiple of 1024 dividing it");
  return requested;
}

// The combine (or dispatch backward) starts at the GEMM's entry and waits on its chunk flags.
constexpr bool kCombineAtGemmEntry = true;

// Group and layer state checked by every entry point.
struct EpGemmContext {
  ep::EPBackend& backend;
  const NVTEEpGroupConfig& group;
  NVTEEpLayerConfig layer_cfg;
  int num_clusters;
  int all_clusters;  // clusters of every SM
  // max_recv_tokens_per_rank == 0: the caller sized the recv buffers with nvte_ep_prepare.
  bool eager;

  size_t alignment() const { return layer_cfg.dispatch_output_per_expert_alignment; }
  // Rows per dispatch readiness flag: the alignment, at most one M tile.
  size_t rows_per_slot() const { return std::min(alignment(), kRowsPerTile); }
};

// comm_sms: the group's SM count of the NCCL EP call that the entry point overlaps; the GEMM runs
// on the other SMs.
EpGemmContext make_context(const NVTEEpLayerConfig* layer_cfg, int NVTEEpGroupConfig::*comm_sms) {
  const NVTEEpLayerConfig cfg = ep::normalize_layer_config(layer_cfg);
  const size_t alignment = cfg.dispatch_output_per_expert_alignment;
  NVTE_CHECK(alignment == 64 || alignment == 128 || alignment == kRowsPerTile,
             "dispatch_output_per_expert_alignment must be 64, 128 or ", kRowsPerTile);
  ep::EPBackend& backend = ep::EPBackend::get();
  const NVTEEpGroupConfig& group = backend.group_config();
  NVTE_CHECK(group.num_comm_sms > 0, "the overlapped ops need an explicit num_comm_sms");
  NVTE_CHECK(!group.zero_copy, "the overlapped ops do not support zero-copy");
  // The GEMMs take the SMs left beside the ep_bootstrap counts, which these variables would
  // change for the NCCL EP kernels only.
  for (const char* name : {"NCCL_EP_COMM_SMS", "NCCL_EP_DISPATCH_SMS", "NCCL_EP_COMBINE_SMS",
                           "NCCL_EP_DISPATCH_BWD_SMS", "NCCL_EP_COMBINE_BWD_SMS"}) {
    NVTE_CHECK(std::getenv(name) == nullptr,
               "the overlapped ops split the SMs by the ep_bootstrap "
               "counts; unset ",
               name);
  }
  const int sm_count = transformer_engine::cuda::sm_count();
  NVTE_CHECK(sm_count - group.*comm_sms >= static_cast<int>(kClusterSize), "fewer than ",
             kClusterSize, " SMs left beside the ", group.*comm_sms, " communication SMs");
  // The GEMM tails take the clusters of every SM.
  const int max_clusters =
      transformer_engine::cuda::max_active_clusters(static_cast<int>(kClusterSize));
  NVTE_CHECK(max_clusters == sm_count / static_cast<int>(kClusterSize), "the device runs ",
             max_clusters, " clusters of ", kClusterSize, " SMs at once, not ", sm_count, " / ",
             kClusterSize);
  EpGemmContext ctx{backend,
                    group,
                    cfg,
                    (sm_count - group.*comm_sms) / static_cast<int>(kClusterSize),
                    sm_count / static_cast<int>(kClusterSize),
                    group.max_recv_tokens_per_rank == 0};
  return ctx;
}

// A caller-owned [num_experts] int32 count tensor.
void check_counts(NVTETensor counts, size_t num_experts, const char* name) {
  NVTE_CHECK(counts != nullptr, name, " must not be null");
  NVTE_CHECK(nvte_tensor_shape(counts).ndim == 1 && nvte_tensor_numel(counts) == num_experts &&
                 nvte_tensor_type(counts) == kNVTEInt32,
             name, " must be an int32 tensor of ", num_experts, " elements");
}

// A caller-owned float32 [rows] tensor.
void check_row_floats(NVTETensor t, size_t rows, const char* name) {
  NVTE_CHECK(t != nullptr, name, " must not be null");
  NVTE_CHECK(nvte_tensor_shape(t).ndim == 1 && nvte_tensor_numel(t) == rows &&
                 nvte_tensor_type(t) == kNVTEFloat32,
             name, " must be a float32 tensor of recv_capacity (", rows, ") elements");
}

// The GEMM reads the per-expert row ends from expert_token_offsets[1:] and requires them 16-byte
// aligned; callers front-pad the [num_experts + 1] int32 layout by 3 elements to achieve that.
void check_expert_token_offsets(const NVTETensor expert_token_offsets, size_t num_experts) {
  NVTE_CHECK(nvte_tensor_numel(expert_token_offsets) == num_experts + 1 &&
                 nvte_tensor_type(expert_token_offsets) == kNVTEInt32,
             "expert_token_offsets must be an int32 tensor of num_experts + 1 elements");
  const int32_t* row_ends = static_cast<int32_t*>(nvte_tensor_data(expert_token_offsets)) + 1;
  NVTE_CHECK(reinterpret_cast<uintptr_t>(row_ends) % 16 == 0,
             "expert_token_offsets[1:] (the per-expert row ends) must start 16-byte aligned; "
             "front-pad the num_experts + 1 int32 layout by 3 elements");
}

// Rows: the group capacity, or any multiple of the flag rows in eager mode.
void check_rows(const EpGemmContext& ctx, size_t rows, const char* name) {
  NVTE_CHECK(rows % ctx.rows_per_slot() == 0, name, " rows (", rows, ") must be a multiple of ",
             ctx.rows_per_slot());
  NVTE_CHECK(ctx.eager || rows == static_cast<size_t>(ctx.group.max_recv_tokens_per_rank), name,
             " has ", rows, " rows, max_recv_tokens_per_rank is ",
             ctx.group.max_recv_tokens_per_rank);
}

// Whether the forward dispatch counts the routing itself and writes the caller's expert layout. As
// NCCL EP's dispatch_pull_count_capable for the overlapped group (pull-push flow in one LSA team):
// not in eager mode and at most 65536 / 32 local experts. Without drops only: with them the
// dispatch reports the pre-drop row total.
bool dispatch_writes_layout(const EpGemmContext& ctx, size_t num_local_experts) {
  return !ctx.eager && !ctx.group.drop_on_overflow && num_local_experts <= 65536 / 32;
}

// The forward combine_gemm GEMM of [rows, k] inputs and n output columns.
ep::EpCombineGemmConfig combine_fwd_config(const EpGemmContext& ctx, size_t rows, size_t k,
                                           size_t n, size_t num_experts, bool mxfp8) {
  return ep::EpCombineGemmConfig{rows,
                                 k,
                                 n,
                                 num_experts,
                                 ctx.num_clusters,
                                 combine_chunk(n),
                                 kCombineAtGemmEntry,
                                 false,
                                 mxfp8,
                                 ctx.alignment(),
                                 ctx.all_clusters};
}

// Scratch of one call: GEMM workspace, row ends (one reserved word before them, which unset_row_ends
// sets), unit alphas, optional unit probs, optional weight-gradient workspace, and the words the
// call's launch resets.
struct GemmScratch {
  uint8_t* base;
  uint8_t* wgrad = nullptr;
  int32_t* row_ends = nullptr;  // [num_experts]
  float* alpha = nullptr;       // [num_experts]
  float* ones = nullptr;        // [rows]
  int32_t* signal = nullptr;    // [signal_capacity] readiness flags
  int signal_capacity = 0;
  int num_experts = 0;
  ep::WorkspaceReset reset;
};

GemmScratch claim_scratch(size_t gemm_bytes, size_t num_experts, size_t rows, bool unit_probs,
                          size_t wgrad_bytes) {
  const size_t row_ends = align_up(gemm_bytes + sizeof(int32_t));
  const size_t alpha = row_ends + align_up(sizeof(int32_t) * num_experts);
  const size_t ones = alpha + align_up(sizeof(float) * num_experts);
  const size_t wgrad = ones + (unit_probs ? align_up(sizeof(float) * rows) : 0);
  const size_t end = wgrad + align_up(wgrad_bytes);
  GemmScratch scratch{get_scratch(end)};
  if (wgrad_bytes > 0) {
    scratch.wgrad = scratch.base + wgrad;
  }
  scratch.row_ends = reinterpret_cast<int32_t*>(scratch.base + row_ends);
  scratch.alpha = reinterpret_cast<float*>(scratch.base + alpha);
  scratch.num_experts = static_cast<int>(num_experts);
  scratch.reset.fill(scratch.alpha, scratch.num_experts, ep::kFillOne);
  if (unit_probs) {
    scratch.ones = reinterpret_cast<float*>(scratch.base + ones);
    scratch.reset.fill(scratch.ones, static_cast<int>(rows), ep::kFillOne);
  }
  return scratch;
}

// Words of a GEMM that waits on the dispatch tile flags: its tile counter, flags and scheduler
// state.
void set_signal_words(GemmScratch& scratch, const ep::WorkspaceLayout& layout, size_t slots) {
  scratch.signal = reinterpret_cast<int32_t*>(scratch.base + layout.signal);
  scratch.signal_capacity = static_cast<int>(slots);
  scratch.reset.fill(scratch.base + layout.counter, 1, ep::kFillZero);
  scratch.reset.fill(scratch.signal, scratch.signal_capacity, ep::kFillMinusOne);
  scratch.reset.fill(scratch.base + layout.state, (layout.total_bytes - layout.state) / 4,
                     ep::kFillZero);
}

// Words of a GEMM that publishes column chunks: its first zeroed_bytes (scheduler words, zero at
// entry) and chunk flags.
void set_chunk_words(GemmScratch& scratch, const ep::WorkspaceLayout& layout, size_t chunks,
                     size_t zeroed_bytes) {
  scratch.signal = reinterpret_cast<int32_t*>(scratch.base + layout.signal);
  scratch.signal_capacity = static_cast<int>(chunks);
  scratch.reset.fill(scratch.base, zeroed_bytes / 4, ep::kFillZero);
  scratch.reset.fill(scratch.signal, scratch.signal_capacity, ep::kFillMinusOne);
}

// Row ends the dispatch writes: unset until it does, together with the reserved word before them.
void unset_row_ends(GemmScratch& scratch) {
  scratch.reset.fill(scratch.row_ends - 1, scratch.num_experts + 1, ep::kFillMinusOne);
}

// Row ends computed by the reset from the padded int32 counts per local expert; write_layout_base
// also writes the reserved word before them, expert 0's start, as 0.
void set_row_ends(GemmScratch& scratch, NVTETensor counts, bool write_layout_base = false) {
  scratch.reset.counts = static_cast<const int32_t*>(nvte_tensor_data(counts));
  scratch.reset.row_ends = scratch.row_ends;
  scratch.reset.num_experts = scratch.num_experts;
  if (write_layout_base) {
    scratch.reset.fill(scratch.row_ends - 1, 1, ep::kFillZero);
  }
}

// Compiled GEMM and workspace layout of a configuration.
template <typename Config>
tvm::ffi::Function get_gemm(const Config& config, ep::WorkspaceLayout* layout) {
  auto kernel = config.get_kernel();
  NVTE_CHECK(kernel.has_value(),
             "the EP grouped GEMM is not available: needs NVTE_ENABLE_CUTEDSL_BACKEND=1, "
             "NVTE_MOE_OVERLAP and SM 10.0, 10.3 or 10.7");
  const auto offsets = config.workspace_layout();
  NVTE_CHECK(offsets.has_value(), "cannot query the GEMM workspace layout");
  *layout = *offsets;
  return *kernel;
}

// The dispatch GEMM on `clusters` more clusters than the first launch's, which share its workspace.
tvm::ffi::Function get_tail_gemm(ep::EpDispatchGemmConfig config, int clusters,
                                 const ep::WorkspaceLayout& layout, const char* name) {
  config.num_clusters = clusters;
  ep::WorkspaceLayout tail_layout;
  const tvm::ffi::Function tail = get_gemm(config, &tail_layout);
  NVTE_CHECK(tail_layout.total_bytes == layout.total_bytes &&
                 tail_layout.counter == layout.counter && tail_layout.signal == layout.signal &&
                 tail_layout.state == layout.state,
             name, " has another workspace layout");
  return tail;
}

// A weight gradient chained behind a backward GEMM: weight_grad[e] = grad^T x over the rows of
// expert e, written or added to.
struct Wgrad {
  NVTETensor grad;
  NVTETensor x;
  NVTETensor weight_grad;
  ep::EpWgradConfig config;  // each launch sets its clusters and wait options
  ep::WorkspaceLayout layout;
};

// Checks the weight-gradient operands of a backward entry; nullopt without weight_grad. MXFP8
// reads both operands columnwise.
std::optional<Wgrad> make_wgrad(const EpGemmContext& ctx, NVTETensor grad, NVTETensor x,
                                NVTETensor weight_grad, bool accumulate, size_t rows,
                                size_t num_experts, bool mxfp8) {
  if (weight_grad == nullptr) {
    NVTE_CHECK(x == nullptr, "wgrad_input needs weight_grad");
    return std::nullopt;
  }
  NVTE_CHECK(x != nullptr, "weight_grad needs wgrad_input");
  NVTE_CHECK(rows > 0, "the chained weight gradient needs received rows");
  NVTE_CHECK(nvte_tensor_shape(grad).ndim == 2 && nvte_tensor_shape(x).ndim == 2,
             "wgrad_input and the gradient must be 2D");
  const NVTEShape shape = nvte_tensor_shape(weight_grad);
  NVTE_CHECK(shape.ndim == 3 && shape.data[0] == num_experts,
             "weight_grad must be [num_local_experts, m, n]");
  NVTE_CHECK(reinterpret_cast<uintptr_t>(nvte_tensor_data(weight_grad)) % 16 == 0,
             "weight_grad must be 16-byte aligned");
  const size_t m = shape.data[1];
  const size_t n = shape.data[2];
  NVTE_CHECK(nvte_tensor_shape(grad).data[1] == m && nvte_tensor_shape(x).data[0] == rows &&
                 nvte_tensor_shape(x).data[1] == n,
             "weight_grad must be [num_local_experts, ", nvte_tensor_shape(grad).data[1],
             ", n] with wgrad_input [recv_capacity, n]");
  if (mxfp8) {
    check_mxfp8(x, true, "wgrad_input");
    // Both operands are read columnwise: [rows, cols] E4M3 data and rows * cols / 32 scales.
    const auto check_columnwise = [rows](NVTETensor t, size_t cols, const char* name) {
      const NVTEBasicTensor data = nvte_get_tensor_param(t, kNVTEColumnwiseData);
      const NVTEBasicTensor scales = nvte_get_tensor_param(t, kNVTEColumnwiseScaleInv);
      size_t scale_count = 1;
      for (size_t i = 0; i < scales.shape.ndim; ++i) {
        scale_count *= scales.shape.data[i];
      }
      NVTE_CHECK(data.data_ptr != nullptr && scales.data_ptr != nullptr,
                 "the MXFP8 weight gradient needs columnwise ", name, " data and scales");
      NVTE_CHECK(data.shape.ndim == 2 && data.shape.data[0] == rows && data.shape.data[1] == cols,
                 "the columnwise MXFP8 ", name, " must be [", rows, ", ", cols, "]");
      NVTE_CHECK(scale_count == rows * cols / 32, "the columnwise MXFP8 ", name, " needs ",
                 rows * cols / 32, " scales, got ", scale_count);
    };
    check_columnwise(grad, m, "gradient");
    check_columnwise(x, n, "wgrad_input");
  } else {
    NVTE_CHECK(nvte_tensor_type(x) == kNVTEBFloat16, "wgrad_input must be bfloat16");
  }
  const NVTEDType out_type = nvte_tensor_type(weight_grad);
  NVTE_CHECK(out_type == kNVTEBFloat16 || out_type == kNVTEFloat32,
             "weight_grad must be bfloat16 or float32");
  const size_t routed_rows =
      static_cast<size_t>(ctx.group.max_tokens_per_rank) * ctx.layer_cfg.top_k;
  Wgrad wgrad{grad,
              x,
              weight_grad,
              {rows, m, n, num_experts, 1, out_type == kNVTEFloat32, accumulate, false, 0, mxfp8,
               false, ctx.alignment(), routed_rows},
              {}};
  const auto layout = wgrad.config.workspace_layout();
  NVTE_CHECK(layout.has_value(), "cannot query the weight-gradient workspace layout");
  wgrad.layout = *layout;
  return wgrad;
}

// Whether the MXFP8 weight gradient reads per-expert descriptors that a descriptors_only launch
// writes: an added or FP32 one, and on SM 10.7 a written BF16 one with long experts. The others
// read global descriptors.
bool needs_descriptors(const Wgrad& wgrad) {
  return wgrad.config.mxfp8 && wgrad.config.reads_descriptors();
}

// One launch of a weight gradient on num_clusters clusters (none: no launch), a programmatic
// dependent of the kernel before it. Launches sharing the call's workspace split the tiles.
// wait_primary_at_start: no load before that kernel completes. ready_slot_rows: no load before
// every dispatch flag of the expert rows is set. descriptors_only (MXFP8): a plain launch that
// writes the per-expert descriptors the other launches read; it follows the scratch reset.
void launch_wgrad(const Wgrad& wgrad, const GemmScratch& scratch, int num_clusters,
                  bool wait_primary_at_start, size_t ready_slot_rows, cudaStream_t stream,
                  bool descriptors_only) {
  if (num_clusters <= 0) {
    return;
  }
  ep::EpWgradConfig config = wgrad.config;
  config.num_clusters = num_clusters;
  config.wait_primary_at_start = wait_primary_at_start;
  config.ready_slot_rows = ready_slot_rows;
  config.descriptors_only = descriptors_only;
  ep::WorkspaceLayout layout;
  const tvm::ffi::Function gemm = get_gemm(config, &layout);
  // The call sized the scratch from its base configuration.
  NVTE_CHECK(
      layout.total_bytes <= wgrad.layout.total_bytes && layout.counter == wgrad.layout.counter,
      "weight-gradient workspace layout differs between launches");
  auto out = wrap_rowwise(wgrad.weight_grad, false);
  auto row_ends = wrap_raw(scratch.row_ends, kNVTEInt32, config.num_experts);
  auto ws = wrap_raw(scratch.wgrad, kNVTEByte, layout.total_bytes);
  if (config.mxfp8) {
    auto grad = wrap_columnwise(wgrad.grad, true);
    auto x = wrap_columnwise(wgrad.x, true);
    auto grad_scales = wrap_scales(wgrad.grad, true, config.rows, config.m);
    auto x_scales = wrap_scales(wgrad.x, true, config.rows, config.n);
    gemm(&grad, &x, &grad_scales, &x_scales, &out, &row_ends, &ws, static_cast<void*>(stream));
    return;
  }
  auto grad = wrap_rowwise(wgrad.grad, true);
  auto x = wrap_rowwise(wgrad.x, true);
  if (ready_slot_rows > 0) {
    auto signal = wrap_raw(scratch.signal, kNVTEInt32, config.rows / ready_slot_rows);
    gemm(&grad, &x, &out, &row_ends, &ws, &signal, static_cast<void*>(stream));
    return;
  }
  gemm(&grad, &x, &out, &row_ends, &ws, static_cast<void*>(stream));
}

// The columnwise MXFP8 copy of t's received rows, requantized from its rowwise data and compact
// scales; columnwise scales GEMM-swizzled per expert slab. A programmatic dependent of the kernel
// before it. ready_slot_rows > 0: a tile starts once the dispatch has set the flags of its rows.
void launch_requant(NVTETensor t, const GemmScratch& scratch, size_t rows, size_t cols,
                    size_t num_experts, size_t ready_slot_rows, cudaStream_t stream) {
  const ep::EpRequantConfig config{rows, cols, num_experts, ready_slot_rows};
  const auto kernel = config.get_kernel();
  NVTE_CHECK(kernel.has_value(),
             "the EP MXFP8 requant is not available: needs NVTE_ENABLE_CUTEDSL_BACKEND=1, "
             "NVTE_MOE_OVERLAP and SM 10.0, 10.3 or 10.7");
  auto codes = wrap_rowwise(t, true);
  auto row_scales = wrap_scales(t, false, rows, cols);
  auto col_codes = wrap_columnwise(t, true);
  auto col_scales = wrap_scales(t, true, rows, cols);
  auto row_ends = wrap_raw(scratch.row_ends, kNVTEInt32, num_experts);
  if (ready_slot_rows > 0) {
    auto signal = wrap_raw(scratch.signal, kNVTEInt32, rows / ready_slot_rows);
    (*kernel)(&codes, &row_scales, &col_codes, &col_scales, &row_ends, &signal,
              static_cast<void*>(stream));
    return;
  }
  (*kernel)(&codes, &row_scales, &col_codes, &col_scales, &row_ends, static_cast<void*>(stream));
}

}  // namespace

namespace transformer_engine {
namespace ep {

void release_gemm_scratch() {
  std::lock_guard<std::mutex> lock(scratch_mutex);
  scratch_blocks()[cuda::current_device()].release();
}

}  // namespace ep
}  // namespace transformer_engine

extern "C" {

void nvte_ep_dispatch_gemm_fwd(NVTETensor handle_mem, const NVTETensor topk_idx,
                               const NVTETensor tokens, NVTECommWindow tokens_win,
                               const NVTETensor topk_weights, const NVTETensor weight,
                               NVTETensor recv_tokens, NVTETensor recv_topk_weights,
                               NVTETensor gemm_out, NVTETensor act_out,
                               NVTETensor recv_tokens_per_expert, NVTETensor expert_token_offsets,
                               NVTETensor total_recv_tokens, const NVTEEpLayerConfig* layer_cfg,
                               cudaStream_t stream) {
  using namespace transformer_engine;
  const EpGemmContext ctx = make_context(layer_cfg, &NVTEEpGroupConfig::dispatch_fwd_num_sms);
  checkCuDriverContext(stream);

  const size_t rows = nvte_tensor_shape(recv_tokens).data[0];
  const size_t hidden = nvte_tensor_shape(tokens).data[1];
  const size_t num_experts = nvte_tensor_shape(weight).data[0];
  const size_t n = nvte_tensor_shape(weight).data[1];
  check_rows(ctx, rows, "recv_tokens");
  NVTE_CHECK(num_experts * ctx.group.ep_size == static_cast<size_t>(ctx.group.num_experts),
             "weight has ", num_experts, " experts, expected num_experts / ep_size = ",
             ctx.group.num_experts / ctx.group.ep_size);
  NVTE_CHECK(nvte_tensor_shape(act_out).data[1] * 2 == n, "weight rows (", n,
             ") must be 2 * the act_out columns: SwiGLU");
  check_counts(recv_tokens_per_expert, num_experts, "recv_tokens_per_expert");
  check_row_floats(recv_topk_weights, rows, "recv_topk_weights");
  NVTE_CHECK(total_recv_tokens == nullptr || (nvte_tensor_numel(total_recv_tokens) == 1 &&
                                              nvte_tensor_type(total_recv_tokens) == kNVTEInt32),
             "total_recv_tokens must be an int32 tensor of 1 element");
  const bool mxfp8 = is_mxfp8(tokens);
  if (mxfp8) {
    check_mxfp8(tokens, false, "tokens");
    check_mxfp8(recv_tokens, false, "recv_tokens");
    check_mxfp8(weight, true, "weight");
    check_mxfp8(act_out, true, "act_out");
    NVTE_CHECK(nvte_tensor_type(gemm_out) == kNVTEBFloat16, "gemm_out must be bfloat16");
    check_mxfp8_dims(hidden, n / 2, ctx.alignment());
  } else {
    NVTE_CHECK(
        nvte_tensor_type(tokens) == kNVTEBFloat16 && nvte_tensor_type(weight) == kNVTEBFloat16,
        "tokens and weight must be bfloat16 or MXFP8");
  }

  // No count exchange before the dispatch when the dispatch writes the GEMM's expert layout.
  const bool device_layout = dispatch_writes_layout(ctx, num_experts);

  // The GEMM runs beside the dispatch, and its tail on the SMs the dispatch frees. Both launches
  // claim tiles from one workspace.
  ep::WorkspaceLayout layout;
  const ep::EpDispatchGemmConfig config{
      rows, hidden, n, num_experts, ctx.num_clusters, false, mxfp8, ctx.alignment(), device_layout};
  std::vector<tvm::ffi::Function> gemms{get_gemm(config, &layout)};
  const int tail_clusters = ctx.all_clusters - ctx.num_clusters;
  if (tail_clusters > 0) {
    gemms.push_back(get_tail_gemm(config, tail_clusters, layout, "the GEMM tail"));
  }
  GemmScratch scratch = claim_scratch(layout.total_bytes, num_experts, rows, false, 0);
  set_signal_words(scratch, layout, rows / ctx.rows_per_slot());
  // The GEMM reads the expert row ends from the caller's [num_experts + 1] int32
  // expert_token_offsets (expert 0's start, then the per-expert ends); the dispatch writes them in
  // capacity mode.
  check_expert_token_offsets(expert_token_offsets, num_experts);
  scratch.row_ends = static_cast<int32_t*>(nvte_tensor_data(expert_token_offsets)) + 1;

  // The dispatch publishes one flag per slot of rows; the GEMM behind it consumes ready tiles.
  TensorWrapper signal(scratch.signal,
                       std::vector<size_t>{static_cast<size_t>(scratch.signal_capacity)},
                       DType::kInt32);
  const ep::EpSignal dispatch_signal{signal.data(), ctx.rows_per_slot()};
  NVTECommWindow no_window{};
  // Expert offsets, then the row total: the expert row ends sit one word in.
  int32_t* const expert_layout = scratch.row_ends - 1;
  if (device_layout) {
    // Flags, scheduler state and unset layout words, before the dispatch that sets them. The fused
    // dispatch writes the GEMM's expert layout and the padded counts in one scan.
    unset_row_ends(scratch);
    ep::reset_workspace(scratch.reset, stream);
    TensorWrapper offsets(expert_layout, std::vector<size_t>{num_experts}, DType::kInt32);
    TensorWrapper total(expert_layout + num_experts, std::vector<size_t>{1}, DType::kInt32);
    ctx.backend.prepare_and_dispatch(handle_mem, topk_idx, tokens, tokens_win, topk_weights,
                                     no_window, recv_tokens, no_window, recv_topk_weights,
                                     no_window, recv_tokens_per_expert, total.data(), ctx.layer_cfg,
                                     stream, offsets.data(), &dispatch_signal);
  } else {
    // Capacity mode runs the routing prepare; eager mode ran nvte_ep_prepare already.
    if (!ctx.eager) {
      ctx.backend.prepare(handle_mem, topk_idx, recv_tokens_per_expert, total_recv_tokens,
                          ctx.layer_cfg, stream);
    }
    // Flags, scheduler state and expert row ends, before the dispatch that sets the flags. The
    // reset also writes the caller offsets' leading 0 (expert 0's start), which the dispatch
    // layout scan does not touch here.
    set_row_ends(scratch, recv_tokens_per_expert, /*write_layout_base=*/true);
    ep::reset_workspace(scratch.reset, stream);
    ctx.backend.dispatch(handle_mem, topk_idx, tokens, tokens_win, topk_weights, no_window,
                         recv_tokens, no_window, recv_topk_weights, no_window, &ctx.layer_cfg,
                         stream, &dispatch_signal);
  }

  // The GEMM is a programmatic dependent of the dispatch: tiles start as their flags are set. The
  // tail is the GEMM's programmatic dependent.
  auto a = wrap_rowwise(recv_tokens, true);
  auto b = wrap_rowwise(weight, false);
  auto c = wrap_rowwise(gemm_out, true);
  auto d = wrap_rowwise(act_out, true);
  auto row_ends = wrap_raw(scratch.row_ends, kNVTEInt32, num_experts);
  auto alpha = wrap_raw(scratch.alpha, kNVTEFloat32, num_experts);
  auto ws = wrap_raw(scratch.base, kNVTEByte, layout.total_bytes);
  if (mxfp8) {
    auto sfb = wrap_scales(weight, false, num_experts * n, hidden);
    auto sfa = wrap_scales(recv_tokens, false, rows, hidden);
    auto sfd = wrap_scales(act_out, false, rows, n / 2);
    auto norm_const = wrap_raw(scratch.alpha, kNVTEFloat32, 1);
    for (const tvm::ffi::Function& gemm : gemms) {
      gemm(&a, &b, &sfb, &sfa, &c, &d, &sfd, &row_ends, &alpha, &norm_const, &ws,
           static_cast<void*>(stream));
    }
  } else {
    auto prob = wrap_rowwise(recv_topk_weights, false);
    for (const tvm::ffi::Function& gemm : gemms) {
      gemm(&a, &b, &c, &d, &row_ends, &alpha, &prob, &ws, static_cast<void*>(stream));
    }
  }
}

void nvte_ep_combine_gemm_fwd(NVTETensor handle_mem, const NVTETensor gemm_in,
                              const NVTETensor recv_topk_weights,
                              const NVTETensor expert_token_offsets, const NVTETensor weight,
                              NVTETensor expert_out, NVTETensor result,
                              const NVTEEpLayerConfig* layer_cfg, cudaStream_t stream) {
  using namespace transformer_engine;
  const EpGemmContext ctx = make_context(layer_cfg, &NVTEEpGroupConfig::combine_fwd_num_sms);
  checkCuDriverContext(stream);

  const size_t rows = nvte_tensor_shape(gemm_in).data[0];
  const size_t k = nvte_tensor_shape(gemm_in).data[1];
  const size_t num_experts = nvte_tensor_shape(weight).data[0];
  const size_t n = nvte_tensor_shape(weight).data[1];
  check_rows(ctx, rows, "gemm_in");
  NVTE_CHECK(nvte_tensor_shape(weight).data[2] == k, "weight columns must equal gemm_in columns");
  const bool mxfp8 = is_mxfp8(gemm_in);
  if (mxfp8) {
    check_mxfp8(gemm_in, true, "gemm_in");
    check_mxfp8(weight, true, "weight");
    check_mxfp8_dims(n, k, ctx.alignment());
  } else {
    NVTE_CHECK(
        nvte_tensor_type(gemm_in) == kNVTEBFloat16 && nvte_tensor_type(weight) == kNVTEBFloat16,
        "gemm_in and weight must be bfloat16 or MXFP8");
  }
  check_expert_token_offsets(expert_token_offsets, num_experts);
  const bool weighted = recv_topk_weights != nullptr;
  NVTE_CHECK(!weighted || (nvte_tensor_type(recv_topk_weights) == kNVTEFloat32 &&
                           nvte_tensor_numel(recv_topk_weights) == rows),
             "recv_topk_weights must be float32 [recv_capacity]");

  // The GEMM launches on every SM; the clusters past those beside the combine stop after the
  // first chunk and leave their SMs to the combine.
  ep::WorkspaceLayout layout;
  const ep::EpCombineGemmConfig config = combine_fwd_config(ctx, rows, k, n, num_experts, mxfp8);
  const size_t n_chunk = config.n_chunk;
  const tvm::ffi::Function gemm = get_gemm(config, &layout);
  // Own workspace for the signal, scheduler and alpha words; the expert layout is the caller's
  // expert_token_offsets the dispatch wrote.
  GemmScratch scratch = claim_scratch(layout.total_bytes, num_experts, rows, !weighted, 0);
  set_chunk_words(scratch, layout, n / n_chunk, layout.signal);
  ep::reset_workspace(scratch.reset, stream);

  // The GEMM fills the flags; the combine is its programmatic dependent.
  auto a = wrap_rowwise(gemm_in, true);
  auto b = wrap_rowwise(weight, false);
  auto d = wrap_rowwise(expert_out, true);
  auto row_ends = wrap_raw(static_cast<int32_t*>(nvte_tensor_data(expert_token_offsets)) + 1,
                           kNVTEInt32, num_experts);
  auto alpha = wrap_raw(scratch.alpha, kNVTEFloat32, num_experts);
  auto prob = weighted ? wrap_rowwise(recv_topk_weights, false)
                       : wrap_raw(scratch.ones, kNVTEFloat32, rows);
  auto ws = wrap_raw(scratch.base, kNVTEByte, layout.total_bytes);
  if (mxfp8) {
    auto sfb = wrap_scales(weight, false, num_experts * n, k);
    auto sfa = wrap_scales(gemm_in, false, rows, k);
    gemm(&a, &b, &sfb, &sfa, &d, &row_ends, &alpha, &prob, &ws, static_cast<void*>(stream));
  } else {
    gemm(&a, &b, &d, &row_ends, &alpha, &prob, &ws, static_cast<void*>(stream));
  }

  NVTECommWindow no_window{};
  TensorWrapper signal(scratch.base + layout.signal, std::vector<size_t>{n / n_chunk},
                       DType::kInt32);
  const ep::EpSignal combine_signal{signal.data(), n_chunk};
  ctx.backend.combine(handle_mem, expert_out, no_window, result, &ctx.layer_cfg, stream,
                      &combine_signal);
}

void nvte_ep_combine_gemm_bwd(NVTETensor handle_mem, const NVTETensor grad_result,
                              NVTECommWindow grad_result_win, const NVTETensor gemm_out,
                              const NVTETensor recv_topk_weights,
                              const NVTETensor recv_tokens_per_expert, const NVTETensor weight,
                              const NVTETensor wgrad_input, NVTETensor grad_expert_out,
                              NVTETensor grad_gemm_out, NVTETensor grad_recv_topk_weights,
                              NVTETensor act_out, NVTETensor weight_grad,
                              int accumulate_weight_grad, const NVTEEpLayerConfig* layer_cfg,
                              cudaStream_t stream) {
  using namespace transformer_engine;
  const EpGemmContext ctx = make_context(layer_cfg, &NVTEEpGroupConfig::dispatch_bwd_num_sms);
  checkCuDriverContext(stream);

  const size_t rows = nvte_tensor_shape(grad_expert_out).data[0];
  const size_t hidden = nvte_tensor_shape(grad_expert_out).data[1];
  const size_t num_experts = nvte_tensor_shape(weight).data[0];
  const size_t ffn = nvte_tensor_shape(weight).data[2];
  check_rows(ctx, rows, "grad_expert_out");
  NVTE_CHECK(nvte_tensor_shape(weight).data[1] == hidden &&
                 nvte_tensor_shape(gemm_out).data[1] == 2 * ffn &&
                 nvte_tensor_shape(grad_gemm_out).data[1] == 2 * ffn,
             "weight must be [experts, hidden, ffn] with 2 * ffn the gemm_out columns: SwiGLU");
  NVTE_CHECK(nvte_tensor_type(gemm_out) == kNVTEBFloat16, "gemm_out must be bfloat16");
  const bool mxfp8 = is_mxfp8(grad_result);
  if (mxfp8) {
    check_mxfp8(grad_result, false, "grad_result");
    check_mxfp8(grad_expert_out, false, "grad_expert_out");
    check_mxfp8(weight, true, "weight");
    check_mxfp8(grad_gemm_out, true, "grad_gemm_out");
    check_mxfp8(act_out, true, "act_out");
    NVTE_CHECK(nvte_tensor_shape(act_out).data[1] == ffn, "act_out must have ffn columns");
    check_mxfp8_dims(hidden, ffn, ctx.alignment());
  } else {
    NVTE_CHECK(
        nvte_tensor_type(grad_result) == kNVTEBFloat16 && nvte_tensor_type(weight) == kNVTEBFloat16,
        "grad_result and weight must be bfloat16 or MXFP8");
    NVTE_CHECK(act_out == nullptr, "act_out is written for MXFP8 only");
  }
  check_counts(recv_tokens_per_expert, num_experts, "recv_tokens_per_expert");
  check_row_floats(recv_topk_weights, rows, "recv_topk_weights");
  check_row_floats(grad_recv_topk_weights, rows, "grad_recv_topk_weights");

  // The MXFP8 weight gradient reads the columnwise grad_expert_out that this call writes.
  const bool requant =
      mxfp8 && nvte_get_tensor_param(grad_expert_out, kNVTEColumnwiseData).data_ptr != nullptr;
  NVTE_CHECK(!mxfp8 || weight_grad == nullptr || requant,
             "the MXFP8 weight gradient needs the columnwise grad_expert_out");
  const std::optional<Wgrad> wgrad = make_wgrad(ctx, grad_expert_out, wgrad_input, weight_grad,
                                                accumulate_weight_grad, rows, num_experts, mxfp8);

  ep::WorkspaceLayout layout;
  const ep::EpDispatchGemmConfig config{
      rows, hidden, ffn, num_experts, ctx.num_clusters, true, mxfp8, ctx.alignment()};
  const tvm::ffi::Function gemm = get_gemm(config, &layout);
  GemmScratch scratch = claim_scratch(layout.total_bytes, num_experts, rows, false,
                                      wgrad ? wgrad->layout.total_bytes : 0);
  set_signal_words(scratch, layout, rows / ctx.rows_per_slot());
  if (wgrad) {
    scratch.reset.fill(scratch.wgrad + wgrad->layout.counter, 1, ep::kFillZero);
  }
  // The GEMM adds each row's routing-weight gradient into grad_recv_topk_weights.
  scratch.reset.fill(nvte_tensor_data(grad_recv_topk_weights), static_cast<int>(rows),
                     ep::kFillZero);
  set_row_ends(scratch, recv_tokens_per_expert);
  ep::reset_workspace(scratch.reset, stream);
  if (wgrad && needs_descriptors(*wgrad)) {
    launch_wgrad(*wgrad, scratch, 1, false, 0, stream, true);
  }

  // The reverse dispatch publishes one flag per slot; the GEMM consumes ready tiles.
  TensorWrapper signal(scratch.signal,
                       std::vector<size_t>{static_cast<size_t>(scratch.signal_capacity)},
                       DType::kInt32);
  const ep::EpSignal dispatch_signal{signal.data(), ctx.rows_per_slot()};
  NVTECommWindow no_window{};
  ctx.backend.combine_bwd(handle_mem, grad_result, grad_result_win, grad_expert_out, no_window,
                          &ctx.layer_cfg, stream, &dispatch_signal);

  // FC2 data gradient with the SwiGLU backward in its epilogue.
  auto a = wrap_rowwise(grad_expert_out, true);
  auto c = wrap_rowwise(gemm_out, true);
  auto d = wrap_rowwise(grad_gemm_out, true);
  auto dprob = wrap_rowwise(grad_recv_topk_weights, false);
  auto row_ends = wrap_raw(scratch.row_ends, kNVTEInt32, num_experts);
  auto alpha = wrap_raw(scratch.alpha, kNVTEFloat32, num_experts);
  auto prob = wrap_rowwise(recv_topk_weights, false);
  auto ws = wrap_raw(scratch.base, kNVTEByte, layout.total_bytes);
  if (mxfp8) {
    // The columnwise forward weight; the epilogue also recomputes act_out columnwise.
    auto b = wrap_columnwise(weight, false);
    auto sfb = wrap_scales(weight, true, num_experts * hidden, ffn);
    auto sfa = wrap_scales(grad_expert_out, false, rows, hidden);
    auto d_col = wrap_columnwise(grad_gemm_out, true);
    auto sfd_row = wrap_scales(grad_gemm_out, false, rows, 2 * ffn);
    auto sfd_col = wrap_scales(grad_gemm_out, true, rows, 2 * ffn);
    auto h = wrap_columnwise(act_out, true);
    auto sfh_col = wrap_scales(act_out, true, rows, ffn);
    auto norm_const = wrap_raw(scratch.alpha, kNVTEFloat32, 1);
    gemm(&a, &b, &sfb, &sfa, &c, &d, &d_col, &sfd_row, &sfd_col, &h, &sfh_col, &dprob, &row_ends,
         &alpha, &prob, &norm_const, &ws, static_cast<void*>(stream));
    // The columnwise grad_expert_out, a dependent of the GEMM, which starts it at entry: each
    // tile waits for the reverse dispatch's flags of its rows.
    if (requant) {
      launch_requant(grad_expert_out, scratch, rows, hidden, num_experts, ctx.rows_per_slot(),
                     stream);
    }
    // Its tail on the SMs the reverse dispatch frees takes the tiles the GEMM has not claimed.
    const int tail_clusters = ctx.all_clusters - ctx.num_clusters;
    if (tail_clusters > 0) {
      const tvm::ffi::Function tail = get_tail_gemm(config, tail_clusters, layout, "the dGLU tail");
      tail(&a, &b, &sfb, &sfa, &c, &d, &d_col, &sfd_row, &sfd_col, &h, &sfh_col, &dprob, &row_ends,
           &alpha, &prob, &norm_const, &ws, static_cast<void*>(stream));
    }
    // The weight gradient once the GEMM and the requant complete: its base, which then lets its
    // tail in.
    if (wgrad) {
      launch_wgrad(*wgrad, scratch, ctx.num_clusters, true, 0, stream, false);
      launch_wgrad(*wgrad, scratch, ctx.all_clusters - ctx.num_clusters, false, 0, stream, false);
    }
    return;
  }
  auto b = wrap_rowwise(weight, false);
  gemm(&a, &b, &c, &d, &dprob, &row_ends, &alpha, &prob, &ws, static_cast<void*>(stream));
  // The weight gradient once every row is received: its tail on the SMs the reverse dispatch
  // frees, its base on those the GEMM frees.
  if (wgrad) {
    launch_wgrad(*wgrad, scratch, ctx.all_clusters - ctx.num_clusters, false, ctx.rows_per_slot(),
                 stream, false);
    launch_wgrad(*wgrad, scratch, ctx.num_clusters, false, ctx.rows_per_slot(), stream, false);
  }
}

void nvte_ep_dispatch_gemm_bwd(NVTETensor handle_mem, const NVTETensor grad_gemm_out,
                               const NVTETensor grad_recv_topk_weights,
                               const NVTETensor recv_tokens_per_expert, const NVTETensor weight,
                               const NVTETensor wgrad_input, NVTETensor grad_recv_tokens,
                               NVTETensor grad_tokens, NVTETensor grad_topk_weights,
                               NVTETensor weight_grad, int accumulate_weight_grad,
                               const NVTEEpLayerConfig* layer_cfg, cudaStream_t stream) {
  using namespace transformer_engine;
  const EpGemmContext ctx = make_context(layer_cfg, &NVTEEpGroupConfig::combine_bwd_num_sms);
  checkCuDriverContext(stream);

  const size_t rows = nvte_tensor_shape(grad_gemm_out).data[0];
  const size_t k = nvte_tensor_shape(grad_gemm_out).data[1];
  const size_t hidden = nvte_tensor_shape(grad_recv_tokens).data[1];
  const size_t num_experts = nvte_tensor_shape(weight).data[0];
  check_rows(ctx, rows, "grad_gemm_out");
  NVTE_CHECK(nvte_tensor_shape(weight).data[1] == k && nvte_tensor_shape(weight).data[2] == hidden,
             "weight must be [experts, 2 * ffn, hidden]");
  check_counts(recv_tokens_per_expert, num_experts, "recv_tokens_per_expert");
  check_row_floats(grad_recv_topk_weights, rows, "grad_recv_topk_weights");
  const bool mxfp8 = is_mxfp8(grad_gemm_out);
  if (mxfp8) {
    check_mxfp8(grad_gemm_out, true, "grad_gemm_out");
    check_mxfp8(weight, true, "weight");
    NVTE_CHECK(nvte_tensor_type(grad_recv_tokens) == kNVTEBFloat16,
               "grad_recv_tokens must be bfloat16");
    check_mxfp8_dims(hidden, k / 2, ctx.alignment());
  } else {
    NVTE_CHECK(nvte_tensor_type(grad_gemm_out) == kNVTEBFloat16 &&
                   nvte_tensor_type(weight) == kNVTEBFloat16,
               "grad_gemm_out and weight must be bfloat16 or MXFP8");
  }
  const std::optional<Wgrad> wgrad = make_wgrad(ctx, grad_gemm_out, wgrad_input, weight_grad,
                                                accumulate_weight_grad, rows, num_experts, mxfp8);
  // MXFP8: the call writes the columnwise wgrad_input from its rowwise data when that is set.
  const bool requant =
      wgrad && mxfp8 && nvte_get_tensor_param(wgrad_input, kNVTERowwiseData).data_ptr != nullptr;
  // With the weight gradient the dispatch backward starts behind the weight gradient's base, which
  // waits for the GEMM to complete: the GEMM publishes nothing and runs in the stock tile order, and
  // the one flag of the whole row is reset to 0, published.
  const bool plain = wgrad && ctx.num_clusters > 0;
  const size_t n_chunk = plain ? 0 : combine_chunk(hidden);
  const size_t chunk = plain ? hidden : n_chunk;

  // With the weight gradient the GEMM runs alone on every SM, and the weight gradient beside the
  // dispatch backward.
  ep::WorkspaceLayout layout;
  const ep::EpCombineGemmConfig config{rows,
                                       k,
                                       hidden,
                                       num_experts,
                                       wgrad ? ctx.all_clusters : ctx.num_clusters,
                                       n_chunk,
                                       !plain && kCombineAtGemmEntry,
                                       true,
                                       mxfp8,
                                       ctx.alignment()};
  const tvm::ffi::Function gemm = get_gemm(config, &layout);
  GemmScratch scratch = claim_scratch(layout.total_bytes, num_experts, rows, true,
                                      wgrad ? wgrad->layout.total_bytes : 0);
  set_chunk_words(scratch, layout, plain ? 0 : hidden / chunk,
                  plain ? layout.total_bytes : layout.signal);
  if (wgrad) {
    scratch.reset.fill(scratch.wgrad + wgrad->layout.counter, 1, ep::kFillZero);
  }
  set_row_ends(scratch, recv_tokens_per_expert);
  ep::reset_workspace(scratch.reset, stream);
  if (wgrad && needs_descriptors(*wgrad)) {
    launch_wgrad(*wgrad, scratch, 1, false, 0, stream, true);
  }
  // The columnwise wgrad_input, before the GEMM.
  if (requant) {
    launch_requant(wgrad_input, scratch, rows, hidden, num_experts, 0, stream);
  }

  // The GEMM fills the flags; the dispatch backward is its programmatic dependent.
  auto a = wrap_rowwise(grad_gemm_out, true);
  auto d = wrap_rowwise(grad_recv_tokens, true);
  auto row_ends = wrap_raw(scratch.row_ends, kNVTEInt32, num_experts);
  auto alpha = wrap_raw(scratch.alpha, kNVTEFloat32, num_experts);
  auto prob = wrap_raw(scratch.ones, kNVTEFloat32, rows);
  auto ws = wrap_raw(scratch.base, kNVTEByte, layout.total_bytes);
  if (mxfp8) {
    // The columnwise forward weight.
    auto b = wrap_columnwise(weight, false);
    auto sfb = wrap_scales(weight, true, num_experts * k, hidden);
    auto sfa = wrap_scales(grad_gemm_out, false, rows, k);
    gemm(&a, &b, &sfb, &sfa, &d, &row_ends, &alpha, &prob, &ws, static_cast<void*>(stream));
  } else {
    auto b = wrap_rowwise(weight, false);
    gemm(&a, &b, &d, &row_ends, &alpha, &prob, &ws, static_cast<void*>(stream));
  }
  // The weight gradient's base starts once the GEMM completes and lets the dispatch backward in.
  if (wgrad) {
    launch_wgrad(*wgrad, scratch, ctx.num_clusters, true, 0, stream, false);
  }

  NVTECommWindow no_window{};
  TensorWrapper signal(scratch.base + layout.signal, std::vector<size_t>{hidden / chunk},
                       DType::kInt32);
  const ep::EpSignal combine_signal{signal.data(), chunk};
  ctx.backend.dispatch_bwd(handle_mem, grad_recv_tokens, no_window, grad_recv_topk_weights,
                           no_window, grad_tokens, grad_topk_weights, &ctx.layer_cfg, stream,
                           &combine_signal);
  // Its tail takes the tiles left on the communication SMs.
  if (wgrad) {
    launch_wgrad(*wgrad, scratch, ctx.all_clusters - ctx.num_clusters, false, 0, stream, false);
  }
}

}  // extern "C"

#else  // !(NVTE_WITH_NCCL_EP && NVTE_WITH_CUTEDSL): throwing stubs.

namespace {
[[noreturn]] void ep_gemm_not_built() {
  NVTE_ERROR(
      "The EP grouped GEMMs need TransformerEngine built with NVTE_WITH_NCCL_EP=1 and "
      "NVTE_WITH_CUTEDSL=1.");
}
}  // namespace

namespace transformer_engine {
namespace ep {
void release_gemm_scratch() {}
}  // namespace ep
}  // namespace transformer_engine

extern "C" {

void nvte_ep_dispatch_gemm_fwd(NVTETensor, NVTETensor, NVTETensor, NVTECommWindow, NVTETensor,
                               NVTETensor, NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                               NVTETensor, NVTETensor, NVTETensor, const NVTEEpLayerConfig*,
                               cudaStream_t) {
  ep_gemm_not_built();
}

void nvte_ep_combine_gemm_fwd(NVTETensor, NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                              NVTETensor, NVTETensor, const NVTEEpLayerConfig*, cudaStream_t) {
  ep_gemm_not_built();
}

void nvte_ep_combine_gemm_bwd(NVTETensor, NVTETensor, NVTECommWindow, NVTETensor, NVTETensor,
                              NVTETensor, NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                              NVTETensor, NVTETensor, NVTETensor, int, const NVTEEpLayerConfig*,
                              cudaStream_t) {
  ep_gemm_not_built();
}

void nvte_ep_dispatch_gemm_bwd(NVTETensor, NVTETensor, NVTETensor, NVTETensor, NVTETensor,
                               NVTETensor, NVTETensor, NVTETensor, NVTETensor, NVTETensor, int,
                               const NVTEEpLayerConfig*, cudaStream_t) {
  ep_gemm_not_built();
}

}  // extern "C"

#endif
