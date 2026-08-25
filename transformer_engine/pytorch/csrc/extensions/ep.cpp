/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifdef NVTE_WITH_NCCL_EP

#include "transformer_engine/ep.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <torch/extension.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include "transformer_engine/comm_window.h"

// torch's NCCL symm-mem headers (zero-copy path) exist only since torch 2.11;
// without them NCCL_HAS_SYMMEM_SUPPORT stays undefined and zero-copy is compiled out.
#if __has_include(<torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>)
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#endif

#ifdef NCCL_HAS_SYMMEM_SUPPORT
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#endif

#include "../common.h"
#include "../extensions.h"
#include "transformer_engine/gemm.h"

namespace transformer_engine::pytorch {

namespace {

// EP process group name, captured at ep_initialize. Used by the symm-mem
// window resolver below to look up SymmetricMemory for payload tensors.
// Empty until ep_initialize.
std::string g_ep_group_name;  // NOLINT(runtime/string)

// True while the EP backend holds a borrowed reference to torch's NCCL comm.
bool g_ep_initialized = false;

// Zero-copy IO toggle captured at ep_initialize. Atomic so the Python-side
// toggle is safe against concurrent ep_dispatch/combine (which release the GIL).
std::atomic<bool> g_zero_copy_enabled{false};

// Sentinel returned by maybe_make_window when zero-copy is off or the tensor
// is not symm-mem-backed; the backend treats it as "no window, use staged copy".
constexpr NVTECommWindow kNoWindow = {nullptr, 0};

// Resolve ``t`` to an NCCL symm-mem window for the zero-copy one-sided path.
// Returns ``kNoWindow`` when symm-mem support isn't compiled in, zero-copy is
// disabled, no group is set, or ``t`` isn't symm-mem-backed; callers pass the
// resulting window unconditionally to the backend.
NVTECommWindow maybe_make_window(const at::Tensor& t) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  if (!g_zero_copy_enabled.load(std::memory_order_relaxed)) return kNoWindow;
  if (g_ep_group_name.empty()) return kNoWindow;
  c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> sm;
  try {
    sm = c10d::symmetric_memory::rendezvous(t, g_ep_group_name);
  } catch (const std::exception&) {
    return kNoWindow;  // Tensor not symm-mem-backed; fall back to staged copy.
  }
  if (sm == nullptr) return kNoWindow;
  auto* nccl_sm = dynamic_cast<c10d::symmetric_memory::NCCLSymmetricMemory*>(sm.get());
  NVTE_CHECK(nccl_sm != nullptr,
             "Symm-mem backend mismatch: expected NCCLSymmetricMemory. Set the backend to "
             "\"NCCL\" before allocating EP payload buffers.");
  // rendezvous resolves ``t`` by its storage base, so get_offset() is the allocation's offset in
  // the NCCL window. Add ``t``'s own storage offset so a slice/view of a symm-mem allocation
  // (e.g. the scale region carved from a shared recv buffer) resolves to its true position in the
  // window rather than the allocation base.
  const uint64_t offset =
      static_cast<uint64_t>(nccl_sm->get_offset()) +
      static_cast<uint64_t>(t.storage_offset()) * static_cast<uint64_t>(t.element_size());
  return NVTECommWindow{static_cast<ncclWindow_t>(nccl_sm->get_window()), offset};
#else
  (void)t;
  return kNoWindow;
#endif
}

// When zero-copy is enabled, the named tensor must be symm-mem-backed on the
// EP group. Throws a clear error otherwise. No-op when zero-copy is off or
// symm-mem support isn't compiled in. Mirrors maybe_make_window's resolution
// path but turns the "not symm-mem" outcome into a hard error.
void check_symm_mem_required(const at::Tensor& t, const char* name) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  if (!g_zero_copy_enabled.load(std::memory_order_relaxed)) return;
  NVTE_CHECK(!g_ep_group_name.empty(),
             "Zero-copy is enabled but EP group name is unset; call ep_initialize first.");
  c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> sm;
  try {
    sm = c10d::symmetric_memory::rendezvous(t, g_ep_group_name);
  } catch (const std::exception&) {
    sm = nullptr;
  }
  NVTE_CHECK(sm != nullptr, "ep zero-copy: ", name,
             " must be symm-mem-backed on the EP group (allocate via symm_mem_alloc).");
#else
  (void)t;
  (void)name;
#endif
}

// Validates topk_idx is int32 or int64 and returns the matching TE dtype.
DType check_topk_idx_dtype(at::Tensor topk_idx) {
  NVTE_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
  NVTE_CHECK(topk_idx.scalar_type() == at::kLong || topk_idx.scalar_type() == at::kInt,
             "topk_idx must be int32 or int64; got dtype=", c10::toString(topk_idx.scalar_type()));
  return GetTransformerEngineDType(topk_idx.scalar_type());
}

using Shape = std::vector<size_t>;

// EP block scaling supports only E4M3 MXFP8 today. A future block-scaled recipe (e.g. NVFP4)
// would also set is_scaled but carry a non-FP8 token dtype, so key the guard on the FP8 dtype.
bool is_mxfp8_scaled(bool is_scaled, const at::Tensor& data) {
  return is_scaled && data.scalar_type() == at::kFloat8_e4m3fn;
}

// Validate the scale-inverse pair of a block-scaled EP op and return the scale column count
// (hidden/block). Both scales must be 2D contiguous with matching cols dividing H and numels
// equal to their row counts times cols; the recv scale must be symm-mem-backed under zero-copy.
size_t check_mxfp8_scale_pair(const at::Tensor& send_scale, const at::Tensor& recv_scale,
                              size_t send_rows, size_t recv_rows, size_t H, const char* recv_name) {
  NVTE_CHECK(send_scale.dim() >= 2 && recv_scale.dim() >= 2,
             "scale-inverses must be at least 2D [., H/block]");
  NVTE_CHECK(send_scale.is_contiguous() && recv_scale.is_contiguous(),
             "scale-inverses must be contiguous");
  const size_t sc_cols = static_cast<size_t>(send_scale.size(-1));
  NVTE_CHECK(sc_cols > 0 && H % sc_cols == 0, "scale cols (", sc_cols,
             ") must be a non-zero divisor of hidden (", H, ")");
  NVTE_CHECK(static_cast<size_t>(recv_scale.size(-1)) == sc_cols,
             "recv scale cols must match send scale cols");
  NVTE_CHECK(static_cast<size_t>(send_scale.numel()) == send_rows * sc_cols,
             "send scale numel must equal rows * cols");
  NVTE_CHECK(static_cast<size_t>(recv_scale.numel()) == recv_rows * sc_cols,
             "recv scale numel must equal rows * cols");
  check_symm_mem_required(recv_scale, recv_name);
  return sc_cols;
}

}  // namespace

bool ep_get_zero_copy() { return g_zero_copy_enabled.load(std::memory_order_relaxed); }

bool ep_zero_copy_supported() {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
  return true;
#else
  return false;
#endif
}

// ── Bootstrap ────────────────────────────────────────────────────────────────
// Borrows torch's NCCL host comm (from ``ProcessGroupNCCL._comm_ptr()``).
// ``group_name`` is captured for the symm-mem window resolver.

void ep_initialize(uintptr_t comm_ptr, const std::string& group_name, int64_t num_experts,
                   int64_t max_tokens_per_rank, int64_t max_recv_tokens_per_rank,
                   int64_t hidden_dim, int64_t max_num_sms, pybind11::object max_token_dtype,
                   bool zero_copy, int64_t num_topk, bool drop_on_overflow) {
  NVTE_CHECK(!group_name.empty(), "group_name must be non-empty (used for symm-mem lookup)");
  NVTE_CHECK(comm_ptr != 0, "comm_ptr must be non-null (torch NCCL host comm pointer)");
  NVTE_CHECK(!g_ep_initialized, "ep_initialize called twice without ep_finalize");

  auto ep_comm = reinterpret_cast<ncclComm_t>(comm_ptr);
  int ep_size = 0;
  NVTE_CHECK(ncclCommCount(ep_comm, &ep_size) == ncclSuccess, "ncclCommCount failed");
  auto torch_dtype = max_token_dtype.cast<at::ScalarType>();
  NVTEEpGroupConfig cfg{
      .struct_size = sizeof(NVTEEpGroupConfig),
      .ep_size = ep_size,
      .num_experts = static_cast<int>(num_experts),
      .max_tokens_per_rank = static_cast<int>(max_tokens_per_rank),
      .max_recv_tokens_per_rank = static_cast<int>(max_recv_tokens_per_rank),
      .hidden_dim = static_cast<int>(hidden_dim),
      .num_comm_sms = static_cast<int>(max_num_sms),
      .max_token_dtype = static_cast<NVTEDType>(GetTransformerEngineDType(torch_dtype)),
      .zero_copy = zero_copy ? 1 : 0,
      .num_topk = static_cast<int>(num_topk),
      .drop_on_overflow = drop_on_overflow ? 1 : 0,
  };
  // Release the GIL only around the native init. It must stay held while pybind11 casts
  // the ``max_token_dtype`` object above and destroys the by-value ``pybind11::object``
  // parameter on return; releasing it across those trips pybind11's dec_ref GIL assertion.
  {
    pybind11::gil_scoped_release nogil;
    nvte_ep_initialize(static_cast<void*>(ep_comm), &cfg);
  }
  g_zero_copy_enabled.store(zero_copy, std::memory_order_relaxed);
  g_ep_initialized = true;
  g_ep_group_name = group_name;
}

void ep_finalize() {
  if (!g_ep_initialized) return;
  // The borrowed comm is owned by torch's symm-mem layer; don't destroy it.
  nvte_ep_shutdown();
  g_ep_initialized = false;
  g_ep_group_name.clear();
  g_zero_copy_enabled.store(false, std::memory_order_relaxed);
}

namespace {

NVTEEpLayerConfig make_layer_cfg(int64_t top_k, int64_t dispatch_output_per_expert_alignment) {
  return NVTEEpLayerConfig{
      .struct_size = sizeof(NVTEEpLayerConfig),
      .top_k = static_cast<int>(top_k),
      .dispatch_output_per_expert_alignment =
          static_cast<size_t>(dispatch_output_per_expert_alignment),
  };
}

}  // namespace

int64_t ep_handle_mem_size(int64_t top_k, int64_t dispatch_output_per_expert_alignment) {
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment);
  return static_cast<int64_t>(nvte_ep_handle_mem_size(&layer_cfg));
}

// ── Per-step ops ─────────────────────────────────────────────────────────────

void ep_prepare(at::Tensor handle_mem, at::Tensor topk_idx, at::Tensor tokens_per_expert,
                int64_t top_k, int64_t dispatch_output_per_expert_alignment,
                at::Tensor total_recv_tokens) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(topk_idx.dim() >= 2, "topk_idx must be at least 2D [..., top_k]");
  auto idx_dtype = check_topk_idx_dtype(topk_idx);
  // NCCL EP requires all prepare int output counters to share a dtype.
  NVTE_CHECK(tokens_per_expert.scalar_type() == at::kLong, "tokens_per_expert must be int64");
  NVTE_CHECK(total_recv_tokens.scalar_type() == at::kLong, "total_recv_tokens must be int64");
  const size_t T_flat = topk_idx.numel() / topk_idx.size(-1);
  const size_t topk_n = static_cast<size_t>(topk_idx.size(-1));

  auto topk_idx_te =
      makeTransformerEngineTensor(topk_idx.data_ptr(), Shape{T_flat, topk_n}, idx_dtype);
  auto tokens_per_expert_te = makeTransformerEngineTensor(
      tokens_per_expert.data_ptr(), Shape{static_cast<size_t>(tokens_per_expert.numel())},
      DType::kInt64);
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  // [1] int64 scalar recv-slot total; lets the caller size dispatch outputs
  // (eager) or detect overflow past recv_capacity_per_rank (graph mode).
  auto total_recv_tokens_te = makeTransformerEngineTensor(
      total_recv_tokens.data_ptr(), Shape{static_cast<size_t>(total_recv_tokens.numel())},
      DType::kInt64);

  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment);
  nvte_ep_prepare(handle_mem_te.data(), topk_idx_te.data(), tokens_per_expert_te.data(),
                  total_recv_tokens_te.data(), &layer_cfg, stream);
}

// tokens_scale_inv / recv_scale_inv are set only for block-scaled dispatch (for
// now MXFP8): tokens/recv_tokens carry e4m3 data and the scale tensors carry the
// unswizzled e8m0 scale-inverses [T, H/block]. Both null => bf16/fp16/fp32.
void ep_dispatch(at::Tensor handle_mem, at::Tensor topk_idx, at::Tensor tokens,
                 at::Tensor topk_weights, at::Tensor recv_tokens, at::Tensor recv_topk_weights,
                 std::optional<at::Tensor> tokens_scale_inv,
                 std::optional<at::Tensor> recv_scale_inv) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(tokens.dim() >= 2, "tokens must be at least 2D [..., H]");
  NVTE_CHECK(topk_idx.dim() >= 2, "topk_idx must be at least 2D [..., top_k]");
  NVTE_CHECK(topk_weights.dim() >= 2, "topk_weights must be at least 2D [..., top_k]");
  NVTE_CHECK(recv_tokens.dim() >= 2, "recv_tokens must be at least 2D [..., recv_pr, H]");
  auto idx_dtype = check_topk_idx_dtype(topk_idx);
  NVTE_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
  NVTE_CHECK(topk_weights.is_contiguous(), "topk_weights must be contiguous");
  NVTE_CHECK(recv_tokens.is_contiguous(), "recv_tokens must be contiguous");
  NVTE_CHECK(recv_topk_weights.is_contiguous(), "recv_topk_weights must be contiguous");

  const size_t H = static_cast<size_t>(tokens.size(-1));
  const size_t T_flat = tokens.numel() / H;
  const size_t topk_n = static_cast<size_t>(topk_idx.size(-1));
  const size_t recv_pr = recv_tokens.numel() / H;

  NVTE_CHECK(static_cast<size_t>(topk_weights.size(-1)) == topk_n,
             "topk_weights last dim must equal topk_idx last dim");
  NVTE_CHECK(static_cast<size_t>(topk_idx.numel()) == T_flat * topk_n,
             "topk_idx token count must equal tokens token count");
  NVTE_CHECK(static_cast<size_t>(topk_weights.numel()) == T_flat * topk_n,
             "topk_weights token count must equal tokens token count");
  NVTE_CHECK(static_cast<size_t>(recv_topk_weights.numel()) == recv_pr,
             "recv_topk_weights total size must equal recv_tokens recv_pr");
  NVTE_CHECK(recv_tokens.scalar_type() == tokens.scalar_type(), "recv_tokens dtype (",
             c10::toString(recv_tokens.scalar_type()), ") must match tokens dtype (",
             c10::toString(tokens.scalar_type()), ")");
  check_symm_mem_required(recv_tokens, "recv_tokens");
  check_symm_mem_required(recv_topk_weights, "recv_topk_weights");

  // Block-scaled dispatch: tokens carry e4m3 data and the scale tensors carry
  // unswizzled e8m0 scale-inverses [T, H/block]. Scales ride in the tensor; the
  // backend keys on the tensor's scaling mode. is_mxfp8 is split from is_scaled
  // so future block-scaled recipes can reuse the scale-routing plumbing while
  // building their own TE tensors; only MXFP8 is supported for now.
  const bool is_scaled = tokens_scale_inv.has_value();
  const bool is_mxfp8 = is_mxfp8_scaled(is_scaled, tokens);
  size_t sc_cols = 0;
  if (is_scaled) {
    NVTE_CHECK(recv_scale_inv.has_value(),
               "recv_scale_inv must be provided together with tokens_scale_inv");
    NVTE_CHECK(is_mxfp8, "EP dispatch currently supports only E4M3 MXFP8 block scaling");
    sc_cols = check_mxfp8_scale_pair(*tokens_scale_inv, *recv_scale_inv, T_flat, recv_pr, H,
                                     "recv_scale_inv");
  }

  auto tok_dtype = GetTransformerEngineDType(tokens.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto topk_idx_te =
      makeTransformerEngineTensor(topk_idx.data_ptr(), Shape{T_flat, topk_n}, idx_dtype);
  auto tokens_te =
      is_mxfp8 ? makeTransformerEngineTensor(tokens.data_ptr(), Shape{T_flat, H}, tok_dtype,
                                             nullptr, nullptr, tokens_scale_inv->data_ptr(),
                                             Shape{T_flat, sc_cols}, NVTE_MXFP8_1D_SCALING)
               : makeTransformerEngineTensor(tokens.data_ptr(), Shape{T_flat, H}, tok_dtype);
  auto topk_w_te =
      makeTransformerEngineTensor(topk_weights.data_ptr(), Shape{T_flat, topk_n}, DType::kFloat32);
  auto recv_tokens_te =
      is_mxfp8 ? makeTransformerEngineTensor(recv_tokens.data_ptr(), Shape{recv_pr, H}, tok_dtype,
                                             nullptr, nullptr, recv_scale_inv->data_ptr(),
                                             Shape{recv_pr, sc_cols}, NVTE_MXFP8_1D_SCALING)
               : makeTransformerEngineTensor(recv_tokens.data_ptr(), Shape{recv_pr, H}, tok_dtype);
  auto recv_topk_w_te =
      makeTransformerEngineTensor(recv_topk_weights.data_ptr(), Shape{recv_pr}, DType::kFloat32);

  // top_k / alignment are carried by the cached layer_cfg seeded at ep_prepare;
  // per-step ops look them up by handle_mem pointer in the backend.
  NVTECommWindow tokens_win = maybe_make_window(tokens);
  NVTECommWindow topk_w_win = maybe_make_window(topk_weights);
  NVTECommWindow recv_tokens_win = maybe_make_window(recv_tokens);
  NVTECommWindow recv_topk_w_win = maybe_make_window(recv_topk_weights);
  // Block-scaled zero-copy: the scale-inverse rides on the data tensor's window.
  // Send scales (tokens_scale_inv) stay staged like the send data; recv scales
  // are the one-sided write target and must be symm-mem-backed under zero-copy.
  if (is_scaled) {
    const NVTECommWindow tsi_win = maybe_make_window(*tokens_scale_inv);
    const NVTECommWindow rsi_win = maybe_make_window(*recv_scale_inv);
    tokens_win.scale_window = tsi_win.window;
    tokens_win.scale_offset = tsi_win.offset;
    recv_tokens_win.scale_window = rsi_win.window;
    recv_tokens_win.scale_offset = rsi_win.offset;
  }
  nvte_ep_dispatch(handle_mem_te.data(), topk_idx_te.data(), tokens_te.data(), tokens_win,
                   topk_w_te.data(), topk_w_win, recv_tokens_te.data(), recv_tokens_win,
                   recv_topk_w_te.data(), recv_topk_w_win, stream);
}

std::vector<at::Tensor> ep_prepare_and_dispatch(
    at::Tensor handle_mem, at::Tensor topk_idx, at::Tensor tokens, at::Tensor topk_weights,
    at::Tensor tokens_per_expert, at::Tensor total_recv_tokens, int64_t top_k,
    int64_t dispatch_output_per_expert_alignment, std::optional<at::Tensor> recv_tokens,
    std::optional<at::Tensor> recv_topk_weights, std::optional<at::Tensor> recv_scale_inv,
    std::optional<at::Tensor> tokens_scale_inv) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(tokens.dim() == 2, "dispatch tokens must be 2D [T, H]");
  const bool is_scaled = tokens_scale_inv.has_value();
  // Eager sizes the recv outputs from the per-step count and allocates them here; a caller
  // that supplies them (non-eager) uses static recv capacity, so no count read is needed.
  const bool caller_omitted_recv = !recv_tokens.has_value();

  ep_prepare(handle_mem, topk_idx, tokens_per_expert, top_k, dispatch_output_per_expert_alignment,
             total_recv_tokens);

  at::Tensor rt, rw, rsi;
  if (caller_omitted_recv) {
    // Prepare writes the per-step recv total straight into pinned host total_recv_tokens
    // (UVA), so a stream sync makes it host-readable with no D2H copy. Kept in one C++ op
    // so no Python runs between the count read and the dispatch launch below.
    NVTE_CHECK(total_recv_tokens.is_pinned(), "eager total_recv_tokens must be pinned host memory");
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    const int64_t num_recv = *total_recv_tokens.data_ptr<int64_t>();
    // Recv outputs sized to the host count (eager forbids zero-copy, so plain allocations).
    const int64_t H = tokens.size(-1);
    rt = at::empty({num_recv, H}, tokens.options());
    rw = at::empty({num_recv}, topk_weights.options());
    if (is_scaled)
      rsi = at::empty({num_recv, tokens_scale_inv->size(-1)}, tokens_scale_inv->options());
  } else {
    rt = *recv_tokens;
    rw = *recv_topk_weights;
    if (is_scaled) {
      NVTE_CHECK(recv_scale_inv.has_value(),
                 "recv_scale_inv must be provided together with tokens_scale_inv");
      rsi = *recv_scale_inv;
    }
  }

  ep_dispatch(handle_mem, topk_idx, tokens, topk_weights, rt, rw, tokens_scale_inv,
              is_scaled ? std::optional<at::Tensor>(rsi) : std::nullopt);
  if (!caller_omitted_recv) return {};  // caller buffers were mutated in place
  if (is_scaled) return {rt, rw, rsi};
  return {rt, rw};
}

void ep_combine(at::Tensor handle_mem, at::Tensor expert_out, at::Tensor result) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(expert_out.dim() >= 2, "expert_out must be at least 2D [..., recv_pr, H]");
  NVTE_CHECK(result.dim() >= 2, "result must be at least 2D [..., H]");
  NVTE_CHECK(expert_out.is_contiguous(), "expert_out must be contiguous");

  const size_t H = static_cast<size_t>(expert_out.size(-1));
  const size_t recv_pr = expert_out.numel() / H;
  const size_t T_flat = result.numel() / H;
  NVTE_CHECK(static_cast<size_t>(result.size(-1)) == H,
             "result hidden dim must equal expert_out hidden dim");
  NVTE_CHECK(result.scalar_type() == expert_out.scalar_type(), "result dtype (",
             c10::toString(result.scalar_type()), ") must match expert_out dtype (",
             c10::toString(expert_out.scalar_type()), ")");
  check_symm_mem_required(expert_out, "expert_out");

  auto eo_dtype = GetTransformerEngineDType(expert_out.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto expert_out_te =
      makeTransformerEngineTensor(expert_out.data_ptr(), Shape{recv_pr, H}, eo_dtype);
  auto result_te = makeTransformerEngineTensor(result.data_ptr(), Shape{T_flat, H}, eo_dtype);

  NVTECommWindow expert_out_win = maybe_make_window(expert_out);
  nvte_ep_combine(handle_mem_te.data(), expert_out_te.data(), expert_out_win, result_te.data(),
                  stream);
}

void ep_dispatch_bwd(at::Tensor handle_mem, at::Tensor grad, at::Tensor g_recv_topk_weights,
                     at::Tensor grad_tokens, at::Tensor grad_topk_weights) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(grad.dim() >= 2, "grad must be at least 2D [..., recv_pr, H]");
  NVTE_CHECK(grad_tokens.dim() >= 2, "grad_tokens must be at least 2D [..., H]");
  NVTE_CHECK(grad_topk_weights.dim() >= 2, "grad_topk_weights must be at least 2D [..., top_k]");
  NVTE_CHECK(grad.is_contiguous(), "grad must be contiguous");
  NVTE_CHECK(g_recv_topk_weights.is_contiguous(), "g_recv_topk_weights must be contiguous");

  const size_t H = static_cast<size_t>(grad.size(-1));
  const size_t recv_pr = grad.numel() / H;
  const size_t T_flat = grad_tokens.numel() / H;
  const size_t topk_n = static_cast<size_t>(grad_topk_weights.size(-1));
  NVTE_CHECK(static_cast<size_t>(g_recv_topk_weights.numel()) == recv_pr,
             "g_recv_topk_weights total size must equal grad recv_pr");
  NVTE_CHECK(static_cast<size_t>(grad_tokens.size(-1)) == H,
             "grad_tokens hidden dim must equal grad H");
  NVTE_CHECK(static_cast<size_t>(grad_topk_weights.numel()) == T_flat * topk_n,
             "grad_topk_weights numel (", grad_topk_weights.numel(),
             ") must equal T_flat * top_k (", T_flat * topk_n, ")");
  NVTE_CHECK(grad_tokens.scalar_type() == grad.scalar_type(), "grad_tokens dtype (",
             c10::toString(grad_tokens.scalar_type()), ") must match grad dtype (",
             c10::toString(grad.scalar_type()), ")");
  // Upstream grads are autograd-allocated, so they take the staged-copy path.

  auto g_dtype = GetTransformerEngineDType(grad.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto grad_te = makeTransformerEngineTensor(grad.data_ptr(), Shape{recv_pr, H}, g_dtype);
  auto g_recv_w_te =
      makeTransformerEngineTensor(g_recv_topk_weights.data_ptr(), Shape{recv_pr}, DType::kFloat32);
  auto grad_tokens_te =
      makeTransformerEngineTensor(grad_tokens.data_ptr(), Shape{T_flat, H}, g_dtype);
  auto grad_topk_w_te = makeTransformerEngineTensor(grad_topk_weights.data_ptr(),
                                                    Shape{T_flat, topk_n}, DType::kFloat32);

  NVTECommWindow grad_win = maybe_make_window(grad);
  NVTECommWindow g_recv_w_win = maybe_make_window(g_recv_topk_weights);
  nvte_ep_dispatch_bwd(handle_mem_te.data(), grad_te.data(), grad_win, g_recv_w_te.data(),
                       g_recv_w_win, grad_tokens_te.data(), grad_topk_w_te.data(), stream);
}

void ep_combine_bwd(at::Tensor handle_mem, at::Tensor grad, at::Tensor grad_expert_out,
                    std::optional<at::Tensor> grad_scale_inv,
                    std::optional<at::Tensor> grad_expert_out_scale_inv) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(grad.dim() >= 2, "grad must be at least 2D [..., H]");
  NVTE_CHECK(grad_expert_out.dim() >= 2, "grad_expert_out must be at least 2D [..., recv_pr, H]");
  NVTE_CHECK(grad.is_contiguous(), "grad must be contiguous");
  NVTE_CHECK(grad_expert_out.is_contiguous(), "grad_expert_out must be contiguous");

  const size_t H = static_cast<size_t>(grad.size(-1));
  const size_t T_flat = grad.numel() / H;
  const size_t recv_pr = grad_expert_out.numel() / H;
  NVTE_CHECK(static_cast<size_t>(grad_expert_out.size(-1)) == H,
             "grad_expert_out hidden dim must match grad H");
  NVTE_CHECK(grad_expert_out.scalar_type() == grad.scalar_type(), "grad_expert_out dtype (",
             c10::toString(grad_expert_out.scalar_type()), ") must match grad dtype (",
             c10::toString(grad.scalar_type()), ")");
  // grad is autograd-allocated (staged-copy path); grad_expert_out is the
  // EpBuffer-owned scatter target and must be symm-mem in zero-copy mode.
  check_symm_mem_required(grad_expert_out, "grad_expert_out");

  // Block-scaled (MXFP8) backward: grad/grad_expert_out carry e4m3 data and the scale
  // tensors carry the unswizzled e8m0 scale-inverses [., H/block]; the reverse-direction
  // dispatch forwards them like the forward path. is_mxfp8 is split from is_scaled so
  // future block-scaled recipes can reuse the plumbing; only MXFP8 is supported for now.
  const bool is_scaled = grad_scale_inv.has_value();
  const bool is_mxfp8 = is_mxfp8_scaled(is_scaled, grad);
  size_t sc_cols = 0;
  if (is_scaled) {
    NVTE_CHECK(grad_expert_out_scale_inv.has_value(),
               "grad_expert_out_scale_inv must be provided together with grad_scale_inv");
    NVTE_CHECK(is_mxfp8, "EP combine backward currently supports only E4M3 MXFP8 block scaling");
    sc_cols = check_mxfp8_scale_pair(*grad_scale_inv, *grad_expert_out_scale_inv, T_flat, recv_pr,
                                     H, "grad_expert_out_scale_inv");
  }

  auto g_dtype = GetTransformerEngineDType(grad.scalar_type());
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  auto grad_te = is_mxfp8
                     ? makeTransformerEngineTensor(grad.data_ptr(), Shape{T_flat, H}, g_dtype,
                                                   nullptr, nullptr, grad_scale_inv->data_ptr(),
                                                   Shape{T_flat, sc_cols}, NVTE_MXFP8_1D_SCALING)
                     : makeTransformerEngineTensor(grad.data_ptr(), Shape{T_flat, H}, g_dtype);
  auto grad_expert_out_te =
      is_mxfp8
          ? makeTransformerEngineTensor(grad_expert_out.data_ptr(), Shape{recv_pr, H}, g_dtype,
                                        nullptr, nullptr, grad_expert_out_scale_inv->data_ptr(),
                                        Shape{recv_pr, sc_cols}, NVTE_MXFP8_1D_SCALING)
          : makeTransformerEngineTensor(grad_expert_out.data_ptr(), Shape{recv_pr, H}, g_dtype);

  // grad is autograd-allocated (staged); grad_expert_out resolves to a symm-mem
  // window in zero-copy mode, else kNoWindow for the staged path.
  NVTECommWindow grad_win = maybe_make_window(grad);
  NVTECommWindow grad_expert_out_win = maybe_make_window(grad_expert_out);
  // Block-scaled zero-copy: the scale-inverse rides on the data tensor's window,
  // mirroring the forward dispatch. Send scales stay staged like the send data;
  // recv scales are the one-sided write target and must be symm-mem-backed.
  if (is_scaled) {
    const NVTECommWindow gsi_win = maybe_make_window(*grad_scale_inv);
    const NVTECommWindow gesi_win = maybe_make_window(*grad_expert_out_scale_inv);
    grad_win.scale_window = gsi_win.window;
    grad_win.scale_offset = gsi_win.offset;
    grad_expert_out_win.scale_window = gesi_win.window;
    grad_expert_out_win.scale_offset = gesi_win.offset;
  }
  nvte_ep_combine_bwd(handle_mem_te.data(), grad_te.data(), grad_win, grad_expert_out_te.data(),
                      grad_expert_out_win, stream);
}

void register_ep_bindings(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("ep_initialize", &ep_initialize,
        "Initialize the EP backend; borrows torch's NCCL comm pointed to by ``comm_ptr``.",
        py::arg("comm_ptr"), py::arg("group_name"), py::arg("num_experts"),
        py::arg("max_tokens_per_rank"), py::arg("max_recv_tokens_per_rank"), py::arg("hidden_dim"),
        py::arg("max_num_sms") = 0, py::arg("max_token_dtype"), py::arg("zero_copy") = false,
        py::arg("num_topk") = 0, py::arg("drop_on_overflow") = false);
  m.def("ep_finalize", &ep_finalize, "Tear down the EP backend. Idempotent.",
        py::call_guard<py::gil_scoped_release>());
  m.def("ep_get_zero_copy", &ep_get_zero_copy, "Return the current EP zero-copy toggle state.");
  m.def("ep_zero_copy_supported", &ep_zero_copy_supported,
        "Return True when the extension was built with NCCL symm-mem (zero-copy) support.");
  m.def("ep_handle_mem_size", &ep_handle_mem_size,
        "Return the handle_mem byte size for the given layer config.", py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment") = 0);
  m.def("ep_prepare", &ep_prepare, "EP prepare", py::arg("handle_mem"), py::arg("topk_idx"),
        py::arg("tokens_per_expert"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"), py::arg("total_recv_tokens"),
        py::call_guard<py::gil_scoped_release>());
  m.def("ep_dispatch", &ep_dispatch, "EP dispatch", py::arg("handle_mem"), py::arg("topk_idx"),
        py::arg("tokens"), py::arg("topk_weights"), py::arg("recv_tokens"),
        py::arg("recv_topk_weights"), py::arg("tokens_scale_inv") = std::nullopt,
        py::arg("recv_scale_inv") = std::nullopt, py::call_guard<py::gil_scoped_release>());
  m.def("ep_prepare_and_dispatch", &ep_prepare_and_dispatch, "Fused EP prepare + dispatch",
        py::arg("handle_mem"), py::arg("topk_idx"), py::arg("tokens"), py::arg("topk_weights"),
        py::arg("tokens_per_expert"), py::arg("total_recv_tokens"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"), py::arg("recv_tokens") = std::nullopt,
        py::arg("recv_topk_weights") = std::nullopt, py::arg("recv_scale_inv") = std::nullopt,
        py::arg("tokens_scale_inv") = std::nullopt, py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine", &ep_combine, "EP combine", py::call_guard<py::gil_scoped_release>());
  m.def("ep_dispatch_bwd", &ep_dispatch_bwd, "EP dispatch backward",
        py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine_bwd", &ep_combine_bwd, "EP combine backward", py::arg("handle_mem"),
        py::arg("grad"), py::arg("grad_expert_out"), py::arg("grad_scale_inv") = std::nullopt,
        py::arg("grad_expert_out_scale_inv") = std::nullopt,
        py::call_guard<py::gil_scoped_release>());
}

}  // namespace transformer_engine::pytorch

#endif  // NVTE_WITH_NCCL_EP
