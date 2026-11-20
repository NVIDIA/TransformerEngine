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

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include "transformer_engine/comm_window.h"
#include "transformer_engine/ep_gemm.h"

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

#ifdef NCCL_HAS_SYMMEM_SUPPORT
// Offset of a symm-mem allocation relative to the start of its NCCL window.
// Newer torch places the signal pad at the front of the allocation and exposes
// get_window_offset() for it; on older torch the window starts at the buffer
// base, where get_offset() is already window-relative.
template <typename T>
auto symm_mem_window_offset(T* sm, int) -> decltype(sm->get_window_offset()) {
  return sm->get_window_offset();
}
template <typename T>
size_t symm_mem_window_offset(T* sm, ...) {
  return sm->get_offset();
}
#endif

// Resolve ``t`` to its NCCL symm-mem window. Returns ``kNoWindow`` when symm-mem
// support isn't compiled in, no group is set, or ``t`` isn't symm-mem-backed.
NVTECommWindow make_window(const at::Tensor& t) {
#ifdef NCCL_HAS_SYMMEM_SUPPORT
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
  // NCCL EP consumes window-relative offsets. Add ``t``'s own storage offset so a
  // slice/view of a symm-mem allocation (e.g. the scale region carved from a shared
  // recv buffer) resolves to its true position in the window.
  const uint64_t offset =
      static_cast<uint64_t>(symm_mem_window_offset(nccl_sm, 0)) +
      static_cast<uint64_t>(t.storage_offset()) * static_cast<uint64_t>(t.element_size());
  return NVTECommWindow{static_cast<ncclWindow_t>(nccl_sm->get_window()), offset};
#else
  (void)t;
  return kNoWindow;
#endif
}

// Resolve ``t`` to an NCCL symm-mem window for the zero-copy one-sided path.
// Returns ``kNoWindow`` when zero-copy is disabled or make_window finds none;
// callers pass the resulting window unconditionally to the backend.
NVTECommWindow maybe_make_window(const at::Tensor& t) {
  if (!g_zero_copy_enabled.load(std::memory_order_relaxed)) return kNoWindow;
  return make_window(t);
}

// Window of a symm-mem input and, for MXFP8, of its scales, whatever the
// zero-copy setting. Throws when either is not symm-mem-backed on the EP group.
NVTECommWindow make_input_window(const at::Tensor& t, const std::optional<at::Tensor>& scale_inv) {
  NVTECommWindow win = make_window(t);
  NVTE_CHECK(win.window != nullptr, "symm_input needs a symm-mem input on the EP group");
  if (scale_inv.has_value()) {
    const NVTECommWindow scale_win = make_window(*scale_inv);
    NVTE_CHECK(scale_win.window != nullptr,
               "symm_input needs symm-mem input scales on the EP group");
    win.scale_window = scale_win.window;
    win.scale_offset = scale_win.offset;
  }
  return win;
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

// Returns the TE dtype for a topk_idx torch dtype; errors unless int32 or int64.
DType get_topk_idx_te_dtype(at::ScalarType dtype) {
  NVTE_CHECK(dtype == at::kLong || dtype == at::kInt,
             "topk_idx must be int32 or int64; got dtype=", c10::toString(dtype));
  return GetTransformerEngineDType(dtype);
}

// Returns the TE dtype of topk_idx; errors unless it is contiguous int32 or int64.
DType get_contiguous_topk_idx_te_dtype(at::Tensor topk_idx) {
  NVTE_CHECK(topk_idx.is_contiguous(), "topk_idx must be contiguous");
  return get_topk_idx_te_dtype(topk_idx.scalar_type());
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

// MXFP8 TE tensor of E4M3 rowwise and/or columnwise data (undefined: absent) and E8M0 scales.
TensorWrapper make_mxfp8_tensor(const at::Tensor& rowwise, const at::Tensor& rowwise_scale_inv,
                                const at::Tensor& columnwise,
                                const at::Tensor& columnwise_scale_inv, bool swizzled) {
  TensorWrapper t(NVTE_MXFP8_1D_SCALING);
  for (const bool col : {false, true}) {
    const at::Tensor& data = col ? columnwise : rowwise;
    const at::Tensor& scale_inv = col ? columnwise_scale_inv : rowwise_scale_inv;
    if (!data.defined()) {
      continue;
    }
    NVTE_CHECK(data.scalar_type() == at::kFloat8_e4m3fn && data.is_contiguous() &&
                   scale_inv.defined() && scale_inv.is_contiguous(),
               "MXFP8 operands need contiguous E4M3 data and contiguous scales");
    if (col) {
      t.set_columnwise_data(data.data_ptr(), DType::kFloat8E4M3, getTensorShape(data));
      t.set_columnwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0,
                                 getTensorShape(scale_inv));
    } else {
      t.set_rowwise_data(data.data_ptr(), DType::kFloat8E4M3, getTensorShape(data));
      t.set_rowwise_scale_inv(scale_inv.data_ptr(), DType::kFloat8E8M0, getTensorShape(scale_inv));
    }
  }
  t.set_with_gemm_swizzled_scales(swizzled);
  return t;
}

// Columnwise-only MXFP8 TE tensor with GEMM-swizzled scales.
TensorWrapper make_columnwise_mxfp8_tensor(const at::Tensor& data, const at::Tensor& scale_inv) {
  return make_mxfp8_tensor(at::Tensor(), at::Tensor(), data, scale_inv, true);
}

// Rowwise MXFP8 TE tensor when scale_inv is given, else the plain tensor.
TensorWrapper make_maybe_mxfp8_tensor(const at::Tensor& data,
                                      const std::optional<at::Tensor>& scale_inv, bool swizzled) {
  if (!scale_inv.has_value()) {
    return makeTransformerEngineTensor(data);
  }
  return make_mxfp8_tensor(data, *scale_inv, at::Tensor(), at::Tensor(), swizzled);
}

// Optional weight gradient of an overlapped backward GEMM and the activations it reads: plain, or
// columnwise MXFP8 data with its GEMM-swizzled scales.
class WeightGradTensors {
 public:
  // input_rowwise: the rowwise MXFP8 input the call requantizes into the columnwise input.
  WeightGradTensors(const std::optional<at::Tensor>& input,
                    const std::optional<at::Tensor>& weight_grad,
                    const std::optional<at::Tensor>& input_scale_inv = std::nullopt,
                    const std::optional<at::Tensor>& input_rowwise = std::nullopt,
                    const std::optional<at::Tensor>& input_rowwise_scale_inv = std::nullopt)
      : enabled_(weight_grad.has_value()) {
    NVTE_CHECK(input.has_value() == enabled_, "wgrad_input and weight_grad must be given together");
    NVTE_CHECK(enabled_ || !input_scale_inv.has_value(), "wgrad_input_scale_inv needs weight_grad");
    NVTE_CHECK(input_rowwise.has_value() == input_rowwise_scale_inv.has_value() &&
                   (input_scale_inv.has_value() || !input_rowwise.has_value()),
               "the rowwise wgrad_input needs its scales and a columnwise MXFP8 wgrad_input");
    if (enabled_) {
      NVTE_CHECK(input->is_contiguous() && weight_grad->is_contiguous(),
                 "wgrad_input and weight_grad must be contiguous");
      input_ = input_scale_inv.has_value()
                   ? make_mxfp8_tensor(input_rowwise.value_or(at::Tensor()),
                                       input_rowwise_scale_inv.value_or(at::Tensor()), *input,
                                       *input_scale_inv, true)
                   : makeTransformerEngineTensor(*input);
      output_ = makeTransformerEngineTensor(*weight_grad);
    }
  }
  NVTETensor input() { return enabled_ ? input_.data() : nullptr; }
  NVTETensor output() { return enabled_ ? output_.data() : nullptr; }

 private:
  bool enabled_;
  TensorWrapper input_;
  TensorWrapper output_;
};

// A tensor the native call reads or writes through a raw pointer: dense, on the current device.
void check_dense_cuda(const at::Tensor& t, const char* name) {
  NVTE_CHECK(t.is_cuda() && t.is_contiguous() && t.device().index() == c10::cuda::current_device(),
             name, " must be a contiguous tensor on the current CUDA device");
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
// ``comm_ptr`` is torch's borrowed NCCL host comm (see get_nccl_comm_ptr).
// ``group_name`` is captured for the symm-mem window resolver.

void ep_initialize(uintptr_t comm_ptr, const std::string& group_name, int64_t num_experts,
                   int64_t max_tokens_per_rank, int64_t max_recv_tokens_per_rank,
                   int64_t hidden_dim, int64_t max_num_sms, pybind11::object max_token_dtype,
                   bool zero_copy, int64_t num_topk, bool drop_on_overflow,
                   int64_t dispatch_fwd_num_sms, int64_t combine_fwd_num_sms,
                   int64_t dispatch_bwd_num_sms, int64_t combine_bwd_num_sms) {
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
      .dispatch_fwd_num_sms = static_cast<int>(dispatch_fwd_num_sms),
      .combine_fwd_num_sms = static_cast<int>(combine_fwd_num_sms),
      .dispatch_bwd_num_sms = static_cast<int>(dispatch_bwd_num_sms),
      .combine_bwd_num_sms = static_cast<int>(combine_bwd_num_sms),
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

NVTEEpLayerConfig make_layer_cfg(int64_t top_k, int64_t dispatch_output_per_expert_alignment,
                                 DType topk_idx_dtype = DType::kInt32) {
  return NVTEEpLayerConfig{
      .struct_size = sizeof(NVTEEpLayerConfig),
      .top_k = static_cast<int>(top_k),
      .dispatch_output_per_expert_alignment =
          static_cast<size_t>(dispatch_output_per_expert_alignment),
      .topk_idx_dtype = static_cast<NVTEDType>(topk_idx_dtype),
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
  auto idx_dtype = get_contiguous_topk_idx_te_dtype(topk_idx);
  // NCCL EP requires all prepare int output counters to share a dtype: int64, or int32 for the
  // overlapped GEMMs.
  NVTE_CHECK(
      tokens_per_expert.scalar_type() == at::kLong || tokens_per_expert.scalar_type() == at::kInt,
      "tokens_per_expert must be int64 or int32");
  NVTE_CHECK(total_recv_tokens.scalar_type() == tokens_per_expert.scalar_type(),
             "total_recv_tokens must have the dtype of tokens_per_expert");
  const DType count_dtype = GetTransformerEngineDType(tokens_per_expert.scalar_type());
  const size_t T_flat = topk_idx.numel() / topk_idx.size(-1);
  const size_t topk_n = static_cast<size_t>(topk_idx.size(-1));

  auto topk_idx_te =
      makeTransformerEngineTensor(topk_idx.data_ptr(), Shape{T_flat, topk_n}, idx_dtype);
  auto tokens_per_expert_te = makeTransformerEngineTensor(
      tokens_per_expert.data_ptr(), Shape{static_cast<size_t>(tokens_per_expert.numel())},
      count_dtype);
  auto handle_mem_te = makeTransformerEngineTensor(
      handle_mem.data_ptr(), Shape{static_cast<size_t>(handle_mem.numel())}, DType::kByte);
  // [1] scalar recv-slot total; lets the caller size dispatch outputs
  // (eager) or detect overflow past recv_capacity_per_rank (graph mode).
  auto total_recv_tokens_te = makeTransformerEngineTensor(
      total_recv_tokens.data_ptr(), Shape{static_cast<size_t>(total_recv_tokens.numel())},
      count_dtype);

  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment, idx_dtype);
  nvte_ep_prepare(handle_mem_te.data(), topk_idx_te.data(), tokens_per_expert_te.data(),
                  total_recv_tokens_te.data(), &layer_cfg, stream);
}

// tokens_scale_inv / recv_scale_inv are set only for block-scaled dispatch (for
// now MXFP8): tokens/recv_tokens carry e4m3 data and the scale tensors carry the
// unswizzled e8m0 scale-inverses [T, H/block]. Both null => bf16/fp16/fp32.
void ep_dispatch(at::Tensor handle_mem, at::Tensor topk_idx, at::Tensor tokens,
                 at::Tensor topk_weights, at::Tensor recv_tokens, at::Tensor recv_topk_weights,
                 int64_t top_k, int64_t dispatch_output_per_expert_alignment,
                 std::optional<at::Tensor> tokens_scale_inv,
                 std::optional<at::Tensor> recv_scale_inv) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(tokens.dim() >= 2, "tokens must be at least 2D [..., H]");
  NVTE_CHECK(topk_idx.dim() >= 2, "topk_idx must be at least 2D [..., top_k]");
  NVTE_CHECK(topk_weights.dim() >= 2, "topk_weights must be at least 2D [..., top_k]");
  NVTE_CHECK(recv_tokens.dim() >= 2, "recv_tokens must be at least 2D [..., recv_pr, H]");
  auto idx_dtype = get_contiguous_topk_idx_te_dtype(topk_idx);
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

  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment, idx_dtype);
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
  nvte_ep_dispatch_v2(handle_mem_te.data(), topk_idx_te.data(), tokens_te.data(), tokens_win,
                      topk_w_te.data(), topk_w_win, recv_tokens_te.data(), recv_tokens_win,
                      recv_topk_w_te.data(), recv_topk_w_win, &layer_cfg, stream);
}

namespace {

// Non-eager path with caller-supplied recv buffers: routes through the fused NCCL count-mode
// dispatch (nvte_ep_prepare_and_dispatch) instead of a separate ep_prepare + ep_dispatch.
// tokens_per_expert is derived from the dispatch's own scan rather than a prior AllGather
// round trip; recv buffers are caller-owned (static capacity), so no allocation or host read.
// Received row order within an expert's block can differ from the AllGather-based path.
void ep_prepare_and_dispatch_fused(at::Tensor handle_mem, at::Tensor topk_idx, at::Tensor tokens,
                                   at::Tensor topk_weights, at::Tensor recv_tokens,
                                   at::Tensor recv_topk_weights, at::Tensor tokens_per_expert,
                                   at::Tensor total_recv_tokens, int64_t top_k,
                                   int64_t dispatch_output_per_expert_alignment,
                                   std::optional<at::Tensor> tokens_scale_inv,
                                   std::optional<at::Tensor> recv_scale_inv) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(tokens.dim() >= 2, "tokens must be at least 2D [..., H]");
  NVTE_CHECK(topk_idx.dim() >= 2, "topk_idx must be at least 2D [..., top_k]");
  NVTE_CHECK(topk_weights.dim() >= 2, "topk_weights must be at least 2D [..., top_k]");
  NVTE_CHECK(recv_tokens.dim() >= 2, "recv_tokens must be at least 2D [..., recv_pr, H]");
  auto idx_dtype = get_contiguous_topk_idx_te_dtype(topk_idx);
  NVTE_CHECK(tokens.is_contiguous(), "tokens must be contiguous");
  NVTE_CHECK(topk_weights.is_contiguous(), "topk_weights must be contiguous");
  NVTE_CHECK(topk_weights.scalar_type() == at::kFloat, "topk_weights must be float32");
  NVTE_CHECK(recv_tokens.is_contiguous(), "recv_tokens must be contiguous");
  NVTE_CHECK(recv_topk_weights.is_contiguous(), "recv_topk_weights must be contiguous");
  NVTE_CHECK(recv_topk_weights.scalar_type() == at::kFloat, "recv_topk_weights must be float32");
  NVTE_CHECK(tokens_per_expert.is_contiguous(), "tokens_per_expert must be contiguous");
  NVTE_CHECK(
      tokens_per_expert.scalar_type() == at::kInt || tokens_per_expert.scalar_type() == at::kLong,
      "tokens_per_expert must be int32 or int64");
  NVTE_CHECK(total_recv_tokens.is_contiguous(), "total_recv_tokens must be contiguous");
  NVTE_CHECK(
      total_recv_tokens.scalar_type() == at::kInt || total_recv_tokens.scalar_type() == at::kLong,
      "total_recv_tokens must be int32 or int64");

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
  auto tokens_per_expert_te = makeTransformerEngineTensor(
      tokens_per_expert.data_ptr(), Shape{static_cast<size_t>(tokens_per_expert.numel())},
      GetTransformerEngineDType(tokens_per_expert.scalar_type()));
  auto total_recv_tokens_te = makeTransformerEngineTensor(
      total_recv_tokens.data_ptr(), Shape{static_cast<size_t>(total_recv_tokens.numel())},
      GetTransformerEngineDType(total_recv_tokens.scalar_type()));

  NVTECommWindow tokens_win = maybe_make_window(tokens);
  NVTECommWindow topk_w_win = maybe_make_window(topk_weights);
  NVTECommWindow recv_tokens_win = maybe_make_window(recv_tokens);
  NVTECommWindow recv_topk_w_win = maybe_make_window(recv_topk_weights);
  if (is_scaled) {
    const NVTECommWindow tsi_win = maybe_make_window(*tokens_scale_inv);
    const NVTECommWindow rsi_win = maybe_make_window(*recv_scale_inv);
    tokens_win.scale_window = tsi_win.window;
    tokens_win.scale_offset = tsi_win.offset;
    recv_tokens_win.scale_window = rsi_win.window;
    recv_tokens_win.scale_offset = rsi_win.offset;
  }
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment, idx_dtype);
  nvte_ep_prepare_and_dispatch(
      handle_mem_te.data(), topk_idx_te.data(), tokens_te.data(), tokens_win, topk_w_te.data(),
      topk_w_win, recv_tokens_te.data(), recv_tokens_win, recv_topk_w_te.data(), recv_topk_w_win,
      tokens_per_expert_te.data(), total_recv_tokens_te.data(), &layer_cfg, stream);
}

}  // namespace

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

  // Opt-in: the fused NCCL count-mode dispatch replaces the AllGather-based prepare + dispatch.
  // Off by default; the standard path stays unfused.
  static const bool fused_prepare_dispatch = [] {
    const char* env = std::getenv("NVTE_EP_FUSED_PREPARE_DISPATCH");
    const bool enabled = env != nullptr && env[0] == '1';
    if (enabled) {
      TORCH_WARN_ONCE(
          "NVTE_EP_FUSED_PREPARE_DISPATCH is set: fused prepare+dispatch only supports EP within a "
          "single NVLink domain.");
    }
    return enabled;
  }();

  if (fused_prepare_dispatch && !caller_omitted_recv) {
    ep_prepare_and_dispatch_fused(handle_mem, topk_idx, tokens, topk_weights, *recv_tokens,
                                  *recv_topk_weights, tokens_per_expert, total_recv_tokens, top_k,
                                  dispatch_output_per_expert_alignment, tokens_scale_inv,
                                  recv_scale_inv);
    return {};
  }

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

  ep_dispatch(handle_mem, topk_idx, tokens, topk_weights, rt, rw, top_k,
              dispatch_output_per_expert_alignment, tokens_scale_inv,
              is_scaled ? std::optional<at::Tensor>(rsi) : std::nullopt);
  if (!caller_omitted_recv) return {};  // caller buffers were mutated in place
  if (is_scaled) return {rt, rw, rsi};
  return {rt, rw};
}

void ep_combine(at::Tensor handle_mem, at::Tensor expert_out, at::Tensor result, int64_t top_k,
                int64_t dispatch_output_per_expert_alignment, at::ScalarType topk_idx_dtype) {
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
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment,
                                  get_topk_idx_te_dtype(topk_idx_dtype));
  nvte_ep_combine_v2(handle_mem_te.data(), expert_out_te.data(), expert_out_win, result_te.data(),
                     &layer_cfg, stream);
}

void ep_dispatch_bwd(at::Tensor handle_mem, at::Tensor grad, at::Tensor g_recv_topk_weights,
                     at::Tensor grad_tokens, at::Tensor grad_topk_weights, int64_t top_k,
                     int64_t dispatch_output_per_expert_alignment, at::ScalarType topk_idx_dtype) {
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
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment,
                                  get_topk_idx_te_dtype(topk_idx_dtype));
  nvte_ep_dispatch_bwd_v2(handle_mem_te.data(), grad_te.data(), grad_win, g_recv_w_te.data(),
                          g_recv_w_win, grad_tokens_te.data(), grad_topk_w_te.data(), &layer_cfg,
                          stream);
}

void ep_combine_bwd(at::Tensor handle_mem, at::Tensor grad, at::Tensor grad_expert_out,
                    int64_t top_k, int64_t dispatch_output_per_expert_alignment,
                    at::ScalarType topk_idx_dtype, std::optional<at::Tensor> grad_scale_inv,
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
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment,
                                  get_topk_idx_te_dtype(topk_idx_dtype));
  nvte_ep_combine_bwd_v2(handle_mem_te.data(), grad_te.data(), grad_win, grad_expert_out_te.data(),
                         grad_expert_out_win, &layer_cfg, stream);
}

std::vector<at::Tensor> ep_dispatch_gemm_fwd(
    at::Tensor handle_mem, at::Tensor topk_idx, at::Tensor tokens, at::Tensor topk_weights,
    at::Tensor weight, std::optional<at::Tensor> recv_tokens_opt,
    std::optional<at::Tensor> recv_topk_weights_opt, at::Tensor tokens_per_expert,
    std::optional<at::Tensor> gemm_out_opt, std::optional<at::Tensor> act_out_opt, int64_t top_k,
    int64_t dispatch_output_per_expert_alignment, std::optional<at::Tensor> total_recv_tokens,
    int64_t eager_row_bucket, std::optional<at::Tensor> tokens_scale_inv,
    std::optional<at::Tensor> recv_scale_inv, std::optional<at::Tensor> weight_scale_inv,
    std::optional<at::Tensor> act_out_scale_inv, bool symm_input) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  const bool eager = !recv_tokens_opt.has_value();
  NVTE_CHECK(recv_topk_weights_opt.has_value() != eager && gemm_out_opt.has_value() != eager &&
                 act_out_opt.has_value() != eager,
             "pass all of the recv and GEMM outputs, or none (eager)");
  const bool mxfp8 = tokens_scale_inv.has_value();
  NVTE_CHECK(weight_scale_inv.has_value() == mxfp8 &&
                 recv_scale_inv.has_value() == (mxfp8 && !eager) &&
                 act_out_scale_inv.has_value() == (mxfp8 && !eager),
             "pass all of the MXFP8 scales, or none; eager allocates the output scales");
  NVTE_CHECK(!total_recv_tokens.has_value() || total_recv_tokens->scalar_type() == at::kInt,
             "total_recv_tokens must be int32");
  at::Tensor recv_tokens, recv_topk_weights, gemm_out, act_out;
  if (eager) {
    // Prepare, read the recv total on the host and size the outputs to whole row buckets, all
    // in this call; the fused entry then skips the prepare.
    NVTE_CHECK(total_recv_tokens.has_value() && total_recv_tokens->is_pinned(),
               "eager needs total_recv_tokens in pinned host memory");
    NVTE_CHECK(eager_row_bucket > 0, "eager needs eager_row_bucket > 0");
    ep_prepare(handle_mem, topk_idx, tokens_per_expert, top_k, dispatch_output_per_expert_alignment,
               *total_recv_tokens);
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    const int64_t total = *total_recv_tokens->data_ptr<int32_t>();
    const int64_t buckets = std::max<int64_t>((total + eager_row_bucket - 1) / eager_row_bucket, 1);
    const int64_t rows = buckets * eager_row_bucket;
    const int64_t hidden = tokens.size(1);
    const int64_t ffn = weight.size(1) / 2;
    recv_tokens = at::empty({rows, hidden}, tokens.options());
    recv_topk_weights = at::empty({rows}, topk_weights.options());
    gemm_out = at::empty({rows, 2 * ffn}, tokens.options().dtype(at::kBFloat16));
    act_out = at::empty({rows, ffn}, tokens.options());
    if (mxfp8) {
      const int64_t scale_cols = tokens_scale_inv->size(-1);
      NVTE_CHECK(scale_cols > 0 && hidden % scale_cols == 0, "eager MXFP8 tokens_scale_inv cols (",
                 scale_cols, ") must be a non-zero divisor of hidden (", hidden, ")");
      const int64_t block = hidden / scale_cols;
      recv_scale_inv = at::empty({rows, scale_cols}, tokens_scale_inv->options());
      act_out_scale_inv = at::empty({rows * ffn / block}, tokens_scale_inv->options());
    }
  } else {
    recv_tokens = *recv_tokens_opt;
    recv_topk_weights = *recv_topk_weights_opt;
    gemm_out = *gemm_out_opt;
    act_out = *act_out_opt;
  }
  NVTE_CHECK(tokens.is_contiguous() && recv_tokens.is_contiguous() && weight.is_contiguous() &&
                 gemm_out.is_contiguous() && act_out.is_contiguous(),
             "operands must be contiguous");
  check_dense_cuda(tokens_per_expert, "tokens_per_expert");
  check_dense_cuda(recv_topk_weights, "recv_topk_weights");
  // The expert layout the GEMMs read: expert 0's start (0), then each expert's padded row end.
  // Front-pad by 3 so the row ends (expert_token_offsets[1:]) start 16-byte aligned, as the GEMM
  // needs.
  auto expert_token_offsets =
      at::empty({weight.size(0) + 4}, tokens.options().dtype(at::kInt)).slice(0, 3);
  auto handle_mem_te = makeTransformerEngineTensor(handle_mem);
  auto topk_idx_te = makeTransformerEngineTensor(topk_idx);
  auto tokens_te = make_maybe_mxfp8_tensor(tokens, tokens_scale_inv, false);
  auto topk_weights_te = makeTransformerEngineTensor(topk_weights);
  auto weight_te = make_maybe_mxfp8_tensor(weight, weight_scale_inv, true);
  auto recv_tokens_te = make_maybe_mxfp8_tensor(recv_tokens, recv_scale_inv, false);
  auto recv_topk_weights_te = makeTransformerEngineTensor(recv_topk_weights);
  auto tokens_per_expert_te = makeTransformerEngineTensor(tokens_per_expert);
  auto gemm_out_te = makeTransformerEngineTensor(gemm_out);
  auto act_out_te = make_maybe_mxfp8_tensor(act_out, act_out_scale_inv, true);
  auto expert_token_offsets_te = makeTransformerEngineTensor(expert_token_offsets);
  // In eager mode the prepare above wrote the total.
  const bool report_total = total_recv_tokens.has_value() && !eager;
  TensorWrapper total_recv_tokens_te;
  if (report_total) {
    total_recv_tokens_te = makeTransformerEngineTensor(*total_recv_tokens);
  }
  const NVTECommWindow tokens_win =
      symm_input ? make_input_window(tokens, tokens_scale_inv) : kNoWindow;
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment,
                                  get_contiguous_topk_idx_te_dtype(topk_idx));
  nvte_ep_dispatch_gemm_fwd(handle_mem_te.data(), topk_idx_te.data(), tokens_te.data(), tokens_win,
                            topk_weights_te.data(), weight_te.data(), recv_tokens_te.data(),
                            recv_topk_weights_te.data(), gemm_out_te.data(), act_out_te.data(),
                            tokens_per_expert_te.data(), expert_token_offsets_te.data(),
                            report_total ? total_recv_tokens_te.data() : nullptr, &layer_cfg,
                            stream);
  if (mxfp8) {
    return {recv_tokens,          recv_topk_weights, gemm_out,          act_out,
            expert_token_offsets, *recv_scale_inv,   *act_out_scale_inv};
  }
  return {recv_tokens, recv_topk_weights, gemm_out, act_out, expert_token_offsets};
}

void ep_combine_gemm_fwd(at::Tensor handle_mem, at::Tensor gemm_in, at::Tensor expert_token_offsets,
                         at::Tensor weight, at::Tensor expert_out, at::Tensor result, int64_t top_k,
                         int64_t dispatch_output_per_expert_alignment,
                         at::ScalarType topk_idx_dtype, std::optional<at::Tensor> recv_topk_weights,
                         std::optional<at::Tensor> gemm_in_scale_inv,
                         std::optional<at::Tensor> weight_scale_inv) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(gemm_in.is_contiguous() && weight.is_contiguous() && expert_out.is_contiguous() &&
                 result.is_contiguous(),
             "operands must be contiguous");
  check_dense_cuda(expert_token_offsets, "expert_token_offsets");
  NVTE_CHECK(gemm_in_scale_inv.has_value() == weight_scale_inv.has_value(),
             "pass both MXFP8 scales, or none");
  auto handle_mem_te = makeTransformerEngineTensor(handle_mem);
  auto gemm_in_te = make_maybe_mxfp8_tensor(gemm_in, gemm_in_scale_inv, true);
  TensorWrapper recv_topk_weights_te;
  if (recv_topk_weights.has_value()) {
    recv_topk_weights_te = makeTransformerEngineTensor(*recv_topk_weights);
  }
  auto expert_token_offsets_te = makeTransformerEngineTensor(expert_token_offsets);
  auto weight_te = make_maybe_mxfp8_tensor(weight, weight_scale_inv, true);
  auto expert_out_te = makeTransformerEngineTensor(expert_out);
  auto result_te = makeTransformerEngineTensor(result);
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment,
                                  get_topk_idx_te_dtype(topk_idx_dtype));
  nvte_ep_combine_gemm_fwd(handle_mem_te.data(), gemm_in_te.data(),
                           recv_topk_weights.has_value() ? recv_topk_weights_te.data() : nullptr,
                           expert_token_offsets_te.data(), weight_te.data(), expert_out_te.data(),
                           result_te.data(), &layer_cfg, stream);
}

void ep_combine_gemm_bwd(
    at::Tensor handle_mem, at::Tensor grad_result, at::Tensor gemm_out,
    at::Tensor recv_topk_weights, at::Tensor tokens_per_expert, at::Tensor weight,
    at::Tensor grad_expert_out, at::Tensor grad_gemm_out, at::Tensor grad_recv_topk_weights,
    int64_t top_k, int64_t dispatch_output_per_expert_alignment, at::ScalarType topk_idx_dtype,
    std::optional<at::Tensor> grad_result_scale_inv,
    std::optional<at::Tensor> grad_expert_out_scale_inv, std::optional<at::Tensor> weight_scale_inv,
    std::optional<at::Tensor> grad_gemm_out_scale_inv,
    std::optional<at::Tensor> grad_gemm_out_columnwise,
    std::optional<at::Tensor> grad_gemm_out_columnwise_scale_inv, std::optional<at::Tensor> act_out,
    std::optional<at::Tensor> act_out_scale_inv, std::optional<at::Tensor> wgrad_input,
    std::optional<at::Tensor> weight_grad, bool accumulate_weight_grad,
    std::optional<at::Tensor> grad_expert_out_columnwise,
    std::optional<at::Tensor> grad_expert_out_columnwise_scale_inv,
    std::optional<at::Tensor> wgrad_input_scale_inv, bool symm_input) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(grad_result.is_contiguous() && gemm_out.is_contiguous() && weight.is_contiguous() &&
                 grad_expert_out.is_contiguous() && grad_gemm_out.is_contiguous(),
             "operands must be contiguous");
  check_dense_cuda(tokens_per_expert, "tokens_per_expert");
  check_dense_cuda(grad_recv_topk_weights, "grad_recv_topk_weights");
  const bool mxfp8 = grad_result_scale_inv.has_value();
  NVTE_CHECK(grad_expert_out_scale_inv.has_value() == mxfp8 &&
                 weight_scale_inv.has_value() == mxfp8 &&
                 grad_gemm_out_scale_inv.has_value() == mxfp8 &&
                 grad_gemm_out_columnwise.has_value() == mxfp8 &&
                 grad_gemm_out_columnwise_scale_inv.has_value() == mxfp8 &&
                 act_out.has_value() == mxfp8 && act_out_scale_inv.has_value() == mxfp8,
             "pass all of the MXFP8 operands, or none");
  // The columnwise grad_expert_out of the MXFP8 weight gradient, written by the call.
  const bool grad_expert_out_columns = grad_expert_out_columnwise.has_value();
  NVTE_CHECK(grad_expert_out_columnwise_scale_inv.has_value() == grad_expert_out_columns &&
                 (mxfp8 || !grad_expert_out_columns),
             "pass the MXFP8 columnwise grad_expert_out with its scales, or neither");
  auto handle_mem_te = makeTransformerEngineTensor(handle_mem);
  auto grad_result_te = make_maybe_mxfp8_tensor(grad_result, grad_result_scale_inv, false);
  auto gemm_out_te = makeTransformerEngineTensor(gemm_out);
  auto recv_topk_weights_te = makeTransformerEngineTensor(recv_topk_weights);
  auto tokens_per_expert_te = makeTransformerEngineTensor(tokens_per_expert);
  // MXFP8 reads the forward weight columnwise.
  auto weight_te = mxfp8 ? make_columnwise_mxfp8_tensor(weight, *weight_scale_inv)
                         : makeTransformerEngineTensor(weight);
  auto grad_expert_out_te =
      grad_expert_out_columns
          ? make_mxfp8_tensor(grad_expert_out, *grad_expert_out_scale_inv,
                              *grad_expert_out_columnwise, *grad_expert_out_columnwise_scale_inv,
                              false)
          : make_maybe_mxfp8_tensor(grad_expert_out, grad_expert_out_scale_inv, false);
  auto grad_gemm_out_te =
      mxfp8 ? make_mxfp8_tensor(grad_gemm_out, *grad_gemm_out_scale_inv, *grad_gemm_out_columnwise,
                                *grad_gemm_out_columnwise_scale_inv, true)
            : makeTransformerEngineTensor(grad_gemm_out);
  auto grad_recv_topk_weights_te = makeTransformerEngineTensor(grad_recv_topk_weights);
  TensorWrapper act_out_te;
  if (mxfp8) {
    act_out_te = make_columnwise_mxfp8_tensor(*act_out, *act_out_scale_inv);
  }
  NVTE_CHECK(mxfp8 || !wgrad_input_scale_inv.has_value(),
             "wgrad_input_scale_inv is for MXFP8 only");
  WeightGradTensors wgrad(wgrad_input, weight_grad, wgrad_input_scale_inv);
  const NVTECommWindow grad_result_win =
      symm_input ? make_input_window(grad_result, grad_result_scale_inv) : kNoWindow;
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment,
                                  get_topk_idx_te_dtype(topk_idx_dtype));
  nvte_ep_combine_gemm_bwd(handle_mem_te.data(), grad_result_te.data(), grad_result_win,
                           gemm_out_te.data(), recv_topk_weights_te.data(),
                           tokens_per_expert_te.data(), weight_te.data(), wgrad.input(),
                           grad_expert_out_te.data(), grad_gemm_out_te.data(),
                           grad_recv_topk_weights_te.data(), mxfp8 ? act_out_te.data() : nullptr,
                           wgrad.output(), accumulate_weight_grad, &layer_cfg, stream);
}

void ep_dispatch_gemm_bwd(
    at::Tensor handle_mem, at::Tensor grad_gemm_out, at::Tensor grad_recv_topk_weights,
    at::Tensor tokens_per_expert, at::Tensor weight, at::Tensor grad_recv_tokens,
    at::Tensor grad_tokens, at::Tensor grad_topk_weights, int64_t top_k,
    int64_t dispatch_output_per_expert_alignment, at::ScalarType topk_idx_dtype,
    std::optional<at::Tensor> grad_gemm_out_scale_inv, std::optional<at::Tensor> weight_scale_inv,
    std::optional<at::Tensor> grad_gemm_out_columnwise,
    std::optional<at::Tensor> grad_gemm_out_columnwise_scale_inv,
    std::optional<at::Tensor> wgrad_input, std::optional<at::Tensor> wgrad_input_scale_inv,
    std::optional<at::Tensor> weight_grad, bool accumulate_weight_grad,
    std::optional<at::Tensor> wgrad_input_rowwise,
    std::optional<at::Tensor> wgrad_input_rowwise_scale_inv) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  NVTE_CHECK(
      grad_gemm_out.is_contiguous() && weight.is_contiguous() && grad_recv_tokens.is_contiguous(),
      "operands must be contiguous");
  check_dense_cuda(tokens_per_expert, "tokens_per_expert");
  check_dense_cuda(grad_recv_topk_weights, "grad_recv_topk_weights");
  const bool mxfp8 = grad_gemm_out_scale_inv.has_value();
  NVTE_CHECK(weight_scale_inv.has_value() == mxfp8, "pass both MXFP8 scales, or none");
  // The MXFP8 weight gradient reads grad_gemm_out and wgrad_input columnwise.
  const bool columnwise = grad_gemm_out_columnwise.has_value();
  NVTE_CHECK(grad_gemm_out_columnwise_scale_inv.has_value() == columnwise &&
                 wgrad_input_scale_inv.has_value() == columnwise && (mxfp8 || !columnwise),
             "pass the MXFP8 columnwise operands of the weight gradient together, or none");
  auto handle_mem_te = makeTransformerEngineTensor(handle_mem);
  auto grad_gemm_out_te =
      columnwise
          ? make_mxfp8_tensor(grad_gemm_out, *grad_gemm_out_scale_inv, *grad_gemm_out_columnwise,
                              *grad_gemm_out_columnwise_scale_inv, true)
          : make_maybe_mxfp8_tensor(grad_gemm_out, grad_gemm_out_scale_inv, true);
  auto grad_recv_topk_weights_te = makeTransformerEngineTensor(grad_recv_topk_weights);
  auto tokens_per_expert_te = makeTransformerEngineTensor(tokens_per_expert);
  // MXFP8 reads the forward weight columnwise.
  auto weight_te = mxfp8 ? make_columnwise_mxfp8_tensor(weight, *weight_scale_inv)
                         : makeTransformerEngineTensor(weight);
  auto grad_recv_tokens_te = makeTransformerEngineTensor(grad_recv_tokens);
  auto grad_tokens_te = makeTransformerEngineTensor(grad_tokens);
  auto grad_topk_weights_te = makeTransformerEngineTensor(grad_topk_weights);
  WeightGradTensors wgrad(wgrad_input, weight_grad, wgrad_input_scale_inv, wgrad_input_rowwise,
                          wgrad_input_rowwise_scale_inv);
  auto layer_cfg = make_layer_cfg(top_k, dispatch_output_per_expert_alignment,
                                  get_topk_idx_te_dtype(topk_idx_dtype));
  nvte_ep_dispatch_gemm_bwd(handle_mem_te.data(), grad_gemm_out_te.data(),
                            grad_recv_topk_weights_te.data(), tokens_per_expert_te.data(),
                            weight_te.data(), wgrad.input(), grad_recv_tokens_te.data(),
                            grad_tokens_te.data(), grad_topk_weights_te.data(), wgrad.output(),
                            accumulate_weight_grad, &layer_cfg, stream);
}

void register_ep_bindings(pybind11::module_& m) {
  namespace py = pybind11;
  m.def("ep_initialize", &ep_initialize,
        "Initialize the EP backend; borrows torch's NCCL comm pointed to by ``comm_ptr``.",
        py::arg("comm_ptr"), py::arg("group_name"), py::arg("num_experts"),
        py::arg("max_tokens_per_rank"), py::arg("max_recv_tokens_per_rank"), py::arg("hidden_dim"),
        py::arg("max_num_sms") = 0, py::arg("max_token_dtype"), py::arg("zero_copy") = false,
        py::arg("num_topk") = 0, py::arg("drop_on_overflow") = false,
        py::arg("dispatch_fwd_num_sms") = 0, py::arg("combine_fwd_num_sms") = 0,
        py::arg("dispatch_bwd_num_sms") = 0, py::arg("combine_bwd_num_sms") = 0);
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
        py::arg("recv_topk_weights"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"), py::arg("tokens_scale_inv") = std::nullopt,
        py::arg("recv_scale_inv") = std::nullopt, py::call_guard<py::gil_scoped_release>());
  m.def("ep_prepare_and_dispatch", &ep_prepare_and_dispatch, "Fused EP prepare + dispatch",
        py::arg("handle_mem"), py::arg("topk_idx"), py::arg("tokens"), py::arg("topk_weights"),
        py::arg("tokens_per_expert"), py::arg("total_recv_tokens"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"), py::arg("recv_tokens") = std::nullopt,
        py::arg("recv_topk_weights") = std::nullopt, py::arg("recv_scale_inv") = std::nullopt,
        py::arg("tokens_scale_inv") = std::nullopt, py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine", &ep_combine, "EP combine", py::arg("handle_mem"), py::arg("expert_out"),
        py::arg("result"), py::arg("top_k"), py::arg("dispatch_output_per_expert_alignment"),
        py::arg("topk_idx_dtype"), py::call_guard<py::gil_scoped_release>());
  m.def("ep_dispatch_gemm_fwd", &ep_dispatch_gemm_fwd,
        "EP dispatch overlapped with the first GEMM. Returns recv_tokens, recv_topk_weights, "
        "gemm_out, act_out and expert_token_offsets, then the recv and act_out scales when "
        "MXFP8; in eager mode omit them and they are sized and allocated here",
        py::arg("handle_mem"), py::arg("topk_idx"), py::arg("tokens"), py::arg("topk_weights"),
        py::arg("weight"), py::arg("recv_tokens"), py::arg("recv_topk_weights"),
        py::arg("tokens_per_expert"), py::arg("gemm_out"), py::arg("act_out"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"),
        py::arg("total_recv_tokens") = std::nullopt, py::arg("eager_row_bucket") = 0,
        py::arg("tokens_scale_inv") = std::nullopt, py::arg("recv_scale_inv") = std::nullopt,
        py::arg("weight_scale_inv") = std::nullopt, py::arg("act_out_scale_inv") = std::nullopt,
        py::arg("symm_input") = false, py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine_gemm_fwd", &ep_combine_gemm_fwd, "EP second GEMM overlapped with the combine",
        py::arg("handle_mem"), py::arg("gemm_in"), py::arg("expert_token_offsets"),
        py::arg("weight"), py::arg("expert_out"), py::arg("result"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"), py::arg("topk_idx_dtype"),
        py::arg("recv_topk_weights") = std::nullopt, py::arg("gemm_in_scale_inv") = std::nullopt,
        py::arg("weight_scale_inv") = std::nullopt, py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine_gemm_bwd", &ep_combine_gemm_bwd,
        "Backward of the second GEMM and the combine, with the activation backward",
        py::arg("handle_mem"), py::arg("grad_result"), py::arg("gemm_out"),
        py::arg("recv_topk_weights"), py::arg("tokens_per_expert"), py::arg("weight"),
        py::arg("grad_expert_out"), py::arg("grad_gemm_out"), py::arg("grad_recv_topk_weights"),
        py::arg("top_k"), py::arg("dispatch_output_per_expert_alignment"),
        py::arg("topk_idx_dtype"), py::arg("grad_result_scale_inv") = std::nullopt,
        py::arg("grad_expert_out_scale_inv") = std::nullopt,
        py::arg("weight_scale_inv") = std::nullopt,
        py::arg("grad_gemm_out_scale_inv") = std::nullopt,
        py::arg("grad_gemm_out_columnwise") = std::nullopt,
        py::arg("grad_gemm_out_columnwise_scale_inv") = std::nullopt,
        py::arg("act_out") = std::nullopt, py::arg("act_out_scale_inv") = std::nullopt,
        py::arg("wgrad_input") = std::nullopt, py::arg("weight_grad") = std::nullopt,
        py::arg("accumulate_weight_grad") = false,
        py::arg("grad_expert_out_columnwise") = std::nullopt,
        py::arg("grad_expert_out_columnwise_scale_inv") = std::nullopt,
        py::arg("wgrad_input_scale_inv") = std::nullopt, py::arg("symm_input") = false,
        py::call_guard<py::gil_scoped_release>());
  m.def(
      "ep_dispatch_gemm_bwd", &ep_dispatch_gemm_bwd, "Backward of the first GEMM and the dispatch",
      py::arg("handle_mem"), py::arg("grad_gemm_out"), py::arg("grad_recv_topk_weights"),
      py::arg("tokens_per_expert"), py::arg("weight"), py::arg("grad_recv_tokens"),
      py::arg("grad_tokens"), py::arg("grad_topk_weights"), py::arg("top_k"),
      py::arg("dispatch_output_per_expert_alignment"), py::arg("topk_idx_dtype"),
      py::arg("grad_gemm_out_scale_inv") = std::nullopt, py::arg("weight_scale_inv") = std::nullopt,
      py::arg("grad_gemm_out_columnwise") = std::nullopt,
      py::arg("grad_gemm_out_columnwise_scale_inv") = std::nullopt,
      py::arg("wgrad_input") = std::nullopt, py::arg("wgrad_input_scale_inv") = std::nullopt,
      py::arg("weight_grad") = std::nullopt, py::arg("accumulate_weight_grad") = false,
      py::arg("wgrad_input_rowwise") = std::nullopt,
      py::arg("wgrad_input_rowwise_scale_inv") = std::nullopt,
      py::call_guard<py::gil_scoped_release>());
  m.def("ep_dispatch_bwd", &ep_dispatch_bwd, "EP dispatch backward", py::arg("handle_mem"),
        py::arg("grad"), py::arg("g_recv_topk_weights"), py::arg("grad_tokens"),
        py::arg("grad_topk_weights"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"), py::arg("topk_idx_dtype"),
        py::call_guard<py::gil_scoped_release>());
  m.def("ep_combine_bwd", &ep_combine_bwd, "EP combine backward", py::arg("handle_mem"),
        py::arg("grad"), py::arg("grad_expert_out"), py::arg("top_k"),
        py::arg("dispatch_output_per_expert_alignment"), py::arg("topk_idx_dtype"),
        py::arg("grad_scale_inv") = std::nullopt,
        py::arg("grad_expert_out_scale_inv") = std::nullopt,
        py::call_guard<py::gil_scoped_release>());
}

}  // namespace transformer_engine::pytorch

#endif  // NVTE_WITH_NCCL_EP
