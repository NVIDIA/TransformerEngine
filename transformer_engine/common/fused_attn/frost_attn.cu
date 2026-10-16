/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "../tvm_ffi_bridge.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "frost_attn.h"

namespace transformer_engine {
namespace fused_attn {
namespace {

struct Operand {
  int role;
  size_t bytes;
  size_t alignment;
};

struct Plan {
  tvm::ffi::Function function;
  size_t workspace_bytes;
  bool length_form;
  std::vector<int64_t> prefix;
  std::vector<Operand> operands;
};

bool eligible(const FusedAttnConfig &c) {
  if (cuda::sm_arch(cuda::current_device()) != 100 || c.deterministic || !c.is_training ||
      c.qkv_dtype != kNVTEBFloat16 || c.o_dtype != kNVTEBFloat16 || c.do_dtype != kNVTEBFloat16 ||
      c.dqkv_dtype != kNVTEBFloat16 || c.head_dim_qk != c.head_dim_v ||
      (c.head_dim_qk != 128 && c.head_dim_qk != 256) || c.bias_type != NVTE_NO_BIAS ||
      c.softmax_type != NVTE_VANILLA_SOFTMAX || c.dropout != 0 || c.is_paged_kv ||
      c.bottom_right_diagonal || c.window_size_left != -1 ||
      (c.window_size_right != -1 && c.window_size_right != 0) ||
      (c.window_size_right == 0 && !c.is_causal) || !std::isfinite(c.attn_scale) ||
      c.attn_scale <= 0 || c.batch_size == 0 || c.batch_size > 65535 || c.num_gqa_groups == 0 ||
      c.num_attn_heads < c.num_gqa_groups || c.num_attn_heads % c.num_gqa_groups != 0 ||
      c.num_attn_heads > 65535 || c.max_seqlen_q == 0 || c.max_seqlen_kv == 0 ||
      c.max_seqlen_q > INT32_MAX - 256 || c.max_seqlen_kv > INT32_MAX - 256)
    return false;
  const auto format = c.is_ragged_q ? NVTE_THD : NVTE_BSHD;
  if (c.o_format != format || c.do_format != format || c.dqkv_layout != c.qkv_layout) return false;
  if (c.qkv_layout == NVTE_BSHD_BSHD_BSHD) {
    return !c.is_padding && c.max_seqlen_q % 128 == 0 && c.max_seqlen_kv % 128 == 0;
  }
  if (c.qkv_layout != NVTE_THD_THD_THD || !c.is_padding || !c.uses_ragged_stats ||
      c.num_tokens_q == 0 || c.num_tokens_kv == 0 || c.num_tokens_q > INT32_MAX ||
      c.num_tokens_kv > INT32_MAX)
    return false;
  if (c.head_dim_qk == 128) return true;
  // D256's current OSS engine declines ragged attention. This separate promise
  // enables an equivalent dense declaration; validate its values on EVERY call.
  return getenv<bool>("NVTE_FUSED_ATTN_FROST_UNIFORM") &&
         c.num_tokens_q == c.batch_size * c.max_seqlen_q &&
         c.num_tokens_kv == c.batch_size * c.max_seqlen_kv;
}

const Plan *get_plan(const FusedAttnConfig &c, cudaStream_t stream) {
  // Process lifetime, like TVMFFIConfigCache: graph executables can outlive Python.
  static auto *cache = new std::map<FusedAttnConfig, std::unique_ptr<Plan>>;
  static auto *mutex = new std::mutex;
  static std::atomic<uint64_t> serial{0};
  {
    std::lock_guard<std::mutex> lock(*mutex);
    auto found = cache->find(c);
    if (found != cache->end()) return found->second.get();
  }
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  NVTE_CHECK(capture == cudaStreamCaptureStatusNone,
             "Warm up cuDNN OSS attention before CUDA graph capture");
  auto &central = tvm_ffi_bridge::TVMFFICentral::getInstance();
  if (!central.get_cutedsl_backend_enabled() || !central.get_tvm_ffi_available()) return nullptr;
  auto prepare = tvm::ffi::Function::GetGlobal("get_frost_attention_backward");
  if (!prepare) return nullptr;
  const std::string key = "transformer_engine.frost.backward." + std::to_string(serial++);
  const bool thd = c.is_ragged_q && c.head_dim_qk == 128;
  // Never hold a C++ cache lock while entering Python: another Python thread
  // might hold the GIL and be waiting for that lock.
  auto result =
      (*prepare)(tvm::ffi::String(key), static_cast<int64_t>(c.batch_size),
                 static_cast<int64_t>(c.num_attn_heads), static_cast<int64_t>(c.num_gqa_groups),
                 static_cast<int64_t>(c.max_seqlen_q), static_cast<int64_t>(c.max_seqlen_kv),
                 static_cast<int64_t>(c.head_dim_qk), static_cast<int64_t>(c.num_tokens_q),
                 static_cast<int64_t>(c.num_tokens_kv), thd, c.is_causal,
                 static_cast<double>(c.attn_scale), cuda::current_device());
  // A bounded scalar string keeps the optional bridge free of TVM object-type
  // registration symbols, so loading TE with the DSL backend disabled still works.
  auto encoded = result.try_cast<tvm::ffi::String>();
  auto declined = result.try_cast<bool>();
  NVTE_CHECK(encoded || (declined && !*declined), "Invalid cuDNN OSS preparation response");
  std::unique_ptr<Plan> plan;
  if (encoded) {
    NVTE_CHECK(encoded->size() < 4096, "Oversized cuDNN OSS native plan metadata");
    std::istringstream input{std::string(*encoded)};
    std::vector<int64_t> v;
    int64_t value;
    while (input >> value) v.push_back(value);
    NVTE_CHECK(input.eof(), "Malformed cuDNN OSS native plan metadata");
    auto at = [&v](size_t i) { return v[i]; };
    NVTE_CHECK(v.size() >= 5 && at(0) == 1, "Unknown cuDNN OSS native plan metadata");
    const auto prefix_count = at(3), operand_count = at(4);
    NVTE_CHECK(at(1) > 0 && at(2) == static_cast<int64_t>(thd) && prefix_count == (thd ? 7 : 0) &&
                   operand_count >= 9 && operand_count <= 20 &&
                   v.size() == 5 + prefix_count + operand_count * 3,
               "Malformed cuDNN OSS native plan metadata");
    auto function = tvm::ffi::Function::GetGlobal(key);
    NVTE_CHECK(function.has_value(), "cuDNN OSS did not register its native function");
    plan = std::make_unique<Plan>(Plan{*function, static_cast<size_t>(at(1)), thd, {}, {}});
    size_t cursor = 5;
    for (int64_t i = 0; i < prefix_count; ++i) plan->prefix.push_back(at(cursor++));
    for (int64_t i = 0; i < operand_count; ++i) {
      const auto role = at(cursor++), bytes = at(cursor++), alignment = at(cursor++);
      NVTE_CHECK(role >= -1 && role < 20 && bytes >= 0 && alignment > 0 &&
                     (alignment & (alignment - 1)) == 0,
                 "Invalid cuDNN OSS operand metadata");
      plan->operands.push_back(
          {static_cast<int>(role), static_cast<size_t>(bytes), static_cast<size_t>(alignment)});
    }
  }
  std::lock_guard<std::mutex> lock(*mutex);
  auto inserted = cache->emplace(c, std::move(plan));
  return inserted.first->second.get();
}

bool overlaps(uintptr_t a, size_t a_bytes, uintptr_t b, size_t b_bytes) {
  NVTE_CHECK(a_bytes <= UINTPTR_MAX - a && b_bytes <= UINTPTR_MAX - b,
             "cuDNN OSS pointer range overflows");
  return a < b + b_bytes && b < a + a_bytes;
}

__global__ void validate_uniform(const int32_t *cu_q, const int32_t *cu_kv,
                                 const int32_t *origins_q, const int32_t *origins_kv, int64_t batch,
                                 int64_t sq, int64_t sk) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i <= batch && (cu_q[i] != i * sq || cu_kv[i] != i * sk || origins_q[i] != i * sq ||
                     origins_kv[i] != i * sk)) {
    printf("NVTE_FUSED_ATTN_FROST_UNIFORM: invalid sequence/offset at %d\n", static_cast<int>(i));
    asm volatile("trap;");
  }
}

__global__ void unpack_stats(const float *packed, float *dense, int64_t size, int64_t heads,
                             int64_t sequence) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    const int64_t s = i % sequence, h = (i / sequence) % heads;
    const int64_t b = i / (sequence * heads);
    dense[i] = packed[(b * sequence + s) * heads + h];
  }
}

}  // namespace

bool frost_attention_backward(const FusedAttnConfig &cfg, const FrostPointers &pointers,
                              void *workspace, size_t *workspace_size, cudaStream_t stream) {
  const int mode = getenv<int>("NVTE_FUSED_ATTN_FROST");
  if (mode == 0) return false;
  NVTE_CHECK(mode == 1 || mode == 2, "NVTE_FUSED_ATTN_FROST must be 0, 1 or 2");
  cfg.check_derived();
  FusedAttnConfig cache_cfg = cfg;
  cache_cfg.device_id = cuda::current_device();
  const Plan *plan = eligible(cfg) ? get_plan(cache_cfg, stream) : nullptr;
  if (!plan) {
    NVTE_CHECK(mode != 2,
               "Requested cuDNN OSS backward unavailable; check shape, mask, "
               "cuDNN Frontend installation and NVTE_ENABLE_CUTEDSL_BACKEND=1");
    return false;
  }
  NVTE_CHECK(cache_cfg.device_id == cuda::current_device(),
             "cuDNN OSS plan belongs to another device");
  const bool uniform = cfg.is_ragged_q && cfg.head_dim_qk == 256;
  const size_t stats_bytes =
      uniform ? cfg.batch_size * cfg.num_attn_heads * cfg.max_seqlen_q * 4 : 0;
  NVTE_CHECK(plan->workspace_bytes <= SIZE_MAX - 127 - stats_bytes, "cuDNN OSS workspace overflow");
  const size_t kernel_bytes = (plan->workspace_bytes + 127) & ~size_t(127);
  const size_t required = kernel_bytes + stats_bytes;
  if (workspace == nullptr) {
    *workspace_size = required;
    return true;
  }
  NVTE_CHECK(*workspace_size >= required, "cuDNN OSS workspace is too small");
  *workspace_size = required;
  NVTE_CHECK(reinterpret_cast<uintptr_t>(workspace) % 128 == 0,
             "cuDNN OSS workspace must be 128-byte aligned");
  // Validate caller pointers before any launch or Stats conversion.
  for (const auto &op : plan->operands) {
    if (op.role < 0) continue;
    const auto address = reinterpret_cast<uintptr_t>(pointers[op.role]);
    NVTE_CHECK(address != 0 && address % op.alignment == 0,
               "cuDNN OSS operand is null or misaligned, role=", op.role);
    NVTE_CHECK(!overlaps(address, op.bytes, reinterpret_cast<uintptr_t>(workspace), required),
               "cuDNN OSS operand overlaps workspace");
    if (op.role >= 6 && op.role <= 8) {
      for (const auto &other : plan->operands) {
        if (other.role < 0 || other.role == op.role) continue;
        NVTE_CHECK(!overlaps(address, op.bytes, reinterpret_cast<uintptr_t>(pointers[other.role]),
                             other.bytes),
                   "cuDNN OSS gradient aliases an operand");
      }
    }
  }
  auto actual = pointers;
  if (uniform) {
    const size_t metadata_bytes = (cfg.batch_size + 1) * sizeof(int32_t);
    for (int i : {9, 10, 11, 12}) {
      const auto address = reinterpret_cast<uintptr_t>(pointers[i]);
      NVTE_CHECK(address != 0 && address % alignof(int32_t) == 0,
                 "THD metadata is null or misaligned");
      NVTE_CHECK(
          !overlaps(address, metadata_bytes, reinterpret_cast<uintptr_t>(workspace), required),
          "THD metadata overlaps workspace");
      for (const auto &op : plan->operands) {
        if (op.role >= 6 && op.role <= 8) {
          NVTE_CHECK(!overlaps(address, metadata_bytes,
                               reinterpret_cast<uintptr_t>(pointers[op.role]), op.bytes),
                     "cuDNN OSS gradient aliases THD metadata");
        }
      }
    }
    validate_uniform<<<(cfg.batch_size + 256) / 256, 256, 0, stream>>>(
        static_cast<const int32_t *>(pointers[9]), static_cast<const int32_t *>(pointers[10]),
        static_cast<const int32_t *>(pointers[11]), static_cast<const int32_t *>(pointers[12]),
        cfg.batch_size, cfg.max_seqlen_q, cfg.max_seqlen_kv);
    NVTE_CHECK_CUDA(cudaGetLastError());
    actual[5] = static_cast<char *>(workspace) + kernel_bytes;
    unpack_stats<<<(stats_bytes / 4 + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(pointers[5]), static_cast<float *>(actual[5]), stats_bytes / 4,
        cfg.num_attn_heads, cfg.max_seqlen_q);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
  std::array<tvm::ffi::AnyView, 64> frame;
  size_t cursor = 0;
  for (auto prefix : plan->prefix) frame[cursor++] = prefix;
  for (const auto &op : plan->operands) {
    if (op.role >= 0) frame[cursor] = reinterpret_cast<int64_t>(actual[op.role]);
    ++cursor;
  }
  frame[cursor++] = reinterpret_cast<int64_t>(workspace);
  frame[cursor++] = static_cast<double>(cfg.attn_scale);
  if (plan->length_form) frame[cursor++] = int32_t(3);
  frame[cursor++] = reinterpret_cast<void *>(stream);
  tvm::ffi::Any result;
  plan->function.CallPacked(frame.data(), static_cast<int32_t>(cursor), &result);
  return true;
}

}  // namespace fused_attn
}  // namespace transformer_engine
