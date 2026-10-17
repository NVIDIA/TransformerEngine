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
#include "frost_output_policy.h"

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

struct UniformLayout {
  size_t q_pitch;
  size_t kv_pitch;
  size_t q_origin;
};

bool uniform_layout(const FusedAttnConfig &c, UniformLayout *layout) {
  if (!getenv<bool>("NVTE_FUSED_ATTN_FROST_UNIFORM")) return false;
  *layout = {c.max_seqlen_q, c.max_seqlen_kv, 0};
  if (c.num_tokens_q == c.batch_size * c.max_seqlen_q &&
      c.num_tokens_kv == c.batch_size * c.max_seqlen_kv)
    return true;
  if (!getenv<bool>("NVTE_FUSED_ATTN_FROST_ALLGATHER") || !c.is_causal_bottom_right ||
      c.max_seqlen_q % 128 != 0 || c.num_tokens_kv % c.batch_size != 0)
    return false;
  const size_t pq = 2 * c.max_seqlen_q;
  const size_t pk = c.num_tokens_kv / c.batch_size;
  const size_t cp = pk / pq;
  if ((cp != 2 && cp != 4 && cp != 8) || pk % pq != 0 || c.num_tokens_q != c.batch_size * pq ||
      c.max_seqlen_kv > pk || c.max_seqlen_kv % c.max_seqlen_q != 0)
    return false;
  *layout = {pq, pk, c.max_seqlen_kv > pk / 2 ? c.max_seqlen_q : 0};
  return true;
}

bool eligible(const FusedAttnConfig &c) {
  const bool causal = c.is_causal || c.is_causal_bottom_right;
  if (cuda::sm_arch(cuda::current_device()) != 100 || c.deterministic || !c.is_training ||
      c.qkv_dtype != kNVTEBFloat16 || c.o_dtype != kNVTEBFloat16 || c.do_dtype != kNVTEBFloat16 ||
      c.dqkv_dtype != kNVTEBFloat16 || c.head_dim_qk != c.head_dim_v ||
      (c.head_dim_qk != 128 && c.head_dim_qk != 256) || c.bias_type != NVTE_NO_BIAS ||
      c.softmax_type != NVTE_VANILLA_SOFTMAX || c.dropout != 0 || c.is_paged_kv ||
      (c.bottom_right_diagonal && (c.head_dim_qk != 256 || !c.is_causal_bottom_right)) ||
      c.window_size_left != -1 || (c.window_size_right != -1 && c.window_size_right != 0) ||
      (c.window_size_right == 0 && !causal) || !std::isfinite(c.attn_scale) || c.attn_scale <= 0 ||
      c.batch_size == 0 || c.batch_size > 65535 || c.num_gqa_groups == 0 ||
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
  UniformLayout layout;
  return uniform_layout(c, &layout);
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
  UniformLayout layout{0, 0, 0};
  if (c.is_ragged_q && c.head_dim_qk == 256) {
    NVTE_CHECK(uniform_layout(c, &layout), "Invalid cuDNN OSS uniform layout");
  }
  // Never hold a C++ cache lock while entering Python: another Python thread
  // might hold the GIL and be waiting for that lock.
  auto result = (*prepare)(
      tvm::ffi::String(key), static_cast<int64_t>(c.batch_size),
      static_cast<int64_t>(c.num_attn_heads), static_cast<int64_t>(c.num_gqa_groups),
      static_cast<int64_t>(c.max_seqlen_q), static_cast<int64_t>(c.max_seqlen_kv),
      static_cast<int64_t>(c.head_dim_qk), static_cast<int64_t>(c.num_tokens_q),
      static_cast<int64_t>(c.num_tokens_kv), thd, c.is_causal || c.is_causal_bottom_right,
      static_cast<double>(c.attn_scale), cuda::current_device(), c.bottom_right_diagonal,
      static_cast<int64_t>(layout.q_pitch), static_cast<int64_t>(layout.kv_pitch));
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
                                 int64_t sq, int64_t sk, int64_t pq, int64_t pk, int64_t origin) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i <= batch &&
      (cu_q[i] != i * sq || cu_kv[i] != i * sk ||
       origins_q[i] != i * pq + (i < batch ? origin : 0) || origins_kv[i] != i * pk)) {
    printf("NVTE_FUSED_ATTN_FROST_UNIFORM: invalid sequence/offset at %d\n", static_cast<int>(i));
    asm volatile("trap;");
  }
}

__global__ void unpack_stats(const float *packed, float *dense, int64_t size, int64_t heads,
                             int64_t sequence, int64_t pitch, int64_t origin) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    const int64_t s = i % sequence, h = (i / sequence) % heads;
    const int64_t b = i / (sequence * heads);
    dense[i] = packed[(b * pitch + origin + s) * heads + h];
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
  const bool compact_d128 = frost_d128_compact_outputs(
      cfg.qkv_layout, cfg.dqkv_layout, cfg.dqkv_dtype, cfg.head_dim_qk, cfg.head_dim_v,
      cfg.batch_size, cfg.num_tokens_q, cfg.num_tokens_kv, cfg.max_seqlen_q, cfg.max_seqlen_kv,
      cfg.num_attn_heads, cfg.num_gqa_groups);
  const bool uniform = cfg.is_ragged_q && cfg.head_dim_qk == 256;
  UniformLayout layout{0, 0, 0};
  if (uniform) {
    NVTE_CHECK(uniform_layout(cfg, &layout), "Invalid cuDNN OSS uniform layout");
  }
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
  // Validate the full declared physical buffers, including holes, before any
  // launch or pointer adjustment. Packed Stats is larger than the dense scratch.
  auto bound_bytes = [&](const Operand &op) -> size_t {
    if (!uniform || op.role < 0 || op.role > 8) return op.bytes;
    if (op.role == 5) return cfg.num_tokens_q * cfg.num_attn_heads * sizeof(float);
    const bool kv = op.role == 1 || op.role == 2 || op.role == 7 || op.role == 8;
    return (kv ? cfg.num_tokens_kv * cfg.num_gqa_groups : cfg.num_tokens_q * cfg.num_attn_heads) *
           cfg.head_dim_qk * 2;
  };
  // Validate caller pointers before any launch or Stats conversion.
  for (const auto &op : plan->operands) {
    if (op.role < 0) continue;
    const auto address = reinterpret_cast<uintptr_t>(pointers[op.role]);
    NVTE_CHECK(address != 0 && address % op.alignment == 0,
               "cuDNN OSS operand is null or misaligned, role=", op.role);
    NVTE_CHECK(
        !overlaps(address, bound_bytes(op), reinterpret_cast<uintptr_t>(workspace), required),
        "cuDNN OSS operand overlaps workspace");
    if (op.role >= 6 && op.role <= 8) {
      for (const auto &other : plan->operands) {
        if (other.role < 0 || other.role == op.role) continue;
        NVTE_CHECK(!overlaps(address, bound_bytes(op),
                             reinterpret_cast<uintptr_t>(pointers[other.role]), bound_bytes(other)),
                   "cuDNN OSS gradient aliases an operand");
      }
    }
  }
  // D128 keeps native THD Stats and descriptors. This guard proves that every
  // physical output row is live; it does not lower THD to a dense engine.
  if (compact_d128) {
    validate_uniform<<<(cfg.batch_size + 256) / 256, 256, 0, stream>>>(
        static_cast<const int32_t *>(pointers[9]), static_cast<const int32_t *>(pointers[10]),
        static_cast<const int32_t *>(pointers[11]), static_cast<const int32_t *>(pointers[12]),
        cfg.batch_size, cfg.max_seqlen_q, cfg.max_seqlen_kv, cfg.max_seqlen_q, cfg.max_seqlen_kv,
        0);
    NVTE_CHECK_CUDA(cudaGetLastError());
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
                               reinterpret_cast<uintptr_t>(pointers[op.role]), bound_bytes(op)),
                     "cuDNN OSS gradient aliases THD metadata");
        }
      }
    }
    validate_uniform<<<(cfg.batch_size + 256) / 256, 256, 0, stream>>>(
        static_cast<const int32_t *>(pointers[9]), static_cast<const int32_t *>(pointers[10]),
        static_cast<const int32_t *>(pointers[11]), static_cast<const int32_t *>(pointers[12]),
        cfg.batch_size, cfg.max_seqlen_q, cfg.max_seqlen_kv, layout.q_pitch, layout.kv_pitch,
        layout.q_origin);
    NVTE_CHECK_CUDA(cudaGetLastError());
    actual[5] = static_cast<char *>(workspace) + kernel_bytes;
    unpack_stats<<<(stats_bytes / 4 + 255) / 256, 256, 0, stream>>>(
        static_cast<const float *>(pointers[5]), static_cast<float *>(actual[5]), stats_bytes / 4,
        cfg.num_attn_heads, cfg.max_seqlen_q, layout.q_pitch, layout.q_origin);
    NVTE_CHECK_CUDA(cudaGetLastError());
  }
  if (uniform && layout.q_origin != 0) {
    const size_t offset = layout.q_origin * cfg.num_attn_heads * cfg.head_dim_qk * 2;
    for (int role : {0, 3, 4, 6}) actual[role] = static_cast<char *>(actual[role]) + offset;
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

namespace {

struct ForwardArgument {
  int kind;
  int64_t value;
  tvm::ffi::Any constant;
};

struct ForwardPlan {
  tvm::ffi::Function function;
  size_t workspace_bytes;
  size_t workspace_alignment;
  std::array<Operand, 9> operands;
  std::vector<ForwardArgument> arguments;
};

bool forward_eligible(const FusedAttnConfig &c) {
  // The current FE forward host derives compact physical positions from lengths.
  // Separate padded origins therefore require an explicit, device-checked promise.
  return getenv<bool>("NVTE_FUSED_ATTN_FROST_FWD_COMPACT") &&
         cuda::sm_arch(cuda::current_device()) == 100 && c.is_training && !c.return_max_logit &&
         c.qkv_dtype == kNVTEBFloat16 && c.o_dtype == kNVTEBFloat16 && c.head_dim_qk == 128 &&
         c.head_dim_v == 128 && c.bias_type == NVTE_NO_BIAS &&
         c.softmax_type == NVTE_VANILLA_SOFTMAX && c.dropout == 0 && !c.is_paged_kv &&
         c.qkv_layout == NVTE_THD_THD_THD && c.o_format == NVTE_THD && c.is_padding &&
         c.is_ragged_q && c.uses_ragged_stats && c.is_causal && !c.is_causal_bottom_right &&
         c.window_size_left == -1 && (c.window_size_right == -1 || c.window_size_right == 0) &&
         std::isfinite(c.attn_scale) && c.attn_scale > 0 && c.batch_size > 0 &&
         c.batch_size <= 65535 && c.num_gqa_groups > 0 && c.num_attn_heads <= 65535 &&
         c.num_attn_heads >= c.num_gqa_groups && c.num_attn_heads % c.num_gqa_groups == 0 &&
         c.max_seqlen_q > 0 && c.max_seqlen_kv > 0 && c.max_seqlen_q <= INT32_MAX - 256 &&
         c.max_seqlen_kv <= INT32_MAX - 256 && c.num_tokens_q > 0 && c.num_tokens_kv > 0 &&
         c.num_tokens_q <= INT32_MAX && c.num_tokens_kv <= INT32_MAX;
}

const ForwardPlan *get_forward_plan(const FusedAttnConfig &c, cudaStream_t stream) {
  static auto *cache = new std::map<FusedAttnConfig, std::unique_ptr<ForwardPlan>>;
  static auto *mutex = new std::mutex;
  static std::atomic<uint64_t> serial{0};
  {
    std::lock_guard<std::mutex> lock(*mutex);
    auto it = cache->find(c);
    if (it != cache->end()) return it->second.get();
  }
  cudaStreamCaptureStatus capture;
  NVTE_CHECK_CUDA(cudaStreamIsCapturing(stream, &capture));
  NVTE_CHECK(capture == cudaStreamCaptureStatusNone,
             "Warm up cuDNN OSS forward before CUDA graph capture");
  auto &central = tvm_ffi_bridge::TVMFFICentral::getInstance();
  if (!central.get_cutedsl_backend_enabled() || !central.get_tvm_ffi_available()) return nullptr;
  auto prepare = tvm::ffi::Function::GetGlobal("get_frost_attention_forward");
  auto constant = tvm::ffi::Function::GetGlobal("get_frost_attention_forward_constant");
  if (!prepare || !constant) return nullptr;
  const std::string key = "transformer_engine.frost.forward." + std::to_string(serial++);
  // Python is entered only on a cache miss, without holding a C++ cache lock.
  auto result =
      (*prepare)(tvm::ffi::String(key), static_cast<int64_t>(c.batch_size),
                 static_cast<int64_t>(c.num_attn_heads), static_cast<int64_t>(c.num_gqa_groups),
                 static_cast<int64_t>(c.max_seqlen_q), static_cast<int64_t>(c.max_seqlen_kv),
                 static_cast<int64_t>(c.num_tokens_q), static_cast<int64_t>(c.num_tokens_kv),
                 static_cast<double>(c.attn_scale), c.device_id);
  auto encoded = result.try_cast<tvm::ffi::String>();
  auto declined = result.try_cast<bool>();
  NVTE_CHECK(encoded || (declined && !*declined), "Invalid cuDNN OSS forward preparation response");
  std::unique_ptr<ForwardPlan> plan;
  if (encoded) {
    NVTE_CHECK(encoded->size() < 4096, "Oversized cuDNN OSS forward metadata");
    std::istringstream input{std::string(*encoded)};
    std::vector<int64_t> v;
    int64_t value;
    while (input >> value) v.push_back(value);
    NVTE_CHECK(input.eof() && v.size() >= 22 && v[0] == 1 && v[1] > 0 && v[2] >= 16 &&
                   (v[2] & (v[2] - 1)) == 0 && v[3] == 27 && v.size() == 22 + 2 * v[3],
               "Malformed cuDNN OSS forward metadata");
    auto function = tvm::ffi::Function::GetGlobal(key);
    NVTE_CHECK(function.has_value(), "cuDNN OSS did not register its native forward entry");
    plan = std::make_unique<ForwardPlan>(
        ForwardPlan{*function, static_cast<size_t>(v[1]), static_cast<size_t>(v[2]), {}, {}});
    size_t cursor = 4;
    for (int role = 0; role < 9; ++role) {
      const auto bytes = v[cursor++], alignment = v[cursor++];
      NVTE_CHECK(bytes > 0 && alignment > 0 && (alignment & (alignment - 1)) == 0,
                 "Invalid cuDNN OSS forward operand metadata");
      plan->operands[role] = {role, static_cast<size_t>(bytes), static_cast<size_t>(alignment)};
    }
    for (int64_t i = 0; i < v[3]; ++i) {
      const auto kind = v[cursor++], index = v[cursor++];
      NVTE_CHECK((kind == 0 && index == i) || (kind == 1 && index >= 0 && index < 7) ||
                     (kind == 2 && index >= 0 && index < v[1]) || (kind == 3 && index == 0),
                 "Invalid cuDNN OSS forward argument metadata");
      tvm::ffi::Any immutable;
      if (kind == 0) immutable = (*constant)(tvm::ffi::String(key), index);
      plan->arguments.push_back({static_cast<int>(kind), index, std::move(immutable)});
    }
  }
  std::lock_guard<std::mutex> lock(*mutex);
  return cache->emplace(c, std::move(plan)).first->second.get();
}

__global__ void validate_compact_forward(const int32_t *cu_q, const int32_t *cu_kv,
                                         const int32_t *origins_q, const int32_t *origins_kv,
                                         int64_t batch, int64_t sq, int64_t sk, int64_t tq,
                                         int64_t tk) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i > batch) return;
  const int64_t q = cu_q[i], k = cu_kv[i];
  bool valid = q >= 0 && q <= tq && k >= 0 && k <= tk && origins_q[i] == q && origins_kv[i] == k;
  if (i == 0) valid = valid && q == 0 && k == 0;
  if (i < batch) {
    const int64_t lq = static_cast<int64_t>(cu_q[i + 1]) - q;
    const int64_t lk = static_cast<int64_t>(cu_kv[i + 1]) - k;
    valid = valid && lq >= 0 && lq <= sq && lk >= 0 && lk <= sk && (lq == 0 || lk > 0);
  }
  if (!valid) {
    printf("NVTE_FUSED_ATTN_FROST_FWD_COMPACT: invalid sequence/offset at %d\n",
           static_cast<int>(i));
    asm volatile("trap;");
  }
}

}  // namespace

bool frost_attention_forward(const FusedAttnConfig &cfg, const FrostForwardPointers &pointers,
                             void *workspace, size_t *workspace_size, cudaStream_t stream) {
  const int mode = getenv<int>("NVTE_FUSED_ATTN_FROST_FWD");
  if (mode == 0) return false;
  NVTE_CHECK(mode == 1 || mode == 2, "NVTE_FUSED_ATTN_FROST_FWD must be 0, 1 or 2");
  cfg.check_derived();
  FusedAttnConfig cached = cfg;
  cached.device_id = cuda::current_device();
  const ForwardPlan *plan = forward_eligible(cfg) ? get_forward_plan(cached, stream) : nullptr;
  if (!plan) {
    NVTE_CHECK(mode != 2,
               "Requested cuDNN OSS forward unavailable; check compact THD D128 causal scope, "
               "NVTE_FUSED_ATTN_FROST_FWD_COMPACT=1 and compatible FE/CuTeDSL installation");
    return false;
  }
  if (workspace == nullptr) {
    *workspace_size = plan->workspace_bytes;
    return true;
  }
  NVTE_CHECK(*workspace_size >= plan->workspace_bytes, "cuDNN OSS forward workspace is too small");
  *workspace_size = plan->workspace_bytes;
  const auto address = reinterpret_cast<uintptr_t>(workspace);
  NVTE_CHECK(address % plan->workspace_alignment == 0, "cuDNN OSS forward workspace misaligned");
  for (const auto &op : plan->operands) {
    const auto ptr = reinterpret_cast<uintptr_t>(pointers[op.role]);
    NVTE_CHECK(ptr != 0 && ptr % op.alignment == 0,
               "cuDNN OSS forward operand is null or misaligned, role=", op.role);
    NVTE_CHECK(!overlaps(ptr, op.bytes, address, plan->workspace_bytes),
               "cuDNN OSS forward operand overlaps workspace");
    if (op.role == 3 || op.role == 4) {
      for (const auto &other : plan->operands) {
        if (other.role == op.role) continue;
        NVTE_CHECK(!overlaps(ptr, op.bytes, reinterpret_cast<uintptr_t>(pointers[other.role]),
                             other.bytes),
                   "cuDNN OSS forward output aliases an operand");
      }
    }
  }
  validate_compact_forward<<<(cfg.batch_size + 256) / 256, 256, 0, stream>>>(
      static_cast<const int32_t *>(pointers[5]), static_cast<const int32_t *>(pointers[6]),
      static_cast<const int32_t *>(pointers[7]), static_cast<const int32_t *>(pointers[8]),
      cfg.batch_size, cfg.max_seqlen_q, cfg.max_seqlen_kv, cfg.num_tokens_q, cfg.num_tokens_kv);
  NVTE_CHECK_CUDA(cudaGetLastError());
  std::array<tvm::ffi::AnyView, 27> frame;
  for (size_t i = 0; i < plan->arguments.size(); ++i) {
    const auto &arg = plan->arguments[i];
    if (arg.kind == 0) frame[i] = static_cast<tvm::ffi::AnyView>(arg.constant);
    if (arg.kind == 1) frame[i] = reinterpret_cast<int64_t>(pointers[arg.value]);
    if (arg.kind == 2) frame[i] = static_cast<int64_t>(address + arg.value);
    if (arg.kind == 3) frame[i] = reinterpret_cast<void *>(stream);
  }
  tvm::ffi::Any result;
  plan->function.CallPacked(frame.data(), static_cast<int32_t>(frame.size()), &result);
  return true;
}

}  // namespace fused_attn
}  // namespace transformer_engine
