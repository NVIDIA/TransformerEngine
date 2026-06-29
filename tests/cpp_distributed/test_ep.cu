/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * EP pipeline tests: smallest-scope first.
 *
 *   EPDispatchTest/PrepareAndDispatch       : exact recv values + per-expert counts
 *   EPCombineTest/Combine                   : round-trip: out == top_k * tokens
 *   EPCombineBwdTest/CombineBwdCheck        : exact grad_expert values
 *   EPDispatchBwdTest/DispatchBwdCheck      : exact grad_tokens
 *   EPDispatchBwdGradWeightsTest/RoundTrip  : exact per-(t, k) grad_topk_weights
 *   EPPipelineTest/FullForwardBackward     : fwd + bwd NaN/Inf check
 *
 * Routing: token t on rank r -> expert (r * num_tokens * top_k + t * top_k + k) % num_experts
 * Token values: rank r, token t -> all hidden dims = (r+1)*0.01 + t*0.001
 *
 * Closed-form expected values:
 *   dispatch recv:  multiset of source-token values routed to this rank's experts
 *   combine:        result[t] == top_k * tokens[t]
 *   combine_bwd:    grad_expert[slot] == d_result[t] (no weighting)
 *   dispatch_bwd:   grad_tokens[t]    == top_k * d_result[t]
 */

#include "test_ep_common.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

// -- Deterministic routing helpers ---------------------------------------------

// Token value for (rank, t): (rank * num_tokens + t + 1) / 256. Step 1/256 is
// bf16-exact and unique across (rank, t) when rank * num_tokens + t < 256.
static inline float token_value(int rank, int t, int num_tokens) {
  return static_cast<float>(rank * num_tokens + t + 1) * (1.0f / 256.0f);
}

// Per-element host-side conversion helpers used by templated test code.
inline float        tok_to_float(nv_bfloat16 v) { return __bfloat162float(v); }
inline float        tok_to_float(__half v)      { return __half2float(v); }
inline float        tok_to_float(float v)       { return v; }

template <typename T> T tok_from_float(float v);
template <> inline nv_bfloat16 tok_from_float<nv_bfloat16>(float v) { return __float2bfloat16(v); }
template <> inline __half      tok_from_float<__half>     (float v) { return __float2half(v); }
template <> inline float       tok_from_float<float>      (float v) { return v; }

template <typename T = nv_bfloat16>
static std::vector<T> generate_tokens(int rank, int num_tokens, int hidden_dim) {
  std::vector<T> v(num_tokens * hidden_dim);
  for (int t = 0; t < num_tokens; ++t) {
    T val = tok_from_float<T>(token_value(rank, t, num_tokens));
    for (int h = 0; h < hidden_dim; ++h)
      v[t * hidden_dim + h] = val;
  }
  return v;
}

static std::vector<int32_t> expected_recv_tokens_per_expert(
    int recv_rank, int num_processes, int num_tokens, int top_k,
    int num_experts, int num_local_experts) {
  int base = recv_rank * num_local_experts;
  std::vector<int32_t> cnt(num_local_experts, 0);
  for (int src = 0; src < num_processes; ++src) {
    auto idx = routing_balanced(src, num_tokens, top_k, num_experts, num_local_experts);
    for (int t = 0; t < num_tokens; ++t)
      for (int k = 0; k < top_k; ++k) {
        int64_t e = idx[t * top_k + k];
        if (e >= base && e < base + num_local_experts) ++cnt[e - base];
      }
  }
  return cnt;
}

template <typename T = nv_bfloat16>
static std::vector<float> expected_recv_values_sorted(
    int recv_rank, int num_processes, int num_tokens, int top_k,
    int num_experts, int num_local_experts) {
  int base = recv_rank * num_local_experts;
  std::vector<float> vals;
  for (int src = 0; src < num_processes; ++src) {
    auto idx = routing_balanced(src, num_tokens, top_k, num_experts, num_local_experts);
    for (int t = 0; t < num_tokens; ++t)
      for (int k = 0; k < top_k; ++k) {
        int64_t e = idx[t * top_k + k];
        if (e >= base && e < base + num_local_experts) {
          float raw = token_value(src, t, num_tokens);
          vals.push_back(tok_to_float(tok_from_float<T>(raw)));
        }
      }
  }
  std::sort(vals.begin(), vals.end());
  return vals;
}

// 2^-5 relative tolerance for BF16 (matches mantissa precision with margin),
// plus a small atol floor for near-zero expected values.
static constexpr float kBf16Rtol = 1.0f / 32.0f;
static constexpr float kBf16Atol = 1e-3f;
static float bf16_tol(float magnitude) {
  return kBf16Atol + kBf16Rtol * std::fabs(magnitude);
}

template <typename T = nv_bfloat16>
static bool check_no_nan_inf(const T* dev, int count, const char* name) {
  std::vector<T> h(count);
  cudaMemcpy(h.data(), dev, count * sizeof(T), cudaMemcpyDeviceToHost);
  for (int i = 0; i < count; ++i) {
    float v = tok_to_float(h[i]);
    if (std::isnan(v) || std::isinf(v)) {
      fprintf(stderr, "Rank %d: %s in %s[%d]\n",
              g_process_id, std::isnan(v) ? "NaN" : "Inf", name, i);
      return false;
    }
  }
  return true;
}

// -- Forward buffer set with RAII ----------------------------------------------

template <typename T = nv_bfloat16>
struct EPBuffers {
  // Forward
  DevBuf<int64_t>     topk_idx;
  DevBuf<float>       topk_weights;
  DevBuf<T>           tokens;
  DevBuf<int32_t>     recv_tokens_per_expert;
  DevBuf<uint8_t>     handle_mem;
  DevBuf<T>           recv_tokens;
  DevBuf<float>       recv_topk_weights;
  DevBuf<T>           result;
  // Backward
  DevBuf<T>           grad_result;
  DevBuf<T>           grad_expert;
  DevBuf<T>           grad_tokens;
  DevBuf<float>       g_recv_topk_weights;
  DevBuf<float>       grad_topk_weights;

  size_t handle_mem_size = 0;
  size_t recv_capacity   = 0;
  int    top_k_          = 0;
  size_t alignment_      = 0;
  NVTEEpLayerConfig layer_cfg_{};

  void alloc(int num_tokens, int top_k, int hidden_dim, int num_local_experts,
             int ep_size, int max_tokens_per_rank, size_t alignment = 0) {
    top_k_ = top_k;
    alignment_ = alignment;
    layer_cfg_ = NVTE_EP_LAYER_CONFIG_INIT;
    layer_cfg_.top_k = top_k;
    layer_cfg_.dispatch_output_per_expert_alignment = alignment;
    recv_capacity = static_cast<size_t>(ep_size) * max_tokens_per_rank * 2;

    topk_idx.alloc(num_tokens * top_k);
    topk_weights.alloc(num_tokens * top_k);
    tokens.alloc(num_tokens * hidden_dim);
    recv_tokens_per_expert.alloc(num_local_experts);
    recv_tokens.alloc(recv_capacity * hidden_dim);
    recv_topk_weights.alloc(recv_capacity);
    result.alloc(num_tokens * hidden_dim);

    handle_mem_size = nvte_ep_handle_mem_size(&layer_cfg_);
    handle_mem.alloc(handle_mem_size);

    grad_result.alloc(num_tokens * hidden_dim);
    grad_expert.alloc(recv_capacity * hidden_dim);
    grad_tokens.alloc(num_tokens * hidden_dim);
    g_recv_topk_weights.alloc(recv_capacity);
    grad_topk_weights.alloc(num_tokens * top_k);
  }
};

// Bundled NVTETensor views over an EPBuffers, with the shapes the EP C API
// expects.
template <typename T = nv_bfloat16>
struct EPTensors {
  TensorWrapper topk_idx, topk_weights, recv_tokens_per_expert, handle_mem, tokens;
  TensorWrapper recv_tokens, recv_topk_weights, result;
  TensorWrapper grad_result, grad_expert, grad_tokens;
  TensorWrapper g_recv_topk_weights, grad_topk_weights;

  int    top_k_     = 0;
  size_t alignment_ = 0;
  NVTEEpLayerConfig layer_cfg_{};

  EPTensors(EPBuffers<T>& b, int num_tokens, int top_k, int hidden_dim,
            int num_local_experts) {
    top_k_ = top_k;
    alignment_ = b.alignment_;
    layer_cfg_ = NVTE_EP_LAYER_CONFIG_INIT;
    layer_cfg_.top_k = top_k;
    layer_cfg_.dispatch_output_per_expert_alignment = b.alignment_;
    constexpr DType kTokDType = test::TypeInfo<T>::dtype;
    using Shape = std::vector<size_t>;
    topk_idx          = TensorWrapper(b.topk_idx.get(),
                            Shape{(size_t)num_tokens, (size_t)top_k}, DType::kInt64);
    topk_weights      = TensorWrapper(b.topk_weights.get(),
                            Shape{(size_t)num_tokens, (size_t)top_k}, DType::kFloat32);
    recv_tokens_per_expert      = TensorWrapper(b.recv_tokens_per_expert.get(),
                            Shape{(size_t)num_local_experts}, DType::kInt32);
    handle_mem        = TensorWrapper(b.handle_mem.get(),
                            Shape{b.handle_mem_size}, DType::kByte);
    tokens            = TensorWrapper(b.tokens.get(),
                            Shape{(size_t)num_tokens, (size_t)hidden_dim}, kTokDType);
    recv_tokens       = TensorWrapper(b.recv_tokens.get(),
                            Shape{b.recv_capacity, (size_t)hidden_dim}, kTokDType);
    recv_topk_weights = TensorWrapper(b.recv_topk_weights.get(),
                            Shape{b.recv_capacity}, DType::kFloat32);
    result            = TensorWrapper(b.result.get(),
                            Shape{(size_t)num_tokens, (size_t)hidden_dim}, kTokDType);
    grad_result       = TensorWrapper(b.grad_result.get(),
                            Shape{(size_t)num_tokens, (size_t)hidden_dim}, kTokDType);
    grad_expert       = TensorWrapper(b.grad_expert.get(),
                            Shape{b.recv_capacity, (size_t)hidden_dim}, kTokDType);
    grad_tokens       = TensorWrapper(b.grad_tokens.get(),
                            Shape{(size_t)num_tokens, (size_t)hidden_dim}, kTokDType);
    g_recv_topk_weights = TensorWrapper(b.g_recv_topk_weights.get(),
                            Shape{b.recv_capacity}, DType::kFloat32);
    grad_topk_weights = TensorWrapper(b.grad_topk_weights.get(),
                            Shape{(size_t)num_tokens, (size_t)top_k}, DType::kFloat32);
  }
};

// -- Shared fixture base -------------------------------------------------------

class EpOpTestBase : public ::testing::Test {
 protected:
  int ep_size_, num_experts_, num_local_experts_, hidden_dim_;
  int max_tokens_per_rank_, top_k_, num_tokens_;

  void SetUp() override {
    if (g_sm_major < 9)
      GTEST_SKIP() << "EP requires SM_90+ (device is SM_" << g_sm_major << "0)";
    ASSERT_GE(g_num_processes, 2);
    ASSERT_TRUE(g_ep_initialized);

    ep_size_             = g_ep_size;
    num_experts_         = g_num_experts;
    num_local_experts_   = num_experts_ / ep_size_;
    hidden_dim_          = g_hidden_dim;
    max_tokens_per_rank_ = g_max_tokens_per_rank;
    top_k_               = 2;
    num_tokens_          = 32;
  }

  template <typename T = nv_bfloat16>
  void upload_inputs(EPBuffers<T>& buf, int rank = -1) {
    if (rank < 0) rank = g_process_id;
    auto h_idx = routing_balanced(rank, num_tokens_, top_k_,
                                   num_experts_, num_local_experts_);
    std::vector<float> h_w(num_tokens_ * top_k_, 1.0f / top_k_);
    auto h_tok = generate_tokens<T>(rank, num_tokens_, hidden_dim_);

    NVTE_CHECK_CUDA(cudaMemcpy(buf.topk_idx.get(),     h_idx.data(),
                          h_idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(buf.topk_weights.get(), h_w.data(),
                          h_w.size()   * sizeof(float),   cudaMemcpyHostToDevice));
    NVTE_CHECK_CUDA(cudaMemcpy(buf.tokens.get(),       h_tok.data(),
                          h_tok.size() * sizeof(T),       cudaMemcpyHostToDevice));
  }

  // NVTE_CHECK_CUDA (fprintf+exit) so this non-void helper stays legal.
  template <typename T = nv_bfloat16>
  int read_total_recv(const EPBuffers<T>& buf) const {
    std::vector<int32_t> cnt(num_local_experts_);
    NVTE_CHECK_CUDA(cudaMemcpy(cnt.data(), buf.recv_tokens_per_expert.get(),
                              num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
    int total = 0;
    for (int c : cnt) total += c;
    return total;
  }
};

// Pull non-dependent base members into the typed-test scope as local consts so
// the bodies can reference them unqualified.
#define EP_PULL_FIXTURE()                                       \
  const int ep_size_             = this->ep_size_;              \
  const int num_experts_         = this->num_experts_;          \
  const int num_local_experts_   = this->num_local_experts_;    \
  const int hidden_dim_          = this->hidden_dim_;           \
  const int max_tokens_per_rank_ = this->max_tokens_per_rank_;  \
  const int top_k_               = this->top_k_;                \
  const int num_tokens_          = this->num_tokens_

// =============================================================================
// EPDispatchTest: exact recv values and per-expert counts.
// =============================================================================

template <typename T> class EPDispatchTest : public EpOpTestBase {};
using EPBf16Only = ::testing::Types<nv_bfloat16>;
TYPED_TEST_SUITE(EPDispatchTest, EPBf16Only);

TYPED_TEST(EPDispatchTest, PrepareAndDispatch) {
  using Tok = TypeParam;
  EP_PULL_FIXTURE();
  EPBuffers<Tok> buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  this->template upload_inputs<Tok>(buf);
  EPTensors<Tok> t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  NVTE_CHECK_CUDA(cudaMemset(buf.recv_tokens.get(), 0, buf.recv_tokens.bytes()));

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(t.handle_mem.data(), t.topk_idx.data(), t.recv_tokens_per_expert.data(), nullptr, &t.layer_cfg_, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(t.handle_mem.data(), t.topk_idx.data(),
                                   t.tokens.data(), NVTECommWindow{}, t.topk_weights.data(),
                                   NVTECommWindow{}, t.recv_tokens.data(), NVTECommWindow{},
                                   t.recv_topk_weights.data(), NVTECommWindow{}, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  // 1. Per-expert counts.
  std::vector<int32_t> got_counts(num_local_experts_);
  NVTE_CHECK_CUDA(cudaMemcpy(got_counts.data(), buf.recv_tokens_per_expert.get(),
                        num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
  auto exp_counts = expected_recv_tokens_per_expert(g_process_id, g_num_processes, num_tokens_, top_k_,
                                          num_experts_, num_local_experts_);
  int total_recv = 0;
  for (int i = 0; i < num_local_experts_; ++i) {
    EXPECT_EQ(got_counts[i], exp_counts[i]) << "local expert " << i;
    total_recv += exp_counts[i];
  }
  ASSERT_LE(total_recv, static_cast<int>(buf.recv_capacity))
      << "total_recv exceeded recv_capacity; overflow would corrupt downstream memory";

  // 2. Recv values: read only the filled prefix per local-expert zone, not the
  // whole recv buffer; avoids false positives from legitimate-zero token values.
  std::vector<Tok> h_recv(buf.recv_capacity * hidden_dim_);
  NVTE_CHECK_CUDA(cudaMemcpy(h_recv.data(), buf.recv_tokens.get(),
                        h_recv.size() * sizeof(Tok), cudaMemcpyDeviceToHost));

  std::vector<float> got_vals;
  got_vals.reserve(total_recv);
  size_t slot = 0;
  for (int e = 0; e < num_local_experts_; ++e) {
    for (int i = 0; i < got_counts[e]; ++i) {
      got_vals.push_back(tok_to_float(h_recv[slot * hidden_dim_]));
      ++slot;
    }
  }
  std::sort(got_vals.begin(), got_vals.end());

  auto exp_vals = expected_recv_values_sorted<Tok>(g_process_id, g_num_processes, num_tokens_,
                                                   top_k_, num_experts_, num_local_experts_);

  ASSERT_EQ(got_vals.size(), exp_vals.size());
  for (size_t i = 0; i < exp_vals.size(); ++i)
    EXPECT_EQ(got_vals[i], exp_vals[i])
        << "recv value mismatch at sorted index " << i;

  // 3. recv_topk_weights: every filled slot must equal the per-token weight (1/top_k).
  std::vector<float> h_w(buf.recv_capacity);
  NVTE_CHECK_CUDA(cudaMemcpy(h_w.data(), buf.recv_topk_weights.get(),
                        h_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const float exp_w = 1.0f / static_cast<float>(top_k_);
  for (int i = 0; i < total_recv; ++i)
    EXPECT_NEAR(h_w[i], exp_w, 1e-6f) << "recv_topk_weights[" << i << "]";

  if (g_process_id == 0)
    printf("  PrepareAndDispatch: passed (recv=%d, values + weights exact)\n", total_recv);

  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPCombineTest: round-trip identity expert -> result == top_k * tokens.
// =============================================================================

template <typename T> class EPCombineTest : public EpOpTestBase {};
TYPED_TEST_SUITE(EPCombineTest, EPBf16Only);

TYPED_TEST(EPCombineTest, Combine) {
  using Tok = TypeParam;
  EP_PULL_FIXTURE();
  EPBuffers<Tok> buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  this->template upload_inputs<Tok>(buf);
  EPTensors<Tok> t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(t.handle_mem.data(), t.topk_idx.data(), t.recv_tokens_per_expert.data(), nullptr, &t.layer_cfg_, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(t.handle_mem.data(), t.topk_idx.data(),
                                   t.tokens.data(), NVTECommWindow{}, t.topk_weights.data(),
                                   NVTECommWindow{}, t.recv_tokens.data(), NVTECommWindow{},
                                   t.recv_topk_weights.data(), NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(t.handle_mem.data(), t.recv_tokens.data(), NVTECommWindow{},
                                  t.result.data(), stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<Tok> h_result(num_tokens_ * hidden_dim_);
  NVTE_CHECK_CUDA(cudaMemcpy(h_result.data(), buf.result.get(),
                        h_result.size() * sizeof(Tok), cudaMemcpyDeviceToHost));
  auto h_tok = generate_tokens<Tok>(g_process_id, num_tokens_, hidden_dim_);
  for (int tok = 0; tok < num_tokens_; ++tok) {
    float exp = tok_to_float(h_tok[tok * hidden_dim_]) * static_cast<float>(top_k_);
    for (int p = 0; p < hidden_dim_; ++p) {
      float got = tok_to_float(h_result[tok * hidden_dim_ + p]);
      EXPECT_NEAR(got, exp, bf16_tol(exp))
          << "token " << tok << " rank " << g_process_id << " hidden " << p;
    }
  }

  if (g_process_id == 0)
    printf("  Combine: passed (result == top_k * tokens)\n");

  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPCombineBwdTest: filled slots in grad_expert == d_result (unweighted).
// =============================================================================

template <typename T> class EPCombineBwdTest : public EpOpTestBase {};
TYPED_TEST_SUITE(EPCombineBwdTest, EPBf16Only);

TYPED_TEST(EPCombineBwdTest, CombineBwdCheck) {
  using Tok = TypeParam;
  EP_PULL_FIXTURE();
  EPBuffers<Tok> buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  this->template upload_inputs<Tok>(buf);
  EPTensors<Tok> t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(t.handle_mem.data(), t.topk_idx.data(), t.recv_tokens_per_expert.data(), nullptr, &t.layer_cfg_, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(t.handle_mem.data(), t.topk_idx.data(),
                                   t.tokens.data(), NVTECommWindow{}, t.topk_weights.data(),
                                   NVTECommWindow{}, t.recv_tokens.data(), NVTECommWindow{},
                                   t.recv_topk_weights.data(), NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(t.handle_mem.data(), t.recv_tokens.data(), NVTECommWindow{},
                                  t.result.data(), stream));

  std::vector<Tok> h_grad_r(num_tokens_ * hidden_dim_, tok_from_float<Tok>(0.1f));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(buf.grad_result.get(), h_grad_r.data(),
                             h_grad_r.size() * sizeof(Tok),
                             cudaMemcpyHostToDevice, stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(), 0, buf.grad_expert.bytes(), stream));

  ASSERT_NO_THROW(nvte_ep_combine_bwd(t.handle_mem.data(), t.grad_result.data(), NVTECommWindow{},
                                      t.grad_expert.data(), NVTECommWindow{}, stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  int total_recv = this->template read_total_recv<Tok>(buf);

  std::vector<int32_t> cnt(num_local_experts_);
  NVTE_CHECK_CUDA(cudaMemcpy(cnt.data(), buf.recv_tokens_per_expert.get(),
                        num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
  std::vector<Tok> h_ge(buf.recv_capacity * hidden_dim_);
  NVTE_CHECK_CUDA(cudaMemcpy(h_ge.data(), buf.grad_expert.get(),
                        h_ge.size() * sizeof(Tok), cudaMemcpyDeviceToHost));

  // Walk filled slots by per-expert zone (no v != 0 heuristic).
  const float kExpGrad = tok_to_float(tok_from_float<Tok>(0.1f));
  size_t slot = 0;
  int filled = 0;
  for (int e = 0; e < num_local_experts_; ++e) {
    for (int i = 0; i < cnt[e]; ++i) {
      for (int p = 0; p < hidden_dim_; ++p) {
        float v = tok_to_float(h_ge[slot * hidden_dim_ + p]);
        EXPECT_NEAR(v, kExpGrad, bf16_tol(kExpGrad))
            << "grad_expert expert " << e << " slot " << i
            << " (linear " << slot << ") hidden " << p;
      }
      ++filled; ++slot;
    }
  }
  EXPECT_EQ(filled, total_recv);

  if (g_process_id == 0)
    printf("  CombineBwdCheck: passed (filled=%d)\n", filled);

  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPDispatchBwdTest: grad_tokens == top_k * d_result.
// =============================================================================

template <typename T> class EPDispatchBwdTest : public EpOpTestBase {};
TYPED_TEST_SUITE(EPDispatchBwdTest, EPBf16Only);

TYPED_TEST(EPDispatchBwdTest, DispatchBwdCheck) {
  using Tok = TypeParam;
  EP_PULL_FIXTURE();
  EPBuffers<Tok> buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  this->template upload_inputs<Tok>(buf);
  EPTensors<Tok> t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(t.handle_mem.data(), t.topk_idx.data(), t.recv_tokens_per_expert.data(), nullptr, &t.layer_cfg_, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(t.handle_mem.data(), t.topk_idx.data(),
                                   t.tokens.data(), NVTECommWindow{}, t.topk_weights.data(),
                                   NVTECommWindow{}, t.recv_tokens.data(), NVTECommWindow{},
                                   t.recv_topk_weights.data(), NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(t.handle_mem.data(), t.recv_tokens.data(), NVTECommWindow{},
                                  t.result.data(), stream));

  std::vector<Tok> h_grad(num_tokens_ * hidden_dim_, tok_from_float<Tok>(0.1f));
  NVTE_CHECK_CUDA(cudaMemcpyAsync(buf.grad_result.get(), h_grad.data(),
                             h_grad.size() * sizeof(Tok),
                             cudaMemcpyHostToDevice, stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(),         0, buf.grad_expert.bytes(),         stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(buf.g_recv_topk_weights.get(), 0, buf.g_recv_topk_weights.bytes(), stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(buf.grad_topk_weights.get(),   0, buf.grad_topk_weights.bytes(),   stream));

  ASSERT_NO_THROW(nvte_ep_combine_bwd(t.handle_mem.data(), t.grad_result.data(), NVTECommWindow{},
                                      t.grad_expert.data(), NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(t.handle_mem.data(), t.grad_expert.data(), NVTECommWindow{},
                                       t.g_recv_topk_weights.data(), NVTECommWindow{},
                                       t.grad_tokens.data(), t.grad_topk_weights.data(), stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<Tok> h_gt(num_tokens_ * hidden_dim_);
  NVTE_CHECK_CUDA(cudaMemcpy(h_gt.data(), buf.grad_tokens.get(),
                        h_gt.size() * sizeof(Tok), cudaMemcpyDeviceToHost));
  const float kExpGrad = static_cast<float>(top_k_) * tok_to_float(tok_from_float<Tok>(0.1f));
  for (int tok = 0; tok < num_tokens_; ++tok)
    for (int p = 0; p < hidden_dim_; ++p)
      EXPECT_NEAR(tok_to_float(h_gt[tok * hidden_dim_ + p]), kExpGrad,
                  bf16_tol(kExpGrad))
          << "grad_tokens token " << tok << " hidden " << p;

  if (g_process_id == 0)
    printf("  DispatchBwdCheck: passed (grad_tokens == %.2f)\n", kExpGrad);

  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPDispatchBwdGradWeightsTest: round-trip per-(t, k) weights.
// =============================================================================

template <typename T> class EPDispatchBwdGradWeightsTest : public EpOpTestBase {};
TYPED_TEST_SUITE(EPDispatchBwdGradWeightsTest, EPBf16Only);

TYPED_TEST(EPDispatchBwdGradWeightsTest, RoundTrip) {
  using Tok = TypeParam;
  EP_PULL_FIXTURE();
  EPBuffers<Tok> buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  this->template upload_inputs<Tok>(buf);
  EPTensors<Tok> t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  // Distinct per-(rank, t, k) weights so each slot carries a unique value.
  // Global integer counter over (rank, tok, k) keeps every slot unique.
  std::vector<float> h_w(num_tokens_ * top_k_);
  for (int tok = 0; tok < num_tokens_; ++tok)
    for (int k = 0; k < top_k_; ++k)
      h_w[tok * top_k_ + k] = static_cast<float>(
          (g_process_id * num_tokens_ + tok) * top_k_ + k + 1);
  NVTE_CHECK_CUDA(cudaMemcpy(buf.topk_weights.get(), h_w.data(),
                        h_w.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(t.handle_mem.data(), t.topk_idx.data(), t.recv_tokens_per_expert.data(), nullptr, &t.layer_cfg_, stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(buf.recv_topk_weights.get(), 0,
                             buf.recv_topk_weights.bytes(), stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(t.handle_mem.data(), t.topk_idx.data(),
                                   t.tokens.data(), NVTECommWindow{}, t.topk_weights.data(),
                                   NVTECommWindow{}, t.recv_tokens.data(), NVTECommWindow{},
                                   t.recv_topk_weights.data(), NVTECommWindow{}, stream));

  // Sentinel: NaN so any (t, k) the bwd kernel fails to write is immediately visible.
  std::vector<float> h_nan(num_tokens_ * top_k_,
                           std::numeric_limits<float>::quiet_NaN());
  NVTE_CHECK_CUDA(cudaMemcpyAsync(buf.grad_topk_weights.get(), h_nan.data(),
                             h_nan.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  NVTE_CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(), 0, buf.grad_expert.bytes(), stream));

  // g_recv_topk_weights := recv_topk_weights (the round-trip input).
  auto g_recv_t = TensorWrapper(buf.recv_topk_weights.get(),
                                   std::vector<size_t>{buf.recv_capacity}, DType::kFloat32);
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(t.handle_mem.data(), t.grad_expert.data(),
                                       NVTECommWindow{}, g_recv_t.data(), NVTECommWindow{},
                                       t.grad_tokens.data(), t.grad_topk_weights.data(), stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<float> h_grad_w(num_tokens_ * top_k_);
  NVTE_CHECK_CUDA(cudaMemcpy(h_grad_w.data(), buf.grad_topk_weights.get(),
                        h_grad_w.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float kTol = 1e-5f;
  int errs = 0, k0_eq_k1 = 0;
  for (int tok = 0; tok < num_tokens_; ++tok) {
    for (int k = 0; k < top_k_; ++k) {
      float got = h_grad_w[tok * top_k_ + k];
      float exp = h_w[tok * top_k_ + k];
      if (std::isnan(got) || std::fabs(got - exp) > kTol) {
        if (errs < 8)
          fprintf(stderr, "Rank %d: grad_topk_weights[%d, %d]: got %.6f, expected %.6f\n",
                  g_process_id, tok, k, got, exp);
        ++errs;
      }
    }
    if (top_k_ >= 2 &&
        std::fabs(h_grad_w[tok * top_k_ + 0] - h_grad_w[tok * top_k_ + 1]) < 1e-7f)
      ++k0_eq_k1;
  }
  EXPECT_EQ(errs, 0);
  EXPECT_EQ(k0_eq_k1, 0) << "per-token-average regression: grad[t, 0] == grad[t, 1]";

  if (g_process_id == 0 && errs == 0 && k0_eq_k1 == 0)
    printf("  RoundTrip: passed (%d (t, k) gradients)\n", num_tokens_ * top_k_);

  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// Integrated FwdBwd: NaN/Inf check end-to-end.
// =============================================================================

class EPPipelineTest : public EpOpTestBase, public ::testing::WithParamInterface<DType> {
 protected:
  template <typename Tok>
  void run_full_forward_backward() {
    EPBuffers<Tok> buf;
    buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
              ep_size_, max_tokens_per_rank_);
    upload_inputs<Tok>(buf);
    EPTensors<Tok> t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

    cudaStream_t stream;
    NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

    ASSERT_NO_THROW(nvte_ep_prepare(t.handle_mem.data(), t.topk_idx.data(), t.recv_tokens_per_expert.data(), nullptr, &t.layer_cfg_, stream));
    ASSERT_NO_THROW(nvte_ep_dispatch(t.handle_mem.data(), t.topk_idx.data(),
                                     t.tokens.data(), NVTECommWindow{}, t.topk_weights.data(),
                                     NVTECommWindow{}, t.recv_tokens.data(), NVTECommWindow{},
                                     t.recv_topk_weights.data(), NVTECommWindow{}, stream));
    ASSERT_NO_THROW(nvte_ep_combine(t.handle_mem.data(), t.recv_tokens.data(), NVTECommWindow{},
                                    t.result.data(), stream));

    std::vector<Tok> h_grad(num_tokens_ * hidden_dim_, tok_from_float<Tok>(0.1f));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(buf.grad_result.get(), h_grad.data(),
                               h_grad.size() * sizeof(Tok),
                               cudaMemcpyHostToDevice, stream));
    NVTE_CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(),         0, buf.grad_expert.bytes(),         stream));
    NVTE_CHECK_CUDA(cudaMemsetAsync(buf.g_recv_topk_weights.get(), 0, buf.g_recv_topk_weights.bytes(), stream));
    NVTE_CHECK_CUDA(cudaMemsetAsync(buf.grad_topk_weights.get(),   0, buf.grad_topk_weights.bytes(),   stream));

    ASSERT_NO_THROW(nvte_ep_combine_bwd(t.handle_mem.data(), t.grad_result.data(), NVTECommWindow{},
                                        t.grad_expert.data(), NVTECommWindow{}, stream));
    ASSERT_NO_THROW(nvte_ep_dispatch_bwd(t.handle_mem.data(), t.grad_expert.data(), NVTECommWindow{},
                                         t.g_recv_topk_weights.data(), NVTECommWindow{},
                                         t.grad_tokens.data(), t.grad_topk_weights.data(), stream));
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

    ASSERT_TRUE(check_no_nan_inf<Tok>(buf.result.get(),      num_tokens_ * hidden_dim_, "result"));
    ASSERT_TRUE(check_no_nan_inf<Tok>(buf.grad_tokens.get(), num_tokens_ * hidden_dim_, "grad_tokens"));

    NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
  }
};

TEST_P(EPPipelineTest, FullForwardBackward) {
  const DType dtype = GetParam();
  // NCCL EP backend currently asserts ncclBfloat16 in ncclEpDispatch
  // (contrib/nccl_ep/nccl_ep.cc); skip FP16/FP32 until the backend supports them.
  if (dtype != DType::kBFloat16) {
    GTEST_SKIP() << test::typeName(dtype) << " not yet supported by NCCL EP backend";
  }
  switch (dtype) {
    case DType::kBFloat16: run_full_forward_backward<nv_bfloat16>(); break;
    case DType::kFloat16:  run_full_forward_backward<__half>     (); break;
    case DType::kFloat32:  run_full_forward_backward<float>      (); break;
    default: FAIL() << "unsupported token dtype " << static_cast<int>(dtype);
  }
  if (g_process_id == 0)
    printf("  FullForwardBackward[%s]: passed\n", test::typeName(dtype).c_str());
}

INSTANTIATE_TEST_SUITE_P(
    Dtypes, EPPipelineTest,
    ::testing::Values(DType::kBFloat16, DType::kFloat16, DType::kFloat32),
    [](const ::testing::TestParamInfo<DType>& info) {
      return test::typeName(info.param);
    });

// =============================================================================
// EPZeroCopyTest: dispatch/combine with NCCL symmetric-memory windows attached
// to payload tensors (zero-copy fast path via ncclEpTensorCreateFromWindow).
// Symm-mem requirements per spec: input&output of Dispatch, input of Combine,
// input&output of Combine bwd, input of Dispatch bwd.
// =============================================================================

namespace {

// Caller-owned ncclMemAlloc'd buffer with a registered symmetric window.
// Frees in destructor (deregister + ncclMemFree). Non-copyable, move-only.
struct SymmBuf {
  void*        ptr   = nullptr;
  size_t       bytes = 0;
  ncclWindow_t win   = nullptr;

  SymmBuf() = default;
  SymmBuf(const SymmBuf&) = delete;
  SymmBuf& operator=(const SymmBuf&) = delete;
  SymmBuf(SymmBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes), win(o.win) {
    o.ptr = nullptr; o.win = nullptr; o.bytes = 0;
  }
  ~SymmBuf() {
    if (win)  ncclCommWindowDeregister(g_ep_comm, win);
    if (ptr)  ncclMemFree(ptr);
  }

  void alloc(size_t n_bytes) {
    bytes = n_bytes;
    NVTE_CHECK_NCCL(ncclMemAlloc(&ptr, bytes));
    NVTE_CHECK_CUDA(cudaMemset(ptr, 0, bytes));
    NVTE_CHECK_NCCL(ncclCommWindowRegister(g_ep_comm, ptr, bytes, &win,
                                           NCCL_WIN_COLL_SYMMETRIC));
  }
};

// Build an NVTECommWindow descriptor pointing at a SymmBuf's window (offset 0).
static inline NVTECommWindow symm_window(const SymmBuf& b) {
  return NVTECommWindow{b.win, /*offset=*/0};
}

}  // namespace

// Tests rebootstrap the backend to zero_copy=ON for the symm phase via
// ep_reinitialize(); TearDown restores OFF for the rest of the suite.
template <typename T>
class EPZeroCopyTest : public EpOpTestBase {
 protected:
  void TearDown() override {
    if (g_ep_initialized) ep_reinitialize(/*zero_copy=*/0);
  }
};
TYPED_TEST_SUITE(EPZeroCopyTest, EPBf16Only);

// Identity round-trip with symm-mem on dispatch i/o + combine input. Bit-exact
// vs HBM reference (same routing, same input).
TYPED_TEST(EPZeroCopyTest, IdentityAllSymm) {
  using Tok = TypeParam;
  EP_PULL_FIXTURE();
  constexpr DType kTokDType = test::TypeInfo<Tok>::dtype;

  // HBM reference run.
  EPBuffers<Tok> ref_buf;
  ref_buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
                ep_size_, max_tokens_per_rank_);
  this->template upload_inputs<Tok>(ref_buf);
  EPTensors<Tok> ref_t(ref_buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NO_THROW(nvte_ep_prepare(ref_t.handle_mem.data(), ref_t.topk_idx.data(), ref_t.recv_tokens_per_expert.data(), nullptr, &ref_t.layer_cfg_, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(ref_t.handle_mem.data(), ref_t.topk_idx.data(),
                                   ref_t.tokens.data(), NVTECommWindow{}, ref_t.topk_weights.data(),
                                   NVTECommWindow{}, ref_t.recv_tokens.data(), NVTECommWindow{},
                                   ref_t.recv_topk_weights.data(), NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(ref_t.handle_mem.data(), ref_t.recv_tokens.data(), NVTECommWindow{},
                                  ref_t.result.data(), stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<Tok> ref_recv(ref_buf.recv_capacity * hidden_dim_);
  std::vector<Tok> ref_result(num_tokens_ * hidden_dim_);
  NVTE_CHECK_CUDA(cudaMemcpy(ref_recv.data(),   ref_buf.recv_tokens.get(),
                        ref_recv.size() * sizeof(Tok), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(ref_result.data(), ref_buf.result.get(),
                        ref_result.size() * sizeof(Tok), cudaMemcpyDeviceToHost));

  // Switch backend to zero_copy=ON for the symm phase.
  ep_reinitialize(/*zero_copy=*/1);

  // Symm-mem run: tokens, recv_tokens, combine_input (== recv_tokens) all symm.
  EPBuffers<Tok> sym_buf;  // alloc all buffers except the symm ones.
  sym_buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
                ep_size_, max_tokens_per_rank_);
  this->template upload_inputs<Tok>(sym_buf);

  SymmBuf sym_tokens, sym_recv;
  sym_tokens.alloc(num_tokens_           * hidden_dim_ * sizeof(Tok));
  sym_recv  .alloc(sym_buf.recv_capacity * hidden_dim_ * sizeof(Tok));

  // Stage same tokens into the symm-mem input.
  auto h_tok = generate_tokens<Tok>(g_process_id, num_tokens_, hidden_dim_);
  NVTE_CHECK_CUDA(cudaMemcpy(sym_tokens.ptr, h_tok.data(),
                        h_tok.size() * sizeof(Tok), cudaMemcpyHostToDevice));

  EPTensors<Tok> sym_t(sym_buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);
  // Replace the tokens/recv_tokens views with ones pointing at the symm buffers.
  sym_t.tokens      = TensorWrapper(sym_tokens.ptr,
                          std::vector<size_t>{(size_t)num_tokens_, (size_t)hidden_dim_}, kTokDType);
  sym_t.recv_tokens = TensorWrapper(sym_recv.ptr,
                          std::vector<size_t>{sym_buf.recv_capacity, (size_t)hidden_dim_}, kTokDType);

  ASSERT_NO_THROW(nvte_ep_prepare(sym_t.handle_mem.data(), sym_t.topk_idx.data(), sym_t.recv_tokens_per_expert.data(), nullptr, &sym_t.layer_cfg_, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(sym_t.handle_mem.data(), sym_t.topk_idx.data(),
                                   sym_t.tokens.data(), symm_window(sym_tokens),
                                   sym_t.topk_weights.data(), NVTECommWindow{},
                                   sym_t.recv_tokens.data(), symm_window(sym_recv),
                                   sym_t.recv_topk_weights.data(), NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(sym_t.handle_mem.data(), sym_t.recv_tokens.data(),
                                  symm_window(sym_recv), sym_t.result.data(), stream));
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<Tok> sym_recv_host(sym_buf.recv_capacity * hidden_dim_);
  std::vector<Tok> sym_result(num_tokens_ * hidden_dim_);
  NVTE_CHECK_CUDA(cudaMemcpy(sym_recv_host.data(), sym_recv.ptr,
                        sym_recv_host.size() * sizeof(Tok), cudaMemcpyDeviceToHost));
  NVTE_CHECK_CUDA(cudaMemcpy(sym_result.data(),    sym_buf.result.get(),
                        sym_result.size() * sizeof(Tok), cudaMemcpyDeviceToHost));

  // Compare per filled recv slot (HBM ref vs symm) and full result.
  int total_recv = this->template read_total_recv<Tok>(sym_buf);
  for (int i = 0; i < total_recv * hidden_dim_; ++i)
    ASSERT_EQ(tok_to_float(sym_recv_host[i]), tok_to_float(ref_recv[i]))
        << "recv mismatch at " << i;
  for (size_t i = 0; i < sym_result.size(); ++i)
    ASSERT_EQ(tok_to_float(sym_result[i]), tok_to_float(ref_result[i]))
        << "result mismatch at " << i;

  if (g_process_id == 0)
    printf("  IdentityAllSymm: passed (recv_slots=%d, bit-exact vs HBM)\n", total_recv);

  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
}


// -- main ----------------------------------------------------------------------

int main(int argc, char* argv[]) {
  if (!ep_bootstrap(argc, argv)) return 0;
  int ret = RUN_ALL_TESTS();
  ep_teardown();
  return ret;
}
