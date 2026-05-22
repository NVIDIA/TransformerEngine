/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * EP pipeline tests: smallest-scope first.
 *
 *   EPDispatchTest/PrepareAndDispatch       — exact recv values + per-expert counts
 *   EPCombineTest/Combine                   — round-trip: out == top_k * tokens
 *   EPCombineBwdTest/CombineBwdCheck        — exact grad_expert values
 *   EPDispatchBwdTest/DispatchBwdCheck      — exact grad_tokens
 *   EPDispatchBwdGradWeightsTest/RoundTrip  — exact per-(t, k) grad_topk_weights
 *   EPPipelineTest/FullForwardBackward     — fwd + bwd NaN/Inf check
 *
 * Routing: token t on rank r → expert (r * num_local_experts + t * top_k + k) % num_experts
 * Token values: rank r, token t → all hidden dims = (r+1)*0.01 + t*0.001
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

// ── Deterministic routing helpers ─────────────────────────────────────────────

// Token value for (rank, t): (rank * num_tokens + t + 1) / 256. Step 1/256 is
// bf16-exact and unique across (rank, t) when rank * num_tokens + t < 256.
static inline float token_value(int rank, int t, int num_tokens) {
  return static_cast<float>(rank * num_tokens + t + 1) * (1.0f / 256.0f);
}

static std::vector<nv_bfloat16> generate_tokens(int rank, int num_tokens, int hidden_dim) {
  std::vector<nv_bfloat16> v(num_tokens * hidden_dim);
  for (int t = 0; t < num_tokens; ++t) {
    nv_bfloat16 val = __float2bfloat16(token_value(rank, t, num_tokens));
    for (int h = 0; h < hidden_dim; ++h)
      v[t * hidden_dim + h] = val;
  }
  return v;
}

static std::vector<int32_t> expected_token_counts(
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
          vals.push_back(__bfloat162float(__float2bfloat16(raw)));
        }
      }
  }
  std::sort(vals.begin(), vals.end());
  return vals;
}

// BF16 has 7 mantissa bits; relative ULP ≈ 2^-7. Use 4× headroom for
// accumulation noise inside dispatch/combine.
static float bf16_tol(float magnitude) {
  return 4.f * std::ldexp(std::fabs(magnitude) + 1e-3f, -7);
}

static bool check_no_nan_inf(const nv_bfloat16* dev, int count, const char* name) {
  std::vector<nv_bfloat16> h(count);
  cudaMemcpy(h.data(), dev, count * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost);
  for (int i = 0; i < count; ++i) {
    float v = __bfloat162float(h[i]);
    if (std::isnan(v) || std::isinf(v)) {
      fprintf(stderr, "Rank %d: %s in %s[%d]\n",
              g_process_id, std::isnan(v) ? "NaN" : "Inf", name, i);
      return false;
    }
  }
  return true;
}

// ── Forward buffer set with RAII ──────────────────────────────────────────────

struct EPBuffers {
  // Forward
  DevBuf<int64_t>     topk_idx;
  DevBuf<float>       topk_weights;
  DevBuf<nv_bfloat16> tokens;
  DevBuf<int32_t>     token_counts;
  DevBuf<uint8_t>     handle_mem;
  DevBuf<nv_bfloat16> recv_tokens;
  DevBuf<float>       recv_topk_weights;
  DevBuf<nv_bfloat16> result;
  // Backward
  DevBuf<nv_bfloat16> grad_result;
  DevBuf<nv_bfloat16> grad_expert;
  DevBuf<nv_bfloat16> grad_tokens;
  DevBuf<float>       g_recv_topk_weights;
  DevBuf<float>       grad_topk_weights;

  uint64_t handle_id      = 0;
  size_t handle_mem_size = 0;
  size_t recv_capacity   = 0;
  int    top_k_          = 0;

  void alloc(int num_tokens, int top_k, int hidden_dim, int num_local_experts,
             int ep_size, int max_tokens_per_rank, size_t alignment = 0) {
    top_k_ = top_k;
    recv_capacity = static_cast<size_t>(ep_size) * max_tokens_per_rank * 2;

    topk_idx.alloc(num_tokens * top_k);
    topk_weights.alloc(num_tokens * top_k);
    tokens.alloc(num_tokens * hidden_dim);
    token_counts.alloc(num_local_experts);
    recv_tokens.alloc(recv_capacity * hidden_dim);
    recv_topk_weights.alloc(recv_capacity);
    result.alloc(num_tokens * hidden_dim);

    NVTEEpLayerConfig cfg{num_local_experts, top_k, alignment};
    handle_id = nvte_ep_register_layer(cfg, &handle_mem_size);
    handle_mem.alloc(handle_mem_size);

    grad_result.alloc(num_tokens * hidden_dim);
    grad_expert.alloc(recv_capacity * hidden_dim);
    grad_tokens.alloc(num_tokens * hidden_dim);
    g_recv_topk_weights.alloc(recv_capacity);
    grad_topk_weights.alloc(num_tokens * top_k);
  }
};

// Bundled NVTETensor views over an EPBuffers — one place to update the shape
// conventions when the C-API evolves.
struct EPTensors {
  TensorHandle topk_idx, topk_weights, token_counts, handle_mem, tokens;
  TensorHandle recv_tokens, recv_topk_weights, result;
  TensorHandle grad_result, grad_expert, grad_tokens;
  TensorHandle g_recv_topk_weights, grad_topk_weights;

  EPTensors(EPBuffers& b, int num_tokens, int top_k, int hidden_dim,
            int num_local_experts) {
    topk_idx          = make_nvte_tensor(b.topk_idx.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
    topk_weights      = make_nvte_tensor(b.topk_weights.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEFloat32);
    token_counts      = make_nvte_tensor(b.token_counts.get(),
                            {(size_t)num_local_experts}, kNVTEInt32);
    handle_mem        = make_nvte_tensor(b.handle_mem.get(),
                            {b.handle_mem_size}, kNVTEByte);
    tokens            = make_nvte_tensor(b.tokens.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim}, kNVTEBFloat16);
    recv_tokens       = make_nvte_tensor(b.recv_tokens.get(),
                            {b.recv_capacity, (size_t)hidden_dim}, kNVTEBFloat16);
    recv_topk_weights = make_nvte_tensor(b.recv_topk_weights.get(),
                            {b.recv_capacity}, kNVTEFloat32);
    result            = make_nvte_tensor(b.result.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim}, kNVTEBFloat16);
    grad_result       = make_nvte_tensor(b.grad_result.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim}, kNVTEBFloat16);
    grad_expert       = make_nvte_tensor(b.grad_expert.get(),
                            {b.recv_capacity, (size_t)hidden_dim}, kNVTEBFloat16);
    grad_tokens       = make_nvte_tensor(b.grad_tokens.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim}, kNVTEBFloat16);
    g_recv_topk_weights = make_nvte_tensor(b.g_recv_topk_weights.get(),
                            {b.recv_capacity}, kNVTEFloat32);
    grad_topk_weights = make_nvte_tensor(b.grad_topk_weights.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEFloat32);
  }
};

// ── Shared fixture base ───────────────────────────────────────────────────────

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

  void upload_inputs(EPBuffers& buf, int rank = -1) {
    if (rank < 0) rank = g_process_id;
    auto h_idx = routing_balanced(rank, num_tokens_, top_k_,
                                   num_experts_, num_local_experts_);
    std::vector<float> h_w(num_tokens_ * top_k_, 1.0f / top_k_);
    auto h_tok = generate_tokens(rank, num_tokens_, hidden_dim_);

    CHECK_CUDA(cudaMemcpy(buf.topk_idx.get(),     h_idx.data(),
                          h_idx.size() * sizeof(int64_t),     cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(buf.topk_weights.get(), h_w.data(),
                          h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(buf.tokens.get(),       h_tok.data(),
                          h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  }

  NVTEEpLayerConfig layer_config(size_t alignment = 0) const {
    return NVTEEpLayerConfig{num_local_experts_, top_k_, alignment};
  }

  // ASSERT_CUDA_OK (fprintf+exit) so this non-void helper stays legal.
  int read_total_recv(const EPBuffers& buf) const {
    std::vector<int32_t> cnt(num_local_experts_);
    ASSERT_CUDA_OK(cudaMemcpy(cnt.data(), buf.token_counts.get(),
                              num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
    int total = 0;
    for (int c : cnt) total += c;
    return total;
  }
};

// =============================================================================
// EPDispatchTest: exact recv values and per-expert counts.
// =============================================================================

class EPDispatchTest : public EpOpTestBase {};

TEST_F(EPDispatchTest, PrepareAndDispatch) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);
  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  CHECK_CUDA(cudaMemset(buf.recv_tokens.get(), 0, buf.recv_tokens.bytes()));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                   t.tokens.tensor, NVTECommWindow{}, t.topk_weights.tensor,
                                   NVTECommWindow{}, t.recv_tokens.tensor, NVTECommWindow{},
                                   t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // 1. Per-expert counts.
  std::vector<int32_t> got_counts(num_local_experts_);
  CHECK_CUDA(cudaMemcpy(got_counts.data(), buf.token_counts.get(),
                        num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
  auto exp_counts = expected_token_counts(g_process_id, g_num_processes, num_tokens_, top_k_,
                                          num_experts_, num_local_experts_);
  int total_recv = 0;
  for (int i = 0; i < num_local_experts_; ++i) {
    EXPECT_EQ(got_counts[i], exp_counts[i]) << "local expert " << i;
    total_recv += exp_counts[i];
  }
  ASSERT_LE(total_recv, static_cast<int>(buf.recv_capacity))
      << "total_recv exceeded recv_capacity — overflow would corrupt downstream memory";

  // 2. Recv values: read only the filled prefix per local-expert zone, not the
  // whole recv buffer — avoids false positives from legitimate-zero token values.
  std::vector<nv_bfloat16> h_recv(buf.recv_capacity * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_recv.data(), buf.recv_tokens.get(),
                        h_recv.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

  std::vector<float> got_vals;
  got_vals.reserve(total_recv);
  size_t slot = 0;
  for (int e = 0; e < num_local_experts_; ++e) {
    for (int i = 0; i < got_counts[e]; ++i) {
      got_vals.push_back(__bfloat162float(h_recv[slot * hidden_dim_]));
      ++slot;
    }
  }
  std::sort(got_vals.begin(), got_vals.end());

  auto exp_vals = expected_recv_values_sorted(g_process_id, g_num_processes, num_tokens_,
                                              top_k_, num_experts_, num_local_experts_);

  ASSERT_EQ(got_vals.size(), exp_vals.size());
  for (size_t i = 0; i < exp_vals.size(); ++i)
    EXPECT_NEAR(got_vals[i], exp_vals[i], bf16_tol(exp_vals[i]))
        << "recv value mismatch at sorted index " << i;

  // 3. recv_topk_weights: every filled slot must equal the per-token weight (1/top_k).
  std::vector<float> h_w(buf.recv_capacity);
  CHECK_CUDA(cudaMemcpy(h_w.data(), buf.recv_topk_weights.get(),
                        h_w.size() * sizeof(float), cudaMemcpyDeviceToHost));
  const float exp_w = 1.0f / static_cast<float>(top_k_);
  for (int i = 0; i < total_recv; ++i)
    EXPECT_NEAR(h_w[i], exp_w, 1e-6f) << "recv_topk_weights[" << i << "]";

  if (g_process_id == 0)
    printf("  PrepareAndDispatch: passed (recv=%d, values + weights exact)\n", total_recv);

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPCombineTest: round-trip identity expert → result == top_k * tokens.
// =============================================================================

class EPCombineTest : public EpOpTestBase {};

TEST_F(EPCombineTest, Combine) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);
  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                   t.tokens.tensor, NVTECommWindow{}, t.topk_weights.tensor,
                                   NVTECommWindow{}, t.recv_tokens.tensor, NVTECommWindow{},
                                   t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.recv_tokens.tensor, NVTECommWindow{},
                                  t.result.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<nv_bfloat16> h_result(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_result.data(), buf.result.get(),
                        h_result.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  auto h_tok = generate_tokens(g_process_id, num_tokens_, hidden_dim_);
  // Spot-check 3 hidden-dim positions per token to catch partial-row writes.
  const int probes[3] = {0, hidden_dim_ / 2, hidden_dim_ - 1};
  for (int tok = 0; tok < num_tokens_; ++tok) {
    float exp = __bfloat162float(h_tok[tok * hidden_dim_]) * static_cast<float>(top_k_);
    for (int p : probes) {
      float got = __bfloat162float(h_result[tok * hidden_dim_ + p]);
      EXPECT_NEAR(got, exp, bf16_tol(exp))
          << "token " << tok << " rank " << g_process_id << " hidden " << p;
    }
  }

  if (g_process_id == 0)
    printf("  Combine: passed (result == top_k * tokens)\n");

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPCombineBwdTest: filled slots in grad_expert == d_result (unweighted).
// =============================================================================

class EPCombineBwdTest : public EpOpTestBase {};

TEST_F(EPCombineBwdTest, CombineBwdCheck) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);
  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                   t.tokens.tensor, NVTECommWindow{}, t.topk_weights.tensor,
                                   NVTECommWindow{}, t.recv_tokens.tensor, NVTECommWindow{},
                                   t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.recv_tokens.tensor, NVTECommWindow{},
                                  t.result.tensor, stream));

  std::vector<nv_bfloat16> h_grad_r(num_tokens_ * hidden_dim_, __float2bfloat16(0.1f));
  CHECK_CUDA(cudaMemcpyAsync(buf.grad_result.get(), h_grad_r.data(),
                             h_grad_r.size() * sizeof(nv_bfloat16),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(), 0, buf.grad_expert.bytes(), stream));

  ASSERT_NO_THROW(nvte_ep_combine_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_result.tensor, NVTECommWindow{},
                                      t.grad_expert.tensor, NVTECommWindow{}, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  int total_recv = read_total_recv(buf);

  std::vector<int32_t> cnt(num_local_experts_);
  CHECK_CUDA(cudaMemcpy(cnt.data(), buf.token_counts.get(),
                        num_local_experts_ * sizeof(int32_t), cudaMemcpyDeviceToHost));
  std::vector<nv_bfloat16> h_ge(buf.recv_capacity * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_ge.data(), buf.grad_expert.get(),
                        h_ge.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

  // Walk filled slots by per-expert zone (no v != 0 heuristic).
  const float kExpGrad = 0.1f;
  size_t slot = 0;
  int filled = 0;
  for (int e = 0; e < num_local_experts_; ++e) {
    for (int i = 0; i < cnt[e]; ++i) {
      float v = __bfloat162float(h_ge[slot * hidden_dim_]);
      EXPECT_NEAR(v, kExpGrad, bf16_tol(kExpGrad))
          << "grad_expert expert " << e << " slot " << i << " (linear " << slot << ")";
      ++filled; ++slot;
    }
  }
  EXPECT_EQ(filled, total_recv);

  if (g_process_id == 0)
    printf("  CombineBwdCheck: passed (filled=%d)\n", filled);

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPDispatchBwdTest: grad_tokens == top_k * d_result.
// =============================================================================

class EPDispatchBwdTest : public EpOpTestBase {};

TEST_F(EPDispatchBwdTest, DispatchBwdCheck) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);
  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                   t.tokens.tensor, NVTECommWindow{}, t.topk_weights.tensor,
                                   NVTECommWindow{}, t.recv_tokens.tensor, NVTECommWindow{},
                                   t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.recv_tokens.tensor, NVTECommWindow{},
                                  t.result.tensor, stream));

  std::vector<nv_bfloat16> h_grad(num_tokens_ * hidden_dim_, __float2bfloat16(0.1f));
  CHECK_CUDA(cudaMemcpyAsync(buf.grad_result.get(), h_grad.data(),
                             h_grad.size() * sizeof(nv_bfloat16),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(),         0, buf.grad_expert.bytes(),         stream));
  CHECK_CUDA(cudaMemsetAsync(buf.g_recv_topk_weights.get(), 0, buf.g_recv_topk_weights.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(buf.grad_topk_weights.get(),   0, buf.grad_topk_weights.bytes(),   stream));

  ASSERT_NO_THROW(nvte_ep_combine_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_result.tensor, NVTECommWindow{},
                                      t.grad_expert.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_expert.tensor, NVTECommWindow{},
                                       t.g_recv_topk_weights.tensor, NVTECommWindow{},
                                       t.grad_tokens.tensor, t.grad_topk_weights.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<nv_bfloat16> h_gt(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_gt.data(), buf.grad_tokens.get(),
                        h_gt.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  const float kExpGrad = static_cast<float>(top_k_) * 0.1f;
  for (int tok = 0; tok < num_tokens_; ++tok)
    EXPECT_NEAR(__bfloat162float(h_gt[tok * hidden_dim_]), kExpGrad, bf16_tol(kExpGrad))
        << "grad_tokens token " << tok;

  if (g_process_id == 0)
    printf("  DispatchBwdCheck: passed (grad_tokens == %.2f)\n", kExpGrad);

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EPDispatchBwdGradWeightsTest: round-trip per-(t, k) weights.
// =============================================================================

class EPDispatchBwdGradWeightsTest : public EpOpTestBase {};

TEST_F(EPDispatchBwdGradWeightsTest, RoundTrip) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);
  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  // Distinct per-(rank, t, k) weights so each slot carries a unique value.
  std::vector<float> h_w(num_tokens_ * top_k_);
  for (int tok = 0; tok < num_tokens_; ++tok)
    for (int k = 0; k < top_k_; ++k)
      h_w[tok * top_k_ + k] = 0.1f + 0.01f * tok + 0.001f * k +
                              0.0001f * (g_process_id + 1);
  CHECK_CUDA(cudaMemcpy(buf.topk_weights.get(), h_w.data(),
                        h_w.size() * sizeof(float), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
  CHECK_CUDA(cudaMemsetAsync(buf.recv_topk_weights.get(), 0,
                             buf.recv_topk_weights.bytes(), stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                   t.tokens.tensor, NVTECommWindow{}, t.topk_weights.tensor,
                                   NVTECommWindow{}, t.recv_tokens.tensor, NVTECommWindow{},
                                   t.recv_topk_weights.tensor, NVTECommWindow{}, stream));

  // Sentinel: NaN so any (t, k) the bwd kernel fails to write is immediately visible.
  std::vector<float> h_nan(num_tokens_ * top_k_,
                           std::numeric_limits<float>::quiet_NaN());
  CHECK_CUDA(cudaMemcpyAsync(buf.grad_topk_weights.get(), h_nan.data(),
                             h_nan.size() * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(), 0, buf.grad_expert.bytes(), stream));

  // g_recv_topk_weights := recv_topk_weights (the round-trip input).
  auto g_recv_t = make_nvte_tensor(buf.recv_topk_weights.get(),
                                   {buf.recv_capacity}, kNVTEFloat32);
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_expert.tensor,
                                       NVTECommWindow{}, g_recv_t.tensor, NVTECommWindow{},
                                       t.grad_tokens.tensor, t.grad_topk_weights.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<float> h_grad_w(num_tokens_ * top_k_);
  CHECK_CUDA(cudaMemcpy(h_grad_w.data(), buf.grad_topk_weights.get(),
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

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// Integrated FwdBwd: NaN/Inf check end-to-end.
// =============================================================================

class EPPipelineTest : public EpOpTestBase {};

TEST_F(EPPipelineTest, FullForwardBackward) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);
  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                   t.tokens.tensor, NVTECommWindow{}, t.topk_weights.tensor,
                                   NVTECommWindow{}, t.recv_tokens.tensor, NVTECommWindow{},
                                   t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.recv_tokens.tensor, NVTECommWindow{},
                                  t.result.tensor, stream));

  std::vector<nv_bfloat16> h_grad(num_tokens_ * hidden_dim_, __float2bfloat16(0.1f));
  CHECK_CUDA(cudaMemcpyAsync(buf.grad_result.get(), h_grad.data(),
                             h_grad.size() * sizeof(nv_bfloat16),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemsetAsync(buf.grad_expert.get(),         0, buf.grad_expert.bytes(),         stream));
  CHECK_CUDA(cudaMemsetAsync(buf.g_recv_topk_weights.get(), 0, buf.g_recv_topk_weights.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(buf.grad_topk_weights.get(),   0, buf.grad_topk_weights.bytes(),   stream));

  ASSERT_NO_THROW(nvte_ep_combine_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_result.tensor, NVTECommWindow{},
                                      t.grad_expert.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_expert.tensor, NVTECommWindow{},
                                       t.g_recv_topk_weights.tensor, NVTECommWindow{},
                                       t.grad_tokens.tensor, t.grad_topk_weights.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  ASSERT_TRUE(check_no_nan_inf(buf.result.get(),      num_tokens_ * hidden_dim_, "result"));
  ASSERT_TRUE(check_no_nan_inf(buf.grad_tokens.get(), num_tokens_ * hidden_dim_, "grad_tokens"));

  if (g_process_id == 0) printf("  FullForwardBackward: passed\n");

  CHECK_CUDA(cudaStreamDestroy(stream));
}

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
    ASSERT_NCCL_OK(ncclMemAlloc(&ptr, bytes));
    CHECK_CUDA(cudaMemset(ptr, 0, bytes));
    ASSERT_NCCL_OK(ncclCommWindowRegister(g_ep_comm, ptr, bytes, &win,
                                           NCCL_WIN_COLL_SYMMETRIC));
  }
};

// Build an NVTECommWindow descriptor pointing at a SymmBuf's window (offset 0).
static inline NVTECommWindow symm_window(const SymmBuf& b) {
  return NVTECommWindow{b.win, /*offset=*/0};
}

}  // namespace

class EPZeroCopyTest : public EpOpTestBase {};

// Identity round-trip with symm-mem on dispatch i/o + combine input. Bit-exact
// vs HBM reference (same routing, same input).
TEST_F(EPZeroCopyTest, IdentityAllSymm) {
  // HBM reference run.
  EPBuffers ref_buf;
  ref_buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
                ep_size_, max_tokens_per_rank_);
  upload_inputs(ref_buf);
  EPTensors ref_t(ref_buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t ref_hid = ref_buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{ref_hid, ref_t.handle_mem.tensor}, ref_t.topk_idx.tensor, ref_t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{ref_hid, ref_t.handle_mem.tensor}, ref_t.topk_idx.tensor,
                                   ref_t.tokens.tensor, NVTECommWindow{}, ref_t.topk_weights.tensor,
                                   NVTECommWindow{}, ref_t.recv_tokens.tensor, NVTECommWindow{},
                                   ref_t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{ref_hid, ref_t.handle_mem.tensor}, ref_t.recv_tokens.tensor, NVTECommWindow{},
                                  ref_t.result.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<nv_bfloat16> ref_recv(ref_buf.recv_capacity * hidden_dim_);
  std::vector<nv_bfloat16> ref_result(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(ref_recv.data(),   ref_buf.recv_tokens.get(),
                        ref_recv.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(ref_result.data(), ref_buf.result.get(),
                        ref_result.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

  // Symm-mem run: tokens, recv_tokens, combine_input (== recv_tokens) all symm.
  EPBuffers sym_buf;  // alloc all buffers except the symm ones.
  sym_buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
                ep_size_, max_tokens_per_rank_);
  upload_inputs(sym_buf);

  SymmBuf sym_tokens, sym_recv;
  sym_tokens.alloc(num_tokens_           * hidden_dim_ * sizeof(nv_bfloat16));
  sym_recv  .alloc(sym_buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16));

  // Stage same tokens into the symm-mem input.
  auto h_tok = generate_tokens(g_process_id, num_tokens_, hidden_dim_);
  CHECK_CUDA(cudaMemcpy(sym_tokens.ptr, h_tok.data(),
                        h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  EPTensors sym_t(sym_buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);
  // Replace the tokens/recv_tokens views with ones pointing at the symm buffers.
  sym_t.tokens      = make_nvte_tensor(sym_tokens.ptr,
                          {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  sym_t.recv_tokens = make_nvte_tensor(sym_recv.ptr,
                          {sym_buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);

  uint64_t sym_hid = sym_buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{sym_hid, sym_t.handle_mem.tensor}, sym_t.topk_idx.tensor, sym_t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{sym_hid, sym_t.handle_mem.tensor}, sym_t.topk_idx.tensor,
                                   sym_t.tokens.tensor, symm_window(sym_tokens),
                                   sym_t.topk_weights.tensor, NVTECommWindow{},
                                   sym_t.recv_tokens.tensor, symm_window(sym_recv),
                                   sym_t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{sym_hid, sym_t.handle_mem.tensor}, sym_t.recv_tokens.tensor,
                                  symm_window(sym_recv), sym_t.result.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<nv_bfloat16> sym_recv_host(sym_buf.recv_capacity * hidden_dim_);
  std::vector<nv_bfloat16> sym_result(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(sym_recv_host.data(), sym_recv.ptr,
                        sym_recv_host.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(sym_result.data(),    sym_buf.result.get(),
                        sym_result.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));

  // Compare per filled recv slot (HBM ref vs symm) and full result.
  int total_recv = read_total_recv(sym_buf);
  for (int i = 0; i < total_recv * hidden_dim_; ++i)
    ASSERT_EQ(__bfloat162float(sym_recv_host[i]), __bfloat162float(ref_recv[i]))
        << "recv mismatch at " << i;
  for (size_t i = 0; i < sym_result.size(); ++i)
    ASSERT_EQ(__bfloat162float(sym_result[i]), __bfloat162float(ref_result[i]))
        << "result mismatch at " << i;

  if (g_process_id == 0)
    printf("  IdentityAllSymm: passed (recv_slots=%d, bit-exact vs HBM)\n", total_recv);

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// Same buffers, 2 iterations — catches window-lifecycle regressions where the
// symm-mem registration goes stale between calls.
TEST_F(EPZeroCopyTest, IdentityAllSymmRepeated) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);

  SymmBuf sym_tokens, sym_recv;
  sym_tokens.alloc(num_tokens_       * hidden_dim_ * sizeof(nv_bfloat16));
  sym_recv  .alloc(buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16));
  auto h_tok = generate_tokens(g_process_id, num_tokens_, hidden_dim_);
  CHECK_CUDA(cudaMemcpy(sym_tokens.ptr, h_tok.data(),
                        h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);
  t.tokens      = make_nvte_tensor(sym_tokens.ptr,
                      {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  t.recv_tokens = make_nvte_tensor(sym_recv.ptr,
                      {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  for (int iter = 0; iter < 2; ++iter) {
    ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
    ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                     t.tokens.tensor, symm_window(sym_tokens),
                                     t.topk_weights.tensor, NVTECommWindow{},
                                     t.recv_tokens.tensor, symm_window(sym_recv),
                                     t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
    ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.recv_tokens.tensor,
                                    symm_window(sym_recv), t.result.tensor, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    std::vector<nv_bfloat16> h_res(num_tokens_ * hidden_dim_);
    CHECK_CUDA(cudaMemcpy(h_res.data(), buf.result.get(),
                          h_res.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    for (int tok = 0; tok < num_tokens_; ++tok) {
      float exp = __bfloat162float(h_tok[tok * hidden_dim_]) * static_cast<float>(top_k_);
      float got = __bfloat162float(h_res[tok * hidden_dim_]);
      ASSERT_NEAR(got, exp, bf16_tol(exp)) << "iter " << iter << " tok " << tok;
    }
  }

  if (g_process_id == 0)
    printf("  IdentityAllSymmRepeated: passed (2 iters)\n");

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// Full forward+backward with symm-mem on every spec-mandated buffer:
// dispatch i/o, combine input, combine_bwd i/o, dispatch_bwd input.
// TODO: flaky on rank 0 (grad_tokens partial-zero) when run after the prior
// EPZeroCopyTest cases in the same binary; passes in isolation. Re-enable once
// the root cause (likely NCCL EP NVLS write→read coherence on grad_expert) is
// understood. Tracked separately.
TEST_F(EPZeroCopyTest, DISABLED_FullPipelineSymm) {
  EPBuffers buf;
  buf.alloc(num_tokens_, top_k_, hidden_dim_, num_local_experts_,
            ep_size_, max_tokens_per_rank_);
  upload_inputs(buf);

  // Symm-mem: tokens (dispatch input), recv_tokens (dispatch output AND
  // combine input), grad_result (combine_bwd input), grad_expert
  // (combine_bwd output AND dispatch_bwd input).
  SymmBuf sym_tokens, sym_recv, sym_grad_result, sym_grad_expert;
  sym_tokens     .alloc(num_tokens_       * hidden_dim_ * sizeof(nv_bfloat16));
  sym_recv       .alloc(buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16));
  sym_grad_result.alloc(num_tokens_       * hidden_dim_ * sizeof(nv_bfloat16));
  sym_grad_expert.alloc(buf.recv_capacity * hidden_dim_ * sizeof(nv_bfloat16));

  auto h_tok = generate_tokens(g_process_id, num_tokens_, hidden_dim_);
  CHECK_CUDA(cudaMemcpy(sym_tokens.ptr, h_tok.data(),
                        h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  EPTensors t(buf, num_tokens_, top_k_, hidden_dim_, num_local_experts_);
  t.tokens       = make_nvte_tensor(sym_tokens.ptr,
                       {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  t.recv_tokens  = make_nvte_tensor(sym_recv.ptr,
                       {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  t.grad_result  = make_nvte_tensor(sym_grad_result.ptr,
                       {(size_t)num_tokens_, (size_t)hidden_dim_}, kNVTEBFloat16);
  t.grad_expert  = make_nvte_tensor(sym_grad_expert.ptr,
                       {buf.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  uint64_t handle_id = buf.handle_id;
  ASSERT_NO_THROW(nvte_ep_prepare(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor, t.token_counts.tensor, /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.topk_idx.tensor,
                                   t.tokens.tensor, symm_window(sym_tokens),
                                   t.topk_weights.tensor, NVTECommWindow{},
                                   t.recv_tokens.tensor, symm_window(sym_recv),
                                   t.recv_topk_weights.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.recv_tokens.tensor,
                                  symm_window(sym_recv), t.result.tensor, stream));

  std::vector<nv_bfloat16> h_grad(num_tokens_ * hidden_dim_, __float2bfloat16(0.1f));
  CHECK_CUDA(cudaMemcpyAsync(sym_grad_result.ptr, h_grad.data(),
                             h_grad.size() * sizeof(nv_bfloat16),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemsetAsync(sym_grad_expert.ptr,                0, sym_grad_expert.bytes,            stream));
  CHECK_CUDA(cudaMemsetAsync(buf.g_recv_topk_weights.get(), 0, buf.g_recv_topk_weights.bytes(), stream));
  CHECK_CUDA(cudaMemsetAsync(buf.grad_topk_weights.get(),   0, buf.grad_topk_weights.bytes(),   stream));

  ASSERT_NO_THROW(nvte_ep_combine_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_result.tensor,
                                      symm_window(sym_grad_result), t.grad_expert.tensor,
                                      symm_window(sym_grad_expert), stream));
  ASSERT_NO_THROW(nvte_ep_dispatch_bwd(NVTEEpHandle{handle_id, t.handle_mem.tensor}, t.grad_expert.tensor,
                                       symm_window(sym_grad_expert),
                                       t.g_recv_topk_weights.tensor, NVTECommWindow{},
                                       t.grad_tokens.tensor, t.grad_topk_weights.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  ASSERT_TRUE(check_no_nan_inf(buf.result.get(),      num_tokens_ * hidden_dim_, "result"));
  ASSERT_TRUE(check_no_nan_inf(buf.grad_tokens.get(), num_tokens_ * hidden_dim_, "grad_tokens"));

  std::vector<nv_bfloat16> h_gt(num_tokens_ * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_gt.data(), buf.grad_tokens.get(),
                        h_gt.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  const float kExpGrad = static_cast<float>(top_k_) * 0.1f;
  for (int tok = 0; tok < num_tokens_; ++tok)
    EXPECT_NEAR(__bfloat162float(h_gt[tok * hidden_dim_]), kExpGrad, bf16_tol(kExpGrad))
        << "grad_tokens token " << tok;

  if (g_process_id == 0) printf("  FullPipelineSymm: passed\n");

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (!ep_bootstrap(argc, argv)) return 0;
  int ret = RUN_ALL_TESTS();
  ep_teardown();
  return ret;
}
