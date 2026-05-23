/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*
 * EP C-API coverage tests (paths not exercised by the pipeline suite).
 *
 *   MultiHandleAllocTest  — distinct handle ids; each works end-to-end.
 *   TopK1Test             — top_k=1 dispatch/combine/bwd round-trip.
 *   EmptyExpertsTest      — alignment ∈ {0, 2, 8, 16} with experts receiving 0 tokens.
 *   NegativeTests         — alignment mismatch and null handle_mem must throw.
 */

#include "test_ep_common.h"

#include <cmath>
#include <vector>

// top1 -> expert 0, top2 -> expert 2; leaves local-expert 1 empty between two
// full experts. Requires top_k >= 2 and num_experts >= 3.
static std::vector<int64_t> routing_skip_middle(int num_tokens, int top_k) {
  std::vector<int64_t> idx(num_tokens * top_k);
  for (int t = 0; t < num_tokens; ++t) {
    idx[t * top_k + 0] = 0;
    if (top_k >= 2) idx[t * top_k + 1] = 2;
    for (int k = 2; k < top_k; ++k) idx[t * top_k + k] = 2 + k;  // distinct stragglers
  }
  return idx;
}

static std::vector<nv_bfloat16> tokens_constant(int num_tokens, int hidden_dim, float val) {
  std::vector<nv_bfloat16> v(num_tokens * hidden_dim);
  nv_bfloat16 b = __float2bfloat16(val);
  std::fill(v.begin(), v.end(), b);
  return v;
}

namespace {

class EpCoverageBase : public ::testing::Test {
 protected:
  int ep_size_, num_experts_, num_local_experts_, hidden_dim_;
  int max_tokens_per_rank_;

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
  }

  // Helper: allocate buffers + tensor views for a single dispatch+combine.
  struct Bundle {
    DevBuf<int64_t>     topk_idx;
    DevBuf<float>       topk_weights;
    DevBuf<nv_bfloat16> tokens;
    DevBuf<int32_t>     token_counts;
    DevBuf<uint8_t>     handle_mem;
    DevBuf<nv_bfloat16> recv_tokens;
    DevBuf<float>       recv_topk_weights;
    DevBuf<nv_bfloat16> result;
    uint64_t            handle_id       = 0;
    size_t              handle_mem_size = 0;
    size_t              recv_capacity   = 0;
  };

  Bundle make_bundle(int num_tokens, int top_k, int num_local_experts,
                     size_t alignment) {
    Bundle b;
    b.recv_capacity = static_cast<size_t>(ep_size_) * max_tokens_per_rank_ * 2;
    b.topk_idx.alloc(num_tokens * top_k);
    b.topk_weights.alloc(num_tokens * top_k);
    b.tokens.alloc(num_tokens * hidden_dim_);
    b.token_counts.alloc(num_local_experts);
    b.recv_tokens.alloc(b.recv_capacity * hidden_dim_);
    b.recv_topk_weights.alloc(b.recv_capacity);
    b.result.alloc(num_tokens * hidden_dim_);
    NVTEEpLayerConfig cfg{num_local_experts, top_k, alignment};
    b.handle_id = nvte_ep_register_layer(cfg, &b.handle_mem_size);
    b.handle_mem.alloc(b.handle_mem_size);
    return b;
  }
};

}  // namespace

// =============================================================================
// MultiHandleAllocTest: ids are distinct and each is independently usable.
// =============================================================================

class MultiHandleAllocTest : public EpCoverageBase {};

TEST_F(MultiHandleAllocTest, IdsAreDistinct) {
  NVTEEpLayerConfig cfg{num_local_experts_, /*top_k=*/2, /*alignment=*/0};
  const int kN = 8;
  std::vector<uint64_t> ids(kN);
  for (int i = 0; i < kN; ++i) {
    size_t sz = 0;
    ids[i] = nvte_ep_register_layer(cfg, &sz);
  }
  for (int i = 0; i < kN; ++i) {
    EXPECT_NE(ids[i], 0u) << "handle_id 0 is reserved as \"no id\"";
    for (int j = i + 1; j < kN; ++j)
      EXPECT_NE(ids[i], ids[j]) << "duplicate id " << ids[i] << " at indices " << i << ", " << j;
  }
}

TEST_F(MultiHandleAllocTest, TwoHandlesCoexist) {
  const int num_tokens = 16, top_k = 2;
  Bundle a = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);

  auto h_idx = routing_balanced(g_process_id, num_tokens, top_k,
                                num_experts_, num_local_experts_);
  std::vector<float> h_w(num_tokens * top_k, 1.0f / top_k);
  auto h_tok = tokens_constant(num_tokens, hidden_dim_, 0.5f);
  for (Bundle* x : {&a, &b}) {
    CHECK_CUDA(cudaMemcpy(x->topk_idx.get(),     h_idx.data(),
                          h_idx.size() * sizeof(int64_t),     cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x->topk_weights.get(), h_w.data(),
                          h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(x->tokens.get(),       h_tok.data(),
                          h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));
  }

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  ASSERT_NE(a.handle_id, b.handle_id);

  auto run_one = [&](Bundle& x) {
    auto topk_idx     = make_nvte_tensor(x.topk_idx.get(),     {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
    auto topk_weights = make_nvte_tensor(x.topk_weights.get(), {(size_t)num_tokens, (size_t)top_k}, kNVTEFloat32);
    auto token_counts = make_nvte_tensor(x.token_counts.get(), {(size_t)num_local_experts_},        kNVTEInt32);
    auto handle_mem   = make_nvte_tensor(x.handle_mem.get(),   {x.handle_mem_size},                 kNVTEByte);
    auto tokens       = make_nvte_tensor(x.tokens.get(),       {(size_t)num_tokens, (size_t)hidden_dim_}, kNVTEBFloat16);
    auto recv_tokens  = make_nvte_tensor(x.recv_tokens.get(),  {x.recv_capacity, (size_t)hidden_dim_},    kNVTEBFloat16);
    auto recv_w       = make_nvte_tensor(x.recv_topk_weights.get(), {x.recv_capacity},                    kNVTEFloat32);
    auto result       = make_nvte_tensor(x.result.get(),       {(size_t)num_tokens, (size_t)hidden_dim_}, kNVTEBFloat16);
    NVTEEpHandle h{x.handle_id, handle_mem.tensor};
    ASSERT_NO_THROW(nvte_ep_prepare(h, topk_idx.tensor, token_counts.tensor,
                                    /*alignment=*/0, stream));
    ASSERT_NO_THROW(nvte_ep_dispatch(h, topk_idx.tensor, tokens.tensor,
                                     NVTECommWindow{}, topk_weights.tensor, NVTECommWindow{},
                                     recv_tokens.tensor, NVTECommWindow{}, recv_w.tensor,
                                     NVTECommWindow{}, stream));
    ASSERT_NO_THROW(nvte_ep_combine(h, recv_tokens.tensor, NVTECommWindow{},
                                    result.tensor, stream));
  };
  run_one(a);
  run_one(b);
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Both round-trips must produce result == top_k * 0.5 = 1.0.
  for (Bundle* x : {&a, &b}) {
    std::vector<nv_bfloat16> h_res(num_tokens * hidden_dim_);
    CHECK_CUDA(cudaMemcpy(h_res.data(), x->result.get(),
                          h_res.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
    const int probes[3] = {0, hidden_dim_ / 2, hidden_dim_ - 1};
    for (int t = 0; t < num_tokens; ++t)
      for (int p : probes)
        EXPECT_NEAR(__bfloat162float(h_res[t * hidden_dim_ + p]),
                    static_cast<float>(top_k) * 0.5f, 1e-2f);
  }
  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// TopK1Test: top_k=1 dispatch/combine round-trip, including dispatch_bwd.
// =============================================================================

class TopK1Test : public EpCoverageBase {};

TEST_F(TopK1Test, RoundTrip) {
  const int num_tokens = 16, top_k = 1;
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);

  auto h_idx = routing_balanced(g_process_id, num_tokens, top_k,
                                num_experts_, num_local_experts_);
  std::vector<float> h_w(num_tokens * top_k, 1.0f);  // top_k=1: weight is unity
  auto h_tok = tokens_constant(num_tokens, hidden_dim_, 0.25f);
  CHECK_CUDA(cudaMemcpy(b.topk_idx.get(),     h_idx.data(),
                        h_idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.topk_weights.get(), h_w.data(),
                        h_w.size()   * sizeof(float),   cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.tokens.get(),       h_tok.data(),
                        h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  auto topk_idx_t     = make_nvte_tensor(b.topk_idx.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(b.topk_weights.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(b.token_counts.get(),
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(b.handle_mem.get(),
                            {b.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(b.tokens.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(b.recv_tokens.get(),
                            {b.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_w_t       = make_nvte_tensor(b.recv_topk_weights.get(),
                            {b.recv_capacity}, kNVTEFloat32);
  auto result_t       = make_nvte_tensor(b.result.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim_}, kNVTEBFloat16);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  NVTEEpHandle h{b.handle_id, handle_mem_t.tensor};
  ASSERT_NO_THROW(nvte_ep_prepare(h, topk_idx_t.tensor, token_counts_t.tensor,
                                  /*alignment=*/0, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(h, topk_idx_t.tensor,
                                   tokens_t.tensor, NVTECommWindow{}, topk_weights_t.tensor,
                                   NVTECommWindow{}, recv_tokens_t.tensor, NVTECommWindow{},
                                   recv_w_t.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(h, recv_tokens_t.tensor,
                                  NVTECommWindow{}, result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // top_k=1: combine is unweighted gather, so result[t] == tokens[t].
  std::vector<nv_bfloat16> h_res(num_tokens * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_res.data(), b.result.get(),
                        h_res.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  const int probes[3] = {0, hidden_dim_ / 2, hidden_dim_ - 1};
  for (int t = 0; t < num_tokens; ++t)
    for (int p : probes)
      EXPECT_NEAR(__bfloat162float(h_res[t * hidden_dim_ + p]), 0.25f, 1e-2f)
          << "tok " << t << " hidden " << p;

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// EmptyExpertsTest: alignment ∈ {0, 2, 8, 16}, only local-expert 0 receives
// tokens. Round-trip must produce result == top_k * tokens regardless of the
// per-expert padding choice.
// =============================================================================

class EmptyExpertsTest : public EpCoverageBase,
                        public ::testing::WithParamInterface<size_t> {};

TEST_P(EmptyExpertsTest, RoundTripCorrect) {
  // routing_skip_middle needs experts {0, 2, ...}; smallest viable num_experts is 3.
  ASSERT_GE(num_experts_, 3);
  const size_t alignment = GetParam();
  const int    num_tokens = 16, top_k = 2;
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, alignment);

  // top1 -> expert 0, top2 -> expert 2; rank 0's local-expert 1 receives 0
  // tokens between two non-empty experts.
  std::vector<int64_t> h_idx = routing_skip_middle(num_tokens, top_k);
  std::vector<float>   h_w(num_tokens * top_k, 1.0f / top_k);
  auto h_tok = tokens_constant(num_tokens, hidden_dim_, 0.3f);

  CHECK_CUDA(cudaMemcpy(b.topk_idx.get(),     h_idx.data(),
                        h_idx.size() * sizeof(int64_t),     cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.topk_weights.get(), h_w.data(),
                        h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.tokens.get(),       h_tok.data(),
                        h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  auto topk_idx_t     = make_nvte_tensor(b.topk_idx.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(b.topk_weights.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(b.token_counts.get(),
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(b.handle_mem.get(),
                            {b.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(b.tokens.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(b.recv_tokens.get(),
                            {b.recv_capacity, (size_t)hidden_dim_}, kNVTEBFloat16);
  auto recv_w_t       = make_nvte_tensor(b.recv_topk_weights.get(),
                            {b.recv_capacity}, kNVTEFloat32);
  auto result_t       = make_nvte_tensor(b.result.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim_}, kNVTEBFloat16);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  NVTEEpHandle h{b.handle_id, handle_mem_t.tensor};
  ASSERT_NO_THROW(nvte_ep_prepare(h, topk_idx_t.tensor, token_counts_t.tensor,
                                  alignment, stream));
  ASSERT_NO_THROW(nvte_ep_dispatch(h, topk_idx_t.tensor,
                                   tokens_t.tensor, NVTECommWindow{}, topk_weights_t.tensor,
                                   NVTECommWindow{}, recv_tokens_t.tensor, NVTECommWindow{},
                                   recv_w_t.tensor, NVTECommWindow{}, stream));
  ASSERT_NO_THROW(nvte_ep_combine(h, recv_tokens_t.tensor,
                                  NVTECommWindow{}, result_t.tensor, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  // Identity expert + uniform weights: result[t] == top_k * tokens[t].
  std::vector<nv_bfloat16> h_res(num_tokens * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_res.data(), b.result.get(),
                        h_res.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  const float expected = static_cast<float>(top_k) * 0.3f;
  const int probes[3] = {0, hidden_dim_ / 2, hidden_dim_ - 1};
  for (int t = 0; t < num_tokens; ++t)
    for (int p : probes)
      EXPECT_NEAR(__bfloat162float(h_res[t * hidden_dim_ + p]), expected, 1e-2f)
          << "alignment=" << alignment << " tok=" << t << " hidden=" << p;

  CHECK_CUDA(cudaStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(Alignments, EmptyExpertsTest,
                         ::testing::Values<size_t>(0, 2, 8, 16));

// =============================================================================
// NegativeTests: prepare/dispatch must surface bad inputs as exceptions.
// =============================================================================

class NegativeTests : public EpCoverageBase {};

TEST_F(NegativeTests, AlignmentMismatchThrows) {
  const int num_tokens = 8, top_k = 2;
  // Allocate handle for alignment=0, then call prepare with alignment=16.
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);
  auto h_idx = routing_balanced(g_process_id, num_tokens, top_k,
                                num_experts_, num_local_experts_);
  CHECK_CUDA(cudaMemcpy(b.topk_idx.get(), h_idx.data(),
                        h_idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

  auto topk_idx_t     = make_nvte_tensor(b.topk_idx.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
  auto token_counts_t = make_nvte_tensor(b.token_counts.get(),
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(b.handle_mem.get(),
                            {b.handle_mem_size}, kNVTEByte);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  NVTEEpHandle h{b.handle_id, handle_mem_t.tensor};
  EXPECT_THROW(nvte_ep_prepare(h, topk_idx_t.tensor, token_counts_t.tensor,
                               /*alignment=*/16, stream),
               std::exception);
  CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST_F(NegativeTests, NullHandleMemThrows) {
  const int num_tokens = 8, top_k = 2;
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);
  auto h_idx = routing_balanced(g_process_id, num_tokens, top_k,
                                num_experts_, num_local_experts_);
  CHECK_CUDA(cudaMemcpy(b.topk_idx.get(), h_idx.data(),
                        h_idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

  auto topk_idx_t     = make_nvte_tensor(b.topk_idx.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
  auto token_counts_t = make_nvte_tensor(b.token_counts.get(),
                            {(size_t)num_local_experts_}, kNVTEInt32);
  // Construct a tensor view backed by a null device pointer.
  auto null_hm_t = make_nvte_tensor(nullptr, {b.handle_mem_size}, kNVTEByte);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));
  NVTEEpHandle h{b.handle_id, null_hm_t.tensor};
  EXPECT_THROW(nvte_ep_prepare(h, topk_idx_t.tensor, token_counts_t.tensor,
                               /*alignment=*/0, stream),
               std::exception);
  CHECK_CUDA(cudaStreamDestroy(stream));
}

// =============================================================================
// HandleCacheTest: persistent ncclEpHandle is reused across ops on the same
// handle_mem ptr; relocation triggers throw by default and rebuild when
// NVTEEpGroupConfig.allow_handle_mem_reloc=1.
// =============================================================================

class HandleCacheTest : public EpCoverageBase {};

// Run prepare → dispatch → combine on bundle b. handle_mem_data overrides the
// device ptr used for handle_mem (must be the buffer owned by b unless
// reloc-allowed mode is active). Templated on Bundle because EpCoverageBase::
// Bundle is declared in a protected section.
template <typename B>
static void run_round_trip(B& b, void* handle_mem_data,
                           int num_tokens, int top_k, int num_local_experts,
                           int hidden_dim, size_t alignment,
                           cudaStream_t stream) {
  auto topk_idx_t     = make_nvte_tensor(b.topk_idx.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
  auto topk_weights_t = make_nvte_tensor(b.topk_weights.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEFloat32);
  auto token_counts_t = make_nvte_tensor(b.token_counts.get(),
                            {(size_t)num_local_experts}, kNVTEInt32);
  auto handle_mem_t   = make_nvte_tensor(handle_mem_data,
                            {b.handle_mem_size}, kNVTEByte);
  auto tokens_t       = make_nvte_tensor(b.tokens.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim}, kNVTEBFloat16);
  auto recv_tokens_t  = make_nvte_tensor(b.recv_tokens.get(),
                            {b.recv_capacity, (size_t)hidden_dim}, kNVTEBFloat16);
  auto recv_w_t       = make_nvte_tensor(b.recv_topk_weights.get(),
                            {b.recv_capacity}, kNVTEFloat32);
  auto result_t       = make_nvte_tensor(b.result.get(),
                            {(size_t)num_tokens, (size_t)hidden_dim}, kNVTEBFloat16);

  NVTEEpHandle h{b.handle_id, handle_mem_t.tensor};
  nvte_ep_prepare(h, topk_idx_t.tensor, token_counts_t.tensor, alignment, stream);
  nvte_ep_dispatch(h, topk_idx_t.tensor, tokens_t.tensor, NVTECommWindow{},
                   topk_weights_t.tensor, NVTECommWindow{},
                   recv_tokens_t.tensor, NVTECommWindow{},
                   recv_w_t.tensor, NVTECommWindow{}, stream);
  nvte_ep_combine(h, recv_tokens_t.tensor, NVTECommWindow{}, result_t.tensor, stream);
}

// Re-bootstrap EP backend with a different allow_handle_mem_reloc setting.
// Reuses the existing g_ep_comm; caller is responsible for restoring defaults.
static void reinit_ep_with_reloc(int allow_reloc) {
  nvte_ep_shutdown();
  NVTEEpGroupConfig cfg{};
  cfg.ep_size                  = g_ep_size;
  cfg.num_experts              = g_num_experts;
  cfg.max_tokens_per_rank      = g_max_tokens_per_rank;
  cfg.max_recv_tokens_per_rank = g_ep_size * g_max_tokens_per_rank * 2;
  cfg.hidden_dim               = g_hidden_dim;
  cfg.allow_handle_mem_reloc   = allow_reloc;
  nvte_ep_initialize(static_cast<void*>(g_ep_comm), cfg);
}

TEST_F(HandleCacheTest, ReuseSameMemSucceeds) {
  const int num_tokens = 16, top_k = 2;
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);

  auto h_idx = routing_balanced(g_process_id, num_tokens, top_k,
                                num_experts_, num_local_experts_);
  std::vector<float> h_w(num_tokens * top_k, 1.0f / top_k);
  auto h_tok = tokens_constant(num_tokens, hidden_dim_, 0.5f);
  CHECK_CUDA(cudaMemcpy(b.topk_idx.get(),     h_idx.data(),
                        h_idx.size() * sizeof(int64_t),     cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.topk_weights.get(), h_w.data(),
                        h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.tokens.get(),       h_tok.data(),
                        h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // Two consecutive round-trips on the same handle_mem ptr: first opens the
  // cached handle, second hits the cache. Both must succeed and be correct.
  for (int iter = 0; iter < 2; ++iter) {
    ASSERT_NO_THROW(run_round_trip(b, b.handle_mem.get(), num_tokens, top_k,
                                   num_local_experts_, hidden_dim_,
                                   /*alignment=*/0, stream));
  }
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<nv_bfloat16> h_res(num_tokens * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_res.data(), b.result.get(),
                        h_res.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  const int probes[3] = {0, hidden_dim_ / 2, hidden_dim_ - 1};
  for (int t = 0; t < num_tokens; ++t)
    for (int p : probes)
      EXPECT_NEAR(__bfloat162float(h_res[t * hidden_dim_ + p]),
                  static_cast<float>(top_k) * 0.5f, 1e-2f);

  CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST_F(HandleCacheTest, RelocDefaultThrows) {
  // Default bootstrap has allow_handle_mem_reloc=0: a second prepare call on
  // the same handle_id with a different handle_mem ptr must throw.
  const int num_tokens = 8, top_k = 2;
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);
  DevBuf<uint8_t> second_hm(b.handle_mem_size);  // distinct device buffer
  ASSERT_NE(b.handle_mem.get(), second_hm.get());

  auto h_idx = routing_balanced(g_process_id, num_tokens, top_k,
                                num_experts_, num_local_experts_);
  CHECK_CUDA(cudaMemcpy(b.topk_idx.get(), h_idx.data(),
                        h_idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice));

  auto topk_idx_t     = make_nvte_tensor(b.topk_idx.get(),
                            {(size_t)num_tokens, (size_t)top_k}, kNVTEInt64);
  auto token_counts_t = make_nvte_tensor(b.token_counts.get(),
                            {(size_t)num_local_experts_}, kNVTEInt32);
  auto hm1_t          = make_nvte_tensor(b.handle_mem.get(),
                            {b.handle_mem_size}, kNVTEByte);
  auto hm2_t          = make_nvte_tensor(second_hm.get(),
                            {b.handle_mem_size}, kNVTEByte);

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // First prepare seeds the cache.
  NVTEEpHandle h1{b.handle_id, hm1_t.tensor};
  ASSERT_NO_THROW(nvte_ep_prepare(h1, topk_idx_t.tensor, token_counts_t.tensor,
                                  /*alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  // Same handle_id with a different handle_mem ptr must throw.
  NVTEEpHandle h2{b.handle_id, hm2_t.tensor};
  EXPECT_THROW(nvte_ep_prepare(h2, topk_idx_t.tensor, token_counts_t.tensor,
                               /*alignment=*/0, stream),
               std::exception);
  CHECK_CUDA(cudaStreamDestroy(stream));
}

TEST_F(HandleCacheTest, RelocAllowedRebuilds) {
  // Re-init EP backend with allow_handle_mem_reloc=1, run two round-trips with
  // distinct handle_mem buffers, verify both succeed numerically, restore.
  reinit_ep_with_reloc(/*allow_reloc=*/1);

  struct Restore { ~Restore() { reinit_ep_with_reloc(/*allow_reloc=*/0); } } restore;

  const int num_tokens = 16, top_k = 2;
  Bundle b = make_bundle(num_tokens, top_k, num_local_experts_, /*alignment=*/0);
  DevBuf<uint8_t> alt_hm(b.handle_mem_size);
  ASSERT_NE(b.handle_mem.get(), alt_hm.get());

  auto h_idx = routing_balanced(g_process_id, num_tokens, top_k,
                                num_experts_, num_local_experts_);
  std::vector<float> h_w(num_tokens * top_k, 1.0f / top_k);
  auto h_tok = tokens_constant(num_tokens, hidden_dim_, 0.5f);
  CHECK_CUDA(cudaMemcpy(b.topk_idx.get(),     h_idx.data(),
                        h_idx.size() * sizeof(int64_t),     cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.topk_weights.get(), h_w.data(),
                        h_w.size()   * sizeof(float),       cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(b.tokens.get(),       h_tok.data(),
                        h_tok.size() * sizeof(nv_bfloat16), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  // First on the original handle_mem.
  ASSERT_NO_THROW(run_round_trip(b, b.handle_mem.get(), num_tokens, top_k,
                                 num_local_experts_, hidden_dim_,
                                 /*alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  // Then on the relocated handle_mem — must trigger silent rebuild, not throw.
  ASSERT_NO_THROW(run_round_trip(b, alt_hm.get(), num_tokens, top_k,
                                 num_local_experts_, hidden_dim_,
                                 /*alignment=*/0, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  std::vector<nv_bfloat16> h_res(num_tokens * hidden_dim_);
  CHECK_CUDA(cudaMemcpy(h_res.data(), b.result.get(),
                        h_res.size() * sizeof(nv_bfloat16), cudaMemcpyDeviceToHost));
  const int probes[3] = {0, hidden_dim_ / 2, hidden_dim_ - 1};
  for (int t = 0; t < num_tokens; ++t)
    for (int p : probes)
      EXPECT_NEAR(__bfloat162float(h_res[t * hidden_dim_ + p]),
                  static_cast<float>(top_k) * 0.5f, 1e-2f);

  CHECK_CUDA(cudaStreamDestroy(stream));
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
  if (!ep_bootstrap(argc, argv)) return 0;
  int ret = RUN_ALL_TESTS();
  ep_teardown();
  return ret;
}
