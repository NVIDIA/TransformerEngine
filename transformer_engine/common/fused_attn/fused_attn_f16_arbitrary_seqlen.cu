/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>

#include <map>
#include <mutex>  // [SHARED-CACHE]
#include <vector>

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_f16_arbitrary_seqlen.h"
#include "graph_cache_debug.h"  // [FUSED-ATTN-CACHE]
#include "utils.h"

namespace transformer_engine {
namespace fused_attn {

void fused_attn_arbitrary_seqlen_fwd_impl(
    const FusedAttnConfig &cfg, void *devPtrQ, void *devPtrK, void *devPtrV, void *devPtrBias,
    void *devPtrSoftmaxOffset, void *devPtrS1, void *devPtrS2, void *devPtrO,
    void *devPtrDropoutSeed, void *devPtrDropoutOffset, void *devPtrCuSeqlensQ,
    void *devPtrCuSeqlensKV, void *devPtrPageTableK, void *devPtrPageTableV,
    void *devPtrSeqOffsetsQ, void *devPtrSeqOffsetsKV, void *workspace, size_t *workspace_size,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const cudnn_frontend::DataType_t tensorType =
      get_cudnn_fe_dtype(static_cast<DType>(cfg.qkv_dtype));

  int64_t b = static_cast<int64_t>(cfg.batch_size);
  const int64_t h = static_cast<int64_t>(cfg.num_attn_heads);
  const int64_t hg = static_cast<int64_t>(cfg.num_gqa_groups);
  int64_t s_q = static_cast<int64_t>(cfg.max_seqlen_q);
  int64_t s_kv = static_cast<int64_t>(cfg.max_seqlen_kv);
  const int64_t d_qk = static_cast<int64_t>(cfg.head_dim_qk);
  const int64_t d_v = static_cast<int64_t>(cfg.head_dim_v);
  int64_t bucketed_batch_size = static_cast<int64_t>(cfg.bucketed_batch_size);
  int64_t bucketed_num_tokens_q = static_cast<int64_t>(cfg.bucketed_num_tokens_q);
  int64_t bucketed_num_tokens_kv = static_cast<int64_t>(cfg.bucketed_num_tokens_kv);
  int64_t num_pages_k = static_cast<int64_t>(cfg.num_pages_k);
  int64_t num_pages_v = static_cast<int64_t>(cfg.num_pages_v);
  int64_t page_size_k = static_cast<int64_t>(cfg.page_size_k);
  int64_t page_size_v = static_cast<int64_t>(cfg.page_size_v);
  int64_t max_pages_per_seq_k = static_cast<int64_t>(cfg.max_pages_per_seq_k);
  int64_t max_pages_per_seq_v = static_cast<int64_t>(cfg.max_pages_per_seq_v);
  int64_t bias_b = static_cast<int64_t>(cfg.bias_batch_size);
  int64_t bias_h = static_cast<int64_t>(cfg.bias_num_heads);
  int64_t bias_sq = static_cast<int64_t>(cfg.bias_seqlen_q);
  int64_t bias_skv = static_cast<int64_t>(cfg.bias_seqlen_kv);
  const bool is_training = cfg.is_training;
  const bool return_max_logit = cfg.return_max_logit;
  float scaling_factor = cfg.attn_scale;
  const float dropout_probability = cfg.dropout;
  const NVTE_QKV_Layout qkv_layout = cfg.qkv_layout;
  const NVTE_Bias_Type bias_type = cfg.bias_type;
  const NVTE_Mask_Type mask_type = cfg.attn_mask_type;
  const NVTE_Softmax_Type softmax_type = cfg.softmax_type;
  const int64_t window_size_left = cfg.window_size_left;
  const int64_t window_size_right = cfg.window_size_right;
  bool bottom_right_diagonal = cfg.bottom_right_diagonal;

  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_causal_bottom_right = cfg.is_causal_bottom_right;
  bool is_padding = cfg.is_padding;
  bool is_softmax_offset = (softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX);
  bool is_dropout = (is_training && dropout_probability != 0.0f);
  bool is_ragged_q = cfg.is_ragged_q;
  bool is_ragged_kv = cfg.is_ragged_kv;
  const auto cudnn_runtime_version = cudnnGetVersion();
  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);
  bool use_ragged_stats = is_ragged_q && cudnn_runtime_version >= 90600 && sm_arch_ != 120;

  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  bool is_paged_kv = cfg.is_paged_kv;
  if (is_paged_kv) {
    NVTE_CHECK(is_padding, "Paged attention requires padding mask!");
  }

  // Newer versions of cuDNN SDPA can accept sequence lengths directly as a cumulative
  // tensor, and can accept ragged offsets in arbitrary units (such as tokens) instead
  // of elements. Take advantage of this if possible to avoid 2 extra kernel calls.
  // Defined on FusedAttnConfig so make_cache_key() keys the graph on the matching batch
  // handling (real batch here, bucketed batch on the legacy path); keep the two in sync.
  const bool use_cu_seqlens_directly = cfg.uses_cu_seqlens_directly;

  // keep original batch size because cu_seqlens are created with [b+1] shape
  int64_t actual_b = b;
  if ((is_ragged_q || is_ragged_kv) && cudnn_runtime_version >= 90600) {
    NVTE_CHECK(is_padding, "Ragged QKV input requires padding or padding_causal mask!");
    // On SM 120, cuDNN support check treats layouts with stride[0] > dim[1]*dim[2]*dim[3]
    // as interleaved and rejects them. Use BHSD-like dimensions/strides with max_seqlen at plan build
    // so the check passes; ragged offset still provides variable-length boundaries.
    if (sm_arch_ != 120) {
      // replace batch size and maximum sequence lengths with maximum token counts
      // for query and key/value so the graph is static within each quantization bucket.
      // When passing cu_seqlens* directly to cuDNN SDPA, keep the true batch size:
      // cuDNN reads the user's [actual_b+1] cu_seqlens buffers, so a quantized batch
      // would read out of bounds.
      if (!use_cu_seqlens_directly) {
        b = bucketed_batch_size;
      }
      s_q = is_ragged_q ? bucketed_num_tokens_q : s_q;
      s_kv = is_ragged_kv ? bucketed_num_tokens_kv : s_kv;
    }
  }

  const DType ragged_offset_type =
      use_cu_seqlens_directly
          ? DType::kInt32  // cu_seqlens* are given to us as int32; keep it that way.
          : (cudnn_runtime_version >= 90500 ? DType::kInt64 : DType::kInt32);

  // Ragged offset multipliers (elements per token); shared with the legacy conversion
  // kernel (cu_seqlens_padded_to_offsets) so the two paths cannot drift apart.
  const RaggedOffsetMultipliers offset_mults(layout_group, h, hg, d_qk, d_v);

  bool generate_stats = true;  // Always return stats
  const FusedAttnConfig cache_cfg = cfg.make_cache_key();
  try {
    namespace fe = cudnn_frontend;
    using graph_and_tensors =
        std::tuple<std::shared_ptr<fe::graph::Graph>,
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // V
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // O
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // S1
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // S2
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q / cu_seq_len_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv / cu_seq_len_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // page_table_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // page_table_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FusedAttnConfig, graph_and_tensors>;
    // [SHARED-CACHE] Process-wide graph cache (was `static thread_local`) so a compiled graph
    // is reused across threads instead of rebuilt per thread. Safe because cuDNN >= 9.0 allows
    // concurrent execution of a shared plan and cudnn-frontend >= 1.25.0 has a thread-safe
    // execute(). The TE minimum-cuDNN-version bump that formalizes this requirement is a follow-up PR.
    static CacheType sdpa_f16_fprop_cache;
    static std::mutex sdpa_f16_fprop_cache_mutex;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType &cache, const FusedAttnConfig &descriptor) -> graph_and_tensors {
      // [SHARED-CACHE] Lock only the map lookup; copy the entry out and release before building
      // so concurrent first-misses on different keys build in parallel. graph->execute() runs
      // unlocked after get_graph() returns; built graphs are shared across threads.
      graph_and_tensors cached_graph{};
      bool cache_hit = false;
      {
        std::lock_guard<std::mutex> shared_cache_lock(sdpa_f16_fprop_cache_mutex);
        auto it = cache.find(descriptor);
        cache_hit = (it != cache.end());
        if (cache_hit) cached_graph = it->second;
      }
      graph_cache_debug::note_cache_lookup("fwd", cache_hit, cfg);  // [FUSED-ATTN-CACHE]
      if ((is_ragged_q || is_ragged_kv) && cudnn_runtime_version >= 90600 &&      // [FUSED-ATTN-CACHE]
          sm_arch_ != 120) {                                                     // [FUSED-ATTN-CACHE]
        graph_cache_debug::note_thd_lookup(                                 // [FUSED-ATTN-CACHE]
            "fwd", cache_hit, !cache_hit || graph_cache_debug::cache_disabled(),
            /*legacy=*/!use_cu_seqlens_directly);                                // [FUSED-ATTN-CACHE]
      }                                                                          // [FUSED-ATTN-CACHE]
      if (cache_hit && !graph_cache_debug::cache_disabled()) {     // [FUSED-ATTN-CACHE]
        return cached_graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(tensorType)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, attn_scale, softmax_offset;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> page_table_k, page_table_v;
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o,
          offset_stats;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d_qk, q_stride.data(), qkv_layout,
                            NVTE_QKV_Matrix::NVTE_Q_Matrix);
      if (is_paged_kv) {
        generateMatrixStrides(num_pages_k, hg, page_size_k, page_size_v, d_qk, k_stride.data(),
                              qkv_layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
        generateMatrixStrides(num_pages_v, hg, page_size_k, page_size_v, d_v, v_stride.data(),
                              qkv_layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
      } else {
        generateMatrixStrides(b, hg, s_q, s_kv, d_qk, k_stride.data(), qkv_layout,
                              NVTE_QKV_Matrix::NVTE_K_Matrix);
        generateMatrixStrides(b, hg, s_q, s_kv, d_v, v_stride.data(), qkv_layout,
                              NVTE_QKV_Matrix::NVTE_V_Matrix);
      }

      Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d_qk})
                                .set_stride(q_stride));
      if (is_ragged_q) {
        offset_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_q")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        Q->set_ragged_offset(offset_q);
        if (use_cu_seqlens_directly) {
          Q->set_ragged_offset_multiplier(offset_mults.q);
        }
      }
      K = mha_graph->tensor(fe::graph::Tensor_attributes().set_name("K").set_stride(k_stride));
      V = mha_graph->tensor(fe::graph::Tensor_attributes().set_name("V").set_stride(v_stride));
      if (is_paged_kv) {
        K->set_dim({num_pages_k, hg, page_size_k, d_qk});
        V->set_dim({num_pages_v, hg, page_size_v, d_v});
      } else if (is_ragged_kv) {
        offset_k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_k")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        offset_v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_v")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        K->set_dim({b, hg, s_kv, d_qk}).set_ragged_offset(offset_k);
        V->set_dim({b, hg, s_kv, d_v}).set_ragged_offset(offset_v);
        if (use_cu_seqlens_directly) {
          K->set_ragged_offset_multiplier(offset_mults.k);
          V->set_ragged_offset_multiplier(offset_mults.v);
        }
      } else {
        K->set_dim({b, hg, s_kv, d_qk});
        V->set_dim({b, hg, s_kv, d_v});
      }

      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      fe::graph::SDPA_attributes sdpa_options;
      sdpa_options = fe::graph::SDPA_attributes()
                         .set_name("flash_attention")
                         .set_generate_stats(generate_stats)
                         .set_attn_scale(attn_scale);

      fe::DiagonalAlignment_t const &diagonal_alignment =
          bottom_right_diagonal ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
                                : fe::DiagonalAlignment_t::TOP_LEFT;
      sdpa_options.set_diagonal_alignment(diagonal_alignment);
      if (cudnn_runtime_version >= 90200 && window_size_left != -1) {
        sdpa_options.set_diagonal_band_left_bound(window_size_left + 1);
      }
      if (cudnn_runtime_version >= 90600 && window_size_right != -1) {
        sdpa_options.set_diagonal_band_right_bound(window_size_right);
      }
      if (is_causal || is_causal_bottom_right) {
        sdpa_options.set_diagonal_band_right_bound(0);
      }

      sdpa_options.set_alibi_mask(is_alibi);

      if (is_bias) {
        bias = mha_graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("bias")
                .set_dim({bias_b, bias_h, bias_sq, bias_skv})
                .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
        sdpa_options.set_bias(bias);
      }

      if (is_padding) {
        if (use_cu_seqlens_directly) {
          // seq_q/seq_kv keep their tuple slots but hold (b+1)-shaped cu_seqlen tensors.
          seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("cu_seq_len_q")
                                        .set_dim({b + 1, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
          seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("cu_seq_len_kv")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::INT32));
          sdpa_options.set_padding_mask(is_padding)
              .set_cu_seq_len_q(seq_q)
              .set_cu_seq_len_kv(seq_kv);
          // cu_seq_len (and the ragged offset multiplier) are unified-engine-only.
          // Pin the implementation so an unsupported config fails with the unified
          // engine's specific error instead of auto-selection's generic failure.
          sdpa_options.set_implementation(fe::AttentionImplementation_t::UNIFIED);
        } else {
          seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("seq_q")
                                        .set_dim({b, 1, 1, 1})
                                        .set_stride({1, 1, 1, 1})
                                        .set_data_type(fe::DataType_t::INT32));
          seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("seq_kv")
                                         .set_dim({b, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::INT32));
          sdpa_options.set_padding_mask(is_padding).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
        }
      }

      if (is_paged_kv) {
        page_table_k =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("page_table_k")
                                  .set_dim({b, 1, max_pages_per_seq_k, 1})
                                  .set_stride({{max_pages_per_seq_k, max_pages_per_seq_v, 1, 1}})
                                  .set_data_type(fe::DataType_t::INT32));
        page_table_v =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("page_table_v")
                                  .set_dim({b, 1, max_pages_per_seq_v, 1})
                                  .set_stride({{max_pages_per_seq_v, max_pages_per_seq_v, 1, 1}})
                                  .set_data_type(fe::DataType_t::INT32));
        sdpa_options.set_paged_attention_k_table(page_table_k);
        sdpa_options.set_paged_attention_v_table(page_table_v);
        sdpa_options.set_paged_attention_max_seq_len_kv(static_cast<int32_t>(s_kv));
      }

      if (is_dropout) {
        dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("Seed")
                                             .set_dim({1, 1, 1, 1})
                                             .set_stride({1, 1, 1, 1})
                                             .set_data_type(fe::DataType_t::INT64));
        dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("Offset")
                                               .set_dim({1, 1, 1, 1})
                                               .set_stride({1, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::INT64));
        sdpa_options.set_dropout(dropout_probability, dropout_seed, dropout_offset);
      }

      if (is_softmax_offset) {
        softmax_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("softmax_offset")
                                               .set_dim({1, h, 1, 1})
                                               .set_stride({h, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::FLOAT));
        sdpa_options.set_sink_token(softmax_offset);
      }

      std::shared_ptr<fe::graph::Tensor_attributes> Max;
      if (use_ragged_stats) {
        offset_stats =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("offset_stats")
                                  .set_dim({b + 1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
      }
      if (return_max_logit) {
        Max = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Max")
                                    .set_dim({b, h, s_q, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));
        if (use_ragged_stats) {
          Max->set_stride({h * s_q, 1, h, 1}).set_ragged_offset(offset_stats);
          if (use_cu_seqlens_directly) {
            Max->set_ragged_offset_multiplier(offset_mults.stats);
          }
        } else {
          Max->set_stride({h * s_q, s_q, 1, 1});
        }
        sdpa_options.set_logit_max(Max);
      }

      auto [O, Stats] = mha_graph->sdpa(Q, K, V, std::move(sdpa_options));

      std::vector<int64_t> o_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d_v, o_stride.data(), qkv_layout,
                            NVTE_QKV_Matrix::NVTE_O_Matrix);
      O->set_output(true).set_dim({b, h, s_q, d_v}).set_stride(o_stride);
      if (is_ragged_q) {
        offset_o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_o")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        O->set_ragged_offset(offset_o);
        if (use_cu_seqlens_directly) {
          O->set_ragged_offset_multiplier(offset_mults.o);
        }
      }

      Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({b, h, s_q, 1});
      if (use_ragged_stats) {
        Stats->set_stride({h * s_q, 1, h, 1}).set_ragged_offset(offset_stats);
        if (use_cu_seqlens_directly) {
          Stats->set_ragged_offset_multiplier(offset_mults.stats);
        }
      } else {
        Stats->set_stride({h * s_q, s_q, 1, 1});
      }

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // O
          key_tensors_tuple = std::make_tuple(Q, K, V, attn_scale, O);
      auto Stats_tuple =
          return_max_logit ? std::make_tuple(Stats, Max) : std::make_tuple(Stats, nullptr);
      auto bias_tuple = is_bias ? std::make_tuple(bias) : std::make_tuple(nullptr);
      auto softmax_offset_tuple =
          is_softmax_offset ? std::make_tuple(softmax_offset) : std::make_tuple(nullptr);
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto page_table_tuple = is_paged_kv ? std::make_tuple(page_table_k, page_table_v)
                                          : std::make_tuple(nullptr, nullptr);
      auto offset_qo_tuple =
          is_ragged_q ? std::make_tuple(offset_q, offset_o) : std::make_tuple(nullptr, nullptr);
      auto offset_kv_tuple =
          is_ragged_kv ? std::make_tuple(offset_k, offset_v) : std::make_tuple(nullptr, nullptr);
      auto offset_s_tuple =
          use_ragged_stats ? std::make_tuple(offset_stats) : std::make_tuple(nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support());  // no-handle overload (handle version is deprecated)
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans());    // no-handle overload (handle version is deprecated)

      auto return_tuple =
          std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, Stats_tuple, bias_tuple,
                         softmax_offset_tuple, padding_tuple, page_table_tuple, offset_qo_tuple,
                         offset_kv_tuple, offset_s_tuple, dropout_tuple);
      graph_cache_debug::note_fwd_build();  // [FUSED-ATTN-CACHE]
      // [SHARED-CACHE] Lock only for insert. If another thread inserted this key while we built,
      // reuse theirs and discard ours so all threads share one graph (rare duplicate build).
      {
        std::lock_guard<std::mutex> shared_cache_lock(sdpa_f16_fprop_cache_mutex);
        auto inserted = cache.insert({descriptor, return_tuple});
        return graph_cache_debug::cache_disabled() ? return_tuple : inserted.first->second;
      }
    };

    auto [mha_graph, Q, K, V, attn_scale, O, S1, S2, bias, softmax_offset, seq_q, seq_kv,
          page_table_k, page_table_v, offset_q, offset_o, offset_k, offset_v, offset_stats,
          dropout_seed, dropout_offset] = get_graph(sdpa_f16_fprop_cache, cache_cfg);

    // Exit to request upper level API to allocate memory if needed
    // n.b. Care should be taken to align each of the added worksapce tensors to their type.
    // We do this by adding padding at the end of each separate allocation.
    // When passing cu_seqlens* directly to cuDNN SDPA, no conversion workspace is
    // needed: cuDNN consumes the user's cu_seqlens buffers as-is.
    auto plan_workspace_size = alignTo<16>(mha_graph->get_workspace_size());
    const size_t num_bytes_per_seqlen = alignTo<16>(b * sizeof(int32_t));
    const size_t num_bytes_per_ragged_offset =
        alignTo<16>(((b + 1) * typeToNumBits(ragged_offset_type)) / 8);
    size_t actual_seqlen_workspace_size = 0;
    size_t seqlen_offsets_workspace_size = 0;
    if (!use_cu_seqlens_directly) {
      if (is_padding) {
        actual_seqlen_workspace_size = 2 * num_bytes_per_seqlen;
      }
      if (is_ragged_q || is_ragged_kv) {
        const size_t count =
            2 * (static_cast<size_t>(is_ragged_q) + static_cast<size_t>(is_ragged_kv));
        seqlen_offsets_workspace_size =
            (use_ragged_stats ? count + 1 : count) * num_bytes_per_ragged_offset;
      }
    }
    if (workspace == nullptr) {
      *workspace_size =
          plan_workspace_size + actual_seqlen_workspace_size + seqlen_offsets_workspace_size;
      return;
    }
    graph_cache_debug::note_fwd_exec();  // [FUSED-ATTN-CACHE]

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK},  {V, devPtrV}, {attn_scale, &scaling_factor},
        {O, devPtrO}, {S1, devPtrS1}};

    if (return_max_logit) {
      variant_pack[S2] = devPtrS2;
    }

    if (is_bias) {
      variant_pack[bias] = devPtrBias;
    }

    if (is_padding) {
      if (use_cu_seqlens_directly) {
        variant_pack[seq_q] = devPtrCuSeqlensQ;
        variant_pack[seq_kv] = devPtrCuSeqlensKV;
      } else {
        constexpr size_t nthreads_per_block = 128;
        const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
        void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
        void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + num_bytes_per_seqlen;
        cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
            actual_b, b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
            static_cast<const int32_t *>(devPtrCuSeqlensKV),
            static_cast<int32_t *>(devActualSeqlenQ), static_cast<int32_t *>(devActualSeqlenKV));
        NVTE_CHECK_CUDA(cudaGetLastError());
        variant_pack[seq_q] = devActualSeqlenQ;
        variant_pack[seq_kv] = devActualSeqlenKV;
      }
    }

    if (is_paged_kv) {
      variant_pack[page_table_k] = devPtrPageTableK;
      variant_pack[page_table_v] = devPtrPageTableV;
    }

    if (use_cu_seqlens_directly) {
      // The token-unit cu_seqlens_padded buffers serve as the ragged offsets; the engine
      // applies the per-tensor multipliers set at graph build time.
      if (is_ragged_q) {
        variant_pack[offset_q] = devPtrSeqOffsetsQ;
        variant_pack[offset_o] = devPtrSeqOffsetsQ;
      }
      if (is_ragged_kv) {
        void *devOffsetsKV = offset_mults.kv_from_q ? devPtrSeqOffsetsQ : devPtrSeqOffsetsKV;
        variant_pack[offset_k] = devOffsetsKV;
        variant_pack[offset_v] = devOffsetsKV;
      }
      if (use_ragged_stats) {
        variant_pack[offset_stats] = devPtrSeqOffsetsQ;
      }
    } else if (is_ragged_q || is_ragged_kv) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block) / nthreads_per_block;
      void *devOffsets =
          static_cast<int8_t *>(workspace) + plan_workspace_size + actual_seqlen_workspace_size;
      void *devOffsetsQ = nullptr;
      void *devOffsetsO = nullptr;
      if (is_ragged_q) {
        devOffsetsQ = devOffsets;
        devOffsetsO = static_cast<int8_t *>(devOffsetsQ) + num_bytes_per_ragged_offset;
      }
      void *devOffsetsK = nullptr;
      void *devOffsetsV = nullptr;
      if (is_ragged_kv) {
        devOffsetsK = static_cast<int8_t *>(devOffsets) +
                      static_cast<int>(is_ragged_q) * 2 * num_bytes_per_ragged_offset;
        devOffsetsV = static_cast<int8_t *>(devOffsetsK) + num_bytes_per_ragged_offset;
      }
      void *devOffsetsS = nullptr;
      if (use_ragged_stats) {
        devOffsetsS = static_cast<int8_t *>(devOffsets) +
                      (static_cast<int>(is_ragged_q) + static_cast<int>(is_ragged_kv)) * 2 *
                          num_bytes_per_ragged_offset;
      }
      cu_seqlens_padded_to_offsets<<<grid, nthreads_per_block, 0, stream>>>(
          offset_mults, actual_b, b, static_cast<int32_t *>(devPtrSeqOffsetsQ),
          static_cast<int32_t *>(devPtrSeqOffsetsKV), ragged_offset_type, devOffsetsQ, devOffsetsK,
          devOffsetsV, devOffsetsO, devOffsetsS);
      NVTE_CHECK_CUDA(cudaGetLastError());
      if (is_ragged_q) {
        variant_pack[offset_q] = devOffsetsQ;
        variant_pack[offset_o] = devOffsetsO;
      }
      if (is_ragged_kv) {
        variant_pack[offset_k] = devOffsetsK;
        variant_pack[offset_v] = devOffsetsV;
      }
      if (use_ragged_stats) {
        variant_pack[offset_stats] = devOffsetsS;
      }
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    if (is_softmax_offset) {
      variant_pack[softmax_offset] = devPtrSoftmaxOffset;
    }

    NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException &e) {
    NVTE_ERROR(e.what());
  }
}  // NOLINT(readability/fn_size)

void fused_attn_arbitrary_seqlen_bwd_impl(
    const FusedAttnConfig &cfg, void *devPtrQ, void *devPtrKTranspose, void *devPtrVTranspose,
    void *devPtrO, void *devPtrSoftmaxStats, void *devPtrBias, void *devPtrSoftmaxOffset,
    void *devPtrdQ, void *devPtrdK, void *devPtrdV, void *devPtrdO, void *devPtrdBias,
    void *devPtrdSoftmaxOffset, void *devPtrDropoutSeed, void *devPtrDropoutOffset,
    void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV, void *devPtrSeqOffsetsQ,
    void *devPtrSeqOffsetsKV, void *workspace, size_t *workspace_size, cudaStream_t stream,
    cudnnHandle_t handle) {
  using namespace transformer_engine;

  const cudnn_frontend::DataType_t tensorType =
      get_cudnn_fe_dtype(static_cast<DType>(cfg.qkv_dtype));

  int64_t b = static_cast<int64_t>(cfg.batch_size);
  const int64_t h = static_cast<int64_t>(cfg.num_attn_heads);
  const int64_t hg = static_cast<int64_t>(cfg.num_gqa_groups);
  int64_t s_q = static_cast<int64_t>(cfg.max_seqlen_q);
  int64_t s_kv = static_cast<int64_t>(cfg.max_seqlen_kv);
  const int64_t d_qk = static_cast<int64_t>(cfg.head_dim_qk);
  const int64_t d_v = static_cast<int64_t>(cfg.head_dim_v);
  int64_t bucketed_batch_size = static_cast<int64_t>(cfg.bucketed_batch_size);
  int64_t bucketed_num_tokens_q = static_cast<int64_t>(cfg.bucketed_num_tokens_q);
  int64_t bucketed_num_tokens_kv = static_cast<int64_t>(cfg.bucketed_num_tokens_kv);
  int64_t bias_b = static_cast<int64_t>(cfg.bias_batch_size);
  int64_t bias_h = static_cast<int64_t>(cfg.bias_num_heads);
  int64_t bias_sq = static_cast<int64_t>(cfg.bias_seqlen_q);
  int64_t bias_skv = static_cast<int64_t>(cfg.bias_seqlen_kv);
  float scaling_factor = cfg.attn_scale;
  const float dropout_probability = cfg.dropout;
  const NVTE_QKV_Layout qkv_layout = cfg.qkv_layout;
  const NVTE_Bias_Type bias_type = cfg.bias_type;
  const NVTE_Mask_Type mask_type = cfg.attn_mask_type;
  const NVTE_Softmax_Type softmax_type = cfg.softmax_type;
  const int64_t window_size_left = cfg.window_size_left;
  const int64_t window_size_right = cfg.window_size_right;
  bool bottom_right_diagonal = cfg.bottom_right_diagonal;
  const bool deterministic = cfg.deterministic;

  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_causal_bottom_right = cfg.is_causal_bottom_right;
  bool is_padding = cfg.is_padding;
  bool is_softmax_offset = (softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX);
  bool is_dropout = (dropout_probability != 0.0f);
  bool is_ragged_q = cfg.is_ragged_q;
  bool is_ragged_kv = cfg.is_ragged_kv;
  const auto cudnn_runtime_version = cudnnGetVersion();
  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);
  bool use_ragged_stats = is_ragged_q && cudnn_runtime_version >= 90600 && sm_arch_ != 120;

  // keep original batch size because cu_seqlens are created with [b+1] shape
  int64_t actual_b = b;
  if ((is_ragged_q || is_ragged_kv) && cudnn_runtime_version >= 90600) {
    NVTE_CHECK(is_padding, "Ragged QKV input requires padding or padding_causal mask!");
    // On SM 120, cuDNN support check requires BHSD-like strides with max_seqlen (see fwd).
    if (sm_arch_ != 120) {
      // replace batch size and maximum sequence lengths with maximum token counts
      // for query and key/value so the graph is static within each quantization bucket
      b = bucketed_batch_size;
      s_q = is_ragged_q ? bucketed_num_tokens_q : s_q;
      s_kv = is_ragged_kv ? bucketed_num_tokens_kv : s_kv;
    }
  }
  // We choose between 32-bit and 64-bit offsets depending on need.
  // This allows us to support older cuDNN runtimes gracefully.
  const DType ragged_offset_type = cudnn_runtime_version >= 90500 ? DType::kInt64 : DType::kInt32;
  const FusedAttnConfig cache_cfg = cfg.make_cache_key();

  try {
    namespace fe = cudnn_frontend;
    using graph_and_tensors =
        std::tuple<std::shared_ptr<fe::graph::Graph>,
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dO
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dQ
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dK
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dV
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dBias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // d_softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FusedAttnConfig, graph_and_tensors>;
    static CacheType sdpa_f16_bprop_cache;         // [SHARED-CACHE] process-wide (was thread_local)
    static std::mutex sdpa_f16_bprop_cache_mutex;  // [SHARED-CACHE]

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType &cache, const FusedAttnConfig &descriptor) -> graph_and_tensors {
      // [SHARED-CACHE] Lock only the map lookup; copy the entry out and release before building
      // so concurrent first-misses on different keys build in parallel. graph->execute() runs
      // unlocked after get_graph() returns; built graphs are shared across threads.
      graph_and_tensors cached_graph{};
      bool cache_hit = false;
      {
        std::lock_guard<std::mutex> shared_cache_lock(sdpa_f16_bprop_cache_mutex);
        auto it = cache.find(descriptor);
        cache_hit = (it != cache.end());
        if (cache_hit) cached_graph = it->second;
      }
      graph_cache_debug::note_cache_lookup("bwd", cache_hit, cfg);  // [FUSED-ATTN-CACHE]
      if ((is_ragged_q || is_ragged_kv) && cudnn_runtime_version >= 90600 &&      // [FUSED-ATTN-CACHE]
          sm_arch_ != 120) {                                                     // [FUSED-ATTN-CACHE]
        // The backward impl has no cu_seqlens-directly path; it always buckets the batch.
        graph_cache_debug::note_thd_lookup(  // [FUSED-ATTN-CACHE]
            "bwd", cache_hit, !cache_hit || graph_cache_debug::cache_disabled(),
            /*legacy=*/true);                                                    // [FUSED-ATTN-CACHE]
      }                                                                          // [FUSED-ATTN-CACHE]
      if (cache_hit && !graph_cache_debug::cache_disabled()) {     // [FUSED-ATTN-CACHE]
        return cached_graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(tensorType)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> q, k, v, o, dO, stats, attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, dBias, softmax_offset, d_softmax_offset,
          seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o,
          offset_stats;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      std::vector<int64_t> o_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d_qk, q_stride.data(), qkv_layout,
                            NVTE_QKV_Matrix::NVTE_Q_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d_qk, k_stride.data(), qkv_layout,
                            NVTE_QKV_Matrix::NVTE_K_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d_v, v_stride.data(), qkv_layout,
                            NVTE_QKV_Matrix::NVTE_V_Matrix);
      generateMatrixStrides(b, h, s_q, s_kv, d_v, o_stride.data(), qkv_layout,
                            NVTE_QKV_Matrix::NVTE_O_Matrix);

      q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d_qk})
                                .set_stride(q_stride));
      k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_dim({b, hg, s_kv, d_qk})
                                .set_stride(k_stride));
      v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_dim({b, hg, s_kv, d_v})
                                .set_stride(v_stride));
      o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("O")
                                .set_dim({b, h, s_q, d_v})
                                .set_stride(o_stride));
      dO = mha_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("dO")
                                 .set_dim({b, h, s_q, d_v})
                                 .set_stride(o_stride));
      if (is_ragged_q) {
        offset_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_q")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        offset_o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_o")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        q->set_ragged_offset(offset_q);
        o->set_ragged_offset(offset_o);
        dO->set_ragged_offset(offset_o);
      }
      if (is_ragged_kv) {
        offset_k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_k")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        offset_v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_v")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        k->set_ragged_offset(offset_k);
        v->set_ragged_offset(offset_v);
      }

      stats = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("stats")
                                    .set_dim({b, h, s_q, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));
      if (use_ragged_stats) {
        offset_stats =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("offset_stats")
                                  .set_dim({b + 1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        stats->set_stride({h * s_q, 1, h, 1}).set_ragged_offset(offset_stats);
      } else {
        stats->set_stride({h * s_q, s_q, 1, 1});
      }

      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      fe::graph::SDPA_backward_attributes sdpa_backward_options;
      sdpa_backward_options = fe::graph::SDPA_backward_attributes()
                                  .set_name("flash_attention_backward")
                                  .set_attn_scale(attn_scale);

      if (use_ragged_stats) {
        sdpa_backward_options.set_max_total_seq_len_q(s_q);
      }
      if (is_ragged_kv && cudnn_runtime_version >= 90600 && sm_arch_ != 120) {
        sdpa_backward_options.set_max_total_seq_len_kv(s_kv);
      }

      fe::DiagonalAlignment_t const &diagonal_alignment =
          bottom_right_diagonal ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
                                : fe::DiagonalAlignment_t::TOP_LEFT;
      sdpa_backward_options.set_diagonal_alignment(diagonal_alignment);

      if (cudnn_runtime_version >= 90200 && window_size_left != -1) {
        sdpa_backward_options.set_diagonal_band_left_bound(window_size_left + 1);
      }
      if (cudnn_runtime_version >= 90600 && window_size_right != -1) {
        sdpa_backward_options.set_diagonal_band_right_bound(window_size_right);
      }
      if (is_causal || is_causal_bottom_right) {
        sdpa_backward_options.set_diagonal_band_right_bound(0);
      }

      if (cudnn_runtime_version >= 90000) {
        sdpa_backward_options.set_deterministic_algorithm(deterministic);
      }

      sdpa_backward_options.set_alibi_mask(is_alibi);

      if (is_bias) {
        bias = mha_graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("bias")
                .set_dim({bias_b, bias_h, bias_sq, bias_skv})
                .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
        sdpa_backward_options.set_bias(bias);
        // bias shapes [1, 1, s, s], [b, 1, s, s], [b, h, s, s], [1, h, s, s] are supported for dbias calculation
        // bias shape [1, 1, 1, s] is not supported for dbias calculation as of cuDNN 9.18
        if (!((bias_b == 1) && (bias_h == 1) && (bias_sq == 1))) {
          dBias = mha_graph->tensor(
              fe::graph::Tensor_attributes()
                  .set_name("dBias")
                  .set_dim({bias_b, bias_h, bias_sq, bias_skv})
                  .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
          sdpa_backward_options.set_dbias(dBias);
        }
      }

      if (is_padding) {
        seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("seq_q")
                                      .set_dim({b, 1, 1, 1})
                                      .set_stride({1, 1, 1, 1})
                                      .set_data_type(fe::DataType_t::INT32));
        seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("seq_kv")
                                       .set_dim({b, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
        sdpa_backward_options.set_padding_mask(is_padding)
            .set_seq_len_q(seq_q)
            .set_seq_len_kv(seq_kv);
      }

      if (is_dropout) {
        dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
                                             .set_name("Seed")
                                             .set_dim({1, 1, 1, 1})
                                             .set_stride({1, 1, 1, 1})
                                             .set_data_type(fe::DataType_t::INT64));
        dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("Offset")
                                               .set_dim({1, 1, 1, 1})
                                               .set_stride({1, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::INT64));
        sdpa_backward_options.set_dropout(dropout_probability, dropout_seed, dropout_offset);
      }

      if (is_softmax_offset) {
        softmax_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                               .set_name("softmax_offset")
                                               .set_dim({1, h, 1, 1})
                                               .set_stride({h, 1, 1, 1})
                                               .set_data_type(fe::DataType_t::FLOAT));
        sdpa_backward_options.set_sink_token(softmax_offset);
        d_softmax_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
                                                 .set_name("d_softmax_offset")
                                                 .set_dim({1, h, 1, 1})
                                                 .set_stride({h, 1, 1, 1})
                                                 .set_data_type(fe::DataType_t::FLOAT));
        sdpa_backward_options.set_dsink_token(d_softmax_offset);
      }

      auto [dQ, dK, dV] = mha_graph->sdpa_backward(q, k, v, o, dO, stats, sdpa_backward_options);

      dQ->set_output(true).set_dim({b, h, s_q, d_qk}).set_stride(q_stride);
      dK->set_output(true).set_dim({b, hg, s_kv, d_qk}).set_stride(k_stride);
      dV->set_output(true).set_dim({b, hg, s_kv, d_v}).set_stride(v_stride);
      if (is_ragged_q) {
        dQ->set_ragged_offset(offset_q);
      }
      if (is_ragged_kv) {
        dK->set_ragged_offset(offset_k);
        dV->set_ragged_offset(offset_v);
      }

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // k
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // v
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // o
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // stats
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // dV
          key_tensors_tuple = std::make_tuple(q, k, v, o, dO, stats, attn_scale, dQ, dK, dV);
      auto bias_tuple = is_bias ? std::make_tuple(bias, dBias) : std::make_tuple(nullptr, nullptr);
      auto softmax_offset_tuple = is_softmax_offset
                                      ? std::make_tuple(softmax_offset, d_softmax_offset)
                                      : std::make_tuple(nullptr, nullptr);
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto offset_qo_tuple =
          is_ragged_q ? std::make_tuple(offset_q, offset_o) : std::make_tuple(nullptr, nullptr);
      auto offset_kv_tuple =
          is_ragged_kv ? std::make_tuple(offset_k, offset_v) : std::make_tuple(nullptr, nullptr);
      auto offset_s_tuple =
          use_ragged_stats ? std::make_tuple(offset_stats) : std::make_tuple(nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support());  // no-handle overload (handle version is deprecated)
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans());    // no-handle overload (handle version is deprecated)

      auto return_tuple = std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, bias_tuple,
                                         softmax_offset_tuple, padding_tuple, offset_qo_tuple,
                                         offset_kv_tuple, offset_s_tuple, dropout_tuple);
      graph_cache_debug::note_bwd_build();  // [FUSED-ATTN-CACHE]
      // [SHARED-CACHE] Lock only for insert. If another thread inserted this key while we built,
      // reuse theirs and discard ours so all threads share one graph (rare duplicate build).
      {
        std::lock_guard<std::mutex> shared_cache_lock(sdpa_f16_bprop_cache_mutex);
        auto inserted = cache.insert({descriptor, return_tuple});
        return graph_cache_debug::cache_disabled() ? return_tuple : inserted.first->second;
      }
    };

    auto [mha_graph, q, k, v, o, dO, stats, attn_scale, dQ, dK, dV, bias, dBias, softmax_offset,
          d_softmax_offset, seq_q, seq_kv, offset_q, offset_o, offset_k, offset_v, offset_stats,
          dropout_seed, dropout_offset] = get_graph(sdpa_f16_bprop_cache, cache_cfg);

    // Exit to request upper level API to allocate memory if needed
    // n.b. Care should be taken to align each of the added worksapce tensors to their type.
    // We do this by adding padding at the end of each separate allocation.
    auto plan_workspace_size = alignTo<16>(mha_graph->get_workspace_size());
    const size_t num_bytes_per_seqlen = alignTo<16>(b * sizeof(int32_t));
    const size_t actual_seqlen_workspace_size = is_padding ? 2 * num_bytes_per_seqlen : 0;
    const size_t num_bytes_per_ragged_offset =
        alignTo<16>(((b + 1) * typeToNumBits(ragged_offset_type)) / 8);
    size_t seqlen_offsets_workspace_size = 0;
    if (is_ragged_q || is_ragged_kv) {
      size_t count = 2 * (static_cast<size_t>(is_ragged_q) + static_cast<size_t>(is_ragged_kv));
      if (use_ragged_stats) {
        seqlen_offsets_workspace_size = (count + 1) * num_bytes_per_ragged_offset;
      } else {
        seqlen_offsets_workspace_size = count * num_bytes_per_ragged_offset;
      }
    }
    if (workspace == nullptr) {
      *workspace_size =
          plan_workspace_size + actual_seqlen_workspace_size + seqlen_offsets_workspace_size;
      return;
    }
    graph_cache_debug::note_bwd_exec();  // [FUSED-ATTN-CACHE]

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {q, devPtrQ},
        {k, devPtrKTranspose},
        {v, devPtrVTranspose},
        {o, devPtrO},
        {dO, devPtrdO},
        {stats, devPtrSoftmaxStats},
        {attn_scale, &scaling_factor},
        {dQ, devPtrdQ},
        {dK, devPtrdK},
        {dV, devPtrdV},
    };

    if (is_bias) {
      variant_pack[bias] = devPtrBias;
      if (dBias != nullptr) {
        variant_pack[dBias] = devPtrdBias;
      }
    }

    if (is_padding) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
      void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
      void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + num_bytes_per_seqlen;
      cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
          actual_b, b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
          static_cast<const int32_t *>(devPtrCuSeqlensKV), static_cast<int32_t *>(devActualSeqlenQ),
          static_cast<int32_t *>(devActualSeqlenKV));
      NVTE_CHECK_CUDA(cudaGetLastError());
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;
    }

    if (is_ragged_q || is_ragged_kv) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block) / nthreads_per_block;
      void *devOffsets =
          static_cast<int8_t *>(workspace) + plan_workspace_size + actual_seqlen_workspace_size;
      void *devOffsetsQ = nullptr;
      void *devOffsetsO = nullptr;
      if (is_ragged_q) {
        devOffsetsQ = devOffsets;
        devOffsetsO = static_cast<int8_t *>(devOffsetsQ) + num_bytes_per_ragged_offset;
      }
      void *devOffsetsK = nullptr;
      void *devOffsetsV = nullptr;
      if (is_ragged_kv) {
        devOffsetsK = static_cast<int8_t *>(devOffsets) +
                      static_cast<int>(is_ragged_q) * 2 * num_bytes_per_ragged_offset;
        devOffsetsV = static_cast<int8_t *>(devOffsetsK) + num_bytes_per_ragged_offset;
      }
      void *devOffsetsS = nullptr;
      if (use_ragged_stats) {
        devOffsetsS = static_cast<int8_t *>(devOffsets) +
                      (static_cast<int>(is_ragged_q) + static_cast<int>(is_ragged_kv)) * 2 *
                          num_bytes_per_ragged_offset;
      }
      const RaggedOffsetMultipliers offset_mults(nvte_get_qkv_layout_group(qkv_layout), h, hg, d_qk,
                                                 d_v);
      cu_seqlens_padded_to_offsets<<<grid, nthreads_per_block, 0, stream>>>(
          offset_mults, actual_b, b, static_cast<int32_t *>(devPtrSeqOffsetsQ),
          static_cast<int32_t *>(devPtrSeqOffsetsKV), ragged_offset_type, devOffsetsQ, devOffsetsK,
          devOffsetsV, devOffsetsO, devOffsetsS);
      NVTE_CHECK_CUDA(cudaGetLastError());
      if (is_ragged_q) {
        variant_pack[offset_q] = devOffsetsQ;
        variant_pack[offset_o] = devOffsetsO;
      }
      if (is_ragged_kv) {
        variant_pack[offset_k] = devOffsetsK;
        variant_pack[offset_v] = devOffsetsV;
      }
      if (use_ragged_stats) {
        variant_pack[offset_stats] = devOffsetsS;
      }
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    if (is_softmax_offset) {
      variant_pack[softmax_offset] = devPtrSoftmaxOffset;
      variant_pack[d_softmax_offset] = devPtrdSoftmaxOffset;
    }

    NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException &e) {
    NVTE_ERROR(e.what());
  }
}
}  // namespace fused_attn

using namespace transformer_engine::fused_attn;
void fused_attn_arbitrary_seqlen_fwd(const FusedAttnConfig &cfg, const Tensor *input_Q,
                                     const Tensor *input_K, const Tensor *input_V,
                                     const Tensor *input_Bias, const Tensor *input_SoftmaxOffset,
                                     Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors,
                                     const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
                                     const Tensor *cu_seqlens_q_padded,
                                     const Tensor *cu_seqlens_kv_padded, const Tensor *page_table_k,
                                     const Tensor *page_table_v, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const size_t batch = cfg.batch_size;
  const size_t num_attn_heads = cfg.num_attn_heads;
  const size_t max_seqlen_q = cfg.max_seqlen_q;
  const size_t num_tokens_q = cfg.num_tokens_q;
  const bool return_max_logit = cfg.return_max_logit;
  const NVTE_QKV_Layout qkv_layout = cfg.qkv_layout;
  const NVTE_Bias_Type bias_type = cfg.bias_type;
  const NVTE_Softmax_Type softmax_type = cfg.softmax_type;

  const auto QKV_type = input_Q->data.dtype;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;
  void *devPtrO = output_O->data.dptr;
  void *devPtrS1 = nullptr;
  void *devPtrS2 = nullptr;
  void *devPtrBias = nullptr;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
  }
  void *devPtrSoftmaxOffset = nullptr;
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    devPtrSoftmaxOffset = input_SoftmaxOffset->data.dptr;
  }

  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);

  void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = cu_seqlens_kv_padded->data.dptr;
  void *devPtrPageTableK = page_table_k ? page_table_k->data.dptr : nullptr;
  void *devPtrPageTableV = page_table_v ? page_table_v->data.dptr : nullptr;

  FusedAttnConfig graph_cfg = cfg;
  graph_cfg.derive();

  size_t i = 0;
  if (Aux_CTX_Tensors->size == 0) {
    const auto cudnn_runtime_version = cudnnGetVersion();

    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_S->data.dptr = nullptr;
    // sm120 does not use ragged stats: the graph declares a dense
    // [b, h, s_q, 1] stats tensor, so allocate to match (same as Max below).
    if ((q_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) &&
        (sm_arch_ != 120)) {
      output_S->data.shape = {num_tokens_q, num_attn_heads, 1};
    } else {
      output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
    }
    output_S->data.dtype = DType::kFloat32;

    if (return_max_logit) {
      Tensor *output_Max = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_Max->data.dptr = nullptr;
      if ((q_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) &&
          (sm_arch_ != 120)) {
        output_Max->data.shape = {num_tokens_q, num_attn_heads, 1};
      } else {
        output_Max->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      }
      output_Max->data.dtype = DType::kFloat32;
    }

    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = nullptr;
    output_rng_state->data.shape = {2};
    output_rng_state->data.dtype = DType::kInt64;

    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {graph_cfg.bias_batch_size, graph_cfg.bias_num_heads,
                                 graph_cfg.bias_seqlen_q, graph_cfg.bias_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    }

    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      Tensor *output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = nullptr;
      output_softmax_offset->data.shape = {1, num_attn_heads, 1, 1};
      output_softmax_offset->data.dtype = DType::kFloat32;
    }

    Aux_CTX_Tensors->size = i;
  } else if (Aux_CTX_Tensors->size >= 2) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    devPtrS1 = output_S->data.dptr;

    if (return_max_logit) {
      Tensor *output_Max = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      devPtrS2 = output_Max->data.dptr;
    }
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_bias->data.dptr = devPtrBias;
    }
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      Tensor *output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = devPtrSoftmaxOffset;
    }
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_fwd_impl(
      graph_cfg, devPtrQ, devPtrK, devPtrV, devPtrBias, devPtrSoftmaxOffset, devPtrS1, devPtrS2,
      devPtrO, devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlensQ, devPtrCuSeqlensKV,
      devPtrPageTableK, devPtrPageTableV, devPtrSeqOffsetsQ, devPtrSeqOffsetsKV,
      workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}

void fused_attn_arbitrary_seqlen_bwd(const FusedAttnConfig &cfg, const Tensor *input_Q,
                                     const Tensor *input_K, const Tensor *input_V,
                                     const Tensor *input_O, const Tensor *input_dO,
                                     const Tensor *input_Bias, const Tensor *input_SoftmaxOffset,
                                     Tensor *output_S, Tensor *output_dQ, Tensor *output_dK,
                                     Tensor *output_dV, Tensor *output_dBias,
                                     Tensor *output_dSoftmaxOffset, const Tensor *cu_seqlens_q,
                                     const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
                                     const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const NVTE_Bias_Type bias_type = cfg.bias_type;
  const NVTE_Softmax_Type softmax_type = cfg.softmax_type;

  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;
  void *devPtrO = input_O->data.dptr;
  void *devPtrdO = input_dO->data.dptr;
  void *devPtrBias = nullptr;
  void *devPtrdBias = nullptr;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    devPtrdBias = output_dBias->data.dptr;
  }

  FusedAttnConfig graph_cfg = cfg;
  graph_cfg.derive();

  void *devPtrdQ = output_dQ->data.dptr;
  void *devPtrdK = output_dK->data.dptr;
  void *devPtrdV = output_dV->data.dptr;
  void *devPtrSoftmaxStats = nullptr;
  devPtrSoftmaxStats = output_S->data.dptr;
  void *devPtrSoftmaxOffset = nullptr;
  void *devPtrdSoftmaxOffset = nullptr;
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    devPtrSoftmaxOffset = input_SoftmaxOffset->data.dptr;
    devPtrdSoftmaxOffset = output_dSoftmaxOffset->data.dptr;
  }

  void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = cu_seqlens_kv_padded->data.dptr;

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_bwd_impl(
      graph_cfg, devPtrQ, devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias,
      devPtrSoftmaxOffset, devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdBias,
      devPtrdSoftmaxOffset, devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlensQ,
      devPtrCuSeqlensKV, devPtrSeqOffsetsQ, devPtrSeqOffsetsKV, workspace->data.dptr,
      &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = {workspace_size};
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = {1};
    workspace->data.dtype = DType::kByte;
    return;
  } else {
    NVTE_ERROR("Unexpected workspace_size.");
  }
}

std::string is_supported_f16_fwd(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  FusedAttnConfig graph_cfg = cfg;
  graph_cfg.is_forward = true;
  graph_cfg.derive();

  size_t workspace_size = 0;
  try {
    fused_attn::fused_attn_arbitrary_seqlen_fwd_impl(
        graph_cfg,
        /*devPtrQ=*/nullptr, /*devPtrK=*/nullptr, /*devPtrV=*/nullptr, /*devPtrBias=*/nullptr,
        /*devPtrSoftmaxOffset=*/nullptr, /*devPtrS1=*/nullptr, /*devPtrS2=*/nullptr,
        /*devPtrO=*/nullptr, /*devPtrDropoutSeed=*/nullptr, /*devPtrDropoutOffset=*/nullptr,
        /*devPtrCuSeqlensQ=*/nullptr, /*devPtrCuSeqlensKV=*/nullptr,
        /*devPtrPageTableK=*/nullptr, /*devPtrPageTableV=*/nullptr,
        /*devPtrSeqOffsetsQ=*/nullptr, /*devPtrSeqOffsetsKV=*/nullptr,
        /*workspace=*/nullptr, &workspace_size,
        /*stream=*/static_cast<cudaStream_t>(0), handle);
    return "";
  } catch (const std::exception &e) {
    return e.what();
  } catch (...) {
    return "is_supported_f16_fwd: unknown failure.";
  }
}

std::string is_supported_f16_bwd(const FusedAttnConfig &cfg, cudnnHandle_t handle) {
  FusedAttnConfig graph_cfg = cfg;
  graph_cfg.is_forward = false;
  graph_cfg.derive();

  size_t workspace_size = 0;
  try {
    fused_attn::fused_attn_arbitrary_seqlen_bwd_impl(
        graph_cfg,
        /*devPtrQ=*/nullptr, /*devPtrKTranspose=*/nullptr,
        /*devPtrVTranspose=*/nullptr, /*devPtrO=*/nullptr, /*devPtrSoftmaxStats=*/nullptr,
        /*devPtrBias=*/nullptr, /*devPtrSoftmaxOffset=*/nullptr, /*devPtrdQ=*/nullptr,
        /*devPtrdK=*/nullptr, /*devPtrdV=*/nullptr, /*devPtrdO=*/nullptr,
        /*devPtrdBias=*/nullptr, /*devPtrdSoftmaxOffset=*/nullptr,
        /*devPtrDropoutSeed=*/nullptr, /*devPtrDropoutOffset=*/nullptr,
        /*devPtrCuSeqlensQ=*/nullptr, /*devPtrCuSeqlensKV=*/nullptr,
        /*devPtrSeqOffsetsQ=*/nullptr, /*devPtrSeqOffsetsKV=*/nullptr,
        /*workspace=*/nullptr, &workspace_size,
        /*stream=*/static_cast<cudaStream_t>(0), handle);
    return "";
  } catch (const std::exception &e) {
    return e.what();
  } catch (...) {
    return "is_supported_f16_bwd: unknown failure.";
  }
}

}  // namespace transformer_engine
