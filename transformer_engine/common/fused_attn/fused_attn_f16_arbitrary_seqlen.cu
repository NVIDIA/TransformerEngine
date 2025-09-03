/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cudnn_frontend.h>
#include <cudnn_frontend_utils.h>

#include <map>
#include <vector>

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/cuda_runtime.h"
#include "../util/system.h"
#include "fused_attn_f16_arbitrary_seqlen.h"
#include "utils.h"

#if (CUDNN_VERSION >= 8900)
#define Q_ID 1
#define K_ID 2
#define V_ID 3
#define O_ID 4
#define S_ID 5
#define B_ID 6
#define D_CONST_ID 7
#define S_CONST_ID 8
#define Q_SEQLEN_ID 9
#define K_SEQLEN_ID 10
#define dQ_ID 11
#define dK_ID 12
#define dV_ID 13
#define dO_ID 14
#define MASK_VAL_ID 15
#define dS_ID 16
#define D_SEED_ID 17
#define D_OFFSET_ID 18
#define S_STATS_ID 19
#define S_SUM_ID 20
#define SCALE_PROB 21
#define K_TRANSPOSE_ID 22
#define dQ_ACCUM_ID 23

#define VIRTUAL_ID 30

namespace transformer_engine {
namespace fused_attn {
void fused_attn_arbitrary_seqlen_fwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d_qk, int64_t d_v,
    int64_t max_b, int64_t max_t_q, int64_t max_t_kv, int64_t num_pages_k, int64_t num_pages_v,
    int64_t page_size_k, int64_t page_size_v, int64_t max_pages_per_seq_k,
    int64_t max_pages_per_seq_v, int64_t bias_b, int64_t bias_h, bool is_training,
    float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, int64_t window_size_left,
    int64_t window_size_right, void *devPtrQ, void *devPtrK, void *devPtrV, void *devPtrBias,
    void *devPtrSoftmaxStats, void *devPtrO, void *devPtrDropoutSeed, void *devPtrDropoutOffset,
    void *devPtrCuSeqlensQ, void *devPtrCuSeqlensKV, void *devPtrPageTableK, void *devPtrPageTableV,
    void *devPtrSeqOffsetsQ, void *devPtrSeqOffsetsKV, cudnn_frontend::DataType_t tensorType,
    void *workspace, size_t *workspace_size, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_bottom_right = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK) ||
                          (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK));
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK));
  if (is_bottom_right && s_q == s_kv && !is_padding) {
    is_causal = true;
    is_bottom_right = false;
  }
  bool is_dropout = (is_training && dropout_probability != 0.0f);
  NVTE_QKV_Format q_format = nvte_get_q_format(layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(layout);
  bool is_ragged_q = (q_format == NVTE_QKV_Format::NVTE_THD);
  bool is_ragged_kv = (kv_format == NVTE_QKV_Format::NVTE_THD);
  const auto cudnn_runtime_version = cudnnGetVersion();

  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
  bool is_paged_kv = (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD);
  if (is_paged_kv) {
    NVTE_CHECK(is_padding, "Paged attention requires padding mask!");
  }

  // keep original batch size because cu_seqlens are created with [b+1] shape
  int64_t actual_b = b;
  if ((is_ragged_q || is_ragged_kv) && cudnn_runtime_version >= 90600) {
    NVTE_CHECK(is_padding, "Ragged QKV input requires padding or padding_causal mask!");
    // replace batch size and maximum sequence lengths with maximum token counts
    // for query and key/value so the graph is static within each quantization bucket
    b = max_b;
    s_q = is_ragged_q ? max_t_q : s_q;
    s_kv = is_ragged_kv ? max_t_kv : s_kv;
  }
  const DType ragged_offset_type = cudnn_runtime_version >= 90500 ? DType::kInt64 : DType::kInt32;

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d_qk,
                               d_v,
                               num_pages_k,
                               num_pages_v,
                               page_size_k,
                               page_size_v,
                               max_pages_per_seq_k,
                               max_pages_per_seq_v,
                               bias_b,
                               bias_h,
                               scaling_factor,
                               is_training,
                               dropout_probability,
                               layout,
                               bias_type,
                               mask_type,
                               window_size_left,
                               window_size_right,
                               true,
                               tensorType,
                               tensorType};

    namespace fe = cudnn_frontend;
    using graph_and_tensors =
        std::tuple<std::shared_ptr<fe::graph::Graph>,
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // V
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // O
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // page_table_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // page_table_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_f16_fprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType &cache, const FADescriptor_v1 &descriptor) -> graph_and_tensors {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(tensorType)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> page_table_k, page_table_v;
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o,
          offset_stats;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d_qk, q_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_Q_Matrix);
      if (is_paged_kv) {
        generateMatrixStrides(num_pages_k, hg, page_size_k, page_size_v, d_qk, k_stride.data(),
                              layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
        generateMatrixStrides(num_pages_v, hg, page_size_k, page_size_v, d_v, v_stride.data(),
                              layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
      } else {
        generateMatrixStrides(b, hg, s_q, s_kv, d_qk, k_stride.data(), layout,
                              NVTE_QKV_Matrix::NVTE_K_Matrix);
        generateMatrixStrides(b, hg, s_q, s_kv, d_v, v_stride.data(), layout,
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
                         .set_is_inference(false)
                         .set_causal_mask(is_causal)
                         .set_causal_mask_bottom_right(is_bottom_right)
                         .set_attn_scale(attn_scale);

      if (cudnn_runtime_version >= 90200 && window_size_left != -1) {
        sdpa_options.set_diagonal_band_left_bound(window_size_left + 1);
      }

      sdpa_options.set_alibi_mask(is_alibi);

      if (is_bias) {
        bias = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("bias")
                                     .set_dim({bias_b, bias_h, s_q, s_kv})
                                     .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_options.set_bias(bias);
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
        sdpa_options.set_padding_mask(is_padding).set_seq_len_q(seq_q).set_seq_len_kv(seq_kv);
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

      auto [O, Stats] = mha_graph->sdpa(Q, K, V, sdpa_options);

      std::vector<int64_t> o_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d_v, o_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_O_Matrix);
      O->set_output(true).set_dim({b, h, s_q, d_v}).set_stride(o_stride);
      if (is_ragged_q) {
        offset_o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("offset_o")
                                         .set_dim({b + 1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        O->set_ragged_offset(offset_o);
      }

      Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT).set_dim({b, h, s_q, 1});
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
        offset_stats =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("offset_stats")
                                  .set_dim({b + 1, 1, 1, 1})
                                  .set_stride({1, 1, 1, 1})
                                  .set_data_type(get_cudnn_fe_dtype(ragged_offset_type)));
        Stats->set_stride({h * s_q, 1, h, 1}).set_ragged_offset(offset_stats);
      } else {
        Stats->set_stride({h * s_q, s_q, 1, 1});
      }

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // O
          key_tensors_tuple = std::make_tuple(Q, K, V, attn_scale, O);
      auto Stats_tuple = std::make_tuple(Stats);
      auto bias_tuple = is_bias ? std::make_tuple(bias) : std::make_tuple(nullptr);
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto page_table_tuple = is_paged_kv ? std::make_tuple(page_table_k, page_table_v)
                                          : std::make_tuple(nullptr, nullptr);
      auto offset_qo_tuple =
          is_ragged_q ? std::make_tuple(offset_q, offset_o) : std::make_tuple(nullptr, nullptr);
      auto offset_kv_tuple =
          is_ragged_kv ? std::make_tuple(offset_k, offset_v) : std::make_tuple(nullptr, nullptr);
      auto offset_s_tuple = (is_ragged_q && cudnn_runtime_version >= 90600)
                                ? std::make_tuple(offset_stats)
                                : std::make_tuple(nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple = std::tuple_cat(
          std::make_tuple(mha_graph), key_tensors_tuple, Stats_tuple, bias_tuple, padding_tuple,
          page_table_tuple, offset_qo_tuple, offset_kv_tuple, offset_s_tuple, dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };

    auto [mha_graph, Q, K, V, attn_scale, O, Stats, bias, seq_q, seq_kv, page_table_k, page_table_v,
          offset_q, offset_o, offset_k, offset_v, offset_stats, dropout_seed, dropout_offset] =
        get_graph(sdpa_f16_fprop_cache, descriptor);

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
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
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

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void *> variant_pack = {
        {Q, devPtrQ}, {K, devPtrK},
        {V, devPtrV}, {attn_scale, &scaling_factor},
        {O, devPtrO}, {Stats, devPtrSoftmaxStats}};

    if (is_bias) {
      variant_pack[bias] = devPtrBias;
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

    if (is_paged_kv) {
      variant_pack[page_table_k] = devPtrPageTableK;
      variant_pack[page_table_v] = devPtrPageTableV;
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
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
        devOffsetsS = static_cast<int8_t *>(devOffsets) +
                      (static_cast<int>(is_ragged_q) + static_cast<int>(is_ragged_kv)) * 2 *
                          num_bytes_per_ragged_offset;
      }
      const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
      cu_seqlens_padded_to_offsets<<<grid, nthreads_per_block, 0, stream>>>(
          layout_group, actual_b, b, h, hg, d_qk, d_v, static_cast<int32_t *>(devPtrSeqOffsetsQ),
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
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
        variant_pack[offset_stats] = devOffsetsS;
      }
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }
    NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException &e) {
    NVTE_ERROR(e.what());
  }
}

void fused_attn_arbitrary_seqlen_bwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d_qk, int64_t d_v,
    int64_t max_b, int64_t max_t_q, int64_t max_t_kv, int64_t bias_b, int64_t bias_h,
    float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, int64_t window_size_left,
    int64_t window_size_right, bool deterministic, void *devPtrQ, void *devPtrKTranspose,
    void *devPtrVTranspose, void *devPtrO, void *devPtrSoftmaxStats, void *devPtrBias,
    void *devPtrdQ, void *devPtrdK, void *devPtrdV, void *devPtrdO, void *devPtrdBias,
    void *devPtrDropoutSeed, void *devPtrDropoutOffset, void *devPtrCuSeqlensQ,
    void *devPtrCuSeqlensKV, void *devPtrSeqOffsetsQ, void *devPtrSeqOffsetsKV,
    cudnn_frontend::DataType_t tensorType, void *workspace, size_t *workspace_size,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_bottom_right = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_BOTTOM_RIGHT_MASK) ||
                          (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK));
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK));
  if (is_bottom_right && s_q == s_kv && !is_padding) {
    is_causal = true;
    is_bottom_right = false;
  }
  bool is_dropout = (dropout_probability != 0.0f);
  NVTE_QKV_Format q_format = nvte_get_q_format(layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(layout);
  bool is_ragged_q = (q_format == NVTE_QKV_Format::NVTE_THD);
  bool is_ragged_kv = (kv_format == NVTE_QKV_Format::NVTE_THD);
  const auto cudnn_runtime_version = cudnnGetVersion();
  const int device_id = cuda::current_device();
  const int sm_arch_ = cuda::sm_arch(device_id);

  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
  bool is_paged_kv = (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD);
  if (is_paged_kv) {
    NVTE_CHECK(is_padding, "Paged attention requires padding mask!");
  }

  // keep original batch size because cu_seqlens are created with [b+1] shape
  int64_t actual_b = b;
  if ((is_ragged_q || is_ragged_kv) && cudnn_runtime_version >= 90600) {
    NVTE_CHECK(is_padding, "Ragged QKV input requires padding or padding_causal mask!");
    // replace batch size and maximum sequence lengths with maximum token counts
    // for query and key/value so the graph is static within each quantization bucket
    b = max_b;
    s_q = is_ragged_q ? max_t_q : s_q;
    s_kv = is_ragged_kv ? max_t_kv : s_kv;
  }

  // We choose between 32-bit and 64-bit offsets depending on need.
  // This allows us to support older cuDNN runtimes gracefully.
  const DType ragged_offset_type = cudnn_runtime_version >= 90500 ? DType::kInt64 : DType::kInt32;

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d_qk,
                               d_v,
                               0,
                               0,
                               0,
                               0,
                               0,
                               0,
                               bias_b,
                               bias_h,
                               scaling_factor,
                               true,
                               dropout_probability,
                               layout,
                               bias_type,
                               mask_type,
                               window_size_left,
                               window_size_right,
                               deterministic,
                               tensorType,
                               tensorType};

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
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_f16_bprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType &cache, const FADescriptor_v1 &descriptor) -> graph_and_tensors {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(tensorType)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> q, k, v, o, dO, stats, attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, dBias, seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o,
          offset_stats;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      std::vector<int64_t> o_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d_qk, q_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_Q_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d_qk, k_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_K_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d_v, v_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_V_Matrix);
      generateMatrixStrides(b, h, s_q, s_kv, d_v, o_stride.data(), layout,
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
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
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
                                  .set_causal_mask(is_causal)
                                  .set_causal_mask_bottom_right(is_bottom_right)
                                  .set_attn_scale(attn_scale);

      if (is_ragged_q && cudnn_runtime_version >= 90600) {
        sdpa_backward_options.set_max_total_seq_len_q(s_q);
      }
      if (is_ragged_kv && cudnn_runtime_version >= 90600) {
        sdpa_backward_options.set_max_total_seq_len_kv(s_kv);
      }

      if (cudnn_runtime_version >= 90200 && window_size_left != -1) {
        sdpa_backward_options.set_diagonal_band_left_bound(window_size_left + 1);
      }

      if (cudnn_runtime_version >= 90000) {
        sdpa_backward_options.set_deterministic_algorithm(deterministic);
      }

      sdpa_backward_options.set_alibi_mask(is_alibi);

      if (is_bias) {
        bias = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("bias")
                                     .set_dim({bias_b, bias_h, s_q, s_kv})
                                     .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
        dBias = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("dBias")
                                      .set_dim({bias_b, bias_h, s_q, s_kv})
                                      .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
        sdpa_backward_options.set_bias(bias);
        // shapes [1, 1, s, s], [b, 1, s, s], [b, h, s, s]
        // are not supported for dbias calculation but they are
        // supported for forward bias calculation
        if ((bias_b == 1) && (bias_h == h)) {
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
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto offset_qo_tuple =
          is_ragged_q ? std::make_tuple(offset_q, offset_o) : std::make_tuple(nullptr, nullptr);
      auto offset_kv_tuple =
          is_ragged_kv ? std::make_tuple(offset_k, offset_v) : std::make_tuple(nullptr, nullptr);
      auto offset_s_tuple = (is_ragged_q && cudnn_runtime_version >= 90600)
                                ? std::make_tuple(offset_stats)
                                : std::make_tuple(nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple =
          std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, bias_tuple, padding_tuple,
                         offset_qo_tuple, offset_kv_tuple, offset_s_tuple, dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };

    auto [mha_graph, q, k, v, o, dO, stats, attn_scale, dQ, dK, dV, bias, dBias, seq_q, seq_kv,
          offset_q, offset_o, offset_k, offset_v, offset_stats, dropout_seed, dropout_offset] =
        get_graph(sdpa_f16_bprop_cache, descriptor);

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
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
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
      if ((bias_b == 1) && (bias_h == h)) {
        variant_pack[dBias] = devPtrdBias;
      } else {
        variant_pack[dBias] = nullptr;
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
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
        devOffsetsS = static_cast<int8_t *>(devOffsets) +
                      (static_cast<int>(is_ragged_q) + static_cast<int>(is_ragged_kv)) * 2 *
                          num_bytes_per_ragged_offset;
      }
      const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
      cu_seqlens_padded_to_offsets<<<grid, nthreads_per_block, 0, stream>>>(
          layout_group, actual_b, b, h, hg, d_qk, d_v, static_cast<int32_t *>(devPtrSeqOffsetsQ),
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
      if (is_ragged_q && cudnn_runtime_version >= 90600) {
        variant_pack[offset_stats] = devOffsetsS;
      }
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException &e) {
    NVTE_ERROR(e.what());
  }
}
}  // namespace fused_attn

using namespace transformer_engine::fused_attn;
void fused_attn_arbitrary_seqlen_fwd_qkvpacked(
    size_t batch, size_t num_attn_heads, size_t max_seqlen, size_t head_dim, size_t num_tokens,
    bool is_training, float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, int64_t window_size_left,
    int64_t window_size_right, const Tensor *input_QKV, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens, const Tensor *cu_seqlens_padded,
    const Tensor *rng_state, Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_QKV->data.dtype;
  void *devPtrQKV = input_QKV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    stride = (typeToNumBits(QKV_type) * num_attn_heads * head_dim) / 8;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    stride = (typeToNumBits(QKV_type) * head_dim) / 8;
  }
  void *devPtrQ = static_cast<void *>(devPtrQKV);
  void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

  void *devPtrBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    bias_b = input_Bias->data.shape[0];
    bias_h = input_Bias->data.shape[1];
  }

  void *devPtrO = output_O->data.dptr;
  void *devPtrS = nullptr;
  void *devPtrCuSeqlens = cu_seqlens->data.dptr;
  void *devPtrSeqOffsets = cu_seqlens_padded->data.dptr;

  size_t max_batch_size = 0;
  size_t max_tokens = 0;
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    max_batch_size = get_max_batch_size(batch);
    max_tokens = get_max_tokens(num_tokens);
  }

  if (Aux_CTX_Tensors->size == 0) {
    const auto cudnn_runtime_version = cudnnGetVersion();
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if (qkv_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) {
        output_S->data.shape = {max_tokens, num_attn_heads, 1};
      } else {
        output_S->data.shape = {batch, num_attn_heads, max_seqlen, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen, max_seqlen};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if (qkv_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) {
        output_S->data.shape = {max_tokens, num_attn_heads, 1};
      } else {
        output_S->data.shape = {batch, num_attn_heads, max_seqlen, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_fwd_impl(
      batch, num_attn_heads, num_attn_heads, max_seqlen, max_seqlen, head_dim, head_dim,
      max_batch_size, max_tokens, max_tokens, 0, 0, 0, 0, 0, 0, bias_b, bias_h, is_training,
      attn_scale, p_dropout, qkv_layout, bias_type, mask_type, window_size_left, window_size_right,
      devPtrQ, devPtrK, devPtrV, devPtrBias, devPtrS, devPtrO, devPtrDropoutSeed,
      devPtrDropoutOffset, devPtrCuSeqlens, devPtrCuSeqlens, nullptr, nullptr, devPtrSeqOffsets,
      devPtrSeqOffsets, get_cudnn_fe_dtype(QKV_type), workspace->data.dptr, &workspace_size, stream,
      handle);

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

void fused_attn_arbitrary_seqlen_bwd_qkvpacked(
    size_t batch, size_t num_attn_heads, size_t max_seqlen, size_t head_dim, size_t num_tokens,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, int64_t window_size_left, int64_t window_size_right,
    bool deterministic, const Tensor *input_QKV, const Tensor *input_O, const Tensor *input_dO,
    const Tensor *input_Bias, Tensor *output_S, Tensor *output_dQKV, Tensor *output_dBias,
    const Tensor *cu_seqlens, const Tensor *cu_seqlens_padded, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_QKV->data.dtype;
  void *devPtrQKV = input_QKV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    stride = (typeToNumBits(QKV_type) * num_attn_heads * head_dim) / 8;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    stride = (typeToNumBits(QKV_type) * head_dim) / 8;
  }
  void *devPtrQ = devPtrQKV;
  void *devPtrK = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + stride);
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrQKV) + 2 * stride);

  void *devPtrO = input_O->data.dptr;
  void *devPtrdO = input_dO->data.dptr;
  void *devPtrBias = nullptr;
  void *devPtrdBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    devPtrdBias = output_dBias->data.dptr;
    bias_b = output_dBias->data.shape[0];
    bias_h = output_dBias->data.shape[1];
  }

  size_t max_batch_size = 0;
  size_t max_tokens = 0;
  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if (qkv_format == NVTE_QKV_Format::NVTE_THD) {
    max_batch_size = get_max_batch_size(batch);
    max_tokens = get_max_tokens(num_tokens);
  }

  void *devPtrdQKV = output_dQKV->data.dptr;
  void *devPtrdQ = devPtrdQKV;
  void *devPtrdK = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + stride);
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdQKV) + 2 * stride);

  void *devPtrSoftmaxStats = nullptr;
  devPtrSoftmaxStats = output_S->data.dptr;

  void *devPtrCuSeqlens = cu_seqlens->data.dptr;
  void *devPtrSeqOffsets = cu_seqlens_padded->data.dptr;

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_bwd_impl(
      batch, num_attn_heads, num_attn_heads, max_seqlen, max_seqlen, head_dim, head_dim,
      max_batch_size, max_tokens, max_tokens, bias_b, bias_h, attn_scale, p_dropout, qkv_layout,
      bias_type, mask_type, window_size_left, window_size_right, deterministic, devPtrQ, devPtrK,
      devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias, devPtrdQ, devPtrdK, devPtrdV, devPtrdO,
      devPtrdBias, devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlens, devPtrCuSeqlens,
      devPtrSeqOffsets, devPtrSeqOffsets, get_cudnn_fe_dtype(QKV_type), workspace->data.dptr,
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
void fused_attn_arbitrary_seqlen_fwd_kvpacked(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim, size_t num_tokens_q, size_t num_tokens_kv,
    size_t num_pages_k, size_t num_pages_v, size_t page_size_k, size_t page_size_v,
    size_t max_pages_per_seq_k, size_t max_pages_per_seq_v, bool is_training, float attn_scale,
    float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    int64_t window_size_left, int64_t window_size_right, const Tensor *input_Q,
    const Tensor *input_KV, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *cu_seqlens_q_padded, const Tensor *cu_seqlens_kv_padded,
    const Tensor *page_table_k, const Tensor *page_table_v, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_Q->data.dtype;
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = (typeToNumBits(QKV_type) * num_gqa_groups * head_dim) / 8;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = (typeToNumBits(QKV_type) * head_dim) / 8;
  }
  void *devPtrK = devPtrKV;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

  void *devPtrBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    bias_b = input_Bias->data.shape[0];
    bias_h = input_Bias->data.shape[1];
  }

  void *devPtrO = output_O->data.dptr;
  void *devPtrS = nullptr;

  void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = cu_seqlens_kv_padded->data.dptr;
  void *devPtrPageTableK = page_table_k->data.dptr;
  void *devPtrPageTableV = page_table_v->data.dptr;

  size_t max_batch_size = 0;
  size_t max_tokens_q = 0;
  size_t max_tokens_kv = 0;
  if (q_format == NVTE_QKV_Format::NVTE_THD || kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_batch_size = get_max_batch_size(batch);
  }
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_q = get_max_tokens(num_tokens_q);
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_kv = get_max_tokens(num_tokens_kv);
  }

  if (Aux_CTX_Tensors->size == 0) {
    const auto cudnn_runtime_version = cudnnGetVersion();
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if (q_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) {
        output_S->data.shape = {max_tokens_q, num_attn_heads, 1};
      } else {
        output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if (q_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) {
        output_S->data.shape = {max_tokens_q, num_attn_heads, 1};
      } else {
        output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_fwd_impl(
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim, head_dim,
      max_batch_size, max_tokens_q, max_tokens_kv, num_pages_k, num_pages_v, page_size_k,
      page_size_v, max_pages_per_seq_k, max_pages_per_seq_v, bias_b, bias_h, is_training,
      attn_scale, p_dropout, qkv_layout, bias_type, mask_type, window_size_left, window_size_right,
      devPtrQ, devPtrK, devPtrV, devPtrBias, devPtrS, devPtrO, devPtrDropoutSeed,
      devPtrDropoutOffset, devPtrCuSeqlensQ, devPtrCuSeqlensKV, devPtrPageTableK, devPtrPageTableV,
      devPtrSeqOffsetsQ, devPtrSeqOffsetsKV, get_cudnn_fe_dtype(QKV_type), workspace->data.dptr,
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

void fused_attn_arbitrary_seqlen_bwd_kvpacked(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim, size_t num_tokens_q, size_t num_tokens_kv,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, int64_t window_size_left, int64_t window_size_right,
    bool deterministic, const Tensor *input_Q, const Tensor *input_KV, const Tensor *input_O,
    const Tensor *input_dO, const Tensor *input_Bias, Tensor *output_S, Tensor *output_dQ,
    Tensor *output_dKV, Tensor *output_dBias, const Tensor *cu_seqlens_q,
    const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
    const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state, Tensor *workspace,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_Q->data.dtype;
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = (typeToNumBits(QKV_type) * num_gqa_groups * head_dim) / 8;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = (typeToNumBits(QKV_type) * head_dim) / 8;
  }
  void *devPtrK = devPtrKV;
  void *devPtrV = static_cast<void *>(static_cast<int8_t *>(devPtrKV) + stride);

  void *devPtrO = input_O->data.dptr;
  void *devPtrdO = input_dO->data.dptr;
  void *devPtrBias = nullptr;
  void *devPtrdBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    devPtrdBias = output_dBias->data.dptr;
    bias_b = output_dBias->data.shape[0];
    bias_h = output_dBias->data.shape[1];
  }

  size_t max_batch_size = 0;
  size_t max_tokens_q = 0;
  size_t max_tokens_kv = 0;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  if (q_format == NVTE_QKV_Format::NVTE_THD || kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_batch_size = get_max_batch_size(batch);
  }
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_q = get_max_tokens(num_tokens_q);
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_kv = get_max_tokens(num_tokens_kv);
  }

  void *devPtrdQ = output_dQ->data.dptr;
  void *devPtrdKV = output_dKV->data.dptr;
  void *devPtrdK = devPtrdKV;
  void *devPtrdV = static_cast<void *>(static_cast<int8_t *>(devPtrdKV) + stride);

  void *devPtrSoftmaxStats = nullptr;
  devPtrSoftmaxStats = output_S->data.dptr;

  void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = cu_seqlens_kv_padded->data.dptr;

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_bwd_impl(
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim, head_dim,
      max_batch_size, max_tokens_q, max_tokens_kv, bias_b, bias_h, attn_scale, p_dropout,
      qkv_layout, bias_type, mask_type, window_size_left, window_size_right, deterministic, devPtrQ,
      devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias, devPtrdQ, devPtrdK, devPtrdV,
      devPtrdO, devPtrdBias, devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlensQ,
      devPtrCuSeqlensKV, devPtrSeqOffsetsQ, devPtrSeqOffsetsKV, get_cudnn_fe_dtype(QKV_type),
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

void fused_attn_arbitrary_seqlen_fwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, size_t num_tokens_q,
    size_t num_tokens_kv, size_t num_pages_k, size_t num_pages_v, size_t page_size_k,
    size_t page_size_v, size_t max_pages_per_seq_k, size_t max_pages_per_seq_v, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, int64_t window_size_left, int64_t window_size_right,
    const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V, const Tensor *input_Bias,
    Tensor *output_O, NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q,
    const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
    const Tensor *cu_seqlens_kv_padded, const Tensor *page_table_k, const Tensor *page_table_v,
    const Tensor *rng_state, Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_Q->data.dtype;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;
  void *devPtrO = output_O->data.dptr;
  void *devPtrS = nullptr;
  void *devPtrBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    bias_b = input_Bias->data.shape[0];
    bias_h = input_Bias->data.shape[1];
  }

  void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = cu_seqlens_kv_padded->data.dptr;
  void *devPtrPageTableK = page_table_k->data.dptr;
  void *devPtrPageTableV = page_table_v->data.dptr;

  size_t max_batch_size = 0;
  size_t max_tokens_q = 0;
  size_t max_tokens_kv = 0;
  if (q_format == NVTE_QKV_Format::NVTE_THD || kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_batch_size = get_max_batch_size(batch);
  }
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_q = get_max_tokens(num_tokens_q);
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_kv = get_max_tokens(num_tokens_kv);
  }

  if (Aux_CTX_Tensors->size == 0) {
    const auto cudnn_runtime_version = cudnnGetVersion();
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if (q_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) {
        output_S->data.shape = {max_tokens_q, num_attn_heads, 1};
      } else {
        output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      if (q_format == NVTE_QKV_Format::NVTE_THD && cudnn_runtime_version >= 90600) {
        output_S->data.shape = {max_tokens_q, num_attn_heads, 1};
      } else {
        output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      }
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_fwd_impl(
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v,
      max_batch_size, max_tokens_q, max_tokens_kv, num_pages_k, num_pages_v, page_size_k,
      page_size_v, max_pages_per_seq_k, max_pages_per_seq_v, bias_b, bias_h, is_training,
      attn_scale, p_dropout, qkv_layout, bias_type, mask_type, window_size_left, window_size_right,
      devPtrQ, devPtrK, devPtrV, devPtrBias, devPtrS, devPtrO, devPtrDropoutSeed,
      devPtrDropoutOffset, devPtrCuSeqlensQ, devPtrCuSeqlensKV, devPtrPageTableK, devPtrPageTableV,
      devPtrSeqOffsetsQ, devPtrSeqOffsetsKV, get_cudnn_fe_dtype(QKV_type), workspace->data.dptr,
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

void fused_attn_arbitrary_seqlen_bwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, size_t num_tokens_q,
    size_t num_tokens_kv, float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, int64_t window_size_left,
    int64_t window_size_right, bool deterministic, const Tensor *input_Q, const Tensor *input_K,
    const Tensor *input_V, const Tensor *input_O, const Tensor *input_dO, const Tensor *input_Bias,
    Tensor *output_S, Tensor *output_dQ, Tensor *output_dK, Tensor *output_dV, Tensor *output_dBias,
    const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
    const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state, Tensor *workspace,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  const auto QKV_type = input_Q->data.dtype;
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrK = input_K->data.dptr;
  void *devPtrV = input_V->data.dptr;
  void *devPtrO = input_O->data.dptr;
  void *devPtrdO = input_dO->data.dptr;
  void *devPtrBias = nullptr;
  void *devPtrdBias = nullptr;
  size_t bias_b = 0;
  size_t bias_h = 0;
  if ((bias_type != NVTE_Bias_Type::NVTE_NO_BIAS) && (bias_type != NVTE_Bias_Type::NVTE_ALIBI)) {
    devPtrBias = input_Bias->data.dptr;
    devPtrdBias = output_dBias->data.dptr;
    bias_b = output_dBias->data.shape[0];
    bias_h = output_dBias->data.shape[1];
  }

  size_t max_batch_size = 0;
  size_t max_tokens_q = 0;
  size_t max_tokens_kv = 0;
  NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
  NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
  if (q_format == NVTE_QKV_Format::NVTE_THD || kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_batch_size = get_max_batch_size(batch);
  }
  if (q_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_q = get_max_tokens(num_tokens_q);
  }
  if (kv_format == NVTE_QKV_Format::NVTE_THD) {
    max_tokens_kv = get_max_tokens(num_tokens_kv);
  }

  void *devPtrdQ = output_dQ->data.dptr;
  void *devPtrdK = output_dK->data.dptr;
  void *devPtrdV = output_dV->data.dptr;
  void *devPtrSoftmaxStats = nullptr;
  devPtrSoftmaxStats = output_S->data.dptr;

  void *devPtrCuSeqlensQ = cu_seqlens_q->data.dptr;
  void *devPtrCuSeqlensKV = cu_seqlens_kv->data.dptr;
  void *devPtrSeqOffsetsQ = cu_seqlens_q_padded->data.dptr;
  void *devPtrSeqOffsetsKV = cu_seqlens_kv_padded->data.dptr;

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_bwd_impl(
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v,
      max_batch_size, max_tokens_q, max_tokens_kv, bias_b, bias_h, attn_scale, p_dropout,
      qkv_layout, bias_type, mask_type, window_size_left, window_size_right, deterministic, devPtrQ,
      devPtrK, devPtrV, devPtrO, devPtrSoftmaxStats, devPtrBias, devPtrdQ, devPtrdK, devPtrdV,
      devPtrdO, devPtrdBias, devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlensQ,
      devPtrCuSeqlensKV, devPtrSeqOffsetsQ, devPtrSeqOffsetsKV, get_cudnn_fe_dtype(QKV_type),
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
}  // namespace transformer_engine
#endif  // CUDNN_VERSION >= 8900
