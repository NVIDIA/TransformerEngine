/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "../common.h"
#include "../cudnn_utils.h"
#include "../util/system.h"
#include "fused_attn_fp8.h"
#include "utils.h"

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

// fused attention FWD FP8 with FE 1.0+
void fused_attn_fp8_fwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d_qk, int64_t d_v,
    bool is_training, float scaling_factor, float dropout_probability, NVTE_QKV_Layout qkv_layout,
    NVTE_QKV_Format o_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_Softmax_Type softmax_type, int64_t window_size_left, int64_t window_size_right,
    bool bottom_right_diagonal, void* devPtrQ, void* devPtrK, void* devPtrV,
    void* devPtrSoftmaxOffset, void* devPtrM, void* devPtrO, void* devPtrDescaleQ,
    void* devPtrDescaleK, void* devPtrDescaleV, void* devPtrDescaleS, void* devPtrScaleS,
    void* devPtrScaleO, void* devPtrAmaxO, void* devPtrAmaxS, void* devPtrcuSeqlensQ,
    void* devPtrcuSeqlensKV, void* devPtrDropoutSeed, void* devPtrDropoutOffset,
    cudnn_frontend::DataType_t qkv_tensor_type, cudnn_frontend::DataType_t o_tensor_type,
    NVTEScalingMode scaling_mode, NVTE_QKV_Format qkv_scale_inv_format, void* workspace,
    size_t* workspace_size, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  const auto cudnn_runtime_version = cudnnGetVersion();
  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_dropout = (is_training && dropout_probability != 0.0f);
  bool is_softmax_offset = (softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX);
  auto bias_b = b;
  auto bias_h = h;
  auto bias_sq = s_q;
  auto bias_skv = s_kv;
  NVTE_CHECK(~is_bias, "FP8 fused attention does not support pre/post_scale_bias yet!");
  NVTE_CHECK(~is_alibi, "FP8 fused attention does not support ALiBi yet!");
  bool is_delayed_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (o_tensor_type == cudnn_frontend::DataType_t::FP8_E4M3 ||
                             o_tensor_type == cudnn_frontend::DataType_t::FP8_E5M2);
  bool is_current_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (o_tensor_type == cudnn_frontend::DataType_t::HALF ||
                             o_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  bool is_mxfp8 = (scaling_mode == NVTE_MXFP8_1D_SCALING) &&
                  (o_tensor_type == cudnn_frontend::DataType_t::HALF ||
                   o_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  NVTE_CHECK(
      is_delayed_scaling || is_current_scaling || is_mxfp8,
      "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 recipes!");
  NVTE_CHECK(!is_mxfp8 || cudnn_runtime_version >= 92100,
             "MXFP8 fused attention requires cuDNN 9.21.0 or later!");

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
                               bias_sq,
                               bias_skv,
                               scaling_factor,
                               is_training,
                               dropout_probability,
                               qkv_layout,
                               o_format,
                               NVTE_QKV_Format_NOT_SET,
                               NVTE_QKV_Layout_NOT_SET,
                               qkv_scale_inv_format,
                               NVTE_QKV_Format_NOT_SET,
                               bias_type,
                               mask_type,
                               softmax_type,
                               window_size_left,
                               window_size_right,
                               bottom_right_diagonal,
                               true,
                               qkv_tensor_type,
                               o_tensor_type,
                               cudnn_frontend::DataType_t::NOT_SET,
                               cudnn_frontend::DataType_t::NOT_SET,
                               false};

    namespace fe = cudnn_frontend;
    using graph_and_tensors =
        std::tuple<std::shared_ptr<fe::graph::Graph>,
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // V
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // O
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_fp8_fprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType& cache, const FADescriptor_v1& descriptor) -> graph_and_tensors {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();
      mha_graph->set_io_data_type(qkv_tensor_type)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> Q, K, V, attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_q, descale_k, descale_v;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_s, scale_s, scale_o;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, softmax_offset, seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      // Q, K, V, attn_scale
      std::vector<int64_t> q_strides(4), k_strides(4), v_strides(4);
      generateMatrixStridesWithLayout(b, h, hg, s_q, s_kv, d_qk, d_v, q_strides.data(),
                                      k_strides.data(), v_strides.data(), qkv_layout);
      Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d_qk})
                                .set_stride(q_strides)
                                .set_data_type(qkv_tensor_type));
      K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_dim({b, hg, s_kv, d_qk})
                                .set_stride(k_strides)
                                .set_data_type(qkv_tensor_type));
      V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_dim({b, hg, s_kv, d_v})
                                .set_stride(v_strides)
                                .set_data_type(qkv_tensor_type));
      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      // Descale_q, Descale_k, Descale_v, Descale_s, Scale_s, Scale_o
      if (is_delayed_scaling || is_current_scaling) {
        descale_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_q")
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
        descale_k = mha_graph->tensor_like(descale_q, "Descale_q");
        descale_v = mha_graph->tensor_like(descale_q, "Descale_v");
        descale_s = mha_graph->tensor_like(descale_q, "Descale_s");
        scale_s = mha_graph->tensor_like(descale_q, "Scale_s");
        if (is_delayed_scaling) {
          scale_o = mha_graph->tensor_like(descale_q, "Scale_o");
        }
        if (is_current_scaling) {
          scale_o = mha_graph->tensor(1.0f);
        }
      } else if (is_mxfp8) {
        NVTE_QKV_Format q_scale_inv_format = (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET)
                                                 ? qkv_scale_inv_format
                                                 : nvte_get_q_format(qkv_layout);
        NVTE_QKV_Format kv_scale_inv_format = (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET)
                                                  ? qkv_scale_inv_format
                                                  : nvte_get_kv_format(qkv_layout);
        std::vector<int64_t> q_scale_strides(4);
        std::vector<int64_t> k_scale_strides(4);
        std::vector<int64_t> v_scale_strides(4);
        auto padded = pad_s_d_for_mxfp8(s_q, s_kv, d_qk, d_v);
        generateMatrixStridesWithFormat(b, h, padded.s_q_padded, padded.d_qk_scale_padded,
                                        q_scale_strides.data(), q_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_padded, padded.d_qk_scale_padded,
                                        k_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_scale_padded, padded.d_v_padded,
                                        v_scale_strides.data(), kv_scale_inv_format);
        descale_q =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_q")
                                  .set_dim({b, h, padded.s_q_padded, padded.d_qk_scale_padded})
                                  .set_stride(q_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_k =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_k")
                                  .set_dim({b, hg, padded.s_kv_padded, padded.d_qk_scale_padded})
                                  .set_stride(k_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_v =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_v")
                                  .set_dim({b, hg, padded.s_kv_scale_padded, padded.d_v_padded})
                                  .set_stride(v_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
      }

      fe::graph::SDPA_fp8_attributes sdpa_options;
      sdpa_options = fe::graph::SDPA_fp8_attributes()
                         .set_name("sdpa_fp8")
                         .set_generate_stats(true)
                         .set_causal_mask(is_causal)
                         .set_attn_scale(attn_scale);

      fe::DiagonalAlignment_t const& diagonal_alignment =
          bottom_right_diagonal ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
                                : fe::DiagonalAlignment_t::TOP_LEFT;
      sdpa_options.set_diagonal_alignment(diagonal_alignment);

      if (cudnn_runtime_version >= 92100) {
        if (window_size_left != -1) {
          sdpa_options.set_diagonal_band_left_bound(window_size_left + 1);
        }
        if (window_size_right != -1) {
          sdpa_options.set_diagonal_band_right_bound(window_size_right);
        }
      }

      // sdpa_options.set_alibi_mask(is_alibi);
      // if (is_bias) {
      //     bias = mha_graph->tensor(fe::graph::Tensor_attributes()
      //                     .set_name("bias")
      //                     .set_dim({bias_b, bias_h, bias_sq, bias_skv})
      //                     .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
      //     sdpa_options.set_bias(bias);
      // }

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

      std::shared_ptr<fe::graph::Tensor_attributes> O, Stats, amax_s, amax_o;
      if (is_delayed_scaling || is_current_scaling) {
        auto outputs = mha_graph->sdpa_fp8(Q, K, V, descale_q, descale_k, descale_v, descale_s,
                                           scale_s, scale_o, sdpa_options);
        O = outputs[0];
        Stats = outputs[1];
        amax_s = outputs[2];
        amax_o = outputs[3];
        amax_s->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
      } else if (is_mxfp8) {
        auto outputs = mha_graph->sdpa_fp8(Q, K, V, descale_q, descale_k, descale_v, sdpa_options);
        O = outputs[0];
        Stats = outputs[1];
        amax_o = outputs[2];
      }

      std::vector<int64_t> o_strides(4);
      generateMatrixStridesWithFormat(b, h, s_q, d_v, o_strides.data(), o_format);
      O->set_output(true)
          .set_dim({b, h, s_q, d_v})
          .set_stride(o_strides)
          .set_data_type(o_tensor_type);
      amax_o->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);

      Stats->set_output(true)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({b, h, s_q, 1})
          .set_stride({h * s_q, s_q, 1, 1});

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_k
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_v
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_o
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_s
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // amax_o
          key_tensors_tuple =
              is_mxfp8 ? std::make_tuple(Q, K, V, descale_q, descale_k, descale_v, nullptr, nullptr,
                                         nullptr, attn_scale, O, nullptr, amax_o)
                       : std::make_tuple(Q, K, V, descale_q, descale_k, descale_v, descale_s,
                                         scale_s, scale_o, attn_scale, O, amax_s, amax_o);
      auto Stats_tuple = std::make_tuple(Stats);
      auto bias_tuple = is_bias ? std::make_tuple(bias) : std::make_tuple(nullptr);
      auto softmax_offset_tuple =
          is_softmax_offset ? std::make_tuple(softmax_offset) : std::make_tuple(nullptr);
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));
      auto return_tuple =
          std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, Stats_tuple, bias_tuple,
                         softmax_offset_tuple, padding_tuple, dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };

    auto [mha_graph, Q, K, V, descale_q, descale_k, descale_v, descale_s, scale_s, scale_o,
          attn_scale, O, amax_s, amax_o, Stats, bias, softmax_offset, seq_q, seq_kv, dropout_seed,
          dropout_offset] = get_graph(sdpa_fp8_fprop_cache, descriptor);

    auto plan_workspace_size = mha_graph->get_workspace_size();

    // Exit to request upper level API to allocate memory if needed
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    if (workspace == nullptr) {
      *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // Build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {descale_q, devPtrDescaleQ},
        {descale_k, devPtrDescaleK},
        {descale_v, devPtrDescaleV},
        {attn_scale, &scaling_factor},
        {O, devPtrO},
        {Stats, devPtrM}};

    if (is_delayed_scaling) {
      variant_pack[scale_o] = devPtrScaleO;
    }
    if (is_delayed_scaling || is_current_scaling) {
      variant_pack[descale_s] = devPtrDescaleS;
      variant_pack[scale_s] = devPtrScaleS;
      variant_pack[amax_s] = devPtrAmaxS;
      variant_pack[amax_o] = devPtrAmaxO;
    }

    /* if (is_bias) {
       variant_pack[bias] = devPtrBias;
    } */

    if (is_padding) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
      void* devActualSeqlenQ = static_cast<int8_t*>(workspace) + plan_workspace_size;
      void* devActualSeqlenKV = static_cast<int8_t*>(devActualSeqlenQ) + b * sizeof(int32_t);
      cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
          b, b, static_cast<const int32_t*>(devPtrcuSeqlensQ),  // TODO(pass max_b)
          static_cast<const int32_t*>(devPtrcuSeqlensKV), static_cast<int32_t*>(devActualSeqlenQ),
          static_cast<int32_t*>(devActualSeqlenKV));
      NVTE_CHECK_CUDA(cudaGetLastError());
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;
    }

    if (is_dropout) {
      variant_pack[dropout_seed] = devPtrDropoutSeed;
      variant_pack[dropout_offset] = devPtrDropoutOffset;
    }

    if (is_softmax_offset) {
      variant_pack[softmax_offset] = devPtrSoftmaxOffset;
    }

    NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
  } catch (cudnn_frontend::cudnnException& e) {
    NVTE_ERROR(e.what());
  }
}

// fused attention BWD FP8 with FE 1.0+
void fused_attn_fp8_bwd_impl(
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d_qk, int64_t d_v,
    float scaling_factor, float dropout_probability, NVTE_QKV_Layout qkv_layout,
    NVTE_QKV_Format o_format, NVTE_QKV_Format do_format, NVTE_QKV_Layout dqkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, NVTE_Softmax_Type softmax_type,
    int64_t window_size_left, int64_t window_size_right, bool bottom_right_diagonal,
    bool deterministic, void* devPtrQ, void* devPtrK, void* devPtrV, void* devPtrM, void* devPtrO,
    void* devPtrdO, void* devPtrSoftmaxOffset, void* devPtrdQ, void* devPtrdK, void* devPtrdV,
    void* devPtrdSoftmaxOffset, void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
    void* devPtrDescaleO, void* devPtrDescaledO, void* devPtrDescaleS, void* devPtrDescaledP,
    void* devPtrScaleS, void* devPtrScaledP, void* devPtrScaledQ, void* devPtrScaledK,
    void* devPtrScaledV, void* devPtrAmaxdP, void* devPtrAmaxdQ, void* devPtrAmaxdK,
    void* devPtrAmaxdV, void* devPtrQ_t, void* devPtrK_t, void* devPtrdO_f16, void* devPtrdO_t,
    void* devPtrDescaleQ_t, void* devPtrDescaleK_t, void* devPtrDescaledO_t, void* devPtrcuSeqlensQ,
    void* devPtrcuSeqlensKV, void* devPtrDropoutSeed, void* devPtrDropoutOffset,
    cudnn_frontend::DataType_t qkv_tensor_type, cudnn_frontend::DataType_t o_tensor_type,
    cudnn_frontend::DataType_t do_tensor_type, cudnn_frontend::DataType_t dqkv_tensor_type,
    NVTEScalingMode scaling_mode, NVTE_QKV_Format qkv_scale_inv_format,
    NVTE_QKV_Format do_scale_inv_format, void* workspace, size_t* workspace_size,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  const auto cudnn_runtime_version = cudnnGetVersion();
  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_dropout = (dropout_probability != 0.0f);
  bool is_softmax_offset = (softmax_type != NVTE_Softmax_Type::NVTE_VANILLA_SOFTMAX);
  auto bias_b = b;
  auto bias_h = h;
  auto bias_sq = s_q;
  auto bias_skv = s_kv;
  NVTE_CHECK(~is_bias, "FP8 fused attention does not support pre/post_scale_bias yet!");
  NVTE_CHECK(~is_alibi, "FP8 fused attention does not support ALiBi yet!");
  bool is_delayed_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (dqkv_tensor_type == cudnn_frontend::DataType_t::FP8_E4M3 ||
                             dqkv_tensor_type == cudnn_frontend::DataType_t::FP8_E5M2);
  bool is_current_scaling = (scaling_mode == NVTE_DELAYED_TENSOR_SCALING) &&
                            (dqkv_tensor_type == cudnn_frontend::DataType_t::HALF ||
                             dqkv_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  bool is_mxfp8 = (scaling_mode == NVTE_MXFP8_1D_SCALING) &&
                  (dqkv_tensor_type == cudnn_frontend::DataType_t::HALF ||
                   dqkv_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);
  NVTE_CHECK(
      is_delayed_scaling || is_current_scaling || is_mxfp8,
      "FP8 fused attention only supports FP8DelayedScaling or FP8CurrentScaling or MXFP8 recipes!");
  NVTE_CHECK(!is_mxfp8 || cudnn_runtime_version >= 92100,
             "MXFP8 fused attention requires cuDNN 9.21.0 or later!");

  bool is_O_in_F16 = (o_tensor_type == cudnn_frontend::DataType_t::HALF ||
                      o_tensor_type == cudnn_frontend::DataType_t::BFLOAT16);

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
                               bias_sq,
                               bias_skv,
                               scaling_factor,
                               true,
                               dropout_probability,
                               qkv_layout,
                               o_format,
                               do_format,
                               dqkv_layout,
                               qkv_scale_inv_format,
                               do_scale_inv_format,
                               bias_type,
                               mask_type,
                               softmax_type,
                               window_size_left,
                               window_size_right,
                               bottom_right_diagonal,
                               deterministic,
                               qkv_tensor_type,
                               o_tensor_type,
                               do_tensor_type,
                               dqkv_tensor_type,
                               false};

    namespace fe = cudnn_frontend;
    using graph_and_tensors =
        std::tuple<std::shared_ptr<fe::graph::Graph>,
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Q_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // K_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // V
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // O
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // Stats
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dO
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dO_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dO_f16
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // attn_scale
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_q_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_k_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_o
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_dO
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_dO_t
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // descale_dP
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dQ
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dK
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dV
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_s
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // scale_dP
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dQ
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dK
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dV
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dQ
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dK
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dV
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // amax_dP
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // bias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dBias
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // d_softmax_offset
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // seq_kv
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // dropout_seed
                   std::shared_ptr<fe::graph::Tensor_attributes>>;  // dropout_offset

    using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
    static thread_local CacheType sdpa_fp8_bprop_cache;

    // Get plan from cache if cache is available, otherwise create one
    auto get_graph = [&](CacheType& cache, const FADescriptor_v1& descriptor) -> graph_and_tensors {
      // if hit, return
      auto it = cache.find(descriptor);
      if (it != cache.end()) {
        auto graph = it->second;
        return graph;
      }

      // otherwise, build the op_graph and the plan. Then update cache
      auto mha_graph = std::make_shared<fe::graph::Graph>();

      mha_graph->set_io_data_type(qkv_tensor_type)
          .set_intermediate_data_type(fe::DataType_t::FLOAT)
          .set_compute_data_type(fe::DataType_t::FLOAT);

      std::shared_ptr<fe::graph::Tensor_attributes> Q, Q_t, K, K_t, V, O, dO, dO_t, dO_f16, Stats,
          attn_scale;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_q, descale_q_t, descale_k, descale_k_t,
          descale_v;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_s, descale_o;
      std::shared_ptr<fe::graph::Tensor_attributes> descale_dP, descale_dO, descale_dO_t;
      std::shared_ptr<fe::graph::Tensor_attributes> scale_s, scale_dP;
      std::shared_ptr<fe::graph::Tensor_attributes> scale_dQ, scale_dK, scale_dV;
      std::shared_ptr<fe::graph::Tensor_attributes> bias, dBias, softmax_offset, d_softmax_offset;
      std::shared_ptr<fe::graph::Tensor_attributes> seq_q, seq_kv;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      // Q, K, V, O, dO, stats, attn_scale
      std::vector<int64_t> q_strides(4), k_strides(4), v_strides(4), o_strides(4), dO_strides(4);
      generateMatrixStridesWithLayout(b, h, hg, s_q, s_kv, d_qk, d_v, q_strides.data(),
                                      k_strides.data(), v_strides.data(), qkv_layout);
      generateMatrixStridesWithFormat(b, h, s_q, d_v, o_strides.data(), o_format);
      generateMatrixStridesWithFormat(b, h, s_q, d_v, dO_strides.data(), do_format);
      Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("Q")
                                .set_dim({b, h, s_q, d_qk})
                                .set_stride(q_strides)
                                .set_data_type(qkv_tensor_type));
      K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("K")
                                .set_dim({b, hg, s_kv, d_qk})
                                .set_stride(k_strides)
                                .set_data_type(qkv_tensor_type));
      V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("V")
                                .set_dim({b, hg, s_kv, d_v})
                                .set_stride(v_strides)
                                .set_data_type(qkv_tensor_type));
      O = mha_graph->tensor(fe::graph::Tensor_attributes()
                                .set_name("O")
                                .set_dim({b, h, s_q, d_v})
                                .set_stride(o_strides)
                                .set_data_type(o_tensor_type));
      dO = mha_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("dO")
                                 .set_dim({b, h, s_q, d_v})
                                 .set_stride(dO_strides)
                                 .set_data_type(do_tensor_type));
      Stats = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Stats")
                                    .set_dim({b, h, s_q, 1})
                                    .set_stride({h * s_q, s_q, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));
      attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("attn_scale")
                                         .set_dim({1, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_is_pass_by_value(true)
                                         .set_data_type(fe::DataType_t::FLOAT));

      // Descale_q, Descale_k, Descale_v, Descale_s, Scale_s, Descale_dP, Scale_dP, Descale_o, Descale_dO, Scale_dQ, Scale_dK, Scale_dV
      if (is_delayed_scaling || is_current_scaling) {
        descale_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("Descale_q")
                                          .set_dim({1, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::FLOAT));
        descale_k = mha_graph->tensor_like(descale_q, "Descale_q");
        descale_v = mha_graph->tensor_like(descale_q, "Descale_v");
        descale_s = mha_graph->tensor_like(descale_q, "Descale_s");
        scale_s = mha_graph->tensor_like(descale_q, "Scale_s");
        descale_dP = mha_graph->tensor_like(descale_q, "Descale_dP");
        scale_dP = mha_graph->tensor_like(descale_q, "Scale_dP");
        if (is_current_scaling && is_O_in_F16) {
          descale_o = mha_graph->tensor(1.0f);
        } else {
          descale_o = mha_graph->tensor_like(descale_q, "Descale_O");
        }
        descale_dO = mha_graph->tensor_like(descale_q, "Descale_dO");
        if (is_delayed_scaling) {
          scale_dQ = mha_graph->tensor_like(descale_q, "Scale_dQ");
          scale_dK = mha_graph->tensor_like(descale_q, "Scale_dK");
          scale_dV = mha_graph->tensor_like(descale_q, "Scale_dV");
        }
        if (is_current_scaling) {
          scale_dQ = mha_graph->tensor(1.0f);
          scale_dK = mha_graph->tensor(1.0f);
          scale_dV = mha_graph->tensor(1.0f);
        }
      } else if (is_mxfp8) {
        NVTE_QKV_Format q_format = nvte_get_q_format(qkv_layout);
        NVTE_QKV_Format kv_format = nvte_get_kv_format(qkv_layout);
        NVTE_QKV_Format q_scale_inv_format =
            (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET) ? qkv_scale_inv_format : q_format;
        NVTE_QKV_Format kv_scale_inv_format =
            (qkv_scale_inv_format != NVTE_QKV_Format_NOT_SET) ? qkv_scale_inv_format : kv_format;
        NVTE_QKV_Format do_scale_format_ =
            (do_scale_inv_format != NVTE_QKV_Format_NOT_SET) ? do_scale_inv_format : do_format;
        // Q_t, K_t, dO_t, dO_f16
        std::vector<int64_t> q_t_strides(4), k_t_strides(4), dO_t_strides(4);
        generateMatrixStridesWithFormat(b, h, s_q, d_qk, q_t_strides.data(), q_format);
        generateMatrixStridesWithFormat(b, hg, s_kv, d_qk, k_t_strides.data(), kv_format);
        generateMatrixStridesWithFormat(b, h, s_q, d_v, dO_t_strides.data(), do_format);
        Q_t = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("Q_t")
                                    .set_dim({b, h, s_q, d_qk})
                                    .set_stride(q_t_strides)
                                    .set_data_type(qkv_tensor_type));
        K_t = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("K_t")
                                    .set_dim({b, hg, s_kv, d_qk})
                                    .set_stride(k_t_strides)
                                    .set_data_type(qkv_tensor_type));
        dO_t = mha_graph->tensor(fe::graph::Tensor_attributes()
                                     .set_name("dO_t")
                                     .set_dim({b, h, s_q, d_v})
                                     .set_stride(dO_t_strides)
                                     .set_data_type(do_tensor_type));
        dO_f16 = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("dO_f16")
                                       .set_dim({b, h, s_q, d_v})
                                       .set_stride(dO_strides)
                                       .set_data_type(o_tensor_type));
        // Descale_q, Descale_q_t, Descale_k, Descale_k_t, Descale_v, Descale_dO, Descale_dO_t
        auto padded = pad_s_d_for_mxfp8(s_q, s_kv, d_qk, d_v);
        std::vector<int64_t> q_scale_strides(4), q_t_scale_strides(4), k_scale_strides(4),
            k_t_scale_strides(4), v_scale_strides(4), dO_scale_strides(4), dO_t_scale_strides(4);
        generateMatrixStridesWithFormat(b, h, padded.s_q_padded, padded.d_qk_scale_padded,
                                        q_scale_strides.data(), q_scale_inv_format);
        generateMatrixStridesWithFormat(b, h, padded.s_q_scale_padded, padded.d_qk_padded,
                                        q_t_scale_strides.data(), q_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_padded, padded.d_qk_scale_padded,
                                        k_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_scale_padded, padded.d_qk_padded,
                                        k_t_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, hg, padded.s_kv_padded, padded.d_v_scale_padded,
                                        v_scale_strides.data(), kv_scale_inv_format);
        generateMatrixStridesWithFormat(b, h, padded.s_q_padded, padded.d_v_scale_padded,
                                        dO_scale_strides.data(), do_scale_format_);
        generateMatrixStridesWithFormat(b, h, padded.s_q_scale_padded, padded.d_v_padded,
                                        dO_t_scale_strides.data(), do_scale_format_);
        descale_q =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_q")
                                  .set_dim({b, h, padded.s_q_padded, padded.d_qk_scale_padded})
                                  .set_stride(q_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_q_t =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_q_t")
                                  .set_dim({b, h, padded.s_q_scale_padded, padded.d_qk_padded})
                                  .set_stride(q_t_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_k =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_k")
                                  .set_dim({b, hg, padded.s_kv_padded, padded.d_qk_scale_padded})
                                  .set_stride(k_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_k_t =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_k_t")
                                  .set_dim({b, hg, padded.s_kv_scale_padded, padded.d_qk_padded})
                                  .set_stride(k_t_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_v =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_v")
                                  .set_dim({b, hg, padded.s_kv_padded, padded.d_v_scale_padded})
                                  .set_stride(v_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_dO =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_dO")
                                  .set_dim({b, h, padded.s_q_padded, padded.d_v_scale_padded})
                                  .set_stride(dO_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
        descale_dO_t =
            mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Descale_dO_t")
                                  .set_dim({b, h, padded.s_q_scale_padded, padded.d_v_padded})
                                  .set_stride(dO_t_scale_strides)
                                  .set_data_type(fe::DataType_t::FP8_E8M0)
                                  .set_reordering_type(fe::TensorReordering_t::F8_128x4));
      }

      fe::graph::SDPA_fp8_backward_attributes sdpa_backward_options;
      sdpa_backward_options = fe::graph::SDPA_fp8_backward_attributes()
                                  .set_name("sdpa_fp8_backward")
                                  .set_causal_mask(is_causal)
                                  .set_attn_scale(attn_scale);

      fe::DiagonalAlignment_t const& diagonal_alignment =
          bottom_right_diagonal ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
                                : fe::DiagonalAlignment_t::TOP_LEFT;
      sdpa_backward_options.set_diagonal_alignment(diagonal_alignment);

      if (cudnn_runtime_version >= 92100) {
        if (window_size_left != -1) {
          sdpa_backward_options.set_diagonal_band_left_bound(window_size_left + 1);
        }
        if (window_size_right != -1) {
          sdpa_backward_options.set_diagonal_band_right_bound(window_size_right);
        }
      }

      // sdpa_backward_options.set_alibi_mask(is_alibi);

      // if (is_bias) {
      //     bias = mha_graph->tensor(fe::graph::Tensor_attributes()
      //                     .set_name("bias")
      //                     .set_dim({bias_b, bias_h, bias_sq, bias_skv})
      //                     .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
      //     dBias = mha_graph->tensor(fe::graph::Tensor_attributes()
      //                     .set_name("dBias")
      //                     .set_dim({bias_b, bias_h, bias_sq, bias_skv})
      //                     .set_stride({bias_h * bias_sq * bias_skv, bias_sq * bias_skv, bias_skv, 1}));
      //     sdpa_backward_options.set_bias(bias);
      // bias shapes [1, 1, s, s], [b, 1, s, s], [b, h, s, s], [1, h, s, s] are supported for dbias calculation
      // bias shape [1, 1, 1, s] is not supported for dbias calculation as of cuDNN 9.18
      // if (!((bias_b == 1) && (bias_h == 1) && (bias_sq == 1))) {
      //    sdpa_backward_options.set_dbias(dBias);
      //  }
      // }

      if (cudnn_runtime_version >= 91900) {
        sdpa_backward_options.set_deterministic_algorithm(deterministic);
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

      std::shared_ptr<fe::graph::Tensor_attributes> dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP;
      if (is_delayed_scaling || is_current_scaling) {
        std::tie(dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP) =
            std::apply([](const auto&... elems) { return std::make_tuple(elems...); },
                       mha_graph->sdpa_fp8_backward(Q, K, V, O, dO, Stats, descale_q, descale_k,
                                                    descale_v, descale_o, descale_dO, descale_s,
                                                    descale_dP, scale_s, scale_dQ, scale_dK,
                                                    scale_dV, scale_dP, sdpa_backward_options));
      } else if (is_mxfp8) {
        std::tie(dQ, dK, dV, amax_dQ, amax_dK, amax_dV) = std::apply(
            [](const auto&... elems) { return std::make_tuple(elems...); },
            mha_graph->sdpa_fp8_backward(Q, Q_t, K, K_t, V, O, dO_f16, dO, dO_t, Stats, descale_q,
                                         descale_q_t, descale_k, descale_k_t, descale_v, descale_dO,
                                         descale_dO_t, sdpa_backward_options));
      }
      std::vector<int64_t> dq_strides(4), dk_strides(4), dv_strides(4);
      generateMatrixStridesWithLayout(b, h, hg, s_q, s_kv, d_qk, d_v, dq_strides.data(),
                                      dk_strides.data(), dv_strides.data(), dqkv_layout);
      dQ->set_output(true)
          .set_dim({b, h, s_q, d_qk})
          .set_stride(dq_strides)
          .set_data_type(dqkv_tensor_type);
      dK->set_output(true)
          .set_dim({b, hg, s_kv, d_qk})
          .set_stride(dk_strides)
          .set_data_type(dqkv_tensor_type);
      dV->set_output(true)
          .set_dim({b, hg, s_kv, d_v})
          .set_stride(dv_strides)
          .set_data_type(dqkv_tensor_type);
      amax_dQ->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);
      amax_dK->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);
      amax_dV->set_output(!is_mxfp8)
          .set_dim({1, 1, 1, 1})
          .set_stride({1, 1, 1, 1})
          .set_data_type(fe::DataType_t::FLOAT);
      if (is_delayed_scaling || is_current_scaling) {
        amax_dP->set_output(true)
            .set_dim({1, 1, 1, 1})
            .set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::FLOAT);
      }

      std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // K
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // V
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // O
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // Stats
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_q
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_k
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_v
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_o
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dO
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dP
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dK
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dV
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_s
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dP
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // dV
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dQ
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dK
                 std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dV
                 std::shared_ptr<fe::graph::Tensor_attributes>>  // amax_dP
          key_tensors_tuple = std::make_tuple(
              Q, K, V, O, Stats, dO, attn_scale, descale_q, descale_k, descale_v, descale_o,
              descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP,
              dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP);
      auto mxfp8_tensors_tuple =
          is_mxfp8 ? std::make_tuple(Q_t, K_t, dO_f16, dO_t, descale_q_t, descale_k_t, descale_dO_t)
                   : std::make_tuple(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
      auto bias_tuple = is_bias ? std::make_tuple(bias, dBias) : std::make_tuple(nullptr, nullptr);
      auto softmax_offset_tuple = is_softmax_offset
                                      ? std::make_tuple(softmax_offset, d_softmax_offset)
                                      : std::make_tuple(nullptr, nullptr);
      auto padding_tuple =
          is_padding ? std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple =
          std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, mxfp8_tensors_tuple,
                         bias_tuple, softmax_offset_tuple, padding_tuple, dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };
    auto [mha_graph, Q, K, V, O, Stats, dO, attn_scale, descale_q, descale_k, descale_v, descale_o,
          descale_dO, descale_s, descale_dP, scale_s, scale_dQ, scale_dK, scale_dV, scale_dP, dQ,
          dK, dV, amax_dQ, amax_dK, amax_dV, amax_dP, Q_t, K_t, dO_f16, dO_t, descale_q_t,
          descale_k_t, descale_dO_t, bias, dBias, softmax_offset, d_softmax_offset, seq_q, seq_kv,
          dropout_seed, dropout_offset] = get_graph(sdpa_fp8_bprop_cache, descriptor);

    auto plan_workspace_size = mha_graph->get_workspace_size();

    // Exit to request upper level API to allocate memory if needed
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    if (workspace == nullptr) {
      *workspace_size = plan_workspace_size + actual_seqlen_workspace_size;
      return;
    }

    // cuDNN stream check needs to be moved here to support dummy kernel calls with
    // null streams for sizing the cuDNN workspace.
    NVTE_CHECK_CUDNN(cudnnSetStream(handle, stream));

    // build variant pack
    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {Q, devPtrQ},
        {K, devPtrK},
        {V, devPtrV},
        {O, devPtrO},
        {Stats, devPtrM},
        {dO, devPtrdO},
        {attn_scale, &scaling_factor},
        {descale_q, devPtrDescaleQ},
        {descale_k, devPtrDescaleK},
        {descale_v, devPtrDescaleV},
        {descale_dO, devPtrDescaledO},
        {dQ, devPtrdQ},
        {dK, devPtrdK},
        {dV, devPtrdV},
    };
    if (is_delayed_scaling || is_current_scaling) {
      variant_pack[descale_s] = devPtrDescaleS;
      variant_pack[descale_dP] = devPtrDescaledP;
      variant_pack[scale_s] = devPtrScaleS;
      variant_pack[scale_dP] = devPtrScaledP;
      variant_pack[amax_dP] = devPtrAmaxdP;
      variant_pack[amax_dQ] = devPtrAmaxdQ;
      variant_pack[amax_dK] = devPtrAmaxdK;
      variant_pack[amax_dV] = devPtrAmaxdV;
    }
    if (is_delayed_scaling || (is_current_scaling && !is_O_in_F16)) {
      variant_pack[descale_o] = devPtrDescaleO;
    }
    if (is_delayed_scaling) {
      variant_pack[scale_dQ] = devPtrScaledQ;
      variant_pack[scale_dK] = devPtrScaledK;
      variant_pack[scale_dV] = devPtrScaledV;
    }
    if (is_mxfp8) {
      variant_pack[Q_t] = devPtrQ_t;
      variant_pack[K_t] = devPtrK_t;
      variant_pack[dO_f16] = devPtrdO_f16;
      variant_pack[dO_t] = devPtrdO_t;
      variant_pack[descale_q_t] = devPtrDescaleQ_t;
      variant_pack[descale_k_t] = devPtrDescaleK_t;
      variant_pack[descale_dO_t] = devPtrDescaledO_t;
    }

    /* if (is_bias) {
       variant_pack[bias] = devPtrBias;
       if ((bias_b == 1) && (bias_h == h)) {
         variant_pack[dBias] = devPtrdBias;
       } else {
         variant_pack[dBias] = nullptr;
       }
    } */

    if (is_padding) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
      void* devActualSeqlenQ = static_cast<int8_t*>(workspace) + plan_workspace_size;
      void* devActualSeqlenKV = static_cast<int8_t*>(devActualSeqlenQ) + b * sizeof(int32_t);
      cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
          b, b, static_cast<const int32_t*>(devPtrcuSeqlensQ),  // TODO(pass max_b)
          static_cast<const int32_t*>(devPtrcuSeqlensKV), static_cast<int32_t*>(devActualSeqlenQ),
          static_cast<int32_t*>(devActualSeqlenKV));
      NVTE_CHECK_CUDA(cudaGetLastError());
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;
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
  } catch (cudnn_frontend::cudnnException& e) {
    NVTE_ERROR(e.what());
  }
}  // NOLINT(readability/fn_size)

}  // namespace fused_attn

// fused attention FWD FP8 with separate Q, K, V
void fused_attn_fp8_fwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, bool is_training, float attn_scale,
    float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
    NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_Softmax_Type softmax_type, size_t window_size_left, size_t window_size_right,
    bool bottom_right_diagonal, const Tensor* input_Q, const Tensor* input_K, const Tensor* input_V,
    const Tensor* input_SoftmaxOffset, Tensor* input_output_S, Tensor* output_O,
    NVTETensorPack* Aux_CTX_Tensors, const Tensor* cu_seqlens_q, const Tensor* cu_seqlens_kv,
    const Tensor* rng_state, Tensor* workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  void *devPtrQ = nullptr, *devPtrK = nullptr, *devPtrV = nullptr;
  void *devPtrDescaleQ = nullptr, *devPtrDescaleK = nullptr, *devPtrDescaleV = nullptr;
  void *devPtrO = nullptr, *devPtrAmaxO = nullptr, *devPtrScaleO = nullptr;
  void *devPtrAmaxS = nullptr, *devPtrScaleS = nullptr, *devPtrDescaleS = nullptr;
  devPtrQ = input_Q->data.dptr;
  devPtrDescaleQ = input_Q->scale_inv.dptr;
  devPtrK = input_K->data.dptr;
  devPtrDescaleK = input_K->scale_inv.dptr;
  devPtrO = output_O->data.dptr;
  if (input_Q->scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    devPtrV = input_V->data.dptr;
    devPtrDescaleV = input_V->scale_inv.dptr;
    devPtrScaleO = output_O->scale.dptr;
    devPtrAmaxS = input_output_S->amax.dptr;
    devPtrScaleS = input_output_S->scale.dptr;
    devPtrDescaleS = input_output_S->scale_inv.dptr;
    devPtrAmaxO = output_O->amax.dptr;
  } else if (input_Q->scaling_mode == NVTE_MXFP8_1D_SCALING) {
    devPtrV = input_V->columnwise_data.dptr;
    devPtrDescaleV = input_V->columnwise_scale_inv.dptr;
  }
  void* devPtrSoftmaxOffset = nullptr;
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    devPtrSoftmaxOffset = input_SoftmaxOffset->data.dptr;
  }
  void* devPtrM = nullptr;
  if (Aux_CTX_Tensors->size == 0) {
    int i = 0;
    Tensor* output_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_M->data.dptr = nullptr;
    output_M->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
    output_M->data.dtype = DType::kFloat32;
    Tensor* output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = nullptr;
    output_rng_state->data.shape = {2};
    output_rng_state->data.dtype = DType::kInt64;
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      Tensor* output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = nullptr;
      output_softmax_offset->data.shape = {1, num_attn_heads, 1, 1};
      output_softmax_offset->data.dtype = DType::kFloat32;
    }
    Aux_CTX_Tensors->size = i;
  } else if (Aux_CTX_Tensors->size >= 2) {
    int i = 0;
    Tensor* output_M = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    devPtrM = output_M->data.dptr;
    Tensor* output_rng_state = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    if (softmax_type != NVTE_VANILLA_SOFTMAX) {
      Tensor* output_softmax_offset = convertNVTETensorCheck(Aux_CTX_Tensors->tensors[i++]);
      output_softmax_offset->data.dptr = devPtrSoftmaxOffset;
    }
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void* devPtrcuSeqlensQ =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  void* devPtrDropoutSeed =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  const DType O_type = output_O->data.dtype;
  size_t workspace_size = 0;

  NVTE_QKV_Format qkv_format = nvte_get_qkv_format(qkv_layout);
  if ((qkv_format == NVTE_QKV_Format::NVTE_BSHD) || (qkv_format == NVTE_QKV_Format::NVTE_SBHD) ||
      (qkv_format == NVTE_QKV_Format::NVTE_BHSD)) {
    fused_attn::fused_attn_fp8_fwd_impl(
        batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v,
        is_training, attn_scale, p_dropout, qkv_layout, o_format, bias_type, mask_type,
        softmax_type, window_size_left, window_size_right, bottom_right_diagonal, devPtrQ, devPtrK,
        devPtrV, devPtrSoftmaxOffset, devPtrM, devPtrO, devPtrDescaleQ, devPtrDescaleK,
        devPtrDescaleV, devPtrDescaleS, devPtrScaleS, devPtrScaleO, devPtrAmaxO, devPtrAmaxS,
        devPtrcuSeqlensQ, devPtrcuSeqlensKV, devPtrDropoutSeed, devPtrDropoutOffset,
        get_cudnn_fe_dtype(QKV_type), get_cudnn_fe_dtype(O_type), input_Q->scaling_mode,
        qkv_scale_inv_format, workspace->data.dptr, &workspace_size, stream, handle);
  } else {
    NVTE_ERROR("FP8 fused attention only supports qkv_format=BSHD, SBHD, or BHSD.\n");
  }

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
  }
}
// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format, NVTE_QKV_Format do_format,
    NVTE_QKV_Layout dqkv_layout, NVTE_QKV_Format qkv_scale_inv_format,
    NVTE_QKV_Format do_scale_inv_format, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    NVTE_Softmax_Type softmax_type, size_t window_size_left, size_t window_size_right,
    bool bottom_right_diagonal, bool deterministic, const Tensor* input_Q, const Tensor* input_K,
    const Tensor* input_V, const Tensor* input_O, const Tensor* input_dO,
    const Tensor* input_dO_f16, const Tensor* input_M, const Tensor* input_S,
    const Tensor* input_SoftmaxOffset, Tensor* input_output_dP, const Tensor* output_dQ,
    const Tensor* output_dK, const Tensor* output_dV, Tensor* output_dSoftmaxOffset,
    const Tensor* cu_seqlens_q, const Tensor* cu_seqlens_kv, const Tensor* rng_state,
    Tensor* workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  void* devPtrQ = input_Q->data.dptr;
  void* devPtrK = input_K->data.dptr;
  void* devPtrV = input_V->data.dptr;
  void* devPtrDescaleQ = input_Q->scale_inv.dptr;
  void* devPtrDescaleK = input_K->scale_inv.dptr;
  void* devPtrDescaleV = input_V->scale_inv.dptr;
  void *devPtrQ_t = nullptr, *devPtrK_t = nullptr, *devPtrDescaleQ_t = nullptr,
       *devPtrDescaleK_t = nullptr;
  if (input_Q->scaling_mode == NVTE_MXFP8_1D_SCALING) {
    devPtrQ_t = input_Q->columnwise_data.dptr;
    devPtrDescaleQ_t = input_Q->columnwise_scale_inv.dptr;
    devPtrK_t = input_K->columnwise_data.dptr;
    devPtrDescaleK_t = input_K->columnwise_scale_inv.dptr;
  }

  void* devPtrO = input_O->data.dptr;
  const DType O_type = input_O->data.dtype;
  void* devPtrDescaleO = nullptr;
  if (O_type == DType::kFloat8E4M3 || O_type == DType::kFloat8E5M2) {
    devPtrDescaleO = input_O->scale_inv.dptr;
  }
  void* devPtrdO = input_dO->data.dptr;
  void* devPtrDescaledO = input_dO->scale_inv.dptr;
  void *devPtrdO_t = nullptr, *devPtrdO_f16 = nullptr, *devPtrDescaledO_t = nullptr;
  if (input_dO->scaling_mode == NVTE_MXFP8_1D_SCALING) {
    devPtrdO_t = input_dO->columnwise_data.dptr;
    devPtrdO_f16 = input_dO_f16->data.dptr;
    devPtrDescaledO_t = input_dO->columnwise_scale_inv.dptr;
  }

  void* devPtrM = input_M->data.dptr;

  void *devPtrScaleS = nullptr, *devPtrDescaleS = nullptr, *devPtrAmaxdP = nullptr,
       *devPtrScaledP = nullptr, *devPtrDescaledP = nullptr;
  if (input_Q->scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    devPtrScaleS = input_S->scale.dptr;
    devPtrDescaleS = input_S->scale_inv.dptr;
    devPtrAmaxdP = input_output_dP->amax.dptr;
    devPtrScaledP = input_output_dP->scale.dptr;
    devPtrDescaledP = input_output_dP->scale_inv.dptr;
  }

  void* devPtrSoftmaxOffset = nullptr;
  void* devPtrdSoftmaxOffset = nullptr;
  if (softmax_type != NVTE_VANILLA_SOFTMAX) {
    devPtrSoftmaxOffset = input_SoftmaxOffset->data.dptr;
    devPtrdSoftmaxOffset = output_dSoftmaxOffset->data.dptr;
  }

  void* devPtrdQ = output_dQ->data.dptr;
  void* devPtrdK = output_dK->data.dptr;
  void* devPtrdV = output_dV->data.dptr;
  void *devPtrAmaxdQ = nullptr, *devPtrAmaxdK = nullptr, *devPtrAmaxdV = nullptr,
       *devPtrScaledQ = nullptr, *devPtrScaledK = nullptr, *devPtrScaledV = nullptr;
  if (input_Q->scaling_mode == NVTE_DELAYED_TENSOR_SCALING) {
    devPtrAmaxdQ = output_dQ->amax.dptr;
    devPtrAmaxdK = output_dK->amax.dptr;
    devPtrAmaxdV = output_dV->amax.dptr;
    devPtrScaledQ = output_dQ->scale.dptr;
    devPtrScaledK = output_dK->scale.dptr;
    devPtrScaledV = output_dV->scale.dptr;
  }

  void* devPtrcuSeqlensQ =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV =
      reinterpret_cast<void*>(reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  void* devPtrDropoutSeed =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset =
      reinterpret_cast<void*>(reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  const DType dO_type = input_dO->data.dtype;
  const DType dQKV_type = output_dQ->data.dtype;
  size_t workspace_size = 0;

  NVTE_QKV_Format dqkv_format = nvte_get_qkv_format(dqkv_layout);
  if ((dqkv_format == NVTE_QKV_Format::NVTE_BSHD) || (dqkv_format == NVTE_QKV_Format::NVTE_SBHD) ||
      (dqkv_format == NVTE_QKV_Format::NVTE_BHSD)) {
    fused_attn::fused_attn_fp8_bwd_impl(
        batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim_qk, head_dim_v,
        attn_scale, p_dropout, qkv_layout, o_format, do_format, dqkv_layout, bias_type, mask_type,
        softmax_type, window_size_left, window_size_right, bottom_right_diagonal, deterministic,
        devPtrQ, devPtrK, devPtrV, devPtrM, devPtrO, devPtrdO, devPtrSoftmaxOffset, devPtrdQ,
        devPtrdK, devPtrdV, devPtrdSoftmaxOffset, devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
        devPtrDescaleO, devPtrDescaledO, devPtrDescaleS, devPtrDescaledP, devPtrScaleS,
        devPtrScaledP, devPtrScaledQ, devPtrScaledK, devPtrScaledV, devPtrAmaxdP, devPtrAmaxdQ,
        devPtrAmaxdK, devPtrAmaxdV, devPtrQ_t, devPtrK_t, devPtrdO_f16, devPtrdO_t,
        devPtrDescaleQ_t, devPtrDescaleK_t, devPtrDescaledO_t, devPtrcuSeqlensQ, devPtrcuSeqlensKV,
        devPtrDropoutSeed, devPtrDropoutOffset, get_cudnn_fe_dtype(QKV_type),
        get_cudnn_fe_dtype(O_type), get_cudnn_fe_dtype(dO_type), get_cudnn_fe_dtype(dQKV_type),
        input_dO->scaling_mode, qkv_scale_inv_format, do_scale_inv_format, workspace->data.dptr,
        &workspace_size, stream, handle);
  } else {
    NVTE_ERROR("FP8 fused attention only supports dqkv_format=BSHD, SBHD, or BHSD.\n");
  }

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
  }
}
}  // namespace transformer_engine
