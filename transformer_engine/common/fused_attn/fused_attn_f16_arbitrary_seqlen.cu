/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d, int64_t bias_b,
    int64_t bias_h, bool is_training, float scaling_factor, float dropout_probability,
    NVTE_QKV_Layout layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, void *devPtrQ,
    void *devPtrK, void *devPtrV, void *devPtrBias, void *devPtrSoftmaxStats, void *devPtrO,
    void *devPtrDropoutSeed, void *devPtrDropoutOffset, void *devPtrCuSeqlensQ,
    void *devPtrCuSeqlensKV, void *devPtrSeqOffsetsQ, void *devPtrSeqOffsetsKV,
    cudnn_frontend::DataType_t tensorType, void *workspace, size_t *workspace_size,
    cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;
  bool is_bias = (bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);
  bool is_alibi = (bias_type == NVTE_Bias_Type::NVTE_ALIBI);
  bool is_causal = ((mask_type == NVTE_Mask_Type::NVTE_CAUSAL_MASK) ||
                    (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_dropout = (is_training && dropout_probability != 0.0f);
  bool is_ragged = (nvte_get_qkv_format(layout) == NVTE_QKV_Format::NVTE_THD);
  if (is_ragged) {
    NVTE_CHECK(is_padding, "Ragged QKV input requires padding or padding_causal mask!");
  }

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d,
                               bias_b,
                               bias_h,
                               scaling_factor,
                               is_training,
                               dropout_probability,
                               layout,
                               bias_type,
                               mask_type,
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
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_q
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_k
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_v
                   std::shared_ptr<fe::graph::Tensor_attributes>,   // offset_o
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
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      offset_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_q")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
      offset_k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_k")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
      offset_v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_v")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
      offset_o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_o")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));

      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_Q_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_K_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_V_Matrix);

      if (is_ragged) {
        Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride(q_stride)
                                  .set_ragged_offset(offset_q));
        K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(k_stride)
                                  .set_ragged_offset(offset_k));
        V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(v_stride)
                                  .set_ragged_offset(offset_v));
      } else {
        Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride(q_stride));
        K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(k_stride));
        V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(v_stride));
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
                         .set_attn_scale(attn_scale);

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
      generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_O_Matrix);
      if (is_ragged) {
        O->set_output(true)
            .set_dim({b, h, s_q, d})
            .set_stride(o_stride)
            .set_ragged_offset(offset_o);
      } else {
        O->set_output(true).set_dim({b, h, s_q, d}).set_stride(o_stride);
      }

      Stats->set_output(true)
          .set_data_type(fe::DataType_t::FLOAT)
          .set_dim({b, h, s_q, 1})
          .set_stride({h * s_q, s_q, 1, 1});

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
      auto offset_tuple = is_ragged ? std::make_tuple(offset_q, offset_k, offset_v, offset_o)
                                    : std::make_tuple(nullptr, nullptr, nullptr, nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple = std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, Stats_tuple,
                                         bias_tuple, padding_tuple, offset_tuple, dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };

    auto [mha_graph, Q, K, V, attn_scale, O, Stats, bias, seq_q, seq_kv, offset_q, offset_k,
          offset_v, offset_o, dropout_seed, dropout_offset] =
        get_graph(sdpa_f16_fprop_cache, descriptor);

    auto plan_workspace_size = mha_graph->get_workspace_size();
    // Exit to request upper level API to allocate memory if needed
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    size_t seqlen_offsets_workspace_size = 4 * (b + 1) * sizeof(int32_t);
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
      void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
      cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
          b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
          static_cast<const int32_t *>(devPtrCuSeqlensKV), static_cast<int32_t *>(devActualSeqlenQ),
          static_cast<int32_t *>(devActualSeqlenKV));
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;
    }

    if (is_ragged) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block) / nthreads_per_block;
      void *devOffsetsQ =
          static_cast<int8_t *>(workspace) + plan_workspace_size + actual_seqlen_workspace_size;
      void *devOffsetsK = static_cast<int8_t *>(devOffsetsQ) + (b + 1) * sizeof(int32_t);
      void *devOffsetsV = static_cast<int8_t *>(devOffsetsK) + (b + 1) * sizeof(int32_t);
      void *devOffsetsO = static_cast<int8_t *>(devOffsetsV) + (b + 1) * sizeof(int32_t);
      NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
      cu_seqlens_padded_to_offsets<<<grid, nthreads_per_block, 0, stream>>>(
          layout_group, b, h, hg, d, static_cast<int32_t *>(devPtrSeqOffsetsQ),
          static_cast<int32_t *>(devPtrSeqOffsetsKV), static_cast<int32_t *>(devOffsetsQ),
          static_cast<int32_t *>(devOffsetsK), static_cast<int32_t *>(devOffsetsV),
          static_cast<int32_t *>(devOffsetsO));
      variant_pack[offset_q] = devOffsetsQ;
      variant_pack[offset_k] = devOffsetsK;
      variant_pack[offset_v] = devOffsetsV;
      variant_pack[offset_o] = devOffsetsO;
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
    int64_t b, int64_t h, int64_t hg, int64_t s_q, int64_t s_kv, int64_t d, int64_t bias_b,
    int64_t bias_h, float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type, void *devPtrQ, void *devPtrKTranspose,
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
  bool is_padding = ((mask_type == NVTE_Mask_Type::NVTE_PADDING_MASK) ||
                     (mask_type == NVTE_Mask_Type::NVTE_PADDING_CAUSAL_MASK));
  bool is_dropout = (dropout_probability != 0.0f);
  bool is_ragged = (nvte_get_qkv_format(layout) == NVTE_QKV_Format::NVTE_THD);

  try {
    FADescriptor_v1 descriptor{b,
                               h,
                               hg,
                               s_q,
                               s_kv,
                               d,
                               bias_b,
                               bias_h,
                               scaling_factor,
                               true,
                               dropout_probability,
                               layout,
                               bias_type,
                               mask_type,
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
      std::shared_ptr<fe::graph::Tensor_attributes> offset_q, offset_k, offset_v, offset_o;
      std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

      offset_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_q")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
      offset_k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_k")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
      offset_v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_v")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
      offset_o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                       .set_name("offset_o")
                                       .set_dim({b + 1, 1, 1, 1})
                                       .set_stride({1, 1, 1, 1})
                                       .set_data_type(fe::DataType_t::INT32));
      std::vector<int64_t> q_stride(4);
      std::vector<int64_t> k_stride(4);
      std::vector<int64_t> v_stride(4);
      std::vector<int64_t> o_stride(4);
      generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_Q_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_K_Matrix);
      generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_V_Matrix);
      generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(), layout,
                            NVTE_QKV_Matrix::NVTE_O_Matrix);

      if (is_ragged) {
        q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride(q_stride)
                                  .set_ragged_offset(offset_q));
        k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(k_stride)
                                  .set_ragged_offset(offset_k));
        v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(v_stride)
                                  .set_ragged_offset(offset_v));
        o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("O")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride(o_stride)
                                  .set_ragged_offset(offset_o));
        dO = mha_graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("dO")
                                   .set_dim({b, h, s_q, d})
                                   .set_stride(o_stride)
                                   .set_ragged_offset(offset_o));
      } else {
        q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("Q")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride(q_stride));
        k = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("K")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(k_stride));
        v = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("V")
                                  .set_dim({b, hg, s_kv, d})
                                  .set_stride(v_stride));
        o = mha_graph->tensor(fe::graph::Tensor_attributes()
                                  .set_name("O")
                                  .set_dim({b, h, s_q, d})
                                  .set_stride(o_stride));
        dO = mha_graph->tensor(fe::graph::Tensor_attributes()
                                   .set_name("dO")
                                   .set_dim({b, h, s_q, d})
                                   .set_stride(o_stride));
      }
      stats = mha_graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name("stats")
                                    .set_dim({b, h, s_q, 1})
                                    .set_stride({h * s_q, s_q, 1, 1})
                                    .set_data_type(fe::DataType_t::FLOAT));

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
                                  .set_attn_scale(attn_scale);

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

      if (is_ragged) {
        dQ->set_output(true)
            .set_dim({b, h, s_q, d})
            .set_stride(q_stride)
            .set_ragged_offset(offset_q);
        dK->set_output(true)
            .set_dim({b, hg, s_kv, d})
            .set_stride(k_stride)
            .set_ragged_offset(offset_k);
        dV->set_output(true)
            .set_dim({b, hg, s_kv, d})
            .set_stride(v_stride)
            .set_ragged_offset(offset_v);
      } else {
        dQ->set_output(true).set_dim({b, h, s_q, d}).set_stride(q_stride);
        dK->set_output(true).set_dim({b, hg, s_kv, d}).set_stride(k_stride);
        dV->set_output(true).set_dim({b, hg, s_kv, d}).set_stride(v_stride);
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
      auto offset_tuple = is_ragged ? std::make_tuple(offset_q, offset_k, offset_v, offset_o)
                                    : std::make_tuple(nullptr, nullptr, nullptr, nullptr);
      auto dropout_tuple = is_dropout ? std::make_tuple(dropout_seed, dropout_offset)
                                      : std::make_tuple(nullptr, nullptr);

      NVTE_CHECK_CUDNN_FE(mha_graph->validate());
      NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
      NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
      NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

      auto return_tuple = std::tuple_cat(std::make_tuple(mha_graph), key_tensors_tuple, bias_tuple,
                                         padding_tuple, offset_tuple, dropout_tuple);
      cache.insert({descriptor, return_tuple});

      return return_tuple;
    };

    auto [mha_graph, q, k, v, o, dO, stats, attn_scale, dQ, dK, dV, bias, dBias, seq_q, seq_kv,
          offset_q, offset_k, offset_v, offset_o, dropout_seed, dropout_offset] =
        get_graph(sdpa_f16_bprop_cache, descriptor);

    auto plan_workspace_size = mha_graph->get_workspace_size();

    // Exit to request upper level API to allocate memory if needed
    size_t actual_seqlen_workspace_size = 2 * b * sizeof(int32_t);
    size_t seqlen_offsets_workspace_size = 4 * (b + 1) * sizeof(int32_t);
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
      void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
      cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
          b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
          static_cast<const int32_t *>(devPtrCuSeqlensKV), static_cast<int32_t *>(devActualSeqlenQ),
          static_cast<int32_t *>(devActualSeqlenKV));
      variant_pack[seq_q] = devActualSeqlenQ;
      variant_pack[seq_kv] = devActualSeqlenKV;
    }

    if (is_ragged) {
      constexpr size_t nthreads_per_block = 128;
      const size_t grid = (b + nthreads_per_block) / nthreads_per_block;
      void *devOffsetsQ =
          static_cast<int8_t *>(workspace) + plan_workspace_size + actual_seqlen_workspace_size;
      void *devOffsetsK = static_cast<int8_t *>(devOffsetsQ) + (b + 1) * sizeof(int32_t);
      void *devOffsetsV = static_cast<int8_t *>(devOffsetsK) + (b + 1) * sizeof(int32_t);
      void *devOffsetsO = static_cast<int8_t *>(devOffsetsV) + (b + 1) * sizeof(int32_t);
      NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(layout);
      cu_seqlens_padded_to_offsets<<<grid, nthreads_per_block, 0, stream>>>(
          layout_group, b, h, hg, d, static_cast<int32_t *>(devPtrSeqOffsetsQ),
          static_cast<int32_t *>(devPtrSeqOffsetsKV), static_cast<int32_t *>(devOffsetsQ),
          static_cast<int32_t *>(devOffsetsK), static_cast<int32_t *>(devOffsetsV),
          static_cast<int32_t *>(devOffsetsO));
      variant_pack[offset_q] = devOffsetsQ;
      variant_pack[offset_k] = devOffsetsK;
      variant_pack[offset_v] = devOffsetsV;
      variant_pack[offset_o] = devOffsetsO;
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
    size_t batch, size_t num_attn_heads, size_t max_seqlen, size_t head_dim, bool is_training,
    float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type mask_type, const Tensor *input_QKV, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens, const Tensor *cu_seqlens_padded,
    const Tensor *rng_state, Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_QKV->data.dtype;
  void *devPtrQKV = input_QKV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    stride = typeToSize(QKV_type) * num_attn_heads * head_dim;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    stride = typeToSize(QKV_type) * head_dim;
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

  if (Aux_CTX_Tensors->size == 0) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {batch, num_attn_heads, max_seqlen, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen, max_seqlen};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {batch, num_attn_heads, max_seqlen, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_fwd_impl(
      batch, num_attn_heads, num_attn_heads, max_seqlen, max_seqlen, head_dim, bias_b, bias_h,
      is_training, attn_scale, p_dropout, qkv_layout, bias_type, mask_type, devPtrQ, devPtrK,
      devPtrV, devPtrBias, devPtrS, devPtrO, devPtrDropoutSeed, devPtrDropoutOffset,
      devPtrCuSeqlens, devPtrCuSeqlens, devPtrSeqOffsets, devPtrSeqOffsets,
      get_cudnn_fe_dtype(QKV_type), workspace->data.dptr, &workspace_size, stream, handle);

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
    size_t batch, size_t num_attn_heads, size_t max_seqlen, size_t head_dim, float attn_scale,
    float p_dropout, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    const Tensor *input_QKV, const Tensor *input_O, const Tensor *input_dO,
    const Tensor *input_Bias, Tensor *output_S, Tensor *output_dQKV, Tensor *output_dBias,
    const Tensor *cu_seqlens, const Tensor *cu_seqlens_padded, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_QKV->data.dtype;
  void *devPtrQKV = input_QKV->data.dptr;

  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_3HD) {
    stride = typeToSize(QKV_type) * num_attn_heads * head_dim;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_H3D) {
    stride = typeToSize(QKV_type) * head_dim;
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
      batch, num_attn_heads, num_attn_heads, max_seqlen, max_seqlen, head_dim, bias_b, bias_h,
      attn_scale, p_dropout, qkv_layout, bias_type, mask_type, devPtrQ, devPtrK, devPtrV, devPtrO,
      devPtrSoftmaxStats, devPtrBias, devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdBias,
      devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlens, devPtrCuSeqlens, devPtrSeqOffsets,
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
void fused_attn_arbitrary_seqlen_fwd_kvpacked(
    size_t batch, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim, bool is_training, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    const Tensor *input_Q, const Tensor *input_KV, const Tensor *input_Bias, Tensor *output_O,
    NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *cu_seqlens_q_padded, const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_Q->data.dtype;
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = typeToSize(QKV_type) * num_gqa_groups * head_dim;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = typeToSize(QKV_type) * head_dim;
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

  if (Aux_CTX_Tensors->size == 0) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_fwd_impl(
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
      is_training, attn_scale, p_dropout, qkv_layout, bias_type, mask_type, devPtrQ, devPtrK,
      devPtrV, devPtrBias, devPtrS, devPtrO, devPtrDropoutSeed, devPtrDropoutOffset,
      devPtrCuSeqlensQ, devPtrCuSeqlensKV, devPtrSeqOffsetsQ, devPtrSeqOffsetsKV,
      get_cudnn_fe_dtype(QKV_type), workspace->data.dptr, &workspace_size, stream, handle);

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
    size_t max_seqlen_kv, size_t head_dim, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    const Tensor *input_Q, const Tensor *input_KV, const Tensor *input_O, const Tensor *input_dO,
    const Tensor *input_Bias, Tensor *output_S, Tensor *output_dQ, Tensor *output_dKV,
    Tensor *output_dBias, const Tensor *cu_seqlens_q, const Tensor *cu_seqlens_kv,
    const Tensor *cu_seqlens_q_padded, const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
    Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_Q->data.dtype;
  void *devPtrQ = input_Q->data.dptr;
  void *devPtrKV = input_KV->data.dptr;
  NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(qkv_layout);
  size_t stride = 0;
  if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_2HD) {
    stride = typeToSize(QKV_type) * num_gqa_groups * head_dim;
  } else if (layout_group == NVTE_QKV_Layout_Group::NVTE_HD_H2D) {
    stride = typeToSize(QKV_type) * head_dim;
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
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
      attn_scale, p_dropout, qkv_layout, bias_type, mask_type, devPtrQ, devPtrK, devPtrV, devPtrO,
      devPtrSoftmaxStats, devPtrBias, devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdBias,
      devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlensQ, devPtrCuSeqlensKV,
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

void fused_attn_arbitrary_seqlen_fwd(size_t batch, size_t num_attn_heads, size_t num_gqa_groups,
                                     size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim,
                                     bool is_training, float attn_scale, float p_dropout,
                                     NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                                     NVTE_Mask_Type mask_type, const Tensor *input_Q,
                                     const Tensor *input_K, const Tensor *input_V,
                                     const Tensor *input_Bias, Tensor *output_O,
                                     NVTETensorPack *Aux_CTX_Tensors, const Tensor *cu_seqlens_q,
                                     const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
                                     const Tensor *cu_seqlens_kv_padded, const Tensor *rng_state,
                                     Tensor *workspace, cudaStream_t stream, cudnnHandle_t handle) {
  using namespace transformer_engine;

  const auto QKV_type = input_Q->data.dtype;
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

  if (Aux_CTX_Tensors->size == 0) {
    if ((bias_type != NVTE_NO_BIAS) && (bias_type != NVTE_ALIBI)) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
      Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
      output_bias->data.dptr = nullptr;
      output_bias->data.shape = {bias_b, bias_h, max_seqlen_q, max_seqlen_kv};
      output_bias->data.dtype = QKV_type;
    } else {
      Aux_CTX_Tensors->size = 2;
      Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
      output_S->data.dptr = nullptr;
      output_S->data.shape = {batch, num_attn_heads, max_seqlen_q, 1};
      output_S->data.dtype = DType::kFloat32;
      Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 2) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_S = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[0]);
    devPtrS = output_S->data.dptr;
    Tensor *output_rng_state = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[1]);
    output_rng_state->data.dptr = rng_state->data.dptr;
    Tensor *output_bias = reinterpret_cast<Tensor *>(Aux_CTX_Tensors->tensors[2]);
    output_bias->data.dptr = devPtrBias;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void *devPtrDropoutSeed = rng_state->data.dptr;
  void *devPtrDropoutOffset =
      reinterpret_cast<void *>(reinterpret_cast<uint64_t *>(rng_state->data.dptr) + 1);

  size_t workspace_size = 0;

  fused_attn_arbitrary_seqlen_fwd_impl(
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
      is_training, attn_scale, p_dropout, qkv_layout, bias_type, mask_type, devPtrQ, devPtrK,
      devPtrV, devPtrBias, devPtrS, devPtrO, devPtrDropoutSeed, devPtrDropoutOffset,
      devPtrCuSeqlensQ, devPtrCuSeqlensKV, devPtrSeqOffsetsQ, devPtrSeqOffsetsKV,
      get_cudnn_fe_dtype(QKV_type), workspace->data.dptr, &workspace_size, stream, handle);

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
    size_t max_seqlen_kv, size_t head_dim, float attn_scale, float p_dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type mask_type,
    const Tensor *input_Q, const Tensor *input_K, const Tensor *input_V, const Tensor *input_O,
    const Tensor *input_dO, const Tensor *input_Bias, Tensor *output_S, Tensor *output_dQ,
    Tensor *output_dK, Tensor *output_dV, Tensor *output_dBias, const Tensor *cu_seqlens_q,
    const Tensor *cu_seqlens_kv, const Tensor *cu_seqlens_q_padded,
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
      batch, num_attn_heads, num_gqa_groups, max_seqlen_q, max_seqlen_kv, head_dim, bias_b, bias_h,
      attn_scale, p_dropout, qkv_layout, bias_type, mask_type, devPtrQ, devPtrK, devPtrV, devPtrO,
      devPtrSoftmaxStats, devPtrBias, devPtrdQ, devPtrdK, devPtrdV, devPtrdO, devPtrdBias,
      devPtrDropoutSeed, devPtrDropoutOffset, devPtrCuSeqlensQ, devPtrCuSeqlensKV,
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
}  // namespace transformer_engine
#endif  // CUDNN_VERSION >= 8900
