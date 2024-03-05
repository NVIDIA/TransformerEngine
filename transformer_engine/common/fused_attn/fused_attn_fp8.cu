/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "transformer_engine/fused_attn.h"

#include "../common.h"
#include "utils.h"
#include "fused_attn_fp8.h"

namespace transformer_engine {
namespace fused_attn {

using namespace transformer_engine;

// fused attention BWD FP8
void fused_attn_fp8_bwd_impl_v1(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            float scaling_factor, float dropout_probability, NVTE_QKV_Layout layout,
            void* devPtrQ, void* devPtrK, void* devPtrV,
            void* devPtrM, void* devPtrZInv,
            void* devPtrO, void* devPtrdO,
            void* devPtrdQ, void* devPtrdK, void* devPtrdV,
            void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
            void* devPtrDescaleO, void* devPtrDescaledO,
            void* devPtrDescaleS, void* devPtrDescaledS,
            void* devPtrScaleS, void* devPtrScaledS,
            void* devPtrScaledQ, void* devPtrScaledK, void* devPtrScaledV,
            void* devPtrAmaxdS,
            void* devPtrAmaxdQ, void* devPtrAmaxdK, void* devPtrAmaxdV,
            void* devPtrcuSeqlensQ, void* devPtrcuSeqlensKV,
            void* devPtrDropoutSeed, void* devPtrDropoutOffset,
            cudnn_frontend::DataType_t tensorType,
            void* workspace,
            size_t* workspace_size,
            cudaStream_t stream,
            cudnnHandle_t handle) {
    using namespace transformer_engine;
    bool is_bias = false;
    bool is_causal = true;
    bool is_padding = false;
    bool is_training = true;
    bool is_dropout = (is_training && dropout_probability != 0.0f);
    auto bias_type = NVTE_Bias_Type::NVTE_NO_BIAS;
    auto mask_type = NVTE_Mask_Type::NVTE_CAUSAL_MASK;
    auto hg = h;

    try {
        FADescriptor_v1 descriptor{b,                   h,
                                   hg,                  s_q,
                                   s_kv,                d,
                                   scaling_factor,      true,
                                   dropout_probability, layout,
                                   bias_type,           mask_type,
                                   tensorType};

        namespace fe = cudnn_frontend;
        using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
              std::shared_ptr<fe::graph::Tensor_attributes>,  // q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // k
              std::shared_ptr<fe::graph::Tensor_attributes>,  // v
              std::shared_ptr<fe::graph::Tensor_attributes>,  // o
              std::shared_ptr<fe::graph::Tensor_attributes>,  // stats
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
              std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
              std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_k
              std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_v
              std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_o
              std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dO
              std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_s
              std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dS
              std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dQ
              std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dK
              std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dV
              std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_s
              std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dS
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dV
              std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dQ
              std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dK
              std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dV
              std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dS
              std::shared_ptr<fe::graph::Tensor_attributes>,  // bias
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dBias
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_kv
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dropout_seed
              std::shared_ptr<fe::graph::Tensor_attributes> >;  // dropout_offset

        using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
        static thread_local CacheType sdpa_fp8_bprop_cache;

        // Get plan from cache if cache is available, otherwise create one
        auto get_graph = [&](CacheType &cache, const FADescriptor_v1 &descriptor)
            -> graph_and_tensors {
            // if hit, return
            auto it = cache.find(descriptor);
            if (it != cache.end()) {
                auto graph = it->second;
                return graph;
            }

            // otherwise, build the op_graph and the plan. Then update cache
            auto mha_graph = std::make_shared<fe::graph::Graph>();

            auto data_type_forward_tensors = fe::DataType_t::FP8_E4M3; // according to the mix recipe
            auto data_type_backward_tensors = fe::DataType_t::FP8_E4M3; // should be e5m2 in TE, but devtech kernel is hardcoded for e4m3

            //mha_graph->set_io_data_type(tensorType)
            mha_graph->set_io_data_type(data_type_forward_tensors)
                    .set_intermediate_data_type(fe::DataType_t::FLOAT)
                    .set_compute_data_type(fe::DataType_t::FLOAT);

            std::shared_ptr<fe::graph::Tensor_attributes> q, k, v, o, dO, stats, attn_scale;
            std::shared_ptr<fe::graph::Tensor_attributes> descale_q, descale_k, descale_v;
            std::shared_ptr<fe::graph::Tensor_attributes> descale_s, descale_o, descale_dS, descale_dO;
            std::shared_ptr<fe::graph::Tensor_attributes> scale_s, scale_dS;
            std::shared_ptr<fe::graph::Tensor_attributes> scale_dQ, scale_dK, scale_dV;
            std::shared_ptr<fe::graph::Tensor_attributes> bias, dBias, seq_q, seq_kv;
            std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

            std::vector<int64_t> q_stride(4);
            std::vector<int64_t> k_stride(4);
            std::vector<int64_t> v_stride(4);
            std::vector<int64_t> o_stride(4);
            generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
            generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
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

            descale_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("Descale_q")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
            descale_k  = mha_graph->tensor_like(descale_q, "Descale_q");
            descale_v  = mha_graph->tensor_like(descale_q, "Descale_V");
            descale_s  = mha_graph->tensor_like(descale_q, "Descale_S");
            descale_o  = mha_graph->tensor_like(descale_q, "Descale_O");
            descale_dS = mha_graph->tensor_like(descale_q, "Descale_dS");
            descale_dO = mha_graph->tensor_like(descale_q, "Descale_dO");
            scale_s    = mha_graph->tensor_like(descale_q, "Scale_S");
            scale_dS   = mha_graph->tensor_like(descale_q, "Scale_dS");
            scale_dQ   = mha_graph->tensor_like(descale_q, "Scale_dQ");
            scale_dK   = mha_graph->tensor_like(descale_q, "Scale_dK");
            scale_dV   = mha_graph->tensor_like(descale_q, "Scale_dV");

            fe::graph::SDPA_fp8_backward_attributes sdpa_backward_options;
            sdpa_backward_options = fe::graph::SDPA_fp8_backward_attributes()
                            .set_name("sdpa_fp8_backward")
                            .set_causal_mask(is_causal)
                            .set_attn_scale(attn_scale);

            //sdpa_backward_options.set_alibi_mask(is_alibi);

            //if (is_bias) {
            //    bias = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("bias")
            //                    .set_dim({bias_b, bias_h, s_q, s_kv})
            //                    .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
            //    dBias = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("dBias")
            //                    .set_dim({bias_b, bias_h, s_q, s_kv})
            //                    .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
            //    sdpa_backward_options.set_bias(bias);
            //    // shapes [1, 1, s, s], [b, 1, s, s], [b, h, s, s]
            //    // are not supported for dbias calculation but they are
            //    // supported for forward bias calculation
            //    if ((bias_b == 1) && (bias_h == h)) {
            //      sdpa_backward_options.set_dbias(dBias);
            //    }
            //}

            //if (is_padding) {
            //    seq_q  = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("seq_q")
            //                    .set_dim({b, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT32));
            //    seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("seq_kv")
            //                    .set_dim({b, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT32));
            //    sdpa_backward_options.set_padding_mask(is_padding)
            //                    .set_seq_len_q(seq_q)
            //                    .set_seq_len_kv(seq_kv);
            //}

            //if (is_dropout) {
            //    dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("Seed")
            //                    .set_dim({1, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT64));
            //    dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("Offset")
            //                    .set_dim({1, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT64));
            //    sdpa_backward_options.set_dropout(
            //                    dropout_probability, dropout_seed, dropout_offset);
            //}

            auto [dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dS] = mha_graph->sdpa_fp8_backward(
                q, k, v, o, stats, dO,
                descale_q, descale_k, descale_v,
                descale_o, descale_dO, descale_s, descale_dS,
                scale_s, scale_dQ, scale_dK, scale_dV, scale_dS,
                sdpa_backward_options);

            dQ->set_output(true)
                    .set_dim({b, h, s_q, d})
                    .set_stride(q_stride);
            dK->set_output(true)
                    .set_dim({b, hg, s_kv, d})
                    .set_stride(k_stride);
            dV->set_output(true)
                    .set_dim({b, hg, s_kv, d})
                    .set_stride(v_stride);
            amax_dQ->set_output(true)
                    .set_dim({1, 1, 1, 1})
                    .set_data_type(fe::DataType_t::FLOAT);
            amax_dK->set_output(true)
                    .set_dim({1, 1, 1, 1})
                    .set_data_type(fe::DataType_t::FLOAT);
            amax_dV->set_output(true)
                    .set_dim({1, 1, 1, 1})
                    .set_data_type(fe::DataType_t::FLOAT);
            amax_dS->set_output(true)
                    .set_dim({1, 1, 1, 1})
                    .set_data_type(fe::DataType_t::FLOAT);

            dO->set_data_type(data_type_backward_tensors);
            dQ->set_data_type(data_type_backward_tensors);
            dK->set_data_type(data_type_backward_tensors);
            dV->set_data_type(data_type_backward_tensors);

            std::tuple<std::shared_ptr<fe::graph::Tensor_attributes>,  // q
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // k
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // v
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // o
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // stats
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // dO
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // attn_scale
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_q
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_k
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_v
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_o
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dO
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_s
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // descale_dS
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dQ
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dK
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dV
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_s
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // scale_dS
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // dQ
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // dK
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // dV
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dQ
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dK
                    std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_dV
                    std::shared_ptr<fe::graph::Tensor_attributes> >  // amax_dS
            key_tensors_tuple = std::make_tuple(
                q, k, v, o, stats, dO, attn_scale,
                descale_q, descale_k, descale_v,
                descale_o, descale_dO, descale_s, descale_dS,
                scale_s, scale_dQ, scale_dK, scale_dV, scale_dS,
                dQ, dK, dV,
                amax_dQ, amax_dK, amax_dV, amax_dS);
            auto bias_tuple = is_bias ?
                std::make_tuple(bias, dBias) : std::make_tuple(nullptr, nullptr);
            auto padding_tuple = is_padding ?
                std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
            auto dropout_tuple = is_dropout ?
                std::make_tuple(dropout_seed, dropout_offset) : std::make_tuple(nullptr, nullptr);

            NVTE_CHECK_CUDNN_FE(mha_graph->validate());
            NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
            NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

            auto return_tuple = std::tuple_cat(
                std::make_tuple(mha_graph), key_tensors_tuple,
                bias_tuple, padding_tuple, dropout_tuple);
            cache.insert({descriptor, return_tuple});

            return return_tuple;
        };

        auto [mha_graph, q, k, v, o, stats, dO, attn_scale,
            descale_q, descale_k, descale_v,
            descale_o, descale_dO, descale_s, descale_dS,
            scale_s, scale_dQ, scale_dK, scale_dV, scale_dS,
            dQ, dK, dV, amax_dQ, amax_dK, amax_dV, amax_dS,
            bias, dBias, seq_q, seq_kv, dropout_seed, dropout_offset] = get_graph(
            sdpa_fp8_bprop_cache, descriptor);

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
            {q, devPtrQ},
            {k, devPtrK},
            {v, devPtrV},
            {o, devPtrO},
            {stats, devPtrM},
            {dO, devPtrdO},
            {attn_scale, &scaling_factor},
            {descale_q, devPtrDescaleQ},
            {descale_k, devPtrDescaleK},
            {descale_v, devPtrDescaleV},
            {descale_o, devPtrDescaleO},
            {descale_dO, devPtrDescaledO},
            {descale_s, devPtrDescaleS},
            {descale_dS, devPtrDescaledS},
            {scale_s, devPtrScaleS},
            {scale_dQ, devPtrScaledQ},
            {scale_dK, devPtrScaledK},
            {scale_dV, devPtrScaledV},
            {scale_dS, devPtrScaledS},
            {dQ, devPtrdQ},
            {dK, devPtrdK},
            {dV, devPtrdV},
            {amax_dQ, devPtrAmaxdQ},
            {amax_dK, devPtrAmaxdK},
            {amax_dV, devPtrAmaxdV},
            {amax_dS, devPtrAmaxdS},
        };

        //if (is_bias) {
        //    variant_pack[bias] = devPtrBias;
        //    if ((bias_b == 1) && (bias_h == h)) {
        //      variant_pack[dBias] = devPtrdBias;
        //    } else {
        //      variant_pack[dBias] = nullptr;
        //    }
        //}

        //if (is_padding) {
        //    constexpr size_t nthreads_per_block = 128;
        //    const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
        //    void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
        //    void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
        //    cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
        //        b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
        //        static_cast<const int32_t *>(devPtrCuSeqlensKV),
        //        static_cast<int32_t *>(devActualSeqlenQ),
        //        static_cast<int32_t *>(devActualSeqlenKV));
        //    variant_pack[seq_q]  = devActualSeqlenQ;
        //    variant_pack[seq_kv] = devActualSeqlenKV;
        //}

        //if (is_dropout) {
        //    variant_pack[dropout_seed] = devPtrDropoutSeed;
        //    variant_pack[dropout_offset] = devPtrDropoutOffset;
        //}

        NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
    } catch (cudnn_frontend::cudnnException &e) {
        NVTE_ERROR(e.what());
    }

}

// fused attention FWD FP8
void fused_attn_fp8_fwd_impl_v1(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            bool is_training, float scaling_factor,
            float dropout_probability, NVTE_QKV_Layout layout,
            void* devPtrQ, void* devPtrK, void* devPtrV,
            void* devPtrM, void* devPtrZInv,
            void* devPtrO,
            void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
            void* devPtrDescaleS, void* devPtrScaleS, void* devPtrScaleO,
            void* devPtrAmaxO, void* devPtrAmaxS,
            void* devPtrcuSeqlensQ, void* devPtrcuSeqlensKV,
            void* devPtrDropoutSeed, void* devPtrDropoutOffset,
            cudnn_frontend::DataType_t tensorType,
            void* workspace,
            size_t* workspace_size,
            cudaStream_t stream,
            cudnnHandle_t handle) {
    using namespace transformer_engine;
    bool is_bias = false;
    //bool is_alibi = false;
    bool is_causal = true;
    bool is_padding = false;
    is_training = true; //false;
    bool is_dropout = (is_training && dropout_probability != 0.0f);
    auto bias_type = NVTE_Bias_Type::NVTE_NO_BIAS;
    auto mask_type = NVTE_Mask_Type::NVTE_CAUSAL_MASK;
    auto hg = h;

    try {
        FADescriptor_v1 descriptor{b,                   h,
                                   hg,                  s_q,
                                   s_kv,                d,
                                   scaling_factor,      is_training,
                                   dropout_probability, layout,
                                   bias_type,           mask_type,
                                   tensorType};

        namespace fe = cudnn_frontend;
        using graph_and_tensors = std::tuple<std::shared_ptr<fe::graph::Graph>,
              std::shared_ptr<fe::graph::Tensor_attributes>,  // Q
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
              std::shared_ptr<fe::graph::Tensor_attributes>,  // amax_o
              std::shared_ptr<fe::graph::Tensor_attributes>,  // Stats
              std::shared_ptr<fe::graph::Tensor_attributes>,  // bias
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_q
              std::shared_ptr<fe::graph::Tensor_attributes>,  // seq_kv
              std::shared_ptr<fe::graph::Tensor_attributes>,  // dropout_seed
              std::shared_ptr<fe::graph::Tensor_attributes> >;  // dropout_offset

        using CacheType = std::map<FADescriptor_v1, graph_and_tensors>;
        static thread_local CacheType sdpa_fp8_fprop_cache;

        // Get plan from cache if cache is available, otherwise create one
        auto get_graph = [&](CacheType &cache, const FADescriptor_v1 &descriptor)
            -> graph_and_tensors {
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
            std::shared_ptr<fe::graph::Tensor_attributes> descale_q, descale_k, descale_v;
            std::shared_ptr<fe::graph::Tensor_attributes> descale_s, scale_s, scale_o;
            std::shared_ptr<fe::graph::Tensor_attributes> bias, seq_q, seq_kv;
            std::shared_ptr<fe::graph::Tensor_attributes> dropout_seed, dropout_offset;

            std::vector<int64_t> q_stride(4);
            std::vector<int64_t> k_stride(4);
            std::vector<int64_t> v_stride(4);
            generateMatrixStrides(b, h, s_q, s_kv, d, q_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_Q_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, k_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_K_Matrix);
            generateMatrixStrides(b, hg, s_q, s_kv, d, v_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_V_Matrix);
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

            attn_scale = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("attn_scale")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_is_pass_by_value(true)
                            .set_data_type(fe::DataType_t::FLOAT));

            descale_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("Descale_q")
                            .set_dim({1, 1, 1, 1})
                            .set_stride({1, 1, 1, 1})
                            .set_data_type(fe::DataType_t::FLOAT));
            descale_k = mha_graph->tensor_like(descale_q, "Descale_q");
            descale_v = mha_graph->tensor_like(descale_q, "Descale_V");
            descale_s = mha_graph->tensor_like(descale_q, "Descale_S");
            scale_s   = mha_graph->tensor_like(descale_q, "Scale_S");
            scale_o   = mha_graph->tensor_like(descale_q, "Scale_O");

            fe::graph::SDPA_fp8_attributes sdpa_options;
            sdpa_options = fe::graph::SDPA_fp8_attributes()
                            .set_name("sdpa_fp8")
                            .set_is_inference(!is_training)
                            .set_causal_mask(is_causal)
                            .set_attn_scale(attn_scale);

            //sdpa_options.set_alibi_mask(is_alibi);
            //if (is_bias) {
            //    bias = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("bias")
            //                    .set_dim({bias_b, bias_h, s_q, s_kv})
            //                    .set_stride({bias_h * s_q * s_kv, s_q * s_kv, s_kv, 1}));
            //    sdpa_options.set_bias(bias);
            //}

            //if (is_padding) {
            //    seq_q  = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("seq_q")
            //                    .set_dim({b, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT32));
            //    seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("seq_kv")
            //                    .set_dim({b, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT32));
            //    sdpa_options.set_padding_mask(is_padding)
            //                    .set_seq_len_q(seq_q)
            //                    .set_seq_len_kv(seq_kv);
            //}

            //if (is_dropout) {
            //    dropout_seed = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("Seed")
            //                    .set_dim({1, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT64));
            //    dropout_offset = mha_graph->tensor(fe::graph::Tensor_attributes()
            //                    .set_name("Offset")
            //                    .set_dim({1, 1, 1, 1})
            //                    .set_stride({1, 1, 1, 1})
            //                    .set_data_type(fe::DataType_t::INT64));
            //    sdpa_options.set_dropout(
            //                    dropout_probability, dropout_seed, dropout_offset);
            //}

            auto [O, Stats, amax_s, amax_o] = mha_graph->sdpa_fp8(
                Q, K, V, descale_q, descale_k, descale_v, descale_s,
                scale_s, scale_o, sdpa_options);

            std::vector<int64_t> o_stride(4);
            generateMatrixStrides(b, h, s_q, s_kv, d, o_stride.data(),
                    layout, NVTE_QKV_Matrix::NVTE_O_Matrix);
            O->set_output(true).set_dim({b, h, s_q, d}).set_stride(o_stride);
            amax_o->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);
            amax_s->set_output(true).set_dim({1, 1, 1, 1}).set_data_type(fe::DataType_t::FLOAT);

            if (is_training) {
                Stats->set_output(true).set_data_type(fe::DataType_t::FLOAT)//;
                        .set_dim({b, h, s_q, 1})
                        .set_stride({h * s_q, s_q, 1, 1});
            }

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
                    std::shared_ptr<fe::graph::Tensor_attributes> >  // amax_o
            key_tensors_tuple = std::make_tuple(Q, K, V, descale_q, descale_k, descale_v,
                descale_s, scale_s, scale_o, attn_scale, O, amax_s, amax_o);
            auto Stats_tuple = is_training ? std::make_tuple(Stats) : std::make_tuple(nullptr);
            auto bias_tuple = is_bias ? std::make_tuple(bias) : std::make_tuple(nullptr);
            auto padding_tuple = is_padding ?
                std::make_tuple(seq_q, seq_kv) : std::make_tuple(nullptr, nullptr);
            auto dropout_tuple = is_dropout ?
                std::make_tuple(dropout_seed, dropout_offset) : std::make_tuple(nullptr, nullptr);

            NVTE_CHECK_CUDNN_FE(mha_graph->validate());
            NVTE_CHECK_CUDNN_FE(mha_graph->build_operation_graph(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->create_execution_plans({fe::HeurMode_t::A}));
            NVTE_CHECK_CUDNN_FE(mha_graph->check_support(handle));
            NVTE_CHECK_CUDNN_FE(mha_graph->build_plans(handle));

            auto return_tuple = std::tuple_cat(
                std::make_tuple(mha_graph), key_tensors_tuple,
                Stats_tuple, bias_tuple, padding_tuple, dropout_tuple);
            cache.insert({descriptor, return_tuple});

            return return_tuple;
        };

        auto [mha_graph, Q, K, V, descale_q, descale_k, descale_v, descale_s,
            scale_s, scale_o, attn_scale, O, amax_s, amax_o, Stats,
            bias, seq_q, seq_kv, dropout_seed, dropout_offset] = get_graph(
                sdpa_fp8_fprop_cache, descriptor);

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
            {descale_s, devPtrDescaleS},
            {scale_s, devPtrScaleS},
            {scale_o, devPtrScaleO},
            {attn_scale, &scaling_factor},
            {O, devPtrO},
            {amax_s, devPtrAmaxS},
            {amax_o, devPtrAmaxO}};

        if (is_training) {
            variant_pack[Stats] = devPtrM; //nullptr; // temporary
        }

        //if (is_bias) {
        //    variant_pack[bias] = devPtrBias;
        //}

        //if (is_padding) {
        //    constexpr size_t nthreads_per_block = 128;
        //    const size_t grid = (b + nthreads_per_block - 1) / nthreads_per_block;
        //    void *devActualSeqlenQ = static_cast<int8_t *>(workspace) + plan_workspace_size;
        //    void *devActualSeqlenKV = static_cast<int8_t *>(devActualSeqlenQ) + b * sizeof(int32_t);
        //    cu_seqlens_to_actual_seqlens<<<grid, nthreads_per_block, 0, stream>>>(
        //        b, static_cast<const int32_t *>(devPtrCuSeqlensQ),
        //        static_cast<const int32_t *>(devPtrCuSeqlensKV),
        //        static_cast<int32_t *>(devActualSeqlenQ),
        //        static_cast<int32_t *>(devActualSeqlenKV));
        //    variant_pack[seq_q]  = devActualSeqlenQ;
        //    variant_pack[seq_kv] = devActualSeqlenKV;
        //}

        //if (is_dropout) {
        //    variant_pack[dropout_seed] = devPtrDropoutSeed;
        //    variant_pack[dropout_offset] = devPtrDropoutOffset;
        //}

        NVTE_CHECK_CUDNN_FE(mha_graph->execute(handle, variant_pack, workspace));
    } catch (cudnn_frontend::cudnnException &e) {
        NVTE_ERROR(e.what());
    }
}

// fused attention FWD FP8
void fused_attn_fp8_fwd_impl(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            bool isTraining, float attnScale,
            float dropoutProbability, NVTE_QKV_Layout layout,
            void* devPtrQ, void* devPtrK, void* devPtrV,
            void* devPtrM, void* devPtrZInv,
            void* devPtrO,
            void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
            void* devPtrDescaleS, void* devPtrScaleS, void* devPtrScaleO,
            void* devPtrAmaxO, void* devPtrAmaxS,
            void* devPtrcuSeqlensQ, void* devPtrcuSeqlensKV,
            void* devPtrDropoutSeed, void* devPtrDropoutOffset,
            cudnnDataType_t tensorType,
            void* workspace_ptr,
            size_t* workspace_size,
            cudaStream_t stream,
            cudnnHandle_t handle_) {
//  try {
//      FADescriptor descriptor{
//              b, h, s_q, s_kv, d,
//              attnScale, isTraining, dropoutProbability, layout,
//              NVTE_Bias_Type::NVTE_NO_BIAS, NVTE_Mask_Type::NVTE_PADDING_MASK, tensorType, false};
//
//      using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
//      static thread_local CacheType fa_fprop_cache;
//
//      // Get plan from cache if cache is available, otherwise create one
//      auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
//          // If hit, return
//          auto it = cache.find(descriptor);
//          if (it != cache.end()) {
//            auto plan = it->second;
//            return plan;
//          }
//
//          // Otherwise, build the op_graph and the plan. Then update cache
//          std::vector<cudnn_frontend::Operation const*> all_ops;
//          std::vector<cudnn_frontend::Operation> ops;
//
////          cudnn_frontend::throw_if(dropoutProbability != 0.0f && !isTraining,
////                          "Dropout probability should be 0.0f for inference mode",
////                          CUDNN_STATUS_BAD_PARAM);
////          cudnn_frontend::throw_if(dropoutProbability == 1.0f,
////                          "Dropout probability cannot be 1.0",
////                          CUDNN_STATUS_BAD_PARAM);
//
//          int64_t raggedDim[4] =  {b + 1, 1, 1, 1};
//          int64_t raggedStride[4] = {1, 1, 1, 1};
//          // Create offset tensors
//          auto QKVOffsetTensor = tensor_create(
//                          CUDNN_DATA_INT32, tensor_name_to_uid["QKV_RAGGED"],
//                          raggedDim, raggedStride, false, false);
//          auto ORaggedOffsetTensor = tensor_create(
//                          CUDNN_DATA_INT32, tensor_name_to_uid["O_RAGGED"],
//                          raggedDim, raggedStride, false, false);
//
//          int64_t seqlen_dim[4] =  {b, 1, 1, 1};
//          int64_t seqlen_stride[4] = {1, 1, 1, 1};
//          // Create override tensors
//          auto seqlenMNKTensor = tensor_create(
//                          CUDNN_DATA_INT32, tensor_name_to_uid["MNK_OVERRIDE"],
//                          seqlen_dim, seqlen_stride, false, false);
//
//          // Create shared ptrs to ragged offset tensors
//          // for multiple tensors to use ragged offset
//          std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensorPtr =
//                  std::make_shared<cudnn_frontend::Tensor>(std::move(QKVOffsetTensor));
//          std::shared_ptr<cudnn_frontend::Tensor> ORaggedOffsetTensorPtr =
//                  std::make_shared<cudnn_frontend::Tensor>(std::move(ORaggedOffsetTensor));
//
//          // Create Q and K tensors that are used in different places
//          int64_t q_dim[4] = {b, h, s_q, d};
//          int64_t q_stride[4];
//          generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout,
//                          NVTE_QKV_Matrix::NVTE_Q_Matrix);
//
//          int64_t k_dim[4] =  {b, h, s_kv, d};
//          int64_t k_stride[4];
//          generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout,
//                          NVTE_QKV_Matrix::NVTE_K_Matrix);
//
//          auto qTensor = tensor_create_with_offset(
//                          tensorType, tensor_name_to_uid["Q"],
//                          q_dim, q_stride, false, false,
//                          QKVRaggedOffsetTensorPtr);
//          auto kTensor = tensor_create_with_offset(
//                          tensorType, tensor_name_to_uid["K"],
//                          k_dim, k_stride, false, false,
//                          QKVRaggedOffsetTensorPtr);
//
//          // Q * K.T
//          auto afterQKTensor = createQKBMM(
//                          b, h, s_q, s_kv, d, layout, tensorType,
//                          &ops, qTensor, kTensor,
//                          seqlenMNKTensor, QKVRaggedOffsetTensorPtr);
//
//          // QK.T * attn scale
//          auto AfterAttnScale_before_dequan_Q_tensor = createScale(
//                          afterQKTensor,  // input tensor
//                          "AttnScale",  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          true,  // scale is by value
//                          &ops);
//
//          // QK.T * attn scale * dequant_Q
//          auto AfterAttnScale_before_dequan_K_tensor = createScale(
//                          AfterAttnScale_before_dequan_Q_tensor,  // input tensor
//                          "descaleQ",  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // QK.T * attn scale * dequant_Q * dequant_K
//          auto AfterAttnScale_tensor = createScale(
//                          AfterAttnScale_before_dequan_K_tensor,  // input tensor
//                          "descaleK",  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          auto BeforeDropoutTensor = createSoftmaxForward(
//                          b, h, s_q, s_kv, &ops,
//                          AfterAttnScale_tensor, isTraining);
//
//          auto AfterDropout_before_quan_S = createDropoutForward(
//                          b, h, s_q, s_kv, dropoutProbability,
//                          &ops, BeforeDropoutTensor);
//
//          // Amax for S
//          createAmax("amaxS", BeforeDropoutTensor, &ops);
//
//          // After softmax * dropout * scale S -> fp8 input to next bmm with V
//          auto AfterMultiplyDropout = createScale(
//                          AfterDropout_before_quan_S,  // input tensor
//                          "scaleS",  // scale tensor
//                          tensorType,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // After softmax * Dropout * V
//          auto OTensor_before_dequan_S_tensor = createSVBMM(
//                          b, h, s_q, s_kv, d, layout, tensorType,
//                          &ops, AfterMultiplyDropout,
//                          seqlenMNKTensor, QKVRaggedOffsetTensorPtr);
//
//          // O * dequant_S
//          auto OTensor_before_dequan_V_tensor = createScale(
//                          OTensor_before_dequan_S_tensor,  // input tensor
//                          "descaleS",  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // O * dequant_S * dequant_V
//          auto OTensor_before_quan_O_tensor = createScale(
//                          OTensor_before_dequan_V_tensor,  // input tensor
//                          "descaleV",  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // O * dequant_S * dequant_V * scale O
//          auto OTensor = createScaleWithOffset(
//                          OTensor_before_quan_O_tensor,  // input tensor
//                          "scaleO",  // scale tensor
//                          layout,  // qkv layout
//                          tensorType,  // output tensor type
//                          false,  // output not virtual
//                          false,  // scale is by value
//                          &ops,
//                          ORaggedOffsetTensorPtr,  // ragged offset
//                          "O");
//
//          // Amax for O
//          createAmax("amaxO", OTensor_before_quan_O_tensor, &ops);
//
//          for (unsigned int i = 0; i < ops.size(); i++) {
//              all_ops.push_back(&ops[i]);
//          }
//
//          // Create an Operation Graph
//          auto opGraph = cudnn_frontend::OperationGraphBuilder()
//                             .setHandle(handle_)
//                             .setOperationGraph(all_ops.size(), all_ops.data())
//                             .build();
//
//          cudnn_frontend::EngineConfigList filtered_configs;
//          auto statuses = cudnn_frontend::get_heuristics_list<1>(
//                          {"heuristics_instant"}, opGraph,
//                          allowAllConfig, filtered_configs, true);
//
//          if (filtered_configs.size() == 0) {
//              cudnn_frontend::set_error_and_throw_exception(
//                      nullptr,
//                      CUDNN_STATUS_NOT_SUPPORTED,
//                      "run_mha_fprop: No config returned by the heuristics");
//          }
//
//          auto plan = cudnn_frontend::ExecutionPlanBuilder()
//                  .setHandle(handle_)
//                  .setEngineConfig(filtered_configs[0], opGraph.getTag())
//                  .build();
//          cache.insert({descriptor, plan});
//          return plan;
//      };  // end of get_plan
//
//      auto plan = get_plan(fa_fprop_cache, descriptor);
//      size_t wkspace_size = static_cast<size_t>(plan.getWorkspaceSize());
//
//      // Exit to request upper level API to allocate memory if needed
//      if (workspace_ptr == nullptr) {
//          *workspace_size = wkspace_size + ((b + 1) * 2 + b) * sizeof(int32_t);
//          return;
//      }
//
//      // cuDNN stream check needs to be moved here to support dummy kernel calls with
//      // null streams for sizing the cuDNN workspace.
//      NVTE_CHECK_CUDNN(cudnnSetStream(handle_, stream));
//
//      int32_t* qkv_ragged_offset = reinterpret_cast<int32_t*>(
//                  reinterpret_cast<int8_t*>(workspace_ptr) + wkspace_size);
//      int32_t* o_ragged_offset = reinterpret_cast<int32_t*>(
//                  reinterpret_cast<int8_t*>(workspace_ptr)
//                  + wkspace_size + (b + 1) * sizeof(int32_t));
//      int32_t* actual_seqlens_q = reinterpret_cast<int32_t*>(
//                  reinterpret_cast<int8_t*>(workspace_ptr)
//                  + wkspace_size + (b + 1) * 2 * sizeof(int32_t));
//      // FP8 currently only supports self-attention, so doesn't use devPtrcuSeqlensKV
//      dim3 blockDims(128);
//      dim3 gridDims((b + blockDims.x)/blockDims.x);
//      cu_seqlens_to_offsets<<<gridDims, blockDims, 0, stream>>>(
//                      b, h, d, reinterpret_cast<int32_t*>(devPtrcuSeqlensQ),
//                      actual_seqlens_q, qkv_ragged_offset, o_ragged_offset);
//      void* devPtrQKVRaggedOffset = reinterpret_cast<void *>(qkv_ragged_offset);
//      void* devPtrORaggedOffset = reinterpret_cast<void *>(o_ragged_offset);
//      void* devPtrMNKOverride = reinterpret_cast<void *>(actual_seqlens_q);
//
//      float dropoutScale = 1.0f/(1.0f - dropoutProbability);
//
//      std::set<std::pair<uint64_t, void*>> data_ptrs;
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q"], devPtrQ));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K"], devPtrK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K_TRANSPOSE"], devPtrK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V"], devPtrV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["AttnScale"], &attnScale));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["DROPOUT_SCALE"], &dropoutScale));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["DROPOUT_SEED"], devPtrDropoutSeed));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["DROPOUT_OFFSET"], devPtrDropoutOffset));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["O"], devPtrO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleQ"], devPtrDescaleQ));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleK"], devPtrDescaleK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleV"], devPtrDescaleV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleS"], devPtrDescaleS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["scaleS"], devPtrScaleS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["scaleO"], devPtrScaleO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["amaxO"], devPtrAmaxO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["amaxS"], devPtrAmaxS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["QKV_RAGGED"], devPtrQKVRaggedOffset));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["O_RAGGED"], devPtrORaggedOffset));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["MNK_OVERRIDE"], devPtrMNKOverride));
//
//      // If training, then we need to write out M and Z_INV
//      if (isTraining) {
//          data_ptrs.emplace(std::pair<uint64_t, void*>(
//                                  tensor_name_to_uid["M"], devPtrM));
//          data_ptrs.emplace(std::pair<uint64_t, void*>(
//                                  tensor_name_to_uid["Z_INV"], devPtrZInv));
//      }
//
//      auto variantPack  = cudnn_frontend::VariantPackBuilder()
//                             .setWorkspacePointer(workspace_ptr)
//                             .setDataPointers(data_ptrs)
//                             .build();
//      cudnnStatus_t status = cudnnBackendExecute(
//                      handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
//
////      cudnn_frontend::throw_if(
////                      [status]() { return (status != CUDNN_STATUS_SUCCESS); },
////                      "Plan execute error", status);
//  } catch (cudnn_frontend::cudnnException& e) {
//      struct cudaDeviceProp prop;
//      NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
//
//      // This example is only for GH100 cards (cudnn Version >= 8900)
//      if (!((prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8900))
//                      && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH
//                              || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
//          std::cout << "Example is only supported for GH100 (cuDNN >= 8900) GPUs" << std::endl;
//      }  else {
//          std::cout << "[ERROR] Exception " << e.what() << std::endl;
//      }
//  }
}

// fused attention BWD FP8
void fused_attn_fp8_bwd_impl(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d,
            float attnScale, float dropoutProbability, NVTE_QKV_Layout layout,
            void* devPtrQ, void* devPtrK, void* devPtrV,
            void* devPtrM, void* devPtrZInv,
            void* devPtrO, void* devPtrdO,
            void* devPtrdQ, void* devPtrdK, void* devPtrdV,
            void* devPtrDescaleQ, void* devPtrDescaleK, void* devPtrDescaleV,
            void* devPtrDescaleO, void* devPtrDescaledO,
            void* devPtrDescaleS, void* devPtrDescaledS,
            void* devPtrScaleS, void* devPtrScaledS,
            void* devPtrScaledQ, void* devPtrScaledK, void* devPtrScaledV,
            void* devPtrAmaxdS,
            void* devPtrAmaxdQ, void* devPtrAmaxdK, void* devPtrAmaxdV,
            void* devPtrcuSeqlensQ, void* devPtrcuSeqlensKV,
            void* devPtrDropoutSeed, void* devPtrDropoutOffset,
            cudnnDataType_t tensorType,
            void* workspace_ptr,
            size_t* workspace_size,
            cudaStream_t stream,
            cudnnHandle_t handle_) {
//  try {
//      FADescriptor descriptor{
//              b, h, s_q, s_kv, d,
//              attnScale, false, dropoutProbability, layout,
//              NVTE_Bias_Type::NVTE_NO_BIAS, NVTE_Mask_Type::NVTE_PADDING_MASK, tensorType, false};
//
//      using CacheType = std::map<FADescriptor, cudnn_frontend::ExecutionPlan>;
//      static thread_local CacheType fa_bprop_cache;
//
//      // Get plan from cache if cache is available, otherwise create one
//      auto get_plan = [&](CacheType &cache, const FADescriptor &descriptor) {
//          // If hit, return
//          auto it = cache.find(descriptor);
//          if (it != cache.end()) {
//            auto plan = it->second;
//            return plan;
//          }
//
//          // Otherwise, build the op_graph and the plan. Then update cache
//          std::vector<cudnn_frontend::Operation const*> all_ops;
//          std::vector<cudnn_frontend::Operation> ops;
//
////          cudnn_frontend::throw_if(dropoutProbability == 1.0f,
////                          "Dropout probability cannot be 1.0",
////                          CUDNN_STATUS_BAD_PARAM);
//
//          int64_t raggedDim[4] =  {b + 1, 1, 1, 1};
//          int64_t raggedStride[4] = {1, 1, 1, 1};
//          // Create offset tensors
//          auto QKVOffsetTensor = tensor_create(
//                          CUDNN_DATA_INT32, tensor_name_to_uid["QKV_RAGGED"],
//                          raggedDim, raggedStride, false, false);
//          auto ORaggedOffsetTensor = tensor_create(
//                          CUDNN_DATA_INT32, tensor_name_to_uid["O_RAGGED"],
//                          raggedDim, raggedStride, false, false);
//
//          // Create shared ptrs to ragged offset tensors for multiple tensors
//          std::shared_ptr<cudnn_frontend::Tensor> QKVRaggedOffsetTensorPtr =
//                  std::make_shared<cudnn_frontend::Tensor>(std::move(QKVOffsetTensor));
//          std::shared_ptr<cudnn_frontend::Tensor> ORaggedOffsetTensorPtr =
//                  std::make_shared<cudnn_frontend::Tensor>(std::move(ORaggedOffsetTensor));
//
//          // Create Q and K tensors that are used in different places
//          int64_t q_dim[4] = {b, h, s_q, d};
//          int64_t q_stride[4];
//          generateMatrixStrides(b, h, s_q, s_kv, d, q_stride, layout,
//                          NVTE_QKV_Matrix::NVTE_Q_Matrix);
//
//          int64_t k_dim[4] =  {b, h, s_kv, d};
//          int64_t k_stride[4];
//          generateMatrixStrides(b, h, s_q, s_kv, d, k_stride, layout,
//                          NVTE_QKV_Matrix::NVTE_K_Matrix);
//
//          auto qTensor = tensor_create_with_offset(
//                          tensorType, tensor_name_to_uid["Q"],
//                          q_dim, q_stride, false, false, QKVRaggedOffsetTensorPtr);
//          auto kTensor = tensor_create_with_offset(
//                          tensorType, tensor_name_to_uid["K"],
//                          k_dim, k_stride, false, false, QKVRaggedOffsetTensorPtr);
//
//          int64_t scale_dim[4] = {1, 1, 1, 1};
//          int64_t scale_stride[4] = {1, 1, 1, 1};
//
//          // Create attnScale tensor for multiple ops to use
//          auto attnScaleTensor = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["AttnScale"],
//                          scale_dim, scale_stride, false, true);  // is by value
//
//          // Create descale Q K dO dS global tensors since they are used in multiple places
//          auto descaleQTensor = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaleQ"],
//                          scale_dim, scale_stride, false, false);
//          auto descaleKTensor = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaleK"],
//                          scale_dim, scale_stride, false, false);
//          auto descaledOTensor = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaledO"],
//                          scale_dim, scale_stride, false, false);
//          auto descaledSTensor = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["descaledS"],
//                          scale_dim, scale_stride, false, false);
//
//          int64_t seqlen_dim[4] =  {b, 1, 1, 1};
//          int64_t seqlen_stride[4] = {1, 1, 1, 1};
//          // Create MNK override tensor
//          auto seqlenMNKTensor = tensor_create(
//                          CUDNN_DATA_INT32, tensor_name_to_uid["MNK_OVERRIDE"],
//                          seqlen_dim, seqlen_stride, false, false);
//
//          int64_t O_dim[4] =  {b, h, s_q, d};
//          int64_t O_stride[4];
//          generateMatrixStrides(b, h, s_q, s_kv, d, O_stride, layout,
//                          NVTE_QKV_Matrix::NVTE_O_Matrix);
//          // Create O and loss tensor
//          auto OTensor = tensor_create_with_offset(
//                          tensorType, tensor_name_to_uid["O"],
//                          O_dim, O_stride, false, false, ORaggedOffsetTensorPtr);
//          // dO is used in multiple places and E5M2
//          auto dOTensor = tensor_create_with_offset(
//                          CUDNN_DATA_FP8_E5M2, tensor_name_to_uid["dO"],
//                          O_dim, O_stride, false, false, ORaggedOffsetTensorPtr);
//
//          // Q * K.T
//          auto afterQKTensor = createQKBMM(
//                          b, h, s_q, s_kv, d, layout, tensorType,
//                          &ops, qTensor, kTensor,
//                          seqlenMNKTensor, QKVRaggedOffsetTensorPtr);
//
//          // QK.T * attn scale
//          auto AfterAttnScale_before_dequan_Q_tensor = createScale(
//                          afterQKTensor,  // input tensor
//                          attnScaleTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          true,  // scale is by value
//                          &ops,
//                          1999  /*UID offset*/);
//
//          // QK.T * attn scale * dequant_Q
//          auto AfterAttnScale_before_dequan_K_tensor = createScale(
//                          AfterAttnScale_before_dequan_Q_tensor,  // input tensor
//                          descaleQTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2000  /*UID offset*/);
//
//          // QK.T * attn scale * dequant_Q * dequant_K
//          auto AfterAttnScale_tensor = createScale(
//                          AfterAttnScale_before_dequan_K_tensor,  // input tensor
//                          descaleKTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2001  /*UID offset*/);
//
//          auto beforeDropout_QKt_Tensor = createSoftmaxBackward(
//                          b, h, s_q, s_kv, &ops, AfterAttnScale_tensor);
//
//          int64_t afterBMM1_dim[4] = {b, h, s_q, s_kv};
//          int64_t afterBMM1_stride[4] = {h * s_q * s_kv, s_q * s_kv, s_kv, 1};
//
//          // mask for the dropout. Used in different places
//          auto dropoutMaskTensor = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 200,
//                          afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual
//
//          auto AfterDropout_before_quan_S = createDropoutBackward(
//                          b, h, s_q, s_kv, dropoutProbability,
//                          &ops, beforeDropout_QKt_Tensor, dropoutMaskTensor);
//
//          // After softmax * scale S -> fp8 input to next bmm with V
//          auto AfterMultiply = createScale(
//                          AfterDropout_before_quan_S,  // input tensor
//                          "scaleS",  // scale tensor
//                          tensorType,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // After softmax * dO
//          auto dVTensor_before_dequan_S = createSdOBMM(
//                          b, h, s_q, s_kv, d, tensorType,
//                          &ops, AfterMultiply, dOTensor, seqlenMNKTensor);
//
//          // O * dequant_S
//          auto dVTensor_before_dequan_dO = createScale(
//                          dVTensor_before_dequan_S,  // input tensor
//                          "descaleS",  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // O * dequant_S * dequant_dO
//          auto dVTensor_before_quan_dV = createScale(
//                          dVTensor_before_dequan_dO,  // input tensor
//                          descaledOTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2002  /*UID offset*/);
//
//          // O * dequant_S * dequant_dO * scale dV
//          auto dVTensor = createScaleWithOffset(
//                          dVTensor_before_quan_dV,  // input tensor
//                          "scaledV",  // scale tensor
//                          layout,  // qkv layout
//                          CUDNN_DATA_FP8_E5M2,  // output tensor type
//                          false,  // output not virtual
//                          false,  // scale is by value
//                          &ops,
//                          QKVRaggedOffsetTensorPtr,  // ragged offset
//                          "dV"  /*Output tensor name*/);
//
//          // Amax for dV
//          createAmax("amaxdV", dVTensor_before_quan_dV, &ops);
//
//          auto dS_before_dequan_dO_Tensor = createdOVBMM(
//                          b, h, s_q, s_kv, d, layout, tensorType,
//                          &ops, dOTensor, seqlenMNKTensor, QKVRaggedOffsetTensorPtr);
//
//          // dS * dequant_dO
//          auto dS_before_dequan_V = createScale(
//                          dS_before_dequan_dO_Tensor,  // input tensor
//                          descaledOTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2003  /*UID offset*/);
//
//          // O * dequant_S * dequant_dV
//          auto dS_after_dequan = createScale(
//                          dS_before_dequan_V,  // input tensor
//                          "descaleV",  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // RNG Multiply
//          auto beforeDropoutScale_dOVt_Tensor = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 350,
//                          afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual
//          // After dropout mask and scale
//          auto dS_after_dropout = tensor_create(
//                          CUDNN_DATA_FLOAT, tensor_name_to_uid["VIRTUAL"] + 351,
//                          afterBMM1_dim, afterBMM1_stride, true, false);  // is virtual
//
//          // Define the multiply mask descriptor
//          auto mulDesc = pw_desc_create(CUDNN_DATA_FLOAT, CUDNN_POINTWISE_MUL);
//
//          // Create a multiply mask Node
//          auto maskMul_op = binary_pw_op_create(
//                          dS_after_dequan, dropoutMaskTensor,
//                          beforeDropoutScale_dOVt_Tensor, mulDesc);
//
//          ops.push_back(std::move(maskMul_op));
//
//          // scale after dropout for dO and O chain
//          auto dropoutScale_dOVt_OdO_Tensor = tensor_create(
//                          tensorType, tensor_name_to_uid["DROPOUT_SCALE_dOVt_OdO"],
//                          scale_dim, scale_stride, false, true);  // is by value
//
//          // Create a multiply dropout scale Node
//          auto mul_dropout_scale_op = binary_pw_op_create(
//                          beforeDropoutScale_dOVt_Tensor,
//                          dropoutScale_dOVt_OdO_Tensor,
//                          dS_after_dropout, mulDesc);
//
//          ops.push_back(std::move(mul_dropout_scale_op));
//
//          // O * dequant_O
//          auto O_after_dequan_Tensor = createScale(OTensor,  // input tensor
//                                          "descaleO",  // scale tensor
//                                          CUDNN_DATA_FLOAT,  // output tensor type
//                                          true,  // output is virtual
//                                          false,  // scale is by value
//                                          &ops);
//
//          // dO * dequant_dO
//          auto dO_after_dequan_Tensor = createScale(dOTensor,  // input tensor
//                                          descaledOTensor,  // scale tensor
//                                          CUDNN_DATA_FLOAT,  // output tensor type
//                                          true,  // output is virtual
//                                          false,  // scale is by value
//                                          &ops,
//                                          2004  /*UID offset*/);
//
//          // row reduction sum[(dO * dequant_dO) * (O * dequant_O) * (1 - p)]
//          auto O_dO_after_rowsum = createdOAndORowReductionChain(
//                          b, h, s_q, s_kv, d, layout,
//                          &ops, O_after_dequan_Tensor,
//                          dO_after_dequan_Tensor, dropoutScale_dOVt_OdO_Tensor);
//
//          // (dS_after_dropout - O_dO_after_rowsum) * AfterDropout_before_quan_S * attnScale
//          auto S_mul_dS_minus_O_dO = createBiasSubtractionSoftmaxMulChain(
//              b, h, s_q, s_kv, d, layout,
//              &ops, dS_after_dropout,
//              AfterDropout_before_quan_S, O_dO_after_rowsum,
//              attnScaleTensor);
//
//
//          // S_mul_dS_minus_O_dO * scaledS
//          auto S_mul_dS_minus_O_dO_after_quan_dS = createScale(
//                          S_mul_dS_minus_O_dO,  // input tensor
//                          "scaledS",  // scale tensor
//                          CUDNN_DATA_FP8_E5M2,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops);
//
//          // Amax for dS
//          createAmax("amaxdS", S_mul_dS_minus_O_dO, &ops);
//
//          // dS @ K
//          auto After_dS_K = createdSKBMM(
//                          b, h, s_q, s_kv, d, &ops,
//                          S_mul_dS_minus_O_dO_after_quan_dS,
//                          kTensor, seqlenMNKTensor);
//
//          // (dS * K) * descale dS
//          auto After_dS_K_before_dequan_K = createScale(
//                          After_dS_K,  // input tensor
//                          descaledSTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2006  /*UID offset*/);
//
//          // (dS * K) * descale dS * descale K
//          auto After_dS_K_before_quan_dQ = createScale(
//                          After_dS_K_before_dequan_K,  // input tensor
//                          descaleKTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2007  /*UID offset*/);
//
//          // (dS * K) * descale dS * descale K * scale dQ
//          auto dQ = createScaleWithOffset(
//                          After_dS_K_before_quan_dQ,  // input tensor
//                          "scaledQ",  // scale tensor
//                          layout,  // qkv layout
//                          CUDNN_DATA_FP8_E5M2,  // output tensor type
//                          false,  // output not virtual
//                          false,  // scale is by value
//                          &ops,
//                          QKVRaggedOffsetTensorPtr,  // ragged offset
//                          "dQ");
//
//          // Amax for dQ
//          createAmax("amaxdQ", After_dS_K_before_quan_dQ, &ops);
//
//          // dS.T @ Q
//          auto After_dSTranspose_Q = createdSQBMM(
//                          b, h, s_q, s_kv, d, layout, &ops,
//                          S_mul_dS_minus_O_dO_after_quan_dS,
//                          qTensor, seqlenMNKTensor);
//
//          // (dS.T * Q) * descale dS
//          auto After_dSTranspose_Q_before_dequan_Q = createScale(
//                          After_dSTranspose_Q,  // input tensor
//                          descaledSTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2009  /*UID offset*/);
//
//          // (dS.T * Q) * descale dS * descale Q
//          auto After_dSTranspose_Q_before_quan_dK = createScale(
//                          After_dSTranspose_Q_before_dequan_Q,  // input tensor
//                          descaleQTensor,  // scale tensor
//                          CUDNN_DATA_FLOAT,  // output tensor type
//                          true,  // output is virtual
//                          false,  // scale is by value
//                          &ops,
//                          2010  /*UID offset*/);
//
//          // (dS.T * Q) * descale dS * descale Q * scale dK
//          auto dK = createScaleWithOffset(
//                          After_dSTranspose_Q_before_quan_dK,  // input tensor
//                          "scaledK",  // scale tensor
//                          layout,  // qkv layout
//                          CUDNN_DATA_FP8_E5M2,  // output tensor type
//                          false,  // output not virtual
//                          false,  // scale is by value
//                          &ops,
//                          QKVRaggedOffsetTensorPtr,  // ragged offset
//                          "dK");
//
//          // Amax for dK
//          createAmax("amaxdK", After_dSTranspose_Q_before_quan_dK, &ops);
//
//          for (unsigned int i = 0; i < ops.size(); i++) {
//              all_ops.push_back(&ops[i]);
//          }
//
//          // Create an Operation Graph
//          auto opGraph = cudnn_frontend::OperationGraphBuilder()
//                             .setHandle(handle_)
//                             .setOperationGraph(all_ops.size(), all_ops.data())
//                             .build();
//
//          cudnn_frontend::EngineConfigList filtered_configs;
//          auto statuses = cudnn_frontend::get_heuristics_list<1>(
//                          {"heuristics_instant"}, opGraph,
//                          allowAllConfig, filtered_configs, true);
//
//          if (filtered_configs.size() == 0) {
//              cudnn_frontend::set_error_and_throw_exception(
//                      nullptr,
//                      CUDNN_STATUS_NOT_SUPPORTED,
//                      "run_mha_bprop: No config returned by the heuristics");
//          }
//
//          auto plan = cudnn_frontend::ExecutionPlanBuilder()
//                  .setHandle(handle_)
//                  .setEngineConfig(filtered_configs[0], opGraph.getTag())
//                  .build();
//          cache.insert({descriptor, plan});
//          return plan;
//      };
//
//      auto plan = get_plan(fa_bprop_cache, descriptor);
//      size_t wkspace_size = static_cast<size_t>(plan.getWorkspaceSize());
//
//      // Exit to request upper level API to allocate memory if needed
//      if (workspace_ptr == nullptr) {
//          *workspace_size = wkspace_size + ((b + 1) * 2 + b) * sizeof(int32_t);
//          return;
//      }
//
//      // cuDNN stream check needs to be moved here to support dummy kernel calls with
//      // null streams for sizing the cuDNN workspace.
//      NVTE_CHECK_CUDNN(cudnnSetStream(handle_, stream));
//
//      int32_t* qkv_ragged_offset = reinterpret_cast<int32_t*>(
//                  reinterpret_cast<int8_t*>(workspace_ptr) + wkspace_size);
//      int32_t* o_ragged_offset = reinterpret_cast<int32_t*>(
//                  reinterpret_cast<int8_t*>(workspace_ptr)
//                  + wkspace_size + (b + 1) * sizeof(int32_t));
//      int32_t* actual_seqlens_q = reinterpret_cast<int32_t*>(
//                  reinterpret_cast<int8_t*>(workspace_ptr)
//                  + wkspace_size + (b + 1) * 2 * sizeof(int32_t));
//      // FP8 currently only supports self-attention, so doesn't use devPtrcuSeqlensKV
//      dim3 blockDims(128);
//      dim3 gridDims((b + blockDims.x)/blockDims.x);
//      cu_seqlens_to_offsets<<<gridDims, blockDims, 0, stream>>>(
//                      b, h, d, reinterpret_cast<int32_t*>(devPtrcuSeqlensQ),
//                      actual_seqlens_q, qkv_ragged_offset, o_ragged_offset);
//      void* devPtrQKVRaggedOffset = reinterpret_cast<void *>(qkv_ragged_offset);
//      void* devPtrORaggedOffset = reinterpret_cast<void *>(o_ragged_offset);
//      void* devPtrMNKOverride = reinterpret_cast<void *>(actual_seqlens_q);
//
//      std::set<std::pair<uint64_t, void*>> data_ptrs;
//      float dropoutScale = 1.0f/(1.0f - dropoutProbability);
//      float dropoutScale_dOVt_OdO = 1.0f - dropoutProbability;
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Q"], devPtrQ));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["K"], devPtrK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["K_TRANSPOSE"], devPtrK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["V"], devPtrV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["V_TRANSPOSE"], devPtrV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dQ"], devPtrdQ));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dK"], devPtrdK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dV"], devPtrdV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["dO"], devPtrdO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["AttnScale"], &attnScale));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["DROPOUT_SCALE"], &dropoutScale));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["DROPOUT_SCALE_dOVt_OdO"],
//                              &dropoutScale_dOVt_OdO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["DROPOUT_SEED"], devPtrDropoutSeed));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["DROPOUT_OFFSET"], devPtrDropoutOffset));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["M"], devPtrM));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["Z_INV"], devPtrZInv));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(tensor_name_to_uid["O"], devPtrO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleQ"], devPtrDescaleQ));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleK"], devPtrDescaleK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleV"], devPtrDescaleV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleS"], devPtrDescaleS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaledS"], devPtrDescaledS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaleO"], devPtrDescaleO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["descaledO"], devPtrDescaledO));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["scaleS"], devPtrScaleS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["scaledS"], devPtrScaledS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["scaledQ"], devPtrScaledQ));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["scaledK"], devPtrScaledK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["scaledV"], devPtrScaledV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["amaxdS"], devPtrAmaxdS));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["amaxdQ"], devPtrAmaxdQ));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["amaxdK"], devPtrAmaxdK));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["amaxdV"], devPtrAmaxdV));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["QKV_RAGGED"], devPtrQKVRaggedOffset));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["O_RAGGED"], devPtrORaggedOffset));
//      data_ptrs.emplace(std::pair<uint64_t, void*>(
//                              tensor_name_to_uid["MNK_OVERRIDE"], devPtrMNKOverride));
//
//      auto variantPack  = cudnn_frontend::VariantPackBuilder()
//                             .setWorkspacePointer(workspace_ptr)
//                             .setDataPointers(data_ptrs)
//                             .build();
//      cudnnStatus_t status = cudnnBackendExecute(
//                      handle_, plan.get_raw_desc(), variantPack.get_raw_desc());
//
////      cudnn_frontend::throw_if(
////                      [status]() { return (status != CUDNN_STATUS_SUCCESS); },
////                      "Plan execute error", status);
//  } catch (cudnn_frontend::cudnnException& e) {
//      struct cudaDeviceProp prop;
//      NVTE_CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
//
//      // This example is only for GH100 cards (cudnn Version >= 8900)
//      if (!((prop.major == 9 && prop.minor == 0 && CUDNN_VERSION >= 8900))
//                      && (e.getCudnnStatus() == CUDNN_STATUS_ARCH_MISMATCH
//                              || e.getCudnnStatus() == CUDNN_STATUS_NOT_SUPPORTED)) {
//          std::cout << "Example is only supported for GH100 (cuDNN >= 8900) GPUs" << std::endl;
//      }  else {
//          std::cout << "[ERROR] Exception " << e.what() << std::endl;
//      }
//  }
}

//#endif

}  // namespace fused_attn

#if (CUDNN_VERSION >= 8900)
// fused attention FWD FP8 with packed QKV
void fused_attn_fp8_fwd_qkvpacked(
            size_t b, size_t h, size_t max_seqlen, size_t d,
            bool is_training, float attn_scale,
            float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_QKV,
            Tensor *input_output_S,
            Tensor *output_O,
            NVTETensorPack* Aux_CTX_Tensors,
            const Tensor *cu_seqlens,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  // QKV shape is [total_seqs, 3, h, d]
  void* devPtrQKV = input_QKV->data.dptr;
  void* devPtrQ = reinterpret_cast<void *>(devPtrQKV);
  void* devPtrK = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + h * d);
  void* devPtrV = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + 2 * h * d);
  void* devPtrDescaleQ = input_QKV->scale_inv.dptr;
  void* devPtrDescaleK = input_QKV->scale_inv.dptr;
  void* devPtrDescaleV = input_QKV->scale_inv.dptr;

  void* devPtrO = output_O->data.dptr;
  void* devPtrAmaxO = output_O->amax.dptr;
  void* devPtrScaleO = output_O->scale.dptr;

  void* devPtrM = nullptr;
  void* devPtrZInv = nullptr;
  if (Aux_CTX_Tensors->size == 0) {
    if (is_training) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
      Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
      Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
      output_M->data.dptr = nullptr;
      output_M->data.shape = {b, h, max_seqlen, 1};
      output_M->data.dtype = DType::kFloat32;
      output_ZInv->data.dptr = nullptr;
      output_ZInv->data.shape = {b, h, max_seqlen, 1};
      output_ZInv->data.dtype = DType::kFloat32;
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
    Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
    Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
    devPtrM = output_M->data.dptr;
    devPtrZInv = output_ZInv->data.dptr;
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  void* devPtrAmaxS = input_output_S->amax.dptr;
  void* devPtrScaleS = input_output_S->scale.dptr;
  void* devPtrDescaleS = input_output_S->scale_inv.dptr;

  void* devPtrcuSeqlens = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_QKV->data.dtype;
  size_t workspace_size = 0;

  fused_attn::fused_attn_fp8_fwd_impl(
                  b, h, max_seqlen, max_seqlen, d,
                  is_training, attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleS, devPtrScaleS, devPtrScaleO,
                  devPtrAmaxO, devPtrAmaxS,
                  devPtrcuSeqlens, devPtrcuSeqlens,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
// fused attention BWD FP8 with packed QKV
void fused_attn_fp8_bwd_qkvpacked(
            size_t b, size_t h, size_t max_seqlen, size_t d,
            float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_QKV,
            const Tensor *input_O,
            const Tensor *input_dO,
            const Tensor *input_M,
            const Tensor *input_ZInv,
            const Tensor *input_S,
            Tensor *input_output_dP,
            const Tensor *output_dQKV,
            const Tensor *cu_seqlens,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  // QKV shape is [total_seqs, 3, h, d]
  void* devPtrQKV = input_QKV->data.dptr;
  void* devPtrQ = reinterpret_cast<void *>(devPtrQKV);
  void* devPtrK = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + h * d);
  void* devPtrV = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrQKV) + 2 * h * d);
  void* devPtrDescaleQ = input_QKV->scale_inv.dptr;
  void* devPtrDescaleK = input_QKV->scale_inv.dptr;
  void* devPtrDescaleV = input_QKV->scale_inv.dptr;

  void* devPtrO = input_O->data.dptr;
  void* devPtrDescaleO = input_O->scale_inv.dptr;
  void* devPtrdO = input_dO->data.dptr;
  void* devPtrDescaledO = input_dO->scale_inv.dptr;

  void* devPtrM = input_M->data.dptr;
  void* devPtrZInv = input_ZInv->data.dptr;

  void* devPtrScaleS = input_S->scale.dptr;
  void* devPtrDescaleS = input_S->scale_inv.dptr;
  void* devPtrAmaxdS = input_output_dP->amax.dptr;
  void* devPtrScaledS = input_output_dP->scale.dptr;
  void* devPtrDescaledS = input_output_dP->scale_inv.dptr;

  // dQKV shape is [total_seqs, 3, h, d]
  void* devPtrdQKV = output_dQKV->data.dptr;
  void* devPtrdQ = reinterpret_cast<void *>(devPtrdQKV);
  void* devPtrdK = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrdQKV) + h * d);
  void* devPtrdV = reinterpret_cast<void *>(reinterpret_cast<int8_t*>(devPtrdQKV) + 2 * h * d);
  void* devPtrAmaxdQ = output_dQKV->amax.dptr;
  void* devPtrAmaxdK = output_dQKV->amax.dptr;
  void* devPtrAmaxdV = output_dQKV->amax.dptr;
  void* devPtrScaledQ = output_dQKV->scale.dptr;
  void* devPtrScaledK = output_dQKV->scale.dptr;
  void* devPtrScaledV = output_dQKV->scale.dptr;

  void* devPtrcuSeqlens = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_QKV->data.dtype;
  size_t workspace_size = 0;

  fused_attn::fused_attn_fp8_bwd_impl(
                  b, h, max_seqlen, max_seqlen, d,
                  attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO, devPtrdO,
                  devPtrdQ, devPtrdK, devPtrdV,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleO, devPtrDescaledO,
                  devPtrDescaleS, devPtrDescaledS,
                  devPtrScaleS, devPtrScaledS,
                  devPtrScaledQ, devPtrScaledK, devPtrScaledV,
                  devPtrAmaxdS,
                  devPtrAmaxdQ, devPtrAmaxdK, devPtrAmaxdV,
                  devPtrcuSeqlens, devPtrcuSeqlens,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
// fused attention FWD FP8 with separate Q, K, V
void fused_attn_fp8_fwd(
            size_t b, size_t h, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
            bool is_training, float attn_scale,
            float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_Q,
            const Tensor *input_K,
            const Tensor *input_V,
            Tensor *input_output_S,
            Tensor *output_O,
            NVTETensorPack* Aux_CTX_Tensors,
            const Tensor *cu_seqlens_q,
            const Tensor *cu_seqlens_kv,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  void* devPtrQ = input_Q->data.dptr;
  void* devPtrK = input_K->data.dptr;
  void* devPtrV = input_V->data.dptr;
  void* devPtrDescaleQ = input_Q->scale_inv.dptr;
  void* devPtrDescaleK = input_Q->scale_inv.dptr;
  void* devPtrDescaleV = input_Q->scale_inv.dptr;

  void* devPtrO = output_O->data.dptr;
  void* devPtrAmaxO = output_O->amax.dptr;
  void* devPtrScaleO = output_O->scale.dptr;

  void* devPtrM = nullptr;
  void* devPtrZInv = nullptr;
  if (Aux_CTX_Tensors->size == 0) {
    if (is_training) {
      Aux_CTX_Tensors->size = 3;
      Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
      Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
      Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
      output_M->data.dptr = nullptr;
      output_M->data.shape = {b, h, max_seqlen_q, 1};
      output_M->data.dtype = DType::kFloat32;
      output_ZInv->data.dptr = nullptr;
      output_ZInv->data.shape = {b, h, max_seqlen_q, 1};
      output_ZInv->data.dtype = DType::kFloat32;
      output_rng_state->data.dptr = nullptr;
      output_rng_state->data.shape = {2};
      output_rng_state->data.dtype = DType::kInt64;
    }
  } else if (Aux_CTX_Tensors->size == 3) {
    Tensor *output_M = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[0]);
    Tensor *output_ZInv = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[1]);
    Tensor *output_rng_state = reinterpret_cast<Tensor*>(Aux_CTX_Tensors->tensors[2]);
    devPtrM = output_M->data.dptr;
    devPtrZInv = output_ZInv->data.dptr;
    output_rng_state->data.dptr = rng_state->data.dptr;
  } else {
    NVTE_ERROR("Unexpected Aux_CTX_Tensors->size.");
  }

  //void* devPtrStats = input_output_S->data.dptr;
  void* devPtrAmaxS = input_output_S->amax.dptr;
  void* devPtrScaleS = input_output_S->scale.dptr;
  void* devPtrDescaleS = input_output_S->scale_inv.dptr;

  void* devPtrcuSeqlensQ = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  size_t workspace_size = 0;

//  fused_attn::fused_attn_fp8_fwd_impl(
//                  b, h, max_seqlen_q, max_seqlen_kv, d,
//                  is_training, attn_scale, p_dropout, qkv_layout,
//                  devPtrQ, devPtrK, devPtrV,
//                  devPtrM, devPtrZInv,
//                  devPtrO,
//                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
//                  devPtrDescaleS, devPtrScaleS, devPtrScaleO,
//                  devPtrAmaxO, devPtrAmaxS,
//                  devPtrcuSeqlensQ, devPtrcuSeqlensKV,
//                  devPtrDropoutSeed, devPtrDropoutOffset,
//                  get_cudnn_dtype(QKV_type),
//                  workspace->data.dptr, &workspace_size, stream, handle);
  fused_attn::fused_attn_fp8_fwd_impl_v1(
                  b, h, max_seqlen_q, max_seqlen_kv, d,
                  is_training, attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleS, devPtrScaleS, devPtrScaleO,
                  devPtrAmaxO, devPtrAmaxS,
                  devPtrcuSeqlensQ, devPtrcuSeqlensKV,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_fe_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
// fused attention BWD FP8 with separate Q, K, V
void fused_attn_fp8_bwd(
            size_t b, size_t h, size_t max_seqlen_q, size_t max_seqlen_kv, size_t d,
            float attn_scale, float p_dropout, NVTE_QKV_Layout qkv_layout,
            const Tensor *input_Q,
            const Tensor *input_K,
            const Tensor *input_V,
            const Tensor *input_O,
            const Tensor *input_dO,
            const Tensor *input_M,
            const Tensor *input_ZInv,
            const Tensor *input_S,
            Tensor *input_output_dP,
            const Tensor *output_dQ,
            const Tensor *output_dK,
            const Tensor *output_dV,
            const Tensor *cu_seqlens_q,
            const Tensor *cu_seqlens_kv,
            const Tensor *rng_state,
            Tensor *workspace,
            cudaStream_t stream,
            cudnnHandle_t handle) {
  using namespace transformer_engine;
  void* devPtrQ = input_Q->data.dptr;
  void* devPtrK = input_K->data.dptr;
  void* devPtrV = input_V->data.dptr;
  void* devPtrDescaleQ = input_Q->scale_inv.dptr;
  void* devPtrDescaleK = input_Q->scale_inv.dptr;
  void* devPtrDescaleV = input_Q->scale_inv.dptr;

  void* devPtrO = input_O->data.dptr;
  void* devPtrDescaleO = input_O->scale_inv.dptr;
  void* devPtrdO = input_dO->data.dptr;
  void* devPtrDescaledO = input_dO->scale_inv.dptr;

  void* devPtrM = input_M->data.dptr;
  void* devPtrZInv = input_ZInv->data.dptr;

  void* devPtrScaleS = input_S->scale.dptr;
  void* devPtrDescaleS = input_S->scale_inv.dptr;
  void* devPtrAmaxdS = input_output_dP->amax.dptr;
  void* devPtrScaledS = input_output_dP->scale.dptr;
  void* devPtrDescaledS = input_output_dP->scale_inv.dptr;

  void* devPtrdQ = output_dQ->data.dptr;
  void* devPtrdK = output_dK->data.dptr;
  void* devPtrdV = output_dV->data.dptr;
  void* devPtrAmaxdQ = output_dQ->amax.dptr;
  void* devPtrAmaxdK = output_dQ->amax.dptr;
  void* devPtrAmaxdV = output_dQ->amax.dptr;
  void* devPtrScaledQ = output_dQ->scale.dptr;
  void* devPtrScaledK = output_dQ->scale.dptr;
  void* devPtrScaledV = output_dQ->scale.dptr;

  void* devPtrcuSeqlensQ = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_q->data.dptr));
  void* devPtrcuSeqlensKV = reinterpret_cast<void *>(
                  reinterpret_cast<int32_t*>(cu_seqlens_kv->data.dptr));
  void* devPtrDropoutSeed = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr));
  void* devPtrDropoutOffset = reinterpret_cast<void *>(
                  reinterpret_cast<uint64_t*>(rng_state->data.dptr) + 1);

  const DType QKV_type = input_Q->data.dtype;
  size_t workspace_size = 0;

//  fused_attn::fused_attn_fp8_bwd_impl(
//                  b, h, max_seqlen_q, max_seqlen_kv, d,
//                  attn_scale, p_dropout, qkv_layout,
//                  devPtrQ, devPtrK, devPtrV,
//                  devPtrM, devPtrZInv,
//                  devPtrO, devPtrdO,
//                  devPtrdQ, devPtrdK, devPtrdV,
//                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
//                  devPtrDescaleO, devPtrDescaledO,
//                  devPtrDescaleS, devPtrDescaledS,
//                  devPtrScaleS, devPtrScaledS,
//                  devPtrScaledQ, devPtrScaledK, devPtrScaledV,
//                  devPtrAmaxdS,
//                  devPtrAmaxdQ, devPtrAmaxdK, devPtrAmaxdV,
//                  devPtrcuSeqlensQ, devPtrcuSeqlensKV,
//                  devPtrDropoutSeed, devPtrDropoutOffset,
//                  get_cudnn_dtype(QKV_type),
//                  workspace->data.dptr, &workspace_size, stream, handle);
  fused_attn::fused_attn_fp8_bwd_impl_v1(
                  b, h, max_seqlen_q, max_seqlen_kv, d,
                  attn_scale, p_dropout, qkv_layout,
                  devPtrQ, devPtrK, devPtrV,
                  devPtrM, devPtrZInv,
                  devPtrO, devPtrdO,
                  devPtrdQ, devPtrdK, devPtrdV,
                  devPtrDescaleQ, devPtrDescaleK, devPtrDescaleV,
                  devPtrDescaleO, devPtrDescaledO,
                  devPtrDescaleS, devPtrDescaledS,
                  devPtrScaleS, devPtrScaledS,
                  devPtrScaledQ, devPtrScaledK, devPtrScaledV,
                  devPtrAmaxdS,
                  devPtrAmaxdQ, devPtrAmaxdK, devPtrAmaxdV,
                  devPtrcuSeqlensQ, devPtrcuSeqlensKV,
                  devPtrDropoutSeed, devPtrDropoutOffset,
                  get_cudnn_fe_dtype(QKV_type),
                  workspace->data.dptr, &workspace_size, stream, handle);

  if (workspace_size > 0) {
    if (workspace->data.dptr == nullptr) {
      workspace->data.shape = { workspace_size };
      workspace->data.dtype = DType::kByte;
      return;
    }
  } else if (workspace_size == 0) {
    workspace->data.shape = { 1 };
    workspace->data.dtype = DType::kByte;
    return;
  }
}
#endif  // end of CUDNN>=8900
}  // namespace transformer_engine
