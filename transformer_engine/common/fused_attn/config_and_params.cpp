/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "config_and_params.h"

#include <cstring>

namespace {

void bool_to_uint8(bool in, void *out) {
  *reinterpret_cast<uint8_t *>(out) = static_cast<uint8_t>(in);
}

void uint8_to_bool(const void *in, bool &out) {
  out = static_cast<bool>(*reinterpret_cast<const uint8_t *>(in));
}

}  // namespace

namespace transformer_engine {

namespace fused_attn {
// Forward declarations from fused_attn/utils.h. Declared here to avoid pulling the heavy
// cuDNN frontend header into this plain C++ translation unit.
size_t get_max_batch_size(size_t batch_size);
size_t get_max_tokens(size_t num_tokens);
}  // namespace fused_attn

void populate_fused_attn_config(FusedAttnConfig *cfg) {
  NVTE_CHECK(cfg != nullptr, "FusedAttnConfig must not be NULL.");

  const int64_t b = static_cast<int64_t>(cfg->batch_size);
  const int64_t h = static_cast<int64_t>(cfg->num_attn_heads);
  const int64_t sq = static_cast<int64_t>(cfg->max_seqlen_q);
  const int64_t skv = static_cast<int64_t>(cfg->max_seqlen_kv);

  const NVTE_QKV_Format q_format = nvte_get_q_format(cfg->qkv_layout);
  const NVTE_QKV_Format kv_format = nvte_get_kv_format(cfg->qkv_layout);
  const NVTE_QKV_Layout_Group layout_group = nvte_get_qkv_layout_group(cfg->qkv_layout);
  const bool is_paged_kv = (layout_group == NVTE_QKV_Layout_Group::NVTE_Paged_KV_HD_HD_HD);
  const bool has_bias = (cfg->bias_type == NVTE_Bias_Type::NVTE_POST_SCALE_BIAS);

  const size_t num_tokens_q =
      cfg->num_tokens_q != 0 ? cfg->num_tokens_q : static_cast<size_t>(b * sq);
  const size_t num_tokens_kv =
      cfg->num_tokens_kv != 0 ? cfg->num_tokens_kv : static_cast<size_t>(b * skv);

  // Bucket the THD (ragged) batch and token counts so the support probes and the runtime
  // dispatch quantize into the same bucket, i.e. build and cache the same cuDNN graph.
  const bool is_ragged_q = (q_format == NVTE_QKV_Format::NVTE_THD);
  const bool is_ragged_kv = (kv_format == NVTE_QKV_Format::NVTE_THD);
  cfg->bucketed_batch_size =
      (is_ragged_q || is_ragged_kv) ? fused_attn::get_max_batch_size(cfg->batch_size) : 0;
  cfg->bucketed_num_tokens_q = is_ragged_q ? fused_attn::get_max_tokens(num_tokens_q) : 0;
  cfg->bucketed_num_tokens_kv = is_ragged_kv ? fused_attn::get_max_tokens(num_tokens_kv) : 0;

  if (is_paged_kv) {
    if (cfg->num_pages_k == 0) {
      cfg->num_pages_k = static_cast<size_t>(b);
    }
    if (cfg->num_pages_v == 0) {
      cfg->num_pages_v = static_cast<size_t>(b);
    }
    if (cfg->page_size_k == 0) {
      cfg->page_size_k = static_cast<size_t>(skv);
    }
    if (cfg->page_size_v == 0) {
      cfg->page_size_v = static_cast<size_t>(skv);
    }
    if (cfg->max_pages_per_seq_k == 0) {
      cfg->max_pages_per_seq_k = 1;
    }
    if (cfg->max_pages_per_seq_v == 0) {
      cfg->max_pages_per_seq_v = 1;
    }
  }

  if (has_bias) {
    if (cfg->bias_batch_size == 0) {
      cfg->bias_batch_size = static_cast<size_t>(b);
    }
    if (cfg->bias_num_heads == 0) {
      cfg->bias_num_heads = static_cast<size_t>(h);
    }
    if (cfg->bias_seqlen_q == 0) {
      cfg->bias_seqlen_q = static_cast<size_t>(sq);
    }
    if (cfg->bias_seqlen_kv == 0) {
      cfg->bias_seqlen_kv = static_cast<size_t>(skv);
    }
  }
}

}  // namespace transformer_engine

NVTEFusedAttnConfig nvte_create_fused_attn_config() {
  return new transformer_engine::FusedAttnConfig(
      transformer_engine::make_default_fused_attn_config());
}

void nvte_destroy_fused_attn_config(NVTEFusedAttnConfig config) {
  delete transformer_engine::get_fused_attn_config_mutable(config);
}

void nvte_get_fused_attn_config_attribute(NVTEFusedAttnConfig config,
                                          NVTEFusedAttnConfigAttribute attr, void *buf,
                                          size_t size_in_bytes, size_t *size_written) {
  using namespace transformer_engine;

  NVTE_CHECK(attr < kNVTEFusedAttnConfigNumAttributes,
             "Invalid NVTEFusedAttnConfigAttribute (got ", static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnConfig::attr_sizes[attr];
  if (size_written != nullptr) {
    *size_written = attr_size;
  }
  if (buf == nullptr) {
    return;
  }
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for fused attention config attribute (attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ",
             size_in_bytes, " bytes)");

  const auto &cfg = *get_fused_attn_config(config);
  switch (attr) {
    case kNVTEFusedAttnConfigIsTraining:
      bool_to_uint8(cfg.is_training, buf);
      break;
    case kNVTEFusedAttnConfigDeterministic:
      bool_to_uint8(cfg.deterministic, buf);
      break;
    case kNVTEFusedAttnConfigCudaGraph:
      bool_to_uint8(cfg.cuda_graph, buf);
      break;
    case kNVTEFusedAttnConfigReturnMaxLogit:
      bool_to_uint8(cfg.return_max_logit, buf);
      break;
    case kNVTEFusedAttnConfigQKVLayout:
      std::memcpy(buf, &cfg.qkv_layout, attr_size);
      break;
    case kNVTEFusedAttnConfigOFormat:
      std::memcpy(buf, &cfg.o_format, attr_size);
      break;
    case kNVTEFusedAttnConfigDOFormat:
      std::memcpy(buf, &cfg.do_format, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVLayout:
      std::memcpy(buf, &cfg.dqkv_layout, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVScaleInvFormat:
      std::memcpy(buf, &cfg.qkv_scale_inv_format, attr_size);
      break;
    case kNVTEFusedAttnConfigDOScaleInvFormat:
      std::memcpy(buf, &cfg.do_scale_inv_format, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasType:
      std::memcpy(buf, &cfg.bias_type, attr_size);
      break;
    case kNVTEFusedAttnConfigAttnMaskType:
      std::memcpy(buf, &cfg.attn_mask_type, attr_size);
      break;
    case kNVTEFusedAttnConfigSoftmaxType:
      std::memcpy(buf, &cfg.softmax_type, attr_size);
      break;
    case kNVTEFusedAttnConfigScalingMode:
      std::memcpy(buf, &cfg.scaling_mode, attr_size);
      break;
    case kNVTEFusedAttnConfigAttnScale:
      std::memcpy(buf, &cfg.attn_scale, attr_size);
      break;
    case kNVTEFusedAttnConfigDropout:
      std::memcpy(buf, &cfg.dropout, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenQ:
      std::memcpy(buf, &cfg.max_seqlen_q, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenKV:
      std::memcpy(buf, &cfg.max_seqlen_kv, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeLeft:
      std::memcpy(buf, &cfg.window_size_left, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeRight:
      std::memcpy(buf, &cfg.window_size_right, attr_size);
      break;
    case kNVTEFusedAttnConfigBottomRightDiagonal:
      bool_to_uint8(cfg.bottom_right_diagonal, buf);
      break;
    case kNVTEFusedAttnConfigQKVDtype:
      std::memcpy(buf, &cfg.qkv_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigODtype:
      std::memcpy(buf, &cfg.o_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigDODtype:
      std::memcpy(buf, &cfg.do_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVDtype:
      std::memcpy(buf, &cfg.dqkv_dtype, attr_size);
      break;
    case kNVTEFusedAttnConfigBatchSize:
      std::memcpy(buf, &cfg.batch_size, attr_size);
      break;
    case kNVTEFusedAttnConfigNumAttnHeads:
      std::memcpy(buf, &cfg.num_attn_heads, attr_size);
      break;
    case kNVTEFusedAttnConfigNumGqaGroups:
      std::memcpy(buf, &cfg.num_gqa_groups, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimQK:
      std::memcpy(buf, &cfg.head_dim_qk, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimV:
      std::memcpy(buf, &cfg.head_dim_v, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesK:
      std::memcpy(buf, &cfg.num_pages_k, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesV:
      std::memcpy(buf, &cfg.num_pages_v, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeK:
      std::memcpy(buf, &cfg.page_size_k, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeV:
      std::memcpy(buf, &cfg.page_size_v, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqK:
      std::memcpy(buf, &cfg.max_pages_per_seq_k, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqV:
      std::memcpy(buf, &cfg.max_pages_per_seq_v, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasBatchSize:
      std::memcpy(buf, &cfg.bias_batch_size, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasNumHeads:
      std::memcpy(buf, &cfg.bias_num_heads, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenQ:
      std::memcpy(buf, &cfg.bias_seqlen_q, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenKV:
      std::memcpy(buf, &cfg.bias_seqlen_kv, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensQ:
      std::memcpy(buf, &cfg.num_tokens_q, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensKV:
      std::memcpy(buf, &cfg.num_tokens_kv, attr_size);
      break;
    case kNVTEFusedAttnConfigBucketedBatchSize:
      std::memcpy(buf, &cfg.bucketed_batch_size, attr_size);
      break;
    case kNVTEFusedAttnConfigBucketedNumTokensQ:
      std::memcpy(buf, &cfg.bucketed_num_tokens_q, attr_size);
      break;
    case kNVTEFusedAttnConfigBucketedNumTokensKV:
      std::memcpy(buf, &cfg.bucketed_num_tokens_kv, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}

void nvte_set_fused_attn_config_attribute(NVTEFusedAttnConfig config,
                                          NVTEFusedAttnConfigAttribute attr, const void *buf,
                                          size_t size_in_bytes) {
  using namespace transformer_engine;

  NVTE_CHECK(attr < kNVTEFusedAttnConfigNumAttributes,
             "Invalid NVTEFusedAttnConfigAttribute (got ", static_cast<int>(attr), ")");
  const auto &attr_size = FusedAttnConfig::attr_sizes[attr];
  NVTE_CHECK(size_in_bytes >= attr_size,
             "Buffer is too small for fused attention config attribute (attribute ",
             static_cast<int>(attr), " needs ", attr_size, " bytes, but buffer has ",
             size_in_bytes, " bytes)");
  NVTE_CHECK(buf != nullptr, "Invalid buffer (got NULL)");

  auto &cfg = *get_fused_attn_config_mutable(config);
  switch (attr) {
    case kNVTEFusedAttnConfigIsTraining:
      uint8_to_bool(buf, cfg.is_training);
      break;
    case kNVTEFusedAttnConfigDeterministic:
      uint8_to_bool(buf, cfg.deterministic);
      break;
    case kNVTEFusedAttnConfigCudaGraph:
      uint8_to_bool(buf, cfg.cuda_graph);
      break;
    case kNVTEFusedAttnConfigReturnMaxLogit:
      uint8_to_bool(buf, cfg.return_max_logit);
      break;
    case kNVTEFusedAttnConfigQKVLayout:
      std::memcpy(&cfg.qkv_layout, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigOFormat:
      std::memcpy(&cfg.o_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDOFormat:
      std::memcpy(&cfg.do_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVLayout:
      std::memcpy(&cfg.dqkv_layout, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigQKVScaleInvFormat:
      std::memcpy(&cfg.qkv_scale_inv_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDOScaleInvFormat:
      std::memcpy(&cfg.do_scale_inv_format, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasType:
      std::memcpy(&cfg.bias_type, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigAttnMaskType:
      std::memcpy(&cfg.attn_mask_type, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigSoftmaxType:
      std::memcpy(&cfg.softmax_type, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigScalingMode:
      std::memcpy(&cfg.scaling_mode, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigAttnScale:
      std::memcpy(&cfg.attn_scale, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDropout:
      std::memcpy(&cfg.dropout, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenQ:
      std::memcpy(&cfg.max_seqlen_q, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxSeqlenKV:
      std::memcpy(&cfg.max_seqlen_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeLeft:
      std::memcpy(&cfg.window_size_left, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigWindowSizeRight:
      std::memcpy(&cfg.window_size_right, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBottomRightDiagonal:
      uint8_to_bool(buf, cfg.bottom_right_diagonal);
      break;
    case kNVTEFusedAttnConfigQKVDtype:
      std::memcpy(&cfg.qkv_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigODtype:
      std::memcpy(&cfg.o_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDODtype:
      std::memcpy(&cfg.do_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigDQKVDtype:
      std::memcpy(&cfg.dqkv_dtype, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBatchSize:
      std::memcpy(&cfg.batch_size, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumAttnHeads:
      std::memcpy(&cfg.num_attn_heads, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumGqaGroups:
      std::memcpy(&cfg.num_gqa_groups, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimQK:
      std::memcpy(&cfg.head_dim_qk, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigHeadDimV:
      std::memcpy(&cfg.head_dim_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesK:
      std::memcpy(&cfg.num_pages_k, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumPagesV:
      std::memcpy(&cfg.num_pages_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeK:
      std::memcpy(&cfg.page_size_k, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigPageSizeV:
      std::memcpy(&cfg.page_size_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqK:
      std::memcpy(&cfg.max_pages_per_seq_k, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigMaxPagesPerSeqV:
      std::memcpy(&cfg.max_pages_per_seq_v, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasBatchSize:
      std::memcpy(&cfg.bias_batch_size, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasNumHeads:
      std::memcpy(&cfg.bias_num_heads, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenQ:
      std::memcpy(&cfg.bias_seqlen_q, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBiasSeqlenKV:
      std::memcpy(&cfg.bias_seqlen_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensQ:
      std::memcpy(&cfg.num_tokens_q, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigNumTokensKV:
      std::memcpy(&cfg.num_tokens_kv, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBucketedBatchSize:
      std::memcpy(&cfg.bucketed_batch_size, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBucketedNumTokensQ:
      std::memcpy(&cfg.bucketed_num_tokens_q, buf, attr_size);
      break;
    case kNVTEFusedAttnConfigBucketedNumTokensKV:
      std::memcpy(&cfg.bucketed_num_tokens_kv, buf, attr_size);
      break;
    default:
      NVTE_ERROR("Unsupported NVTEFusedAttnConfigAttribute (got ", static_cast<int>(attr), ")");
  }
}
