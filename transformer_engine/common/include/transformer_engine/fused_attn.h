/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn.h
 *  \brief Enums and functions for fused attention.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_

#include "stdint.h"
#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \enum NVTE_QKV_Layout
 *  \brief Memory layouts of QKV tensors.
 *   `S`, `B`, `H`, `D`, and `T` stand for sequence length, batch size, number of heads,
 *   head size, and the total number of tokens in a batch, i.e. `t = sum(s_i) for i = 0...b-1`.
 *   `SBHD` and `BSHD`-based layouts are used when sequences in a batch are of equal length
 *   or padded to the same length, and `THD`-based layouts are used when sequences have
 *   different lengths in a batch. `Paged_KV`-based layouts are used for paged attention.
 */
enum NVTE_QKV_Layout {
  NVTE_SB3HD = 0,                    /*!< SB3HD layout */
  NVTE_SBH3D = 1,                    /*!< SBH3D layout */
  NVTE_SBHD_SB2HD = 2,               /*!< SBHD_SB2HD layout */
  NVTE_SBHD_SBH2D = 3,               /*!< SBHD_SBH2D layout */
  NVTE_SBHD_SBHD_SBHD = 4,           /*!< SBHD_SBHD_SBHD layout */
  NVTE_BS3HD = 5,                    /*!< BS3HD layout */
  NVTE_BSH3D = 6,                    /*!< BSH3D layout */
  NVTE_BSHD_BS2HD = 7,               /*!< BSHD_BS2HD layout */
  NVTE_BSHD_BSH2D = 8,               /*!< BSHD_BSH2D layout */
  NVTE_BSHD_BSHD_BSHD = 9,           /*!< BSHD_BSHD_BSHD layout */
  NVTE_T3HD = 10,                    /*!< T3HD layout */
  NVTE_TH3D = 11,                    /*!< TH3D layout */
  NVTE_THD_T2HD = 12,                /*!< THD_T2HD layout */
  NVTE_THD_TH2D = 13,                /*!< THD_TH2D layout */
  NVTE_THD_THD_THD = 14,             /*!< THD_THD_THD layout */
  NVTE_SBHD_BSHD_BSHD = 15,          /*!< SBHD_BSHD_BSHD layout */
  NVTE_BSHD_SBHD_SBHD = 16,          /*!< BSHD_SBHD_SBHD layout */
  NVTE_THD_BSHD_BSHD = 17,           /*!< THD_BSHD_BSHD layout */
  NVTE_THD_SBHD_SBHD = 18,           /*!< THD_SBHD_SBHD layout */
  NVTE_Paged_KV_BSHD_BSHD_BSHD = 19, /*!< Paged_KV_BSHD_BSHD_BSHD layout */
  NVTE_Paged_KV_BSHD_SBHD_SBHD = 20, /*!< Paged_KV_BSHD_SBHD_SBHD layout */
  NVTE_Paged_KV_SBHD_BSHD_BSHD = 21, /*!< Paged_KV_SBHD_BSHD_BSHD layout */
  NVTE_Paged_KV_SBHD_SBHD_SBHD = 22, /*!< Paged_KV_SBHD_SBHD_SBHD layout */
  NVTE_Paged_KV_THD_BSHD_BSHD = 23,  /*!< Paged_KV_THD_BSHD_BSHD layout */
  NVTE_Paged_KV_THD_SBHD_SBHD = 24,  /*!< Paged_KV_THD_SBHD_SBHD layout */
  NVTE_BHSD_BHSD_BHSD = 25,          /*!< BHSD_BHSD_BHSD layout */
  NVTE_QKV_Layout_NOT_SET,           /*!< Not set */
};

/*! \enum NVTE_QKV_Layout_Group
 *  \brief QKV layout groups
 */
enum NVTE_QKV_Layout_Group {
  /*! 3HD QKV layouts, i.e. BS3HD, SB3HD, T3HD */
  NVTE_3HD = 0,
  /*! H3D QKV layouts, i.e. BSH3D, SBH3D, TH3D */
  NVTE_H3D = 1,
  /*! HD_2HD QKV layouts, i.e. BSHD_BS2HD, SBHD_SB2HD, THD_T2HD */
  NVTE_HD_2HD = 2,
  /*! HD_H2D QKV layouts, i.e. BSHD_BSH2D, SBHD_SBH2D, THD_TH2D */
  NVTE_HD_H2D = 3,
  /*! HD_HD_HD QKV layouts, i.e. BSHD_BSHD_BSHD, SBHD_SBHD_SBHD, THD_THD_THD */
  NVTE_HD_HD_HD = 4,
  /*! Paged_KV_HD_HD_HD QKV layouts, e.g. Paged_KV_BSHD_BSHD_BSHD, Paged_KV_THD_SBHD_SBHD */
  NVTE_Paged_KV_HD_HD_HD = 5,
  /*! SD_SD_SD QKV layouts, e.g. BHSD_BHSD_BHSD */
  NVTE_SD_SD_SD = 6,
};

/*! \enum NVTE_QKV_Format
 *  \brief QKV formats
 */
enum NVTE_QKV_Format {
  /*! SBHD QKV format, i.e. SB3HD, SBH3D, SBHD_SB2HD, SBHD_SBH2D, SBHD_SBHD_SBHD, Paged_KV_SBHD_SBHD_SBHD */
  NVTE_SBHD = 0,
  /*! BSHD QKV format, i.e. BS3HD, BSH3D, BSHD_BS2HD, BSHD_BSH2D, BSHD_BSHD_BSHD, Paged_KV_BSHD_BSHD_BSHD */
  NVTE_BSHD = 1,
  /*! THD QKV format, i.e. T3HD, TH3D, THD_T2HD, THD_TH2D, THD_THD_THD */
  NVTE_THD = 2,
  /*! BSHD format for Q and SBHD format for KV, i.e. BSHD_SBHD_SBHD, Paged_KV_BSHD_SBHD_SBHD */
  NVTE_BSHD_2SBHD = 3,
  /*! SBHD format for Q and BSHD format for KV, i.e. SBHD_BSHD_BSHD, Paged_KV_SBHD_BSHD_BSHD */
  NVTE_SBHD_2BSHD = 4,
  /*! THD format for Q and BSHD format for KV, i.e. THD_BSHD_BSHD, Paged_KV_THD_BSHD_BSHD */
  NVTE_THD_2BSHD = 5,
  /*! THD format for Q and SBHD format for KV, i.e. THD_SBHD_SBHD, Paged_KV_THD_SBHD_SBHD */
  NVTE_THD_2SBHD = 6,
  /*! BHSD QKV format, e.g. BHSD_BHSD_BHSD */
  NVTE_BHSD = 7,
  /*! Not set */
  NVTE_QKV_Format_NOT_SET,
};

/*! \enum NVTE_Bias_Type
 *  \brief Bias types
 */
enum NVTE_Bias_Type {
  /*! No bias */
  NVTE_NO_BIAS = 0,
  /*! Bias before scale */
  NVTE_PRE_SCALE_BIAS = 1,
  /*! Bias after scale */
  NVTE_POST_SCALE_BIAS = 2,
  /*! ALiBi */
  NVTE_ALIBI = 3,
};

/*! \enum NVTE_Mask_Type
 *  \brief Attention mask types
 */
enum NVTE_Mask_Type {
  /*! No masking */
  NVTE_NO_MASK = 0,
  /*! Padding attention mask */
  NVTE_PADDING_MASK = 1,
  /*! Causal attention mask (aligned to the top left corner) */
  NVTE_CAUSAL_MASK = 2,
  /*! Padding and causal attention mask (aligned to the top left corner) */
  NVTE_PADDING_CAUSAL_MASK = 3,
  /*! Causal attention mask (aligned to the bottom right corner) */
  NVTE_CAUSAL_BOTTOM_RIGHT_MASK = 4,
  /*! Padding and causal attention mask (aligned to the bottom right corner) */
  NVTE_PADDING_CAUSAL_BOTTOM_RIGHT_MASK = 5,
};

/*! \enum NVTE_Softmax_Type
 *  \brief Attention softmax types as described in
 *  Efficient Streaming Language Models with Attention Sinks (https://arxiv.org/pdf/2309.17453v3).
 *  For a given attention score S = Q*K^T, different softmax types perform different operations on S,
 *  NVTE_VANILLA_SOFTMAX: S[:,:,:,i] = exp(S[:,:,:,i])/sum(exp(S[:,:,:,:]), dim=-1),
 *  NVTE_OFF_BY_ONE_SOFTMAX: S[:,:,:,i] = exp(S[:,:,:,i])/(1 + sum(exp(S[:,:,:,:]), dim=-1)), and
 *  NVTE_LEARNABLE_SOFTMAX: S[:,j,:,i] = exp(S[:,j,:,i])/(exp(alpha[j]) + sum(exp(S[:,j,:,:]), dim=-1)),
 *  where alpha is a learnable parameter of shape [H].
 */
enum NVTE_Softmax_Type {
  /*! Vanilla softmax */
  NVTE_VANILLA_SOFTMAX = 0,
  /*! Off-by-one softmax */
  NVTE_OFF_BY_ONE_SOFTMAX = 1,
  /*! Learnable softmax */
  NVTE_LEARNABLE_SOFTMAX = 2,
};

/*! \enum NVTE_Fused_Attn_Backend
 *  \brief Fused attention backends
 */
enum NVTE_Fused_Attn_Backend {
  /*! No supported backend */
  NVTE_No_Backend = -1,
  /*! cuDNN-based FP16/BF16 fused attention for any sequence length */
  NVTE_F16_arbitrary_seqlen = 1,
  /*! cuDNN-based FP8 fused attention */
  NVTE_FP8 = 2,
};

/*!  \brief Get QKV layout group for a given QKV layout.
 *
 *  \param[in]     qkv_layout       QKV layout, e.g. sbh3d.
 *
 *  \return        qkv layout group, e.g. h3d.
 */
NVTE_QKV_Layout_Group nvte_get_qkv_layout_group(NVTE_QKV_Layout qkv_layout);

/*!  \brief Get QKV format for a given QKV layout.
 *
 *  \param[in]     qkv_layout       QKV layout, e.g. sbh3d.
 *
 *  \return        qkv format, e.g. sbhd.
 */
NVTE_QKV_Format nvte_get_qkv_format(NVTE_QKV_Layout qkv_layout);

/*!  \brief Get Q format for a given QKV layout.
 *
 *  \param[in]     qkv_layout       QKV layout, e.g. sbhd_bshd_bshd.
 *
 *  \return        q format, e.g. sbhd.
 */
NVTE_QKV_Format nvte_get_q_format(NVTE_QKV_Layout qkv_layout);

/*!  \brief Get KV format for a given QKV layout.
 *
 *  \param[in]     qkv_layout       QKV layout, e.g. sbhd_bshd_bshd.
 *
 *  \return        kv format, e.g. bshd.
 */
NVTE_QKV_Format nvte_get_kv_format(NVTE_QKV_Layout qkv_layout);

/*! \struct NVTEFusedAttnConfig
 *  \brief Inputs to nvte_get_fused_attn_backend_v2().
 *
 * Holds algorithm/policy fields plus tensor-derived shape and dtype metadata
 * needed to determine which fused attention backend supports a given
 * configuration, and to uniquely identify the cuDNN execution plan that will
 * be cached for it. Field values are passed directly so backend support can
 * be probed without allocating any tensors.
 *
 * Distinct from NVTEFusedAttnFwdParams / NVTEFusedAttnBwdParams, which
 * additionally bind tensors and execution context for an actual call.
 *
 * Field naming follows snake_case throughout. Direction-only fields are
 * grouped together: callers querying for forward should leave bwd-only fields
 * (do_format, dqkv_layout, do_scale_inv_format, do_dtype, dqkv_dtype,
 * deterministic) at their defaults, and vice versa.
 *
 * Versioning rules:
 *  - struct_size MUST be set to sizeof(NVTEFusedAttnConfig) by the caller
 *    (use NVTE_FUSED_ATTN_CONFIG_INIT).
 *  - New fields may only be appended at the end; existing fields are never
 *    reordered, removed, or resized. The library reads only fields that are
 *    in range according to struct_size and uses safe defaults otherwise.
 */
typedef struct NVTEFusedAttnConfig {
  size_t struct_size; /*!< MUST equal sizeof(NVTEFusedAttnConfig). */
  uint32_t reserved0; /*!< Padding for layout stability; set to 0. */
  uint32_t reserved1; /*!< Padding for layout stability; set to 0. */

  /* ---- algorithm / policy ---- */
  NVTE_QKV_Layout qkv_layout;           /*!< QKV tensors' layout. */
  NVTE_QKV_Format o_format;             /*!< Output O tensor format. */
  NVTE_QKV_Format do_format;            /*!< Output-grad dO tensor format (bwd). */
  NVTE_QKV_Layout dqkv_layout;          /*!< Gradient dQKV tensor layout (bwd). */
  NVTE_QKV_Format qkv_scale_inv_format; /*!< QKV scale_inv tensor format (FP8). */
  NVTE_QKV_Format do_scale_inv_format;  /*!< dO scale_inv tensor format (FP8 bwd). */
  NVTE_Bias_Type bias_type;             /*!< Attention bias type. */
  NVTE_Mask_Type attn_mask_type;        /*!< Attention mask type. */
  NVTE_Softmax_Type softmax_type;       /*!< Attention softmax type. */
  float attn_scale;                     /*!< Pre-softmax attention scale factor. */
  float dropout;                        /*!< Dropout probability. */
  size_t max_seqlen_q;                  /*!< Max sequence length for Q. */
  size_t max_seqlen_kv;                 /*!< Max sequence length for K, V. */
  int64_t window_size_left;             /*!< Sliding window size (left half); -1 = unlimited. */
  int64_t window_size_right;            /*!< Sliding window size (right half); -1 = unlimited. */
  bool bottom_right_diagonal; /*!< Whether causal mask aligns to the bottom-right diagonal. */
  bool cuda_graph;            /*!< Whether CUDA graph capture is enabled. */

  /* ---- tensor-derived metadata (passed as values; no tensor required) ---- */
  NVTEDType q_dtype;     /*!< Data type of Tensor Q. */
  NVTEDType kv_dtype;    /*!< Data type of Tensors K, V. */
  NVTEDType o_dtype;     /*!< Data type of Tensor O. */
  NVTEDType do_dtype;    /*!< Data type of Tensor dO (bwd). */
  NVTEDType dqkv_dtype;  /*!< Data type of Tensors dQ, dK, dV (bwd). */
  size_t batch_size;     /*!< Batch size. */
  size_t num_attn_heads; /*!< Number of heads in Q. */
  size_t num_gqa_groups; /*!< Number of heads in K, V. */
  size_t head_dim_qk;    /*!< Head dimension of Q, K. */
  size_t head_dim_v;     /*!< Head dimension of V. */

  /* paged KV cache shape (set 0 when not using paged attention). */
  size_t num_pages_k;         /*!< Total number of K cache pages. */
  size_t num_pages_v;         /*!< Total number of V cache pages. */
  size_t page_size_k;         /*!< Tokens per K cache page. */
  size_t page_size_v;         /*!< Tokens per V cache page. */
  size_t max_pages_per_seq_k; /*!< Max K pages per sequence in the batch. */
  size_t max_pages_per_seq_v; /*!< Max V pages per sequence in the batch. */

  /* attention-bias broadcast shape (set 0 when not using a bias tensor). */
  size_t bias_batch_size; /*!< Bias broadcast dim for batch. */
  size_t bias_num_heads;  /*!< Bias broadcast dim for heads. */
  size_t bias_seqlen_q;   /*!< Bias broadcast dim for Q sequence length. */
  size_t bias_seqlen_kv;  /*!< Bias broadcast dim for K/V sequence length. */

  /* ---- direction-affecting behavior flags ---- */
  bool is_training;      /*!< Whether the model is in training mode. */
  bool return_max_logit; /*!< Whether to produce Max along with Stats (fwd-only). */
  bool deterministic;    /*!< Whether determinism is required (bwd-only). */

  /* ---- Future fields appended here only. ---- */
} NVTEFusedAttnConfig;

/*! \brief Default-initialize an NVTEFusedAttnConfig.
 *
 * Sets struct_size and the categorical fields (layouts, formats, masks,
 * window sizes) to safe NOT_SET / no-op defaults. Numeric and tensor-derived
 * fields, paged-KV shape, bias broadcast shape, and direction flags all
 * default to zero/false; callers must set the fields relevant to their query.
 */
#define NVTE_FUSED_ATTN_CONFIG_INIT                    \
  {                                                    \
      .struct_size = sizeof(NVTEFusedAttnConfig),      \
      .qkv_layout = NVTE_QKV_Layout_NOT_SET,           \
      .o_format = NVTE_QKV_Format_NOT_SET,             \
      .do_format = NVTE_QKV_Format_NOT_SET,            \
      .dqkv_layout = NVTE_QKV_Layout_NOT_SET,          \
      .qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET, \
      .do_scale_inv_format = NVTE_QKV_Format_NOT_SET,  \
      .bias_type = NVTE_NO_BIAS,                       \
      .attn_mask_type = NVTE_NO_MASK,                  \
      .softmax_type = NVTE_VANILLA_SOFTMAX,            \
      .window_size_left = -1,                          \
      .window_size_right = -1,                         \
  }

/*! \brief Get fused attention backend based on input parameters.
 *
 *  \param[in]     cfg    Algorithm/policy + tensor metadata describing the
 *                        attention configuration to query.
 *
 *  \return Backend able to execute this configuration, or NVTE_No_Backend if none.
 */
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend_v2(const NVTEFusedAttnConfig *cfg);

/*! \brief Get fused attention backend based on input parameters (deprecated).
 *
 * This has been deprecated in favor of nvte_get_fused_attn_backend_v2.
 *
 *  \param[in]     is_training         Whether the model is in training mode.
 *  \param[in]     q_dtype             The data type of Tensor Q.
 *  \param[in]     kv_dtype            The data type of Tensors K, V.
 *  \param[in]     qkv_layout          The layout of Tensors Q, K, V.
 *  \param[in]     bias_type           The attention bias type.
 *  \param[in]     attn_mask_type      The attention mask type.
 *  \param[in]     softmax_type        The attention softmax type.
 *  \param[in]     dropout             The dropout probability.
 *  \param[in]     num_attn_heads      The number of heads in Q.
 *  \param[in]     num_gqa_groups      The number of heads in K, V.
 *  \param[in]     max_seqlen_q        The sequence length of Q.
 *  \param[in]     max_seqlen_kv       The sequence length of K, V.
 *  \param[in]     head_dim_qk         The head dimension of Q, K.
 *  \param[in]     head_dim_v          The head dimension of V.
 *  \param[in]     window_size_left    Sliding window size (the left half).
 *  \param[in]     window_size_right   Sliding window size (the right half).
 *  \param[in]     return_max_logit    Whether to produce Max along with Stats.
 *  \param[in]     cuda_graph          Whether cuda graph capture is enabled or not.
 *  \param[in]     deterministic       Whether determinism is required or not.
 */
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    bool is_training, NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout,
    NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
    float dropout, size_t num_attn_heads, size_t num_gqa_groups, size_t max_seqlen_q,
    size_t max_seqlen_kv, size_t head_dim_qk, size_t head_dim_v, int64_t window_size_left,
    int64_t window_size_right, bool return_max_logit, bool cuda_graph, bool deterministic);

/*! \struct NVTEFusedAttnFwdParams
 *  \brief All inputs and configuration for nvte_fused_attn_fwd_v2().
 *
 * Bundles the tensor bindings, algorithm configuration, behavior flags, and
 * execution context for one forward call. Tensors that do not apply to a given
 * call (e.g. Bias when bias_type == NVTE_NO_BIAS) may be left as nullptr.
 *
 * For semantics of cu_seqlens_q_padded / cu_seqlens_kv_padded with THD layouts,
 * see the notes on nvte_fused_attn_fwd().
 *
 * Versioning rules:
 *  - struct_size MUST be set to sizeof(NVTEFusedAttnFwdParams) by the caller
 *    (use NVTE_FUSED_ATTN_FWD_PARAMS_INIT).
 *  - New fields may only be appended at the end; existing fields are never
 *    reordered, removed, or resized. The library reads only fields that are
 *    in range according to struct_size and uses safe defaults otherwise.
 */
typedef struct NVTEFusedAttnFwdParams {
  size_t struct_size; /*!< MUST equal sizeof(NVTEFusedAttnFwdParams). */
  uint32_t reserved0; /*!< Padding for layout stability; set to 0. */
  uint32_t reserved1; /*!< Padding for layout stability; set to 0. */

  /* ---- input tensors ---- */
  NVTETensor Q;                   /*!< The Q tensor. */
  NVTETensor K;                   /*!< The K tensor. */
  NVTETensor V;                   /*!< The V tensor. */
  NVTETensor Bias;                /*!< The Bias tensor. */
  NVTETensor SoftmaxOffset;       /*!< The SoftmaxOffset tensor. */
  NVTETensor cu_seqlens_q;        /*!< Cumulative sequence lengths for Q, [batch_size + 1]. */
  NVTETensor cu_seqlens_kv;       /*!< Cumulative sequence lengths for K and V, [batch_size + 1]. */
  NVTETensor cu_seqlens_q_padded; /*!< Cumulative sequence offsets for Q, [batch_size + 1]. */
  NVTETensor cu_seqlens_kv_padded; /*!< Cumulative sequence offsets for KV, [batch_size + 1]. */
  NVTETensor page_table_k; /*!< Page table for K cache, [batch_size, max_pages_per_seq_k]. */
  NVTETensor page_table_v; /*!< Page table for V cache, [batch_size, max_pages_per_seq_v]. */
  NVTETensor rng_state;    /*!< Seed and offset of CUDA random number generator. */

  /* ---- output / inout tensors ---- */
  NVTETensor S;                    /*!< The S tensor (in/out). */
  NVTETensor O;                    /*!< The output O tensor. */
  NVTETensorPack *Aux_CTX_Tensors; /*!< Auxiliary output tensors when training,
                                         e.g. softmax stats, optional Max, rng_state. */

  /* ---- sizes ---- */
  size_t max_seqlen_q;  /*!< Max sequence length for Q;
                             it may be >= max(seqlen_q_i) for i = 0, ..., batch_size - 1. */
  size_t max_seqlen_kv; /*!< Max sequence length for K and V;
                             it may be >= max(seqlen_kv_i) for i = 0, ..., batch_size - 1. */

  /* ---- algorithm config ---- */
  NVTE_QKV_Layout qkv_layout;           /*!< QKV tensors' layout. */
  NVTE_QKV_Format o_format;             /*!< Output format. */
  NVTE_QKV_Format qkv_scale_inv_format; /*!< Format of scale-inverse tensors for QKV;
                                                NVTE_QKV_Format_NOT_SET = infer from qkv_layout. */
  NVTE_Bias_Type bias_type;             /*!< Bias type. */
  NVTE_Mask_Type attn_mask_type;        /*!< Attention mask type. */
  NVTE_Softmax_Type softmax_type;       /*!< Attention softmax type. */

  /* ---- numerics / windowing ---- */
  float attn_scale;           /*!< Scaling factor for Q * K.T. */
  float dropout;              /*!< Dropout probability. */
  int64_t window_size_left;   /*!< Sliding window size (left half); -1 = unlimited. */
  int64_t window_size_right;  /*!< Sliding window size (right half); -1 = unlimited. */
  bool bottom_right_diagonal; /*!< Whether to align sliding window and ALiBi diagonal
                                       to the bottom right corner of the softmax matrix. */

  /* ---- behavior ---- */
  bool is_training;      /*!< Whether this is in training mode or inference. */
  bool return_max_logit; /*!< Whether to produce Max along with Stats. */
  bool cuda_graph;       /*!< Whether CUDA graph capture is enabled. */

  /* ---- execution ---- */
  NVTETensor workspace; /*!< Workspace tensor. */
  cudaStream_t stream;  /*!< CUDA stream used for this operation. */

  /* ---- Future fields appended here only. ---- */
} NVTEFusedAttnFwdParams;

/*! \brief Default-initialize an NVTEFusedAttnFwdParams.
 *
 * Sets struct_size and all enums/scalars to safe defaults. Tensors and the
 * Aux_CTX_Tensors pack default to nullptr; callers must set the fields
 * relevant to their call.
 */
#define NVTE_FUSED_ATTN_FWD_PARAMS_INIT                \
  {                                                    \
      .struct_size = sizeof(NVTEFusedAttnFwdParams),   \
      .qkv_layout = NVTE_QKV_Layout_NOT_SET,           \
      .o_format = NVTE_QKV_Format_NOT_SET,             \
      .qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET, \
      .bias_type = NVTE_NO_BIAS,                       \
      .attn_mask_type = NVTE_NO_MASK,                  \
      .softmax_type = NVTE_VANILLA_SOFTMAX,            \
      .attn_scale = 1.0f,                              \
      .window_size_left = -1,                          \
      .window_size_right = -1,                         \
  }

/*! \brief Compute dot product attention with separate Q, K and V.
 *
 * Computes:
 *  - P = Q * Transpose(K) + Bias
 *  - S = ScaleMaskSoftmax(P)
 *  - D = Dropout(S)
 *  - O = D * Transpose(V)
 *
 * See the notes on nvte_fused_attn_fwd() for cu_seqlens_q_padded /
 * cu_seqlens_kv_padded semantics with THD layouts.
 *
 *  \param[in,out] params  All inputs, configuration, and execution context.
 *                         Output tensors and the Aux_CTX_Tensors pack are
 *                         written through pointers the caller has supplied
 *                         in the struct.
 */
void nvte_fused_attn_fwd_v2(const NVTEFusedAttnFwdParams *params);

/*! \brief Compute dot product attention with separate Q, K and V (deprecated).
 *
 * This has been deprecated in favor of nvte_fused_attn_fwd_v2.
 *
 * Computes:
 *  - P = Q * Transpose(K) + Bias
 *  - S = ScaleMaskSoftmax(P)
 *  - D = Dropout(S)
 *  - O = D * Transpose(V)
 *
 * Notes:
 *
 * Tensors `cu_seqlens_q_padded` and `cu_seqlens_kv_padded`
 * help identify the correct offsets of different sequences in tensors Q, K, V and O.
 * When the QKV format (`nvte_get_qkv_format(qkv_layout)`) is `bshd` or `sbhd`,
 * offset tensors are not used in the attention calculation and can be set to empty `NVTETensor`s.
 * When the QKV format is `thd`, these tensors should follow the following rules.
 * When there is no padding between sequences, the offset tensors should be equal to
 * `cu_seqlens_q` and `cu_seqlens_kv` respectively.
 * When there is padding between sequences, users are responsible to adjust the offsets as needed.
 * For example, a tensor of 4 sequences `[a, PAD, b, b, c, PAD, PAD, d, d]` should have
 * `cu_seqlens = [0, 1, 3, 4, 6]` and `cu_seqlens_padded= [0, 2, 4, 7, 9]`.
 *
 *  \param[in]     Q                         The Q tensor.
 *  \param[in]     K                         The K tensor.
 *  \param[in]     V                         The V tensor.
 *  \param[in]     Bias                      The Bias tensor.
 *  \param[in]     SoftmaxOffset             The SoftmaxOffset tensor.
 *  \param[in,out] S                         The S tensor.
 *  \param[out]    O                         The output O tensor.
 *  \param[out]    Aux_CTX_Tensors           Auxiliary output tensors when training,
 *                                           e.g. softmax stats, optional Max, rng_state.
 *  \param[in]     cu_seqlens_q              Cumulative sequence lengths for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv             Cumulative sequence lengths for K and V, [batch_size + 1].
 *  \param[in]     cu_seqlens_q_padded       Cumulative sequence offsets for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv_padded      Cumulative sequence offsets for KV, [batch_size + 1].
 *  \param[in]     page_table_k              Page table for K cache, [batch_size, max_pages_per_seq_k].
 *  \param[in]     page_table_v              Page table for V cache, [batch_size, max_pages_per_seq_v].
 *  \param[in]     rng_state                 Seed and offset of CUDA random number generator.
 *  \param[in]     max_seqlen_q              Max sequence length used for computing for Q.
 *                                           it may be >= max(seqlen_q_i) for i=0,...batch_size-1.
 *  \param[in]     max_seqlen_kv             Max sequence length used for computing for K and V.
 *                                           it may be >= max(seqlen_kv_i) for i=0,...batch_size-1.
 *  \param[in]     is_training               Whether this is in training mode or inference.
 *  \param[in]     return_max_logit          Whether to produce Max along with Stats.
 *  \param[in]     cuda_graph                Whether cuda graph capture is enabled or not.
 *  \param[in]     attn_scale                Scaling factor for Q * K.T.
 *  \param[in]     dropout                   Dropout probability.
 *  \param[in]     qkv_layout                QKV tensors' layout.
 *  \param[in]     o_format                  Output format.
 *  \param[in]     qkv_scale_inv_format      Format of scale-inverse tensors for QKV;
 *                                           if NVTE_QKV_Format_NOT_SET, inferred from qkv_layout.
 *  \param[in]     bias_type                 Bias type.
 *  \param[in]     attn_mask_type            Attention mask type.
 *  \param[in]     softmax_type              Attention softmax type.
 *  \param[in]     window_size_left          Sliding window size (the left half).
 *  \param[in]     window_size_right         Sliding window size (the right half).
 *  \param[in]     bottom_right_diagonal     Whether to align sliding window and ALiBi diagonal to the bottom right corner of the softmax matrix.
 *  \param[in]     workspace                 Workspace tensor.
 *  \param[in]     stream                    CUDA stream used for this operation.
 */
void nvte_fused_attn_fwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor Bias, const NVTETensor SoftmaxOffset, NVTETensor S,
                         NVTETensor O, NVTETensorPack *Aux_CTX_Tensors,
                         const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                         const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, const NVTETensor page_table_k,
                         const NVTETensor page_table_v, const NVTETensor rng_state,
                         size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                         bool return_max_logit, bool cuda_graph, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
                         NVTE_QKV_Format qkv_scale_inv_format, NVTE_Bias_Type bias_type,
                         NVTE_Mask_Type attn_mask_type, NVTE_Softmax_Type softmax_type,
                         int64_t window_size_left, int64_t window_size_right,
                         bool bottom_right_diagonal, NVTETensor workspace, cudaStream_t stream);

/*! \struct NVTEFusedAttnBwdParams
 *  \brief All inputs and configuration for nvte_fused_attn_bwd_v2().
 *
 * Bundles the tensor bindings, algorithm configuration, behavior flags, and
 * execution context for one backward call. Auxiliary tensors saved by the
 * forward pass (e.g. softmax stats, RNG state, saved Bias / SoftmaxOffset)
 * are accessed via Aux_CTX_Tensors and are NOT separately set as fields here.
 *
 * For semantics of cu_seqlens_q_padded / cu_seqlens_kv_padded with THD layouts,
 * see the notes on nvte_fused_attn_bwd().
 *
 * Versioning rules:
 *  - struct_size MUST be set to sizeof(NVTEFusedAttnBwdParams) by the caller
 *    (use NVTE_FUSED_ATTN_BWD_PARAMS_INIT).
 *  - New fields may only be appended at the end; existing fields are never
 *    reordered, removed, or resized. The library reads only fields that are
 *    in range according to struct_size and uses safe defaults otherwise.
 */
typedef struct NVTEFusedAttnBwdParams {
  size_t struct_size; /*!< MUST equal sizeof(NVTEFusedAttnBwdParams). */
  uint32_t reserved0; /*!< Padding for layout stability; set to 0. */
  uint32_t reserved1; /*!< Padding for layout stability; set to 0. */

  /* ---- input tensors ---- */
  NVTETensor Q;                   /*!< The Q tensor. */
  NVTETensor K;                   /*!< The K tensor. */
  NVTETensor V;                   /*!< The V tensor. */
  NVTETensor O;                   /*!< The O tensor from forward. */
  NVTETensor dO;                  /*!< The gradient of the O tensor. */
  NVTETensor S;                   /*!< The S tensor. */
  NVTETensor cu_seqlens_q;        /*!< Cumulative sequence lengths for Q, [batch_size + 1]. */
  NVTETensor cu_seqlens_kv;       /*!< Cumulative sequence lengths for K and V, [batch_size + 1]. */
  NVTETensor cu_seqlens_q_padded; /*!< Cumulative sequence offsets for Q, [batch_size + 1]. */
  NVTETensor cu_seqlens_kv_padded; /*!< Cumulative sequence offsets for KV, [batch_size + 1]. */
  const NVTETensorPack *Aux_CTX_Tensors; /*!< Auxiliary tensors from forward context,
                                                    e.g. softmax stats, optional Max, rng_state. */

  /* ---- output / inout tensors ---- */
  NVTETensor dP;             /*!< The gradient of the P tensor (in/out). */
  NVTETensor dQ;             /*!< The gradient of the Q tensor. */
  NVTETensor dK;             /*!< The gradient of the K tensor. */
  NVTETensor dV;             /*!< The gradient of the V tensor. */
  NVTETensor dBias;          /*!< The gradient of the Bias tensor. */
  NVTETensor dSoftmaxOffset; /*!< The gradient of the SoftmaxOffset tensor. */

  /* ---- sizes ---- */
  size_t max_seqlen_q;  /*!< Max sequence length for Q;
                             it may be >= max(seqlen_q_i) for i = 0, ..., batch_size - 1. */
  size_t max_seqlen_kv; /*!< Max sequence length for K and V;
                             it may be >= max(seqlen_kv_i) for i = 0, ..., batch_size - 1. */

  /* ---- algorithm config ---- */
  NVTE_QKV_Layout qkv_layout;           /*!< QKV tensors' layout. */
  NVTE_QKV_Layout dqkv_layout;          /*!< QKV gradient tensors' layout. */
  NVTE_QKV_Format o_format;             /*!< Output format. */
  NVTE_QKV_Format do_format;            /*!< Output gradient's format. */
  NVTE_QKV_Format qkv_scale_inv_format; /*!< Format of scale-inverse tensors for QKV;
                                                NVTE_QKV_Format_NOT_SET = infer from qkv_layout. */
  NVTE_QKV_Format do_scale_inv_format;  /*!< Format of scale-inverse tensors for dO;
                                                NVTE_QKV_Format_NOT_SET = infer from output layout. */
  NVTE_Bias_Type bias_type;             /*!< Bias type. */
  NVTE_Mask_Type attn_mask_type;        /*!< Attention mask type. */
  NVTE_Softmax_Type softmax_type;       /*!< Attention softmax type. */

  /* ---- numerics / windowing ---- */
  float attn_scale;           /*!< Scaling factor for Q * K.T. */
  float dropout;              /*!< Dropout probability. */
  int64_t window_size_left;   /*!< Sliding window size (left half); -1 = unlimited. */
  int64_t window_size_right;  /*!< Sliding window size (right half); -1 = unlimited. */
  bool bottom_right_diagonal; /*!< Whether to align sliding window and ALiBi diagonal
                                       to the bottom right corner of the softmax matrix. */

  /* ---- behavior ---- */
  bool deterministic; /*!< Whether to execute with deterministic behaviours. */
  bool cuda_graph;    /*!< Whether CUDA graph capture is enabled. */

  /* ---- execution ---- */
  NVTETensor workspace; /*!< Workspace tensor. */
  cudaStream_t stream;  /*!< CUDA stream used for this operation. */

  /* ---- Future fields appended here only. ---- */
} NVTEFusedAttnBwdParams;

/*! \brief Default-initialize an NVTEFusedAttnBwdParams.
 *
 * Sets struct_size and all enums/scalars to safe defaults. Tensors and the
 * Aux_CTX_Tensors pack default to nullptr; callers must set the fields
 * relevant to their call.
 */
#define NVTE_FUSED_ATTN_BWD_PARAMS_INIT                \
  {                                                    \
      .struct_size = sizeof(NVTEFusedAttnBwdParams),   \
      .qkv_layout = NVTE_QKV_Layout_NOT_SET,           \
      .dqkv_layout = NVTE_QKV_Layout_NOT_SET,          \
      .o_format = NVTE_QKV_Format_NOT_SET,             \
      .do_format = NVTE_QKV_Format_NOT_SET,            \
      .qkv_scale_inv_format = NVTE_QKV_Format_NOT_SET, \
      .do_scale_inv_format = NVTE_QKV_Format_NOT_SET,  \
      .bias_type = NVTE_NO_BIAS,                       \
      .attn_mask_type = NVTE_NO_MASK,                  \
      .softmax_type = NVTE_VANILLA_SOFTMAX,            \
      .attn_scale = 1.0f,                              \
      .window_size_left = -1,                          \
      .window_size_right = -1,                         \
  }

/*! \brief Compute the backward of the dot product attention with separate Q, K and V.
 *
 * See the notes on nvte_fused_attn_bwd() for cu_seqlens_q_padded /
 * cu_seqlens_kv_padded semantics with THD layouts.
 *
 *  \param[in,out] params  All inputs, configuration, and execution context.
 *                         Output tensors are written through pointers the
 *                         caller has supplied in the struct.
 */
void nvte_fused_attn_bwd_v2(const NVTEFusedAttnBwdParams *params);

/*! \brief Compute the backward of the dot product attention with separate Q, K and V (deprecated).
 *
 * This has been deprecated in favor of nvte_fused_attn_bwd_v2.
 *
 * Notes:
 *
 * Tensors `cu_seqlens_q_padded` and `cu_seqlens_kv_padded`
 * help identify the correct offsets of different sequences in tensors Q, K, V and O.
 * When the QKV format (`nvte_get_qkv_format(qkv_layout)`) is `bshd` or `sbhd`,
 * offset tensors are not used in the attention calculation and can be set to empty `NVTETensor`s.
 * When the QKV format is `thd`, these tensors should follow the following rules.
 * When there is no padding between sequences, the offset tensors should be equal to
 * `cu_seqlens_q` and `cu_seqlens_kv` respectively.
 * When there is padding between sequences, users are responsible to adjust the offsets as needed.
 * For example, a tensor of 4 sequences `[a, PAD, b, b, c, PAD, PAD, d, d]` should have
 * `cu_seqlens = [0, 1, 3, 4, 6]` and `cu_seqlens_padded= [0, 2, 4, 7, 9]`.
 *
 *  \param[in]     Q                         The Q tensor.
 *  \param[in]     K                         The K tensor.
 *  \param[in]     V                         The V tensor.
 *  \param[in]     O                         The O tensor from forward.
 *  \param[in]     dO                        The gradient of the O tensor.
 *  \param[in]     S                         The S tensor.
 *  \param[in,out] dP                        The gradient of the P tensor.
 *  \param[in]     Aux_CTX_Tensors           Auxiliary tensors from context when in training mode,
 *                                           e.g. softmax stats, optional Max, rng_state.
 *  \param[out]    dQ                        The gradient of the Q tensor.
 *  \param[out]    dK                        The gradient of the K tensor.
 *  \param[out]    dV                        The gradient of the V tensor.
 *  \param[out]    dBias                     The gradient of the Bias tensor.
 *  \param[out]    dSoftmaxOffset            The gradient of the SoftmaxOffset tensor.
 *  \param[in]     cu_seqlens_q              Cumulative sequence lengths for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv             Cumulative sequence lengths for K and V, [batch_size + 1].
 *  \param[in]     cu_seqlens_q_padded       Cumulative sequence offsets for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv_padded      Cumulative sequence offsets for KV, [batch_size + 1].
 *  \param[in]     max_seqlen_q              Max sequence length used for computing for Q.
 *                                           it may be >= max(seqlen_q_i) for i=0,...batch_size-1.
 *  \param[in]     max_seqlen_kv             Max sequence length used for computing for K and V.
 *                                           it may be >= max(seqlen_kv_i) for i=0,...batch_size-1.
 *  \param[in]     attn_scale                Scaling factor for Q * K.T.
 *  \param[in]     dropout                   Dropout probability.
 *  \param[in]     qkv_layout                QKV tensors' layout.
 *  \param[in]     o_format                  Output format.
 *  \param[in]     do_format                 Output gradient's format.
 *  \param[in]     dqkv_layout               QKV gradient tensors' layout.
 *  \param[in]     qkv_scale_inv_format      Format of scale-inverse tensors for QKV;
 *                                           if NVTE_QKV_Format_NOT_SET, inferred from qkv_layout.
 *  \param[in]     do_scale_inv_format       Format of scale-inverse tensors for dO;
 *                                           if NVTE_QKV_Format_NOT_SET, inferred from the output layout.
 *  \param[in]     bias_type                 Bias type.
 *  \param[in]     attn_mask_type            Attention mask type.
 *  \param[in]     softmax_type              Attention softmax type.
 *  \param[in]     window_size_left          Sliding window size (the left half).
 *  \param[in]     window_size_right         Sliding window size (the right half).
 *  \param[in]     bottom_right_diagonal     Whether to align sliding window and ALiBi diagonal to the bottom right corner of the softmax matrix.
 *  \param[in]     deterministic             Whether to execute with deterministic behaviours.
 *  \param[in]     cuda_graph                Whether cuda graph capture is enabled or not.
 *  \param[in]     workspace                 Workspace tensor.
 *  \param[in]     stream                    CUDA stream used for this operation.
 */
void nvte_fused_attn_bwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor O, const NVTETensor dO, const NVTETensor S, NVTETensor dP,
                         const NVTETensorPack *Aux_CTX_Tensors, NVTETensor dQ, NVTETensor dK,
                         NVTETensor dV, NVTETensor dBias, NVTETensor dSoftmaxOffset,
                         const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                         const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, size_t max_seqlen_q,
                         size_t max_seqlen_kv, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_QKV_Format o_format,
                         NVTE_QKV_Format do_format, NVTE_QKV_Layout dqkv_layout,
                         NVTE_QKV_Format qkv_scale_inv_format, NVTE_QKV_Format do_scale_inv_format,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         NVTE_Softmax_Type softmax_type, int64_t window_size_left,
                         int64_t window_size_right, bool bottom_right_diagonal, bool deterministic,
                         bool cuda_graph, NVTETensor workspace, cudaStream_t stream);

/*!  \brief Update the RNG state with the seed and calculated offset.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     rng_state_dst             RNG state to store seed and offset.
 *  \param[in]     seed                      Seed for RNG state.
 *  \param[in]     q_max_seqlen              Max sequence length used for computing for Q.
 *                                           it may be >= max(seqlen_q_i) for i=0,...batch_size-1.
 *  \param[in]     kv_max_seqlen             Max sequence length used for computing for K and V.
 *                                           it may be >= max(seqlen_kv_i) for i=0,...batch_size-1.
 *  \param[in]     backend                   Fused attention backend.
 *  \param[in]     stream                    CUDA stream used for this operation.
 */
void nvte_populate_rng_state_async(NVTETensor rng_state_dst, const NVTETensor seed,
                                   size_t q_max_seqlen, size_t kv_max_seqlen,
                                   NVTE_Fused_Attn_Backend backend, cudaStream_t stream);

/*!  \brief Get KV format for a given QKV layout.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     cu_seqlens               Cumulative sequence lengths, [batch_size + 1].
 *  \param[in]     workspace                Workspace tensor.
 *  \param[in]     len                      batch_size x sequence_length.
 *  \param[in]     stream                   CUDA stream used for this operation.
 */
uint32_t nvte_get_runtime_num_segments(NVTETensor cu_seqlens, NVTETensor workspace, size_t len,
                                       cudaStream_t stream);

/*!  \brief Set the seed and offset for RNG state.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[out]    rng_state_ptr            A size 2 array storing the RNG's seed and offset respectively.
 *  \param[in]     captured                 Whether a CUDA graph is being captured.
 *  \param[in]     seed_ptr                 Seed pointer.
 *  \param[in]     seed_val                 Seed value.
 *  \param[in]     offset_ptr               Offset pointer.
 *  \param[in]     offset_val               Offset value.
 *  \param[in]     offset_intragraph        Intragraph offset in RNG states. For use with CUDA Graphs.
 *  \param[in]     stream                   CUDA stream used for this operation.
 */
void nvte_extract_seed_and_offset(int64_t *rng_state_ptr, int captured, int64_t *seed_ptr,
                                  uint64_t seed_val, int64_t *offset_ptr, uint64_t offset_val,
                                  uint32_t offset_intragraph, cudaStream_t stream);

/*!  \brief Copy keys and values into the KV cache.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     new_k               Key tensor.
 *  \param[in]     new_v               Value tensor.
 *  \param[out]    k_cache             Key cache.
 *  \param[out]    v_cache             Value cache.
 *  \param[in]     page_table          Page table for K cache, [batch_size, max_pages_per_seq].
 *  \param[in]     cu_new_lens         Cumulative sequence lengths.
 *  \param[in]     cu_cached_lens      Cached cumulative sequence lengths.
 *  \param[in]     qkv_format          QKV format, e.g. sbhd.
 *  \param[in]     b                   Batch size.
 *  \param[in]     max_ctx_len         Maximum context length.
 *  \param[in]     max_seq_len         Maximum sequence length.
 *  \param[in]     max_pages_per_seq   Maximum number of pages per sequence.
 *  \param[in]     is_non_paged        Whether the cache is paged or not.
 *  \param[in]     stream              CUDA stream used for this operation.
 */
void nvte_copy_to_kv_cache(NVTETensor new_k, NVTETensor new_v, NVTETensor k_cache,
                           NVTETensor v_cache, NVTETensor page_table, NVTETensor cu_new_lens,
                           NVTETensor cu_cached_lens, NVTE_QKV_Format qkv_format, int b,
                           int max_ctx_len, int max_seq_len, int max_pages_per_seq,
                           int is_non_paged, cudaStream_t stream);

/*!  \brief Extract the first half (half_idx=0) or second half (half_idx=1) of a THD tensor.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     tensor              Input tensor.
 *  \param[in]     cu_seqlens          Cumulative sequence lengths, [batch_size + 1].
 *  \param[out]    half                Output tensor.
 *  \param[in]     half_idx            Whether to read first or second half of input tensor.
 *  \param[in]     stream              CUDA stream used for this operation.
 */
void nvte_cp_thd_read_half_tensor(const NVTETensor &tensor, const NVTETensor &cu_seqlens,
                                  NVTETensor half, int half_idx, cudaStream_t stream);

/*!  \brief Correct the second half of the softmax LSE (LogSumExp) for context parallelism.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[out]    lse                 Output tensor.
 *  \param[in]     lse_per_step        Input tensor.
 *  \param[in]     cu_seqlens          Cumulative sequence lengths, [batch_size + 1].
 *  \param[in]     lse_packed          Whether or not lse_per_step is packed.
 *  \param[in]     stream              CUDA stream used for this operation.
 */
void nvte_cp_thd_second_half_lse_correction(NVTETensor lse, const NVTETensor &lse_per_step,
                                            const NVTETensor &cu_seqlens, int lse_packed,
                                            cudaStream_t stream);

/*!  \brief Read the second half of the softmax LSE (LogSumExp) for context parallelism.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     lse                      Input tensor.
 *  \param[in]     cu_seqlens               Cumulative sequence lengths, [batch_size + 1].
 *  \param[out]    half_lse                 Output tensor.
 *  \param[in]     lse_packed               Whether or the softmax LSE is in packed format.
 *  \param[in]     second_half_lse_seqlen   Sequence length.
 *  \param[in]     stream                   CUDA stream used for this operation.
 */
void nvte_cp_thd_read_second_half_lse(const NVTETensor &lse, const NVTETensor &cu_seqlens,
                                      NVTETensor half_lse, int lse_packed,
                                      int second_half_lse_seqlen, cudaStream_t stream);

/*!  \brief Correct the THD format output of context parallelism in forward pass.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[out]    out                   Output tensor.
 *  \param[in]     out_per_step          THD format output of context parallelism in forward pass.
 *  \param[in]     lse                   Softmax LSE.
 *  \param[in]     lse_per_step          Softmax LSE per step.
 *  \param[in]     cu_seqlens            Cumulative sequence lengths, [batch_size + 1].
 *  \param[in]     only_second_half      Whether or not to correct only second half.
 *  \param[in]     lse_packed            Whether or the softmax LSE is in packed format.
 *  \param[in]     stream                CUDA stream used for this operation.
 */
void nvte_cp_thd_out_correction(NVTETensor out, const NVTETensor &out_per_step,
                                const NVTETensor &lse, const NVTETensor &lse_per_step,
                                const NVTETensor &cu_seqlens, int only_second_half, int lse_packed,
                                cudaStream_t stream);

/*!  \brief Correct the THD format output of context parallelism in forward pass.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[out]    grad                Output tensor.
 *  \param[in]     grad_per_step       THD format gradient of context parallelism.
 *  \param[in]     cu_seqlens          Cumulative sequence lengths, [batch_size + 1].
 *  \param[in]     first_half          One of ("add", "copy", "none") correction op for first half.
 *  \param[in]     second_half         One of ("add", "copy", "none") correction op for second half.
                                       Must be different from first_half.
 *  \param[in]     stream              CUDA stream used for this operation.
 */
void nvte_cp_thd_grad_correction(NVTETensor grad, const NVTETensor &grad_per_step,
                                 const NVTETensor &cu_seqlens, const char *first_half,
                                 const char *second_half, cudaStream_t stream);

/*!  \brief Generate partitioned indices for inputs in THD format.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     cu_seqlens          Cumulative sequence lengths, [batch_size + 1].
 *  \param[out]    output              Output tensor.
 *  \param[in]     total_tokens        Total number of tokens.
 *  \param[in]     world_size          Total number of devices for context parallelism.
 *  \param[in]     rank                Device ID for current device.
 *  \param[in]     stream              CUDA stream used for this operation.
 */
void nvte_cp_thd_get_partitioned_indices(const NVTETensor &cu_seqlens, NVTETensor output,
                                         int total_tokens, int world_size, int rank,
                                         cudaStream_t stream);

/*!  \brief Convert tensor from THD to BSHD format.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     tensor           Input tensor.
 *  \param[in]     cu_seqlens       Cumulative sequence lengths, [batch_size + 1].
 *  \param[out]    new_tensor       Output tensor.
 *  \param[in]     b                Batch size.
 *  \param[in]     max_seq_len      Maximum sequence length.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_convert_thd_to_bshd(NVTETensor tensor, NVTETensor cu_seqlens, NVTETensor new_tensor,
                              int b, int max_seq_len, cudaStream_t stream);

/*!  \brief Convert tensor from BSHD to THD format.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     tensor           Input tensor.
 *  \param[in]     cu_seqlens       Cumulative sequence lengths, [batch_size + 1].
 *  \param[out]    new_tensor       Output tensor.
 *  \param[in]     t                Packed sequence length.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_convert_bshd_to_thd(NVTETensor tensor, NVTETensor cu_seqlens, NVTETensor new_tensor,
                              int t, cudaStream_t stream);

/*!  \brief Prepare QKV tensor for Flash Attention forward kernel.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     qkvi             Input tensor.
 *  \param[out]    qkv              Output tensor.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_prepare_flash_attn_fwd(NVTETensor qkvi, NVTETensor qkv, cudaStream_t stream);

/*!  \brief Prepare QKV tensor for Flash Attention backward kernel.
 *
 * \warning   This API is **experimental** and subject to change.
 *
 *  \param[in]     q                Input query tensor.
 *  \param[in]     k                Input key tensor.
 *  \param[in]     v                Input value tensor.
 *  \param[out]    qkv              Output tensor.
 *  \param[in]     stream           CUDA stream used for this operation.
 */
void nvte_prepare_flash_attn_bwd(NVTETensor q, NVTETensor k, NVTETensor v, NVTETensor qkv,
                                 cudaStream_t stream);

/*!  \brief Transpose multiple tensors from BSHD/SBHD to BHSD.
 *
 *  Each input tensor is 4D in BSHD or SBHD layout, and the corresponding output tensor
 *  is 4D in BHSD layout. Output tensors are pre-allocated and may have a larger last dimension.
 *
 *  \param[in]     inputs           List of input tensors.
 *  \param[in,out] outputs          List of output tensors.
 *  \param[in]     num_tensors      Number of tensors in the list.
 *  \param[in]     original_format  Original QKV format (NVTE_BSHD or NVTE_SBHD).
 *  \param[in]     stream           CUDA stream.
 */
void nvte_multi_tensor_transpose_to_bhsd(NVTETensor *inputs, NVTETensor *outputs,
                                         size_t num_tensors, NVTE_QKV_Format original_format,
                                         cudaStream_t stream);

/*!  \brief Pad the last dimension of multiple 2D tensors with zeros in one kernel launch.
 *
 *  Each tensor copies a row-major (rows, in_cols) input to a (rows, out_cols) output,
 *  zero-filling the region [in_cols, out_cols) in every row.
 *  Outputs must be pre-allocated with out_cols >= in_cols and matching dtype.
 *
 *  \param[in]     inputs       List of input tensors.
 *  \param[in,out] outputs      List of output tensors.
 *  \param[in]     num_tensors  Number of tensors in the list.
 *  \param[in]     stream       CUDA stream.
 */
void nvte_multi_tensor_pad_last_dim(NVTETensor *inputs, NVTETensor *outputs, size_t num_tensors,
                                    cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"

#include <array>
#include <cstddef>
#include <utility>

/*! \brief Parses a QKV tensor shape into canonical (b, h, s, d, t) dimensions
 *         and converts between QKV formats.
 */
class AttentionShape {
 public:
  inline AttentionShape(NVTE_QKV_Format fmt, const size_t *shape) : canonical_{} {
    auto [ndim, order] = dim_order(fmt);
    for (size_t i = 0; i < ndim; ++i) canonical_[order[i]] = shape[i];
  }

  size_t b() const { return canonical_[0]; }
  size_t h() const { return canonical_[1]; }
  size_t s() const { return canonical_[2]; }
  size_t d() const { return canonical_[3]; }
  size_t t() const { return canonical_[4]; }

  inline void to_format(NVTE_QKV_Format dst_fmt, size_t *dst_shape) const {
    auto [ndim, order] = dim_order(dst_fmt);
    for (size_t i = 0; i < ndim; ++i) dst_shape[i] = canonical_[order[i]];
  }

 private:
  static inline std::pair<size_t, std::array<int, 4>> dim_order(NVTE_QKV_Format fmt) {
    switch (fmt) {
      case NVTE_QKV_Format::NVTE_BSHD:
        return {4, {0, 2, 1, 3}};  // b s h d
      case NVTE_QKV_Format::NVTE_SBHD:
        return {4, {2, 0, 1, 3}};  // s b h d
      case NVTE_QKV_Format::NVTE_BHSD:
        return {4, {0, 1, 2, 3}};  // b h s d
      case NVTE_QKV_Format::NVTE_THD:
        return {3, {4, 1, 3, -1}};  // t h d
      default:
        return {0, {}};
    }
  }
  size_t canonical_[5] = {};
};

#endif  // __cplusplus

#endif
