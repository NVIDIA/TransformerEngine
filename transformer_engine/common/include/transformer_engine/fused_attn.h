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

#include <cudnn.h>

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

/*! \brief Opaque fused-attention configuration handle. */
typedef void *NVTEFusedAttnConfig;

/*! \enum NVTEFusedAttnConfigAttribute
 *  \brief Attribute types for ``NVTEFusedAttnConfig``.
 *
 *  This enum is used to index the ``FusedAttnConfig`` struct. The order of its fields must match that of
 *  the declaration fields and the ``attr_sizes`` array of that struct. New fields may only be appended
 *  at the end, and existing fields are never to be reordered, removed, or resized.
 */
enum NVTEFusedAttnConfigAttribute {
  // basic attention knobs
  kNVTEFusedAttnConfigIsTraining = 0,
  kNVTEFusedAttnConfigDeterministic,
  kNVTEFusedAttnConfigCudaGraph,
  kNVTEFusedAttnConfigReturnMaxLogit,
  kNVTEFusedAttnConfigAttnMaskType,
  kNVTEFusedAttnConfigBiasType,
  kNVTEFusedAttnConfigWindowSizeLeft,
  kNVTEFusedAttnConfigWindowSizeRight,
  kNVTEFusedAttnConfigBottomRightDiagonal,
  kNVTEFusedAttnConfigSoftmaxType,
  kNVTEFusedAttnConfigScalingMode,
  kNVTEFusedAttnConfigDropout,
  // data types
  kNVTEFusedAttnConfigQKVDtype,
  kNVTEFusedAttnConfigODtype,
  kNVTEFusedAttnConfigDODtype,
  kNVTEFusedAttnConfigDQKVDtype,
  // data and scale layout
  kNVTEFusedAttnConfigQKVLayout,
  kNVTEFusedAttnConfigOFormat,
  kNVTEFusedAttnConfigDOFormat,
  kNVTEFusedAttnConfigDQKVLayout,
  kNVTEFusedAttnConfigQKVScaleInvFormat,
  kNVTEFusedAttnConfigDOScaleInvFormat,
  // attention scaling
  kNVTEFusedAttnConfigAttnScale,
  // tensor dimensions
  kNVTEFusedAttnConfigBatchSize,
  kNVTEFusedAttnConfigNumAttnHeads,
  kNVTEFusedAttnConfigNumGqaGroups,
  kNVTEFusedAttnConfigHeadDimQK,
  kNVTEFusedAttnConfigHeadDimV,
  kNVTEFusedAttnConfigMaxSeqlenQ,
  kNVTEFusedAttnConfigMaxSeqlenKV,
  kNVTEFusedAttnConfigNumTokensQ,
  kNVTEFusedAttnConfigNumTokensKV,
  // paged KV dimensions
  kNVTEFusedAttnConfigNumPagesK,
  kNVTEFusedAttnConfigNumPagesV,
  kNVTEFusedAttnConfigPageSizeK,
  kNVTEFusedAttnConfigPageSizeV,
  kNVTEFusedAttnConfigMaxPagesPerSeqK,
  kNVTEFusedAttnConfigMaxPagesPerSeqV,
  // bias dimensions
  kNVTEFusedAttnConfigBiasBatchSize,
  kNVTEFusedAttnConfigBiasNumHeads,
  kNVTEFusedAttnConfigBiasSeqlenQ,
  kNVTEFusedAttnConfigBiasSeqlenKV,
  kNVTEFusedAttnConfigNumAttributes
};

/*! \brief Create a default-initialized fused-attention configuration.
 *
 *  Categorical fields (layouts, formats, masks, window sizes, scaling mode) are
 *  set to safe NOT_SET / no-op defaults. Numeric and tensor-derived fields,
 *  paged-KV shape, bias broadcast shape, and direction flags default to
 *  zero/false; callers must set the fields relevant to their query.
 *
 *  \return A new configuration handle. Must be destroyed with
 *          ``nvte_destroy_fused_attn_config()``.
 */
NVTEFusedAttnConfig nvte_create_fused_attn_config(void);

/*! \brief Destroy a fused-attention configuration handle. */
void nvte_destroy_fused_attn_config(NVTEFusedAttnConfig config);

/*! \brief Query an attribute in a fused-attention configuration. */
void nvte_get_fused_attn_config_attribute(NVTEFusedAttnConfig config,
                                          NVTEFusedAttnConfigAttribute attr, void *buf,
                                          size_t size_in_bytes, size_t *size_written);

/*! \brief Set an attribute in a fused-attention configuration. */
void nvte_set_fused_attn_config_attribute(NVTEFusedAttnConfig config,
                                          NVTEFusedAttnConfigAttribute attr, const void *buf,
                                          size_t size_in_bytes);

/*! \brief Opaque fused-attention forward-parameter handle. */
typedef void *NVTEFusedAttnFwdParams;

/*! \enum NVTEFusedAttnFwdParamsAttribute
 *  \brief Attribute types for ``NVTEFusedAttnFwdParams``.
 */
enum NVTEFusedAttnFwdParamsAttribute {
  kNVTEFusedAttnFwdParamsQ = 0,
  kNVTEFusedAttnFwdParamsK,
  kNVTEFusedAttnFwdParamsV,
  kNVTEFusedAttnFwdParamsBias,
  kNVTEFusedAttnFwdParamsSoftmaxOffset,
  kNVTEFusedAttnFwdParamsCuSeqlensQ,
  kNVTEFusedAttnFwdParamsCuSeqlensKV,
  kNVTEFusedAttnFwdParamsCuSeqlensQPadded,
  kNVTEFusedAttnFwdParamsCuSeqlensKVPadded,
  kNVTEFusedAttnFwdParamsPageTableK,
  kNVTEFusedAttnFwdParamsPageTableV,
  kNVTEFusedAttnFwdParamsRngState,
  kNVTEFusedAttnFwdParamsS,
  kNVTEFusedAttnFwdParamsO,
  kNVTEFusedAttnFwdParamsAuxCtxTensors,
  kNVTEFusedAttnFwdParamsMaxSeqlenQ,
  kNVTEFusedAttnFwdParamsMaxSeqlenKV,
  kNVTEFusedAttnFwdParamsQKVLayout,
  kNVTEFusedAttnFwdParamsOFormat,
  kNVTEFusedAttnFwdParamsQKVScaleInvFormat,
  kNVTEFusedAttnFwdParamsBiasType,
  kNVTEFusedAttnFwdParamsAttnMaskType,
  kNVTEFusedAttnFwdParamsSoftmaxType,
  kNVTEFusedAttnFwdParamsAttnScale,
  kNVTEFusedAttnFwdParamsDropout,
  kNVTEFusedAttnFwdParamsWindowSizeLeft,
  kNVTEFusedAttnFwdParamsWindowSizeRight,
  kNVTEFusedAttnFwdParamsBottomRightDiagonal,
  kNVTEFusedAttnFwdParamsIsTraining,
  kNVTEFusedAttnFwdParamsReturnMaxLogit,
  kNVTEFusedAttnFwdParamsCudaGraph,
  kNVTEFusedAttnFwdParamsWorkspace,
  kNVTEFusedAttnFwdParamsStream,
  kNVTEFusedAttnFwdParamsNumAttributes
};

/*! \brief Create a default-initialized fused-attention forward-parameter object. */
NVTEFusedAttnFwdParams nvte_create_fused_attn_fwd_params(void);

/*! \brief Destroy a fused-attention forward-parameter handle. */
void nvte_destroy_fused_attn_fwd_params(NVTEFusedAttnFwdParams params);

/*! \brief Query an attribute in a fused-attention forward-parameter object. */
void nvte_get_fused_attn_fwd_params_attribute(NVTEFusedAttnFwdParams params,
                                              NVTEFusedAttnFwdParamsAttribute attr, void *buf,
                                              size_t size_in_bytes, size_t *size_written);

/*! \brief Set an attribute in a fused-attention forward-parameter object. */
void nvte_set_fused_attn_fwd_params_attribute(NVTEFusedAttnFwdParams params,
                                              NVTEFusedAttnFwdParamsAttribute attr, const void *buf,
                                              size_t size_in_bytes);

/*! \brief Opaque fused-attention backward-parameter handle. */
typedef void *NVTEFusedAttnBwdParams;

/*! \enum NVTEFusedAttnBwdParamsAttribute
 *  \brief Attribute types for ``NVTEFusedAttnBwdParams``.
 */
enum NVTEFusedAttnBwdParamsAttribute {
  kNVTEFusedAttnBwdParamsQ = 0,
  kNVTEFusedAttnBwdParamsK,
  kNVTEFusedAttnBwdParamsV,
  kNVTEFusedAttnBwdParamsO,
  kNVTEFusedAttnBwdParamsDO,
  kNVTEFusedAttnBwdParamsS,
  kNVTEFusedAttnBwdParamsDP,
  kNVTEFusedAttnBwdParamsAuxCtxTensors,
  kNVTEFusedAttnBwdParamsDQ,
  kNVTEFusedAttnBwdParamsDK,
  kNVTEFusedAttnBwdParamsDV,
  kNVTEFusedAttnBwdParamsDBias,
  kNVTEFusedAttnBwdParamsDSoftmaxOffset,
  kNVTEFusedAttnBwdParamsCuSeqlensQ,
  kNVTEFusedAttnBwdParamsCuSeqlensKV,
  kNVTEFusedAttnBwdParamsCuSeqlensQPadded,
  kNVTEFusedAttnBwdParamsCuSeqlensKVPadded,
  kNVTEFusedAttnBwdParamsMaxSeqlenQ,
  kNVTEFusedAttnBwdParamsMaxSeqlenKV,
  kNVTEFusedAttnBwdParamsQKVLayout,
  kNVTEFusedAttnBwdParamsOFormat,
  kNVTEFusedAttnBwdParamsDOFormat,
  kNVTEFusedAttnBwdParamsDQKVLayout,
  kNVTEFusedAttnBwdParamsQKVScaleInvFormat,
  kNVTEFusedAttnBwdParamsDOScaleInvFormat,
  kNVTEFusedAttnBwdParamsBiasType,
  kNVTEFusedAttnBwdParamsAttnMaskType,
  kNVTEFusedAttnBwdParamsSoftmaxType,
  kNVTEFusedAttnBwdParamsAttnScale,
  kNVTEFusedAttnBwdParamsDropout,
  kNVTEFusedAttnBwdParamsWindowSizeLeft,
  kNVTEFusedAttnBwdParamsWindowSizeRight,
  kNVTEFusedAttnBwdParamsBottomRightDiagonal,
  kNVTEFusedAttnBwdParamsDeterministic,
  kNVTEFusedAttnBwdParamsCudaGraph,
  kNVTEFusedAttnBwdParamsWorkspace,
  kNVTEFusedAttnBwdParamsStream,
  kNVTEFusedAttnBwdParamsNumAttributes
};

/*! \brief Create a default-initialized fused-attention backward-parameter object. */
NVTEFusedAttnBwdParams nvte_create_fused_attn_bwd_params(void);

/*! \brief Destroy a fused-attention backward-parameter handle. */
void nvte_destroy_fused_attn_bwd_params(NVTEFusedAttnBwdParams params);

/*! \brief Query an attribute in a fused-attention backward-parameter object. */
void nvte_get_fused_attn_bwd_params_attribute(NVTEFusedAttnBwdParams params,
                                              NVTEFusedAttnBwdParamsAttribute attr, void *buf,
                                              size_t size_in_bytes, size_t *size_written);

/*! \brief Set an attribute in a fused-attention backward-parameter object. */
void nvte_set_fused_attn_bwd_params_attribute(NVTEFusedAttnBwdParams params,
                                              NVTEFusedAttnBwdParamsAttribute attr, const void *buf,
                                              size_t size_in_bytes);

/*! \brief Get fused attention backend based on input parameters.
 *
 *  This call exercises cudnn-frontend's support checks by building (and caching)
 *  the cuDNN execution graph for the supported configurations. The configuration
 *  parameters are a superset of those of ``nvte_fused_attn_fwd`` and
 *  ``nvte_fused_attn_bwd`` to maintain a consistent signature between graph
 *  building and runtime calls.
 *
 *  \param[in]     cfg     Attention configuration created with
 *                         ``nvte_create_fused_attn_config()`` (or the C++
 *                         ``FusedAttnConfigWrapper``).
 *  \param[out]    message Empty on success, otherwise a diagnostic string describing
 *                         why the configuration was rejected. The string pointer
 *                         refers to a per-thread buffer owned by the library and
 *                         remains valid only until the next call to
 *                         ``nvte_get_fused_attn_backend_v2`` on the same thread;
 *                         callers that need to retain the message across further
 *                         calls must copy it. Pass NULL to skip diagnostics.
 *
 *  \return Backend able to execute this configuration, or ``NVTE_No_Backend`` if none.
 */
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend_v2(NVTEFusedAttnConfig cfg,
                                                       const char **message);

/*! \brief Get fused attention backend based on input parameters.
 *
 *  \deprecated This function has been deprecated in favor of nvte_get_fused_attn_backend_v2.
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

/*! \brief Compute dot product attention with separate Q, K and V.
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
void nvte_fused_attn_fwd_v2(NVTEFusedAttnFwdParams params);

/*! \brief Compute dot product attention with separate Q, K and V.
 *
 *  \deprecated This function has been deprecated in favor of nvte_fused_attn_fwd_v2.
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

/*! \brief Compute the backward of the dot product attention with separate Q, K and V.
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
void nvte_fused_attn_bwd_v2(NVTEFusedAttnBwdParams params);

/*! \brief Compute the backward of the dot product attention with separate Q, K and V.
 *
 *  \deprecated This function has been deprecated in favor of nvte_fused_attn_bwd_v2.
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

/*!  \brief Reorder THD tensor from sequence order to dual-chunk CP rank order.
 *
 * Uses the padded THD sequence lengths to place each sequence's two CP chunks
 * in the order consumed by each CP rank.
 *
 *  \param[in]     inp           Input THD tensor [total_tokens, ...].
 *  \param[in]     cu_seqlens    Padded cumulative sequence lengths, [batch_size + 1], int32.
 *  \param[out]    out           Output tensor, same shape/dtype as inp.
 *  \param[in]     world_size    Context-parallel size.
 *  \param[in]     total_tokens  Total padded tokens (= inp.shape[0]).
 *  \param[in]     stream        CUDA stream used for this operation.
 */
void nvte_thd_sequence_order_to_cp_rank_order(const NVTETensor &inp, const NVTETensor &cu_seqlens,
                                              NVTETensor out, int world_size, int total_tokens,
                                              cudaStream_t stream);

/*!  \brief Reorder THD tensor from dual-chunk CP rank order to sequence order.
 *
 * Uses the padded THD sequence lengths to restore each sequence's dual-chunk
 * CP entries to sequence order.
 *
 *  \param[in]     inp           Input THD tensor [total_tokens, ...].
 *  \param[in]     cu_seqlens    Padded cumulative sequence lengths, [batch_size + 1], int32.
 *  \param[out]    out           Output tensor, same shape/dtype as inp.
 *  \param[in]     world_size    Context-parallel size.
 *  \param[in]     total_tokens  Total padded tokens (= inp.shape[0]).
 *  \param[in]     stream        CUDA stream used for this operation.
 */
void nvte_thd_cp_rank_order_to_sequence_order(const NVTETensor &inp, const NVTETensor &cu_seqlens,
                                              NVTETensor out, int world_size, int total_tokens,
                                              cudaStream_t stream);

/*!  \brief Copy valid token entries from a per-split THD tensor to a rank-local accumulator.
 *
 * For each dual-chunk CP step/split, copies each sequence's valid range at
 * its padded THD token offsets and leaves padded entries untouched.
 *
 *  \param[in]     inp                 Per-split THD source tensor [total_tokens, ...].
 *  \param[in]     cu_seqlens_padded   Padded cumulative sequence lengths, [batch_size + 1], int32.
 *  \param[in]     cu_seqlens          Valid cumulative sequence lengths, [batch_size + 1], int32.
 *  \param[in,out] out                 Rank-local accumulator, same shape/dtype as inp.
 *  \param[in]     total_tokens        Total padded tokens (= inp.shape[0]).
 *  \param[in]     stream              CUDA stream used for this operation.
 */
void nvte_thd_copy_valid_tokens_from_per_split_to_rank_local(const NVTETensor &inp,
                                                             const NVTETensor &cu_seqlens_padded,
                                                             const NVTETensor &cu_seqlens,
                                                             NVTETensor out, int total_tokens,
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

/*! \class FusedAttnConfigWrapper
 *  \brief C++ helper for constructing an ``NVTEFusedAttnConfig``.
 *
 *  Owns an opaque ``NVTEFusedAttnConfig`` handle created via
 *  ``nvte_create_fused_attn_config()``. Provides typed, chainable setters for
 *  every field.
 */
class FusedAttnConfigWrapper {
 public:
  FusedAttnConfigWrapper() : cfg_{nvte_create_fused_attn_config()} {}

  FusedAttnConfigWrapper(const FusedAttnConfigWrapper &) = delete;
  FusedAttnConfigWrapper &operator=(const FusedAttnConfigWrapper &) = delete;

  FusedAttnConfigWrapper(FusedAttnConfigWrapper &&other) noexcept : cfg_{other.cfg_} {
    other.cfg_ = nullptr;
  }

  FusedAttnConfigWrapper &operator=(FusedAttnConfigWrapper &&other) noexcept {
    if (this != &other) {
      nvte_destroy_fused_attn_config(cfg_);
      cfg_ = other.cfg_;
      other.cfg_ = nullptr;
    }
    return *this;
  }

  ~FusedAttnConfigWrapper() {
    if (cfg_ != nullptr) {
      nvte_destroy_fused_attn_config(cfg_);
    }
  }

  operator NVTEFusedAttnConfig() const noexcept { return cfg_; }
  NVTEFusedAttnConfig get() const noexcept { return cfg_; }

  FusedAttnConfigWrapper &set_is_training(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigIsTraining, &u8_val,
                                         sizeof(u8_val));
    return *this;
  }
  FusedAttnConfigWrapper &set_deterministic(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigDeterministic, &u8_val,
                                         sizeof(u8_val));
    return *this;
  }
  FusedAttnConfigWrapper &set_cuda_graph(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigCudaGraph, &u8_val,
                                         sizeof(u8_val));
    return *this;
  }
  FusedAttnConfigWrapper &set_return_max_logit(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigReturnMaxLogit, &u8_val,
                                         sizeof(u8_val));
    return *this;
  }
  FusedAttnConfigWrapper &set_qkv_layout(NVTE_QKV_Layout val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigQKVLayout, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_o_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigOFormat, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_do_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigDOFormat, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_dqkv_layout(NVTE_QKV_Layout val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigDQKVLayout, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_qkv_scale_inv_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigQKVScaleInvFormat, &val,
                                         sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_do_scale_inv_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigDOScaleInvFormat, &val,
                                         sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_bias_type(NVTE_Bias_Type val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigBiasType, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_attn_mask_type(NVTE_Mask_Type val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigAttnMaskType, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_softmax_type(NVTE_Softmax_Type val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigSoftmaxType, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_scaling_mode(NVTEScalingMode val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigScalingMode, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_attn_scale(float val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigAttnScale, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_dropout(float val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigDropout, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_max_seqlen_q(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigMaxSeqlenQ, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_max_seqlen_kv(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigMaxSeqlenKV, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_window_size_left(int64_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigWindowSizeLeft, &val,
                                         sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_window_size_right(int64_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigWindowSizeRight, &val,
                                         sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_bottom_right_diagonal(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigBottomRightDiagonal, &u8_val,
                                         sizeof(u8_val));
    return *this;
  }
  FusedAttnConfigWrapper &set_qkv_dtype(NVTEDType val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigQKVDtype, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_o_dtype(NVTEDType val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigODtype, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_do_dtype(NVTEDType val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigDODtype, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_dqkv_dtype(NVTEDType val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigDQKVDtype, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_batch_size(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigBatchSize, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_num_attn_heads(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigNumAttnHeads, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_num_gqa_groups(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigNumGqaGroups, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_head_dim_qk(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigHeadDimQK, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_head_dim_v(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigHeadDimV, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_num_pages_k(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigNumPagesK, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_num_pages_v(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigNumPagesV, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_page_size_k(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigPageSizeK, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_page_size_v(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigPageSizeV, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_max_pages_per_seq_k(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigMaxPagesPerSeqK, &val,
                                         sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_max_pages_per_seq_v(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigMaxPagesPerSeqV, &val,
                                         sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_bias_batch_size(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigBiasBatchSize, &val,
                                         sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_bias_num_heads(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigBiasNumHeads, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_bias_seqlen_q(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigBiasSeqlenQ, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_bias_seqlen_kv(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigBiasSeqlenKV, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_num_tokens_q(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigNumTokensQ, &val, sizeof(val));
    return *this;
  }
  FusedAttnConfigWrapper &set_num_tokens_kv(size_t val) noexcept {
    nvte_set_fused_attn_config_attribute(cfg_, kNVTEFusedAttnConfigNumTokensKV, &val, sizeof(val));
    return *this;
  }

 private:
  NVTEFusedAttnConfig cfg_ = nullptr;
};

/*! \class FusedAttnFwdParamsWrapper
 *  \brief C++ helper for constructing an ``NVTEFusedAttnFwdParams``.
 */
class FusedAttnFwdParamsWrapper {
 public:
  FusedAttnFwdParamsWrapper() : params_{nvte_create_fused_attn_fwd_params()} {}
  FusedAttnFwdParamsWrapper(const FusedAttnFwdParamsWrapper &) = delete;
  FusedAttnFwdParamsWrapper &operator=(const FusedAttnFwdParamsWrapper &) = delete;
  FusedAttnFwdParamsWrapper(FusedAttnFwdParamsWrapper &&other) noexcept : params_{other.params_} {
    other.params_ = nullptr;
  }
  FusedAttnFwdParamsWrapper &operator=(FusedAttnFwdParamsWrapper &&other) noexcept {
    if (this != &other) {
      nvte_destroy_fused_attn_fwd_params(params_);
      params_ = other.params_;
      other.params_ = nullptr;
    }
    return *this;
  }
  ~FusedAttnFwdParamsWrapper() {
    if (params_ != nullptr) {
      nvte_destroy_fused_attn_fwd_params(params_);
    }
  }
  operator NVTEFusedAttnFwdParams() const noexcept { return params_; }
  NVTEFusedAttnFwdParams get() const noexcept { return params_; }
  FusedAttnFwdParamsWrapper &set_Q(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsQ, &val, sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_K(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsK, &val, sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_V(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsV, &val, sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_Bias(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsBias, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_SoftmaxOffset(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsSoftmaxOffset, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_cu_seqlens_q(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsCuSeqlensQ, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_cu_seqlens_kv(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsCuSeqlensKV, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_cu_seqlens_q_padded(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsCuSeqlensQPadded, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_cu_seqlens_kv_padded(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsCuSeqlensKVPadded,
                                             &val, sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_page_table_k(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsPageTableK, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_page_table_v(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsPageTableV, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_rng_state(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsRngState, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_S(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsS, &val, sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_O(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsO, &val, sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_Aux_CTX_Tensors(NVTETensorPack *val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsAuxCtxTensors, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_max_seqlen_q(size_t val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsMaxSeqlenQ, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_max_seqlen_kv(size_t val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsMaxSeqlenKV, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_qkv_layout(NVTE_QKV_Layout val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsQKVLayout, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_o_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsOFormat, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_qkv_scale_inv_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsQKVScaleInvFormat,
                                             &val, sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_bias_type(NVTE_Bias_Type val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsBiasType, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_attn_mask_type(NVTE_Mask_Type val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsAttnMaskType, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_softmax_type(NVTE_Softmax_Type val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsSoftmaxType, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_attn_scale(float val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsAttnScale, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_dropout(float val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsDropout, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_window_size_left(int64_t val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsWindowSizeLeft, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_window_size_right(int64_t val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsWindowSizeRight, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_bottom_right_diagonal(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsBottomRightDiagonal,
                                             &u8_val, sizeof(u8_val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_is_training(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsIsTraining, &u8_val,
                                             sizeof(u8_val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_return_max_logit(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsReturnMaxLogit,
                                             &u8_val, sizeof(u8_val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_cuda_graph(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsCudaGraph, &u8_val,
                                             sizeof(u8_val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_workspace(NVTETensor val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsWorkspace, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnFwdParamsWrapper &set_stream(cudaStream_t val) noexcept {
    nvte_set_fused_attn_fwd_params_attribute(params_, kNVTEFusedAttnFwdParamsStream, &val,
                                             sizeof(val));
    return *this;
  }

 private:
  NVTEFusedAttnFwdParams params_ = nullptr;
};

/*! \class FusedAttnBwdParamsWrapper
 *  \brief C++ helper for constructing an ``NVTEFusedAttnBwdParams``.
 */
class FusedAttnBwdParamsWrapper {
 public:
  FusedAttnBwdParamsWrapper() : params_{nvte_create_fused_attn_bwd_params()} {}
  FusedAttnBwdParamsWrapper(const FusedAttnBwdParamsWrapper &) = delete;
  FusedAttnBwdParamsWrapper &operator=(const FusedAttnBwdParamsWrapper &) = delete;
  FusedAttnBwdParamsWrapper(FusedAttnBwdParamsWrapper &&other) noexcept : params_{other.params_} {
    other.params_ = nullptr;
  }
  FusedAttnBwdParamsWrapper &operator=(FusedAttnBwdParamsWrapper &&other) noexcept {
    if (this != &other) {
      nvte_destroy_fused_attn_bwd_params(params_);
      params_ = other.params_;
      other.params_ = nullptr;
    }
    return *this;
  }
  ~FusedAttnBwdParamsWrapper() {
    if (params_ != nullptr) {
      nvte_destroy_fused_attn_bwd_params(params_);
    }
  }
  operator NVTEFusedAttnBwdParams() const noexcept { return params_; }
  NVTEFusedAttnBwdParams get() const noexcept { return params_; }
  FusedAttnBwdParamsWrapper &set_Q(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsQ, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_K(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsK, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_V(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsV, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_O(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsO, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dO(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDO, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_S(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsS, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dP(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDP, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_Aux_CTX_Tensors(const NVTETensorPack *val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsAuxCtxTensors, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dQ(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDQ, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dK(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDK, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dV(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDV, &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dBias(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDBias, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dSoftmaxOffset(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDSoftmaxOffset, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_cu_seqlens_q(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsCuSeqlensQ, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_cu_seqlens_kv(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsCuSeqlensKV, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_cu_seqlens_q_padded(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsCuSeqlensQPadded, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_cu_seqlens_kv_padded(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsCuSeqlensKVPadded,
                                             &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_max_seqlen_q(size_t val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsMaxSeqlenQ, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_max_seqlen_kv(size_t val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsMaxSeqlenKV, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_qkv_layout(NVTE_QKV_Layout val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsQKVLayout, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_o_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsOFormat, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_do_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDOFormat, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dqkv_layout(NVTE_QKV_Layout val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDQKVLayout, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_qkv_scale_inv_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsQKVScaleInvFormat,
                                             &val, sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_do_scale_inv_format(NVTE_QKV_Format val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDOScaleInvFormat, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_bias_type(NVTE_Bias_Type val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsBiasType, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_attn_mask_type(NVTE_Mask_Type val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsAttnMaskType, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_softmax_type(NVTE_Softmax_Type val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsSoftmaxType, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_attn_scale(float val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsAttnScale, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_dropout(float val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDropout, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_window_size_left(int64_t val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsWindowSizeLeft, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_window_size_right(int64_t val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsWindowSizeRight, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_bottom_right_diagonal(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsBottomRightDiagonal,
                                             &u8_val, sizeof(u8_val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_deterministic(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsDeterministic, &u8_val,
                                             sizeof(u8_val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_cuda_graph(bool val) noexcept {
    const uint8_t u8_val = static_cast<uint8_t>(val);
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsCudaGraph, &u8_val,
                                             sizeof(u8_val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_workspace(NVTETensor val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsWorkspace, &val,
                                             sizeof(val));
    return *this;
  }
  FusedAttnBwdParamsWrapper &set_stream(cudaStream_t val) noexcept {
    nvte_set_fused_attn_bwd_params_attribute(params_, kNVTEFusedAttnBwdParamsStream, &val,
                                             sizeof(val));
    return *this;
  }

 private:
  NVTEFusedAttnBwdParams params_ = nullptr;
};
#endif  // __cplusplus

#endif
