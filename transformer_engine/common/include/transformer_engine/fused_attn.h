/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file fused_attn.h
 *  \brief Enums and functions for fused attention.
 */

#ifndef TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_
#define TRANSFORMER_ENGINE_FUSED_ATTN_FP8_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \enum NVTE_QKV_Layout
 *  \brief Memory layouts of QKV tensors.
 *   `S`, `B`, `H`, `D`, and `T` stand for sequence length, batch size, number of heads,
 *   head size, and the total number of sequences in a batch, i.e. `t = sum(s_i) for i = 0...b-1`.
 *   `SBHD` and `BSHD`-based layouts are used when sequences in a batch are of equal length
 *   or padded to the same length, and `THD`-based layouts are used when sequences have
 *   different lengths in a batch.
 */
enum NVTE_QKV_Layout {
  NVTE_SB3HD = 0,          /*!< SB3HD layout */
  NVTE_SBH3D = 1,          /*!< SBH3D layout */
  NVTE_SBHD_SB2HD = 2,     /*!< SBHD_SB2HD layout */
  NVTE_SBHD_SBH2D = 3,     /*!< SBHD_SBH2D layout */
  NVTE_SBHD_SBHD_SBHD = 4, /*!< SBHD_SBHD_SBHD layout */
  NVTE_BS3HD = 5,          /*!< BS3HD layout */
  NVTE_BSH3D = 6,          /*!< BSH3D layout */
  NVTE_BSHD_BS2HD = 7,     /*!< BSHD_BS2HD layout */
  NVTE_BSHD_BSH2D = 8,     /*!< BSHD_BSH2D layout */
  NVTE_BSHD_BSHD_BSHD = 9, /*!< BSHD_BSHD_BSHD layout */
  NVTE_T3HD = 10,          /*!< T3HD layout */
  NVTE_TH3D = 11,          /*!< TH3D layout */
  NVTE_THD_T2HD = 12,      /*!< THD_T2HD layout */
  NVTE_THD_TH2D = 13,      /*!< THD_TH2D layout */
  NVTE_THD_THD_THD = 14,   /*!< THD_THD_THD layout */
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
};

/*! \enum NVTE_QKV_Format
 *  \brief QKV formats
 */
enum NVTE_QKV_Format {
  /*! SBHD QKV format, i.e. SB3HD, SBH3D, SBHD_SB2HD, SBHD_SBH2D, SBHD_SBHD_SBHD */
  NVTE_SBHD = 0,
  /*! BSHD QKV format, i.e. BS3HD, BSH3D, BSHD_BS2HD, BSHD_BSH2D, BSHD_BSHD_BSHD */
  NVTE_BSHD = 1,
  /*! THD QKV format, i.e. T3HD, TH3D, THD_T2HD, THD_TH2D, THD_THD_THD */
  NVTE_THD = 2,
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
  /*! Causal attention mask */
  NVTE_CAUSAL_MASK = 2,
  /*! Padding and causal attention mask */
  NVTE_PADDING_CAUSAL_MASK = 3,
};

/*! \enum NVTE_Fused_Attn_Backend
 *  \brief Fused attention backends
 */
enum NVTE_Fused_Attn_Backend {
  /*! No supported backend */
  NVTE_No_Backend = -1,
  /*! cuDNN-based FP16/BF16 fused attention for <= 512 sequence length */
  NVTE_F16_max512_seqlen = 0,
  /*! cuDNN-based FP16/BF16 fused attention for any sequence length */
  NVTE_F16_arbitrary_seqlen = 1,
  /*! cuDNN-based FP8 fused attention for <= 512 sequence length */
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

/*! \brief Get fused attention backend based on input parameters.
 *
 *  \param[in]     q_dtype          The data type of Tensor Q.
 *  \param[in]     kv_dtype         The data type of Tensors K, V.
 *  \param[in]     qkv_layout       The layout of Tensors Q, K, V.
 *  \param[in]     bias_type        The attention bias type.
 *  \param[in]     attn_mask_type   The attention mask type.
 *  \param[in]     dropout          The dropout probability.
 *  \param[in]     num_attn_heads   The number of heads in Q.
 *  \param[in]     num_gqa_groups   The number of heads in K, V.
 *  \param[in]     max_seqlen_q     The sequence length of Q.
 *  \param[in]     max_seqlen_kv    The sequence length of K, V.
 *  \param[in]     head_dim         The head dimension of Q, K, V.
 */
NVTE_Fused_Attn_Backend nvte_get_fused_attn_backend(
    NVTEDType q_dtype, NVTEDType kv_dtype, NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
    NVTE_Mask_Type attn_mask_type, float dropout, size_t num_attn_heads, size_t num_gqa_groups,
    size_t max_seqlen_q, size_t max_seqlen_kv, size_t head_dim);

/*! \brief Compute dot product attention with packed QKV input.
 *
 * Computes:
 *  - P = Q * Transpose(K) + Bias
 *  - S = ScaleMaskSoftmax(P)
 *  - D = Dropout(S)
 *  - O = D * Transpose(V)
 *
 * Support Matrix:
   \verbatim
   | backend | precision |        qkv layout       |           bias           |                 mask                  | dropout |  sequence length  | head_dim         |
   |   0     | FP16/BF16 |       BS3HD,SB3HD       |   NO/POST_SCALE_BIAS     | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   | <= 512, % 64 == 0 |    64            |
   |   1     | FP16/BF16 | BS3HD,SB3HD,BSH3D,SBH3D | NO/POST_SCALE_BIAS/ALIBI | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   |  > 512, % 64 == 0 | <= 128, % 8 == 0 |
   |   2     |   FP8     |          T3HD           |          NO_BIAS         |               PADDING_MASK            |   Yes   | <= 512, % 64 == 0 |    64            |
   \endverbatim
 *
 * Notes:
 *
 * Tensor `cu_seqlens_padded` helps identify the correct offsets of different sequences
 * in tensors Q, K, V and O.
 * When the QKV format (`nvte_get_qkv_format(qkv_layout)`) is `bshd` or `sbhd`,
 * the offset tensor is not used in the attention calculation and can be set to empty `NVTETensor`.
 * When the QKV format is `thd`, this tensor should follow the following rules.
 * When there is no padding between sequences, the offset tensor should be equal to `cu_seqlens`,
 * When there is padding between sequences, users are responsible to adjust the offsets as needed.
 * For example, a tensor of 4 sequences `[a, PAD, b, b, c, PAD, PAD, d, d]` should have
 * `cu_seqlens = [0, 1, 3, 4, 6]` and `cu_seqlens_padded= [0, 2, 4, 7, 9]`.
 *
 *  \param[in]     QKV                      The QKV tensor in packed format, H3D or 3HD.
 *  \param[in]     Bias                     The Bias tensor.
 *  \param[in,out] S                        The S tensor.
 *  \param[out]    O                        The output O tensor.
 *  \param[out]    Aux_CTX_Tensors          Auxiliary output tensors when training,
 *                                          e.g. M, ZInv, rng_state.
 *  \param[in]     cu_seqlens               Cumulative sequence lengths, [batch_size + 1].
 *  \param[in]     cu_seqlens_padded        Cumulative sequence offsets for QKV, [batch_size + 1].
 *  \param[in]     rng_state                Seed and offset of CUDA random number generator.
 *  \param[in]     max_seqlen               Max sequence length used for computing,
 *                                          it may be >= max(seqlen_i) for i=0,...batch_size-1.
 *  \param[in]     is_training              Whether this is in training mode or inference.
 *  \param[in]     attn_scale               Scaling factor for Q * K.T.
 *  \param[in]     dropout                  Dropout probability.
 *  \param[in]     qkv_layout               QKV tensor's layout.
 *  \param[in]     bias_type                Bias type.
 *  \param[in]     attn_mask_type           Attention mask type.
 *  \param[in]     workspace                Workspace tensor.
 *  \param[in]     stream                   CUDA stream used for this operation.
 */
void nvte_fused_attn_fwd_qkvpacked(const NVTETensor QKV, const NVTETensor Bias, NVTETensor S,
                                   NVTETensor O, NVTETensorPack* Aux_CTX_Tensors,
                                   const NVTETensor cu_seqlens, const NVTETensor cu_seqlens_padded,
                                   const NVTETensor rng_state, size_t max_seqlen, bool is_training,
                                   float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                   NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                   NVTETensor workspace, cudaStream_t stream);

/*! \brief Compute the backward of the dot product attention with packed QKV input.
 *
 * Support Matrix:
   \verbatim
   | backend | precision |        qkv layout       |           bias           |                 mask                  | dropout |  sequence length  | head_dim         |
   |   0     | FP16/BF16 |       BS3HD,SB3HD       |   NO/POST_SCALE_BIAS     | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   | <= 512, % 64 == 0 |    64            |
   |   1     | FP16/BF16 | BS3HD,SB3HD,BSH3D,SBH3D | NO/POST_SCALE_BIAS/ALIBI | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   |  > 512, % 64 == 0 | <= 128, % 8 == 0 |
   |   2     |   FP8     |          T3HD           |          NO_BIAS         |               PADDING_MASK            |   Yes   | <= 512, % 64 == 0 |    64            |
   \endverbatim
 *
 * Notes:
 *
 * Tensor `cu_seqlens_padded` helps identify the correct offsets of different sequences
 * in tensors Q, K, V and O.
 * When the QKV format (`nvte_get_qkv_format(qkv_layout)`) is `bshd` or `sbhd`,
 * the offset tensor is not used in the attention calculation and can be set to empty `NVTETensor`.
 * When the QKV format is `thd`, this tensor should follow the following rules.
 * When there is no padding between sequences, the offset tensor should be equal to `cu_seqlens`,
 * When there is padding between sequences, users are responsible to adjust the offsets as needed.
 * For example, a tensor of 4 sequences `[a, PAD, b, b, c, PAD, PAD, d, d]` should have
 * `cu_seqlens = [0, 1, 3, 4, 6]` and `cu_seqlens_padded= [0, 2, 4, 7, 9]`.
 *
 *  \param[in]     QKV                      The QKV tensor in packed format, H3D or 3HD.
 *  \param[in]     O                        The O tensor from forward.
 *  \param[in]     dO                       The gradient of the O tensor.
 *  \param[in]     S                        The S tensor.
 *  \param[in,out] dP                       The gradient of the P tensor.
 *  \param[in]     Aux_CTX_Tensors          Auxiliary tensors from context when in training mode,
 *                                          e.g. M, ZInv, rng_state.
 *  \param[out]    dQKV                     The gradient of the QKV tensor.
 *  \param[out]    dBias                    The gradient of the Bias tensor.
 *  \param[in]     cu_seqlens               Cumulative sequence lengths, [batch_size + 1].
 *  \param[in]     cu_seqlens_padded        Cumulative sequence offsets for QKV, [batch_size + 1].
 *  \param[in]     max_seqlen               Max sequence length used for computing,
 *                                          it may be >= max(seqlen_i) for i=0,...batch_size-1.
 *  \param[in]     attn_scale               Scaling factor for Q * K.T.
 *  \param[in]     dropout                  Dropout probability.
 *  \param[in]     qkv_layout               QKV tensor's layout.
 *  \param[in]     bias_type                Bias type.
 *  \param[in]     attn_mask_type           Attention mask type.
 *  \param[in]     workspace                Workspace tensor.
 *  \param[in]     stream                   CUDA stream used for this operation.
 */
void nvte_fused_attn_bwd_qkvpacked(const NVTETensor QKV, const NVTETensor O, const NVTETensor dO,
                                   const NVTETensor S, NVTETensor dP,
                                   const NVTETensorPack* Aux_CTX_Tensors, NVTETensor dQKV,
                                   NVTETensor dBias, const NVTETensor cu_seqlens,
                                   const NVTETensor cu_seqlens_padded, size_t max_seqlen,
                                   float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                   NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                   NVTETensor workspace, cudaStream_t stream);

/*! \brief Compute dot product attention with packed KV input.
 *
 * Computes:
 *  - P = Q * Transpose(K) + Bias
 *  - S = ScaleMaskSoftmax(P)
 *  - D = Dropout(S)
 *  - O = D * Transpose(V)
 *
 * Support Matrix:
   \verbatim
   | backend | precision |                 qkv layout                  |           bias           |                 mask                  | dropout |  sequence length  | head_dim         |
   |   0     | FP16/BF16 |            BSHD_BS2HD,SBHD_SB2HD            |   NO/POST_SCALE_BIAS     | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   | <= 512, % 64 == 0 |    64            |
   |   1     | FP16/BF16 | BSHD_BS2HD,BSHD_BSH2D,SBHD_SB2HD,SBHD_SBH2D | NO/POST_SCALE_BIAS/ALIBI | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   |  > 512, % 64 == 0 | <= 128, % 8 == 0 |
   \endverbatim
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
 *  \param[in]     Q                         The Q tensor, in HD layouts.
 *  \param[in]     KV                        The KV tensor, in 2HD or H2D layouts.
 *  \param[in]     Bias                      The Bias tensor.
 *  \param[in,out] S                         The S tensor.
 *  \param[out]    O                         The output O tensor.
 *  \param[out]    Aux_CTX_Tensors           Auxiliary output tensors when training,
 *                                           e.g. M, ZInv, rng_state.
 *  \param[in]     cu_seqlens_q              Cumulative sequence lengths for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv             Cumulative sequence lengths for KV, [batch_size + 1].
 *  \param[in]     cu_seqlens_q_padded       Cumulative sequence offsets for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv_padded      Cumulative sequence offsets for KV, [batch_size + 1].
 *  \param[in]     rng_state                 Seed and offset of CUDA random number generator.
 *  \param[in]     max_seqlen_q              Max sequence length used for computing for Q.
 *                                           it may be >= max(seqlen_q_i) for i=0,...batch_size-1.
 *  \param[in]     max_seqlen_kv             Max sequence length used for computing for KV.
 *                                           it may be >= max(seqlen_kv_i) for i=0,...batch_size-1.
 *  \param[in]     is_training               Whether this is in training mode or inference.
 *  \param[in]     attn_scale                Scaling factor for Q * K.T.
 *  \param[in]     dropout                   Dropout probability.
 *  \param[in]     qkv_layout                QKV tensor's layout.
 *  \param[in]     bias_type                 Bias type.
 *  \param[in]     attn_mask_type            Attention mask type.
 *  \param[in]     workspace                 Workspace tensor.
 *  \param[in]     stream                    CUDA stream used for this operation.
 */
void nvte_fused_attn_fwd_kvpacked(const NVTETensor Q, const NVTETensor KV, const NVTETensor Bias,
                                  NVTETensor S, NVTETensor O, NVTETensorPack* Aux_CTX_Tensors,
                                  const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
                                  const NVTETensor cu_seqlens_q_padded,
                                  const NVTETensor cu_seqlens_kv_padded, const NVTETensor rng_state,
                                  size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                                  float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                                  NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                                  NVTETensor workspace, cudaStream_t stream);

/*! \brief Compute the backward of the dot product attention with packed KV input.
 *
 * Support Matrix:
   \verbatim
   | backend | precision |                 qkv layout                  |           bias           |                 mask                  | dropout |  sequence length  | head_dim         |
   |   0     | FP16/BF16 |            BSHD_BS2HD,SBHD_SB2HD            |   NO/POST_SCALE_BIAS     | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   | <= 512, % 64 == 0 |    64            |
   |   1     | FP16/BF16 | BSHD_BS2HD,BSHD_BSH2D,SBHD_SB2HD,SBHD_SBH2D | NO/POST_SCALE_BIAS/ALIBI | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   |  > 512, % 64 == 0 | <= 128, % 8 == 0 |
   \endverbatim
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
 *  \param[in]     Q                         The Q tensor, in HD layouts.
 *  \param[in]     KV                        The KV tensor, in H2D or 2HD layouts.
 *  \param[in]     O                         The O tensor from forward.
 *  \param[in]     dO                        The gradient of the O tensor.
 *  \param[in]     S                         The S tensor.
 *  \param[in,out] dP                        The gradient of the P tensor.
 *  \param[in]     Aux_CTX_Tensors           Auxiliary tensors from context when in training mode,
 *                                           e.g. M, ZInv, rng_state.
 *  \param[out]    dQ                        The gradient of the Q tensor.
 *  \param[out]    dKV                       The gradient of the KV tensor.
 *  \param[out]    dBias                     The gradient of the Bias tensor.
 *  \param[in]     cu_seqlens_q              Cumulative sequence lengths for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv             Cumulative sequence lengths for KV, [batch_size + 1].
 *  \param[in]     cu_seqlens_q_padded       Cumulative sequence offsets for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv_padded      Cumulative sequence offsets for KV, [batch_size + 1].
 *  \param[in]     max_seqlen_q              Max sequence length used for computing for Q.
 *                                           it may be >= max(seqlen_q_i) for i=0,...batch_size-1.
 *  \param[in]     max_seqlen_kv             Max sequence length used for computing for KV.
 *                                           it may be >= max(seqlen_kv_i) for i=0,...batch_size-1.
 *  \param[in]     attn_scale                Scaling factor for Q * K.T.
 *  \param[in]     dropout                   Dropout probability.
 *  \param[in]     qkv_layout                QKV tensor's layout.
 *  \param[in]     bias_type                 Bias type.
 *  \param[in]     attn_mask_type            Attention mask type.
 *  \param[in]     workspace                 Workspace tensor.
 *  \param[in]     stream                    CUDA stream used for this operation.
 */
void nvte_fused_attn_bwd_kvpacked(
    const NVTETensor Q, const NVTETensor KV, const NVTETensor O, const NVTETensor dO,
    const NVTETensor S, NVTETensor dP, const NVTETensorPack* Aux_CTX_Tensors, NVTETensor dQ,
    NVTETensor dKV, NVTETensor dBias, const NVTETensor cu_seqlens_q, const NVTETensor cu_seqlens_kv,
    const NVTETensor cu_seqlens_q_padded, const NVTETensor cu_seqlens_kv_padded,
    size_t max_seqlen_q, size_t max_seqlen_kv, float attn_scale, float dropout,
    NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
    NVTETensor workspace, cudaStream_t stream);

/*! \brief Compute dot product attention with separate Q, K and V.
 *
 * Computes:
 *  - P = Q * Transpose(K) + Bias
 *  - S = ScaleMaskSoftmax(P)
 *  - D = Dropout(S)
 *  - O = D * Transpose(V)
 *
 * Support Matrix:
   \verbatim
   | backend | precision |                qkv layout                   |           bias           |                 mask                  | dropout |  sequence length  | head_dim         |
   |   0     | FP16/BF16 |     BS3HD,SB3HD,BSHD_BS2HD,SBHD_SB2HD       |   NO/POST_SCALE_BIAS     | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   | <= 512, % 64 == 0 |    64            |
   |   1     | FP16/BF16 |          BS3HD,SB3HD,BSH3D,SBH3D            | NO/POST_SCALE_BIAS/ALIBI | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   |  > 512, % 64 == 0 | <= 128, % 8 == 0 |
   |         |           | BSHD_BS2HD,BSHD_BSH2D,SBHD_SB2HD,SBHD_SBH2D |                          |                                       |         |                   |                  |
   |         |           |       BSHD_BSHD_BSHD,SBHD_SBHD_SBHD         |                          |                                       |         |                   |                  |
   |   2     |   FP8     |                 T3HD                        |          NO_BIAS         |               PADDING_MASK            |   Yes   | <= 512, % 64 == 0 |    64            |
   \endverbatim
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
 *  \param[in,out] S                         The S tensor.
 *  \param[out]    O                         The output O tensor.
 *  \param[out]    Aux_CTX_Tensors           Auxiliary output tensors when training,
 *                                           e.g. M, ZInv, rng_state.
 *  \param[in]     cu_seqlens_q              Cumulative sequence lengths for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv             Cumulative sequence lengths for K and V, [batch_size + 1].
 *  \param[in]     cu_seqlens_q_padded       Cumulative sequence offsets for Q, [batch_size + 1].
 *  \param[in]     cu_seqlens_kv_padded      Cumulative sequence offsets for KV, [batch_size + 1].
 *  \param[in]     rng_state                 Seed and offset of CUDA random number generator.
 *  \param[in]     max_seqlen_q              Max sequence length used for computing for Q.
 *                                           it may be >= max(seqlen_q_i) for i=0,...batch_size-1.
 *  \param[in]     max_seqlen_kv             Max sequence length used for computing for K and V.
 *                                           it may be >= max(seqlen_kv_i) for i=0,...batch_size-1.
 *  \param[in]     is_training               Whether this is in training mode or inference.
 *  \param[in]     attn_scale                Scaling factor for Q * K.T.
 *  \param[in]     dropout                   Dropout probability.
 *  \param[in]     qkv_layout                QKV tensors' layout.
 *  \param[in]     bias_type                 Bias type.
 *  \param[in]     attn_mask_type            Attention mask type.
 *  \param[in]     workspace                 Workspace tensor.
 *  \param[in]     stream                    CUDA stream used for this operation.
 */
void nvte_fused_attn_fwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor Bias, NVTETensor S, NVTETensor O,
                         NVTETensorPack* Aux_CTX_Tensors, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, const NVTETensor rng_state,
                         size_t max_seqlen_q, size_t max_seqlen_kv, bool is_training,
                         float attn_scale, float dropout, NVTE_QKV_Layout qkv_layout,
                         NVTE_Bias_Type bias_type, NVTE_Mask_Type attn_mask_type,
                         NVTETensor workspace, cudaStream_t stream);

/*! \brief Compute the backward of the dot product attention with separate Q, K and V.
 *
 * Support Matrix:
   \verbatim
   | backend | precision |                qkv layout                   |           bias           |                 mask                  | dropout |  sequence length  | head_dim         |
   |   0     | FP16/BF16 |     BS3HD,SB3HD,BSHD_BS2HD,SBHD_SB2HD       |   NO/POST_SCALE_BIAS     | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   | <= 512, % 64 == 0 |    64            |
   |   1     | FP16/BF16 |          BS3HD,SB3HD,BSH3D,SBH3D            | NO/POST_SCALE_BIAS/ALIBI | NO/PADDING/CAUSAL/PADDING_CAUSAL_MASK |   Yes   |  > 512, % 64 == 0 | <= 128, % 8 == 0 |
   |         |           | BSHD_BS2HD,BSHD_BSH2D,SBHD_SB2HD,SBHD_SBH2D |                          |                                       |         |                   |                  |
   |         |           |       BSHD_BSHD_BSHD,SBHD_SBHD_SBHD         |                          |                                       |         |                   |                  |
   |   2     |   FP8     |                 T3HD                        |          NO_BIAS         |               PADDING_MASK            |   Yes   | <= 512, % 64 == 0 |    64            |
   \endverbatim
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
 *                                           e.g. M, ZInv, rng_state.
 *  \param[out]    dQ                        The gradient of the Q tensor.
 *  \param[out]    dK                        The gradient of the K tensor.
 *  \param[out]    dV                        The gradient of the V tensor.
 *  \param[out]    dBias                     The gradient of the Bias tensor.
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
 *  \param[in]     bias_type                 Bias type.
 *  \param[in]     attn_mask_type            Attention mask type.
 *  \param[in]     workspace                 Workspace tensor.
 *  \param[in]     stream                    CUDA stream used for this operation.
 */
void nvte_fused_attn_bwd(const NVTETensor Q, const NVTETensor K, const NVTETensor V,
                         const NVTETensor O, const NVTETensor dO, const NVTETensor S, NVTETensor dP,
                         const NVTETensorPack* Aux_CTX_Tensors, NVTETensor dQ, NVTETensor dK,
                         NVTETensor dV, NVTETensor dBias, const NVTETensor cu_seqlens_q,
                         const NVTETensor cu_seqlens_kv, const NVTETensor cu_seqlens_q_padded,
                         const NVTETensor cu_seqlens_kv_padded, size_t max_seqlen_q,
                         size_t max_seqlen_kv, float attn_scale, float dropout,
                         NVTE_QKV_Layout qkv_layout, NVTE_Bias_Type bias_type,
                         NVTE_Mask_Type attn_mask_type, NVTETensor workspace, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
