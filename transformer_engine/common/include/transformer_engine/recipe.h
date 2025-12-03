/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file recipe.h
 *  \brief Functions handling FP8 recipes.
 */

#ifndef TRANSFORMER_ENGINE_RECIPE_H_
#define TRANSFORMER_ENGINE_RECIPE_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Update FP8 scaling factors with delayed scaling recipe.
 *
 * The amax history is rotated by -1 (e.g. the first entry shifts to
 * the last, the last entry shifts to the second to last) and the
 * first entry is set to zero. The scaling factor is estimated so the
 * FP8 tensor's maximum absolute value is
 * @f$ 2^{-\text{margin}} \text{max}_\text{fp8\_dtype} @f$.
 *
 *  \param[in] amax_history             History of maximum absolute values.
 *                                      Shape: [history_length, num_scales]
 *  \param[in] scale                    Scaling factor for casting to FP8. Shape: [num_scales]
 *  \param[out] updated_amax_history    Updated history of maximum absolute values.
 *                                      Shape: [history_length, num_scales]
 *  \param[out] updated_scale           Updated scaling factor for casting to FP8.
 *                                      Shape: [num_scales]
 *  \param[in] amax_compute_algo        Method to reduce amax history. Options are "max" and
 *                                      "most_recent".
 *  \param[in] fp8_dtype                FP8 datatype.
 *  \param[in] margin                   Scaling factor margin.
 *  \param[in] stream                   CUDA stream.
 */
void nvte_delayed_scaling_recipe_amax_and_scale_update(
    const NVTETensor amax_history, const NVTETensor scale, NVTETensor updated_amax_history,
    NVTETensor updated_scale, const char* amax_compute_algo, NVTEDType fp8_dtype, float margin,
    cudaStream_t stream);

/*! \brief Bulk-update FP8 scaling factors with delayed scaling recipe after amax reduction.
 *
 * Operations performed include, updating the most recent amax history
 * with the relevant segment of global reduction buffer if it's not 0,
 * rotating the amax history based on the rule below, and updating the
 * scales.
 *
 * The amax history is rotated by -1 (e.g. the first entry shifts to
 * the last, the last entry shifts to the second to last) and the
 * first entry is set to zero. The scaling factor is estimated so the
 * FP8 tensor's maximum absolute value is
 * @f$ 2^{-\text{margin}} \text{max}_\text{fp8\_dtype} @f$.
 *
 *  \param[in] amax_reduction_buffer    The contiguous buffer used for amax reduction.
 *                                      Shape: [num_scales * num_tensors]
 *  \param[in,out] amax_histories       List of amax histories of maximum absolute values.
 *                                      Shape: num_tensors x [history_length, num_scales]
 *  \param[in,out] scales               List of scaling factors for casting to FP8.
 *                                      Shape: num_tensors x [num_scales]
 *  \param[in] amax_compute_algo        Method to reduce amax history. Options are "max" and
 *                                      "most_recent".
 *  \param[in] fp8_dtype                FP8 datatype.
 *  \param[in] margin                   Scaling factor margin.
 *  \param[in] stream                   CUDA stream.
 */
void nvte_delayed_scaling_recipe_amax_and_scale_update_after_reduction(
    const NVTETensor amax_reduction_buffer, std::vector<NVTETensor> amax_histories,
    std::vector<NVTETensor> scales, const char* amax_compute_algo, NVTEDType fp8_dtype,
    float margin, cudaStream_t stream);

/*! \brief Compute an FP8 tensor's amax.
 *
 *  The amax (maximum absolute value) of the input tensor is computed
 *  and written to the amax buffer of the output tensor.
 *
 *  \param[in]     input            Input tensor. Must be unquantized.
 *  \param[in,out] output           Output tensor. Must be an FP8 tensor with per-tensor scaling.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_compute_amax(const NVTETensor input, NVTETensor output, cudaStream_t stream);

/*! \brief Compute an FP8 tensor's amax with quantization config.
 *
 *  The amax (maximum absolute value) of the input tensor is computed
 *  and written to the amax buffer of the output tensor, using the provided
 *  quantization configuration.
 *  One useful config is the noop tensor, which is needed by cuda graph.
 *
 *  \param[in]     input            Input tensor. Must be unquantized.
 *  \param[in,out] output           Output tensor. Must be an FP8 tensor with per-tensor scaling.
 *  \param[in]     config           Quantization configuration.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_compute_amax_with_config(const NVTETensor input, NVTETensor output,
                                   const NVTEQuantizationConfig config, cudaStream_t stream);

/*! \brief Update an FP8 tensor's scale based on its amax.
 *
 *  This is only supported for FP8 tensors with per-tensor scaling.
 *  Options are primarily intended for FP8 current-scaling recipes.
 *
 *  \param[in,out] output           FP8 tensor with per-tensor scaling.
 *  \param[in]     config           Quantization configuration.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_compute_scale_from_amax(NVTETensor output, const NVTEQuantizationConfig config,
                                  cudaStream_t stream);

/*! \brief Compute partial amax for FP8 blockwise scaling.
 *
 *  This function computes the maximum absolute values for each block of the original tensor.
 *  `inp` contains a continuous segment from the flattened original tensor. For each block,
 *  if it overlaps with the range [start_offset, start_offset+inp.length), the amax is
 *  computed from inp; otherwise, the amax is set to 0.
 *
 *  Example: Original tensor (logically 512x512) divided into 16 blocks of size 128x128.
 *  `inp` contains continuous elements starting from position start_offset
 *  in the flattened original tensor.
 *
 *  Logical view - Original Tensor (e.g., 512x512) divided into 16 blocks of size 128x128:
 *  ┌─────────┬─────────┬─────────┬─────────┐
 *  │ Block0  │ Block1  │ Block2  │ Block3  │  Each block: 128x128
 *  │ 128x128 │ 128x128 │ 128x128 │ 128x128 │
 *  ├─────────┼─────────┼─────────┼─────────┤
 *  │ Block4  │ Block5  │ Block6  │ Block7  │
 *  ├─────────┼─────────┼─────────┼─────────┤
 *  │ Block8  │ Block9  │ Block10 │ Block11 │
 *  ├─────────┼─────────┼─────────┼─────────┤
 *  │ Block12 │ Block13 │ Block14 │ Block15 │
 *  └─────────┴─────────┴─────────┴─────────┘
 *
 *  Physical view - Flattened in row-major order:
 *  ┌────────────────────────────────────────────────────────────────┐
 *  │[0...128][128...256][256...384][384...512]...[261632...262143]  │
 *  └────────────────────────────────────────────────────────────────┘
 *                 ^            ^
 *            start_offset  start_offset + inp.length
 *
 *  For each 128x128 block, compute amax:
 *  - If the block overlaps with [start_offset, start_offset+inp.length), compute amax
 *  - If the block is completely outside this range, set amax = 0
 *
 *  amax output (one value per 128x128 block), block 1 and block 2 are non-zero because they
 *  overlap with the [start_offset, start_offset+inp.length) range:
 *  ┌───────┬───────┬───────┬───────┐
 *  │    0  │ amax  │ amax  │    0  │  Block0-3
 *  ├───────┼───────┼───────┼───────┤
 *  │    0  │    0  │    0  │    0  │  Block4-7
 *  ├───────┼───────┼───────┼───────┤
 *  │    0  │    0  │    0  │    0  │  Block8-11
 *  ├───────┼───────┼───────┼───────┤
 *  │    0  │    0  │    0  │    0  │  Block12-15
 *  └───────┴───────┴───────┴───────┘
 *
 *  \param[in]     inp              Input tensor (continuous slice of flattened original tensor).
 *  \param[in,out] amax             Output tensor for maximum absolute values per block.
 *  \param[in]     h                Height dimension of the logical tensor.
 *  \param[in]     w                Width dimension of the logical tensor.
 *  \param[in]     amax_stride_h    Stride in height dimension for amax tensor.
 *  \param[in]     amax_stride_w    Stride in width dimension for amax tensor.
 *  \param[in]     start_offset     Starting offset in the flattened tensor.
 *  \param[in]     block_len        Length of a quantization block to process.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_fp8_block_scaling_compute_partial_amax(const NVTETensor inp, NVTETensor amax, size_t h,
                                                 size_t w, size_t amax_stride_h,
                                                 size_t amax_stride_w, size_t start_offset,
                                                 size_t block_len, cudaStream_t stream);

/*! \brief Perform partial FP8 casting with blockwise scaling.
 *
 *  This function casts the input tensor to FP8 format using blockwise scaling factors.
 *  `inp` contains a continuous segment from the flattened original tensor.
 *
 *  \param[in]     inp              Input tensor.
 *  \param[out]    out              Output tensor in FP8 format.
 *  \param[in]     scale            Scaling factors per block.
 *  \param[in]     h                Height dimension of the tensor.
 *  \param[in]     w                Width dimension of the tensor.
 *  \param[in]     scale_stride_h   Stride in height dimension for scale tensor.
 *  \param[in]     scale_stride_w   Stride in width dimension for scale tensor.
 *  \param[in]     start_offset     Starting offset for partial computation.
 *  \param[in]     block_len        Length of the block to process.
 *  \param[in]     out_dtype        Output FP8 datatype.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_fp8_block_scaling_partial_cast(const NVTETensor inp, NVTETensor out,
                                         const NVTETensor scale, size_t h, size_t w,
                                         size_t scale_stride_h, size_t scale_stride_w,
                                         size_t start_offset, size_t block_len,
                                         const NVTEDType out_dtype, cudaStream_t stream);

/*! \brief Compute partial amax for MXFP8 scaling.
 *
 *  This function computes the maximum absolute values along both row and column dimensions.
 *  input contains a continuous segment from the flattened original tensor. For each row/column
 *  block, if it overlaps with the range starting from start_offset, the amax is computed from
 *  `input`; otherwise, the amax is set to 0.
 *
 *  Example: Original tensor (64 rows x 64 cols).
 *  Rowwise amax granularity: 1x32 (each row divided into 2 blocks)
 *  Columnwise amax granularity: 32x1 (each column divided into 2 blocks)
 *  input contains a continuous segment starting from start_offset.
 *
 *  Logical view - Original Tensor (64x64) with 1x32 and 32x1 blocks:
 *
 *  Rowwise blocks (1x32): Each row has 2 blocks
 *       ┌──────────────┬──────────────┐
 *  row0 │  Block_r0_0  │  Block_r0_1  │  (cols 0-31, 32-63)
 *       ├──────────────┼──────────────┤
 *  row1 │  Block_r1_0  │  Block_r1_1  │
 *       ├──────────────┼──────────────┤
 *   ... │     ...      │     ...      │
 *       ├──────────────┼──────────────┤
 *  row63│  Block_r63_0 │  Block_r63_1 │
 *       └──────────────┴──────────────┘
 *
 *  Columnwise blocks (32x1): Each column has 2 blocks
 *       ┌───┬───┬─────┬───┬───┐
 *       │c0 │c1 │ ... │c62│c63│
 *  ┌────┼───┼───┼─────┼───┼───┤
 *  │Blk0│   │   │     │   │   │  rows 0-31
 *  ├────┼───┼───┼─────┼───┼───┤
 *  │Blk1│   │   │     │   │   │  rows 32-63
 *  └────┴───┴───┴─────┴───┴───┘
 *
 *  Physical view - Flattened in row-major order:
 *  Total elements: 64*64 = 4096
 *  ┌──────────────────────────────────────────────────────┐
 *  │[0...63][64...127][128...191]...[4032...4095]         │
 *  └──────────────────────────────────────────────────────┘
 *       ^                ^
 *     start_offset=60  start_offset + input.length=130
 *
 *  Row-wise amax output (one value per 1x32 block):
 *  ┌────────┬────────┐
 *  │ amax   │  amax  │  row0 (block0 and block1 partially covered)
 *  ├────────┼────────┤
 *  │    0   │    0   │  row1 (not covered)
 *  ├────────┼────────┤
 *  │   ...  │   ...  │
 *  ├────────┼────────┤
 *  │    0   │    0   │  row63 (not covered)
 *  └────────┴────────┘
 *
 *  Column-wise amax output (one value per 32x1 block):
 *  ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┐
 *  │  amax  │  amax  │  amax  │ amax   │  amax  │ amax   │ amax   │ ... row 0-31
 *  ├────────┼────────┼────────┼────────┼────────┼────────┼────────┤
 *  │ amax=0 │ amax=0 │ amax=0 │ amax=0 │ amax=0 │ amax=0 │ amax=0 │ ... row 32-62
 *  └────────┴────────┴────────┴────────┴────────┴────────┴────────┘
 *    col0     col1     col2     col3     col4     col5     col6
 *
 *  For each 1x32 or 32x1 block, if it overlaps with [start_offset, start_offset+input.length),
 *  compute amax; otherwise set to 0.
 *
 *  \param[in]     input            Input tensor (continuous segment of flattened original tensor).
 *  \param[in,out] amax_rowwise     Output tensor for row-wise maximum absolute values.
 *  \param[in,out] amax_colwise     Output tensor for column-wise maximum absolute values.
 *  \param[in]     rows             Number of rows in the logical tensor.
 *  \param[in]     cols             Number of columns in the logical tensor.
 *  \param[in]     start_offset     Starting offset in the flattened tensor.
 *  \param[in]     stream           CUDA stream used for the operation.
 */
void nvte_mxfp8_scaling_compute_partial_amax(const NVTETensor input, NVTETensor amax_rowwise,
                                             NVTETensor amax_colwise, int rows, int cols,
                                             size_t start_offset, cudaStream_t stream);

/*! \brief Perform partial MXFP8 casting.
 *
 *  This function casts the input tensor to MXFP8 format, producing both row-wise and
 *  column-wise scaled outputs. input contains a continuous segment from the flattened
 *  original tensor.
 *
 *  \param[in]     input               Input (continuous segment of flattened original tensor).
 *  \param[out]    output_rowwise      Output tensor with row-wise scaling (MXFP8 format).
 *  \param[out]    output_colwise      Output tensor with column-wise scaling (MXFP8 format).
 *  \param[in]     scale_inv_rowwise   Inverse scaling factors for row-wise scaling.
 *  \param[in]     scale_inv_colwise   Inverse scaling factors for column-wise scaling.
 *  \param[in]     rows                Number of rows in the logical tensor.
 *  \param[in]     cols                Number of columns in the logical tensor.
 *  \param[in]     start_offset        Starting offset in the flattened tensor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_mxfp8_scaling_partial_cast(const NVTETensor input, NVTETensor output_rowwise,
                                     NVTETensor output_colwise, const NVTETensor scale_inv_rowwise,
                                     const NVTETensor scale_inv_colwise, int rows, int cols,
                                     size_t start_offset, cudaStream_t stream);

/*! \brief Compute per-tensor scaling factor for NVFP4 format.
 *
 *  This function computes the scaling factor (alpha) for NVFP4 quantization based
 *  on the input tensors A and B, with options for using row-wise amax values.
 *
 *  \param[in]     inpA                Input tensor A.
 *  \param[in]     use_rowwise_amax_A  Whether to use row-wise amax for tensor A.
 *  \param[in]     inpB                Input tensor B.
 *  \param[in]     use_rowwise_amax_B  Whether to use row-wise amax for tensor B.
 *  \param[in]     alpha_in            Input scaling factor.
 *  \param[out]    alpha_out           Output scaling factor.
 *  \param[in]     stream              CUDA stream used for the operation.
 */
void nvte_nvfp4_compute_per_tensor_scale(const NVTETensor inpA, const bool use_rowwise_amax_A,
                                         const NVTETensor inpB, const bool use_rowwise_amax_B,
                                         float alpha_in, NVTETensor alpha_out, cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // TRANSFORMER_ENGINE_RECIPE_H_
