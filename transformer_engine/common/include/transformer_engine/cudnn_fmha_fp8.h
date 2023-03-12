/*
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

//#pragma once

//#include <iostream>
//#include <inttypes.h>
//#include <stdlib.h>
//#include <string.h>
//#include <ctype.h>
//#include <cuda_runtime.h>
//#include <assert.h>
//#include <tuple>
//#include <functional>

//#include <cudnn.h>
////#include "fp16_dev.h"
////#include "fp16_emu.h"
////#include "helpers.h"
//
////#include "transformer_engine/transformer_engine.h"
//#include "../../common.h"
////#include "../common.h"


#ifndef TRANSFORMER_ENGINE_CUDNN_FMHA_FP8_H_
#define TRANSFORMER_ENGINE_CUDNN_FMHA_FP8_H_

#include "transformer_engine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CUDNN_FRONTEND_UNUSED(X) ((void)X)

//#define THRESHOLD 2.0e-2

//using namespace transformer_engine;

enum class MHA_Layout {
    NOT_INTERLEAVED = 0,
    QKV_INTERLEAVED = 1,
    KV_INTERLEAVED = 2
};

enum class MHA_Matrix {
    Q_Matrix = 0, // queries
    K_Matrix = 1, // keys
    V_Matrix = 2, // values
    S_Matrix = 3, // output of GEMM1
    O_Matrix = 4, // final output
};

//void generateMHAStrides(int64_t b, int64_t h, int64_t s_q, int64_t s_kv, int64_t d, int64_t* strideA, MHA_Layout layout, MHA_Matrix matrix);

//#if (CUDNN_VERSION >= 8900)

//void 
//run_fp8_flash_mha_fprop(int64_t b, 
//                int64_t h, 
//                int64_t s_q,
//                int64_t s_kv,
//                int64_t d,
//                float attnScale,
//                MHA_Layout layout,
//                void* devPtrQKV, 
//                void* devPtrM,
//                void* devPtrZInv,
//                void* devPtrO,
//                void* devPtrDescaleQ,
//                void* devPtrDescaleK,
//                void* devPtrDescaleV,
//                void* devPtrDescaleS,
//                void* devPtrScaleS,
//                void* devPtrScaleO,
//                void* devPtrAmaxO,
//                void* devPtrAmaxS,
//                void* devPtrQKVRaggedOffset,
//                void* devPtrORaggeDOffset,
//                void* devPtrMOverride,
//                void* devPtrNOverride,
//                void* devPtrKOverride,
//                cudnnDataType_t tensorType);
//
//void fp8_flash_mha_fprop(int64_t b, 
//                int64_t h, 
//                int64_t s_q,
//                int64_t s_kv,
//                int64_t d,
//                float attnScale,
//                MHA_Layout layout,
//		const transformer_engine::Tensor *inputQKV,
//		const transformer_engine::Tensor *inputM,
//                const transformer_engine::Tensor *inputZInv,
//                const transformer_engine::Tensor *inputO,
//                const transformer_engine::Tensor *inputDescaleQ,
//                const transformer_engine::Tensor *inputDescaleK,
//                const transformer_engine::Tensor *inputDescaleV,
//                const transformer_engine::Tensor *inputDescaleS,
//                const transformer_engine::Tensor *inputScaleS,
//                const transformer_engine::Tensor *inputScaleO,
//                const transformer_engine::Tensor *inputAmaxS,
//                const transformer_engine::Tensor *inputAmaxO,
//                const transformer_engine::Tensor *inputQKVRaggedOffset,
//                const transformer_engine::Tensor *inputORaggedOffset,
//		const transformer_engine::Tensor *inputActualSeqlenQ,
//		const transformer_engine::Tensor *inputActualSeqlenK,
//		const transformer_engine::Tensor *inputActualSeqlenO,
//                void* workspace,
//                size_t workspaceSize,
//                cudaStream_t stream);

void nvte_cudnn_fmha_fwd(const NVTETensor QKV,
		const NVTETensor M,
                const NVTETensor ZInv,
                const NVTETensor O,
                const NVTETensor DescaleQ,
                const NVTETensor DescaleK,
                const NVTETensor DescaleV,
                const NVTETensor DescaleS,
                const NVTETensor ScaleS,
                const NVTETensor ScaleO,
                const NVTETensor AmaxS,
                const NVTETensor AmaxO,
                const NVTETensor QKVRaggedOffset,
                const NVTETensor ORaggedOffset,
		const NVTETensor ActualSeqlenQ,
		const NVTETensor ActualSeqlenK,
		const NVTETensor ActualSeqlenO,
		NVTETensor workspace,
		cudaStream_t stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif
