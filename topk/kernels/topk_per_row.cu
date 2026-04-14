// Adapted from:
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/indexerTopK.cu
//
// Key differences from that reference:
//  - No TRT-LLM-specific includes; uses standalone CUB and PyTorch/pybind11.
//  - Single dispatch entry point for both prefill and decode (via `is_prefill`).
//  - Simplified decode path: rowStart is always 0, stride1 is always 1.
//  - cudaFuncSetAttribute called when topK is large enough that static + dynamic
//    smem would exceed the 48 KB default carveout.

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>

// ---------------------------------------------------------------------------
// Device helpers (identical to TRT-LLM reference)
// ---------------------------------------------------------------------------

template <int step>
static inline __device__ uint32_t extractBinIdx(float x)
{
    if constexpr (step == 0) {
        __half hx = __float2half(x);
        uint16_t bits = __half_as_ushort(hx);
        bits = (bits & 0x8000) ? bits : ~bits & 0x7fff;
        return bits >> 5;
    } else {
        uint32_t bits = __float_as_uint(x);
        bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
        if constexpr (step == 1) return bits >> 21;
        if constexpr (step == 2) return (bits >> 10) & 0x7ff;
        if constexpr (step == 3) return bits & 0x3ff;
    }
}

template <int shift>
static inline __device__ bool isPartialMatch(float x, uint32_t pattern)
{
    if constexpr (shift == 0) return true;
    uint32_t bits = __float_as_uint(x);
    bits = (bits & 0x80000000) ? bits : ~bits & 0x7fffffff;
    return (bits ^ pattern) >> shift == 0;
}

template <typename T, typename idxT, typename Func>
__device__ void vectorized_process(size_t thread_rank, size_t num_threads, T const* in, idxT len, Func f)
{
    constexpr int WARP_SIZE = 32;
    using WideT = float4;
    if constexpr (sizeof(T) >= sizeof(WideT)) {
        for (idxT i = thread_rank; i < len; i += num_threads) f(in[i], i);
    } else {
        static_assert(sizeof(WideT) % sizeof(T) == 0);
        constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
        union { WideT scalar; T array[items_per_scalar]; } wide;

        int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
            ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T)) : 0;
        if (skip_cnt > len) skip_cnt = len;

        WideT const* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
        idxT const len_cast  = (len - skip_cnt) / items_per_scalar;

        for (idxT i = thread_rank; i < len_cast; i += num_threads) {
            wide.scalar = in_cast[i];
            idxT const real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
            for (int j = 0; j < items_per_scalar; ++j) f(wide.array[j], real_i + j);
        }
        static_assert(WARP_SIZE >= items_per_scalar);
        if (thread_rank < (size_t)skip_cnt) f(in[thread_rank], thread_rank);
        idxT const remain_i = skip_cnt + len_cast * items_per_scalar + thread_rank;
        if (remain_i < len) f(in[remain_i], remain_i);
    }
}

// ---------------------------------------------------------------------------
// Histogram step (identical to TRT-LLM reference)
// ---------------------------------------------------------------------------

template <int step, int kNumThreadsPerBlock, int kNumBins, int kNumFinalItems,
          bool multipleBlocksPerRow, bool mergeBlocks,
          typename SmemFinalType, typename SmemOutputType>
__device__ bool processHistogramStep(int const* indices, float const* logits, int rowEnd,
    uint32_t& logitPattern, int& thresholdBinIdx, SmemOutputType& smemOutput,
    int* smemThresholdBinIdx, int* smemFinalDstIdx, int* smemFinalBinSize,
    int* smemFoundTopKValues, SmemFinalType& smemFinal, int stride1, int rowStart, int topK)
{
#pragma unroll
    for (int idx = threadIdx.x; idx < kNumBins; idx += kNumThreadsPerBlock)
        smemFinal.histo.data[idx] = 0;
    __syncthreads();

    constexpr auto patternShift = step < 2 ? 0 : step == 2 ? 21 : 10;
    if constexpr (step == 2)
        logitPattern = static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;
    else if constexpr (step == 3)
        logitPattern |= static_cast<uint32_t>(thresholdBinIdx & 0x7ff) << patternShift;

    auto distributeToBins = [&](float logit, int = 0) {
        if (isPartialMatch<patternShift>(logit, logitPattern))
            atomicAdd(&smemFinal.histo.data[extractBinIdx<step>(logit)], 1);
    };

    if (stride1 == 1)
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, distributeToBins);
    else
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
            distributeToBins(logits[idx * stride1], idx);
    __syncthreads();

    int lastValue = smemFoundTopKValues[0];
    for (int round = 0; round < kNumBins / kNumThreadsPerBlock; round++) {
        int idx = threadIdx.x + kNumThreadsPerBlock * round;
        int binCount = smemFinal.histo.data[idx];
        __syncthreads();

        int prefixSum{0}, totalSum{0};
        using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;
        Scan(smemFinal.histo.scan).ExclusiveSum(binCount, prefixSum, totalSum);
        prefixSum += lastValue;
        totalSum  += lastValue;
        smemFinal.histo.data[idx] = prefixSum;
        __syncthreads();

        bool foundThreshold = false;
        if (prefixSum < topK) {
            int nextPrefixSum = threadIdx.x == kNumThreadsPerBlock - 1
                ? totalSum : smemFinal.histo.data[idx + 1];
            if (nextPrefixSum >= topK) {
                smemThresholdBinIdx[0] = idx;
                smemFinalBinSize[0]    = nextPrefixSum - prefixSum;
                foundThreshold         = true;
            }
        }
        if (__syncthreads_or(foundThreshold)) break;
        lastValue = totalSum;
    }
    __syncthreads();

    thresholdBinIdx = smemThresholdBinIdx[0];

    auto processBins = [&](float logit, int idx) {
        if (isPartialMatch<patternShift>(logit, logitPattern)) {
            uint32_t binIdx = extractBinIdx<step>(logit);
            if (binIdx < (uint32_t)thresholdBinIdx) {
                int dstIdx = atomicAdd(&smemFoundTopKValues[0], 1);
                if constexpr (mergeBlocks) {
                    smemOutput[dstIdx] = indices[idx];
                } else if constexpr (multipleBlocksPerRow) {
                    smemOutput[dstIdx] = idx + rowStart;
                    reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                } else {
                    smemOutput[dstIdx] = idx;
                }
            }
            if constexpr (step < 3) {
                if (binIdx == (uint32_t)thresholdBinIdx && smemFinalBinSize[0] <= kNumFinalItems) {
                    int dstIdx = atomicAdd(&smemFinalDstIdx[0], 1);
                    smemFinal.items.logits[dstIdx] = logit;
                    if constexpr (mergeBlocks)
                        smemFinal.items.indices[dstIdx] = indices[idx];
                    else if constexpr (multipleBlocksPerRow)
                        smemFinal.items.indices[dstIdx] = idx + rowStart;
                    else
                        smemFinal.items.indices[dstIdx] = idx;
                }
            } else {
                if (binIdx == (uint32_t)thresholdBinIdx) {
                    int dstIdx = atomicAdd(&smemFinal.histo.data[binIdx], 1);
                    if (dstIdx < topK) {
                        if constexpr (mergeBlocks)
                            smemOutput[dstIdx] = indices[idx];
                        else if constexpr (multipleBlocksPerRow) {
                            smemOutput[dstIdx] = idx + rowStart;
                            reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = logit;
                        } else
                            smemOutput[dstIdx] = idx;
                    }
                }
            }
        }
    };

    if (stride1 == 1)
        vectorized_process(threadIdx.x, kNumThreadsPerBlock, logits + rowStart, rowEnd - rowStart, processBins);
    else
        for (int idx = rowStart + threadIdx.x; idx < rowEnd; idx += kNumThreadsPerBlock)
            processBins(logits[idx * stride1], idx);
    __syncthreads();

    return smemFinalBinSize[0] > kNumFinalItems;
}

// ---------------------------------------------------------------------------
// Core algorithm — identical to TRT-LLM's topKPerRowJob
// ---------------------------------------------------------------------------

template <int kNumThreadsPerBlock, int kNumBins, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __device__ void topKPerRowJob(int const* indices, float const* logits,
    int rowStart, int rowEnd, int* outIndices, float* outLogits, int stride1, int topK)
{
    static constexpr int kNumFinalItems = 2048;
    static constexpr int kNumFinalItemsPerThread = kNumFinalItems / kNumThreadsPerBlock;

    using FinalSort = cub::BlockRadixSort<float, kNumThreadsPerBlock, kNumFinalItemsPerThread, int>;
    using FinalSortTempStorage = std::conditional_t<useRadixSort, typename FinalSort::TempStorage, int>;
    using Scan = cub::BlockScan<int, kNumThreadsPerBlock>;

    struct FinalItems   { int indices[kNumFinalItems]; float logits[kNumFinalItems]; };
    struct Histogram    { typename Scan::TempStorage scan; int data[kNumBins]; };

    __shared__ union {
        FinalItems          items;
        FinalSortTempStorage finalSort;
        Histogram           histo;
    } smemFinal;

    extern __shared__ int32_t smemOutput[];   // topK or 2*topK int32s allocated at launch

    __shared__ int smemThresholdBinIdx[1];
    __shared__ int smemFinalDstIdx[1];
    __shared__ int smemFinalBinSize[1];
    __shared__ int smemFoundTopKValues[1];

    int rowLen = rowEnd - rowStart;
    if (rowLen <= topK) {
        for (int rowIt = threadIdx.x; rowIt < rowLen; rowIt += kNumThreadsPerBlock) {
            if constexpr (multipleBlocksPerRow) {
                outIndices[rowIt] = rowIt + rowStart;
                outLogits[rowIt]  = logits[rowIt + rowStart];
            } else {
                outIndices[rowIt] = rowIt;
            }
        }
        for (int rowIt = rowLen + threadIdx.x; rowIt < topK; rowIt += kNumThreadsPerBlock) {
            outIndices[rowIt] = -1;
            if constexpr (multipleBlocksPerRow) outLogits[rowIt] = -FLT_MAX;
        }
        return;
    }

    if (threadIdx.x == 0) { smemFinalDstIdx[0] = 0; smemFoundTopKValues[0] = 0; }
    __syncthreads();

    int thresholdBinIdx = -1;
    uint32_t logitPattern = 0;

    bool cont = processHistogramStep<0, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
        smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    if (cont)
        cont = processHistogramStep<1, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    if (cont)
        cont = processHistogramStep<2, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);
    if (cont)
        processHistogramStep<3, kNumThreadsPerBlock, kNumBins, kNumFinalItems, multipleBlocksPerRow, mergeBlocks>(
            indices, logits, rowEnd, logitPattern, thresholdBinIdx, smemOutput,
            smemThresholdBinIdx, smemFinalDstIdx, smemFinalBinSize, smemFoundTopKValues, smemFinal, stride1, rowStart, topK);

    if (!cont) {
        if constexpr (useRadixSort) {
            float finalLogits[kNumFinalItemsPerThread];
            int   finalIndices[kNumFinalItemsPerThread];
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) finalLogits[ii] = -FLT_MAX;
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                if (srcIdx < smemFinalDstIdx[0]) {
                    finalLogits[ii]  = smemFinal.items.logits[srcIdx];
                    finalIndices[ii] = smemFinal.items.indices[srcIdx];
                }
            }
            __syncthreads();
            FinalSort(smemFinal.finalSort).SortDescendingBlockedToStriped(finalLogits, finalIndices);
            int baseIdx = smemFoundTopKValues[0];
#pragma unroll
            for (int ii = 0; ii < kNumFinalItemsPerThread; ++ii) {
                int srcIdx = ii * kNumThreadsPerBlock + threadIdx.x;
                int dstIdx = baseIdx + srcIdx;
                if (dstIdx < topK) {
                    smemOutput[dstIdx] = finalIndices[ii];
                    if constexpr (multipleBlocksPerRow)
                        reinterpret_cast<float*>(smemOutput + topK)[dstIdx] = finalLogits[ii];
                }
            }
        } else {
            auto baseIdx = smemFoundTopKValues[0];
            for (int i = threadIdx.x; i < smemFinalDstIdx[0]; i += kNumThreadsPerBlock) {
                int outIndex = 0;
                auto logit = smemFinal.items.logits[i];
                for (int j = 0; j < smemFinalDstIdx[0]; j++) {
                    auto other = smemFinal.items.logits[j];
                    if (logit < other || (logit == other && i < j)) outIndex++;
                }
                if (outIndex + baseIdx < topK) {
                    smemOutput[outIndex + baseIdx] = smemFinal.items.indices[i];
                    if constexpr (multipleBlocksPerRow)
                        reinterpret_cast<float*>(smemOutput + topK)[outIndex + baseIdx] = smemFinal.items.logits[i];
                }
            }
        }
        __syncthreads();
    }

    // Store results — TRT-LLM fix: stride1==1 uses vectorized_process which
    // already offsets by rowStart, so the stored index is row-local.
    for (int i = threadIdx.x; i < topK; i += kNumThreadsPerBlock) {
        if constexpr (multipleBlocksPerRow) {
            outIndices[i] = smemOutput[i];
            outLogits[i]  = reinterpret_cast<float*>(smemOutput + topK)[i];
        } else {
            outIndices[i] = stride1 == 1 ? smemOutput[i] : smemOutput[i] - rowStart;
        }
    }
}

// ---------------------------------------------------------------------------
// Global kernels
// ---------------------------------------------------------------------------

template <int kNumThreadsPerBlock, bool useRadixSort>
static __global__ __launch_bounds__(kNumThreadsPerBlock)
void topKPerRowPrefill(float const* logits, int const* rowStarts, int const* rowEnds,
                       int* outIndices, int stride0, int stride1, int topK, int offsetIndex)
{
    static constexpr int kNumBins = 2048;
    int rowIdx   = blockIdx.x + offsetIndex;
    int rowStart = rowStarts[rowIdx];
    int rowEnd   = rowEnds[rowIdx];
    outIndices += static_cast<int64_t>(rowIdx) * topK;
    logits     += static_cast<int64_t>(rowIdx) * stride0;
    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort>(
        nullptr, logits, rowStart, rowEnd, outIndices, nullptr, stride1, topK);
}

template <int kNumThreadsPerBlock, bool useRadixSort,
          bool multipleBlocksPerRow = false, bool mergeBlocks = false>
static __global__ __launch_bounds__(kNumThreadsPerBlock)
void topKPerRowDecode(float const* logits, int const* rowEnds, int* outIndices,
                      int stride0, int stride1, int topK,
                      float* outLogits = nullptr, int numBlocksToMerge = 0,
                      int const* indices = nullptr)
{
    static constexpr int kNumBins = 2048;
    int rowIdx   = blockIdx.x;
    int rowStart = 0;
    int rowEnd   = rowEnds[rowIdx];

    if constexpr (!multipleBlocksPerRow && !mergeBlocks) {
        outIndices += static_cast<int64_t>(rowIdx) * topK;
    } else if constexpr (multipleBlocksPerRow) {
        auto const blockSize = rowEnd / gridDim.y;
        rowStart   = blockSize * blockIdx.y;
        rowEnd     = gridDim.y == blockIdx.y + 1 ? rowEnd : rowStart + blockSize;
        outIndices += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
        outLogits  += static_cast<int64_t>(rowIdx) * gridDim.y * topK + blockIdx.y * topK;
    } else if constexpr (mergeBlocks) {
        rowEnd      = numBlocksToMerge * topK;
        indices    += static_cast<int64_t>(rowIdx) * numBlocksToMerge * topK;
        outIndices += static_cast<int64_t>(rowIdx) * topK;
    }
    logits += static_cast<int64_t>(rowIdx) * stride0;

    topKPerRowJob<kNumThreadsPerBlock, kNumBins, useRadixSort, multipleBlocksPerRow, mergeBlocks>(
        indices, logits, rowStart, rowEnd, outIndices, outLogits, stride1, topK);
}

// ---------------------------------------------------------------------------
// Shared-memory attribute helper
// ---------------------------------------------------------------------------
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")

// Raise a kernel's dynamic-smem cap when static + requested dynamic smem
// would exceed the 48 KB default.  Queries the exact static smem size via
// cudaFuncGetAttributes so we never over-request.
template <typename KernelFunc>
static void ensure_smem(KernelFunc func, size_t needed_dynamic_bytes)
{
    constexpr size_t kDefaultLimit = 48 * 1024;
    cudaFuncAttributes attrs{};
    TORCH_CHECK(cudaFuncGetAttributes(&attrs, func) == cudaSuccess,
                "cudaFuncGetAttributes failed");
    if (attrs.sharedSizeBytes + needed_dynamic_bytes <= kDefaultLimit) return;

    int dev = 0; cudaGetDevice(&dev);
    int optin_max = 0;
    cudaDeviceGetAttribute(&optin_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    int dyn_limit = optin_max - static_cast<int>(attrs.sharedSizeBytes);
    TORCH_CHECK(dyn_limit >= static_cast<int>(needed_dynamic_bytes),
                "Requested smem (", needed_dynamic_bytes, " B dynamic + ",
                attrs.sharedSizeBytes, " B static) exceeds device limit (", optin_max, " B).");
    TORCH_CHECK(cudaFuncSetAttribute(func, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_limit)
                    == cudaSuccess,
                "cudaFuncSetAttribute failed");
}

// ---------------------------------------------------------------------------
// Dispatch (mirrors TRT-LLM's invokeIndexerTopK{Decode,Prefill})
// ---------------------------------------------------------------------------

static void vllm_topk_interface(at::Tensor logits, at::Tensor outIndicesAux,
                                at::Tensor outLogitsAux, at::Tensor outIndices,
                                at::Tensor rowEnds, bool isPrefill, int topK)
{
    CHECK_CUDA(logits);
    CHECK_CUDA(outIndicesAux);
    CHECK_CUDA(outLogitsAux);
    CHECK_CUDA(outIndices);
    CHECK_CUDA(rowEnds);

    const auto logitsPtr        = logits.data_ptr<float>();
    const auto outIndicesAuxPtr = outIndicesAux.data_ptr<int32_t>();
    const auto outLogitsAuxPtr  = outLogitsAux.data_ptr<float>();
    const auto outIndicesPtr    = outIndices.data_ptr<int32_t>();
    const auto rowEndsPtr       = rowEnds.data_ptr<int32_t>();
    const int64_t numRows    = logits.size(0);
    const int64_t numColumns = logits.size(1);
    const int stride0 = numColumns;
    const int stride1 = 1;
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int kSortingAlgorithmThreshold = 12288;
    constexpr int kSplitWorkThreshold        = 200 * 1000;
    constexpr int kNumThreadsPerBlock        = 512;
    constexpr int kMultiBlockConfig          = 10;

    const size_t smemSingle = static_cast<size_t>(topK) * sizeof(int32_t);
    const size_t smemDouble = 2 * smemSingle;

    if (isPrefill) {
        // Prefill: variable-length rows via rowStarts / rowEnds.
        // We pass zeros as rowStarts (our rowEnds tensor holds the full lengths).
        // Allocate a zero rowStarts on device.
        auto rowStarts = at::zeros({numRows}, torch::dtype(torch::kInt32).device(logits.device()));
        const int* rowStartsPtr = rowStarts.data_ptr<int32_t>();

        int numInsertionBlocks = std::min(static_cast<int>(numRows), kSortingAlgorithmThreshold);
        topKPerRowPrefill<kNumThreadsPerBlock, false>
            <<<numInsertionBlocks, kNumThreadsPerBlock, smemSingle, stream>>>(
                logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr, stride0, stride1, topK, 0);

        if (numRows > kSortingAlgorithmThreshold) {
            int numRadixBlocks = numRows - kSortingAlgorithmThreshold;
            topKPerRowPrefill<kNumThreadsPerBlock, true>
                <<<numRadixBlocks, kNumThreadsPerBlock, smemSingle, stream>>>(
                    logitsPtr, rowStartsPtr, rowEndsPtr, outIndicesPtr,
                    stride0, stride1, topK, kSortingAlgorithmThreshold);
        }
    } else {
        if (numColumns < kSortingAlgorithmThreshold) {
            topKPerRowDecode<kNumThreadsPerBlock, false>
                <<<numRows, kNumThreadsPerBlock, smemSingle, stream>>>(
                    logitsPtr, rowEndsPtr, outIndicesPtr, stride0, stride1, topK);
        } else if (numColumns < kSplitWorkThreshold) {
            topKPerRowDecode<kNumThreadsPerBlock, true>
                <<<numRows, kNumThreadsPerBlock, smemSingle, stream>>>(
                    logitsPtr, rowEndsPtr, outIndicesPtr, stride0, stride1, topK);
        } else {
            // Two-pass: split each row across kMultiBlockConfig sub-blocks, then merge.
            auto* pass1 = topKPerRowDecode<kNumThreadsPerBlock, true,  true,  false>;
            auto* pass2 = topKPerRowDecode<kNumThreadsPerBlock, false, false, true>;
            // Raise the smem cap only when static+dynamic would exceed 48 KB.
            ensure_smem(pass1, smemDouble);
            ensure_smem(pass2, smemSingle);

            pass1<<<dim3(numRows, kMultiBlockConfig), kNumThreadsPerBlock, smemDouble, stream>>>(
                logitsPtr, rowEndsPtr, outIndicesAuxPtr,
                stride0, stride1, topK, outLogitsAuxPtr, 0, nullptr);

            pass2<<<numRows, kNumThreadsPerBlock, smemSingle, stream>>>(
                outLogitsAuxPtr, rowEndsPtr, outIndicesPtr,
                kMultiBlockConfig * topK, 1, topK,
                nullptr, kMultiBlockConfig, outIndicesAuxPtr);
        }
    }

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "topk_per_row kernel failed: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Python-facing allocate / invoke
// ---------------------------------------------------------------------------

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
allocate_topk_per_row_buffers(torch::Tensor logits, int k)
{
    const auto numRows         = logits.size(0);
    const int  kMultiBlockConfig = 10;
    auto outIndicesAux = torch::empty({numRows, kMultiBlockConfig, k},
        torch::dtype(torch::kInt32).device(logits.device()));
    auto outLogitsAux  = torch::empty({numRows, kMultiBlockConfig, k},
        torch::dtype(torch::kFloat).device(logits.device()));
    auto outIndices    = torch::empty({numRows, k},
        torch::dtype(torch::kInt32).device(logits.device()));
    return {outIndicesAux, outLogitsAux, outIndices};
}

void invoke_topk_per_row(torch::Tensor logits, torch::Tensor lengths, int k,
                         torch::Tensor outIndicesAux, torch::Tensor outLogitsAux,
                         torch::Tensor indices, bool isPrefill)
{
    vllm_topk_interface(logits, outIndicesAux, outLogitsAux, indices, lengths, isPrefill, k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("allocate_buffers", &allocate_topk_per_row_buffers,
          "Allocate buffers for Top K Per Row kernel",
          py::arg("score"), py::arg("k"));
    m.def("topk_kernel", &invoke_topk_per_row,
          "Call Top K Per Row kernel with pre-allocated buffers",
          py::arg("score"), py::arg("lengths"), py::arg("k"),
          py::arg("out_indices_aux"), py::arg("out_logits_aux"),
          py::arg("indices"), py::arg("is_prefill"));
}
