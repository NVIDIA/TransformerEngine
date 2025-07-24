#pragma once

#include <cstdlib>
#include <cooperative_groups.h>

namespace transformer_engine {
namespace mxfp8_kernel {
namespace spec {

namespace cg = cooperative_groups;
namespace ptx = transformer_engine::ptx;



inline bool is_cast_only_enabled() {
    const char* env = std::getenv("ENABLE_CAST_ONLY");
    return env != nullptr && (env[0] == '1');
}


template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename IType, typename OType>
inline bool hasSpec() { return false; }

// IType could be [float, fp16, bf16]
// OType could be [fp8e5m2, fp8e4m3]
template <>
inline bool hasSpec<false, false, false, float, fp8e5m2>() { return is_cast_only_enabled(); }
template <>
inline bool hasSpec<false, false, false, float, fp8e4m3>() { return is_cast_only_enabled(); }
template <>
inline bool hasSpec<false, false, false, fp16, fp8e5m2>() { return is_cast_only_enabled(); }
template <>
inline bool hasSpec<false, false, false, fp16, fp8e4m3>() { return is_cast_only_enabled(); }
template <>
inline bool hasSpec<false, false, false, bf16, fp8e5m2>() { return is_cast_only_enabled(); }
template <>
inline bool hasSpec<false, false, false, bf16, fp8e4m3>() { return is_cast_only_enabled(); }

template <int32_t _M, int32_t _N>
struct Layout {
    static constexpr int32_t M = _M;
    static constexpr int32_t N = _N;
    static constexpr int32_t num = M * N;
};

template <typename IType, typename OType,
          bool rowwise, bool colwise>
struct CastTraits;

// 1x32
template <typename _IType, typename _OType>
struct CastTraits<_IType, _OType, /*rowwise=*/true, /*colwise=*/false> {
    static constexpr bool isRowwise = true;
    static constexpr bool isColwise = false;
    using IType = _IType;
    using OType = _OType;

    static constexpr int32_t chunkElems = 32;
    using threadLayout = Layout<8, 4>;
    static constexpr int32_t numThreadsPerChunk = 1;
    static constexpr int32_t warpDimM = threadLayout::M;
    static constexpr int32_t warpDimN = threadLayout::N * chunkElems;
    using inputUnitType = uint4;
    static constexpr int32_t numUnitsPerChunk = chunkElems * sizeof(IType) / sizeof(inputUnitType);
    using outputUnitType = uint4;
    static constexpr int32_t numOutUnitsPerChunk = chunkElems * sizeof(OType) / sizeof(outputUnitType);

    using warpLayout = Layout<4, 1>;
    static constexpr int32_t blockDimM = warpLayout::M * warpDimM;
    static constexpr int32_t blockDimN = warpLayout::N * warpDimN;

    static constexpr int32_t numThreads = warpLayout::num * 32;
    static constexpr size_t smem = 0ul;
};

// 1x32
template <typename CastTraits,
          std::enable_if_t<CastTraits::isRowwise && !CastTraits::isColwise, int> = 0>
__global__ void cast_mxfp8_kernel(
    typename CastTraits::IType * __restrict__ input,
    typename CastTraits::OType * __restrict__ output,
    e8m0_t * __restrict__ scales_rowwise,
    float * __restrict__ amax_ptr,
    int32_t rows,
    int32_t cols,
    int32_t scale_stride_rowwise,
    int32_t scale_stride_colwise
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using IType2 = typename ptx::FPx2<typename CastTraits::IType>;
    constexpr int32_t numItersIType2 = sizeof(typename CastTraits::inputUnitType) / sizeof(IType2);
    using OType2 = typename ptx::FPx2<typename CastTraits::OType>;

    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);

    float block_amax = 0.0f;

    int32_t offset = blockIdx.y * CastTraits::blockDimM * cols
                    + blockIdx.x * CastTraits::blockDimN
                    + threadIdx.z * CastTraits::warpDimM * cols
                    + threadIdx.y * CastTraits::warpDimN
                    + (threadIdx.x / CastTraits::threadLayout::N) * cols
                    + (threadIdx.x % CastTraits::threadLayout::N) * CastTraits::chunkElems;

    typename CastTraits::inputUnitType * input_units = reinterpret_cast<typename CastTraits::inputUnitType *>(input + offset);


    if constexpr (std::is_same_v<typename CastTraits::IType, float>) {

    } else {
        IType2 thread_amax2{0.f, 0.f};

        typename CastTraits::inputUnitType rInput[CastTraits::numUnitsPerChunk];
        rInput[0] = input_units[0];
        #pragma unroll
        for (int32_t i = 1; i < CastTraits::numUnitsPerChunk; i++) {
            rInput[i] = input_units[i];

            IType2 * rInput2 = reinterpret_cast<IType2 *>(&rInput[i - 1]);
            #pragma unroll
            for (int32_t j = 0; j < numItersIType2; j++) {
                ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[j]);
            }
        }

        IType2 * rInput2 = reinterpret_cast<IType2 *>(&rInput[CastTraits::numUnitsPerChunk - 1]);
        #pragma unroll
        for (int32_t j = 0; j < numItersIType2; j++) {
            ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[j]);
        }
        typename CastTraits::IType thread_amax = ptx::get_amax(thread_amax2.x, thread_amax2.y);
        float thread_amax_fp32 = static_cast<float>(thread_amax);
        // FIXME: this may not be needed
        // thread_amax_fp32 = fabsf(thread_amax_fp32);

        e8m0_t biased_exponent = ptx::float_to_e8m0(
            thread_amax_fp32 * Quantized_Limits<typename CastTraits::OType>::max_norm_rcp);

        // write biased_exponent
        int32_t scale_offset = blockIdx.y * CastTraits::blockDimM * scale_stride_rowwise
                               + blockIdx.x * (CastTraits::blockDimN / CastTraits::chunkElems)
                               + threadIdx.z * CastTraits::warpDimM * scale_stride_rowwise
                               + threadIdx.y * CastTraits::warpDimN / CastTraits::chunkElems
                               + (threadIdx.x / CastTraits::threadLayout::N) * scale_stride_rowwise
                               + (threadIdx.x % CastTraits::threadLayout::N) * 1;
        // method-1: point-wise writing
        scales_rowwise[scale_offset] = biased_exponent;
        // method-2: packed to int32_t, then STG.32
        // uint32_t packed_scale = biased_exponent << ((threadIdx.x % 4) * 8);
        // uint32_t group_mask = 0xF << ((threadIdx.x / 4) * 4);
        // packed_scale = __reduce_or_sync(group_mask, packed_scale);
        // if (threadIdx.x % 4 == 0) {
        //     *reinterpret_cast<uint32_t *>(scales_rowwise + scale_offset) = packed_scale;
        // }


        // scaling input
        float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
        ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

        {
            typename CastTraits::outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
            IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput);
            OType2 *rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
            #pragma unroll
            for (int32_t i = 0; i < CastTraits::chunkElems / 2; i++) {
                IType2 in = rInput2[i];
                OType2 out;
                ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                rOutput2[i] = out;
            }
            // write output
            typename CastTraits::outputUnitType * output_units = reinterpret_cast<typename CastTraits::outputUnitType *>(output + offset);
            #pragma unroll
            for (int32_t i = 0; i < CastTraits::numOutUnitsPerChunk; i++) {
                output_units[i] = rOutput[i];
            }
        }
    }
#endif // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}


// 32x32
template <typename IType, typename OType>
struct CastTraits<IType, OType, /*rowwise=*/true, /*colwise=*/true> {
    static constexpr bool isRowwise = true;
    static constexpr bool isColwise = true;
};

// 32x32
template <typename CastTraits,
          std::enable_if_t<CastTraits::isRowwise && CastTraits::isColwise, int> = 0>
__global__ void cast_mxfp8_kernel() {

}

} // namespace spec
} // namespace mxfp8_kernel
} // namespace transformer_engine