#pragma once

#include <cstdlib>
#include <cooperative_groups.h>

namespace transformer_engine {
namespace mxfp8_kernel {
namespace spec {

namespace cg = cooperative_groups;
namespace ptx = transformer_engine::ptx;

namespace {

template <typename IType, typename OType>
struct _Quantized_Limits;

template <>
struct _Quantized_Limits<fp16, fp8e5m2> {
    static constexpr uint16_t max_norm_rcp{0x125};
};

template <>
struct _Quantized_Limits<bf16, fp8e5m2> {
    static constexpr uint16_t max_norm_rcp{0x3792};
};

template <>
struct _Quantized_Limits<fp16, fp8e4m3> {
    static constexpr uint16_t max_norm_rcp{0x1892};
};

template <>
struct _Quantized_Limits<bf16, fp8e4m3> {
    static constexpr uint16_t max_norm_rcp{0x3b12};
};


template <typename OType, typename IType>
__device__ __forceinline__
e8m0_t to_e8m0(IType amax) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) && (defined _ENABLE_MXFMA)
    constexpr uint16_t max_norm_rcp = _Quantized_Limits<IType, OType>::max_norm_rcp;

    float amax_fp32;
    if constexpr (std::is_same_v<IType, fp16>) {
        asm volatile (
            "fma.rn.f32.f16 %0, %1, %2, 0.0;"
            : "=f"(amax_fp32)
            : "h"(reinterpret_cast<uint16_t&>(amax)),
              "h"(max_norm_rcp)
        );
    } else if constexpr (std::is_same_v<IType, bf16>) {
        asm volatile (
            "fma.rn.f32.bf16 %0, %1, %2, 0.0;"
            : "=f"(amax_fp32)
            : "h"(reinterpret_cast<uint16_t&>(amax)),
              "h"(max_norm_rcp)
        );
    }
    return ptx::float_to_e8m0(amax_fp32);
#else
    float amax_fp32 = static_cast<float>(amax);
    return ptx::float_to_e8m0(
        __fmaf_ieee_rn(amax_fp32, Quantized_Limits<OType>::max_norm_rcp, 0.0f)
    );
#endif
}

template <typename T>
struct alignas(4 * sizeof(T)) FPx4 {
    T x, y, z, w;
};
using floatx4 = FPx4<float>;
using bf16x4 = FPx4<bf16>;
using fp16x4 = FPx4<fp16>;
using fp8e4m3x4 = FPx4<fp8e4m3>;
using fp8e5m2x4 = FPx4<fp8e5m2>;

static_assert(sizeof(floatx4) == 16);
static_assert(sizeof(bf16x4) == 8);
static_assert(sizeof(fp16x4) == 8);
static_assert(sizeof(fp8e4m3x4) == 4);
static_assert(sizeof(fp8e5m2x4) == 4);


__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, const bf16x4 &in, const floatx4 &scale) {
    bf16x2 const * in2 = reinterpret_cast<bf16x2 const*>(&in);
    floatx2 const * scale2 = reinterpret_cast<floatx2 const*>(&scale);
    asm volatile (
        "{\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "prmt.b32 val2, 0x0, %1, 0x7632;\n\t"
        "prmt.b32 val1, 0x0, %1, 0x5410;\n\t"
        "prmt.b32 val4, 0x0, %2, 0x7632;\n\t"
        "prmt.b32 val3, 0x0, %2, 0x5410;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 result_1_2;\n\t"
        ".reg.b64 result_3_4;\n\t"
        "fma.rn.f32x2 result_1_2, val_1_2, %3, 0x0;\n\t"
        "fma.rn.f32x2 result_3_4, val_3_4, %4, 0x0;\n\t"
        "mov.b32 {val1, val2}, result_1_2;\n\t"
        "mov.b32 {val3, val4}, result_3_4;\n\t"
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val1, val2, val3, val4}, 0x0\n\t"
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[0])),
          "l"(reinterpret_cast<const uint64_t&>(scale2[1]))
    );
}


} // anonymous namespace


inline bool is_cast_only_enabled() {
    // static bool enabled = [](){
    //     const char* env = std::getenv("ENABLE_CAST_ONLY");
    //     return env != nullptr && (env[0] == '1');
    // }();
    // return enabled;

    // FIXME: when finish debugging, remove this
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
    // using threadLayout = Layout<8, 4>;
    using threadLayout = Layout<1, 32>;
    static constexpr int32_t numThreadsPerChunk = 1;
    static constexpr int32_t warpDimM = threadLayout::M;
    static constexpr int32_t warpDimN = threadLayout::N * chunkElems;
    using inputUnitType = uint4;
    static constexpr int32_t numUnitsPerChunk = chunkElems * sizeof(IType) / sizeof(inputUnitType);
    using outputUnitType = uint4;
    static constexpr int32_t numOutUnitsPerChunk = chunkElems * sizeof(OType) / sizeof(outputUnitType);

    using warpLayout = Layout<4, 1>;
    static constexpr int32_t blockIterDimM = warpLayout::M * warpDimM;
    static constexpr int32_t blockIterDimN = warpLayout::N * warpDimN;

    using iterLayout = Layout<1, 1>;
    static constexpr int32_t blockDimM = iterLayout::M * blockIterDimM;
    static constexpr int32_t blockDimN = iterLayout::N * blockIterDimN;

    static constexpr int32_t numStages = 1;
    static constexpr int32_t numPrefetch = numStages - 1;

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

    uint2 block_coords;
    block_coords.y = blockIdx.y * CastTraits::blockDimM
                + threadIdx.z * CastTraits::warpDimM
                + (threadIdx.x / CastTraits::threadLayout::N);
    block_coords.x = blockIdx.x * CastTraits::blockDimN
                + threadIdx.y * CastTraits::warpDimN
                + (threadIdx.x % CastTraits::threadLayout::N) * CastTraits::chunkElems;

    typename CastTraits::inputUnitType rInput[CastTraits::numStages][CastTraits::numUnitsPerChunk];
    // prologue
    #pragma unroll
    for (int32_t iter = 0; iter < CastTraits::numPrefetch; iter++) {
        int32_t iter_m = iter / CastTraits::iterLayout::N;
        int32_t iter_n = iter % CastTraits::iterLayout::N;

        uint2 coords;
        coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
        coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;

        if (coords.y < rows && coords.x < cols) {
            int32_t offset = coords.y * cols + coords.x;
            typename CastTraits::inputUnitType * input_units = 
                reinterpret_cast<typename CastTraits::inputUnitType *>(input + offset);
        
            #pragma unroll
            for (int32_t i = 0; i < CastTraits::numUnitsPerChunk; i++) {
                rInput[iter][i] = input_units[i];
            }
        }
    }
    // mainloop
    #pragma unroll
    for (int32_t iter = CastTraits::numPrefetch; iter < CastTraits::iterLayout::num; iter++) {
        {
            // load data
            int32_t iter_m = iter / CastTraits::iterLayout::N;
            int32_t iter_n = iter % CastTraits::iterLayout::N;

            uint2 coords;
            coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
            coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;

            if (coords.y < rows && coords.x < cols) {
                int32_t offset = coords.y * cols + coords.x;
                typename CastTraits::inputUnitType * input_units = 
                    reinterpret_cast<typename CastTraits::inputUnitType *>(input + offset);
            
                #pragma unroll
                for (int32_t i = 0; i < CastTraits::numUnitsPerChunk; i++) {
                    rInput[iter % CastTraits::numStages][i] = input_units[i];
                }
            }
        }
        int32_t process_iter = iter - CastTraits::numPrefetch;
        int32_t iter_m = process_iter / CastTraits::iterLayout::N;
        int32_t iter_n = process_iter % CastTraits::iterLayout::N;
        uint2 coords;
        coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
        coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;
        if (coords.y >= rows || coords.x >= cols) { return; }

        if constexpr (std::is_same_v<typename CastTraits::IType, float>) {

        } else {
            IType2 thread_amax2{0.f, 0.f};
            IType2 * rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
            #pragma unroll
            for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
                ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[j]);
            }
            typename CastTraits::IType thread_amax = ptx::get_amax(thread_amax2.x, thread_amax2.y);
            e8m0_t biased_exponent = to_e8m0<typename CastTraits::OType>(thread_amax);
            
            // write biased_exponent
            scales_rowwise[coords.y * scale_stride_rowwise + coords.x / CastTraits::chunkElems] = biased_exponent;

            // scaling input
            float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
            ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
            
            typename CastTraits::outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
            OType2 * rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
            #pragma unroll
            for (int32_t i = 0; i < CastTraits::chunkElems / 2; i++) {
                IType2 in = rInput2[i];
                OType2 out;
                ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                rOutput2[i] = out;
            }
            typename CastTraits::outputUnitType * output_units = 
                reinterpret_cast<typename CastTraits::outputUnitType *>(output + coords.y * cols + coords.x);
            #pragma unroll
            for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
                output_units[j] = rOutput[j];
            }
        }
    }

    // epilogue
    #pragma unroll
    for (int32_t iter = CastTraits::iterLayout::num; iter < CastTraits::iterLayout::num + CastTraits::numPrefetch; iter++) {
        int32_t process_iter = iter - CastTraits::numPrefetch;
        int32_t iter_m = process_iter / CastTraits::iterLayout::N;
        int32_t iter_n = process_iter % CastTraits::iterLayout::N;
        uint2 coords;
        coords.y = block_coords.y + iter_m * CastTraits::blockIterDimM;
        coords.x = block_coords.x + iter_n * CastTraits::blockIterDimN;
        if (coords.y >= rows || coords.x >= cols) { return; }

        if constexpr (std::is_same_v<typename CastTraits::IType, float>) {
            // FIXME: implement
        } else {
            IType2 thread_amax2{0.f, 0.f};
            IType2 * rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
            #pragma unroll
            for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
                ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[j]);
            }
            typename CastTraits::IType thread_amax = ptx::get_amax(thread_amax2.x, thread_amax2.y);
            e8m0_t biased_exponent = to_e8m0<typename CastTraits::OType>(thread_amax);
            
            // write biased_exponent
            scales_rowwise[coords.y * scale_stride_rowwise + coords.x / CastTraits::chunkElems] = biased_exponent;

            // scaling input
            float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
            ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
            
            typename CastTraits::outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
            OType2 * rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
            #pragma unroll
            for (int32_t i = 0; i < CastTraits::chunkElems / 2; i++) {
                IType2 in = rInput2[i];
                OType2 out;
                ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                rOutput2[i] = out;
            }
            typename CastTraits::outputUnitType * output_units = 
                reinterpret_cast<typename CastTraits::outputUnitType *>(output + coords.y * cols + coords.x);
            #pragma unroll
            for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
                output_units[j] = rOutput[j];
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