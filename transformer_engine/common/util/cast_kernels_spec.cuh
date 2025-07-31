#pragma once

#include <cstdlib>
#include <cuda/ptx>
#include "swizzle.cuh"
#include "state_counter.cuh"

namespace transformer_engine {
namespace mxfp8_kernel {
namespace spec {

namespace ptx = transformer_engine::ptx;
namespace cuda_ptx = cuda::ptx;

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
    if constexpr (std::is_same_v<IType, float>) {
        return ptx::float_to_e8m0(
            __fmaf_ieee_rn(amax, Quantized_Limits<OType>::max_norm_rcp, 0.0f)
        );
    } else {
        float amax_fp32 = static_cast<float>(amax);
        return ptx::float_to_e8m0(
            __fmaf_ieee_rn(amax_fp32, Quantized_Limits<OType>::max_norm_rcp, 0.0f)
        );
    }
#endif
}

#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
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
void mul_cvt_4x(fp8e4m3x4 &out, const bf16x4 &in, const ptx::floatx2 &scale) {
    ptx::bf16x2 const * in2 = reinterpret_cast<ptx::bf16x2 const*>(&in);
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
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, const bf16x4 &in, const ptx::floatx2 &scale) {
    ptx::bf16x2 const * in2 = reinterpret_cast<ptx::bf16x2 const*>(&in);
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
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, const fp16x4 &in, const ptx::floatx2 &scale) {
    ptx::fp16x2 const * in2 = reinterpret_cast<ptx::fp16x2 const*>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b16 val1_f16;\n\t"
        ".reg.b16 val2_f16;\n\t"
        ".reg.b16 val3_f16;\n\t"
        ".reg.b16 val4_f16;\n\t"
        "mov.b32 {val1_f16, val2_f16}, %1;\n\t"
        "mov.b32 {val3_f16, val4_f16}, %2;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "cvt.f32.f16 val1, val1_f16;\n\t"
        "cvt.f32.f16 val2, val2_f16;\n\t"
        "cvt.f32.f16 val3, val3_f16;\n\t"
        "cvt.f32.f16 val4, val4_f16;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, const fp16x4 &in, const ptx::floatx2 &scale) {
    ptx::fp16x2 const * in2 = reinterpret_cast<ptx::fp16x2 const*>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b16 val1_f16;\n\t"
        ".reg.b16 val2_f16;\n\t"
        ".reg.b16 val3_f16;\n\t"
        ".reg.b16 val4_f16;\n\t"
        "mov.b32 {val1_f16, val2_f16}, %1;\n\t"
        "mov.b32 {val3_f16, val4_f16}, %2;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "cvt.f32.f16 val1, val1_f16;\n\t"
        "cvt.f32.f16 val2, val2_f16;\n\t"
        "cvt.f32.f16 val3, val3_f16;\n\t"
        "cvt.f32.f16 val4, val4_f16;\n\t"
        ".reg.b64 val_1_2;\n\t"
        ".reg.b64 val_3_4;\n\t"
        "mov.b64 val_1_2, {val1, val2};\n\t"
        "mov.b64 val_3_4, {val3, val4};\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 val_1_2, val_1_2, %3, zeros;\n\t"
        "fma.rn.f32x2 val_3_4, val_3_4, %3, zeros;\n\t"
        "mov.b64 {val1, val2}, val_1_2;\n\t"
        "mov.b64 {val3, val4}, val_3_4;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out))
        : "r"(reinterpret_cast<const uint32_t&>(in2[0])),
          "r"(reinterpret_cast<const uint32_t&>(in2[1])),
          "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e5m2x4 &out, floatx4 &in, const ptx::floatx2 &scale) {
    ptx::floatx2 * in2 = reinterpret_cast<ptx::floatx2 *>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 %1, %1, %3, zeros;\n\t"
        "fma.rn.f32x2 %2, %2, %3, zeros;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "mov.b64 {val1, val2}, %1;\n\t"
        "mov.b64 {val3, val4}, %2;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e5m2x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e5m2x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out)),
          "+l"(reinterpret_cast<uint64_t&>(in2[0])),
          "+l"(reinterpret_cast<uint64_t&>(in2[1]))
        : "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void mul_cvt_4x(fp8e4m3x4 &out, floatx4 &in, const ptx::floatx2 &scale) {
    ptx::floatx2 * in2 = reinterpret_cast<ptx::floatx2 *>(&in);
    asm volatile (
        "{\n\t"
        ".reg.b64 zeros;\n\t"
        "mov.b64 zeros, {0x0, 0x0};\n\t"
        "fma.rn.f32x2 %1, %1, %3, zeros;\n\t"
        "fma.rn.f32x2 %2, %2, %3, zeros;\n\t"
        ".reg.b32 val1;\n\t"
        ".reg.b32 val2;\n\t"
        ".reg.b32 val3;\n\t"
        ".reg.b32 val4;\n\t"
        "mov.b64 {val1, val2}, %1;\n\t"
        "mov.b64 {val3, val4}, %2;\n\t"
    #if (defined _LOOSE_PRECISION)
        "cvt.rs.satfinite.e4m3x4.f32 %0, {val4, val3, val2, val1}, %4;\n\t"
    #else
        ".reg.b16 r1;\n\t"
        ".reg.b16 r2;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r1, val2, val1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 r2, val4, val3;\n\t"
        "mov.b32 %0, {r1, r2};\n\t"
    #endif
        "}\n\t"
        : "=r"(reinterpret_cast<uint32_t&>(out)),
          "+l"(reinterpret_cast<uint64_t&>(in2[0])),
          "+l"(reinterpret_cast<uint64_t&>(in2[1]))
        : "l"(reinterpret_cast<const uint64_t&>(scale)),
          "r"(0x80008000)
    );
}

__device__ __forceinline__
void abs_max_2x(float &dst, const float &p1, const float &p2, const float &p3) {
    asm volatile (
        "max.abs.f32 %0, %1, %2, %3;"
        : "=f"(dst)
        : "f"(p1), "f"(p2), "f"(p3)
    );
}

#endif // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

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

    static constexpr bool _use_cvt_4x = true;

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
    int32_t rows,
    int32_t cols,
    int32_t scale_stride_rowwise,
    int32_t scale_stride_colwise
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using IType2 = typename ptx::FPx2<typename CastTraits::IType>;
    constexpr int32_t numItersIType2 = sizeof(typename CastTraits::inputUnitType) / sizeof(IType2);
    using OType2 = typename ptx::FPx2<typename CastTraits::OType>;

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
            float thread_amax = 0.f;
            IType2 * rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
            #pragma unroll
            for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
                abs_max_2x(thread_amax, thread_amax, rInput2[j].x, rInput2[j].y);
            }
            e8m0_t biased_exponent = to_e8m0<typename CastTraits::OType>(thread_amax);
            scales_rowwise[coords.y * scale_stride_rowwise + coords.x / CastTraits::chunkElems] = biased_exponent;

            float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
            ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

            typename CastTraits::outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
            if constexpr (CastTraits::_use_cvt_4x) {
                using OType4 = FPx4<typename CastTraits::OType>;
                using IType4 = FPx4<typename CastTraits::IType>;
                IType4 * rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
                OType4 * rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
                #pragma unroll
                for (int32_t j = 0; j < CastTraits::chunkElems / 4; j++) {
                    IType4 in = rInput4[j];
                    OType4 out;
                    mul_cvt_4x(out, in, block_scale_inverse_2x);
                    rOutput4[j] = out;
                }
            } else {
                OType2 * rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
                #pragma unroll
                for (int32_t j = 0; j < CastTraits::chunkElems / 2; j++) {
                    IType2 in = rInput2[j];
                    OType2 out;
                    ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                    rOutput2[j] = out;
                }
            }
            typename CastTraits::outputUnitType * output_units = 
                reinterpret_cast<typename CastTraits::outputUnitType *>(output + coords.y * cols + coords.x);
            #pragma unroll
            for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
                output_units[j] = rOutput[j];
            }
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
            if constexpr (CastTraits::_use_cvt_4x) {
                using OType4 = FPx4<typename CastTraits::OType>;
                using IType4 = FPx4<typename CastTraits::IType>;
                IType4 * rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
                OType4 * rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
                #pragma unroll
                for (int32_t i = 0; i < CastTraits::chunkElems / 4; i++) {
                    IType4 in = rInput4[i];
                    OType4 out;
                    mul_cvt_4x(out, in, block_scale_inverse_2x);
                    rOutput4[i] = out;
                }
            } else {
                OType2 * rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
                #pragma unroll
                for (int32_t i = 0; i < CastTraits::chunkElems / 2; i++) {
                    IType2 in = rInput2[i];
                    OType2 out;
                    ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                    rOutput2[i] = out;
                }
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
            float thread_amax = 0.f;
            IType2 * rInput2 = reinterpret_cast<IType2 *>(&rInput[process_iter % CastTraits::numStages]);
            #pragma unroll
            for (int32_t j = 0; j < numItersIType2 * CastTraits::numUnitsPerChunk; j++) {
                abs_max_2x(thread_amax, thread_amax, rInput2[j].x, rInput2[j].y);
            }
            e8m0_t biased_exponent = to_e8m0<typename CastTraits::OType>(thread_amax);
            scales_rowwise[coords.y * scale_stride_rowwise + coords.x / CastTraits::chunkElems] = biased_exponent;

            float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
            ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};

            typename CastTraits::outputUnitType rOutput[CastTraits::numOutUnitsPerChunk];
            if constexpr (CastTraits::_use_cvt_4x) {
                using OType4 = FPx4<typename CastTraits::OType>;
                using IType4 = FPx4<typename CastTraits::IType>;
                IType4 * rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
                OType4 * rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
                #pragma unroll
                for (int32_t j = 0; j < CastTraits::chunkElems / 4; j++) {
                    IType4 in = rInput4[j];
                    OType4 out;
                    mul_cvt_4x(out, in, block_scale_inverse_2x);
                    rOutput4[j] = out;
                }
            } else {
                OType2 * rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
                #pragma unroll
                for (int32_t j = 0; j < CastTraits::chunkElems / 2; j++) {
                    IType2 in = rInput2[j];
                    OType2 out;
                    ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                    rOutput2[j] = out;
                }
            }
            typename CastTraits::outputUnitType * output_units = 
                reinterpret_cast<typename CastTraits::outputUnitType *>(output + coords.y * cols + coords.x);
            #pragma unroll
            for (int32_t j = 0; j < CastTraits::numOutUnitsPerChunk; j++) {
                output_units[j] = rOutput[j];
            }
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
            if constexpr (CastTraits::_use_cvt_4x) {
                using OType4 = FPx4<typename CastTraits::OType>;
                using IType4 = FPx4<typename CastTraits::IType>;
                IType4 * rInput4 = reinterpret_cast<IType4 *>(&rInput[process_iter % CastTraits::numStages]);
                OType4 * rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
                #pragma unroll
                for (int32_t i = 0; i < CastTraits::chunkElems / 4; i++) {
                    IType4 in = rInput4[i];
                    OType4 out;
                    mul_cvt_4x(out, in, block_scale_inverse_2x);
                    rOutput4[i] = out;
                }
            } else {
                OType2 * rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
                #pragma unroll
                for (int32_t i = 0; i < CastTraits::chunkElems / 2; i++) {
                    IType2 in = rInput2[i];
                    OType2 out;
                    ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                    rOutput2[i] = out;
                }
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
template <typename _IType, typename _OType>
struct CastTraits<_IType, _OType, /*rowwise=*/true, /*colwise=*/true> {
    static constexpr bool isRowwise = true;
    static constexpr bool isColwise = true;
    using IType = _IType;
    using OType = _OType;

    static constexpr int32_t rowChunkElems = 32;
    static constexpr int32_t colChunkElems = 32;
    static constexpr int32_t transChunkElems = 4;

    using rowThreadLayout = Layout<32, 1>; // 32x1 
    using colThreadLayout = Layout<rowThreadLayout::N, rowThreadLayout::M>; // 1x32
    static_assert(rowThreadLayout::num == colThreadLayout::num, 
        "rowThreadLayout::num must be equal to colThreadLayout::num");
    static_assert(rowThreadLayout::num == 32, "rowThreadLayout::num must be 32");
    
    using rowWarpDim = Layout<rowThreadLayout::M, rowThreadLayout::N * rowChunkElems>;
    using colWarpDim = Layout<colThreadLayout::M * colChunkElems, colThreadLayout::N>;
    using warpDim = Layout<std::max(rowWarpDim::M, colWarpDim::M), std::max(rowWarpDim::N, colWarpDim::N)>;

    using warpLayout = Layout<4, 1>;
    using blockIterDim = Layout<warpLayout::M * warpDim::M, warpLayout::N * warpDim::N>;

    using iterLayout = Layout<1, 4>;
    using blockDIM = Layout<iterLayout::M * blockIterDim::M, iterLayout::N * blockIterDim::N>;

    static constexpr int32_t numStages = 3;

    static constexpr int32_t numRegStages = 2;
    static constexpr int32_t numPrefetch = numRegStages - 1;

    using inputUnitType = uint4;
    static constexpr int32_t rowNumUnitsPerChunk = rowChunkElems * sizeof(IType) / sizeof(inputUnitType);
    // TODO: set condition for float
    using inputElemSwz = swz::Swizzle<2, 3, 3>;
    using inputUnitSwz = swz::Swizzle<2, 0, 3>;

    using colIndexSwz = swz::Swizzle<5, 0, 5>;

    using rowOutputUnitType = uint4;
    static constexpr int32_t rowNumOutUnitsPerChunk = rowChunkElems * sizeof(OType) / sizeof(rowOutputUnitType);

    using colOutputSwz = swz::Swizzle<1, 4, 3>;

    static constexpr bool _use_cvt_4x = true;

    static constexpr int32_t numThreads = (warpLayout::num + 1) * 32;
    static constexpr size_t smemPerWarp = warpDim::num * sizeof(IType);
    static constexpr size_t smemPerBlock = smemPerWarp * warpLayout::num;

    static constexpr size_t smemOutputPerWarp = warpDim::num * sizeof(OType);
    static constexpr size_t smemOutputPerBlock = smemOutputPerWarp * warpLayout::num;

    static constexpr size_t smem = smemPerBlock * numStages + smemOutputPerBlock * numStages + 128ul;
};

#define ALIGN_TO(x, align) (((x) + (align) - 1) & ~((align) - 1))

// 32x32
template <typename CastTraits,
          std::enable_if_t<CastTraits::isRowwise && CastTraits::isColwise, int> = 0>
__global__ void cast_mxfp8_kernel(
    const __grid_constant__ CUtensorMap tensor_map_input,
    typename CastTraits::OType * __restrict__ rowwise_output,
    const __grid_constant__ CUtensorMap tensor_map_colwise_output,
    e8m0_t * __restrict__ scales_rowwise,
    e8m0_t * __restrict__ scales_colwise,
    int32_t rows,
    int32_t cols,
    int32_t scale_stride_rowwise,
    int32_t scale_stride_colwise
) {
#if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    using IType2 = typename ptx::FPx2<typename CastTraits::IType>;
    using OType2 = typename ptx::FPx2<typename CastTraits::OType>;
    constexpr int32_t numItersIType2 = sizeof(typename CastTraits::inputUnitType) / sizeof(IType2);

    int32_t warpId = threadIdx.y;
    int32_t leader = ptx::elect_one_sync();
    uint2 block_coords;
    block_coords.y = blockIdx.y * CastTraits::blockDIM::M;
    block_coords.x = blockIdx.x * CastTraits::blockDIM::N;

    extern __shared__ char smem[];
    char *smemAligned = (char*)(ALIGN_TO((intptr_t)smem, 128));
    typename CastTraits::IType *sInput = reinterpret_cast<typename CastTraits::IType *>(smemAligned);
    typename CastTraits::IType *sInputWarp = sInput + warpId * CastTraits::warpDim::num;
    typename CastTraits::OType *sColOutput = reinterpret_cast<typename CastTraits::OType *>(
        sInput + CastTraits::blockIterDim::num * CastTraits::numStages);
    typename CastTraits::OType *sColOutputWarp = sColOutput + warpId * CastTraits::warpDim::num;

    // TODO: maybe we can assign a different barrier for each warp
    __shared__ uint64_t producer[CastTraits::numStages], consumer[CastTraits::numStages];

    if (warpId == 0 && leader) {
        #pragma unroll
        for (int32_t i = 0; i < CastTraits::numStages; i++) {
            cuda_ptx::mbarrier_init(&producer[i], 1);
            cuda_ptx::mbarrier_init(&consumer[i], CastTraits::warpLayout::num * 32);
        }
        cuda_ptx::fence_mbarrier_init(cuda_ptx::sem_release, cuda_ptx::scope_cluster);
    }
    __syncthreads();

    if (warpId == CastTraits::warpLayout::num) {
        
        PipeState<CastTraits::numStages, true> write_state;
        if (leader) {
            #pragma unroll
            for (int32_t iter = 0; iter < CastTraits::iterLayout::num; iter++) {
                int32_t iter_m = iter / CastTraits::iterLayout::N;
                int32_t iter_n = iter % CastTraits::iterLayout::N;

                if (block_coords.x + iter_n * CastTraits::blockIterDim::N >= cols ||
                    block_coords.y + iter_m * CastTraits::blockIterDim::M >= rows) {
                    break;
                }

                while (!cuda_ptx::mbarrier_try_wait_parity(
                    &consumer[write_state.index()], 
                    write_state.phase()));

                cuda_ptx::cp_async_bulk_tensor(
                    cuda_ptx::space_shared,
                    cuda_ptx::space_global,
                    /*dstMem=*/sInput + write_state.index() * CastTraits::blockIterDim::num,
                    &tensor_map_input,
                    /*tensorCoords=*/{int32_t(block_coords.x + iter_n * CastTraits::blockIterDim::N),
                                      int32_t(block_coords.y + iter_m * CastTraits::blockIterDim::M)},
                    &producer[write_state.index()]
                );
                cuda_ptx::mbarrier_arrive_expect_tx(
                    cuda_ptx::sem_release,
                    cuda_ptx::scope_cta,
                    cuda_ptx::space_shared,
                    &producer[write_state.index()],
                    CastTraits::blockIterDim::num * sizeof(typename CastTraits::IType)
                );
                write_state++;
            }
        }

    } else {
        PipeState<CastTraits::numStages> read_state;
        for (int32_t iter = 0; iter < CastTraits::iterLayout::num; iter++) {
            int32_t iter_m = iter / CastTraits::iterLayout::N;
            int32_t iter_n = iter % CastTraits::iterLayout::N;

            if (block_coords.x + iter_n * CastTraits::blockIterDim::N >= cols ||
                block_coords.y + iter_m * CastTraits::blockIterDim::M >= rows) {
                break;
            }

            while (!cuda_ptx::mbarrier_try_wait_parity(
                &producer[read_state.index()],
                read_state.phase()));
            {
                // rowwise
                {
                    typename CastTraits::inputUnitType *sInputUnit = 
                        reinterpret_cast<typename CastTraits::inputUnitType *>(
                            sInputWarp + CastTraits::blockIterDim::num * read_state.index());
                    typename CastTraits::inputUnitType rInput[CastTraits::rowNumUnitsPerChunk];
                    #pragma unroll
                    for (int32_t i = 0; i < CastTraits::rowNumUnitsPerChunk; i++) {
                        int32_t offset = threadIdx.x * CastTraits::rowNumUnitsPerChunk + i;
                        rInput[i] = sInputUnit[CastTraits::inputUnitSwz::swz(offset)];
                    }
                    if constexpr (std::is_same_v<typename CastTraits::IType, float>) {

                    } else {
                        IType2 thread_amax2{0.f, 0.f};
                        IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput);
                        #pragma unroll
                        for (int32_t i = 0; i < numItersIType2 * CastTraits::rowNumUnitsPerChunk; i++) {
                            ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[i]);
                        }
                        typename CastTraits::IType thread_amax = ptx::get_amax(thread_amax2.x, thread_amax2.y);
                        e8m0_t biased_exponent = to_e8m0<typename CastTraits::OType>(thread_amax);
                        
                        uint2 coords;
                        coords.y = block_coords.y 
                                    + (warpId / CastTraits::warpLayout::N) * CastTraits::warpDim::M
                                    + (threadIdx.x / CastTraits::rowThreadLayout::N)
                                    + iter_m * CastTraits::blockIterDim::M;
                        coords.x = block_coords.x
                                    + (warpId % CastTraits::warpLayout::N) * CastTraits::warpDim::N
                                    + (threadIdx.x % CastTraits::rowThreadLayout::N) * CastTraits::rowChunkElems
                                    + iter_n * CastTraits::blockIterDim::N;
                        scales_rowwise[coords.y * scale_stride_rowwise + coords.x / CastTraits::rowChunkElems] = biased_exponent;

                        float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
                        ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
                        typename CastTraits::rowOutputUnitType rOutput[CastTraits::rowNumOutUnitsPerChunk];
                        if constexpr (CastTraits::_use_cvt_4x) {
                            using OType4 = FPx4<typename CastTraits::OType>;
                            using IType4 = FPx4<typename CastTraits::IType>;
                            IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput);
                            OType4 *rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
                            #pragma unroll
                            for (int32_t i = 0; i < CastTraits::rowChunkElems / 4; i++) {
                                IType4 in = rInput4[i];
                                OType4 out;
                                mul_cvt_4x(out, in, block_scale_inverse_2x);
                                rOutput4[i] = out;
                            }
                        } else {
                            OType2 *rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
                            #pragma unroll
                            for (int32_t i = 0; i < CastTraits::rowChunkElems / 2; i++) {
                                IType2 in = rInput[i];
                                OType2 out;
                                ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                                rOutput2[i] = out;
                            }
                        }
                        typename CastTraits::rowOutputUnitType *rowOutputUnit = 
                            reinterpret_cast<typename CastTraits::rowOutputUnitType *>(
                                rowwise_output + coords.y * cols + coords.x);
                        #pragma unroll
                        for (int32_t i = 0; i < CastTraits::rowNumOutUnitsPerChunk; i++) {
                            rowOutputUnit[i] = rOutput[i];
                        }
                    }
                }
                // colwise
                {
                    typename CastTraits::IType rInput[CastTraits::colChunkElems];
                    typename CastTraits::OType rOutput[CastTraits::colChunkElems];
                    #pragma unroll
                    for (int32_t i = 0; i < CastTraits::colChunkElems; i++) {
                        int32_t row = CastTraits::colIndexSwz::swz(i * CastTraits::rowChunkElems + threadIdx.x) - i * CastTraits::rowChunkElems;
                        int32_t offset = row * CastTraits::rowChunkElems + threadIdx.x;
                        rInput[i] = sInputWarp[CastTraits::inputElemSwz::swz(offset)];
                    }
                    cuda_ptx::mbarrier_arrive_expect_tx(
                        cuda_ptx::sem_release,
                        cuda_ptx::scope_cta,
                        cuda_ptx::space_shared,
                        &consumer[read_state.index()],
                        0u
                    );

                    if constexpr (std::is_same_v<typename CastTraits::IType, float>) {

                    } else {
                        IType2 thread_amax2{0.f, 0.f};
                        IType2 *rInput2 = reinterpret_cast<IType2 *>(&rInput);
                        #pragma unroll
                        for (int32_t i = 0; i < CastTraits::colChunkElems / 2; i++) {
                            ptx::abs_max_2x(thread_amax2, thread_amax2, rInput2[i]);
                        }
                        typename CastTraits::IType thread_amax = ptx::get_amax(thread_amax2.x, thread_amax2.y);
                        e8m0_t biased_exponent = to_e8m0<typename CastTraits::OType>(thread_amax);
                        uint2 coords;
                        coords.y = block_coords.y 
                                    + (warpId / CastTraits::warpLayout::N) * CastTraits::warpDim::M
                                    + (threadIdx.x / CastTraits::colThreadLayout::N) * CastTraits::colChunkElems
                                    + iter_m * CastTraits::blockIterDim::M;
                        coords.x = block_coords.x
                                    + (warpId % CastTraits::warpLayout::N) * CastTraits::warpDim::N
                                    + (threadIdx.x % CastTraits::colThreadLayout::N)
                                    + iter_n * CastTraits::blockIterDim::N;
                        scales_colwise[(coords.y / CastTraits::colChunkElems) * scale_stride_colwise + coords.x] = biased_exponent;

                        float block_scale_inverse = ptx::exp2f_rcp(biased_exponent);
                        ptx::floatx2 block_scale_inverse_2x{block_scale_inverse, block_scale_inverse};
                        if constexpr (CastTraits::_use_cvt_4x) {
                            using OType4 = FPx4<typename CastTraits::OType>;
                            using IType4 = FPx4<typename CastTraits::IType>;
                            IType4 *rInput4 = reinterpret_cast<IType4 *>(&rInput);
                            OType4 *rOutput4 = reinterpret_cast<OType4 *>(&rOutput);
                            #pragma unroll
                            for (int32_t i = 0; i < CastTraits::colChunkElems / 4; i++) {
                                IType4 in = rInput4[i];
                                OType4 out;
                                mul_cvt_4x(out, in, block_scale_inverse_2x);
                                rOutput4[i] = out;
                            }
                        } else {
                            OType2 *rOutput2 = reinterpret_cast<OType2 *>(&rOutput);
                            #pragma unroll
                            for (int32_t i = 0; i < CastTraits::colChunkElems / 2; i++) {
                                IType2 in = rInput[i];
                                OType2 out;
                                ptx::mul_cvt_2x(out, in, block_scale_inverse_2x);
                                rOutput2[i] = out;
                            }
                        }
                        ptx::cp_async_bulk_wait_group_read<CastTraits::numStages - 1>();
                        ptx::numbered_barrier_sync(CastTraits::warpLayout::num * 32, 8);
                        #pragma unroll
                        for (int32_t i = 0; i < CastTraits::colChunkElems; i++) {
                            int32_t row = CastTraits::colIndexSwz::swz(i * CastTraits::rowChunkElems + threadIdx.x) - i * CastTraits::rowChunkElems;
                            int32_t offset = row * CastTraits::rowChunkElems + threadIdx.x;
                            sColOutputWarp[CastTraits::colOutputSwz::swz(offset)] = rOutput[i];
                        }
                        ptx::fence_proxy_async_shared_cta();
                        ptx::numbered_barrier_sync(CastTraits::warpLayout::num * 32, 8);
                        if (warpId == 0 && leader) {
                            cuda_ptx::cp_async_bulk_tensor(
                                cuda_ptx::space_global,
                                cuda_ptx::space_shared,
                                &tensor_map_colwise_output,
                                /*tensorCoords=*/{int32_t(block_coords.x + iter_n * CastTraits::blockIterDim::N),
                                                  int32_t(block_coords.y + iter_m * CastTraits::blockIterDim::M)},
                                /*srcMem=*/sColOutput + read_state.index() * CastTraits::blockIterDim::num
                            );
                        }
                        cuda_ptx::cp_async_bulk_commit_group();
                    }
                }
            }
            read_state++;
        }
        ptx::cp_async_bulk_wait_group_read<0>();
    }



#endif // #if (defined __CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
}

} // namespace spec
} // namespace mxfp8_kernel
} // namespace transformer_engine