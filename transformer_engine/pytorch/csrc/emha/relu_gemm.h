/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#pragma once


#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/numeric_types.h>


template <typename T>
inline __device__ uint32_t my_relu(uint32_t);

template <>
inline __device__ uint32_t my_relu<cutlass::half_t>(uint32_t x) {
  constexpr uint32_t zero = 0;
  uint32_t y;
  asm volatile("max.NaN.f16x2 %0, %1, %2;\n" : "=r"(y) : "r"(x), "r"(zero));
  return y;
}

template <>
inline __device__ uint32_t my_relu<cutlass::bfloat16_t>(uint32_t x) {
  constexpr uint32_t zero = 0;
  uint32_t y;
  asm volatile("max.NaN.bf16x2 %0, %1, %2;\n" : "=r"(y) : "r"(x), "r"(zero));
  return y;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Complex elementwise transformation on A operand
    cutlass::ComplexTransform TransformA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Complex elementwise transformation on B operand
    cutlass::ComplexTransform TransformB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    cutlass::gemm::SharedMemoryClearOption SharedMemoryClear =
        cutlass::gemm::SharedMemoryClearOption::kNone>
struct ReluGemmUniversal {
  using DefaultGemmKernel =
      typename cutlass::gemm::kernel::DefaultGemmUniversal<
          ElementA, LayoutA, TransformA, kAlignmentA, ElementB, LayoutB,
          TransformB, kAlignmentB, ElementC, LayoutC, ElementAccumulator,
          OperatorClass, ArchTag, ThreadblockShape, WarpShape, InstructionShape,
          EpilogueOutputOp, ThreadblockSwizzle, Stages, Operator,
          SharedMemoryClear>::DefaultGemmKernel;

  constexpr static bool kAccumulatorsInRowMajor = false;
  using Mma = cutlass::gemm::threadblock::DefaultMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, ThreadblockShape, WarpShape, InstructionShape,
      Stages, Operator, kAccumulatorsInRowMajor, SharedMemoryClear>;

  static_assert(std::is_same<typename Mma::ThreadblockMma,
                             typename DefaultGemmKernel::Mma>::value);

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      cutlass::arch::OpClassTensorOp, Stages, Operator, kAccumulatorsInRowMajor,
      Mma::CacheOpA, Mma::CacheOpB>;

  static_assert(std::is_same<MmaCore, typename Mma::MmaCore>::value);

  using DefaultMmaTensorOp = cutlass::gemm::warp::DefaultMmaTensorOp<
      typename MmaCore::WarpShape, typename MmaCore::InstructionShape,
      typename MmaCore::ElementA, typename MmaCore::SmemLayoutA,
      typename MmaCore::ElementB, typename MmaCore::SmemLayoutB,
      typename MmaCore::ElementC, typename MmaCore::LayoutC,
      typename MmaCore::Operator, MmaCore::WarpCount::kK>;

  using MmaTensorOp = cutlass::gemm::warp::MmaTensorOp<
      typename MmaCore::WarpShape, typename MmaCore::ElementA,
      typename MmaCore::SmemLayoutA, typename MmaCore::ElementB,
      typename MmaCore::SmemLayoutB, typename MmaCore::ElementC,
      typename MmaCore::LayoutC, typename DefaultMmaTensorOp::Policy,
      MmaCore::WarpCount::kK, kAccumulatorsInRowMajor>;

  static_assert(
      std::is_same<MmaTensorOp, typename MmaCore::MmaTensorOp>::value);

  struct MyMmaTensorOp : public MmaTensorOp {
    using Base = MmaTensorOp;

    CUTLASS_DEVICE
    MyMmaTensorOp() : Base() {}

    CUTLASS_DEVICE
    void transform(typename Base::TransformedFragmentA &dst_A,
                   typename Base::TransformedFragmentB &dst_B,
                   typename Base::FragmentA const &A,
                   typename Base::FragmentB const &B) const {
      Base::transform(dst_A, dst_B, A, B);

      constexpr size_t N = Base::TransformedFragmentA::kElements;
      static_assert(N == Base::FragmentA::kElements);
      static_assert(N % 2 == 0);

      uint32_t *tmp = reinterpret_cast<uint32_t *>(&dst_A);
#pragma unroll
      for (int it = 0; it < N / 2; it++) {
        tmp[it] = my_relu<ElementC>(tmp[it]);
      }
    }
  };

  using MmaPolicy = cutlass::gemm::threadblock::MmaPolicy<
      MyMmaTensorOp, cutlass::MatrixShape<0, 0>, cutlass::MatrixShape<0, 0>,
      MmaCore::WarpCount::kK>;

  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, typename Mma::IteratorA,
      typename MmaCore::SmemIteratorA, Mma::MmaCore::kCacheOpA,
      typename Mma::IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC, MmaPolicy, Stages>;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ThreadblockMma, typename DefaultGemmKernel::Epilogue, ThreadblockSwizzle>;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
