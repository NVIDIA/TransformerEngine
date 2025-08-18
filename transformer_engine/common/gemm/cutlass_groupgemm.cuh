/***************************************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 **************************************************************************************************/

//
// Copyright (c) 2025 Shopee Inc. All Rights Reserved.
//

/**
 * @file: cutlass_groupgemm.cuh
 * @author: min.yang@shopee.com, yangfan.bai@shopee.com, finch.li@shopee.com
 * @date: 2025-08-08 16:20:00
 * @brief: cutlass group gemm kernel.
 **/

#include <transformer_engine/transformer_engine.h>

#include <cub/cub.cuh>
#include <type_traits>

#include "../common.h"
#include "../util/logging.h"
#include "common/util/system.h"
#include "cute/tensor.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/complex.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

namespace grouped_gemm {

template <bool transa>
using GroupedGemmInputALayout =
    std::conditional_t<transa, ::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor>;

template <bool transb>
using GroupedGemmInputBLayout =
    std::conditional_t<transb, ::cutlass::layout::ColumnMajor, ::cutlass::layout::RowMajor>;

using ProblemShapeType = cute::Shape<int, int, int>;
using ProblemShape = cutlass::gemm::GroupProblemShape<ProblemShapeType>;  // <M,N,K> per group
template <typename ScheduleConfig>
struct GemmGivenSchedule {
  using ElementA = typename ScheduleConfig::DataType;  // Element type for A matrix operand
  using ElementB = typename ScheduleConfig::DataType;  // Element type for B matrix operand
  using ElementC = typename ScheduleConfig::DataType;  // Element type for C and D matrix operands

  // A matrix configuration
  using LayoutA = typename ScheduleConfig::LayoutA;  // Layout type for A matrix operand
  static constexpr int AlignmentA =
      128 / cutlass::sizeof_bits<
                ElementA>::value;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using LayoutB = typename ScheduleConfig::LayoutB;  // Layout type for B matrix operand
  static constexpr int AlignmentB =
      128 / cutlass::sizeof_bits<
                ElementB>::value;  // Alignment of B matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using LayoutC = typename ScheduleConfig::LayoutC;  // Layout type for C and D matrix operands
  static constexpr int AlignmentC =
      128 / cutlass::sizeof_bits<
                ElementC>::value;  // Alignment of C matrix in units of elements (up to 16 bytes)

  // Core kernel configurations
  using ElementAccumulator = float;  // Element type for internal accumulation
  using ArchTag =
      cutlass::arch::Sm90;  // Tag indicating the minimum SM that supports the intended feature
  using OperatorClass = cutlass::arch::OpClassTensorOp;  // Operator class tag
  using StageCountType =
      cutlass::gemm::collective::StageCountAuto;  // Stage count maximized based on the tile size

  using TileShape = typename ScheduleConfig::TileShape;  // Threadblock-level tile size
  using ClusterShape =
      typename ScheduleConfig::ClusterShape;  // Shape of the threadblocks in a cluster
  using KernelSchedule = typename ScheduleConfig::KernelSchedule;      // Kernel to launch
  using EpilogueSchedule = typename ScheduleConfig::EpilogueSchedule;  // Epilogue to launch

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC*, AlignmentC, ElementC, LayoutC*, AlignmentC, EpilogueSchedule,
      cutlass::epilogue::fusion::LinearCombination<ElementC, ElementAccumulator>>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, ElementA, LayoutA*, AlignmentA, ElementB, LayoutB*, AlignmentB,
      ElementAccumulator, TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      KernelSchedule>::CollectiveOp;

  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

template <typename DataType_, bool trans_a, bool trans_b>
struct ScheduleConfig {
  using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong;
  using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_2, cute::_1>;

  // TODO(Alan): Add tuning for different scenarios to select the optimal configuration,
  //             as the current configuration may not be the best.

  // using KernelSchedule = cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative;
  // using EpilogueSchedule = cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative;
  // using TileShape = Shape<cute::_256, cute::_128, cute::_128>;
  // using ClusterShape = Shape<cute::_1, cute::_2, cute::_1>;

  using LayoutA = GroupedGemmInputALayout<trans_a>;
  using LayoutB = GroupedGemmInputBLayout<trans_b>;
  using LayoutC = cutlass::layout::RowMajor;
  using DataType = DataType_;
};

template <typename DataType_, bool trans_a, bool trans_b>
using GemmGrouped = typename GemmGivenSchedule<ScheduleConfig<DataType_, trans_a, trans_b>>::Gemm;

template <typename GemmT, typename ElementA, typename ElementB, typename ElementC, typename StrideA,
          typename StrideB, typename StrideC>
typename GemmT::Arguments MakeArguments(int num_experts, void* problem_sizes_host,
                                        void* problem_sizes, const ElementA** ptr_A,
                                        StrideA* stride_A, const ElementB** ptr_B,
                                        StrideB* stride_B, ElementC** ptr_C, StrideC* stride_C,
                                        float alpha, float beta, int device, int math_sm_count) {
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.

  cutlass::KernelHardwareInfo kernel_hw_info =
      cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename GemmT::GemmKernel>(
          device, math_sm_count);

  typename GemmT::Arguments arguments;
  decltype(arguments.epilogue.thread) fusion_args;

  fusion_args.alpha = alpha;
  fusion_args.beta = beta;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.beta_ptr_array = nullptr;
  // Single alpha and beta for all groups
  fusion_args.dAlpha = {cute::_0{}, cute::_0{}, 0};
  fusion_args.dBeta = {cute::_0{}, cute::_0{}, 0};

  arguments = typename GemmT::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, (ProblemShapeType*)problem_sizes, (ProblemShapeType const*)problem_sizes_host},
      {ptr_A, stride_A, ptr_B, stride_B},
      {
          fusion_args,
          (beta > 0.0) ? (const ElementC**)ptr_C : nullptr,
          stride_C,
          ptr_C,
          stride_C,
      },
      kernel_hw_info};

  return arguments;
}

template <typename StrideT>
StrideT infer_stride(int rows, int cols, bool is_col_major) {
  if (is_col_major) {
    return cutlass::make_cute_packed_stride(StrideT{}, {rows, cols, 0});  // ColumnMajor
  } else {
    return cutlass::make_cute_packed_stride(StrideT{}, {rows, cols, 0});  // RowMajor
  }
}

template <typename T>
inline __device__ __host__ T DIV_UP(T m, T n) {
  return (m + n - 1) / n;
}

template <typename T>
void debug_type() {
  std::cout << typeid(T).name() << std::endl;
}

int64_t inline getGemmCoordSize(int64_t num_gemms) {
  return (int64_t)(DIV_UP(num_gemms * sizeof(ProblemShapeType), 128UL) * 128UL);
}

int64_t inline getPtrSize(int64_t num_gemms) {
  return (int64_t)(DIV_UP(num_gemms * sizeof(half*), 128UL) * 128UL);
}

int64_t inline getLddSize(int64_t num_gemms) {
  return (int64_t)(DIV_UP(num_gemms * sizeof(int64_t), 128UL) * 128UL);
}

template <bool trans_a, bool trans_b, typename Element>
void CutlassGroupedGemm(bool transa, bool transb, const NVTETensor* A, const NVTETensor* B,
                        NVTETensor* D, NVTETensor* workspace, float alpha, float beta,
                        int num_gemms, cudaStream_t stream, int device, int math_sm_count) {
  using Gemm = GemmGrouped<Element, trans_a, trans_b>;
  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementC = typename Gemm::ElementC;

  using StrideA = typename Gemm::GemmKernel::InternalStrideA;
  using StrideB = typename Gemm::GemmKernel::InternalStrideB;
  using StrideC = typename Gemm::GemmKernel::InternalStrideC;

  typename Gemm::Arguments arguments;
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  auto gemm_coord_size = getGemmCoordSize(num_gemms);
  auto ptr_size = getPtrSize(num_gemms);
  auto ldd_size = getLddSize(num_gemms);
  auto param_workspace_size = 3 * ptr_size + 3 * ldd_size + gemm_coord_size;

  auto total_workspace_size = param_workspace_size + workspace_size;
  transformer_engine::Tensor* wspace = transformer_engine::convertNVTETensor(workspace[0]);

  NVTE_CHECK(total_workspace_size < wspace->numel(), "Insufficient workspace[0] size: required=",
             static_cast<int64_t>(total_workspace_size),
             ", available=", static_cast<int64_t>(wspace->numel()), " for CUTLASS grouped GEMM.");

  char* all_workspace = (char*)(wspace->data.dptr);

  char* workspace_ptr = nullptr;

  char* host_workspace = (char*)std::malloc(param_workspace_size);

  ProblemShapeType* problem_sizes_host = reinterpret_cast<ProblemShapeType*>(host_workspace);

  ElementA** ptr_A_host = reinterpret_cast<ElementA**>(host_workspace + gemm_coord_size);
  ElementB** ptr_B_host = reinterpret_cast<ElementB**>(host_workspace + gemm_coord_size + ptr_size);
  ElementC** ptr_C_host =
      reinterpret_cast<ElementC**>(host_workspace + gemm_coord_size + 2 * ptr_size);
  StrideA* lda_host =
      reinterpret_cast<StrideA*>(host_workspace + gemm_coord_size + 3 * ptr_size + 0 * ldd_size);
  StrideB* ldb_host =
      reinterpret_cast<StrideB*>(host_workspace + gemm_coord_size + 3 * ptr_size + 1 * ldd_size);
  StrideC* ldc_host =
      reinterpret_cast<StrideC*>(host_workspace + gemm_coord_size + 3 * ptr_size + 2 * ldd_size);

  for (size_t i = 0; i < num_gemms; i++) {
    const transformer_engine::Tensor* inputA = transformer_engine::convertNVTETensorCheck(A[i]);
    const transformer_engine::Tensor* inputB = transformer_engine::convertNVTETensorCheck(B[i]);
    transformer_engine::Tensor* outputD = transformer_engine::convertNVTETensor(D[i]);

    const int m = transa ? inputA->data.shape[1] : inputA->data.shape[0];
    const int k = transa ? inputA->data.shape[0] : inputA->data.shape[1];
    const int n = transb ? inputB->data.shape[0] : inputB->data.shape[1];

    auto problem = ProblemShapeType(m, n, k);
    problem_sizes_host[i] = problem;

    ptr_A_host[i] = (ElementA*)inputA->data.dptr;
    ptr_B_host[i] = (ElementB*)inputB->data.dptr;
    ptr_C_host[i] = (ElementC*)outputD->data.dptr;

    lda_host[i] =
        infer_stride<StrideA>(m, k, std::is_same_v<LayoutA, cutlass::layout::ColumnMajor>);
    ldb_host[i] =
        infer_stride<StrideB>(n, k, std::is_same_v<LayoutB, cutlass::layout::ColumnMajor>);
    ldc_host[i] =
        infer_stride<StrideC>(m, n, std::is_same_v<LayoutC, cutlass::layout::ColumnMajor>);
  }

  cudaMemcpyAsync(all_workspace, host_workspace, param_workspace_size, cudaMemcpyHostToDevice,
                  stream);

  char* param_workspace = all_workspace;
  ProblemShapeType* problem_sizes_device = reinterpret_cast<ProblemShapeType*>(param_workspace);
  const ElementA** ptr_A =
      reinterpret_cast<const ElementA**>((char*)param_workspace + gemm_coord_size);
  const ElementB** ptr_B =
      reinterpret_cast<const ElementB**>((char*)param_workspace + gemm_coord_size + 1 * ptr_size);
  ElementC** ptr_C =
      reinterpret_cast<ElementC**>((char*)param_workspace + gemm_coord_size + 2 * ptr_size);

  StrideA* lda = reinterpret_cast<StrideA*>((char*)param_workspace + gemm_coord_size +
                                            3 * ptr_size + 0 * ldd_size);
  StrideB* ldb = reinterpret_cast<StrideB*>((char*)param_workspace + gemm_coord_size +
                                            3 * ptr_size + 1 * ldd_size);
  StrideC* ldc = reinterpret_cast<StrideC*>((char*)param_workspace + gemm_coord_size +
                                            3 * ptr_size + 2 * ldd_size);

  workspace_ptr = all_workspace + param_workspace_size;

  arguments = MakeArguments<Gemm, ElementA, ElementB, ElementC, StrideA, StrideB, StrideC>(
      num_gemms, problem_sizes_host, problem_sizes_device, ptr_A, lda, ptr_B, ldb, ptr_C, ldc,
      alpha, beta, device, math_sm_count);

  Gemm gemm;

  // Check can implement the kernel.
  if (gemm.can_implement(arguments) != cutlass::Status::kSuccess) {
    NVTE_CHECK(false, "Failed to implement CUTLASS Grouped GEMM");
  }

  // Initialize the kernel.
  if (gemm.initialize(arguments, workspace_ptr) != cutlass::Status::kSuccess) {
    NVTE_CHECK(false, "Failed to initialize CUTLASS Grouped GEMM");
  }

  // Execute the kernel in the current stream.
  if (gemm.run(stream) != cutlass::Status::kSuccess) {
    NVTE_CHECK(false, "Failed to run CUTLASS Grouped GEMM");
  }

  std::free(host_workspace);
}

}  // namespace grouped_gemm

void cutlass_grouped_gemm(const NVTETensor* A, const NVTETensor* B, NVTETensor* D, int num_gemms,
                          bool transa, bool transb, bool grad, NVTETensor* workspace,
                          bool accumulate, int device, int math_sm_count, cudaStream_t stream);
