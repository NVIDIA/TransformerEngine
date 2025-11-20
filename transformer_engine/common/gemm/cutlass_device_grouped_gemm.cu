/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <float.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cstdint>
#include <mutex>
#include <vector>

#include "../common.h"
#include "../util/handle_manager.h"
#include "../util/logging.h"
#include "common/util/cuda_runtime.h"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/group_array_problem_shape.hpp"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/device_memory.h"

using namespace cute;

/*****
 * Fprop and Dgrad
 ******/

template <typename Sm1xxBlkScaledConfig, typename UnderlyingProblemShape, typename ElementA,
          typename ElementD, typename ElementSF, typename StrideA, typename StrideB,
          typename StrideD, typename LayoutSFA, typename LayoutSFB, bool transB>
__global__ void setGroupedGemmArguments(int num_experts, const int64_t *gemm_m_per_expert,
                                        int gemm_n, int gemm_k, ElementA *ptr_A, ElementSF *ptr_SFA,
                                        ElementD *ptr_D, UnderlyingProblemShape *problem_sizes,
                                        ElementA **ptr_A_list, ElementSF **ptr_SFA_list,
                                        StrideA *stride_A_list, LayoutSFA *layout_SFA_list,
                                        StrideB *stride_B_list, LayoutSFB *layout_SFB_list,
                                        ElementD **ptr_D_list, StrideD *stride_D_list) {
  int m_offset = 0;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    for (int expert_id = 0; expert_id < num_experts; expert_id++) {
      int gemm_m = int(gemm_m_per_expert[expert_id]);
      problem_sizes[expert_id] = cute::make_shape(gemm_m, gemm_n, gemm_k);

      ptr_A_list[expert_id] = ptr_A + m_offset * gemm_k;
      ptr_SFA_list[expert_id] = ptr_SFA + m_offset * ((gemm_k + 127) / 128 * 4);
      stride_A_list[expert_id] = cute::make_stride(int64_t(gemm_k), _1{}, _0{});
      layout_SFA_list[expert_id] =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(gemm_m, gemm_n, gemm_k, 1));

      if constexpr (transB) {
        stride_B_list[expert_id] = cute::make_stride(int64_t(gemm_k), _1{}, _0{});
      } else {
        stride_B_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_n), _0{});
      }
      layout_SFB_list[expert_id] =
          Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(gemm_m, gemm_n, gemm_k, 1));

      ptr_D_list[expert_id] = ptr_D + m_offset * gemm_n;
      stride_D_list[expert_id] = cute::make_stride(int64_t(gemm_n), _1{}, _0{});

      m_offset += gemm_m;
    }
  }
}

template <typename ElementInput, typename ElementSF, typename ElementC, bool DGrad, bool TransB>
void generic_moe_gemm_kernelLauncher(ElementInput *A, ElementSF *SFA, const void **ptr_B_list,
                                     const void **ptr_SFB_list, ElementC *D,
                                     const int64_t *gemm_m_per_expert, int gemm_n, int gemm_k,
                                     int num_experts, size_t workspaceSize, void *workspace,
                                     cudaStream_t stream, int *kernel_occupancy = nullptr) {
  static_assert(cute::is_same_v<ElementInput, cutlass::float_e4m3_t> ||
                    cute::is_same_v<ElementInput, cutlass::float_e5m2_t>,
                "Unsupported input type. Expected e4m3 or e5m2.");
  static_assert(cute::is_same_v<ElementSF, cutlass::float_ue8m0_t>,
                "Unsupported SF type. Expected ue8m0.");
  static_assert(cute::is_same_v<ElementC, cutlass::bfloat16_t> ||
                    cute::is_same_v<ElementC, cutlass::half_t> || cute::is_same_v<ElementC, float>,
                "Unsupported output type. Expected bf16/fp16/fp32.");

  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

  using ElementA = cutlass::mx_float8_t<ElementInput>;  // Element type for A matrix operand
  using LayoutA = cutlass::layout::RowMajor;            // Layout type for A matrix operand
  constexpr int AlignmentA = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::mx_float8_t<ElementInput>;  // Element type for B matrix operand
  using LayoutB =
      cute::conditional_t<TransB, cutlass::layout::ColumnMajor,
                          cutlass::layout::RowMajor>;  // Layout type for B matrix operand
  constexpr int AlignmentB = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementD = ElementC;
  using LayoutC = cutlass::layout::RowMajor;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;

  // Core kernel configurations
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;

  // Runtime Cluster Shape
  using ClusterShape = Shape<int32_t, int32_t, _1>;

  struct MMA2SMConfig {
    using MmaTileShape =
        cute::conditional_t<DGrad, Shape<_256, _256, _128>, Shape<_256, _256, _128>>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;  // Kernel to launch
    using EpilogueSchedule =
        cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;  // Epilogue to launch
  };

  using CollectiveEpilogue2SM = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator, void,
      LayoutC *, AlignmentC, ElementD, LayoutC *, AlignmentD,
      typename MMA2SMConfig::EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop2SM = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass, ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *,
      AlignmentB, ElementAccumulator, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
      typename MMA2SMConfig::KernelSchedule>::CollectiveOp;
  using GemmKernel2SM = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop2SM,
                                                             CollectiveEpilogue2SM>;
  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

  using StrideA = typename GemmGrouped::GemmKernel::InternalStrideA;
  using StrideB = typename GemmGrouped::GemmKernel::InternalStrideB;
  using StrideC = typename GemmGrouped::GemmKernel::InternalStrideC;
  using StrideD = typename GemmGrouped::GemmKernel::InternalStrideD;

  using LayoutSFA = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig =
      typename GemmGrouped::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  auto get_aligned_offset = [](size_t current_offset, size_t alignment) -> size_t {
    return (current_offset + alignment - 1) & ~(alignment - 1);
  };

  if (workspace == nullptr) {
    throw std::runtime_error("TE CUTLASS device grouped gemm workspace is null");
  }

  size_t offset = get_aligned_offset(num_experts * 2 * sizeof(uint64_t), 128); // inputA_and_SF_addrs
  auto ptr_A = reinterpret_cast<typename GemmGrouped::ElementA *>(A);
  auto ptr_A_list = reinterpret_cast<typename GemmGrouped::ElementA **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementA *), 128);

  auto ptr_D = reinterpret_cast<typename GemmGrouped::ElementD *>(D);
  auto ptr_D_list = reinterpret_cast<typename GemmGrouped::ElementD **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementD *), 128);

  auto ptr_SFA = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(SFA);
  auto ptr_SFA_list = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(
      offset + num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), 128);

  auto stride_A_list = reinterpret_cast<StrideA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideA), 128);
  auto stride_B_list = reinterpret_cast<StrideB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideB), 128);
  auto stride_D_list = reinterpret_cast<StrideD *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideD), 128);

  auto layout_SFA_list =
      reinterpret_cast<LayoutSFA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFA), 128);
  auto layout_SFB_list =
      reinterpret_cast<LayoutSFB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFB), 128);

  auto problem_sizes = reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(
      reinterpret_cast<char *>(workspace) + offset);
  offset =
      get_aligned_offset(offset + num_experts * sizeof(ProblemShape::UnderlyingProblemShape), 128);

  setGroupedGemmArguments<Sm1xxBlkScaledConfig, ProblemShape::UnderlyingProblemShape,
                          typename GemmGrouped::ElementA, typename GemmGrouped::ElementD,
                          typename GemmGrouped::GemmKernel::ElementSF, StrideA, StrideB, StrideD,
                          LayoutSFA, LayoutSFB, TransB><<<1, 32, 0, stream>>>(
      num_experts, gemm_m_per_expert, gemm_n, gemm_k, ptr_A, ptr_SFA, ptr_D, problem_sizes,
      ptr_A_list, ptr_SFA_list, stride_A_list, layout_SFA_list, stride_B_list, layout_SFB_list,
      ptr_D_list, stride_D_list);

  typename GemmGrouped::Arguments args;
  decltype(args.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  // Set alpha and beta to 1 and 0 for the fusion operation
  fusion_args.alpha = 1;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.beta = 0;
  fusion_args.beta_ptr_array = nullptr;
  fusion_args.dBeta = {_0{}, _0{}, 0};

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  if (!is_static_v<ClusterShape>) {
    hw_info.cluster_shape = DGrad ? dim3(2, 2, 1) : dim3(4, 4, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  }

  typename GemmGrouped::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongN;

  args = typename GemmGrouped::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes, nullptr},
      {const_cast<const typename GemmGrouped::ElementA **>(ptr_A_list), stride_A_list,
       reinterpret_cast<const typename GemmGrouped::ElementB **>(ptr_B_list), stride_B_list,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFA_list),
       layout_SFA_list,
       reinterpret_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFB_list),
       layout_SFB_list},
      {fusion_args, nullptr, stride_D_list, ptr_D_list, stride_D_list},
      hw_info,
      scheduler};

  GemmGrouped gemm;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmGrouped::get_workspace_size(args);
  if (workspaceSize < offset + workspace_size) {  // 16MB limit
    throw std::runtime_error("TE CUTLASS device grouped gemm calculated workspace size (" +
                             std::to_string(offset + workspace_size) + ") exceeds buffer size (" +
                             std::to_string(workspaceSize) + ")\n");
  }

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "TE CUTLASS device grouped gemm will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }

  auto init_status = gemm.initialize(args, reinterpret_cast<char *>(workspace) + offset);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass device grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass device grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }
}

// Only mxfp8 is supported for now
// A is splited tensor list, B is single Tensor, D is single Tensor
void nvte_device_cutlass_grouped_gemm(const void **A_and_SF_addrs, const NVTETensor *B,
                                      NVTETensor *D, const int64_t *m_splits, const int gemm_m,
                                      const NVTETensor *bias, NVTETensor *pre_gelu_out,
                                      const int num_gemms, bool transa, bool transb, bool grad,
                                      NVTETensor *workspace, size_t workspaceSize,
                                      bool use_split_accumulator, int math_sm_count,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_device_cutlass_grouped_gemm);
  using namespace transformer_engine;

  // Process B
  const transformer_engine::Tensor *inputB = convertNVTETensor(B[0]);
  if (transb) {
    NVTE_CHECK(inputB->has_columnwise_data(), "Input B is missing column-wise usage");
  } else {
    NVTE_CHECK(inputB->has_data(), "Input B is missing row-wise usage");
  }
  void *raw_inputB_ptr = transb ? inputB->columnwise_data.dptr : inputB->data.dptr;
  void *raw_inputB_SF_ptr = transb ? inputB->columnwise_scale_inv.dptr : inputB->scale_inv.dptr;

  // Process D
  const transformer_engine::Tensor *outputD = convertNVTETensor(D[0]);
  NVTE_CHECK(outputD->has_data(), "Input D is missing row-wise usage");
  void *raw_outputD_ptr = outputD->data.dptr;

  // Get GEMM shape
  const int gemm_k = transb ? inputB->flat_first_dim() : inputB->flat_last_dim();

  if ((gemm_k & 0x1F) != 0) {
    throw std::runtime_error("gemm_k of grouped gemm with variable M must be a multiple of 32.");
  }

  // Dispatch
  using transformer_engine::DType;
  const DType ab_dtype = transb ? inputB->columnwise_data.dtype : inputB->data.dtype;
  const DType d_dtype = outputD->data.dtype;

  auto workspace_ptr = convertNVTETensor(workspace[0])->data.dptr;

  auto dispatch_layout = [&](auto ab_dtype, auto d_dtype) {
    // dispatch based on input layout and dgrad flag
    using ABType = decltype(ab_dtype);
    using ABSFType = cutlass::float_ue8m0_t;
    using DType = decltype(d_dtype);

    ABType *inputB_ptr = reinterpret_cast<ABType *>(raw_inputB_ptr);
    ABSFType *inputB_SF_ptr = reinterpret_cast<ABSFType *>(raw_inputB_SF_ptr);
    DType *outputD_ptr = reinterpret_cast<DType *>(raw_outputD_ptr);

    // Swap A and B
    if (transa) {
      if (grad) {
        generic_moe_gemm_kernelLauncher<ABType, ABSFType, DType, true, true>(
            inputB_ptr, inputB_SF_ptr, A_and_SF_addrs, A_and_SF_addrs + num_gemms, outputD_ptr,
            m_splits, gemm_m, gemm_k, num_gemms, workspaceSize, workspace_ptr, stream);
      } else {
        generic_moe_gemm_kernelLauncher<ABType, ABSFType, DType, false, true>(
            inputB_ptr, inputB_SF_ptr, A_and_SF_addrs, A_and_SF_addrs + num_gemms, outputD_ptr,
            m_splits, gemm_m, gemm_k, num_gemms, workspaceSize, workspace_ptr, stream);
      }
    } else {
      if (grad) {
        generic_moe_gemm_kernelLauncher<ABType, ABSFType, DType, true, false>(
            inputB_ptr, inputB_SF_ptr, A_and_SF_addrs, A_and_SF_addrs + num_gemms, outputD_ptr,
            m_splits, gemm_m, gemm_k, num_gemms, workspaceSize, workspace_ptr, stream);
      } else {
        generic_moe_gemm_kernelLauncher<ABType, ABSFType, DType, false, false>(
            inputB_ptr, inputB_SF_ptr, A_and_SF_addrs, A_and_SF_addrs + num_gemms, outputD_ptr,
            m_splits, gemm_m, gemm_k, num_gemms, workspaceSize, workspace_ptr, stream);
      }
    }
  };

  auto dispatch_output_dtype = [&](auto ab_dtype) {
    // dispatch based on D dtype
    switch (d_dtype) {
      case DType::kBFloat16:
        dispatch_layout(ab_dtype, cutlass::bfloat16_t{});
        break;
      case DType::kFloat16:
        dispatch_layout(ab_dtype, cutlass::half_t{});
        break;
      case DType::kFloat32:
        dispatch_layout(ab_dtype, float{});
        break;
      default:
        throw std::runtime_error("Unsupported output dtype. Expected BF16/FP16/FP32.");
    }
  };

  auto dispatch = [&]() {
    // dispatch based on A/B dtype
    switch (ab_dtype) {
      case DType::kFloat8E4M3:
        dispatch_output_dtype(cutlass::float_e4m3_t{});
        break;
      case DType::kFloat8E5M2:
        dispatch_output_dtype(cutlass::float_e5m2_t{});
        break;
      default:
        throw std::runtime_error("Unsupported input dtype. A/B must be FP8 e4m3 or e5m2.");
    }
  };

  dispatch();
}

/*****
 * Wgrad
 ******/

namespace {
static inline const char *dtype_to_cstr(transformer_engine::DType dt) {
  using transformer_engine::DType;
  switch (dt) {
    case DType::kFloat8E4M3:
      return "Float8E4M3";
    case DType::kFloat8E5M2:
      return "Float8E5M2";
    case DType::kBFloat16:
      return "BFloat16";
    case DType::kFloat16:
      return "Float16";
    case DType::kFloat32:
      return "Float32";
    default:
      return "Unknown";
  }
}
}  // namespace

// Wgrad accumulate policy: supports optional per-expert bitmap (up to 1024 experts)
// and global accumulate switch. Passed by value to device kernel to avoid cudaMemcpyAsync.
struct WgradAccumulatePolicy {
  unsigned long long words[16];
  bool partial_wgrad_accumulate;  // true only when partial accumulate is enabled and mask provided
  bool accumulate;                // global accumulate switch

  __host__ __device__ inline bool need_accumulate(int expert_id) const {
    if (!accumulate) return false;
    if (!partial_wgrad_accumulate) return true;
    int chunk = (expert_id >> 6);
    int bit = expert_id & 63;
    unsigned long long w = words[chunk];
    return ((w >> bit) & 1ull) != 0ull;
  }

  __host__ inline void set(bool do_accumulate, const bool *mask, int num_experts) {
    accumulate = do_accumulate;
    partial_wgrad_accumulate = (do_accumulate && mask != nullptr);

    // Initialize words only when mask is used; otherwise words are don't-care
    if (partial_wgrad_accumulate) {
      for (int k = 0; k < 16; ++k) words[k] = ~0ull;
      for (int i = 0; i < num_experts; ++i) {
        if (mask[i]) continue;
        int chunk = (i >> 6);
        int bit = i & 63;
        // Already checked num_experts <= 1024 when initializing GroupedLinear op.
        // Skip out-of-range check here to avoid redundant computation.
        words[chunk] &= ~(1ull << bit);
      }
    }
  }
};

template <typename Sm1xxBlkScaledConfig, typename UnderlyingProblemShape, typename ElementA,
          typename ElementB, typename ElementC, typename ElementD, typename ElementAccumulator,
          typename ElementSF, typename StrideA, typename StrideB, typename StrideD,
          typename LayoutSFA, typename LayoutSFB, bool transD>
__global__ void setGroupedGemmWgradArguments(
    int num_experts, int gemm_m, int gemm_n, const int64_t *gemm_k_per_expert, int total_gemm_k,
    ElementA *ptr_A, ElementSF *ptr_SFA, ElementB *ptr_B, ElementSF *ptr_SFB,
    UnderlyingProblemShape *problem_sizes, ElementA **ptr_A_list, ElementSF **ptr_SFA_list,
    StrideA *stride_A_list, LayoutSFA *layout_SFA_list, ElementB **ptr_B_list,
    ElementSF **ptr_SFB_list, StrideB *stride_B_list, LayoutSFB *layout_SFB_list,
    ElementD **ptr_D_list, StrideD *stride_D_list, ElementC **ptr_C_list,
    ElementAccumulator **beta_ptr_list, ElementAccumulator *beta_zero, ElementAccumulator *beta_one,
    WgradAccumulatePolicy accumulate_policy) {
  int k_offset = 0;
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *beta_zero = 0;
    *beta_one = 1;
#pragma unroll
    for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
      int gemm_k = int(gemm_k_per_expert[expert_id]);
      if (!accumulate_policy.need_accumulate(expert_id)) {
        ptr_C_list[expert_id] = nullptr;
        beta_ptr_list[expert_id] = beta_zero;
      } else {
        ptr_C_list[expert_id] = ptr_D_list[expert_id];
        beta_ptr_list[expert_id] = beta_one;
      }

      problem_sizes[expert_id] = cute::make_shape(gemm_m, gemm_n, gemm_k);
      if (gemm_k == 0) {
        // If gemm_k is 0, we need to set the problem_sizes to 0, 0, 0 to skip the gemm
        problem_sizes[expert_id] = cute::make_shape(0, 0, 0);
        continue;
      }

      ptr_A_list[expert_id] = ptr_A + gemm_m * k_offset;
      ptr_SFA_list[expert_id] = ptr_SFA + 128 * ((k_offset + 127) / 128 * 4);
      stride_A_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_m), _0{});
      auto temp_sfa_layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
          cute::make_shape(gemm_m, gemm_n, total_gemm_k, 1));
      layout_SFA_list[expert_id] = cute::make_layout(
          get<0>(temp_sfa_layout),
          make_layout(get<0>(get<1>(temp_sfa_layout)),
                      make_layout(gemm_k / 128, get<1>(get<1>(temp_sfa_layout.stride())))),

          get<2>(temp_sfa_layout));

      ptr_B_list[expert_id] = ptr_B + gemm_n * k_offset;
      ptr_SFB_list[expert_id] = ptr_SFB + 128 * ((k_offset + 127) / 128 * 4);
      stride_B_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_n), _0{});
      auto temp_sfb_layout = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
          cute::make_shape(gemm_m, gemm_n, total_gemm_k, 1));
      layout_SFB_list[expert_id] = cute::make_layout(
          get<0>(temp_sfb_layout),
          make_layout(get<0>(get<1>(temp_sfb_layout)),
                      make_layout(gemm_k / 128, get<1>(get<1>(temp_sfb_layout.stride())))),
          get<2>(temp_sfb_layout));

      if constexpr (transD) {
        stride_D_list[expert_id] = cute::make_stride(_1{}, int64_t(gemm_m), _0{});
      } else {
        stride_D_list[expert_id] = cute::make_stride(int64_t(gemm_n), _1{}, _0{});
      }

      k_offset += gemm_k;
    }
  }

  // Parallel zero-fill for experts with gemm_k == 0 when not accumulating into D
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  int total_elems = gemm_m * gemm_n;

  // Try 16-byte vectorized stores when alignment permits, fallback to scalar otherwise
  constexpr int bytes_per_vec = 16;
  constexpr int elem_size = int(sizeof(ElementD));
  constexpr int elems_per_vec = bytes_per_vec / elem_size;  // 8 for bf16
  int total_vec = total_elems / elems_per_vec;

#pragma unroll
  for (int expert_id = 0; expert_id < num_experts; ++expert_id) {
    if (int(gemm_k_per_expert[expert_id]) != 0 || accumulate_policy.need_accumulate(expert_id)) {
      continue;  // Skip zero-fill if gemm_k is not 0 or current expert need accumulate
    }

    ElementD *d_ptr = ptr_D_list[expert_id];
    bool aligned16 = ((reinterpret_cast<size_t>(d_ptr) & (bytes_per_vec - 1)) == 0);

    if (aligned16 && elems_per_vec > 0) {
      int4 zero4;
      zero4.x = 0;
      zero4.y = 0;
      zero4.z = 0;
      zero4.w = 0;
      int4 *vec_ptr = reinterpret_cast<int4 *>(d_ptr);

#pragma unroll
      for (int j = tid; j < total_vec; j += stride) {
        vec_ptr[j] = zero4;
      }
#pragma unroll
      for (int k = total_vec * elems_per_vec + tid; k < total_elems; k += stride) {
        d_ptr[k] = ElementD(0);
      }
    } else {
#pragma unroll
      for (int i = tid; i < total_elems; i += stride) {
        d_ptr[i] = ElementD(0);
      }
    }
  }
}

template <typename ElementInput, typename ElementSF, typename ElementC, bool TransD>
void generic_moe_gemm_wgrad_kernelLauncher(ElementInput *A, ElementSF *SFA, ElementInput *B,
                                           ElementSF *SFB, void **ptr_D_list, int gemm_m,
                                           int gemm_n, const int64_t *gemm_k_per_expert,
                                           int total_gemm_k, int num_experts, bool accumulate,
                                           bool *accumulate_mask, size_t workspaceSize,
                                           void *workspace, cudaStream_t stream,
                                           int *kernel_occupancy = nullptr) {
  static_assert(cute::is_same_v<ElementInput, cutlass::float_e4m3_t> ||
                    cute::is_same_v<ElementInput, cutlass::float_e5m2_t>,
                "Unsupported input type. Expected e4m3 or e5m2.");
  static_assert(cute::is_same_v<ElementSF, cutlass::float_ue8m0_t>,
                "Unsupported SF type. Expected ue8m0.");
  static_assert(cute::is_same_v<ElementC, cutlass::bfloat16_t> ||
                    cute::is_same_v<ElementC, cutlass::half_t> || cute::is_same_v<ElementC, float>,
                "Unsupported output type. Expected bf16/fp16/fp32.");

  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;  // <M,N,K> per group

  using ElementA = cutlass::mx_float8_t<ElementInput>;  // Element type for A matrix operand
  using LayoutA = cutlass::layout::ColumnMajor;         // Layout type for A matrix operand
  constexpr int AlignmentA = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // B matrix configuration
  using ElementB = cutlass::mx_float8_t<ElementInput>;  // Element type for B matrix operand
  using LayoutB = cutlass::layout::RowMajor;            // Layout type for B matrix operand
  constexpr int AlignmentB = 32;  // Alignment of A matrix in units of elements (up to 16 bytes)

  // C/D matrix configuration
  using ElementD = ElementC;
  using LayoutC = typename cutlass::platform::conditional<TransD, cutlass::layout::ColumnMajor,
                                                          cutlass::layout::RowMajor>::type;
  constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
  constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
  using ElementAccumulator = float;

  // Core kernel configurations
  using ArchTag = cutlass::arch::Sm100;
  using EpilogueOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using MainloopOperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
  using StageCountType = cutlass::gemm::collective::StageCountAuto;

  // Runtime Cluster Shape
  using ClusterShape = Shape<int32_t, int32_t, _1>;

  struct MMA2SMConfig {
    using MmaTileShape = Shape<_256, _128, _128>;
    using KernelSchedule =
        cutlass::gemm::KernelPtrArrayTmaWarpSpecialized2SmMxf8f6f4Sm100;  // Kernel to launch
    using EpilogueSchedule =
        cutlass::epilogue::PtrArrayTmaWarpSpecialized2Sm;  // Epilogue to launch
  };

  using CollectiveEpilogue2SM = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, EpilogueOperatorClass, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      ElementC, LayoutC *, AlignmentC, ElementD, LayoutC *, AlignmentD,
      typename MMA2SMConfig::EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop2SM = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, MainloopOperatorClass, ElementA, LayoutA *, AlignmentA, ElementB, LayoutB *,
      AlignmentB, ElementAccumulator, typename MMA2SMConfig::MmaTileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue2SM::SharedStorage))>,
      typename MMA2SMConfig::KernelSchedule>::CollectiveOp;
  using GemmKernel2SM = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop2SM,
                                                             CollectiveEpilogue2SM>;
  using GemmGrouped = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel2SM>;

  using StrideA = typename GemmGrouped::GemmKernel::InternalStrideA;
  using StrideB = typename GemmGrouped::GemmKernel::InternalStrideB;
  using StrideC = typename GemmGrouped::GemmKernel::InternalStrideC;
  using StrideD = typename GemmGrouped::GemmKernel::InternalStrideD;

  using LayoutSFA = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFA;
  using LayoutSFB = typename GemmGrouped::GemmKernel::CollectiveMainloop::InternalLayoutSFB;
  using Sm1xxBlkScaledConfig =
      typename GemmGrouped::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

  using RasterOrderOptions = cutlass::gemm::kernel::detail::RasterOrderOptions;

  // Helper function to calculate aligned offset
  auto get_aligned_offset = [](size_t current_offset, size_t alignment) -> size_t {
    return (current_offset + alignment - 1) & ~(alignment - 1);
  };

  if (workspace == nullptr) {
    throw std::runtime_error("TE CUTLASS device grouped gemm workspace is null");
  }

  size_t offset = get_aligned_offset(num_experts * sizeof(uint64_t), 128); // outputD_addrs
  auto ptr_A = reinterpret_cast<typename GemmGrouped::ElementA *>(A);
  auto ptr_A_list = reinterpret_cast<typename GemmGrouped::ElementA **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementA *),
                              size_t(128));

  auto ptr_B = reinterpret_cast<typename GemmGrouped::ElementB *>(B);
  auto ptr_B_list = reinterpret_cast<typename GemmGrouped::ElementB **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementB *),
                              size_t(128));

  auto ptr_C_list = reinterpret_cast<typename GemmGrouped::ElementC **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(typename GemmGrouped::ElementC *),
                              size_t(128));

  auto ptr_SFA = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(SFA);
  auto ptr_SFA_list = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(
      offset + num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), size_t(128));

  auto ptr_SFB = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF *>(SFB);
  auto ptr_SFB_list = reinterpret_cast<typename GemmGrouped::GemmKernel::ElementSF **>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(
      offset + num_experts * sizeof(typename GemmGrouped::GemmKernel::ElementSF *), size_t(128));

  auto stride_A_list = reinterpret_cast<StrideA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideA), size_t(128));
  auto stride_B_list = reinterpret_cast<StrideB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideB), size_t(128));
  auto stride_D_list = reinterpret_cast<StrideD *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(StrideD), size_t(128));

  auto layout_SFA_list =
      reinterpret_cast<LayoutSFA *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFA), size_t(128));
  auto layout_SFB_list =
      reinterpret_cast<LayoutSFB *>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(LayoutSFB), size_t(128));

  auto problem_sizes = reinterpret_cast<ProblemShape::UnderlyingProblemShape *>(
      reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(ProblemShape::UnderlyingProblemShape),
                              size_t(128));

  auto beta_ptr_list =
      reinterpret_cast<ElementAccumulator **>(reinterpret_cast<char *>(workspace) + offset);
  offset = get_aligned_offset(offset + num_experts * sizeof(ElementAccumulator *), size_t(128));
  auto beta_zero =
      reinterpret_cast<ElementAccumulator *>(reinterpret_cast<char *>(workspace) + offset);
  auto beta_one = reinterpret_cast<ElementAccumulator *>(reinterpret_cast<char *>(workspace) +
                                                         offset + sizeof(ElementAccumulator));
  offset = get_aligned_offset(offset + 2 * sizeof(ElementAccumulator), size_t(128));

  // Build accumulate decision:
  // - accumulate == false      -> no expert accumulates
  // - accumulate == true
  //     - accumulate_mask == nullptr -> all experts accumulate (supports any num_experts)
  //     - accumulate_mask != nullptr -> partial accumulate; supports up to 1024 experts
  WgradAccumulatePolicy accumulate_policy{};
  accumulate_policy.set(accumulate, accumulate_mask, num_experts);

  // Launch setGroupedGemmWgradArguments
  constexpr int kVecBytes = 16;
  constexpr int kElemSize = int(sizeof(typename GemmGrouped::ElementD));
  const int elems_per_vec = kVecBytes > kElemSize ? kVecBytes / kElemSize : 1;
  const int total_elems = gemm_m * gemm_n;
  const int work_units = (total_elems + elems_per_vec - 1) / elems_per_vec;
  const int threads_per_block = 256;
  const int blocks = (work_units + threads_per_block - 1) / threads_per_block;
  setGroupedGemmWgradArguments<Sm1xxBlkScaledConfig, ProblemShape::UnderlyingProblemShape,
                               typename GemmGrouped::ElementA, typename GemmGrouped::ElementB,
                               typename GemmGrouped::ElementC, typename GemmGrouped::ElementD,
                               ElementAccumulator, typename GemmGrouped::GemmKernel::ElementSF,
                               StrideA, StrideB, StrideD, LayoutSFA, LayoutSFB, TransD>
      <<<blocks, threads_per_block, 0, stream>>>(
          num_experts, gemm_m, gemm_n, gemm_k_per_expert, total_gemm_k, ptr_A, ptr_SFA, ptr_B,
          ptr_SFB, problem_sizes, ptr_A_list, ptr_SFA_list, stride_A_list, layout_SFA_list,
          ptr_B_list, ptr_SFB_list, stride_B_list, layout_SFB_list,
          reinterpret_cast<typename GemmGrouped::ElementD **>(ptr_D_list), stride_D_list,
          ptr_C_list, beta_ptr_list, beta_zero, beta_one, accumulate_policy);

  // Check for CUDA errors after kernel launch
  cudaError_t cuda_error = cudaGetLastError();
  if (cuda_error != cudaSuccess) {
    std::string err_msg = "Failed to run setGroupedGemmWgradArguments. CUDA Error: " +
                          std::string(cudaGetErrorString(cuda_error));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }

  typename GemmGrouped::Arguments args;
  decltype(args.epilogue.thread) fusion_args;
  fusion_args.alpha_ptr = nullptr;
  fusion_args.beta_ptr = nullptr;
  // Set alpha and beta
  fusion_args.alpha = 1;
  fusion_args.alpha_ptr_array = nullptr;
  fusion_args.dAlpha = {_0{}, _0{}, 0};
  fusion_args.beta = 0;
  fusion_args.beta_ptr_array = beta_ptr_list;
  fusion_args.dBeta = {_0{}, _0{}, 1};

  cutlass::KernelHardwareInfo hw_info;
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  hw_info.device_id = 0;
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

  if (!is_static_v<ClusterShape>) {
    hw_info.cluster_shape = dim3(2, 2, 1);
    hw_info.cluster_shape_fallback = dim3(2, 1, 1);
  }

  typename GemmGrouped::GemmKernel::TileSchedulerArguments scheduler;
  scheduler.raster_order = RasterOrderOptions::AlongM;

  args = typename GemmGrouped::Arguments{
      cutlass::gemm::GemmUniversalMode::kGrouped,
      {num_experts, problem_sizes, nullptr},
      {const_cast<const typename GemmGrouped::ElementA **>(ptr_A_list), stride_A_list,
       const_cast<const typename GemmGrouped::ElementB **>(ptr_B_list), stride_B_list,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFA_list),
       layout_SFA_list,
       const_cast<const typename GemmGrouped::GemmKernel::ElementSF **>(ptr_SFB_list),
       layout_SFB_list},
      {fusion_args, const_cast<const typename GemmGrouped::ElementC **>(ptr_C_list), stride_D_list,
       reinterpret_cast<typename GemmGrouped::ElementD **>(ptr_D_list), stride_D_list},
      hw_info,
      scheduler};

  GemmGrouped gemm;

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = GemmGrouped::get_workspace_size(args);
  if (workspaceSize < offset + workspace_size) {  // 16MB limit
    throw std::runtime_error("TE CUTLASS device grouped gemm calculated workspace size (" +
                             std::to_string(offset + workspace_size) + ") exceeds buffer size (" +
                             std::to_string(workspaceSize) + ")\n");
  }

  auto can_implement = gemm.can_implement(args);
  if (can_implement != cutlass::Status::kSuccess) {
    std::string err_msg = "TE CUTLASS device grouped gemm will fail for params. Error: " +
                          std::string(cutlassGetStatusString(can_implement));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }

  auto init_status = gemm.initialize(args, reinterpret_cast<char *>(workspace) + offset);
  if (init_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to initialize cutlass grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(init_status));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }

  auto run_status = gemm.run(stream);
  if (run_status != cutlass::Status::kSuccess) {
    std::string err_msg = "Failed to run cutlass grouped gemm. Error: " +
                          std::string(cutlassGetStatusString(run_status));
    throw std::runtime_error("TE CUTLASS device grouped gemm error: " + err_msg);
  }
}

// Only mxfp8 is supported for now
// A and B is single Tensor, D is splited tensor list
void nvte_device_cutlass_grouped_gemm_wgrad(
    const NVTETensor *A, const NVTETensor *B, void **outputD_ptr_list,
    transformer_engine::DType D_type, const int64_t *m_splits, const NVTETensor *bias,
    NVTETensor *pre_gelu_out, const int num_gemms, bool transa, bool transb, NVTETensor *workspace,
    size_t workspaceSize, bool accumulate, bool *accumulate_mask, bool use_split_accumulator,
    int math_sm_count, cudaStream_t stream) {
  NVTE_API_CALL(nvte_device_cutlass_grouped_gemm_wgrad);
  using namespace transformer_engine;

  NVTE_CHECK(!transa && transb, "wgrad grouped gemm currently only support NT layout.");

  // Process A
  const transformer_engine::Tensor *inputA = convertNVTETensor(A[0]);
  if (transa) {
    NVTE_CHECK(inputA->has_data(), "Input A is missing row-wise usage");
  } else {
    NVTE_CHECK(inputA->has_columnwise_data(), "Input A is missing column-wise usage");
  }
  void *raw_inputA_ptr = transa ? inputA->data.dptr : inputA->columnwise_data.dptr;
  void *raw_inputA_SF_ptr = transa ? inputA->scale_inv.dptr : inputA->columnwise_scale_inv.dptr;

  // Process B
  const transformer_engine::Tensor *inputB = convertNVTETensor(B[0]);
  if (transb) {
    NVTE_CHECK(inputB->has_columnwise_data(), "Input B is missing column-wise usage");
  } else {
    NVTE_CHECK(inputB->has_data(), "Input B is missing row-wise usage");
  }
  void *raw_inputB_ptr = transb ? inputB->columnwise_data.dptr : inputB->data.dptr;
  void *raw_inputB_SF_ptr = transb ? inputB->columnwise_scale_inv.dptr : inputB->scale_inv.dptr;

  // Get GEMM shape
  const int gemm_m = transa ? inputA->flat_first_dim() : inputA->flat_last_dim();
  const int gemm_n = transb ? inputB->flat_last_dim() : inputB->flat_first_dim();
  const int total_gemm_k = transa ? inputA->flat_last_dim() : inputA->flat_first_dim();

  if ((gemm_m & 0x1F) != 0 || (gemm_n & 0xF) != 0) {
    throw std::runtime_error(
        "gemm_m and gemm_n of grouped gemm with variable K must be multiples of 32.");
  }

  // Dispatch
  using transformer_engine::DType;
  const DType a_dtype = transa ? inputA->data.dtype : inputA->columnwise_data.dtype;
  const DType b_dtype = transb ? inputB->columnwise_data.dtype : inputB->data.dtype;
  NVTE_CHECK(a_dtype == b_dtype, transformer_engine::concat_strings(
                                     "Input A/B dtypes mismatch for wgrad. A=",
                                     dtype_to_cstr(b_dtype), ", B=", dtype_to_cstr(b_dtype)));

  auto workspace_ptr = convertNVTETensor(workspace[0])->data.dptr;

  bool transD = true;  // transD should mirror fprop's transB; currently always true

  auto dispatch_layout = [&](auto ab_dtype, auto out_dtype) {
    // dispatch based on output layout
    using ABType = decltype(ab_dtype);
    using ABSFType = cutlass::float_ue8m0_t;
    using OutType = decltype(out_dtype);

    ABType *inputA_ptr = reinterpret_cast<ABType *>(raw_inputA_ptr);
    ABSFType *inputA_SF_ptr = reinterpret_cast<ABSFType *>(raw_inputA_SF_ptr);
    ABType *inputB_ptr = reinterpret_cast<ABType *>(raw_inputB_ptr);
    ABSFType *inputB_SF_ptr = reinterpret_cast<ABSFType *>(raw_inputB_SF_ptr);

    if (transD) {
      generic_moe_gemm_wgrad_kernelLauncher<ABType, ABSFType, OutType, true>(
          inputA_ptr, inputA_SF_ptr, inputB_ptr, inputB_SF_ptr, outputD_ptr_list, gemm_m, gemm_n,
          m_splits, total_gemm_k, num_gemms, accumulate, accumulate_mask, workspaceSize,
          workspace_ptr, stream);
    } else {
      generic_moe_gemm_wgrad_kernelLauncher<ABType, ABSFType, OutType, false>(
          inputA_ptr, inputA_SF_ptr, inputB_ptr, inputB_SF_ptr, outputD_ptr_list, gemm_m, gemm_n,
          m_splits, total_gemm_k, num_gemms, accumulate, accumulate_mask, workspaceSize,
          workspace_ptr, stream);
    }
  };
  auto dispatch_output_dtype = [&](auto ab_dtype) {
    switch (D_type) {
      case DType::kFloat32:
        dispatch_layout(ab_dtype, float{});
        break;
      case DType::kBFloat16:
        dispatch_layout(ab_dtype, cutlass::bfloat16_t{});
        break;
      case DType::kFloat16:
        dispatch_layout(ab_dtype, cutlass::half_t{});
        break;
      default:
        throw std::runtime_error("Unsupported output dtype. Expected Float32/BFloat16/Float16");
    }
  };

  auto dispatch = [&]() {
    // dispatch based on A/B dtype and D type
    switch (a_dtype) {
      case DType::kFloat8E4M3:
        dispatch_output_dtype(cutlass::float_e4m3_t{});
        break;
      case DType::kFloat8E5M2:
        dispatch_output_dtype(cutlass::float_e5m2_t{});
        break;
      default:
        throw std::runtime_error("Unsupported input dtype. A/B must be FP8 e4m3 or e5m2.");
    }
  };

  dispatch();
}