/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// NVRTC-backed registry registration for LayerNorm/RMSNorm forward + backward
// launchers. When NVTE_BUILD_LEGACY_STATIC_NORM=OFF, the per-config
// REGISTER_NORM_LAUNCHER macro expands to a call into one of the
// register_*_tuned/general functions defined here. Each registers a closure
// in the TeNormalizationRegistry that, on first invocation, compiles the
// matching template instantiation via NVRTC and from then on dispatches via
// the cached CUfunction.

#include "rtc_dispatch.h"

#include <cstdint>
#include <string>

#include "../util/cuda_driver.h"
#include "../util/rtc.h"
#include "../util/string.h"
#include "common.h"

// NVRTC source strings for the four kernel families. These are tiny stub
// files (each is a couple of #include lines that pull in kernel_traits + the
// matching kernel header); NVRTC then instantiates a specific
// (Kernel_traits<W, I, O, C, …>) on demand via nvrtcAddNameExpression.
#include "string_code_normalization_layernorm_rtc_ln_fwd_kernel_cu.h"
#include "string_code_normalization_layernorm_rtc_ln_bwd_kernel_cu.h"
#include "string_code_normalization_rmsnorm_rtc_rmsnorm_fwd_kernel_cu.h"
#include "string_code_normalization_rmsnorm_rtc_rmsnorm_bwd_kernel_cu.h"

namespace transformer_engine {
namespace normalization {
namespace rtc_norm {

namespace {

// Map our DType enum onto the C++ type names used inside the norm RTC sources.
// Aliases come from normalization/common.h (`using bf16 = nv_bfloat16;` etc.).
const char* cpp_name_for(DType dt) {
  switch (dt) {
    case DType::kFloat32:
      return "::transformer_engine::normalization::fp32";
    case DType::kFloat16:
      return "::transformer_engine::normalization::fp16";
    case DType::kBFloat16:
      return "::transformer_engine::normalization::bf16";
    case DType::kFloat8E4M3:
      return "::transformer_engine::normalization::fp8e4m3";
    case DType::kFloat8E5M2:
      return "::transformer_engine::normalization::fp8e5m2";
    default:
      NVTE_ERROR("Unsupported DType for norm RTC dispatch");
  }
}

int byte_size_of(DType dt) {
  switch (dt) {
    case DType::kFloat32:
      return 4;
    case DType::kFloat16:
    case DType::kBFloat16:
      return 2;
    case DType::kFloat8E4M3:
    case DType::kFloat8E5M2:
      return 1;
    default:
      NVTE_ERROR("Unsupported DType for norm RTC dispatch");
  }
}

// Build a Kernel_traits template-argument list as a string. The C++ argument
// list matches:
//   Kernel_traits<weight_t, input_t, output_t, compute_t, index_t, HIDDEN_SIZE,
//                 CTAS_PER_ROW, WARPS_M, WARPS_N, BYTES_PER_LDG>
std::string kernel_traits_expr(DType wt, DType it, DType ot, DType ct, int hidden_size,
                               int ctas_per_row, int warps_m, int warps_n, int bytes_per_ldg) {
  return concat_strings("::transformer_engine::normalization::Kernel_traits<", cpp_name_for(wt),
                        ", ", cpp_name_for(it), ", ", cpp_name_for(ot), ", ", cpp_name_for(ct),
                        ", uint32_t, ", hidden_size, ", ", ctas_per_row, ", ", warps_m, ", ",
                        warps_n, ", ", bytes_per_ldg, ">");
}

// Stats<T, CTAS_PER_ROW, WARPS_M, WARPS_N>::SMEM_BYTES expressed in host code.
// stats_t = TypeToVec2<compute_t>::Type — for compute_t == fp32 that's float2
// (8 bytes); for fp16 it's half2 (4 bytes); for bf16 it's nv_bfloat162 (4
// bytes). Matches the formulas in utils.cuh:
//   WARPS_N == 1 → 0
//   else       → WARPS_M * WARPS_N * sizeof(stats_t) * 2
int stats_smem_bytes(DType ctype, int warps_m, int warps_n) {
  if (warps_n == 1) return 0;
  int sizeof_stats_t;
  switch (ctype) {
    case DType::kFloat32:
      sizeof_stats_t = 8;
      break;
    case DType::kFloat16:
    case DType::kBFloat16:
      sizeof_stats_t = 4;
      break;
    default:
      NVTE_ERROR("Unsupported compute dtype for norm smem calc");
  }
  return warps_m * warps_n * sizeof_stats_t * 2;
}

// Reducer<reduce_t, CTAS_PER_ROW, WARPS_M, WARPS_N>::SMEM_BYTES.
// reduce_t = TypeToVec2<compute_t>::Type — same sizes as stats_t.
int reducer_smem_bytes(DType ctype, int warps_m, int warps_n) {
  return stats_smem_bytes(ctype, warps_m, warps_n);
}

// Kernel_traits::SMEM_BYTES (used by backward launchers):
//   SMEM_BYTES_DGRAD = Reducer::SMEM_BYTES
//   SMEM_BYTES_WGRAD = (CTAS_PER_ROW > 1) ? 0 : WARPS_M * HIDDEN * sizeof(compute_t)
//   SMEM_BYTES       = DGRAD + WGRAD
int bwd_smem_bytes(DType ctype, int hidden_size, int ctas_per_row, int warps_m, int warps_n) {
  const int dgrad = reducer_smem_bytes(ctype, warps_m, warps_n);
  const int wgrad =
      (ctas_per_row > 1) ? 0 : warps_m * hidden_size * byte_size_of(ctype);
  return dgrad + wgrad;
}

// Common per-launch configure/launch helper. `kernel_expr` is the full
// templated kernel symbol (used as the nvrtcAddNameExpression argument); the
// closure captures everything needed.
template <typename ParamsT>
void register_launcher(const std::string& label, const std::string& kernel_expr,
                       const char* rtc_source, const char* filename, TupleKeyType key,
                       int threads_per_cta, int dynamic_smem_bytes, int ctas_per_row,
                       bool needs_cooperative, int barrier_bytes_per_col,
                       int workspace_bytes_per_col, int dgamma_part_bytes_per_col) {
  auto closure = [label, kernel_expr, rtc_source, filename, threads_per_cta, dynamic_smem_bytes,
                  ctas_per_row, needs_cooperative, barrier_bytes_per_col,
                  workspace_bytes_per_col, dgamma_part_bytes_per_col](
                     LaunchParams<ParamsT>& launch_params, const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(label)) {
      mgr.compile(label, kernel_expr, rtc_source, filename);
    }

    if (configure_params) {
      const int ctas_per_sm =
          mgr.occupancy_max_active_blocks_per_sm(label, threads_per_cta, dynamic_smem_bytes);
      launch_params.params.ctas_per_row = ctas_per_row;
      launch_params.params.ctas_per_col =
          launch_params.multiprocessorCount * ctas_per_sm / ctas_per_row;
      if (ctas_per_row > 1) {
        launch_params.barrier_bytes =
            barrier_bytes_per_col * launch_params.params.ctas_per_col;
        launch_params.workspace_bytes =
            workspace_bytes_per_col * launch_params.params.ctas_per_col;
      }
      if (dgamma_part_bytes_per_col > 0) {
        launch_params.dgamma_part_bytes =
            dgamma_part_bytes_per_col * launch_params.params.ctas_per_col;
      }
      return;
    }

    // Real launch.
    if (dynamic_smem_bytes >= 48 * 1024) {
      mgr.set_function_attribute(label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                 dynamic_smem_bytes);
    }
    const auto stream = launch_params.stream;
    const auto ctas_per_col = launch_params.params.ctas_per_col;
    if (ctas_per_row == 1) {
      mgr.launch(label, dim3(ctas_per_col), dim3(threads_per_cta), dynamic_smem_bytes, stream,
                 launch_params.params);
    } else {
      mgr.launch_cooperative(label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta),
                             dynamic_smem_bytes, stream, launch_params.params);
    }
    (void)needs_cooperative;
  };
  TeNormalizationRegistry<ParamsT>::registerFunction(key, std::move(closure));
}

// Backward "general" launcher additionally launches a finalize kernel after
// the main kernel — host-side replicating the static path needs awareness of
// that. Same dispatch shape but with a second kernel symbol.
template <typename ParamsT>
void register_bwd_general_launcher(const std::string& main_label,
                                   const std::string& main_kernel_expr,
                                   const std::string& finalize_label,
                                   const std::string& finalize_kernel_expr,
                                   const char* rtc_source, const char* filename, TupleKeyType key,
                                   int threads_per_cta, int dynamic_smem_bytes,
                                   int finalize_threads_per_cta, int finalize_ctas,
                                   int hidden_size, int ctype_bytes, int warps_m) {
  auto closure = [main_label, main_kernel_expr, finalize_label, finalize_kernel_expr, rtc_source,
                  filename, threads_per_cta, dynamic_smem_bytes, finalize_threads_per_cta,
                  finalize_ctas, hidden_size, ctype_bytes, warps_m](
                     LaunchParams<ParamsT>& launch_params, const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(main_label)) {
      mgr.compile(main_label, main_kernel_expr, rtc_source, filename);
    }
    if (!mgr.is_compiled(finalize_label)) {
      mgr.compile(finalize_label, finalize_kernel_expr, rtc_source, filename);
    }

    // ctas_per_col + ctas_per_row for "general" backward is computed exactly
    // like the static launcher: ceil_div on hidden_size and on rows, capped by
    // SM count × occupancy.
    auto ceil_div = [](int x, int y) { return (x + y - 1) / y; };
    const int rows = launch_params.params.rows;
    const int cols = launch_params.params.cols;
    int ctas_per_col = launch_params.params.ctas_per_col;
    int ctas_per_row = launch_params.params.ctas_per_row;
    if (configure_params) {
      const int ctas_per_sm = mgr.occupancy_max_active_blocks_per_sm(main_label, threads_per_cta,
                                                                     dynamic_smem_bytes);
      const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
      ctas_per_row = ceil_div(cols, hidden_size);
      ctas_per_col = std::min(ceil_div(rows, warps_m), max_ctas / std::max(ctas_per_row, 1));
      launch_params.params.ctas_per_row = ctas_per_row;
      launch_params.params.ctas_per_col = ctas_per_col;
      if (ctas_per_row > 1) {
        launch_params.barrier_bytes = 2 * ctas_per_col * sizeof(int);
        launch_params.workspace_bytes =
            ctas_per_col * warps_m * ctas_per_row * (ctype_bytes * 2) * 2;
      }
      launch_params.dgamma_part_bytes = ctas_per_col * cols * ctype_bytes;
      return;
    }

    if (dynamic_smem_bytes >= 48 * 1024) {
      mgr.set_function_attribute(main_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                 dynamic_smem_bytes);
    }
    const auto stream = launch_params.stream;
    if (ctas_per_row == 1) {
      mgr.launch(main_label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta),
                 dynamic_smem_bytes, stream, launch_params.params);
    } else {
      mgr.launch_cooperative(main_label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta),
                             dynamic_smem_bytes, stream, launch_params.params);
    }
    mgr.launch(finalize_label, dim3(finalize_ctas), dim3(finalize_threads_per_cta), 0, stream,
               launch_params.params);
  };
  TeNormalizationRegistry<ParamsT>::registerFunction(key, std::move(closure));
}

}  // namespace

// ============================================================================
// LayerNorm Forward
// ============================================================================

void register_ln_fwd_tuned(DType wt, DType it, DType ot, DType ct, int hidden, int cr, int wm,
                           int wn, int bl) {
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::LayerNorm,
                           NVTE_Norm_Stage::Forward, wt, it, ot, ct, 0, hidden, false, true);
  const std::string label =
      concat_strings("ln_fwd_tuned,w=", static_cast<int>(wt), ",i=", static_cast<int>(it),
                     ",o=", static_cast<int>(ot), ",c=", static_cast<int>(ct), ",h=", hidden,
                     ",cr=", cr, ",wm=", wm, ",wn=", wn, ",bl=", bl);
  const std::string traits_expr = kernel_traits_expr(wt, it, ot, ct, hidden, cr, wm, wn, bl);
  const std::string kernel_expr =
      concat_strings("&::transformer_engine::normalization::ln_fwd_tuned_kernel<", traits_expr,
                     ">");
  const int threads_per_cta = wm * wn * 32;
  const int smem_bytes = stats_smem_bytes(ct, wm, wn);
  // tuned path multi-CTA workspace formula:
  //   barrier_bytes = 2 * ctas_per_col * sizeof(index_t == uint32_t == 4 bytes)
  //   workspace_bytes = ctas_per_col * WARPS_M * CTAS_PER_ROW * sizeof(stats_t) * 2
  const int sizeof_stats_t =
      (ct == DType::kFloat32) ? 8 : 4;  // float2 vs half2/bf162
  const int barrier_per_col = 2 * 4;
  const int workspace_per_col = wm * cr * sizeof_stats_t * 2;
  register_launcher<ForwardKernelParams>(
      label, kernel_expr, string_code_normalization_layernorm_rtc_ln_fwd_kernel_cu,
      "ln_fwd_kernel.cu", key, threads_per_cta, smem_bytes, cr, /*needs_cooperative=*/cr > 1,
      barrier_per_col, workspace_per_col, /*dgamma_part_bytes_per_col=*/0);
}

void register_ln_fwd_general(DType wt, DType it, DType ot, DType ct, int hidden, int wm, int wn,
                             int bl) {
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::LayerNorm,
                           NVTE_Norm_Stage::Forward, wt, it, ot, ct, 0, hidden, false, false);
  const std::string label =
      concat_strings("ln_fwd_general,w=", static_cast<int>(wt), ",i=", static_cast<int>(it),
                     ",o=", static_cast<int>(ot), ",c=", static_cast<int>(ct), ",h=", hidden,
                     ",wm=", wm, ",wn=", wn, ",bl=", bl);
  // "general" path always uses CTAS_PER_ROW=1 in the Kernel_traits.
  const std::string traits_expr = kernel_traits_expr(wt, it, ot, ct, hidden, 1, wm, wn, bl);
  const std::string kernel_expr =
      concat_strings("&::transformer_engine::normalization::ln_fwd_general_kernel<", traits_expr,
                     ">");
  const int threads_per_cta = wm * wn * 32;
  const auto closure = [label, kernel_expr, threads_per_cta, hidden, wm, ct](
                           LaunchParams<ForwardKernelParams>& launch_params,
                           const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(label)) {
      mgr.compile(label, kernel_expr, string_code_normalization_layernorm_rtc_ln_fwd_kernel_cu,
                  "ln_fwd_kernel.cu");
    }
    auto ceil_div = [](int x, int y) { return (x + y - 1) / y; };
    const int rows = launch_params.params.rows;
    const int cols = launch_params.params.cols;
    int ctas_per_col = launch_params.params.ctas_per_col;
    int ctas_per_row = launch_params.params.ctas_per_row;
    if (configure_params) {
      const int ctas_per_sm =
          mgr.occupancy_max_active_blocks_per_sm(label, threads_per_cta, /*smem=*/0);
      const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
      ctas_per_row = ceil_div(cols, hidden);
      ctas_per_col = std::min(ceil_div(rows, wm), max_ctas / std::max(ctas_per_row, 1));
      launch_params.params.ctas_per_row = ctas_per_row;
      launch_params.params.ctas_per_col = ctas_per_col;
      if (ctas_per_row > 1) {
        launch_params.barrier_bytes = 2 * ctas_per_col * sizeof(int);
        // compute_t bytes
        const int ctype_bytes = byte_size_of(ct);
        launch_params.workspace_bytes = ctas_per_col * wm * ctas_per_row * ctype_bytes * 2;
      }
      return;
    }
    const auto stream = launch_params.stream;
    if (ctas_per_row == 1) {
      mgr.launch(label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta), 0, stream,
                 launch_params.params);
    } else {
      mgr.launch_cooperative(label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta), 0,
                             stream, launch_params.params);
    }
  };
  TeNormalizationRegistry<ForwardKernelParams>::registerFunction(key, std::move(closure));
}

// ============================================================================
// RMSNorm Forward (same shape as LayerNorm Forward)
// ============================================================================

void register_rmsnorm_fwd_tuned(DType wt, DType it, DType ot, DType ct, int hidden, int cr, int wm,
                                int wn, int bl) {
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::RMSNorm,
                           NVTE_Norm_Stage::Forward, wt, it, ot, ct, 0, hidden, false, true);
  const std::string label =
      concat_strings("rmsnorm_fwd_tuned,w=", static_cast<int>(wt), ",i=", static_cast<int>(it),
                     ",o=", static_cast<int>(ot), ",c=", static_cast<int>(ct), ",h=", hidden,
                     ",cr=", cr, ",wm=", wm, ",wn=", wn, ",bl=", bl);
  const std::string traits_expr = kernel_traits_expr(wt, it, ot, ct, hidden, cr, wm, wn, bl);
  const std::string kernel_expr = concat_strings(
      "&::transformer_engine::normalization::rmsnorm_fwd_tuned_kernel<", traits_expr, ">");
  const int threads_per_cta = wm * wn * 32;
  const int smem_bytes = stats_smem_bytes(ct, wm, wn);
  const int sizeof_stats_t = (ct == DType::kFloat32) ? 8 : 4;
  const int barrier_per_col = 2 * 4;
  const int workspace_per_col = wm * cr * sizeof_stats_t * 2;
  register_launcher<ForwardKernelParams>(
      label, kernel_expr, string_code_normalization_rmsnorm_rtc_rmsnorm_fwd_kernel_cu,
      "rmsnorm_fwd_kernel.cu", key, threads_per_cta, smem_bytes, cr, /*needs_cooperative=*/cr > 1,
      barrier_per_col, workspace_per_col, 0);
}

void register_rmsnorm_fwd_general(DType wt, DType it, DType ot, DType ct, int hidden, int wm,
                                  int wn, int bl) {
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::RMSNorm,
                           NVTE_Norm_Stage::Forward, wt, it, ot, ct, 0, hidden, false, false);
  const std::string label =
      concat_strings("rmsnorm_fwd_general,w=", static_cast<int>(wt), ",i=", static_cast<int>(it),
                     ",o=", static_cast<int>(ot), ",c=", static_cast<int>(ct), ",h=", hidden,
                     ",wm=", wm, ",wn=", wn, ",bl=", bl);
  const std::string traits_expr = kernel_traits_expr(wt, it, ot, ct, hidden, 1, wm, wn, bl);
  const std::string kernel_expr = concat_strings(
      "&::transformer_engine::normalization::rmsnorm_fwd_general_kernel<", traits_expr, ">");
  const int threads_per_cta = wm * wn * 32;
  const auto closure = [label, kernel_expr, threads_per_cta, hidden, wm, ct](
                           LaunchParams<ForwardKernelParams>& launch_params,
                           const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(label)) {
      mgr.compile(label, kernel_expr,
                  string_code_normalization_rmsnorm_rtc_rmsnorm_fwd_kernel_cu,
                  "rmsnorm_fwd_kernel.cu");
    }
    auto ceil_div = [](int x, int y) { return (x + y - 1) / y; };
    const int rows = launch_params.params.rows;
    const int cols = launch_params.params.cols;
    int ctas_per_col = launch_params.params.ctas_per_col;
    int ctas_per_row = launch_params.params.ctas_per_row;
    if (configure_params) {
      const int ctas_per_sm =
          mgr.occupancy_max_active_blocks_per_sm(label, threads_per_cta, 0);
      const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
      ctas_per_row = ceil_div(cols, hidden);
      ctas_per_col = std::min(ceil_div(rows, wm), max_ctas / std::max(ctas_per_row, 1));
      launch_params.params.ctas_per_row = ctas_per_row;
      launch_params.params.ctas_per_col = ctas_per_col;
      if (ctas_per_row > 1) {
        launch_params.barrier_bytes = 2 * ctas_per_col * sizeof(int);
        const int ctype_bytes = byte_size_of(ct);
        launch_params.workspace_bytes = ctas_per_col * wm * ctas_per_row * ctype_bytes * 2;
      }
      return;
    }
    const auto stream = launch_params.stream;
    if (ctas_per_row == 1) {
      mgr.launch(label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta), 0, stream,
                 launch_params.params);
    } else {
      mgr.launch_cooperative(label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta), 0,
                             stream, launch_params.params);
    }
  };
  TeNormalizationRegistry<ForwardKernelParams>::registerFunction(key, std::move(closure));
}

// ============================================================================
// LayerNorm Backward (main kernel + finalize kernel)
// ============================================================================

void register_ln_bwd_tuned(DType wt, DType it, DType ot, DType ct, int hidden, int cr, int wm,
                           int wn, int bl_main, int bl_final) {
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::LayerNorm,
                           NVTE_Norm_Stage::Backward, wt, it, ot, ct, 0, hidden, false, true);
  const std::string label =
      concat_strings("ln_bwd_tuned,w=", static_cast<int>(wt), ",i=", static_cast<int>(it),
                     ",o=", static_cast<int>(ot), ",c=", static_cast<int>(ct), ",h=", hidden,
                     ",cr=", cr, ",wm=", wm, ",wn=", wn, ",bl=", bl_main, ",blf=", bl_final);
  const std::string main_label = concat_strings(label, ",main");
  const std::string finalize_label = concat_strings(label, ",finalize");
  const std::string main_traits = kernel_traits_expr(wt, it, ot, ct, hidden, cr, wm, wn, bl_main);
  const std::string main_kexpr = concat_strings(
      "&::transformer_engine::normalization::ln_bwd_tuned_kernel<", main_traits, ">");
  // Kernel_traits_finalize<HIDDEN_SIZE, weight_t, input_t, output_t, compute_t, index_t,
  //                         THREADS_PER_CTA (32 * 32), BYTES_PER_LDG_FINAL>
  const std::string finalize_traits =
      concat_strings("::transformer_engine::normalization::Kernel_traits_finalize<", hidden, ", ",
                     cpp_name_for(wt), ", ", cpp_name_for(it), ", ", cpp_name_for(ot), ", ",
                     cpp_name_for(ct), ", uint32_t, 1024, ", bl_final, ">");
  const std::string finalize_kexpr = concat_strings(
      "&::transformer_engine::normalization::ln_bwd_finalize_tuned_kernel<", finalize_traits, ">");
  const int threads_per_cta = wm * wn * 32;
  const int smem_bytes = bwd_smem_bytes(ct, hidden, cr, wm, wn);
  const int sizeof_reduce_t = (ct == DType::kFloat32) ? 8 : 4;
  // tuned backward: workspace = ctas_per_col * WARPS_M * CTAS_PER_ROW * sizeof(reduce_t) * 2
  //                 dgamma_part_bytes = ctas_per_col * cols * sizeof(compute_t)
  //                 barrier_bytes = 2 * ctas_per_col * sizeof(index_t)
  const int barrier_per_col = 2 * 4;
  const int workspace_per_col = wm * cr * sizeof_reduce_t * 2;
  const int dgamma_part_per_col = hidden * byte_size_of(ct);

  // Finalize kernel dims: Kernel_traits_finalize::THREADS_PER_CTA == 1024
  //                     Kernel_traits_finalize::CTAS = HIDDEN_SIZE / 32 (since COLS%32==0)
  //   COLS = HIDDEN_SIZE * sizeof(compute_t) / BYTES_PER_LDG_FINAL
  //   CTAS = COLS / 32
  const int colspass = hidden * byte_size_of(ct) / bl_final;
  const int finalize_ctas = colspass / 32;
  const int finalize_threads_per_cta = 1024;

  auto closure = [main_label, main_kexpr, finalize_label, finalize_kexpr, threads_per_cta,
                  smem_bytes, cr, barrier_per_col, workspace_per_col, dgamma_part_per_col,
                  finalize_ctas, finalize_threads_per_cta](
                     LaunchParams<BackwardKernelParams>& launch_params,
                     const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(main_label)) {
      mgr.compile(main_label, main_kexpr,
                  string_code_normalization_layernorm_rtc_ln_bwd_kernel_cu, "ln_bwd_kernel.cu");
    }
    if (!mgr.is_compiled(finalize_label)) {
      mgr.compile(finalize_label, finalize_kexpr,
                  string_code_normalization_layernorm_rtc_ln_bwd_kernel_cu, "ln_bwd_kernel.cu");
    }
    if (configure_params) {
      const int ctas_per_sm =
          mgr.occupancy_max_active_blocks_per_sm(main_label, threads_per_cta, smem_bytes);
      launch_params.params.ctas_per_row = cr;
      launch_params.params.ctas_per_col =
          launch_params.multiprocessorCount * ctas_per_sm / cr;
      if (cr > 1) {
        launch_params.barrier_bytes = barrier_per_col * launch_params.params.ctas_per_col;
        launch_params.workspace_bytes = workspace_per_col * launch_params.params.ctas_per_col;
      }
      launch_params.dgamma_part_bytes = dgamma_part_per_col * launch_params.params.ctas_per_col;
      return;
    }
    if (smem_bytes >= 48 * 1024) {
      mgr.set_function_attribute(main_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                 smem_bytes);
    }
    const auto stream = launch_params.stream;
    const auto ctas_per_col = launch_params.params.ctas_per_col;
    if (cr == 1) {
      mgr.launch(main_label, dim3(ctas_per_col), dim3(threads_per_cta), smem_bytes, stream,
                 launch_params.params);
    } else {
      mgr.launch_cooperative(main_label, dim3(cr * ctas_per_col), dim3(threads_per_cta),
                             smem_bytes, stream, launch_params.params);
    }
    mgr.launch(finalize_label, dim3(finalize_ctas), dim3(finalize_threads_per_cta), 0, stream,
               launch_params.params);
  };
  TeNormalizationRegistry<BackwardKernelParams>::registerFunction(key, std::move(closure));
}

void register_ln_bwd_general(DType wt, DType it, DType ot, DType ct, int hidden, int wm, int wn,
                             int bl_main, int bl_final) {
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::LayerNorm,
                           NVTE_Norm_Stage::Backward, wt, it, ot, ct, 0, hidden, false, false);
  const std::string label =
      concat_strings("ln_bwd_general,w=", static_cast<int>(wt), ",i=", static_cast<int>(it),
                     ",o=", static_cast<int>(ot), ",c=", static_cast<int>(ct), ",h=", hidden,
                     ",wm=", wm, ",wn=", wn, ",bl=", bl_main, ",blf=", bl_final);
  const std::string main_label = concat_strings(label, ",main");
  const std::string finalize_label = concat_strings(label, ",finalize");
  const std::string traits = kernel_traits_expr(wt, it, ot, ct, hidden, 1, wm, wn, bl_main);
  const std::string main_kexpr = concat_strings(
      "&::transformer_engine::normalization::ln_bwd_general_kernel<", traits, ">");
  // ln_bwd_finalize_general_kernel<weight_t, compute_t, WARPS_M_FINAL=4, WARPS_N_FINAL=1,
  //                                BYTES_PER_LDG_FINAL, THREADS_PER_WARP=32>
  const std::string finalize_kexpr =
      concat_strings("&::transformer_engine::normalization::ln_bwd_finalize_general_kernel<",
                     cpp_name_for(wt), ", ", cpp_name_for(ct), ", 4, 1, ", bl_final, ", 32>");
  const int threads_per_cta = wm * wn * 32;
  // general bwd uses ctas_per_row = ceil_div(cols, HIDDEN_SIZE); smem=0 for main kernel call.
  const int ctype_bytes = byte_size_of(ct);
  const int finalize_threads_per_warp = 32;
  const int finalize_warps_n = 1;
  const int finalize_warps_m = 4;
  const int finalize_elts_n_per_cta =
      finalize_threads_per_warp * finalize_warps_n * bl_final / ctype_bytes;

  auto closure = [main_label, main_kexpr, finalize_label, finalize_kexpr, threads_per_cta, hidden,
                  wm, ctype_bytes, finalize_elts_n_per_cta, finalize_warps_n, finalize_warps_m,
                  finalize_threads_per_warp](LaunchParams<BackwardKernelParams>& launch_params,
                                              const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(main_label)) {
      mgr.compile(main_label, main_kexpr,
                  string_code_normalization_layernorm_rtc_ln_bwd_kernel_cu, "ln_bwd_kernel.cu");
    }
    if (!mgr.is_compiled(finalize_label)) {
      mgr.compile(finalize_label, finalize_kexpr,
                  string_code_normalization_layernorm_rtc_ln_bwd_kernel_cu, "ln_bwd_kernel.cu");
    }
    auto ceil_div = [](int x, int y) { return (x + y - 1) / y; };
    const int rows = launch_params.params.rows;
    const int cols = launch_params.params.cols;
    int ctas_per_col = launch_params.params.ctas_per_col;
    int ctas_per_row = launch_params.params.ctas_per_row;
    if (configure_params) {
      const int ctas_per_sm =
          mgr.occupancy_max_active_blocks_per_sm(main_label, threads_per_cta, 0);
      const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
      ctas_per_row = ceil_div(cols, hidden);
      ctas_per_col = std::min(ceil_div(rows, wm), max_ctas / std::max(ctas_per_row, 1));
      launch_params.params.ctas_per_row = ctas_per_row;
      launch_params.params.ctas_per_col = ctas_per_col;
      if (ctas_per_row > 1) {
        launch_params.barrier_bytes = 2 * ctas_per_col * sizeof(int);
        launch_params.workspace_bytes = ctas_per_col * wm * ctas_per_row * (ctype_bytes * 2) * 2;
      }
      launch_params.dgamma_part_bytes = ctas_per_col * cols * ctype_bytes;
      return;
    }
    const auto stream = launch_params.stream;
    if (ctas_per_row == 1) {
      mgr.launch(main_label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta), 0, stream,
                 launch_params.params);
    } else {
      mgr.launch_cooperative(main_label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta),
                             0, stream, launch_params.params);
    }
    const dim3 fin_block(finalize_threads_per_warp * finalize_warps_n, finalize_warps_m);
    const dim3 fin_grid(ceil_div(cols, finalize_elts_n_per_cta), 1);
    mgr.launch(finalize_label, fin_grid, fin_block, 0, stream, launch_params.params);
  };
  TeNormalizationRegistry<BackwardKernelParams>::registerFunction(key, std::move(closure));
}

// ============================================================================
// RMSNorm Backward (main kernel + finalize kernel; same shape as LayerNorm bwd)
// ============================================================================

void register_rmsnorm_bwd_tuned(DType wt, DType it, DType ot, DType ct, int hidden, int cr, int wm,
                                int wn, int bl_main, int bl_final, bool with_add) {
  const auto stage =
      with_add ? NVTE_Norm_Stage::BackwardAdd : NVTE_Norm_Stage::Backward;
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::RMSNorm, stage, wt, it, ot, ct, 0,
                           hidden, false, true);
  const std::string add_tag = with_add ? "_add" : "";
  const std::string label = concat_strings("rmsnorm_bwd_tuned", add_tag, ",w=",
                                            static_cast<int>(wt), ",i=", static_cast<int>(it),
                                            ",o=", static_cast<int>(ot), ",c=",
                                            static_cast<int>(ct), ",h=", hidden, ",cr=", cr,
                                            ",wm=", wm, ",wn=", wn, ",bl=", bl_main,
                                            ",blf=", bl_final);
  const std::string main_label = concat_strings(label, ",main");
  const std::string finalize_label = concat_strings(label, ",finalize");
  const std::string traits = kernel_traits_expr(wt, it, ot, ct, hidden, cr, wm, wn, bl_main);
  const char* add_flag = with_add ? "true" : "false";
  const std::string main_kexpr =
      concat_strings("&::transformer_engine::normalization::rmsnorm_bwd_tuned_kernel<", traits,
                     ", ", add_flag, ">");
  const std::string finalize_traits =
      concat_strings("::transformer_engine::normalization::Kernel_traits_finalize<", hidden, ", ",
                     cpp_name_for(wt), ", ", cpp_name_for(it), ", ", cpp_name_for(ot), ", ",
                     cpp_name_for(ct), ", uint32_t, 1024, ", bl_final, ">");
  const std::string finalize_kexpr = concat_strings(
      "&::transformer_engine::normalization::rmsnorm_bwd_finalize_tuned_kernel<", finalize_traits,
      ">");
  const int threads_per_cta = wm * wn * 32;
  const int smem_bytes = bwd_smem_bytes(ct, hidden, cr, wm, wn);
  const int sizeof_reduce_t = (ct == DType::kFloat32) ? 8 : 4;
  const int barrier_per_col = 2 * 4;
  const int workspace_per_col = wm * cr * sizeof_reduce_t * 2;
  const int dgamma_part_per_col = hidden * byte_size_of(ct);
  const int colspass = hidden * byte_size_of(ct) / bl_final;
  const int finalize_ctas = colspass / 32;
  const int finalize_threads_per_cta = 1024;

  auto closure = [main_label, main_kexpr, finalize_label, finalize_kexpr, threads_per_cta,
                  smem_bytes, cr, barrier_per_col, workspace_per_col, dgamma_part_per_col,
                  finalize_ctas, finalize_threads_per_cta](
                     LaunchParams<BackwardKernelParams>& launch_params,
                     const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(main_label)) {
      mgr.compile(main_label, main_kexpr,
                  string_code_normalization_rmsnorm_rtc_rmsnorm_bwd_kernel_cu,
                  "rmsnorm_bwd_kernel.cu");
    }
    if (!mgr.is_compiled(finalize_label)) {
      mgr.compile(finalize_label, finalize_kexpr,
                  string_code_normalization_rmsnorm_rtc_rmsnorm_bwd_kernel_cu,
                  "rmsnorm_bwd_kernel.cu");
    }
    if (configure_params) {
      const int ctas_per_sm =
          mgr.occupancy_max_active_blocks_per_sm(main_label, threads_per_cta, smem_bytes);
      launch_params.params.ctas_per_row = cr;
      launch_params.params.ctas_per_col =
          launch_params.multiprocessorCount * ctas_per_sm / cr;
      if (cr > 1) {
        launch_params.barrier_bytes = barrier_per_col * launch_params.params.ctas_per_col;
        launch_params.workspace_bytes = workspace_per_col * launch_params.params.ctas_per_col;
      }
      launch_params.dgamma_part_bytes = dgamma_part_per_col * launch_params.params.ctas_per_col;
      return;
    }
    if (smem_bytes >= 48 * 1024) {
      mgr.set_function_attribute(main_label, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                 smem_bytes);
    }
    const auto stream = launch_params.stream;
    const auto ctas_per_col = launch_params.params.ctas_per_col;
    if (cr == 1) {
      mgr.launch(main_label, dim3(ctas_per_col), dim3(threads_per_cta), smem_bytes, stream,
                 launch_params.params);
    } else {
      mgr.launch_cooperative(main_label, dim3(cr * ctas_per_col), dim3(threads_per_cta),
                             smem_bytes, stream, launch_params.params);
    }
    mgr.launch(finalize_label, dim3(finalize_ctas), dim3(finalize_threads_per_cta), 0, stream,
               launch_params.params);
  };
  TeNormalizationRegistry<BackwardKernelParams>::registerFunction(key, std::move(closure));
}

void register_rmsnorm_bwd_general(DType wt, DType it, DType ot, DType ct, int hidden, int wm,
                                  int wn, int bl_main, int bl_final, bool with_add) {
  const auto stage =
      with_add ? NVTE_Norm_Stage::BackwardAdd : NVTE_Norm_Stage::Backward;
  const auto key = get_key(NVTE_Norm_Backend::Te, NVTE_Norm_Type::RMSNorm, stage, wt, it, ot, ct, 0,
                           hidden, false, false);
  const std::string add_tag = with_add ? "_add" : "";
  const std::string label = concat_strings("rmsnorm_bwd_general", add_tag, ",w=",
                                            static_cast<int>(wt), ",i=", static_cast<int>(it),
                                            ",o=", static_cast<int>(ot), ",c=",
                                            static_cast<int>(ct), ",h=", hidden, ",wm=", wm,
                                            ",wn=", wn, ",bl=", bl_main, ",blf=", bl_final);
  const std::string main_label = concat_strings(label, ",main");
  const std::string finalize_label = concat_strings(label, ",finalize");
  const std::string traits = kernel_traits_expr(wt, it, ot, ct, hidden, 1, wm, wn, bl_main);
  const char* add_flag = with_add ? "true" : "false";
  const std::string main_kexpr =
      concat_strings("&::transformer_engine::normalization::rmsnorm_bwd_general_kernel<", traits,
                     ", ", add_flag, ">");
  const std::string finalize_kexpr =
      concat_strings("&::transformer_engine::normalization::rmsnorm_bwd_finalize_general_kernel<",
                     cpp_name_for(wt), ", ", cpp_name_for(ct), ", 4, 1, ", bl_final, ", 32>");
  const int threads_per_cta = wm * wn * 32;
  const int ctype_bytes = byte_size_of(ct);
  const int finalize_warps_m = 4;
  const int finalize_warps_n = 1;
  const int finalize_threads_per_warp = 32;
  const int finalize_elts_n_per_cta =
      finalize_threads_per_warp * finalize_warps_n * bl_final / ctype_bytes;

  auto closure = [main_label, main_kexpr, finalize_label, finalize_kexpr, threads_per_cta, hidden,
                  wm, ctype_bytes, finalize_elts_n_per_cta, finalize_warps_n, finalize_warps_m,
                  finalize_threads_per_warp](LaunchParams<BackwardKernelParams>& launch_params,
                                              const bool configure_params) {
    auto& mgr = rtc::KernelManager::instance();
    if (!mgr.is_compiled(main_label)) {
      mgr.compile(main_label, main_kexpr,
                  string_code_normalization_rmsnorm_rtc_rmsnorm_bwd_kernel_cu,
                  "rmsnorm_bwd_kernel.cu");
    }
    if (!mgr.is_compiled(finalize_label)) {
      mgr.compile(finalize_label, finalize_kexpr,
                  string_code_normalization_rmsnorm_rtc_rmsnorm_bwd_kernel_cu,
                  "rmsnorm_bwd_kernel.cu");
    }
    auto ceil_div = [](int x, int y) { return (x + y - 1) / y; };
    const int rows = launch_params.params.rows;
    const int cols = launch_params.params.cols;
    int ctas_per_col = launch_params.params.ctas_per_col;
    int ctas_per_row = launch_params.params.ctas_per_row;
    if (configure_params) {
      const int ctas_per_sm =
          mgr.occupancy_max_active_blocks_per_sm(main_label, threads_per_cta, 0);
      const int max_ctas = launch_params.multiprocessorCount * ctas_per_sm;
      ctas_per_row = ceil_div(cols, hidden);
      ctas_per_col = std::min(ceil_div(rows, wm), max_ctas / std::max(ctas_per_row, 1));
      launch_params.params.ctas_per_row = ctas_per_row;
      launch_params.params.ctas_per_col = ctas_per_col;
      if (ctas_per_row > 1) {
        launch_params.barrier_bytes = 2 * ctas_per_col * sizeof(int);
        launch_params.workspace_bytes = ctas_per_col * wm * ctas_per_row * (ctype_bytes * 2) * 2;
      }
      launch_params.dgamma_part_bytes = ctas_per_col * cols * ctype_bytes;
      return;
    }
    const auto stream = launch_params.stream;
    if (ctas_per_row == 1) {
      mgr.launch(main_label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta), 0, stream,
                 launch_params.params);
    } else {
      mgr.launch_cooperative(main_label, dim3(ctas_per_row * ctas_per_col), dim3(threads_per_cta),
                             0, stream, launch_params.params);
    }
    const dim3 fin_block(finalize_threads_per_warp * finalize_warps_n, finalize_warps_m);
    const dim3 fin_grid(ceil_div(cols, finalize_elts_n_per_cta), 1);
    mgr.launch(finalize_label, fin_grid, fin_block, 0, stream, launch_params.params);
  };
  TeNormalizationRegistry<BackwardKernelParams>::registerFunction(key, std::move(closure));
}

}  // namespace rtc_norm
}  // namespace normalization
}  // namespace transformer_engine
