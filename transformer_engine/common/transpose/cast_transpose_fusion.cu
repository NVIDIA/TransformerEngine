/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_runtime.h>
#include <transformer_engine/transpose.h>

#include <cfloat>
#include <functional>
#include <numeric>
#include <type_traits>

#include "../util/math.h"
#include "../util/rtc.h"
#include "../util/string.h"
#include "../utils.cuh"
#include "cast_transpose.h"
#include "common/common.h"

namespace transformer_engine {

namespace detail {

// String with RTC kernel implementation
#include "string_code_transpose_rtc_cast_transpose_fusion_cu.h"

// STUFF TO TUNE
constexpr size_t n_warps_per_tile = 8;
constexpr size_t desired_load_size = 8;
constexpr size_t desired_store_size = 8;
constexpr size_t desired_load_size_dact = 4;  // dAct fusion kernels use more registers
constexpr size_t desired_store_size_dact = 4;

constexpr size_t threads_per_warp = static_cast<size_t>(THREADS_PER_WARP);
constexpr size_t max_threads_per_block = 256;
constexpr size_t reduce_dbias_num_threads = 256;
constexpr size_t cast_transpose_num_threads = n_warps_per_tile * threads_per_warp;
constexpr size_t n_warps_per_block = cast_transpose_num_threads / threads_per_warp;
static_assert(cast_transpose_num_threads <= max_threads_per_block);

/* Performance heuristics for optimized kernel parameters */
struct KernelConfig {
  size_t load_size = 0;   // Vector load size
  size_t store_size = 0;  // Vector store size to transposed output

  bool valid = false;     // Whether config is valid
  bool is_dact = false;   // Whether dact is used
  size_t num_blocks = 0;  // Number of CUDA blocks

  size_t active_sm_count = 0;         // Number of active SMs
  size_t elements_per_load = 0;       // Elements per L1 cache load
  size_t elements_per_load_dact = 0;  // Elements per L1 cache load dact
  size_t elements_per_store_c = 0;    // Elements per L1 cache store to cast output
  size_t elements_per_store_t = 0;    // Elements per L1 cache store to transposed output

  KernelConfig(size_t row_length, size_t num_rows, size_t itype_size, size_t itype2_size,
               size_t otype_size, size_t load_size_, size_t store_size_, size_t sm_count,
               bool is_dact_)
      : load_size{load_size_}, store_size{store_size_}, is_dact{is_dact_} {
    if (is_dact) {
      if (load_size > desired_load_size_dact || store_size > desired_store_size_dact) {
        return;
      }
    }

    // Check that tiles are correctly aligned
    constexpr size_t cache_line_size = 128;
    if (load_size % itype_size != 0 || store_size % otype_size != 0 ||
        cache_line_size % itype_size != 0 || cache_line_size % otype_size != 0) {
      return;
    }
    /* row_tile_elements */
    const size_t tile_size_x = (load_size * THREADS_PER_WARP) / itype_size;
    /* col_tile_elements */
    const size_t tile_size_y = (store_size * THREADS_PER_WARP) / otype_size;
    const size_t num_tiles_x = row_length / tile_size_x;
    const size_t num_tiles_y = num_rows / tile_size_y;

    valid = (row_length % tile_size_x == 0 && num_rows % tile_size_y == 0);
    if (!valid) {
      return;
    }

    // Number of CUDA blocks
    num_blocks = num_tiles_x * num_tiles_y;

    // Parameters for performance model
    constexpr size_t warps_per_sm = 16;  // Rough estimate for saturated SMs
    active_sm_count = std::min(DIVUP(num_blocks * n_warps_per_tile, warps_per_sm), sm_count);
    elements_per_load = (std::min(cache_line_size, tile_size_x * itype_size) / itype_size);
    elements_per_load_dact = (std::min(cache_line_size, tile_size_x * itype2_size) / itype2_size);
    elements_per_store_c = (std::min(cache_line_size, tile_size_x * otype_size) / otype_size);
    elements_per_store_t = (std::min(cache_line_size, tile_size_y * otype_size) / otype_size);
  }

  /* Compare by estimated cost */
  bool operator<(const KernelConfig &other) const {
    if (this->valid && other.valid) {
      // cost ~ (1/elements_per_load
      //         + 1/elements_per_load_dact
      //         + 1/elements_per_store_c
      //         + 1/elements_per_store_t) / active_sms
      // Note: Integer arithmetic ensures stable ordering
      const auto &l1 = this->elements_per_load;
      const auto &la1 = this->elements_per_load_dact;
      const auto &sc1 = this->elements_per_store_c;
      const auto &st1 = this->elements_per_store_t;
      const auto &p1 = this->active_sm_count;
      const auto &l2 = other.elements_per_load;
      const auto &la2 = other.elements_per_load_dact;
      const auto &sc2 = other.elements_per_store_c;
      const auto &st2 = other.elements_per_store_t;
      const auto &p2 = other.active_sm_count;
      const auto scale1 = l1 * sc1 * st1 * p1 * (is_dact ? la1 : 1);
      const auto scale2 = l2 * sc2 * st2 * p2 * (is_dact ? la2 : 1);
      const auto scale = scale1 * scale2;
      const auto cost1 =
          (scale / l1 + scale / sc1 + scale / st1 + (is_dact ? (scale / la1) : 0)) / p1;
      const auto cost2 =
          (scale / l2 + scale / sc2 + scale / st2 + (is_dact ? (scale / la2) : 0)) / p2;

      return cost1 < cost2;
    } else {
      return this->valid && !other.valid;
    }
  }
};

template <bool IS_DBIAS, bool IS_FULL_TILE, int nvec_in, int nvec_out, typename OVec, typename CVec,
          typename CType>
inline __device__ void cast_and_transpose_regs(const CVec (&in)[nvec_out],
                                               OVec (&out_trans)[nvec_in],
                                               CVec &out_dbias,  // NOLINT(*)
                                               typename OVec::type *output_cast_tile,
                                               const size_t current_place, const size_t stride,
                                               const CType scale,
                                               CType &amax,  // NOLINT(*)
                                               const int dbias_shfl_src_lane,
                                               const bool valid_store) {
  using OType = typename OVec::type;
  using OVecC = Vec<OType, nvec_in>;

  CVec step_dbias;
  if constexpr (IS_DBIAS) {
    step_dbias.clear();
  }

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    OVecC out_cast;
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      const CType tmp = in[i].data.elt[j];
      if constexpr (IS_DBIAS) {
        step_dbias.data.elt[j] += tmp;  // dbias: thread tile local accumulation
      }
      out_cast.data.elt[j] = static_cast<OType>(tmp * scale);
      out_trans[j].data.elt[i] = static_cast<OType>(tmp * scale);  // thread tile transpose

      __builtin_assume(amax >= 0);
      amax = fmaxf(fabsf(tmp), amax);
    }
    if (IS_FULL_TILE || valid_store) {
      out_cast.store_to(output_cast_tile, current_place + stride * i);
    }
  }

  if constexpr (IS_DBIAS) {
#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      CType elt = step_dbias.data.elt[j];
      elt = __shfl_sync(0xffffffff, elt, dbias_shfl_src_lane);  // shuffle data in a warp
      out_dbias.data.elt[j] += elt;
    }
  }
}

void populate_cast_transpose_dbias_workspace_config(const Tensor &cast_output, /*cast*/
                                                    Tensor *workspace, const int nvec_out) {
  const size_t row_length = cast_output.flat_last_dim();
  const size_t num_rows = cast_output.flat_first_dim();

  const size_t tile_size_y = (nvec_out * THREADS_PER_WARP);
  NVTE_CHECK(num_rows % nvec_out == 0, "Unsupported shape.");

  const size_t num_rows_partial_dbias = DIVUP(num_rows, tile_size_y);

  if (workspace->data.dptr == nullptr) {
    workspace->data.shape = {num_rows_partial_dbias, row_length};
    workspace->data.dtype = DType::kFloat32;
  } else {
    // Check that workspace matches expected size
    const size_t workspace_size = get_buffer_size_bytes(
        std::accumulate(workspace->data.shape.begin(), workspace->data.shape.end(), 1,
                        std::multiplies<size_t>()),
        workspace->data.dtype);
    const size_t required_size =
        get_buffer_size_bytes(num_rows_partial_dbias, row_length, DType::kFloat32);
    NVTE_CHECK(!workspace->data.shape.empty(), "Invalid workspace dims (expected (",
               num_rows_partial_dbias, ",", row_length, "), found ())");
    NVTE_CHECK(workspace_size >= required_size, "Invalid workspace (expected dims=(",
               num_rows_partial_dbias, ",", row_length, "), dtype=", to_string(DType::kFloat32),
               "; found dims=", workspace->data.shape,
               ", dtype=", typeToNumBits(workspace->data.dtype), " bits)");
  }
}

template <int nvec, typename ComputeType, typename OutputType>
__global__ void __launch_bounds__(reduce_dbias_num_threads)
    reduce_dbias_kernel(OutputType *const dbias_output, const ComputeType *const dbias_partial,
                        const int row_length, const int num_rows) {
  using ComputeVec = Vec<ComputeType, nvec>;
  using OutputVec = Vec<OutputType, nvec>;

  const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

  if (thread_id * nvec >= row_length) {
    return;
  }

  const ComputeType *const thread_in_base = dbias_partial + thread_id * nvec;
  OutputType *const thread_out_base = dbias_output + thread_id * nvec;

  const int stride_in_vec = row_length / nvec;

  ComputeVec ldg_vec;
  ComputeVec acc_vec;
  acc_vec.clear();
  for (int i = 0; i < num_rows; ++i) {
    ldg_vec.load_from(thread_in_base, i * stride_in_vec);
#pragma unroll
    for (int e = 0; e < nvec; ++e) {
      acc_vec.data.elt[e] += ldg_vec.data.elt[e];
    }
  }

  OutputVec stg_vec;
#pragma unroll
  for (int e = 0; e < nvec; ++e) {
    stg_vec.data.elt[e] = OutputType(acc_vec.data.elt[e]);
  }
  stg_vec.store_to(thread_out_base, 0);
}

template <typename InputType>
void reduce_dbias(const Tensor &workspace, Tensor *dbias, const size_t row_length,
                  const size_t num_rows, const int nvec_out, cudaStream_t stream) {
  constexpr int reduce_dbias_store_bytes = 8;  // stg.64
  constexpr int reduce_dbias_nvec = reduce_dbias_store_bytes / sizeof(InputType);

  NVTE_CHECK(row_length % reduce_dbias_nvec == 0, "Unsupported shape.");

  const size_t reduce_dbias_row_length = row_length;
  const size_t reduce_dbias_num_rows =
      DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
  const size_t reduce_dbias_num_blocks =
      DIVUP(row_length, reduce_dbias_num_threads * reduce_dbias_nvec);

  using DbiasOutputType = fp32;
  reduce_dbias_kernel<reduce_dbias_nvec, DbiasOutputType, InputType>
      <<<reduce_dbias_num_blocks, reduce_dbias_num_threads, 0, stream>>>(
          reinterpret_cast<InputType *>(dbias->data.dptr),
          reinterpret_cast<const fp32 *>(workspace.data.dptr), reduce_dbias_row_length,
          reduce_dbias_num_rows);
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ComputeType, typename Param,
          int nvec_in, int nvec_out, typename ParamOP,
          ComputeType (*OP)(ComputeType, const ParamOP &)>
__global__ void __launch_bounds__(cast_transpose_num_threads)
    cast_transpose_fused_kernel_notaligned(const Param param, const size_t row_length,
                                           const size_t num_rows, const size_t num_tiles) {
  static_assert(!(IS_DACT && IS_ACT), "forward and backward activation are mutually exclusive");
  using IType = typename Param::InputType;
  using IType2 = typename Param::InputType2;
  using OType = typename Param::OutputType;
  using CType = typename Param::ComputeType;
  using IVec = Vec<IType, nvec_in>;
  using IVec2 = Vec<IType2, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const unsigned int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x =
      (row_length + nvec_in * THREADS_PER_WARP - 1) / (nvec_in * THREADS_PER_WARP);
  const size_t tile_id =
      blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) + warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) {
    return;
  }

  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const size_t tile_offset =
      (tile_id_x * nvec_in + tile_id_y * row_length * nvec_out) * THREADS_PER_WARP;
  const size_t tile_offset_transp =
      (tile_id_y * nvec_out + tile_id_x * num_rows * nvec_in) * THREADS_PER_WARP;

  const IType *const my_input_tile = param.input + tile_offset;
  const IType2 *const my_act_input_tile = param.act_input + tile_offset;
  OType *const my_output_c_tile = param.output_c + tile_offset;
  OType *const my_output_t_tile = param.output_t + tile_offset_transp;
  CType *const my_partial_dbias_tile =
      param.workspace + (tile_id_x * (nvec_in * THREADS_PER_WARP) + tile_id_y * row_length);

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
  const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
  const unsigned int tile_length =
      row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP : row_length_rest;
  const unsigned int tile_height =
      row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP : row_height_rest;

  OVec *const my_scratch =
      reinterpret_cast<OVec *>(scratch) +
      (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) * (THREADS_PER_WARP + 1);

  CVec *const my_dbias_scratch = reinterpret_cast<CVec *>(scratch);

  IVec in[2][nvec_out];
  IVec2 act_in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space[n_iterations][nvec_in];

  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  size_t current_row = (tile_id_y * THREADS_PER_WARP + warp_id_in_tile * n_iterations) * nvec_out;
  unsigned int my_place =
      (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
  CType amax = 0;
  const CType scale = param.scale_ptr != nullptr ? *param.scale_ptr : 1;

  CVec partial_dbias;
  if constexpr (IS_DBIAS) {
    partial_dbias.clear();
  }

  {
    const bool valid_load = my_place < tile_length && warp_id_in_tile * n_iterations < tile_height;
#pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
      if (valid_load) {
        const size_t ld_offset = current_stride + my_place + stride * i;
        in[0][i].load_from(my_input_tile, ld_offset);
        if constexpr (IS_DACT) {
          act_in[0][i].load_from(my_act_input_tile, ld_offset);
        }
      } else {
        in[0][i].clear();
        if constexpr (IS_DACT) {
          act_in[0][i].clear();
        }
      }
    }
  }

#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
      const bool valid_load =
          my_place_in < tile_length && warp_id_in_tile * n_iterations + i + 1 < tile_height;
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        if (valid_load) {
          const size_t ld_offset = current_stride + my_place_in + stride * (nvec_out + j);
          in[current_in][j].load_from(my_input_tile, ld_offset);
          if constexpr (IS_DACT) {
            act_in[current_in][j].load_from(my_act_input_tile, ld_offset);
          }
        } else {
          in[current_in][j].clear();
          if constexpr (IS_DACT) {
            act_in[current_in][j].clear();
          }
        }
      }
    }
    CVec after_dact[nvec_out];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < nvec_out; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < nvec_in; ++k) {
        if constexpr (IS_DACT) {
          after_dact[j].data.elt[k] = CType(in[current_in ^ 1][j].data.elt[k]) *
                                      OP(act_in[current_in ^ 1][j].data.elt[k], {});
        } else if constexpr (IS_ACT) {
          after_dact[j].data.elt[k] = OP(in[current_in ^ 1][j].data.elt[k], {});
        } else {
          after_dact[j].data.elt[k] = CType(in[current_in ^ 1][j].data.elt[k]);
        }
      }
    }
    const int dbias_shfl_src_lane =
        (my_id_in_warp + i + warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    constexpr bool IS_FULL_TILE = false;
    const bool valid_store =
        (my_place < tile_length) && (warp_id_in_tile * n_iterations + i < tile_height);

    cast_and_transpose_regs<IS_DBIAS, IS_FULL_TILE>(after_dact, out_space[i], partial_dbias,
                                                    my_output_c_tile, current_place, stride, scale,
                                                    amax, dbias_shfl_src_lane, valid_store);

    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
    current_row += nvec_out;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; warp_id_in_tile * n_iterations + j < tile_length; ++j) {
      const bool valid_store = my_place < tile_height;
      if (valid_store) {
        my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile,
                                                                current_stride + my_place);
      }
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  if constexpr (IS_DBIAS) {
    my_dbias_scratch[threadIdx.x] = partial_dbias;
    __syncthreads();
    if (warp_id_in_tile == 0) {
#pragma unroll
      for (unsigned int i = 1; i < n_warps_per_tile; ++i) {
        CVec tmp = my_dbias_scratch[threadIdx.x + i * THREADS_PER_WARP];
#pragma unroll
        for (unsigned int j = 0; j < nvec_in; ++j) {
          partial_dbias.data.elt[j] += tmp.data.elt[j];
        }
      }
      if (my_id_in_warp < tile_length) {
        partial_dbias.store_to(my_partial_dbias_tile, my_id_in_warp);
      }
    }
  }

  // Reduce amax over block
  if (param.amax != nullptr) {
    amax = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(amax, warp_id);
    if (threadIdx.x == 0) {
      static_assert(std::is_same<CType, float>::value);
      atomicMaxFloat(param.amax, amax);
    }
  }

  // Update scale-inverse
  if (blockIdx.x == 0 && threadIdx.x == 0 && param.scale_inv != nullptr) {
    reciprocal<CType>(param.scale_inv, scale);
  }
}

static const char *ActTypeToString[] = {
    "none",      // 0
    "sigmoid",   // 1
    "dsigmoid",  // 2
    "gelu",      // 3
    "dgelu",     // 4
    "qgelu",     // 5
    "dqgelu",    // 6
    "silu",      // 7
    "dsilu",     // 8
    "relu",      // 9
    "drelu",     // 10
    "srelu",     // 11
    "dsrelu"     // 12
};

template <typename ComputeType, typename ParamOP, ComputeType (*OP)(ComputeType, const ParamOP &)>
constexpr int get_activation_type() {
  constexpr decltype(OP) ActivationList[] = {
      nullptr,                              // 0
      &sigmoid<ComputeType, ComputeType>,   // 1
      &dsigmoid<ComputeType, ComputeType>,  // 2
      &gelu<ComputeType, ComputeType>,      // 3
      &dgelu<ComputeType, ComputeType>,     // 4
      &qgelu<ComputeType, ComputeType>,     // 5
      &dqgelu<ComputeType, ComputeType>,    // 6
      &silu<ComputeType, ComputeType>,      // 7
      &dsilu<ComputeType, ComputeType>,     // 8
      &relu<ComputeType, ComputeType>,      // 9
      &drelu<ComputeType, ComputeType>,     // 10
      &srelu<ComputeType, ComputeType>,     // 11
      &dsrelu<ComputeType, ComputeType>     // 12
  };
#pragma unroll
  for (int i = 0; i < sizeof(ActivationList) / sizeof(ActivationList[0]); ++i) {
    if (OP == ActivationList[i]) {
      return i;
    }
  }
  return 0;
}

template <bool IS_DBIAS, bool IS_DACT, bool IS_ACT, typename ComputeType, typename ParamOP,
          ComputeType (*OP)(ComputeType, const ParamOP &)>
void cast_transpose_fused(const Tensor &input, const Tensor *act_input, Tensor *output,
                          Tensor *dbias, Tensor *workspace, cudaStream_t stream) {
  // Check tensors, unless querying dbias workspace
  if (!IS_DBIAS || workspace->data.dptr != nullptr) {
    CheckInputTensor(input, "cast_transpose_fused_input");
    CheckOutputTensor(*output, "output");
    if constexpr (IS_DBIAS) {
      NVTE_CHECK(dbias != nullptr && dbias->has_data());
      CheckOutputTensor(*dbias, "dbias");
    }
    if constexpr (IS_DACT) {
      NVTE_CHECK(act_input != nullptr && act_input->has_data());
      CheckInputTensor(*act_input, "act_input");
    }
  }

  // Check that inputs and outputs are available
  NVTE_CHECK(input.has_data(), "Input is not allocated");
  NVTE_CHECK(output->has_data(), "Output rowwise data is not allocated");
  NVTE_CHECK(output->has_columnwise_data(), "Output columnwise data is not allocated");

  // Flatten tensor to 2D
  NVTE_CHECK(input.data.shape == output->data.shape,
             "Input and output shapes do not match (input=", input.data.shape,
             ", output=", output->data.shape);
  const size_t row_length = input.flat_last_dim();
  const size_t num_rows = input.flat_first_dim();

  // Check that cast and transposed output data matches
  NVTE_CHECK(output->data.dtype == output->columnwise_data.dtype,
             "Cast and transposed output types must match.");
  NVTE_CHECK(output->scale_inv.dptr == output->columnwise_scale_inv.dptr,
             "Cast and transposed outputs need to share scale-inverse tensor.");

  if constexpr (IS_DBIAS) {
    NVTE_CHECK(dbias->data.dtype == input.data.dtype, "DBias must have the same type as input.");
    NVTE_CHECK(dbias->data.shape == std::vector<size_t>{row_length}, "Wrong shape of DBias.");
  }
  if constexpr (IS_DACT) {
    NVTE_CHECK(input.dtype() == act_input->dtype(), "Types of both inputs must match.");
    NVTE_CHECK(input.data.shape == act_input->data.shape, "Shapes of both inputs must match.");
  }

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OutputType, using InputType2 = InputType;
          using Param = CTDBiasDActParam<InputType, InputType2, OutputType, ComputeType>;

          constexpr int itype_size = sizeof(InputType);
          constexpr int itype2_size = sizeof(InputType2);
          constexpr int otype_size = sizeof(OutputType);

          const bool aligned =
              (row_length % THREADS_PER_WARP == 0) && (num_rows % THREADS_PER_WARP == 0);
          const bool jit_compiled = aligned && rtc::is_enabled();

          size_t load_size = (IS_DACT ? desired_load_size_dact : desired_load_size);
          size_t store_size = (IS_DACT ? desired_store_size_dact : desired_store_size);
          size_t num_blocks;

          if (jit_compiled) {
            // Pick kernel config
            std::vector<KernelConfig> kernel_configs;
            kernel_configs.reserve(16);
            const size_t sm_count = static_cast<size_t>(cuda::sm_count());
            auto add_config = [&](size_t load_size_config, size_t store_size_config) {
              kernel_configs.emplace_back(row_length, num_rows, itype_size, itype2_size, otype_size,
                                          load_size_config, store_size_config, sm_count, IS_DACT);
            };
            add_config(8, 8);
            add_config(4, 8);
            add_config(8, 4);
            add_config(4, 4);
            add_config(2, 8);
            add_config(8, 2);
            add_config(2, 4);
            add_config(4, 2);
            add_config(2, 2);
            add_config(1, 8);
            add_config(8, 1);
            add_config(1, 4);
            add_config(4, 1);
            add_config(1, 2);
            add_config(2, 1);
            add_config(1, 1);

            // Select the kernel configuration with the lowest cost
            const auto &kernel_config =
                *std::min_element(kernel_configs.begin(), kernel_configs.end());
            NVTE_CHECK(kernel_config.valid, "invalid kernel config");
            load_size = kernel_config.load_size;
            store_size = kernel_config.store_size;
            num_blocks = kernel_config.num_blocks;
          }

          const size_t nvec_in = load_size / itype_size;
          const size_t nvec_out = store_size / otype_size;
          const size_t tile_size_x = nvec_in * threads_per_warp;
          const size_t tile_size_y = nvec_out * threads_per_warp;
          const size_t num_tiles_x = DIVUP(row_length, tile_size_x);
          const size_t num_tiles_y = DIVUP(num_rows, tile_size_y);
          const size_t num_tiles = num_tiles_x * num_tiles_y;

          NVTE_CHECK(row_length % nvec_in == 0, "Unsupported shape.");
          NVTE_CHECK(num_rows % nvec_out == 0, "Unsupported shape.");

          if (!jit_compiled) {
            num_blocks = DIVUP(num_tiles * n_warps_per_tile, n_warps_per_block);
          } if constexpr (IS_DBIAS) {
            // Check workspace size
            populate_cast_transpose_dbias_workspace_config(*output, workspace, nvec_out);
            if (workspace->data.dptr == nullptr) {
              return;
            }
          }

          size_t VecOutputTypeSize;
          switch (nvec_out) {
            case 1:
              VecOutputTypeSize = sizeof(Vec<OutputType, 1>);
              break;
            case 2:
              VecOutputTypeSize = sizeof(Vec<OutputType, 2>);
              break;
            case 4:
              VecOutputTypeSize = sizeof(Vec<OutputType, 4>);
              break;
            case 8:
              VecOutputTypeSize = sizeof(Vec<OutputType, 8>);
              break;
          } size_t shared_size_transpose = cast_transpose_num_threads / n_warps_per_tile *
                                           (threads_per_warp + 1) * VecOutputTypeSize;

          if constexpr (IS_DBIAS) {
            size_t VecComputeTypeSize;
            switch (nvec_in) {
              case 1:
                VecComputeTypeSize = sizeof(Vec<ComputeType, 1>);
                break;
              case 2:
                VecComputeTypeSize = sizeof(Vec<ComputeType, 2>);
                break;
              case 4:
                VecComputeTypeSize = sizeof(Vec<ComputeType, 4>);
                break;
              case 8:
                VecComputeTypeSize = sizeof(Vec<ComputeType, 8>);
                break;
            }
            const size_t shared_size_dbias = cast_transpose_num_threads * VecComputeTypeSize;
            if (shared_size_transpose < shared_size_dbias) {
              shared_size_transpose = shared_size_dbias;
            }
          }

          Param param;
          param.input = reinterpret_cast<const InputType *>(input.data.dptr);
          param.output_c = reinterpret_cast<OutputType *>(output->data.dptr);
          param.output_t = reinterpret_cast<OutputType *>(output->columnwise_data.dptr);
          param.scale_ptr = reinterpret_cast<const ComputeType *>(output->scale.dptr);
          param.amax = reinterpret_cast<ComputeType *>(output->amax.dptr);
          param.scale_inv = reinterpret_cast<ComputeType *>(output->scale_inv.dptr);
          if constexpr (IS_DBIAS) {
            param.workspace = reinterpret_cast<ComputeType *>(workspace->data.dptr);
          } if constexpr (IS_DACT) {
            param.act_input = reinterpret_cast<const InputType2 *>(act_input->data.dptr);
          }

          // Runtime-compiled tuned kernel
          if (jit_compiled) {
            constexpr const char *itype_name = TypeInfo<InputType>::name;
            constexpr const char *itype2_name = TypeInfo<InputType2>::name;
            constexpr const char *otype_name = TypeInfo<OutputType>::name;

            int actType = 0;
            if constexpr (IS_DACT || IS_ACT) {
              actType = get_activation_type<ComputeType, ParamOP, OP>();
            }

            // Compile NVRTC kernel if needed and launch
            auto &rtc_manager = rtc::KernelManager::instance();
            const std::string kernel_label = concat_strings(
                "cast_transpose_fusion"
                ",itype=",
                itype_name, ",itype2=", itype2_name, ",otype=", otype_name,
                ",load_size=", load_size, ",store_size=", store_size, ",IS_DBIAS=", IS_DBIAS,
                ",IS_DACT=", IS_DACT, ",IS_ACT=", IS_ACT,
                ",activationType=", ActTypeToString[actType]);

            if (!rtc_manager.is_compiled(kernel_label)) {
              std::string code = string_code_transpose_rtc_cast_transpose_fusion_cu;
              code = regex_replace(code, "__ITYPE__", itype_name);
              code = regex_replace(code, "__ITYPE2__", itype2_name);
              code = regex_replace(code, "__OTYPE__", otype_name);
              code = regex_replace(code, "__LOAD_SIZE__", load_size);
              code = regex_replace(code, "__STORE_SIZE__", store_size);
              code = regex_replace(code, "__WARPS_PER_TILE__", n_warps_per_tile);
              code = regex_replace(code, "__BLOCK_SIZE__", cast_transpose_num_threads);
              code = regex_replace(code, "__IS_DBIAS__", IS_DBIAS);
              code = regex_replace(code, "__IS_DACT__", IS_DACT);
              code = regex_replace(code, "__IS_ACT__", IS_ACT);
              code = regex_replace(code, "__ACTIVATION_TYPE__", actType);

              rtc_manager.compile(
                  kernel_label, "cast_transpose_fusion_kernel_optimized", code,
                  "transformer_engine/common/transpose/rtc/cast_transpose_fusion.cu");
            }

            rtc_manager.set_cache_config(kernel_label, CU_FUNC_CACHE_PREFER_SHARED);

            rtc_manager.launch(kernel_label, num_blocks, cast_transpose_num_threads,
                               shared_size_transpose, stream, param, row_length, num_rows,
                               num_tiles);
          } else {  // Statically-compiled general kernel
            constexpr size_t load_size = IS_DACT ? desired_load_size_dact : desired_load_size;
            constexpr size_t store_size = IS_DACT ? desired_store_size_dact : desired_store_size;
            constexpr size_t nvec_in = load_size / itype_size;
            constexpr size_t nvec_out = store_size / otype_size;

            NVTE_CHECK(row_length % nvec_in == 0, "Unsupported shape.");
            NVTE_CHECK(num_rows % nvec_out == 0, "Unsupported shape.");

            NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                cast_transpose_fused_kernel_notaligned<IS_DBIAS, IS_DACT, IS_ACT, ComputeType,
                                                       Param, nvec_in, nvec_out, Empty, OP>,
                cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            cast_transpose_fused_kernel_notaligned<IS_DBIAS, IS_DACT, IS_ACT, ComputeType, Param,
                                                   nvec_in, nvec_out, Empty, OP>
                <<<num_blocks, cast_transpose_num_threads, shared_size_transpose, stream>>>(
                    param, row_length, num_rows, num_tiles);
            NVTE_CHECK_CUDA(cudaGetLastError());
          }

          if constexpr (IS_DBIAS) {
            reduce_dbias<InputType>(*workspace, dbias, row_length, num_rows, nvec_out, stream);
          });  // NOLINT(*)
  );           // NOLINT(*)
}

template <int nvec_in, int nvec_out, typename CType, typename IType, typename OType,
          typename ParamOP, CType (*OP1)(CType, const ParamOP &),
          CType (*OP2)(CType, const ParamOP &)>
__global__ void __launch_bounds__(cast_transpose_num_threads)
    dgated_act_cast_transpose_kernel(const IType *const input, const IType *const act_input,
                                     OType *const output_c, OType *const output_t,
                                     const CType *const scale_ptr, CType *const amax,
                                     CType *const scale_inv, const size_t row_length,
                                     const size_t num_rows, const size_t num_tiles) {
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x = row_length / (nvec_in * THREADS_PER_WARP);
  const size_t tile_id =
      blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) + warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) {
    return;
  }

  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType *const my_input_tile =
      input + (tile_id_x * nvec_in + tile_id_y * row_length * nvec_out) * THREADS_PER_WARP;
  const IType *const my_act_input_tile =
      act_input + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP;
  const IType *const my_gate_input_tile =
      act_input + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP +
      row_length;
  OType *const my_output_c_tile_0 =
      output_c + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP;
  OType *const my_output_c_tile_1 =
      output_c + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP +
      row_length;
  OType *const my_output_t_tile_0 =
      output_t + (tile_id_y * nvec_out + tile_id_x * num_rows * nvec_in) * THREADS_PER_WARP;
  OType *const my_output_t_tile_1 =
      output_t + (tile_id_y * nvec_out + tile_id_x * num_rows * nvec_in) * THREADS_PER_WARP +
      row_length * num_rows;
  OVec *const my_scratch =
      reinterpret_cast<OVec *>(scratch) +
      (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) * (THREADS_PER_WARP + 1);

  IVec in[2][nvec_out];
  IVec act_in[2][nvec_out];
  IVec gate_in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space_0[n_iterations][nvec_in];
  OVec out_space_1[n_iterations][nvec_in];

  const size_t stride = row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  unsigned int my_place =
      (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
  const size_t stride2 = 2 * row_length / nvec_in;
  size_t current_stride2 = warp_id_in_tile * n_iterations * nvec_out * stride2;
  CType max = 0;
  const CType scale = scale_ptr != nullptr ? *scale_ptr : 1;

  CVec partial_dbias;

#pragma unroll
  for (unsigned int i = 0; i < nvec_out; ++i) {
    in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
    act_in[0][i].load_from(my_act_input_tile, current_stride2 + my_place + stride2 * i);
    gate_in[0][i].load_from(my_gate_input_tile, current_stride2 + my_place + stride2 * i);
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride2 + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
#pragma unroll
      for (unsigned int j = 0; j < nvec_out; ++j) {
        in[current_in][j].load_from(my_input_tile,
                                    current_stride + my_place_in + stride * (nvec_out + j));
        act_in[current_in][j].load_from(my_act_input_tile,
                                        current_stride2 + my_place_in + stride2 * (nvec_out + j));
        gate_in[current_in][j].load_from(my_gate_input_tile,
                                         current_stride2 + my_place_in + stride2 * (nvec_out + j));
      }
    }
    CVec after_dact[nvec_out];   // NOLINT(*)
    CVec after_dgate[nvec_out];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < nvec_out; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < nvec_in; ++k) {
        after_dact[j].data.elt[k] = OP1(act_in[current_in ^ 1][j].data.elt[k], {}) *
                                    CType(in[current_in ^ 1][j].data.elt[k]) *
                                    CType(gate_in[current_in ^ 1][j].data.elt[k]);
        after_dgate[j].data.elt[k] = CType(in[current_in ^ 1][j].data.elt[k]) *
                                     OP2(act_in[current_in ^ 1][j].data.elt[k], {});
      }
    }
    OVec out_trans_0[nvec_in];  // NOLINT(*)
    OVec out_trans_1[nvec_in];  // NOLINT(*)

    constexpr bool IS_DBIAS = false;
    constexpr bool IS_FULL_TILE = true;
    constexpr bool valid_store = true;
    constexpr int dbias_shfl_src_lane = 0;

    cast_and_transpose_regs<IS_DBIAS, IS_FULL_TILE>(after_dact, out_trans_0, partial_dbias,
                                                    my_output_c_tile_0, current_place, stride2,
                                                    scale, max, dbias_shfl_src_lane, valid_store);

    cast_and_transpose_regs<IS_DBIAS, IS_FULL_TILE>(after_dgate, out_trans_1, partial_dbias,
                                                    my_output_c_tile_1, current_place, stride2,
                                                    scale, max, dbias_shfl_src_lane, valid_store);

#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space_0[i][j].data.vec = out_trans_0[j].data.vec;
      out_space_1[i][j].data.vec = out_trans_1[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
    current_stride2 += nvec_out * stride2;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space_0[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile_0,
                                                              current_stride + my_place);
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space_1[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile_1,
                                                              current_stride + my_place);
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  // Reduce amax over block
  if (amax != nullptr) {
    max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);
    if (threadIdx.x == 0) {
      static_assert(std::is_same<CType, float>::value);
      atomicMaxFloat(amax, max);
    }
  }

  // Update scale-inverse
  if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv != nullptr) {
    reciprocal<CType>(scale_inv, scale);
  }
}

template <int nvec_in, int nvec_out, typename CType, typename IType, typename OType,
          typename ParamOP, CType (*OP1)(CType, const ParamOP &),
          CType (*OP2)(CType, const ParamOP &)>
__global__ void __launch_bounds__(cast_transpose_num_threads)
    dgated_act_cast_transpose_kernel_notaligned(const IType *const input,
                                                const IType *const act_input, OType *const output_c,
                                                OType *const output_t, const CType *const scale_ptr,
                                                CType *const amax, CType *const scale_inv,
                                                const size_t row_length, const size_t num_rows,
                                                const size_t num_tiles) {
  using IVec = Vec<IType, nvec_in>;
  using OVec = Vec<OType, nvec_out>;
  using CVec = Vec<CType, nvec_in>;

  extern __shared__ char scratch[];

  const int warp_id = threadIdx.x / THREADS_PER_WARP;
  const int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
  const size_t num_tiles_x =
      (row_length + nvec_in * THREADS_PER_WARP - 1) / (nvec_in * THREADS_PER_WARP);
  const size_t tile_id =
      blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) + warp_id / n_warps_per_tile;
  if (tile_id >= num_tiles) return;
  const size_t tile_id_x = tile_id % num_tiles_x;
  const size_t tile_id_y = tile_id / num_tiles_x;

  const IType *const my_input_tile =
      input + (tile_id_x * nvec_in + tile_id_y * row_length * nvec_out) * THREADS_PER_WARP;
  const IType *const my_act_input_tile =
      act_input + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP;
  const IType *const my_gate_input_tile =
      act_input + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP +
      row_length;
  OType *const my_output_c_tile_0 =
      output_c + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP;
  OType *const my_output_c_tile_1 =
      output_c + (tile_id_x * nvec_in + tile_id_y * row_length * 2 * nvec_out) * THREADS_PER_WARP +
      row_length;
  OType *const my_output_t_tile_0 =
      output_t + (tile_id_y * nvec_out + tile_id_x * num_rows * nvec_in) * THREADS_PER_WARP;
  OType *const my_output_t_tile_1 =
      output_t + (tile_id_y * nvec_out + tile_id_x * num_rows * nvec_in) * THREADS_PER_WARP +
      row_length * num_rows;
  const size_t stride = row_length / nvec_in;
  const size_t stride2 = 2 * row_length / nvec_in;
  const size_t output_stride = num_rows / nvec_out;
  const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
  const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
  const unsigned int tile_length =
      row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP : row_length_rest;
  const unsigned int tile_height =
      row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP : row_height_rest;

  OVec *const my_scratch =
      reinterpret_cast<OVec *>(scratch) +
      (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) * (THREADS_PER_WARP + 1);

  IVec in[2][nvec_out];
  IVec act_in[2][nvec_out];
  IVec gate_in[2][nvec_out];
  const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
  constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
  OVec out_space_0[n_iterations][nvec_in];
  OVec out_space_1[n_iterations][nvec_in];

  size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
  size_t current_stride2 = warp_id_in_tile * n_iterations * nvec_out * stride2;
  unsigned int my_place =
      (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
  CType max = 0;
  const CType scale = scale_ptr != nullptr ? *scale_ptr : 1;

  CVec partial_dbias;

  {
    const bool valid_load = my_place < tile_length && warp_id_in_tile * n_iterations < tile_height;
#pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
      if (valid_load) {
        in[0][i].load_from(my_input_tile, current_stride + my_place + stride * i);
        act_in[0][i].load_from(my_act_input_tile, current_stride2 + my_place + stride2 * i);
        gate_in[0][i].load_from(my_gate_input_tile, current_stride2 + my_place + stride2 * i);
      } else {
        in[0][i].clear();
        act_in[0][i].clear();
        gate_in[0][i].clear();
      }
    }
  }
#pragma unroll
  for (unsigned int i = 0; i < n_iterations; ++i) {
    const size_t current_place = current_stride2 + my_place;
    const unsigned int my_place_in = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    const unsigned int current_in = (i + 1) % 2;
    if (i < n_iterations - 1) {
      {
        const bool valid_load =
            my_place_in < tile_length && warp_id_in_tile * n_iterations + i + 1 < tile_height;
#pragma unroll
        for (unsigned int j = 0; j < nvec_out; ++j) {
          if (valid_load) {
            in[current_in][j].load_from(my_input_tile,
                                        current_stride + my_place_in + stride * (nvec_out + j));
            act_in[current_in][j].load_from(
                my_act_input_tile, current_stride2 + my_place_in + stride2 * (nvec_out + j));
            gate_in[current_in][j].load_from(
                my_gate_input_tile, current_stride2 + my_place_in + stride2 * (nvec_out + j));
          } else {
            in[current_in][j].clear();
            act_in[current_in][j].clear();
            gate_in[current_in][j].clear();
          }
        }
      }
    }
    CVec after_dact[nvec_out];   // NOLINT(*)
    CVec after_dgate[nvec_out];  // NOLINT(*)
#pragma unroll
    for (unsigned int j = 0; j < nvec_out; ++j) {
#pragma unroll
      for (unsigned int k = 0; k < nvec_in; ++k) {
        after_dact[j].data.elt[k] = OP1(act_in[current_in ^ 1][j].data.elt[k], {}) *
                                    CType(in[current_in ^ 1][j].data.elt[k]) *
                                    CType(gate_in[current_in ^ 1][j].data.elt[k]);
        after_dgate[j].data.elt[k] = CType(in[current_in ^ 1][j].data.elt[k]) *
                                     OP2(act_in[current_in ^ 1][j].data.elt[k], {});
      }
    }
    OVec out_trans_0[nvec_in];  // NOLINT(*)
    OVec out_trans_1[nvec_in];  // NOLINT(*)

    constexpr bool IS_DBIAS = false;
    constexpr bool IS_FULL_TILE = false;
    constexpr int dbias_shfl_src_lane = 0;
    const bool valid_store =
        (my_place < tile_length) && (warp_id_in_tile * n_iterations + i < tile_height);

    cast_and_transpose_regs<IS_DBIAS, IS_FULL_TILE>(after_dact, out_trans_0, partial_dbias,
                                                    my_output_c_tile_0, current_place, stride2,
                                                    scale, max, dbias_shfl_src_lane, valid_store);
    cast_and_transpose_regs<IS_DBIAS, IS_FULL_TILE>(after_dgate, out_trans_1, partial_dbias,
                                                    my_output_c_tile_1, current_place, stride2,
                                                    scale, max, dbias_shfl_src_lane, valid_store);

#pragma unroll
    for (unsigned int j = 0; j < nvec_in; ++j) {
      out_space_0[i][j].data.vec = out_trans_0[j].data.vec;
      out_space_1[i][j].data.vec = out_trans_1[j].data.vec;
    }
    my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
    current_stride += nvec_out * stride;
    current_stride2 += nvec_out * stride2;
  }

  for (unsigned int i = 0; i < nvec_in; ++i) {
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space_0[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; warp_id_in_tile * n_iterations + j < tile_length; ++j) {
      const bool valid_store = my_place < tile_height;
      if (valid_store) {
        my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile_0,
                                                                current_stride + my_place);
      }
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
#pragma unroll
    for (unsigned int j = 0; j < n_iterations; ++j) {
      my_scratch[(my_id_in_warp + THREADS_PER_WARP - j - warp_id_in_tile * n_iterations) %
                 THREADS_PER_WARP] = out_space_1[j][i];
    }
    __syncthreads();
    my_place =
        (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) % THREADS_PER_WARP;
    current_stride = i * output_stride + warp_id_in_tile * n_iterations * output_stride * nvec_in;
    for (unsigned int j = 0; warp_id_in_tile * n_iterations + j < tile_length; ++j) {
      const bool valid_store = my_place < tile_height;
      if (valid_store) {
        my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile_1,
                                                                current_stride + my_place);
      }
      my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
      current_stride += output_stride * nvec_in;
    }
    __syncthreads();
  }

  // Reduce amax over block
  if (amax != nullptr) {
    max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);
    if (threadIdx.x == 0) {
      static_assert(std::is_same<CType, float>::value);
      atomicMaxFloat(amax, max);
    }
  }

  // Update scale-inverse
  if (blockIdx.x == 0 && threadIdx.x == 0 && scale_inv != nullptr) {
    reciprocal<CType>(scale_inv, scale);
  }
}

template <typename ComputeType, typename ParamOP, ComputeType (*OP1)(ComputeType, const ParamOP &),
          ComputeType (*OP2)(ComputeType, const ParamOP &)>
void dgated_act_cast_transpose(const Tensor &input, const Tensor &gated_act_input, Tensor *output,
                               cudaStream_t stream) {
  CheckInputTensor(input, "dgated_act_cast_transpose_input");
  CheckInputTensor(gated_act_input, "dgated_act_cast_transpose_gated_act_input");
  CheckOutputTensor(*output, "dgated_act_cast_transpose_output");

  NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(gated_act_input.data.shape.size() == 2, "Input must have 2 dimensions.");
  NVTE_CHECK(output->has_data() && output->has_columnwise_data(),
             "Both rowwise and columnwise data need to be allocated.");
  NVTE_CHECK(output->data.shape.size() == 2, "C output must have 2 dimensions.");
  NVTE_CHECK(output->columnwise_data.shape.size() == 2, "T output must have 2 dimensions.");
  const size_t row_length = input.data.shape[1];
  const size_t num_rows = input.data.shape[0];

  NVTE_CHECK(gated_act_input.data.shape[0] == num_rows, "Wrong dimension of output.");
  NVTE_CHECK(gated_act_input.data.shape[1] == row_length * 2, "Wrong dimension of output.");
  NVTE_CHECK(output->data.shape[0] == num_rows, "Wrong dimension of output.");
  NVTE_CHECK(output->data.shape[1] == row_length * 2, "Wrong dimension of output.");
  NVTE_CHECK(output->columnwise_data.shape[0] == row_length * 2, "Wrong dimension of T output.");
  NVTE_CHECK(output->columnwise_data.shape[1] == num_rows, "Wrong dimension of T output.");

  NVTE_CHECK(input.data.dtype == gated_act_input.data.dtype, "Types of both inputs must match.");

  NVTE_CHECK(output->data.dtype == output->columnwise_data.dtype,
             "C and T outputs need to have the same type.");
  NVTE_CHECK(output->scale_inv.dptr == output->columnwise_scale_inv.dptr,
             "C and T outputs need to share scale inverse tensor.");

  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.dtype(), InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          output->dtype(), OutputType, using InputType2 = InputType;
          /* dact fusion kernel uses more registers */
          constexpr int desired_load_size_dact = 4;
          constexpr int desired_store_size_dact = 4; constexpr int itype_size = sizeof(InputType);
          constexpr int otype_size = sizeof(OutputType);
          constexpr int nvec_in = desired_load_size_dact / itype_size;
          constexpr int nvec_out = desired_store_size_dact / otype_size;

          NVTE_CHECK(row_length % nvec_in == 0, "Unsupported shape.");
          NVTE_CHECK(num_rows % nvec_out == 0, "Unsupported shape.");
          const size_t n_tiles =
              DIVUP(row_length, static_cast<size_t>(nvec_in * THREADS_PER_WARP)) *
              DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
          const size_t n_warps_per_block = cast_transpose_num_threads / THREADS_PER_WARP;
          const size_t n_blocks = DIVUP(n_tiles * n_warps_per_tile, n_warps_per_block);

          const bool full_tile = row_length % (nvec_in * THREADS_PER_WARP) == 0 &&
                                 num_rows % (nvec_out * THREADS_PER_WARP) == 0;
          const size_t shmem_size = cast_transpose_num_threads / n_warps_per_tile *
                                    (THREADS_PER_WARP + 1) * sizeof(Vec<OutputType, nvec_out>);
          if (full_tile) {
            NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                dgated_act_cast_transpose_kernel<nvec_in, nvec_out, ComputeType, InputType,
                                                 OutputType, Empty, OP1, OP2>,
                cudaFuncAttributePreferredSharedMemoryCarveout, 100));

            dgated_act_cast_transpose_kernel<nvec_in, nvec_out, ComputeType, InputType, OutputType,
                                             Empty, OP1, OP2>
                <<<n_blocks, cast_transpose_num_threads, shmem_size, stream>>>(
                    reinterpret_cast<const InputType *>(input.data.dptr),
                    reinterpret_cast<const InputType *>(gated_act_input.data.dptr),
                    reinterpret_cast<OutputType *>(output->data.dptr),
                    reinterpret_cast<OutputType *>(output->columnwise_data.dptr),
                    reinterpret_cast<const fp32 *>(output->scale.dptr),
                    reinterpret_cast<fp32 *>(output->amax.dptr),
                    reinterpret_cast<fp32 *>(output->scale_inv.dptr), row_length, num_rows,
                    n_tiles);
            NVTE_CHECK_CUDA(cudaGetLastError());
          } else {
            NVTE_CHECK_CUDA(cudaFuncSetAttribute(
                dgated_act_cast_transpose_kernel_notaligned<nvec_in, nvec_out, ComputeType,
                                                            InputType, OutputType, Empty, OP1, OP2>,
                cudaFuncAttributePreferredSharedMemoryCarveout, 100));
            dgated_act_cast_transpose_kernel_notaligned<nvec_in, nvec_out, ComputeType, InputType,
                                                        OutputType, Empty, OP1, OP2>
                <<<n_blocks, cast_transpose_num_threads, shmem_size, stream>>>(
                    reinterpret_cast<const InputType *>(input.data.dptr),
                    reinterpret_cast<const InputType *>(gated_act_input.data.dptr),
                    reinterpret_cast<OutputType *>(output->data.dptr),
                    reinterpret_cast<OutputType *>(output->columnwise_data.dptr),
                    reinterpret_cast<const fp32 *>(output->scale.dptr),
                    reinterpret_cast<fp32 *>(output->amax.dptr),
                    reinterpret_cast<fp32 *>(output->scale_inv.dptr), row_length, num_rows,
                    n_tiles);
            NVTE_CHECK_CUDA(cudaGetLastError());
          });  // NOLINT(*)
  );           // NOLINT(*)
}

// Explicit template instantiation
template void cast_transpose_fused<true, false, false, float, transformer_engine::Empty, nullptr>(
    const Tensor &, const Tensor *, Tensor *, Tensor *, Tensor *, cudaStream_t);
#define NVTE_INSTANTIATE_ACTIVATION(op)                                                    \
  template void cast_transpose_fused<false, false, true, float, transformer_engine::Empty, \
                                     transformer_engine::op<float, float>>(                \
      const Tensor &, const Tensor *, Tensor *, Tensor *, Tensor *, cudaStream_t);         \
  template void cast_transpose_fused<false, true, false, float, transformer_engine::Empty, \
                                     transformer_engine::d##op<float, float>>(             \
      const Tensor &, const Tensor *, Tensor *, Tensor *, Tensor *, cudaStream_t);
NVTE_INSTANTIATE_ACTIVATION(relu);
NVTE_INSTANTIATE_ACTIVATION(srelu);
NVTE_INSTANTIATE_ACTIVATION(gelu);
NVTE_INSTANTIATE_ACTIVATION(qgelu);
NVTE_INSTANTIATE_ACTIVATION(silu);
#undef NVTE_INSTANTIATE_ACTIVATION

}  // namespace detail

}  // namespace transformer_engine

using ComputeType = typename transformer_engine::fp32;

void nvte_cast_transpose_dbias(const NVTETensor input, NVTETensor output, NVTETensor dbias,
                               NVTETensor workspace, cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose_dbias);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = false;
  constexpr bool IS_ACT = false;

  constexpr const NVTETensor activation_input = nullptr;

  cast_transpose_fused<IS_DBIAS, IS_DACT, IS_ACT, ComputeType, Empty, nullptr>(
      *convertNVTETensorCheck(input), convertNVTETensor(activation_input),
      convertNVTETensor(output), convertNVTETensor(dbias), convertNVTETensor(workspace), stream);
}

void nvte_cast_transpose_dbias_dgelu(const NVTETensor input, const NVTETensor act_input,
                                     NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose_dbias_dgelu);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  cast_transpose_fused<IS_DBIAS, IS_DACT, IS_ACT, ComputeType, Empty, dgelu<fp32, fp32>>(
      *convertNVTETensorCheck(input), convertNVTETensorCheck(act_input),
      convertNVTETensorCheck(output), convertNVTETensorCheck(dbias), convertNVTETensor(workspace),
      stream);
}

void nvte_cast_transpose_dbias_dsilu(const NVTETensor input, const NVTETensor silu_input,
                                     NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose_dbias_dsilu);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  cast_transpose_fused<IS_DBIAS, IS_DACT, IS_ACT, ComputeType, Empty, dsilu<fp32, fp32>>(
      *convertNVTETensorCheck(input), convertNVTETensorCheck(silu_input),
      convertNVTETensorCheck(output), convertNVTETensorCheck(dbias), convertNVTETensor(workspace),
      stream);
}

void nvte_cast_transpose_dbias_drelu(const NVTETensor input, const NVTETensor relu_input,
                                     NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                     cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose_dbias_drelu);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  cast_transpose_fused<IS_DBIAS, IS_DACT, IS_ACT, ComputeType, Empty, drelu<fp32, fp32>>(
      *convertNVTETensorCheck(input), convertNVTETensorCheck(relu_input),
      convertNVTETensorCheck(output), convertNVTETensorCheck(dbias), convertNVTETensor(workspace),
      stream);
}

void nvte_cast_transpose_dbias_dsrelu(const NVTETensor input, const NVTETensor srelu_input,
                                      NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose_dbias_dsrelu);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  cast_transpose_fused<IS_DBIAS, IS_DACT, IS_ACT, ComputeType, Empty, dsrelu<fp32, fp32>>(
      *convertNVTETensorCheck(input), convertNVTETensorCheck(srelu_input),
      convertNVTETensorCheck(output), convertNVTETensorCheck(dbias), convertNVTETensor(workspace),
      stream);
}

void nvte_cast_transpose_dbias_dqgelu(const NVTETensor input, const NVTETensor qgelu_input,
                                      NVTETensor output, NVTETensor dbias, NVTETensor workspace,
                                      cudaStream_t stream) {
  NVTE_API_CALL(nvte_cast_transpose_dbias_dqgelu);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  constexpr bool IS_DBIAS = true;
  constexpr bool IS_DACT = true;
  constexpr bool IS_ACT = false;

  cast_transpose_fused<IS_DBIAS, IS_DACT, IS_ACT, ComputeType, Empty, dqgelu<fp32, fp32>>(
      *convertNVTETensorCheck(input), convertNVTETensorCheck(qgelu_input),
      convertNVTETensorCheck(output), convertNVTETensorCheck(dbias), convertNVTETensor(workspace),
      stream);
}

void nvte_dgeglu_cast_transpose(const NVTETensor input, const NVTETensor gated_act_input,
                                NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dgeglu_cast_transpose);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  dgated_act_cast_transpose<ComputeType, Empty, dgelu<fp32, fp32>, gelu<fp32, fp32>>(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(gated_act_input),
      convertNVTETensorCheck(output), stream);
}

void nvte_dswiglu_cast_transpose(const NVTETensor input, const NVTETensor swiglu_input,
                                 NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dswiglu_cast_transpose);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  dgated_act_cast_transpose<ComputeType, Empty, dsilu<fp32, fp32>, silu<fp32, fp32>>(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(swiglu_input),
      convertNVTETensorCheck(output), stream);
}

void nvte_dreglu_cast_transpose(const NVTETensor input, const NVTETensor gated_act_input,
                                NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dreglu_cast_transpose);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  dgated_act_cast_transpose<ComputeType, Empty, drelu<fp32, fp32>, relu<fp32, fp32>>(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(gated_act_input),
      convertNVTETensorCheck(output), stream);
}

void nvte_dsreglu_cast_transpose(const NVTETensor input, const NVTETensor gated_act_input,
                                 NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dsreglu_cast_transpose);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  dgated_act_cast_transpose<ComputeType, Empty, dsrelu<fp32, fp32>, srelu<fp32, fp32>>(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(gated_act_input),
      convertNVTETensorCheck(output), stream);
}

void nvte_dqgeglu_cast_transpose(const NVTETensor input, const NVTETensor gated_act_input,
                                 NVTETensor output, cudaStream_t stream) {
  NVTE_API_CALL(nvte_dqgeglu_cast_transpose);
  using namespace transformer_engine;
  using namespace transformer_engine::detail;

  dgated_act_cast_transpose<ComputeType, Empty, dqgelu<fp32, fp32>, qgelu<fp32, fp32>>(
      *convertNVTETensorCheck(input), *convertNVTETensorCheck(gated_act_input),
      convertNVTETensorCheck(output), stream);
}
