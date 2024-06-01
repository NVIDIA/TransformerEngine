/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transpose.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <iostream>
#include <type_traits>
#include "../utils.cuh"
#include "../common.h"
#include "../util/math.h"

namespace transformer_engine {

// STUFF TO TUNE
constexpr unsigned int n_warps_per_tile = 4;
constexpr unsigned int max_threads_per_block = 256;
static_assert(n_warps_per_tile * THREADS_PER_WARP <= max_threads_per_block);
constexpr unsigned int cast_transpose_num_threads = n_warps_per_tile * THREADS_PER_WARP;

template <bool full_tile, int nvec_in, int nvec_out, typename IVec, typename OVec, typename CType>
inline __device__ void cast_and_transpose_regs(const IVec (&in)[nvec_out],
                                               OVec (&out_trans)[nvec_in],
                                               typename OVec::type *output_cast_tile,
                                               const size_t current_place,
                                               const size_t stride,
                                               CType &max,  // NOLINT(*)
                                               const CType scale,
                                               const bool valid_store) {
    using T = typename OVec::type;
    using OVecC = Vec<T, nvec_in>;
    #pragma unroll
    for (unsigned int i = 0; i < nvec_out; ++i) {
        OVecC out_cast;
        #pragma unroll
        for (unsigned int j = 0; j < nvec_in; ++j) {
            const CType tmp = static_cast<CType>(in[i].data.elt[j]);
            const T elt_o = T(scale * tmp);

            out_cast.data.elt[j]     = elt_o;
            out_trans[j].data.elt[i] = elt_o;  // thread tile transpose

            __builtin_assume(max >= 0);
            max = fmaxf(fabsf(tmp), max);
        }
        if (full_tile || valid_store) {
            out_cast.store_to(output_cast_tile, current_place + stride * i);
        }
    }
}

template <int nvec_in, int nvec_out,
          typename CType, typename IType, typename OType, typename ParamOP,
          CType (*OP1)(CType, const ParamOP&), 
          CType (*OP2)(CType, const ParamOP&)>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
dgated_act_cast_transpose_kernel(const IType * const input,
                                 const IType * const act_input,
                                 OType * const output_c,
                                 OType * const output_t,
                                 const CType * const scale_ptr,
                                 CType * const amax,
                                 CType * const scale_inv,
                                 const size_t row_length,
                                 const size_t num_rows,
                                 const size_t num_tiles) {
    using IVec = Vec<IType, nvec_in>;
    using OVec = Vec<OType, nvec_out>;
    using CVec = Vec<CType, nvec_in>;

    extern __shared__ char scratch[];

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
    const size_t num_tiles_x = row_length / (nvec_in * THREADS_PER_WARP);
    const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                            warp_id / n_warps_per_tile;
    if (tile_id >= num_tiles) {
        return;
    }
  
    const size_t tile_id_x = tile_id % num_tiles_x;
    const size_t tile_id_y = tile_id / num_tiles_x;

    const IType * const my_input_tile = input + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * nvec_out) *
                                                THREADS_PER_WARP;
    const IType * const my_act_input_tile = act_input + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP;
    const IType * const my_gate_input_tile = act_input + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP + row_length;
    OType * const my_output_c_tile_0 = output_c + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP;
    OType * const my_output_c_tile_1 = output_c + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP + row_length;
    OType * const my_output_t_tile_0 = output_t + (tile_id_y * nvec_out +
                                                tile_id_x * num_rows * nvec_in) *
                                                THREADS_PER_WARP;
    OType * const my_output_t_tile_1 = output_t + (tile_id_y * nvec_out +
                                                tile_id_x * num_rows * nvec_in) *
                                                THREADS_PER_WARP + row_length * num_rows;
    OVec * const my_scratch = reinterpret_cast<OVec*>(scratch) +
                                (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                                (THREADS_PER_WARP + 1);

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
    unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                            warp_id_in_tile * n_iterations) %
                            THREADS_PER_WARP;
    const size_t stride2 = 2 * row_length / nvec_in;
    size_t current_stride2 = warp_id_in_tile * n_iterations * nvec_out * stride2;
    CType max = 0;
    const CType scale = scale_ptr != nullptr ? *scale_ptr : 1;
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
        CVec after_dact[nvec_out];  // NOLINT(*)
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
        cast_and_transpose_regs<true>(after_dact, out_trans_0, my_output_c_tile_0,
                                    current_place, stride2, max, scale, true);
        OVec out_trans_1[nvec_in];  // NOLINT(*)
        cast_and_transpose_regs<true>(after_dgate, out_trans_1, my_output_c_tile_1,
                                    current_place, stride2, max, scale, true);
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
            my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                    j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space_0[j][i];
        }
        __syncthreads();
        my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
                THREADS_PER_WARP;
        current_stride = i * output_stride +
                        warp_id_in_tile * n_iterations * output_stride * nvec_in;
        for (unsigned int j = 0; j < n_iterations; ++j) {
            my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile_0,
                                                                    current_stride + my_place);
            my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
            current_stride += output_stride * nvec_in;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned int j = 0; j < n_iterations; ++j) {
            my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                    j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space_1[j][i];
        }
        __syncthreads();
        my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
                THREADS_PER_WARP;
        current_stride = i * output_stride +
                        warp_id_in_tile * n_iterations * output_stride * nvec_in;
        for (unsigned int j = 0; j < n_iterations; ++j) {
            my_scratch[j + warp_id_in_tile * n_iterations].store_to(my_output_t_tile_1,
                                                                    current_stride + my_place);
            my_place = (my_place + THREADS_PER_WARP - 1) % THREADS_PER_WARP;
            current_stride += output_stride * nvec_in;
        }
        __syncthreads();
    }

    /* warp tile amax reduce*/
    max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);

    if (threadIdx.x == 0) {
        static_assert(std::is_same<CType, float>::value);
        if (amax != nullptr) {
            atomicMaxFloat(amax, max);
        }
        if (scale_inv != nullptr) {
            reciprocal<float>(scale_inv, scale);
        }
    }
}

template <int nvec_in, int nvec_out,
         typename CType, typename IType, typename OType,
         typename ParamOP,
         CType (*OP1)(CType, const ParamOP&),
         CType (*OP2)(CType, const ParamOP&)>
__global__ void
__launch_bounds__(cast_transpose_num_threads)
dgated_act_cast_transpose_kernel_notaligned(const IType * const input,
                                        const IType * const act_input,
                                        OType * const output_c,
                                        OType * const output_t,
                                        const CType * const scale_ptr,
                                        CType * const amax,
                                        CType * const scale_inv,
                                        const size_t row_length,
                                        const size_t num_rows,
                                        const size_t num_tiles) {
    using IVec = Vec<IType, nvec_in>;
    using OVec = Vec<OType, nvec_out>;
    using CVec = Vec<CType, nvec_in>;

    extern __shared__ char scratch[];

    const int warp_id = threadIdx.x / THREADS_PER_WARP;
    const int my_id_in_warp = threadIdx.x % THREADS_PER_WARP;
    const size_t num_tiles_x = (row_length + nvec_in * THREADS_PER_WARP - 1) /
                                (nvec_in * THREADS_PER_WARP);
    const size_t tile_id = blockIdx.x * blockDim.x / (THREADS_PER_WARP * n_warps_per_tile) +
                            warp_id / n_warps_per_tile;
    if (tile_id >= num_tiles) return;
    const size_t tile_id_x = tile_id % num_tiles_x;
    const size_t tile_id_y = tile_id / num_tiles_x;

    const IType * const my_input_tile = input + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * nvec_out) *
                                                THREADS_PER_WARP;
    const IType * const my_act_input_tile = act_input + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP;
    const IType * const my_gate_input_tile = act_input + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP + row_length;
    OType * const my_output_c_tile_0 = output_c + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP;
    OType * const my_output_c_tile_1 = output_c + (tile_id_x * nvec_in +
                                                tile_id_y * row_length * 2 * nvec_out) *
                                                THREADS_PER_WARP + row_length;
    OType * const my_output_t_tile_0 = output_t + (tile_id_y * nvec_out +
                                                tile_id_x * num_rows * nvec_in) *
                                                THREADS_PER_WARP;
    OType * const my_output_t_tile_1 = output_t + (tile_id_y * nvec_out +
                                                tile_id_x * num_rows * nvec_in) *
                                                THREADS_PER_WARP + row_length * num_rows;
    const size_t stride = row_length / nvec_in;
    const size_t stride2 = 2 * row_length / nvec_in;
    const size_t output_stride = num_rows / nvec_out;
    const size_t row_length_rest = stride - tile_id_x * THREADS_PER_WARP;
    const size_t row_height_rest = output_stride - tile_id_y * THREADS_PER_WARP;
    const unsigned int tile_length = row_length_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                        : row_length_rest;
    const unsigned int tile_height = row_height_rest > THREADS_PER_WARP ? THREADS_PER_WARP
                                                                        : row_height_rest;

    OVec * const my_scratch = reinterpret_cast<OVec*>(scratch) +
                                (my_id_in_warp + warp_id / n_warps_per_tile * THREADS_PER_WARP) *
                                (THREADS_PER_WARP + 1);

    IVec in[2][nvec_out];
    IVec act_in[2][nvec_out];
    IVec gate_in[2][nvec_out];
    const unsigned int warp_id_in_tile = warp_id % n_warps_per_tile;
    constexpr unsigned int n_iterations = THREADS_PER_WARP / n_warps_per_tile;
    OVec out_space_0[n_iterations][nvec_in];
    OVec out_space_1[n_iterations][nvec_in];

    size_t current_stride = warp_id_in_tile * n_iterations * nvec_out * stride;
    size_t current_stride2 = warp_id_in_tile * n_iterations * nvec_out * stride2;
    unsigned int my_place = (my_id_in_warp + THREADS_PER_WARP -
                            warp_id_in_tile * n_iterations) %
                            THREADS_PER_WARP;
    CType max = 0;
    const CType scale = scale_ptr != nullptr ? *scale_ptr : 1;
    {
        const bool valid_load = my_place < tile_length &&
                                warp_id_in_tile * n_iterations < tile_height;
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
                const bool valid_load = my_place_in < tile_length &&
                                        warp_id_in_tile * n_iterations + i + 1 < tile_height;
                #pragma unroll
                for (unsigned int j = 0; j < nvec_out; ++j) {
                    if (valid_load) {
                        in[current_in][j].load_from(my_input_tile,
                                                    current_stride + my_place_in + stride * (nvec_out + j));
                        act_in[current_in][j].load_from(my_act_input_tile,
                                                    current_stride2 + my_place_in + stride2 * (nvec_out + j));
                        gate_in[current_in][j].load_from(my_gate_input_tile,
                                                    current_stride2 + my_place_in + stride2 * (nvec_out + j));
                    } else {
                        in[current_in][j].clear();
                        act_in[current_in][j].clear();
                        gate_in[current_in][j].clear();
                    }
                }
            }
        }
        CVec after_dact[nvec_out];  // NOLINT(*)
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
        const bool valid_store = my_place < tile_length &&
                                warp_id_in_tile * n_iterations + i < tile_height;
        cast_and_transpose_regs<false>(after_dact, out_trans_0, my_output_c_tile_0,
                                    current_place, stride2, max, scale, valid_store);
        cast_and_transpose_regs<false>(after_dgate, out_trans_1, my_output_c_tile_1,
                                    current_place, stride2, max, scale, valid_store);
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
            my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                        j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space_0[j][i];
        }
        __syncthreads();
        my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
                THREADS_PER_WARP;
        current_stride = i * output_stride +
                        warp_id_in_tile * n_iterations * output_stride * nvec_in;
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
            my_scratch[(my_id_in_warp + THREADS_PER_WARP -
                        j - warp_id_in_tile * n_iterations) % THREADS_PER_WARP] = out_space_1[j][i];
        }
        __syncthreads();
        my_place = (my_id_in_warp + THREADS_PER_WARP - warp_id_in_tile * n_iterations) %
                THREADS_PER_WARP;
        current_stride = i * output_stride +
                        warp_id_in_tile * n_iterations * output_stride * nvec_in;
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

    /* warp tile amax reduce*/
    max = reduce_max<cast_transpose_num_threads / THREADS_PER_WARP>(max, warp_id);

    if (threadIdx.x == 0) {
        static_assert(std::is_same<CType, float>::value);
        if (amax != nullptr)  {
            atomicMaxFloat(amax, max);
        }
        if (scale_inv != nullptr) {
            reciprocal<float>(scale_inv, scale);
        }
    }
}

template <typename ComputeType, typename ParamOP,
         ComputeType (*OP1)(ComputeType, const ParamOP&),
         ComputeType (*OP2)(ComputeType, const ParamOP&)>
void dgated_act_cast_transpose(const Tensor &input,
                           const Tensor &gated_act_input,
                           Tensor *cast_output,
                           Tensor *transposed_output,
                           cudaStream_t stream) {
    CheckInputTensor(input, "dgated_act_cast_transpose_input");
    CheckInputTensor(gated_act_input, "dgated_act_cast_transpose_gated_act_input");
    CheckOutputTensor(*cast_output, "dgated_act_cast_transpose_cast_output");
    CheckOutputTensor(*transposed_output, "dgated_act_cast_transpose_transposed_output");

    NVTE_CHECK(input.data.shape.size() == 2, "Input must have 2 dimensions.");
    NVTE_CHECK(gated_act_input.data.shape.size() == 2, "Input must have 2 dimensions.");
    NVTE_CHECK(cast_output->data.shape.size() == 2, "C output must have 2 dimensions.");
    NVTE_CHECK(transposed_output->data.shape.size() == 2, "T output must have 2 dimensions.");
    const size_t row_length = input.data.shape[1];
    const size_t num_rows = input.data.shape[0];

    NVTE_CHECK(gated_act_input.data.shape[0] == num_rows, "Wrong dimension of output.");
    NVTE_CHECK(gated_act_input.data.shape[1] == row_length * 2, "Wrong dimension of output.");
    NVTE_CHECK(cast_output->data.shape[0] == num_rows, "Wrong dimension of output.");
    NVTE_CHECK(cast_output->data.shape[1] == row_length * 2, "Wrong dimension of output.");
    NVTE_CHECK(transposed_output->data.shape[0] == row_length * 2, "Wrong dimension of T output.");
    NVTE_CHECK(transposed_output->data.shape[1] == num_rows, "Wrong dimension of T output.");

    NVTE_CHECK(input.data.dtype == gated_act_input.data.dtype, "Types of both inputs must match.");

    NVTE_CHECK(cast_output->data.dtype == transposed_output->data.dtype, "C and T outputs need to have the same type.");
    NVTE_CHECK(cast_output->amax.dptr == transposed_output->amax.dptr, "C and T outputs need to share amax tensor.");
    NVTE_CHECK(cast_output->scale.dptr == transposed_output->scale.dptr, "C and T outputs need to share scale tensor.");
    NVTE_CHECK(cast_output->scale_inv.dptr == transposed_output->scale_inv.dptr, "C and T outputs need to share scale inverse tensor.");

    TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(input.data.dtype, InputType,
        TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(cast_output->data.dtype, OutputType,
            using InputType2 = InputType;
            /* dact fusion kernel uses more registers */
            constexpr int desired_load_size_dact = 4;
            constexpr int desired_store_size_dact = 4;
            constexpr int itype_size = sizeof(InputType);
            constexpr int otype_size = sizeof(OutputType);
            constexpr int nvec_in = desired_load_size_dact / itype_size;
            constexpr int nvec_out = desired_store_size_dact / otype_size;

            NVTE_CHECK(row_length % nvec_in  == 0, "Unsupported shape.");
            NVTE_CHECK(num_rows   % nvec_out == 0, "Unsupported shape.");
            const size_t n_tiles = DIVUP(row_length, static_cast<size_t>(nvec_in * THREADS_PER_WARP)) *
                                    DIVUP(num_rows, static_cast<size_t>(nvec_out * THREADS_PER_WARP));
            const size_t n_warps_per_block = cast_transpose_num_threads / THREADS_PER_WARP;
            const size_t n_blocks = DIVUP(n_tiles * n_warps_per_tile, n_warps_per_block);

            const bool full_tile = row_length % (nvec_in * THREADS_PER_WARP) == 0 &&
                                    num_rows % (nvec_out * THREADS_PER_WARP) == 0;
            const size_t shmem_size = cast_transpose_num_threads / n_warps_per_tile * 
                                       (THREADS_PER_WARP + 1) * sizeof(Vec<OutputType, nvec_out>);
            if (full_tile) {
                cudaFuncSetAttribute(
                    dgated_act_cast_transpose_kernel
                        <nvec_in, nvec_out, ComputeType, InputType, OutputType, Empty, OP1, OP2>,
                    cudaFuncAttributePreferredSharedMemoryCarveout,
                    100);

                dgated_act_cast_transpose_kernel
                    <nvec_in, nvec_out, ComputeType, InputType, OutputType, Empty, OP1, OP2>
                    <<<n_blocks, cast_transpose_num_threads, shmem_size, stream>>>(
                        reinterpret_cast<const InputType *>(input.data.dptr),
                        reinterpret_cast<const InputType *>(gated_act_input.data.dptr),
                        reinterpret_cast<OutputType *>(cast_output->data.dptr),
                        reinterpret_cast<OutputType *>(transposed_output->data.dptr),
                        reinterpret_cast<const fp32 *>(cast_output->scale.dptr),
                        reinterpret_cast<fp32 *>(cast_output->amax.dptr),
                        reinterpret_cast<fp32 *>(cast_output->scale_inv.dptr),
                        row_length, num_rows, n_tiles);
            } else {
                cudaFuncSetAttribute(
                    dgated_act_cast_transpose_kernel_notaligned
                        <nvec_in, nvec_out, ComputeType, InputType, OutputType, Empty, OP1, OP2>,
                    cudaFuncAttributePreferredSharedMemoryCarveout,
                    100);
                dgated_act_cast_transpose_kernel_notaligned
                    <nvec_in, nvec_out, ComputeType, InputType, OutputType, Empty, OP1, OP2>
                    <<<n_blocks, cast_transpose_num_threads, shmem_size, stream>>>(
                        reinterpret_cast<const InputType *>(input.data.dptr),
                        reinterpret_cast<const InputType *>(gated_act_input.data.dptr),
                        reinterpret_cast<OutputType *>(cast_output->data.dptr),
                        reinterpret_cast<OutputType *>(transposed_output->data.dptr),
                        reinterpret_cast<const fp32 *>(cast_output->scale.dptr),
                        reinterpret_cast<fp32 *>(cast_output->amax.dptr),
                        reinterpret_cast<fp32 *>(cast_output->scale_inv.dptr),
                        row_length, num_rows, n_tiles);
            }
        ); // NOLINT(*)
    );  // NOLINT(*)
}

}  // namespace transformer_engine

void nvte_dgeglu_cast_transpose(const NVTETensor input,
                                const NVTETensor gated_act_input,
                                NVTETensor cast_output,
                                NVTETensor transposed_output,
                                cudaStream_t stream) {
    NVTE_API_CALL(nvte_dgeglu_cast_transpose);
    using namespace transformer_engine;

    constexpr auto dActivation = &dgelu<fp32, fp32>;
    constexpr auto  Activation =  &gelu<fp32, fp32>;

    dgated_act_cast_transpose<fp32, Empty, dActivation, Activation>(
        *reinterpret_cast<const Tensor*>(input),
        *reinterpret_cast<const Tensor*>(gated_act_input),
        reinterpret_cast<Tensor*>(cast_output),
        reinterpret_cast<Tensor*>(transposed_output),
        stream);
}

void nvte_dswiglu_cast_transpose(const NVTETensor input,
                                 const NVTETensor swiglu_input,
                                 NVTETensor cast_output,
                                 NVTETensor transposed_output,
                                 cudaStream_t stream) {
    NVTE_API_CALL(nvte_dswiglu_cast_transpose);
    using namespace transformer_engine;

    constexpr auto dActivation = &dsilu<fp32, fp32>;
    constexpr auto  Activation =  &silu<fp32, fp32>;

    dgated_act_cast_transpose<fp32, Empty, dActivation, Activation>(
        *reinterpret_cast<const Tensor*>(input),
        *reinterpret_cast<const Tensor*>(swiglu_input),
        reinterpret_cast<Tensor*>(cast_output),
        reinterpret_cast<Tensor*>(transposed_output),
        stream);
}

void nvte_dreglu_cast_transpose(const NVTETensor input,
                                const NVTETensor gated_act_input,
                                NVTETensor cast_output,
                                NVTETensor transposed_output,
                                cudaStream_t stream) {
    NVTE_API_CALL(nvte_dreglu_cast_transpose);
    using namespace transformer_engine;

    constexpr auto dActivation = &drelu<fp32, fp32>;
    constexpr auto  Activation =  &relu<fp32, fp32>;

    dgated_act_cast_transpose<fp32, Empty, dActivation, Activation>(
        *reinterpret_cast<const Tensor*>(input),
        *reinterpret_cast<const Tensor*>(gated_act_input),
        reinterpret_cast<Tensor*>(cast_output),
        reinterpret_cast<Tensor*>(transposed_output),
        stream);
}

void nvte_dsreglu_cast_transpose(const NVTETensor input,
                                 const NVTETensor gated_act_input,
                                 NVTETensor cast_output,
                                 NVTETensor transposed_output,
                                 cudaStream_t stream) {
    NVTE_API_CALL(nvte_dsreglu_cast_transpose);
    using namespace transformer_engine;

    constexpr auto dActivation = &dsrelu<fp32, fp32>;
    constexpr auto  Activation =  &srelu<fp32, fp32>;

    dgated_act_cast_transpose<fp32, Empty, dActivation, Activation>(
        *reinterpret_cast<const Tensor*>(input),
        *reinterpret_cast<const Tensor*>(gated_act_input),
        reinterpret_cast<Tensor*>(cast_output),
        reinterpret_cast<Tensor*>(transposed_output),
        stream);
}

void nvte_dqgeglu_cast_transpose(const NVTETensor input,
                                 const NVTETensor gated_act_input,
                                 NVTETensor cast_output,
                                 NVTETensor transposed_output,
                                 cudaStream_t stream) {
    NVTE_API_CALL(nvte_dqgeglu_cast_transpose);
    using namespace transformer_engine;

    constexpr auto dActivation = &dqgelu<fp32, fp32>;
    constexpr auto  Activation =  &qgelu<fp32, fp32>;

    dgated_act_cast_transpose<fp32, Empty, dActivation, Activation>(
        *reinterpret_cast<const Tensor*>(input),
        *reinterpret_cast<const Tensor*>(gated_act_input),
        reinterpret_cast<Tensor*>(cast_output),
        reinterpret_cast<Tensor*>(transposed_output),
        stream);
}
