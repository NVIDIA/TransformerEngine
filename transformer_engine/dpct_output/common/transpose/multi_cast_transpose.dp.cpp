/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <transformer_engine/transpose.h>
#include <iostream>
#include <cfloat>
#include <vector>
#include "../utils.dp.hpp"
#include "../common.h"

namespace transformer_engine {

namespace {

// Parameters to tune
constexpr int n_warps_per_tile = 4;
constexpr int threads_per_block = THREADS_PER_WARP * n_warps_per_tile;
constexpr int desired_load_size = 8;
constexpr int desired_store_size = 8;
constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB

struct MultiCastTransposeArgs {
  // (input) Data buffers for input tensors
  void* input_list[kMaxTensorsPerKernel];
  // (output) Data buffers for cast output tensors
  void* output_c_list[kMaxTensorsPerKernel];
  // (output) Data buffers for transpose output tensors
  void* output_t_list[kMaxTensorsPerKernel];
  // (input) Scaling factor for output tensors
  void* scale_list[kMaxTensorsPerKernel];
  // (output) AMAX's of input tensors
  void* amax_list[kMaxTensorsPerKernel];
  // Input matrix heights
  int num_rows_list[kMaxTensorsPerKernel];
  // Input matrix widths
  int row_length_list[kMaxTensorsPerKernel];
  // Prefix sum (with leading zero) of CUDA blocks needed for each
  // tensor
  int block_range[kMaxTensorsPerKernel+1];
  // Number of tensors being processed by kernel
  int num_tensors;
};

template <
  int nvec_in,
  int nvec_out,
  bool aligned,
  typename CType,
  typename IType,
  typename OType>
/*
DPCT1110:146: The total declared local variable size in device function multi_cast_transpose_kernel exceeds 128 bytes and may cause high register pressure. Consult with your hardware vendor to find the total register size available and adjust the code, or use smaller sub-group size to avoid high register pressure.
*/
void

multi_cast_transpose_kernel(MultiCastTransposeArgs args,
                            const sycl::nd_item<3> &item_ct1,
                            sycl::local_accessor<OVecT, 2> shared_output_t,
                            float *staging) {
  using IVec = Vec<IType, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = n_warps_per_tile;
  const int tid = item_ct1.get_local_id(2);
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = item_ct1.get_group(2);

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int n_iterations = THREADS_PER_WARP / n_warps_per_tile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id+1] <= bid) {
    ++tensor_id;
  }
  const IType* input = reinterpret_cast<const IType*>(args.input_list[tensor_id]);
  OType* output_c = reinterpret_cast<OType*>(args.output_c_list[tensor_id]);
  OType* output_t = reinterpret_cast<OType*>(args.output_t_list[tensor_id]);
  const CType* scale_ptr = reinterpret_cast<CType*>(args.scale_list[tensor_id]);
  const CType scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  CType* amax = reinterpret_cast<CType*>(args.amax_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads n_iterations subtiles, casts to output
  // type, and transposes in registers.
  OVecT local_output_t[nvec_in][n_iterations];
  CType local_amax = 0;
  #pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
    #pragma unroll
    for (int i2 = 0; i2 < nvec_out; ++i2) {
      const int row = tile_row + i1 * nvec_out + i2;
      const int col = tile_col + j1 * nvec_in;
      IVec local_input;
      OVecC local_output_c;
      if constexpr (aligned) {
        local_input.load_from(&input[row * row_length + col]);
      } else {
        local_input.clear();
        if (row < num_rows) {
          #pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              local_input.data.elt[j2] = input[row * row_length + col + j2];
            }
          }
        }
      }
      #pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        const CType x = CType(local_input.data.elt[j2]);
        const OType y = OType(scale * x);
        local_output_c.data.elt[j2] = y;
        local_output_t[j2][iter].data.elt[i2] = y;
        __builtin_assume(local_amax >= 0);
        local_amax = sycl::fmax(sycl::fabs((float)x), local_amax);
      }
      if constexpr (aligned) {
        local_output_c.store_to(&output_c[row * row_length + col]);
      } else {
        if (row < num_rows) {
          #pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              output_c[row * row_length + col + j2] = local_output_c.data.elt[j2];
            }
          }
        }
      }
    }
  }

  // Copy transposed output from registers to global memory

#pragma unroll
  for (int j2 = 0; j2 < nvec_in; ++j2) {
    #pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidy + iter * bdimy;
      const int j1 = tidx;
      shared_output_t[j1][i1] = local_output_t[j2][iter];
    }
    /*
    DPCT1118:147: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:504: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidx;
      const int j1 = tidy + iter * bdimy;
      const int row = tile_row + i1 * nvec_out;
      const int col = tile_col + j1 * nvec_in + j2;
      if constexpr (aligned) {
        shared_output_t[j1][i1].store_to(&output_t[col * num_rows + row]);
      } else {
        if (col < row_length) {
          #pragma unroll
          for (int i2 = 0; i2 < nvec_out; ++i2) {
            if (row + i2 < num_rows) {
              output_t[col * num_rows + row + i2] = shared_output_t[j1][i1].data.elt[i2];
            }
          }
        }
      }
    }
    /*
    DPCT1118:148: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:505: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
  }

  // Finalize fp8 factors
  local_amax =
      reduce_max<n_warps_per_tile>(local_amax, tidy, item_ct1, staging);
  if (tid == 0) {
    static_assert(std::is_same<CType, float>::value);
    if (amax != nullptr) atomicMaxFloat(amax, local_amax);
  }
}

}  // namespace

void multi_cast_transpose(const std::vector<Tensor *> input_list,
                          std::vector<Tensor *> cast_output_list,
                          std::vector<Tensor *> transposed_output_list,
                          dpct::queue_ptr stream) {
  // Check that number of tensors is valid
  NVTE_CHECK(cast_output_list.size() == input_list.size(),
             "Number of input and C output tensors must match");
  NVTE_CHECK(transposed_output_list.size() == input_list.size(),
             "Number of input and T output tensors must match");
  if (input_list.empty()) {
    return;
  }

  // Check that tensor properties are valid
  DType itype = input_list[0]->data.dtype;
  DType otype = cast_output_list[0]->data.dtype;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    const auto& input = *input_list[tensor_id];
    const auto& cast_output = *cast_output_list[tensor_id];
    const auto& transposed_output = *transposed_output_list[tensor_id];
    CheckInputTensor(input, "multi_cast_transpose_input_" + std::to_string(tensor_id));
    CheckInputTensor(cast_output, "multi_cast_output_" + std::to_string(tensor_id));
    CheckInputTensor(transposed_output, "multi_transpose_output_" + std::to_string(tensor_id));

    NVTE_CHECK(input.data.dtype == itype,
               "Input tensor types do not match.");
    NVTE_CHECK(cast_output.data.dtype == otype,
               "C output tensor types do not match.");
    NVTE_CHECK(transposed_output.data.dtype == otype,
               "T output tensor types do not match.");

    NVTE_CHECK(input.data.shape.size() == 2,
               "Input tensor must have 2 dimensions.");
    NVTE_CHECK(cast_output.data.shape == input.data.shape,
               "C output tensor shape does not match input tensor.");
    NVTE_CHECK(transposed_output.data.shape.size() == 2,
               "T output tensor shape does not match input tensor.");
    NVTE_CHECK(transposed_output.data.shape[0] == input.data.shape[1],
               "T output tensor shape does not match input tensor.");
    NVTE_CHECK(transposed_output.data.shape[1] == input.data.shape[0],
               "T output tensor shape does not match input tensor.");
  }

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  const int tile_dim_m = THREADS_PER_WARP * desired_store_size / typeToSize(otype);
  const int tile_dim_n = THREADS_PER_WARP * desired_load_size / typeToSize(itype);

  // Add tensors to kernel argument struct
  MultiCastTransposeArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    // Launch kernel if argument struct is full
    if (kernel_args_aligned.num_tensors == kMaxTensorsPerKernel) {
      /*
      DPCT1038:149: When the kernel function name is used as a macro argument,
      the migration result may be incorrect. You need to verify the definition
      of the macro.
      */
            TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
                itype, InputType,
                TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
                    otype, OutputType,
                    constexpr int nvec_in =
                        desired_load_size / sizeof(InputType);
                    constexpr int nvec_out =
                        desired_store_size / sizeof(OutputType);
                    const int n_blocks =
                        kernel_args_aligned
                            .block_range[kernel_args_aligned.num_tensors];
                    {
                        dpct::has_capability_or_fail(stream->get_device(),
                                                     {sycl::aspect::fp16});

                        stream->submit([&](sycl::handler &cgh) {
                            /*
                            DPCT1101:537: 'THREADS_PER_WARP' expression was
                            replaced with a value. Modify the code to use the
                            original expression, provided in comments, if it is
                            correct.
                            */
                            /*
                            DPCT1101:538: 'THREADS_PER_WARP+1' expression was
                            replaced with a value. Modify the code to use the
                            original expression, provided in comments, if it is
                            correct.
                            */
                            sycl::local_accessor<OVecT, 2>
                                shared_output_t_acc_ct1(
                                    sycl::range<2>(32 /*THREADS_PER_WARP*/,
                                                   33 /*THREADS_PER_WARP+1*/),
                                    cgh);
                            sycl::local_accessor<float, 1> staging_acc_ct1(
                                sycl::range<1>(n_warps_per_tile), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(1, 1, n_blocks) *
                                        sycl::range<3>(1, 1, threads_per_block),
                                    sycl::range<3>(1, 1, threads_per_block)),
                                [=](sycl::nd_item<3>
                                        item_ct1) [[intel::reqd_sub_group_size(
                                    32)]] {
                                    multi_cast_transpose_kernel<
                                        nvec_in, nvec_out, true, fp32,
                                        InputType, OutputType>(
                                        kernel_args_aligned, item_ct1,
                                        shared_output_t_acc_ct1,
                                        staging_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                                });
                        });
                    }); // NOLINT(*)
            );          // NOLINT(*)
      kernel_args_aligned.num_tensors = 0;
    }
    if (kernel_args_unaligned.num_tensors == kMaxTensorsPerKernel) {
      /*
      DPCT1038:150: When the kernel function name is used as a macro argument,
      the migration result may be incorrect. You need to verify the definition
      of the macro.
      */
            TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
                itype, InputType,
                TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
                    otype, OutputType,
                    constexpr int nvec_in =
                        desired_load_size / sizeof(InputType);
                    constexpr int nvec_out =
                        desired_store_size / sizeof(OutputType);
                    const int n_blocks =
                        kernel_args_unaligned
                            .block_range[kernel_args_unaligned.num_tensors];
                    {
                        dpct::has_capability_or_fail(stream->get_device(),
                                                     {sycl::aspect::fp16});

                        stream->submit([&](sycl::handler &cgh) {
                            /*
                            DPCT1101:539: 'THREADS_PER_WARP' expression was
                            replaced with a value. Modify the code to use the
                            original expression, provided in comments, if it is
                            correct.
                            */
                            /*
                            DPCT1101:540: 'THREADS_PER_WARP+1' expression was
                            replaced with a value. Modify the code to use the
                            original expression, provided in comments, if it is
                            correct.
                            */
                            sycl::local_accessor<OVecT, 2>
                                shared_output_t_acc_ct1(
                                    sycl::range<2>(32 /*THREADS_PER_WARP*/,
                                                   33 /*THREADS_PER_WARP+1*/),
                                    cgh);
                            sycl::local_accessor<float, 1> staging_acc_ct1(
                                sycl::range<1>(n_warps_per_tile), cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(
                                    sycl::range<3>(1, 1, n_blocks) *
                                        sycl::range<3>(1, 1, threads_per_block),
                                    sycl::range<3>(1, 1, threads_per_block)),
                                [=](sycl::nd_item<3>
                                        item_ct1) [[intel::reqd_sub_group_size(
                                    32)]] {
                                    multi_cast_transpose_kernel<
                                        nvec_in, nvec_out, false, fp32,
                                        InputType, OutputType>(
                                        kernel_args_unaligned, item_ct1,
                                        shared_output_t_acc_ct1,
                                        staging_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                                });
                        });
                    }); // NOLINT(*)
            );          // NOLINT(*)
      kernel_args_unaligned.num_tensors = 0;
    }

    // Calculate number of thread blocks needed for tensor
    const int num_rows = input_list[tensor_id]->data.shape[0];
    const int row_length = input_list[tensor_id]->data.shape[1];
    const int num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Figure out whether to use aligned or unaligned kernel
    const bool aligned = ((num_tiles_m * tile_dim_m == num_rows)
                          && (num_tiles_n * tile_dim_n == row_length));
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id]->data.dptr);
    kernel_args.output_c_list[pos] = cast_output_list[tensor_id]->data.dptr;
    kernel_args.output_t_list[pos] = transposed_output_list[tensor_id]->data.dptr;
    kernel_args.scale_list[pos] = cast_output_list[tensor_id]->scale.dptr;
    kernel_args.amax_list[pos] = cast_output_list[tensor_id]->amax.dptr;
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.block_range[pos+1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  if (kernel_args_aligned.num_tensors > 0) {
    /*
    DPCT1038:151: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
        TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
            itype, InputType,
            TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
                otype, OutputType,
                constexpr int nvec_in = desired_load_size / sizeof(InputType);
                constexpr int nvec_out =
                    desired_store_size / sizeof(OutputType);
                const int n_blocks =
                    kernel_args_aligned
                        .block_range[kernel_args_aligned.num_tensors];
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->submit([&](sycl::handler &cgh) {
                        /*
                        DPCT1101:541: 'THREADS_PER_WARP' expression was replaced
                        with a value. Modify the code to use the original
                        expression, provided in comments, if it is correct.
                        */
                        /*
                        DPCT1101:542: 'THREADS_PER_WARP+1' expression was
                        replaced with a value. Modify the code to use the
                        original expression, provided in comments, if it is
                        correct.
                        */
                        sycl::local_accessor<OVecT, 2> shared_output_t_acc_ct1(
                            sycl::range<2>(32 /*THREADS_PER_WARP*/,
                                           33 /*THREADS_PER_WARP+1*/),
                            cgh);
                        sycl::local_accessor<float, 1> staging_acc_ct1(
                            sycl::range<1>(n_warps_per_tile), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, n_blocks) *
                                    sycl::range<3>(1, 1, threads_per_block),
                                sycl::range<3>(1, 1, threads_per_block)),
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(32)]] {
                                    multi_cast_transpose_kernel<
                                        nvec_in, nvec_out, true, fp32,
                                        InputType, OutputType>(
                                        kernel_args_aligned, item_ct1,
                                        shared_output_t_acc_ct1,
                                        staging_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                                });
                    });
                }); // NOLINT(*)
        );          // NOLINT(*)
  }
  if (kernel_args_unaligned.num_tensors > 0) {
    /*
    DPCT1038:152: When the kernel function name is used as a macro argument, the
    migration result may be incorrect. You need to verify the definition of the
    macro.
    */
        TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
            itype, InputType,
            TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
                otype, OutputType,
                constexpr int nvec_in = desired_load_size / sizeof(InputType);
                constexpr int nvec_out =
                    desired_store_size / sizeof(OutputType);
                const int n_blocks =
                    kernel_args_unaligned
                        .block_range[kernel_args_unaligned.num_tensors];
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->submit([&](sycl::handler &cgh) {
                        /*
                        DPCT1101:543: 'THREADS_PER_WARP' expression was replaced
                        with a value. Modify the code to use the original
                        expression, provided in comments, if it is correct.
                        */
                        /*
                        DPCT1101:544: 'THREADS_PER_WARP+1' expression was
                        replaced with a value. Modify the code to use the
                        original expression, provided in comments, if it is
                        correct.
                        */
                        sycl::local_accessor<OVecT, 2> shared_output_t_acc_ct1(
                            sycl::range<2>(32 /*THREADS_PER_WARP*/,
                                           33 /*THREADS_PER_WARP+1*/),
                            cgh);
                        sycl::local_accessor<float, 1> staging_acc_ct1(
                            sycl::range<1>(n_warps_per_tile), cgh);

                        cgh.parallel_for(
                            sycl::nd_range<3>(
                                sycl::range<3>(1, 1, n_blocks) *
                                    sycl::range<3>(1, 1, threads_per_block),
                                sycl::range<3>(1, 1, threads_per_block)),
                            [=](sycl::nd_item<3> item_ct1)
                                [[intel::reqd_sub_group_size(32)]] {
                                    multi_cast_transpose_kernel<
                                        nvec_in, nvec_out, false, fp32,
                                        InputType, OutputType>(
                                        kernel_args_unaligned, item_ct1,
                                        shared_output_t_acc_ct1,
                                        staging_acc_ct1
                                            .get_multi_ptr<
                                                sycl::access::decorated::no>()
                                            .get());
                                });
                    });
                }); // NOLINT(*)
        );          // NOLINT(*)
  }
}

}  // namespace transformer_engine

void nvte_multi_cast_transpose(size_t num_tensors, const NVTETensor *input_list,
                               NVTETensor *cast_output_list,
                               NVTETensor *transposed_output_list,
                               dpct::queue_ptr stream) {
  NVTE_API_CALL(nvte_multi_cast_transpose);
  using namespace transformer_engine;
  std::vector<Tensor*> input_list_,
    cast_output_list_, transposed_output_list_;
  for (size_t i = 0; i < num_tensors; ++i) {
    input_list_.push_back(reinterpret_cast<Tensor*>(const_cast<NVTETensor&>(input_list[i])));
    cast_output_list_.push_back(reinterpret_cast<Tensor*>(cast_output_list[i]));
    transposed_output_list_.push_back(reinterpret_cast<Tensor*>(transposed_output_list[i]));
  }
  multi_cast_transpose(input_list_,
                       cast_output_list_,
                       transposed_output_list_,
                       stream);
}
