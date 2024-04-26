#define DPCT_COMPAT_RT_VERSION 12010
/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <assert.h>
#include <stdint.h>

#include <cfloat>
#include <limits>

#include <transformer_engine/softmax.h>
#include "../common.h"
#include "../utils.dp.hpp"
#include "../util/logging.h"


namespace transformer_engine {

template <typename Datatype, int ELEMENTS_PER_LDG>
SYCL_EXTERNAL __inline__ void copy_vector(Datatype *dst, const Datatype *src);

SYCL_EXTERNAL template <>
__inline__ void copy_vector<bf16, 1>(bf16 *dst, const bf16 *src) {
  *dst = *src;
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<bf16, 4>(bf16 *dst, const bf16 *src) {
  *((sycl::float2 *)dst) = *((sycl::float2 *)src); // NOLINT(*)
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<fp16, 1>(fp16 *dst, const fp16 *src) {
  *dst = *src;
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<fp16, 4>(fp16 *dst, const fp16 *src) {
  *((sycl::float2 *)dst) = *((sycl::float2 *)src); // NOLINT(*)
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<uint8_t, 1>(uint8_t *dst, const uint8_t *src) {
  *dst = *src;
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<uint8_t, 4>(uint8_t *dst, const uint8_t *src) {
  *((sycl::half2 *)dst) = *((sycl::half2 *)src); // NOLINT(*)
}

template <typename Datatype, int ELEMENTS_PER_LDG>
SYCL_EXTERNAL __inline__ void copy_zero_vector(Datatype *dst);

SYCL_EXTERNAL template <> __inline__ void copy_zero_vector<bf16, 1>(bf16 *dst) {
  *dst = 0.0f;
}

SYCL_EXTERNAL template <> __inline__ void copy_zero_vector<bf16, 4>(bf16 *dst) {
  *((sycl::float2 *)dst) = sycl::float2(0.0f, 0.0f); // NOLINT(*)
}

SYCL_EXTERNAL template <>
__inline__ void copy_zero_vector<fp16, 1>(fp16 *dst) {
        *dst = 0.0f;
}

SYCL_EXTERNAL template <> __inline__ void copy_zero_vector<fp16, 4>(fp16 *dst) {
  *((sycl::float2 *)dst) = sycl::float2(0.0f, 0.0f); // NOLINT(*)
}

template<typename T>
struct Add {
  SYCL_EXTERNAL __dpct_inline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __dpct_inline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
SYCL_EXTERNAL __dpct_inline__ T
WARP_SHFL_XOR_NATIVE(T value, int laneMask, const sycl::nd_item<3> &item_ct1,
                     int width = 0, unsigned int mask = 0xffffffff) {
#if DPCT_COMPAT_RT_VERSION >= 9000
    /*
    DPCT1023:184: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_xor_sync.
    */
    /*
    DPCT1096:547: The right-most dimension of the work-group used in the SYCL
    kernel that calls this function may be less than "32". The function
    "dpct::permute_sub_group_by_xor" may return an unexpected result on the CPU
    device. Modify the size of the work-group to ensure that the value of the
    right-most dimension is a multiple of "32".
    */
    if (!width) width = item_ct1.get_sub_group().get_local_range().get(0);
    return dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), value,
                                          laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE,
          template <typename> class ReduceOp>
SYCL_EXTERNAL __dpct_inline__ void
warp_reduce(acc_t *sum, const sycl::nd_item<3> &item_ct1) {
    ReduceOp<acc_t> r;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, item_ct1, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

/*
 * Extended softmax (from native aten pytorch) with following additional features
 * 1) input scaling
 * 2) Implicit time (diagonal masking)
 */
template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
/*
DPCT1110:185: The total declared local variable size in device function
scaled_upper_triang_masked_softmax_warp_forward exceeds 128 bytes and may cause
high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to
avoid high register pressure.
*/
void scaled_upper_triang_masked_softmax_warp_forward(
    output_t *dst, const input_t *src, const acc_t scale, int micro_batch_size,
    int stride, int element_count, const sycl::nd_item<3> &item_ct1) {
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
    // warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < THREADS_PER_WARP) ?
                                                          next_power_of_two : THREADS_PER_WARP;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

    size_t first_batch = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                          item_ct1.get_local_id(1)) *
                             item_ct1.get_group_range(2) * WARP_BATCH +
                         item_ct1.get_group(2);
    int local_seq = item_ct1.get_group(2) + 1;
    int warp_iteration_limit = (local_seq + ELEMENTS_PER_LDG_STG * WARP_SIZE - 1)/ WARP_SIZE;

    // micro_batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = micro_batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = item_ct1.get_local_id(2);

    size_t thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    src += thread_offset;
    dst += thread_offset;

    // load data from global memory
    acc_t elements[WARP_BATCH][WARP_ITERATIONS];
    input_t temp_data[ELEMENTS_PER_LDG_STG];
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : local_seq;

#pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it+=ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

            if (element_index < batch_element_count) {
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_data,
                                                           src + i*element_count*stride
                                                               + it*WARP_SIZE);

#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if ((element_index + element) < batch_element_count) {
                        elements[i][it+element] = (acc_t)temp_data[element] * scale;
                    } else {
                        elements[i][it + element] = -std::numeric_limits<acc_t>::infinity();
                    }
                }
            } else {
#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    elements[i][it + element] = -std::numeric_limits<acc_t>::infinity();
                }
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_BATCH];
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        max_value[i] = elements[i][0];
#pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            max_value[i] = (max_value[i] > elements[i][it]) ? max_value[i] : elements[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value, item_ct1);

    acc_t sum[WARP_BATCH] { 0.0f };
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
#pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (it < warp_iteration_limit) {
                elements[i][it] = sycl::exp((elements[i][it] - max_value[i]));
                sum[i] += elements[i][it];
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum, item_ct1);

    // store result
    output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
#pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it+=ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

            if (element_index < local_seq) {
#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (element_index + element < local_seq) {
                        out[element] = elements[i][it + element] / sum[i];
                    } else {
                        out[element] = 0.0f;
                    }
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + i * element_count * stride
                                                                + it * WARP_SIZE,
                                                                out);
            } else if (element_index < element_count) {
                copy_zero_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + i * element_count * stride
                                                                     + it * WARP_SIZE);
            } else {
                break;
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
/*
DPCT1110:186: The total declared local variable size in device function
scaled_upper_triang_masked_softmax_warp_backward exceeds 128 bytes and may cause
high register pressure. Consult with your hardware vendor to find the total
register size available and adjust the code, or use smaller sub-group size to
avoid high register pressure.
*/
void scaled_upper_triang_masked_softmax_warp_backward(
    output_t *gradInput, const input_t *grad, const input_t *output,
    acc_t scale, int micro_batch_size, int stride, int element_count,
    const sycl::nd_item<3> &item_ct1) {
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
    // warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < THREADS_PER_WARP) ?
                                                    next_power_of_two : THREADS_PER_WARP;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

    size_t first_batch = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                          item_ct1.get_local_id(1)) *
                             item_ct1.get_group_range(2) * WARP_BATCH +
                         item_ct1.get_group(2);
    int local_seq = item_ct1.get_group(2) + 1;

    // micro_batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = micro_batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = item_ct1.get_local_id(2);

    // the first element to process by the current thread
    size_t thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS] { 0.0f };
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS] { 0.0f };
    input_t temp_grad[ELEMENTS_PER_LDG_STG];
    input_t temp_output[ELEMENTS_PER_LDG_STG];
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        int batch_element_count = (i >= local_batches) ? 0 : local_seq;

#pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it+=ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < batch_element_count) {
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_grad,
                                                           grad + i * element_count * stride
                                                                + it * WARP_SIZE);
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_output,
                                                           output + i * element_count * stride
                                                                  + it * WARP_SIZE);

#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (element_index + element < batch_element_count) {
                        output_reg[i][it + element] = (acc_t)temp_output[element];
                    }
                }
#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (element_index + element < batch_element_count) {
                        grad_reg[i][it + element] = (acc_t)temp_grad[element] *
                                                            output_reg[i][it + element];
                    }
                }
            }
        }
    }

    acc_t sum[WARP_BATCH];
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        sum[i] = grad_reg[i][0];
#pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[i] += grad_reg[i][it];
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum, item_ct1);

    // store result
#pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
#pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it+=ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;
            if (element_index < element_count) {
                // compute gradients
                output_t out[ELEMENTS_PER_LDG_STG];
#pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    out[element] = (output_t)(scale * (grad_reg[i][it + element] -
                                                        output_reg[i][it + element] * sum[i]));
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(gradInput + i * element_count * stride
                                                                      + it * WARP_SIZE, out);
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_upper_triang_masked_softmax_forward(
    output_t *dst, const input_t *src, const input_t scale,
    int softmax_elements, int softmax_elements_stride, int attn_batches,
    dpct::queue_ptr stream) {
    NVTE_CHECK(softmax_elements >= 0 && softmax_elements <= 16384, "Unsupported shape.");
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;
        int seq_len = softmax_elements;
        int batch_count = attn_batches * seq_len;

        // This value must match the WARP_SIZE constexpr
        // value computed inside softmax_warp_forward.
        int warp_size = (next_power_of_two < THREADS_PER_WARP) ? next_power_of_two
                                                               : THREADS_PER_WARP;

        // This value must match the WARP_BATCH constexpr
        // value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        NVTE_CHECK(attn_batches % batches_per_block == 0, "Unsupported shape.");

        int blocks_per_seq = attn_batches / batches_per_block;
        sycl::range<3> blocks(1, blocks_per_seq, seq_len);
        sycl::range<3> threads(1, warps_per_block, warp_size);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0:  // 1
                /*
                DPCT1049:187: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 0>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 1:  // 2
                /*
                DPCT1049:188: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 1>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 2:  // 4
                /*
                DPCT1049:189: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 2>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 3:  // 8
                /*
                DPCT1049:190: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 3>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 4:  // 16
                /*
                DPCT1049:191: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 4>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 5:  // 32
                /*
                DPCT1049:192: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 5>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 6:  // 64
                /*
                DPCT1049:193: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 6>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 7:  // 128
                /*
                DPCT1049:194: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 7>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 8:  // 256
                /*
                DPCT1049:195: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 8>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 9:  // 512
                /*
                DPCT1049:196: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 9>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 10:  // 1024
                /*
                DPCT1049:197: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 10>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 11:  // 2048
                /*
                DPCT1049:198: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 11>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 12:  // 4096
                /*
                DPCT1049:199: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 12>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 13:  // 8192
                /*
                DPCT1049:200: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 13>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 14:  // 16384
                /*
                DPCT1049:201: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_forward<
                                input_t, output_t, acc_t, 14>(
                                dst, src, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            default:
                break;
        }
    }
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_upper_triang_masked_softmax_backward(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const acc_t scale, int softmax_elements, int softmax_elements_stride,
    int attn_batches, dpct::queue_ptr stream) {
    NVTE_CHECK(softmax_elements >= 0 && softmax_elements <= 16384, "Unsupported shape.");
    if (softmax_elements == 0) {
       return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;
        int seq_len = softmax_elements;
        int batch_count = attn_batches * seq_len;

        // This value must match the WARP_SIZE constexpr
        // value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < THREADS_PER_WARP) ? next_power_of_two
                                                               : THREADS_PER_WARP;

        // This value must match the WARP_BATCH constexpr
        // value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        NVTE_CHECK(attn_batches % batches_per_block == 0, "Unsupported shape.");

        int blocks_per_seq = attn_batches / batches_per_block;
        sycl::range<3> blocks(1, blocks_per_seq, seq_len);
        sycl::range<3> threads(1, warps_per_block, warp_size);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 0:  // 1
                /*
                DPCT1049:202: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 0>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 1:  // 2
                /*
                DPCT1049:203: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 1>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 2:  // 4
                /*
                DPCT1049:204: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 2>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 3:  // 8
                /*
                DPCT1049:205: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 3>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 4:  // 16
                /*
                DPCT1049:206: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 4>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 5:  // 32
                /*
                DPCT1049:207: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 5>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 6:  // 64
                /*
                DPCT1049:208: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 6>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 7:  // 128
                /*
                DPCT1049:209: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 7>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 8:  // 256
                /*
                DPCT1049:210: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 8>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 9:  // 512
                /*
                DPCT1049:211: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 9>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 10:  // 1024
                /*
                DPCT1049:212: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 10>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 11:  // 2048
                /*
                DPCT1049:213: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 11>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 12:  // 4096
                /*
                DPCT1049:214: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 12>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 13:  // 8192
                /*
                DPCT1049:215: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 13>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            case 14:  // 16384
                /*
                DPCT1049:216: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
            {
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(blocks * threads, threads),
                    [=](sycl::nd_item<3> item_ct1)
                        [[intel::reqd_sub_group_size(32)]] {
                            scaled_upper_triang_masked_softmax_warp_backward<
                                input_t, output_t, acc_t, 14>(
                                grad_input, grad, output, scale, batch_count,
                                softmax_elements_stride, softmax_elements,
                                item_ct1);
                        });
            }
                break;
            default:
                break;
        }
    }
}

void scaled_upper_triang_masked_softmax_forward(const Tensor input,
                                                Tensor *softmax_results,
                                                float scale_factor,
                                                dpct::queue_ptr stream) {

    const int attn_batches = input.data.shape[0];
    const int seq_len = input.data.shape[1];

    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(input.data.dtype, softmax_type,
        dispatch_scaled_upper_triang_masked_softmax_forward<softmax_type, softmax_type, float>(
            reinterpret_cast<softmax_type*>(softmax_results->data.dptr),
            reinterpret_cast<const softmax_type*>(input.data.dptr),
            scale_factor,
            seq_len,
            seq_len,
            attn_batches,
            stream););
}

void scaled_upper_triang_masked_softmax_backward(Tensor output_grads,
                                                 const Tensor incoming_grads,
                                                 const Tensor softmax_results,
                                                 float scale_factor,
                                                 dpct::queue_ptr stream) {

    const int attn_batches = output_grads.data.shape[0];
    const int seq_len = output_grads.data.shape[1];

    // Softmax Grad
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(output_grads.data.dtype, softmax_type,
        dispatch_scaled_upper_triang_masked_softmax_backward<softmax_type, softmax_type, float>(
            reinterpret_cast<softmax_type*>(output_grads.data.dptr),
            reinterpret_cast<softmax_type const*>(incoming_grads.data.dptr),
            reinterpret_cast<softmax_type const*>(softmax_results.data.dptr),
            scale_factor,
            seq_len,
            seq_len,
            attn_batches,
            stream););
}

}  // end namespace transformer_engine

void nvte_scaled_upper_triang_masked_softmax_forward(const NVTETensor input,
                                                     NVTETensor softmax_results,
                                                     float scale_factor,
                                                     dpct::queue_ptr stream) {
    using namespace transformer_engine;
    scaled_upper_triang_masked_softmax_forward(
        *reinterpret_cast<const Tensor*>(input),
        reinterpret_cast<Tensor*>(softmax_results),
        scale_factor,
        stream);
}

void nvte_scaled_upper_triang_masked_softmax_backward(
    const NVTETensor incoming_grads, const NVTETensor softmax_results,
    NVTETensor output_grads, float scale_factor, dpct::queue_ptr stream) {
    using namespace transformer_engine;
    scaled_upper_triang_masked_softmax_backward(
        *reinterpret_cast<Tensor*>(output_grads),
        *reinterpret_cast<const Tensor*>(incoming_grads),
        *reinterpret_cast<const Tensor*>(softmax_results),
        scale_factor,
        stream);
}
