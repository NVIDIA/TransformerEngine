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
#include <array>
#include <functional>

#include <transformer_engine/softmax.h>
#include "../common.h"
#include "../utils.dp.hpp"
#include "../util/logging.h"
#include <cmath>

namespace transformer_engine {

template <typename Datatype, int ELEMENTS_PER_LDG>
SYCL_EXTERNAL __inline__ void copy_vector(Datatype *dst, const Datatype *src);

SYCL_EXTERNAL template <>
__inline__ void copy_vector<bf16, 1>(bf16 *dst, const bf16 *src) {
    *dst = *src;
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<bf16, 4>(bf16 *dst, const bf16 *src) {
    *((uint64_t*) dst) = *((uint64_t*) src);    // NOLINT(*)
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<fp16, 1>(fp16 *dst, const fp16 *src) {
    *dst = *src;
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<fp16, 4>(fp16 *dst, const fp16 *src) {
    *((uint64_t*) dst) = *((uint64_t*) src);    // NOLINT(*)
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<uint8_t, 1>(uint8_t *dst, const uint8_t *src) {
    *dst = *src;
}

SYCL_EXTERNAL template <>
__inline__ void copy_vector<uint8_t, 4>(uint8_t *dst, const uint8_t *src) {
    *((uint32_t*) dst) = *((uint32_t*) src);      // NOLINT(*)
}

template <typename Datatype, int ELEMENTS_PER_LDG>
SYCL_EXTERNAL __inline__ void copy_zero_vector(Datatype *dst);

SYCL_EXTERNAL template <> __inline__ void copy_zero_vector<bf16, 1>(bf16 *dst) {
    *dst = 0.0f;
}

SYCL_EXTERNAL template <> __inline__ void copy_zero_vector<bf16, 4>(bf16 *dst) {
    *((sycl::float2 *)dst) = sycl::float2(0.0f, 0.0f); // NOLINT(*)
}

SYCL_EXTERNAL template <> __inline__ void copy_zero_vector<fp16, 1>(fp16 *dst) {
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
    DPCT1023:30: The SYCL sub-group does not support mask options for
    dpct::permute_sub_group_by_xor. You can specify
    "--use-experimental-features=masked-sub-group-operation" to use the
    experimental helper function to migrate __shfl_xor_sync.
    */
    /*
    DPCT1096:545: The right-most dimension of the work-group used in the SYCL
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

template <typename acc_t, int WARP_ROWS, int WARP_SIZE,
          template <typename> class ReduceOp>
SYCL_EXTERNAL __dpct_inline__ void
warp_reduce(acc_t *sum, const sycl::nd_item<3> &item_ct1) {
    ReduceOp<acc_t> r;
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        #pragma unroll
        for (int i = 0;  i < WARP_ROWS;  ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, item_ct1, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

/*
 * Extended softmax (from native aten pytorch) with the following additional features
 * 1) input scaling
 * 2) implicit causal masking
 * 
 * works for all cases:
 *  k > q
 *  k < q
 *  k = q
 * 
 * where:
 * microbatches = batches * attn_heads * query_seq_len
 * rows = query_seq_len
 * cols = key_seq_len
 */
template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
/*
DPCT1110:31: The total declared local variable size in device function
scaled_aligned_causal_masked_softmax_warp_forward exceeds 128 bytes and may
cause high register pressure. Consult with your hardware vendor to find the
total register size available and adjust the code, or use smaller sub-group size
to avoid high register pressure.
*/
void scaled_aligned_causal_masked_softmax_warp_forward(
    output_t *dst, const input_t *src, const acc_t scale,
    const int microbatches, const int rows, const int cols,
    const sycl::nd_item<3> &item_ct1) {
    // 1) WARP_WIDTH must match the value of warp_size
    // 2) WARP_ROWS must match the value of rows_per_warp
    // of the dispatch_scaled_aligned_causal_masked_softmax_forward method.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_WIDTH = (next_power_of_two < THREADS_PER_WARP) ? next_power_of_two
                                                                      : THREADS_PER_WARP;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_WIDTH;
    constexpr int WARP_ROWS = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

    const int global_row_idx =
        (item_ct1.get_group(2) * item_ct1.get_local_range(1) +
         item_ct1.get_local_id(1)) *
        WARP_ROWS;
    const int col = item_ct1.get_local_id(2) * ELEMENTS_PER_LDG_STG;

    const size_t thread_offset = global_row_idx * cols + col;

    src += thread_offset;
    dst += thread_offset;

    // load data from global memory into registers WITH scaling
    acc_t elements[WARP_ROWS][WARP_ITERATIONS];
    input_t temp_data[ELEMENTS_PER_LDG_STG];

    #pragma unroll
    for (int w = 0; w < WARP_ROWS; ++w) {
        const int microbatch = global_row_idx + w;
        const int i = microbatch % rows;                    // local row index of attention matrix
        const int masked_elements = i + cols - rows + 1;

        if (microbatch >= microbatches) {
            break;
        }

        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it += ELEMENTS_PER_LDG_STG) {
            const int j = col + it * WARP_WIDTH;
            const int itr_idx = w * cols + it * WARP_WIDTH;

            if (j < masked_elements) {
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_data, src + itr_idx);
                #pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (j + element < masked_elements) {
                        elements[w][it + element] = (acc_t)temp_data[element] * scale;
                    } else {
                        elements[w][it + element] = (acc_t)( -10'000 );
                    }
                }
            } else {
                #pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    elements[w][it + element] = (acc_t)( -10'000 );
                }
            }
        }
    }

    // compute max_value
    acc_t max_value[WARP_ROWS];
    #pragma unroll
    for (int w = 0;  w < WARP_ROWS;  ++w) {
        max_value[w] = elements[w][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            max_value[w] =
                (max_value[w] > elements[w][it]) ? max_value[w] : elements[w][it];
        }
    }
    warp_reduce<acc_t, WARP_ROWS, WARP_WIDTH, Max>(max_value, item_ct1);

    acc_t sum[WARP_ROWS] { 0.0f };
    #pragma unroll
    for (int w = 0;  w < WARP_ROWS;  ++w) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            elements[w][it] =
                sycl::native::exp((elements[w][it] - max_value[w]));
            sum[w] += elements[w][it];
        }
    }
    warp_reduce<acc_t, WARP_ROWS, WARP_WIDTH, Add>(sum, item_ct1);

    output_t out[ELEMENTS_PER_LDG_STG] { 0.0f };
    // store result
    #pragma unroll
    for (int w = 0;  w < WARP_ROWS;  ++w) {
        const int microbatch = global_row_idx + w;
        const int i = microbatch % rows;
        const int masked_elements = i + cols - rows + 1;

        // out of Attention matrix bounds (rows)
        if (microbatch >= microbatches) {
            break;
        }

        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it += ELEMENTS_PER_LDG_STG) {
            const int j = col + it * WARP_WIDTH;              // index of the first column
            const int itr_idx = w * cols + it * WARP_WIDTH;

            if (j < masked_elements) {
                #pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (j + element < masked_elements) {
                        out[element] = elements[w][it + element] / sum[w];
                    } else {
                        out[element] = (output_t)( 0.0f );
                    }
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + itr_idx, out);
            } else if (j < cols) {
                copy_zero_vector<output_t, ELEMENTS_PER_LDG_STG>(dst + itr_idx);
            } else {
                break;
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
/*
DPCT1110:32: The total declared local variable size in device function
scaled_aligned_causal_masked_softmax_warp_backward exceeds 128 bytes and may
cause high register pressure. Consult with your hardware vendor to find the
total register size available and adjust the code, or use smaller sub-group size
to avoid high register pressure.
*/
void scaled_aligned_causal_masked_softmax_warp_backward(
    output_t *gradInput, const input_t *grad, const input_t *softmax_output,
    const acc_t scale, const int microbatches, const int rows, const int cols,
    const sycl::nd_item<3> &item_ct1) {
    // 1) WARP_WIDTH must match the value of warp_size
    // 2) WARP_ROWS must match the value of rows_per_warp
    // of the dispatch_scaled_aligned_causal_masked_softmax_forward method.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_WIDTH = (next_power_of_two < THREADS_PER_WARP) ? next_power_of_two
                                                                      : THREADS_PER_WARP;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_WIDTH;
    constexpr int WARP_ROWS = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

    const int global_row_idx =
        (item_ct1.get_group(2) * item_ct1.get_local_range(1) +
         item_ct1.get_local_id(1)) *
        WARP_ROWS;
    const int col = item_ct1.get_local_id(2) * ELEMENTS_PER_LDG_STG;

    const size_t thread_offset = global_row_idx * cols + col;

    grad += thread_offset;
    softmax_output += thread_offset;
    gradInput += thread_offset;

    // load data from global memory into registers
    acc_t grad_reg[WARP_ROWS][WARP_ITERATIONS] { 0.0f };
    acc_t softmax_output_reg[WARP_ROWS][WARP_ITERATIONS] { 0.0f };
    input_t temp_grad[ELEMENTS_PER_LDG_STG];
    input_t temp_output[ELEMENTS_PER_LDG_STG];

    #pragma unroll
    for (int w = 0; w < WARP_ROWS; ++w) {
        const int microbatch = global_row_idx + w;
        const int i = microbatch % rows;                    // local row index of attention matrix
        const int masked_elements = i + cols - rows + 1;

        if (microbatch >= microbatches) {
            break;
        }

        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it += ELEMENTS_PER_LDG_STG) {
            const int j = col + it * WARP_WIDTH;                // index of the first column
            const int itr_idx = w * cols + it * WARP_WIDTH;

            if (j < masked_elements) {
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_grad, grad + itr_idx);
                copy_vector<input_t, ELEMENTS_PER_LDG_STG>(temp_output, softmax_output + itr_idx);
                #pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (j + element < masked_elements) {
                        softmax_output_reg[w][it + element] = (acc_t)temp_output[element];
                        grad_reg[w][it + element] =
                            (acc_t)temp_grad[element] * softmax_output_reg[w][it + element];
                    }
                }
            }
        }
    }

    acc_t sum[WARP_ROWS];
    #pragma unroll
    for (int w = 0; w < WARP_ROWS; ++w) {
        sum[w] = grad_reg[w][0];
        #pragma unroll
        for (int it = 1;  it < WARP_ITERATIONS;  ++it) {
            sum[w] += grad_reg[w][it];
        }
    }

    warp_reduce<acc_t, WARP_ROWS, WARP_WIDTH, Add>(sum, item_ct1);

    // store result
    #pragma unroll
    for (int w = 0;  w < WARP_ROWS;  ++w) {
        const int microbatch = global_row_idx + w;
        if (microbatch >= microbatches) {
            break;
        }

        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it += ELEMENTS_PER_LDG_STG) {
            const int j = col + it * WARP_WIDTH;              // index of the first column
            const int itr_idx = w * cols + it * WARP_WIDTH;

            if (j < cols) {
                output_t out[ELEMENTS_PER_LDG_STG];
                #pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    out[element] = (output_t)(scale * (grad_reg[w][it + element] -
                                                    softmax_output_reg[w][it + element] * sum[w]));
                }
                copy_vector<output_t, ELEMENTS_PER_LDG_STG>(gradInput + itr_idx, out);
            }
        }
    }
}

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
void call_kernel_scaled_aligned_causal_masked_softmax_forward(
    sycl::range<3> grid_size, sycl::range<3> block_size, const int shmem_size,
    dpct::queue_ptr stream, output_t *dst, const input_t *src,
    const acc_t scale, const int microbatches, const int query_seq_len,
    const int key_seq_len) {
    /*
    DPCT1049:33: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->parallel_for(
        sycl::nd_range<3>(grid_size * block_size, block_size),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            scaled_aligned_causal_masked_softmax_warp_forward<
                input_t, output_t, acc_t, log2_elements>(
                dst, src, scale, microbatches, query_seq_len, key_seq_len,
                item_ct1);
        });
}

template <typename input_t, typename output_t, typename acc_t,
          int log2_elements>
void call_kernel_scaled_aligned_causal_masked_softmax_backward(
    sycl::range<3> grid_size, sycl::range<3> block_size, const int shmem_size,
    dpct::queue_ptr stream, output_t *gradInput, const input_t *grad,
    const input_t *output, const acc_t scale, const int microbatches,
    const int query_seq_len, const int key_seq_len) {
    /*
    DPCT1049:34: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    dpct::has_capability_or_fail(stream->get_device(), {sycl::aspect::fp16});

    stream->parallel_for(
        sycl::nd_range<3>(grid_size * block_size, block_size),
        [=](sycl::nd_item<3> item_ct1) [[intel::reqd_sub_group_size(32)]] {
            scaled_aligned_causal_masked_softmax_warp_backward<
                input_t, output_t, acc_t, log2_elements>(
                gradInput, grad, output, scale, microbatches, query_seq_len,
                key_seq_len, item_ct1);
        });
}

template<typename input_t, typename output_t, typename acc_t>
struct FunctionWrapper {
    using ForwardType = std::function<void(
        sycl::range<3> grid_size, sycl::range<3> block_size,
        const int shmem_size, dpct::queue_ptr stream, output_t *dst,
        const input_t *src, const acc_t scale, const int microbatches,
        const int query_seq_len, const int key_seq_len)>;
    using BackwardType = std::function<void(
        sycl::range<3> grid_size, sycl::range<3> block_size,
        const int shmem_size, dpct::queue_ptr stream, output_t *gradInput,
        const input_t *grad, const input_t *output, const acc_t scale,
        const int microbatches, const int query_seq_len,
        const int key_seq_len)>;
};


constexpr int MIN_SUPPORTED_POWER = 4;
constexpr int MAX_SUPPORTED_POWER = 14;
constexpr int MIN_POWER = MIN_SUPPORTED_POWER - 1;
constexpr int MAX_POWER = MAX_SUPPORTED_POWER + 1;

// Recursively instantiate the function for the limit of "log2_elements",
// i.e. "MAX_POWER" defined above.
template <typename input_t, typename output_t, typename acc_t, int log2_elements>
struct CompileTimeLoopForward {
    using ForwardFuncType = typename FunctionWrapper<input_t, output_t, acc_t>::ForwardType;
    static void populate(std::array<ForwardFuncType, MAX_POWER>* arr) {
        CompileTimeLoopForward<input_t, output_t, acc_t, log2_elements - 1>::populate(arr);
        (*arr)[log2_elements] = &call_kernel_scaled_aligned_causal_masked_softmax_forward<
                              output_t, input_t, acc_t, log2_elements>;
    }
};

template <typename input_t, typename output_t, typename acc_t>
struct CompileTimeLoopForward<input_t, output_t, acc_t, MIN_POWER> {
    using ForwardFuncType = typename FunctionWrapper<input_t, output_t, acc_t>::ForwardType;
    static void populate(std::array<ForwardFuncType, MAX_POWER>* arr) {
        (*arr)[MIN_POWER] = nullptr;
    }
};

template <typename input_t, typename output_t, typename acc_t, int log2_elements>
struct CompileTimeLoopBackward {
    using BackwardFuncType = typename FunctionWrapper<input_t, output_t, acc_t>::BackwardType;
    static void populate(std::array<BackwardFuncType, MAX_POWER>* arr) {
        CompileTimeLoopBackward<input_t, output_t, acc_t, log2_elements - 1>::populate(arr);
        (*arr)[log2_elements] = &call_kernel_scaled_aligned_causal_masked_softmax_backward<
                              output_t, input_t, acc_t, log2_elements>;
    }
};

template <typename input_t, typename output_t, typename acc_t>
struct CompileTimeLoopBackward<input_t, output_t, acc_t, MIN_POWER> {
    using BackwardFuncType = typename FunctionWrapper<input_t, output_t, acc_t>::BackwardType;
    static void populate(std::array<BackwardFuncType, MAX_POWER>* arr) {
        (*arr)[MIN_POWER] = nullptr;
    }
};

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_aligned_causal_masked_softmax_forward(
    output_t *dst, const input_t *src, const input_t scale, int query_seq_len,
    int key_seq_len, int batches, int attn_heads, dpct::queue_ptr stream) {
    NVTE_CHECK(key_seq_len >= 0 && key_seq_len <= 16384, "Unsupported shape.");

    if (key_seq_len == 0) {
        return;
    }
    int log2_elements = log2_ceil(key_seq_len);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_WIDTH constexpr
    // value computed inside scaled_aligned_causal_masked_softmax_warp_forward.
    int warp_width = (next_power_of_two < THREADS_PER_WARP) ? next_power_of_two
                                                            : THREADS_PER_WARP;

    // This value must match the WARP_ROWS constexpr
    // value computed inside scaled_aligned_causal_masked_softmax_warp_forward.
    int microbatches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = threads_per_block / warp_width;
    int microbatches_per_block = warps_per_block * microbatches_per_warp;
    int microbatches = batches * attn_heads * query_seq_len;
    int blocks = DIVUP(microbatches, microbatches_per_block);

    sycl::range<3> block_size(1, warps_per_block, warp_width);
    sycl::range<3> grid_size(1, 1, blocks);

    // create an array of pointers to functions
    using ForwardFuncType = typename FunctionWrapper<input_t, output_t, acc_t>::ForwardType;
    static std::array<ForwardFuncType, MAX_POWER> forwardFunctionArray;
    static bool is_initialized = false;
    if (!is_initialized) {
        CompileTimeLoopForward<input_t, output_t, acc_t, MAX_SUPPORTED_POWER>::populate(
            &forwardFunctionArray);
        is_initialized = true;
    }
    // Call the corresponding kernel
    forwardFunctionArray[log2_elements](grid_size, block_size, 0, stream, dst, src, scale,
                                        microbatches, query_seq_len, key_seq_len);
}

template <typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_aligned_causal_masked_softmax_backward(
    output_t *grad_input, const input_t *grad, const input_t *output,
    const acc_t scale, int query_seq_len, int key_seq_len, int batches,
    int attn_heads, dpct::queue_ptr stream) {
    NVTE_CHECK(key_seq_len >= 0 && key_seq_len <= 16384, "Unsupported shape.");

    if (key_seq_len == 0) {
        return;
    }
    int log2_elements = log2_ceil(key_seq_len);
    const int next_power_of_two = 1 << log2_elements;

    // This value must match the WARP_WIDTH constexpr
    // value computed inside scaled_aligned_causal_masked_softmax_warp_forward.
    int warp_width = (next_power_of_two < THREADS_PER_WARP) ? next_power_of_two : THREADS_PER_WARP;

    // This value must match the WARP_ROWS constexpr
    // value computed inside scaled_aligned_causal_masked_softmax_warp_forward.
    int microbatches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

    // use 128 threads per block to maximimize gpu utilization
    constexpr int threads_per_block = 128;

    int warps_per_block = threads_per_block / warp_width;
    int microbatches_per_block = warps_per_block * microbatches_per_warp;
    int microbatches = batches * attn_heads * query_seq_len;
    int blocks = DIVUP(microbatches, microbatches_per_block);

    sycl::range<3> block_size(1, warps_per_block, warp_width);
    sycl::range<3> grid_size(1, 1, blocks);

    // create an array of pointers to functions
    using BackwardFuncType = typename FunctionWrapper<input_t, output_t, acc_t>::BackwardType;
    static std::array<BackwardFuncType, MAX_POWER> backwardFunctionArray;
    static bool is_initialized = false;
    if (!is_initialized) {
        CompileTimeLoopBackward<input_t, output_t, acc_t, MAX_SUPPORTED_POWER>::populate(
            &backwardFunctionArray);
        is_initialized = true;
    }
    // Call the corresponding kernel
    backwardFunctionArray[log2_elements](grid_size, block_size, 0, stream, grad_input, grad,
                                         output, scale, microbatches, query_seq_len, key_seq_len);
}

void scaled_aligned_causal_masked_softmax_forward(const Tensor &input,
                                                  Tensor *softmax_results,
                                                  float scale_factor,
                                                  dpct::queue_ptr stream) {

    const int batches = input.data.shape[0];
    const int attn_heads = input.data.shape[1];
    const int query_seq_len = input.data.shape[2];
    const int key_seq_len = input.data.shape[3];

    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(input.data.dtype, softmax_type,
        dispatch_scaled_aligned_causal_masked_softmax_forward<softmax_type, softmax_type, float>(
            reinterpret_cast<softmax_type*>(softmax_results->data.dptr),
            reinterpret_cast<const softmax_type*>(input.data.dptr),
            scale_factor,
            query_seq_len,
            key_seq_len,
            batches,
            attn_heads,
            stream););
}

void scaled_aligned_causal_masked_softmax_backward(Tensor output_grads,
                                                   const Tensor incoming_grads,
                                                   const Tensor softmax_results,
                                                   float scale_factor,
                                                   dpct::queue_ptr stream) {

    // output grads is a 4d tensor with dimensions [batches, attn_heads, seq_len, seq_len]
    const int batches = output_grads.data.shape[0];
    const int attn_heads = output_grads.data.shape[1];
    const int query_seq_len = output_grads.data.shape[2];
    const int key_seq_len = output_grads.data.shape[3];

    // Softmax Grad
    TRANSFORMER_ENGINE_TYPE_SWITCH_16BIT(output_grads.data.dtype, softmax_type,
        dispatch_scaled_aligned_causal_masked_softmax_backward<softmax_type, softmax_type, float>(
            reinterpret_cast<softmax_type*>(output_grads.data.dptr),
            reinterpret_cast<softmax_type const*>(incoming_grads.data.dptr),
            reinterpret_cast<softmax_type const*>(softmax_results.data.dptr),
            scale_factor,
            query_seq_len,
            key_seq_len,
            batches,
            attn_heads,
            stream););
}
}  // end namespace transformer_engine

void nvte_scaled_aligned_causal_masked_softmax_forward(
    const NVTETensor input, NVTETensor softmax_results, float scale_factor,
    dpct::queue_ptr stream) {
    NVTE_API_CALL(nvte_scaled_aligned_causal_masked_softmax_forward);
    using namespace transformer_engine;
    scaled_aligned_causal_masked_softmax_forward(
        *reinterpret_cast<const Tensor*>(input),
        reinterpret_cast<Tensor*>(softmax_results),
        scale_factor,
        stream);
}

void nvte_scaled_aligned_causal_masked_softmax_backward(
    const NVTETensor incoming_grads, const NVTETensor softmax_results,
    NVTETensor output_grads, float scale_factor, dpct::queue_ptr stream) {
    NVTE_API_CALL(nvte_scaled_aligned_causal_masked_softmax_backward);
    using namespace transformer_engine;
    scaled_aligned_causal_masked_softmax_backward(
        *reinterpret_cast<Tensor*>(output_grads),
        *reinterpret_cast<const Tensor*>(incoming_grads),
        *reinterpret_cast<const Tensor*>(softmax_results),
        scale_factor,
        stream);
}
