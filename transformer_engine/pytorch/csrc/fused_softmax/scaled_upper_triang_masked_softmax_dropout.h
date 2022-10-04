/*************************************************************************
 * Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_SCALED_UPPER_TRIANG_SOFTMAX_H_
#define TRANSFORMER_ENGINE_SCALED_UPPER_TRIANG_SOFTMAX_H_

#include <assert.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <limits>
#include <stdint.h>
#include <c10/macros/Macros.h>

#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <ATen/cuda/detail/UnpackRaw.cuh>
#include "philox.h"

namespace {

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_vector(Datatype *dst, const Datatype *src);

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 1>(c10::BFloat16 *dst,
                                                         const c10::BFloat16 *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<c10::BFloat16, 4>(c10::BFloat16 *dst,
                                                         const c10::BFloat16 *src) {
  *((float2*) dst) = *((float2*) src);  // NOLINT(*)
}

template <>
__device__ __inline__ void copy_vector<c10::Half, 1>(c10::Half *dst,
                                                     const c10::Half *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<c10::Half, 4>(c10::Half *dst,
                                                     const c10::Half *src) {
  *((float2*) dst) = *((float2*) src);  // NOLINT(*)
}

template <>
__device__ __inline__ void copy_vector<uint8_t, 1>(uint8_t *dst,
                                                   const uint8_t *src) {
  *dst = *src;
}

template <>
__device__ __inline__ void copy_vector<uint8_t, 4>(uint8_t *dst,
                                                   const uint8_t *src) {
  *((half2*) dst) = *((half2*) src);  // NOLINT(*)
}

template <typename Datatype, int ELEMENTS_PER_LDG>
__device__ __inline__ void copy_zero_vector(Datatype *dst);

template <>
__device__ __inline__ void copy_zero_vector<c10::BFloat16, 1>(c10::BFloat16 *dst) {
  *dst = 0.0;
}

template <>
__device__ __inline__ void copy_zero_vector<c10::BFloat16, 4>(c10::BFloat16 *dst) {
  *((float2*) dst) = make_float2(0.0f, 0.0f);  // NOLINT(*)
}

template <>
__device__ __inline__ void copy_zero_vector<c10::Half, 1>(c10::Half *dst) { *dst = 0.0; }

template <>
__device__ __inline__ void copy_zero_vector<c10::Half, 4>(c10::Half *dst) {
  *((float2*) dst) = make_float2(0.0f, 0.0f);  // NOLINT(*)
}


int log2_ceil(int value) {
    int log2_value = 0;
    while ((1 << log2_value) < value) ++log2_value;
    return log2_value;
}

template<typename T>
struct Add {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a + b;
  }
};

template<typename T>
struct Max {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return a < b ? b : a;
  }
};

template <typename T>
__device__ __forceinline__ T WARP_SHFL_XOR_NATIVE(T value, int laneMask, int width = warpSize,
                                                  unsigned int mask = 0xffffffff) {
#if CUDA_VERSION >= 9000
    return __shfl_xor_sync(mask, value, laneMask, width);
#else
    return __shfl_xor(value, laneMask, width);
#endif
}

template <typename acc_t, int WARP_BATCH, int WARP_SIZE, template<typename> class ReduceOp>
__device__ __forceinline__ void warp_reduce(acc_t* sum) {
    ReduceOp<acc_t> r;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
        for (int i = 0;  i < WARP_BATCH;  ++i) {
            acc_t b = WARP_SHFL_XOR_NATIVE(sum[i], offset, WARP_SIZE);
            sum[i] = r(sum[i], b);
        }
    }
}

/*
 * Extended softmax (from native aten pytorch) with following additional features
 * 1) input scaling
 * 2) Implicit time (diagonal masking)
 */
template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void scaled_upper_triang_masked_softmax_warp_forward(
    output_t *dst,
    const input_t *src,
    const acc_t scale,
    int micro_batch_size,
    int stride,
    int element_count,
    const float p_keep,
    at::PhiloxCudaState philox_args
    ) {
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
    // warp_size of method warp_softmax_forward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ?
                                                          next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;
    static_assert(ELEMENTS_PER_LDG_STG == 4);

    //const output_t rp_keep = 1.f / p_keep;
    const float rp_keep = 1.f / p_keep;

    auto seeds = at::cuda::philox::unpack(philox_args);

    int tidx_global =
        ((blockIdx.y * gridDim.x + blockIdx.x) * blockDim.y + threadIdx.y) *
            blockDim.x +
        threadIdx.x;

    Philox ph(std::get<0>(seeds), tidx_global, std::get<1>(seeds));
    
    int first_batch = (blockDim.y * blockIdx.y + threadIdx.y) * gridDim.x * WARP_BATCH + blockIdx.x;
    int local_seq = blockIdx.x + 1;
    int warp_iteration_limit = (local_seq + ELEMENTS_PER_LDG_STG * WARP_SIZE - 1)/ WARP_SIZE;

    // micro_batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = micro_batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    src += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    dst += first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;

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
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Max>(max_value);

    acc_t sum[WARP_BATCH] { 0.0f };
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  ++it) {
            if (it < warp_iteration_limit) {
                elements[i][it] = std::exp((elements[i][it] - max_value[i]));
                sum[i] += elements[i][it];
            }
        }
    }
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

    // store result
    output_t out[ELEMENTS_PER_LDG_STG];
    #pragma unroll
    for (int i = 0;  i < WARP_BATCH;  ++i) {
        if (i >= local_batches)
            break;
        float inv_sum = 1.f / sum[i];
        #pragma unroll
        for (int it = 0;  it < WARP_ITERATIONS;  it+=ELEMENTS_PER_LDG_STG) {
            int element_index = ELEMENTS_PER_LDG_STG * local_idx + it * WARP_SIZE;

            if (element_index < local_seq) {
                float4 tmp = uniform4(ph());
                float rand[4] = {tmp.x, tmp.y, tmp.z, tmp.w};
                static_assert(ELEMENTS_PER_LDG_STG == 4);
                #pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (element_index + element < local_seq) {
                        //output_t dmask =  rand[element] < p_keep ? 1.f : -1.f;
                        //output_t val =  elements[i][it + element] / sum[i];
                        float val = elements[i][it + element] * inv_sum;
                        // cast down.
                        output_t out_val = rand[element] < p_keep ? val : -val;
                        // cast back up for scaling.
                        val = float(out_val) * rp_keep;
                        // cast back down again for storing.
                        out[element] = val;
                    } else {
                        out[element] = 0;
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

template <typename input_t, typename output_t, typename acc_t, int log2_elements>
__global__ void scaled_upper_triang_masked_softmax_warp_backward(
                                                                 output_t *gradInput,
                                                                 input_t *grad,
                                                                 const input_t *output,
                                                                 acc_t scale,
                                                                 int micro_batch_size,
                                                                 int stride,
                                                                 int element_count,
                                                                 const acc_t p_keep,
                                                                 const acc_t rp_keep) {
    // WARP_SIZE and WARP_BATCH must match the return values batches_per_warp and
    // warp_size of method warp_softmax_backward_kernel.
    constexpr int next_power_of_two = 1 << log2_elements;
    constexpr int WARP_SIZE = (next_power_of_two < C10_WARP_SIZE) ?
        next_power_of_two : C10_WARP_SIZE;
    constexpr int WARP_ITERATIONS = next_power_of_two / WARP_SIZE;
    constexpr int WARP_BATCH = (next_power_of_two <= 128) ? 2 : 1;
    constexpr int ELEMENTS_PER_LDG_STG = (WARP_ITERATIONS < 4) ? 1 : 4;

    int first_batch = (blockDim.y * blockIdx.y + threadIdx.y) * gridDim.x * WARP_BATCH + blockIdx.x;
    int local_seq = blockIdx.x + 1;

    // micro_batch_size might not be a multiple of WARP_BATCH. Check how
    // many batches have to computed within this WARP.
    int local_batches = micro_batch_size - first_batch;
    if (local_batches > WARP_BATCH)
        local_batches = WARP_BATCH;

    // there might be multiple batches per warp. compute the index within the batch
    int local_idx = threadIdx.x;

    // the first element to process by the current thread
    int thread_offset = first_batch * stride + ELEMENTS_PER_LDG_STG * local_idx;
    grad += thread_offset;
    output += thread_offset;
    gradInput += thread_offset;

    // load data from global memory
    acc_t grad_reg[WARP_BATCH][WARP_ITERATIONS] { 0.0f };
    acc_t output_reg[WARP_BATCH][WARP_ITERATIONS] { 0.0f };
    input_t temp_grad[ELEMENTS_PER_LDG_STG] {input_t(0.f)};
    input_t temp_output[ELEMENTS_PER_LDG_STG] {input_t(0.f)};
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
                // grad_reg: should hold dS * S
                // output_reg: should hold S
                #pragma unroll
                for (int element = 0; element < ELEMENTS_PER_LDG_STG; ++element) {
                    if (element_index + element < batch_element_count) {
                        float S_dmask = temp_output[element];
                        float dD = temp_grad[element];

                        float dmask = float(S_dmask > 0.f) * rp_keep;
                        float dS = dD * dmask;

                        input_t S_ = fabsf(S_dmask) * p_keep;
                        float S = S_;

                        output_reg[i][it + element] = S;
                        grad_reg[i][it + element] = dS * S;
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
    warp_reduce<acc_t, WARP_BATCH, WARP_SIZE, Add>(sum);

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

}  // end of anonymous namespace

template<typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_upper_triang_masked_softmax_forward(
                                                         output_t *dst,
                                                         const input_t *src,
                                                         const input_t scale,
                                                         int softmax_elements,
                                                         int softmax_elements_stride,
                                                         int attn_batches,
                                                         const float p_keep,
                                                         c10::optional<at::Generator> gen_
                                                        ) {
    TORCH_INTERNAL_ASSERT(softmax_elements >= 256 && softmax_elements <= 2048);
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;
        int seq_len = softmax_elements;
        int batch_count = attn_batches * seq_len;


        // This value must match the WARP_SIZE constexpr
        // value computed inside softmax_warp_forward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr
        // value computed inside softmax_warp_forward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        TORCH_INTERNAL_ASSERT(attn_batches % batches_per_block == 0);

        int blocks_per_seq = attn_batches / batches_per_block;
        dim3 blocks(seq_len, blocks_per_seq, 1);
        dim3 threads(warp_size, warps_per_block, 1);

        // Every warp processes one sequences. Elts_per_thread should be ceildiv(seqlen,32).
        TORCH_INTERNAL_ASSERT(warp_size == 32);
        TORCH_INTERNAL_ASSERT(batches_per_warp == 1);
        TORCH_INTERNAL_ASSERT(warps_per_block == 4);
        TORCH_INTERNAL_ASSERT(batches_per_block == 4);
        at::PhiloxCudaState philox_args;
        {
            auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(gen_, at::cuda::detail::getDefaultCUDAGenerator());
            // number of times random will be generated per thread, to offset philox
            // counter in thc random state
            const int64_t counter_offset = (seq_len + warp_size - 1) / warp_size;
            // See Note [Acquire lock when using random generators]
            std::lock_guard<std::mutex> lock(gen->mutex_);
            philox_args = gen->philox_cuda_state(counter_offset);
        }


        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 8:  // 256
                scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t, acc_t, 8>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,
                                                                               src,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               philox_args);
                break;
            case 9:  // 512
                scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t, acc_t, 9>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,
                                                                               src,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               philox_args);
                break;
            case 10:  // 1024
                scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t, acc_t, 10>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,
                                                                               src,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               philox_args);
                break;
            case 11:  // 2048
                scaled_upper_triang_masked_softmax_warp_forward<input_t, output_t, acc_t, 11>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(dst,
                                                                               src,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               philox_args);
                break;
            default:
                break;
        }
    }
}

template<typename input_t, typename output_t, typename acc_t>
void dispatch_scaled_upper_triang_masked_softmax_backward(
                                                          output_t *grad_input,
                                                          input_t *grad,
                                                          const input_t *output,
                                                          const acc_t scale,
                                                          int softmax_elements,
                                                          int softmax_elements_stride,
                                                          int attn_batches,
                                                          const acc_t p_keep,
                                                          const acc_t rp_keep) {
    TORCH_INTERNAL_ASSERT(softmax_elements >= 256 && softmax_elements <= 2048);
    if (softmax_elements == 0) {
        return;
    } else {
        int log2_elements = log2_ceil(softmax_elements);
        const int next_power_of_two = 1 << log2_elements;
        int seq_len = softmax_elements;
        int batch_count = attn_batches * seq_len;

        // This value must match the WARP_SIZE constexpr
        // value computed inside softmax_warp_backward.
        int warp_size = (next_power_of_two < C10_WARP_SIZE) ? next_power_of_two : C10_WARP_SIZE;

        // This value must match the WARP_BATCH constexpr
        // value computed inside softmax_warp_backward.
        int batches_per_warp = (next_power_of_two <= 128) ? 2 : 1;

        // use 128 threads per block to maximimize gpu utilization
        constexpr int threads_per_block = 128;

        int warps_per_block = (threads_per_block / warp_size);
        int batches_per_block = warps_per_block * batches_per_warp;
        TORCH_INTERNAL_ASSERT(attn_batches % batches_per_block == 0);

        int blocks_per_seq = attn_batches / batches_per_block;
        dim3 blocks(seq_len, blocks_per_seq, 1);
        dim3 threads(warp_size, warps_per_block, 1);
        // Launch code would be more elegant if C++ supported FOR CONSTEXPR
        switch (log2_elements) {
            case 8:  // 256
                scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t, acc_t, 8>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input,
                                                                               grad, output,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               rp_keep);
                break;
            case 9:  // 512
                scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t, acc_t, 9>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input,
                                                                               grad, output,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               rp_keep);
                break;
            case 10:  // 1024
                scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t, acc_t, 10>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input,
                                                                               grad, output,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               rp_keep);
                break;
            case 11:  // 2048
                scaled_upper_triang_masked_softmax_warp_backward<input_t, output_t, acc_t, 11>
                    <<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(grad_input,
                                                                               grad, output,
                                                                               scale,
                                                                               batch_count,
                                                                               softmax_elements_stride,
                                                                               softmax_elements,
                                                                               p_keep,
                                                                               rp_keep);
                break;
            default:
                break;
        }
    }
}

#endif  // TRANSFORMER_ENGINE_SCALED_UPPER_TRIANG_SOFTMAX_H_
