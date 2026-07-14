/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* Vectorization model (gated):
 *
 * With no GLU interleave, the row is laid out as:
 *   [ act[0:H] | gate[0:H] ]
 * With GLU interleave (e.g. 32):
 *   [ act[0:32] | gate[0:32] | act[32:64] | gate[32:64] | ... ]
 *
 * Backward uses fused SiLU / ClampedSiLU specializations (one sigmoid) matching
 * gated_fp8.cuh; other ActOP/DActOP pairs call the ops directly.
 */

#include <algorithm>
#include <type_traits>

#include "../common.h"
#include "../util/math.h"
#include "./scaled_activation.h"

namespace transformer_engine {
namespace {

using namespace detail::scaled_activation;

using WarpReducer = Reducer<float, 1, 1, 1>;

// blockDim.x must be a multiple of warp size and <= kReductionThreads.
__device__ __forceinline__ float block_reduce_sum(float value, float *smem) {
  const int lane = threadIdx.x % THREADS_PER_WARP;
  const int warp = threadIdx.x / THREADS_PER_WARP;
  const int num_warps = blockDim.x / THREADS_PER_WARP;
  Empty params = {};
  WarpReducer reducer(params, /*bidm=*/0, /*bidn=*/0, /*warp_m=*/0, /*warp_n=*/0, lane,
                      /*smem=*/nullptr);
  Sum<float> sum;

  value = reducer.reduce(value, sum);
  if (lane == 0) {
    smem[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < num_warps ? smem[lane] : 0.0f;
  return warp == 0 ? reducer.reduce(value, sum) : value;
}

// ---------------------------------------------------------------------------
// Device helpers: fused Act / DAct (IEEE sigmoid / expf)
// ---------------------------------------------------------------------------

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
__device__ __forceinline__ float gated_forward_value(const float act_in, const float gate_in,
                                                     const ParamOP &param) {
  if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
    const float gate = fminf(fmaxf(-param.limit, gate_in), param.limit) + param.glu_linear_offset;
    return ActOP(act_in, param) * gate;
  } else {
    return ActOP(act_in, param) * gate_in;
  }
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
__device__ __forceinline__ void gated_backward_values(const float act_in, const float gate_in,
                                                      const ParamOP &param, float *dact,
                                                      float *dgate, float *unscaled) {
  Empty empty = {};
  float act_x = 0.0f;
  float dact_x = 0.0f;
  float gate = gate_in;
  bool dgate_mask = true;

  if constexpr (std::is_same<ParamOP, ClampedSwiGLUParam>::value) {
    dgate_mask = gate_in <= param.limit && gate_in >= -param.limit;
    gate = fminf(fmaxf(-param.limit, gate_in), param.limit) + param.glu_linear_offset;
    const bool dact_mask = act_in <= param.limit;
    const float clamped_act_in = fminf(act_in, param.limit);
    const float s = sigmoid<float, float>(param.alpha * clamped_act_in, empty);
    act_x = clamped_act_in * s;
    dact_x = dact_mask ? s + param.alpha * clamped_act_in * s * (1.0f - s) : 0.0f;
  } else if constexpr ((ActOP == &silu<fp32, fp32>) && (DActOP == &dsilu<fp32, fp32>)) {
    const float s = sigmoid<float, float>(act_in, empty);
    act_x = act_in * s;
    dact_x = s + act_in * s * (1.0f - s);
  } else {
    act_x = ActOP(act_in, param);
    dact_x = DActOP(act_in, param);
  }

  *unscaled = act_x * gate;
  *dact = dact_x * gate;
  *dgate = dgate_mask ? act_x : 0.0f;
}

// ---------------------------------------------------------------------------
// Gated kernels
// ---------------------------------------------------------------------------

template <int nvec, typename InputT, typename ScaleT, typename OutputT, typename ParamOP,
          float (*ActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kThreads, 4)
    scaled_gated_forward_kernel(const InputT *__restrict__ input,
                                const ScaleT *__restrict__ act_scales, OutputT *__restrict__ output,
                                const size_t rows, const size_t hidden, const size_t segment_size,
                                const size_t num_segments, const size_t num_vectors_per_segment,
                                const ParamOP param) {
  const size_t total_vectors = rows * num_segments * num_vectors_per_segment;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_segment;
    const size_t segment = (tid / num_vectors_per_segment) % num_segments;
    const size_t row = tid / (num_vectors_per_segment * num_segments);
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;

    VectorizedLoader<InputT, nvec, true> act_loader(input + input_segment_offset, segment_size);
    VectorizedLoader<InputT, nvec, true> gate_loader(input + input_segment_offset + segment_size,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> output_storer(output + output_segment_offset,
                                                        segment_size);
    act_loader.load(vector_idx, segment_size);
    gate_loader.load(vector_idx, segment_size);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      const float unscaled = gated_forward_value<ParamOP, ActOP>(
          static_cast<float>(act_loader.separate()[lane]),
          static_cast<float>(gate_loader.separate()[lane]), param);
      output_storer.separate()[lane] = static_cast<OutputT>(unscaled * scale);
    }
    output_storer.store(vector_idx, segment_size);
  }
}

template <int nvec, typename GradT, typename InputT, typename ScaleT, typename OutputT,
          typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kThreads, 4)
    scaled_gated_backward_kernel(const GradT *__restrict__ grad_output,
                                 const InputT *__restrict__ input,
                                 const ScaleT *__restrict__ act_scales,
                                 OutputT *__restrict__ grad_input, const size_t rows,
                                 const size_t hidden, const size_t segment_size,
                                 const size_t num_segments, const size_t num_vectors_per_segment,
                                 const ParamOP param) {
  const size_t total_vectors = rows * num_segments * num_vectors_per_segment;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_segment;
    const size_t segment = (tid / num_vectors_per_segment) % num_segments;
    const size_t row = tid / (num_vectors_per_segment * num_segments);
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;

    VectorizedLoader<GradT, nvec, true> grad_loader(grad_output + output_segment_offset,
                                                    segment_size);
    VectorizedLoader<InputT, nvec, true> act_loader(input + input_segment_offset, segment_size);
    VectorizedLoader<InputT, nvec, true> gate_loader(input + input_segment_offset + segment_size,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> act_storer(grad_input + input_segment_offset,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> gate_storer(
        grad_input + input_segment_offset + segment_size, segment_size);
    grad_loader.load(vector_idx, segment_size);
    act_loader.load(vector_idx, segment_size);
    gate_loader.load(vector_idx, segment_size);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      float dact = 0.0f;
      float dgate = 0.0f;
      float unscaled = 0.0f;
      gated_backward_values<ParamOP, ActOP, DActOP>(
          static_cast<float>(act_loader.separate()[lane]),
          static_cast<float>(gate_loader.separate()[lane]), param, &dact, &dgate, &unscaled);
      (void)unscaled;
      const float grad = static_cast<float>(grad_loader.separate()[lane]) * scale;
      act_storer.separate()[lane] = static_cast<OutputT>(grad * dact);
      gate_storer.separate()[lane] = static_cast<OutputT>(grad * dgate);
    }
    act_storer.store(vector_idx, segment_size);
    gate_storer.store(vector_idx, segment_size);
  }
}

template <int nvec, typename GradT, typename InputT, typename ScaleT, typename OutputT,
          typename GradScaleT, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kReductionThreads, 4)
    scaled_gated_backward_with_scale_grad_kernel(
        const GradT *__restrict__ grad_output, const InputT *__restrict__ input,
        const ScaleT *__restrict__ act_scales, OutputT *__restrict__ grad_input,
        GradScaleT *__restrict__ grad_act_scales, const size_t rows, const size_t hidden,
        const size_t segment_size, const size_t num_segments, const size_t num_vectors_per_segment,
        const ParamOP param) {
  __shared__ float smem[kReductionWarps];
  const size_t row = blockIdx.x;
  (void)rows;
  float scale_grad = 0.0f;
  const float scale = static_cast<float>(act_scales[row]);

  const size_t row_vectors = num_segments * num_vectors_per_segment;
  for (size_t row_vector_idx = threadIdx.x; row_vector_idx < row_vectors;
       row_vector_idx += blockDim.x) {
    const size_t segment = row_vector_idx / num_vectors_per_segment;
    const size_t vector_idx = row_vector_idx % num_vectors_per_segment;
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;
    VectorizedLoader<GradT, nvec, true> grad_loader(grad_output + output_segment_offset,
                                                    segment_size);
    VectorizedLoader<InputT, nvec, true> act_loader(input + input_segment_offset, segment_size);
    VectorizedLoader<InputT, nvec, true> gate_loader(input + input_segment_offset + segment_size,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> act_storer(grad_input + input_segment_offset,
                                                     segment_size);
    VectorizedStorer<OutputT, nvec, true> gate_storer(
        grad_input + input_segment_offset + segment_size, segment_size);

    grad_loader.load(vector_idx, segment_size);
    act_loader.load(vector_idx, segment_size);
    gate_loader.load(vector_idx, segment_size);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      float dact = 0.0f;
      float dgate = 0.0f;
      float unscaled = 0.0f;
      gated_backward_values<ParamOP, ActOP, DActOP>(
          static_cast<float>(act_loader.separate()[lane]),
          static_cast<float>(gate_loader.separate()[lane]), param, &dact, &dgate, &unscaled);
      const float grad = static_cast<float>(grad_loader.separate()[lane]);
      scale_grad += grad * unscaled;

      const float scaled_grad = grad * scale;
      act_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dact);
      gate_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dgate);
    }
    act_storer.store(vector_idx, segment_size);
    gate_storer.store(vector_idx, segment_size);
  }

  scale_grad = block_reduce_sum(scale_grad, smem);
  if (threadIdx.x == 0) {
    grad_act_scales[row] = static_cast<GradScaleT>(scale_grad);
  }
}

// ---------------------------------------------------------------------------
// Unary kernels
// ---------------------------------------------------------------------------

template <int nvec, typename InputT, typename ScaleT, typename OutputT, typename ParamOP,
          float (*ActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kThreads, 4)
    scaled_unary_forward_kernel(const InputT *__restrict__ input,
                                const ScaleT *__restrict__ act_scales, OutputT *__restrict__ output,
                                const size_t rows, const size_t hidden,
                                const size_t num_vectors_per_row, const ParamOP param) {
  const size_t total_vectors = rows * num_vectors_per_row;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_row;
    const size_t row = tid / num_vectors_per_row;
    VectorizedLoader<InputT, nvec, true> input_loader(input + row * hidden, hidden);
    VectorizedStorer<OutputT, nvec, true> output_storer(output + row * hidden, hidden);
    input_loader.load(vector_idx, hidden);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      const float unscaled = ActOP(static_cast<float>(input_loader.separate()[lane]), param);
      output_storer.separate()[lane] = static_cast<OutputT>(unscaled * scale);
    }
    output_storer.store(vector_idx, hidden);
  }
}

template <int nvec, typename GradT, typename InputT, typename ScaleT, typename OutputT,
          typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kThreads, 4)
    scaled_unary_backward_kernel(const GradT *__restrict__ grad_output,
                                 const InputT *__restrict__ input,
                                 const ScaleT *__restrict__ act_scales,
                                 OutputT *__restrict__ grad_input, const size_t rows,
                                 const size_t hidden, const size_t num_vectors_per_row,
                                 const ParamOP param) {
  const size_t total_vectors = rows * num_vectors_per_row;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_row;
    const size_t row = tid / num_vectors_per_row;
    VectorizedLoader<GradT, nvec, true> grad_loader(grad_output + row * hidden, hidden);
    VectorizedLoader<InputT, nvec, true> input_loader(input + row * hidden, hidden);
    VectorizedStorer<OutputT, nvec, true> grad_input_storer(grad_input + row * hidden, hidden);
    grad_loader.load(vector_idx, hidden);
    input_loader.load(vector_idx, hidden);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      const float grad = static_cast<float>(grad_loader.separate()[lane]) * scale;
      grad_input_storer.separate()[lane] = static_cast<OutputT>(
          grad * DActOP(static_cast<float>(input_loader.separate()[lane]), param));
    }
    grad_input_storer.store(vector_idx, hidden);
  }
}

template <int nvec, typename GradT, typename InputT, typename ScaleT, typename OutputT,
          typename GradScaleT, typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
__global__ void __launch_bounds__(kReductionThreads, 4)
    scaled_unary_backward_with_scale_grad_kernel(
        const GradT *__restrict__ grad_output, const InputT *__restrict__ input,
        const ScaleT *__restrict__ act_scales, OutputT *__restrict__ grad_input,
        GradScaleT *__restrict__ grad_act_scales, const size_t rows, const size_t hidden,
        const size_t num_vectors_per_row, const ParamOP param) {
  __shared__ float smem[kReductionWarps];
  const size_t row = blockIdx.x;
  (void)rows;
  float scale_grad = 0.0f;
  const float scale = static_cast<float>(act_scales[row]);

  VectorizedLoader<GradT, nvec, true> grad_loader(grad_output + row * hidden, hidden);
  VectorizedLoader<InputT, nvec, true> input_loader(input + row * hidden, hidden);
  VectorizedStorer<OutputT, nvec, true> grad_input_storer(grad_input + row * hidden, hidden);
  for (size_t vector_idx = threadIdx.x; vector_idx < num_vectors_per_row;
       vector_idx += blockDim.x) {
    grad_loader.load(vector_idx, hidden);
    input_loader.load(vector_idx, hidden);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      const float x = static_cast<float>(input_loader.separate()[lane]);
      const float unscaled = ActOP(x, param);
      const float grad = static_cast<float>(grad_loader.separate()[lane]);
      scale_grad += grad * unscaled;
      grad_input_storer.separate()[lane] = static_cast<OutputT>(grad * scale * DActOP(x, param));
    }
    grad_input_storer.store(vector_idx, hidden);
  }

  scale_grad = block_reduce_sum(scale_grad, smem);
  if (threadIdx.x == 0) {
    grad_act_scales[row] = static_cast<GradScaleT>(scale_grad);
  }
}

// ---------------------------------------------------------------------------
// Tensor checks
// ---------------------------------------------------------------------------

void check_gated_forward_tensors(const Tensor *input, const Tensor *act_scales,
                                 const Tensor *output, const int64_t glu_interleave_size,
                                 const char *api_name, size_t *rows, size_t *hidden) {
  const auto input_dims = input->flat_2d_dims();
  const auto output_dims = output->flat_2d_dims();
  NVTE_CHECK(input_dims[0] == output_dims[0], api_name, ": input/output row mismatch.");
  NVTE_CHECK(input_dims[1] == output_dims[1] * 2, api_name,
             ": gated input last dimension must be twice output last dimension.");
  NVTE_CHECK(glu_interleave_size >= 0, api_name, ": glu_interleave_size must be non-negative.");
  if (glu_interleave_size > 0) {
    NVTE_CHECK(glu_interleave_size % 32 == 0, api_name,
               ": nonzero glu_interleave_size must be a multiple of 32.");
    NVTE_CHECK(output_dims[1] % static_cast<size_t>(glu_interleave_size) == 0, api_name,
               ": output last dimension must be divisible by glu_interleave_size.");
  }
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  *rows = input_dims[0];
  *hidden = output_dims[1];
}

void check_gated_backward_tensors(const Tensor *grad_output, const Tensor *input,
                                  const Tensor *act_scales, const Tensor *grad_input,
                                  const Tensor *grad_act_scales, const int64_t glu_interleave_size,
                                  const char *api_name, size_t *rows, size_t *hidden) {
  const auto grad_dims = grad_output->flat_2d_dims();
  const auto input_dims = input->flat_2d_dims();
  const auto grad_input_dims = grad_input->flat_2d_dims();
  NVTE_CHECK(grad_dims[0] == input_dims[0] && input_dims[0] == grad_input_dims[0], api_name,
             ": input/grad row mismatch.");
  NVTE_CHECK(input_dims[1] == grad_dims[1] * 2 && grad_input_dims[1] == input_dims[1], api_name,
             ": gated backward dimensions are inconsistent.");
  NVTE_CHECK(glu_interleave_size >= 0, api_name, ": glu_interleave_size must be non-negative.");
  if (glu_interleave_size > 0) {
    NVTE_CHECK(glu_interleave_size % 32 == 0, api_name,
               ": nonzero glu_interleave_size must be a multiple of 32.");
    NVTE_CHECK(grad_dims[1] % static_cast<size_t>(glu_interleave_size) == 0, api_name,
               ": grad last dimension must be divisible by glu_interleave_size.");
  }
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  if (grad_act_scales != nullptr) {
    NVTE_CHECK(grad_act_scales->numel() == input_dims[0], api_name,
               ": grad_act_scales must have one value per row.");
  }
  *rows = input_dims[0];
  *hidden = grad_dims[1];
}

void check_unary_forward_tensors(const Tensor *input, const Tensor *act_scales,
                                 const Tensor *output, const char *api_name, size_t *rows,
                                 size_t *hidden) {
  const auto input_dims = input->flat_2d_dims();
  const auto output_dims = output->flat_2d_dims();
  NVTE_CHECK(input_dims[0] == output_dims[0] && input_dims[1] == output_dims[1], api_name,
             ": input/output shapes must match.");
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  *rows = input_dims[0];
  *hidden = output_dims[1];
}

void check_unary_backward_tensors(const Tensor *grad_output, const Tensor *input,
                                  const Tensor *act_scales, const Tensor *grad_input,
                                  const Tensor *grad_act_scales, const char *api_name, size_t *rows,
                                  size_t *hidden) {
  const auto grad_dims = grad_output->flat_2d_dims();
  const auto input_dims = input->flat_2d_dims();
  const auto grad_input_dims = grad_input->flat_2d_dims();
  NVTE_CHECK(grad_dims[0] == input_dims[0] && input_dims[0] == grad_input_dims[0], api_name,
             ": input/grad row mismatch.");
  NVTE_CHECK(grad_dims[1] == input_dims[1] && input_dims[1] == grad_input_dims[1], api_name,
             ": unary backward dimensions are inconsistent.");
  NVTE_CHECK(act_scales->numel() == input_dims[0], api_name,
             ": act_scales must have one value per row.");
  if (grad_act_scales != nullptr) {
    NVTE_CHECK(grad_act_scales->numel() == input_dims[0], api_name,
               ": grad_act_scales must have one value per row.");
  }
  *rows = input_dims[0];
  *hidden = grad_dims[1];
}

}  // namespace

// ---------------------------------------------------------------------------
// Launch implementations
// ---------------------------------------------------------------------------

using namespace detail::scaled_activation;

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void launch_scaled_gated_forward(const NVTETensor nvte_input, const NVTETensor nvte_act_scales,
                                 NVTETensor nvte_output, ParamOP param, int64_t glu_interleave_size,
                                 cudaStream_t stream, const char *api_name) {
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *output = convertNVTETensorCheck(nvte_output);
  size_t rows = 0;
  size_t hidden = 0;
  check_gated_forward_tensors(input, act_scales, output, glu_interleave_size, api_name, &rows,
                              &hidden);
  if (rows == 0 || hidden == 0) return;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(output->data.dtype, OutputT, {
        constexpr int nvec = 32 / static_cast<int>(sizeof(InputT));
        const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
        const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
        auto output_ptr = reinterpret_cast<OutputT *>(output->data.dptr);
        const size_t segment_size =
            glu_interleave_size > 0 ? static_cast<size_t>(glu_interleave_size) : hidden;
        const size_t num_segments = glu_interleave_size > 0 ? hidden / segment_size : 1;
        const auto align = row_vector_alignment(segment_size, nvec, input_ptr,
                                                input_ptr + segment_size, output_ptr);
        const bool use_vector = align == Alignment::SAME_ALIGNED;
        const size_t num_vectors =
            use_vector ? get_num_aligned_elements(input_ptr, segment_size, nvec, sizeof(InputT))
                       : segment_size;
        const int blocks = static_cast<int>(std::min<size_t>(
            DIVUP(rows * num_segments * num_vectors, static_cast<size_t>(kThreads)), 65535));
        if (use_vector) {
          scaled_gated_forward_kernel<nvec, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                segment_size, num_segments, num_vectors, param);
        } else {
          scaled_gated_forward_kernel<1, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                segment_size, num_segments, segment_size, param);
        }
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void launch_scaled_gated_backward(const NVTETensor nvte_grad_output, const NVTETensor nvte_input,
                                  const NVTETensor nvte_act_scales, NVTETensor nvte_grad_input,
                                  NVTETensor nvte_grad_act_scales, ParamOP param,
                                  int64_t glu_interleave_size, cudaStream_t stream,
                                  const char *api_name) {
  const Tensor *grad_output = convertNVTETensorCheck(nvte_grad_output);
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *grad_input = convertNVTETensorCheck(nvte_grad_input);
  Tensor *grad_act_scales =
      nvte_grad_act_scales == nullptr ? nullptr : convertNVTETensorCheck(nvte_grad_act_scales);
  size_t rows = 0;
  size_t hidden = 0;
  check_gated_backward_tensors(grad_output, input, act_scales, grad_input, grad_act_scales,
                               glu_interleave_size, api_name, &rows, &hidden);
  if (rows == 0 || hidden == 0) return;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_output->data.dtype, GradT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_input->data.dtype, OutputT, {
          constexpr int nvec =
              32 / static_cast<int>(std::max({sizeof(GradT), sizeof(InputT), sizeof(OutputT)}));
          const auto grad_ptr = reinterpret_cast<const GradT *>(grad_output->data.dptr);
          const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
          const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
          auto grad_input_ptr = reinterpret_cast<OutputT *>(grad_input->data.dptr);
          const size_t segment_size =
              glu_interleave_size > 0 ? static_cast<size_t>(glu_interleave_size) : hidden;
          const size_t num_segments = glu_interleave_size > 0 ? hidden / segment_size : 1;
          const auto align = row_vector_alignment(segment_size, nvec, grad_ptr, input_ptr,
                                                  input_ptr + segment_size, grad_input_ptr,
                                                  grad_input_ptr + segment_size);
          const bool use_vector = align == Alignment::SAME_ALIGNED;
          const size_t num_vectors =
              use_vector ? get_num_aligned_elements(input_ptr, segment_size, nvec, sizeof(InputT))
                         : segment_size;
          if (grad_act_scales == nullptr) {
            const int blocks = static_cast<int>(std::min<size_t>(
                DIVUP(rows * num_segments * num_vectors, static_cast<size_t>(kThreads)), 65535));
            if (use_vector) {
              scaled_gated_backward_kernel<nvec, GradT, InputT, ScaleT, OutputT, ParamOP, ActOP,
                                           DActOP><<<blocks, kThreads, 0, stream>>>(
                  grad_ptr, input_ptr, scale_ptr, grad_input_ptr, rows, hidden, segment_size,
                  num_segments, num_vectors, param);
            } else {
              scaled_gated_backward_kernel<1, GradT, InputT, ScaleT, OutputT, ParamOP, ActOP,
                                           DActOP><<<blocks, kThreads, 0, stream>>>(
                  grad_ptr, input_ptr, scale_ptr, grad_input_ptr, rows, hidden, segment_size,
                  num_segments, segment_size, param);
            }
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              auto grad_act_scales_ptr = reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr);
              const int reduction_threads = choose_reduction_threads(num_segments * num_vectors);
              if (use_vector) {
                scaled_gated_backward_with_scale_grad_kernel<nvec, GradT, InputT, ScaleT, OutputT,
                                                             GradScaleT, ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, segment_size, num_segments, num_vectors, param);
              } else {
                scaled_gated_backward_with_scale_grad_kernel<1, GradT, InputT, ScaleT, OutputT,
                                                             GradScaleT, ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, segment_size, num_segments, segment_size, param);
              }
            });
          }
        });
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &)>
void launch_scaled_unary_forward(const NVTETensor nvte_input, const NVTETensor nvte_act_scales,
                                 NVTETensor nvte_output, ParamOP param, cudaStream_t stream,
                                 const char *api_name) {
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *output = convertNVTETensorCheck(nvte_output);
  size_t rows = 0;
  size_t hidden = 0;
  check_unary_forward_tensors(input, act_scales, output, api_name, &rows, &hidden);
  if (rows == 0 || hidden == 0) return;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(output->data.dtype, OutputT, {
        constexpr int nvec = 32 / static_cast<int>(sizeof(InputT));
        const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
        const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
        auto output_ptr = reinterpret_cast<OutputT *>(output->data.dptr);
        const auto align = row_vector_alignment(hidden, nvec, input_ptr, output_ptr);
        const bool use_vector = align == Alignment::SAME_ALIGNED;
        const size_t num_vectors =
            use_vector ? get_num_aligned_elements(input_ptr, hidden, nvec, sizeof(InputT)) : hidden;
        const int blocks = static_cast<int>(
            std::min<size_t>(DIVUP(rows * num_vectors, static_cast<size_t>(kThreads)), 65535));
        if (use_vector) {
          scaled_unary_forward_kernel<nvec, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                num_vectors, param);
        } else {
          scaled_unary_forward_kernel<1, InputT, ScaleT, OutputT, ParamOP, ActOP>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                hidden, param);
        }
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <typename ParamOP, float (*ActOP)(float, const ParamOP &),
          float (*DActOP)(float, const ParamOP &)>
void launch_scaled_unary_backward(const NVTETensor nvte_grad_output, const NVTETensor nvte_input,
                                  const NVTETensor nvte_act_scales, NVTETensor nvte_grad_input,
                                  NVTETensor nvte_grad_act_scales, ParamOP param,
                                  cudaStream_t stream, const char *api_name) {
  const Tensor *grad_output = convertNVTETensorCheck(nvte_grad_output);
  const Tensor *input = convertNVTETensorCheck(nvte_input);
  const Tensor *act_scales = convertNVTETensorCheck(nvte_act_scales);
  Tensor *grad_input = convertNVTETensorCheck(nvte_grad_input);
  Tensor *grad_act_scales =
      nvte_grad_act_scales == nullptr ? nullptr : convertNVTETensorCheck(nvte_grad_act_scales);
  size_t rows = 0;
  size_t hidden = 0;
  check_unary_backward_tensors(grad_output, input, act_scales, grad_input, grad_act_scales,
                               api_name, &rows, &hidden);
  if (rows == 0 || hidden == 0) return;

  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_output->data.dtype, GradT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_input->data.dtype, OutputT, {
          constexpr int nvec =
              32 / static_cast<int>(std::max({sizeof(GradT), sizeof(InputT), sizeof(OutputT)}));
          const auto grad_ptr = reinterpret_cast<const GradT *>(grad_output->data.dptr);
          const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
          const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
          auto grad_input_ptr = reinterpret_cast<OutputT *>(grad_input->data.dptr);
          const auto align =
              row_vector_alignment(hidden, nvec, grad_ptr, input_ptr, grad_input_ptr);
          const bool use_vector = align == Alignment::SAME_ALIGNED;
          const size_t num_vectors =
              use_vector ? get_num_aligned_elements(input_ptr, hidden, nvec, sizeof(InputT))
                         : hidden;
          if (grad_act_scales == nullptr) {
            const int blocks = static_cast<int>(
                std::min<size_t>(DIVUP(rows * num_vectors, static_cast<size_t>(kThreads)), 65535));
            if (use_vector) {
              scaled_unary_backward_kernel<nvec, GradT, InputT, ScaleT, OutputT, ParamOP, ActOP,
                                           DActOP><<<blocks, kThreads, 0, stream>>>(
                  grad_ptr, input_ptr, scale_ptr, grad_input_ptr, rows, hidden, num_vectors, param);
            } else {
              scaled_unary_backward_kernel<1, GradT, InputT, ScaleT, OutputT, ParamOP, ActOP,
                                           DActOP><<<blocks, kThreads, 0, stream>>>(
                  grad_ptr, input_ptr, scale_ptr, grad_input_ptr, rows, hidden, hidden, param);
            }
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              auto grad_act_scales_ptr = reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr);
              const int reduction_threads = choose_reduction_threads(num_vectors);
              if (use_vector) {
                scaled_unary_backward_with_scale_grad_kernel<nvec, GradT, InputT, ScaleT, OutputT,
                                                             GradScaleT, ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, num_vectors, param);
              } else {
                scaled_unary_backward_with_scale_grad_kernel<1, GradT, InputT, ScaleT, OutputT,
                                                             GradScaleT, ParamOP, ActOP, DActOP>
                    <<<static_cast<int>(rows), reduction_threads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, hidden, param);
              }
            });
          }
        });
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Explicit instantiations
// ---------------------------------------------------------------------------

template void launch_scaled_gated_forward<Empty, silu<fp32, fp32>>(const NVTETensor,
                                                                   const NVTETensor, NVTETensor,
                                                                   Empty, int64_t, cudaStream_t,
                                                                   const char *);
template void launch_scaled_gated_backward<Empty, silu<fp32, fp32>, dsilu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, Empty, int64_t,
    cudaStream_t, const char *);

template void launch_scaled_gated_forward<ClampedSwiGLUParam, clamped_silu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, NVTETensor, ClampedSwiGLUParam, int64_t, cudaStream_t,
    const char *);
template void launch_scaled_gated_backward<ClampedSwiGLUParam, clamped_silu<fp32, fp32>,
                                           clamped_dsilu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor, NVTETensor,
    ClampedSwiGLUParam, int64_t, cudaStream_t, const char *);

template void launch_scaled_unary_forward<Empty, srelu<fp32, fp32>>(const NVTETensor,
                                                                    const NVTETensor, NVTETensor,
                                                                    Empty, cudaStream_t,
                                                                    const char *);
template void launch_scaled_unary_backward<Empty, srelu<fp32, fp32>, dsrelu<fp32, fp32>>(
    const NVTETensor, const NVTETensor, const NVTETensor, NVTETensor, NVTETensor, Empty,
    cudaStream_t, const char *);

}  // namespace transformer_engine
