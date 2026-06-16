/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/* Scaled activations: apply an activation, multiply by a per-row scale
 * (act_scales[row]), do all math in fp32, and cast once at the store. The
 * backward path optionally also reduces the gradient of the per-row scale.
 *
 * The six __global__ kernels below:
 *
 *   # | Kernel                                        | Activation             | Dir | grad_act_scales | Launch
 *  ---+-----------------------------------------------+------------------------+-----+-----------------+--------------------
 *   1 | scaled_gated_forward_kernel                   | SwiGLU / ClampedSwiGLU | fwd | --              | vectorized row segments
 *   2 | scaled_srelu_forward_kernel                   | SReLU (unary)          | fwd | --              | vectorized flat grid
 *   3 | scaled_gated_backward_kernel                  | SwiGLU / ClampedSwiGLU | bwd | no              | vectorized row segments
 *   4 | scaled_srelu_backward_kernel                  | SReLU                  | bwd | no              | vectorized flat grid
 *   5 | scaled_gated_backward_with_scale_grad_kernel  | SwiGLU / ClampedSwiGLU | bwd | yes             | vectorized, one block per row
 *   6 | scaled_srelu_backward_with_scale_grad_kernel  | SReLU                  | bwd | yes             | vectorized, one block per row
 *
 * The "with scale grad" variants compute grad_act_scales[row] = sum_j dY * unscaled,
 * a per-row reduction that requires the one-block-per-row launch; when
 * grad_act_scales is null the cheaper flat element-wise grid is used instead.
 *
 * Vectorization model:
 *
 * Gated activations consume two FC1 streams per row: an activation stream and a
 * gate stream. With no GLU interleave, the row is laid out as:
 *
 *   [ act[0:H] | gate[0:H] ]
 *
 * With GLU interleave, e.g. interleave=32, the row is laid out as independent
 * act/gate segments:
 *
 *   [ act[0:32] | gate[0:32] | act[32:64] | gate[32:64] | ... ]
 *
 * Vector loads:
 *
 *   interleave=0:
 *     input  [ act0 | act1 | ... | actN | gate0 | gate1 | ... | gateN ]
 *                 |                         |
 *                 v                         v
 *     load   act vector i             gate vector i
 *     store  output vector i = activation(act vector i) * gate vector i * scale[row]
 *
 *   interleave=32:
 *     input  [ act0 | gate0 | act1 | gate1 | ... | actN | gateN ]
 *                 |     |      |     |
 *                 v     v      v     v
 *     load     act0  gate0  act1  gate1
 *     store  output vector i = activation(act vector i) * gate vector i * scale[row]
 *
 * Only fully aligned segments use vector loads. Everything else uses the same
 * kernels with nvec=1, i.e. regular elementwise loads/stores.
 */

#include <transformer_engine/activation.h>

#include <algorithm>

#include "../common.h"
#include "../util/math.h"
#include "../util/vectorized_pointwise.h"

namespace transformer_engine {
namespace {

enum class ScaledActivation {
  kSwiGLU,
  kClampedSwiGLU,
  kSReLU,
};

template <ScaledActivation Act>
__device__ __forceinline__ float gated_forward_value(const float act_in, const float gate_in,
                                                     const ClampedSwiGLUParam &param) {
  if constexpr (Act == ScaledActivation::kSwiGLU) {
    Empty empty = {};
    return silu<float, float>(act_in, empty) * gate_in;
  } else {
    const float gate = fminf(fmaxf(-param.limit, gate_in), param.limit) + param.glu_linear_offset;
    return clamped_silu<float, float>(act_in, param) * gate;
  }
}

template <ScaledActivation Act>
__device__ __forceinline__ void gated_backward_values(const float act_in, const float gate_in,
                                                      const ClampedSwiGLUParam &param, float *dact,
                                                      float *dgate,
                                                      float *unscaled) {
  if constexpr (Act == ScaledActivation::kSwiGLU) {
    Empty empty = {};
    const float act = silu<float, float>(act_in, empty);
    *unscaled = act * gate_in;
    *dact = dsilu<float, float>(act_in, empty) * gate_in;
    *dgate = act;
  } else {
    const bool dgate_mask = gate_in <= param.limit && gate_in >= -param.limit;
    const float gate = fminf(fmaxf(-param.limit, gate_in), param.limit) + param.glu_linear_offset;
    const float act = clamped_silu<float, float>(act_in, param);
    *unscaled = act * gate;
    *dact = clamped_dsilu<float, float>(act_in, param) * gate;
    *dgate = dgate_mask ? act : 0.0f;
  }
}

constexpr int kThreads = unary_kernel_threads;

template <typename T>
constexpr int vector_width() {
  return 32 / static_cast<int>(sizeof(T));
}

inline int launch_blocks(const size_t work_items) {
  return static_cast<int>(
      std::min<size_t>(DIVUP(work_items, static_cast<size_t>(kThreads)), 65535));
}

template <typename... Ptrs>
Alignment row_vector_alignment(const size_t lead_dim, const int nvec, const Ptrs... ptrs) {
  if (nvec == 1) {
    return Alignment::SAME_ALIGNED;
  }
  // GLU interleave is handled as independent row-local segments. Keep the scalar
  // fallback for odd segment widths or unaligned pointers so vector stores never
  // cross from an activation segment into its paired gate segment.
  if (lead_dim % static_cast<size_t>(nvec) != 0) {
    return Alignment::DIFFERENT;
  }
  const auto align = CheckAlignment(lead_dim, nvec, ptrs...);
  return align == Alignment::SAME_ALIGNED ? Alignment::SAME_ALIGNED : Alignment::DIFFERENT;
}

template <int nvec, bool aligned>
__device__ __forceinline__ bool vector_lane_index(const size_t vector_idx, const int lane,
                                                  const int alignment, const size_t length,
                                                  size_t *index) {
  size_t idx = vector_idx * static_cast<size_t>(nvec) + static_cast<size_t>(lane);
  if constexpr (!aligned) {
    if (idx < static_cast<size_t>(alignment)) {
      return false;
    }
    idx -= static_cast<size_t>(alignment);
  }
  if (idx >= length) {
    return false;
  }
  *index = idx;
  return true;
}

template <int nvec, bool aligned, typename InputT, typename ScaleT, typename OutputT,
          ScaledActivation Act>
__global__ void scaled_gated_forward_kernel(const InputT *input, const ScaleT *act_scales,
                                            OutputT *output, const size_t rows,
                                            const size_t hidden, const size_t segment_size,
                                            const size_t num_segments,
                                            const size_t num_vectors_per_segment,
                                            const ClampedSwiGLUParam param) {
  const size_t total_vectors = rows * num_segments * num_vectors_per_segment;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_segment;
    const size_t segment = (tid / num_vectors_per_segment) % num_segments;
    const size_t row = tid / (num_vectors_per_segment * num_segments);
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;

    VectorizedLoader<InputT, nvec, aligned> act_loader(input + input_segment_offset,
                                                       segment_size);
    VectorizedLoader<InputT, nvec, aligned> gate_loader(
        input + input_segment_offset + segment_size, segment_size);
    VectorizedStorer<OutputT, nvec, aligned> output_storer(output + output_segment_offset,
                                                           segment_size);
    if (vector_idx >= act_loader.num_aligned_elements()) {
      continue;
    }

    act_loader.load(vector_idx, segment_size);
    gate_loader.load(vector_idx, segment_size);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      size_t col = 0;
      if (vector_lane_index<nvec, aligned>(vector_idx, lane, act_loader.alignment(),
                                           segment_size, &col)) {
        const float unscaled =
            gated_forward_value<Act>(static_cast<float>(act_loader.separate()[lane]),
                                     static_cast<float>(gate_loader.separate()[lane]), param);
        output_storer.separate()[lane] = static_cast<OutputT>(unscaled * scale);
      }
    }
    output_storer.store(vector_idx, segment_size);
  }
}

template <int nvec, bool aligned, typename InputT, typename ScaleT, typename OutputT>
__global__ void scaled_srelu_forward_kernel(const InputT *input, const ScaleT *act_scales,
                                            OutputT *output, const size_t total,
                                            const size_t hidden,
                                            const size_t num_vectors) {
  Empty empty = {};
  VectorizedLoader<InputT, nvec, aligned> input_loader(input, total);
  VectorizedStorer<OutputT, nvec, aligned> output_storer(output, total);
  for (size_t vector_idx = blockIdx.x * blockDim.x + threadIdx.x; vector_idx < num_vectors;
       vector_idx += gridDim.x * blockDim.x) {
    if (vector_idx >= input_loader.num_aligned_elements()) {
      continue;
    }
    input_loader.load(vector_idx, total);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      size_t idx = 0;
      if (vector_lane_index<nvec, aligned>(vector_idx, lane, input_loader.alignment(), total,
                                           &idx)) {
        const size_t row = idx / hidden;
        const float unscaled = srelu<float, float>(static_cast<float>(input_loader.separate()[lane]),
                                                   empty);
        const float scale = static_cast<float>(act_scales[row]);
        output_storer.separate()[lane] = static_cast<OutputT>(unscaled * scale);
      }
    }
    output_storer.store(vector_idx, total);
  }
}

template <int nvec, bool aligned, typename GradT, typename InputT, typename ScaleT,
          typename OutputT, ScaledActivation Act>
__global__ void scaled_gated_backward_kernel(
    const GradT *grad_output, const InputT *input, const ScaleT *act_scales, OutputT *grad_input,
    const size_t rows, const size_t hidden, const size_t segment_size, const size_t num_segments,
    const size_t num_vectors_per_segment, const ClampedSwiGLUParam param) {
  const size_t total_vectors = rows * num_segments * num_vectors_per_segment;
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total_vectors;
       tid += gridDim.x * blockDim.x) {
    const size_t vector_idx = tid % num_vectors_per_segment;
    const size_t segment = (tid / num_vectors_per_segment) % num_segments;
    const size_t row = tid / (num_vectors_per_segment * num_segments);
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;

    VectorizedLoader<GradT, nvec, aligned> grad_loader(grad_output + output_segment_offset,
                                                       segment_size);
    VectorizedLoader<InputT, nvec, aligned> act_loader(input + input_segment_offset,
                                                       segment_size);
    VectorizedLoader<InputT, nvec, aligned> gate_loader(
        input + input_segment_offset + segment_size, segment_size);
    VectorizedStorer<OutputT, nvec, aligned> act_storer(grad_input + input_segment_offset,
                                                        segment_size);
    VectorizedStorer<OutputT, nvec, aligned> gate_storer(
        grad_input + input_segment_offset + segment_size, segment_size);
    if (vector_idx >= act_loader.num_aligned_elements()) {
      continue;
    }

    grad_loader.load(vector_idx, segment_size);
    act_loader.load(vector_idx, segment_size);
    gate_loader.load(vector_idx, segment_size);
    const float scale = static_cast<float>(act_scales[row]);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      size_t col = 0;
      if (vector_lane_index<nvec, aligned>(vector_idx, lane, act_loader.alignment(),
                                           segment_size, &col)) {
        float dact = 0.0f;
        float dgate = 0.0f;
        float unscaled = 0.0f;
        gated_backward_values<Act>(static_cast<float>(act_loader.separate()[lane]),
                                   static_cast<float>(gate_loader.separate()[lane]), param, &dact,
                                   &dgate, &unscaled);
        (void)unscaled;
        const float grad = static_cast<float>(grad_loader.separate()[lane]) * scale;
        act_storer.separate()[lane] = static_cast<OutputT>(grad * dact);
        gate_storer.separate()[lane] = static_cast<OutputT>(grad * dgate);
      }
    }
    act_storer.store(vector_idx, segment_size);
    gate_storer.store(vector_idx, segment_size);
  }
}

template <int nvec, bool aligned, typename GradT, typename InputT, typename ScaleT,
          typename OutputT>
__global__ void scaled_srelu_backward_kernel(const GradT *grad_output, const InputT *input,
                                             const ScaleT *act_scales, OutputT *grad_input,
                                             const size_t total, const size_t hidden,
                                             const size_t num_vectors) {
  Empty empty = {};
  VectorizedLoader<GradT, nvec, aligned> grad_loader(grad_output, total);
  VectorizedLoader<InputT, nvec, aligned> input_loader(input, total);
  VectorizedStorer<OutputT, nvec, aligned> grad_input_storer(grad_input, total);
  for (size_t vector_idx = blockIdx.x * blockDim.x + threadIdx.x; vector_idx < num_vectors;
       vector_idx += gridDim.x * blockDim.x) {
    if (vector_idx >= input_loader.num_aligned_elements()) {
      continue;
    }
    grad_loader.load(vector_idx, total);
    input_loader.load(vector_idx, total);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      size_t idx = 0;
      if (vector_lane_index<nvec, aligned>(vector_idx, lane, input_loader.alignment(), total,
                                           &idx)) {
        const size_t row = idx / hidden;
        const float scale = static_cast<float>(act_scales[row]);
        const float grad = static_cast<float>(grad_loader.separate()[lane]) * scale;
        grad_input_storer.separate()[lane] =
            static_cast<OutputT>(grad * dsrelu<float, float>(
                                            static_cast<float>(input_loader.separate()[lane]),
                                            empty));
      }
    }
    grad_input_storer.store(vector_idx, total);
  }
}

template <int nvec, bool aligned, typename GradT, typename InputT, typename ScaleT,
          typename OutputT, typename GradScaleT, ScaledActivation Act>
__global__ void scaled_gated_backward_with_scale_grad_kernel(
    const GradT *grad_output, const InputT *input, const ScaleT *act_scales, OutputT *grad_input,
    GradScaleT *grad_act_scales, const size_t rows, const size_t hidden,
    const size_t segment_size, const size_t num_segments, const size_t num_vectors_per_segment,
    const ClampedSwiGLUParam param) {
  __shared__ float smem[kThreads];
  const size_t row = blockIdx.x;
  (void)rows;
  float scale_grad = 0.0f;

  for (size_t segment = 0; segment < num_segments; ++segment) {
    const size_t input_segment_offset = row * hidden * 2 + segment * segment_size * 2;
    const size_t output_segment_offset = row * hidden + segment * segment_size;
    VectorizedLoader<GradT, nvec, aligned> grad_loader(grad_output + output_segment_offset,
                                                       segment_size);
    VectorizedLoader<InputT, nvec, aligned> act_loader(input + input_segment_offset,
                                                       segment_size);
    VectorizedLoader<InputT, nvec, aligned> gate_loader(
        input + input_segment_offset + segment_size, segment_size);
    VectorizedStorer<OutputT, nvec, aligned> act_storer(grad_input + input_segment_offset,
                                                        segment_size);
    VectorizedStorer<OutputT, nvec, aligned> gate_storer(
        grad_input + input_segment_offset + segment_size, segment_size);

    for (size_t vector_idx = threadIdx.x; vector_idx < num_vectors_per_segment;
         vector_idx += blockDim.x) {
      if (vector_idx >= act_loader.num_aligned_elements()) {
        continue;
      }
      grad_loader.load(vector_idx, segment_size);
      act_loader.load(vector_idx, segment_size);
      gate_loader.load(vector_idx, segment_size);
#pragma unroll
      for (int lane = 0; lane < nvec; ++lane) {
        size_t col = 0;
        if (vector_lane_index<nvec, aligned>(vector_idx, lane, act_loader.alignment(),
                                             segment_size, &col)) {
          float dact = 0.0f;
          float dgate = 0.0f;
          float unscaled = 0.0f;
          gated_backward_values<Act>(static_cast<float>(act_loader.separate()[lane]),
                                     static_cast<float>(gate_loader.separate()[lane]), param, &dact,
                                     &dgate, &unscaled);
          const float grad = static_cast<float>(grad_loader.separate()[lane]);
          scale_grad += grad * unscaled;

          const float scale = static_cast<float>(act_scales[row]);
          const float scaled_grad = grad * scale;
          act_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dact);
          gate_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dgate);
        }
      }
      act_storer.store(vector_idx, segment_size);
      gate_storer.store(vector_idx, segment_size);
    }
  }

  smem[threadIdx.x] = scale_grad;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    grad_act_scales[row] = static_cast<GradScaleT>(smem[0]);
  }
}

template <int nvec, bool aligned, typename GradT, typename InputT, typename ScaleT,
          typename OutputT, typename GradScaleT>
__global__ void scaled_srelu_backward_with_scale_grad_kernel(
    const GradT *grad_output, const InputT *input, const ScaleT *act_scales, OutputT *grad_input,
    GradScaleT *grad_act_scales, const size_t rows, const size_t hidden,
    const size_t num_vectors_per_row) {
  __shared__ float smem[kThreads];
  const size_t row = blockIdx.x;
  (void)rows;
  float scale_grad = 0.0f;
  Empty empty = {};

  VectorizedLoader<GradT, nvec, aligned> grad_loader(grad_output + row * hidden, hidden);
  VectorizedLoader<InputT, nvec, aligned> input_loader(input + row * hidden, hidden);
  VectorizedStorer<OutputT, nvec, aligned> grad_input_storer(grad_input + row * hidden, hidden);
  for (size_t vector_idx = threadIdx.x; vector_idx < num_vectors_per_row;
       vector_idx += blockDim.x) {
    if (vector_idx >= input_loader.num_aligned_elements()) {
      continue;
    }
    grad_loader.load(vector_idx, hidden);
    input_loader.load(vector_idx, hidden);
#pragma unroll
    for (int lane = 0; lane < nvec; ++lane) {
      size_t col = 0;
      if (vector_lane_index<nvec, aligned>(vector_idx, lane, input_loader.alignment(), hidden,
                                           &col)) {
        const float unscaled =
            srelu<float, float>(static_cast<float>(input_loader.separate()[lane]), empty);
        const float grad = static_cast<float>(grad_loader.separate()[lane]);
        scale_grad += grad * unscaled;

        const float scale = static_cast<float>(act_scales[row]);
        const float scaled_grad = grad * scale;
        const float dact =
            dsrelu<float, float>(static_cast<float>(input_loader.separate()[lane]), empty);
        grad_input_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dact);
      }
    }
    grad_input_storer.store(vector_idx, hidden);
  }

  smem[threadIdx.x] = scale_grad;
  __syncthreads();
  for (int offset = kThreads / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      smem[threadIdx.x] += smem[threadIdx.x + offset];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    grad_act_scales[row] = static_cast<GradScaleT>(smem[0]);
  }
}

void check_scale_tensor(const Tensor *act_scales, const size_t rows, const char *api_name) {
  NVTE_CHECK(act_scales->numel() == rows, api_name, ": act_scales must have one value per row.");
}

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
  check_scale_tensor(act_scales, input_dims[0], api_name);
  *rows = input_dims[0];
  *hidden = output_dims[1];
}

void check_unary_forward_tensors(const Tensor *input, const Tensor *act_scales,
                                 const Tensor *output, const char *api_name, size_t *rows,
                                 size_t *hidden) {
  const auto input_dims = input->flat_2d_dims();
  const auto output_dims = output->flat_2d_dims();
  NVTE_CHECK(input_dims[0] == output_dims[0] && input_dims[1] == output_dims[1], api_name,
             ": input/output shapes must match.");
  check_scale_tensor(act_scales, input_dims[0], api_name);
  *rows = input_dims[0];
  *hidden = output_dims[1];
}

void check_grad_scale_tensor(const Tensor *grad_act_scales, const size_t rows,
                             const char *api_name) {
  if (grad_act_scales != nullptr) {
    NVTE_CHECK(grad_act_scales->numel() == rows, api_name,
               ": grad_act_scales must have one value per row.");
  }
}

void check_gated_backward_tensors(const Tensor *grad_output, const Tensor *input,
                                  const Tensor *act_scales, const Tensor *grad_input,
                                  const Tensor *grad_act_scales,
                                  const int64_t glu_interleave_size, const char *api_name,
                                  size_t *rows, size_t *hidden) {
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
  check_scale_tensor(act_scales, input_dims[0], api_name);
  check_grad_scale_tensor(grad_act_scales, input_dims[0], api_name);
  *rows = input_dims[0];
  *hidden = grad_dims[1];
}

void check_unary_backward_tensors(const Tensor *grad_output, const Tensor *input,
                                  const Tensor *act_scales, const Tensor *grad_input,
                                  const Tensor *grad_act_scales, const char *api_name,
                                  size_t *rows, size_t *hidden) {
  const auto grad_dims = grad_output->flat_2d_dims();
  const auto input_dims = input->flat_2d_dims();
  const auto grad_input_dims = grad_input->flat_2d_dims();
  NVTE_CHECK(grad_dims[0] == input_dims[0] && input_dims[0] == grad_input_dims[0], api_name,
             ": input/grad row mismatch.");
  NVTE_CHECK(grad_dims[1] == input_dims[1] && input_dims[1] == grad_input_dims[1], api_name,
             ": unary backward dimensions are inconsistent.");
  check_scale_tensor(act_scales, input_dims[0], api_name);
  check_grad_scale_tensor(grad_act_scales, input_dims[0], api_name);
  *rows = input_dims[0];
  *hidden = grad_dims[1];
}

template <ScaledActivation Act>
void launch_scaled_gated_forward(const NVTETensor nvte_input, const NVTETensor nvte_act_scales,
                                 NVTETensor nvte_output, const int64_t glu_interleave_size,
                                 const ClampedSwiGLUParam param, cudaStream_t stream,
                                 const char *api_name) {
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
        constexpr int nvec =
            sizeof(InputT) == sizeof(OutputT) ? vector_width<InputT>() : 1;
        const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
        const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
        auto output_ptr = reinterpret_cast<OutputT *>(output->data.dptr);
        const size_t segment_size =
            glu_interleave_size > 0 ? static_cast<size_t>(glu_interleave_size) : hidden;
        const size_t num_segments = glu_interleave_size > 0 ? hidden / segment_size : 1;
        const auto align =
            row_vector_alignment(segment_size, nvec, input_ptr, input_ptr + segment_size,
                                 output_ptr);
        const bool use_vector = align == Alignment::SAME_ALIGNED;
        const size_t num_vectors =
            use_vector ? get_num_aligned_elements(input_ptr, segment_size, nvec, sizeof(InputT))
                       : segment_size;
        const int blocks = launch_blocks(rows * num_segments * num_vectors);
        if (use_vector) {
          scaled_gated_forward_kernel<nvec, true, InputT, ScaleT, OutputT, Act>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                segment_size, num_segments, num_vectors, param);
        } else {
          scaled_gated_forward_kernel<1, true, InputT, ScaleT, OutputT, Act>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                segment_size, num_segments, segment_size, param);
        }
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

void launch_scaled_srelu_forward(const NVTETensor nvte_input, const NVTETensor nvte_act_scales,
                                 NVTETensor nvte_output, cudaStream_t stream,
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
        constexpr int nvec =
            sizeof(InputT) == sizeof(OutputT) ? vector_width<InputT>() : 1;
        const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
        const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
        auto output_ptr = reinterpret_cast<OutputT *>(output->data.dptr);
        const size_t total = rows * hidden;
        const auto align = CheckAlignment(total, nvec, input_ptr, output_ptr);
        const bool use_vector = align == Alignment::SAME_ALIGNED;
        const size_t num_vectors =
            use_vector ? get_num_aligned_elements(input_ptr, total, nvec, sizeof(InputT)) : total;
        const int blocks = launch_blocks(num_vectors);
        if (use_vector) {
          scaled_srelu_forward_kernel<nvec, true, InputT, ScaleT, OutputT>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, total, hidden,
                                                num_vectors);
        } else {
          scaled_srelu_forward_kernel<1, true, InputT, ScaleT, OutputT>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, total, hidden,
                                                total);
        }
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

template <ScaledActivation Act>
void launch_scaled_gated_backward(const NVTETensor nvte_grad_output, const NVTETensor nvte_input,
                                  const NVTETensor nvte_act_scales, NVTETensor nvte_grad_input,
                                  NVTETensor nvte_grad_act_scales,
                                  const int64_t glu_interleave_size,
                                  const ClampedSwiGLUParam param, cudaStream_t stream,
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
          constexpr int nvec = sizeof(GradT) == sizeof(InputT) &&
                                       sizeof(InputT) == sizeof(OutputT)
                                   ? vector_width<GradT>()
                                   : 1;
          const auto grad_ptr = reinterpret_cast<const GradT *>(grad_output->data.dptr);
          const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
          const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
          auto grad_input_ptr = reinterpret_cast<OutputT *>(grad_input->data.dptr);
          const size_t segment_size =
              glu_interleave_size > 0 ? static_cast<size_t>(glu_interleave_size) : hidden;
          const size_t num_segments = glu_interleave_size > 0 ? hidden / segment_size : 1;
          const auto align = row_vector_alignment(
              segment_size, nvec, grad_ptr, input_ptr, input_ptr + segment_size, grad_input_ptr,
              grad_input_ptr + segment_size);
          const bool use_vector = align == Alignment::SAME_ALIGNED;
          const size_t num_vectors =
              use_vector ? get_num_aligned_elements(input_ptr, segment_size, nvec, sizeof(InputT))
                         : segment_size;
          if (grad_act_scales == nullptr) {
            const int blocks = launch_blocks(rows * num_segments * num_vectors);
            if (use_vector) {
              scaled_gated_backward_kernel<nvec, true, GradT, InputT, ScaleT, OutputT, Act>
                  <<<blocks, kThreads, 0, stream>>>(grad_ptr, input_ptr, scale_ptr, grad_input_ptr,
                                                    rows, hidden, segment_size, num_segments,
                                                    num_vectors, param);
            } else {
              scaled_gated_backward_kernel<1, true, GradT, InputT, ScaleT, OutputT, Act>
                  <<<blocks, kThreads, 0, stream>>>(grad_ptr, input_ptr, scale_ptr, grad_input_ptr,
                                                    rows, hidden, segment_size, num_segments,
                                                    segment_size, param);
            }
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              auto grad_act_scales_ptr =
                  reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr);
              if (use_vector) {
                scaled_gated_backward_with_scale_grad_kernel<
                    nvec, true, GradT, InputT, ScaleT, OutputT, GradScaleT, Act>
                    <<<static_cast<int>(rows), kThreads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, segment_size, num_segments, num_vectors, param);
              } else {
                scaled_gated_backward_with_scale_grad_kernel<
                    1, true, GradT, InputT, ScaleT, OutputT, GradScaleT, Act>
                    <<<static_cast<int>(rows), kThreads, 0, stream>>>(
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

void launch_scaled_srelu_backward(const NVTETensor nvte_grad_output, const NVTETensor nvte_input,
                                  const NVTETensor nvte_act_scales, NVTETensor nvte_grad_input,
                                  NVTETensor nvte_grad_act_scales, cudaStream_t stream,
                                  const char *api_name) {
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
          constexpr int nvec = sizeof(GradT) == sizeof(InputT) &&
                                       sizeof(InputT) == sizeof(OutputT)
                                   ? vector_width<GradT>()
                                   : 1;
          const auto grad_ptr = reinterpret_cast<const GradT *>(grad_output->data.dptr);
          const auto input_ptr = reinterpret_cast<const InputT *>(input->data.dptr);
          const auto scale_ptr = reinterpret_cast<const ScaleT *>(act_scales->data.dptr);
          auto grad_input_ptr = reinterpret_cast<OutputT *>(grad_input->data.dptr);
          if (grad_act_scales == nullptr) {
            const size_t total = rows * hidden;
            const auto align = CheckAlignment(total, nvec, grad_ptr, input_ptr, grad_input_ptr);
            const bool use_vector = align == Alignment::SAME_ALIGNED;
            const size_t num_vectors =
                use_vector ? get_num_aligned_elements(input_ptr, total, nvec, sizeof(InputT))
                           : total;
            const int blocks = launch_blocks(num_vectors);
            if (use_vector) {
              scaled_srelu_backward_kernel<nvec, true, GradT, InputT, ScaleT, OutputT>
                  <<<blocks, kThreads, 0, stream>>>(grad_ptr, input_ptr, scale_ptr, grad_input_ptr,
                                                    total, hidden, num_vectors);
            } else {
              scaled_srelu_backward_kernel<1, true, GradT, InputT, ScaleT, OutputT>
                  <<<blocks, kThreads, 0, stream>>>(grad_ptr, input_ptr, scale_ptr, grad_input_ptr,
                                                    total, hidden, total);
            }
          } else {
            const auto align = row_vector_alignment(hidden, nvec, grad_ptr, input_ptr,
                                                    grad_input_ptr);
            const bool use_vector = align == Alignment::SAME_ALIGNED;
            const size_t num_vectors =
                use_vector ? get_num_aligned_elements(input_ptr, hidden, nvec, sizeof(InputT))
                           : hidden;
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              auto grad_act_scales_ptr =
                  reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr);
              if (use_vector) {
                scaled_srelu_backward_with_scale_grad_kernel<
                    nvec, true, GradT, InputT, ScaleT, OutputT, GradScaleT>
                    <<<static_cast<int>(rows), kThreads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, num_vectors);
              } else {
                scaled_srelu_backward_with_scale_grad_kernel<
                    1, true, GradT, InputT, ScaleT, OutputT, GradScaleT>
                    <<<static_cast<int>(rows), kThreads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, hidden);
              }
            });
          }
        });
      });
    });
  });
  NVTE_CHECK_CUDA(cudaGetLastError());
}

}  // namespace
}  // namespace transformer_engine

void nvte_scaled_swiglu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                        int64_t glu_interleave_size, cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_swiglu);
  using namespace transformer_engine;
  Empty empty = {};
  (void)empty;
  ClampedSwiGLUParam param = {};
  launch_scaled_gated_forward<ScaledActivation::kSwiGLU>(
      input, act_scales, output, glu_interleave_size, param, stream, "nvte_scaled_swiglu");
}

void nvte_scaled_dswiglu(const NVTETensor grad, const NVTETensor input,
                         const NVTETensor act_scales, NVTETensor grad_input,
                         NVTETensor grad_act_scales, int64_t glu_interleave_size,
                         cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_dswiglu);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {};
  launch_scaled_gated_backward<ScaledActivation::kSwiGLU>(
      grad, input, act_scales, grad_input, grad_act_scales, glu_interleave_size, param, stream,
      "nvte_scaled_dswiglu");
}

void nvte_scaled_clamped_swiglu(const NVTETensor input, const NVTETensor act_scales,
                                NVTETensor output, float limit, float alpha,
                                float glu_linear_offset, int64_t glu_interleave_size,
                                cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_clamped_swiglu);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  launch_scaled_gated_forward<ScaledActivation::kClampedSwiGLU>(
      input, act_scales, output, glu_interleave_size, param, stream,
      "nvte_scaled_clamped_swiglu");
}

void nvte_scaled_clamped_dswiglu(const NVTETensor grad, const NVTETensor input,
                                 const NVTETensor act_scales, NVTETensor grad_input,
                                 NVTETensor grad_act_scales, float limit, float alpha,
                                 float glu_linear_offset, int64_t glu_interleave_size,
                                 cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_clamped_dswiglu);
  using namespace transformer_engine;
  ClampedSwiGLUParam param = {limit, alpha, glu_linear_offset};
  launch_scaled_gated_backward<ScaledActivation::kClampedSwiGLU>(
      grad, input, act_scales, grad_input, grad_act_scales, glu_interleave_size, param, stream,
      "nvte_scaled_clamped_dswiglu");
}

void nvte_scaled_srelu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_srelu);
  using namespace transformer_engine;
  launch_scaled_srelu_forward(input, act_scales, output, stream, "nvte_scaled_srelu");
}

void nvte_scaled_dsrelu(const NVTETensor grad, const NVTETensor input,
                        const NVTETensor act_scales, NVTETensor grad_input,
                        NVTETensor grad_act_scales, cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_dsrelu);
  using namespace transformer_engine;
  launch_scaled_srelu_backward(grad, input, act_scales, grad_input, grad_act_scales, stream,
                               "nvte_scaled_dsrelu");
}
