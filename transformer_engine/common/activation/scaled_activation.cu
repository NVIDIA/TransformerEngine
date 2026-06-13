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
 *   1 | scaled_gated_forward_kernel                   | SwiGLU / ClampedSwiGLU | fwd | --              | flat element grid
 *   2 | scaled_srelu_forward_kernel                   | SReLU (unary)          | fwd | --              | flat element grid
 *   3 | scaled_gated_backward_kernel                  | SwiGLU / ClampedSwiGLU | bwd | no              | flat element grid
 *   4 | scaled_srelu_backward_kernel                  | SReLU                  | bwd | no              | flat element grid
 *   5 | scaled_gated_backward_with_scale_grad_kernel  | SwiGLU / ClampedSwiGLU | bwd | yes             | one block per row
 *   6 | scaled_srelu_backward_with_scale_grad_kernel  | SReLU                  | bwd | yes             | one block per row
 *
 * The "with scale grad" variants compute grad_act_scales[row] = sum_j dY * unscaled,
 * a per-row reduction that requires the one-block-per-row launch; when
 * grad_act_scales is null the cheaper flat element-wise grid is used instead.
 */

#include <transformer_engine/activation.h>

#include <algorithm>

#include "../common.h"
#include "../util/math.h"

namespace transformer_engine {
namespace {

enum class ScaledActivation {
  kSwiGLU,
  kClampedSwiGLU,
  kSReLU,
};

__device__ __forceinline__ void glu_input_indices(const size_t row, const size_t col,
                                                  const size_t hidden,
                                                  const int64_t glu_interleave_size,
                                                  size_t *act_idx, size_t *linear_idx) {
  if (glu_interleave_size > 0) {
    const size_t interleave = static_cast<size_t>(glu_interleave_size);
    const size_t block = col / interleave;
    const size_t lane = col % interleave;
    const size_t base = row * hidden * 2 + block * interleave * 2 + lane;
    *act_idx = base;
    *linear_idx = base + interleave;
  } else {
    const size_t base = row * hidden * 2;
    *act_idx = base + col;
    *linear_idx = base + hidden + col;
  }
}

template <ScaledActivation Act>
__device__ __forceinline__ float gated_forward_value(const float act_in, const float linear_in,
                                                     const ClampedSwiGLUParam &param) {
  if constexpr (Act == ScaledActivation::kSwiGLU) {
    Empty empty = {};
    return silu<float, float>(act_in, empty) * linear_in;
  } else {
    const float linear =
        fminf(fmaxf(-param.limit, linear_in), param.limit) + param.glu_linear_offset;
    return clamped_silu<float, float>(act_in, param) * linear;
  }
}

template <ScaledActivation Act>
__device__ __forceinline__ void gated_backward_values(const float act_in, const float linear_in,
                                                      const ClampedSwiGLUParam &param,
                                                      float *dact, float *dlinear,
                                                      float *unscaled) {
  if constexpr (Act == ScaledActivation::kSwiGLU) {
    Empty empty = {};
    const float act = silu<float, float>(act_in, empty);
    *unscaled = act * linear_in;
    *dact = dsilu<float, float>(act_in, empty) * linear_in;
    *dlinear = act;
  } else {
    const bool dlinear_mask = linear_in <= param.limit && linear_in >= -param.limit;
    const float linear =
        fminf(fmaxf(-param.limit, linear_in), param.limit) + param.glu_linear_offset;
    const float act = clamped_silu<float, float>(act_in, param);
    *unscaled = act * linear;
    *dact = clamped_dsilu<float, float>(act_in, param) * linear;
    *dlinear = dlinear_mask ? act : 0.0f;
  }
}

template <typename InputT, typename ScaleT, typename OutputT, ScaledActivation Act>
__global__ void scaled_gated_forward_kernel(const InputT *input, const ScaleT *act_scales,
                                            OutputT *output, const size_t rows,
                                            const size_t hidden,
                                            const int64_t glu_interleave_size,
                                            const ClampedSwiGLUParam param) {
  const size_t total = rows * hidden;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const size_t row = idx / hidden;
    const size_t col = idx % hidden;
    size_t act_idx = 0;
    size_t linear_idx = 0;
    glu_input_indices(row, col, hidden, glu_interleave_size, &act_idx, &linear_idx);

    const float unscaled = gated_forward_value<Act>(static_cast<float>(input[act_idx]),
                                                    static_cast<float>(input[linear_idx]), param);
    const float scale = static_cast<float>(act_scales[row]);
    output[idx] = static_cast<OutputT>(unscaled * scale);
  }
}

template <typename InputT, typename ScaleT, typename OutputT>
__global__ void scaled_srelu_forward_kernel(const InputT *input, const ScaleT *act_scales,
                                            OutputT *output, const size_t rows,
                                            const size_t hidden) {
  const size_t total = rows * hidden;
  Empty empty = {};
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const size_t row = idx / hidden;
    const float unscaled = srelu<float, float>(static_cast<float>(input[idx]), empty);
    const float scale = static_cast<float>(act_scales[row]);
    output[idx] = static_cast<OutputT>(unscaled * scale);
  }
}

template <typename GradT, typename InputT, typename ScaleT, typename OutputT,
          ScaledActivation Act>
__global__ void scaled_gated_backward_kernel(const GradT *grad_output, const InputT *input,
                                             const ScaleT *act_scales, OutputT *grad_input,
                                             const size_t rows, const size_t hidden,
                                             const int64_t glu_interleave_size,
                                             const ClampedSwiGLUParam param) {
  const size_t total = rows * hidden;
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const size_t row = idx / hidden;
    const size_t col = idx % hidden;
    size_t act_idx = 0;
    size_t linear_idx = 0;
    glu_input_indices(row, col, hidden, glu_interleave_size, &act_idx, &linear_idx);

    float dact = 0.0f;
    float dlinear = 0.0f;
    float unscaled = 0.0f;
    gated_backward_values<Act>(static_cast<float>(input[act_idx]),
                               static_cast<float>(input[linear_idx]), param, &dact, &dlinear,
                               &unscaled);
    (void)unscaled;
    const float scale = static_cast<float>(act_scales[row]);
    const float grad = static_cast<float>(grad_output[idx]) * scale;
    grad_input[act_idx] = static_cast<OutputT>(grad * dact);
    grad_input[linear_idx] = static_cast<OutputT>(grad * dlinear);
  }
}

template <typename GradT, typename InputT, typename ScaleT, typename OutputT>
__global__ void scaled_srelu_backward_kernel(const GradT *grad_output, const InputT *input,
                                             const ScaleT *act_scales, OutputT *grad_input,
                                             const size_t rows, const size_t hidden) {
  const size_t total = rows * hidden;
  Empty empty = {};
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += gridDim.x * blockDim.x) {
    const size_t row = idx / hidden;
    const float scale = static_cast<float>(act_scales[row]);
    const float grad = static_cast<float>(grad_output[idx]) * scale;
    grad_input[idx] =
        static_cast<OutputT>(grad * dsrelu<float, float>(static_cast<float>(input[idx]), empty));
  }
}

template <typename GradT, typename InputT, typename ScaleT, typename OutputT,
          typename GradScaleT, ScaledActivation Act>
__global__ void scaled_gated_backward_with_scale_grad_kernel(
    const GradT *grad_output, const InputT *input, const ScaleT *act_scales, OutputT *grad_input,
    GradScaleT *grad_act_scales, const size_t rows, const size_t hidden,
    const int64_t glu_interleave_size, const ClampedSwiGLUParam param) {
  constexpr int kThreads = 256;
  __shared__ float smem[kThreads];
  const size_t row = blockIdx.x;
  float scale_grad = 0.0f;

  for (size_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const size_t grad_idx = row * hidden + col;
    size_t act_idx = 0;
    size_t linear_idx = 0;
    glu_input_indices(row, col, hidden, glu_interleave_size, &act_idx, &linear_idx);

    float dact = 0.0f;
    float dlinear = 0.0f;
    float unscaled = 0.0f;
    gated_backward_values<Act>(static_cast<float>(input[act_idx]),
                               static_cast<float>(input[linear_idx]), param, &dact, &dlinear,
                               &unscaled);
    const float grad = static_cast<float>(grad_output[grad_idx]);
    scale_grad += grad * unscaled;

    const float scale = static_cast<float>(act_scales[row]);
    const float scaled_grad = grad * scale;
    grad_input[act_idx] = static_cast<OutputT>(scaled_grad * dact);
    grad_input[linear_idx] = static_cast<OutputT>(scaled_grad * dlinear);
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

template <typename GradT, typename InputT, typename ScaleT, typename OutputT, typename GradScaleT>
__global__ void scaled_srelu_backward_with_scale_grad_kernel(
    const GradT *grad_output, const InputT *input, const ScaleT *act_scales, OutputT *grad_input,
    GradScaleT *grad_act_scales, const size_t rows, const size_t hidden) {
  constexpr int kThreads = 256;
  __shared__ float smem[kThreads];
  const size_t row = blockIdx.x;
  float scale_grad = 0.0f;
  Empty empty = {};

  for (size_t col = threadIdx.x; col < hidden; col += blockDim.x) {
    const size_t idx = row * hidden + col;
    const float unscaled = srelu<float, float>(static_cast<float>(input[idx]), empty);
    const float grad = static_cast<float>(grad_output[idx]);
    scale_grad += grad * unscaled;

    const float scale = static_cast<float>(act_scales[row]);
    const float scaled_grad = grad * scale;
    const float dact = dsrelu<float, float>(static_cast<float>(input[idx]), empty);
    grad_input[idx] = static_cast<OutputT>(scaled_grad * dact);
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

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<size_t>(DIVUP(rows * hidden, static_cast<size_t>(threads)), 65535));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(output->data.dtype, OutputT, {
        scaled_gated_forward_kernel<InputT, ScaleT, OutputT, Act>
            <<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const InputT *>(input->data.dptr),
                reinterpret_cast<const ScaleT *>(act_scales->data.dptr),
                reinterpret_cast<OutputT *>(output->data.dptr), rows, hidden, glu_interleave_size,
                param);
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

  constexpr int threads = 256;
  const int blocks = static_cast<int>(std::min<size_t>(DIVUP(rows * hidden, static_cast<size_t>(threads)), 65535));
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(output->data.dtype, OutputT, {
        scaled_srelu_forward_kernel<InputT, ScaleT, OutputT>
            <<<blocks, threads, 0, stream>>>(
                reinterpret_cast<const InputT *>(input->data.dptr),
                reinterpret_cast<const ScaleT *>(act_scales->data.dptr),
                reinterpret_cast<OutputT *>(output->data.dptr), rows, hidden);
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

  constexpr int threads = 256;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_output->data.dtype, GradT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_input->data.dtype, OutputT, {
          if (grad_act_scales == nullptr) {
            const int blocks =
                static_cast<int>(std::min<size_t>(DIVUP(rows * hidden, static_cast<size_t>(threads)), 65535));
            scaled_gated_backward_kernel<GradT, InputT, ScaleT, OutputT, Act>
                <<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const GradT *>(grad_output->data.dptr),
                    reinterpret_cast<const InputT *>(input->data.dptr),
                    reinterpret_cast<const ScaleT *>(act_scales->data.dptr),
                    reinterpret_cast<OutputT *>(grad_input->data.dptr), rows, hidden,
                    glu_interleave_size, param);
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              scaled_gated_backward_with_scale_grad_kernel<GradT, InputT, ScaleT, OutputT,
                                                           GradScaleT, Act>
                  <<<static_cast<int>(rows), threads, 0, stream>>>(
                      reinterpret_cast<const GradT *>(grad_output->data.dptr),
                      reinterpret_cast<const InputT *>(input->data.dptr),
                      reinterpret_cast<const ScaleT *>(act_scales->data.dptr),
                      reinterpret_cast<OutputT *>(grad_input->data.dptr),
                      reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr), rows, hidden,
                      glu_interleave_size, param);
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

  constexpr int threads = 256;
  TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_output->data.dtype, GradT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(input->data.dtype, InputT, {
      TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(act_scales->data.dtype, ScaleT, {
        TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_input->data.dtype, OutputT, {
          if (grad_act_scales == nullptr) {
            const int blocks =
                static_cast<int>(std::min<size_t>(DIVUP(rows * hidden, static_cast<size_t>(threads)), 65535));
            scaled_srelu_backward_kernel<GradT, InputT, ScaleT, OutputT>
                <<<blocks, threads, 0, stream>>>(
                    reinterpret_cast<const GradT *>(grad_output->data.dptr),
                    reinterpret_cast<const InputT *>(input->data.dptr),
                    reinterpret_cast<const ScaleT *>(act_scales->data.dptr),
                    reinterpret_cast<OutputT *>(grad_input->data.dptr), rows, hidden);
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              scaled_srelu_backward_with_scale_grad_kernel<GradT, InputT, ScaleT, OutputT,
                                                           GradScaleT>
                  <<<static_cast<int>(rows), threads, 0, stream>>>(
                      reinterpret_cast<const GradT *>(grad_output->data.dptr),
                      reinterpret_cast<const InputT *>(input->data.dptr),
                      reinterpret_cast<const ScaleT *>(act_scales->data.dptr),
                      reinterpret_cast<OutputT *>(grad_input->data.dptr),
                      reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr), rows, hidden);
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
