/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/activation.h>

#include <algorithm>

#include "../common.h"
#include "../util/math.h"
#include "./scaled_activation.h"

namespace transformer_engine {
namespace {

using namespace detail::scaled_activation;

template <int nvec, typename InputT, typename ScaleT, typename OutputT>
__global__ void __launch_bounds__(kThreads, 4) scaled_srelu_forward_kernel(
    const InputT *__restrict__ input, const ScaleT *__restrict__ act_scales,
    OutputT *__restrict__ output, const size_t rows, const size_t hidden,
    const size_t num_vectors_per_row) {
  Empty empty = {};
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
      const float unscaled =
          srelu<float, float>(static_cast<float>(input_loader.separate()[lane]), empty);
      output_storer.separate()[lane] = static_cast<OutputT>(unscaled * scale);
    }
    output_storer.store(vector_idx, hidden);
  }
}

template <int nvec, typename GradT, typename InputT, typename ScaleT, typename OutputT>
__global__ void __launch_bounds__(kThreads, 4) scaled_srelu_backward_kernel(
    const GradT *__restrict__ grad_output, const InputT *__restrict__ input,
    const ScaleT *__restrict__ act_scales, OutputT *__restrict__ grad_input, const size_t rows,
    const size_t hidden, const size_t num_vectors_per_row) {
  Empty empty = {};
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
          grad * dsrelu<float, float>(static_cast<float>(input_loader.separate()[lane]), empty));
    }
    grad_input_storer.store(vector_idx, hidden);
  }
}

template <int nvec, typename GradT, typename InputT, typename ScaleT, typename OutputT,
          typename GradScaleT>
__global__ void __launch_bounds__(kReductionThreads, 4) scaled_srelu_backward_with_scale_grad_kernel(
    const GradT *__restrict__ grad_output, const InputT *__restrict__ input,
    const ScaleT *__restrict__ act_scales, OutputT *__restrict__ grad_input,
    GradScaleT *__restrict__ grad_act_scales, const size_t rows, const size_t hidden,
    const size_t num_vectors_per_row) {
  __shared__ float smem[kReductionWarps];
  const size_t row = blockIdx.x;
  (void)rows;
  float scale_grad = 0.0f;
  Empty empty = {};
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
      const float unscaled =
          srelu<float, float>(static_cast<float>(input_loader.separate()[lane]), empty);
      const float grad = static_cast<float>(grad_loader.separate()[lane]);
      scale_grad += grad * unscaled;

      const float scaled_grad = grad * scale;
      const float dact =
          dsrelu<float, float>(static_cast<float>(input_loader.separate()[lane]), empty);
      grad_input_storer.separate()[lane] = static_cast<OutputT>(scaled_grad * dact);
    }
    grad_input_storer.store(vector_idx, hidden);
  }

  scale_grad = block_reduce_sum(scale_grad, smem);
  if (threadIdx.x == 0) {
    grad_act_scales[row] = static_cast<GradScaleT>(scale_grad);
  }
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
        // Same element nvec for input/output; each uses its own vector type.
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
          scaled_srelu_forward_kernel<nvec, InputT, ScaleT, OutputT>
              <<<blocks, kThreads, 0, stream>>>(input_ptr, scale_ptr, output_ptr, rows, hidden,
                                                num_vectors);
        } else {
          scaled_srelu_forward_kernel<1, InputT, ScaleT, OutputT><<<blocks, kThreads, 0, stream>>>(
              input_ptr, scale_ptr, output_ptr, rows, hidden, hidden);
        }
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
          // Same element nvec across Grad/Input/Output; size by widest element.
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
              scaled_srelu_backward_kernel<nvec, GradT, InputT, ScaleT, OutputT>
                  <<<blocks, kThreads, 0, stream>>>(grad_ptr, input_ptr, scale_ptr, grad_input_ptr,
                                                    rows, hidden, num_vectors);
            } else {
              scaled_srelu_backward_kernel<1, GradT, InputT, ScaleT, OutputT>
                  <<<blocks, kThreads, 0, stream>>>(grad_ptr, input_ptr, scale_ptr, grad_input_ptr,
                                                    rows, hidden, hidden);
            }
          } else {
            TRANSFORMER_ENGINE_TYPE_SWITCH_NON_FP8ONLY(grad_act_scales->data.dtype, GradScaleT, {
              auto grad_act_scales_ptr = reinterpret_cast<GradScaleT *>(grad_act_scales->data.dptr);
              if (use_vector) {
                scaled_srelu_backward_with_scale_grad_kernel<nvec, GradT, InputT, ScaleT, OutputT,
                                                             GradScaleT>
                    <<<static_cast<int>(rows), kReductionThreads, 0, stream>>>(
                        grad_ptr, input_ptr, scale_ptr, grad_input_ptr, grad_act_scales_ptr, rows,
                        hidden, num_vectors);
              } else {
                scaled_srelu_backward_with_scale_grad_kernel<1, GradT, InputT, ScaleT, OutputT,
                                                             GradScaleT>
                    <<<static_cast<int>(rows), kReductionThreads, 0, stream>>>(
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

void nvte_scaled_srelu(const NVTETensor input, const NVTETensor act_scales, NVTETensor output,
                       cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_srelu);
  using namespace transformer_engine;
  launch_scaled_srelu_forward(input, act_scales, output, stream, "nvte_scaled_srelu");
}

void nvte_scaled_dsrelu(const NVTETensor grad, const NVTETensor input, const NVTETensor act_scales,
                        NVTETensor grad_input, NVTETensor grad_act_scales, cudaStream_t stream) {
  NVTE_API_CALL(nvte_scaled_dsrelu);
  using namespace transformer_engine;
  launch_scaled_srelu_backward(grad, input, act_scales, grad_input, grad_act_scales, stream,
                               "nvte_scaled_dsrelu");
}
