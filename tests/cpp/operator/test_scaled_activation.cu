/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <tuple>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>

#include "../test_common.h"

using namespace transformer_engine;

namespace {

enum class ScaledActivationCase {
  kSwiGLU,
  kClampedSwiGLU,
  kSReLU,
};

constexpr float kClampedLimit = 1.3f;
constexpr float kClampedAlpha = 1.702f;
constexpr float kClampedLinearOffset = 0.5f;

const char *activation_name(ScaledActivationCase activation) {
  switch (activation) {
    case ScaledActivationCase::kSwiGLU:
      return "scaled_swiglu";
    case ScaledActivationCase::kClampedSwiGLU:
      return "scaled_clamped_swiglu";
    case ScaledActivationCase::kSReLU:
      return "scaled_srelu";
  }
  return "unknown";
}

inline float sigmoid(const float x) { return 1.0f / (1.0f + expf(-x)); }

inline float qgelu_alpha(const float x, const float alpha) { return x * sigmoid(alpha * x); }

inline float dqgelu_alpha(const float x, const float alpha) {
  const float sig = sigmoid(alpha * x);
  return alpha * x * sig * (1.0f - sig) + sig;
}

inline float silu_ref(const float x) { return x * sigmoid(x); }

inline float dsilu_ref(const float x) {
  const float sig = sigmoid(x);
  return x * sig * (1.0f - sig) + sig;
}

inline float srelu_ref(const float x) { return x > 0.0f ? x * x : 0.0f; }

inline float dsrelu_ref(const float x) { return fmaxf(0.0f, 2.0f * x); }

inline void glu_indices(const size_t row, const size_t col, const size_t hidden,
                        const int64_t interleave, size_t *act_idx, size_t *linear_idx) {
  if (interleave > 0) {
    const size_t block = col / static_cast<size_t>(interleave);
    const size_t lane = col % static_cast<size_t>(interleave);
    const size_t base = row * hidden * 2 + block * static_cast<size_t>(interleave) * 2 + lane;
    *act_idx = base;
    *linear_idx = base + static_cast<size_t>(interleave);
  } else {
    const size_t base = row * hidden * 2;
    *act_idx = base + col;
    *linear_idx = base + hidden + col;
  }
}

inline float gated_unscaled(const ScaledActivationCase activation, const float act_in,
                            const float linear_in) {
  switch (activation) {
    case ScaledActivationCase::kSwiGLU:
      return silu_ref(act_in) * linear_in;
    case ScaledActivationCase::kClampedSwiGLU: {
      const float act = qgelu_alpha(fminf(kClampedLimit, act_in), kClampedAlpha);
      const float linear =
          fminf(fmaxf(-kClampedLimit, linear_in), kClampedLimit) + kClampedLinearOffset;
      return act * linear;
    }
    case ScaledActivationCase::kSReLU:
      return srelu_ref(act_in);
  }
  return 0.0f;
}

inline void gated_grads(const ScaledActivationCase activation, const float act_in,
                        const float linear_in, float *dact, float *dlinear, float *unscaled) {
  switch (activation) {
    case ScaledActivationCase::kSwiGLU: {
      const float act = silu_ref(act_in);
      *unscaled = act * linear_in;
      *dact = dsilu_ref(act_in) * linear_in;
      *dlinear = act;
      return;
    }
    case ScaledActivationCase::kClampedSwiGLU: {
      const bool dlinear_mask = linear_in <= kClampedLimit && linear_in >= -kClampedLimit;
      const float act = qgelu_alpha(fminf(kClampedLimit, act_in), kClampedAlpha);
      const float dact_base =
          act_in <= kClampedLimit ? dqgelu_alpha(fminf(kClampedLimit, act_in), kClampedAlpha)
                                  : 0.0f;
      const float linear =
          fminf(fmaxf(-kClampedLimit, linear_in), kClampedLimit) + kClampedLinearOffset;
      *unscaled = act * linear;
      *dact = dact_base * linear;
      *dlinear = dlinear_mask ? act : 0.0f;
      return;
    }
    case ScaledActivationCase::kSReLU:
      *unscaled = srelu_ref(act_in);
      *dact = dsrelu_ref(act_in);
      *dlinear = 0.0f;
      return;
  }
}

template <typename DataT, typename ScaleT>
void compute_reference(ScaledActivationCase activation, const DataT *input, const ScaleT *scales,
                       const DataT *grad_output, DataT *output, DataT *grad_input,
                       DataT *grad_scales, const size_t rows, const size_t hidden,
                       const int64_t interleave, const bool compute_grad_scales) {
  const bool is_gated = activation != ScaledActivationCase::kSReLU;
  const size_t input_cols = is_gated ? hidden * 2 : hidden;
  std::fill(grad_input, grad_input + rows * input_cols, static_cast<DataT>(0.0f));

  for (size_t row = 0; row < rows; ++row) {
    const float scale = static_cast<float>(scales[row]);
    float scale_grad = 0.0f;
    for (size_t col = 0; col < hidden; ++col) {
      const size_t out_idx = row * hidden + col;
      float unscaled = 0.0f;
      float dact = 0.0f;
      float dlinear = 0.0f;
      if (is_gated) {
        size_t act_idx = 0;
        size_t linear_idx = 0;
        glu_indices(row, col, hidden, interleave, &act_idx, &linear_idx);
        const float act_in = static_cast<float>(input[act_idx]);
        const float linear_in = static_cast<float>(input[linear_idx]);
        unscaled = gated_unscaled(activation, act_in, linear_in);
        gated_grads(activation, act_in, linear_in, &dact, &dlinear, &unscaled);

        const float scaled_grad = static_cast<float>(grad_output[out_idx]) * scale;
        grad_input[act_idx] = static_cast<DataT>(scaled_grad * dact);
        grad_input[linear_idx] = static_cast<DataT>(scaled_grad * dlinear);
      } else {
        const float x = static_cast<float>(input[out_idx]);
        unscaled = srelu_ref(x);
        const float scaled_grad = static_cast<float>(grad_output[out_idx]) * scale;
        grad_input[out_idx] = static_cast<DataT>(scaled_grad * dsrelu_ref(x));
      }

      output[out_idx] = static_cast<DataT>(unscaled * scale);
      scale_grad += static_cast<float>(grad_output[out_idx]) * unscaled;
    }
    if (compute_grad_scales) {
      grad_scales[row] = static_cast<DataT>(scale_grad);
    }
  }
}

template <typename DataT, typename ScaleT>
void run_scaled_activation_test(ScaledActivationCase activation, const size_t rows,
                                const size_t hidden, const int64_t interleave,
                                const bool compute_grad_scales) {
  using namespace test;
  const DType data_type = TypeInfo<DataT>::dtype;
  const DType scale_type = TypeInfo<ScaleT>::dtype;
  const bool is_gated = activation != ScaledActivationCase::kSReLU;
  const size_t input_cols = is_gated ? hidden * 2 : hidden;

  Tensor input("input", std::vector<size_t>{rows, input_cols}, data_type);
  Tensor scales("act_scales", std::vector<size_t>{rows}, scale_type);
  Tensor output("output", std::vector<size_t>{rows, hidden}, data_type);
  Tensor grad_output("grad_output", std::vector<size_t>{rows, hidden}, data_type);
  Tensor grad_input("grad_input", std::vector<size_t>{rows, input_cols}, data_type);
  Tensor grad_scales("grad_scales", std::vector<size_t>{rows}, data_type);

  fillUniform(&input);
  fillUniform(&scales);
  fillUniform(&grad_output);

  std::unique_ptr<DataT[]> ref_output = std::make_unique<DataT[]>(rows * hidden);
  std::unique_ptr<DataT[]> ref_grad_input = std::make_unique<DataT[]>(rows * input_cols);
  std::unique_ptr<DataT[]> ref_grad_scales = std::make_unique<DataT[]>(rows);

  compute_reference(activation, input.rowwise_cpu_dptr<DataT>(), scales.rowwise_cpu_dptr<ScaleT>(),
                    grad_output.rowwise_cpu_dptr<DataT>(), ref_output.get(),
                    ref_grad_input.get(), ref_grad_scales.get(), rows, hidden, interleave,
                    compute_grad_scales);

  switch (activation) {
    case ScaledActivationCase::kSwiGLU:
      nvte_scaled_swiglu(input.data(), scales.data(), output.data(), interleave, 0);
      nvte_scaled_dswiglu(grad_output.data(), input.data(), scales.data(), grad_input.data(),
                          compute_grad_scales ? grad_scales.data() : nullptr, interleave, 0);
      break;
    case ScaledActivationCase::kClampedSwiGLU:
      nvte_scaled_clamped_swiglu(input.data(), scales.data(), output.data(), kClampedLimit,
                                 kClampedAlpha, kClampedLinearOffset, interleave, 0);
      nvte_scaled_clamped_dswiglu(
          grad_output.data(), input.data(), scales.data(), grad_input.data(),
          compute_grad_scales ? grad_scales.data() : nullptr, kClampedLimit, kClampedAlpha,
          kClampedLinearOffset, interleave, 0);
      break;
    case ScaledActivationCase::kSReLU:
      nvte_scaled_srelu(input.data(), scales.data(), output.data(), 0);
      nvte_scaled_dsrelu(grad_output.data(), input.data(), scales.data(), grad_input.data(),
                         compute_grad_scales ? grad_scales.data() : nullptr, 0);
      break;
  }

  NVTE_CHECK_CUDA(cudaDeviceSynchronize());
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  auto [atol, rtol] = getTolerances(data_type);
  if (data_type == DType::kFloat32) {
    atol = 5e-5;
    rtol = 5e-5;
  }
  compareResults("scaled_activation_output", output, ref_output.get(), atol, rtol);
  compareResults("scaled_activation_grad_input", grad_input, ref_grad_input.get(), atol, rtol);
  if (compute_grad_scales) {
    compareResults("scaled_activation_grad_scales", grad_scales, ref_grad_scales.get(), atol, rtol);
  }
}

class ScaledActivationTest
    : public ::testing::TestWithParam<
          std::tuple<ScaledActivationCase, DType, DType, std::pair<size_t, size_t>, int64_t,
                     bool>> {
};

std::string test_name_generator(
    const testing::TestParamInfo<ScaledActivationTest::ParamType> &info) {
  const auto activation = std::get<0>(info.param);
  const auto data_type = std::get<1>(info.param);
  const auto scale_type = std::get<2>(info.param);
  const auto shape = std::get<3>(info.param);
  const auto interleave = std::get<4>(info.param);
  const auto compute_grad_scales = std::get<5>(info.param);
  return std::string(activation_name(activation)) + "_data_" + test::typeName(data_type) +
         "_scale_" + test::typeName(scale_type) + "_m_" + std::to_string(shape.first) + "_h_" +
         std::to_string(shape.second) + "_interleave_" + std::to_string(interleave) +
         (compute_grad_scales ? "_with_scale_grad" : "_no_scale_grad");
}

}  // namespace

TEST_P(ScaledActivationTest, ForwardBackward) {
  const auto activation = std::get<0>(GetParam());
  const auto data_type = std::get<1>(GetParam());
  const auto scale_type = std::get<2>(GetParam());
  const auto shape = std::get<3>(GetParam());
  const auto interleave = std::get<4>(GetParam());
  const auto compute_grad_scales = std::get<5>(GetParam());

  if (activation == ScaledActivationCase::kSReLU && interleave != 0) {
    GTEST_SKIP() << "SReLU is not a GLU activation.";
  }
  if (activation != ScaledActivationCase::kSReLU && interleave > 0 &&
      shape.second % static_cast<size_t>(interleave) != 0) {
    GTEST_SKIP() << "Hidden size must be divisible by GLU interleave.";
  }

  using namespace test;
  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(data_type, DataT, {
    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(scale_type, ScaleT, {
      run_scaled_activation_test<DataT, ScaleT>(activation, shape.first, shape.second, interleave,
                                                compute_grad_scales);
    });
  });
}

// Test axes (the six tuple elements consumed by ScaledActivationTest):
//   1. Activation         : SwiGLU and ClampedSwiGLU are gated (input is [M, 2H]);
//                           SReLU is unary (input is [M, H], no gate split).
//   2. Data dtype         : dtype of the activation input/output tensors.
//   3. Scale dtype        : dtype of act_scales / grad_act_scales.
//   4. Shape {rows, hidden}: rows = M (tokens), hidden = H (output width; gated input is 2H).
//   5. GLU interleave      : 0 = contiguous [a | b]; 32 = interleaved a/b blocks. Only valid
//                            for gated activations with hidden % 32 == 0; SReLU skips != 0.
//   6. compute_grad_scales : whether the backward also reduces grad_act_scales.

// Interleave is swept over {0, 32}; invalid combinations -- SReLU with any nonzero interleave, or
// a gated activation whose hidden is not divisible by the interleave -- are skipped at runtime by
// the GTEST_SKIP guards in the test body.
INSTANTIATE_TEST_SUITE_P(
    OperatorTest_ScaledActivation, ScaledActivationTest,
    ::testing::Combine(
        ::testing::Values(ScaledActivationCase::kSwiGLU, ScaledActivationCase::kClampedSwiGLU,
                          ScaledActivationCase::kSReLU),
        ::testing::Values(DType::kFloat32, DType::kBFloat16),   // data dtype
        ::testing::Values(DType::kFloat32, DType::kBFloat16),   // scale dtype
        ::testing::Values(std::pair<size_t, size_t>{17, 64},    // odd rows, aligned hidden
                          std::pair<size_t, size_t>{32, 32},    // minimal aligned square
                          std::pair<size_t, size_t>{128, 128},  // square
                          std::pair<size_t, size_t>{256, 64},   // many rows, narrow hidden
                          std::pair<size_t, size_t>{1024, 2048},  // large FFN-ish width
                          std::pair<size_t, size_t>{1, 1},      // single element
                          std::pair<size_t, size_t>{1, 96},     // single row
                          std::pair<size_t, size_t>{96, 1},     // single hidden column
                          std::pair<size_t, size_t>{13, 100}),  // non-power-of-two
        ::testing::Values(0, 32),                                // contiguous + interleaved
        ::testing::Values(false, true)),                         // grad_act_scales off / on
    test_name_generator);
