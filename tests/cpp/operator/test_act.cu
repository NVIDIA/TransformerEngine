/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cmath>
#include <cstring>
#include <memory>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <transformer_engine/activation.h>
#include <transformer_engine/cast.h>
#include "../test_common.h"

using namespace transformer_engine;

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_act_cast(const IT *input_h,
                          OT *output_h,
                          const CT scale,
                          CT *amax_h,
                          const size_t N,
                          const size_t H) {
  CT amax  = 0.;

  #pragma omp parallel for schedule(static) reduction(max: amax) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);
      elt = act(elt);
      output_h[i * H + j] = static_cast<OT>(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <float (*dact)(const float), typename IT, typename OT>
void compute_ref_dact_cast(const IT *input_h,
                           const IT *grad_h,
                           OT *output_h,
                           const size_t N,
                           const size_t H) {
  using CT = float;
  #pragma omp parallel for schedule(static) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT elt = static_cast<CT>(input_h[i * H + j]);
      elt = dact(elt);
      CT grad = static_cast<CT>(grad_h[i * H + j]);
      output_h[i * H + j] = static_cast<OT>(grad * elt);
    }
  }
}

template <float (*act)(const float), typename IT, typename OT, typename CT>
void compute_ref_glu_act_cast(const IT *input_h, OT *output_h, const CT scale, CT *amax_h,
                              const size_t N, const size_t H) {
  CT amax = 0.;

  const int col = H * 2;

  #pragma omp parallel for schedule(static) reduction(max: amax) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT gelu_elt = static_cast<CT>(input_h[i * col + j]);
      gelu_elt = act(gelu_elt);
      CT gate_elt = static_cast<CT>(input_h[i * col + H + j]);
      CT elt = gelu_elt * gate_elt;
      output_h[i * H + j] = static_cast<OT>(scale * elt);
      amax = std::abs(elt) > amax ? std::abs(elt) : amax;
    }
  }

  *amax_h = amax;
}

template <float (*dact)(const float), float (*act)(const float),
          typename IT, typename OT>
void compute_ref_dglu_act_cast(const IT *input_h, const IT *grad_h, OT *output_h,
                               const size_t N, const size_t H) {
  const int col = H * 2;
  using CT = float;

  #pragma omp parallel for schedule(static) proc_bind(spread)
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < H; j++) {
      CT grad = static_cast<CT>(grad_h[i * H + j]);
      CT gelu_elt = static_cast<CT>(input_h[i * col + j]);
      CT gate_elt = static_cast<CT>(input_h[i * col + H + j]);
      output_h[i * col + H + j] = static_cast<OT>(grad * act(gelu_elt));
      gelu_elt = dact(gelu_elt);
      CT elt = gelu_elt * gate_elt;
      output_h[i * col + j] = static_cast<OT>(grad * elt);
    }
  }
}


template <float (*ref_act)(const float),
          float (*ref_dact)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
          void (*nvte_dact)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTest(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input({ N, H }, itype);
  Tensor output({ N, H }, otype);
  Tensor igrad({ N, H }, itype);
  Tensor ograd({ N, H }, itype);

  fillUniform(&input);
  fillUniform(&ograd);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N*H);
  std::unique_ptr<IType[]> ref_igrad = std::make_unique<IType[]>(N*H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_act_cast<ref_act>(input.rowwise_cpu_dptr<IType>(), ref_output.get(),
                                output.scale(), &ref_amax, N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
    auto [atol_amax, rtol_amax] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol_amax, rtol_amax);
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_act", output, ref_output.get(), atol, rtol);

  nvte_dact(ograd.data(), input.data(), igrad.data(), 0);

  compute_ref_dact_cast<ref_dact>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                  ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);
  }
}

std::vector<size_t> getDBiasWorkspaceShape(size_t batch_size, size_t hidden_size, DType in_dtype, DType out_dtype) {
  auto input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto dact_input_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_shape = std::vector<size_t>{batch_size, hidden_size};
  auto output_trans_shape = std::vector<size_t>{hidden_size, batch_size};
  auto dbias_shape = std::vector<size_t>{hidden_size};

  // Evil hack to specify TE impl
  // Note: nvte_quantize_dbias_dgelu chooses its internal impl based
  // on what pointers are allocated, e.g. whether to output with
  // column-wise data. However, we don't have access to any allocated
  // buffers in this function. We pass a dummy pointer as a
  // workaround.
  int temp = 0;

  auto input_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), input_shape, in_dtype);
  auto dact_input_tensor =
      TensorWrapper(reinterpret_cast<void *>(&temp), dact_input_shape, in_dtype);
  auto output_tensor = TensorWrapper();
  output_tensor.set_rowwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_shape);
  output_tensor.set_columnwise_data(reinterpret_cast<void *>(&temp), out_dtype, output_trans_shape);
  auto dbias_tensor = TensorWrapper(reinterpret_cast<void *>(&temp), dbias_shape, in_dtype);

  TensorWrapper dummy_workspace;

  // For now, all dbias_dact(-s) have the same workspace size
  nvte_quantize_dbias_dgelu(input_tensor.data(), dact_input_tensor.data(), output_tensor.data(),
                            dbias_tensor.data(), dummy_workspace.data(), nullptr);

  auto work_shape = std::vector<size_t>(dummy_workspace.shape().data, dummy_workspace.shape().data + dummy_workspace.shape().ndim);
  return work_shape;
  // return pybind11::make_tuple(std::make_pair(work_shape, dummy_workspace.dtype()));
}

template <float (*ref_dact)(const float),
          void (*nvte_dact)(const NVTETensor, const NVTETensor,
                               NVTETensor, NVTETensor, NVTETensor,
                               cudaStream_t),
         typename IType, typename OType>
void performTestDActZeroGradInput(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  // const NVTETensor input, const NVTETensor activation_input,
  //  NVTETensor output, NVTETensor dbias, NVTETensor workspace,
  //  cudaStream_t stream

  Tensor input({ N, H }, itype);
  Tensor igrad({ N, H }, otype);
  Tensor ograd({ N, H }, itype);
  Tensor dbias({ H }, itype);
  auto workspace_shape = getDBiasWorkspaceShape(N, H, itype, otype);
  Tensor workspace(workspace_shape, DType::kFloat32);

  fillUniform(&input);
  // fillUniform(&ograd);
  igrad.set_scale(1.f);
  fillCase<IType>(&ograd, zeros);

  std::unique_ptr<OType[]> ref_igrad = std::make_unique<OType[]>(N*H);

  nvte_dact(ograd.data(), input.data(), igrad.data(), dbias.data(), workspace.data(), 0);

  compute_ref_dact_cast<ref_dact>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                  ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);

    // TODO compare amax, scale_inv
  }
}

template <float (*ref_dact)(const float),
          void (*nvte_dact)(const NVTETensor, const NVTETensor,
                               NVTETensor, NVTETensor, NVTETensor,
                               cudaStream_t),
         typename IType, typename OType>
void performTestDAct(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  // const NVTETensor input, const NVTETensor activation_input,
  //  NVTETensor output, NVTETensor dbias, NVTETensor workspace,
  //  cudaStream_t stream

  Tensor input({ N, H }, itype);
  Tensor igrad({ N, H }, otype);
  Tensor ograd({ N, H }, itype);
  Tensor dbias({ H }, itype);
  auto workspace_shape = getDBiasWorkspaceShape(N, H, itype, otype);
  Tensor workspace(workspace_shape, DType::kFloat32);

  fillUniform(&input);
  fillUniform(&ograd);
  igrad.set_scale(1.f);

  std::unique_ptr<OType[]> ref_igrad = std::make_unique<OType[]>(N*H);

  nvte_dact(ograd.data(), input.data(), igrad.data(), dbias.data(), workspace.data(), 0);

  compute_ref_dact_cast<ref_dact>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                  ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);

    // TODO compare amax, scale_inv
  }
}

template <float (*ref_act)(const float),
          float (*ref_dact)(const float),
          void (*nvte_act)(const NVTETensor, NVTETensor, cudaStream_t),
          void (*nvte_dact)(const NVTETensor, const NVTETensor, NVTETensor, cudaStream_t),
         typename IType, typename OType>
void performTestGLU(const size_t N, const size_t H) {
  using namespace test;

  DType itype = TypeInfo<IType>::dtype;
  DType otype = TypeInfo<OType>::dtype;

  Tensor input({N, H * 2}, itype);
  Tensor output({N, H}, otype);
  Tensor igrad({ N, H * 2 }, itype);
  Tensor ograd({ N, H }, itype);

  fillUniform(&input);
  fillUniform(&ograd);
  setRandomScale(&output);

  std::unique_ptr<OType[]> ref_output = std::make_unique<OType[]>(N * H);
  std::unique_ptr<IType[]> ref_igrad = std::make_unique<IType[]>(2 * N * H);

  nvte_act(input.data(), output.data(), 0);

  float ref_amax;
  compute_ref_glu_act_cast<ref_act>(input.rowwise_cpu_dptr<IType>(), ref_output.get(),
                                    output.scale(), &ref_amax, N, H);

  cudaDeviceSynchronize();
  auto err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  if (otype == DType::kFloat8E4M3 || otype == DType::kFloat8E5M2) {
    auto [atol, rtol] = getTolerances(DType::kFloat32);
    compareResults("amax", output.amax(), ref_amax, atol, rtol);
    if (output.scaling_mode() == NVTE_DELAYED_TENSOR_SCALING) {
      const float ref_scale = 1.f / output.scale();
      compareResults("scale_inv", *output.rowwise_cpu_scale_inv_ptr<float>(), ref_scale, atol, rtol);
    }
  }
  auto [atol, rtol] = getTolerances(otype);
  compareResults("output_gelu", output, ref_output.get(), atol, rtol);

  nvte_dact(ograd.data(), input.data(), igrad.data(), 0);

  compute_ref_dglu_act_cast<ref_dact, ref_act>(input.rowwise_cpu_dptr<IType>(), ograd.rowwise_cpu_dptr<IType>(),
                                               ref_igrad.get(), N, H);

  cudaDeviceSynchronize();
  err = cudaGetLastError();
  ASSERT_EQ(err, cudaSuccess) << cudaGetErrorString(err);

  {
    auto [atol, rtol] = getTolerances(otype);
    compareResults("igrad_act", igrad, ref_igrad.get(), atol, rtol);
  }
}


class ActTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                transformer_engine::DType,
                                                                std::pair<size_t, size_t>>> {};

TEST_P(ActTestSuite, TestGELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<gelu, dgelu, nvte_gelu, nvte_dgelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestSILU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<silu, dsilu, nvte_silu, nvte_dsilu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestRELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<relu, drelu, nvte_relu, nvte_drelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestQGELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<qgelu, dqgelu, nvte_qgelu, nvte_dqgelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestSRELU) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTest<srelu, dsrelu, nvte_srelu, nvte_dsrelu,
                    InputType, OutputType>(size.first, size.second);
      );
    );
}

TEST_P(ActTestSuite, TestGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<gelu, dgelu, nvte_geglu, nvte_dgeglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestReGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<relu, drelu, nvte_reglu, nvte_dreglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestSwiGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<silu, dsilu, nvte_swiglu, nvte_dswiglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestQGeGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<qgelu, dqgelu, nvte_qgeglu, nvte_dqgeglu, InputType,
                         OutputType>(size.first, size.second);););
}

TEST_P(ActTestSuite, TestSReGLU) {
  using namespace transformer_engine;
  using namespace test;

  const DType input_type = std::get<0>(GetParam());
  const DType output_type = std::get<1>(GetParam());
  const auto size = std::get<2>(GetParam());

  TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
      input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(
          output_type, OutputType,
          performTestGLU<srelu, dsrelu, nvte_sreglu, nvte_dsreglu, InputType,
                         OutputType>(size.first, size.second);););
}

class DActZeroGradTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                transformer_engine::DType,
                                                                std::pair<size_t, size_t>>> {};

TEST_P(DActZeroGradTestSuite, TestDGELUDBias) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTestDActZeroGradInput<dgelu, nvte_quantize_dbias_dgelu, InputType, OutputType>(size.first, size.second);
      );
    );
}

class DActTestSuite : public ::testing::TestWithParam<std::tuple<transformer_engine::DType,
                                                                transformer_engine::DType,
                                                                std::pair<size_t, size_t>>> {};

TEST_P(DActTestSuite, TestDGELUDBias) {
    using namespace transformer_engine;
    using namespace test;

    const DType input_type = std::get<0>(GetParam());
    const DType output_type = std::get<1>(GetParam());
    const auto size = std::get<2>(GetParam());

    TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(input_type, InputType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_ALL(output_type, OutputType,
        performTestDAct<dgelu, nvte_quantize_dbias_dgelu, InputType, OutputType>(size.first, size.second);
      );
    );
}


namespace {

std::vector<std::pair<size_t, size_t>> act_test_cases = {{2048, 12288},
                                                         {768, 2816},
                                                         {256, 65536},
                                                         {65536, 128},
                                                         {256, 256},
                                                         {257, 259},
                                                         {128, 128+1}};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    ActTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::ValuesIn(test::all_fp_types),
        ::testing::ValuesIn(act_test_cases)),
    [](const testing::TestParamInfo<ActTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DActZeroGradTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E5M2, DType::kFloat8E4M3),
        ::testing::Values(std::make_pair<size_t, size_t>(128, 128))),
    [](const testing::TestParamInfo<DActZeroGradTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });

INSTANTIATE_TEST_SUITE_P(
    OperatorTest,
    DActTestSuite,
    ::testing::Combine(
        ::testing::Values(DType::kFloat32, DType::kBFloat16, DType::kFloat16),
        ::testing::Values(DType::kFloat8E5M2, DType::kFloat8E4M3),
        ::testing::Values(std::make_pair<size_t, size_t>(128, 128))),
    [](const testing::TestParamInfo<DActTestSuite::ParamType>& info) {
      std::string name = test::typeName(std::get<0>(info.param)) + "X" +
                         test::typeName(std::get<1>(info.param)) + "X" +
                         std::to_string(std::get<2>(info.param).first) + "X" +
                         std::to_string(std::get<2>(info.param).second);
      return name;
    });
