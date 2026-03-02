/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <transformer_engine/cast.h>
#include <transformer_engine/multi_tensor.h>

#include "../test_common.h"

using namespace transformer_engine;
using namespace test;

namespace {

uint8_t fp8_to_u8(fp8e4m3 v) {
  uint8_t out = 0;
  std::memcpy(&out, &v, sizeof(uint8_t));
  return out;
}

uint8_t fp8_to_u8(fp8e5m2 v) {
  uint8_t out = 0;
  std::memcpy(&out, &v, sizeof(uint8_t));
  return out;
}

void run_mxfp8_adam_test(DType fp8_dtype) {
  const std::vector<size_t> shape1{64, 128};
  const std::vector<size_t> shape2{32, 64};
  const float lr = 1e-3f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const int step = 1;
  const int mode = 1;
  const int bias_correction = 1;
  const float weight_decay = 0.0f;

  // Run with 25 tensors > 24[MXFP8_MAX_TENSORS] to check
  // the chunking logic
  const size_t tensor_count = 25;
  std::vector<std::vector<size_t>> shapes;
  shapes.reserve(tensor_count);
  for (size_t i = 0; i < tensor_count; ++i) {
    shapes.push_back((i % 2 == 0) ? shape1 : shape2);
  }

  std::vector<std::string> names;
  names.reserve(tensor_count * 11);
  std::vector<Tensor> g;
  std::vector<Tensor> p;
  std::vector<Tensor> m;
  std::vector<Tensor> v;
  std::vector<Tensor> p_ref_t;
  std::vector<Tensor> m_ref_t;
  std::vector<Tensor> v_ref_t;
  std::vector<Tensor> q_ref;
  std::vector<Tensor> dq;
  std::vector<Tensor> dq_ref;
  std::vector<Tensor> q;
  g.reserve(tensor_count);
  p.reserve(tensor_count);
  m.reserve(tensor_count);
  v.reserve(tensor_count);
  p_ref_t.reserve(tensor_count);
  m_ref_t.reserve(tensor_count);
  v_ref_t.reserve(tensor_count);
  q_ref.reserve(tensor_count);
  dq.reserve(tensor_count);
  dq_ref.reserve(tensor_count);
  q.reserve(tensor_count);

  for (size_t i = 0; i < tensor_count; ++i) {
    const std::vector<size_t> &shape = shapes[i];
    names.push_back("g" + std::to_string(i));
    g.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);
    names.push_back("p" + std::to_string(i));
    p.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);
    names.push_back("m" + std::to_string(i));
    m.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);
    names.push_back("v" + std::to_string(i));
    v.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);

    fillUniform(&g.back());
    fillUniform(&p.back());
    std::fill_n(m.back().rowwise_cpu_dptr<float>(), product(m.back().rowwise_shape()), 0.0f);
    std::fill_n(v.back().rowwise_cpu_dptr<float>(), product(v.back().rowwise_shape()), 0.0f);
    m.back().from_cpu();
    v.back().from_cpu();

    names.push_back("p_ref_" + std::to_string(i));
    p_ref_t.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);
    names.push_back("m_ref_" + std::to_string(i));
    m_ref_t.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);
    names.push_back("v_ref_" + std::to_string(i));
    v_ref_t.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);
    const size_t n = shape[0] * shape[1];
    std::memcpy(p_ref_t.back().rowwise_cpu_dptr<float>(), p.back().rowwise_cpu_dptr<float>(),
                n * sizeof(float));
    std::memcpy(m_ref_t.back().rowwise_cpu_dptr<float>(), m.back().rowwise_cpu_dptr<float>(),
                n * sizeof(float));
    std::memcpy(v_ref_t.back().rowwise_cpu_dptr<float>(), v.back().rowwise_cpu_dptr<float>(),
                n * sizeof(float));
    p_ref_t.back().from_cpu();
    m_ref_t.back().from_cpu();
    v_ref_t.back().from_cpu();

    names.push_back("q_ref_" + std::to_string(i));
    q_ref.emplace_back(names.back().c_str(), shape, fp8_dtype, true, true, NVTE_MXFP8_1D_SCALING);
    q_ref.back().set_with_gemm_swizzled_scales(false);

    names.push_back("dq" + std::to_string(i));
    dq.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);
    names.push_back("dq_ref_" + std::to_string(i));
    dq_ref.emplace_back(names.back().c_str(), shape, DType::kFloat32, true, false);

    names.push_back("q" + std::to_string(i));
    q.emplace_back(names.back().c_str(), shape, fp8_dtype, true, true, NVTE_MXFP8_1D_SCALING);
    q.back().set_with_gemm_swizzled_scales(false);
  }

  Tensor noop("noop", std::vector<size_t>{1}, DType::kInt32, true, false);
  int zero = 0;
  std::memcpy(noop.rowwise_cpu_dptr<int>(), &zero, sizeof(int));
  noop.from_cpu();

  std::vector<std::vector<NVTETensor>> lists(8);
  std::vector<TensorWrapper> extra_wrappers;
  extra_wrappers.reserve(tensor_count * 4);

  auto add_tensor = [&](Tensor &g, Tensor &p, Tensor &m, Tensor &v, Tensor &q) {
    lists[0].push_back(g.data());
    lists[1].push_back(p.data());
    lists[2].push_back(m.data());
    lists[3].push_back(v.data());

    extra_wrappers.emplace_back(q.rowwise_dptr(), q.rowwise_shape(), fp8_dtype);
    lists[4].push_back(extra_wrappers.back().data());
    extra_wrappers.emplace_back(q.columnwise_dptr(), q.columnwise_shape(), fp8_dtype);
    lists[5].push_back(extra_wrappers.back().data());
    extra_wrappers.emplace_back(q.rowwise_scale_inv_dptr(), q.rowwise_scale_inv_shape(),
                                DType::kByte);
    lists[6].push_back(extra_wrappers.back().data());
    extra_wrappers.emplace_back(q.columnwise_scale_inv_dptr(), q.columnwise_scale_inv_shape(),
                                DType::kByte);
    lists[7].push_back(extra_wrappers.back().data());
  };

  for (size_t i = 0; i < tensor_count; ++i) {
    add_tensor(g[i], p[i], m[i], v[i], q[i]);
  }

  std::vector<NVTETensor *> list_ptrs;
  list_ptrs.reserve(lists.size());
  for (auto &l : lists) {
    list_ptrs.push_back(l.data());
  }

  nvte_multi_tensor_adam_mxfp8_cuda(65536, noop.data(), list_ptrs.data(), lists.size(),
                                    lists[0].size(), static_cast<NVTEDType>(fp8_dtype), lr, beta1,
                                    beta2, eps, step, mode, bias_correction, weight_decay, 0);
  
  std::vector<std::vector<NVTETensor>> ref_lists(4);
  for (size_t i = 0; i < tensor_count; ++i) {
    ref_lists[0].push_back(g[i].data());
    ref_lists[1].push_back(p_ref_t[i].data());
    ref_lists[2].push_back(m_ref_t[i].data());
    ref_lists[3].push_back(v_ref_t[i].data());
  }
  std::vector<NVTETensor *> ref_list_ptrs;
  ref_list_ptrs.reserve(ref_lists.size());
  for (auto &l : ref_lists) {
    ref_list_ptrs.push_back(l.data());
  }

  nvte_multi_tensor_adam_cuda(65536, noop.data(), ref_list_ptrs.data(), ref_lists.size(),
                              ref_lists[0].size(), lr, beta1, beta2, eps, step, mode,
                              bias_correction, weight_decay, 0);

  for (size_t i = 0; i < tensor_count; ++i) {
    nvte_quantize(p_ref_t[i].data(), q_ref[i].data(), 0);
    nvte_dequantize(q[i].data(), dq[i].data(), 0);
    nvte_dequantize(q_ref[i].data(), dq_ref[i].data(), 0);
  }

  cudaDeviceSynchronize();

  for (size_t i = 0; i < tensor_count; ++i) {
    q[i].to_cpu();
    p[i].to_cpu();
    m[i].to_cpu();
    v[i].to_cpu();
    q_ref[i].to_cpu();
    dq[i].to_cpu();
    dq_ref[i].to_cpu();
    p_ref_t[i].to_cpu();
    m_ref_t[i].to_cpu();
    v_ref_t[i].to_cpu();
  }

  for (size_t i = 0; i < lists[0].size(); ++i) {
    const Tensor &g_i = g[i];
    const Tensor &p_i = p[i];
    const Tensor &m_i = m[i];
    const Tensor &v_i = v[i];
    Tensor &q_i = q[i];
    const Tensor &p_ref_t_i = p_ref_t[i];
    const Tensor &m_ref_t_i = m_ref_t[i];
    const Tensor &v_ref_t_i = v_ref_t[i];
    Tensor &q_ref_i = q_ref[i];

    compareResults("p", p_i, p_ref_t_i.rowwise_cpu_dptr<float>(), true, 0.0, 0.0, true, 0);
    compareResults("m", m_i, m_ref_t_i.rowwise_cpu_dptr<float>(), true, 0.0, 0.0, true, 0);
    compareResults("v", v_i, v_ref_t_i.rowwise_cpu_dptr<float>(), true, 0.0, 0.0, true, 0);

    const Tensor &dq_i = dq[i];
    const Tensor &dq_ref_i = dq_ref[i];
    compareResults("dequantized", dq_i, dq_ref_i.rowwise_cpu_dptr<float>(), true, 0.0, 0.0, true,
                   0);

    const size_t rs = q_i.rowwise_scale_inv_shape().data[1];
    const size_t cs = q_i.columnwise_scale_inv_shape().data[1];
    const size_t rowwise_scale_size = q_i.rowwise_scale_inv_shape().data[0] * rs;
    const size_t colwise_scale_size = q_i.columnwise_scale_inv_shape().data[0] * cs;
    compareResults("rowwise_scale", q_i.rowwise_cpu_scale_inv_ptr<uint8_t>(),
                   q_ref_i.rowwise_cpu_scale_inv_ptr<uint8_t>(), rowwise_scale_size, 0.0f);
    compareResults("colwise_scale", q_i.columnwise_cpu_scale_inv_ptr<uint8_t>(),
                   q_ref_i.columnwise_cpu_scale_inv_ptr<uint8_t>(), colwise_scale_size, 0.0f);

    uint8_t *row_data = nullptr;
    uint8_t *col_data = nullptr;
    uint8_t *row_data_ref = nullptr;
    uint8_t *col_data_ref = nullptr;
    if (fp8_dtype == DType::kFloat8E4M3) {
      row_data = reinterpret_cast<uint8_t *>(q_i.rowwise_cpu_dptr<fp8e4m3>());
      col_data = reinterpret_cast<uint8_t *>(q_i.columnwise_cpu_dptr<fp8e4m3>());
      row_data_ref = reinterpret_cast<uint8_t *>(q_ref_i.rowwise_cpu_dptr<fp8e4m3>());
      col_data_ref = reinterpret_cast<uint8_t *>(q_ref_i.columnwise_cpu_dptr<fp8e4m3>());
    } else {
      row_data = reinterpret_cast<uint8_t *>(q_i.rowwise_cpu_dptr<fp8e5m2>());
      col_data = reinterpret_cast<uint8_t *>(q_i.columnwise_cpu_dptr<fp8e5m2>());
      row_data_ref = reinterpret_cast<uint8_t *>(q_ref_i.rowwise_cpu_dptr<fp8e5m2>());
      col_data_ref = reinterpret_cast<uint8_t *>(q_ref_i.columnwise_cpu_dptr<fp8e5m2>());
    }
    const size_t data_size = q_i.rowwise_shape().data[0] * q_i.rowwise_shape().data[1];
    compareResults("rowwise_data", row_data, row_data_ref, data_size, 0.0f);
    compareResults("colwise_data", col_data, col_data_ref, data_size, 0.0f);
  }
}

}  // namespace

TEST(MultiTensorAdamMXFP8, E4M3) { run_mxfp8_adam_test(DType::kFloat8E4M3); }

TEST(MultiTensorAdamMXFP8, E5M2) { run_mxfp8_adam_test(DType::kFloat8E5M2); }
