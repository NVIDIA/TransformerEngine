/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <cuda.h>
#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <transformer_engine/comm_gemm.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../test_common.h"
#include "common.h"

using transformer_engine::DType;
using transformer_engine::TypeInfo;

#define CHECK_MPI(expr)                                              \
  do {                                                               \
    int err = (expr);                                                \
    if (err != MPI_SUCCESS) {                                        \
      char err_str[MPI_MAX_ERROR_STRING + 1]{};                      \
      int _len{};                                                    \
      MPI_Error_string(err, err_str, &_len);                         \
      EXPECT_TRUE(false) << "MPI error: " << err << ": " << err_str; \
    }                                                                \
  } while (false)

#define CHECK_NCCL(expr)                                                              \
  do {                                                                                \
    ncclResult_t err = (expr);                                                        \
    if (err != ncclSuccess) {                                                         \
      EXPECT_TRUE(false) << "NCCL error: " << err << ": " << ncclGetErrorString(err); \
    }                                                                                 \
  } while (false)

#define CHECK_CU(expr)                                          \
  do {                                                          \
    CUresult err = (expr);                                      \
    if (err != CUDA_SUCCESS) {                                  \
      const char* str{};                                        \
      CUresult e_str = cuGetErrorString(err, &str);             \
      if (e_str != CUDA_SUCCESS) str = "(unknown)";             \
      EXPECT_TRUE(false) << "CU error: " << err << ": " << str; \
    }                                                           \
  } while (false)

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  CHECK_MPI(MPI_Init(&argc, &argv));
  auto ret = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return ret;
}

bool IsMulticastSupported(int device_id) {
  int supported = 0;
  CHECK_CU(cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device_id));
  return supported;
}

template <typename T>
std::vector<T> CopyMatrix(const std::vector<T>& data, size_t mstart, size_t nstart, size_t msize,
                          size_t nsize, size_t ld) {
  std::vector<T> ret(msize * nsize);
  size_t dst = 0;
  for (size_t j = nstart; j < nstart + nsize; ++j) {
    for (size_t i = mstart; i < mstart + msize; ++i) {
      ret[dst++] = data[j * ld + i];
    }
  }
  return ret;
}

template <typename T>
test::Tensor Make(size_t m, size_t n, float scale) {
  test::Tensor ret("", std::vector{n, m}, TypeInfo<T>::dtype);
  ret.set_scale(scale);
  ret.set_scale_inv(1.0 / scale);
  return ret;
}

template <typename T>
test::Tensor MakeFromData(const std::vector<T>& data, size_t mstart, size_t nstart, size_t msize,
                          size_t nsize, size_t ld, float scale) {
  test::Tensor ret("", std::vector{nsize, msize}, TypeInfo<T>::dtype);
  ret.set_scale(scale);
  ret.set_scale_inv(1.0 / scale);
  auto local = CopyMatrix(data, mstart, nstart, msize, nsize, ld);
  NVTE_CHECK_CUDA(cudaMemcpy(ret.rowwise_dptr(), local.data(), local.size() * sizeof local[0],
                             cudaMemcpyDefault));
  return ret;
}

template <typename T>
float GetScale(float amax) {
  if constexpr (sizeof(T) > 1) return 1.0;
  return static_cast<float>(static_cast<T>(std::numeric_limits<float>::max())) / amax;
}

struct Params {
  DType a_type;
  DType b_type;
  DType d_type;
  bool transa;
  bool transb;
  size_t m;
  size_t n;
  size_t k;
  float tol;
};

class CommGemmFixure : public ::testing::TestWithParam<Params> {
 protected:
  CommGemmFixure() {
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks_));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    NVTE_CHECK_CUDA(cudaSetDevice(rank_));
    ncclUniqueId id{};
    if (rank_ == 0) CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
    CHECK_NCCL(ncclCommInitRank(&comm_, nranks_, id, rank_));
    ctx_ = nvte_comm_gemm_ctx_create(comm_, nranks_, rank_);
  }
  ~CommGemmFixure() {
    nvte_comm_gemm_ctx_destroy(ctx_);
    ncclCommDestroy(comm_);
  }

  struct PatternDims {
    int64_t a_rows_start;
    int64_t a_rows_num;
    int64_t a_cols_start;
    int64_t a_cols_num;
    int64_t b_rows_start;
    int64_t b_rows_num;
    int64_t b_cols_start;
    int64_t b_cols_num;
    int64_t d_rows_start;
    int64_t d_rows_num;
    int64_t d_cols_start;
    int64_t d_cols_num;
  };

  virtual PatternDims DistributeTensors(int64_t m, int64_t n, int64_t k) = 0;

  virtual void CommGemm(int64_t m, int64_t n, int64_t k, const NVTETensor a, const NVTETensor b,
                        const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                        bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                        cudaStream_t stream) = 0;

  template <typename AType, typename BType, typename DType, typename BiasType>
  void Run(bool transa, bool transb, size_t m, size_t n, size_t k, float tol) {
    cudaStream_t stream{};
    NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr float MAX_IN = 1.0;
    std::mt19937 rng(12);
    std::uniform_real_distribution<float> dist(0.0, MAX_IN);

    float a_scale = GetScale<AType>(MAX_IN);
    float b_scale = GetScale<BType>(MAX_IN);
    float d_scale = GetScale<DType>(MAX_IN * MAX_IN * k);
    float bias_scale = GetScale<BiasType>(MAX_IN);

    std::vector<AType> adata(m * k);
    std::generate(adata.begin(), adata.end(),
                  [&rng, &dist, a_scale] { return static_cast<AType>(dist(rng) * a_scale); });
    std::vector<BType> bdata(k * n);
    std::generate(bdata.begin(), bdata.end(),
                  [&rng, &dist, b_scale] { return static_cast<BType>(dist(rng) * b_scale); });
    std::vector<BiasType> biasdata(m * n);
    std::generate(biasdata.begin(), biasdata.end(), [&rng, &dist, bias_scale] {
      return static_cast<BiasType>(dist(rng) * bias_scale);
    });

    auto ga = transa ? MakeFromData<AType>(adata, 0, 0, k, m, k, a_scale)
                     : MakeFromData<AType>(adata, 0, 0, m, k, m, a_scale);
    auto gb = transb ? MakeFromData<BType>(bdata, 0, 0, n, k, n, b_scale)
                     : MakeFromData<BType>(bdata, 0, 0, k, n, k, b_scale);
    auto gbias = MakeFromData<BiasType>(biasdata, 0, 0, m, n, m, bias_scale);
    auto gd = Make<DType>(m, n, d_scale);
    auto gaux = Make<DType>(m, n, d_scale);

    auto dims = DistributeTensors(m, n, k);
    auto a = transa ? MakeFromData<AType>(adata, dims.a_rows_start, dims.a_cols_start,
                                          dims.a_rows_num, dims.a_cols_num, k, a_scale)
                    : MakeFromData<AType>(adata, dims.a_cols_start, dims.a_rows_start,
                                          dims.a_cols_num, dims.a_rows_num, m, a_scale);
    auto b = transb ? MakeFromData<BType>(bdata, dims.b_cols_start, dims.b_rows_start,
                                          dims.b_cols_num, dims.b_rows_num, n, b_scale)
                    : MakeFromData<BType>(bdata, dims.b_rows_start, dims.b_cols_start,
                                          dims.b_rows_num, dims.b_cols_num, k, b_scale);
    auto bias = MakeFromData<BiasType>(biasdata, dims.d_rows_start, dims.d_cols_start,
                                       dims.d_rows_num, dims.d_cols_num, m, bias_scale);
    auto d = Make<DType>(dims.d_rows_num, dims.d_cols_num, d_scale);
    auto aux = Make<DType>(dims.d_rows_num, dims.d_cols_num, d_scale);

    bool grad = false;
    bool accumulate = false;
    CommGemm(m, n, k, a.data(), b.data(), d.data(), bias.data(), aux.data(), transa, transb, grad,
             accumulate, 0 /*comm_sm_count*/, stream);
    auto workspace = Make<uint8_t>(1, 32 << 20, 1.0);
    nvte_cublas_gemm(ga.data(), gb.data(), gd.data(), gbias.data(), gaux.data(), transa, transb,
                     grad, workspace.data(), accumulate, false /* use_split_accumulator */,
                     0 /* math_sm_count */, stream);
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
    std::vector<DType> out(dims.d_rows_num * dims.d_cols_num);
    NVTE_CHECK_CUDA(
        cudaMemcpy(out.data(), d.rowwise_dptr(), out.size() * sizeof out[0], cudaMemcpyDefault));
    std::vector<DType> out_golden_global(m * n);
    NVTE_CHECK_CUDA(cudaMemcpy(out_golden_global.data(), gd.rowwise_dptr(),
                               out_golden_global.size() * sizeof out_golden_global[0],
                               cudaMemcpyDefault));

    auto out_golden = CopyMatrix(out_golden_global, dims.d_rows_start, dims.d_cols_start,
                                 dims.d_rows_num, dims.d_cols_num, m);
    NVTE_CHECK(out.size() == out_golden.size());
    for (size_t i = 0; i < out.size(); ++i) {
      EXPECT_NEAR(static_cast<float>(out[i]), static_cast<float>(out_golden[i]), tol * k);
    }
  }

  NVTECommGemmCtx* ctx_{};
  int nranks_{};
  int rank_{};
  ncclComm_t comm_{};
};

struct AgGemm : public CommGemmFixure {
  PatternDims DistributeTensors(int64_t m, int64_t n, int64_t k) override {
    auto a_cols_num = nvte_comm_gemm_numroc(ctx_, m);
    auto b_cols_num = nvte_comm_gemm_numroc(ctx_, n);

    int64_t a_cols_start{};
    int64_t b_cols_start{};
    MPI_Exscan(&a_cols_num, &a_cols_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&b_cols_num, &b_cols_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    return PatternDims{
        .a_rows_start = 0,
        .a_rows_num = k,
        .a_cols_start = a_cols_start,
        .a_cols_num = a_cols_num,
        .b_rows_start = 0,
        .b_rows_num = k,
        .b_cols_start = b_cols_start,
        .b_cols_num = b_cols_num,
        .d_rows_start = a_cols_start,
        .d_rows_num = a_cols_num,
        .d_cols_start = 0,
        .d_cols_num = n,
    };
  }

  void CommGemm(int64_t m, int64_t n, int64_t k, const NVTETensor a, const NVTETensor b,
                const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                cudaStream_t stream) override {
    nvte_all_gather_gemm(ctx_, m, n, k, a, b, d, bias, pre_act_out, transa, transb, grad,
                         accumulate, comm_sm_count, stream, kNVTECommGemmAlgoDefault);
  }
};

struct GemmRs : public CommGemmFixure {
  PatternDims DistributeTensors(int64_t m, int64_t n, int64_t k) override {
    auto rows_num = nvte_comm_gemm_numroc(ctx_, k);
    auto d_cols_num = nvte_comm_gemm_numroc(ctx_, n);

    int64_t rows_start{};
    int64_t d_cols_start{};
    MPI_Exscan(&rows_num, &rows_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Exscan(&d_cols_num, &d_cols_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    return PatternDims{
        .a_rows_start = rows_start,
        .a_rows_num = rows_num,
        .a_cols_start = 0,
        .a_cols_num = m,
        .b_rows_start = rows_start,
        .b_rows_num = rows_num,
        .b_cols_start = 0,
        .b_cols_num = n,
        .d_rows_start = 0,
        .d_rows_num = m,
        .d_cols_start = d_cols_start,
        .d_cols_num = d_cols_num,
    };
  }

  void CommGemm(int64_t m, int64_t n, int64_t k, const NVTETensor a, const NVTETensor b,
                const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                cudaStream_t stream) override {
    nvte_gemm_reduce_scatter(ctx_, m, n, k, a, b, d, bias, pre_act_out, transa, transb, grad,
                             accumulate, comm_sm_count, stream, kNVTECommGemmAlgoDefault);
  }
};

struct GemmAr : public CommGemmFixure {
  PatternDims DistributeTensors(int64_t m, int64_t n, int64_t k) override {
    auto rows_num = nvte_comm_gemm_numroc(ctx_, k);

    int64_t rows_start{};
    MPI_Exscan(&rows_num, &rows_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    return PatternDims{
        .a_rows_start = rows_start,
        .a_rows_num = rows_num,
        .a_cols_start = 0,
        .a_cols_num = m,
        .b_rows_start = rows_start,
        .b_rows_num = rows_num,
        .b_cols_start = 0,
        .b_cols_num = n,
        .d_rows_start = 0,
        .d_rows_num = m,
        .d_cols_start = 0,
        .d_cols_num = n,
    };
  }

  void CommGemm(int64_t m, int64_t n, int64_t k, const NVTETensor a, const NVTETensor b,
                const NVTETensor d, const NVTETensor bias, const NVTETensor pre_act_out,
                bool transa, bool transb, bool grad, bool accumulate, int comm_sm_count,
                cudaStream_t stream) override {
    nvte_gemm_all_reduce(ctx_, m, n, k, a, b, d, bias, pre_act_out, transa, transb, grad,
                         accumulate, comm_sm_count, stream, kNVTECommGemmAlgoDefault);
  }

  void SetUp() override {
    if (!IsMulticastSupported(rank_))
      GTEST_SKIP() << "Multicast is not supported on device " << rank_;
  }
};

TEST_P(AgGemm, Gemm) {
  auto [a_type, b_type, d_type, transa, transb, m, n, k, tol] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
      a_type, AType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          b_type, BType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
              d_type, DType, Run<AType, BType, DType, DType>(transa, transb, m, n, k, tol);)));
}

TEST_P(GemmRs, Gemm) {
  auto [a_type, b_type, d_type, transa, transb, m, n, k, tol] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
      a_type, AType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          b_type, BType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
              d_type, DType, Run<AType, BType, DType, DType>(transa, transb, m, n, k, tol);)));
}

TEST_P(GemmAr, Gemm) {
  auto [a_type, b_type, d_type, transa, transb, m, n, k, tol] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
      a_type, AType,
      TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
          b_type, BType,
          TRANSFORMER_ENGINE_TYPE_SWITCH_OUTPUT(
              d_type, DType, Run<AType, BType, DType, DType>(transa, transb, m, n, k, tol);)));
}

std::string ParamSuffix(const testing::TestParamInfo<Params>& info) {
  const auto [a_type, b_type, d_type, transa, transb, m, n, k, _tol] = info.param;
  std::ostringstream ss;
  ss << static_cast<int>(a_type) << "_" << static_cast<int>(b_type) << "_"
     << static_cast<int>(d_type) << "_" << (transa ? "T" : "N") << (transb ? "T" : "N") << "_" << m
     << "x" << n << "x" << k;
  return ss.str();
}

INSTANTIATE_TEST_SUITE_P(AgGemm, AgGemm,
                         testing::Values(Params{DType::kFloat16, DType::kFloat16, DType::kFloat16,
                                                false, false, 256, 128, 64, 1e-3},
                                         Params{DType::kFloat16, DType::kFloat16, DType::kFloat16,
                                                false, true, 256, 128, 64, 1e-3},
                                         Params{DType::kFloat16, DType::kFloat16, DType::kFloat16,
                                                true, false, 256, 128, 64, 1e-3},
                                         Params{DType::kBFloat16, DType::kBFloat16,
                                                DType::kBFloat16, false, false, 256, 128, 64, 1e-3},
                                         Params{DType::kBFloat16, DType::kBFloat16,
                                                DType::kBFloat16, false, true, 256, 128, 64, 1e-3},
                                         Params{DType::kBFloat16, DType::kBFloat16,
                                                DType::kBFloat16, true, false, 256, 128, 64, 1e-3},
                                         Params{DType::kFloat8E4M3, DType::kFloat8E4M3,
                                                DType::kFloat16, true, false, 256, 128, 64, 1e-3},
                                         Params{DType::kFloat8E4M3, DType::kFloat8E5M2,
                                                DType::kFloat16, true, false, 256, 128, 64, 1e-3},
                                         Params{DType::kFloat8E5M2, DType::kFloat8E4M3,
                                                DType::kFloat16, true, false, 256, 128, 64, 1e-3}),
                         &ParamSuffix);

INSTANTIATE_TEST_SUITE_P(GemmRs, GemmRs,
                         testing::Values(Params{DType::kFloat16, DType::kFloat16, DType::kFloat16,
                                                false, false, 64, 128, 256, 5e-2},
                                         Params{DType::kFloat16, DType::kFloat16, DType::kFloat16,
                                                false, true, 64, 128, 256, 5e-2},
                                         Params{DType::kFloat16, DType::kFloat16, DType::kFloat16,
                                                true, false, 64, 128, 256, 5e-2},
                                         Params{DType::kBFloat16, DType::kBFloat16,
                                                DType::kBFloat16, false, false, 64, 128, 256, 5e-2},
                                         Params{DType::kBFloat16, DType::kBFloat16,
                                                DType::kBFloat16, false, true, 64, 128, 256, 5e-2},
                                         Params{DType::kBFloat16, DType::kBFloat16,
                                                DType::kBFloat16, true, false, 64, 128, 256, 5e-2},
                                         Params{DType::kFloat8E4M3, DType::kFloat8E4M3,
                                                DType::kFloat16, true, false, 64, 128, 256, 5e-2},
                                         Params{DType::kFloat8E4M3, DType::kFloat8E5M2,
                                                DType::kFloat16, true, false, 64, 128, 256, 5e-2},
                                         Params{DType::kFloat8E5M2, DType::kFloat8E4M3,
                                                DType::kFloat16, true, false, 64, 128, 256, 5e-2}),
                         &ParamSuffix);

INSTANTIATE_TEST_SUITE_P(
    GemmAr, GemmAr,
    testing::Values(Params{DType::kFloat16, DType::kFloat16, DType::kFloat16, true, false, 64,
                           64 * 4, 64 * 4, 5e-2},
                    Params{DType::kBFloat16, DType::kBFloat16, DType::kBFloat16, true, false, 64,
                           64 * 4, 64 * 4, 5e-2},
                    Params{DType::kFloat8E5M2, DType::kFloat8E4M3, DType::kFloat16, true, false,
                           128, 128 * 4, 128 * 4, 5e-2},
                    Params{DType::kFloat8E4M3, DType::kFloat8E5M2, DType::kFloat16, true, false,
                           128, 128 * 4, 128 * 4, 5e-2},
                    Params{DType::kFloat8E4M3, DType::kFloat8E4M3, DType::kFloat16, true, false,
                           128, 128 * 4, 128 * 4, 5e-2}),
    &ParamSuffix);
