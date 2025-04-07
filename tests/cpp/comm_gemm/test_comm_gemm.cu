#include <gtest/gtest.h>
#include <mpi.h>
#include <transformer_engine/comm_gemm.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>

#include "common.h"

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

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  CHECK_MPI(MPI_Init(&argc, &argv));
  auto ret = RUN_ALL_TESTS();
  CHECK_MPI(MPI_Finalize());
  return ret;
}

struct Params {
  transformer_engine::DType dtype;
  size_t m;
  size_t n;
  size_t k;
};

class CommGemmTest : public ::testing::TestWithParam<Params> {
 protected:
  CommGemmTest() {
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks_));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    int local_device = rank_;
    ctx_ = nvte_comm_gemm_ctx_create(nranks_, rank_, local_device);
    NVTE_CHECK_CUDA(cudaSetDevice(rank_));
  }
  ~CommGemmTest() { nvte_comm_gemm_ctx_destroy(ctx_); }

  template <typename T>
  transformer_engine::Tensor MakeDeviceTensor(size_t m, size_t n, T fill_value,
                                              transformer_engine::DType dtype) {
    std::vector<T> values(m * n, fill_value);
    void* dptr{};
    NVTE_CHECK_CUDA(cudaMalloc(&dptr, values.size() * sizeof values[0]));
    NVTE_CHECK_CUDA(
        cudaMemcpy(dptr, values.data(), values.size() * sizeof values[0], cudaMemcpyDefault));

    transformer_engine::Tensor ret;
    ret.data = {dptr, {m, n}, dtype};
    return ret;
  }

  template <typename T>
  void TestAgGemm(size_t m, size_t n, size_t k, transformer_engine::DType dtype) {
    cudaStream_t stream{};
    NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

    // std::mt19937 rng(12);
    // std::uniform_real_distribution<double> dist(0.0, 0.1);
    //
    // std::vector<T> a(m * k);
    // std::generate(a.begin(), a.end(), [&rng, &dist] { return dist(rng); });
    // std::vector<T> b(k * n);
    // std::generate(b.begin(), b.end(), [&rng, &dist] { return dist(rng); });
    // std::vector<T> d(m * n);
    // std::generate(d.begin(), d.end(), [&rng, &dist] { return dist(rng); });

    auto a = MakeDeviceTensor<T>(k, m / nranks_, 1.0, dtype);
    auto b = MakeDeviceTensor<T>(k, n / nranks_, 2.0, dtype);
    auto d = MakeDeviceTensor<T>(m / nranks_, n, 0.0, dtype);

    transformer_engine::Tensor bias;
    transformer_engine::Tensor pre_act_out;
    nvte_comm_gemm(ctx_, m, n, k, &a, &b, &d, &bias, &pre_act_out, true /*transa*/,
                   false /*transb*/, false /*grad*/, false /*accumulate*/, 0 /*comm_sm_count*/,
                   stream);

    std::vector<T> out(m * n);
    NVTE_CHECK_CUDA(cudaMemcpyAsync(out.data(), d.data.dptr, out.size() * sizeof out[0],
                                    cudaMemcpyDefault, stream));
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
  }

  CommGemmCtx* ctx_{};
  int nranks_{};
  int rank_{};
};

TEST_P(CommGemmTest, AgGemm) {
  auto [dtype, m, n, k] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, DType, { TestAgGemm<DType>(m, n, k, dtype); });

  // std::vector<DType> a(m * k);
  // std::generate(a.begin(), a.end(), [&rng, &dist] { return dist(rng); });
  // std::vector<DType> b(k * n);
  // std::generate(b.begin(), b.end(), [&rng, &dist] { return dist(rng); });
  // std::vector<DType> d(m * n);
  // std::generate(d.begin(), d.end(), [&rng, &dist] { return dist(rng); });
  // test::Tensor a("a", {k, m}, dtype);
  // test::Tensor b("b", {n, k}, dtype);
  // test::Tensor d("d", {n, m}, dtype);
  // test::Tensor bias;
  // test::Tensor pre_act_out;
  // // test::Tensor bias("bias", {m, n}, dtype);
  // // test::Tensor pre_act_out("pre_act_out", {m, n}, dtype);
  // test::Tensor workspace("workspace", {32 << 20}, transformer_engine::DType::kByte);
  // bool transa = true;
  // bool transb = false;
  // bool grad = false;
  // bool accumulate = false;
  // cudaStream_t stream{};
  // NVTE_CHECK_CUDA(cudaStreamCreate(&stream));
  // auto pa = a.rowwise_cpu_dptr<float>();
  // std::iota(pa, pa + m * k, 0);
  // a.from_cpu();
  // auto pb = b.rowwise_cpu_dptr<float>();
  // std::iota(pb, pb + n * k, 10);
  // b.from_cpu();
  // // nvte_cublas_gemm(a.data(), b.data(), d.data(), bias.data(), pre_act_out.data(), transa, transb,
  // //                  grad, workspace.data(), accumulate, false /* use_split_accumulator */,
  // //                  0 /* math_sm_count */, stream);
  // NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  // NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
  // nvte_comm_gemm(ctx_, m, n, k, a.data(), b.data(), d.data(), bias.data(), pre_act_out.data(),
  //                transa, transb, grad, accumulate, 0 /* comm_sm_count */);
  // d.to_cpu();
  // auto pd = d.rowwise_cpu_dptr<float>();
  // if (rank_ == 0) {
  //   for (size_t i = 0; i < m * n; ++i) {
  //     std::cerr << i << ": " << pd[i] << "\n";
  //   }
  // }
}

INSTANTIATE_TEST_SUITE_P(CommGemmShapes, CommGemmTest,
                         testing::Values(Params{transformer_engine::DType::kFloat32, 8, 4, 2}));
