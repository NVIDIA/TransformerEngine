#include <gtest/gtest.h>
#include <mpi.h>
#include <transformer_engine/comm_gemm.h>
#include <transformer_engine/gemm.h>

#include <algorithm>
#include <iostream>
#include <numeric>

#include "../test_common.h"

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

struct TestDims {
  size_t m;
  size_t n;
  size_t k;
};

class CommGemmTest : public ::testing::TestWithParam<TestDims> {
 protected:
  CommGemmTest() {
    int nranks{};
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nranks));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank_));
    int local_device = rank_;
    ctx_ = nvte_comm_gemm_ctx_create(nranks, rank_, local_device);
  }
  ~CommGemmTest() { nvte_comm_gemm_ctx_destroy(ctx_); }

  CommGemmCtx* ctx_{};
  int rank_;
};

TEST_P(CommGemmTest, GEMM) {
  auto [m, n, k] = GetParam();
  auto dtype = transformer_engine::DType::kFloat32;
  test::Tensor a("a", {k, m}, dtype);
  test::Tensor b("b", {n, k}, dtype);
  test::Tensor d("d", {n, m}, dtype);
  test::Tensor bias;
  test::Tensor pre_act_out;
  // test::Tensor bias("bias", {m, n}, dtype);
  // test::Tensor pre_act_out("pre_act_out", {m, n}, dtype);
  test::Tensor workspace("workspace", {32 << 20}, transformer_engine::DType::kByte);
  bool transa = false;
  bool transb = false;
  bool grad = false;
  bool accumulate = false;
  cudaStream_t stream{};
  NVTE_CHECK_CUDA(cudaStreamCreate(&stream));
  auto pa = a.rowwise_cpu_dptr<float>();
  std::iota(pa, pa + m * k, 0);
  a.from_cpu();
  auto pb = b.rowwise_cpu_dptr<float>();
  std::iota(pb, pb + n * k, 10);
  b.from_cpu();
  nvte_cublas_gemm(a.data(), b.data(), d.data(), bias.data(), pre_act_out.data(), transa, transb,
                   grad, workspace.data(), accumulate, false /* use_split_accumulator */,
                   0 /* math_sm_count */, stream);
  NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
  NVTE_CHECK_CUDA(cudaStreamDestroy(stream));
  d.to_cpu();
  auto pd = d.rowwise_cpu_dptr<float>();
  if (rank_ == 0) {
    for (size_t i = 0; i < m * n; ++i) {
      std::cerr << i << ": " << pd[i] << "\n";
    }
  }
  nvte_comm_gemm(ctx_, m, n, k, a.data(), b.data(), d.data(), bias.data(), pre_act_out.data(),
                 transa, transb, grad, accumulate, 0 /* comm_sm_count */);
}

INSTANTIATE_TEST_SUITE_P(CommGemmShapes, CommGemmTest, testing::Values(TestDims{2, 3, 4}));
