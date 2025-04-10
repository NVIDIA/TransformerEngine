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
#include <sstream>

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
    NVTE_CHECK_CUDA(cudaSetDevice(rank_));
    ctx_ = nvte_comm_gemm_ctx_create(nranks_, rank_, local_device);
  }
  ~CommGemmTest() { nvte_comm_gemm_ctx_destroy(ctx_); }

  template <typename T>
  transformer_engine::Tensor MakeDeviceTensor(size_t m, size_t n, transformer_engine::DType dtype,
                                              cudaStream_t stream) {
    void* dptr{};
    NVTE_CHECK_CUDA(cudaMallocAsync(&dptr, m * n * sizeof(T), stream));
    NVTE_CHECK_CUDA(cudaMemsetAsync(dptr, 0, m * n * sizeof(T), stream));

    transformer_engine::Tensor ret;
    ret.data = {dptr, {n, m}, dtype};
    return ret;
  }

  template <typename T>
  std::vector<T> CopyLocalMatrix(const std::vector<T>& data, size_t mstart, size_t nstart,
                                 size_t msize, size_t nsize, size_t ld) {
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
  transformer_engine::Tensor MakeDeviceTensorFromData(const std::vector<T>& data, size_t mstart,
                                                      size_t nstart, size_t msize, size_t nsize,
                                                      size_t ld, transformer_engine::DType dtype,
                                                      cudaStream_t stream) {
    auto values = CopyLocalMatrix(data, mstart, nstart, msize, nsize, ld);
    void* dptr{};
    NVTE_CHECK_CUDA(cudaMallocAsync(&dptr, values.size() * sizeof values[0], stream));
    NVTE_CHECK_CUDA(cudaMemcpyAsync(dptr, values.data(), values.size() * sizeof values[0],
                                    cudaMemcpyDefault, stream));

    transformer_engine::Tensor ret;
    ret.data = {dptr, {nsize, msize}, dtype};
    return ret;
  }

  template <typename T>
  void TestAgGemm(size_t m, size_t n, size_t k, transformer_engine::DType dtype) {
    cudaStream_t stream{};
    NVTE_CHECK_CUDA(cudaStreamCreate(&stream));

    std::mt19937 rng(12);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<T> adata(m * k);
    std::generate(adata.begin(), adata.end(), [&rng, &dist] { return dist(rng); });
    std::vector<T> bdata(k * n);
    std::generate(bdata.begin(), bdata.end(), [&rng, &dist] { return dist(rng); });

    auto ga = MakeDeviceTensorFromData<T>(adata, 0, 0, k, m, k, dtype, stream);
    auto gb = MakeDeviceTensorFromData<T>(bdata, 0, 0, k, n, k, dtype, stream);
    auto gd = MakeDeviceTensor<T>(m, n, dtype, stream);

    auto a_cols = nvte_comm_gemm_numroc(ctx_, m);
    auto b_cols = nvte_comm_gemm_numroc(ctx_, n);

    int64_t a_cols_start{};
    int64_t b_cols_start{};
    MPI_Scan(&a_cols, &a_cols_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Scan(&b_cols, &b_cols_start, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    a_cols_start -= a_cols;
    b_cols_start -= b_cols;

    auto a = MakeDeviceTensorFromData<T>(adata, 0, a_cols_start, k, a_cols, k, dtype, stream);
    auto b = MakeDeviceTensorFromData<T>(bdata, 0, b_cols_start, k, b_cols, k, dtype, stream);
    auto d = MakeDeviceTensor<T>(a_cols, n, dtype, stream);

    transformer_engine::Tensor bias;
    transformer_engine::Tensor pre_act_out;
    bool transa = true;
    bool transb = false;
    bool grad = false;
    bool accumulate = false;
    nvte_comm_gemm(ctx_, m, n, k, &a, &b, &d, &bias, &pre_act_out, transa, transb, grad, accumulate,
                   0 /*comm_sm_count*/, stream);
    auto workspace =
        MakeDeviceTensor<uint8_t>(1, 32 << 20, transformer_engine::DType::kByte, stream);
    nvte_cublas_gemm(&ga, &gb, &gd, &bias, &pre_act_out, transa, transb, grad, &workspace,
                     accumulate, false /* use_split_accumulator */, 0 /* math_sm_count */, stream);

    std::vector<T> out(a_cols * n);
    NVTE_CHECK_CUDA(cudaMemcpyAsync(out.data(), d.data.dptr, out.size() * sizeof out[0],
                                    cudaMemcpyDefault, stream));
    std::vector<T> out_golden_global(m * n);
    NVTE_CHECK_CUDA(cudaMemcpyAsync(out_golden_global.data(), gd.data.dptr,
                                    out_golden_global.size() * sizeof out_golden_global[0],
                                    cudaMemcpyDefault, stream));
    NVTE_CHECK_CUDA(cudaStreamSynchronize(stream));
    NVTE_CHECK_CUDA(cudaStreamDestroy(stream));

    auto out_golden = CopyLocalMatrix(out_golden_global, a_cols_start, 0, a_cols, n, m);
    NVTE_CHECK(out.size() == out_golden.size());
    for (size_t i = 0; i < out.size(); ++i) {
      std::ostringstream ss;
      ss << rank_ << ": " << i << ": " << static_cast<float>(out[i]) << " "
         << static_cast<float>(out_golden[i]);
      std::cerr << ss.str() << "\n";
    }
  }

  CommGemmCtx* ctx_{};
  int nranks_{};
  int rank_{};
};

TEST_P(CommGemmTest, AgGemm) {
  auto [dtype, m, n, k] = GetParam();
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(dtype, DType, { TestAgGemm<DType>(m, n, k, dtype); });
}

INSTANTIATE_TEST_SUITE_P(CommGemmShapes, CommGemmTest,
                         testing::Values(Params{transformer_engine::DType::kFloat16, 8, 4, 2}));
