/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

// Tests that caching offsets/first_dims from gmem into smem via cp.async
// produces identical results to reading directly from gmem in
// get_current_tensor_id().

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(expr)                                  \
  do {                                                    \
    cudaError_t _e = (expr);                              \
    ASSERT_EQ(_e, cudaSuccess) << cudaGetErrorString(_e); \
  } while (0)

// ---------------------------------------------------------------------------
// Device-side implementation (mirrors the file under test)
// ---------------------------------------------------------------------------

enum ShapeRepresentation {
  SAME_BOTH_DIMS = 0,
  VARYING_FIRST_DIM = 1,
  VARYING_LAST_DIM = 2,
  VARYING_BOTH_DIMS = 3
};

// Exact copy of get_current_tensor_id from the file under test
__device__ __forceinline__ size_t get_current_tensor_id(
    const ShapeRepresentation shape_rep, const size_t num_tensors, const size_t current_offset,
    const size_t first_logical_dim, const size_t last_logical_dim,
    const int64_t *const __restrict__ offsets_ptr) {
  if (shape_rep == SAME_BOTH_DIMS) {
    const size_t current_row = current_offset / last_logical_dim;
    const size_t rows_per_tensor = first_logical_dim / num_tensors;
    return current_row / rows_per_tensor;
  } else {
    size_t low = 0, hi = num_tensors;
    while (low < hi) {
      const size_t mid = low + (hi - low) / 2;
      const size_t mid_offset = static_cast<size_t>(offsets_ptr[mid]);
      if (mid_offset <= current_offset)
        low = mid + 1;
      else
        hi = mid;
    }
    return (low == 0) ? 0 : (low - 1);
  }
}

// ---------------------------------------------------------------------------
// Test kernel
//
// For every query offset in `queries[]`:
//   result_gmem[i]  = get_current_tensor_id(..., gmem offsets pointer)
//   result_smem[i]  = get_current_tensor_id(..., smem-cached offsets pointer)
//
// The smem path uses the same cp.async + wait_all + __syncthreads() sequence
// as the production kernel.
// ---------------------------------------------------------------------------

constexpr int kMaxTensors = 64;

struct SmemStorage {
  int64_t offsets[kMaxTensors + 1];
};

__global__ void test_offset_caching_kernel(
    const int64_t *gmem_offsets,  // [num_tensors+1]
    const size_t *queries,        // [num_queries]  — offsets to look up
    size_t *result_gmem,          // [num_queries]
    size_t *result_smem,          // [num_queries]
    size_t num_tensors, size_t num_queries, ShapeRepresentation shape_rep, size_t first_logical_dim,
    size_t last_logical_dim) {
  extern __shared__ char smem_raw[];
  SmemStorage &smem = *reinterpret_cast<SmemStorage *>(smem_raw);

  // ---- cooperative cp.async load (mirrors production kernel lines 385-397) ----
  for (size_t i = threadIdx.x; i <= num_tensors; i += blockDim.x) {
    uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(&smem.offsets[i]));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" ::"r"(dst),
                 "l"(reinterpret_cast<unsigned long long>(&gmem_offsets[i])));
  }
  asm volatile("cp.async.commit_group;\n" ::);
  asm volatile("cp.async.wait_all;\n" ::);
  __syncthreads();

  const int64_t *const offsets_smem = smem.offsets;

  // ---- each thread handles one query ----------------------------------------
  for (size_t q = threadIdx.x; q < num_queries; q += blockDim.x) {
    size_t offset = queries[q];

    result_gmem[q] = get_current_tensor_id(shape_rep, num_tensors, offset, first_logical_dim,
                                           last_logical_dim, gmem_offsets);

    result_smem[q] = get_current_tensor_id(shape_rep, num_tensors, offset, first_logical_dim,
                                           last_logical_dim, offsets_smem);
  }
}

// ---------------------------------------------------------------------------
// Host-side reference — pure C++, no CUDA
// ---------------------------------------------------------------------------

static size_t ref_get_tensor_id(ShapeRepresentation shape_rep, size_t num_tensors,
                                size_t current_offset, size_t first_logical_dim,
                                size_t last_logical_dim, const std::vector<int64_t> &offsets) {
  if (shape_rep == SAME_BOTH_DIMS) {
    size_t current_row = current_offset / last_logical_dim;
    size_t rows_per_tensor = first_logical_dim / num_tensors;
    return current_row / rows_per_tensor;
  } else {
    size_t low = 0, hi = num_tensors;
    while (low < hi) {
      size_t mid = low + (hi - low) / 2;
      if (static_cast<size_t>(offsets[mid]) <= current_offset)
        low = mid + 1;
      else
        hi = mid;
    }
    return (low == 0) ? 0 : (low - 1);
  }
}

// ---------------------------------------------------------------------------
// Helper: run kernel + compare
// ---------------------------------------------------------------------------

static void run_test(ShapeRepresentation shape_rep, const std::vector<int64_t> &offsets_host,
                     const std::vector<size_t> &queries_host, size_t first_logical_dim,
                     size_t last_logical_dim) {
  const size_t num_tensors = offsets_host.size() - 1;  // offsets has num_tensors+1 entries
  const size_t num_queries = queries_host.size();

  // --- allocate device memory -----------------------------------------------
  int64_t *d_offsets = nullptr;
  size_t *d_queries = nullptr, *d_result_gmem = nullptr, *d_result_smem = nullptr;

  CUDA_CHECK(cudaMalloc(&d_offsets, (num_tensors + 1) * sizeof(int64_t)));
  CUDA_CHECK(cudaMalloc(&d_queries, num_queries * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_result_gmem, num_queries * sizeof(size_t)));
  CUDA_CHECK(cudaMalloc(&d_result_smem, num_queries * sizeof(size_t)));

  CUDA_CHECK(cudaMemcpy(d_offsets, offsets_host.data(), (num_tensors + 1) * sizeof(int64_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_queries, queries_host.data(), num_queries * sizeof(size_t),
                        cudaMemcpyHostToDevice));

  // --- launch ---------------------------------------------------------------
  int smem_bytes = sizeof(SmemStorage);
  test_offset_caching_kernel<<<1, 128, smem_bytes>>>(
      d_offsets, d_queries, d_result_gmem, d_result_smem, num_tensors, num_queries, shape_rep,
      first_logical_dim, last_logical_dim);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- copy results back ----------------------------------------------------
  std::vector<size_t> h_gmem(num_queries), h_smem(num_queries);
  CUDA_CHECK(cudaMemcpy(h_gmem.data(), d_result_gmem, num_queries * sizeof(size_t),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_smem.data(), d_result_smem, num_queries * sizeof(size_t),
                        cudaMemcpyDeviceToHost));

  // --- verify: gmem == smem == host_ref -------------------------------------
  for (size_t q = 0; q < num_queries; ++q) {
    size_t ref = ref_get_tensor_id(shape_rep, num_tensors, queries_host[q], first_logical_dim,
                                   last_logical_dim, offsets_host);
    EXPECT_EQ(h_gmem[q], ref) << "query=" << queries_host[q] << " gmem result mismatch at q=" << q;
    EXPECT_EQ(h_smem[q], ref) << "query=" << queries_host[q] << " smem result mismatch at q=" << q;
    EXPECT_EQ(h_gmem[q], h_smem[q]) << "query=" << queries_host[q] << " gmem != smem at q=" << q;
  }

  cudaFree(d_offsets);
  cudaFree(d_queries);
  cudaFree(d_result_gmem);
  cudaFree(d_result_smem);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

// Case 1: SAME_BOTH_DIMS — 4 tensors, equal rows each
TEST(OffsetCaching, SameBothDims) {
  // 4 tensors, each with 128 rows, hidden=256
  // offsets (in elements): 0, 128*256, 2*128*256, 3*128*256, 4*128*256
  const size_t hidden = 256;
  const size_t rows_per = 128;
  const size_t N = 4;

  std::vector<int64_t> offsets(N + 1);
  for (size_t i = 0; i <= N; ++i) offsets[i] = static_cast<int64_t>(i * rows_per * hidden);

  // Query every row's first element
  std::vector<size_t> queries;
  for (size_t t = 0; t < N; ++t)
    for (size_t r = 0; r < rows_per; ++r)
      queries.push_back(static_cast<size_t>(offsets[t]) + r * hidden);

  run_test(SAME_BOTH_DIMS, offsets, queries,
           /*first_logical_dim=*/N * rows_per,
           /*last_logical_dim=*/hidden);
}

// Case 2: VARYING_FIRST_DIM — tensors with different numbers of rows
TEST(OffsetCaching, VaryingFirstDim) {
  // 3 tensors with row counts 128, 256, 192; hidden=512
  const size_t hidden = 512;
  const std::vector<size_t> row_counts = {128, 256, 192};
  const size_t N = row_counts.size();

  std::vector<int64_t> offsets(N + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < N; ++i)
    offsets[i + 1] = offsets[i] + static_cast<int64_t>(row_counts[i] * hidden);

  // Query: first element, last element, and middle element of each tensor
  std::vector<size_t> queries;
  for (size_t t = 0; t < N; ++t) {
    size_t start = static_cast<size_t>(offsets[t]);
    size_t end = static_cast<size_t>(offsets[t + 1]) - 1;
    size_t mid = (start + end) / 2;
    queries.push_back(start);
    queries.push_back(mid);
    queries.push_back(end);
  }

  run_test(VARYING_FIRST_DIM, offsets, queries,
           /*first_logical_dim=*/0,  // unused for binary search path
           /*last_logical_dim=*/hidden);
}

// Case 3: boundary — query lands exactly on an offset boundary
TEST(OffsetCaching, ExactBoundary) {
  const size_t hidden = 128;
  const std::vector<size_t> row_counts = {128, 128, 256, 64};
  const size_t N = row_counts.size();

  std::vector<int64_t> offsets(N + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < N; ++i)
    offsets[i + 1] = offsets[i] + static_cast<int64_t>(row_counts[i] * hidden);

  // Query exactly at each offset boundary (should map to the tensor that starts there)
  std::vector<size_t> queries;
  for (size_t t = 0; t < N; ++t) queries.push_back(static_cast<size_t>(offsets[t]));

  run_test(VARYING_FIRST_DIM, offsets, queries,
           /*first_logical_dim=*/0,
           /*last_logical_dim=*/hidden);
}

// Case 4: single tensor — degenerate case
TEST(OffsetCaching, SingleTensor) {
  const size_t hidden = 256;
  const size_t rows = 512;

  std::vector<int64_t> offsets = {0, static_cast<int64_t>(rows * hidden)};
  std::vector<size_t> queries = {0, rows * hidden / 2, rows * hidden - 1};

  run_test(VARYING_FIRST_DIM, offsets, queries,
           /*first_logical_dim=*/rows,
           /*last_logical_dim=*/hidden);
}

// Case 5: maximum tensors (kMaxTensors=64)
TEST(OffsetCaching, MaxTensors) {
  const size_t hidden = 128;
  const size_t rows_each = 128;
  const size_t N = 64;

  std::vector<int64_t> offsets(N + 1);
  offsets[0] = 0;
  for (size_t i = 0; i < N; ++i)
    offsets[i + 1] = offsets[i] + static_cast<int64_t>(rows_each * hidden);

  // One query per tensor
  std::vector<size_t> queries;
  for (size_t t = 0; t < N; ++t) queries.push_back(static_cast<size_t>(offsets[t]));

  run_test(VARYING_FIRST_DIM, offsets, queries,
           /*first_logical_dim=*/0,
           /*last_logical_dim=*/hidden);
}
