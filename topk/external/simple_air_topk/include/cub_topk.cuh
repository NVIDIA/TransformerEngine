#ifndef CUB_TOPK_CUH_
#define CUB_TOPK_CUH_
#include "nv_util.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
constexpr int BLOCK_DIM = 256;
}

namespace nv {

template <typename idxT> __global__ void init_idx(idxT *idx, idxT len) {
  for (idxT i = blockIdx.x * blockDim.x + threadIdx.x; i < len;
       i += blockDim.x * gridDim.x) {
    idx[i] = i;
  }
}

template <typename T, typename idxT>
void cub_topk(void *buf, size_t &buf_size, const T *in, idxT len, idxT k,
              T *out, idxT *out_idx = nullptr, bool greater = true,
              cudaStream_t stream = 0, idxT *lengths = nullptr) {
  idxT *in_idx = nullptr;
  T *out_for_sort = nullptr;
  idxT *out_idx_for_sort = nullptr;
  void *cub_buf = nullptr;

  size_t cub_buf_size = 0;
  if (greater) {
    cub::DeviceRadixSort::SortPairsDescending(
        nullptr, cub_buf_size, in, out_for_sort, in_idx, out_idx_for_sort, len);
  } else {
    cub::DeviceRadixSort::SortPairs(nullptr, cub_buf_size, in, out_for_sort,
                                    in_idx, out_idx_for_sort, len);
  }

  std::vector<size_t> sizes = {len * sizeof(*in_idx),
                               len * sizeof(*out_for_sort),
                               len * sizeof(*out_idx_for_sort), cub_buf_size};

  size_t total_size = calc_aligned_size(sizes);
  if (!buf) {
    buf_size = total_size;
    return;
  }
  std::vector<void *> aligned_pointers = calc_aligned_pointers(buf, sizes);
  in_idx = static_cast<idxT *>(aligned_pointers[0]);
  out_for_sort = static_cast<T *>(aligned_pointers[1]);
  out_idx_for_sort = static_cast<idxT *>(aligned_pointers[2]);
  cub_buf = aligned_pointers[3];

  init_idx<<<(len - 1) / BLOCK_DIM + 1, BLOCK_DIM, 0, stream>>>(in_idx, len);
  if (greater) {
    cub::DeviceRadixSort::SortPairsDescending(
        cub_buf, cub_buf_size, in, out_for_sort, in_idx, out_idx_for_sort, len,
        0, sizeof(T) * 8, stream);
  } else {
    cub::DeviceRadixSort::SortPairs(cub_buf, cub_buf_size, in, out_for_sort,
                                    in_idx, out_idx_for_sort, len, 0,
                                    sizeof(T) * 8, stream);
  }
  cudaMemcpyAsync(out, out_for_sort, k * sizeof(*out), cudaMemcpyDeviceToDevice,
                  stream);
  // TODO: use Sort instead of SortPairs when out_idx==nullptr
  if (out_idx) {
    cudaMemcpyAsync(out_idx, out_idx_for_sort, k * sizeof(*out_idx),
                    cudaMemcpyDeviceToDevice, stream);
  }
}

// TODO: remove 2nd loop
template <typename idxT>
__global__ void init_idx(idxT *idx, int batch_size, idxT len) {
  for (idxT j = blockIdx.x * blockDim.x + threadIdx.x; j < len;
       j += blockDim.x * gridDim.x) {
    for (int i = 0; i < batch_size; ++i) {
      idx[i * len + j] = j;
    }
  }
}

template <typename T, typename idxT>
__global__ void copy_result(const T *out_for_sort, const idxT *out_idx_for_sort,
                            T *out, idxT *out_idx, int batch_size, idxT len,
                            idxT k) {
  for (idxT j = blockIdx.x * blockDim.x + threadIdx.x; j < k;
       j += blockDim.x * gridDim.x) {
    for (int i = 0; i < batch_size; ++i) {
      out[i * k + j] = out_for_sort[i * len + j];
      out_idx[i * k + j] = out_idx_for_sort[i * len + j];
    }
  }
}

template <typename idxT> class SegmentOffsetCreator {
public:
  __device__ SegmentOffsetCreator(idxT len) : len_(len) {}
  __device__ idxT operator()(idxT idx) const { return idx * len_; }

private:
  idxT len_;
};

template <typename T, typename idxT>
void cub_segmented_topk(void *buf, size_t &buf_size, const T *in,
                        int batch_size, idxT len, idxT k, T *out,
                        idxT *out_idx = nullptr, bool greater = true,
                        cudaStream_t stream = 0, idxT *lengths = nullptr) {
  const idxT total = batch_size * len;
  idxT *in_idx = nullptr;
  T *out_for_sort = nullptr;
  idxT *out_idx_for_sort = nullptr;
  void *cub_buf = nullptr;

  // cub::CountingInputIterator<idxT> counting_iter(0);
  // cub::TransformInputIterator<idxT,
  //                             SegmentOffsetCreator<idxT>,
  //                             cub::CountingInputIterator<idxT>>
  //     segment_offsets(counting_iter, SegmentOffsetCreator<idxT>(len));
  thrust::counting_iterator<idxT> counting_iter(0);
  thrust::transform_iterator<SegmentOffsetCreator<idxT>,
                             thrust::counting_iterator<idxT>>
      segment_offsets(counting_iter, SegmentOffsetCreator<idxT>(len));

  size_t cub_buf_size = 0;
  if (greater) {
    cub::DeviceSegmentedSort::SortPairsDescending(
        nullptr, cub_buf_size, in, out_for_sort, in_idx, out_idx_for_sort,
        total, batch_size, segment_offsets, segment_offsets + 1);
  } else {
    cub::DeviceSegmentedSort::SortPairs(
        nullptr, cub_buf_size, in, out_for_sort, in_idx, out_idx_for_sort,
        total, batch_size, segment_offsets, segment_offsets + 1);
  }
  std::vector<size_t> sizes = {total * sizeof(*in_idx),
                               total * sizeof(*out_for_sort),
                               total * sizeof(*out_idx_for_sort), cub_buf_size};
  size_t total_size = calc_aligned_size(sizes);
  if (!buf) {
    buf_size = total_size;
    return;
  }
  std::vector<void *> aligned_pointers = calc_aligned_pointers(buf, sizes);
  in_idx = static_cast<idxT *>(aligned_pointers[0]);
  out_for_sort = static_cast<T *>(aligned_pointers[1]);
  out_idx_for_sort = static_cast<idxT *>(aligned_pointers[2]);
  cub_buf = aligned_pointers[3];

  init_idx<<<(len - 1) / BLOCK_DIM + 1, BLOCK_DIM, 0, stream>>>(
      in_idx, batch_size, len);
  if (greater) {
    cub::DeviceSegmentedSort::SortPairsDescending(
        cub_buf, cub_buf_size, in, out_for_sort, in_idx, out_idx_for_sort,
        total, batch_size, segment_offsets, segment_offsets + 1, stream);
  } else {
    cub::DeviceSegmentedSort::SortPairs(
        cub_buf, cub_buf_size, in, out_for_sort, in_idx, out_idx_for_sort,
        total, batch_size, segment_offsets, segment_offsets + 1, stream);
  }

  copy_result<<<(k - 1) / BLOCK_DIM + 1, BLOCK_DIM, 0, stream>>>(
      out_for_sort, out_idx_for_sort, out, out_idx, batch_size, len, k);
}

} // namespace nv
#endif
