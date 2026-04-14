#ifndef CPU_TOPK_H_
#define CPU_TOPK_H_
#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

namespace nv {

// TODO: extract the malloc of in_idx to make it a fair comparison with GPU topk
template <typename T, typename idxT, typename Compare>
void cpu_topk(const T *in, int batch_size, idxT len, idxT k, T *out,
              idxT *out_idx, Compare cmp, bool is_stable = false) {
  if (!out_idx) {
    for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
      std::partial_sort_copy(in + i * len, in + (i + 1) * len, out + i * k,
                             out + (i + 1) * k, cmp);
    }
    return;
  }

  std::vector<idxT> in_idx(len);
  for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
    std::function<bool(idxT, idxT)> func_default = [in, i, len, cmp](idxT a,
                                                                     idxT b) {
      return cmp(in[i * len + a], in[i * len + b]);
    };
    std::function<bool(idxT, idxT)> func_stable = [in, i, len, cmp](idxT a,
                                                                    idxT b) {
      bool res = cmp(in[i * len + a], in[i * len + b]);
      if (std::equal_to<T>{}(in[i * len + a], in[i * len + b])) {
        res = std::less<idxT>{}(a, b);
      }
      return res;
    };
    std::function<bool(idxT, idxT)> *func_ptr = &func_default;
    if (is_stable) {
      func_ptr = &func_stable;
    }

    std::iota(in_idx.begin(), in_idx.end(), 0);

    std::partial_sort_copy(in_idx.cbegin(), in_idx.cend(), out_idx + i * k,
                           out_idx + (i + 1) * k, *func_ptr);

    std::transform(out_idx + i * k, out_idx + (i + 1) * k, out + i * k,
                   [in, i, len](idxT j) { return in[i * len + j]; });
  }
}

// has different API from GPU topk
// it uses host pointer rather than device pointer anyway
template <typename T, typename idxT>
void cpu_topk(const T *in, int batch_size, idxT len, idxT k, T *out,
              idxT *out_idx = nullptr, bool greater = true,
              bool is_stable = false) {
  if (greater) {
    cpu_topk<T, idxT, std::greater<T>>(in, batch_size, len, k, out, out_idx,
                                       std::greater<T>(), is_stable);
  } else {
    cpu_topk<T, idxT, std::less<T>>(in, batch_size, len, k, out, out_idx,
                                    std::less<T>(), is_stable);
  }
}

} // namespace nv
#endif
