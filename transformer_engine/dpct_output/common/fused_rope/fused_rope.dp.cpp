/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <dpct/dnnl_utils.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <transformer_engine/fused_rope.h>

#include "../common.h"
#include "../util/logging.h"
#include "../utils.dp.hpp"

namespace transformer_engine {

template <typename scalar_t>
void fused_rope_block_forward(
    const scalar_t *src, const float *freqs, scalar_t *dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d,
    const sycl::nd_item<3> &item_ct1) {
  int s_id = item_ct1.get_group(2);
#pragma unroll
  for (int d_id = item_ct1.get_local_id(2); d_id < d2;
       d_id += item_ct1.get_local_range(2)) {
    float v_cos, v_sin;
    v_sin = sycl::sincos(
        freqs[s_id * d2 + d_id],
        sycl::address_space_cast<sycl::access::address_space::private_space,
                                 sycl::access::decorated::yes>(&v_cos));
#pragma unroll
    for (int h_id = item_ct1.get_local_id(1); h_id < h;
         h_id += item_ct1.get_local_range(1)) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
                                  : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      dst[offset_dst] =
          v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // copy the rest
  if (d > d2) {
#pragma unroll
    for (int h_id = item_ct1.get_local_id(1); h_id < h;
         h_id += item_ct1.get_local_range(1)) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + item_ct1.get_local_id(2); d_id < d;
           d_id += item_ct1.get_local_range(2)) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
void fused_rope_block_backward(
    const scalar_t *src, const float *freqs, scalar_t *dst,
    const int offset_block, const int offset_block_dst, const int h,
    const int d, const int d2, const int stride_h, const int stride_d,
    const int o_stride_h, const int o_stride_d,
    const sycl::nd_item<3> &item_ct1) {
  int s_id = item_ct1.get_group(2);
#pragma unroll
  for (int d_id = item_ct1.get_local_id(2); d_id < d2;
       d_id += item_ct1.get_local_range(2)) {
    float v_cos = sycl::cos((float)(freqs[s_id * d2 + d_id]));
    float v_sin =
        (d_id + d2 / 2 < d2)
            ? sycl::sin((float)(freqs[s_id * d2 + d_id + d2 / 2]))
            : -sycl::sin((float)(freqs[s_id * d2 + d_id + d2 / 2 - d2]));
#pragma unroll
    for (int h_id = item_ct1.get_local_id(1); h_id < h;
         h_id += item_ct1.get_local_range(1)) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate = (d_id + d2 / 2 < d2)
                                  ? src[offset_src + (d2 / 2) * stride_d]
                                  : src[offset_src + (d2 / 2 - d2) * stride_d];
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }

  // handle the tail
  if (d > d2) {
#pragma unroll
    for (int h_id = item_ct1.get_local_id(1); h_id < h;
         h_id += item_ct1.get_local_range(1)) {
      int offset_head = offset_block + h_id * stride_h;
      int offset_head_dst = offset_block_dst + h_id * o_stride_h;
#pragma unroll
      for (int d_id = d2 + item_ct1.get_local_id(2); d_id < d;
           d_id += item_ct1.get_local_range(2)) {
        dst[offset_head_dst + d_id * o_stride_d] =
            src[offset_head + d_id * stride_d];
      }
    }
  }
}

template <typename scalar_t>
void fused_rope_forward_kernel(
    const scalar_t *src, const float *freqs, scalar_t *dst, const int h,
    const int d, const int d2, const int stride_s, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s,
    const int o_stride_b, const int o_stride_h, const int o_stride_d,
    const sycl::nd_item<3> &item_ct1) {
  int s_id = item_ct1.get_group(2), b_id = item_ct1.get_group(1);
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_forward(src, freqs, dst, offset_block, offset_block_dst, h,
                           d, d2, stride_h, stride_d, o_stride_h, o_stride_d,
                           item_ct1);
}

template <typename scalar_t>
void fused_rope_backward_kernel(
    const scalar_t *src, const float *freqs, scalar_t *dst, const int h,
    const int d, const int d2, const int stride_s, const int stride_b,
    const int stride_h, const int stride_d, const int o_stride_s,
    const int o_stride_b, const int o_stride_h, const int o_stride_d,
    const sycl::nd_item<3> &item_ct1) {
  int s_id = item_ct1.get_group(2), b_id = item_ct1.get_group(1);
  int offset_block = s_id * stride_s + b_id * stride_b;
  int offset_block_dst = s_id * o_stride_s + b_id * o_stride_b;
  fused_rope_block_backward(src, freqs, dst, offset_block, offset_block_dst, h,
                            d, d2, stride_h, stride_d, o_stride_h, o_stride_d,
                            item_ct1);
}

template <typename scalar_t>
void fused_rope_thd_forward_kernel(
    const scalar_t *src, const int *cu_seqlens, const float *freqs,
    scalar_t *dst, const int h, const int d, const int d2, const int stride_t,
    const int stride_h, const int stride_d, const int o_stride_t,
    const int o_stride_h, const int o_stride_d,
    const sycl::nd_item<3> &item_ct1) {
  int s_id = item_ct1.get_group(2), b_id = item_ct1.get_group(1);
  int t_id = s_id + cu_seqlens[b_id];
  if (t_id >= cu_seqlens[b_id + 1]) return;
  int offset_block = t_id * stride_t;
  int offset_block_dst = t_id * o_stride_t;
  fused_rope_block_forward(src, freqs, dst, offset_block, offset_block_dst, h,
                           d, d2, stride_h, stride_d, o_stride_h, o_stride_d,
                           item_ct1);
}

template <typename scalar_t>
void fused_rope_thd_backward_kernel(
    const scalar_t *src, const int *cu_seqlens, const float *freqs,
    scalar_t *dst, const int h, const int d, const int d2, const int stride_t,
    const int stride_h, const int stride_d, const int o_stride_t,
    const int o_stride_h, const int o_stride_d,
    const sycl::nd_item<3> &item_ct1) {
  int s_id = item_ct1.get_group(2), b_id = item_ct1.get_group(1);
  int t_id = s_id + cu_seqlens[b_id];
  if (t_id >= cu_seqlens[b_id + 1]) return;
  int offset_block = t_id * stride_t;
  int offset_block_dst = t_id * o_stride_t;
  fused_rope_block_backward(src, freqs, dst, offset_block, offset_block_dst, h,
                            d, d2, stride_h, stride_d, o_stride_h, o_stride_d,
                            item_ct1);
}

template <typename scalar_t>
void fused_rope_forward_launcher(const scalar_t *input, const float *freqs,
                                 scalar_t *output, const int s, const int b,
                                 const int h, const int d, const int d2,
                                 const int stride_s, const int stride_b,
                                 const int stride_h, const int stride_d,
                                 const int o_stride_s, const int o_stride_b,
                                 const int o_stride_h, const int o_stride_d,
                                 dpct::queue_ptr stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  sycl::range<3> blocks(1, b, s);
  sycl::range<3> threads(1, warps_per_block, THREADS_PER_WARP);

  /*
  DPCT1049:142: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 fused_rope_forward_kernel(
                                     input, freqs, output, h, d, d2, stride_s,
                                     stride_b, stride_h, stride_d, o_stride_s,
                                     o_stride_b, o_stride_h, o_stride_d,
                                     item_ct1);
                             });
    }
  /*
  DPCT1010:496: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  /*
  DPCT1009:497: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  NVTE_CHECK_CUDA(0);
}

template <typename scalar_t>
void fused_rope_backward_launcher(
    const scalar_t *output_grads, const float *freqs, scalar_t *input_grads,
    const int s, const int b, const int h, const int d, const int d2,
    const int stride_s, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s, const int o_stride_b,
    const int o_stride_h, const int o_stride_d, dpct::queue_ptr stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  sycl::range<3> blocks(1, b, s);
  sycl::range<3> threads(1, warps_per_block, THREADS_PER_WARP);

  /*
  DPCT1049:143: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 fused_rope_backward_kernel(
                                     output_grads, freqs, input_grads, h, d, d2,
                                     stride_s, stride_b, stride_h, stride_d,
                                     o_stride_s, o_stride_b, o_stride_h,
                                     o_stride_d, item_ct1);
                             });
    }
  /*
  DPCT1010:498: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  /*
  DPCT1009:499: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  NVTE_CHECK_CUDA(0);
}

template <typename scalar_t>
void fused_rope_thd_forward_launcher(
    const scalar_t *input, const int *cu_seqlens, const float *freqs,
    scalar_t *output, const int max_s, const int b, const int h, const int d,
    const int d2, const int stride_t, const int stride_h, const int stride_d,
    const int o_stride_t, const int o_stride_h, const int o_stride_d,
    dpct::queue_ptr stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  sycl::range<3> blocks(1, b, max_s);
  sycl::range<3> threads(1, warps_per_block, THREADS_PER_WARP);

  /*
  DPCT1049:144: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 fused_rope_thd_forward_kernel(
                                     input, cu_seqlens, freqs, output, h, d, d2,
                                     stride_t, stride_h, stride_d, o_stride_t,
                                     o_stride_h, o_stride_d, item_ct1);
                             });
    }
  /*
  DPCT1010:500: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  /*
  DPCT1009:501: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  NVTE_CHECK_CUDA(0);
}

template <typename scalar_t>
void fused_rope_thd_backward_launcher(
    const scalar_t *output_grads, const int *cu_seqlens, const float *freqs,
    scalar_t *input_grads, const int max_s, const int b, const int h,
    const int d, const int d2, const int stride_t, const int stride_h,
    const int stride_d, const int o_stride_t, const int o_stride_h,
    const int o_stride_d, dpct::queue_ptr stream) {
  int warps_per_block = h < 16 ? 4 : 8;
  sycl::range<3> blocks(1, b, max_s);
  sycl::range<3> threads(1, warps_per_block, THREADS_PER_WARP);

  /*
  DPCT1049:145: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(blocks * threads, threads),
                             [=](sycl::nd_item<3> item_ct1) {
                                 fused_rope_thd_backward_kernel(
                                     output_grads, cu_seqlens, freqs,
                                     input_grads, h, d, d2, stride_t, stride_h,
                                     stride_d, o_stride_t, o_stride_h,
                                     o_stride_d, item_ct1);
                             });
    }
  /*
  DPCT1010:502: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  /*
  DPCT1009:503: SYCL uses exceptions to report errors and does not use the error
  codes. The original code was commented out and a warning string was inserted.
  You need to rewrite this code.
  */
  NVTE_CHECK_CUDA(0);
}

void fused_rope_forward(const Tensor &input, const Tensor &freqs,
                        Tensor *output, const int s, const int b, const int h,
                        const int d, const int d2, const int stride_s,
                        const int stride_b, const int stride_h,
                        const int stride_d, const int o_stride_s,
                        const int o_stride_b, const int o_stride_h,
                        const int o_stride_d, dpct::queue_ptr stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      fused_rope_forward_launcher(
          reinterpret_cast<const scalar_t *>(input.data.dptr),
          reinterpret_cast<const float *>(freqs.data.dptr),
          reinterpret_cast<scalar_t *>(output->data.dptr), s, b, h, d, d2,
          stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
          o_stride_h, o_stride_d, stream););
}

void fused_rope_backward(const Tensor &output_grads, const Tensor &freqs,
                         Tensor *input_grads, const int s, const int b,
                         const int h, const int d, const int d2,
                         const int stride_s, const int stride_b,
                         const int stride_h, const int stride_d,
                         const int o_stride_s, const int o_stride_b,
                         const int o_stride_h, const int o_stride_d,
                         dpct::queue_ptr stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      output_grads.data.dtype, scalar_t,
      fused_rope_backward_launcher(
          reinterpret_cast<const scalar_t *>(output_grads.data.dptr),
          reinterpret_cast<const float *>(freqs.data.dptr),
          reinterpret_cast<scalar_t *>(input_grads->data.dptr), s, b, h, d, d2,
          stride_s, stride_b, stride_h, stride_d, o_stride_s, o_stride_b,
          o_stride_h, o_stride_d, stream););
}

void fused_rope_thd_forward(const Tensor &input, const Tensor &cu_seqlens,
                            const Tensor &freqs, Tensor *output,
                            const int max_s, const int b, const int h,
                            const int d, const int d2, const int stride_t,
                            const int stride_h, const int stride_d,
                            const int o_stride_t, const int o_stride_h,
                            const int o_stride_d, dpct::queue_ptr stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      input.data.dtype, scalar_t,
      fused_rope_thd_forward_launcher(
          reinterpret_cast<const scalar_t *>(input.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(freqs.data.dptr),
          reinterpret_cast<scalar_t *>(output->data.dptr), max_s, b, h, d, d2,
          stride_t, stride_h, stride_d, o_stride_t, o_stride_h, o_stride_d,
          stream););
}

void fused_rope_thd_backward(const Tensor &output_grads,
                             const Tensor &cu_seqlens, const Tensor &freqs,
                             Tensor *input_grads, const int max_s, const int b,
                             const int h, const int d, const int d2,
                             const int stride_t, const int stride_h,
                             const int stride_d, const int o_stride_t,
                             const int o_stride_h, const int o_stride_d,
                             dpct::queue_ptr stream) {
  TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT(
      output_grads.data.dtype, scalar_t,
      fused_rope_thd_backward_launcher(
          reinterpret_cast<const scalar_t *>(output_grads.data.dptr),
          reinterpret_cast<const int *>(cu_seqlens.data.dptr),
          reinterpret_cast<const float *>(freqs.data.dptr),
          reinterpret_cast<scalar_t *>(input_grads->data.dptr), max_s, b, h, d,
          d2, stride_t, stride_h, stride_d, o_stride_t, o_stride_h, o_stride_d,
          stream););
}

}  // end namespace transformer_engine

void nvte_fused_rope_forward(const NVTETensor input, const NVTETensor freqs,
                             NVTETensor output, const int s, const int b,
                             const int h, const int d, const int d2,
                             const int stride_s, const int stride_b,
                             const int stride_h, const int stride_d,
                             const int o_stride_s, const int o_stride_b,
                             const int o_stride_h, const int o_stride_d,
                             dpct::queue_ptr stream) {
  NVTE_API_CALL(nvte_fused_rope_forward);
  using namespace transformer_engine;
  fused_rope_forward(*reinterpret_cast<const Tensor *>(input),
                     *reinterpret_cast<const Tensor *>(freqs),
                     reinterpret_cast<Tensor *>(output), s, b, h, d, d2,
                     stride_s, stride_b, stride_h, stride_d, o_stride_s,
                     o_stride_b, o_stride_h, o_stride_d, stream);
}

void nvte_fused_rope_backward(const NVTETensor output_grads,
                              const NVTETensor freqs, NVTETensor input_grads,
                              const int s, const int b, const int h,
                              const int d, const int d2, const int stride_s,
                              const int stride_b, const int stride_h,
                              const int stride_d, const int o_stride_s,
                              const int o_stride_b, const int o_stride_h,
                              const int o_stride_d, dpct::queue_ptr stream) {
  NVTE_API_CALL(nvte_fused_rope_backward);
  using namespace transformer_engine;
  fused_rope_backward(*reinterpret_cast<const Tensor *>(output_grads),
                      *reinterpret_cast<const Tensor *>(freqs),
                      reinterpret_cast<Tensor *>(input_grads), s, b, h, d, d2,
                      stride_s, stride_b, stride_h, stride_d, o_stride_s,
                      o_stride_b, o_stride_h, o_stride_d, stream);
}

void nvte_fused_rope_thd_forward(const NVTETensor input,
                                 const NVTETensor cu_seqlens,
                                 const NVTETensor freqs, NVTETensor output,
                                 const int max_s, const int b, const int h,
                                 const int d, const int d2, const int stride_t,
                                 const int stride_h, const int stride_d,
                                 const int o_stride_t, const int o_stride_h,
                                 const int o_stride_d, dpct::queue_ptr stream) {
  NVTE_API_CALL(nvte_fused_rope_thd_forward);
  using namespace transformer_engine;
  fused_rope_thd_forward(*reinterpret_cast<const Tensor *>(input),
                         *reinterpret_cast<const Tensor *>(cu_seqlens),
                         *reinterpret_cast<const Tensor *>(freqs),
                         reinterpret_cast<Tensor *>(output), max_s, b, h, d, d2,
                         stride_t, stride_h, stride_d, o_stride_t, o_stride_h,
                         o_stride_d, stream);
}

void nvte_fused_rope_thd_backward(
    const NVTETensor output_grads, const NVTETensor cu_seqlens,
    const NVTETensor freqs, NVTETensor input_grads, const int max_s,
    const int b, const int h, const int d, const int d2, const int stride_t,
    const int stride_h, const int stride_d, const int o_stride_t,
    const int o_stride_h, const int o_stride_d, dpct::queue_ptr stream) {
  NVTE_API_CALL(nvte_fused_rope_thd_backward);
  using namespace transformer_engine;
  fused_rope_thd_backward(*reinterpret_cast<const Tensor *>(output_grads),
                          *reinterpret_cast<const Tensor *>(cu_seqlens),
                          *reinterpret_cast<const Tensor *>(freqs),
                          reinterpret_cast<Tensor *>(input_grads), max_s, b, h,
                          d, d2, stride_t, stride_h, stride_d, o_stride_t,
                          o_stride_h, o_stride_d, stream);
}
