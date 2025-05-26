/*************************************************************************
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/comm_gemm_overlap.h>
#include <transformer_engine/gemm.h>
#include <transformer_engine/transformer_engine.h>

#include <cassert>
#include <numeric>

#include "common/common.h"
#include "common/util/cuda_driver.h"
#include "common/util/cuda_runtime.h"
#include "common/util/logging.h"
#include "common/util/system.h"
#include "userbuffers/userbuffers.h"

#define HALF_BYTES 2
#define UB_MAX_SM 32

using namespace std::placeholders;

namespace transformer_engine {

namespace {

std::vector<size_t> shape_to_vector(const NVTEShape &shape) {
  return std::vector<size_t>(shape.data, shape.data + shape.ndim);
}

}  // namespace

/***************************************************************************************************
 * Comm+GEMM Overlap Common Core
 **************************************************************************************************/

bool ubuf_built_with_mpi() {
#ifdef NVTE_UB_WITH_MPI
  return true;
#else
  return false;
#endif
}

CommOverlapCore::CommOverlapCore(int myrank, int numranks, int mylocal, int numlocal, int mynode,
                                 int numnodes, int tp_size, ExtAllgatherOp allgather_handle,
                                 ExtBarrierOp barrier_handle, int num_splits, int num_max_streams,
                                 int comm_cga_size, int gemm_priority, int comm_priority,
                                 int num_comm_sm, bool set_sm_margin, bool use_ce,
                                 bool atomic_gemm) {
  // Initialize userbuf communicator
  if (!_comm_created) {
    if (myrank == 0) {
      printf("!!! [UB] Create Userbuffers Communicator\n");
    }
#ifdef NVTE_UB_WITH_MPI
    create_communicator_grouped2_mpi(&_ub_comm, 1, 1, tp_size, 1);
#else
    create_communicator_grouped2(&_ub_comm, myrank, numranks, mylocal, numlocal, mynode, numnodes,
                                 allgather_handle, barrier_handle, 1, 1, tp_size, 1);
#endif
    _comm_created = true;
  }
  _use_ce = static_cast<int>(use_ce);
  _num_comm_sm = num_comm_sm;
  _cga_size = comm_cga_size;

  if (gemm_priority == 0 && comm_priority == 0) {
    transformer_engine::cuda::stream_priority_range(&_gemm_priority, &_comm_priority);
  } else {
    _gemm_priority = gemm_priority;
    _comm_priority = comm_priority;
  }
  for (int i = 0; i < std::min(num_max_streams, num_splits); i++) {
    cudaStream_t stream;
    NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, _gemm_priority));
    _stream_compute.push_back(std::move(stream));
  }

  _num_splits = num_splits;
  _rank = _ub_comm->myrank;
  _tp_size = tp_size;
  _tp_id = _rank % _tp_size;

  // Set the number of SMs for GEMM with margin
  int sm_count = transformer_engine::cuda::sm_count();
  _math_sms = (set_sm_margin) ? sm_count - num_comm_sm : sm_count;
  _math_sms -= transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", 0);

  _atomic_gemm = atomic_gemm;
  if (_atomic_gemm) {
    void *counter_ptr;
    size_t counter_bytes = _num_splits * 2 * sizeof(int32_t);
    NVTE_CHECK_CUDA(cudaMalloc(&counter_ptr, counter_bytes));
    NVTE_CHECK_CUDA(cudaMemset(counter_ptr, 0, counter_bytes));
    NVTE_CHECK_CUDA(cudaMemset(counter_ptr, 1, counter_bytes / 2));
    _counter = TensorWrapper(counter_ptr, std::vector<size_t>{static_cast<size_t>(_num_splits * 2)},
                             DType::kInt32);
  }
  // CUDA event creation
  cudaEventCreateWithFlags(&_start_compute, 0);
  cudaEventCreateWithFlags(&_stop_compute, 0);
  cudaEventCreateWithFlags(&_start_comm, 0);
  cudaEventCreateWithFlags(&_stop_comm, 0);

  /*
    Defining the launcher order between the communication and GEMM kernels
    using Fast Dependent Launch when CUDA_DEVICE_MAX_CONNECTIONS>1.
    The event is used to schedule the communication kernel before the GEMM.
    This is needed only for Hopper, which uses persistent CTA execution.
  */
  int max_connection = transformer_engine::getenv<int>("CUDA_DEVICE_MAX_CONNECTIONS", 8);
  int runtime_version = 0;
  cudaRuntimeGetVersion(&runtime_version);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  if (runtime_version >= 12030 && deviceProp.major == 9 && max_connection > 1) {
    cudaEventCreateWithFlags(&_comm_launch_event, cudaEventDisableTiming);
  } else {
    _comm_launch_event = 0;
  }
}

CommOverlapCore::~CommOverlapCore() {
  cudaEventDestroy(_stop_comm);
  cudaEventDestroy(_start_comm);
  cudaEventDestroy(_stop_compute);
  cudaEventDestroy(_start_compute);
  if (_comm_launch_event) cudaEventDestroy(_comm_launch_event);

  if (_atomic_gemm) cudaFree(_counter.dptr());

  for (size_t i = 0; i < _stream_compute.size(); i++) cudaStreamDestroy(_stream_compute[i]);

  if (_comm_created) {
#ifdef NVTE_UB_WITH_MPI
    destroy_communicator_mpi(_ub_comm);
#else
    destroy_communicator(_ub_comm);
#endif
    _comm_created = false;
  }
}

TensorWrapper CommOverlapCore::get_tensor_chunk(const TensorWrapper &source, size_t chunk_offset,
                                                const std::vector<size_t> &chunk_shape) {
  const auto scaling_mode = source.scaling_mode();

  // Tensor dimensions
  std::vector<size_t> shape = shape_to_vector(source.shape());
  auto flatten_shape_to_2d = [](const std::vector<size_t> &shape) -> std::pair<size_t, size_t> {
    if (shape.empty()) {
      return {1, 1};
    }
    size_t height = 1;
    for (size_t i = 0; i < shape.size() - 1; ++i) {
      height *= shape[i];
    }
    return {height, shape.back()};
  };
  size_t height, width, chunk_height, chunk_width;
  std::tie(height, width) = flatten_shape_to_2d(shape);
  std::tie(chunk_height, chunk_width) = flatten_shape_to_2d(chunk_shape);

  // Check tensor dimensions
#define NVTE_DIM_CHECK(cond, message)                                                \
  NVTE_CHECK(cond, message, " (tensor shape=", shape, ", chunk shape=", chunk_shape, \
             ", chunk offset=", chunk_offset, ")")
  NVTE_DIM_CHECK(height > 0 && width > 0, "Attempted to get chunk from empty tensor");
  NVTE_DIM_CHECK(chunk_height > 0 && chunk_width > 0, "Attempted to get empty tensor chunk");
  NVTE_DIM_CHECK(chunk_height <= height && chunk_width <= width,
                 "Attempted to get out-of-bounds tensor chunk");
  if (scaling_mode == NVTEScalingMode::NVTE_MXFP8_1D_SCALING) {
    // MXFP8 scale-inverses are padded to a 2D matrix with dims that
    // are divisible by 128. UB doesn't handle this padding yet.
    NVTE_DIM_CHECK(height % 128 == 0 && width % 128 == 0,
                   "Userbuffers requires MXFP8 tensor dims that are divisible by 128");
    NVTE_DIM_CHECK(chunk_height % 128 == 0 && chunk_width % 128 == 0,
                   "Userbuffers requires MXFP8 tensor chunk dims that are divisible by 128");
  }
#undef NVTE_DIM_CHECK

  // Construct tensor chunk
  TensorWrapper chunk(scaling_mode);
  for (int param_id = 0; param_id < NVTETensorParam::kNVTENumTensorParams; param_id++) {
    auto param_type = static_cast<NVTETensorParam>(param_id);
    auto param = source.get_parameter(param_type);
    auto param_dptr = reinterpret_cast<char *>(param.data_ptr);
    auto param_dtype = static_cast<DType>(param.dtype);
    auto param_shape = shape_to_vector(param.shape);

    if (param_dptr != nullptr) {
      if (param_type == NVTETensorParam::kNVTERowwiseData ||
          param_type == NVTETensorParam::kNVTEColumnwiseData) {
        // Offset data pointer
        param_dptr += chunk_offset * typeToSize(param_dtype);
        param_shape = chunk_shape;

        if (param_type == NVTETensorParam::kNVTEColumnwiseData &&
            source.scaling_mode() == NVTEScalingMode::NVTE_DELAYED_TENSOR_SCALING) {
          // Columnwise shape for FP8 tensor-scaled tensors shifts the last dimension to the front
          auto last_dim = param_shape.back();
          param_shape.pop_back();
          param_shape.insert(param_shape.begin(), last_dim);
        }
      } else if (source.scaling_mode() == NVTEScalingMode::NVTE_MXFP8_1D_SCALING &&
                 (param_type == NVTETensorParam::kNVTERowwiseScaleInv ||
                  param_type == NVTETensorParam::kNVTEColumnwiseScaleInv)) {
        // Calculate offset and size for MXFP8 scale-invs
        size_t chunk_scale_height = chunk_height;
        size_t chunk_scale_width = chunk_width;
        if (param_type == NVTETensorParam::kNVTERowwiseScaleInv) {
          chunk_scale_width /= 32;
        } else {
          chunk_scale_height /= 32;
        }
        param_dptr += (chunk_offset / 32) * typeToSize(param_dtype);
        param_shape = {chunk_scale_height, chunk_scale_width};
      }

      // Set chunked source parameters into the chunked tensor output
      chunk.set_parameter(param_type, reinterpret_cast<void *>(param_dptr), param_dtype,
                          param_shape);
    }
  }
  return chunk;
}

TensorWrapper CommOverlapCore::get_buffer_chunk_like(const TensorWrapper &source,
                                                     size_t chunk_offset,
                                                     const std::vector<size_t> &chunk_shape) {
  // Start with a chunk of the source tensor
  auto chunk = get_tensor_chunk(source, chunk_offset, chunk_shape);

  // Update chunk with offset data pointers from the communication buffer
  auto ubuf_ptr = reinterpret_cast<char *>(_ubuf.dptr()) + (chunk_offset * _ubuf.element_size());
  if (chunk.dptr() != nullptr) {
    chunk.set_rowwise_data(reinterpret_cast<void *>(ubuf_ptr), chunk.dtype(), chunk.shape());
  }
  if (chunk.columnwise_dptr() != nullptr) {
    chunk.set_columnwise_data(reinterpret_cast<void *>(ubuf_ptr), chunk.dtype(),
                              chunk.columnwise_shape());
  }
  return chunk;
}

/***************************************************************************************************
 * Comm+GEMM Overlap Base (Pipelined / Collective)
 **************************************************************************************************/

CommOverlapBase::CommOverlapBase(const std::vector<size_t> &buffer_shape, DType buffer_dtype,
                                 int myrank, int numranks, int mylocal, int numlocal, int mynode,
                                 int numnodes, int tp_size, ExtAllgatherOp allgather_handle,
                                 ExtBarrierOp barrier_handle, int num_splits, int num_max_streams,
                                 int comm_cga_size, int gemm_priority, int comm_priority,
                                 int num_comm_sm, bool set_sm_margin, bool atomic_gemm,
                                 bool rs_overlap_first_gemm)
    : CommOverlapCore(myrank, numranks, mylocal, numlocal, mynode, numnodes, tp_size,
                      allgather_handle, barrier_handle, num_splits, num_max_streams, comm_cga_size,
                      gemm_priority, comm_priority, num_comm_sm, set_sm_margin, false,
                      atomic_gemm) {
  _rs_overlap_first_gemm = rs_overlap_first_gemm;
  _rs_kernel_type = getenv<int>("NVTE_RS_STRIDED_ATOMIC", 0);
  NVTE_CHECK(_rs_kernel_type >= 0 && _rs_kernel_type <= 3,
             "Invalid choice for NVTE_RS_STRIDED_ATOMIC: Must be 0 (non-atomic), 1 (atomic) ",
             "or 2 (multi-atomic).");

  NVTE_CHECK(buffer_shape.size() == 2, "Userbuffer shape must be 2-dimensional!");
  size_t buffer_bytes = buffer_shape[0] * buffer_shape[1] * typeToSize(buffer_dtype);
  void *buffer_ptr;
  _ub_reg = register_user_buffer_collective(&buffer_ptr, buffer_bytes, _ub_comm, true);
  if (_ub_comm->myrank == 0) printf("!!! [UB] Register UBuf %d\n", _ub_reg);
  _ubuf = TensorWrapper(buffer_ptr, buffer_shape, buffer_dtype);

  NVTE_CHECK_CUDA(
      cudaStreamCreateWithPriority(&_stream_comm, cudaStreamNonBlocking, _comm_priority));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_start_d2dcopy, 0));
}

CommOverlapBase::~CommOverlapBase() {
  cudaEventDestroy(_start_d2dcopy);
  cudaStreamDestroy(_stream_comm);
}

/*
** Bulk GEMM + COMM
** This function assumes the communication input is pre-copied to _ubuf
*/
void CommOverlapBase::bulk_overlap(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                   bool transb, TensorWrapper &D, TensorWrapper &bias,
                                   TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
                                   bool accumulate, bool use_split_accumulator,
                                   CommOverlapType comm_type, TensorWrapper &rs_output,
                                   cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[0], _start_comm, 0));

  // Communication: AG and RS
  int comm_elements = (_ubuf.numel() / 2) * _ubuf.element_size();  // UBUF uses 2Byte element size
  if (comm_type == CommOverlapType::AG) {
    allgather2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, _stream_comm,
                                (cudaEvent_t)_comm_launch_event);
  } else {
    if (_ubuf.element_size() == 1) {
      assert(_ubuf_scale_inv_initialized);
      comm_elements *= 2;
      assert(rs_output.numel() == _ubuf.numel() / _tp_size);
      assert(rs_output.size(0) == _ubuf.size(0) / _tp_size);
      assert(rs_output.element_size() == 2);
      char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());
      reducescatter2_userbuff_fp8<__nv_fp8_e5m2>(rs_output_ptr, _ubuf.scale_inv(), _ub_reg, 0,
                                                 comm_elements, _ub_comm, _stream_comm,
                                                 (cudaEvent_t)_comm_launch_event);
    } else {
      reducescatter2_userbuff_inplace(_ub_reg, 0, comm_elements, _ub_comm, _stream_comm,
                                      (cudaEvent_t)_comm_launch_event);
    }
  }

  assert(pre_gelu_out.numel() == 0);
  // When the kernel launch order is defined, enforce the GEMM kernel launch to wait for the communication kernel launch
  if (_comm_launch_event)
    NVTE_CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[0], _comm_launch_event, 0));
  nvte_cublas_gemm(A.data(), B.data(), D.data(), bias.data(), pre_gelu_out.data(), transa, transb,
                   grad, workspace.data(), accumulate, use_split_accumulator, _math_sms,
                   _stream_compute[0]);

  _ub_comm->sms = ori_sms;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, _stream_comm));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, _stream_compute[0]));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));

}  // CommOverlapBase::bulk_overlap

/*
** Split FPROP GEMM + ReduceScatter
*/
void CommOverlapBase::atomic_gemm_overlap_rs(const TensorWrapper &A, bool transa,
                                             const TensorWrapper &B, bool transb, TensorWrapper &D,
                                             TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                                             TensorWrapper &workspace, bool grad, bool accumulate,
                                             bool use_split_accumulator, TensorWrapper &rs_output,
                                             cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;
  // Get GEMM dimensions
  size_t m = transa ? A.size(0) : A.size(1);
  size_t k = transa ? A.size(1) : A.size(0);
  size_t n = _ubuf.size(0);
  size_t m_chunk = m / _num_splits;
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Get input, output, and workspace data pointers
  char *input_a_chunk_ptr = reinterpret_cast<char *>(A.dptr());
  char *output_buf_chunk_ptr = reinterpret_cast<char *>(_ubuf.dptr());
  char *workspace_ptr = reinterpret_cast<char *>(workspace.dptr());
  char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());

  // Reset atomic counters
  int *counter_ptr = reinterpret_cast<int *>(_counter.dptr());
  reset_counters(counter_ptr, _num_splits, false, stream_main);

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[0], _start_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_compute, 0));

  assert(pre_gelu_out.numel() == 0);

  auto output_d = get_buffer_chunk_like(D, 0, {n, m});
  auto workspace_chunk = get_tensor_chunk(workspace, 0, {workspace_size_chunk});
  nvte_cublas_atomic_gemm(A.data(), B.data(), output_d.data(), bias.data(), pre_gelu_out.data(),
                          transa, transb, grad, workspace_chunk.data(), accumulate,
                          use_split_accumulator, _math_sms, _num_splits, 0, true, _counter.data(),
                          _stream_compute[0]);

  for (int i = 0; i < _num_splits; i++) {
    if (_rs_kernel_type == 1) {
      if (i == _num_splits - 1) {
        _ub_comm->sms = UB_MAX_SM;
      }
      if (_ubuf.element_size() == 1) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            D.dtype(), fp8_type,
            reducescatter2_userbuff_strided_atomic_fp8<fp8_type>(
                rs_output_ptr, D.scale_inv(), _ub_reg, i * m_chunk, m_chunk, n, m, m, _num_splits,
                &counter_ptr[i], _ub_comm, _stream_comm););
      } else {
        reducescatter2_userbuff_strided_atomic(rs_output_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                               _num_splits, &counter_ptr[i], _ub_comm,
                                               _stream_comm);
      }
    } else if (_rs_kernel_type == 2) {
      if (_ubuf.element_size() == 1) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            D.dtype(), fp8_type,
            reducescatter2_userbuff_strided_multiatomic_fp8<fp8_type>(
                rs_output_ptr, D.scale_inv(), _ub_reg, m_chunk, m_chunk, n, m, m, _num_splits,
                counter_ptr, _ub_comm, _stream_comm););
      } else {
        reducescatter2_userbuff_strided_multiatomic(rs_output_ptr, _ub_reg, m_chunk, m_chunk, n, m,
                                                    _num_splits, counter_ptr, _ub_comm,
                                                    _stream_comm);
      }
      break;
    } else {
      consumer(counter_ptr, i, _stream_comm);
      if (_ubuf.element_size() == 1) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            D.dtype(), fp8_type,
            reducescatter2_userbuff_stridedoutput_fp8<fp8_type>(rs_output_ptr, D.scale_inv(),
                                                                _ub_reg, i * m_chunk, m_chunk, n, m,
                                                                _ub_comm, _stream_comm););
      } else {
        reducescatter2_userbuff_strided(rs_output_ptr, _ub_reg, i * m_chunk, m_chunk, n, m,
                                        _ub_comm, _stream_comm);
      }
    }

    rs_output_ptr += m_chunk * rs_output.element_size();
  }

  _ub_comm->sms = ori_sms;
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[0]));
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, _stream_comm));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
}  // split_overlap_rs

/*
** Split FPROP GEMM + ReduceScatter
*/
void CommOverlapBase::split_overlap_rs(const TensorWrapper &A, bool transa, const TensorWrapper &B,
                                       bool transb, TensorWrapper &D, TensorWrapper &bias,
                                       TensorWrapper &pre_gelu_out, TensorWrapper &workspace,
                                       bool grad, bool accumulate, bool use_split_accumulator,
                                       TensorWrapper &rs_output, cudaStream_t stream_main) {
  // Get GEMM dimensions
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;
  size_t m = transa ? A.size(0) : A.size(1);
  size_t k = transa ? A.size(1) : A.size(0);
  size_t n = _ubuf.size(0);
  size_t m_chunk = m / _num_splits;
  const std::vector<size_t> input_a_chunk_shape =
      (transa ? std::vector<size_t>{m_chunk, k} : std::vector<size_t>{k, m_chunk});
  const std::vector<size_t> output_chunk_shape = {n, m_chunk};
  size_t input_a_chunk_size = m_chunk * k;
  size_t output_chunk_size = n * m_chunk;
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Helper function to get bias chunk if needed
  auto maybe_get_bias_chunk = [this, &bias, m_chunk](size_t chunk_id) -> TensorWrapper {
    if (bias.dptr() == nullptr) {
      return TensorWrapper();
    }
    return get_tensor_chunk(bias, chunk_id * m_chunk, {m_chunk});
  };

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_compute, 0));

  assert(pre_gelu_out.numel() == 0);

  char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());
  if (_rs_overlap_first_gemm) {
    auto input_a_chunk = get_tensor_chunk(A, 0, input_a_chunk_shape);
    auto output_chunk = get_buffer_chunk_like(D, 0, output_chunk_shape);
    auto bias_chunk = maybe_get_bias_chunk(0);
    auto workspace_chunk = get_tensor_chunk(workspace, 0, {workspace_size_chunk});

    nvte_cublas_gemm(input_a_chunk.data(), B.data(), output_chunk.data(), bias_chunk.data(),
                     pre_gelu_out.data(), transa, transb, grad, workspace_chunk.data(), accumulate,
                     use_split_accumulator, _math_sms, _stream_compute[0]);

    for (int i = 1; i < _num_splits; i++) {
      input_a_chunk = get_tensor_chunk(A, i * input_a_chunk_size, input_a_chunk_shape);
      output_chunk = get_buffer_chunk_like(D, i * output_chunk_size, output_chunk_shape);
      bias_chunk = maybe_get_bias_chunk(i);
      workspace_chunk = get_tensor_chunk(
          workspace, (i % _stream_compute.size()) * workspace_size_chunk, {workspace_size_chunk});

      nvte_cublas_gemm(input_a_chunk.data(), B.data(), output_chunk.data(), bias_chunk.data(),
                       pre_gelu_out.data(), transa, transb, grad, workspace_chunk.data(),
                       accumulate, use_split_accumulator, _math_sms,
                       _stream_compute[i % _stream_compute.size()]);

      NVTE_CHECK_CUDA(
          cudaEventRecord(_start_comm, _stream_compute[(i - 1) % _stream_compute.size()]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

      // Communication chunk
      if (_ubuf.element_size() == 1) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            D.dtype(), fp8_type,
            reducescatter2_userbuff_stridedoutput_fp8<fp8_type>(
                rs_output_ptr, D.scale_inv(), _ub_reg, (i - 1) * output_chunk_size, m_chunk, n, m,
                _ub_comm, _stream_comm););
      } else {
        reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, (i - 1) * output_chunk_size,
                                              m_chunk, n, m, _ub_comm, _stream_comm);
      }

      rs_output_ptr += m_chunk * rs_output.element_size();
    }
    int last_compute_stream_id =
        (_num_splits + _stream_compute.size() - 1) % _stream_compute.size();
    NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, _stream_compute[last_compute_stream_id]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

    // Last communication chunk with max SM
    _ub_comm->sms = UB_MAX_SM;
    if (_ubuf.element_size() == 1) {
      TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
          D.dtype(), fp8_type,
          reducescatter2_userbuff_stridedoutput_fp8<fp8_type>(
              rs_output_ptr, D.scale_inv(), _ub_reg, (_num_splits - 1) * output_chunk_size, m_chunk,
              n, m, _ub_comm, _stream_comm););
    } else {
      reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg,
                                            (_num_splits - 1) * output_chunk_size, m_chunk, n, m,
                                            _ub_comm, _stream_comm);
    }
  } else {
    for (int i = 0; i < _num_splits; i++) {
      auto input_a_chunk = get_tensor_chunk(A, i * input_a_chunk_size, input_a_chunk_shape);
      auto output_chunk = get_buffer_chunk_like(D, i * output_chunk_size, output_chunk_shape);
      auto bias_chunk = maybe_get_bias_chunk(i);
      auto workspace_chunk = get_tensor_chunk(
          workspace, (i % _stream_compute.size()) * workspace_size_chunk, {workspace_size_chunk});

      nvte_cublas_gemm(input_a_chunk.data(), B.data(), output_chunk.data(), bias_chunk.data(),
                       pre_gelu_out.data(), transa, transb, grad, workspace_chunk.data(),
                       accumulate, use_split_accumulator, _math_sms,
                       _stream_compute[i % _stream_compute.size()]);

      NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, _stream_compute[i % _stream_compute.size()]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_comm, _start_comm, 0));

      // Communication chunk. Uses MAX_SM at the last chunk
      if (i == _num_splits - 1) {
        _ub_comm->sms = UB_MAX_SM;
      }
      if (_ubuf.element_size() == 1) {
        TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
            D.dtype(), fp8_type,
            reducescatter2_userbuff_stridedoutput_fp8<fp8_type>(
                rs_output_ptr, D.scale_inv(), _ub_reg, i * output_chunk_size, m_chunk, n, m,
                _ub_comm, _stream_comm););
      } else {
        reducescatter2_userbuff_stridedoutput(rs_output_ptr, _ub_reg, i * output_chunk_size,
                                              m_chunk, n, m, _ub_comm, _stream_comm);
      }

      rs_output_ptr += m_chunk * rs_output.element_size();
    }
  }

  _ub_comm->sms = ori_sms;
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_comm, _stream_comm));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_comm, 0));
}  // CommOverlapBase::split_overlap_rs

/***************************************************************************************************
 * Comm+GEMM Overlap P2P Base (Ring-Exchange)
 **************************************************************************************************/

CommOverlapP2PBase::CommOverlapP2PBase(const std::vector<size_t> &buffer_shape, DType buffer_dtype,
                                       int myrank, int numranks, int mylocal, int numlocal,
                                       int mynode, int numnodes, int tp_size,
                                       ExtAllgatherOp allgather_handle, ExtBarrierOp barrier_handle,
                                       CommOverlapType comm_type, int num_max_streams,
                                       int comm_cga_size, int gemm_priority, int comm_priority,
                                       int num_comm_sm, bool set_sm_margin, bool use_ce,
                                       bool atomic_gemm, bool aggregate)
    : CommOverlapCore(myrank, numranks, mylocal, numlocal, mynode, numnodes, tp_size,
                      allgather_handle, barrier_handle, tp_size, num_max_streams, comm_cga_size,
                      gemm_priority, comm_priority, num_comm_sm, set_sm_margin, use_ce,
                      atomic_gemm) {
  _is_p2p = true;
  _is_reduce_scatter = comm_type == CommOverlapType::RS;
  _aggregate = aggregate;

  // Create workspace tensor with userbuffer
  NVTE_CHECK(buffer_shape.size() == 2, "Userbuffer shape must be 2-dimensional!");
  size_t buffer_bytes = buffer_shape[0] * buffer_shape[1] * typeToSize(buffer_dtype);
  int buffer_chunk_bytes = buffer_bytes / tp_size;
  _num_ubuf_chunks = tp_size;
  if (_is_reduce_scatter) {
    // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
    // outputs for reduction at the end of the pipelining.
    buffer_bytes = buffer_bytes / tp_size * (tp_size * 2 - 1);
    _num_ubuf_chunks = tp_size * 2 - 1;
  }

  void *buffer_ptr;
  _ub_reg = register_user_buffer_collective(&buffer_ptr, buffer_bytes, _ub_comm, true);
  if (_rank == 0) printf("!!! [UBP2P] Register UBuf %d\n", _ub_reg);
  _ubuf = TensorWrapper(
      buffer_ptr,
      std::vector<size_t>{buffer_shape[0] / tp_size * _num_ubuf_chunks, buffer_shape[1]},
      buffer_dtype);

  // Create tensor chunks for easy management
  char *ubuf_byte_ptr = reinterpret_cast<char *>(buffer_ptr);
  for (int i = 0; i < _num_ubuf_chunks; i++) {
    _ubufs.push_back(TensorWrapper(reinterpret_cast<void *>(ubuf_byte_ptr),
                                   std::vector<size_t>{buffer_shape[0] / tp_size, buffer_shape[1]},
                                   buffer_dtype));
    ubuf_byte_ptr += buffer_chunk_bytes;
  }

  _rank_round_tp = (_rank / _tp_size) * _tp_size;
  _next_rank = (_tp_size + _rank + 1) % _tp_size + _rank_round_tp;
  _prev_rank = (_tp_size + _rank + -1) % _tp_size + _rank_round_tp;

  _self_chunk_id = _tp_id;
  if (_atomic_gemm && !_is_reduce_scatter) {
    _use_multiatomic_ag = getenv<bool>("NVTE_AG_P2P_MULTI_ATOMIC");
    if (_use_multiatomic_ag) {
      _use_ce = 0;
      _ub_comm->push = 1;
      if (_rank == 0) {
        printf("!!userbuffers_sendrecv_multi_atomic_shuffle\n");
      }
    }
    _self_chunk_id = 0;
    NVTE_CHECK_CUDA(cudaMemset(_counter.dptr(), 0, sizeof(int32_t)));
  }

  for (int i = 0; i < std::min(num_max_streams, _tp_size); i++) {
    cudaStream_t stream;
    NVTE_CHECK_CUDA(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, _comm_priority));
    _stream_send.push_back(std::move(stream));
  }
  NVTE_CHECK_CUDA(
      cudaStreamCreateWithPriority(&_stream_recv, cudaStreamNonBlocking, _comm_priority));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_send, 0));
  NVTE_CHECK_CUDA(cudaEventCreateWithFlags(&_stop_recv, 0));
}

CommOverlapP2PBase::~CommOverlapP2PBase() {
  cudaEventDestroy(_stop_recv);
  cudaEventDestroy(_stop_send);
  cudaStreamDestroy(_stream_recv);
  for (size_t i = 0; i < _stream_send.size(); i++) cudaStreamDestroy(_stream_send[i]);
}

TensorWrapper CommOverlapP2PBase::get_buffer_chunk_by_id(const TensorWrapper &source,
                                                         size_t chunk_id) {
  // Start with a chunk of the source tensor
  auto chunk = get_tensor_chunk(source, 0, shape_to_vector(_ubufs[chunk_id].shape()));

  // Update chunk with offset data pointers from the communication buffer
  if (chunk.dptr() != nullptr) {
    chunk.set_rowwise_data(_ubufs[chunk_id].dptr(), chunk.dtype(), chunk.shape());
  }
  if (chunk.columnwise_dptr() != nullptr) {
    chunk.set_columnwise_data(_ubufs[chunk_id].dptr(), chunk.dtype(), chunk.columnwise_shape());
  }
  return chunk;
}

/*
** Split AllGather + AtomicGEMM using P2P communication
** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
*/
void CommOverlapP2PBase::atomic_gemm_overlap_ag(
    const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb, TensorWrapper &D,
    TensorWrapper &bias, TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
    bool accumulate, bool use_split_accumulator, TensorWrapper &B_copy, cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;

  // Get GEMM dimensions between TN and NN input layouts
  const size_t m = (transa) ? A.size(0) : A.size(1);
  const size_t n_chunk = _ubufs[0].size(0);
  assert(pre_gelu_out.numel() == 0);

  // Get communication and GEMM output chunk sizes
  const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();

  // Create an GEMM output buffer with N+1 chunks in a contiguous memory
  void *D_buffer_ptr;
  int D_chunk_bytes = n_chunk * m * D.element_size();
  NVTE_CHECK_CUDA(cudaMallocAsync(&D_buffer_ptr, (_tp_size + 1) * D_chunk_bytes, stream_main));
  auto D_buffer = TensorWrapper(D_buffer_ptr, D.shape(), D.dtype(), D.amax(), D.scale(),
                                D.scale_inv(), D.scale_inv_shape(), D.scaling_mode());

  // Reset atomic counters
  int *counter_ptr = reinterpret_cast<int *>(_counter.dptr());
  reset_counters(counter_ptr, _tp_size, true, stream_main);

  // Catch up the default torch stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[0], _start_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));

  auto input_b = get_buffer_chunk_like(B, 0, shape_to_vector(B.shape()));
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();
  auto workspace_chunk = get_tensor_chunk(workspace, 0, {workspace_size_chunk});

  for (int i = 0; i < _tp_size - 1; i++) {
    // Set the userbuffer id. Buffer under send is the input for the current
    // GEMM chunk The initial input chunk is stored _ubuf[rank]. This is to
    // have the AG output in all ranks to be contiguous after the ring
    // exchanges
    int send_chunk_id = i;
    int recv_chunk_id = i + 1;
    int send_offset = comm_bytes * send_chunk_id;
    int recv_offset = comm_bytes * recv_chunk_id;

    if (_use_multiatomic_ag) {
      if (i == 0) {
        _ub_comm->use_ce = 0;
        userbuffers_sendrecv_multiatomic(_ub_reg, _ub_reg, comm_bytes, comm_bytes, comm_bytes,
                                         _ub_comm, _next_rank, _prev_rank, _tp_size, counter_ptr,
                                         true, _stream_recv);
      }
    } else {
      userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, _next_rank,
                       _stream_recv);
      userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, _prev_rank,
                       _stream_recv);
      producer(counter_ptr, recv_chunk_id, _stream_recv);
    }
    if (i == 0) {
      nvte_cublas_atomic_gemm(A.data(), input_b.data(), D_buffer.data(), bias.data(),
                              pre_gelu_out.data(), transa, transb, grad, workspace_chunk.data(),
                              accumulate, use_split_accumulator, _math_sms, 0, _tp_size, false,
                              _counter.data(), stream_main);
    }
  }

  // Store the input activation for backprop
  if (B_copy.numel() > 0) {
    assert(B_copy.numel() == _ubufs[_self_chunk_id].numel());
    assert(B_copy.element_size() == _ubufs[_self_chunk_id].element_size());
    NVTE_CHECK_CUDA(
        cudaMemcpyAsync(B_copy.dptr(), _ubufs[_self_chunk_id].dptr(),
                        _ubufs[_self_chunk_id].numel() * _ubufs[_self_chunk_id].element_size(),
                        cudaMemcpyDeviceToDevice, _stream_send[0]));
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send[0]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
  }

  // Copy the first GEMM output chunk to the end chunk position of D_buffer
  char *src_ptr = reinterpret_cast<char *>(D_buffer.dptr());
  NVTE_CHECK_CUDA(cudaMemcpyAsync(src_ptr + (D.numel() * D.element_size()), src_ptr, D_chunk_bytes,
                                  cudaMemcpyDeviceToDevice, stream_main));

  // Return the last N rows of D_buffer
  NVTE_CHECK_CUDA(cudaMemcpyAsync(D.dptr(), src_ptr + D_chunk_bytes, D.numel() * D.element_size(),
                                  cudaMemcpyDeviceToDevice, stream_main));

  // Clean up buffer allocation
  NVTE_CHECK_CUDA(cudaFreeAsync(D_buffer_ptr, stream_main));

  _ub_comm->sms = ori_sms;
}  // CommOverlapP2PBase::atomic_gemm_overlap_ag

/*
** Split AllGather + GEMM using P2P communication
** This function assumes the input_b is pre-copied to _ubufs[rank_id]. This is needed to have AG
** outputs in each rank to be in the contiguous memory space after all ring exchange phases.
*/
void CommOverlapP2PBase::split_overlap_ag(const TensorWrapper &A, bool transa,
                                          const TensorWrapper &B, bool transb, TensorWrapper &D,
                                          TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                                          TensorWrapper &workspace, bool grad, bool accumulate,
                                          bool use_split_accumulator, TensorWrapper &B_copy,
                                          cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;
  // Get GEMM dimensions between TN and NN input layouts
  const size_t m = (transa) ? A.size(0) : A.size(1);
  const size_t k = (transa) ? A.size(1) : A.size(0);
  const size_t n_chunk = _ubufs[0].size(0);

  // Get communication and GEMM output chunk sizes
  const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
  const bool do_gelu = pre_gelu_out.numel() > 0;
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[0], _start_compute, 0));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }
  if (_aggregate) {
    const int num_steps = _tp_size / 2;

    // Chunk dims
    std::vector<size_t> input_b_chunk_shape =
        (transb ? std::vector<size_t>{k, 2 * n_chunk} : std::vector<size_t>{2 * n_chunk, k});
    std::vector<size_t> output_chunk_shape = {2 * n_chunk, m};
    size_t input_b_chunk_size = 2 * n_chunk * k;
    size_t output_chunk_size = 2 * n_chunk * m;

    // Initial 1X input chunk exchange between neighboring peers
    int send_chunk_id = _tp_id;
    int recv_chunk_id = (_tp_id % 2 == 0) ? _tp_id + 1 : _tp_id - 1;
    int send_offset = comm_bytes * send_chunk_id;
    int recv_offset = comm_bytes * recv_chunk_id;
    int peer_rank = (_tp_id % 2 == 0) ? _next_rank : _prev_rank;
    userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm, peer_rank,
                     _stream_send[0]);
    userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, peer_rank,
                     _stream_recv);
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[0], _stop_recv, 0));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[0], _stop_recv, 0));

    int local_rank_round2 = (_tp_id % 2 == 0) ? _tp_id : _tp_id - 1;
    const int next_rank = (_tp_size + _tp_id + 2) % _tp_size + _rank_round_tp;
    const int prev_rank = (_tp_size + _tp_id - 2) % _tp_size + _rank_round_tp;

    // Ring exchange of 2X inputs chunks
    for (int i = 0; i < num_steps; i++) {
      send_chunk_id = (_tp_size + local_rank_round2 - i * 2) % _tp_size;
      recv_chunk_id = (_tp_size + local_rank_round2 - i * 2 - 2) % _tp_size;
      send_offset = comm_bytes * send_chunk_id;
      recv_offset = comm_bytes * recv_chunk_id;

      // GEMM
      auto input_b_chunk =
          get_buffer_chunk_like(B, input_b_chunk_size * send_chunk_id / 2, input_b_chunk_shape);
      auto output_chunk =
          get_tensor_chunk(D, output_chunk_size * send_chunk_id / 2, output_chunk_shape);
      auto aux_chunk = (do_gelu)
                           ? get_tensor_chunk(pre_gelu_out, output_chunk_size * send_chunk_id / 2,
                                              {n_chunk * 2, k})
                           : TensorWrapper(nullptr, std::vector<size_t>{0}, pre_gelu_out.dtype());
      auto workspace_chunk = get_tensor_chunk(
          workspace, (i % _stream_compute.size()) * workspace_size_chunk, {workspace_size_chunk});

      nvte_cublas_gemm(A.data(), input_b_chunk.data(), output_chunk.data(), bias.data(),
                       aux_chunk.data(), transa, transb, grad, workspace_chunk.data(), accumulate,
                       use_split_accumulator, _math_sms,
                       _stream_compute[i % _stream_compute.size()]);

      if (i < num_steps - 1) {
        // P2P communication
        userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes * 2, _ub_comm,
                         next_rank, _stream_send[0]);
        userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes * 2, _ub_comm,
                         prev_rank, _stream_recv);
        NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[0], _stop_recv, 0));
        NVTE_CHECK_CUDA(
            cudaStreamWaitEvent(_stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
      } else if (B_copy.numel() > 0) {
        assert(B_copy.numel() == _ubufs[_tp_id].numel());
        assert(B_copy.element_size() == _ubufs[_tp_id].element_size());
        NVTE_CHECK_CUDA(cudaMemcpyAsync(B_copy.dptr(), _ubufs[_tp_id].dptr(),
                                        _ubufs[_tp_id].numel() * _ubufs[_tp_id].element_size(),
                                        cudaMemcpyDeviceToDevice, _stream_send[0]));
      }
    }
  } else {
    // Chunk dims
    std::vector<size_t> input_b_chunk_shape =
        (transb ? std::vector<size_t>{k, n_chunk} : std::vector<size_t>{n_chunk, k});
    std::vector<size_t> output_chunk_shape = {n_chunk, m};
    size_t input_b_chunk_size = n_chunk * k;
    size_t output_chunk_size = n_chunk * m;

    for (int i = 0; i < _tp_size; i++) {
      // Set the userbuffer id. Buffer under send is the input for the current
      // GEMM chunk The initial input chunk is stored _ubuf[rank]. This is to
      // have the AG output in all ranks to be contiguous after the ring
      // exchanges
      int send_chunk_id = (_tp_size + _tp_id - i) % _tp_size;
      int recv_chunk_id = (_tp_size + _tp_id - i - 1) % _tp_size;
      int send_offset = comm_bytes * send_chunk_id;
      int recv_offset = comm_bytes * recv_chunk_id;

      // GEMM
      auto input_b_chunk =
          get_buffer_chunk_like(B, input_b_chunk_size * send_chunk_id, input_b_chunk_shape);
      auto output_chunk =
          get_tensor_chunk(D, output_chunk_size * send_chunk_id, output_chunk_shape);
      auto aux_chunk =
          (do_gelu)
              ? get_tensor_chunk(pre_gelu_out, output_chunk_size * send_chunk_id, {n_chunk, k})
              : TensorWrapper(nullptr, std::vector<size_t>{0}, pre_gelu_out.dtype());
      auto workspace_chunk = get_tensor_chunk(
          workspace, (i % _stream_compute.size()) * workspace_size_chunk, {workspace_size_chunk});

      nvte_cublas_gemm(A.data(), input_b_chunk.data(), output_chunk.data(), bias.data(),
                       aux_chunk.data(), transa, transb, grad, workspace_chunk.data(), accumulate,
                       use_split_accumulator, _math_sms,
                       _stream_compute[i % _stream_compute.size()]);

      if (i < _tp_size - 1) {
        // P2P communication
        userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm,
                         _next_rank, _stream_send[0]);
        userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
                         _prev_rank, _stream_recv);
        NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
        NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[0], _stop_recv, 0));
        NVTE_CHECK_CUDA(
            cudaStreamWaitEvent(_stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
      } else if (B_copy.numel() > 0) {
        assert(B_copy.numel() == _ubufs[_tp_id].numel());
        assert(B_copy.element_size() == _ubufs[_tp_id].element_size());
        NVTE_CHECK_CUDA(cudaMemcpyAsync(B_copy.dptr(), _ubufs[_tp_id].dptr(),
                                        _ubufs[_tp_id].numel() * _ubufs[_tp_id].element_size(),
                                        cudaMemcpyDeviceToDevice, _stream_send[0]));
      }
    }
  }

  _ub_comm->sms = ori_sms;
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send[0]));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));
}  // CommOverlapP2PBase::split_overlap_ag

/*
** Split ReduceScatter + GEMM using P2P communication
*/
void CommOverlapP2PBase::atomic_gemm_overlap_rs(
    const TensorWrapper &A, bool transa, const TensorWrapper &B, bool transb, TensorWrapper &D,
    TensorWrapper &bias, TensorWrapper &pre_gelu_out, TensorWrapper &workspace, bool grad,
    bool accumulate, bool use_split_accumulator, TensorWrapper &rs_output,
    cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;

  // Get communication and GEMM input chunk sizes
  const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();

  // Reset counters
  int *counter_ptr = reinterpret_cast<int *>(_counter.dptr());
  reset_counters(counter_ptr, _tp_size, false, stream_main);

  // Catch up the main stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));

  // Atomic GEMM
  // Process GEMM chunks in the order that AG+GEMM places the output chunks.
  auto output_d = get_buffer_chunk_like(D, 0, shape_to_vector(D.shape()));
  nvte_cublas_atomic_gemm(A.data(), B.data(), output_d.data(), bias.data(), pre_gelu_out.data(),
                          transa, transb, grad, workspace.data(), accumulate, use_split_accumulator,
                          _math_sms, 0, _tp_size, true, _counter.data(), stream_main);

  // P2P communication chunk
  for (int i = 1; i < _tp_size; i++) {
    int send_chunk_id = i - 1;
    int recv_chunk_id = send_chunk_id + _tp_size;
    int send_offset = comm_bytes * send_chunk_id;
    int recv_offset = comm_bytes * recv_chunk_id;
    int send_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
    int recv_rank = (_tp_id + i) % _tp_size + _rank_round_tp;

    consumer(counter_ptr, send_chunk_id, _stream_recv);
    userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, send_rank,
                     _stream_recv);
    userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, recv_rank,
                     _stream_recv);
  }
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));

  // Reduce GEMM output chunks
  char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].dptr());
  char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());
  if (_ubuf.element_size() == 1 && rs_output.element_size() == 2) {
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        D.dtype(), fp8_type,
        reduce_fp8_in_bf16_out<fp8_type>(reduce_buf_ptr, rs_output_ptr, D.scale_inv(), _tp_size,
                                         _ubufs[0].numel(), stream_main););
  } else {
    reduce_bf16(reduce_buf_ptr, rs_output_ptr, _tp_size, _ubufs[0].numel(), stream_main);
  }
  _ub_comm->sms = ori_sms;
}

/*
** Split ReduceScatter + GEMM using P2P communication
*/
void CommOverlapP2PBase::split_overlap_rs(const TensorWrapper &A, bool transa,
                                          const TensorWrapper &B, bool transb, TensorWrapper &D,
                                          TensorWrapper &bias, TensorWrapper &pre_gelu_out,
                                          TensorWrapper &workspace, bool grad, bool accumulate,
                                          bool use_split_accumulator, TensorWrapper &rs_output,
                                          cudaStream_t stream_main) {
  int ori_sms = _ub_comm->sms;
  _ub_comm->use_ce = _use_ce;
  _ub_comm->sms = _num_comm_sm;
  _ub_comm->cga_size = _cga_size;

  // Get communication and GEMM input chunk sizes
  size_t m = transa ? A.size(0) : A.size(1);
  size_t k = transa ? A.size(1) : A.size(0);
  size_t n_chunk = _ubufs[0].size(0);
  const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();

  // Get input and workspace data pointers
  size_t input_chunk_size = n_chunk * k;
  size_t output_chunk_size = n_chunk * m;
  size_t workspace_size_chunk = workspace.numel() / _stream_compute.size();

  // Catch up the main stream
  NVTE_CHECK_CUDA(cudaEventRecord(_start_compute, stream_main));
  for (size_t i = 0; i < _stream_send.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[i], _start_compute, 0));
  }
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_compute, 0));
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _start_compute, 0));
  }

  // GEMM and send/recv chunks
  for (int i = 0; i < _tp_size; i++) {
    // GEMM chunk
    int stream_id = i % _stream_compute.size();
    int input_b_chunk_id = (_tp_id + i + 1) % _tp_size;

    auto input_b_chunk = get_tensor_chunk(B, input_b_chunk_id * input_chunk_size, {n_chunk, k});
    auto output_chunk = get_buffer_chunk_by_id(D, i);

    auto workspace_chunk =
        get_tensor_chunk(workspace, stream_id * workspace_size_chunk, {workspace_size_chunk});

    nvte_cublas_gemm(A.data(), input_b_chunk.data(), output_chunk.data(), bias.data(),
                     pre_gelu_out.data(), transa, transb, grad, workspace_chunk.data(), accumulate,
                     use_split_accumulator, _math_sms, _stream_compute[stream_id]);

    if (i > 0) {
      // P2P communication chunk
      int prev_stream_id = (i - 1) % _stream_compute.size();
      int send_offset = comm_bytes * (i - 1);
      int recv_offset = comm_bytes * (i - 1 + _tp_size);
      int send_rank = (_tp_id + i) % _tp_size + _rank_round_tp;
      int recv_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;
      NVTE_CHECK_CUDA(cudaEventRecord(_start_comm, _stream_compute[prev_stream_id]));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_send[prev_stream_id], _start_comm, 0));
      NVTE_CHECK_CUDA(cudaStreamWaitEvent(_stream_recv, _start_comm, 0));
      userbuffers_send(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, send_rank,
                       _stream_send[prev_stream_id]);
      userbuffers_recv(_ub_reg, send_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, recv_rank,
                       _stream_recv);
    }
  }

  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_compute, _stream_compute[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_compute, 0));
  }
  for (size_t i = 0; i < _stream_compute.size(); i++) {
    NVTE_CHECK_CUDA(cudaEventRecord(_stop_send, _stream_send[i]));
    NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_send, 0));
  }
  NVTE_CHECK_CUDA(cudaEventRecord(_stop_recv, _stream_recv));
  NVTE_CHECK_CUDA(cudaStreamWaitEvent(stream_main, _stop_recv, 0));

  // Reduce GEMM output chunks
  char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].dptr());
  char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());
  if (_ubuf.element_size() == 1 && rs_output.element_size() == 2) {
    char *rs_output_ptr = reinterpret_cast<char *>(rs_output.dptr());
    TRANSFORMER_ENGINE_TYPE_SWITCH_FP8ONLY(
        D.dtype(), fp8_type,
        reduce_fp8_in_bf16_out<fp8_type>(reduce_buf_ptr, rs_output_ptr, D.scale_inv(), _tp_size,
                                         _ubufs[0].numel(), stream_main););
  } else {
    reduce_bf16(reduce_buf_ptr, rs_output_ptr, _tp_size, _ubufs[0].numel(), stream_main);
  }

  _ub_comm->sms = ori_sms;
}

}  // namespace transformer_engine
