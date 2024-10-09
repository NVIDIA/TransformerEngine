/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
#define TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_

#include <cuda_runtime_api.h>

#include <string>

namespace transformer_engine {

namespace cuda {

/* \brief Number of accessible devices */
int num_devices();

/* \brief Which device is currently being used */
int current_device();

/* \brief Compute capability of device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Compute capability as int. Last digit is minor revision,
 *         remaining digits are major revision.
 */
int sm_arch(int device_id = -1);

/* \brief Number of multiprocessors on a device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Number of multiprocessors
 */
int sm_count(int device_id = -1);

/* \brief Minimum and maximum stream priorities supported on device
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \param[out] lowest_priority Lowest stream priority value supported by the device.
 *
 * \param[out] highest_priority Highest stream priority value supported by the device.
 */
void stream_priority_range(int *lowest_priority, int *highest_priority, int device_id = -1);

/* \brief Check if device supports CUDA Multicast features
 *
 * \param[in] device_id CUDA device (default is current device)
 *
 * \return Support status for CUDA Multicast for given device ID
 */
bool supports_multicast(int device_id = -1);

/* \brief Path to CUDA Toolkit headers
 *
 * The path can be configured by setting NVTE_CUDA_INCLUDE_DIR in the
 * environment. Otherwise searches in common install paths.
 *
 * \param[in] required Whether to throw exception if not found
 *
 * \return Path to include directory, or an empty string if not found
 */
const std::string &include_directory(bool required = false);

}  // namespace cuda

}  // namespace transformer_engine

#endif  // TRANSFORMER_ENGINE_COMMON_UTIL_CUDA_RUNTIME_H_
