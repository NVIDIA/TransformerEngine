/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file nvfp4_4over6.h
 *  \brief Device-safe NVFP4 4over6 mode enum.
 *
 *  Kept free of host-only dependencies so the same definition can be used by
 *  the public C API and by NVRTC-compiled device kernels.
 */

#ifndef TRANSFORMER_ENGINE_NVFP4_4OVER6_H_
#define TRANSFORMER_ENGINE_NVFP4_4OVER6_H_

/*! \enum NVTENVFP44Over6Mode
 * \brief Method for NVFP4 4over6 quantization.
 */
enum NVTENVFP44Over6Mode {
  kNVTENVFP44Over6Disabled = 0, /*!< 4over6 is not applied */
  kNVTENVFP44Over6MinMAE = 1,   /*!< Select the candidate with lower mean absolute error */
  kNVTENVFP44Over6MinMSE = 2,   /*!< Select the candidate with lower mean squared error */
};

#endif  // TRANSFORMER_ENGINE_NVFP4_4OVER6_H_
