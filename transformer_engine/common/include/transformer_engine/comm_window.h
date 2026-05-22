/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file comm_window.h
 *  \brief Borrowed symmetric-memory window + offset for zero-copy one-sided ops.
 *         Pass ``{NULL, 0}`` to use the raw-pointer path.
 */

#ifndef TRANSFORMER_ENGINE_COMM_WINDOW_H_
#define TRANSFORMER_ENGINE_COMM_WINDOW_H_

#include <nccl.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief NCCL window + byte offset for a zero-copy payload tensor. */
typedef struct {
  ncclWindow_t window; /*!< NCCL window, or NULL to use the raw data pointer. */
  uint64_t offset;     /*!< Byte offset of the payload within ``window``. */
} NVTECommWindow;

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_COMM_WINDOW_H_
