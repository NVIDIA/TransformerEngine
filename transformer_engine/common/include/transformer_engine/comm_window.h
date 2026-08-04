/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

/*! \file comm_window.h
 *  \brief NCCL symmetric-memory window handle for zero-copy ops. Pass
 *         {NULL, 0} to use the raw-pointer path.
 */

#ifndef TRANSFORMER_ENGINE_COMM_WINDOW_H_
#define TRANSFORMER_ENGINE_COMM_WINDOW_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward-declare NCCL's opaque window struct so this header does not pull in
 * <nccl.h>; matches NCCL's typedef (struct ncclWindow_vidmem* ncclWindow_t). */
struct ncclWindow_vidmem;

/*! \brief NCCL window plus byte offset for a zero-copy payload tensor. */
typedef struct {
  struct ncclWindow_vidmem* window; /*!< NCCL window, or NULL to use the raw data pointer. */
  uint64_t offset;                  /*!< Byte offset of the payload within window. */
} NVTECommWindow;

#ifdef __cplusplus
}
#endif

#endif  // TRANSFORMER_ENGINE_COMM_WINDOW_H_
