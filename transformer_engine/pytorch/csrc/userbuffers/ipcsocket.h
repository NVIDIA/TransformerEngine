/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef NCCL_IPCSOCKET_H
#define NCCL_IPCSOCKET_H

// #include "nccl.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <memory.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

typedef enum {
  ncclSuccess = 0,
  ncclUnhandledCudaError = 1,
  ncclSystemError = 2,
  ncclInternalError = 3,
  ncclInvalidArgument = 4,
  ncclInvalidUsage = 5,
  ncclRemoteError = 6,
  ncclInProgress = 7,
  ncclNumResults = 8
} ncclResult_t;

#define NCCL_IPC_SOCKNAME_LEN 64

struct ncclIpcSocket {
  int fd;
  char socketName[NCCL_IPC_SOCKNAME_LEN];
  volatile uint32_t *abortFlag;
};

ncclResult_t ncclIpcSocketInit(struct ncclIpcSocket *handle, int rank, uint64_t hash,
                               volatile uint32_t *abortFlag);
ncclResult_t ncclIpcSocketClose(struct ncclIpcSocket *handle);
ncclResult_t ncclIpcSocketGetFd(struct ncclIpcSocket *handle, int *fd);

ncclResult_t ncclIpcSocketRecvFd(struct ncclIpcSocket *handle, int *fd);
ncclResult_t ncclIpcSocketSendFd(struct ncclIpcSocket *handle, const int fd, int rank,
                                 uint64_t hash);

#endif /* NCCL_IPCSOCKET_H */
