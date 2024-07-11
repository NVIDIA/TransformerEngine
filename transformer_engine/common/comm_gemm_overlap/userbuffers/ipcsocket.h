/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#ifndef TRANSFORMER_ENGINE_COMMON_IPCSOCKET_H
#define TRANSFORMER_ENGINE_COMMON_IPCSOCKET_H

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
  ipcSocketSuccess = 0,
  ipcSocketUnhandledCudaError = 1,
  ipcSocketSystemError = 2,
  ipcSocketInternalError = 3,
  ipcSocketInvalidArgument = 4,
  ipcSocketInvalidUsage = 5,
  ipcSocketRemoteError = 6,
  ipcSocketInProgress = 7,
  ipcSocketNumResults = 8
} ipcSocketResult_t;

const char *getIpcSocketErrorString(ipcSocketResult_t res);

#define IPC_SOCKNAME_LEN 64

typedef struct ipcSocket {
  int fd;
  char socketName[IPC_SOCKNAME_LEN];
  volatile uint32_t *abortFlag;
} ipcSocket;

ipcSocketResult_t ipcSocketInit(ipcSocket *handle, int rank, uint64_t hash,
                                volatile uint32_t *abortFlag);
ipcSocketResult_t ipcSocketClose(ipcSocket *handle);
ipcSocketResult_t ipcSocketGetFd(ipcSocket *handle, int *fd);
ipcSocketResult_t ipcSocketRecvFd(ipcSocket *handle, int *fd);
ipcSocketResult_t ipcSocketSendFd(ipcSocket *handle, const int fd, int rank, uint64_t hash);

#endif /* TRANSFORMER_ENGINE_COMMON_IPCSOCKET_H */
