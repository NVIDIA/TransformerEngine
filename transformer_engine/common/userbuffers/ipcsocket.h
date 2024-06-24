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

static const char *ipcSocketResultStrings[static_cast<int>(ipcSocketNumResults)] = {
    "Success",       "Unhandled CUDA error", "Internal error",       "Invalid argument",
    "Invalid usage", "Remote error",         "Operation in progress"};

#define IPC_SOCKET_CHECK(cmd)                                             \
  do {                                                                    \
    ipcSocketResult_t r = cmd;                                            \
    if (r != ipcSocketSuccess) {                                          \
      printf("Failed, IPC socket error %s:%d : %s\n", __FILE__, __LINE__, \
             ipcSocketResultStrings[static_cast<int>(r)]);                \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

#define IPC_SOCKET_CHECK_GOTO(call, RES, label)                  \
  do {                                                           \
    RES = call;                                                  \
    if (RES != ipcSocketSuccess && RES != ipcSocketInProgress) { \
      goto label;                                                \
    }                                                            \
  } while (0);

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
