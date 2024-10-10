/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "ipcsocket.h"

#include <errno.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define IPC_MAX_MSGLEN 4096

void ipc_warn(const char *format, ...) {
  char buffer[IPC_MAX_MSGLEN];

  va_list args;
  va_start(args, format);

  vsnprintf(buffer, IPC_MAX_MSGLEN - 1, format, args);
  snprintf(buffer + strlen(buffer), IPC_MAX_MSGLEN - strlen(buffer) - 1, " : %s (%d)\n",
           strerror(errno), errno);

  fflush(stdout);
  fputs(buffer, stderr);
  fflush(NULL);

  va_end(args);
}

static const char *ipcSocketResultStrings[static_cast<int>(ipcSocketNumResults)] = {
    "Success",          "Unhandled CUDA error", "System error", "Internal error",
    "Invalid argument", "Invalid usage",        "Remote error", "In progress",
};

const char *ipcSocketGetErrorString(ipcSocketResult_t res) {
  return ipcSocketResultStrings[static_cast<int>(res)];
}

#define USE_ABSTRACT_SOCKET  // Enable Linux abstract socket naming

#define IPC_SOCKNAME_STR "/tmp/ub-ipc-socket-%d-%lx"

/*
 * Create a Unix Domain Socket
 */
ipcSocketResult_t ipcSocketInit(IpcSocketHandle *handle, int rank, uint64_t hash,
                                volatile uint32_t *abortFlag) {
  int fd = -1;
  struct sockaddr_un cliaddr;
  char temp[IPC_SOCKNAME_LEN] = "";

  if (handle == NULL) {
    return ipcSocketInternalError;
  }

  handle->fd = -1;
  handle->socketName[0] = '\0';
  if ((fd = socket(AF_UNIX, SOCK_DGRAM, 0)) < 0) {
    ipc_warn("UDS: Socket creation error");
    return ipcSocketSystemError;
  }

  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  // Create unique name for the socket.
  size_t len = snprintf(temp, IPC_SOCKNAME_LEN, IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    errno = ENAMETOOLONG;
    ipc_warn("UDS: Cannot bind provided name to socket. Name too large");
    return ipcSocketInternalError;
  }
  strncpy(cliaddr.sun_path, temp, len);
#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0';  // Linux abstract socket trick
#else
  unlink(temp);
#endif
  if (bind(fd, (struct sockaddr *)&cliaddr, sizeof(cliaddr)) < 0) {
    ipc_warn("UDS: Binding to socket %s failed", temp);
    close(fd);
    return ipcSocketSystemError;
  }

  handle->fd = fd;
  strcpy(handle->socketName, temp);  // NOLINT(*)

  handle->abortFlag = abortFlag;
  // Mark socket as non-blocking
  if (handle->abortFlag) {
    int flags = fcntl(fd, F_GETFL);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
  }

  return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketGetFd(struct IpcSocketHandle *handle, int *fd) {
  if (handle == NULL) {
    errno = EINVAL;
    ipc_warn("ipcSocketSocketGetFd: pass NULL socket");
    return ipcSocketInvalidArgument;
  }
  if (fd) *fd = handle->fd;
  return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketClose(IpcSocketHandle *handle) {
  if (handle == NULL) {
    return ipcSocketInternalError;
  }
  if (handle->fd <= 0) {
    return ipcSocketSuccess;
  }
#ifndef USE_ABSTRACT_SOCKET
  if (handle->socketName[0] != '\0') {
    unlink(handle->socketName);
  }
#endif
  close(handle->fd);

  return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketRecvMsg(IpcSocketHandle *handle, void *hdr, int hdrLen, int *recvFd) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];

  // Union to guarantee alignment requirements for control array
  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  int ret;

  msg.msg_control = control_un.control;
  msg.msg_controllen = sizeof(control_un.control);

  if (hdr == NULL) {
    iov[0].iov_base = reinterpret_cast<void *>(dummy_buffer);
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }

  msg.msg_iov = iov;
  msg.msg_iovlen = 1;

  while ((ret = recvmsg(handle->fd, &msg, 0)) <= 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      ipc_warn("UDS: Receiving data over socket failed");
      return ipcSocketSystemError;
    }
    if (handle->abortFlag && *handle->abortFlag) return ipcSocketInternalError;
  }

  if (recvFd != NULL) {
    if (((cmptr = CMSG_FIRSTHDR(&msg)) != NULL) && (cmptr->cmsg_len == CMSG_LEN(sizeof(int)))) {
      if ((cmptr->cmsg_level != SOL_SOCKET) || (cmptr->cmsg_type != SCM_RIGHTS)) {
        errno = EBADMSG;
        ipc_warn("UDS: Receiving data over socket %s failed", handle->socketName);
        return ipcSocketSystemError;
      }

      memmove(recvFd, CMSG_DATA(cmptr), sizeof(*recvFd));
    } else {
      errno = ENOMSG;
      ipc_warn("UDS: Receiving data over socket %s failed", handle->socketName);
      return ipcSocketSystemError;
    }
  } else {
    errno = EINVAL;
    ipc_warn("UDS: File descriptor pointer cannot be NULL");
    return ipcSocketInvalidArgument;
  }

  return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketRecvFd(IpcSocketHandle *handle, int *recvFd) {
  return ipcSocketRecvMsg(handle, NULL, 0, recvFd);
}

ipcSocketResult_t ipcSocketSendMsg(IpcSocketHandle *handle, void *hdr, int hdrLen, const int sendFd,
                                   int rank, uint64_t hash) {
  struct msghdr msg = {0, 0, 0, 0, 0, 0, 0};
  struct iovec iov[1];
  char temp[IPC_SOCKNAME_LEN];

  union {
    struct cmsghdr cm;
    char control[CMSG_SPACE(sizeof(int))];
  } control_un;

  struct cmsghdr *cmptr;
  char dummy_buffer[1];
  struct sockaddr_un cliaddr;

  // Construct client address to send this shareable handle to
  bzero(&cliaddr, sizeof(cliaddr));
  cliaddr.sun_family = AF_UNIX;

  size_t len = snprintf(temp, IPC_SOCKNAME_LEN, IPC_SOCKNAME_STR, rank, hash);
  if (len > (sizeof(cliaddr.sun_path) - 1)) {
    errno = ENAMETOOLONG;
    ipc_warn("UDS: Cannot connect to provided name for socket. Name too large");
    return ipcSocketInternalError;
  }
  (void)strncpy(cliaddr.sun_path, temp, len);

#ifdef USE_ABSTRACT_SOCKET
  cliaddr.sun_path[0] = '\0';  // Linux abstract socket trick
#endif

  if (sendFd != -1) {
    msg.msg_control = control_un.control;
    msg.msg_controllen = sizeof(control_un.control);

    cmptr = CMSG_FIRSTHDR(&msg);
    cmptr->cmsg_len = CMSG_LEN(sizeof(int));
    cmptr->cmsg_level = SOL_SOCKET;
    cmptr->cmsg_type = SCM_RIGHTS;
    memmove(CMSG_DATA(cmptr), &sendFd, sizeof(sendFd));
  }

  msg.msg_name = reinterpret_cast<void *>(&cliaddr);
  msg.msg_namelen = sizeof(struct sockaddr_un);

  if (hdr == NULL) {
    iov[0].iov_base = reinterpret_cast<void *>(dummy_buffer);
    iov[0].iov_len = sizeof(dummy_buffer);
  } else {
    iov[0].iov_base = hdr;
    iov[0].iov_len = hdrLen;
  }
  msg.msg_iov = iov;
  msg.msg_iovlen = 1;
  msg.msg_flags = 0;

  ssize_t sendResult;
  while ((sendResult = sendmsg(handle->fd, &msg, 0)) < 0) {
    if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) {
      ipc_warn("UDS: Sending data over socket %s failed", temp);
      return ipcSocketSystemError;
    }
    if (handle->abortFlag && *handle->abortFlag) return ipcSocketInternalError;
  }

  return ipcSocketSuccess;
}

ipcSocketResult_t ipcSocketSendFd(IpcSocketHandle *handle, const int sendFd, int rank,
                                  uint64_t hash) {
  return ipcSocketSendMsg(handle, NULL, 0, sendFd, rank, hash);
}
