/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <Python.h>

#include "tvm_ffi_bridge.h"

namespace transformer_engine {
namespace tvm_ffi_bridge {

// Initialize the Python interpreter and import the CuTeDSL backend module. This is only compiled
// with NVTE_WITH_CUTEDSL=ON in CMake and will be ignored otherwise
bool initialize_python_cutedsl_backend() {
  if (!transformer_engine::getenv<bool>("NVTE_ENABLE_CUTEDSL_BACKEND")) {
    return false;
  }
  const bool embedding_python = !Py_IsInitialized();

  // TE is loaded as a C++ library and it's not loaded from python. We need to launch an embedded
  // python so we can compile CuTeDSL backends
  if (embedding_python) {
    // Py_Initialize leaves the calling thread attached and holding the GIL.
    Py_Initialize();
    if (!Py_IsInitialized()) {
      NVTE_WARN(
          "Failed to initialize CuTeDSL backend: unable to initialize Python interpreter. Using "
          "CUDA backend as fallback.");
      return false;
    }
    PyObject *path = PySys_GetObject("path");
    PyObject *source_dir = PyUnicode_FromString(NVTE_SOURCE_DIR);
    if (path == nullptr || source_dir == nullptr || PyList_Insert(path, 0, source_dir) != 0) {
      Py_XDECREF(source_dir);
      PyErr_Print();
      NVTE_WARN(
          "Failed to initialize CuTeDSL backend: failed to insert source directory into Python "
          "path. Using CUDA backend as fallback.");
      (void)PyEval_SaveThread();
      return false;
    }
    Py_DECREF(source_dir);

    const bool initialized = PyRun_SimpleString(
                                 "import sys\n"
                                 "sys.modules['transformer_engine.pytorch'] = None\n"
                                 "sys.modules['transformer_engine.jax'] = None\n"
                                 "import transformer_engine.common") == 0;
    if (!initialized) {
      PyErr_Print();
      NVTE_WARN(
          "Failed to initialize CuTeDSL backend: unable to import transformer_engine.common from "
          "python. Using CUDA backend as fallback.");
    }
    // Detach the initializing C++ thread and release the GIL so TVM-FFI callbacks can acquire it
    // from any thread. The interpreter has process lifetime, so no later restore is needed.
    (void)PyEval_SaveThread();
    return initialized;
  }

  // Python is already running in this process but we don't know if it has imported TE or not,
  // so we import here just to be sure
  const PyGILState_STATE gil_state = PyGILState_Ensure();
  const bool initialized = PyRun_SimpleString("import transformer_engine.common") == 0;
  if (!initialized) {
    PyErr_Print();
    NVTE_WARN(
        "Failed to initialize CuTeDSL backend: unable to import transformer_engine.common from "
        "python. Using CUDA backend as fallback.");
  }
  PyGILState_Release(gil_state);
  return initialized;
}

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine
