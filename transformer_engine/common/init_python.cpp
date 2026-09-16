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
  const bool embedding_python = !Py_IsInitialized();
  if (embedding_python) {
    Py_Initialize();
  }
  const PyGILState_STATE gil_state = PyGILState_Ensure();

  if (embedding_python) {
    PyObject *path = PySys_GetObject("path");
    PyObject *source_dir = PyUnicode_FromString(NVTE_SOURCE_DIR);
    if (path == nullptr || source_dir == nullptr || PyList_Insert(path, 0, source_dir) != 0) {
      Py_XDECREF(source_dir);
      PyErr_Print();
      PyGILState_Release(gil_state);
      return false;
    }
    Py_DECREF(source_dir);
  }

  const char *initialize = embedding_python
                               ? "import sys\n"
                                 "sys.modules['transformer_engine.pytorch'] = None\n"
                                 "sys.modules['transformer_engine.jax'] = None\n"
                                 "import transformer_engine.common"
                               : "import transformer_engine.common";
  const bool initialized = PyRun_SimpleString(initialize) == 0;
  if (!initialized) {
    PyErr_Print();
  }
  PyGILState_Release(gil_state);
  return initialized;
}

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine
