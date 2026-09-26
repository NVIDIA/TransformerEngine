/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <Python.h>

#include "tvm_ffi_bridge.h"

namespace transformer_engine {
namespace tvm_ffi_bridge {

namespace {

// Everything below must stay inside the CPython Stable ABI: this file is built with
// Py_LIMITED_API so a single libtransformer_engine.so works across CPython 3.10+.
// PyRun_SimpleString in particular is *not* part of the limited API.
bool import_and_register_backends() {
  PyObject *module = PyImport_ImportModule("transformer_engine.common");
  if (module == nullptr) {
    return false;
  }
  bool initialized = false;
  PyObject *loaded = PyObject_CallMethod(module, "_load_tvm_ffi_library", nullptr);
  if (loaded != nullptr && PyObject_IsTrue(loaded) == 1) {
    PyObject *registered = PyObject_CallMethod(module, "_register_cutedsl_backends", nullptr);
    if (registered != nullptr && PyObject_IsTrue(registered) == 1) {
      initialized = true;
    }
    Py_XDECREF(registered);
  }
  Py_XDECREF(loaded);
  Py_DECREF(module);
  return initialized;
}

// transformer_engine/__init__.py pulls in the framework subpackages when they are importable,
// which an embedded interpreter has no reason to pay for. A None entry in sys.modules makes the
// corresponding `import` raise ImportError.
void mask_framework_modules() {
  PyObject *modules = PyImport_GetModuleDict();  // borrowed
  if (modules == nullptr) {
    return;
  }
  PyDict_SetItemString(modules, "transformer_engine.pytorch", Py_None);
  PyDict_SetItemString(modules, "transformer_engine.jax", Py_None);
}

void warn_initialization_failed() {
  if (PyErr_Occurred() != nullptr) {
    PyErr_Print();
  }
  NVTE_WARN(
      "Failed to initialize CuTeDSL backend: Python import, TVM-FFI load, or backend "
      "registration failed. Using CUDA backend as fallback.");
}

}  // namespace

// Initialize the Python interpreter and import the CuTeDSL backend module. This is only compiled
// with NVTE_WITH_CUTEDSL=ON in CMake and will be ignored otherwise
bool initialize_python_cutedsl_backend() {
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
    mask_framework_modules();
    const bool initialized = import_and_register_backends();
    if (!initialized) {
      warn_initialization_failed();
    }
    // Detach the initializing C++ thread and release the GIL so TVM-FFI callbacks can acquire it
    // from any thread. The interpreter has process lifetime, so no later restore is needed.
    (void)PyEval_SaveThread();
    return initialized;
  }

  // Python is already running in this process but we don't know if it has imported TE or not,
  // so we import here just to be sure
  const PyGILState_STATE gil_state = PyGILState_Ensure();
  const bool initialized = import_and_register_backends();
  if (!initialized) {
    warn_initialization_failed();
  }
  PyGILState_Release(gil_state);
  return initialized;
}

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine
