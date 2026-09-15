/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <Python.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

namespace {

class CuTeDSLEnvironment : public ::testing::Environment {
 public:
  // Before all tests start, we need to create a python environment and import transformer_engine
  // which imports tvm_ffi and prepare the CuTeDSL entrypoint, so C++ can ask python to compile
  // CuTeDSL kernels and use that to run C++ tests
  void SetUp() override {
    const char *enable_cutedsl = std::getenv("NVTE_ENABLE_CUTEDSL_BACKEND");
    if (enable_cutedsl == nullptr || std::strcmp(enable_cutedsl, "0") == 0) {
      return;
    }

    Py_Initialize();
    PyObject *path = PySys_GetObject("path");
    PyObject *source_dir = PyUnicode_FromString(NVTE_SOURCE_DIR);
    PyList_Insert(path, 0, source_dir);
    Py_DECREF(source_dir);

    if (PyRun_SimpleString("import sys\n"
                           "sys.modules['transformer_engine.pytorch'] = None\n"
                           "sys.modules['transformer_engine.jax'] = None\n"
                           "import transformer_engine.common") != 0) {
      PyErr_Print();
      FAIL() << "Failed to initialize the CuTeDSL Python backend";
    }
  }
};

::testing::Environment *const cutedsl_environment =
    ::testing::AddGlobalTestEnvironment(new CuTeDSLEnvironment);

}  // namespace
