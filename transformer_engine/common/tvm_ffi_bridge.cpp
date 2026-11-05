/*************************************************************************
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include "tvm_ffi_bridge.h"

#include <fcntl.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>

extern char **environ;

namespace transformer_engine {
namespace tvm_ffi_bridge {

namespace {

// The compiler process imports only the framework-independent part of TE: a None entry in
// sys.modules makes `import transformer_engine.{pytorch,jax}` raise ImportError.
constexpr const char *kPythonPrelude =
    "import sys\n"
    "sys.modules['transformer_engine.pytorch'] = None\n"
    "sys.modules['transformer_engine.jax'] = None\n";

// Run `$NVTE_CUTEDSL_PYTHON -c <prelude + code> args...` and wait for it. Returns whether it
// exited with status 0. If `out` is given, it receives the process's stdout.
bool run_python(const std::string &code, const std::vector<std::string> &args,
                std::string *out = nullptr) {
  const std::string python = getenv<std::string>("NVTE_CUTEDSL_PYTHON", "python3");
  std::vector<std::string> argv_strs = {python, "-c", kPythonPrelude + code};
  argv_strs.insert(argv_strs.end(), args.begin(), args.end());
  std::vector<char *> argv;
  for (auto &s : argv_strs) argv.push_back(s.data());
  argv.push_back(nullptr);

  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  int pipe_fds[2] = {-1, -1};
  if (out != nullptr) {
    if (pipe2(pipe_fds, O_CLOEXEC) != 0) {
      posix_spawn_file_actions_destroy(&actions);
      return false;
    }
    posix_spawn_file_actions_adddup2(&actions, pipe_fds[1], STDOUT_FILENO);
  }
  pid_t pid;
  const int spawn_rc = posix_spawnp(&pid, python.c_str(), &actions, nullptr, argv.data(), environ);
  posix_spawn_file_actions_destroy(&actions);
  if (out != nullptr) {
    close(pipe_fds[1]);
    char buf[4096];
    while (spawn_rc == 0) {
      const ssize_t n = read(pipe_fds[0], buf, sizeof(buf));
      if (n > 0) {
        out->append(buf, n);
      } else if (n == 0 || errno != EINTR) {
        break;
      }
    }
    close(pipe_fds[0]);
  }
  if (spawn_rc != 0) {
    NVTE_WARN("Failed to launch the CuTeDSL compiler process `", python,
              "` (set NVTE_CUTEDSL_PYTHON to a Python interpreter): ", std::strerror(spawn_rc));
    return false;
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

}  // namespace

std::string find_tvm_ffi_library() {
  std::string path;
  if (!run_python("import tvm_ffi.libinfo\nprint(tvm_ffi.libinfo.find_libtvm_ffi())", {}, &path)) {
    return "";
  }
  while (!path.empty() && std::isspace(static_cast<unsigned char>(path.back()))) path.pop_back();
  return path;
}

std::optional<tvm::ffi::Function> compile_and_load_kernel(const std::string &entrypoint,
                                                          const std::string &name,
                                                          const std::vector<std::string> &args) {
  std::string dir = (std::filesystem::temp_directory_path() / "nvte_cutedsl_XXXXXX").string();
  if (mkdtemp(dir.data()) == nullptr) {
    NVTE_WARN("Failed to create a temporary directory for the CuTeDSL kernel `", name, "`.");
    return std::nullopt;
  }
  const std::string library = dir + "/" + name + ".so";

  std::vector<std::string> cli = {"--entrypoint",
                                  entrypoint,
                                  "--name",
                                  name,
                                  "--output",
                                  library,
                                  "--device",
                                  std::to_string(cuda::current_device()),
                                  "--"};
  cli.insert(cli.end(), args.begin(), args.end());
  std::optional<tvm::ffi::Function> fn;
  if (run_python("from transformer_engine.common.CuTeDSL.compile_kernel import main\n"
                 "sys.exit(main(sys.argv[1:]))",
                 cli)) {
    try {
      tvm::ffi::Any module =
          (*tvm::ffi::Function::GetGlobal("ffi.ModuleLoadFromFile"))(tvm::ffi::String(library));
      // Keep the library loaded for the lifetime of the process, as tvm_ffi.load_module does.
      (*tvm::ffi::Function::GetGlobal("ffi.ModuleGlobalsAdd"))(module);
      fn = (*tvm::ffi::Function::GetGlobal("ffi.ModuleGetFunction"))(module, tvm::ffi::String(name),
                                                                     false)
               .try_cast<tvm::ffi::Function>();
    } catch (const std::exception &e) {
      NVTE_WARN("Failed to load the CuTeDSL kernel `", name, "` from ", library, ": ", e.what());
    }
  }
  // The loaded library stays mapped after its file is removed.
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
  return fn;
}

}  // namespace tvm_ffi_bridge
}  // namespace transformer_engine
