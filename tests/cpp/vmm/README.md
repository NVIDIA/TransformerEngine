The physical-block reuse policy can be checked without CUDA, PyTorch, or TE builds:

```sh
g++ -std=c++17 -Wall -Wextra -Werror -pthread -I . \
    tests/cpp/vmm/test_vmm_mempool_core.cpp -o /tmp/te-test-vmm-mempool-core
/tmp/te-test-vmm-mempool-core
```

Run from the TE repository root. The simulated driver rejects releasing mapped
handles, freeing still-mapped virtual addresses, and querying events in capture.
The tests cover differently sized allocations, allocation-stream completion,
physical-domain selection, capture quarantine, cache draining with live tensors,
OOM eviction, and partial mapping/event/access failure rollback.

The same test is registered in TE's C++ CTest suite. To configure it independently
without a CUDA compiler:

```sh
cmake -S tests/cpp/vmm -B /tmp/te-vmm-core-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/te-vmm-core-build
ctest --test-dir /tmp/te-vmm-core-build --output-on-failure
```
