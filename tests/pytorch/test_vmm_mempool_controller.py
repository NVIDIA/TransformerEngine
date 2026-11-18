# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See LICENSE for license information.

"""Host-only tests of ordinary-MemPool isolation and real-Storage ownership.

Run directly with Python/unittest; neither PyTorch, CUDA nor pytest is needed.
The fake torch module deliberately has no allocator-managed API.
"""

import contextlib
import gc
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import weakref


class DType:
    def __init__(self, size):
        self.size = size


class Storage:
    pass


class Tensor:
    def __init__(self, shape, dtype, storage=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.storage = storage if storage is not None else Storage()

    def element_size(self):
        return self.dtype.size

    def numel(self):
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def view(self, value):
        if isinstance(value, DType):
            return Tensor((self.numel() * self.dtype.size // value.size,), value, self.storage)
        assert self.numel() == _product(value)
        return Tensor(value, self.dtype, self.storage)

    def flatten(self):
        return Tensor((self.numel(),), self.dtype, self.storage)

    def __getitem__(self, value):
        return Tensor((value.stop,), self.dtype, self.storage)

    def detach(self):
        return Tensor(self.shape, self.dtype, self.storage)


def _product(shape):
    result = 1
    for dimension in shape:
        result *= dimension
    return result


class Ticket:
    def __init__(self, requested):
        self.mapped_bytes = ((requested + 1) // 2 + 63) // 64 * 128
        self.allocator = self
        self.storage = None
        self.pool = lambda: None
        self.trimmed = False

    def watch(self, backing):
        self.storage = weakref.ref(backing.storage)

    def storage_alive(self):
        return self.storage is not None and self.storage() is not None

    def trim(self, pool_id):
        assert self.pool() is None
        assert not self.storage_alive()
        self.trimmed = True
        return True


class Backend:
    def __init__(self, *args):
        self.tickets = []
        self.drains = 0
        self.allocator = types.SimpleNamespace()
        self.wait_for_reuse = False

    def set_wait_for_reuse(self, wait):
        self.wait_for_reuse = wait

    def ticket(self, requested):
        ticket = Ticket(requested)
        self.tickets.append(ticket)
        return ticket

    def empty_cache(self):
        self.drains += 1

    def collect_completed(self):
        pass

    def stats(self):
        return {"block_bytes": 64}


class OrdinaryTorch(types.ModuleType):
    def __init__(self):
        super().__init__("torch")
        self.capturing = False
        self.current_pool = None
        self.fail_gpu_allocation = False
        self.destroyed_during_capture = []
        self._C = types.SimpleNamespace()  # No _cuda_customManagedPoolAllocator.
        self.uint8 = DType(1)
        self.float32 = DType(4)
        owner = self

        class NativePool:
            next_id = 1

            # Intentionally accepts ONLY the standard allocator argument.
            def __init__(self, allocator):
                self.allocator = allocator
                allocator.pool = weakref.ref(self)
                self.id = (0, NativePool.next_id)
                NativePool.next_id += 1
                self.graph_refs = 0

            def use_count(self):
                return 1 + self.graph_refs

            def __del__(self):
                owner.destroyed_during_capture.append(owner.capturing)

        @contextlib.contextmanager
        def use_mem_pool(pool, device):
            previous = owner.current_pool
            owner.current_pool = pool
            try:
                yield
            finally:
                owner.current_pool = previous

        self.cuda = types.SimpleNamespace(
            MemPool=NativePool,
            use_mem_pool=use_mem_pool,
            current_device=lambda: 0,
            device=lambda _: contextlib.nullcontext(),
            is_current_stream_capturing=lambda: self.capturing,
            memory=types.SimpleNamespace(get_allocator_backend=lambda: "native"),
        )

    def device(self, value, index=None):
        if hasattr(value, "type"):
            return value
        return types.SimpleNamespace(type=value, index=index)

    def empty(self, shape, *, dtype, device):
        if isinstance(shape, int):
            shape = (shape,)
        if device != "cpu" and self.fail_gpu_allocation:
            raise RuntimeError("injected allocation failure")
        return Tensor(shape, dtype)


class TestOrdinaryMemPoolController(unittest.TestCase):
    def setUp(self):
        self.torch = OrdinaryTorch()
        tex = types.ModuleType("transformer_engine_torch")
        tex.VMMAllocationTicket = Ticket
        tex.VMMMemPoolBackend = Backend
        self.modules = patch.dict(
            sys.modules, {"torch": self.torch, "transformer_engine_torch": tex}
        )
        self.modules.start()
        source = Path(__file__).resolve().parents[2] / (
            "transformer_engine/pytorch/tensor/vmm_mempool.py"
        )
        spec = importlib.util.spec_from_file_location("te_vmm_controller_test", source)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.controller = self.module.VMMMemPool(mode="dedicated")

    def tearDown(self):
        self.torch.capturing = False
        self.module._CONTROLLERS.clear()
        gc.collect()
        self.modules.stop()

    def test_no_experimental_api_and_dedicated_pool_per_allocation(self):
        a = self.controller.allocate((2, 128), self.torch.uint8)
        b = self.controller.allocate((2, 128), self.torch.uint8)
        self.assertEqual(a.shape, (2, 128))
        self.assertEqual(b.shape, (2, 128))
        self.assertEqual(len(self.controller._pools), 2)
        self.assertNotEqual(self.controller._pools[0][2], self.controller._pools[1][2])
        self.assertIsNone(self.torch.current_pool)

    def test_view_keeps_storage_and_pool_alive(self):
        root = self.controller.allocate((2, 128), self.torch.float32)
        view = root.flatten()[:32]
        del root
        self.controller.collect()
        self.assertEqual(len(self.controller._pools), 1)
        self.assertTrue(self.controller._backend.tickets[0].storage_alive())
        del view
        self.controller.collect()
        self.assertEqual(len(self.controller._pools), 0)
        self.assertTrue(self.controller._backend.tickets[0].trimmed)

    def test_next_allocation_collects_dead_tensor(self):
        root = self.controller.allocate((2, 64), self.torch.uint8)
        del root
        next_tensor = self.controller.allocate((2, 32), self.torch.uint8)
        self.assertEqual(next_tensor.shape, (2, 32))
        self.assertTrue(self.controller._backend.tickets[0].trimmed)
        self.assertEqual(len(self.controller._pools), 1)

    def test_capture_defers_native_pool_destruction_even_on_failure(self):
        self.torch.capturing = True
        self.torch.fail_gpu_allocation = True
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.controller.allocate((2, 64), self.torch.uint8)
        self.controller.collect()
        self.assertEqual(self.torch.destroyed_during_capture, [])
        self.assertEqual(len(self.controller._pools), 1)
        self.torch.capturing = False
        self.controller.collect()
        self.assertEqual(self.torch.destroyed_during_capture, [False])
        self.assertEqual(len(self.controller._pools), 0)

    def test_native_graph_reference_defers_pool_drain(self):
        tensor = self.controller.allocate((2, 64), self.torch.uint8)
        entry = self.controller._pools[0]
        entry[0].graph_refs = 1
        del tensor
        self.controller.collect()
        self.assertEqual(len(self.controller._pools), 1)
        entry[0].graph_refs = 0
        self.controller.collect()
        self.assertEqual(len(self.controller._pools), 0)

    def test_cache_drain_keeps_live_storage(self):
        tensor = self.controller.allocate((2, 64), self.torch.uint8)
        self.controller.empty_cache()
        self.assertEqual(self.controller._backend.drains, 1)
        self.assertEqual(len(self.controller._pools), 1)
        self.assertEqual(tensor.shape, (2, 64))

    def test_dropping_controller_during_capture_defers_pool_destruction(self):
        self.torch.capturing = True
        tensor = self.controller.allocate((2, 64), self.torch.uint8)
        self.controller = None
        del tensor
        gc.collect()
        self.assertEqual(self.torch.destroyed_during_capture, [])
        self.torch.capturing = False
        self.module.clear_all_vmm_mem_pool_caches()
        self.assertEqual(self.torch.destroyed_during_capture, [False])

    def test_wait_for_reuse_is_explicit_and_default_is_nonblocking(self):
        self.assertFalse(self.controller._backend.wait_for_reuse)
        diagnostic = self.module.VMMMemPool(mode="dedicated", wait_for_reuse=True)
        self.assertTrue(diagnostic._backend.wait_for_reuse)
        self.assertFalse(self.controller._backend.wait_for_reuse)

    def test_managed_mode_reports_missing_pytorch_support(self):
        with self.assertRaisesRegex(RuntimeError, "patched PyTorch"):
            self.module.VMMMemPool(mode="managed")

    def test_managed_mode_uses_one_pool_without_per_tensor_tickets(self):
        ordinary_pool = self.torch.cuda.MemPool
        drains = []

        class ManagedPool(ordinary_pool):
            def __init__(self, allocator, *, allocator_managed=False):
                assert allocator_managed
                super().__init__(allocator)

        def drain(pool):
            if pool.graph_refs:
                raise RuntimeError("pool retained by graph")
            drains.append(pool.id)

        self.torch.cuda.MemPool = ManagedPool
        self.torch.cuda.memory.empty_allocator_cache = drain
        controller = self.module.VMMMemPool(mode="managed")
        a = controller.allocate((2, 64), self.torch.uint8)
        b = controller.allocate((2, 128), self.torch.uint8)
        self.assertEqual(a.shape, (2, 64))
        self.assertEqual(b.shape, (2, 128))
        self.assertEqual(controller._backend.tickets, [])
        self.assertEqual(controller._pools, [])
        self.assertEqual(controller.stats()["managed_pools"], 1)
        self.assertFalse(hasattr(self.torch._C, "_cuda_customManagedPoolAllocator"))
        controller._managed_pool.graph_refs = 1
        with self.assertRaisesRegex(RuntimeError, "retained"):
            controller.empty_cache()
        self.assertEqual(controller._backend.drains, 0)
        controller._managed_pool.graph_refs = 0
        controller.empty_cache()
        self.assertEqual(drains, [controller._managed_pool.id])

    def test_rejects_cache_drain_in_capture(self):
        self.torch.capturing = True
        with self.assertRaisesRegex(RuntimeError, "capture"):
            self.controller.empty_cache()
        self.assertEqual(self.controller._backend.drains, 0)


if __name__ == "__main__":
    unittest.main()
