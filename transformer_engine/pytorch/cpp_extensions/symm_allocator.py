import torch
import os
import gc
import weakref
from typing import List, Tuple, Optional, Dict
from threading import Lock
import torch.distributed._symmetric_memory as symm_mem
from ctypes import pythonapi, c_void_p, py_object


def to_capsule(ptr):
    # Set the return type to py_object to get a Python object (PyCapsule)
    pythonapi.PyCapsule_New.restype = py_object
    pythonapi.PyCapsule_New.argtypes = [c_void_p, c_void_p, c_void_p]
    # Create capsule with a name (optional, can be None) and no destructor
    capsule = pythonapi.PyCapsule_New(ptr, None, None)
    return capsule


class SymmTensor(torch.Tensor):
    """Custom tensor subclass that uses custom memory"""

    @staticmethod
    def __new__(
        cls,
        pool: torch.Tensor,
        offset: int,
        shape: torch.Size,
        dtype: torch.dtype,
        allocator: "SymmAllocator",
    ):
        # Calculate number of elements and bytes
        num_elements = torch.Size(shape).numel()
        element_size = torch.tensor(0, dtype=dtype).element_size()
        nbytes = element_size * num_elements

        # Validate pool
        assert pool.dtype == torch.uint8, f"Expected uint8 pool, got {pool.dtype}"
        assert (
            pool.numel() >= offset + nbytes
        ), f"Pool too small: {pool.numel()} bytes, need {offset + nbytes}"

        # Slice the pool to get the required bytes
        byte_slice = pool[offset : offset + nbytes]

        # Reinterpret the uint8 bytes as the target dtype
        tensor = byte_slice.view(dtype=dtype)
        tensor = tensor.view(*shape)

        # Initialize as a subclass of torch.Tensor
        self = torch.Tensor._make_subclass(cls, tensor)
        if not isinstance(allocator, SymmAllocator):
            raise TypeError(f"Expected SymmAllocator, got {type(allocator)}")
        self._allocator = allocator
        self._ptr = tensor.data_ptr()
        self._offset = offset
        self._size = nbytes
        return self

    def __del__(self):
        """Custom deallocator to return memory to the pool."""
        if hasattr(self, "_allocator") and hasattr(self, "_ptr"):
            self._allocator.free(self._ptr)


class SymmAllocator:
    def __init__(self, size_bytes: int, device: torch.device, dist_group: torch.distributed.group):
        """Initialize the allocator with a preallocated memory pool."""
        # Preallocate the memory pool using torch.empty
        self.reg0_size = 1024  # NVL72*8 plus up to 112 flags
        self.device = device
        self.world_size = torch.distributed.get_world_size(dist_group)
        self.myrank = torch.distributed.get_rank(dist_group)
        self.dist_group = dist_group

        from ..module.base import get_ub

        if os.environ.get("NVTE_USE_UB_FOR_UBNEXT"):
            self.ub_obj = get_ub("ubnext")
            self.internal_pool = self.ub_obj.get_buffer(False).reshape(-1)
            self.mc0_ptr = self.ub_obj.init_ubnext()
            self.pool_size = self.internal_pool.numel()
        else:
            alignment = 2 * 1024 * 1024  # memory is allocated in 2MB pages anyways
            self.pool_size = int((size_bytes + alignment - 1) / alignment) * alignment
            #            symm_mem.set_backend("NCCL")
            self.internal_pool = symm_mem.empty(self.pool_size, dtype=torch.uint8, device=device)
            self.hdl0 = symm_mem.rendezvous(self.internal_pool, dist_group)
            self.mc0_ptr = self.hdl0.multicast_ptr
            self.internal_pool.fill_(0)
            self.internal_pool.view(torch.int64)[: self.world_size].copy_(
                torch.tensor(self.hdl0.buffer_ptrs).view(torch.int64)
            )
            # self.hdl0.barrier(channel=0)
        # Synchronize all processes before proceeding
        torch.distributed.barrier(group=dist_group)

        # Track the raw pointer to the pool
        self.pool_ptr = self.internal_pool.data_ptr()
        # Track allocated segments: (offset, size)
        self.allocated: List[Tuple[int, int]] = []
        # Track free segments: (offset, size)
        self.freelist: List[Tuple[int, int]] = [(self.reg0_size, self.pool_size - self.reg0_size)]
        self.nextpoisoned = None
        self.residual = None
        self.residual_tokens = 0
        self.tensors = weakref.WeakSet()
        self.lock = Lock()

    def allocate(self, nbytes: int) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Allocate nbytes from the pool, returning a pointer and pool reference."""
        with self.lock:
            for i, (offset, size) in enumerate(self.freelist):
                if size >= nbytes:
                    self.freelist.pop(i)
                    self.allocated.append((offset, nbytes))
                    if size > nbytes:
                        self.freelist.append((offset + nbytes, size - nbytes))
                    return self.pool_ptr + offset, self.internal_pool
            return None, None

        # No suitable free segment found
        raise MemoryError(
            f"Preallocated pool exhausted: requested {nbytes} bytes, "
            f"available segments: {self.freelist}"
        )

    def free(self, ptr: int):
        """Free the memory at ptr, returning it to the pool."""
        with self.lock:
            offset = ptr - self.pool_ptr
            for i, (alloc_offset, size) in enumerate(self.allocated):
                if alloc_offset == offset:
                    self.allocated.pop(i)
                    self.freelist.append((offset, size))
                    self.freelist.sort(key=lambda x: x[0])
                    self._merge_free_segments()
                    return
            # Ignore invalid pointers silently
            pass

        raise ValueError(f"Invalid pointer {ptr} not found in allocated segments")

    def _merge_free_segments(self):
        """Merge adjacent free segments to reduce fragmentation."""
        if not self.freelist:
            return
        merged = []
        current_offset, current_size = self.freelist[0]
        for offset, size in self.freelist[1:]:
            if current_offset + current_size == offset:
                # Adjacent segments, merge them
                current_size += size
            else:
                # Non-adjacent, keep current and start new
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        merged.append((current_offset, current_size))
        self.freelist = merged

    def create_tensor(
        self, shape: torch.Size, dtype: torch.dtype = torch.float32
    ) -> Optional[torch.Tensor]:
        """Create a PooledTensor using memory from the pool."""
        nbytes = torch.tensor(0, dtype=dtype).element_size() * torch.Size(shape).numel()
        ptr, pool = self.allocate(nbytes)
        if ptr is None:
            return None
        offset = ptr - self.pool_ptr
        tensor = SymmTensor(pool, offset, torch.Size(shape), dtype, self)
        self.tensors.add(tensor)
        return tensor

    def allreduce_uc(
        self,
        tensor_in: torch.Tensor,
        hidden_size: int = 0,
        residual_in: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        fuse_layernorm: bool = False,
        gamma: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        """Performs in-place allreduce on the given SymmTensor using best algo"""
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        # tensor_out = self.create_tensor(tensor_in.shape, tensor_in.dtype)

        ucptr_in = tensor_in.data_ptr()
        # mcptr_out = tensor_out.data_ptr()
        nbytes = tensor_in.numel() * tensor_in.element_size()

        # Import your pybind module if not imported
        from transformer_engine_torch import allreduce_2shot_uc

        allreduce_2shot_uc(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(ucptr_in),
            to_capsule(ucptr_in),  # out
            nbytes,
            to_capsule(residual_in.data_ptr()) if residual_in is not None else None,
            to_capsule(residual_out.data_ptr()) if residual_out is not None else None,
            fuse_layernorm,
            to_capsule(gamma.data_ptr()) if gamma is not None else None,
            eps if eps is not None else 0.0,
            hidden_size,
        )
        return tensor_in

    def allreduce_simple(
        self,
        tensor_in: torch.Tensor,
        hidden_size: int = 0,
        residual_in: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        fuse_layernorm: bool = False,
        gamma: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        """Performs in-place allreduce on the given SymmTensor using best algo"""
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        # tensor_out = self.create_tensor(tensor_in.shape, tensor_in.dtype)

        mcptr_in = self.mc0_ptr + (tensor_in.data_ptr() - self.internal_pool.data_ptr())
        # mcptr_out = self.hdl.multicast_ptr + (tensor_out.data_ptr() - self.internal_pool.data_ptr())
        nbytes = tensor_in.numel() * tensor_in.element_size()

        # Import your pybind module if not imported
        from transformer_engine_torch import allreduce_2shot_mc

        allreduce_2shot_mc(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(self.mc0_ptr),
            to_capsule(mcptr_in),
            to_capsule(mcptr_in),  # out
            nbytes,
            to_capsule(residual_in.data_ptr()) if residual_in is not None else None,
            to_capsule(residual_out.data_ptr()) if residual_out is not None else None,
            fuse_layernorm,
            to_capsule(gamma.data_ptr()) if gamma is not None else None,
            eps if eps is not None else 0.0,
            hidden_size,
        )
        return tensor_in

    def allreduce_lamport(
        self,
        tensor_in: torch.Tensor,
        hidden_size: int = 0,
        residual_in: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        fuse_layernorm: bool = False,
        gamma: Optional[torch.Tensor] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Performs allreduce using 2-shot multicast Lamport variant:
        - Takes `tensor_in` as input (SymmTensor).
        - Allocates `tensor_out` of same shape and dtype.
        - Runs `allreduce_2shot_mc_lamport` over them.
        - Returns `tensor_out`.
        """
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"
        if self.mc0_ptr is None or self.mc0_ptr == 0:
            return self.allreduce_uc(
                tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps
            )
        from transformer_engine_torch import allreduce_2shot_mc_lamport

        # Allocate output tensor of same shape/dtype
        tensor_out = self.nextpoisoned
        poisonedout = True

        if self.nextpoisoned is None or self.nextpoisoned.shape != tensor_in.shape:
            if self.nextpoisoned is not None:
                del self.nextpoisoned
                self.nextpoisoned = None
            tensor_out = self.create_tensor(tensor_in.shape, tensor_in.dtype)
            poisonedout = False
        if tensor_out is None:
            return self.allreduce_simple(
                tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps
            )

        # alllcate potential output for next allreduce (speculative) and poison it now
        self.nextpoisoned = self.create_tensor(tensor_in.shape, tensor_in.dtype)

        # Calculate mcptr_in and mcptr_out with offset relative to internal_pool
        offset = tensor_in.data_ptr() - self.internal_pool.data_ptr()
        mcptr_in = self.mc0_ptr + offset
        mcptr_out = self.mc0_ptr + (tensor_out.data_ptr() - self.internal_pool.data_ptr())

        # Use clear_ptr to clear output memory before reduction; here we use tensor_out
        # clear_ptr = self.nextpoisoned.data_ptr() if self.nextpoisoned is not None else 0

        nbytes = tensor_in.numel() * tensor_in.element_size()

        # Call your pybind lamport allreduce
        allreduce_2shot_mc_lamport(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(self.mc0_ptr),
            to_capsule(tensor_out.data_ptr()),
            to_capsule(mcptr_in),
            to_capsule(mcptr_out),
            to_capsule(self.nextpoisoned.data_ptr()) if self.nextpoisoned is not None else None,
            nbytes,
            poisonedout,
            to_capsule(residual_in.data_ptr()) if residual_in is not None else None,
            to_capsule(residual_out.data_ptr()) if residual_out is not None else None,
            fuse_layernorm,
            to_capsule(gamma.data_ptr()) if gamma is not None else None,
            eps if eps is not None else 0.0,
            hidden_size,
        )

        return tensor_out


_allocator_map: Dict[torch.distributed.group, Tuple[int, "SymmAllocator"]] = {}


def ubsymm_request_allocator(
    dist_group: torch.distributed.group,
    shape: Optional[torch.Size] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    if shape is not None:
        num_elements = torch.Size(shape).numel()
        element_size = torch.tensor(0, dtype=dtype).element_size()
        tensor_size = num_elements * element_size
    else:
        tensor_size = 0

    if dist_group not in _allocator_map:
        if os.environ.get("NVTE_USE_UB_FOR_UBNEXT"):
            assert not _allocator_map, "Current UBNEXT-UB bypass supports only one process group."
        _allocator_map[dist_group] = (tensor_size, None)
    else:
        old_size, allocator = _allocator_map[dist_group]
        assert allocator is None, "Second element of tuple must be None"
        max_size = max(old_size, tensor_size)
        _allocator_map[dist_group] = (max_size, None)


def ubsymm_get_sym_tensor(
    shape: torch.Size, dtype: torch.dtype, dist_group: torch.distributed.group
) -> torch.Tensor:
    if dtype != torch.bfloat16:
        return None  # Unsupported dtype, do fallback to nccl
    if dist_group not in _allocator_map:
        return None  # No allocator requested earlier, do fallback to nccl
    (max_size, allocator) = _allocator_map[dist_group]
    if allocator is None:
        new_max_size = int(
            os.environ.get("NVTE_UB_SYMM_POOL_SIZE", ((6 * max_size + 1048575) / 1024 / 1024))
        )
        allocator = SymmAllocator(
            new_max_size * 1024 * 1024,
            torch.device(f"cuda:{torch.cuda.current_device()}"),
            dist_group,
        )
        _allocator_map[dist_group] = (new_max_size, allocator)
    return allocator.create_tensor(shape, dtype)


def ubsymm_allreduce(
    tensor_in: SymmTensor,
    residual_global: Optional[torch.Tensor] = None,
    gamma: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
) -> SymmTensor:
    """
    Performs allreduce on the given SymmTensor using best algo
    Four modes:
     standalone allreduce: no residual, no layernorm (residual_global passed by user, both eps and gamma=None)
     first PROJ layer: layernorm fused, global residual in, internal residual out (residual_global passed by user, both eps and gamma not None)
        this allocates internal residual if it wasnt allocated previously or token count is different from previous allreduce
     middle layers: layernorm fused, internal residual in, internal residual out (residual_global=None, both eps and gamma not None)
     Last FC2 layer: no layernorm, internal residual in, no residual out(layer output is actually the global residual) (residual_global=None, fboth eps and gamma=None)
       this is different from standalone once there is no internal residual allocated
    """
    fuse_layernorm = gamma is not None and eps is not None
    internal_residual = tensor_in._allocator.residual
    num_ranks = tensor_in._allocator.world_size
    hidden_size = (
        tensor_in.shape[-1]
        if fuse_layernorm or internal_residual is not None or residual_global is not None
        else tensor_in.numel() // num_ranks
    )
    num_tokens = tensor_in.numel() // hidden_size
    myrank = tensor_in._allocator.myrank
    if residual_global is not None and (
        internal_residual is None or tensor_in._allocator.residual_tokens != num_tokens
    ):
        my_tokens = num_tokens // num_ranks
        extra_tokens = num_tokens % num_ranks
        first_token = myrank * my_tokens
        if myrank < extra_tokens:
            my_tokens += 1
            first_token += myrank
        else:
            first_token += extra_tokens
        if my_tokens == 0:
            my_tokens = 1  # avoid empty residual
        if tensor_in._allocator.residual is not None:
            del tensor_in._allocator.residual
        tensor_in._allocator.residual = torch.empty(
            my_tokens * hidden_size, dtype=tensor_in.dtype, device=tensor_in.device
        )
        tensor_in._allocator.residual_tokens = num_tokens
        internal_residual = tensor_in._allocator.residual

    residual_in = residual_global if residual_global is not None else internal_residual

    residual_out = (
        internal_residual if fuse_layernorm else None
    )  # without layernorm new full residual is output of allreduce
    if tensor_in.numel() > 1048576:
        return tensor_in._allocator.allreduce_simple(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps
        )
    else:
        return tensor_in._allocator.allreduce_lamport(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps
        )


def ubsymm_free_residual(tensor_in: SymmTensor):
    if tensor_in._allocator.residual is not None:
        del tensor_in._allocator.residual
        tensor_in._allocator.residual_tokens = 0
        tensor_in._allocator.residual = None
