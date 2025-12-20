from code import interact
import torch
import os
import gc
import weakref
from typing import List, Tuple, Optional, Dict
from threading import Lock
import torch.distributed._symmetric_memory as symm_mem
from ctypes import pythonapi, c_void_p, py_object
from transformer_engine_torch import ubnext_allreduce_2shot_uc, ubnext_allreduce_2shot_mc, ubnext_allreduce_2shot_mc_lamport, ubnext_allgather_mc, ubnext_alltoall


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
        allocator.tensors.add(self)
        allocator.allocated_change(self._ptr, 1)
        return self

    def __del__(self):
        """Custom deallocator to return memory to the pool."""
        if hasattr(self, "_allocator") and hasattr(self, "_ptr"):
            self._allocator.free(self._ptr)

    def view(self, *shape):
        """
        Returns a new SymmTensor view with the same backing memory but different shape.
        Handles -1 inference manually without temporary tensors.
        """
        if not hasattr(self, '_allocator'):
            return self.as_subclass(torch.Tensor).view(*shape)
        # Convert shape to list for modification
        shape_list = list(shape)
        if len(shape_list) == 1 and isinstance(shape_list[0], (tuple, list)):
            shape_list = list(shape_list[0])
        
        original_numel = self.numel()
        
        # Handle -1 inference
        has_infer_dim = -1 in shape_list
        if has_infer_dim:
            if shape_list.count(-1) > 1:
                raise RuntimeError("Only one dimension can be inferred (contains -1)")
            
            infer_idx = shape_list.index(-1)
            known_size = 1
            for i, dim in enumerate(shape_list):
                if i != infer_idx:
                    if dim < 0:
                        raise RuntimeError("Only -1 is supported for inference")
                    known_size *= dim
            
            if original_numel % known_size != 0:
                raise RuntimeError(
                    f"Shape inference failed: {original_numel} elements not divisible by {known_size}"
                )
            
            inferred_dim = original_numel // known_size
            shape_list[infer_idx] = inferred_dim
            resolved_shape = torch.Size(shape_list)
        else:
            resolved_shape = torch.Size(shape_list)
        
        # Validate total elements match
        if resolved_shape.numel() != original_numel:
            raise RuntimeError(
                f"View size mismatch: original has {original_numel} elements, "
                f"resolved shape {resolved_shape} requires {resolved_shape.numel()} elements"
            )
        
        # Create SymmTensor with resolved shape, same backing memory
        new_tensor = SymmTensor.__new__(
            SymmTensor,
            self._allocator.internal_pool,
            self._offset,
            resolved_shape,
            self.dtype,
            self._allocator
        )
        return new_tensor


'''
    def clone(self):
        new_tensor = torch.empty(self.shape, dtype=self.dtype)
        new_tensor.copy_(self)
        return new_tensor
        
    def clone(self):
        return SymmTensor.__new__(
            SymmTensor,
            self._allocator.internal_pool,
            self._offset,
            self.shape,
            self.dtype,
            self._allocator
        )
        return new_tensor
'''

torch.serialization.add_safe_globals([SymmTensor])

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
            self.ub_obj = get_ub("ubnext",use_fp8=False)
            self.internal_pool = self.ub_obj.get_buffer(False).reshape(-1)
            self.mc0_ptr = self.ub_obj.init_ubnext()
            self.pool_size = self.internal_pool.numel()
        else:
            alignment = 2 * 1024 * 1024  # memory is allocated in 2MB pages anyways
            self.pool_size = int((size_bytes + alignment - 1) / alignment) * alignment
            self.internal_pool = symm_mem.empty(self.pool_size, dtype=torch.uint8, device=device)
            self.hdl0 = symm_mem.rendezvous(self.internal_pool, dist_group)
            self.mc0_ptr = self.hdl0.multicast_ptr
            self.internal_pool.fill_(0)
            self.internal_pool.view(torch.int64)[: self.world_size].copy_(
                torch.tensor(self.hdl0.buffer_ptrs).view(torch.int64)
            )
        # Synchronize all processes before proceeding
        torch.distributed.barrier(group=dist_group)

        # Track the raw pointer to the pool
        self.pool_ptr = self.internal_pool.data_ptr()
        # Track allocated segments: (offset, size, reference count)
        self.allocated: Dict[bool, List[Tuple[int, int, int]]] = {True: [], False: []}
        # Track free segments: (offset, size)
        self.graph_pool_size = int(self.pool_size * float(os.environ.get("NVTE_UBNEXT_GRAPH_POOL_SHARE", 0.9)))
        self.graph_pool_size = int(self.graph_pool_size//4096)*4096

        self.freelist: Dict[bool, List[Tuple[int, int]]] = {True: [(self.reg0_size, self.graph_pool_size - self.reg0_size)], False: [(self.graph_pool_size, self.pool_size - self.graph_pool_size)]}
        self.nextpoisoned: Dict[bool, Optional[SymmTensor]] = {True: None, False: None}
        self.residual = None
        self.residual_global = None
        self.residual_tokens = 0
        self.tensors = weakref.WeakSet()
        self.lock = Lock()
        self.nchunks = 1
        self.current_chunk = 0
        self.dummy = os.environ.get("NVTE_UBNEXT_DUMMY")
        self.debug = os.environ.get("NVTE_UBNEXT_DEBUG")
        if self.debug:
            print(f"Rank {self.myrank} Graph pool size: {self.graph_pool_size}")
            print(f"Rank {self.myrank} Non-graph pool size: {self.pool_size - self.graph_pool_size}")
            print(f"Rank {self.myrank} Reg0 size: {self.reg0_size}")
            print(f"Rank {self.myrank} Total pool size: {self.pool_size}")
        self.used_uc = False
        self.used_simple = False
        self.lamport_out = None
        self.lamport_poisoned = False

    def allocate(self, nbytes: int) -> Tuple[Optional[int], Optional[torch.Tensor]]:
        """Allocate nbytes from the pool, returning a pointer and pool reference."""
        graph_mode = torch.cuda.is_current_stream_capturing()
        with self.lock:
            for i, (offset, size) in enumerate(self.freelist[graph_mode]):
                if size >= nbytes:
                    self.freelist[graph_mode].pop(i)
                    self.allocated[graph_mode].append((offset, nbytes,0))
                    if size > nbytes:
                        self.freelist[graph_mode].append((offset + nbytes, size - nbytes))
                    if self.debug:
                        print(f"Rank {self.myrank} Allocated {nbytes} bytes at {offset} cudagraph caputuring: {graph_mode}")
                    return self.pool_ptr + offset, self.internal_pool
            if self.debug:
                print(f"Rank {self.myrank} No suitable free segment found for {nbytes} bytes, allocated list: {self.allocated[graph_mode]}, free list: {self.freelist[graph_mode]} cudagraph capturing mode: {graph_mode}")
            return None, None

        # No suitable free segment found
        raise MemoryError(
            f"Preallocated pool exhausted in graph:{graph_mode} mode: requested {nbytes} bytes, "
            f"available segments: {self.freelist[graph_mode]}"
        )

    def allocated_change(self, ptr: int, change: int):
        """Free the memory at ptr, returning it to the pool."""
        offset = ptr - self.pool_ptr
        graph_mode  = offset < self.graph_pool_size
        with self.lock:

            for i, (alloc_offset, size,ref_count) in enumerate(self.allocated[graph_mode]):
                if alloc_offset == offset:
                    self.allocated[graph_mode].pop(i)
                    ref_count += change
                    if ref_count == 0:
                        self.freelist[graph_mode].append((offset, size))
                        self.freelist[graph_mode].sort(key=lambda x: x[0])
                        self._merge_free_segments(graph_mode)
                        if self.debug:
                            print(f"Rank {self.myrank} Freed {size} bytes at {offset} cudagraph caputuring: {graph_mode}")
                    else:
                        self.allocated[graph_mode].append((offset, size, ref_count))
                        if self.debug:
                            print(f"Rank {self.myrank} Refcount changed to {ref_count} for {size} bytes at {offset} cudagraph caputuring: {graph_mode}")
                    return
            # Ignore invalid pointers silently
            pass
        raise ValueError(f"Invalid pointer {ptr} not found in allocated segments")

    def free(self, ptr: int):
        self.allocated_change(ptr, -1)

    def _merge_free_segments(self, graph_mode: bool):
        """Merge adjacent free segments to reduce fragmentation."""
        if not self.freelist[graph_mode]:
            return
        merged = []
        current_offset, current_size = self.freelist[graph_mode][0]
        for offset, size in self.freelist[graph_mode][1:]:
            if current_offset + current_size == offset:
                # Adjacent segments, merge them
                current_size += size
            else:
                # Non-adjacent, keep current and start new
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        merged.append((current_offset, current_size))
        self.freelist[graph_mode] = merged

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
        #self.tensors.add(tensor)
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
        smlimit: int = 0,
        cgasize: int = 0,
    ) -> torch.Tensor:
        """Performs in-place allreduce on the given SymmTensor using best algo"""
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        nbytes = tensor_in.numel() * tensor_in.element_size() // tensor_in._allocator.nchunks
        ucptr_in = tensor_in.data_ptr() + nbytes * tensor_in._allocator.current_chunk

        ubnext_allreduce_2shot_uc(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(ucptr_in),
            to_capsule(ucptr_in),  # out
            nbytes,
            to_capsule(residual_in.data_ptr() + nbytes * tensor_in._allocator.current_chunk * residual_in.element_size()) if residual_in is not None else None,
            to_capsule(residual_out.data_ptr() + nbytes * tensor_in._allocator.current_chunk * residual_out.element_size()) if residual_out is not None else None,
            fuse_layernorm,
            to_capsule(gamma.data_ptr()) if gamma is not None else None,
            eps if eps is not None else 0.0,
            hidden_size,
            smlimit,
            cgasize,
            tensor_in._allocator.current_chunk,
            False,
        )
        tensor_in._allocator.current_chunk += 1
        if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks:
            tensor_in._allocator.current_chunk = 0
            tensor_in._allocator.used_uc = False
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
        smlimit: int = 0,
        cgasize: int = 0,
    ) -> torch.Tensor:
        """Performs in-place allreduce on the given SymmTensor using best algo"""
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        nbytes = tensor_in.numel() * tensor_in.element_size() // tensor_in._allocator.nchunks
        mcptr_in = self.mc0_ptr + (tensor_in.data_ptr() - self.internal_pool.data_ptr()) + nbytes * tensor_in._allocator.current_chunk

        ubnext_allreduce_2shot_mc(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(self.mc0_ptr),
            to_capsule(mcptr_in),
            to_capsule(mcptr_in),  # out
            nbytes,
            to_capsule(residual_in.data_ptr() + nbytes * tensor_in._allocator.current_chunk * residual_in.element_size()) if residual_in is not None else None,
            to_capsule(residual_out.data_ptr() + nbytes * tensor_in._allocator.current_chunk * residual_out.element_size()) if residual_out is not None else None,
            fuse_layernorm,
            to_capsule(gamma.data_ptr()) if gamma is not None else None,
            eps if eps is not None else 0.0,
            hidden_size,
            smlimit,
            cgasize,
            tensor_in._allocator.current_chunk,
            False,
        )
        tensor_in._allocator.current_chunk += 1
        tensor_in._allocator.used_simple = True
        if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks:
            tensor_in._allocator.current_chunk = 0
            tensor_in._allocator.used_simple = False
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
        smlimit: int = 0,
        cgasize: int = 0,
    ) -> torch.Tensor:
        """
        Performs allreduce using 2-shot multicast Lamport variant:
        - Takes `tensor_in` as input (SymmTensor).
        - Allocates `tensor_out` of same shape and dtype.
        - Runs `allreduce_2shot_mc_lamport` over them.
        - Returns `tensor_out`.
        """
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"
        if self.mc0_ptr is None or self.mc0_ptr == 0 or tensor_in._allocator.used_uc:
            return self.allreduce_uc(
                tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,
                smlimit, cgasize
            )

        # Allocate output tensor of same shape/dtype
        graph_mode = torch.cuda.is_current_stream_capturing()
        tensor_out = self.nextpoisoned[graph_mode] if tensor_in._allocator.current_chunk == 0 else self.lamport_out
        poisonedout = True if tensor_in._allocator.current_chunk == 0 else self.lamport_poisoned

        if tensor_in._allocator.current_chunk == 0 and (tensor_out is None or tensor_out.shape != tensor_in.shape):
            if self.nextpoisoned[graph_mode] is not None:
                del self.nextpoisoned[graph_mode]
                self.nextpoisoned[graph_mode] = None
            tensor_out = self.create_tensor(tensor_in.shape, tensor_in.dtype)
            poisonedout = False

        if tensor_out is None or tensor_in._allocator.used_simple:
            return self.allreduce_simple(
                tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,
                smlimit, cgasize
            )
        tensor_in._allocator.lamport_out = tensor_out
        tensor_in._allocator.lamport_poisoned = poisonedout
        # allocate potential output for next allreduce (speculative) and poison it now
        if tensor_in._allocator.current_chunk == 0:
            self.nextpoisoned[graph_mode] = self.create_tensor(tensor_in.shape, tensor_in.dtype)

        nbytes = tensor_in.numel() * tensor_in.element_size() // tensor_in._allocator.nchunks
        offset_in = tensor_in.data_ptr() - self.internal_pool.data_ptr()
        offset_out = tensor_out.data_ptr() - self.internal_pool.data_ptr()
        mcptr_in = self.mc0_ptr + offset_in + nbytes * tensor_in._allocator.current_chunk
        mcptr_out = self.mc0_ptr + offset_out + nbytes * tensor_in._allocator.current_chunk

        ubnext_allreduce_2shot_mc_lamport(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(self.mc0_ptr),
            to_capsule(tensor_out.data_ptr() + nbytes * tensor_in._allocator.current_chunk),
            to_capsule(mcptr_in),
            to_capsule(mcptr_out),
            to_capsule(self.nextpoisoned[graph_mode].data_ptr() + nbytes * tensor_in._allocator.current_chunk) if self.nextpoisoned[graph_mode] is not None else None,
            nbytes,
            poisonedout,
            to_capsule(residual_in.data_ptr() + residual_in.numel() // tensor_in._allocator.nchunks * tensor_in._allocator.current_chunk * residual_in.element_size()) if residual_in is not None else None,
            to_capsule(residual_out.data_ptr() + residual_out.numel() // tensor_in._allocator.nchunks * tensor_in._allocator.current_chunk * residual_out.element_size()) if residual_out is not None else None,
            fuse_layernorm,
            to_capsule(gamma.data_ptr()) if gamma is not None else None,
            eps if eps is not None else 0.0,
            hidden_size,
            smlimit,
            cgasize,
            tensor_in._allocator.current_chunk,
            False,
        )
        tensor_in._allocator.current_chunk += 1
        if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks:
            tensor_in._allocator.current_chunk = 0
            self.lamport_out = None
            self.lamport_poisoned = False
        return tensor_out

    def allgather_mc(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
    ) -> SymmTensor:
        """
        Performs allgather using multicast:
        - Takes `tensor_in` as input (torch.Tensor).
        - Allocates `tensor_out` of Nranks * tensor_in.shape shape and dtype.
        - Runs `allgather_mc` over them.
        - Returns `tensor_out`.
        """
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"

        tensor_out = self.create_tensor(torch.Size([self.world_size * tensor_in.shape[0], *tensor_in.shape[1:]]), tensor_in.dtype)
        offset_out = tensor_out.data_ptr() - self.internal_pool.data_ptr()
        mcptr_out = self.mc0_ptr + offset_out
        ubnext_allgather_mc(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(self.mc0_ptr),
            to_capsule(tensor_in.data_ptr()),
            to_capsule(mcptr_out),
            tensor_in.numel() * tensor_in.element_size(),
            smlimit,
        )
        return tensor_out
    
    def alltoall(
        self,
        tensor_in: torch.Tensor,
        smlimit: int = 0,
    ) -> SymmTensor:
        """
        Performs alltoall using multicast:
        - Takes `tensor_in` as input (torch.Tensor).
        - Allocates `tensor_out` of tensor_in.shape shape and dtype.
        - Runs `alltoall_mc` over them.
        - Returns `tensor_out`.
        """
        assert tensor_in.device == self.device, "Tensor device mismatch with allocator device"
        tensor_out = self.create_tensor(tensor_in.shape, tensor_in.dtype)

        ubnext_alltoall(
            self.world_size,
            self.myrank,
            to_capsule(self.internal_pool.data_ptr()),
            to_capsule(self.mc0_ptr),
            to_capsule(tensor_in.data_ptr()),
            to_capsule(tensor_out.data_ptr()),
            tensor_in.numel() * tensor_in.element_size()//self.world_size,
            smlimit,
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
    gamma: Optional[torch.Tensor] = None,
    eps: Optional[float] = None,
    smlimit: int = 0,
    cgasize: int = 0,
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

    if tensor_in._allocator.dummy:
        return tensor_in
    if tensor_in._allocator.debug:
        print(f"UBNEXT ALLREDUCE: {tensor_in.shape} gamma None:{gamma is None} eps None:{eps is None}")

    fuse_layernorm = gamma is not None and eps is not None
    internal_residual = tensor_in._allocator.residual
    residual_global = tensor_in._allocator.residual_global
    num_ranks = tensor_in._allocator.world_size
    hidden_size = (
        tensor_in.shape[-1]
        if fuse_layernorm or internal_residual is not None or residual_global is not None
        else tensor_in.numel() // num_ranks
    )
    assert (tensor_in.numel() // hidden_size) % tensor_in._allocator.nchunks == 0, "Token count must be divisible by nchunks"

    num_tokens = (tensor_in.numel() // hidden_size) // tensor_in._allocator.nchunks
    myrank = tensor_in._allocator.myrank
    if residual_global is not None and tensor_in._allocator.current_chunk == 0 and (
        internal_residual is None or tensor_in._allocator.residual_tokens != num_tokens  or tensor_in._allocator.residual_chunks != tensor_in._allocator.nchunks
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
            my_tokens = 1  # avoid empty residual shard
        if tensor_in._allocator.residual is not None:
            del tensor_in._allocator.residual
        tensor_in._allocator.residual = torch.empty(
            my_tokens * tensor_in._allocator.nchunks * hidden_size, dtype=tensor_in.dtype, device=tensor_in.device
        )
        tensor_in._allocator.residual_tokens = num_tokens
        tensor_in._allocator.residual_chunks = tensor_in._allocator.nchunks
        internal_residual = tensor_in._allocator.residual

    residual_in = residual_global if residual_global is not None else internal_residual

    residual_out = (
        internal_residual if fuse_layernorm else None
    )  # without layernorm new full residual is output of allreduce
    
    if tensor_in._allocator.current_chunk == tensor_in._allocator.nchunks - 1:
        tensor_in._allocator.residual_global = None

    if tensor_in.numel() // tensor_in._allocator.nchunks > 1048576:
        return tensor_in._allocator.allreduce_simple(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,  
            smlimit, cgasize
        )
    else:
        return tensor_in._allocator.allreduce_lamport(
            tensor_in, hidden_size, residual_in, residual_out, fuse_layernorm, gamma, eps,
            smlimit, cgasize
        )

def ubsymm_free_residual(tensor_in: SymmTensor):
    if tensor_in._allocator.residual is not None:
        del tensor_in._allocator.residual
        tensor_in._allocator.residual_tokens = 0
        tensor_in._allocator.residual = None

def ubsymm_restore(tensor: torch.Tensor, dist_group: torch.distributed.group) -> SymmTensor:
    """
    Restores a torch.Tensor to a SymmTensor if its data pointer is within the allocator's internal pool.
    Otherwise, returns the original tensor.
    """
    if dist_group not in _allocator_map:
        return tensor
    (_, allocator) = _allocator_map[dist_group]
    ptr = tensor.data_ptr()
    pool_ptr = allocator.pool_ptr
    pool_size = allocator.pool_size
    if pool_ptr <= ptr < pool_ptr + pool_size:
        offset = ptr - pool_ptr
        # Calculate nbytes for validation
        num_elements = tensor.numel()
        element_size = tensor.element_size()
        nbytes = element_size * num_elements
        # Validate
        if allocator.internal_pool.numel() < offset + nbytes:
            raise ValueError(f"Offset {offset} + {nbytes} bytes exceeds pool size {allocator.internal_pool.numel()}")
        # Create SymmTensor
        symm_tensor = SymmTensor.__new__(
            SymmTensor,
            allocator.internal_pool,
            offset,
            tensor.shape,
            tensor.dtype,
            allocator
        )
        del tensor
        return symm_tensor
    else:
        return tensor

def ubsymm_mem_stats():
    for dist_group, (_, allocator) in _allocator_map.items():
        print(f"Rank {allocator.myrank} Graph pool used size: {sum(size for _, size, _ in allocator.allocated[True])/1024/1024} MB")
        print(f"Rank {allocator.myrank} Non-graph pool used size: {sum(size for _, size, _ in allocator.allocated[False])/1024/1024} MB")