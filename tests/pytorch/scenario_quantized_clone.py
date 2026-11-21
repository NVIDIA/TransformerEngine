# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Exercise quantized clone isolation and gradient flow with CUDA buffers.

Run with ``python tests/pytorch/scenario_quantized_clone.py``. The packed-buffer
lifecycles cover formats whose quantization kernels require newer GPUs. Separate
FP8 current-scaling lifecycles also check decoded values after replacing a clone.
"""

from functools import partial

import torch
import transformer_engine.pytorch as te


SHAPE = (128, 128)


def make_quantized(kind, direction, device, value, *, requires_grad=True):
    """Allocate a real quantized wrapper and initialize every owned buffer."""
    usage = {"rowwise": direction != "column", "columnwise": direction != "row"}
    if kind == "float8":
        quantizer = te.Float8Quantizer(
            torch.ones(1, device=device),
            torch.zeros(1, device=device),
            te.DType.kFloat8E4M3,
            **usage,
        )
    elif kind == "mxfp8":
        quantizer = te.MXFP8Quantizer(te.DType.kFloat8E4M3, **usage)
    elif kind == "nvfp4":
        quantizer = te.NVFP4Quantizer(with_random_sign_mask=False, **usage)
    else:
        quantizer = te.Float8BlockQuantizer(te.DType.kFloat8E4M3, **usage)
    tensor = quantizer.make_empty(
        SHAPE, dtype=torch.float32, device=device, requires_grad=requires_grad
    )
    with torch.no_grad():
        for buffer in buffers(tensor).values():
            buffer.fill_(value)
    return tensor


def make_hybrid(kind, direction, device, value):
    """Place each format in either Hybrid child, including column-only storage."""
    row = make_quantized(
        kind if direction == "row" else "float8", "row", device, value, requires_grad=False
    )
    column = make_quantized(
        kind if direction == "column" else "float8", "column", device, value, requires_grad=False
    )
    return te.HybridQuantizedTensor(
        shape=SHAPE,
        dtype=torch.float32,
        device=device,
        rowwise_storage=row,
        columnwise_storage=column,
        quantizer=te.HybridQuantizer(
            rowwise_quantizer=row._quantizer, columnwise_quantizer=column._quantizer
        ),
        requires_grad=True,
    )


def storages(tensor):
    """Expose the format owners preserved by clone."""
    if isinstance(tensor, te.HybridQuantizedTensor):
        return {"row": tensor.rowwise_sub_storage, "column": tensor.columnwise_sub_storage}
    return {"direct": tensor}


def buffers(tensor):
    """Collect the tensor buffers owned by each storage."""
    return {
        f"{direction}.{name}": value
        for direction, storage in storages(tensor).items()
        for name, value in storage.get_metadata().items()
        if isinstance(value, torch.Tensor)
    }


def clone_and_replace(factory, clone):
    """Clone, backpropagate, replace the clone, and inspect the original."""
    original = factory(1)
    before = {name: value.clone() for name, value in buffers(original).items()}
    copied = clone(original)
    if type(copied) is not type(original) or {
        name: type(storage) for name, storage in storages(copied).items()
    } != {name: type(storage) for name, storage in storages(original).items()}:
        raise RuntimeError("clone changed a quantized storage type")
    copied_buffers = buffers(copied)
    if copied_buffers.keys() != before.keys():
        raise RuntimeError("clone changed buffer presence")
    for name, value in copied_buffers.items():
        if not torch.equal(value, before[name]):
            raise RuntimeError(f"clone changed buffer {name}")

    gradient = torch.full(SHAPE, 3.0, device=original.device)
    copied.backward(gradient)
    if original.grad is None or not torch.equal(original.grad, gradient):
        raise RuntimeError("clone backward changed the incoming gradient")

    replacement = factory(2)
    with torch.no_grad():
        copied.copy_(replacement)
    copied_buffers = buffers(copied)
    for name, value in buffers(original).items():
        if not torch.equal(value, before[name]):
            raise RuntimeError(f"clone.copy_ changed original buffer {name}")
        if value.data_ptr() == copied_buffers[name].data_ptr():
            raise RuntimeError(f"clone shares buffer {name}")
        if not torch.equal(copied_buffers[name], buffers(replacement)[name]):
            raise RuntimeError(f"clone.copy_ did not replace buffer {name}")
    for name, value in buffers(original.detach()).items():
        if value.data_ptr() != buffers(original)[name].data_ptr():
            raise RuntimeError(f"detach lost its alias for {name}")


def fp8_values(device, clone):
    """Check the numerical consequence through the native current-scaling path."""
    quantizer = te.Float8CurrentScalingQuantizer(
        fp8_dtype=te.DType.kFloat8E4M3, device=device, rowwise=True, columnwise=True
    )
    original = quantizer(torch.ones(SHAPE, device=device))
    before = {name: value.clone() for name, value in buffers(original).items()}
    copied = clone(original)
    copied.copy_(quantizer(torch.full(SHAPE, 2.0, device=device)))
    if not torch.equal(original.dequantize(), torch.ones(SHAPE, device=device)):
        raise RuntimeError("clone.copy_ changed original FP8 values")
    if not torch.equal(copied.dequantize(), torch.full(SHAPE, 2.0, device=device)):
        raise RuntimeError("clone.copy_ did not replace FP8 values")
    for name, value in buffers(original).items():
        if not torch.equal(value, before[name]):
            raise RuntimeError(f"clone.copy_ changed original FP8 buffer {name}")


def main():
    """Run every lifecycle and return a failing status for any broken contract."""
    device = torch.device("cuda", torch.cuda.current_device())
    cases = []
    for kind in ("float8", "mxfp8", "nvfp4", "blockwise"):
        for direction in ("row", "column", "both"):
            factory = partial(make_quantized, kind, direction, device)
            cases.append((f"{kind}/{direction}", partial(clone_and_replace, factory)))
        for direction in ("row", "column"):
            factory = partial(make_hybrid, kind, direction, device)
            cases.append((f"hybrid/{kind}/{direction}", partial(clone_and_replace, factory)))
    cases.append(("fp8/current-scaling-values", partial(fp8_values, device)))
    failures = 0
    for name, scenario in cases:
        for api, clone in (("method", lambda tensor: tensor.clone()), ("torch", torch.clone)):
            try:
                scenario(clone)
            except Exception as error:  # pylint: disable=broad-exception-caught
                failures += 1
                print(f"FAIL {name}/{api}: {type(error).__name__}: {error}")
            else:
                print(f"PASS {name}/{api}")
    print(f"Quantized clone lifecycles: {2 * len(cases) - failures} passed, {failures} failed")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
