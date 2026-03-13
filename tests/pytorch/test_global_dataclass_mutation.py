"""
Experiment: can torch.compile handle mutation of a global dataclass?

Analogous to the global-dict experiment, but uses a dataclass instance
stored as a module-level global instead of a plain dict.

Parts:
  1. Read a field from the global dataclass, check if recompilation happens
     when the field value changes.
  2. Write a Python scalar to a dataclass field inside a compiled function.
  3. Write a Tensor to a dataclass field inside a compiled function.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Global dataclass
# ---------------------------------------------------------------------------

@dataclass
class State:
    scale: float = 1.0
    result: Optional[int] = None
    tensor_val: Optional[torch.Tensor] = None


GLOBAL_STATE = State()


# ---------------------------------------------------------------------------
# Functions that access / mutate the global dataclass
# ---------------------------------------------------------------------------


def fn_read_dataclass(x: torch.Tensor) -> torch.Tensor:
    """Read scale from the global dataclass and multiply x by it."""
    return x * GLOBAL_STATE.scale


def fn_write_scalar(x: torch.Tensor, value: int) -> torch.Tensor:
    """Write a Python scalar to the global dataclass, return x unchanged."""
    GLOBAL_STATE.result = value
    return x + 0


def fn_write_tensor(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Write a Tensor to the global dataclass, return x unchanged."""
    GLOBAL_STATE.tensor_val = t
    return x + 0


# ---------------------------------------------------------------------------
# Compiled versions
# ---------------------------------------------------------------------------
compiled_read = torch.compile(fn_read_dataclass, fullgraph=False)
compiled_write_scalar = torch.compile(fn_write_scalar, fullgraph=False)
compiled_write_tensor = torch.compile(fn_write_tensor, fullgraph=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset():
    global GLOBAL_STATE
    GLOBAL_STATE = State()


def unique_graphs():
    return torch._dynamo.utils.counters["stats"].get("unique_graphs", "?")


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Experiment: torch.compile + global dataclass mutation")
    print("=" * 60)

    x = torch.tensor([2.0], device="cpu")

    # -----------------------------------------------------------------------
    # Part 1 – reading a field from the global dataclass
    # -----------------------------------------------------------------------
    print("\n--- Part 1: reading a field from a global dataclass ---")
    reset()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    GLOBAL_STATE.scale = 3.0
    y1 = compiled_read(x)
    g1 = unique_graphs()
    print(f"  GLOBAL_STATE.scale = 3.0  →  compiled_read(x) = {y1.item()}  (expected {x.item() * 3.0})")
    print(f"  unique_graphs after 1st call: {g1}")

    GLOBAL_STATE.scale = 5.0
    y2 = compiled_read(x)
    g2 = unique_graphs()
    print(f"  GLOBAL_STATE.scale = 5.0  →  compiled_read(x) = {y2.item()}  (expected {x.item() * 5.0})")
    print(f"  unique_graphs after 2nd call: {g2}")

    if g2 != g1:
        print(f"  NOTE: Dynamo recompiled (graphs: {g1} -> {g2})")
    else:
        print(f"  NOTE: Dynamo did NOT recompile")

    if abs(y2.item() - x.item() * 5.0) < 1e-6:
        print("  PASS: result reflects updated dataclass field")
    else:
        print("  FAIL: result does NOT reflect updated field (guard baked-in old value)")

    # -----------------------------------------------------------------------
    # Part 2 – writing a Python scalar to the dataclass
    # -----------------------------------------------------------------------
    print("\n--- Part 2: writing a Python scalar to a dataclass field ---")
    reset()
    torch._dynamo.reset()

    print(f"  GLOBAL_STATE.result before call: {GLOBAL_STATE.result}")
    compiled_write_scalar(x, 42)
    print(f"  GLOBAL_STATE.result after call:  {GLOBAL_STATE.result}")

    if GLOBAL_STATE.result == 42:
        print("  PASS: dataclass field mutation (scalar) is visible after compiled call")
    else:
        print("  FAIL: dataclass field mutation (scalar) NOT visible")

    # -----------------------------------------------------------------------
    # Part 3 – writing a Tensor to the dataclass
    # -----------------------------------------------------------------------
    print("\n--- Part 3: writing a Tensor to a dataclass field ---")
    reset()
    torch._dynamo.reset()

    t = torch.tensor(99.0)
    print(f"  GLOBAL_STATE.tensor_val before call: {GLOBAL_STATE.tensor_val}")
    compiled_write_tensor(x, t)
    print(f"  GLOBAL_STATE.tensor_val after call:  {GLOBAL_STATE.tensor_val}")

    if GLOBAL_STATE.tensor_val is not None:
        print("  PASS: dataclass field mutation (Tensor) is visible after compiled call")
    else:
        print("  FAIL: dataclass field mutation (Tensor) NOT visible")

    print("\nDone.")


if __name__ == "__main__":
    run()
