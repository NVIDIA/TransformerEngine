"""
Experiment: can torch.compile handle mutation of a global dictionary?

We test two scenarios:
  1. A compiled function that reads from a global dict.
  2. A compiled function that writes (mutates) a global dict.

In both cases we check whether recompilation or graph breaks occur, and
whether the results are numerically correct.
"""

import torch

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
GLOBAL_DICT: dict = {}

# ---------------------------------------------------------------------------
# Functions that access / mutate the global dict
# ---------------------------------------------------------------------------


def fn_read_global(x: torch.Tensor) -> torch.Tensor:
    """Read a scale factor stored in a global dict and multiply x by it."""
    scale = GLOBAL_DICT.get("scale", 1.0)
    return x * scale


def fn_write_global(x: torch.Tensor, key: str, value) -> torch.Tensor:
    """Write a value into the global dict, then return x unchanged."""
    GLOBAL_DICT[key] = value
    return x + 0  # trivial op so there is a tensor output


# ---------------------------------------------------------------------------
# Compiled versions
# ---------------------------------------------------------------------------
compiled_read = torch.compile(fn_read_global, fullgraph=False)
compiled_write = torch.compile(fn_write_global, fullgraph=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def reset():
    global GLOBAL_DICT
    GLOBAL_DICT = {}


def count_recompilations(fn):
    """Return the number of frames that have been compiled so far."""
    # torch._dynamo.explain() gives per-call stats; we use the simpler
    # guard cache size as a proxy.
    try:
        return torch._dynamo.utils.counters["stats"]["unique_graphs"]
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("Experiment: torch.compile + global dict mutation")
    print("=" * 60)

    x = torch.tensor([2.0], device="cpu")

    # -----------------------------------------------------------------------
    # Part 1 – reading from the global dict
    # -----------------------------------------------------------------------
    print("\n--- Part 1: reading from a global dict ---")
    reset()
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()

    GLOBAL_DICT["scale"] = 3.0
    y1 = compiled_read(x)
    graphs_after_first = torch._dynamo.utils.counters["stats"].get("unique_graphs", "?")
    print(f"  GLOBAL_DICT = {GLOBAL_DICT}")
    print(f"  compiled_read(x) = {y1.item()}  (expected {x.item() * 3.0})")
    print(f"  unique_graphs after 1st call: {graphs_after_first}")

    # Change the dict value and call again – should Dynamo pick up the change?
    GLOBAL_DICT["scale"] = 5.0
    y2 = compiled_read(x)
    graphs_after_second = torch._dynamo.utils.counters["stats"].get("unique_graphs", "?")
    print(f"  After mutating scale to 5.0:")
    print(f"  compiled_read(x) = {y2.item()}  (expected {x.item() * 5.0})")
    print(f"  unique_graphs after 2nd call: {graphs_after_second}")

    if graphs_after_second != graphs_after_first:
        print(f"  NOTE: Dynamo recompiled (graphs: {graphs_after_first} -> {graphs_after_second})")
    else:
        print(f"  NOTE: Dynamo did NOT recompile (same graph count)")

    if abs(y2.item() - x.item() * 5.0) < 1e-6:
        print("  PASS: result reflects updated dict value")
    else:
        print("  FAIL: result does NOT reflect updated dict value (guard baked-in old value)")

    # -----------------------------------------------------------------------
    # Part 2 – writing / mutating the global dict inside the compiled fn
    # -----------------------------------------------------------------------
    print("\n--- Part 2: writing into a global dict ---")
    reset()
    torch._dynamo.reset()

    print(f"  GLOBAL_DICT before call: {GLOBAL_DICT}")
    compiled_write(x, "result", 42)
    print(f"  GLOBAL_DICT after call:  {GLOBAL_DICT}")

    if GLOBAL_DICT.get("result") == 42:
        print("  PASS: global dict mutation is visible after compiled call")
    else:
        print("  FAIL: global dict mutation is NOT visible (side-effect was dropped)")

    # -----------------------------------------------------------------------
    # Part 3 – mutation of value that is a Tensor
    # -----------------------------------------------------------------------
    print("\n--- Part 3: storing a Tensor into the global dict ---")
    reset()
    torch._dynamo.reset()

    compiled_write(x, "tensor_val", torch.tensor(99.0))
    print(f"  GLOBAL_DICT after call: {GLOBAL_DICT}")
    if "tensor_val" in GLOBAL_DICT:
        print("  PASS: tensor stored in global dict is visible")
    else:
        print("  FAIL: tensor NOT stored")

    print("\nDone.")


if __name__ == "__main__":
    run()
