# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# ============================================================================
# [GRAPH-DEBUG] TEMPORARY DEBUG INSTRUMENTATION -- REMOVE AFTER VERIFICATION.
#
# Python-side companion to the C++ instrumentation in
# common/fused_attn/graph_debug.h. Prints the Python call stack that leads into
# each fused-attention backend query / forward / backward call, so the Python
# frames interleave (on stderr) just above the C++ "[GRAPH-DEBUG] fwd/bwd HIT|MISS"
# lines they trigger. This makes it possible to attribute each cuDNN graph-cache
# lookup to the exact Python caller (availability probe vs. module backend
# re-selection vs. actual fwd/bwd execution).
#
# Enable with the SAME switch as the C++ side:
#   export NVTE_FUSED_ATTN_GRAPH_DEBUG=1
# Optionally cap the number of printed frames (default 12):
#   export NVTE_FUSED_ATTN_GRAPH_DEBUG_PY_DEPTH=<n>
#
# To remove all of this instrumentation later:
#   1. Delete this file (graph_debug.py).
#   2. Remove every line tagged with the "[GRAPH-DEBUG]" marker in:
#        - attention/dot_product_attention/utils.py
#        - cpp_extensions/fused_attn.py
# ============================================================================

import os
import sys
import threading
import traceback

_enabled = None
_depth = None


def enabled():
    """True when NVTE_FUSED_ATTN_GRAPH_DEBUG is set (same switch as the C++ side)."""
    global _enabled
    if _enabled is None:
        val = os.getenv("NVTE_FUSED_ATTN_GRAPH_DEBUG", "")
        _enabled = val not in ("", "0")
    return _enabled


def _depth_val():
    global _depth
    if _depth is None:
        val = os.getenv("NVTE_FUSED_ATTN_GRAPH_DEBUG_PY_DEPTH", "")
        try:
            _depth = int(val) if val else 12
        except ValueError:
            _depth = 12
        _depth = max(1, min(_depth, 128))
    return _depth


def pytrace(tag):
    """Print a compact Python call stack to stderr, tagged so it groups with the C++
    [GRAPH-DEBUG] frames that follow. No-op unless NVTE_FUSED_ATTN_GRAPH_DEBUG is set."""
    if not enabled():
        return
    # Drop this frame (pytrace itself); show the most recent frames, oldest first.
    frames = traceback.extract_stack()[:-1][-_depth_val() :]
    out = sys.stderr
    out.write(f"[GRAPH-DEBUG-PY] {tag} | tid={threading.get_ident()}\n")
    for fr in frames:
        code = f"  ->  {fr.line}" if fr.line else ""
        out.write(f"[GRAPH-DEBUG-PY]   {fr.filename}:{fr.lineno} {fr.name}(){code}\n")
    out.flush()
