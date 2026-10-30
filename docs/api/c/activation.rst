..
    Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

    See LICENSE for license information.

activation.h
============

Grouped MXFP8 activation workspace
----------------------------------

Grouped GeLU, SiLU, ReLU, Quick GeLU, and Squared ReLU have ``_v2`` entry points
for both forward and backward. Use the query/allocate/execute protocol described
in :doc:`cast`; descriptors, activation-input descriptors, and mapper metadata
occupy private stream slots in the caller-owned workspace. These calls share
the catalog, alignment, host-submission, and CUDA Graph lifetime rules in
:doc:`workspace`.

The original workspace-less functions reject MXFP8 layouts with varying last
dimensions. Constant last dimensions use launch-local tensor maps and continue
to work through both APIs. Grouped DBias plus activation backward already accepts
workspace, requires constant last dimensions, and resolves its scratch through
the same stream-aware catalog.

.. doxygenfile:: activation.h
