#!/bin/bash

# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

# ============================================================================
# Usage (run on the LOGIN node, not inside a container):
#   CHAIN=3 bash submit_chain.sh \
#       --export=ALL,IMAGE=/path/to/te_pertoken.sqsh,SKIP_BUILD=1,TRAIN_ITERS=60000 \
#       sbatch_moe_nvfp4_singlegpu.sh pertoken
# ============================================================================
set -euo pipefail

CHAIN="${CHAIN:-2}"
DEP_TYPE="${DEP_TYPE:-afterany}"

if [[ $# -lt 1 ]]; then
    echo "usage: CHAIN=<n> bash submit_chain.sh <sbatch args...> <script> [spec]" >&2
        echo "   e.g. CHAIN=3 bash submit_chain.sh --export=ALL,IMAGE=...,SKIP_BUILD=1 \\" >&2
        echo "            sbatch_moe_nvfp4_singlegpu.sh pertoken" >&2
    exit 1
fi

echo "[chain] submitting $CHAIN links (dependency=$DEP_TYPE) for: $*"

prev=""
for (( i=1; i<=CHAIN; i++ )); do
    if [[ -z "$prev" ]]; then
        out="$(sbatch "$@")"
    else
        out="$(sbatch "--dependency=${DEP_TYPE}:${prev}" "$@")"
    fi
    # sbatch prints "Submitted batch job <ID>"
    jid="$(awk '{print $NF}' <<<"$out")"
    if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "[chain] ERROR: could not parse job id from: $out" >&2
        exit 1
    fi
    if [[ -z "$prev" ]]; then
        echo "[chain] link $i/$CHAIN -> job $jid (no dependency, runs first)"
    else
        echo "[chain] link $i/$CHAIN -> job $jid (starts ${DEP_TYPE} job $prev)"
    fi
    prev="$jid"
done

echo "[chain] done. Inspect with: squeue -u \$USER ; scontrol show job <id> | grep -i depend"
