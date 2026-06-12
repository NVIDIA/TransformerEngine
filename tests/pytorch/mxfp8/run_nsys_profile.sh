#!/bin/bash
# Compare CuTeDSL vs C++ MXFP8 quantize by running the SAME benchmark twice —
# once with the CuTeDSL backend disabled, once enabled — and aligning results.
#
# The only difference between the two runs is the env var:
#     NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=0   -> C++ CUDA kernel   ("cpp")
#     NVTE_ENABLE_CUTEDSL_QUANT_BACKEND=1   -> CuTeDSL kernel     ("dsl")
# The Python path (tex.quantize / tex.<act>) is identical, so the comparison is
# apples-to-apples. Each run is wrapped in `nsys profile`; kernel-only GPU time
# is extracted from cuda_gpu_kern_sum and aligned by (combo, shape, dir).
#
# Usage:
#   ./run_nsys_profile.sh                           # default preset, all dirs
#   ./run_nsys_profile.sh --preset square
#   ./run_nsys_profile.sh --shapes '8192,8192' --direction row
#   ./run_nsys_profile.sh --combos plain,gelu
#   ./run_nsys_profile.sh --list-presets
#   WARMUP=20 ITERS=200 ./run_nsys_profile.sh --preset large
#   # Pass bench flags through (e.g. warm-cache CPU timing for the Python table):
#   ./run_nsys_profile.sh --shapes '8192,8192' --no-evict-l2
#
# Outputs (per run, per backend):
#   profile/nsys_kernel_time/nsys_<combo>_<shape>_<dir>_<backend>_<TS>.nsys-rep / .stdout
#   Plus two aligned summary tables (kernel-only GPU time, and bench Python time).

set -e
cd "$(dirname "$0")"
mkdir -p profile/nsys_kernel_time

# Import the repo's transformer_engine (with common/CuTeDSL) and target sm_100a.
export PYTHONPATH="$(cd ../../.. && pwd):${PYTHONPATH}"
export CUTE_DSL_ARCH="${CUTE_DSL_ARCH:-sm_100a}"

# --- arg parsing ---
PRESET="default"
SHAPES_ARG=""
DIR_ARG="all"
COMBOS_ARG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --list-presets)
            python bench_mxfp8_cutedsl.py --list-presets
            exit 0 ;;
        --preset) PRESET="$2"; shift 2 ;;
        --shapes) SHAPES_ARG="$2"; shift 2 ;;
        --direction) DIR_ARG="$2"; shift 2 ;;
        --combo|--combos) COMBOS_ARG="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
[[ -z "$COMBOS_ARG" ]] && COMBOS_ARG="plain"
IFS=',' read -r -a COMBOS <<< "$COMBOS_ARG"

WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}

# --- expand shapes ---
if [[ -n "$SHAPES_ARG" ]]; then
    SHAPES=$(echo "$SHAPES_ARG" | tr ';' '\n')
else
    SHAPES=$(python - <<PY
from bench_mxfp8_cutedsl import SHAPE_PRESETS
for m, n in SHAPE_PRESETS["$PRESET"]:
    print(f"{m},{n}")
PY
)
fi

# --- expand directions ---
case "$DIR_ARG" in
    all)  DIRS=("row" "col" "both") ;;
    row)  DIRS=("row") ;;
    col)  DIRS=("col") ;;
    both) DIRS=("both") ;;
    *) echo "invalid --direction: $DIR_ARG (want row|col|both|all)"; exit 1 ;;
esac

# Parse the dominant MXFP8 quantize kernel (avg_ns inst) from one nsys report.
# Each capture contains exactly one backend, so the dominant matching kernel by
# total time is that backend's quantize kernel. Matches both the C++ kernels
# (quantize_mxfp8_kernel / _cast_only) and the CuTeDSL kernel
# (kernel_cutlass.../MXFP8QuantizeSmemKernel). Util kernels (reduce_dbias, the
# L2-evict fill, RNG) don't match and are ignored.
parse_kernel() {  # $1 = report basename (without .nsys-rep)
    python - "$1" <<'PY'
import csv, io, subprocess, sys
rep = sys.argv[1] + ".nsys-rep"
out = subprocess.run(
    ["nsys", "stats", "--report", "cuda_gpu_kern_sum",
     "--format", "csv", "--force-export=true", rep],
    capture_output=True, text=True,
)
avg = 0.0
inst = 0
best = -1.0
for row in csv.reader(io.StringIO(out.stdout)):
    if len(row) < 9:
        continue
    try:
        total_ns = float(row[1]); inst_ = int(row[2]); avg_ns = float(row[3])
    except ValueError:
        continue
    name = row[-1]
    if (("quantize_mxfp8" in name) or ("kernel_cutlass" in name)
            or ("MXFP8QuantizeSmemKernel" in name) or ("cutedsl" in name)):
        if total_ns > best:
            best, avg, inst = total_ns, avg_ns, inst_
print(f"{avg} {inst}")
PY
}

# Run the bench once under nsys for a given backend. Echoes "avg_ns inst us".
run_backend() {  # $1=enable(0/1) $2=combo $3=M $4=N $5=dir $6=label $7=out
    local ENABLE="$1" COMBO="$2" M="$3" N="$4" DIR="$5" OUT="$7"
    if ! NVTE_ENABLE_CUTEDSL_QUANT_BACKEND="$ENABLE" nsys profile \
        --trace=cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --stats=true \
        --resolve-symbols=false \
        --force-overwrite=true \
        --output="$OUT" \
        python bench_mxfp8_cutedsl.py \
            --warmup "$WARMUP" --iters "$ITERS" \
            --shapes "${M},${N}" --direction "$DIR" \
            --combo "$COMBO" \
            "${EXTRA_ARGS[@]}" \
        > "${OUT}.stdout" 2>&1
    then
        echo "0 0 0"
        return
    fi
    local KERN; KERN=$(parse_kernel "$OUT")        # "avg_ns inst"
    # Python-level us is column 5 of the bench summary row for this shape+dir.
    local US; US=$(awk -v shape="${M}x${N}" -v dir="$DIR" \
        '$3 == shape && $4 == dir { print $5; exit }' "${OUT}.stdout")
    echo "${KERN} ${US:-0}"
}

# --- run both backends per (combo, shape, dir) ---
RESULTS_FILE=$(mktemp)
trap 'rm -f "$RESULTS_FILE"' EXIT
GENERATED_FILES=()

for COMBO in "${COMBOS[@]}"; do
for SHAPE_PAIR in $SHAPES; do
    M=${SHAPE_PAIR%,*}
    N=${SHAPE_PAIR#*,}
    for DIR in "${DIRS[@]}"; do
        TS=$(date +"%Y%m%d-%H%M%S")
        OUT_CPP="profile/nsys_kernel_time/nsys_${COMBO}_${M}x${N}_${DIR}_cpp_${TS}"
        OUT_DSL="profile/nsys_kernel_time/nsys_${COMBO}_${M}x${N}_${DIR}_dsl_${TS}"

        echo "==> ${COMBO} ${M}x${N} ${DIR}: cpp (backend disabled)"
        read -r CPP_NS CPP_INST CPP_US < <(run_backend 0 "$COMBO" "$M" "$N" "$DIR" cpp "$OUT_CPP")
        echo "==> ${COMBO} ${M}x${N} ${DIR}: dsl (backend enabled)"
        read -r DSL_NS DSL_INST DSL_US < <(run_backend 1 "$COMBO" "$M" "$N" "$DIR" dsl "$OUT_DSL")

        # Bytes moved per single-kernel launch. dact / dbias_dact combos read an
        # extra act_input (bf16), matching the bench's byte accounting.
        EXTRA_IN=0
        case "$COMBO" in
            dgelu|dsilu|drelu|dbias_dgelu|dbias_dsilu|dbias_drelu) EXTRA_IN=1 ;;
        esac
        if [[ "$DIR" == "both" ]]; then
            BYTES=$(awk -v m="$M" -v n="$N" -v ei="$EXTRA_IN" \
                'BEGIN{printf "%.0f", (2 + 2*ei)*m*n + 2*m*n + 2*(m*n/32)}')
        else
            BYTES=$(awk -v m="$M" -v n="$N" -v ei="$EXTRA_IN" \
                'BEGIN{printf "%.0f", (2 + 2*ei)*m*n + m*n + (m*n/32)}')
        fi

        # Fields: combo shape dir DSL_ns CPP_ns BYTES DSL_us CPP_us
        echo "${COMBO} ${M}x${N} ${DIR} ${DSL_NS} ${CPP_NS} ${BYTES} ${DSL_US} ${CPP_US}" \
            >> "$RESULTS_FILE"
        GENERATED_FILES+=("$OUT_CPP" "$OUT_DSL")
    done
done
done

# --- print summary ---
print_table() {  # $1 = "kernel" | "python"
    printf "%-10s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
        "combo" "shape" "dir" "DSL us" "C++ us" "DSL GB/s" "C++ GB/s" "DSL/C++"
    printf -- "----------  --------------  -----  ---------  ---------  ---------  ---------  --------\n"
    while read -r COMBO SHAPE DIR DSL_NS CPP_NS BYTES DSL_US CPP_US; do
        if [[ "$1" == "kernel" ]]; then
            D=$(awk -v v="$DSL_NS" 'BEGIN{printf "%.2f", v/1000.0}')
            C=$(awk -v v="$CPP_NS" 'BEGIN{printf "%.2f", v/1000.0}')
            DG=$(awk -v b="$BYTES" -v t="$DSL_NS" 'BEGIN{if(t>0)printf "%.1f",b/t; else print "n/a"}')
            CG=$(awk -v b="$BYTES" -v t="$CPP_NS" 'BEGIN{if(t>0)printf "%.1f",b/t; else print "n/a"}')
            SP=$(awk -v d="$DSL_NS" -v c="$CPP_NS" 'BEGIN{if(d>0)printf "%.3fx",c/d; else print "n/a"}')
        else
            D="$DSL_US"; C="$CPP_US"
            DG=$(awk -v b="$BYTES" -v t="$DSL_US" 'BEGIN{if(t>0)printf "%.1f",b/(t*1000); else print "n/a"}')
            CG=$(awk -v b="$BYTES" -v t="$CPP_US" 'BEGIN{if(t>0)printf "%.1f",b/(t*1000); else print "n/a"}')
            SP=$(awk -v d="$DSL_US" -v c="$CPP_US" 'BEGIN{if(d>0)printf "%.3fx",c/d; else print "n/a"}')
        fi
        printf "%-10s  %-14s  %-5s  %9s  %9s  %9s  %9s  %8s\n" \
            "$COMBO" "$SHAPE" "$DIR" "$D" "$C" "$DG" "$CG" "$SP"
    done < "$RESULTS_FILE"
}

echo
echo "[1/2] Kernel-only GPU time (nsys cuda_gpu_kern_sum avg, WARMUP=${WARMUP} ITERS=${ITERS})"
echo "============================================================================================="
print_table kernel

echo
echo "[2/2] Python-level time from bench (mode set by bench flags; default evict-l2 GPU time)"
echo "============================================================================================="
print_table python

echo
echo "==> Generated files (${#GENERATED_FILES[@]} run(s)):"
for OUT in "${GENERATED_FILES[@]}"; do
    echo "    ${OUT}.nsys-rep   ${OUT}.stdout"
done
