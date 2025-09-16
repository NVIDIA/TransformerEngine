# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
set -e  # Exit on any error

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

if $NUM_GPUS -lt 2; then
    echo "Error: NUM_GPUS must be at least 2"
    exit 1
fi

pushd "$(dirname "$0")" > /dev/null

# Function to wait for completion marker
wait_for_completion_marker() {
    local marker_file=$1
    local process_name=$2
    local max_wait=60
    local elapsed=0

    echo "Waiting for $process_name to complete..."
    while [ ! -f "$marker_file" ]; do
        if [ $elapsed -ge $max_wait ]; then
            echo "❌ Timeout waiting for $process_name to complete"
            return 1
        fi
        sleep 1
        elapsed=$((elapsed + 1))
    done
    echo "✓ $process_name completed successfully"
    return 0
}

# Clean up any old markers
rm -f /tmp/bshd_complete.marker /tmp/thd_complete.marker

echo "Running distributed training for BSHD format..."
torchrun --nproc_per_node=2 --master_port=29501 context_parallel_runner_bshd.py

# Wait for BSHD completion marker
wait_for_completion_marker "/tmp/bshd_complete.marker" "BSHD training"

if [ ! -f "/tmp/bshd_cp1_results.pt" ] || [ ! -f "/tmp/bshd_cp2_rank_0_results.pt" ] || [ ! -f "/tmp/bshd_cp2_rank_1_results.pt" ] || [ ! -f "/tmp/bshd_data.pt" ]; then
    echo "❌ Test data not found. Please run the distributed test first:"
    echo "   torchrun --nproc_per_node=2 --master_port=29501 context_parallel_runner_bshd.py"
    exit 1
fi

# Run pytest with verbose output
python -m pytest test_context_parallel_bshd.py -v -s --tb=short

# Clean up BSHD marker
rm -f /tmp/bshd_complete.marker

echo "Running distributed training for THD format..."
torchrun --nproc_per_node=2 --master_port=29502 context_parallel_runner_thd.py

# Wait for THD completion marker
wait_for_completion_marker "/tmp/thd_complete.marker" "THD training"

if [ ! -f "/tmp/thd_cp1_results.pt" ] || [ ! -f "/tmp/thd_cp2_rank_0_results.pt" ] || [ ! -f "/tmp/thd_cp2_rank_1_results.pt" ] || [ ! -f "/tmp/thd_data.pt" ]; then
    echo "❌ Test data not found. Please run the distributed test first:"
    echo "   torchrun --nproc_per_node=2 --master_port=29502 context_parallel_runner_thd.py"
    exit 1
fi

# Run pytest with verbose output
python -m pytest test_context_parallel_thd.py -v -s --tb=short

# Clean up THD marker
rm -f /tmp/thd_complete.marker

# Clean up all test data files
echo "Cleaning up test data..."
rm -f /tmp/bshd_*.pt /tmp/thd_*.pt

popd > /dev/null

echo "✅ All tests completed successfully!"
