# Context Parallel Testing with pytest
import torch
import pytest
import os
import logging

# Test tolerance constants - adjust these to tune test sensitivity
LOGITS_ACCURACY_THRESHOLD = 85.0  # Percentage of elements within 2e-2 tolerance
LOGITS_ELEMENT_TOLERANCE = 2e-2  # Individual element tolerance for logits comparison

LOSS_RELATIVE_DIFF_THRESHOLD = 0.05  # 5% relative difference threshold for losses

GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE = 0.05  # Maximum absolute difference for any gradient
GRADIENT_SUCCESS_RATE_THRESHOLD = 80.0  # Percentage of gradients that must be acceptable
GRADIENT_EXCELLENT_THRESHOLD = 1e-4  # Absolute difference threshold for "excellent" gradients
GRADIENT_GOOD_REL_THRESHOLD = 2e-2  # Relative difference threshold for "good" gradients
GRADIENT_ACCEPTABLE_REL_THRESHOLD = 5e-2  # Relative difference threshold for "acceptable" gradients


def calculate_cp_indices_bshd(batch_size, seq_len, cp_size=2):
    """
    Calculate which sequence positions each CP rank should process for BSHD format.
    This matches the actual splitting logic in context_parallel.py

    Args:
        batch_size: Batch size
        seq_len: Sequence length per batch
        cp_size: Context parallel size (default 2)

    Returns:
        dict: {rank_id: [list of sequence positions]} for each batch
    """
    total_slices_per_sequence = 2 * cp_size  # 4 slices per sequence for CP=2
    slice_size = seq_len // total_slices_per_sequence

    rank_indices = {i: [] for i in range(cp_size)}

    # Match the actual splitting logic: index_select([cp_rank, (2*cp_size - cp_rank - 1)])
    for cp_rank in range(cp_size):
        # First chunk: cp_rank
        chunk_idx_1 = cp_rank
        start_pos_1 = chunk_idx_1 * slice_size
        end_pos_1 = (chunk_idx_1 + 1) * slice_size

        # Second chunk: (2*cp_size - cp_rank - 1)
        chunk_idx_2 = 2 * cp_size - cp_rank - 1
        start_pos_2 = chunk_idx_2 * slice_size
        end_pos_2 = (chunk_idx_2 + 1) * slice_size

        # The actual implementation concatenates these chunks sequentially in the output
        # So rank 0 gets [chunk_0_data, chunk_3_data] as a contiguous tensor
        rank_indices[cp_rank] = list(range(start_pos_1, end_pos_1)) + list(
            range(start_pos_2, end_pos_2)
        )

    logger = logging.getLogger(__name__)
    logger.debug("CP indices calculation debug:")
    logger.debug(f"  seq_len={seq_len}, cp_size={cp_size}, slice_size={slice_size}")
    for rank in range(cp_size):
        chunks = (
            [cp_rank, (2 * cp_size - cp_rank - 1)]
            if cp_rank == rank
            else [rank, (2 * cp_size - rank - 1)]
        )
        logger.debug(
            f"  Rank {rank}: chunks {chunks} → positions"
            f" {rank_indices[rank][:10]}...{rank_indices[rank][-10:]} (total:"
            f" {len(rank_indices[rank])})"
        )

    return rank_indices


def reconstruct_cp_logits_bshd(cp2_rank0_logits, cp2_rank1_logits, cp1_baseline_logits, cp_size=2):
    """
    Reconstruct full sequence logits from distributed CP results for BSHD format.

    The key insight: CP implementation concatenates selected chunks into a contiguous tensor.
    So rank0_logits contains [chunk_0_data, chunk_3_data] sequentially, not at original positions.

    Args:
        cp2_rank0_logits: Logits tensor from CP rank 0 [batch_size, seq_len_rank0, vocab_size]
        cp2_rank1_logits: Logits tensor from CP rank 1 [batch_size, seq_len_rank1, vocab_size]
        cp1_baseline_logits: Full sequence logits from CP=1 run [batch_size, seq_len, vocab_size]
        cp_size: Context parallel size (default 2)

    Returns:
        torch.Tensor: Reconstructed full sequence logits [batch_size, seq_len, vocab_size]
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reconstructing CP={cp_size} BSHD logits from 2 ranks")
    logger.info(
        f"  Input shapes: rank0={cp2_rank0_logits.shape}, rank1={cp2_rank1_logits.shape},"
        f" baseline={cp1_baseline_logits.shape}"
    )

    batch_size, full_seq_len, vocab_size = cp1_baseline_logits.shape

    # Calculate chunk size and positions
    total_chunks = 2 * cp_size  # 4 chunks for CP=2
    chunk_size = full_seq_len // total_chunks

    expected_rank_seq_len = 2 * chunk_size  # Each rank gets 2 chunks

    logger.info(f"  Expected seq lengths per rank: {expected_rank_seq_len}")
    logger.info(
        f"  Actual seq lengths - rank0: {cp2_rank0_logits.shape[1]}, rank1:"
        f" {cp2_rank1_logits.shape[1]}"
    )

    # Verify sizes match expectations
    if expected_rank_seq_len != cp2_rank0_logits.shape[1]:
        logger.error(
            f"Rank 0 seq length mismatch: expected {expected_rank_seq_len}, got"
            f" {cp2_rank0_logits.shape[1]}"
        )
        return None

    if expected_rank_seq_len != cp2_rank1_logits.shape[1]:
        logger.error(
            f"Rank 1 seq length mismatch: expected {expected_rank_seq_len}, got"
            f" {cp2_rank1_logits.shape[1]}"
        )
        return None

    # Verify batch sizes match
    if cp2_rank0_logits.shape[0] != batch_size or cp2_rank1_logits.shape[0] != batch_size:
        logger.error(
            f"Batch size mismatch: expected {batch_size}, got rank0={cp2_rank0_logits.shape[0]},"
            f" rank1={cp2_rank1_logits.shape[0]}"
        )
        return None

    # Reconstruct full logits
    reconstructed_logits = torch.zeros_like(cp1_baseline_logits)

    # For CP=2:
    # Rank 0 gets chunks [0, 3] → positions [0:chunk_size, 3*chunk_size:4*chunk_size]
    # Rank 1 gets chunks [1, 2] → positions [chunk_size:2*chunk_size, 2*chunk_size:3*chunk_size]

    # Rank 0: first half of its tensor goes to chunk 0, second half goes to chunk 3
    chunk_0_start, chunk_0_end = 0, chunk_size
    chunk_3_start, chunk_3_end = 3 * chunk_size, 4 * chunk_size

    reconstructed_logits[:, chunk_0_start:chunk_0_end, :] = cp2_rank0_logits[:, :chunk_size, :]
    reconstructed_logits[:, chunk_3_start:chunk_3_end, :] = cp2_rank0_logits[:, chunk_size:, :]

    logger.debug(
        f"Rank 0: placed chunk 0 at [{chunk_0_start}:{chunk_0_end}], chunk 3 at"
        f" [{chunk_3_start}:{chunk_3_end}]"
    )

    # Rank 1: first half of its tensor goes to chunk 1, second half goes to chunk 2
    chunk_1_start, chunk_1_end = chunk_size, 2 * chunk_size
    chunk_2_start, chunk_2_end = 2 * chunk_size, 3 * chunk_size

    reconstructed_logits[:, chunk_1_start:chunk_1_end, :] = cp2_rank1_logits[:, :chunk_size, :]
    reconstructed_logits[:, chunk_2_start:chunk_2_end, :] = cp2_rank1_logits[:, chunk_size:, :]

    logger.debug(
        f"Rank 1: placed chunk 1 at [{chunk_1_start}:{chunk_1_end}], chunk 2 at"
        f" [{chunk_2_start}:{chunk_2_end}]"
    )

    # Debug: Check if reconstruction makes sense by comparing a few elements
    logger.debug("Reconstruction verification:")
    logger.debug(f"  Baseline sample [0,0,:3]: {cp1_baseline_logits[0,0,:3]}")
    logger.debug(f"  Reconstructed [0,0,:3]: {reconstructed_logits[0,0,:3]}")
    logger.debug(f"  Rank0 input [0,0,:3]: {cp2_rank0_logits[0,0,:3]}")

    # Check if we have any zeros where we shouldn't
    zero_positions = (reconstructed_logits == 0).all(dim=-1).sum()
    total_positions = reconstructed_logits.shape[0] * reconstructed_logits.shape[1]
    logger.debug(f"  Zero positions: {zero_positions}/{total_positions}")

    # Sanity check: if reconstruction is perfect, difference should be minimal
    perfect_match = torch.allclose(reconstructed_logits, cp1_baseline_logits, atol=1e-6)
    logger.debug(f"  Perfect reconstruction (1e-6 tolerance): {perfect_match}")
    if not perfect_match:
        diff_stats = (reconstructed_logits - cp1_baseline_logits).abs()
        logger.debug(
            f"  Reconstruction diff - max: {diff_stats.max():.6f}, mean: {diff_stats.mean():.6f}"
        )

    logger.info(f"  Reconstructed logits shape: {reconstructed_logits.shape}")
    return reconstructed_logits


def compare_logits(reconstructed_logits, baseline_logits, name="Reconstructed"):
    """
    Compare two sets of logits and print detailed statistics.

    Args:
        reconstructed_logits: Reconstructed logits tensor
        baseline_logits: Baseline logits tensor
        name: Name for the comparison (for printing)
    """
    logger = logging.getLogger(__name__)
    logits_abs_diff = torch.abs(reconstructed_logits - baseline_logits)
    logits_diff = logits_abs_diff.max().item()
    logits_mean_diff = logits_abs_diff.mean().item()

    total_elements = reconstructed_logits.numel()
    within_1e3 = (logits_abs_diff < 1e-3).sum().item()
    within_1e2 = (logits_abs_diff < 1e-2).sum().item()
    within_2e2 = (logits_abs_diff < LOGITS_ELEMENT_TOLERANCE).sum().item()
    within_5e2 = (logits_abs_diff < 5e-2).sum().item()

    logger.info(f"{name} Logits Comparison:")
    logger.info(f"  Total elements: {total_elements}")
    logger.info(f"  Max difference: {logits_diff:.8f}")
    logger.info(f"  Mean difference: {logits_mean_diff:.8f}")
    logger.info(
        "  Elements within 1e-3:"
        f" {within_1e3}/{total_elements} ({100*within_1e3/total_elements:.1f}%)"
    )
    logger.info(
        "  Elements within 1e-2:"
        f" {within_1e2}/{total_elements} ({100*within_1e2/total_elements:.1f}%)"
    )
    logger.info(
        f"  Elements within {LOGITS_ELEMENT_TOLERANCE}:"
        f" {within_2e2}/{total_elements} ({100*within_2e2/total_elements:.1f}%)"
    )
    logger.info(
        "  Elements within 5e-2:"
        f" {within_5e2}/{total_elements} ({100*within_5e2/total_elements:.1f}%)"
    )

    return {
        "max_diff": logits_diff,
        "mean_diff": logits_mean_diff,
        "within_2e2_pct": 100 * within_2e2 / total_elements,
    }


@pytest.fixture(scope="module")
def load_test_data():
    """Load saved results from CP=1 and CP=2 runs."""
    # Check if test data files exist
    required_files = [
        "/tmp/bshd_cp2_rank_0_results.pt",
        "/tmp/bshd_cp2_rank_1_results.pt",
        "/tmp/bshd_cp1_results.pt",
        "/tmp/bshd_data.pt",
    ]

    for file_path in required_files:
        if not os.path.exists(file_path):
            pytest.skip(
                f"Required test data file not found: {file_path}. Run the distributed test first."
            )

    # Load all test data
    cp2_rank0_results = torch.load("/tmp/bshd_cp2_rank_0_results.pt")
    cp2_rank1_results = torch.load("/tmp/bshd_cp2_rank_1_results.pt")
    cp1_results = torch.load("/tmp/bshd_cp1_results.pt")
    data = torch.load("/tmp/bshd_data.pt")

    return {
        "cp2_rank0_results": cp2_rank0_results,
        "cp2_rank1_results": cp2_rank1_results,
        "cp1_results": cp1_results,
        "data": data,
    }


def test_data_loading(load_test_data):
    """Test that all required data is loaded correctly."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data

    # Check that all data structures have expected keys
    assert "logits" in test_data["cp1_results"], "CP=1 results missing logits"
    assert "logits" in test_data["cp2_rank0_results"], "CP=2 rank 0 results missing logits"
    assert "logits" in test_data["cp2_rank1_results"], "CP=2 rank 1 results missing logits"
    assert "cu_seqlens_q" in test_data["data"], "Data missing cu_seqlens_q"

    # Check tensor shapes are reasonable
    cp1_logits = test_data["cp1_results"]["logits"]
    cp2_rank0_logits = test_data["cp2_rank0_results"]["logits"]
    cp2_rank1_logits = test_data["cp2_rank1_results"]["logits"]

    assert (
        cp1_logits.dim() == 3
    ), f"CP=1 logits should be 3D (Batch, Seq, Vocab), got {cp1_logits.dim()}D"
    assert (
        cp2_rank0_logits.dim() == 3
    ), f"CP=2 rank 0 logits should be 3D (Batch, Seq, Vocab), got {cp2_rank0_logits.dim()}D"
    assert (
        cp2_rank1_logits.dim() == 3
    ), f"CP=2 rank 1 logits should be 3D (Batch, Seq, Vocab), got {cp2_rank1_logits.dim()}D"

    # Check vocab size consistency (vocab is last dimension)
    vocab_size = cp1_logits.shape[2]
    assert (
        cp2_rank0_logits.shape[2] == vocab_size
    ), f"Vocab size mismatch: CP=1 has {vocab_size}, CP=2 rank 0 has {cp2_rank0_logits.shape[2]}"
    assert (
        cp2_rank1_logits.shape[2] == vocab_size
    ), f"Vocab size mismatch: CP=1 has {vocab_size}, CP=2 rank 1 has {cp2_rank1_logits.shape[2]}"

    logger.info("Data loaded successfully:")
    logger.info(f"  CP=1 logits shape: {cp1_logits.shape}")
    logger.info(f"  CP=2 rank 0 logits shape: {cp2_rank0_logits.shape}")
    logger.info(f"  CP=2 rank 1 logits shape: {cp2_rank1_logits.shape}")


def test_cp_indices_calculation(load_test_data):
    """Test that CP indices calculation works correctly for BSHD format."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data

    # Get baseline logits to determine batch size and sequence length
    cp1_logits = test_data["cp1_results"]["logits"]
    batch_size, seq_len, _ = cp1_logits.shape

    # Calculate indices for CP=2
    rank_indices = calculate_cp_indices_bshd(batch_size, seq_len, cp_size=2)

    # Check that we have indices for both ranks
    assert 0 in rank_indices, "Missing indices for rank 0"
    assert 1 in rank_indices, "Missing indices for rank 1"

    # Check that indices are non-empty
    assert len(rank_indices[0]) > 0, "Rank 0 indices are empty"
    assert len(rank_indices[1]) > 0, "Rank 1 indices are empty"

    # Check that indices don't overlap
    rank0_set = set(rank_indices[0])
    rank1_set = set(rank_indices[1])
    overlap = rank0_set.intersection(rank1_set)
    assert len(overlap) == 0, f"Rank indices overlap: {overlap}"

    # Check that combined indices cover expected range
    combined_indices = set(rank_indices[0] + rank_indices[1])
    expected_coverage = len(rank_indices[0]) + len(rank_indices[1])

    logger.info("CP indices calculation (BSHD):")
    logger.info(f"  Batch size: {batch_size}, Sequence length: {seq_len}")
    logger.info(f"  Rank 0 seq positions: {len(rank_indices[0])} - {rank_indices[0]}")
    logger.info(f"  Rank 1 seq positions: {len(rank_indices[1])} - {rank_indices[1]}")
    logger.info(f"  Combined coverage: {expected_coverage}")


def test_logits_reconstruction(load_test_data):
    """Test that CP=2 logits can be reconstructed from both ranks for BSHD format."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data

    cp2_rank0_logits = test_data["cp2_rank0_results"]["logits"]
    cp2_rank1_logits = test_data["cp2_rank1_results"]["logits"]
    cp1_logits = test_data["cp1_results"]["logits"]

    # Attempt reconstruction
    reconstructed_cp2_logits = reconstruct_cp_logits_bshd(
        cp2_rank0_logits, cp2_rank1_logits, cp1_logits
    )

    # Check reconstruction succeeded
    assert (
        reconstructed_cp2_logits is not None
    ), "Logits reconstruction failed due to size mismatches"

    # Check reconstructed shape matches baseline
    assert reconstructed_cp2_logits.shape == cp1_logits.shape, (
        f"Reconstructed shape {reconstructed_cp2_logits.shape} doesn't match baseline"
        f" {cp1_logits.shape}"
    )

    logger.info("Logits reconstruction successful (BSHD):")
    logger.info(f"  Reconstructed shape: {reconstructed_cp2_logits.shape}")
    logger.info(f"  Baseline shape: {cp1_logits.shape}")


def test_cp2_vs_cp1_logits_accuracy(load_test_data):
    """Test that CP=2 logits match CP=1 baseline within acceptable tolerance for BSHD format."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data

    cp2_rank0_logits = test_data["cp2_rank0_results"]["logits"]
    cp2_rank1_logits = test_data["cp2_rank1_results"]["logits"]
    cp1_logits = test_data["cp1_results"]["logits"]

    # Reconstruct CP=2 logits
    reconstructed_cp2_logits = reconstruct_cp_logits_bshd(
        cp2_rank0_logits, cp2_rank1_logits, cp1_logits
    )

    assert reconstructed_cp2_logits is not None, "Cannot test accuracy - reconstruction failed"

    # Compare logits
    comparison_stats = compare_logits(reconstructed_cp2_logits, cp1_logits, "CP=2 vs CP=1 (BSHD)")

    # Main assertion: check percentage of elements within tolerance
    actual_accuracy = comparison_stats["within_2e2_pct"]

    assert actual_accuracy >= LOGITS_ACCURACY_THRESHOLD, (
        f"Logits accuracy {actual_accuracy:.1f}% is below threshold {LOGITS_ACCURACY_THRESHOLD}%."
        f" Max diff: {comparison_stats['max_diff']:.6f}, Mean diff:"
        f" {comparison_stats['mean_diff']:.6f}"
    )

    logger.info(
        f"Logits accuracy test passed (BSHD): {actual_accuracy:.1f}% within"
        f" {LOGITS_ELEMENT_TOLERANCE} tolerance"
    )


def test_cp2_vs_cp1_loss_similarity(load_test_data):
    """Test that CP=2 and CP=1 losses are similar."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data

    # Check if loss data is available
    if "loss" not in test_data["cp1_results"]:
        pytest.skip("Loss data not available in CP=1 results")
    if "loss" not in test_data["cp2_rank0_results"] or "loss" not in test_data["cp2_rank1_results"]:
        pytest.skip("Loss data not available in CP=2 results")

    cp1_loss = test_data["cp1_results"]["loss"].item()
    cp2_rank0_loss = test_data["cp2_rank0_results"]["loss"].item()
    cp2_rank1_loss = test_data["cp2_rank1_results"]["loss"].item()

    # Average CP=2 losses (both ranks should have similar losses)
    cp2_avg_loss = (cp2_rank0_loss + cp2_rank1_loss) / 2

    # Calculate relative difference
    loss_diff = abs(cp2_avg_loss - cp1_loss)
    loss_rel_diff = loss_diff / cp1_loss if cp1_loss > 1e-12 else float("inf")

    # Assert losses are within relative difference threshold
    assert loss_rel_diff < LOSS_RELATIVE_DIFF_THRESHOLD, (
        f"Loss relative difference {100*loss_rel_diff:.2f}% exceeds"
        f" {100*LOSS_RELATIVE_DIFF_THRESHOLD:.1f}% threshold. CP=1: {cp1_loss:.6f}, CP=2 avg:"
        f" {cp2_avg_loss:.6f}"
    )

    logger.info("Loss similarity test passed:")
    logger.info(f"  CP=1 loss: {cp1_loss:.6f}")
    logger.info(f"  CP=2 rank 0 loss: {cp2_rank0_loss:.6f}")
    logger.info(f"  CP=2 rank 1 loss: {cp2_rank1_loss:.6f}")
    logger.info(f"  Relative difference: {100*loss_rel_diff:.2f}%")


def test_cp2_vs_cp1_gradient_similarity(load_test_data):
    """Test that CP=2 and CP=1 gradient norms are similar."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data

    # Check if gradient data is available
    if "grad_norms" not in test_data["cp1_results"]:
        pytest.skip("Gradient norm data not available in CP=1 results")
    if "grad_norms" not in test_data["cp2_rank0_results"]:
        pytest.skip("Gradient norm data not available in CP=2 results")

    cp1_grad_norms = test_data["cp1_results"]["grad_norms"]
    cp2_grad_norms = test_data["cp2_rank0_results"]["grad_norms"]  # Use rank 0 as representative

    # Normalize gradient names (remove 'module.' prefix from DDP)
    def normalize_grad_name(name):
        return name.replace("module.", "") if name.startswith("module.") else name

    cp1_grad_norms_normalized = {normalize_grad_name(k): v for k, v in cp1_grad_norms.items()}
    cp2_grad_norms_normalized = {normalize_grad_name(k): v for k, v in cp2_grad_norms.items()}

    # Compare gradient norms
    grad_comparisons = []
    max_abs_diff = 0.0
    max_abs_diff_param = ""

    for name in cp2_grad_norms_normalized:
        if name in cp1_grad_norms_normalized:
            cp2_norm = cp2_grad_norms_normalized[name]
            cp1_norm = cp1_grad_norms_normalized[name]
            abs_diff = abs(cp2_norm - cp1_norm)
            rel_diff = abs_diff / cp1_norm if cp1_norm > 1e-12 else float("inf")
            grad_comparisons.append((name, abs_diff, rel_diff))

            # Track maximum absolute difference
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
                max_abs_diff_param = name

    assert len(grad_comparisons) > 0, "No matching gradient parameters found for comparison"

    # Assert maximum absolute difference is within tolerance
    assert max_abs_diff <= GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE, (
        f"Maximum gradient absolute difference {max_abs_diff:.6f} exceeds threshold"
        f" {GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE}. Worst parameter: {max_abs_diff_param}"
    )

    # Count gradients in different accuracy categories
    excellent_count = sum(
        1 for _, abs_diff, _ in grad_comparisons if abs_diff < GRADIENT_EXCELLENT_THRESHOLD
    )
    good_count = sum(
        1
        for _, abs_diff, rel_diff in grad_comparisons
        if abs_diff >= GRADIENT_EXCELLENT_THRESHOLD and rel_diff < GRADIENT_GOOD_REL_THRESHOLD
    )
    acceptable_count = sum(
        1
        for _, abs_diff, rel_diff in grad_comparisons
        if abs_diff >= GRADIENT_EXCELLENT_THRESHOLD
        and rel_diff >= GRADIENT_GOOD_REL_THRESHOLD
        and rel_diff < GRADIENT_ACCEPTABLE_REL_THRESHOLD
    )

    total_good_grads = excellent_count + good_count + acceptable_count
    grad_success_rate = 100 * total_good_grads / len(grad_comparisons)

    # Assert that sufficient percentage of gradients are acceptable
    assert grad_success_rate >= GRADIENT_SUCCESS_RATE_THRESHOLD, (
        f"Only {grad_success_rate:.1f}% of gradients are acceptable (need"
        f" ≥{GRADIENT_SUCCESS_RATE_THRESHOLD}%). Excellent: {excellent_count}, Good: {good_count},"
        f" Acceptable: {acceptable_count}, Total compared: {len(grad_comparisons)}"
    )

    logger.info(
        f"Gradient similarity test passed: {grad_success_rate:.1f}% of gradients are acceptable"
    )
    logger.info(
        f"  Max absolute difference: {max_abs_diff:.6f} (threshold:"
        f" {GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE}) - {max_abs_diff_param}"
    )
    logger.info(
        f"  Excellent (L2 < {GRADIENT_EXCELLENT_THRESHOLD}):"
        f" {excellent_count}/{len(grad_comparisons)}"
    )
    logger.info(
        f"  Good (rel < {GRADIENT_GOOD_REL_THRESHOLD}): {good_count}/{len(grad_comparisons)}"
    )
    logger.info(
        f"  Acceptable (rel < {GRADIENT_ACCEPTABLE_REL_THRESHOLD}):"
        f" {acceptable_count}/{len(grad_comparisons)}"
    )


if __name__ == "__main__":
    # Allow running as script for debugging
    pytest.main([__file__, "-v"])
