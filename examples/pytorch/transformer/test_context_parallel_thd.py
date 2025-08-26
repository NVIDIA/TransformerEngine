import torch
import pytest
import os
import logging

# Test tolerance constants
LOGITS_ACCURACY_THRESHOLD = 85.0
LOGITS_ELEMENT_TOLERANCE = 2e-2
LOSS_RELATIVE_DIFF_THRESHOLD = 0.05
GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE = 0.05
GRADIENT_SUCCESS_RATE_THRESHOLD = 80.0
GRADIENT_EXCELLENT_THRESHOLD = 1e-4
GRADIENT_GOOD_REL_THRESHOLD = 2e-2
GRADIENT_ACCEPTABLE_REL_THRESHOLD = 5e-2

def calculate_cp_indices(cu_seqlens_padded, cp_size=2):
    """
    Calculate which token indices each CP rank should process.
    
    Args:
        cu_seqlens_padded: Cumulative sequence lengths tensor
        cp_size: Context parallel size (default 2)
    
    Returns:
        dict: {rank_id: [list of token indices]}
    """
    total_slices_per_sequence = 2 * cp_size
    num_sequences = len(cu_seqlens_padded) - 1
    
    rank_indices = {i: [] for i in range(cp_size)}
    
    for i in range(num_sequences):
        seq_start = cu_seqlens_padded[i].item()
        seq_end = cu_seqlens_padded[i+1].item()
        seq_len = seq_end - seq_start
        slice_size = seq_len // total_slices_per_sequence
        
        rank_indices[0].extend(range(seq_start + (0 * slice_size), seq_start + (1 * slice_size)))
        rank_indices[0].extend(range(seq_start + (3 * slice_size), seq_start + (4 * slice_size)))
        
        rank_indices[1].extend(range(seq_start + (1 * slice_size), seq_start + (2 * slice_size)))
        rank_indices[1].extend(range(seq_start + (2 * slice_size), seq_start + (3 * slice_size)))
    
    return rank_indices

def reconstruct_cp_logits(cp2_rank0_logits, cp2_rank1_logits, cp1_baseline_logits, cu_seqlens_padded, cp_size=2):
    """
    Reconstruct full sequence logits from distributed CP results.
    
    Args:
        cp2_rank0_logits: Logits tensor from CP rank 0 [num_tokens_rank0, vocab_size]
        cp2_rank1_logits: Logits tensor from CP rank 1 [num_tokens_rank1, vocab_size]
        cp1_baseline_logits: Full sequence logits from CP=1 run [total_tokens, vocab_size]
        cu_seqlens_padded: Cumulative sequence lengths tensor
        cp_size: Context parallel size (default 2)
    
    Returns:
        torch.Tensor: Reconstructed full sequence logits [total_tokens, vocab_size]
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Reconstructing CP={cp_size} logits from 2 ranks")
    logger.info(f"  Input shapes: rank0={cp2_rank0_logits.shape}, rank1={cp2_rank1_logits.shape}, baseline={cp1_baseline_logits.shape}")
    
    rank_indices = calculate_cp_indices(cu_seqlens_padded, cp_size)
    rank0_indices = rank_indices[0]
    rank1_indices = rank_indices[1]
    
    logger.info(f"  Expected tokens - rank0: {len(rank0_indices)}, rank1: {len(rank1_indices)}")
    logger.info(f"  Actual tokens - rank0: {cp2_rank0_logits.shape[0]}, rank1: {cp2_rank1_logits.shape[0]}")
    
    if len(rank0_indices) != cp2_rank0_logits.shape[0]:
        logger.error(f"Rank 0 size mismatch: expected {len(rank0_indices)}, got {cp2_rank0_logits.shape[0]}")
        return None
        
    if len(rank1_indices) != cp2_rank1_logits.shape[0]:
        logger.error(f"Rank 1 size mismatch: expected {len(rank1_indices)}, got {cp2_rank1_logits.shape[0]}")
        return None
    
    reconstructed_logits = torch.zeros_like(cp1_baseline_logits)
    reconstructed_logits[rank0_indices] = cp2_rank0_logits
    logger.debug(f"Rank 0: placed {len(rank0_indices)} tokens successfully")
    reconstructed_logits[rank1_indices] = cp2_rank1_logits
    logger.debug(f"Rank 1: placed {len(rank1_indices)} tokens successfully")
    
    logger.info(f"  Reconstructed logits shape: {reconstructed_logits.shape}")
    return reconstructed_logits

def compare_logits(reconstructed_logits, baseline_logits, name="Reconstructed"):
    """Compare two sets of logits and return statistics."""
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
    logger.info(f"  Elements within 1e-3: {within_1e3}/{total_elements} ({100*within_1e3/total_elements:.1f}%)")
    logger.info(f"  Elements within 1e-2: {within_1e2}/{total_elements} ({100*within_1e2/total_elements:.1f}%)")
    logger.info(f"  Elements within {LOGITS_ELEMENT_TOLERANCE}: {within_2e2}/{total_elements} ({100*within_2e2/total_elements:.1f}%)")
    logger.info(f"  Elements within 5e-2: {within_5e2}/{total_elements} ({100*within_5e2/total_elements:.1f}%)")
    
    return {
        'max_diff': logits_diff,
        'mean_diff': logits_mean_diff,
        'within_2e2_pct': 100 * within_2e2 / total_elements
    }

@pytest.fixture(scope="module")
def load_test_data():
    """Load saved results from CP=1 and CP=2 runs."""
    required_files = [
        '/tmp/thd_cp2_rank_0_results.pt',
        '/tmp/thd_cp2_rank_1_results.pt', 
        '/tmp/thd_cp1_results.pt',
        '/tmp/thd_data.pt'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            pytest.skip(f"Required test data file not found: {file_path}. Run the distributed test first.")
    
    cp2_rank0_results = torch.load('/tmp/thd_cp2_rank_0_results.pt')
    cp2_rank1_results = torch.load('/tmp/thd_cp2_rank_1_results.pt')
    cp1_results = torch.load('/tmp/thd_cp1_results.pt')
    data = torch.load('/tmp/thd_data.pt')
    
    return {
        'cp2_rank0_results': cp2_rank0_results,
        'cp2_rank1_results': cp2_rank1_results,
        'cp1_results': cp1_results,
        'data': data
    }

def test_data_loading(load_test_data):
    """Test that all required data is loaded correctly."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data
    
    assert 'logits' in test_data['cp1_results'], "CP=1 results missing logits"
    assert 'logits' in test_data['cp2_rank0_results'], "CP=2 rank 0 results missing logits"
    assert 'logits' in test_data['cp2_rank1_results'], "CP=2 rank 1 results missing logits"
    assert 'cu_seqlens_q_padded' in test_data['data'], "Data missing cu_seqlens_q_padded"
    
    cp1_logits = test_data['cp1_results']['logits']
    cp2_rank0_logits = test_data['cp2_rank0_results']['logits']
    cp2_rank1_logits = test_data['cp2_rank1_results']['logits']
    
    assert cp1_logits.dim() == 2, f"CP=1 logits should be 2D, got {cp1_logits.dim()}D"
    assert cp2_rank0_logits.dim() == 2, f"CP=2 rank 0 logits should be 2D, got {cp2_rank0_logits.dim()}D"
    assert cp2_rank1_logits.dim() == 2, f"CP=2 rank 1 logits should be 2D, got {cp2_rank1_logits.dim()}D"
    
    vocab_size = cp1_logits.shape[1]
    assert cp2_rank0_logits.shape[1] == vocab_size, f"Vocab size mismatch: CP=1 has {vocab_size}, CP=2 rank 0 has {cp2_rank0_logits.shape[1]}"
    assert cp2_rank1_logits.shape[1] == vocab_size, f"Vocab size mismatch: CP=1 has {vocab_size}, CP=2 rank 1 has {cp2_rank1_logits.shape[1]}"
    
    logger.info("Data loaded successfully:")
    logger.info(f"  CP=1 logits shape: {cp1_logits.shape}")
    logger.info(f"  CP=2 rank 0 logits shape: {cp2_rank0_logits.shape}")
    logger.info(f"  CP=2 rank 1 logits shape: {cp2_rank1_logits.shape}")

def test_cp_indices_calculation(load_test_data):
    """Test that CP indices calculation works correctly."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data
    cu_seqlens_padded = test_data['data']['cu_seqlens_q_padded']
    
    rank_indices = calculate_cp_indices(cu_seqlens_padded, cp_size=2)
    
    assert 0 in rank_indices, "Missing indices for rank 0"
    assert 1 in rank_indices, "Missing indices for rank 1"
    
    assert len(rank_indices[0]) > 0, "Rank 0 indices are empty"
    assert len(rank_indices[1]) > 0, "Rank 1 indices are empty"
    
    rank0_set = set(rank_indices[0])
    rank1_set = set(rank_indices[1])
    overlap = rank0_set.intersection(rank1_set)
    assert len(overlap) == 0, f"Rank indices overlap: {overlap}"
    
    total_tokens = cu_seqlens_padded[-1].item()
    combined_indices = set(rank_indices[0] + rank_indices[1])
    expected_coverage = len(rank_indices[0]) + len(rank_indices[1])
    
    logger.info("CP indices calculation:")
    logger.info(f"  Total tokens: {total_tokens}")
    logger.info(f"  Rank 0 tokens: {len(rank_indices[0])}")
    logger.info(f"  Rank 1 tokens: {len(rank_indices[1])}")
    logger.info(f"  Combined coverage: {expected_coverage}")

def test_logits_reconstruction(load_test_data):
    """Test that CP=2 logits can be reconstructed from both ranks."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data
    
    cp2_rank0_logits = test_data['cp2_rank0_results']['logits']
    cp2_rank1_logits = test_data['cp2_rank1_results']['logits']
    cp1_logits = test_data['cp1_results']['logits']
    cu_seqlens_padded = test_data['data']['cu_seqlens_q_padded']
    
    reconstructed_cp2_logits = reconstruct_cp_logits(
        cp2_rank0_logits,
        cp2_rank1_logits, 
        cp1_logits, 
        cu_seqlens_padded
    )
    
    assert reconstructed_cp2_logits is not None, "Logits reconstruction failed due to size mismatches"
    
    assert reconstructed_cp2_logits.shape == cp1_logits.shape, \
        f"Reconstructed shape {reconstructed_cp2_logits.shape} doesn't match baseline {cp1_logits.shape}"
    
    logger.info("Logits reconstruction successful:")
    logger.info(f"  Reconstructed shape: {reconstructed_cp2_logits.shape}")
    logger.info(f"  Baseline shape: {cp1_logits.shape}")

def test_cp2_vs_cp1_logits_accuracy(load_test_data):
    """Test that CP=2 logits match CP=1 baseline within acceptable tolerance."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data
    
    cp2_rank0_logits = test_data['cp2_rank0_results']['logits']
    cp2_rank1_logits = test_data['cp2_rank1_results']['logits']
    cp1_logits = test_data['cp1_results']['logits']
    cu_seqlens_padded = test_data['data']['cu_seqlens_q_padded']
    
    reconstructed_cp2_logits = reconstruct_cp_logits(
        cp2_rank0_logits,
        cp2_rank1_logits, 
        cp1_logits, 
        cu_seqlens_padded
    )
    
    assert reconstructed_cp2_logits is not None, "Cannot test accuracy - reconstruction failed"
    
    comparison_stats = compare_logits(reconstructed_cp2_logits, cp1_logits, "CP=2 vs CP=1")
    
    actual_accuracy = comparison_stats['within_2e2_pct']
    
    assert actual_accuracy >= LOGITS_ACCURACY_THRESHOLD, \
        f"Logits accuracy {actual_accuracy:.1f}% is below threshold {LOGITS_ACCURACY_THRESHOLD}%. " \
        f"Max diff: {comparison_stats['max_diff']:.6f}, Mean diff: {comparison_stats['mean_diff']:.6f}"
    
    logger.info(f"Logits accuracy test passed: {actual_accuracy:.1f}% within {LOGITS_ELEMENT_TOLERANCE} tolerance")

def test_cp2_vs_cp1_loss_similarity(load_test_data):
    """Test that CP=2 and CP=1 losses are similar."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data
    
    if 'loss' not in test_data['cp1_results']:
        pytest.skip("Loss data not available in CP=1 results")
    if 'loss' not in test_data['cp2_rank0_results'] or 'loss' not in test_data['cp2_rank1_results']:
        pytest.skip("Loss data not available in CP=2 results")
    
    cp1_loss = test_data['cp1_results']['loss'].item()
    cp2_rank0_loss = test_data['cp2_rank0_results']['loss'].item()
    cp2_rank1_loss = test_data['cp2_rank1_results']['loss'].item()
    
    cp2_avg_loss = (cp2_rank0_loss + cp2_rank1_loss) / 2
    
    loss_diff = abs(cp2_avg_loss - cp1_loss)
    loss_rel_diff = loss_diff / cp1_loss if cp1_loss > 1e-12 else float('inf')
    
    assert loss_rel_diff < LOSS_RELATIVE_DIFF_THRESHOLD, \
        f"Loss relative difference {100*loss_rel_diff:.2f}% exceeds {100*LOSS_RELATIVE_DIFF_THRESHOLD:.1f}% threshold. " \
        f"CP=1: {cp1_loss:.6f}, CP=2 avg: {cp2_avg_loss:.6f}"
    
    logger.info("Loss similarity test passed:")
    logger.info(f"  CP=1 loss: {cp1_loss:.6f}")
    logger.info(f"  CP=2 rank 0 loss: {cp2_rank0_loss:.6f}")
    logger.info(f"  CP=2 rank 1 loss: {cp2_rank1_loss:.6f}")
    logger.info(f"  Relative difference: {100*loss_rel_diff:.2f}%")

def test_cp2_vs_cp1_gradient_similarity(load_test_data):
    """Test that CP=2 and CP=1 gradient norms are similar."""
    logger = logging.getLogger(__name__)
    test_data = load_test_data
    
    if 'grad_norms' not in test_data['cp1_results']:
        pytest.skip("Gradient norm data not available in CP=1 results")
    if 'grad_norms' not in test_data['cp2_rank0_results']:
        pytest.skip("Gradient norm data not available in CP=2 results")
    
    cp1_grad_norms = test_data['cp1_results']['grad_norms']
    cp2_grad_norms = test_data['cp2_rank0_results']['grad_norms']
    
    def normalize_grad_name(name):
        return name.replace('module.', '') if name.startswith('module.') else name
    
    cp1_grad_norms_normalized = {normalize_grad_name(k): v for k, v in cp1_grad_norms.items()}
    cp2_grad_norms_normalized = {normalize_grad_name(k): v for k, v in cp2_grad_norms.items()}
    
    grad_comparisons = []
    max_abs_diff = 0.0
    max_abs_diff_param = ""
    
    for name in cp2_grad_norms_normalized:
        if name in cp1_grad_norms_normalized:
            cp2_norm = cp2_grad_norms_normalized[name]
            cp1_norm = cp1_grad_norms_normalized[name]
            abs_diff = abs(cp2_norm - cp1_norm)
            rel_diff = abs_diff / cp1_norm if cp1_norm > 1e-12 else float('inf')
            grad_comparisons.append((name, abs_diff, rel_diff))
            
            if abs_diff > max_abs_diff:
                max_abs_diff = abs_diff
                max_abs_diff_param = name
    
    assert len(grad_comparisons) > 0, "No matching gradient parameters found for comparison"
    
    assert max_abs_diff <= GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE, \
        f"Maximum gradient absolute difference {max_abs_diff:.6f} exceeds threshold {GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE}. " \
        f"Worst parameter: {max_abs_diff_param}"
    
    excellent_count = sum(1 for _, abs_diff, _ in grad_comparisons if abs_diff < GRADIENT_EXCELLENT_THRESHOLD)
    good_count = sum(1 for _, abs_diff, rel_diff in grad_comparisons 
                    if abs_diff >= GRADIENT_EXCELLENT_THRESHOLD and rel_diff < GRADIENT_GOOD_REL_THRESHOLD)
    acceptable_count = sum(1 for _, abs_diff, rel_diff in grad_comparisons 
                          if abs_diff >= GRADIENT_EXCELLENT_THRESHOLD and rel_diff >= GRADIENT_GOOD_REL_THRESHOLD and rel_diff < GRADIENT_ACCEPTABLE_REL_THRESHOLD)
    
    total_good_grads = excellent_count + good_count + acceptable_count
    grad_success_rate = 100 * total_good_grads / len(grad_comparisons)
    
    assert grad_success_rate >= GRADIENT_SUCCESS_RATE_THRESHOLD, \
        f"Only {grad_success_rate:.1f}% of gradients are acceptable (need ≥{GRADIENT_SUCCESS_RATE_THRESHOLD}%). " \
        f"Excellent: {excellent_count}, Good: {good_count}, Acceptable: {acceptable_count}, " \
        f"Total compared: {len(grad_comparisons)}"
    
    logger.info(f"Gradient similarity test passed: {grad_success_rate:.1f}% of gradients are acceptable")
    logger.info(f"  Max absolute difference: {max_abs_diff:.6f} (threshold: {GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE}) - {max_abs_diff_param}")
    logger.info(f"  Excellent (L2 < {GRADIENT_EXCELLENT_THRESHOLD}): {excellent_count}/{len(grad_comparisons)}")
    logger.info(f"  Good (rel < {GRADIENT_GOOD_REL_THRESHOLD}): {good_count}/{len(grad_comparisons)}")
    logger.info(f"  Acceptable (rel < {GRADIENT_ACCEPTABLE_REL_THRESHOLD}): {acceptable_count}/{len(grad_comparisons)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])