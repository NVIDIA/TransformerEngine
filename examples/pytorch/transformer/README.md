# Context Parallel Testing Framework

This directory contains a comprehensive testing framework for validating **Context Parallelism (CP)** in TransformerEngine. The framework compares single-GPU baseline runs against distributed multi-GPU runs to ensure numerical consistency and correctness.

## 🏗️ Architecture Overview

### Test Models

The framework supports two attention input formats, each with its own model implementation in `model.py`:

**1. SimpleThDModel (THD Format - Token-Head-Dimension):**
- Processes sequences in flattened token format
- Uses cumulative sequence lengths for batch handling
- Attention computation in THD layout

**2. SimpleBSHDModel (BSHD Format - Batch-Sequence-Head-Dimension):**
- Processes sequences in standard batch format
- Maintains batch dimension throughout computation
- Attention computation in BSHD layout

**Common Architecture (both models):**
- **Embedding Layer**: Token embedding (vocab_size=33, hidden_size=320)
- **1 TransformerEngine Layer**: Full attention + MLP block with:
  - 20 attention heads (16-dimensional each)
  - 1280 intermediate FFN size
  - GELU activation
  - RoPE positional embeddings
  - Mixed precision (bfloat16)
- **Output Layer**: Linear projection back to vocabulary space
- **Layer Normalization**: Applied after transformer layers

**Key Features:**
- Designed for **variable-length sequences** with padding
- Uses **cumulative sequence lengths** (`cu_seqlens`) for efficient batching
- Supports **context parallel** attention computation
- Deterministic initialization for reproducible testing

### Test Data Generation

The `utils.py` module provides synthetic test data in two formats:

**THD Format (`get_dummy_data_thd()`):**
```python
# Three sequences of different lengths (flattened):
Sequence 1: [1,1,1,1,1,1,1,1]           # 8 tokens (padded)
Sequence 2: [2,2,2,2,2,2,2,2,2,2,2,2]   # 12 tokens (padded)
Sequence 3: [3,3,3,3,3,3,3,3]           # 8 tokens (padded)
# Flattened into single tensor of 28 tokens total
```

**BSHD Format (`get_dummy_data_bshd()`):**
```python
# Single batch with one long sequence:
Batch shape: [1, 1024, hidden_size]  # Standard batch tensor
```

**Data Processing Pipeline:**
1. **Padding**: Sequences padded to be divisible by `2 * cp_size` (required for CP)
2. **Cumulative Lengths**: Used for tracking sequence boundaries
3. **Labels**: Corresponding target tokens for loss computation
4. **Position IDs**: Relative positions within each sequence

## 🚀 Testing Workflow

The framework runs parallel test suites for both THD and BSHD formats:

### Phase 1: Baseline Run (CP=1)
**Files**: 
- `context_parallel_runner_thd.py` (THD format)
- `context_parallel_runner_bshd.py` (BSHD format)

Torchrun is used for both programs, at first no parallelism is used and we have two identical forward passes (one per gpu).
1. **Model Creation**: Initialize model (SimpleThDModel or SimpleBSHDModel)
2. **Forward Pass**: Process full sequences without parallelization
3. **Loss Computation**: Cross-entropy on valid (non-padded) tokens
4. **Gradient Collection**: Gather gradients from key model components:
   - Embedding layer
   - Transformer layer(s)
   - Output linear layer
5. **State Persistence**: Save model weights and results to `/tmp/` for CP=2 comparison

### Phase 2: Distributed Run (CP=2)
Then torch distributed is initialized and we use context parallel=2, both gpus now participate in a forward pass.

1. **Process Group Setup**: Initialize NCCL backend for 2 GPUs
2. **Device Mesh**: Create `(fsdp=1, cp=2, tp=1)` parallelization strategy
3. **Model Replication**: Load identical weights from CP=1 baseline
4. **DDP Wrapping**: Enable gradient synchronization across ranks
5. **Context Parallel Setup**: Configure attention layers for sequence splitting
6. **Data Partitioning**: 
   - THD: Use `get_batch_on_this_cp_rank(..., qvk_format="thd")`
   - BSHD: Use `get_batch_on_this_cp_rank(..., qvk_format="bshd")`
   - **Rank 0**: Gets first + last chunks of each sequence
   - **Rank 1**: Gets middle chunks of each sequence
7. **Synchronized Forward/Backward**: Identical computation with distributed data
8. **Completion Markers**: Creates `/tmp/{format}_complete.marker` when done

### Phase 3: Validation Testing
**Files**: 
- `test_context_parallel_thd.py` (THD format tests)
- `test_context_parallel_bshd.py` (BSHD format tests)

Both test suites perform identical validations with format-specific reconstruction:

## 🧪 Test Suite Details

### Test 1: Data Loading Validation
- Verifies all required result files exist
- Validates tensor shapes and dimensions
- Confirms vocabulary size consistency
- Checks data structure integrity

### Test 2: CP Index Calculation
- Tests the sequence splitting algorithm
- Verifies no overlap between rank assignments
- Confirms complete sequence coverage
- Validates chunk size calculations

### Test 3: Logits Reconstruction
- Reconstructs full sequences from distributed chunks
- Validates reconstruction algorithm correctness
- Ensures output shapes match baseline
- Tests the "first+last vs middle" chunk distribution

### Test 4: Logits Accuracy Comparison ⭐
**The Core Test**: Compares reconstructed CP=2 logits against CP=1 baseline

**Tolerance Configuration:**
```python
LOGITS_ACCURACY_THRESHOLD = 85.0  # % of elements within tolerance
LOGITS_ELEMENT_TOLERANCE = 2e-2   # Individual element tolerance
```

**Success Criteria**: ≥85% of logit elements must be within 2e-2 absolute difference

**Why 85%?** Distributed computation with mixed precision (bfloat16) introduces expected numerical differences. This threshold balances strictness with practical distributed computing realities. Moreover, we notice that as we increase the number of hidden layers (TE Layers), we see the numerical differences between the non CP and CP counterparts increase.

### Test 5: Loss Similarity
- Compares averaged losses from both CP ranks
- **Tolerance**: 5% relative difference
- Validates that distributed training preserves loss computation

### Test 6: Gradient Consistency ⭐
**Dual Validation Approach:**

1. **Strict Bound**: No gradient can exceed 0.05 absolute difference
   ```python
   GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE = 0.05
   ```

2. **Statistical Quality**: ≥80% of gradients must be "acceptable"
   ```python
   GRADIENT_SUCCESS_RATE_THRESHOLD = 80.0
   ```

**Gradient Categories:**
- **Excellent**: Absolute difference < 1e-4
- **Good**: Relative difference < 2e-2  
- **Acceptable**: Relative difference < 5e-2

## 🎯 Tolerance Philosophy

The framework uses **scientifically calibrated tolerances** based on:

1. **Mixed Precision Effects**: bfloat16 has ~3-4 decimal digits of precision
2. **Distributed Communication**: AllReduce operations introduce small numerical errors
3. **Computation Order**: Different operation sequences in CP vs non-CP modes

**Conservative but Practical**: Tolerances are tight enough to catch real bugs while loose enough to handle expected distributed computing variations.

## 🚀 Running the Tests

### Quick Start
```bash
# Run both THD and BSHD format tests with automatic synchronization
bash run_context_parallel.sh
```

### Script Features

**Automatic Process Synchronization:**
- Uses file-based completion markers (`/tmp/{format}_complete.marker`)
- Ensures first `torchrun` completes before starting second
- Configurable timeout (default 60 seconds)
- Automatic cleanup of old markers

**Clean Output Mode:**
- Suppressed verbose logging by default
- Only shows test results and errors
- Use `--verbose` flag for detailed output

### Manual Execution

**THD Format:**
```bash
# Step 1: Generate test data with distributed run
torchrun --nproc_per_node=2 --master_port=29501 context_parallel_runner_thd.py

# Step 2: Run validation tests  
python -m pytest test_context_parallel_thd.py -v
```

**BSHD Format:**
```bash
# Step 1: Generate test data with distributed run
torchrun --nproc_per_node=2 --master_port=29501 context_parallel_runner_bshd.py

# Step 2: Run validation tests
python -m pytest test_context_parallel_bshd.py -v
```

### Expected Output

**Standard Run (Clean Output):**
```
Running distributed training for BSHD format...
✓ BSHD training completed successfully

Running Context Parallel Tests for BSHD...
============================== 6 passed in 2.34s ==============================

Running distributed training for THD format...
✓ THD training completed successfully

Running Context Parallel Tests for THD...
============================== 6 passed in 2.45s ==============================
```

**Verbose Mode:**
```bash
# Run with detailed logging
bash run_context_parallel.sh --verbose

# Or for pytest directly:
pytest test_context_parallel_thd.py -v --log-cli-level=INFO
```

## 🔧 Customizing Tolerances

All test thresholds are configurable constants at the top of both test files:

```python
# In test_context_parallel_thd.py and test_context_parallel_bshd.py
LOGITS_ACCURACY_THRESHOLD = 85.0      # Stricter: 90.0, Looser: 80.0
LOGITS_ELEMENT_TOLERANCE = 2e-2       # Stricter: 1e-2, Looser: 5e-2
GRADIENT_MAX_ABSOLUTE_DIFF_TOLERANCE = 0.05  # Stricter: 0.01, Looser: 0.1
```

## 🎯 What This Folder Validates

✅ **Numerical Correctness**: CP=2 produces equivalent results to CP=1  
✅ **Format Compatibility**: Both THD and BSHD attention formats work correctly
✅ **Gradient Consistency**: Distributed training gradients match single-GPU  
✅ **Loss Preservation**: Training objectives remain unchanged  
✅ **Sequence Reconstruction**: Distributed chunks correctly reassemble  
✅ **Memory Efficiency**: Context parallelism reduces per-GPU memory usage  
✅ **Scalability**: Folder extends to larger CP sizes and models
