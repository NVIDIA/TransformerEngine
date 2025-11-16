# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Test context parallel integration."""
import logging
import random
import numpy as np
import torch
import torch.distributed as dist
import os
import torch.nn.functional as F
from torch.distributed.tensor.device_mesh import init_device_mesh

from utils import get_dummy_data_thd, collect_gradients, DistributedConfig
from model import SimpleConfig, SimpleThDModel
from transformer_engine.pytorch.attention.dot_product_attention.context_parallel import (
    get_batch_on_this_cp_rank,
)


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

data = get_dummy_data_thd()

DISTRIBUTED_MODE = "RANK" in os.environ and "WORLD_SIZE" in os.environ

device = torch.device("cuda:0")
torch.cuda.set_device(0)
config = SimpleConfig(
    micro_batch_size=1,
    max_seq_length=1024,
    num_hidden_layers=1,
    vocab_size=33,
    hidden_size=320,
    intermediate_size=1280,
    num_attention_heads=20,
    layer_norm_eps=1e-5,
    hidden_dropout_prob=0.0,
    attention_probs_dropout_prob=0.1,
)

model_cp1 = SimpleThDModel(config)
model_cp1 = model_cp1.to(device)

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

batch = get_dummy_data_thd()
batch["input_ids"] = batch["input_ids_padded"]
batch["labels"] = batch["labels_padded"]
batch["position_ids"] = batch["position_ids_padded"]
batch = {
    k: v.to(device, non_blocking=True).contiguous() if isinstance(v, torch.Tensor) else v
    for k, v in batch.items()
}

with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    output_cp1 = model_cp1(batch)

logits_flat_cp1 = output_cp1.view(-1, config.vocab_size)
labels_flat = batch["labels"].view(-1)
valid_mask = labels_flat != -100
if valid_mask.any():
    loss_cp1 = F.cross_entropy(
        logits_flat_cp1[valid_mask], labels_flat[valid_mask], reduction="mean"
    )
else:
    loss_cp1 = logits_flat_cp1.sum() * 0.0

# Compute gradients for CP=1
loss_cp1.backward()

target_layers = [
    "embedding",  # Embedding layers
    "transformer_layers.0",  # First transformer layer
    "transformer_layers.1",  # Second transformer layer
    "linear",  # Language model head
]

grads_cp1 = collect_gradients(model_cp1, layer_patterns=target_layers, max_params=15)

# logger.info(f"CP=1 collected {len(grads_cp1)} parameter gradients")

initial_state_dict = {k: v.cpu().clone() for k, v in model_cp1.state_dict().items()}
torch.save(initial_state_dict, "/tmp/thd_initial_model_state.pt")
# logger.info("Model state saved for CP=2 reuse")

cp1_results = {
    "logits": output_cp1.clone().detach().cpu(),
    "loss": loss_cp1.clone().detach().cpu(),
    "grad_norms": {name: grad.norm().item() for name, grad in grads_cp1.items()},
    "grads": grads_cp1,
}

torch.save(cp1_results, "/tmp/thd_cp1_results.pt")
torch.save(data, "/tmp/thd_data.pt")

# STEP 2: RUN CP=2 (CONTEXT PARALLELISM)
# Skip CP=2 if not in distributed mode
if not DISTRIBUTED_MODE:
    logger.info("=" * 50)
    logger.info("SKIPPING CP=2 - Not running in distributed mode")
    logger.info("To test CP=2, run with: torchrun --nproc_per_node=2 run_context_parallel_thd.py")
    logger.info("=" * 50)
    import sys

    sys.exit(1)

# Run CP=2 in distributed mode
if DISTRIBUTED_MODE:
    logger.info("=" * 50)
    logger.info("RUNNING CP=2 (CONTEXT PARALLELISM)")
    logger.info("=" * 50)

    cp_size = 2
    dist.init_process_group(backend="nccl")
    dist_config = DistributedConfig()
    torch.cuda.set_device(dist_config.local_rank)

    device_mesh = init_device_mesh(
        "cuda",
        mesh_shape=(1, cp_size, 1),
        mesh_dim_names=("fsdp", "cp", "tp"),
    )
    device = torch.device(f"cuda:{dist_config.local_rank}")
    model = SimpleThDModel(config)

    try:
        initial_state_dict = torch.load("/tmp/thd_initial_model_state.pt", map_location="cpu")
        model.load_state_dict(initial_state_dict)
        # logger.info(f"Rank {dist_config.rank}: Model state loaded successfully")
    except Exception as e:
        logger.error(f"Rank {dist_config.rank}: Error loading model state: {e}")
        raise

    model = model.to(device)

    group_fsdp_cp = device_mesh[("fsdp", "cp")]._flatten("dp_cp").get_group()
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[dist_config.local_rank],
        output_device=dist_config.local_rank,
        process_group=group_fsdp_cp,
    )

    cp_group = device_mesh["cp"].get_group()

    if cp_size > 1:
        for i, transformer_layer in enumerate(model.module.transformer_layers):
            logger.debug(f"Rank {dist_config.rank}: Setting CP group for layer {i}")
            transformer_layer.set_context_parallel_group(
                cp_group,
                torch.distributed.get_process_group_ranks(device_mesh["cp"].get_group()),
                torch.cuda.Stream(),
            )

    dist.barrier()

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    batch = get_dummy_data_thd()

    input_ids_padded, labels_padded, position_ids_padded = get_batch_on_this_cp_rank(
        cu_seqlens_padded=batch["cu_seqlens_q_padded"],
        input_ids_padded=batch["input_ids_padded"],
        labels_padded=batch["labels_padded"],
        position_ids_padded=batch["position_ids_padded"],
        cp_group=cp_group,
        qvk_format="thd",
    )

    batch["input_ids"] = input_ids_padded
    batch["labels"] = labels_padded
    batch["position_ids"] = position_ids_padded
    batch = {
        k: v.to(device, non_blocking=True).contiguous() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(batch)

    logits_flat = output.view(-1, config.vocab_size)
    labels_flat = batch["labels"].view(-1)
    valid_mask = labels_flat != -100
    if valid_mask.any():
        loss = F.cross_entropy(logits_flat[valid_mask], labels_flat[valid_mask], reduction="mean")
    else:
        loss = logits_flat.sum() * 0.0

    loss.backward()

    grads = collect_gradients(model, layer_patterns=target_layers, max_params=15)

    cp2_results = {
        "logits": output.clone().detach().cpu(),
        "loss": loss.clone().detach().cpu(),
        "grad_norms": {name: grad.norm().item() for name, grad in grads.items()},
        "grads": grads,
    }

    torch.save(cp2_results, f"/tmp/thd_cp2_rank_{dist_config.rank}_results.pt")
    dist.barrier()

    # Create completion marker on rank 0 to signal all processing is done
    if dist_config.rank == 0:
        with open("/tmp/thd_complete.marker", "w") as f:
            f.write("completed")

    dist.destroy_process_group()
