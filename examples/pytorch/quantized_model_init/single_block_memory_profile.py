# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TE TransformerLayer memory profiler for MXFP8 / quantized_model_init analysis.

Profiles memory behavior of one or more 8B-scale transformer blocks so
allocations are unambiguous in the PyTorch memory visualizer.  Six modes let
you progressively add complexity:

  bare               BF16 baseline, no FSDP2, no qinit
  mxfp8              MXFP8 with quantized_model_init, no FSDP2
  fp8-no-qinit       FP8 autocast WITHOUT qinit (BF16 weights), no FSDP2
  bare-fsdp2         BF16 + FSDP2 sharding
  mxfp8-fsdp2        MXFP8 + quantized_model_init + FSDP2
  fp8-no-qinit-fsdp2 FP8 autocast WITHOUT qinit + FSDP2

Non-FSDP2 modes run with plain ``python``; FSDP2 modes require ``torchrun``.

Use ``--num-layers N`` (default 1) to create multiple layers — needed to
observe cross-layer memory accumulation (TE Issue 1: transpose of unsharded
tensor accumulates and is not released after forward pass).

The ``fp8-no-qinit`` modes test TE Issue 2: quantized weights created during
forward pass are not freed when using ``te.autocast`` without
``quantized_model_init``.

Usage::

    # BF16 baseline (no distributed)
    python single_block_memory_profile.py --mode bare

    # MXFP8 with qinit (no distributed)
    python single_block_memory_profile.py --mode mxfp8

    # FP8 autocast without qinit (no distributed)
    python single_block_memory_profile.py --mode fp8-no-qinit

    # BF16 + FSDP2
    torchrun --nproc-per-node 2 single_block_memory_profile.py --mode bare-fsdp2

    # MXFP8 + qinit + FSDP2 (4 layers, to test transpose accumulation)
    torchrun --nproc-per-node 2 single_block_memory_profile.py --mode mxfp8-fsdp2 --num-layers 4

    # FP8 autocast without qinit + FSDP2
    torchrun --nproc-per-node 2 single_block_memory_profile.py --mode fp8-no-qinit-fsdp2

Snapshots are saved to ``--snapshot-dir`` and can be viewed at
https://pytorch.org/memory_viz
"""

import argparse
import os
from pathlib import Path

import torch
import transformer_engine.pytorch as te
from torch import nn
from torch.nn import functional as f
from transformer_engine.common.recipe import Float8BlockScaling, MXFP8BlockScaling
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
from transformer_engine.pytorch.quantized_tensor import QuantizedTensor


MODEL_SIZES = {
    "8b": {"hidden_size": 4096, "ffn_hidden_size": 14336, "num_attention_heads": 32},
    "70b": {"hidden_size": 8192, "ffn_hidden_size": 28672, "num_attention_heads": 64},
}

SEQ_LEN = 128
BATCH_SIZE = 2
NUM_STEPS = 3
DTYPE = torch.bfloat16


def is_rank0():  # noqa: D103
    return int(os.environ.get("RANK", "0")) == 0


def dist_print(msg):  # noqa: D103
    if is_rank0():
        print(msg)


def log_memory(tag: str):  # noqa: D103
    if not is_rank0():
        return
    alloc = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f"[Memory: {tag}] allocated={alloc:.4f} GB, reserved={reserved:.4f} GB, peak={peak:.4f} GB")


def print_param_info(model):  # noqa: D103
    if not is_rank0():
        return
    print("\n--- Parameter Info ---")
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None  # noqa: N806
    for name, param in model.named_parameters():
        if DTensor is not None and isinstance(param, DTensor):
            local = param._local_tensor
        else:
            local = param
        is_qt = isinstance(local, QuantizedTensor)
        has_hpiv = hasattr(local, "get_high_precision_init_val") and local.get_high_precision_init_val() is not None
        print(
            f"  {name}: shape={list(param.shape)}, local_shape={list(local.shape)}, "
            f"dtype={local.dtype}, quantized={is_qt}, hpiv={has_hpiv}"
        )
    print("--- End Parameter Info ---\n")


def resolve_recipe(args):
    """Return the appropriate FP8 recipe based on --recipe flag and GPU support."""
    if args.recipe == "float8block":
        return Float8BlockScaling()
    if args.recipe == "mxfp8":
        return MXFP8BlockScaling()
    # auto: try MXFP8, fall back to Float8BlockScaling
    try:
        from transformer_engine.pytorch.quantization import check_recipe_support

        recipe = MXFP8BlockScaling()
        check_recipe_support(recipe)
        dist_print("Auto-selected MXFP8BlockScaling")
        return recipe
    except (RuntimeError, ImportError):
        dist_print("MXFP8 not supported on this GPU, falling back to Float8BlockScaling")
        return Float8BlockScaling()


def create_layers(num_layers: int, use_qinit: bool, use_hpiv: bool, device: str, recipe, dims: dict):
    """Create N TransformerLayers, optionally inside quantized_model_init context."""
    hidden, ffn, heads = dims["hidden_size"], dims["ffn_hidden_size"], dims["num_attention_heads"]
    layers = []
    for _ in range(num_layers):
        if use_qinit:
            with te.quantized_model_init(recipe=recipe, enabled=True, preserve_high_precision_init_val=use_hpiv):
                layer = te.TransformerLayer(
                    hidden,
                    ffn,
                    heads,
                    fuse_qkv_params=True,
                    params_dtype=DTYPE,
                    hidden_dropout=0.0,
                    attention_dropout=0.0,
                    device=device,
                )
        else:
            layer = te.TransformerLayer(
                hidden,
                ffn,
                heads,
                fuse_qkv_params=True,
                params_dtype=DTYPE,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                device=device,
            )
        layers.append(layer)
    return nn.ModuleList(layers)


def _snapshot_subdir(mode: str, num_layers: int, no_hpiv: bool = False) -> str:
    """Build snapshot subdirectory name, appending layer count and flags."""
    name = mode
    if no_hpiv and "mxfp8" in mode:
        name += "-no-hpiv"
    if num_layers > 1:
        name += f"-{num_layers}L"
    return name


def _warmup_step(model, optimizer, x, target, use_fp8_autocast: bool, recipe):
    """Run one untimed forward+backward+step to warm up CUDA kernels."""
    optimizer.zero_grad(set_to_none=True)
    if use_fp8_autocast:
        with te.autocast(enabled=True, recipe=recipe):
            output = _forward(model, x)
    else:
        output = _forward(model, x)
    loss = f.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    dist_print(f"  Warmup step done (loss={loss.item():.6f})")


def _forward(model, x):
    """Sequential forward through ModuleList."""
    out = x
    for layer in model:
        out = layer(out)
    return out


def run_bare(args, use_fp8: bool, use_fp8_autocast_only: bool = False):
    """N blocks on one GPU, no FSDP2.

    Args:
        args: CLI arguments.
        use_fp8: If True and not use_fp8_autocast_only, use quantized_model_init.
        use_fp8_autocast_only: If True, BF16 weights + te.autocast (no qinit).
    """
    use_qinit = use_fp8 and not use_fp8_autocast_only
    use_autocast = use_fp8 or use_fp8_autocast_only
    recipe = resolve_recipe(args) if use_autocast else None
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_layers = args.num_layers

    log_memory("before_model_init")

    dims = args.dims
    model = create_layers(
        num_layers=num_layers,
        use_qinit=use_qinit,
        use_hpiv=not args.no_hpiv and use_qinit,
        device="cuda",
        recipe=recipe,
        dims=dims,
    )
    log_memory("after_model_init")
    print_param_info(model)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(), lr=1e-3, master_weights=True, master_weight_dtype=torch.float32
    )
    log_memory("after_optimizer_create")

    if use_qinit and not args.no_hpiv:
        count = 0
        for name, param in model.named_parameters():
            optimizer.initialize_state(param, store_param_remainders=False)
            if hasattr(param, "get_high_precision_init_val"):
                hp_val = param.get_high_precision_init_val()
                if hp_val is not None:
                    optimizer.set_scaled_state(param, "master_param", hp_val.to(device=device, dtype=torch.float32))
                    param.clear_high_precision_init_val()
                    count += 1
        dist_print(f"Seeded {count} master weights from HPIV.")
    log_memory("after_master_weight_seed")

    hidden = dims["hidden_size"]
    x = torch.randn(SEQ_LEN, BATCH_SIZE, hidden, dtype=DTYPE, device=device)
    target = torch.randn(SEQ_LEN, BATCH_SIZE, hidden, dtype=DTYPE, device=device)

    # Warmup: one untimed step to compile CUDA kernels before recording
    _warmup_step(model, optimizer, x, target, use_fp8_autocast=use_autocast, recipe=recipe)
    log_memory("after_warmup")

    # Reset peak stats and start recording AFTER warmup
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=500000)

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with te.autocast(enabled=True, recipe=recipe):
                output = _forward(model, x)
        else:
            output = _forward(model, x)
        loss = f.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        log_memory(f"after_step_{step}")
        dist_print(f"  Step {step}: loss = {loss.item():.6f}")

    if use_fp8_autocast_only:
        mode_name = "fp8-no-qinit"
    elif use_fp8:
        mode_name = "mxfp8"
    else:
        mode_name = "bare"
    _dump_snapshot(args, _snapshot_subdir(mode_name, num_layers, no_hpiv=args.no_hpiv))


def run_fsdp2(args, use_fp8: bool, use_fp8_autocast_only: bool = False):
    """N blocks with FSDP2 sharding.

    Args:
        args: CLI arguments.
        use_fp8: If True and not use_fp8_autocast_only, use quantized_model_init.
        use_fp8_autocast_only: If True, BF16 weights + te.autocast (no qinit).
    """
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor

    use_qinit = use_fp8 and not use_fp8_autocast_only
    use_autocast = use_fp8 or use_fp8_autocast_only
    recipe = resolve_recipe(args) if use_autocast else None
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    num_layers = args.num_layers

    log_memory("before_model_init")

    dims = args.dims
    model = create_layers(
        num_layers=num_layers,
        use_qinit=use_qinit,
        use_hpiv=not args.no_hpiv and use_qinit,
        device="meta",
        recipe=recipe,
        dims=dims,
    )
    log_memory("after_model_init_meta")

    mesh = DeviceMesh("cuda", list(range(world_size)))
    # Per-layer FSDP2 sharding (standard pattern for transformer stacks)
    for layer in model:
        fully_shard(layer, mesh=mesh)
    log_memory("after_fsdp_shard")

    for module in model.modules():
        if isinstance(module, TransformerEngineBaseModule):
            module.reset_parameters()
    log_memory("after_materialize")
    print_param_info(model)

    optimizer = te.optimizers.FusedAdam(
        model.parameters(), lr=1e-3, master_weights=True, master_weight_dtype=torch.float32
    )
    log_memory("after_optimizer_create")

    if use_qinit and not args.no_hpiv:
        count = 0
        for name, param in model.named_parameters():
            optimizer.initialize_state(param, store_param_remainders=False)
            local = param._local_tensor if isinstance(param, DTensor) else param
            if hasattr(local, "get_high_precision_init_val"):
                hp_val = local.get_high_precision_init_val()
                if hp_val is not None:
                    optimizer.set_scaled_state(param, "master_param", hp_val.to(device=device, dtype=torch.float32))
                    local.clear_high_precision_init_val()
                    count += 1
        dist_print(f"Seeded {count} master weights from HPIV.")
    log_memory("after_master_weight_seed")

    hidden = dims["hidden_size"]
    x = torch.randn(SEQ_LEN, BATCH_SIZE, hidden, dtype=DTYPE, device=device)
    target = torch.randn(SEQ_LEN, BATCH_SIZE, hidden, dtype=DTYPE, device=device)

    # Warmup: one untimed step to compile CUDA kernels before recording
    _warmup_step(model, optimizer, x, target, use_fp8_autocast=use_autocast, recipe=recipe)
    log_memory("after_warmup")

    # Reset peak stats and start recording AFTER warmup
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=500000)

    for step in range(NUM_STEPS):
        optimizer.zero_grad(set_to_none=True)
        if use_autocast:
            with te.autocast(enabled=True, recipe=recipe):
                output = _forward(model, x)
        else:
            output = _forward(model, x)
        loss = f.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        log_memory(f"after_step_{step}")
        dist_print(f"  Step {step}: loss = {loss.item():.6f}")

    if use_fp8_autocast_only:
        mode_name = "fp8-no-qinit-fsdp2"
    elif use_fp8:
        mode_name = "mxfp8-fsdp2"
    else:
        mode_name = "bare-fsdp2"
    _dump_snapshot(args, _snapshot_subdir(mode_name, num_layers, no_hpiv=args.no_hpiv))
    dist.destroy_process_group()


def _dump_snapshot(args, subdir: str):
    if is_rank0():
        snap_dir = Path(args.snapshot_dir) / subdir
        snap_dir.mkdir(parents=True, exist_ok=True)
        snap_path = snap_dir / "memory_snapshot.pickle"
        torch.cuda.memory._dump_snapshot(str(snap_path))
        print(f"\nSnapshot saved to {snap_path}")
    torch.cuda.memory._record_memory_history(enabled=None)


def main():  # noqa: D103
    parser = argparse.ArgumentParser(description="TE TransformerLayer memory profiler")
    parser.add_argument(
        "--mode",
        choices=["bare", "mxfp8", "fp8-no-qinit", "bare-fsdp2", "mxfp8-fsdp2", "fp8-no-qinit-fsdp2"],
        required=True,
        help=(
            "bare=BF16, mxfp8=MXFP8+qinit, fp8-no-qinit=BF16 weights+FP8 autocast, *-fsdp2=same with FSDP2 sharding"
        ),
    )
    parser.add_argument("--num-layers", type=int, default=1, help="Number of TransformerLayers (default: 1)")
    parser.add_argument(
        "--model-size",
        choices=list(MODEL_SIZES.keys()),
        default="8b",
        help="Layer dimensions: 8b (~490M params/layer) or 70b (~973M params/layer) (default: 8b)",
    )
    parser.add_argument("--no-hpiv", action="store_true", help="Disable preserve_high_precision_init_val")
    parser.add_argument(
        "--recipe",
        choices=["mxfp8", "float8block", "auto"],
        default="auto",
        help="FP8 recipe. 'auto' uses MXFP8 if supported, else Float8BlockScaling (default: auto)",
    )
    parser.add_argument("--snapshot-dir", type=str, default="/tmp/single_block_snapshots")
    args = parser.parse_args()
    args.dims = MODEL_SIZES[args.model_size]

    dims = args.dims
    dist_print(f"\n{'=' * 60}")
    dist_print(f"Memory Profiler — mode={args.mode}, layers={args.num_layers}, size={args.model_size}")
    dist_print(f"  hidden={dims['hidden_size']}, ffn={dims['ffn_hidden_size']}, heads={dims['num_attention_heads']}")
    dist_print(f"  seq_len={SEQ_LEN}, batch={BATCH_SIZE}, steps={NUM_STEPS}")
    dist_print(f"  recipe={args.recipe}, hpiv={'disabled' if args.no_hpiv else 'enabled'}")
    dist_print(f"{'=' * 60}\n")

    if args.mode == "bare":
        run_bare(args, use_fp8=False)
    elif args.mode == "mxfp8":
        run_bare(args, use_fp8=True)
    elif args.mode == "fp8-no-qinit":
        run_bare(args, use_fp8=False, use_fp8_autocast_only=True)
    elif args.mode == "bare-fsdp2":
        run_fsdp2(args, use_fp8=False)
    elif args.mode == "mxfp8-fsdp2":
        run_fsdp2(args, use_fp8=True)
    elif args.mode == "fp8-no-qinit-fsdp2":
        run_fsdp2(args, use_fp8=False, use_fp8_autocast_only=True)


if __name__ == "__main__":
    main()
