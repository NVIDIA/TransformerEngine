# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Tiny EP fine-tune launcher for TE Mixtral.

Run from docs/examples/te_mixtral:
    python3 run_finetune_ep.py --improvement 1
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 2
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 3
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 4
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 5

Performance tiers (default --ep-size 2 -> 4 experts per rank, so the loop vs
GroupedLinear contrast in tiers 2 and 3 is non-degenerate):
    1 = HF baseline BF16 (device_map="auto", single process, Python expert loop)
    2 = TE EP BF16, naive Python loop over experts (one F.linear per expert)
    3 = TE EP BF16, GroupedLinear (batched expert GEMMs)
    4 = TE EP BF16, GroupedLinear + Fused DeepEP dispatcher
    5 = TE EP FP8 (Float8CurrentScaling), GroupedLinear + Fused DeepEP
"""

import argparse
import os

from utils import HyperParameters, run_hf_baseline_finetune, run_te_mixtral_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TE Mixtral fine-tuning with Expert Parallelism."
    )
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument(
        "--ep-size",
        type=int,
        default=2,
        help="Expert-parallel group size. Default 2 -> 4 experts/rank with 8 GPUs (DP=4).",
    )
    parser.add_argument(
        "--improvement",
        type=int,
        choices=(1, 2, 3, 4, 5),
        default=2,
        help=(
            "Tutorial tier: "
            "1=HF baseline BF16, "
            "2=TE EP BF16 (Python loop, F.linear per expert), "
            "3=TE EP BF16 (GroupedLinear), "
            "4=TE EP BF16 (GroupedLinear + Fused DeepEP), "
            "5=TE EP FP8 (Float8CurrentScaling, GroupedLinear + Fused DeepEP)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=2)
    return parser.parse_args()


IMPROVEMENT_LABELS = {
    1: "HF baseline BF16 (device_map=auto, Python expert loop)",
    2: "TE EP BF16 (Python loop + F.linear per expert)",
    3: "TE EP BF16 (GroupedLinear)",
    4: "TE EP BF16 (GroupedLinear + Fused DeepEP)",
    5: "TE EP FP8 (Float8CurrentScaling, GroupedLinear + Fused DeepEP)",
}


def main() -> None:
    args = parse_args()

    hp = HyperParameters()
    hp.model_name = "mistralai/Mixtral-8x7B-v0.1"
    hp.hf_access_token = args.hf_token
    hp.batch_size = args.batch_size
    hp.max_seq_length = args.max_seq_length
    hp.num_warmup_steps = args.warmup_steps
    hp.num_training_steps = args.train_steps

    if args.improvement == 1:
        hp.expert_parallel_size = 1
        hp.mixed_precision = "bf16"
        hp.dispatcher_type = "alltoall"
        hp.expert_ffn_mode = "grouped"  # unused on Tier 1 (HF baseline)
    elif args.improvement == 2:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "bf16"
        hp.dispatcher_type = "alltoall"
        hp.expert_ffn_mode = "loop"
    elif args.improvement == 3:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "bf16"
        hp.dispatcher_type = "alltoall"
        hp.expert_ffn_mode = "grouped"
    elif args.improvement == 4:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "bf16"
        hp.dispatcher_type = "fused"
        hp.expert_ffn_mode = "grouped"
    elif args.improvement == 5:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "fp8"
        hp.dispatcher_type = "fused"
        hp.expert_ffn_mode = "grouped"

    print(
        f"[Tier {args.improvement}] {IMPROVEMENT_LABELS[args.improvement]}\n"
        f"  mixed_precision={hp.mixed_precision}, ep_size={hp.expert_parallel_size}, "
        f"dispatcher={hp.dispatcher_type}, expert_ffn_mode={hp.expert_ffn_mode}\n"
        f"  batch_size={hp.batch_size}, max_seq_length={hp.max_seq_length}"
    )

    if args.improvement == 1:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != 1:
            raise ValueError(
                "HF baseline (improvement 1) should run as a single process so it can use "
                "device_map='auto'. Run with plain python or torchrun --nproc_per_node=1."
            )
        run_hf_baseline_finetune(hp)
        return

    run_te_mixtral_finetune(hp)


if __name__ == "__main__":
    main()
