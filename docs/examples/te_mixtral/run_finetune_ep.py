# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""EP fine-tune launcher for TE Mixtral.

    python3 run_finetune_ep.py --improvement 0
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 1
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 2
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 3

Improvements (``--ep-size 2`` => 4 experts/rank on 8 GPUs, DP=4):

    0 = HF baseline BF16 (single process, ``device_map="auto"``).
    1 = TE EP BF16, Python loop over experts.
    2 = TE EP BF16, GroupedLinear.
    3 = TE EP MXFP8 + fused MXFP8 grouped-MLP kernel.
"""

import argparse
import os
import sys

# Improvement 3 needs ``NVTE_CUTEDSL_FUSED_GROUPED_MLP=1`` set before TE is imported,
# because the fused-grouped-MLP fusion is registered at module-import time
# inside ``if ForwardGroupedMLP_CuTeGEMMSwiGLU_MXFP8.is_supported(): ...``.
for _i, _arg in enumerate(sys.argv[1:]):
    if _arg == "--improvement" and _i + 2 < len(sys.argv) and sys.argv[_i + 2] == "3":
        os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
        break
    if _arg == "--improvement=3":
        os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"
        break

from utils import HyperParameters, run_hf_baseline_finetune, run_te_mixtral_finetune


IMPROVEMENT_LABELS = {
    0: "HF baseline BF16",
    1: "TE EP BF16, Python expert loop",
    2: "TE EP BF16, GroupedLinear",
    3: "TE EP MXFP8 fused MLP",
}


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
        choices=(0, 1, 2, 3),
        default=1,
        help=(
            "Improvement: "
            "0=HF baseline BF16, "
            "1=TE EP BF16 Python loop, "
            "2=TE EP BF16 GroupedLinear, "
            "3=TE EP MXFP8 fused MLP."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hp = HyperParameters()
    hp.model_name = "mistralai/Mixtral-8x7B-v0.1"
    hp.hf_access_token = args.hf_token
    hp.batch_size = args.batch_size
    hp.max_seq_length = args.max_seq_length
    hp.num_warmup_steps = args.warmup_steps
    hp.num_training_steps = args.train_steps

    if args.improvement == 0:
        hp.expert_parallel_size = 1
        hp.mixed_precision = "bf16"
        hp.expert_ffn_mode = "loop"  # unused: HF baseline doesn't use TE MoE
    elif args.improvement == 1:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "bf16"
        hp.expert_ffn_mode = "loop"
    elif args.improvement == 2:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "bf16"
        hp.expert_ffn_mode = "grouped_op"
    elif args.improvement == 3:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "mxfp8"
        hp.expert_ffn_mode = "grouped_op"
        hp.model_impl = "te_mixtral_mxfp8"

    print(
        f"[Improvement {args.improvement}] {IMPROVEMENT_LABELS[args.improvement]}\n"
        f"  mixed_precision={hp.mixed_precision}, ep_size={hp.expert_parallel_size}, "
        f"expert_ffn_mode={hp.expert_ffn_mode}\n"
        f"  batch_size={hp.batch_size}, max_seq_length={hp.max_seq_length}"
    )

    if args.improvement == 0:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != 1:
            raise ValueError(
                "HF baseline must run as a single process (device_map='auto'). "
                "Use plain python or torchrun --nproc_per_node=1."
            )
        run_hf_baseline_finetune(hp)
        return

    run_te_mixtral_finetune(hp)


if __name__ == "__main__":
    main()
