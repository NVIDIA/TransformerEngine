"""Tiny EP fine-tune launcher for TE Mixtral.

Run from docs/examples/te_mixtral:
    python3 run_finetune_ep.py --improvement 1
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 2
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 3
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 4

Performance tiers:
    1 = HF baseline BF16 (device_map="auto", single process)
    2 = TE naive EP BF16 (AllToAllTokenDispatcher, NCCL all-to-all)
    3 = TE fused EP BF16 (FusedTokenRouter via DeepEP)
    4 = TE fused EP FP8  (FusedTokenRouter via DeepEP + FP8 quantization)
"""

import argparse
import os

from utils import HyperParameters, run_hf_baseline_finetune, run_te_mixtral_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TE Mixtral fine-tuning with Expert Parallelism.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument(
        "--improvement",
        type=int,
        choices=(1, 2, 3, 4),
        default=2,
        help=(
            "Tutorial mode: "
            "1=HF baseline BF16, "
            "2=TE naive EP BF16 (AllToAll), "
            "3=TE fused EP BF16 (DeepEP), "
            "4=TE fused EP FP8 (DeepEP+FP8)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=2)
    return parser.parse_args()


IMPROVEMENT_LABELS = {
    1: "HF baseline BF16 (device_map=auto)",
    2: "TE naive EP BF16 (AllToAllTokenDispatcher)",
    3: "TE fused EP BF16 (FusedTokenRouter / DeepEP)",
    4: "TE fused EP FP8 (FusedTokenRouter / DeepEP + FP8)",
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
    elif args.improvement == 2:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "bf16"
        hp.dispatcher_type = "alltoall"
    elif args.improvement == 3:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "bf16"
        hp.dispatcher_type = "fused"
    elif args.improvement == 4:
        hp.expert_parallel_size = args.ep_size
        hp.mixed_precision = "fp8"
        hp.dispatcher_type = "fused"

    print(
        f"[Tier {args.improvement}] {IMPROVEMENT_LABELS[args.improvement]}\n"
        f"  mixed_precision={hp.mixed_precision}, ep_size={hp.expert_parallel_size}, "
        f"dispatcher={hp.dispatcher_type}\n"
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
