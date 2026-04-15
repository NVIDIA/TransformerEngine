"""Tiny EP fine-tune launcher for TE Mixtral.

Run from docs/examples/te_mixtral:
    python3 run_finetune_ep.py --improvement 1
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 2
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --improvement 3
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
        choices=(1, 2, 3),
        default=2,
        help="Tutorial mode: 1=HF baseline BF16, 2=TE BF16, 3=TE FP8.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    hp = HyperParameters()
    # Smallest Mixtral checkpoint.
    hp.model_name = "mistralai/Mixtral-8x7B-v0.1"
    hp.hf_access_token = args.hf_token
    if args.improvement == 1:
        # Baseline is HF model-parallel path, not EP DTensor path.
        hp.expert_parallel_size = 1
    else:
        hp.expert_parallel_size = args.ep_size
    hp.mixed_precision = "bf16" if args.improvement in (1, 2) else "fp8"
    hp.batch_size = args.batch_size
    hp.num_warmup_steps = args.warmup_steps
    hp.num_training_steps = args.train_steps

    print(
        f"Running Mixtral EP improvement {args.improvement} "
        f"with mixed_precision={hp.mixed_precision}, ep_size={hp.expert_parallel_size}"
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
