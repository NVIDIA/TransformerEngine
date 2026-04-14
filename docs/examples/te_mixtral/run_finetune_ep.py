"""Tiny EP fine-tune launcher for TE Mixtral.

Run from docs/examples/te_mixtral:
    torchrun --standalone --nproc_per_node=8 run_finetune_ep.py --hf-token <YOUR_HF_TOKEN>
"""

import argparse
import os

from utils import HyperParameters, run_te_mixtral_finetune


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TE Mixtral fine-tuning with Expert Parallelism.")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--ep-size", type=int, default=8)
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
    hp.expert_parallel_size = args.ep_size
    hp.batch_size = args.batch_size
    hp.num_warmup_steps = args.warmup_steps
    hp.num_training_steps = args.train_steps

    run_te_mixtral_finetune(hp)


if __name__ == "__main__":
    main()
