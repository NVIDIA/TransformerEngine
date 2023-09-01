# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
from prettytable import PrettyTable


te_path = os.getenv("TE_PATH", "/opt/transformerengine")
mlm_log_dir = os.path.join(te_path, "ci_logs")


convergence_pattern = (
    "validation loss at iteration \d* on validation set | lm loss"
    " value: ([\d.]*)E\+(\d*) | lm loss PPL: ([\d.]*)E\+(\d*)"
)


perf_pattern = "elapsed time per iteration \(ms\): ([\d.]*)"


def get_output_file():
    te_ci_log_dir = "/data/transformer_engine_ci_logs"
    fname = f"te_pytorch_distributed_ci_{os.getenv('CI_PIPELINE_ID', 'unknown_id')}.txt"
    return os.path.join(te_ci_log_dir, fname)


def get_run_metrics(filename):
    """Return the loss, perplexity, and step time for a given megatron-LM logfile."""

    with open(filename, "r") as f:
        data = f.read()

    # Loss and PPL
    convergence_matches = re.findall(convergence_pattern, data)
    loss = round(float(convergence_matches[1][0]) * (10 ** int(convergence_matches[1][1])), 2)
    ppl = round(float(convergence_matches[2][2]) * (10 ** int(convergence_matches[2][3])), 2)

    step_times_str = re.findall(perf_pattern, data)
    step_times = [float(x) for x in step_times_str]
    avg_step_time = round(sum(step_times) / len(step_times), 2)
    return loss, ppl, avg_step_time


def main():
    experiments = []
    for model_config in os.listdir(mlm_log_dir):
        model_config_dir = os.path.join(mlm_log_dir, model_config)
        table = PrettyTable()
        table.title = model_config
        table.field_names = ["Config", "Loss", "Perplexity", "Avg time per step (ms)"]
        for exp in os.listdir(model_config_dir):
            filename = os.path.join(model_config_dir, exp)
            loss, ppl, time_per_step = get_run_metrics(filename)
            table.add_row([exp[:-4], loss, ppl, time_per_step])
        experiments.append(table)


    with open(get_output_file(), "w") as f:
        for table in experiments:
            f.write(str(table))
            f.write("\n")
            print(table)


if __name__ == "__main__":
    main()
