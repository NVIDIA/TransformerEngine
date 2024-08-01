# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
import glob
import datetime
from prettytable import PrettyTable
from matplotlib import pyplot as plt

NUM_MOST_RECENT_RUNS = 100


te_path = os.getenv("TE_PATH", "/opt/transformerengine")
mlm_log_dir = os.path.join(te_path, "ci_logs")
te_ci_log_dir = "/data/transformer_engine_ci_logs"
te_ci_plot_dir = os.path.join(te_ci_log_dir, "plots")


convergence_pattern = (
    "validation loss at iteration \d* on validation set | lm loss"
    " value: ([\d.]*)E\+(\d*) | lm loss PPL: ([\d.]*)E\+(\d*)"
)


perf_pattern = "elapsed time per iteration \(ms\): ([\d.]*)"


def get_output_file():
    now = datetime.datetime.now()
    default_fname = f"unknown_pipeline_id_{now.month}_{now.day}_{now.year}_{now.hour}_{now.minute}"
    fname = f"{os.getenv('CI_PIPELINE_ID', default_fname)}.txt"
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


def print_run_logs():
    tables = []
    raw_logs = []
    for model_config in os.listdir(mlm_log_dir):
        model_config_dir = os.path.join(mlm_log_dir, model_config)
        table = PrettyTable()
        table.title = model_config
        table.field_names = ["Config", "Loss", "Perplexity", "Avg time per step (ms)"]
        for exp in os.listdir(model_config_dir):
            filename = os.path.join(model_config_dir, exp)
            loss, ppl, time_per_step = get_run_metrics(filename)
            exp_name = exp[:-4]
            table.add_row([exp_name, loss, ppl, time_per_step])
            raw_logs.append(f"{model_config} {exp_name} {loss} {ppl} {time_per_step}\n")
        tables.append(table)

    with open(get_output_file(), "w") as f:
        for raw_log in raw_logs:
            f.write(raw_log)
    for table in tables:
        print(table)


def save_plot(title, legend, data, filename, ylabel):
    x = list(range(1, len(data[0]) + 1))
    plt.figure()
    for label, y in zip(legend, data):
        plt.plot(x, y, "-o", label=label)
    plt.title(title)
    plt.legend()
    plt.xlabel(f"Last {NUM_MOST_RECENT_RUNS} runs")
    plt.ylabel(ylabel)
    plt.savefig(os.path.join(te_ci_plot_dir, filename))


def perf_and_loss_plots():
    files = glob.glob(os.path.join(te_ci_log_dir, "*.txt"))
    files.sort(key=os.path.getctime)
    files = files[-NUM_MOST_RECENT_RUNS:]
    data = {}
    for filename in files:
        with open(filename) as file:
            for line in file:
                line = line.strip()
                model_config, exp_name, loss, _, time_per_step = line.split(" ")
                if model_config not in data:
                    data[model_config] = {}
                if exp_name not in data[model_config]:
                    data[model_config][exp_name] = {"loss": [], "perf": []}
                data[model_config][exp_name]["loss"].append(float(loss))
                data[model_config][exp_name]["perf"].append(float(time_per_step))

    for model_config, experiments in data.items():
        lm_loss_data = []
        lm_perf_data = []
        legend = []
        for exp_name, lm_data in experiments.items():
            legend.append(exp_name)
            lm_loss_data.append(lm_data["loss"])
            lm_perf_data.append(lm_data["perf"])
        save_plot(
            model_config + " loss",
            legend,
            lm_loss_data,
            model_config + "_loss.png",
            "LM-Loss",
        )
        save_plot(
            model_config + " perf",
            legend,
            lm_perf_data,
            model_config + "_perf.png",
            "Time per step (ms)",
        )


if __name__ == "__main__":
    print_run_logs()
    perf_and_loss_plots()
