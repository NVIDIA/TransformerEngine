# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Utils for plotting stats in the tutorial"""


import os
import matplotlib.pyplot as plt


def plot_stats(log_dir):

    # print and plot the stats
    stat_file = os.path.join(
        log_dir, "nvdlfw_inspect_statistics_logs", "nvdlfw_inspect_globalrank-0.log"
    )

    min_values = []
    custom_feature_values = []

    with open(stat_file, "r") as f:
        import re

        number_pattern = re.compile(r"[-+]?\d*\.\d+|\d+")

        for line in f:
            if "min" in line:
                matches = number_pattern.findall(line)
                if matches:
                    min_values.append(float(matches[-1]))
            if "percentage_greater_than_threshold" in line:
                matches = number_pattern.findall(line)
                if matches:
                    custom_feature_values.append(float(matches[-1]))

    # plot 2 figures side by side
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(min_values, label="min")
    axs[0].legend()
    axs[0].set_title("Min values")

    axs[1].plot(custom_feature_values, label="percentage_greater_than_threshold_0.1")
    axs[1].legend()
    axs[1].set_title("Percentage greater than threshold 0.1 values")

    plt.tight_layout()
    plt.show()
