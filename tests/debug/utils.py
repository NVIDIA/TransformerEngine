# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os

LOG_FILE = os.path.join("nvdlfw_inspect_logs", "nvdlfw_inspect_globalrank-0.log")


def reset_debug_log():
    if os.path.isfile(LOG_FILE):
        # delete all content
        with open(LOG_FILE, "w") as f:
            pass


def check_debug_log(msg):
    with open(LOG_FILE, "r") as f:
        for line in f.readlines():
            if msg in line:
                return True
    return False
