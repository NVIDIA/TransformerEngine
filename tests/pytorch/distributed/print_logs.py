# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import re
from prettytable import PrettyTable


te_path = os.getenv("TE_PATH", "/opt/transformerengine")
ci_logs_dir = os.path.join(te_path, "ci_logs")

pattern = (
    "validation loss at iteration \d* on validation set | lm loss"
    " value: ([\d.]*)E\+(\d*) | lm loss PPL: ([\d.]*)E\+(\d*)"
)


def get_loss_and_ppl(filename):
    with open(filename, "r") as f:
        data = f.read()
    matches = re.findall(pattern, data)
    loss = float(matches[1][0]) * (10 ** int(matches[1][1]))
    ppl = float(matches[2][2]) * (10 ** int(matches[2][3]))
    return round(loss, 2), round(ppl, 2)


for model_config in os.listdir(ci_logs_dir):
    model_config_dir = os.path.join(ci_logs_dir, model_config)
    table = PrettyTable()
    table.title = model_config
    table.field_names = ["Config", "Loss", "Perplexity"]
    for exp in os.listdir(model_config_dir):
        filename = os.path.join(model_config_dir, exp)
        loss, ppl = get_loss_and_ppl(filename)
        table.add_row([exp[:-4], loss, ppl])
    print(table)
