# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Launch two independent ranks, retaining logs and both natural exit codes."""

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--label", required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("command", nargs=argparse.REMAINDER)
args = parser.parse_args()
command = args.command[1:] if args.command[0] == "--" else args.command
args.output_dir.mkdir(parents=True, exist_ok=True)
store = args.output_dir / (args.label + "-" + uuid.uuid4().hex + ".rdzv")
status_path = args.output_dir / (args.label + ".status.json")
fields = "index,uuid,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw"
before = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader"], text=True
)
started = time.time()
processes, logs, rows = [], [], []
for rank in range(2):
    log_path = args.output_dir / f"{args.label}-rank{rank}.log"
    log = log_path.open("w")
    env = os.environ | {
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": "2",
        "NVTE_TEST_RDZV_PATH": str(store),
    }
    proc = subprocess.Popen(command, env=env, stdout=log, stderr=log)
    processes.append(proc)
    logs.append(log)
    rows.append({"rank": rank, "pid": proc.pid, "log": str(log_path), "started": started})
status = {
    "label": args.label,
    "command": command,
    "cwd": os.getcwd(),
    "gpu_before": before,
    "cpu_load_before": os.getloadavg(),
    "ranks": rows,
}
status_path.write_text(json.dumps(status, indent=2))
for proc, log, row in zip(processes, logs, rows):
    row["exit_code"] = proc.wait()
    row["elapsed_s"] = time.time() - started
    log.close()
    status_path.write_text(json.dumps(status, indent=2))
status["gpu_after"] = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader"], text=True
)
status_path.write_text(json.dumps(status, indent=2))
raise SystemExit(max(row["exit_code"] for row in rows))
