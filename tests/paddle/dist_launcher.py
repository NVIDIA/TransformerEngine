# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Helper functions to launch distributed tests"""

import copy
import os
from pathlib import Path
import subprocess
import time
import unittest

try:
    from paddle.base import core
except ImportError:
    from paddle.fluid import core
from paddle.distributed.utils.launch_utils import (
    TrainerProc,
    find_free_ports,
    get_cluster,
    watch_local_trainers,
)

__all__ = ["TestDistributed"]


def get_cluster_from_args(selected_gpus):
    """Get node information from selected GPUs"""
    cluster_node_ips = "127.0.0.1"
    node_ip = "127.0.0.1"

    node_ips = [x.strip() for x in cluster_node_ips.split(",")]

    node_ips.index(node_ip)

    free_ports = None

    free_ports = find_free_ports(len(selected_gpus))
    if free_ports is not None:
        free_ports = list(free_ports)

    trainer_endpoints = []
    for ip in node_ips:
        trainer_endpoints.append([f"{ip}:{port}" for port in free_ports])
    return get_cluster(node_ips, node_ip, trainer_endpoints, selected_gpus)


def get_gpus(selected_gpus):
    """Get selected GPU string"""
    selected_gpus = [x.strip() for x in selected_gpus.split(",")]
    return selected_gpus


def start_local_trainers(
    cluster,
    pod,
    training_script,
    training_script_args,
    allocator_strategy="auto_growth",
):
    """Launch trainers"""
    current_env = copy.copy(os.environ.copy())
    # paddle broadcast ncclUniqueId use socket, and
    # proxy maybe make trainers unreachable, so delete them.
    # if we set them to "", grpc will log error message "bad uri"
    # so just delete them.
    current_env.pop("http_proxy", None)
    current_env.pop("https_proxy", None)

    procs = []
    for t in pod.trainers:
        proc_env = {
            "FLAGS_selected_gpus": ",".join([str(g) for g in t.gpus]),
            "PADDLE_TRAINER_ID": f"{t.rank}",
            "PADDLE_CURRENT_ENDPOINT": f"{t.endpoint}",
            "PADDLE_TRAINERS_NUM": f"{cluster.trainers_nranks()}",
            "PADDLE_TRAINER_ENDPOINTS": ",".join(cluster.trainers_endpoints()),
            "PYTHONPATH": str(Path(__file__).resolve().parent),
        }

        proc_env["FLAGS_allocator_strategy"] = allocator_strategy
        if allocator_strategy == "auto_growth":
            proc_env["FLAGS_fraction_of_gpu_memory_to_use"] = "0.1"

        current_env.update(proc_env)

        print(f"trainer proc env:{current_env}")

        if os.getenv("WITH_COVERAGE", "OFF") == "ON":
            cmd = "python -m coverage run --branch -p " + training_script
        else:
            cmd = "python -u " + training_script

        print(f"start trainer proc:{cmd} env:{proc_env}")

        fn = None

        proc = subprocess.Popen(
            cmd.split(" ") + training_script_args, env=current_env
        )  # pylint: disable=consider-using-with

        tp = TrainerProc()
        tp.proc = proc
        tp.rank = t.rank
        tp.log_fn = fn
        tp.cmd = cmd

        procs.append(tp)

    return procs


class TestDistributed(unittest.TestCase):
    """Base class for distributed test"""

    @staticmethod
    def run_2gpu(
        target_file_name,
        allocator_strategy="auto_growth",
    ):
        """Run target file in subprocesses"""
        if not core.is_compiled_with_cuda() or core.get_cuda_device_count() == 0:
            return

        selected_gpus = get_gpus("0,1")
        cluster = None
        pod = None

        cluster, pod = get_cluster_from_args(selected_gpus)

        procs = start_local_trainers(
            cluster,
            pod,
            allocator_strategy=allocator_strategy,
            training_script=target_file_name,
            training_script_args=[],
        )

        while True:
            alive = watch_local_trainers(procs, cluster.trainers_endpoints())

            if not alive:
                print(f"Local procs complete, POD info:{pod}")
                break
            time.sleep(3)
