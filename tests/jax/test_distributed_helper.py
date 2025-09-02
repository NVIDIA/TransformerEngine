# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import unittest

import jax
import numpy as np

from utils import pytest_parametrize_wrapper, is_devices_enough
from transformer_engine.jax.sharding import MeshResource, global_mesh_resource
from transformer_engine.jax import fp8_autocast


def generate_mesh_configs():
    configs = []
    if is_devices_enough(2):
        configs.append(
            [2, (1, 2), ("dp", "tpsp"), MeshResource(dp_resource="dp", tpsp_resource="tpsp")]
        )
    if is_devices_enough(4):
        configs.append(
            [4, (2, 2), ("fsdp", "tp"), MeshResource(tp_resource="tp", fsdp_resource="fsdp")]
        )
    return configs


class TestMeshResource(unittest.TestCase):
    def test_fp8_autocast_with_mesh_resource(self):
        for mesh_config in generate_mesh_configs():
            device_count, mesh_shape, mesh_axes, mesh_resource = mesh_config
            devices = np.asarray(jax.devices()[:device_count]).reshape(*mesh_shape)
            mesh = jax.sharding.Mesh(devices, mesh_axes)
            with mesh, fp8_autocast(enabled=False, mesh_resource=mesh_resource):
                self.assertEqual(mesh_resource, global_mesh_resource())
