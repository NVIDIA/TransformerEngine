# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import jax
import jax.numpy as jnp
import transformer_engine.jax

x = jax.device_put(jnp.array([1.0, 2.0, 3.0]), device=jax.devices("cuda")[0])
print(x)

print("OK")