import jax
import jax.numpy as jnp
from transformer_engine.jax import cpp_extensions as tex
from utils import assert_allclose
import pdb

out_dtype = jnp.float32

shape_list = [[128, 256], [256, 256], [512, 128]]
A_list = []
B_list = []
ref_C_list = []

key = jax.random.PRNGKey(0)
subkeys = jax.random.split(key, len(shape_list * 2))
for i in range(len(shape_list)):
    shape = shape_list[i]
    A_i = jax.random.normal(subkeys[2 * i], shape, dtype=out_dtype)
    B_i = jax.random.normal(subkeys[2 * i + 1], shape, dtype=out_dtype)
    ref_C_i = A_i + B_i
    A_list.append(A_i)
    B_list.append(B_i)
    ref_C_list.append(ref_C_i)

# pdb.set_trace()
C_list = tex.grouped_add(A_list, B_list, out_dtype)
for i in range(len(shape_list)):
    assert_allclose(C_list[i], ref_C_list[i])
print("Grouped add test passed.")
