from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
import transformer_engine.jax as te
from transformer_engine.common.recipe import (
    Recipe,
    Float8CurrentScaling,
    MXFP8BlockScaling,
    DelayedScaling,
    NVFP4BlockScaling,
)
from flax import linen as nn


def make_einsum_cls(quantization_recipe):
    def te_einsum(generate_quantizer_set, s, x, kernel, **kwargs):
        def dot_general(x, kernel, dims, *args, **kwargs):
            contracting_dims, batch_dims = dims
            assert batch_dims == ((), ()), "Batch dims not supported in TE/JAX yet"

            quantizer_set = generate_quantizer_set("quantizer_set_for_einsum")
            return te.dense.dense(
                x,
                kernel,
                contracting_dims=contracting_dims,
                quantizer_set=quantizer_set,
            )

        return jnp.einsum(s, x, kernel, _dot_general=dot_general, **kwargs)

    return te.flax.wrap_function_in_te_state_module(te_einsum, quantization_recipe, "einsum")()


class EinsumType(Enum):
    JAX = "jax"
    TE = "te"


def main():

    class SimpleModel(nn.Module):

        einsum_type: EinsumType
        quantization_recipe: Recipe = None

        def _einsum(self, *args, **kwargs):
            if self.einsum_type == EinsumType.JAX:
                return jnp.einsum(*args, **kwargs)
            elif self.einsum_type == EinsumType.TE:
                # It is important that we call make_einsum_cls(recipe) here each time einsum
                # is called. If we were to call make_einsum_cls only once and re-use it, the state for some recipes such as DelayedScaling would become incorrectly shared instead of each call having its own state.
                return make_einsum_cls(self.quantization_recipe)(*args, **kwargs)
            else:
                raise ValueError(f"Unsupported einsum type: {self.einsum_type}")

        @nn.compact
        def __call__(self, x):
            kernel = self.param(
                "kernel", jax.nn.initializers.lecun_normal(), (32, 32), jnp.bfloat16
            )
            return self._einsum("ij,jk->ik", x, kernel)

    def test_model(einsum_type: EinsumType, quantization_recipe: Recipe = None):
        model = SimpleModel(einsum_type=einsum_type, quantization_recipe=quantization_recipe)
        x = jax.random.uniform(jax.random.PRNGKey(2), (32, 32), jnp.bfloat16)
        var_collect = model.init(jax.random.PRNGKey(3), x)
        # It is important to use var_collect here to ensure all state (e.g., quantizer states) is properly handled. If you use var_collect['params'] only, TE's state management will not work correctly for recipes that require state (e.g. DelayedScaling).
        y = model.apply(var_collect, x)
        return y

    # einsum_cls = None, so standard JAX computation
    ref_out = test_model(einsum_type=EinsumType.JAX)

    # einsum using Transformer Engine's Float8CurrentScaling recipe
    te_out = test_model(einsum_type=EinsumType.TE, quantization_recipe=Float8CurrentScaling())

    # Compare outputs
    atol = float(jnp.finfo(jnp.float8_e4m3fn).eps)
    np.testing.assert_allclose(ref_out, te_out, atol=atol)


if __name__ == "__main__":
    main()
