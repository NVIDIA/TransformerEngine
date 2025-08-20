import jax
import jax.numpy as jnp

from transformer_engine.jax import cpp_extensions as tex
from transformer_engine.jax.quantize import (
    ScaledTensor,
    ScaledTensor1x,
    ScaledTensor2x,
    GroupedScaledTensor1x,
    ScalingMode,
    QuantizerFactory,
    QuantizeLayout,
    noop_quantizer_set,
)


def run_grouped_quantize(
    in_dtype,
    input_shape,
    q_dtype,
    scaling_mode,
    q_layout,
    flatten_axis,
    with_group_sizes,
    num_iters: int,
    num_steps: int,
):
    n_groups, m, n = input_shape
    key = jax.random.PRNGKey(0)
    subkeys = jax.random.split(key, 2)

    # *32 so that the input shapes works for MXFP8
    input_shape = (m * 32, n)

    if with_group_sizes:
        group_sizes = jnp.sort(jax.random.randint(subkeys[0], (n_groups - 1,), 0, m))
        group_sizes = jnp.concatenate([jnp.array([0]), group_sizes, jnp.array([m])])
        group_sizes = jnp.diff(group_sizes)
        print(group_sizes)
        assert group_sizes.sum() == m
        # assert jnp.any(group_sizes == 0)  # make sure that at least one group has 0 row
        group_sizes = group_sizes * 32
    else:
        group_sizes = None
        input_shape = (n_groups, input_shape[0] // n_groups, input_shape[1])

    if flatten_axis == -2:
        input_shape = input_shape[:-1] + (2,) + input_shape[-1:]

    x = jax.random.uniform(subkeys[1], input_shape, in_dtype)

    grouped_quantizer = QuantizerFactory.create(
        scaling_mode=scaling_mode,
        q_dtype=q_dtype,
        q_layout=q_layout,
        n_groups=n_groups,
    )

    # @jax.jit
    def f_jit(x, group_sizes):
        out = []
        for _ in range(num_iters):
            output = tex.grouped_quantize(
                x, group_sizes=group_sizes, flatten_axis=flatten_axis, quantizer=grouped_quantizer
            )
            # Prevent JIT from optimizing out any iterations
            out.append(output)
            print(len(out))
        return tuple(out)

    for _ in range(num_steps):
        scaled_tensor = f_jit(x, group_sizes)

    # assert_dequantized_grouped_scaled_tensor(scaled_tensor, x)


def main():
    in_dtype = jnp.bfloat16
    # Note, kMaxTensorsPerKernel=64 so group size needs to be big enough to dispatch multiple kernels to take advantage of PDL
    input_shape = (256, 128, 4096)
    q_dtype = jnp.float8_e4m3fn
    scaling_mode = ScalingMode.DELAYED_TENSOR_SCALING
    flatten_axis = -1
    with_group_sizes = False
    q_layout = QuantizeLayout.ROWWISE_COLWISE

    num_iters = 10
    num_steps = 10
    run_grouped_quantize(
        in_dtype,
        input_shape,
        q_dtype,
        scaling_mode,
        q_layout,
        flatten_axis,
        with_group_sizes,
        num_iters=num_iters,
        num_steps=num_steps,
    )


if __name__ == "__main__":
    main()
