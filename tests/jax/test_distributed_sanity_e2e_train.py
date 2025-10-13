# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Encoder training."""
import argparse
from dataclasses import dataclass
import unittest
from functools import partial

import flax
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec
import numpy as np
import optax
from flax import linen as nn
from flax.training import train_state
import pytest

from utils import pytest_parametrize_wrapper, is_devices_enough

from transformer_engine.common import recipe
import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax
from transformer_engine.jax.quantize import get_supported_quantization_recipes
from transformer_engine.jax.sharding import MeshResource

supported_recipes = [
    pytest.param(recipe, id=(recipe.__class__.__name__ if recipe is not None else "None"))
    for recipe in [None] + get_supported_quantization_recipes()
]

# Device axis names
DEVICE_DP_AXIS = "dp"
DEVICE_TPSP_AXIS = "tpsp"

# Logical axis names
LOGICAL_TPSP_AXIS = "tensor_sequence"
NAMED_BROADCAST_AXIS = "broadcast"

# Flax RNG keys
PARAMS_KEY = "params"
DROPOUT_KEY = "dropout"
SR_KEY = "sr_rng"
INPUT_KEY = "input_rng"


@dataclass
class MeshConfig:
    num_devices: int
    mesh_shape: tuple[int]
    mesh_axis_names: tuple[str]
    mesh_resource: te.sharding.MeshResource


@dataclass
class EncoderArgs:
    quantization_recipe: recipe.Recipe = None
    mesh_config: MeshConfig = None
    batch_size: int = 128
    train_dataset_size: int = 16384
    test_batch_size: int = 128
    test_dataset_size: int = 16384
    max_seq_len: int = 64
    num_embed: int = 512
    epochs: int = 3
    lr: float = 0.0001
    seed: int = 0
    dry_run: bool = False


class Net(nn.Module):
    """NLP Encoder"""

    num_embed: int
    enable_seq_paral: bool

    @nn.compact
    def __call__(self, x, mask, disable_dropout=False):
        x = nn.Embed(num_embeddings=self.num_embed, features=256, dtype=jnp.bfloat16)(x)

        te_Encoder = partial(
            te_flax.TransformerLayer,
            hidden_size=256,
            mlp_hidden_size=1024,
            num_attention_heads=8,
            hidden_dropout=0.1,
            attention_dropout=0.1,
            dropout_rng_name=DROPOUT_KEY,
            layer_type=te_flax.TransformerLayerType.ENCODER,
            self_attn_mask_type="padding",
            enable_relative_embedding=False,
            enable_sequence_parallel=self.enable_seq_paral,
            mlp_activations=("gelu", "linear"),
        )
        x = te_Encoder()(x, attention_mask=mask, deterministic=disable_dropout)

        x = x.reshape(x.shape[0], -1)

        if self.enable_seq_paral:
            # Trigger all-gather to collect a complete tensor alone sequence on each device.
            x = jax.lax.with_sharding_constraint(
                x, jax.sharding.PartitionSpec(DEVICE_DP_AXIS, None)
            )

        x = te_flax.DenseGeneral(
            features=256,
            kernel_axes=(NAMED_BROADCAST_AXIS, LOGICAL_TPSP_AXIS),
            bias_axes=(LOGICAL_TPSP_AXIS,),
        )(x)

        x = te_flax.DenseGeneral(
            features=256,
            kernel_axes=(LOGICAL_TPSP_AXIS, NAMED_BROADCAST_AXIS),
            bias_axes=(NAMED_BROADCAST_AXIS,),
        )(x)

        x = nn.Dense(features=2)(x)
        return x


def train_step(state, inputs, masks, labels, var_collect, rngs):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, inputs, masks, disable_dropout, rngs=rngs)
        one_hot = jax.nn.one_hot(labels.astype(jnp.int32), 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = {**var_collect, PARAMS_KEY: state.params}
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(var_collect)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    var_collect, grads = flax.core.pop(grads, PARAMS_KEY)
    state = state.apply_gradients(grads=grads)

    return state, loss, accuracy, var_collect


def train_epoch(state, train_ds, batch_size, rngs, var_collect, train_fn):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["sentence"])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rngs[INPUT_KEY], train_ds_size)
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        # Split and reassign to 'rngs' to ensure unique rng for each step
        rngs = {key: jax.random.split(rngs[key])[1] for key in rngs}
        batch_inputs = train_ds["sentence"][perm, ...]
        batch_masks = train_ds["mask"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        state, loss, accuracy, var_collect = train_fn(
            state, batch_inputs, batch_masks, batch_labels, var_collect, rngs
        )
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    avg_loss = np.mean(epoch_loss)
    avg_accuracy = np.mean(epoch_accuracy)
    return state, avg_loss, avg_accuracy, var_collect


def eval_step(state, inputs, masks, labels, var_collect, rngs):
    """Computes loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, inputs, masks, disable_dropout, rngs=rngs)
        one_hot = jax.nn.one_hot(labels.astype(jnp.int32), 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = {**var_collect, PARAMS_KEY: state.params}
    loss, logits = loss_fn(var_collect, disable_dropout=True)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


def eval_model(state, test_ds, batch_size, var_collect, eval_fn, rngs):
    """Evaluation loop."""
    test_ds_size = len(test_ds["sentence"])
    num_steps = test_ds_size // batch_size
    valid_size = num_steps * batch_size
    all_loss = []
    all_accuracy = []

    for batch_start in range(0, valid_size, batch_size):
        # Split and reassign to 'rngs' to ensure unique rng for each step
        rngs = {key: jax.random.split(rngs[key])[1] for key in rngs}
        batch_end = batch_start + batch_size
        batch_inputs = test_ds["sentence"][batch_start:batch_end]
        batch_masks = test_ds["mask"][batch_start:batch_end]
        batch_labels = test_ds["label"][batch_start:batch_end]
        loss, accuracy = eval_fn(state, batch_inputs, batch_masks, batch_labels, var_collect, rngs)
        all_loss.append(loss)
        all_accuracy.append(accuracy)

    avg_loss = np.mean(all_loss)
    avg_accuracy = np.mean(all_accuracy)
    return avg_loss, avg_accuracy


def create_synthetic_data(num_samples, max_seq_len, num_embed, seed):
    """Create synthetic dataset."""
    np.random.seed(seed)
    sentences = np.random.randint(0, num_embed, size=(num_samples, max_seq_len), dtype=np.int32)
    sentences = sentences.reshape((num_samples, max_seq_len)).astype(np.int32)
    # Arbitrary function to generate labels
    labels = (sentences.astype(np.int64).sum(axis=1) > (num_embed * max_seq_len / 2)).astype(
        np.int32
    )
    # Create a causal (triangular) mask for each sample
    masks = np.ones((num_samples, 1, max_seq_len, max_seq_len), dtype=np.uint8)
    for i in range(num_samples):
        masks[i, 0] = np.triu(np.ones((max_seq_len, max_seq_len), dtype=np.uint8), k=1)
    dataset = {
        "sentence": sentences,
        "label": labels.astype(np.float32),
        "mask": masks,
    }
    return dataset


def create_synthetic_datasets(max_seq_len, num_embed, train_dataset_size, test_dataset_size):
    """Create synthetic datasets for training and testing."""
    vocab = {}
    word_id = num_embed
    train_ds = create_synthetic_data(train_dataset_size, max_seq_len, num_embed, seed=0)
    test_ds = create_synthetic_data(test_dataset_size, max_seq_len, num_embed, seed=1)
    return train_ds, test_ds, word_id


def check_fp8(state, var_collect, inputs, masks, labels):
    "Check if model includes FP8."
    rngs = {DROPOUT_KEY: jax.random.PRNGKey(0), SR_KEY: jax.random.PRNGKey(0)}
    func_jaxpr = str(jax.make_jaxpr(train_step)(state, inputs, masks, labels, var_collect, rngs))
    assert "f8_e5m2" in func_jaxpr or "f8_e4m3" in func_jaxpr


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)
    train_ds, test_ds, num_embed = create_synthetic_datasets(
        args.max_seq_len, args.num_embed, args.train_dataset_size, args.test_dataset_size
    )

    mesh_config = args.mesh_config
    devices = np.asarray(jax.devices()[: mesh_config.num_devices]).reshape(*mesh_config.mesh_shape)
    mesh = jax.sharding.Mesh(devices, mesh_config.mesh_axis_names)
    with mesh, te.fp8_autocast(
        enabled=args.quantization_recipe is not None,
        fp8_recipe=args.quantization_recipe,
        mesh_resource=mesh_config.mesh_resource,
    ):
        rng = jax.random.PRNGKey(args.seed)
        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        rng, sr_rng = jax.random.split(rng)
        init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng, SR_KEY: sr_rng}

        input_shape = [args.batch_size, args.max_seq_len]
        mask_shape = [args.batch_size, 1, args.max_seq_len, args.max_seq_len]
        label_shape = [args.batch_size]

        # Get the base axis rules and extend them with TE's rules. This must be done inside fp8_autocast
        axis_rules = flax.linen.get_logical_axis_rules()
        # TODO, make these better so it's not just a 1:1 mapping. Make each piece, like activation, have a separate logical name but still map to DP or TPSP
        axis_rules += ((LOGICAL_TPSP_AXIS, DEVICE_TPSP_AXIS),)
        axis_rules = te_flax.extend_logical_axis_rules(axis_rules)

        with flax.linen.logical_axis_rules(axis_rules):

            print(f"Device mesh: {mesh}")
            print(f"Axis rules: {axis_rules}")

            enable_sp = mesh_config.mesh_shape[mesh_config.mesh_axis_names.index("tpsp")] > 1
            encoder = Net(num_embed, enable_sp)
            inputs = jnp.zeros(input_shape, dtype=jnp.int32)
            masks = jnp.zeros(mask_shape, dtype=jnp.uint8)
            abs_var_collect = jax.eval_shape(encoder.init, init_rngs, inputs, masks)

            logical_partition_spec = nn.get_partition_spec(abs_var_collect)

            # Note that `nn.logical_to_mesh_sharding` returns a dict with an extra
            # "params" key that contains the sharding for the parameters.
            params_sharding = nn.logical_to_mesh_sharding(logical_partition_spec, mesh, axis_rules)

            inputs_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_DP_AXIS, None))
            masks_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_DP_AXIS, None, None, None))

            in_shardings = (None, inputs_sharding, masks_sharding)
            out_shardings = {
                key: params_sharding[PARAMS_KEY] if key is PARAMS_KEY else None
                for key in abs_var_collect
            }
            jit_encoder_init = jax.jit(
                encoder.init, in_shardings=in_shardings, out_shardings=out_shardings
            )
            var_collect = jit_encoder_init(init_rngs, inputs, masks)

            # Check if params are sufficiently sharded after initialization
            assert_params_sufficiently_sharded(var_collect, mesh, print_info=False)

            optimizer = optax.adamw(args.lr)
            var_collect, params = flax.core.pop(var_collect, PARAMS_KEY)
            state = train_state.TrainState.create(
                apply_fn=encoder.apply, params=params, tx=optimizer
            )

            abs_state = jax.eval_shape(
                lambda: train_state.TrainState.create(
                    apply_fn=encoder.apply, params=params, tx=optimizer
                )
            )
            logical_state_partition_spec = nn.get_partition_spec(abs_state)
            state_sharding = nn.logical_to_mesh_sharding(
                logical_state_partition_spec, mesh, axis_rules
            )

            # Check if params are sufficiently sharded after jitting the state creation
            assert_params_sufficiently_sharded(state.params, mesh, print_info=False)

            # state_sharding = get_state_sharding(state, params_sharding)
            labels_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_DP_AXIS))

            in_shardings = (
                state_sharding,
                inputs_sharding,
                masks_sharding,
                labels_sharding,
                None,
                None,
            )
            out_shardings = (state_sharding, None, None, None)
            jit_train_step = jax.jit(
                train_step, in_shardings=in_shardings, out_shardings=out_shardings
            )

            in_shardings = (
                state_sharding,
                inputs_sharding,
                masks_sharding,
                labels_sharding,
                None,
                None,
            )
            out_shardings = (None, None)
            jit_eval_step = jax.jit(
                eval_step, in_shardings=in_shardings, out_shardings=out_shardings
            )

            if args.quantization_recipe is not None:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                check_fp8(state, var_collect, inputs, masks, labels)

            if args.dry_run:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                rngs = {DROPOUT_KEY: dropout_rng, SR_KEY: sr_rng_state}
                jit_train_step(state, inputs, masks, labels, var_collect, rngs)
                print("PASSED")
                return None

            for epoch in range(1, args.epochs + 1):
                # Split and reassign to 'rng' to ensure unique rng for each step
                rng, input_rng = jax.random.split(rng)
                rng, dropout_rng = jax.random.split(rng)
                rng, sr_rng = jax.random.split(rng)
                rngs = {INPUT_KEY: input_rng, DROPOUT_KEY: dropout_rng, SR_KEY: sr_rng}

                state, train_loss, train_accuracy, var_collect = train_epoch(
                    state, train_ds, args.batch_size, rngs, var_collect, jit_train_step
                )

                test_loss, test_accuracy = eval_model(
                    state, test_ds, args.test_batch_size, var_collect, jit_eval_step, rngs
                )

                print(
                    f"Epoch: {epoch:>2} "
                    f"Train Loss: {train_loss:.6f} "
                    f"Train Accuracy: {train_accuracy:.6f} "
                    f"Test Loss: {test_loss:.6f} "
                    f"Test Accuracy: {test_accuracy:.6f} "
                )

            return [train_loss, train_accuracy, test_loss, test_accuracy]


def generate_configs():
    configs = []

    configs.append(
        pytest.param(
            MeshConfig(
                num_devices=1,
                mesh_shape=(1, 1),
                mesh_axis_names=(DEVICE_DP_AXIS, DEVICE_TPSP_AXIS),
                mesh_resource=MeshResource(
                    dp_resource=DEVICE_DP_AXIS, tpsp_resource=DEVICE_TPSP_AXIS
                ),
            ),
            id="n1_dp1_tpsp1",
        )
    )

    if is_devices_enough(4):
        configs.append(
            pytest.param(
                MeshConfig(
                    num_devices=4,
                    mesh_shape=(2, 2),
                    mesh_axis_names=(DEVICE_DP_AXIS, DEVICE_TPSP_AXIS),
                    mesh_resource=MeshResource(
                        dp_resource=DEVICE_DP_AXIS, tpsp_resource=DEVICE_TPSP_AXIS
                    ),
                ),
                id="n4_dp2_tp2",
            )
        )

    if is_devices_enough(2):
        configs.append(
            pytest.param(
                MeshConfig(
                    num_devices=2,
                    mesh_shape=(2, 1),
                    mesh_axis_names=(DEVICE_DP_AXIS, DEVICE_TPSP_AXIS),
                    mesh_resource=MeshResource(
                        dp_resource=DEVICE_DP_AXIS, tpsp_resource=DEVICE_TPSP_AXIS
                    ),
                ),
                id="n2_dp2_tpsp1",
            )
        )
        configs.append(
            pytest.param(
                MeshConfig(
                    num_devices=2,
                    mesh_shape=(1, 2),
                    mesh_axis_names=(DEVICE_DP_AXIS, DEVICE_TPSP_AXIS),
                    mesh_resource=MeshResource(
                        dp_resource=DEVICE_DP_AXIS, tpsp_resource=DEVICE_TPSP_AXIS
                    ),
                ),
                id="n2_dp1_tpsp2",
            )
        )

    return configs


def assert_params_sufficiently_sharded(params, mesh, tolerance=0.01, print_info=False):
    """Checks whether most params are sharded across sharding axis.

    (Adapted from https://github.com/AI-Hypercomputer/maxtext/blob/315e551e5942b24656a4250dcfca986fb4135b72/MaxText/maxtext_utils.py#L348)

    This function determines whether the majority of parameters are distributed
    across a specified sharding axes with an acceptable tolerance. It compares the
    current distribution to a scenario where all parameters are fully sharded
    across the axes on which the params are sharded e.g. 'tensor' axis.

    Args:
        params: params of the model state
        mesh: mesh constructed from config
        tolerance: float between 0.0 and 1.0 representing the allowed percentage of
        non-sharded parameters.
    """

    def get_product_num_devices_for_weight_sharding(weight_sharding_axes):
        product_num_devices_for_weight_sharding = 1
        for axis in weight_sharding_axes:
            product_num_devices_for_weight_sharding *= mesh.shape.get(axis, 1)
        return product_num_devices_for_weight_sharding

    def assert_leaf_sharding(path, arr):

        # Is the weight sharded? Get the axes on which it is sharded.
        partition_spec = arr.sharding.spec
        weight_sharding_axes = set(partition_spec) - set([None])  # None is not a sharding axis

        # Total number of devices on the axes on which the weight is sharded.
        product_num_devices_for_weight_sharding = get_product_num_devices_for_weight_sharding(
            weight_sharding_axes
        )

        # Params present in one shard (on one device).
        shard = arr.addressable_shards[0]
        params_per_chip = np.prod(shard.data.shape)

        # Total number of params (across all devicess).
        total_params = jax.numpy.size(arr)

        # Percentage of params that are unsharded.
        unsharded_perc = (
            (params_per_chip / (total_params / product_num_devices_for_weight_sharding) - 1) * 100
            if params_per_chip < total_params
            else 100
        )

        if print_info:
            print(
                f"{path}: {unsharded_perc:.2f}% unsharded, unsharded param shape={arr.shape},"
                f" partition spec={partition_spec}"
            )

        # If the weight is sharded on any axis, then the percentage of
        # unsharded params should be less than the tolerance.
        assert (
            product_num_devices_for_weight_sharding == 1 or unsharded_perc < tolerance
        ), f"{path}: {unsharded_perc:.2f}% unsharded"

    jax.tree_util.tree_map_with_path(
        lambda p, x: assert_leaf_sharding("/".join(str(x) for x in p), x), params
    )


class TestEncoder:
    """Encoder integration tests"""

    EXPECTED_RESULTS_BY_RECIPE = {
        # (num_gpus_dp, num_gpus_tp): (train_loss, train_accuracy, test_loss, test_accuracy)
        None: {
            (1, 1): (0.15, 0.94, 0.40, 0.84),
            (2, 1): (0.15, 0.94, 0.40, 0.84),
            (1, 2): (0.15, 0.94, 0.40, 0.84),
            (2, 2): (0.15, 0.94, 0.40, 0.84),
        },
        "DelayedScaling": {
            (1, 1): (0.15, 0.94, 0.40, 0.84),
            (2, 1): (0.15, 0.94, 0.40, 0.84),
            (1, 2): (0.15, 0.94, 0.40, 0.84),
            (2, 2): (0.15, 0.94, 0.40, 0.84),
        },
        "Float8CurrentScaling": {
            (1, 1): (0.15, 0.94, 0.40, 0.84),
            (2, 1): (0.15, 0.94, 0.40, 0.84),
            (1, 2): (0.15, 0.94, 0.40, 0.84),
            (2, 2): (0.15, 0.94, 0.40, 0.84),
        },
        "MXFP8BlockScaling": {
            (1, 1): (0.15, 0.94, 0.40, 0.84),
            (2, 1): (0.15, 0.94, 0.40, 0.84),
            (1, 2): (0.15, 0.94, 0.40, 0.84),
            (2, 2): (0.15, 0.94, 0.40, 0.84),
        },
        "NVFP4BlockScaling": {
            (1, 1): (0.17, 0.93, 0.36, 0.85),
            (2, 1): (0.17, 0.93, 0.36, 0.85),
            (1, 2): (0.17, 0.93, 0.36, 0.85),
            (2, 2): (0.17, 0.93, 0.36, 0.85),
        },
    }

    def _get_expected_results(self, quantization_recipe, mesh_shape):
        if quantization_recipe is None:
            key = None
        else:
            key = quantization_recipe.__class__.__name__

        assert (
            key in self.EXPECTED_RESULTS_BY_RECIPE
        ), f"Recipe {key} not found in expected results."
        results_dict = self.EXPECTED_RESULTS_BY_RECIPE[key]
        assert (
            mesh_shape in results_dict
        ), f"Mesh shape {mesh_shape} not found in expected results for recipe {key}."
        return results_dict[mesh_shape]

    @pytest_parametrize_wrapper("mesh_config", generate_configs())
    @pytest_parametrize_wrapper("quantization_recipe", supported_recipes)
    def test_encoder(self, quantization_recipe, mesh_config):
        # if quantization_recipe is None or quantization_recipe.__class__.__name__ != "NVFP4BlockScaling":
        #     pytest.skip("Only run NVFP4BlockScaling for CI speed.")
        #     return
        result = train_and_evaluate(
            EncoderArgs(quantization_recipe=quantization_recipe, mesh_config=mesh_config)
        )
        assert result is not None
        expected = self._get_expected_results(
            quantization_recipe, mesh_shape=mesh_config.mesh_shape
        )
        for r, e in zip(result, expected):
            np.testing.assert_allclose(r, e, rtol=1e-2, atol=1e-2)
