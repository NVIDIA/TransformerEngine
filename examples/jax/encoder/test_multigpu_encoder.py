# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Encoder training on multi-GPU with data parallelism"""
import argparse
import unittest
from functools import partial

import flax
import jax
import jax.numpy as jnp
import nltk
import numpy as np
import optax
from datasets import load_dataset
from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec, NamedSharding

from common import is_bf16_supported, get_fp8_recipe_from_name_string
import transformer_engine.jax as te
import transformer_engine.jax.cpp_extensions as tex
import transformer_engine.jax.flax as te_flax
from transformer_engine.jax.quantize import is_fp8_available, ScalingMode


DEVICE_DP_AXIS = "data"
PARAMS_KEY = "params"
PARAMS_AXES_KEY = PARAMS_KEY + "_axes"
DROPOUT_KEY = "dropout"
INPUT_KEY = "input_rng"


class Net(nn.Module):
    """NLP Encoder"""

    num_embed: int

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
        )
        x = te_Encoder()(x, attention_mask=mask, deterministic=disable_dropout)

        x = x.reshape(x.shape[0], -1)

        x = te_flax.DenseGeneral(features=256)(x)

        x = te_flax.DenseGeneral(features=256)(x)

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


def eval_step(state, inputs, masks, labels, var_collect):
    """Computes loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, inputs, masks, disable_dropout)
        one_hot = jax.nn.one_hot(labels.astype(jnp.int32), 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = {**var_collect, PARAMS_KEY: state.params}
    loss, logits = loss_fn(var_collect, disable_dropout=True)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


def eval_model(state, test_ds, batch_size, var_collect, eval_fn):
    """Evaluation loop."""
    test_ds_size = len(test_ds["sentence"])
    num_steps = test_ds_size // batch_size
    valid_size = num_steps * batch_size
    all_loss = []
    all_accuracy = []

    for batch_start in range(0, valid_size, batch_size):
        batch_end = batch_start + batch_size
        batch_inputs = test_ds["sentence"][batch_start:batch_end]
        batch_masks = test_ds["mask"][batch_start:batch_end]
        batch_labels = test_ds["label"][batch_start:batch_end]
        loss, accuracy = eval_fn(state, batch_inputs, batch_masks, batch_labels, var_collect)
        all_loss.append(loss)
        all_accuracy.append(accuracy)

    avg_loss = np.mean(all_loss)
    avg_accuracy = np.mean(all_accuracy)
    return avg_loss, avg_accuracy


def data_preprocess(dataset, vocab, word_id, max_seq_len):
    """Convert tokens to numbers."""
    nltk.download("punkt_tab")
    dataset_size = len(dataset["sentence"])
    output = np.zeros((dataset_size, max_seq_len), dtype=np.int32)
    mask_3d = np.ones((dataset_size, max_seq_len, max_seq_len), dtype=np.uint8)

    for j, sentence in enumerate(dataset["sentence"]):
        tokens = nltk.word_tokenize(sentence)
        tensor = output[j]

        for i, word in enumerate(tokens):
            if i >= max_seq_len:
                break

            if word not in vocab:
                vocab[word] = word_id
                tensor[i] = word_id
                word_id = word_id + 1
            else:
                tensor[i] = vocab[word]

        seq_len = min(len(tokens), max_seq_len)
        mask_2d = mask_3d[j]
        mask_2d[:seq_len, :seq_len] = 0

    new_dataset = {
        "sentence": output,
        "label": dataset["label"].astype(np.float32),
        "mask": mask_3d.reshape((dataset_size, 1, max_seq_len, max_seq_len)),
    }
    return new_dataset, vocab, word_id


def get_datasets(max_seq_len):
    """Load GLUE train and test datasets into memory."""
    vocab = {}
    word_id = 0

    train_ds = load_dataset("glue", "cola", split="train")
    train_ds.set_format(type="np")
    train_ds, vocab, word_id = data_preprocess(train_ds, vocab, word_id, max_seq_len)

    test_ds = load_dataset("glue", "cola", split="validation")
    test_ds.set_format(type="np")
    test_ds, vocab, word_id = data_preprocess(test_ds, vocab, word_id, max_seq_len)
    return train_ds, test_ds, word_id


def check_fp8(state, var_collect, inputs, masks, labels):
    "Check if model includes FP8."
    rngs = {DROPOUT_KEY: jax.random.PRNGKey(0)}
    func_jaxpr = str(jax.make_jaxpr(train_step)(state, inputs, masks, labels, var_collect, rngs))
    assert "f8_e5m2" in func_jaxpr or "f8_e4m3" in func_jaxpr


def get_params_sharding(sharding_rules, abs_var_collect, mesh):
    """Refer params to create params sharding"""
    rules_dict = dict(sharding_rules)

    def to_device_axis(logical_axis):
        partitions = [rules_dict[key] for key in logical_axis]
        return NamedSharding(mesh, PartitionSpec(*partitions))

    params_axes = abs_var_collect.get(PARAMS_AXES_KEY, {})
    params_axes_sharding = jax.tree_util.tree_map(
        to_device_axis, nn_partitioning.get_axis_names(params_axes)
    )
    params_axes_sharding = flax.core.unfreeze(params_axes_sharding)
    params_sharding = jax.tree_util.tree_map(
        lambda x: NamedSharding(mesh, PartitionSpec(None)), abs_var_collect[PARAMS_KEY]
    )
    params_sharding = {**params_sharding, **params_axes_sharding}
    return params_sharding


def get_state_sharding(state, params_sharding):
    """Refer params_sharding to create state sharding"""

    def replace_params(x):
        return params_sharding if isinstance(x, dict) else None

    state_sharding = jax.tree_util.tree_map(
        replace_params, state, is_leaf=lambda x: isinstance(x, dict)
    )
    return state_sharding


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)
    jax.config.update("jax_use_shardy_partitioner", args.enable_shardy)
    train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)

    num_gpu = jax.local_device_count()
    assert args.batch_size % num_gpu == 0, f"Batch size needs to be multiple of {num_gpu}"
    assert args.test_batch_size % num_gpu == 0, f"Test batch size needs to be multiple of {num_gpu}"
    if args.fp8_recipe == "MXFP8BlockScaling":
        assert (
            args.batch_size / num_gpu % 32 == 0
        ), "Batch size needs to be multiple of 32 for MXFP8"
        assert (
            args.test_batch_size / num_gpu % 32 == 0
        ), "Test batch size needs to be multiple of 32 for MXFP8"

    if args.use_fp8:
        fp8_recipe = get_fp8_recipe_from_name_string(args.fp8_recipe)
    else:
        fp8_recipe = None

    device_mesh = mesh_utils.create_device_mesh((num_gpu,))
    with jax.sharding.Mesh(
        devices=device_mesh, axis_names=(DEVICE_DP_AXIS,)
    ) as mesh, te.fp8_autocast(
        enabled=args.use_fp8,
        fp8_recipe=fp8_recipe,
        mesh_resource=te.MeshResource(dp_resource=DEVICE_DP_AXIS),
    ):

        rng = jax.random.PRNGKey(args.seed)
        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}

        input_shape = [args.batch_size, args.max_seq_len]
        mask_shape = [args.batch_size, 1, args.max_seq_len, args.max_seq_len]
        label_shape = [args.batch_size]

        # Add TE logical axis rules to our Flax logical axis rule context. This must be done inside fp8_autocast
        sharding_rules = te_flax.extend_logical_axis_rules(tuple())
        with flax.linen.logical_axis_rules(sharding_rules):
            encoder = Net(num_embed)
            inputs = jnp.zeros(input_shape, dtype=jnp.int32)
            masks = jnp.zeros(mask_shape, dtype=jnp.uint8)
            abs_var_collect = jax.eval_shape(encoder.init, init_rngs, inputs, masks)

            params_sharding = get_params_sharding(sharding_rules, abs_var_collect, mesh)
            inputs_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_DP_AXIS, None))
            masks_sharding = NamedSharding(mesh, PartitionSpec(DEVICE_DP_AXIS, None, None, None))

            in_shardings = (None, inputs_sharding, masks_sharding)
            out_shardings = {
                key: params_sharding if key is PARAMS_KEY else None for key in abs_var_collect
            }
            jit_encoder_init = jax.jit(
                encoder.init, in_shardings=in_shardings, out_shardings=out_shardings
            )
            var_collect = jit_encoder_init(init_rngs, inputs, masks)

            optimizer = optax.adamw(args.lr)
            var_collect, params = flax.core.pop(var_collect, PARAMS_KEY)
            state = train_state.TrainState.create(
                apply_fn=encoder.apply, params=params, tx=optimizer
            )
            state_sharding = get_state_sharding(state, params_sharding)
            labels_sharding = NamedSharding(
                mesh,
                PartitionSpec(
                    DEVICE_DP_AXIS,
                ),
            )
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

            in_shardings = (state_sharding, inputs_sharding, masks_sharding, labels_sharding, None)
            out_shardings = (None, None)
            jit_eval_step = jax.jit(
                eval_step, in_shardings=in_shardings, out_shardings=out_shardings
            )

            if args.use_fp8:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                check_fp8(state, var_collect, inputs, masks, labels)

            if args.dry_run:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                rngs = {DROPOUT_KEY: dropout_rng}
                jit_train_step(state, inputs, masks, labels, var_collect, rngs)
                print("PASSED")
                return None

            for epoch in range(1, args.epochs + 1):
                rng, input_rng = jax.random.split(rng)
                rng, dropout_rng = jax.random.split(rng)
                rngs = {INPUT_KEY: input_rng, DROPOUT_KEY: dropout_rng}

                state, train_loss, train_accuracy, var_collect = train_epoch(
                    state, train_ds, args.batch_size, rngs, var_collect, jit_train_step
                )

                test_loss, test_accuracy = eval_model(
                    state, test_ds, args.test_batch_size, var_collect, jit_eval_step
                )

                print(
                    f"Epoch: {epoch:>2} "
                    f"Train Loss: {train_loss:.6f} "
                    f"Train Accuracy: {train_accuracy:.6f} "
                    f"Test Loss: {test_loss:.6f} "
                    f"Test Accuracy: {test_accuracy:.6f} "
                )

            return [train_loss, train_accuracy, test_loss, test_accuracy]


def encoder_parser(args):
    """Training settings."""
    parser = argparse.ArgumentParser(description="JAX Encoder Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 256)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for testing (default: 256)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32,
        metavar="N",
        help="maximum sequence length (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 3)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        default=False,
        help="Use FP8 for inference and training without recalibration",
    )
    parser.add_argument(
        "--fp8-recipe",
        action="store_true",
        default="DelayedScaling",
        help="Use FP8 recipe (default: DelayedScaling)",
    )
    parser.add_argument(
        "--enable-shardy", action="store_true", default=False, help="Enable Shardy (experimental)."
    )

    return parser.parse_args(args)


class TestEncoder(unittest.TestCase):
    """Encoder unittests"""

    is_fp8_supported, fp8_reason = is_fp8_available(ScalingMode.DELAYED_TENSOR_SCALING)
    is_mxfp8_supported, mxfp8_reason = is_fp8_available(ScalingMode.MXFP8_1D_SCALING)

    def setUp(self):
        """Run 5 epochs for testing"""
        self.args = encoder_parser(["--epochs", "5"])

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74

    @unittest.skipIf(not is_fp8_supported, fp8_reason)
    def test_te_delayed_scaling_fp8(self):
        """Test Transformer Engine with DelayedScaling FP8"""
        self.args.use_fp8 = True
        self.args.fp8_recipe = "DelayedScaling"
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74

    @unittest.skipIf(not is_fp8_supported, fp8_reason)
    def test_te_current_scaling_fp8(self):
        """Test Transformer Engine with CurrentScaling FP8"""
        self.args.use_fp8 = True
        self.args.fp8_recipe = "Float8CurrentScaling"
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74

    @unittest.skipIf(not is_mxfp8_supported, mxfp8_reason)
    def test_te_mxfp8(self):
        """Test Transformer Engine with MXFP8"""
        self.args.use_fp8 = True
        self.args.fp8_recipe = "MXFP8BlockScaling"
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74

    @unittest.skipIf(not is_bf16_supported(), "Device compute capability 8.0+ is required for BF16")
    def test_te_bf16_shardy(self):
        """Test Transformer Engine with BF16"""
        self.args.enable_shardy = True
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74

    @unittest.skipIf(not is_fp8_supported, fp8_reason)
    def test_te_delayed_scaling_fp8_shardy(self):
        """Test Transformer Engine with DelayedScaling FP8"""
        self.args.enable_shardy = True
        self.args.use_fp8 = True
        self.args.fp8_recipe = "DelayedScaling"
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74

    @unittest.skipIf(not is_fp8_supported, fp8_reason)
    def test_te_current_scaling_fp8_shardy(self):
        """Test Transformer Engine with CurrentScaling FP8"""
        self.args.enable_shardy = True
        self.args.use_fp8 = True
        self.args.fp8_recipe = "Float8CurrentScaling"
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74

    @unittest.skipIf(not is_mxfp8_supported, mxfp8_reason)
    @unittest.skipIf(
        tex.gemm_uses_jax_dot(), "`jax.nn.scaled_matmul()` does not support the Shardy partitioner."
    )
    def test_te_mxfp8_shardy(self):
        """Test Transformer Engine with MXFP8"""
        self.args.enable_shardy = True
        self.args.use_fp8 = True
        self.args.fp8_recipe = "MXFP8BlockScaling"
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.52 and actual[1] > 0.74


if __name__ == "__main__":
    train_and_evaluate(encoder_parser(None))
