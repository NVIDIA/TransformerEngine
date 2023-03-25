# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
""" Encoder training on multi-GPU with data parallelism"""
import argparse
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import nltk
import numpy as np
import optax
import tensorflow_datasets as tfds
from cuda import cudart
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from jax.experimental import mesh_utils
from jax.experimental.pjit import pjit

import transformer_engine.jax as te

DEVICE_DP_AXIS = 'data'
PARAMS_KEY = 'params'
PARAMS_AXES_KEY = PARAMS_KEY + '_axes'
DROPOUT_KEY = 'dropout'
INPUT_KEY = 'input_rng'


def check_num_gpu(desired_num_gpu):
    """Check if the number of GPUs are correct."""
    actual_num_gpu = len(jax.local_devices())
    assert actual_num_gpu == desired_num_gpu, f"Number of GPUs is mismatch. " \
        f"{desired_num_gpu} GPUs are assigned, but the actual number of GPUs is {actual_num_gpu}"


def gpu_has_fp8():
    """Check if the GPU has FP8."""
    cudaSuccess = cudart.cudaError_t.cudaSuccess
    ret, gpu_id = cudart.cudaGetDevice()
    assert ret == cudaSuccess
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor
    _, major = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    flag = cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor
    _, minor = cudart.cudaDeviceGetAttribute(flag, gpu_id)
    sm_arch = major * 10 + minor
    return sm_arch >= 89


class Net(nn.Module):
    """NLP Encoder"""
    num_embed: int

    @nn.compact
    def __call__(self, x, mask, disable_dropout=False):
        x = nn.Embed(num_embeddings=self.num_embed, features=256, dtype=jnp.bfloat16)(x)

        te_Encoder = partial(te.TransformerLayer,
                             hidden_size=256,
                             mlp_hidden_size=1024,
                             num_attention_heads=8,
                             hidden_dropout=0.1,
                             attention_dropout=0.1,
                             dropout_rng_name=DROPOUT_KEY,
                             layer_type=te.TransformerLayerType.ENCODER,
                             enable_relative_embedding=False,
                             dtype=jnp.bfloat16)
        x = te_Encoder()(x, attention_mask=mask, deterministic=disable_dropout)

        x = x.reshape(x.shape[0], -1)

        x = te.DenseGeneral(features=256, sharding_type=te.ShardingType.DP, dtype=jnp.bfloat16)(x)

        x = te.DenseGeneral(features=256, sharding_type=te.ShardingType.DP, dtype=jnp.bfloat16)(x)

        x = nn.Dense(features=2, dtype=jnp.bfloat16)(x)
        return x


def train_step(state, inputs, masks, labels, var_collect, rngs, use_fp8):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, inputs, masks, disable_dropout, rngs=rngs)
        one_hot = jax.nn.one_hot(labels, 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = FrozenDict({**var_collect, PARAMS_KEY: state.params})
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(var_collect)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    var_collect, grads = grads.pop(PARAMS_KEY)
    state = state.apply_gradients(grads=grads)
    if use_fp8:
        var_collect = te.update_fp8_metas(var_collect)

    return state, loss, accuracy, var_collect


def train_epoch(state, train_ds, batch_size, rngs, var_collect, use_fp8, train_fn):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['sentence'])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rngs[INPUT_KEY], train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]    # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_inputs = train_ds['sentence'][perm, ...]
        batch_masks = train_ds['mask'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        state, loss, accuracy, var_collect = train_fn(state, batch_inputs, batch_masks,
                                                      batch_labels, var_collect, rngs, use_fp8)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    avg_loss = np.mean(epoch_loss)
    avg_accuracy = np.mean(epoch_accuracy)
    return state, avg_loss, avg_accuracy, var_collect


def eval_step(state, inputs, masks, labels, var_collect):
    """Computes loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, inputs, masks, disable_dropout)
        one_hot = jax.nn.one_hot(labels, 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = FrozenDict({**var_collect, PARAMS_KEY: state.params})
    loss, logits = loss_fn(var_collect, disable_dropout=True)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


def eval_model(state, test_ds, batch_size, var_collect, eval_fn):
    """Evaluation loop."""
    test_ds_size = len(test_ds['sentence'])
    num_steps = test_ds_size // batch_size
    valid_size = num_steps * batch_size
    all_loss = []
    all_accuracy = []

    for batch_start in range(0, valid_size, batch_size):
        batch_end = batch_start + batch_size
        batch_inputs = test_ds['sentence'][batch_start:batch_end]
        batch_masks = test_ds['mask'][batch_start:batch_end]
        batch_labels = test_ds['label'][batch_start:batch_end]
        loss, accuracy = eval_fn(state, batch_inputs, batch_masks, batch_labels, var_collect)
        all_loss.append(loss)
        all_accuracy.append(accuracy)

    avg_loss = np.mean(all_loss)
    avg_accuracy = np.mean(all_accuracy)
    return avg_loss, avg_accuracy


def data_preprocess(dataset, vocab, word_id, max_seq_len):
    """Convert tokens to numbers."""
    nltk.download('punkt')
    dataset_size = len(dataset['sentence'])
    output = np.zeros((dataset_size, max_seq_len), dtype=np.int32)
    mask_3d = np.empty((dataset_size, max_seq_len, max_seq_len), dtype=np.uint8)

    for j, sentence in enumerate(dataset['sentence']):
        tokens = nltk.word_tokenize(sentence.decode("utf-8"))
        tensor = output[j]
        mask_1d = np.zeros((1, max_seq_len), dtype=np.uint8)

        for i, word in enumerate(tokens):
            if i >= max_seq_len:
                break

            if word not in vocab:
                vocab[word] = word_id
                tensor[i] = word_id
                word_id = word_id + 1
            else:
                tensor[i] = vocab[word]

            mask_1d[0, i] = 1

        mask_2d = mask_3d[j]
        np.dot(mask_1d.T, mask_1d, out=mask_2d)
        np.subtract(1, mask_2d, out=mask_2d)

    dataset['sentence'] = output
    dataset['label'] = dataset['label'].astype(np.float32)
    dataset['mask'] = mask_3d.reshape((dataset_size, 1, max_seq_len, max_seq_len))
    return dataset, vocab, word_id


def get_datasets(max_seq_len):
    """Load GLUE train and test datasets into memory."""
    vocab = {}
    word_id = 0
    dataset = 'glue/cola'
    train_ds = tfds.as_numpy(tfds.load(dataset, split='train', batch_size=-1))
    train_ds, vocab, word_id = data_preprocess(train_ds, vocab, word_id, max_seq_len)
    test_ds = tfds.as_numpy(tfds.load(dataset, split='validation', batch_size=-1))
    test_ds, vocab, word_id = data_preprocess(test_ds, vocab, word_id, max_seq_len)
    return train_ds, test_ds, word_id


def check_fp8(state, var_collect, inputs, masks, labels):
    "Check if model includes FP8."
    rngs = {DROPOUT_KEY: jax.random.PRNGKey(0)}
    assert "Float8" in str(
        jax.make_jaxpr(train_step, static_argnums=6)(state, inputs, masks, labels, var_collect,
                                                     rngs, True))


def get_params_pspec(sharding_rules, abs_var_collect):
    """Refer params to create params partition spec"""
    rules_dict = {}
    for key, value in sharding_rules:
        rules_dict[key] = value

    def to_device_axis(logical_axis):
        partitions = [rules_dict[key] for key in logical_axis]
        return jax.sharding.PartitionSpec(*partitions)

    params_axes = abs_var_collect.get(PARAMS_AXES_KEY, {})
    params_axes_pspec = jax.tree_map(to_device_axis, nn.partitioning.get_axis_names(params_axes))
    params_pspec = jax.tree_map(lambda x: jax.sharding.PartitionSpec(), abs_var_collect[PARAMS_KEY])
    params_pspec = FrozenDict({**params_pspec, **params_axes_pspec})
    return params_pspec


def get_state_pspec(state, params_pspec):
    """Refer params_pspec to create state partition spec"""

    def replace_params(x):
        return params_pspec if isinstance(x, FrozenDict) else None

    state_pspec = jax.tree_map(replace_params, state, is_leaf=lambda x: isinstance(x, FrozenDict))
    return state_pspec


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)
    check_num_gpu(args.num_gpu)
    assert args.batch_size % args.num_gpu == 0, f"Batch size needs to be multiple of {args.num_gpu}"
    assert args.test_batch_size % args.num_gpu == 0, \
        f"Test batch size needs to be multiple of {args.num_gpu}"

    if args.use_fp8:
        assert gpu_has_fp8(), "GPU needs to support FP8."

    device_mesh = mesh_utils.create_device_mesh((args.num_gpu,))
    with jax.sharding.Mesh(devices=device_mesh, axis_names=(DEVICE_DP_AXIS,)):

        rng = jax.random.PRNGKey(args.seed)
        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}

        input_shape = [args.batch_size, args.max_seq_len]
        mask_shape = [args.batch_size, 1, args.max_seq_len, args.max_seq_len]
        label_shape = [args.batch_size]

        with te.fp8_autocast(args.use_fp8, sharding_resource=te.ShardingResource(DEVICE_DP_AXIS)):
            train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)
            encoder = Net(num_embed)
            inputs = jnp.zeros(input_shape, dtype=jnp.int32)
            masks = jnp.zeros(mask_shape, dtype=jnp.uint8)
            abs_var_collect = jax.eval_shape(encoder.init, init_rngs, inputs, masks)

            sharding_rules = te.extend_logical_axis_rules(tuple())
            params_pspec = get_params_pspec(sharding_rules, abs_var_collect)
            inputs_pspec = jax.sharding.PartitionSpec(DEVICE_DP_AXIS, None)
            masks_pspec = jax.sharding.PartitionSpec(DEVICE_DP_AXIS, None, None, None)

            in_shardings = (None, inputs_pspec, masks_pspec)
            out_shardings = FrozenDict({key: params_pspec if key is PARAMS_KEY else None \
                                        for key in abs_var_collect})
            pjit_encoder_init = pjit(encoder.init, in_shardings, out_shardings)
            var_collect = pjit_encoder_init(init_rngs, inputs, masks)

            optimizer = optax.adamw(args.lr)
            var_collect, params = var_collect.pop(PARAMS_KEY)
            state = train_state.TrainState.create(apply_fn=encoder.apply,
                                                  params=params,
                                                  tx=optimizer)
            state_pspec = get_state_pspec(state, params_pspec)
            labels_pspec = jax.sharding.PartitionSpec(DEVICE_DP_AXIS,)

            in_shardings = (state_pspec, inputs_pspec, masks_pspec, labels_pspec, None, None)
            out_shardings = (state_pspec, None, None, None)
            pjit_train_step = pjit(train_step, in_shardings, out_shardings, static_argnums=(6,))

            in_shardings = (state_pspec, inputs_pspec, masks_pspec, labels_pspec, None)
            out_shardings = (None, None)
            pjit_eval_step = pjit(eval_step, in_shardings, out_shardings)

            if args.use_fp8:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                check_fp8(state, var_collect, inputs, masks, labels)

            if args.dry_run:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                rngs = {DROPOUT_KEY: dropout_rng}
                pjit_train_step(state, inputs, masks, labels, var_collect, rngs, args.use_fp8)
                print("PASSED")
                return None

            for epoch in range(1, args.epochs + 1):
                rng, input_rng = jax.random.split(rng)
                rng, dropout_rng = jax.random.split(rng)
                rngs = {INPUT_KEY: input_rng, DROPOUT_KEY: dropout_rng}

                state, train_loss, train_accuracy, var_collect = train_epoch(
                    state, train_ds, args.batch_size, rngs, var_collect, args.use_fp8,
                    pjit_train_step)

                test_loss, test_accuracy = eval_model(state, test_ds, args.test_batch_size,
                                                      var_collect, pjit_eval_step)

                print(f"Epoch: {epoch:>2} "
                      f"Train Loss: {train_loss:.6f} "
                      f"Train Accuracy: {train_accuracy:.6f} "
                      f"Test Loss: {test_loss:.6f} "
                      f"Test Accuracy: {test_accuracy:.6f} ")

            return [train_loss, train_accuracy, test_loss, test_accuracy]


def encoder_parser(args):
    """Training settings."""
    parser = argparse.ArgumentParser(description="JAX Encoder Example")
    parser.add_argument(
        "--num-gpu",
        type=int,
        default=8,
        metavar="N",
        help="number of GPUs (default: 8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for testing (default: 64)",
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
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument("--use-fp8",
                        action="store_true",
                        default=False,
                        help="Use FP8 for inference and training without recalibration")

    return parser.parse_args(args)


class TestEncoder(unittest.TestCase):
    """Encoder unittests"""

    @classmethod
    def setUpClass(cls):
        """Run 3 epochs for testing"""
        num_gpu = len(jax.local_devices())
        if num_gpu % 2 != 0:
            num_gpu = 1
        cls.args = encoder_parser(["--epochs", "3", "--num-gpu", str(num_gpu)])

    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.45 and actual[1] > 0.79

    @unittest.skipIf(not gpu_has_fp8(), reason='GPU capability is not enough to run FP8')
    def test_te_fp8(self):
        """Test Transformer Engine with FP8"""
        self.args.use_fp8 = True
        actual = train_and_evaluate(self.args)
        assert actual[0] < 0.45 and actual[1] > 0.79


if __name__ == "__main__":
    train_and_evaluate(encoder_parser(None))
