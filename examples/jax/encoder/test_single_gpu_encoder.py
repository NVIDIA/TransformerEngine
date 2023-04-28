# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
""" Encoder training on single GPU"""
import argparse
import os
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

import transformer_engine.jax as te

PARAMS_KEY = 'params'
DROPOUT_KEY = 'dropout'
INPUT_KEY = 'input_rng'


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

        te_Encoder = partial(te.flax.TransformerLayer,
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

        x = te.flax.DenseGeneral(features=256, dtype=jnp.bfloat16)(x)

        x = te.flax.DenseGeneral(features=256, dtype=jnp.bfloat16)(x)

        x = nn.Dense(features=2, dtype=jnp.bfloat16)(x)
        return x


@partial(jax.jit, static_argnums=6)
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


def train_epoch(state, train_ds, batch_size, rngs, var_collect, use_fp8):
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
        state, loss, accuracy, var_collect = train_step(state, batch_inputs, batch_masks,
                                                        batch_labels, var_collect, rngs, use_fp8)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    avg_loss = np.mean(epoch_loss)
    avg_accuracy = np.mean(epoch_accuracy)
    return state, avg_loss, avg_accuracy, var_collect


@jax.jit
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


def eval_model(state, test_ds, batch_size, var_collect):
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
        loss, accuracy = eval_step(state, batch_inputs, batch_masks, batch_labels, var_collect)
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


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(args)

    if args.use_fp8:
        assert gpu_has_fp8(), "GPU needs to support FP8."

    rng = jax.random.PRNGKey(args.seed)
    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}

    input_shape = [args.batch_size, args.max_seq_len]
    mask_shape = [args.batch_size, 1, args.max_seq_len, args.max_seq_len]
    label_shape = [args.batch_size]

    with te.fp8_autocast(enabled=args.use_fp8):
        train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)
        encoder = Net(num_embed)
        inputs = jnp.zeros(input_shape, dtype=jnp.int32)
        masks = jnp.zeros(mask_shape, dtype=jnp.uint8)
        var_collect = encoder.init(init_rngs, inputs, masks)
        tx = optax.adamw(args.lr)
        state = train_state.TrainState.create(apply_fn=encoder.apply,
                                              params=var_collect[PARAMS_KEY],
                                              tx=tx)

        if args.use_fp8:
            labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
            check_fp8(state, var_collect, inputs, masks, labels)

        if args.dry_run:
            labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
            rngs = {DROPOUT_KEY: dropout_rng}
            train_step(state, inputs, masks, labels, var_collect, rngs, args.use_fp8)
            print("PASSED")
            return None

        for epoch in range(1, args.epochs + 1):
            rng, input_rng = jax.random.split(rng)
            rng, dropout_rng = jax.random.split(rng)
            rngs = {INPUT_KEY: input_rng, DROPOUT_KEY: dropout_rng}

            state, train_loss, train_accuracy, var_collect = train_epoch(
                state, train_ds, args.batch_size, rngs, var_collect, args.use_fp8)

            test_loss, test_accuracy = eval_model(state, test_ds, args.test_batch_size, var_collect)

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
        """Run 4 epochs for testing"""
        cls.args = encoder_parser(["--epochs", "3"])

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
