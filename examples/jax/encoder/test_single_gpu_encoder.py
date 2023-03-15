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
    def __call__(self, x, disable_dropout=False):
        nn_Encoder = partial(te.TransformerLayer,
                             hidden_size=768,
                             mlp_hidden_size=3072,
                             num_attention_heads=12,
                             hidden_dropout=0.1,
                             attention_dropout=0.1,
                             dropout_rng_name=DROPOUT_KEY,
                             layer_type=te.TransformerLayerType.ENCODER,
                             enable_relative_embedding=False,
                             transpose_batch_sequence=False,
                             dtype=jnp.bfloat16)

        x = nn.Embed(num_embeddings=self.num_embed, features=768, dtype=jnp.bfloat16)(x)
        x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
        x = nn.Dropout(rate=0.1)(x, deterministic=disable_dropout)

        for _ in range(3):
            x = nn_Encoder()(x, deterministic=disable_dropout)

        x = x.reshape(x.shape[0], -1)
        x = te.DenseGeneral(features=768, dtype=jnp.bfloat16)(x)
        x = jnp.tanh(x)
        x = nn.Dense(features=2, dtype=jnp.bfloat16)(x)
        return x


@jax.jit
def apply_model(state, inputs, labels, var_collect, rngs=None):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, inputs, disable_dropout, rngs=rngs)
        one_hot = jax.nn.one_hot(labels, 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = FrozenDict({**var_collect, PARAMS_KEY: state.params})

    if rngs is not None:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(var_collect)
    else:
        loss, logits = loss_fn(var_collect, disable_dropout=True)
        grads = None

    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@partial(jax.jit, static_argnums=2)
def update_model(state, grads, use_fp8):
    "Update model params and FP8 meta."
    state = state.apply_gradients(grads=grads[PARAMS_KEY])
    if use_fp8:
        grads = te.update_fp8_metas(grads)
    return state, grads


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
        batch_labels = train_ds['label'][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_inputs, batch_labels, var_collect, rngs)
        state, var_collect = update_model(state, grads, use_fp8)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    avg_loss = np.mean(epoch_loss)
    avg_accuracy = np.mean(epoch_accuracy)
    return state, avg_loss, avg_accuracy, var_collect


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
        batch_labels = test_ds['label'][batch_start:batch_end]
        _, loss, accuracy = apply_model(state, batch_inputs, batch_labels, var_collect)
        all_loss.append(loss)
        all_accuracy.append(accuracy)

    avg_loss = np.mean(all_loss)
    avg_accuracy = np.mean(all_accuracy)
    return avg_loss, avg_accuracy


def data_preprocess(dataset, vocab, word_id, max_seq_len):
    nltk.download('punkt')
    dataset_size = len(dataset['sentence'])
    output = np.zeros((dataset_size, max_seq_len), dtype=np.int32)

    for j, sentence in enumerate(dataset['sentence']):
        tokens = nltk.word_tokenize(sentence.decode("utf-8"))
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

    dataset['sentence'] = output
    dataset['label'] = dataset['label'].astype(np.float32)
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


def check_fp8(state, var_collect, input_shape, output_shape):
    "Check if model includes FP8."
    assert "Float8" in str(
        jax.make_jaxpr(apply_model)(state, jnp.zeros(input_shape, dtype=jnp.int32),
                                    jnp.empty(output_shape, dtype=jnp.bfloat16), var_collect))


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    print(args)

    if args.use_fp8:
        assert gpu_has_fp8(), "GPU needs to support FP8."

    train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)
    rng = jax.random.PRNGKey(args.seed)
    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}

    input_shape = [args.batch_size, args.max_seq_len]
    output_shape = [args.batch_size]

    with te.fp8_autocast(enabled=args.use_fp8):
        encoder = Net(num_embed)
        var_collect = encoder.init(init_rngs, jnp.empty(input_shape, dtype=jnp.int32))
        tx = optax.adamw(args.lr)
        state = train_state.TrainState.create(apply_fn=encoder.apply,
                                              params=var_collect[PARAMS_KEY],
                                              tx=tx)

        if args.use_fp8:
            check_fp8(state, var_collect, input_shape, output_shape)

        if args.dry_run:
            apply_model(state,
                        jnp.zeros(input_shape, dtype=jnp.int32),
                        jnp.zeros(output_shape, dtype=jnp.bfloat16),
                        var_collect,
                        rngs={DROPOUT_KEY: dropout_rng})
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
        default=10,
        metavar="N",
        help="number of epochs to train (default: 14)",
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
        """Run Encoder from Transformer Engine with BF16"""
        cls.args = encoder_parser(["--epochs", "3"])
        cls.desired = train_and_evaluate(cls.args)

    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        assert self.desired[1] > 0.7 and self.desired[3] > 0.68

    @unittest.skipIf(not gpu_has_fp8(), reason='GPU capability is not enough to run FP8')
    def test_te_fp8(self):
        """Test Transformer Engine with FP8"""
        self.args.use_fp8 = True
        actual = train_and_evaluate(self.args)
        np.testing.assert_allclose(actual, self.desired, atol=0.001)


if __name__ == "__main__":
    train_and_evaluate(encoder_parser(None))
