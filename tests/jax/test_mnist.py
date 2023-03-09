# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import os
import tempfile
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
from flax import linen as nn
from flax.training import train_state

from transformer_engine.common.recipe import Format as FP8Format
from transformer_engine.jax import DenseGeneral
from transformer_engine.jax.fp8 import FP8Helper
from utils import is_fp8_supported


class MLPNN(nn.Module):

    use_fp8_dense: bool = True

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))    # flatten
        x = nn.Dense(features=512)(x)
        x = nn.relu(x)

        features = [256, 256, 128]
        for feature in features:
            x = DenseGeneral(features=feature, transpose_batch_sequence=False,
                             dtype=jnp.bfloat16, use_bias=True)(x) \
                if self.use_fp8_dense else nn.Dense(features=feature)(x)
            x = nn.relu(x)

        x = nn.Dense(features=10, use_bias=True)(x)
        return x


def cross_entropy_loss(*, logits, labels):
    labels_onehot = jax.nn.one_hot(labels, num_classes=10)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def compute_metrics(*, logits, labels):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
    }
    return metrics


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist', data_dir="/tmp/tensorflow-datasets/downloads")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


def create_train_state(rng, learning_rate, momentum, use_fp8_dense):
    """Creates initial `TrainState`."""
    cnn = MLPNN(use_fp8_dense=use_fp8_dense)
    variables = cnn.init(rng, jnp.ones([32, 28, 28, 1]))
    tx = optax.sgd(learning_rate, momentum)
    return train_state.TrainState.create(apply_fn=cnn.apply, params=variables['params'],
                                         tx=tx), variables


@partial(jax.jit, static_argnums=(3,))
def train_step(state, others, batch, use_fp8_dense):
    """Train for a single step."""

    def loss_fn(collections):
        logits = MLPNN(use_fp8_dense=use_fp8_dense).apply(collections, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(others)
    state = state.apply_gradients(grads=grads['params'])
    metrics = compute_metrics(logits=logits, labels=batch['label'])
    return state, metrics, grads


def train_epoch(state, variables, train_ds, batch_size, rng, use_fp8_dense):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size]    # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for idx, perm in enumerate(perms):
        idx = idx + 1
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics, grads = train_step(state, variables, batch, use_fp8_dense)

        updated_coll = {'params': state.params}
        if use_fp8_dense:
            updated_coll[FP8Helper.FP8_COLLECTION_NAME] \
                = grads[FP8Helper.FP8_COLLECTION_NAME]
        variables = FP8Helper.update_collections(updated_coll, variables)
        batch_metrics.append(metrics)

        if use_fp8_dense:
            variables = FP8Helper.update_fp8_metas(variables)

    return state, variables


@partial(jax.jit, static_argnums=(2,))
def eval_step(variables, batch, use_fp8_dense):
    logits = MLPNN(use_fp8_dense=use_fp8_dense).apply(variables, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'])


def eval_model(variables, test_ds, batch_size, use_fp8_dense):
    test_ds_size = len(test_ds['image'])
    steps_per_epoch = test_ds_size // batch_size
    perms = np.arange(0, test_ds_size)
    perms = perms[:steps_per_epoch * batch_size]    # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    total_summary = {'correct': 0, 'loss': 0, 'total': 0}
    for _, perm in enumerate(perms):
        batch = {k: v[perm, ...] for k, v in test_ds.items()}
        metrics = eval_step(variables, batch, use_fp8_dense)
        metrics = jax.device_get(metrics)
        summary = jax.tree_map(lambda x: x.item(), metrics)
        total_summary['correct'] += summary['accuracy'] * batch_size
        total_summary['loss'] += summary['loss'] * batch_size
        total_summary['total'] += batch_size
    return total_summary['loss']/total_summary['total'], \
           total_summary['correct']/total_summary['total']


class TestMnist(unittest.TestCase):

    def setUp(self) -> None:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        self.learning_rate = 0.1
        self.momentum = 0.9

        self.num_epochs = 5
        self.batch_size = 512
        self.train_ds, self.test_ds = get_datasets()

        self.margin = 0.0
        self.num_fp8_layers = 3
        self.fp8_meta_update_interval = 1
        self.temp_file = tempfile.NamedTemporaryFile()    # pylint: disable=consider-using-with
        self.fp8_ckpt_path = self.temp_file.name

        self.seed = 0

        acc_bfp16_ = self._mnist_baseline_runner()
        acc_rtol = 0.005
        self.target_accuracy = acc_bfp16_ * (1. - acc_rtol)

    def tearDown(self):
        self.temp_file.close()

    @unittest.skipIf(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    def test_mnist_e4m3(self):
        self._mnist_test_runner(FP8Format.E4M3)

    @unittest.skipIf(not is_fp8_supported(), reason='GPU capability is not enough to run FP8')
    def test_mnist_hybrid(self):
        self._mnist_test_runner(FP8Format.HYBRID)

    # Skip for now due to lack bf16 in TE.Format
    # def test_mnist_bfloa16(self):
    #     self._mnist_test_runner(FP8Format.BFLOAT16)

    def _mnist_baseline_runner(self):
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)

        state, variables = create_train_state(init_rng, self.learning_rate, self.momentum, False)
        del init_rng

        _, accuracy = self._train_model(state, variables, self.num_epochs, rng, False)
        return accuracy

    def _mnist_test_runner(self, fp8_format):
        FP8Helper.initialize(margin=self.margin, fp8_format=fp8_format)

        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)

        state, variables = create_train_state(init_rng, self.learning_rate, self.momentum, True)
        del init_rng

        _, test_accuracy = self._train_model(state, variables, self.num_epochs, rng, True)

        self.assertGreater(
            test_accuracy, self.target_accuracy,
            f"Convergence test failed on MNIST with FP8Fomat.{fp8_format.name}. "
            f"Test Accuracy {test_accuracy:.4f} is lower than target {self.target_accuracy:.4f}")

        FP8Helper.finalize()

    def _train_model(self, state, variables, epochs, rng, use_fp8_dense):
        max_test_acc = 0.0
        for _ in range(0, epochs):
            rng, input_rng = jax.random.split(rng)

            state, variables = train_epoch(state, variables, self.train_ds, self.batch_size,
                                           input_rng, use_fp8_dense)

            _, test_accuracy = eval_model(variables, self.test_ds, self.batch_size, use_fp8_dense)
            max_test_acc = test_accuracy if test_accuracy > max_test_acc else max_test_acc
        return state, max_test_acc


if __name__ == '__main__':
    unittest.main()
