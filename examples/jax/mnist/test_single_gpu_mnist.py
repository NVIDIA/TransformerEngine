# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""MNIST training on single GPU"""
import argparse
import unittest
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import load_dataset
from flax import linen as nn
from flax.training import train_state

import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax

IMAGE_H = 28
IMAGE_W = 28
IMAGE_C = 1
PARAMS_KEY = "params"
DROPOUT_KEY = "dropout"
INPUT_KEY = "input_rng"


class Net(nn.Module):
    """CNN model for MNIST."""

    use_te: bool = False

    @nn.compact
    def __call__(self, x, disable_dropout=False):
        if self.use_te:
            nn_Dense = te_flax.DenseGeneral
        else:
            nn_Dense = nn.Dense

        x = nn.Conv(features=32, kernel_size=(3, 3), strides=1, dtype=jnp.bfloat16)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=1, dtype=jnp.bfloat16)(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Dropout(rate=0.25)(x, deterministic=disable_dropout)
        x = x.reshape(x.shape[0], -1)
        x = nn_Dense(features=128, dtype=jnp.bfloat16)(x)
        x = nn.relu(x)
        x = nn.Dropout(rate=0.5)(x, deterministic=disable_dropout)
        x = nn_Dense(features=16, dtype=jnp.bfloat16)(x)
        x = nn.Dense(features=10, dtype=jnp.bfloat16)(x)
        return x


@jax.jit
def apply_model(state, images, labels, var_collect, rngs=None):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, images, disable_dropout, rngs=rngs)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = {**var_collect, PARAMS_KEY: state.params}

    if rngs is not None:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(var_collect)
    else:
        loss, logits = loss_fn(var_collect, disable_dropout=True)
        grads = None

    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@partial(jax.jit)
def update_model(state, grads):
    """Update model params and FP8 meta."""
    state = state.apply_gradients(grads=grads[PARAMS_KEY])
    return state, grads


def train_epoch(state, train_ds, batch_size, rngs, var_collect):
    """Train for a single epoch."""
    train_ds_size = len(train_ds["image"])
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rngs[INPUT_KEY], train_ds_size)
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_images = train_ds["image"][perm, ...]
        batch_labels = train_ds["label"][perm, ...]
        grads, loss, accuracy = apply_model(state, batch_images, batch_labels, var_collect, rngs)
        state, var_collect = update_model(state, grads)
        epoch_loss.append(loss)
        epoch_accuracy.append(accuracy)

    avg_loss = np.mean(epoch_loss)
    avg_accuracy = np.mean(epoch_accuracy)
    return state, avg_loss, avg_accuracy, var_collect


def eval_model(state, test_ds, batch_size, var_collect):
    """Evaluation loop."""
    test_ds_size = len(test_ds["image"])
    num_steps = test_ds_size // batch_size
    valid_size = num_steps * batch_size
    all_loss = []
    all_accuracy = []

    for batch_start in range(0, valid_size, batch_size):
        batch_end = batch_start + batch_size
        batch_images = test_ds["image"][batch_start:batch_end]
        batch_labels = test_ds["label"][batch_start:batch_end]
        _, loss, accuracy = apply_model(state, batch_images, batch_labels, var_collect)
        all_loss.append(loss)
        all_accuracy.append(accuracy)

    avg_loss = np.mean(all_loss)
    avg_accuracy = np.mean(all_accuracy)
    return avg_loss, avg_accuracy


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    train_ds = load_dataset("mnist", split="train", trust_remote_code=True)
    train_ds.set_format(type="np")
    batch_size = train_ds["image"].shape[0]
    shape = (batch_size, IMAGE_H, IMAGE_W, IMAGE_C)
    new_train_ds = {
        "image": train_ds["image"].astype(np.float32).reshape(shape) / 255.0,
        "label": train_ds["label"],
    }
    test_ds = load_dataset("mnist", split="test", trust_remote_code=True)
    test_ds.set_format(type="np")
    batch_size = test_ds["image"].shape[0]
    shape = (batch_size, IMAGE_H, IMAGE_W, IMAGE_C)
    new_test_ds = {
        "image": test_ds["image"].astype(np.float32).reshape(shape) / 255.0,
        "label": test_ds["label"],
    }
    return new_train_ds, new_test_ds


def check_fp8(state, var_collect, input_shape, label_shape):
    "Check if model includes FP8."
    assert "f8_" in str(
        jax.make_jaxpr(apply_model)(
            state,
            jnp.empty(input_shape, dtype=jnp.bfloat16),
            jnp.empty(label_shape, dtype=jnp.bfloat16),
            var_collect,
        )
    )


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)

    if args.use_fp8:
        args.use_te = True

    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(args.seed)
    rng, params_rng = jax.random.split(rng)
    rng, dropout_rng = jax.random.split(rng)
    init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}

    input_shape = [args.batch_size, IMAGE_H, IMAGE_W, IMAGE_C]
    label_shape = [args.batch_size]

    with te.fp8_autocast(enabled=args.use_fp8):
        cnn = Net(args.use_te)
        var_collect = cnn.init(init_rngs, jnp.empty(input_shape, dtype=jnp.bfloat16))
        tx = optax.sgd(args.lr, args.momentum)
        state = train_state.TrainState.create(
            apply_fn=cnn.apply, params=var_collect[PARAMS_KEY], tx=tx
        )

        if args.use_fp8:
            check_fp8(state, var_collect, input_shape, label_shape)

        if args.dry_run:
            apply_model(
                state,
                jnp.empty(input_shape, dtype=jnp.bfloat16),
                jnp.empty(label_shape, dtype=jnp.bfloat16),
                var_collect,
                {DROPOUT_KEY: dropout_rng},
            )
            print("PASSED")
            return None

        for epoch in range(1, args.epochs + 1):
            rng, input_rng = jax.random.split(rng)
            rng, dropout_rng = jax.random.split(rng)
            rngs = {INPUT_KEY: input_rng, DROPOUT_KEY: dropout_rng}

            state, train_loss, train_accuracy, var_collect = train_epoch(
                state, train_ds, args.batch_size, rngs, var_collect
            )
            test_loss, test_accuracy = eval_model(state, test_ds, args.test_batch_size, var_collect)

            print(
                f"Epoch: {epoch:>2} "
                f"Train Loss: {train_loss:.6f} "
                f"Train Accuracy: {train_accuracy:.6f} "
                f"Test Loss: {test_loss:.6f} "
                f"Test Accuracy: {test_accuracy:.6f} "
            )

    return [train_loss, train_accuracy, test_loss, test_accuracy]


def mnist_parser(args):
    """Training settings."""
    parser = argparse.ArgumentParser(description="JAX MNIST Example")
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
        default=800,
        metavar="N",
        help="input batch size for testing (default: 800)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Momentum (default: 0.9)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        default=False,
        help=(
            "Use FP8 for inference and training without recalibration. "
            "It also enables Transformer Engine implicitly."
        ),
    )
    parser.add_argument(
        "--use-te", action="store_true", default=False, help="Use Transformer Engine"
    )

    return parser.parse_args(args)


class TestMNIST(unittest.TestCase):
    """MNIST unittests"""

    gpu_has_fp8, reason = te.fp8.is_fp8_available()

    @classmethod
    def setUpClass(cls):
        """Run MNIST without Transformer Engine"""
        cls.args = mnist_parser(["--epochs", "5"])

    @staticmethod
    def verify(actual):
        """Check If loss and accuracy match target"""
        desired_traing_loss = 0.055
        desired_traing_accuracy = 0.98
        desired_test_loss = 0.04
        desired_test_accuracy = 0.098
        assert actual[0] < desired_traing_loss
        assert actual[1] > desired_traing_accuracy
        assert actual[2] < desired_test_loss
        assert actual[3] > desired_test_accuracy

    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        self.args.use_te = True
        self.args.use_fp8 = False
        actual = train_and_evaluate(self.args)
        self.verify(actual)

    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_te_fp8(self):
        """Test Transformer Engine with FP8"""
        self.args.use_fp8 = True
        actual = train_and_evaluate(self.args)
        self.verify(actual)


if __name__ == "__main__":
    train_and_evaluate(mnist_parser(None))
