# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import transformer_engine.tensorflow as te


class MNIST(tf.keras.Model):
    def __init__(self, use_te=False):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten(input_shape=(28, 28))
        if use_te:
            self.dense1 = te.Dense(128, kernel_initializer='glorot_uniform',
                                   bias_initializer='zeros')
        else:
            self.dense1 = tf.keras.layers.Dense(128, activation=None)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu(x)
        y = self.dense2(x)
        return y


loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def train_step(inputs, model, optimizer, use_fp8, fp8_recipe=None):
    x, labels = inputs
    with tf.GradientTape(persistent=True) as tape:
        with te.fp8_autocast(enabled=use_fp8, fp8_recipe=fp8_recipe):
            y = model(x, training=True)
        loss = loss_func(labels, y)

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def valid_step(inputs, model):
    x, labels = inputs
    predictions = model(x, training=False)
    loss = loss_func(labels, predictions)

    val_loss.update_state(loss)
    val_accuracy.update_state(labels, predictions)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Tensorflow MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
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
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--seed", type=int, default=12, metavar="S",
        help="random seed (default: 12)"
    )
    parser.add_argument(
        "--use-fp8", action="store_true", default=False,
        help="Use FP8 for inference and training without recalibration"
    )
    parser.add_argument(
        "--use-te", action="store_true", default=False,
        help="Use Transformer Engine"
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    test_batch_size = args.test_batch_size
    num_epoch = args.epochs

    tf.random.set_seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)

    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    nstep_per_epoch = len(ds_train) // batch_size
    nstep_per_valid = len(ds_test) // test_batch_size

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    model = MNIST(use_te=(args.use_te or args.use_fp8))

    optimizer = tf.keras.optimizers.Adam(args.lr)

    fp8_recipe = te.DelayedScaling(
        margin=0, interval=1, fp8_format=te.Format.HYBRID,
        amax_compute_algo='max', amax_history_len=16)

    for i in range(num_epoch):
        ds_train_iter = iter(ds_train)
        for _ in range(nstep_per_epoch):
            inputs = next(ds_train_iter)
            _ = train_step(inputs, model, optimizer, use_fp8=args.use_fp8,
                           fp8_recipe=fp8_recipe)

        val_loss.reset_states()
        val_accuracy.reset_states()
        ds_test_iter = iter(ds_test)
        for _ in range(nstep_per_valid):
            inputs = next(ds_test_iter)
            valid_step(inputs, model)

        print("epoch-{} loss: {} - accuracy: {}".format(
            i, val_loss.result(), val_accuracy.result()))


if __name__ == "__main__":
    main()
