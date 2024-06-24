# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Encoder training with multi-GPU, multiprocessing, and tensor parallelism"""
import argparse
import multiprocessing as mp
import os
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
from jax.experimental.pjit import pjit

import transformer_engine.jax as te
import transformer_engine.jax.flax as te_flax

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
DEVICE_DP_AXIS = "data"
DEVICE_TP_AXIS = "model"
NAMED_BROADCAST_AXIS = "my_broadcast_axis"
NAMED_TP_AXIS = "my_tp_axis"
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
            dtype=jnp.bfloat16,
        )
        x = te_Encoder()(x, attention_mask=mask, deterministic=disable_dropout)

        x = x.reshape(x.shape[0], -1)

        x = te_flax.DenseGeneral(
            features=256,
            kernel_axes=(NAMED_BROADCAST_AXIS, NAMED_TP_AXIS),
            bias_axes=(NAMED_TP_AXIS,),
            dtype=jnp.bfloat16,
        )(x)

        x = te_flax.DenseGeneral(
            features=256,
            kernel_axes=(NAMED_TP_AXIS, NAMED_BROADCAST_AXIS),
            bias_axes=(NAMED_BROADCAST_AXIS,),
            dtype=jnp.bfloat16,
        )(x)

        x = nn.Dense(features=2, dtype=jnp.bfloat16)(x)
        return x


def valid_shard_size(total_size, batch_size, dp_size, tp_size):
    """Get sharded input shape"""
    global_batch_size = dp_size * batch_size
    num_steps = total_size // global_batch_size
    valid_size = num_steps * global_batch_size

    gpu_id = jax.local_devices()[0].id
    tp_group_id = gpu_id // tp_size
    return valid_size, global_batch_size, num_steps, tp_group_id


def shard_array_wrapper(dataset, batch_size, mesh, pspec, enable_partition=False):
    """Generate needed args for jax.make_array_from_single_device_arrays"""
    inputs = jnp.asarray(dataset)
    total_input_size = len(inputs)

    (dp_size, tp_size) = mesh.device_ids.shape
    valid_input_size, global_batch_size, num_steps, tp_group_id = valid_shard_size(
        total_input_size, batch_size, dp_size, tp_size
    )
    inputs = inputs[:valid_input_size]  # skip incomplete batch

    single_input_shape = inputs.shape[1:]
    global_input_shape = (global_batch_size, *single_input_shape)

    named_sharding = jax.sharding.NamedSharding(mesh, pspec)

    if enable_partition:
        inputs = inputs.reshape(dp_size, num_steps, batch_size, *single_input_shape)
        inputs = inputs[tp_group_id]
    return global_input_shape, named_sharding, inputs


def train_step(state, inputs, masks, labels, var_collect, rngs):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(var_collect, disable_dropout=False):
        logits = state.apply_fn(var_collect, inputs, masks, disable_dropout, rngs=rngs)
        one_hot = jax.nn.one_hot(labels, 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = {**var_collect, PARAMS_KEY: state.params}
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(var_collect)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    var_collect, grads = flax.core.pop(grads, PARAMS_KEY)
    state = state.apply_gradients(grads=grads)

    return state, loss, accuracy, var_collect


def train_epoch(
    state,
    train_ds,
    batch_size,
    rngs,
    var_collect,
    train_fn,
    mesh,
    inputs_pspec,
    masks_pspec,
    labels_pspec,
):
    """Train for a single epoch."""

    total_batch_size = len(train_ds["sentence"])
    (dp_size, tp_size) = mesh.device_ids.shape
    valid_size, _, num_steps, tp_group_id = valid_shard_size(
        total_batch_size, batch_size, dp_size, tp_size
    )

    perms = jax.random.permutation(rngs[INPUT_KEY], valid_size)
    perms = perms.reshape(dp_size, num_steps, batch_size)
    perms = perms[tp_group_id]

    global_input_shape, input_named_sharding, sentence = shard_array_wrapper(
        train_ds["sentence"], batch_size, mesh, inputs_pspec
    )
    global_mask_shape, mask_named_sharding, mask = shard_array_wrapper(
        train_ds["mask"], batch_size, mesh, masks_pspec
    )
    global_label_shape, label_named_sharding, label = shard_array_wrapper(
        train_ds["label"], batch_size, mesh, labels_pspec
    )

    epoch_loss = []
    epoch_accuracy = []

    for perm in perms:
        batch_input = sentence[perm, ...]
        batch_mask = mask[perm, ...]
        batch_label = label[perm, ...]

        shard_input = jax.make_array_from_single_device_arrays(
            global_input_shape, input_named_sharding, [batch_input]
        )
        shard_mask = jax.make_array_from_single_device_arrays(
            global_mask_shape, mask_named_sharding, [batch_mask]
        )
        shard_label = jax.make_array_from_single_device_arrays(
            global_label_shape, label_named_sharding, [batch_label]
        )

        state, loss, accuracy, var_collect = train_fn(
            state, shard_input, shard_mask, shard_label, var_collect, rngs
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
        one_hot = jax.nn.one_hot(labels, 2)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    var_collect = {**var_collect, PARAMS_KEY: state.params}
    loss, logits = loss_fn(var_collect, disable_dropout=True)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return loss, accuracy


def eval_model(
    state, test_ds, batch_size, var_collect, eval_fn, mesh, inputs_pspec, masks_pspec, labels_pspec
):
    """Evaluation loop."""
    global_input_shape, input_named_sharding, sentence = shard_array_wrapper(
        test_ds["sentence"], batch_size, mesh, inputs_pspec, enable_partition=True
    )
    global_mask_shape, mask_named_sharding, mask = shard_array_wrapper(
        test_ds["mask"], batch_size, mesh, masks_pspec, enable_partition=True
    )
    global_label_shape, label_named_sharding, label = shard_array_wrapper(
        test_ds["label"], batch_size, mesh, labels_pspec, enable_partition=True
    )

    all_loss = []
    all_accuracy = []

    for batch_input, batch_mask, batch_label in zip(sentence, mask, label):

        shard_input = jax.make_array_from_single_device_arrays(
            global_input_shape, input_named_sharding, [batch_input]
        )
        shard_mask = jax.make_array_from_single_device_arrays(
            global_mask_shape, mask_named_sharding, [batch_mask]
        )
        shard_label = jax.make_array_from_single_device_arrays(
            global_label_shape, label_named_sharding, [batch_label]
        )

        loss, accuracy = eval_fn(state, shard_input, shard_mask, shard_label, var_collect)
        all_loss.append(loss)
        all_accuracy.append(accuracy)

    avg_loss = np.mean(all_loss)
    avg_accuracy = np.mean(all_accuracy)
    return avg_loss, avg_accuracy


def data_preprocess(dataset, vocab, word_id, max_seq_len):
    """Convert tokens to numbers."""
    nltk.download("punkt")
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
    assert "fp8_" in str(
        jax.make_jaxpr(train_step)(state, inputs, masks, labels, var_collect, rngs)
    )


def get_params_pspec(sharding_rules, abs_var_collect):
    """Refer params to create params partition spec"""
    rules_dict = {}
    for key, value in sharding_rules:
        rules_dict[key] = value

    def to_device_axis(logical_axis):
        partitions = [rules_dict[key] for key in logical_axis]
        return jax.sharding.PartitionSpec(*partitions)

    params_axes = abs_var_collect.get(PARAMS_AXES_KEY, {})
    params_axes_pspec = jax.tree_map(to_device_axis, nn_partitioning.get_axis_names(params_axes))
    params_axes_pspec = flax.core.unfreeze(params_axes_pspec)
    params_pspec = jax.tree_map(lambda x: jax.sharding.PartitionSpec(), abs_var_collect[PARAMS_KEY])
    params_pspec = {**params_pspec, **params_axes_pspec}
    return params_pspec


def get_state_pspec(state, params_pspec):
    """Refer params_pspec to create state partition spec"""

    def replace_params(x):
        return params_pspec if isinstance(x, dict) else None

    state_pspec = jax.tree_map(replace_params, state, is_leaf=lambda x: isinstance(x, dict))
    return state_pspec


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)
    train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)

    jax.distributed.initialize(
        coordinator_address=args.coordinator_address,
        num_processes=args.num_process,
        process_id=args.process_id,
        local_device_ids=args.process_id,
    )
    assert jax.local_device_count() == 1, "1 GPU per process"

    num_gpu_tp = 2
    if args.num_process % num_gpu_tp == 0:
        num_gpu_dp = args.num_process // num_gpu_tp
    else:
        assert args.num_process == 1, "number of processes should be multiple of 2, or 1"
        num_gpu_dp = 1
        num_gpu_tp = 1

    assert args.batch_size % num_gpu_dp == 0, f"Batch size needs to be multiple of {num_gpu_dp}"
    assert (
        args.test_batch_size % num_gpu_dp == 0
    ), f"Test batch size needs to be multiple of {num_gpu_dp}"

    device_mesh = mesh_utils.create_device_mesh((num_gpu_dp, num_gpu_tp))
    with jax.sharding.Mesh(
        devices=device_mesh, axis_names=(DEVICE_DP_AXIS, DEVICE_TP_AXIS)
    ) as shard_mesh:

        rng = jax.random.PRNGKey(args.seed)
        rng, params_rng = jax.random.split(rng)
        rng, dropout_rng = jax.random.split(rng)
        init_rngs = {PARAMS_KEY: params_rng, DROPOUT_KEY: dropout_rng}

        input_shape = [args.batch_size, args.max_seq_len]
        mask_shape = [args.batch_size, 1, args.max_seq_len, args.max_seq_len]
        label_shape = [args.batch_size]

        with te.fp8_autocast(
            args.use_fp8, mesh_resource=te.MeshResource(DEVICE_DP_AXIS, DEVICE_TP_AXIS, None, None)
        ):
            encoder = Net(num_embed)
            inputs = jnp.zeros(input_shape, dtype=jnp.int32)
            masks = jnp.zeros(mask_shape, dtype=jnp.uint8)
            abs_var_collect = jax.eval_shape(encoder.init, init_rngs, inputs, masks)

            customized_rules = ((NAMED_BROADCAST_AXIS, None), (NAMED_TP_AXIS, DEVICE_TP_AXIS))
            sharding_rules = te_flax.extend_logical_axis_rules(tuple()) + customized_rules
            params_pspec = get_params_pspec(sharding_rules, abs_var_collect)
            inputs_pspec = jax.sharding.PartitionSpec(DEVICE_DP_AXIS, None)
            masks_pspec = jax.sharding.PartitionSpec(DEVICE_DP_AXIS, None, None, None)

            in_shardings = (None, inputs_pspec, masks_pspec)
            out_shardings = {
                key: params_pspec if key is PARAMS_KEY else None for key in abs_var_collect
            }
            pjit_encoder_init = pjit(encoder.init, in_shardings, out_shardings)
            var_collect = pjit_encoder_init(init_rngs, inputs, masks)

            optimizer = optax.adamw(args.lr)
            var_collect, params = flax.core.pop(var_collect, PARAMS_KEY)
            state = train_state.TrainState.create(
                apply_fn=encoder.apply, params=params, tx=optimizer
            )
            state_pspec = get_state_pspec(state, params_pspec)
            labels_pspec = jax.sharding.PartitionSpec(
                DEVICE_DP_AXIS,
            )

            in_shardings = (state_pspec, inputs_pspec, masks_pspec, labels_pspec, None, None)
            out_shardings = (state_pspec, None, None, None)
            pjit_train_step = pjit(train_step, in_shardings, out_shardings)

            in_shardings = (state_pspec, inputs_pspec, masks_pspec, labels_pspec, None)
            out_shardings = (None, None)
            pjit_eval_step = pjit(eval_step, in_shardings, out_shardings)

            if args.use_fp8:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                check_fp8(state, var_collect, inputs, masks, labels)

            if args.dry_run:
                labels = jnp.zeros(label_shape, dtype=jnp.bfloat16)
                rngs = {DROPOUT_KEY: dropout_rng}
                pjit_train_step(state, inputs, masks, labels, var_collect, rngs)
                print("PASSED")
            else:
                for epoch in range(1, args.epochs + 1):
                    rng, input_rng = jax.random.split(rng)
                    rng, dropout_rng = jax.random.split(rng)
                    rngs = {INPUT_KEY: input_rng, DROPOUT_KEY: dropout_rng}

                    state, train_loss, train_accuracy, var_collect = train_epoch(
                        state,
                        train_ds,
                        args.batch_size,
                        rngs,
                        var_collect,
                        pjit_train_step,
                        shard_mesh,
                        inputs_pspec,
                        masks_pspec,
                        labels_pspec,
                    )

                    test_loss, test_accuracy = eval_model(
                        state,
                        test_ds,
                        args.test_batch_size,
                        var_collect,
                        pjit_eval_step,
                        shard_mesh,
                        inputs_pspec,
                        masks_pspec,
                        labels_pspec,
                    )
                    if args.process_id == 0:
                        print(
                            f"Epoch: {epoch:>2} "
                            f"Train Loss: {train_loss:.6f} "
                            f"Train Accuracy: {train_accuracy:.6f} "
                            f"Test Loss: {test_loss:.6f} "
                            f"Test Accuracy: {test_accuracy:.6f} "
                        )

    jax.distributed.shutdown()
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
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        default=False,
        help="Use FP8 for inference and training without recalibration",
    )
    parser.add_argument(
        "--coordinator-address",
        type=str,
        default="127.0.0.1:1234",
        help=(
            "the IP address of process 0 and a port on                              which that"
            " process should launch a coordinator service                              (default:"
            " 127.0.0.1:1234)"
        ),
    )
    parser.add_argument(
        "--num-process", type=int, default=1, help="number of processes (default: 1)"
    )
    parser.add_argument(
        "--process-id",
        type=int,
        default=0,
        help="the ID number of the current process (default: 0)",
    )

    return parser.parse_args(args)


def query_gpu(q):
    """Query GPU info on the system"""
    gpu_has_fp8, reason = te.fp8.is_fp8_available()
    num_gpu = len(jax.devices())
    q.put([num_gpu, gpu_has_fp8, reason])


def unittest_query_gpu():
    r"""
    It is only used by TestEncoder.
    The `jax.distributed.initialize` must be called before any other JAX or Flax API,
    otherwise `jax.local_devices` will be incorrect.
    Thus, fork another process to query number of GPUs and FP8 capability.
    """
    q = mp.Queue()
    p = mp.Process(target=query_gpu, args=(q,))
    p.start()
    num_gpu, gpu_has_fp8, reason = q.get()
    p.join()
    return num_gpu, gpu_has_fp8, reason


class TestEncoder(unittest.TestCase):
    """Encoder unittests"""

    num_gpu, gpu_has_fp8, reason = unittest_query_gpu()

    def exec(self, use_fp8):
        """Run 3 epochs for testing"""
        num_gpu = self.num_gpu
        tp_size = 2 if num_gpu > 1 and num_gpu % 2 == 0 else 1
        dp_size = num_gpu // tp_size
        batch_size = 64 // dp_size

        arg_list = []
        for i in range(num_gpu):
            args = encoder_parser([])
            args.num_process = num_gpu
            args.use_fp8 = use_fp8
            args.batch_size = batch_size
            args.test_batch_size = batch_size
            args.process_id = i
            arg_list.append(args)

        with mp.Pool(self.num_gpu) as p:
            results = p.map(train_and_evaluate, arg_list)

        return results

    def test_te_bf16(self):
        """Test Transformer Engine with BF16"""
        results = self.exec(False)
        actual = results[0]
        assert actual[0] < 0.45 and actual[1] > 0.79

    @unittest.skipIf(not gpu_has_fp8, reason)
    def test_te_fp8(self):
        """Test Transformer Engine with FP8"""
        results = self.exec(True)
        actual = results[0]
        assert actual[0] < 0.45 and actual[1] > 0.79


if __name__ == "__main__":
    train_and_evaluate(encoder_parser(None))
