"""Encoder training on single GPU"""
import argparse
from functools import partial
import time

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F
from paddle.device.cuda.cuda_graphed_layer import CUDAGraphedLayer
from paddlenlp.data import Dict, Pad, Stack
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import BertTokenizer
import transformer_engine.paddle as te
from transformer_engine.paddle.fp8 import get_global_fp8_state

paddle.seed(102)
np.random.seed(102)
paddle.set_flags({
    "FLAGS_cudnn_deterministic": True,
})

num_embed = 30528
hidden_size = 1024
num_heads = 16
intermediate_size = 4096
num_encoder_layers = 12


class EncoderLayers(nn.Layer):
    """Transformer Encoder"""

    def __init__(self, use_te=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.num_layers = num_encoder_layers

        backend = "transformer_engine"
        if use_te is False:
            backend = "paddle"

        self.encoder = nn.LayerList([
            te.TransformerLayer(self.hidden_size,
                                self.intermediate_size,
                                self.num_heads,
                                layernorm_epsilon=1e-5,
                                hidden_dropout=0.1,
                                attention_dropout=0.1,
                                self_attn_mask_type="padding",
                                apply_residual_connection_post_layernorm=False,
                                output_layernorm=True,
                                layer_type="encoder",
                                backend=backend,
                                assume_static_shape=True) for _ in range(self.num_layers)
        ])

    def forward(self, x, mask):
        for layer in self.encoder:
            x = layer(x, mask)
        return x


class Net(nn.Layer):
    """NLP Encoder"""

    num_embed: int

    def __init__(self, max_seqlen, use_te=False, use_cuda_graph=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embed = num_embed
        self.use_te = use_te
        self.num_cuda_graph_warmup_steps = 3

        self.embedding = nn.Embedding(num_embeddings=self.num_embed, embedding_dim=self.hidden_size)

        self.encoder_layer = EncoderLayers(use_te)
        if use_cuda_graph:
            self.encoder = CUDAGraphedLayer(self.encoder_layer, self.num_cuda_graph_warmup_steps)
        else:
            self.encoder = self.encoder_layer
        self.linear1 = nn.Linear(self.hidden_size * max_seqlen, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 2)

    def forward(self, x, mask):
        x = self.embedding(x)
        x = paddle.cast(x, dtype="bfloat16")
        x = self.encoder(x, mask)
        x = x.reshape((x.shape[0], -1))
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


def pad2d(batch_size, tensor):
    """Padding 2d tensor to static shape [batch_size, ]"""

    dim = tensor.shape[0]
    if dim < batch_size:
        pad = paddle.zeros([batch_size - dim, tensor.shape[1]], dtype=tensor.dtype)
        tensor = paddle.concat(x=[tensor, pad])
    return tensor


def train(args, model, train_data_loader, optimizer, epoch):
    """Training function."""
    model.train()
    for batch_id, (data, mask, labels) in enumerate(train_data_loader):
        data = pad2d(args.batch_size, data)
        labels = pad2d(args.batch_size, labels)

        with paddle.amp.auto_cast(dtype="bfloat16", level="O2"):    # pylint: disable=not-context-manager
            with te.fp8_autocast(enabled=args.use_fp8):
                outputs = model(data, mask)

            loss = F.cross_entropy(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.clear_gradients()

        if args.use_cuda_graph:
            global_fp8_state = get_global_fp8_state()
            global_fp8_bwd_buffer = global_fp8_state.get_fp8_bwd_buffer()
            fp8_meta, tp_group, tp_size = global_fp8_state.get_first_module_state()
            global_fp8_bwd_buffer.finalize(fp8_meta, tp_group, tp_size)

        if batch_id % args.log_interval == 0:
            print(f"Train Epoch: {epoch} "
                  f"[{batch_id * len(data)}/{len(train_data_loader.dataset)} "
                  f"({100. * batch_id / len(train_data_loader):.0f}%)]\t"
                  f"Loss: {loss.item():.6f}")
            if args.dry_run:
                return loss.item()

    return loss.item()


def convert_example(example, tokenizer, max_length=128):
    """convert example"""
    labels = np.array([example["labels"]], dtype="int64")
    example = tokenizer(
        example["sentence"],
        padding="max_length",
        max_length=max_length,
        return_token_type_ids=False,
    )
    mask = np.zeros((1, max_length, max_length), dtype="bool")
    input_ids = example["input_ids"]
    return {
        "input_ids": input_ids,
        "mask": mask,
        "labels": labels,
    }


def load_data(batch_size, max_seqlen, tokenizer):
    """create dataloader"""
    train_ds = load_dataset("glue", "cola", splits="train")
    validation_ds = load_dataset("glue", "cola", splits="dev")

    trans_func = partial(convert_example, tokenizer=tokenizer, max_length=max_seqlen)
    train_ds = train_ds.map(trans_func, lazy=True)
    validation_ds = validation_ds.map(trans_func, lazy=True)

    train_sampler = paddle.io.BatchSampler(train_ds, batch_size=batch_size, shuffle=True)
    validation_sampler = paddle.io.BatchSampler(validation_ds, batch_size=batch_size, shuffle=False)

    return train_ds, validation_ds, train_sampler, validation_sampler


def encoder_parser(args):
    """Training settings."""

    max_seqlen = 512
    batch_size = 16

    parser = argparse.ArgumentParser(description="Paddle Encoder Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=batch_size,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=batch_size,
        metavar="N",
        help="input batch size for testing (default: 64)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=max_seqlen,
        metavar="N",
        help="maximum sequence length (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
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
    parser.add_argument("--use-te",
                        action="store_true",
                        default=False,
                        help="Use Transformer Engine")
    parser.add_argument("--seed", type=int, default=0, metavar="S", help="random seed (default: 0)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        default=False,
        help="Use FP8 for inference and training without recalibration",
    )
    parser.add_argument(
        "--use-cuda-graph",
        action="store_true",
        default=False,
        help="Use cudaGraph for training",
    )

    return parser.parse_args(args)


def train_and_evaluate(args):
    """Execute model training and evaluation loop."""
    print(args)
    # train_ds, test_ds, num_embed = get_datasets(args.max_seq_len)
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    batchify_fn = lambda samples, fn=Dict({
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "mask": Pad(axis=0),
        "labels": Stack(dtype="int64"),
    }): fn(samples)
    train_ds, _, train_sampler, _ = load_data(args.batch_size, args.max_seq_len, tokenizer)

    train_data_loader = paddle.io.DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=batchify_fn,
    )

    model = Net(args.max_seqlen, args.use_te, args.use_cuda_graph)

    optimizer = paddle.optimizer.Adam(learning_rate=args.lr, parameters=model.parameters())
    model = paddle.amp.decorate(models=model, level="O2", dtype="bfloat16")

    for epoch in range(1, args.epochs + 1):
        begin = time.time()
        loss = train(args, model, train_data_loader, optimizer, epoch)
        end = time.time()
        print(f"Train Epoch Elapsed Time: {end - begin}")

    return loss


if __name__ == "__main__":
    paddle.fluid.core.nvprof_enable_record_event()
    train_and_evaluate(encoder_parser(None))
