# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TransformerLayer encoder recompute"""

import sys
import paddle
import transformer_engine.paddle as te


class Net(paddle.nn.Layer):
    """Network use for recompute testing"""

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    def forward(self, inp, mask, enable_recompute, use_reentrant):
        for layer in self.layers:
            if enable_recompute:
                out = te.recompute(layer, inp, mask, use_reentrant=use_reentrant)
            else:
                out = layer(inp, mask)
        return out


def main():
    """Main function"""
    paddle.seed(10)
    batch_size = 16
    hidden_size = 4096
    num_heads = 32
    ffn_hidden_size = 16384
    q_seqlen = 512
    kv_seqlen = 512
    num_layers = 4
    enable_recompute = int(sys.argv[1])
    use_reentrant = int(sys.argv[2])

    layers = paddle.nn.LayerList(
        [
            te.TransformerLayer(
                hidden_size,
                ffn_hidden_size,
                num_heads,
                layer_type="encoder",
            )
            for _ in range(num_layers)
        ]
    )
    model = Net(layers)

    optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

    for _ in range(10):
        inp = paddle.uniform([batch_size, q_seqlen, hidden_size])
        inp.stop_gradient = False
        mask = paddle.zeros(shape=(batch_size, 1, q_seqlen, kv_seqlen), dtype="bool")
        with te.fp8_autocast(enabled=True):
            out = model(inp, mask, enable_recompute, use_reentrant)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    print("Loss: ", float(loss))
    print("Peak memory: ", paddle.device.cuda.max_memory_allocated(0))


if __name__ == "__main__":
    main()
