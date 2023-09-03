# Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
"""Test TE Paddle Recompute"""
import os
import re
import subprocess

import numpy as np
import pytest

from transformer_engine.paddle.fp8 import is_fp8_available

is_fp8_supported, reason = is_fp8_available()


@pytest.mark.skipif(not is_fp8_supported, reason=reason)
@pytest.mark.parametrize('use_reentrant', [False, True])
def test_transformer_encoder_recompute(use_reentrant):
    """
    Test TransformerLayer encoder recompute
    """
    rtol = 1e-5
    atol = 1e-5

    def launch_subprocess_and_check_output(enable_recompute):
        """Launch training in subprocess and check output"""
        try:
            script_content = """
import sys
import paddle
import transformer_engine.paddle as te

class Net(paddle.nn.Layer):
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

    layers = paddle.nn.LayerList([te.TransformerLayer(
        hidden_size,
        ffn_hidden_size,
        num_heads,
        layer_type='encoder',
    ) for _ in range(num_layers)])
    model = Net(layers)

    optimizer = paddle.optimizer.AdamW(learning_rate=0.001, parameters=model.parameters())

    for _ in range(10):
        inp = paddle.uniform([batch_size, q_seqlen, hidden_size])
        inp.stop_gradient=False
        mask = paddle.zeros(shape=(batch_size, 1, q_seqlen, kv_seqlen), dtype='bool')
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
"""
            # Create 'script.py' file
            with open('script.py', 'w', encoding="utf8") as script_file:
                script_file.write(script_content)

            # Launch the subprocess and capture its output
            result = subprocess.check_output(
                ['python', 'script.py',
                 str(int(enable_recompute)),
                 str(int(use_reentrant))],
                stderr=subprocess.STDOUT,
                universal_newlines=True)

            # Print the output
            print(result)

            loss_match = re.search(r'Loss:\s+(-?\d+\.\d+)', result)
            memory_match = re.search(r'Peak memory:\s+(\d+)', result)

            loss_value = float(loss_match.group(1))
            memory_value = int(memory_match.group(1))

            # Return the return code of the subprocess (0 indicates success)
            return loss_value, memory_value

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Subprocess failed with error: {e}") from e
        finally:
            os.remove('script.py')

    loss_recompute, peak_memory_recompute = launch_subprocess_and_check_output(True)
    loss_ref, peak_memory_ref = launch_subprocess_and_check_output(False)

    assert peak_memory_recompute < peak_memory_ref
    np.testing.assert_allclose(loss_recompute, loss_ref, rtol=rtol, atol=atol)
