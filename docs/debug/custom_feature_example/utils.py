# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import time
import torch
import transformer_engine.pytorch as te
import nvdlfw_inspect.api as debug_api
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/{}".format(time.time()))


def init_model() -> torch.nn.Module:
    return te.TransformerLayer(
        hidden_size=1024,
        ffn_hidden_size=1024,
        num_attention_heads=16,
    )


def run_example_fit(model: torch.nn.Module):
    output_tensor_ref = torch.randn(1, 1, 1024).cuda()
    input_tensor = torch.randn(1, 1, 1024).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    for i in range(1000):
        output = model(input_tensor)
        loss = torch.nn.functional.mse_loss(output, output_tensor_ref)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        get_tb_writer().add_scalar("Loss", loss.item(), i)

        debug_api.step()


def get_tb_writer():
    return writer
