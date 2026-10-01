# Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""DSv4's trailing-channel, interleaved RoPE for full-sequence attention."""

import torch


def rotary_embeddings(seq, ratio, width, theta, device):
    """Return token and compressed-window (cos, sin) pairs in FP32."""
    inv_freq = theta ** (-torch.arange(0, width, 2, device=device, dtype=torch.float32) / width)
    angles = torch.outer(torch.arange(seq, device=device, dtype=torch.float32), inv_freq)
    cos = angles.cos().repeat_interleave(2, dim=-1)[None, :, None, :]
    sin = angles.sin().repeat_interleave(2, dim=-1)[None, :, None, :]
    window_positions = torch.arange(seq // ratio, device=device) * ratio
    return (cos, sin), (cos[:, window_positions], sin[:, window_positions])


class _DSv4RotaryEmbedding(torch.nn.Module):
    """Cache shared FP32 token frequencies; compressed positions are window starts."""

    def __init__(self, ratio, width, theta, device, max_seqlen=None):
        super().__init__()
        self.ratio, self.width, self.theta = ratio, width, theta
        cos, sin = (None, None)
        if max_seqlen is not None:
            (cos, sin), _ = rotary_embeddings(max_seqlen, ratio, width, theta, device)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _apply(self, fn):
        cos, sin = self.cos, self.sin
        super()._apply(fn)
        if cos is not None:
            # Module.to(bfloat16) must not round the reusable frequencies.
            self.cos = cos.to(device=self.cos.device)
            self.sin = sin.to(device=self.sin.device)
        return self

    def forward(self, seq, device):
        """Return cached token and compressed-window frequency pairs."""
        if self.cos is None or self.cos.shape[1] < seq:
            (self.cos, self.sin), _ = rotary_embeddings(
                seq, self.ratio, self.width, self.theta, device
            )
        n_comp = seq // self.ratio
        return (self.cos[:, :seq], self.sin[:, :seq]), (
            self.cos[:, : n_comp * self.ratio : self.ratio],
            self.sin[:, : n_comp * self.ratio : self.ratio],
        )


def apply_rotary(x, cos, sin):
    """Rotate trailing interleaved pairs in FP32, then restore input dtype."""
    width = cos.shape[-1]
    tail = x[..., -width:]
    pair = torch.stack((-tail[..., 1::2], tail[..., 0::2]), -1).flatten(-2)
    rotated = (tail.float() * cos + pair.float() * sin).to(x.dtype)
    return torch.cat((x[..., :-width], rotated), -1)
