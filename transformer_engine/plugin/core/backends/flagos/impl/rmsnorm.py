# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import torch
import flag_gems


def rmsnorm_fwd_fl(
    input,
    weight,
    eps,
    ln_out,
    quantizer,
    odtype,
    sm_margin,
    zero_centered_gamma,
):
    with flag_gems.use_gems():
        if zero_centered_gamma:
            weight_adj = 1 + weight
        else:
            weight_adj = weight

        y, rstdevs = flag_gems.rms_norm_forward(
            input,
            [input.shape[-1]],
            weight_adj,
            eps,
        )

        if rstdevs.shape != input.shape[:-1]:
            rstdevs = rstdevs.view(input.shape[:-1])

        return y, None, rstdevs


def rmsnorm_bwd_fl(
    dy,
    x,
    rsigma,
    gamma,
    sm_margin,
    zero_centered_gamma,
    eps,
):
    with flag_gems.use_gems():
        # When zero_centered_gamma is True, forward uses (1 + gamma) as weight
        # So backward needs to use (1 + gamma) for computing dx
        if zero_centered_gamma:
            gamma_adj = 1 + gamma
        else:
            gamma_adj = gamma

        dx, dw = flag_gems.rms_norm_backward(
            dy,
            x,
            rsigma,
            [x.shape[-1]],
            gamma_adj,
            eps,
        )
        return dx, dw
