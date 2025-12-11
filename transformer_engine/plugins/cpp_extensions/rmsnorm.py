# Copyright (c) 2025, BAAI. All rights reserved.
#
# See LICENSE for license information.

import os
import torch
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ..import_utils import safety_import, have_flag_gems

### RMSNORM
HAVE_FLAG_GEMS = have_flag_gems()

if HAVE_FLAG_GEMS:
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
    assert HAVE_FLAG_GEMS, "GEMS is not installed"
    y, rstdevs = flag_gems.rms_norm_forward(
        input,
        [input.shape[-1]],
        weight,
        eps,
    )
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
    assert HAVE_FLAG_GEMS, "GEMS is not installed"
    dx, dw = flag_gems.rms_norm_backward(
        dy,
        x,
        rsigma,
        [x.shape[-1]],
        gamma,
        eps,
    )
    return dx, dw
