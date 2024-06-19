# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.
import operator
import re
from functools import reduce

import jax
from jax.experimental.pjit import pjit, _UNSPECIFIED

from transformer_engine.jax.sharding import MeshResource

from utils import assert_allclose, is_devices_enough


def generate_configs():
    configs = []
    if is_devices_enough(2):
        configs.append([2, (2,), "dp", MeshResource(dp_resource="dp")])
        configs.append([2, (2,), "tp", MeshResource(tp_resource="tp")])

    if is_devices_enough(4):
        TP_size = 2
        DP_size = 2
        configs.append(
            [4, (DP_size, TP_size), ("dp", "tp"), MeshResource(dp_resource="dp", tp_resource="tp")]
        )

    return configs


COLL_AR_KEY = "all-reduce"
COLL_AG_KEY = "all-gather"
COLL_OTHER_KEY = "other"


def generate_collectives_count(allreduce, allgather, other):
    return {COLL_AR_KEY: allreduce, COLL_AG_KEY: allgather, COLL_OTHER_KEY: other}


def assert_equal_collectives(target_hlo, coll_count_ref):
    target_splitted_hlo = target_hlo.splitlines()
    start_symb = "-start"

    def count_bytes(hlo_text):
        bytes_count = 0

        def get_bytes_per_txt(t):
            """
            The pattern of t would be like:
                'f32[]',
                '(f32[1024]{0}',
                'f32[1024]{0})',
                'f8E4M3FN[1024]{0}',
                'i32[1024]{0}',
                'bf16[1024,1024]{0}'
            """
            match = re.search(r"(i|f)(\d+).*\[([0-9,]*)\]", t)
            _, bits_of_type, shape = match.groups()
            bytes_of_type = int(bits_of_type) // 8
            if shape == "":
                num_of_elements = 1
            else:
                num_of_elements = reduce(operator.mul, map(int, shape.split(",")))

            return bytes_of_type * num_of_elements

        # ['xxx-start', '=', '(bf16[xxx]', 'bf16[xxx])', 'xxx-start(', ...]
        if "(" in hlo_text[2]:
            for txt in hlo_text[2:]:
                bytes_count += get_bytes_per_txt(txt)
                if ")" in txt:
                    break
        else:  # ['xxx-start', '=', 'fp32[]', 'xxx-start(', ...]
            bytes_count = get_bytes_per_txt(hlo_text[2])

        return bytes_count

    def count_collectives(splitted_hlo):
        result = generate_collectives_count(0, 0, 0)

        for line in splitted_hlo:
            txt = line.split()
            if len(txt) > 0 and start_symb in txt[0]:
                if COLL_AR_KEY in txt[0]:
                    result[COLL_AR_KEY] += count_bytes(txt)
                elif COLL_AG_KEY in txt[0]:
                    result[COLL_AG_KEY] += count_bytes(txt)
                else:
                    result[COLL_OTHER_KEY] += count_bytes(txt)
        return result

    target_result = count_collectives(target_splitted_hlo)
    assert (
        target_result == coll_count_ref
    ), f"Expected collective count is {coll_count_ref}, but got {target_result}."


def compare_ops(
    target_func,
    ref_func,
    inputs,
    coll_count_ref,
    *,
    grad_args=None,
    metric_fwd_dtype=None,
    metric_bwd_dtype=None,
    in_shardings=_UNSPECIFIED,
    out_shardings=_UNSPECIFIED,
    **kwargs,
):
    assert len(inputs) >= 1

    if metric_fwd_dtype is None:
        metric_fwd_dtype = inputs[0].dtype
    if metric_bwd_dtype is None:
        metric_bwd_dtype = inputs[0].dtype

    if grad_args is None:
        grad_args = tuple(range(len(inputs)))

    target_grad_func = jax.value_and_grad(target_func, argnums=grad_args)
    target_pjitter = pjit(target_grad_func, in_shardings=in_shardings, out_shardings=out_shardings)
    target_fwd, target_grads = target_pjitter(*inputs, **kwargs)
    target_hlo = target_pjitter.lower(*inputs, **kwargs).compile().as_text()

    ref_grad_func = jax.value_and_grad(ref_func, argnums=grad_args)
    ref_pjitter = pjit(ref_grad_func, in_shardings=in_shardings, out_shardings=out_shardings)
    ref_fwd, ref_grads = ref_pjitter(*inputs, **kwargs)

    assert_allclose(target_fwd, ref_fwd, dtype=metric_fwd_dtype)

    for i in range(len(target_grads)):
        assert_allclose(target_grads[i], ref_grads[i], dtype=metric_bwd_dtype)

    assert_equal_collectives(target_hlo, coll_count_ref)
