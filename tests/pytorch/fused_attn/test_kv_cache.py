# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

from collections import OrderedDict
from typing import List
import os
import logging
import math

import pytest
import torch

from test_fused_attn import (
    ModelConfig,
    reset_rng_states,
    _get_attention_backends,
)

from torch.distributions import Exponential
from transformer_engine.pytorch import make_graphed_callables
from transformer_engine.common import recipe
from transformer_engine.pytorch import fp8_autocast, fp8_model_init
from transformer_engine.pytorch.transformer import (
    TransformerLayer,
)
from transformer_engine.pytorch.attention import DotProductAttention, InferenceParams
from transformer_engine.pytorch.attention.dot_product_attention.utils import (
    FlashAttentionUtils as fa_utils,
)
from transformer_engine.pytorch.utils import (
    init_method_normal,
    scaled_init_method_normal,
    is_bf16_compatible,
)

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()


param_types = [torch.float16]
if is_bf16_compatible():
    param_types.append(torch.bfloat16)

model_configs_infer = {
    # test: b,  h, hg,  d,  sq, skv,   p,      mask,      bias
    "infer_0": ModelConfig(
        4, 16, 16, 128, 64, 64, 0.0, "no_mask", "no_bias", total_requests=8, max_ctx_len=16
    ),
    "infer_1": ModelConfig(
        2, 16, 4, 64, 66, 66, 0.0, "no_mask", "no_bias", total_requests=6, max_ctx_len=16
    ),
}

qkv_formats = ["bshd", "sbhd", "thd"]


def to_pretty_string(x: torch.Tensor):
    return "[" + ",".join(["{:>3s}".format(str(i)) for i in x.tolist()]) + "]"


def round_up(a: int, b: int):
    return b * math.ceil(a / b)


class Simulation:
    def __init__(
        self,
        total_requests: int = 10,
        max_seq_len: int = 1024,
        max_ctx_len: int = 128,
        max_batch_size: int = 5,
        poisson_rate: float = 1,
    ):
        self.total_requests = total_requests
        self.max_seq_len = max_seq_len
        self.max_batch_size = max_batch_size
        self.poisson_rate = poisson_rate

        # calculate maximum context/generation length
        self.max_ctx_len = max_ctx_len
        self.max_gen_len = max_seq_len - self.max_ctx_len

        # simulate sequence ids in monotonically increasing fashion
        self.seq_ids = torch.range(0, total_requests - 1, dtype=torch.int32, device="cpu")

        # simulate context lengths in Uniform distribution
        self.context_lens = torch.randint(
            1, self.max_ctx_len, [total_requests], dtype=torch.int32, device="cpu"
        )

        # simulate gen lengths in Exponential distribution
        gen_dist = Exponential(1 / self.max_gen_len)
        gen_lens = gen_dist.sample((total_requests,))
        gen_lens = torch.where(gen_lens > self.max_gen_len, self.max_gen_len, gen_lens).to(
            dtype=torch.int32, device="cpu"
        )
        self.gen_lens = torch.where(gen_lens == 0, 1, gen_lens).to(dtype=torch.int32, device="cpu")

        # simulate arrival times in Poisson distribution
        if poisson_rate is None:
            self.poisson_rate = torch.randint(1, max_batch_size, [1]).item()
        interval_dist = Exponential(self.poisson_rate)
        arrival_intervals = interval_dist.sample((total_requests,))
        self.arrival_times = torch.cumsum(arrival_intervals, dim=0).to(
            dtype=torch.int32, device="cpu"
        )
        self.last_arrival = self.arrival_times.max().item()

        # initialize tensors
        self.reset()

    def reset(self):
        self.t = 0
        self.request_delays = torch.zeros([self.total_requests], dtype=torch.int32, device="cpu")
        self.delayed_seq_ids = torch.Tensor().to(dtype=torch.int32, device="cpu")
        self.serving_times = self.arrival_times
        self.complete_times = self.arrival_times

        # batch info at step t
        self.t_seq_ids = torch.Tensor([]).to(dtype=torch.bool, device="cpu")
        self.t_ctx_lens = torch.Tensor([]).to(dtype=torch.bool, device="cpu")
        self.t_gen_lens = torch.Tensor([]).to(dtype=torch.bool, device="cpu")
        self.t_total_lens = self.t_ctx_lens + self.t_gen_lens
        self.t_batch_size = 0

        # step info from step t-1 to t
        self.step_lens = torch.Tensor([]).to(dtype=torch.int32, device="cpu")

    def print_setup(self, logger):
        logger.info("Simulation:")
        logger.info("  {:<31s}: {}".format("total number of requests", self.total_requests))
        logger.info("  {:<31s}: {}".format("max sequence length per request", self.max_seq_len))
        logger.info("  {:<31s}: {}".format("max context length", self.max_ctx_len))
        logger.info("  {:<31s}: {}".format("max generation length", self.max_gen_len))
        logger.info("  {:<31s}: {}".format("max batch size per iteration", self.max_batch_size))
        logger.info("  {:<31s}: {}".format("Poisson rate", self.poisson_rate))
        logger.info("  {:<17s}: {}".format("sequence ids", to_pretty_string(self.seq_ids)))
        logger.info("  {:<17s}: {}".format("arrival times", to_pretty_string(self.arrival_times)))
        logger.info("  {:<17s}: {}".format("context lengths", to_pretty_string(self.context_lens)))
        logger.info("  {:<17s}: {}".format("generation lengths", to_pretty_string(self.gen_lens)))

    def print_step(self, logger):
        logger.info(f"Step t = {self.t}:")
        logger.info("  {:<15s}: {}".format("t_batch_size", self.t_batch_size))
        logger.info("  {:<15s}: {}".format("t_seq_ids", self.t_seq_ids.tolist()))
        logger.info("  {:<15s}: {}".format("t_ctx_lens", self.t_ctx_lens.tolist()))
        logger.info("  {:<15s}: {}".format("t_gen_lens", self.t_gen_lens.tolist()))
        logger.info("  {:<15s}: {}".format("t_total_lens", self.t_total_lens.tolist()))
        logger.info("  {:<15s}: {}".format("step_lens", self.step_lens.tolist()))

    def print_summary(self, logger):
        logger.info("Summary:")
        logger.info("  {:<18s}: {}".format("total steps taken", self.t))
        logger.info("  {:<18s}: {}".format("arrival_times", to_pretty_string(self.arrival_times)))
        logger.info("  {:<18s}: {}".format("serving_times", to_pretty_string(self.serving_times)))
        logger.info("  {:<18s}: {}".format("total_gen_lens", to_pretty_string(self.gen_lens)))
        logger.info("  {:<18s}: {}".format("complete_times", to_pretty_string(self.complete_times)))

    def add_new_seqs(self, new_seq_ids):
        # get ctx_lens for new seqs
        self.t_seq_ids = torch.cat([self.t_seq_ids, new_seq_ids], dim=0)
        self.t_ctx_lens = torch.cat([self.t_ctx_lens, self.context_lens[new_seq_ids]], dim=0)
        gen_lens = torch.Tensor([0] * len(new_seq_ids)).to(dtype=torch.int32, device="cpu")
        self.t_gen_lens = torch.cat([self.t_gen_lens, gen_lens], dim=0)

        # append new seqs' ctx_lens to step_lens
        self.step_lens = torch.cat([self.step_lens, self.context_lens[new_seq_ids]], dim=0)

    def remove_finished(self):
        # figure out which seqs have finished
        finished = torch.where(self.t_gen_lens - self.gen_lens[self.t_seq_ids] < 0, False, True).to(
            dtype=torch.bool, device="cpu"
        )
        self.t_seq_ids = self.t_seq_ids[~finished]
        self.t_ctx_lens = self.t_ctx_lens[~finished]
        self.t_gen_lens = self.t_gen_lens[~finished]

        # add ones for unfinished seqs to step_lens
        self.step_lens = torch.ones([len(self.t_seq_ids)], dtype=torch.int32, device="cpu")

    def step(self, dynamic_fill: bool = True):
        # remove finished seqs
        if self.t != 0:
            self.remove_finished()

        # get allowed new seqs
        arrived_seq_ids = torch.where(self.arrival_times == self.t, True, False).nonzero().view(-1)
        queuing_seq_ids = torch.cat([self.delayed_seq_ids, arrived_seq_ids], dim=0)
        if dynamic_fill:
            allowed_num_new_seqs = self.max_batch_size - len(self.t_seq_ids)
        else:
            allowed_num_new_seqs = 0 if len(self.t_seq_ids) else self.max_batch_size
        if len(queuing_seq_ids) > allowed_num_new_seqs:
            new_seq_ids = queuing_seq_ids[:allowed_num_new_seqs]
            self.delayed_seq_ids = queuing_seq_ids[allowed_num_new_seqs:]
            self.request_delays[self.delayed_seq_ids.tolist()] += 1
        else:
            new_seq_ids = queuing_seq_ids
            self.delayed_seq_ids = torch.Tensor().to(dtype=torch.int32)

        # add new seqs to batch
        self.add_new_seqs(new_seq_ids)

        # update batch variables
        self.t_batch_size = len(self.t_seq_ids)
        self.t_total_lens = self.t_ctx_lens + self.t_gen_lens


def get_model(
    module: torch.nn.Module,
    config: ModelConfig,
    dtype: torch.dtype,
    backend: str = "FusedAttention",
    qkv_format: str = "bshd",
    num_layers: int = 1,
    mode: str = "reference",
    is_fp8: bool = False,
):
    reset_rng_states()
    sigma = 0.023
    init_method = init_method_normal(sigma)
    output_layer_init_method = scaled_init_method_normal(sigma, num_layers)

    if mode == "reference":
        attn_mask_type = "causal"
        qkv_format = "bshd"
    if mode == "inference":
        attn_mask_type = "padding_causal"

    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=is_fp8,
        fp8_mha=False,
    )

    if module == "TransformerLayer":
        hidden_size = config.head_dim_qk * config.num_heads
        with fp8_model_init(enabled=is_fp8, recipe=fp8_recipe):
            model = [
                TransformerLayer(
                    hidden_size=hidden_size,
                    ffn_hidden_size=4 * hidden_size,
                    num_attention_heads=config.num_heads,
                    num_gqa_groups=config.num_gqa_groups,
                    hidden_dropout=0.0,
                    attention_dropout=config.dropout_p,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layer_number=layer_number,
                    kv_channels=config.head_dim_qk,
                    self_attn_mask_type=attn_mask_type,
                    fuse_qkv_params=False,
                    params_dtype=dtype,
                    attn_input_format=qkv_format,
                )
                .cuda()
                .eval()
                for layer_number in range(1, num_layers + 1)
            ]
    if module == "DotProductAttention":
        with fp8_model_init(enabled=is_fp8, recipe=fp8_recipe):
            model = [
                DotProductAttention(
                    kv_channels=config.head_dim_qk,
                    num_attention_heads=config.num_heads,
                    num_gqa_groups=config.num_gqa_groups,
                    layer_number=layer_number,
                    attention_dropout=config.dropout_p,
                    qkv_format=qkv_format,
                    attn_mask_type=attn_mask_type,
                )
                .cuda()
                .eval()
                for layer_number in range(1, num_layers + 1)
            ]
    return model


def generate_args(
    module: torch.nn.Module,
    config: ModelConfig,
    dtype: torch.dtype,
    qkv_format: str = "bshd",
    mode: str = "full_inputs",
):
    # full inputs used as reference
    if mode == "full_inputs":
        warmup = False
        shapes = []
        if module == "TransformerLayer":
            shapes.append(
                [config.total_requests, config.max_seqlen_kv, config.num_heads * config.head_dim_qk]
            )
        if module == "DotProductAttention":
            shapes.append(
                [config.total_requests, config.max_seqlen_kv, config.num_heads, config.head_dim_qk]
            )
            shapes.append(
                [
                    config.total_requests,
                    config.max_seqlen_kv,
                    config.num_gqa_groups,
                    config.head_dim_qk,
                ]
            )
            shapes.append(
                [
                    config.total_requests,
                    config.max_seqlen_kv,
                    config.num_gqa_groups,
                    config.head_dim_v,
                ]
            )
    # sample args used for cuda graph warmup
    elif mode == "sample_args":
        warmup = True
        shapes = []
        if qkv_format == "bshd":
            shape = [config.batch_size, config.max_ctx_len]
        if qkv_format == "sbhd":
            shape = [config.max_ctx_len, config.batch_size]
        if qkv_format == "thd":
            shape = [config.batch_size * config.max_ctx_len]
        if module == "TransformerLayer":
            shapes.append([*shape, config.num_heads * config.head_dim_qk])
        if module == "DotProductAttention":
            shapes.append([*shape, config.num_heads, config.head_dim_qk])
            shapes.append([*shape, config.num_gqa_groups, config.head_dim_qk])
            shapes.append([*shape, config.num_gqa_groups, config.head_dim_v])

    num_tensors = len(shapes)
    if warmup:
        return [
            torch.ones(
                *shapes[i],
                device="cuda",
                dtype=dtype,
            )
            for i in range(num_tensors)
        ]
    elif module == "TransformerLayer":
        return [
            0.01
            * torch.randint(
                -100,
                100,
                shapes[i],
                device="cuda",
                dtype=dtype,
            )
            for i in range(num_tensors)
        ]
    elif module == "DotProductAttention":
        return [
            0.1
            * torch.randn(
                *shapes[i],
                device="cuda",
                dtype=dtype,
            )
            for i in range(num_tensors)
        ]


def get_tols(module, backend, dtype):
    if module == "TransformerLayer":
        tols = {
            torch.half: (5e-3, 5e-3),
            torch.bfloat16: (3.5e-2, 3.5e-2),
        }
    if module == "DotProductAttention":
        tols = {
            torch.half: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-3),
            torch.float8_e4m3fn: (2e-2, 3e-2),
        }
    return tols[dtype]


@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model", model_configs_infer.keys())
@pytest.mark.parametrize("qkv_format", qkv_formats)
@pytest.mark.parametrize("is_paged", [False, True])
@pytest.mark.parametrize("backend", ["FusedAttention", "FlashAttention", "UnfusedAttention"])
@pytest.mark.parametrize("module", ["TransformerLayer", "DotProductAttention"])
@pytest.mark.parametrize("is_cuda_graph", [False, True])
@pytest.mark.parametrize("is_fp8", [False, True])
def test_kv_cache(dtype, model, qkv_format, is_paged, backend, module, is_cuda_graph, is_fp8):
    reset_rng_states()
    logger = logging.getLogger("test_kv_cache")
    fp8_recipe = recipe.DelayedScaling(
        margin=0,
        fp8_format=recipe.Format.HYBRID,
        amax_history_len=1,
        amax_compute_algo="most_recent",
        fp8_dpa=is_fp8,
        fp8_mha=False,
    )
    fp8_meta = {}
    fp8_meta["recipe"] = fp8_recipe

    config = model_configs_infer[model]
    num_layers = 2 if module == "TransformerLayer" else 1
    # flash-attn v2 requires page_size >= 256
    if backend == "FlashAttention" and not fa_utils.v3_is_installed:
        config_max_seqlen_q = config.max_seqlen_q
        config_max_seqlen_kv = config.max_seqlen_kv
        config.max_seqlen_q = 256
        config.max_seqlen_kv = 256

    # create a real-life simulation
    max_batch_size = config.batch_size
    page_size = None
    total_num_pages = None
    if is_paged:
        page_size = 256 if backend == "FlashAttention" and not fa_utils.v3_is_installed else 1
        config.max_seqlen_kv = round_up(config.max_seqlen_kv, page_size)
        total_num_pages = int(max_batch_size * config.max_seqlen_kv / page_size)
    else:
        config.max_seqlen_kv = round_up(config.max_seqlen_kv, 64)
    sim = Simulation(
        total_requests=config.total_requests,
        max_seq_len=config.max_seqlen_kv,
        max_ctx_len=config.max_ctx_len,
        max_batch_size=max_batch_size,
        poisson_rate=2,
    )
    sim.print_setup(logger)

    # initialize inference_params
    inference_params = InferenceParams(
        max_batch_size=max_batch_size,
        max_sequence_length=config.max_seqlen_kv,
        num_heads_kv=config.num_gqa_groups,
        head_dim_k=config.head_dim_qk,
        head_dim_v=config.head_dim_v,
        dtype=dtype,
        is_paged=is_paged,
        page_size=page_size,
        total_num_pages=total_num_pages,
        max_ctx_len=config.max_ctx_len,
        qkv_format=qkv_format,
    )
    if module == "DotProductAttention":
        for layer_number in range(1, num_layers + 1):
            inference_params.allocate_memory(layer_number)

    # figure out supported backends
    inference_params_qkv_format = "bshd"
    qkv_layout = qkv_format + "_" + "_".join([inference_params_qkv_format] * 2)
    if is_paged:
        qkv_layout = "paged_kv_" + qkv_layout
    available_backends, _, fused_attn_backends = _get_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=qkv_layout,
        window_size=config.window_size,
        pad_between_seqs=False,
        is_training=False,
        fp8=is_fp8,
        fp8_meta=fp8_meta,
        inference_params=inference_params,
    )
    flash_attn_supported, fused_attn_supported, unfused_attn_supported = available_backends
    if backend == "FlashAttention" and not flash_attn_supported:
        pytest.skip("FlashAttention backend is not supported")
    if backend == "FusedAttention" and not fused_attn_supported:
        pytest.skip("FusedAttention backend is not supported")
    if backend == "UnfusedAttention" and not unfused_attn_supported:
        pytest.skip("UnfusedAttention backend is not supported")
    os.environ["NVTE_FLASH_ATTN"] = str(int(backend == "FlashAttention"))
    os.environ["NVTE_FUSED_ATTN"] = str(int(backend == "FusedAttention"))
    os.environ["NVTE_UNFUSED_ATTN"] = str(int(backend == "UnfusedAttention"))
    if backend == "UnfusedAttention" and is_cuda_graph:
        pytest.skip("CUDA graph is not supported for UnfusedAttention backend")
    # TransformerLayer FP8 TN Gemm currently requires %8=0
    if is_fp8 and not (qkv_format == "thd" and module == "DotProductAttention"):
        pytest.skip("BSHD/SBHD <-> THD conversions for FP8 are not supported")

    # create full model
    logger.info("=== Generating all tokens at once ===")
    model = get_model(module, config, dtype, backend, qkv_format, num_layers, mode="reference")

    # generate data for all requests
    full_inputs = generate_args(module, config, dtype, qkv_format="bshd", mode="full_inputs")

    # generate reference results
    if module == "DotProductAttention":
        full_output = full_inputs
        for m in model:
            full_output = m(
                *full_output if isinstance(full_output, List) else full_output,
            )
    if module == "TransformerLayer":
        full_output = full_inputs
        for m in model:
            full_output = m(
                full_output[0] if isinstance(full_output, List) else full_output,
            )

    # create inference model
    logger.info("=== Generating one token at a time ===")
    model = get_model(
        module,
        config,
        dtype,
        backend,
        qkv_format,
        num_layers,
        mode="inference",
        is_fp8=is_fp8,
    )

    # graph the model if necessary
    if is_cuda_graph:
        t_seq_ids = torch.range(0, max_batch_size, dtype=torch.int32, device="cpu")
        step_lens = config.max_ctx_len * torch.ones(max_batch_size, dtype=torch.int32, device="cpu")
        step_dict = OrderedDict(zip(t_seq_ids.tolist(), step_lens.tolist()))
        inference_params.pre_step(step_dict)

        sample_args = generate_args(
            module, config, dtype, qkv_format=qkv_format, mode="sample_args"
        )
        sample_kwargs = {}
        sample_kwargs["cu_seqlens_q"] = torch.linspace(
            0,
            config.batch_size * config.max_ctx_len,
            steps=config.batch_size + 1,
            device="cuda",
            dtype=torch.int32,
        )
        sample_kwargs["cu_seqlens_kv"] = torch.linspace(
            0,
            config.batch_size * config.max_ctx_len,
            steps=config.batch_size + 1,
            device="cuda",
            dtype=torch.int32,
        )
        sample_kwargs["inference_params"] = inference_params
        sample_kwargs["max_seqlen_q"] = config.max_ctx_len
        sample_kwargs["max_seqlen_kv"] = config.max_seqlen_kv

        model = [
            make_graphed_callables(
                model[i],
                sample_args,
                num_warmup_iters=10,
                fp8_enabled=is_fp8,
                sample_kwargs=sample_kwargs,
                fp8_recipe=fp8_recipe,
            )
            for i in range(num_layers)
        ]

        sim.reset()
        inference_params.reset()
        step_dict = OrderedDict()

    # simulate step by step
    # t-1: ...
    #      compute for seq_ids = [0, 1, 2], ctx_lens = [5, 2, 3], gen_lens = [2, 9, 4],
    #              batch_size = 3, step_lens = [1, 1, 1]
    #      increase counter for gen_lens = [3, 10, 5]
    # t:   detect seq 1 is finished since expected_gen_lens = [12, 10, 15]
    #      add two new seqs 3 and 4, with ctx lens 10 and 11
    #      compute for seq_ids = [0, 2, 3, 4], ctx_lens = [5, 3, 10, 11], gen_lens = [3, 5, 0, 0],
    #              batch_size = 4, step_lens = [1, 1, 10, 11]
    #      increase counter for gen_lens = [3, 5, 1, 1]
    max_tokens = config.batch_size * config.max_ctx_len
    while True:
        # prepare batch for the current step
        dynamic_fill = True  # inference_params.is_paged
        sim.step(dynamic_fill=dynamic_fill)
        sim.print_step(logger)

        if sim.t_batch_size == 0:
            # all sequences are finished
            if sim.t > sim.last_arrival:
                sim.serving_times = sim.arrival_times + sim.request_delays
                sim.complete_times = sim.serving_times + sim.gen_lens
                break
            # not finished; run next iteration
            else:
                sim.t += 1
                continue

        # create incremental input
        batch_size = max_batch_size if is_cuda_graph else sim.t_batch_size
        max_seqlen_q = sim.max_ctx_len if is_cuda_graph else max(sim.step_lens).item()
        num_tensors = len(full_inputs)
        if qkv_format == "thd":
            incremental_inputs = []
            for i in range(num_tensors):
                inp = full_inputs[i]
                inc_inp = torch.Tensor().to(dtype=dtype, device="cuda")
                for i, seq in enumerate(sim.t_seq_ids):
                    start = (sim.t_total_lens[i] - sim.step_lens[i]).item()
                    end = sim.t_total_lens[i].item()
                    inc_inp = torch.cat([inc_inp, inp[seq, start:end]], dim=0)
                if is_cuda_graph:
                    inc_inp = torch.cat(
                        [
                            inc_inp,
                            torch.zeros(
                                max_tokens - sum(sim.step_lens),
                                *inp.shape[2:],
                                dtype=dtype,
                                device=inc_inp.device,
                            ),
                        ],
                        dim=0,
                    )
                incremental_inputs.append(inc_inp)
        else:
            incremental_inputs = []
            for i in range(num_tensors):
                inp = full_inputs[i]
                inc_inp = torch.zeros(
                    batch_size,
                    max_seqlen_q,
                    *inp.shape[2:],
                    dtype=dtype,
                    device="cuda",
                )
                for i, seq in enumerate(sim.t_seq_ids):
                    start = (sim.t_total_lens[i] - sim.step_lens[i]).item()
                    end = sim.t_total_lens[i].item()
                    inc_inp[i, : sim.step_lens[i], :] = inp[seq, start:end]
                if qkv_format == "sbhd":
                    inc_inp = inc_inp.transpose(0, 1).contiguous()
                incremental_inputs.append(inc_inp)

        # run step
        batch_size = max_batch_size if is_cuda_graph else sim.t_batch_size
        cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_q[1 : sim.t_batch_size + 1] = torch.cumsum(sim.step_lens, dim=0)
        cu_seqlens_kv = cu_seqlens_q.clone()
        step_dict = OrderedDict(zip(sim.t_seq_ids.tolist(), sim.step_lens.tolist()))
        inference_params.pre_step(step_dict)
        if inference_params.is_paged:
            inference_params.cache_manager.print_cache()
        incremental_output = incremental_inputs
        with fp8_autocast(enabled=is_fp8, fp8_recipe=fp8_recipe):
            for m in model:
                incremental_output = m(
                    *incremental_output,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                    inference_params=inference_params,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_kv=config.max_seqlen_kv,
                )
                incremental_output = [incremental_output]
        incremental_output = incremental_output[0]

        # compare results
        atol, rtol = get_tols(module, backend, dtype=dtype if not is_fp8 else torch.float8_e4m3fn)
        for i, seq in enumerate(sim.t_seq_ids):
            token_index = sim.step_lens[i] - 1
            if qkv_format == "bshd":
                torch.testing.assert_close(
                    full_output[seq, sim.t_total_lens[i] - 1, :],
                    incremental_output[i, sim.step_lens[i] - 1, :],
                    atol=atol,
                    rtol=rtol,
                )
            if qkv_format == "sbhd":
                torch.testing.assert_close(
                    full_output[seq, sim.t_total_lens[i] - 1, :],
                    incremental_output[sim.step_lens[i] - 1, i, :],
                    atol=atol,
                    rtol=rtol,
                )
            if qkv_format == "thd":
                torch.testing.assert_close(
                    full_output[seq, sim.t_total_lens[i] - 1, :],
                    incremental_output[cu_seqlens_q[i + 1] - 1, :],
                    atol=atol,
                    rtol=rtol,
                )

        sim.t += 1
        sim.t_gen_lens = sim.t_gen_lens + 1

    # last value in complete_times should be equal to sim.t
    sim.serving_times = sim.arrival_times + sim.request_delays
    sim.complete_times = sim.serving_times + sim.gen_lens
    sim.print_summary(logger)

    if backend == "FlashAttention" and not fa_utils.v3_is_installed:
        config.max_seqlen_q = config_max_seqlen_q
        config.max_seqlen_kv = config_max_seqlen_kv
