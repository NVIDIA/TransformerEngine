from collections import OrderedDict
import os
import logging

import pytest
import torch

from torch.distributions import Exponential
from transformer_engine.pytorch.attention import (
    DotProductAttention,
    InferenceParams,
)
from transformer_engine.pytorch.utils import is_bf16_compatible
from test_fused_attn import (
    ModelConfig,
    reset_rng_states,
    _get_attention_backends,
)
from tests.pytorch.test_numerics import assert_allclose

# Initialize RNG state
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
_cpu_rng_state = torch.get_rng_state()
_cuda_rng_state = torch.cuda.get_rng_state()

class Batch(object):
    def __init__(self):
        self.batch_size = 0
        self.seq_ids = torch.Tensor([]).to(dtype=torch.bool,device='cpu')
        self.ctx_lens = torch.Tensor([]).to(dtype=torch.bool,device='cpu')
        self.gen_lens = torch.Tensor([]).to(dtype=torch.bool,device='cpu')
        self.total_lens = self.ctx_lens + self.gen_lens
        self.expected_gen_lens = torch.Tensor([]).to(dtype=torch.bool,device='cpu')
        self.finished = torch.Tensor([]).to(dtype=torch.bool,device='cpu')
        self.step_lens_q = torch.Tensor([]).to(dtype=torch.int32,device='cpu')

    def copy(self):
        new_batch = Batch()
        new_batch.batch_size         = self.batch_size
        new_batch.seq_ids            = self.seq_ids
        new_batch.ctx_lens           = self.ctx_lens
        new_batch.gen_lens           = self.gen_lens
        new_batch.total_lens         = self.total_lens
        new_batch.expected_gen_lens  = self.expected_gen_lens
        new_batch.finished           = self.finished
        new_batch.step_lens_q        = self.step_lens_q
        return new_batch

    def print(self, logger, header='current batch:'):
        logger.debug(header)
        logger.debug('  {:<17s}: {}'.format('batch_size',self.batch_size))
        logger.debug('  {:<17s}: {}'.format('seq_ids',self.seq_ids.tolist()))
        logger.debug('  {:<17s}: {}'.format('ctx_lens',self.ctx_lens.tolist()))
        logger.debug('  {:<17s}: {}'.format('gen_lens',self.gen_lens.tolist()))
        logger.debug('  {:<17s}: {}'.format('total_lens',self.total_lens.tolist()))
        logger.debug('  {:<17s}: {}'.format('expected_gen_lens',self.expected_gen_lens.tolist()))
        logger.debug('  {:<17s}: {}'.format('finished',self.finished.tolist()))
        logger.debug('  {:<17s}: {}'.format('step_lens_q',self.step_lens_q.tolist()))

    def add_new_seqs(self, seq_ids, context_lens, expected_gen_lens):
        ctx_lens = context_lens[seq_ids]
        gen_lens = torch.Tensor([0] * len(seq_ids)).to(dtype=torch.int32,device='cpu')
        exp_gen_lens = expected_gen_lens[seq_ids]
        finished = torch.Tensor([False] * len(seq_ids)).to(dtype=torch.bool,device='cpu')

        self.batch_size = self.batch_size + len(seq_ids)
        self.finished = torch.cat([self.finished, finished], dim=0)

        if len(self.seq_ids) == 0:
            self.seq_ids = seq_ids
            self.ctx_lens = ctx_lens
            self.gen_lens = gen_lens
            self.expected_gen_lens = exp_gen_lens
        else:
            self.seq_ids = torch.cat([self.seq_ids, seq_ids],dim=0)
            self.ctx_lens = torch.cat([self.ctx_lens, ctx_lens], dim=0)
            self.gen_lens = torch.cat([self.gen_lens, gen_lens], dim=0)
            self.expected_gen_lens = torch.cat([self.expected_gen_lens, exp_gen_lens], dim=0)
        self.total_lens = self.ctx_lens + self.gen_lens
        self.step_lens_q = torch.cat([self.step_lens_q, ctx_lens], dim=0)

    def remove_finished(self):
        self.finished = torch.where(
                self.gen_lens - self.expected_gen_lens < 0, False, True).to(
                        dtype=torch.bool,device='cpu')
        self.batch_size = self.finished.logical_not().sum().item()
        self.seq_ids = self.seq_ids[~self.finished]
        self.ctx_lens = self.ctx_lens[~self.finished]
        self.gen_lens = self.gen_lens[~self.finished]
        self.total_lens = self.total_lens[~self.finished]
        self.expected_gen_lens = self.expected_gen_lens[~self.finished]
        self.gen_lens = self.gen_lens + 1
        self.total_lens = self.total_lens + 1
        self.step_lens_q = torch.ones([self.batch_size], dtype=torch.int32, device='cpu')

param_types = [torch.float16]
if is_bf16_compatible():
    param_types.append(torch.bfloat16)

model_configs_infer = {
    #    test:             b,  h, hg,  d,  sq, skv,   p,      mask,      bias
    "infer_0": ModelConfig(4, 16, 16, 64, 66, 66, 0.0, "no_mask", "no_bias", total_requests=8),
    "infer_1": ModelConfig(2, 16,  4, 64, 66, 66, 0.0, "no_mask", "no_bias", total_requests=6),
    }

qkv_formats = ['bshd', 'sbhd', 'thd']

def to_pretty_string(x: torch.Tensor):
    return '['+','.join(['{:>3s}'.format(str(i)) for i in x.tolist()])+']'

@pytest.mark.parametrize("dtype", param_types)
@pytest.mark.parametrize("model", model_configs_infer.keys())
@pytest.mark.parametrize("qkv_format", qkv_formats)
@pytest.mark.parametrize("is_paged", [False, True])
@pytest.mark.parametrize("backend", ["FusedAttention", "FlashAttention", "UnfusedAttention"])
@pytest.mark.parametrize("is_cuda_graph", [False, True])
def test_paged_attn(dtype, model, qkv_format, is_paged, backend, is_cuda_graph):
    reset_rng_states()
    logger = logging.getLogger('test_paged_attn')

    config = model_configs_infer[model]
    layer_number = 1

    inference_params_qkv_format = 'bshd'
    if is_paged:
        qkv_layout = "paged_kv_"+inference_params_qkv_format+'_2'+inference_params_qkv_format
    else:
        qkv_layout = '_'.join([inference_params_qkv_format]*3)
    available_backends, fused_attn_backends = _get_attention_backends(
        config,
        qkv_dtype=dtype,
        qkv_layout=qkv_layout,
        window_size=config.window_size,
        pad_between_seqs=False,
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

    total_requests = config.total_requests
    # max_batch_size may be smaller than total_requests
    max_batch_size = config.batch_size
    # maximum KV length (context + generation)
    max_seqlen_kv = config.max_seqlen_kv

    # mask type for inference
    attn_mask_type = "padding"

    # page size in number of tokens (k cache and v cache are separate)
    page_size = 256 if backend == "FlashAttention" else 16

    max_seqlen_kv_roundup = max_seqlen_kv
    if is_paged:
        max_seqlen_kv_roundup = int((max_seqlen_kv + page_size - 1)//page_size * page_size)
    else:
        max_seqlen_kv_roundup = int((max_seqlen_kv + 63)//64 * 64)
    cache_size = max_batch_size * max_seqlen_kv_roundup
    total_num_pages = int(cache_size / page_size)

    context_ratio = 0.25
    gen_ratio = 1 - context_ratio
    max_context_len = int(max_seqlen_kv * context_ratio)
    max_gen_len = int(max_seqlen_kv * gen_ratio)

    # context lengths in Uniform distribution
    context_lens = torch.randint(1, max_context_len, [total_requests], dtype=torch.int32, device='cpu')
    # generation lengths in Exponential distribution
    gen_dist = Exponential(1/max_gen_len)
    gen_lens = gen_dist.sample((total_requests,))
    gen_lens = torch.where(gen_lens>max_gen_len, max_gen_len, gen_lens).to(dtype=torch.int32, device='cpu')
    # arrival times in Poisson distribution
    rate = torch.randint(1, max_batch_size, [1]).item()
    interval_dist = Exponential(rate)
    arrival_intervals = interval_dist.sample((total_requests,))
    arrival_times = torch.cumsum(arrival_intervals,dim=0).to(dtype=torch.int32, device='cpu')
    last_arrival = arrival_times.max().item()

    logger.info("Simulation:")
    logger.info(f"  total num of requests: {total_requests}")
    logger.info(f"  k/v cache size:        {cache_size} tokens")
    logger.info(f"  is_paged:              {is_paged}")
    logger.info(f"  dtype:                 {dtype}")
    if not is_paged:
        logger.info(f"  max_batch_size:        {max_batch_size}")
        logger.info(f"  max_seqlen_kv:         {max_seqlen_kv}")
    else:
        logger.info(f"  total_num_pages:       {total_num_pages}")
        logger.info(f"  page_size:             {page_size}")
    logger.info(f"  context_lens:          {to_pretty_string(context_lens)}")
    logger.info(f"  expected_gen_lens:     {to_pretty_string(gen_lens)}")
    logger.info(f"  arrival_times:         {to_pretty_string(arrival_times)}")

    model = (
        DotProductAttention(
            kv_channels=config.head_dim_qk,
            num_attention_heads=config.num_heads,
            num_gqa_groups=config.num_gqa_groups,
            layer_number=layer_number,
            attention_dropout=0.0,
            attn_mask_type="causal",
            qkv_format='bshd',
        )
        .cuda()
        .eval()
    )

    q = 0.1 * torch.randn(
            (total_requests, max_seqlen_kv, config.num_heads, config.head_dim_qk),
            dtype=dtype, device="cuda")
    k = 0.1 * torch.randn(
            (total_requests, max_seqlen_kv, config.num_gqa_groups, config.head_dim_qk),
            dtype=dtype, device="cuda")
    v = 0.1 * torch.randn(
            (total_requests, max_seqlen_kv, config.num_gqa_groups, config.head_dim_v),
            dtype=dtype, device="cuda")

    logger.info("")
    logger.info("=== Generating all tokens at once ===")
    request_delays = torch.zeros([total_requests],dtype=torch.int32,device='cpu')
    full_output = model(
            query_layer=q,
            key_layer=k,
            value_layer=v,
            qkv_format='bshd',
            attn_mask_type="causal",
    )

    t = 1
    logger.info(f"total steps taken: {t}")
    logger.info(f"arrival_times:     {to_pretty_string(arrival_times)}")
    logger.info(f"gen_lens:          {to_pretty_string(gen_lens)}")
    logger.info(f"serving_times:     {to_pretty_string(arrival_times + request_delays)}")

    logger.info("")
    logger.info("=== Generating one token at a time ===")
    inference_params = InferenceParams(
            max_batch_size=max_batch_size,
            max_seqlen_kv=max_seqlen_kv_roundup,
            num_heads_kv=config.num_gqa_groups,
            head_dim_k=config.head_dim_qk,
            head_dim_v=config.head_dim_v,
            dtype=dtype,
            is_paged=is_paged,
            page_size=page_size,
            total_num_pages=total_num_pages,
            is_cuda_graph=is_cuda_graph,
            num_heads_q=config.num_heads,
            head_dim_q=config.head_dim_qk,
            )
    inference_params.allocate_memory(layer_number)
    inference_params.print()

    request_delays = torch.zeros([total_requests],dtype=torch.int32,device='cpu')
    t = 0
    prev = Batch()
    delayed_seq_ids = torch.Tensor().to(dtype=torch.int32,device='cpu')
    while True:
        logger.debug(f"time step {t}")
        cur = prev.copy()
        if t != 0:
            cur.remove_finished()
        if inference_params.is_paged:
            inference_params.cache_manager.print_cache()

        arrived_seq_ids = torch.where(arrival_times == t, True, False).nonzero().view(-1)
        if inference_params.is_paged:
            allowed_num_new_seqs = max_batch_size - cur.batch_size
        else:
            allowed_num_new_seqs = 0 if cur.batch_size > 0 else max_batch_size
        queuing_seq_ids = torch.cat([delayed_seq_ids, arrived_seq_ids],dim=0)
        logger.debug(f"arrived seq_ids:              {to_pretty_string(arrived_seq_ids)}")
        logger.debug(f"previously delayed seq_ids:   {to_pretty_string(delayed_seq_ids)}")
        logger.debug(f"allowed num of new sequences: {allowed_num_new_seqs}")
        if len(queuing_seq_ids) > allowed_num_new_seqs:
            seq_ids = queuing_seq_ids[:allowed_num_new_seqs]
            delayed_seq_ids = queuing_seq_ids[allowed_num_new_seqs:]
            request_delays[delayed_seq_ids.tolist()] += 1
        else:
            seq_ids = queuing_seq_ids
            delayed_seq_ids = torch.Tensor().to(dtype=torch.int32)
        cur.add_new_seqs(seq_ids, context_lens, gen_lens)
        cur.print(logger)
        if inference_params.is_paged:
            inference_params.cache_manager.print_cache()

        if cur.batch_size == 0:
            # all sequences are finished
            if t > last_arrival:
                break
            # not finished; run next iteration
            else:
                prev = cur.copy()
                t += 1
                continue

        if not is_cuda_graph:
            max_seqlen_q_infer = int((cur.step_lens_q.max().item() + 63)//64 * 64)
        else:
            max_seqlen_q_infer = max_seqlen_kv_roundup

        # create incremental input
        if qkv_format == 'thd':
            incremental_q = torch.Tensor().to(dtype=dtype, device='cuda')
            incremental_k = torch.Tensor().to(dtype=dtype, device='cuda')
            incremental_v = torch.Tensor().to(dtype=dtype, device='cuda')
            for i,seq in enumerate(cur.seq_ids):
                start = (cur.total_lens[i]-cur.step_lens_q[i]).item()
                end = cur.total_lens[i].item()
                incremental_q = torch.cat([incremental_q,
                    q[seq, start:end, :, :]],dim=0)
                incremental_k = torch.cat([incremental_k,
                    k[seq, start:end, :, :].view(-1, config.num_gqa_groups, config.head_dim_qk)], dim=0)
                incremental_v = torch.cat([incremental_v,
                    v[seq, start:end, :, :].view(-1, config.num_gqa_groups, config.head_dim_v)], dim=0)
        else:
            incremental_q = torch.zeros(
                    cur.batch_size, max_seqlen_q_infer, config.num_heads, config.head_dim_qk,
                    dtype=dtype, device='cuda')
            incremental_k = torch.zeros(
                    cur.batch_size, max_seqlen_q_infer, config.num_gqa_groups, config.head_dim_qk,
                    dtype=dtype, device='cuda')
            incremental_v = torch.zeros(
                    cur.batch_size, max_seqlen_q_infer, config.num_gqa_groups, config.head_dim_v,
                    dtype=dtype, device='cuda')
            for i,seq in enumerate(cur.seq_ids):
                start = (cur.total_lens[i]-cur.step_lens_q[i]).item()
                end = cur.total_lens[i].item()
                incremental_q[i, :cur.step_lens_q[i], :, :] = q[seq, start:end, :, :]
                incremental_k[i, :cur.step_lens_q[i], :, :] = k[seq, start:end, :, :]
                incremental_v[i, :cur.step_lens_q[i], :, :] = v[seq, start:end, :, :]
            if qkv_format == 'sbhd':
                incremental_q, incremental_k, incremental_v = [
                    x.transpose(0,1) for x in [incremental_q, incremental_k, incremental_v]]

        cu_seqlens_q = torch.zeros(cur.batch_size + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_q[1:cur.batch_size+1] = torch.cumsum(cur.step_lens_q, dim=0)
        cu_seqlens_kv = torch.zeros(cur.batch_size + 1, dtype=torch.int32, device="cuda")
        cu_seqlens_kv[1:cur.batch_size+1] = torch.cumsum(cur.total_lens, dim=0)

        inference_params.step_dict = OrderedDict(zip(cur.seq_ids.tolist(), cur.step_lens_q.tolist()))

        line_output = model(
            query_layer=incremental_q,
            key_layer=incremental_k,
            value_layer=incremental_v,
            inference_params=inference_params,
            attn_mask_type=attn_mask_type,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q_infer,
            max_seqlen_kv=max_seqlen_kv_roundup,
            qkv_format=qkv_format,
        )

        if backend != "FlashAttention":
            tols = {
                torch.float32: 1e-3,
                torch.half: 1e-3,
                torch.bfloat16: 1e-2,
            }
        else:
            tols = {
                torch.float32: 1e-3,
                torch.half: 4e-3,
                torch.bfloat16: 1e-2,
            }
        for i,seq in enumerate(cur.seq_ids):
            if qkv_format == 'bshd':
                torch.testing.assert_close(
                        full_output[seq,cur.total_lens[i]-1,:],
                        line_output[i,cur.step_lens_q[i]-1,:],
                        atol = tols[dtype],
                        rtol = tols[dtype])
            if qkv_format == 'sbhd':
                torch.testing.assert_close(
                        full_output[seq,cur.total_lens[i]-1,:],
                        line_output[cur.step_lens_q[i]-1,i,:],
                        atol = tols[dtype],
                        rtol = tols[dtype])
            if qkv_format == 'thd':
                torch.testing.assert_close(
                        full_output[seq,cur.total_lens[i]-1,:],
                        line_output[cu_seqlens_q[i+1]-1,:],
                        atol = tols[dtype],
                        rtol = tols[dtype])

        prev = cur.copy()
        t += 1

    logger.info(f"total steps taken: {t}")
    logger.info(f"arrival_times:     {to_pretty_string(arrival_times)}")
    logger.info(f"gen_lens:          {to_pretty_string(gen_lens)}")
    logger.info(f"serving_times:     {to_pretty_string(arrival_times + request_delays)}")
