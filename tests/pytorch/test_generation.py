# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch

import transformer_engine.pytorch as te


class TestInferenceParams:
    def test_setup_before_new_input_bshd(self):
        inference_params = te.attention.InferenceParams(64, 128, qkv_format="bshd")

        inference_params.setup_before_new_input(length=16)
        # Offset before first sequence is equal to 0.
        assert inference_params.sequence_len_offset == 0

        # Offset before second sequence is equal to 16.
        inference_params.setup_before_new_input(length=4)
        assert inference_params.sequence_len_offset == 16

    def test_setup_before_new_input_thd(self):
        inference_params = te.attention.InferenceParams(4, 128, qkv_format="thd")

        inference_params.setup_before_new_input(
            lengths_tensor=torch.Tensor([1, 0, 2, 4]).cuda(), max_input_length=20)

        assert torch.equal(
            inference_params.cached_sequence_lengths,
            torch.Tensor([0, 0, 0, 0]).cuda()
        )
        assert torch.equal(
            inference_params.input_sequence_lengths,
            torch.Tensor([1, 0, 2, 4]).cuda()
        )
        assert inference_params.max_incoming_seq_len == 20


        inference_params.setup_before_new_input(
            lengths_tensor=torch.Tensor([2, 3, 5, 1]).cuda(), max_input_length=10)
        assert torch.equal(
            inference_params.cached_sequence_lengths,
            torch.Tensor([1, 0, 2, 4]).cuda()
        )
        assert torch.equal(
            inference_params.input_sequence_lengths,
            torch.Tensor([2, 3, 5, 1]).cuda()
        )
        assert inference_params.max_incoming_seq_len == 10

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("batch_size", [64, 128, 256])
    @pytest.mark.parametrize("max_seq_len", [128, 256, 512])
    @pytest.mark.parametrize("max_input_len", [32, 128])
    def test_save_to_kv_cache_thd(self, batch_size, max_seq_len, max_input_len, dtype):
        h, d = 16, 256

        inference_params = te.attention.InferenceParams(batch_size, max_seq_len, qkv_format="thd")
        inference_params.allocate_memory_for_kv_cache_if_empty(1, h, d, dtype)

        t = batch_size * max_input_len
        key_layer = torch.randn((t, h, d)).cuda().to(dtype)
        value_layer = torch.randn((t, h, d)).cuda().to(dtype)

        sequence_lengths = [1, 2] * (batch_size // 2)

        # We save the same sequences two time, which should result in sequences of lentgh 2 and 4
        # in the cache
        inference_params.reset()
        inference_params.setup_before_new_input(
            lengths_tensor=torch.tensor(sequence_lengths).cuda(), max_input_length=max_input_len)
        inference_params.save_to_kv_cache(1, key_layer, value_layer)

        inference_params.setup_before_new_input(
            lengths_tensor=torch.tensor(sequence_lengths).cuda(), max_input_length=max_input_len)
        inference_params.save_to_kv_cache(1, key_layer, value_layer)

        key_memory, value_memory = inference_params.key_value_memory_dict[1]

        # Chcek whether the sequences were copied properly.

        def check(memory, layer, b, idx1, idx2):
            # Check if sequence idx in batch b in memory corresponds
            # to the sequence idx2 in batch b in layer.
            assert torch.equal(
                memory[b * max_seq_len + idx1],
                layer[b * max_input_len + idx2, :]
            )

        # even indices
        for b in range(0, batch_size, 2):
            check(key_memory, key_layer, b, 0, 0)
            check(key_memory, key_layer, b, 1, 0)
            assert (key_memory[b * max_seq_len + 2:((b + 1) * max_seq_len)] == 0).all()

            check(value_memory, value_layer, b, 0, 0)
            check(value_memory, value_layer, b, 1, 0)
            assert (value_memory[b * max_seq_len + 2:((b + 1) * max_seq_len)] == 0).all()

        # odd indices
        for b in range(1, batch_size, 2):
            check(key_memory, key_layer, b, 0, 0)
            check(key_memory, key_layer, b, 1, 1)
            check(key_memory, key_layer, b, 2, 0)
            check(key_memory, key_layer, b, 3, 1)
            assert (key_memory[b * max_seq_len + 4:((b + 1) * max_seq_len)] == 0).all()

            check(value_memory, value_layer, b, 0, 0)
            check(value_memory, value_layer, b, 1, 1)
            check(value_memory, value_layer, b, 2, 0)
            check(value_memory, value_layer, b, 3, 1)
            assert (value_memory[b * max_seq_len + 4:((b + 1) * max_seq_len)] == 0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("batch_size", [64, 128, 256])
    @pytest.mark.parametrize("max_seq_len", [128, 256, 512])
    def test_save_to_kv_cache_bshd(self, batch_size, max_seq_len, dtype):
        # This test checks if key_layer and value_layer are copied to cache.
        # Cache size is equal to the size of one key/value layer.
        h, d = 16, 256

        inference_params = te.attention.InferenceParams(batch_size, max_seq_len, qkv_format="bshd")

        inference_params.allocate_memory_for_kv_cache_if_empty(1, h, d, dtype)
        key_layer = torch.randn((max_seq_len, batch_size, h, d)).cuda().to(dtype)
        value_layer = torch.randn((max_seq_len, batch_size, h, d)).cuda().to(dtype)

        inference_params.setup_before_new_input(length=0)
        inference_params.save_to_kv_cache(1, key_layer, value_layer)

        key_memory, value_memory = inference_params.key_value_memory_dict[1]

        assert torch.equal(key_memory, key_layer)
        assert torch.equal(value_memory, value_layer)

    @pytest.mark.parametrize("layer_number", [1, 100])
    @pytest.mark.parametrize("batch_size", [1, 128])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_allocate_memory_for_kv_cache_if_empty(
        self,
        layer_number,
        batch_size,
        dtype
        ):
        nr_heads = 16
        head_dim = 256
        max_sequence_len = 128
        inference_params = te.attention.InferenceParams(
            batch_size, max_sequence_len, qkv_format="bshd")

        assert layer_number not in inference_params.key_value_memory_dict

        inference_params.allocate_memory_for_kv_cache_if_empty(
            layer_number, nr_heads, head_dim, dtype)

        key_memory, value_memory =  inference_params.key_value_memory_dict[layer_number]

        assert key_memory.shape == (max_sequence_len, batch_size, nr_heads, head_dim)
        assert value_memory.shape == (max_sequence_len, batch_size, nr_heads, head_dim)

        # Should not allocate new buffers.
        inference_params.allocate_memory_for_kv_cache_if_empty(
            layer_number, 100, 100, dtype)


        assert key_memory.shape == (max_sequence_len, batch_size, nr_heads, head_dim)
        assert value_memory.shape == (max_sequence_len, batch_size, nr_heads, head_dim)


    def test_set_params_to_thd_attention(self):
        # This test check whether parameteres needed to run thd attention
        # are computed correcly. This parameters are passed to the fused_attn_fwd(..)
        # to indicate which parts of the key/query/value layers are sequences and
        # which of them are offsets.
        batch_size = 4
        channels = 1024
        max_sequence_len = 128
        max_input_len = 20
        inference_params = te.attention.InferenceParams(
            batch_size, max_sequence_len, qkv_format="thd")

        inference_params.setup_before_new_input(
            lengths_tensor=torch.Tensor([1, 1, 1, 1]).cuda(), max_input_length=max_input_len)
        inference_params.setup_before_new_input(
            lengths_tensor=torch.Tensor([1, 0, 2, 4]).cuda(), max_input_length=max_input_len)

        buffers = [
                torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda") for _ in range(6)]
        max_q_len, max_kv_len, buffers = \
            inference_params.set_params_to_thd_attention(buffers, channels)

        cu_seqlens_q, cu_seqlens_kv, seq_offsets_q, \
            seq_offsets_k, seq_offsets_v, seq_offsets_o = buffers

        assert max_q_len == max_input_len
        assert max_kv_len == max_sequence_len
        assert torch.equal(
            cu_seqlens_q,
            torch.tensor([0, 1, 1, 3, 7]).cuda()
        )
        assert torch.equal(
            cu_seqlens_kv,
            torch.tensor([0, 2, 3, 6, 11]).cuda()
        )

        assert torch.equal(
            seq_offsets_q,
            torch.tensor([k * max_input_len * channels for k in range(batch_size + 1)]).cuda()
        )
        assert torch.equal(
            seq_offsets_k,
            torch.tensor([k * max_sequence_len * channels for k in range(batch_size + 1)]).cuda()
        )
        assert torch.equal(
            seq_offsets_v,
            torch.tensor([k * max_sequence_len * channels for k in range(batch_size + 1)]).cuda()
        )
        assert torch.equal(
            seq_offsets_o,
            torch.tensor([k * max_input_len * channels for k in range(batch_size + 1)]).cuda()
        )

# This class tests whether inference_params attribute to TransformerLayer works correctly.
# Namely, whether key and value layers of the
# sequences forwarded to the model once are remembered in the cache.
class TestMemory:
    @pytest.mark.parametrize("gen_phase_length", [1, 2, 4])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
    def test_bshd_memory(self, gen_phase_length, dtype):
        """
            The test contains of:
            - one context phase when sequences with length 64 are passed through the model,
            - gen_phase_length phases when sequences with length 1 are passed through the model.

            The output is compared with the case when all this sequences are passed at one.
        """
        context_phase_length = 64
        batch_size = 64
        max_seq_len = 256
        hidden_dim = 256
        nr_heads = 4
        torch.manual_seed(1234)
        input = torch.randn(
            (batch_size, context_phase_length + gen_phase_length, hidden_dim), dtype=dtype).cuda()
        model = te.TransformerLayer(
            hidden_dim, 256, nr_heads,
            layer_number=1,
            attn_input_format="bshd",
            self_attn_mask_type="causal",
            attention_dropout=0,
            hidden_dropout=0).to(dtype).cuda()

        output_split = torch.Tensor().cuda().to(dtype)
        inference_params = te.attention.InferenceParams(batch_size, max_seq_len, qkv_format="bshd")

        # context phase
        chunk = input[:, :context_phase_length, :]
        inference_params.setup_before_new_input(length=context_phase_length)
        output_split = torch.concat(
            (
                output_split,
                model(chunk, inference_params=inference_params, self_attn_mask_type="causal")
            ), dim=1)

        # generation phase
        for i in range(gen_phase_length):
            chunk = input[:, (context_phase_length + i):(context_phase_length + i + 1), :]
            inference_params.setup_before_new_input(length=1)
            output_split = torch.concat(
                (
                    output_split,
                    model(chunk, inference_params=inference_params, self_attn_mask_type="no_mask")
                ), dim=1)

        # ground truth - one pass input
        output_no_split = model(input)

        torch.testing.assert_close(
            output_no_split,
            output_split,
            atol=1e-3,
            rtol=0
        )

    # torch.float32 does not support thd
    @pytest.mark.parametrize("gen_phase_length", [1, 8, 32])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_thd_memory(self,  gen_phase_length, dtype):
        """
            In thd attention sequences can have various lengths,
            different that 's' dimension of input to the Transformer Layer.

            The test contains of:
            - one context phase when sequences with various lengths(!) are passed through the model,
            - gen_phase_length phases when sequences with length 1 are passed through the model.

            The output is compared with the case when all this sequences are passed at one.
        """
        context_phase_len = 64
        batch_size = 64
        max_seq_len = 256
        hid_dim = 256
        torch.manual_seed(1234)

        # Tensors have shapes [b, s, h, d] and the seqlens are the tensor of shapes [b]
        # which indicate the length of sequences - sequences starts from the begining.
        # This function copies sequences from tensor2 into tensor1.
        # tensor1 should be big enough to fit this sequences.
        def _concat_thd(tensor1, seqlens1, tensor2, seqlens2):
            for b in range(batch_size):
                tensor1[b, seqlens1[b]:(seqlens1[b] + seqlens2[b]), :] = tensor2[b, :seqlens2[b], :]
            seqlens1.copy_(seqlens1 + seqlens2)


        model = te.TransformerLayer(
            hid_dim, 256, 16,
            layer_number=1,
            attn_input_format="thd",
            attention_dropout=0,
            hidden_dropout=0,
            self_attn_mask_type="padding_causal").to(dtype)
        model.eval()

        inference_params = te.attention.InferenceParams(batch_size, max_seq_len, qkv_format="thd")

        total_sequence_lengths = torch.zeros((batch_size,)).cuda().to(torch.int32)
        total_tensor = torch.zeros((batch_size, max_seq_len, hid_dim)).cuda().to(dtype)

        # Sequences split into chunks.
        output_split = None

        # context phase
        sequence_lengths = torch.randint(1, context_phase_len, (batch_size,)).cuda().to(torch.int32)
        chunk = torch.randn((batch_size, context_phase_len, hid_dim)).cuda().to(dtype)
        inference_params.setup_before_new_input(
                max_input_length=context_phase_len, lengths_tensor=sequence_lengths)
        model(chunk, inference_params=inference_params)
        _concat_thd(total_tensor, total_sequence_lengths, chunk, sequence_lengths)

        # generation phase
        for _ in range(gen_phase_length):
            sequence_lengths = torch.ones((batch_size,)).cuda().to(torch.int32)
            chunk = torch.randn((batch_size, 1, hid_dim)).cuda().to(dtype)
            inference_params.setup_before_new_input(
                    max_input_length=1, lengths_tensor=sequence_lengths)
            output_split = model(
                chunk, inference_params=inference_params, self_attn_mask_type="padding")
            _concat_thd(total_tensor, total_sequence_lengths, chunk, sequence_lengths)
        logits_split = output_split[:, - 1, :]

        # Sequences passed in one, concatenated chunk.
        inference_params.reset()
        inference_params.setup_before_new_input(
            max_input_length=max_seq_len, lengths_tensor=total_sequence_lengths)
        output_no_split = model(total_tensor, inference_params=inference_params)
        logits_no_split = output_no_split[
            torch.arange(0, batch_size), total_sequence_lengths - 1, :] # last element of each seq.

        # Final result should be close.
        torch.testing.assert_close(
            logits_no_split,
            logits_split,
            atol=1e-2,
            rtol=1e-2
        )
